"""验证原同细胞loop与低内存快速路径的逐值一致性。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from tools.engines.compartment_adapter import (
    _calculate_mean_cell_radius,
    _compute_cell_vectors_notebook,
    _compute_local_expression_vectors_notebook,
    _compute_rnaflux_loop_notebook,
    _filter_boundary_dict,
    _interpolate_rgb_same_cell,
    _make_shape_df,
    _resolve_compartment_radius,
)
from tools.engines.compartment_same_cell_fast import (
    compute_same_cell_rnaflux_fast,
    interpolate_same_cell_rgb_fast,
)


def _make_same_cell_test_dataframe() -> pd.DataFrame:
    """创建基因比例不同、细胞空间分离且单细胞坐标非共线的测试数据。"""

    gene_patterns = {
        "cell_1": ["G1", "G1", "G2", "G2", "G3", "G3", "G4", "G4"],
        "cell_2": ["G1", "G2", "G2", "G2", "G3", "G4", "G4", "G4"],
        "cell_3": ["G1", "G1", "G1", "G2", "G3", "G3", "G4", "G4"],
        "cell_4": ["G1", "G2", "G2", "G3", "G3", "G3", "G4", "G4"],
    }
    local_coordinates = [(0, 0), (1, 0), (0, 1), (1, 1), (2, 0), (0, 2), (2, 2), (1, 2)]
    rows = []
    for cell_index, (cell_id, genes) in enumerate(gene_patterns.items()):
        x_offset = (cell_index % 2) * 10
        y_offset = (cell_index // 2) * 10
        for gene, (x_value, y_value) in zip(genes, local_coordinates):
            rows.append({"cell": cell_id, "gene": gene, "x": x_value + x_offset, "y": y_value + y_offset, "z": 0, "umi": 1})
    return pd.DataFrame(rows)


class SameCellFastEquivalenceTests(unittest.TestCase):
    """比较原同细胞参考函数和新快速助手的确定性中间结果。"""

    def test_fast_rnaflux_matches_original_loop_exactly(self) -> None:
        """确认查询顺序、查询细胞和RNAflux逐值完全一致。"""

        dataframe = _make_same_cell_test_dataframe()
        sampled_dataframe = dataframe.sample(frac=0.5, random_state=42)
        unique_genes = dataframe["gene"].unique()
        device = torch.device("cpu")
        query_coords, expression_vectors, cell_ids = _compute_local_expression_vectors_notebook(
            dataframe,
            sampled_dataframe,
            unique_genes,
            1.5,
            device,
            same_cell_neighborhood=True,
        )
        _, standard_deviation = _compute_cell_vectors_notebook(dataframe, unique_genes, device)
        expected_rnaflux = _compute_rnaflux_loop_notebook(
            dataframe,
            unique_genes,
            query_coords,
            expression_vectors,
            cell_ids,
            standard_deviation,
            device,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            actual_coords, actual_cells, actual_rnaflux, _ = compute_same_cell_rnaflux_fast(
                dataframe,
                sampled_dataframe,
                unique_genes,
                1.5,
                device,
                Path(temp_dir),
            )
            np.testing.assert_array_equal(actual_coords, np.asarray(query_coords))
            np.testing.assert_array_equal(actual_cells, np.asarray(cell_ids))
            np.testing.assert_array_equal(np.asarray(actual_rnaflux), expected_rnaflux)

    def test_fast_interpolation_matches_original_griddata_exactly(self) -> None:
        """确认无查询点、最近邻和线性插值分支均保持原逐通道结果。"""

        dataframe = _make_same_cell_test_dataframe()
        sampled_dataframe = dataframe.sample(frac=0.5, random_state=42)
        unique_genes = dataframe["gene"].unique()
        device = torch.device("cpu")
        query_coords, _, cell_ids = _compute_local_expression_vectors_notebook(
            dataframe,
            sampled_dataframe,
            unique_genes,
            1.5,
            device,
            same_cell_neighborhood=True,
        )
        normalized_embeddings = np.random.RandomState(42).uniform(0.1, 0.9, size=(len(sampled_dataframe), 3))
        expected_rgb = _interpolate_rgb_same_cell(dataframe, query_coords, cell_ids, normalized_embeddings)
        with tempfile.TemporaryDirectory() as temp_dir:
            actual_coords, _, _, state = compute_same_cell_rnaflux_fast(
                dataframe,
                sampled_dataframe,
                unique_genes,
                1.5,
                device,
                Path(temp_dir),
            )
            actual_rgb = interpolate_same_cell_rgb_fast(
                dataframe,
                actual_coords,
                normalized_embeddings,
                state,
                Path(temp_dir) / "rgb.npy",
            )
            np.testing.assert_array_equal(np.asarray(actual_rgb), expected_rgb)

    def test_fast_interpolation_preserves_original_nan_inf_cleanup(self) -> None:
        """确认快速插值保留原函数最终使用全局标量均值回填异常值的规则。"""

        dataframe = _make_same_cell_test_dataframe()
        sampled_dataframe = dataframe.sample(frac=0.5, random_state=42)
        unique_genes = dataframe["gene"].unique()
        device = torch.device("cpu")
        query_coords, _, cell_ids = _compute_local_expression_vectors_notebook(
            dataframe,
            sampled_dataframe,
            unique_genes,
            1.5,
            device,
            same_cell_neighborhood=True,
        )
        normalized_embeddings = np.random.RandomState(42).uniform(0.1, 0.9, size=(len(sampled_dataframe), 3))
        normalized_embeddings[0, 0] = np.inf
        expected_rgb = _interpolate_rgb_same_cell(dataframe, query_coords, cell_ids, normalized_embeddings)
        with tempfile.TemporaryDirectory() as temp_dir:
            actual_coords, _, _, state = compute_same_cell_rnaflux_fast(
                dataframe,
                sampled_dataframe,
                unique_genes,
                1.5,
                device,
                Path(temp_dir),
            )
            actual_rgb = interpolate_same_cell_rgb_fast(
                dataframe,
                actual_coords,
                normalized_embeddings,
                state,
                Path(temp_dir) / "rgb.npy",
            )
            np.testing.assert_array_equal(np.asarray(actual_rgb), expected_rgb)

    def test_prefiltered_boundaries_match_original_postfilter(self) -> None:
        """确认提前筛边界与原先先转WKT再筛选得到完全相同的表格和顺序。"""

        boundary = {
            "cell_3": pd.DataFrame({"x": [0, 1, 0], "y": [0, 0, 1]}),
            "cell_1": pd.DataFrame({"x": [2, 3, 2], "y": [2, 2, 3]}),
            "cell_2": pd.DataFrame({"x": [4, 5, 4], "y": [4, 4, 5]}),
        }
        selected_cells = {"cell_1", "cell_3"}
        expected = _make_shape_df(boundary, "cell_shape")
        expected = expected[expected["cell"].isin(selected_cells)]
        actual = _make_shape_df(_filter_boundary_dict(boundary, selected_cells), "cell_shape")
        pd.testing.assert_frame_equal(actual.reset_index(drop=True), expected.reset_index(drop=True))

    def test_auto_radius_uses_current_group_mean_cell_radius_divided_by_three(self) -> None:
        """确认自动半径只使用当前组边界，并严格返回平均细胞半径的三分之一。"""

        boundary = {
            "cell_1": pd.DataFrame({"x": [-1.0, 1.0, 1.0, -1.0], "y": [-1.0, -1.0, 1.0, 1.0]}),
            "cell_2": pd.DataFrame({"x": [-2.0, 2.0, 2.0, -2.0], "y": [-2.0, -2.0, 2.0, 2.0]}),
            "excluded_cell": pd.DataFrame({"x": [-100.0, 100.0, 100.0, -100.0], "y": [-100.0, -100.0, 100.0, 100.0]}),
        }
        selected_cells = {"cell_1", "cell_2"}
        expected_mean_radius = (np.sqrt(2.0) + np.sqrt(8.0)) / 2.0
        actual_mean_radius = _calculate_mean_cell_radius(boundary, selected_cells)
        actual_radius = _resolve_compartment_radius("auto", boundary, selected_cells)
        self.assertAlmostEqual(actual_mean_radius, expected_mean_radius)
        self.assertAlmostEqual(actual_radius, expected_mean_radius / 3.0)

    def test_explicit_positive_radius_is_preserved(self) -> None:
        """确认显式绝对半径不经过边界换算，便于固定参数运行。"""

        self.assertEqual(_resolve_compartment_radius(5, {}, set()), 5.0)

    def test_auto_radius_requires_valid_boundaries(self) -> None:
        """确认自动半径在缺少有效边界时给出明确错误，而不是静默使用魔法默认值。"""

        with self.assertRaisesRegex(ValueError, "valid cell boundary"):
            _resolve_compartment_radius("auto", {}, {"cell_1"})


if __name__ == "__main__":
    unittest.main()
