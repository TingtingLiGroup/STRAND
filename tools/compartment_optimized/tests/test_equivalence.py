"""用小型确定性数据验证优化实现与慢速细胞内参考算法的一致性。"""

from __future__ import annotations

import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from minisom import MiniSom
from scipy.interpolate import griddata
from scipy.spatial import cKDTree

from tools.compartment_optimized.core import (
    OptimizedCompartmentConfig,
    build_cell_row_index,
    _iter_contiguous_cell_blocks,
    _sample_positions,
    _som_quantization_error_chunked,
    _som_winners_chunked,
    compute_same_cell_rnaflux,
    interpolate_same_cell_rgb,
    run_optimized_compartment,
    write_tile_coordinate_outputs,
)


def _make_small_dataframe() -> pd.DataFrame:
    """创建细胞连续、基因比例不同且坐标非共线的确定性测试数据。"""

    rows = []
    gene_patterns = [
        ["G1", "G1", "G1", "G2", "G2", "G3", "G3", "G4"],
        ["G1", "G1", "G2", "G2", "G2", "G3", "G4", "G4"],
        ["G1", "G2", "G2", "G3", "G3", "G3", "G4", "G4"],
        ["G1", "G1", "G2", "G3", "G4", "G4", "G4", "G4"],
    ]
    offsets = [(0.0, 0.0), (20.0, 0.0), (0.0, 20.0), (20.0, 20.0)]
    local_xy = [(0, 0), (1, 0), (0, 1), (1, 1), (2, 0), (0, 2), (2, 2), (1, 2)]
    for cell_index, (genes, offset) in enumerate(zip(gene_patterns, offsets)):
        for point_index, (gene, local_point) in enumerate(zip(genes, local_xy)):
            rows.append({
                "cell": f"cell_{cell_index}",
                "gene": gene,
                "x": offset[0] + local_point[0],
                "y": offset[1] + local_point[1],
                "batch": "1",
                "tile": "tile_a" if cell_index < 2 else "tile_b",
                "fov": "fov_1" if cell_index < 2 else "fov_2",
                "z": point_index % 3,
                "umi": 1,
                "point_order": point_index,
            })
    df = pd.DataFrame(rows)
    unique_genes = pd.unique(df["gene"])
    df["gene"] = pd.Categorical(df["gene"], categories=unique_genes, ordered=True)
    df["cell"] = df["cell"].astype("category")
    df["batch"] = df["batch"].astype("category")
    df["tile"] = df["tile"].astype("category")
    df["fov"] = df["fov"].astype("category")
    return df


def _naive_same_cell_rnaflux(df: pd.DataFrame, sample_positions: np.ndarray, radius: float) -> np.ndarray:
    """逐查询点执行KDTree和bincount，作为不做性能优化的细胞内参考实现。"""

    cell_codes = df["cell"].cat.codes.to_numpy(copy=False)
    gene_codes = df["gene"].cat.codes.to_numpy(copy=False)
    num_genes = len(df["gene"].cat.categories)
    cell_profiles = []
    cell_ranges = {}
    for start, stop, cell_code in _iter_contiguous_cell_blocks(cell_codes):
        counts = np.bincount(gene_codes[start:stop], minlength=num_genes).astype(np.float64)
        cell_profiles.append(counts / counts.sum())
        cell_ranges[cell_code] = (start, stop)
    cell_profiles = np.asarray(cell_profiles)
    cell_std = np.std(cell_profiles, axis=0, ddof=1)
    output = np.empty((len(sample_positions), num_genes), dtype=np.float64)
    x_values = df["x"].to_numpy(copy=False)
    y_values = df["y"].to_numpy(copy=False)
    for query_index, position in enumerate(sample_positions):
        cell_code = int(cell_codes[position])
        start, stop = cell_ranges[cell_code]
        transcript_xy = np.column_stack((x_values[start:stop], y_values[start:stop]))
        neighbors = cKDTree(transcript_xy).query_ball_point([x_values[position], y_values[position]], radius)
        local_counts = np.bincount(gene_codes[start:stop][neighbors], minlength=num_genes).astype(np.float32)
        local_profile = local_counts / local_counts.sum(dtype=np.float32)
        output[query_index] = (local_profile.astype(np.float64) - cell_profiles[cell_code]) / cell_std
    return output


class OptimizedCompartmentEquivalenceTests(unittest.TestCase):
    """覆盖抽样、RNAflux、插值、SOM和小型端到端输出。"""

    def test_sampling_matches_pandas(self) -> None:
        """确认轻量位置抽样与Pandas原始抽样顺序完全一致。"""

        df = pd.DataFrame({"value": np.arange(100)})
        expected = df.sample(frac=0.23, random_state=42).index.to_numpy()
        np.testing.assert_array_equal(_sample_positions(len(df), 0.23, 42), expected)

    def test_same_cell_rnaflux_matches_naive_reference(self) -> None:
        """确认批量稀疏邻接计数与逐查询点细胞内参考RNAflux一致。"""

        df = _make_small_dataframe()
        sample_positions = np.array([0, 3, 8, 13, 17, 22, 25, 31], dtype=np.int64)
        expected = _naive_same_cell_rnaflux(df, sample_positions, radius=1.5)
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "rnaflux.npy"
            compute_same_cell_rnaflux(df, sample_positions, 1.5, output_path, workers=2, cell_batch_size=2)
            actual = np.load(output_path)
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)

    def test_same_cell_interpolation_matches_channel_griddata(self) -> None:
        """确认复用一次三角剖分的三通道插值与旧逐通道griddata一致。"""

        df = _make_small_dataframe()
        sample_positions = np.array([0, 1, 2, 8, 9, 10, 16, 17, 18, 24, 25, 26], dtype=np.int64)
        query_cell_codes = df["cell"].cat.codes.to_numpy()[sample_positions]
        query_rgb = np.linspace(0.1, 0.9, len(sample_positions) * 3).reshape(len(sample_positions), 3)
        expected = np.empty((len(df), 3), dtype=np.float64)
        cell_codes = df["cell"].cat.codes.to_numpy()
        xy = df[["x", "y"]].to_numpy()
        for start, stop, cell_code in _iter_contiguous_cell_blocks(cell_codes):
            query_ids = np.flatnonzero(query_cell_codes == cell_code)
            source_xy = xy[sample_positions[query_ids]]
            target_xy = xy[start:stop]
            for channel in range(3):
                linear = griddata(source_xy, query_rgb[query_ids, channel], target_xy, method="linear")
                nearest = griddata(source_xy, query_rgb[query_ids, channel], target_xy, method="nearest")
                expected[start:stop, channel] = np.where(np.isnan(linear), nearest, linear)
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "rgb.npy"
            actual = interpolate_same_cell_rgb(df, sample_positions, query_cell_codes, query_rgb, output_path, workers=2, cell_batch_size=2)
            np.testing.assert_allclose(np.asarray(actual), expected, rtol=1e-12, atol=1e-12)

    def test_chunked_som_quantization_matches_minisom(self) -> None:
        """确认分块量化误差与MiniSom原始全量实现一致。"""

        random_state = np.random.RandomState(7)
        rgb = random_state.random_sample((103, 3))
        som = MiniSom(1, 3, 3, random_seed=42)
        som.train(rgb, 20)
        expected = som.quantization_error(rgb)
        actual = _som_quantization_error_chunked(som, rgb, chunk_size=17)
        self.assertAlmostEqual(actual, expected, places=15)

    def test_chunked_som_winners_match_minisom(self) -> None:
        """确认向量化分块winner与逐点MiniSom winner标签完全一致。"""

        random_state = np.random.RandomState(11)
        rgb = random_state.random_sample((109, 3))
        som = MiniSom(1, 4, 3, random_seed=42)
        som.train(rgb, 20)
        expected = np.array([som.winner(point)[1] for point in rgb], dtype=np.int32)
        with tempfile.TemporaryDirectory() as temp_dir:
            actual = _som_winners_chunked(som, rgb, Path(temp_dir) / "labels.npy", chunk_size=13)
            np.testing.assert_array_equal(np.asarray(actual), expected)

    def test_small_end_to_end_pipeline(self) -> None:
        """确认Dataset4式元数据结构可从coordinates映射并按tile输出每个点。"""

        raw_df = _make_small_dataframe().copy()
        coordinates = raw_df[["cell", "batch", "tile", "fov"]].drop_duplicates("cell").copy()
        cell_boundary = {}
        nuclear_boundary = {}
        for cell_id, cell_points in raw_df.groupby("cell", observed=True):
            x_min, x_max = cell_points["x"].min() - 0.2, cell_points["x"].max() + 0.2
            y_min, y_max = cell_points["y"].min() - 0.2, cell_points["y"].max() + 0.2
            cell_boundary[str(cell_id)] = pd.DataFrame({"x": [x_min, x_max, x_max, x_min], "y": [y_min, y_min, y_max, y_max]})
            nuclear_boundary[str(cell_id)] = pd.DataFrame({"x": [x_min + 0.1, x_max - 0.1, x_max - 0.1, x_min + 0.1], "y": [y_min + 0.1, y_min + 0.1, y_max - 0.1, y_max - 0.1]})
        raw_df = raw_df.drop(columns=["batch", "tile", "fov"])
        for column in ["cell", "gene", "batch", "tile", "fov"]:
            if column in raw_df.columns:
                raw_df[column] = raw_df[column].astype(str)
            if column in coordinates.columns:
                coordinates[column] = coordinates[column].astype(str)
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            pkl_path = temp_path / "small.pkl"
            with pkl_path.open("wb") as output_file:
                pickle.dump({"data_df": raw_df, "coordinates": coordinates, "cell_boundary": cell_boundary, "nuclear_boundary": nuclear_boundary}, output_file)
            config = OptimizedCompartmentConfig(frac=0.5, radius=1.5, n_clusters="2", som_iterations=20, som_chunk_size=7, output_chunk_size=5, workers=2, cell_batch_size=2, output_workers=2)
            metadata = run_optimized_compartment(pkl_path, temp_path / "run", display_col="tile", config=config)
            tile_tables = [pd.read_csv(path) for path in metadata["output_paths"].values()]
            output_rows = sum(len(tile_table) for tile_table in tile_tables)
            self.assertEqual(output_rows, len(raw_df))
            self.assertEqual(metadata["best_k"], 2)
            self.assertTrue(metadata["same_cell_neighborhood"])
            self.assertTrue(metadata["same_cell_interpolation"])
            expected_columns = ["cell", "gene", "x", "y", "z", "umi", "fov", "tile", "subdomain", "rnaflux_r", "rnaflux_g", "rnaflux_b"]
            self.assertEqual(tile_tables[0].columns.tolist(), expected_columns)
            global_labels = np.load(temp_path / "run" / "work" / "subdomain_labels.npy")
            all_rgb = np.load(temp_path / "run" / "work" / "all_point_rgb.npy")
            for tile_label, output_path in metadata["output_paths"].items():
                tile_table = pd.read_csv(output_path)
                expected_rows = coordinates.loc[coordinates["tile"] == tile_label, "cell"]
                row_mask = raw_df["cell"].isin(expected_rows).to_numpy()
                np.testing.assert_allclose(tile_table["x"], raw_df.loc[row_mask, "x"])
                np.testing.assert_allclose(tile_table["y"], raw_df.loc[row_mask, "y"])
                np.testing.assert_array_equal(tile_table["subdomain"], global_labels[row_mask])
                np.testing.assert_allclose(tile_table[["rnaflux_r", "rnaflux_g", "rnaflux_b"]], all_rgb[row_mask])
                embedding_path = Path(metadata["plot_paths"][tile_label]["embedding_png"])
                subdomains_path = Path(metadata["plot_paths"][tile_label]["subdomains_png"])
                self.assertEqual(embedding_path.name, f"{tile_label}_sampled_rnaflux_embedding.png")
                self.assertEqual(subdomains_path.name, f"{tile_label}_sampled_subdomains.png")
                self.assertGreater(embedding_path.stat().st_size, 0)
                self.assertGreater(subdomains_path.stat().st_size, 0)

    def test_noncontiguous_cells_use_stable_disk_index(self) -> None:
        """确认Dataset4式交错细胞可按cell聚合且保持每个细胞内部原始行顺序。"""

        cell_codes = np.array([0, 1, 0, 2, 1, 0], dtype=np.int32)
        with tempfile.TemporaryDirectory() as temp_dir:
            cell_index = build_cell_row_index(cell_codes, Path(temp_dir))
            row_order = np.load(cell_index.row_order_path)
            offsets = np.load(cell_index.offsets_path)
        np.testing.assert_array_equal(row_order, np.array([0, 2, 5, 1, 4, 3], dtype=np.uint32))
        np.testing.assert_array_equal(offsets, np.array([0, 3, 5, 6], dtype=np.int64))

    def test_empty_tile_categories_are_not_reported(self) -> None:
        """确认筛选后保留的空 tile 分类不会创建文件或进入输出元数据。"""

        dataframe = pd.DataFrame({
            "cell": pd.Categorical(["cell_1", "cell_1"]),
            "gene": pd.Categorical(["G1", "G2"]),
            "x": [0.0, 1.0],
            "y": [0.0, 1.0],
            "tile": pd.Categorical(["tile_a", "tile_a"], categories=["tile_a", "tile_b"]),
        })
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            labels_path = temp_path / "labels.npy"
            rgb_path = temp_path / "rgb.npy"
            np.save(labels_path, np.array([0, 1], dtype=np.int32))
            np.save(rgb_path, np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float64))
            output_paths = write_tile_coordinate_outputs(dataframe, labels_path, rgb_path, temp_path / "tiles", "tile", 1)
            self.assertEqual(list(output_paths), ["tile_a"])
            self.assertFalse((temp_path / "tiles" / "subdomains_tile_b.npy").exists())

    def test_shuffled_same_cell_rnaflux_matches_naive_reference(self) -> None:
        """确认细胞行交错后，磁盘索引RNAflux仍与按cell筛选的慢速参考一致。"""

        contiguous_df = _make_small_dataframe()
        shuffled_df = contiguous_df.sample(frac=1.0, random_state=9).reset_index(drop=True)
        sample_positions = np.array([0, 2, 5, 9, 14, 18, 23, 29], dtype=np.int64)
        cell_codes = shuffled_df["cell"].cat.codes.to_numpy(copy=False)
        gene_codes = shuffled_df["gene"].cat.codes.to_numpy(copy=False)
        num_genes = len(shuffled_df["gene"].cat.categories)
        profiles = []
        for cell_code in range(len(shuffled_df["cell"].cat.categories)):
            rows = np.flatnonzero(cell_codes == cell_code)
            counts = np.bincount(gene_codes[rows], minlength=num_genes).astype(np.float64)
            profiles.append(counts / counts.sum())
        profiles = np.asarray(profiles)
        cell_std = profiles.std(axis=0, ddof=1)
        xy = shuffled_df[["x", "y"]].to_numpy()
        expected = np.empty((len(sample_positions), num_genes), dtype=np.float64)
        for query_index, position in enumerate(sample_positions):
            cell_code = int(cell_codes[position])
            rows = np.flatnonzero(cell_codes == cell_code)
            neighbors = cKDTree(xy[rows]).query_ball_point(xy[position], 1.5)
            counts = np.bincount(gene_codes[rows][neighbors], minlength=num_genes).astype(np.float32)
            local_profile = counts / counts.sum(dtype=np.float32)
            expected[query_index] = (local_profile.astype(np.float64) - profiles[cell_code]) / cell_std
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "rnaflux.npy"
            compute_same_cell_rnaflux(shuffled_df, sample_positions, 1.5, output_path)
            actual = np.load(output_path)
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
