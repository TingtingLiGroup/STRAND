from __future__ import annotations

import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from tools.api.colocalization import (
    run_colocalization_from_pkl as run_original,
)
from tools.instant_optimized.api import (
    run_colocalization_from_pkl as run_optimized,
)


def _bundle() -> dict:
    rows = []
    for group, cells in {
        "type-a": ["a1", "a2"],
        "type-b": ["b1", "b2"],
    }.items():
        for cell_index, cell in enumerate(cells):
            for gene, points in {
                "A": [(0.0, 0.0), (0.3, 0.0)],
                "B": [(0.6, 0.0), (2.0, 2.0)],
            }.items():
                for x, y in points:
                    rows.append(
                        {
                            "cell": cell,
                            "gene": gene,
                            "x": x + cell_index * 5,
                            "y": y,
                            "z": 0.0,
                            "celltype": group,
                        }
                    )
    return {
        "data_df": pd.DataFrame(rows),
        "cell_boundary": {},
        "nuclear_boundary": {},
    }


class ApiCompatibilityTests(unittest.TestCase):
    def test_saved_matrices_match_original(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            pkl_path = root / "input.pkl"
            with pkl_path.open("wb") as handle:
                pickle.dump(_bundle(), handle)

            kwargs = {
                "pkl_path": pkl_path,
                "distance_threshold": 1.0,
                "pp_alpha": 0.2,
                "cpb_alpha": 0.5,
                "min_genecount": 2,
                "threads": 1,
                "use_3d": False,
                "precision_mode": "high",
                "do_prefilter": False,
                "save_matrices": True,
                "save_pp_pvals": True,
                "plot": False,
                "region_annotation": False,
            }
            original = run_original(out_dir=root / "original", **kwargs)
            optimized = run_optimized(out_dir=root / "optimized", **kwargs)

            for filename in (
                "instant_cpb_pvals.npy",
                "instant_expected_coloc.npy",
                "instant_gene_counts.npy",
                "instant_pp_pvals.npy",
            ):
                self.assertTrue(
                    np.array_equal(
                        np.load(root / "original" / filename),
                        np.load(root / "optimized" / filename),
                    ),
                    msg=filename,
                )

            original_pairs = pd.read_csv(original["pairs_path"])
            optimized_pairs = pd.read_csv(optimized["pairs_path"])
            self.assertTrue(original_pairs.equals(optimized_pairs))

    def test_groupby_and_sampling_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            pkl_path = root / "input.pkl"
            with pkl_path.open("wb") as handle:
                pickle.dump(_bundle(), handle)

            stale_temp_dir = root / "sampled" / "_tmp_instant_coloc"
            stale_temp_dir.mkdir(parents=True)
            (stale_temp_dir / "stale.txt").write_text("stale")
            sampled = run_optimized(
                pkl_path,
                out_dir=root / "sampled",
                do_prefilter=False,
                sample_cells=3,
                min_genecount=2,
                threads=1,
                use_3d=False,
                plot=False,
                region_annotation=False,
            )
            self.assertTrue(sampled["cell_sampling_summary_path"].exists())
            self.assertTrue(sampled["sampled_cells_path"].exists())
            self.assertFalse(stale_temp_dir.exists())

            grouped = run_optimized(
                pkl_path,
                out_dir=root / "grouped",
                do_prefilter=False,
                min_genecount=2,
                threads=2,
                use_3d=False,
                plot=False,
                region_annotation=False,
                groupby="celltype",
            )
            summary = pd.read_csv(grouped["groupby_summary_path"])
            self.assertEqual(set(summary["group"]), {"type-a", "type-b"})
            self.assertEqual(set(summary["status"]), {"done"})
            self.assertTrue(
                all(
                    (
                        root
                        / "grouped"
                        / group
                        / "instant_significant_pairs.csv"
                    ).exists()
                    for group in ("type-a", "type-b")
                )
            )

    def test_group_name_collision_is_rejected(self) -> None:
        bundle = _bundle()
        bundle["data_df"].loc[
            bundle["data_df"]["celltype"] == "type-a",
            "celltype",
        ] = "A/B"
        bundle["data_df"].loc[
            bundle["data_df"]["celltype"] == "type-b",
            "celltype",
        ] = "A B"

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            pkl_path = root / "input.pkl"
            with pkl_path.open("wb") as handle:
                pickle.dump(bundle, handle)
            with self.assertRaisesRegex(
                ValueError,
                "same output directory",
            ):
                run_optimized(
                    pkl_path,
                    out_dir=root / "grouped",
                    do_prefilter=False,
                    min_genecount=2,
                    threads=1,
                    use_3d=False,
                    plot=False,
                    region_annotation=False,
                    groupby="celltype",
                )


if __name__ == "__main__":
    unittest.main()
