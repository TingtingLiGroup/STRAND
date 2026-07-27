from __future__ import annotations

import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd

from tools.instant_core.InSTAnT import ConditionalGlobalColocalization, Instant
from tools.instant_optimized.core import (
    _poisson_binomial_right_tail_fft,
    run_conditional_global_colocalization,
    run_proximal_pairs,
)


def _small_input() -> pd.DataFrame:
    rows = []
    cell_layouts = {
        "cell-b": [
            ("A", 0.0, 0.0, 0.0),
            ("A", 0.5, 0.0, 0.2),
            ("B", 1.0, 0.0, 0.1),
            ("B", 4.0, 4.0, 1.0),
            ("C", 0.0, 2.0, 0.0),
        ],
        "cell-a": [
            ("A", 10.0, 0.0, 0.0),
            ("A", 10.2, 0.0, 0.1),
            ("B", 10.4, 0.0, 0.2),
            ("C", 13.0, 3.0, 1.0),
            ("C", 13.2, 3.0, 1.2),
        ],
        "cell-c": [
            ("A", 20.0, 0.0, 0.0),
            ("B", 20.0, 1.5, 0.0),
            ("B", 20.0, 3.0, 0.0),
        ],
    }
    for cell_id, transcripts in cell_layouts.items():
        for gene, x, y, z in transcripts:
            rows.append(
                {
                    "gene": gene,
                    "absX": x,
                    "absY": y,
                    "absZ": z,
                    "uID": cell_id,
                }
            )
    return pd.DataFrame(rows)


class EquivalenceTests(unittest.TestCase):
    def test_large_trial_poisson_binomial_tail_matches_binomial(self) -> None:
        """确认大 trial 路径与同概率 Binomial 右尾保持数值一致。"""
        from scipy.stats import binom

        probabilities = np.full(20_000, 0.002, dtype=np.float64)
        observed = 55
        optimized_tail = _poisson_binomial_right_tail_fft(
            probabilities,
            observed,
        )
        reference_tail = float(binom.sf(observed - 1, len(probabilities), 0.002))
        self.assertAlmostEqual(optimized_tail, reference_tail, places=10)

    def test_cpb_profile_reports_progress(self) -> None:
        """确认 profile 模式会报告 CPB 准备、索引和完成进度。"""
        optimized_pp = run_proximal_pairs(
            _small_input(),
            distance_threshold=1.5,
            min_genecount=2,
            threads=1,
            use_3d=False,
            precision_mode="high",
        )
        output_buffer = io.StringIO()

        with redirect_stdout(output_buffer):
            run_conditional_global_colocalization(
                optimized_pp,
                alpha_cellwise=0.2,
                threads=1,
                precision_mode="high",
                profile=True,
            )

        profile_output = output_buffer.getvalue()
        self.assertIn("CPB prepare", profile_output)
        self.assertIn("CPB index ready", profile_output)
        self.assertIn("CPB progress", profile_output)
        self.assertIn("100.0%", profile_output)

    def test_sparse_pp_matches_original(self) -> None:
        input_df = _small_input()
        for precision_mode in ("high", "low"):
            for use_3d in (False, True):
                with self.subTest(
                    precision_mode=precision_mode,
                    use_3d=use_3d,
                ), tempfile.TemporaryDirectory() as temp_dir:
                    input_path = Path(temp_dir) / "input.csv"
                    input_df.to_csv(input_path, index=False)

                    original = Instant(
                        distance_threshold=1.5,
                        threads=1,
                        precision_mode=precision_mode == "high",
                    )
                    original.load_preprocessed_data(
                        str(input_path),
                        force_csv=True,
                    )
                    if use_3d:
                        original.run_ProximalPairs3D(
                            distance_threshold=1.5,
                            min_genecount=3,
                        )
                    else:
                        original.run_ProximalPairs(
                            distance_threshold=1.5,
                            min_genecount=3,
                        )

                    optimized = run_proximal_pairs(
                        input_df,
                        distance_threshold=1.5,
                        min_genecount=3,
                        threads=1,
                        use_3d=use_3d,
                        precision_mode=precision_mode,
                    )

                    self.assertTrue(
                        np.array_equal(optimized.to_dense(), original.all_pval)
                    )
                    self.assertTrue(
                        np.array_equal(
                            optimized.gene_counts.astype(
                                original.all_gene_count.dtype
                            ),
                            original.all_gene_count,
                        )
                    )
                    self.assertEqual(
                        optimized.cell_ids.tolist(),
                        sorted(input_df["uID"].unique()),
                    )
                    self.assertEqual(
                        list(optimized.genes),
                        sorted(input_df["gene"].unique()),
                    )

    def test_sparse_cpb_matches_original(self) -> None:
        input_df = _small_input()
        for precision_mode in ("high", "low"):
            with self.subTest(
                precision_mode=precision_mode
            ), tempfile.TemporaryDirectory() as temp_dir:
                input_path = Path(temp_dir) / "input.csv"
                input_df.to_csv(input_path, index=False)

                original = Instant(
                    distance_threshold=1.5,
                    threads=1,
                    precision_mode=precision_mode == "high",
                )
                original.load_preprocessed_data(str(input_path), force_csv=True)
                original.run_ProximalPairs(
                    distance_threshold=1.5,
                    min_genecount=2,
                )

                original_cpb = ConditionalGlobalColocalization(
                    all_pvals=original.all_pval,
                    transcript_count=original.all_gene_count,
                    alpha_cellwise=0.2,
                    min_transcript=0,
                    threads=1,
                    precision_mode=precision_mode,
                )
                original_pvalues, original_expected = (
                    original_cpb.global_colocalization_serial()
                )

                optimized_pp = run_proximal_pairs(
                    input_df,
                    distance_threshold=1.5,
                    min_genecount=2,
                    threads=1,
                    use_3d=False,
                    precision_mode=precision_mode,
                )
                optimized_cpb = run_conditional_global_colocalization(
                    optimized_pp,
                    alpha_cellwise=0.2,
                    min_transcript=0,
                    threads=1,
                    precision_mode=precision_mode,
                )

                self.assertTrue(
                    np.array_equal(
                        optimized_cpb.observed_significant_cells,
                        original_cpb.obs,
                    )
                )
                self.assertTrue(
                    np.array_equal(
                        optimized_cpb.cpb_pvalues,
                        original_pvalues,
                    )
                )
                self.assertTrue(
                    np.array_equal(
                        optimized_cpb.expected_colocalization,
                        original_expected,
                    )
                )

    def test_parallel_pp_matches_serial(self) -> None:
        input_df = pd.concat([_small_input()] * 8, ignore_index=True)
        input_df["uID"] = [
            f"{cell}-{copy_index}"
            for copy_index in range(8)
            for cell in _small_input()["uID"]
        ]

        serial = run_proximal_pairs(
            input_df,
            distance_threshold=1.5,
            min_genecount=2,
            threads=1,
            use_3d=False,
            precision_mode="high",
        )
        parallel = run_proximal_pairs(
            input_df,
            distance_threshold=1.5,
            min_genecount=2,
            threads=2,
            use_3d=False,
            precision_mode="high",
        )

        self.assertTrue(np.array_equal(parallel.to_dense(), serial.to_dense()))
        self.assertTrue(
            np.array_equal(parallel.gene_counts, serial.gene_counts)
        )

    def test_nonzero_min_transcript_cpb_matches_original(self) -> None:
        input_df = _small_input()
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "input.csv"
            input_df.to_csv(input_path, index=False)
            original = Instant(
                distance_threshold=1.5,
                threads=1,
                precision_mode=True,
            )
            original.load_preprocessed_data(str(input_path), force_csv=True)
            original.run_ProximalPairs(
                distance_threshold=1.5,
                min_genecount=2,
            )
            original_cpb = ConditionalGlobalColocalization(
                all_pvals=original.all_pval,
                transcript_count=original.all_gene_count,
                alpha_cellwise=0.2,
                min_transcript=1,
                threads=1,
                precision_mode="high",
            )
            original_pvalues, original_expected = (
                original_cpb.global_colocalization_serial()
            )

            optimized_pp = run_proximal_pairs(
                input_df,
                distance_threshold=1.5,
                min_genecount=2,
                threads=1,
                use_3d=False,
                precision_mode="high",
            )
            optimized_cpb = run_conditional_global_colocalization(
                optimized_pp,
                alpha_cellwise=0.2,
                min_transcript=1,
                threads=1,
                precision_mode="high",
            )

            self.assertTrue(
                np.array_equal(optimized_cpb.cpb_pvalues, original_pvalues)
            )
            self.assertTrue(
                np.array_equal(
                    optimized_cpb.expected_colocalization,
                    original_expected,
                )
            )

    def test_dense_file_and_plot_view_are_compatible(self) -> None:
        optimized = run_proximal_pairs(
            _small_input(),
            distance_threshold=1.5,
            min_genecount=2,
            threads=1,
            use_3d=False,
            precision_mode="high",
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = optimized.save_dense_npy(
                Path(temp_dir) / "pp.npy"
            )
            saved = np.load(output_path, mmap_mode="r")

            self.assertTrue(np.array_equal(saved, optimized.to_dense()))
            pair_view = optimized.broadcast_values_for_pair("A", "B")
            gene_a = optimized.genes.index("A")
            gene_b = optimized.genes.index("B")
            self.assertEqual(pair_view.shape, saved.shape)
            self.assertTrue(
                np.array_equal(
                    pair_view[:, gene_a, gene_b],
                    saved[:, gene_a, gene_b],
                )
            )
            self.assertEqual(optimized.shape, saved.shape)
            self.assertTrue(
                np.array_equal(
                    optimized[:, gene_a, gene_b],
                    saved[:, gene_a, gene_b],
                )
            )


if __name__ == "__main__":
    unittest.main()
