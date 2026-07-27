from __future__ import annotations

import argparse
import resource
import time

import numpy as np
import pandas as pd

from tools.instant_optimized.core import (
    run_conditional_global_colocalization,
    run_proximal_pairs,
)


def _synthetic_input(
    *,
    num_cells: int,
    num_genes: int,
    active_genes_per_cell: int,
    transcripts_per_gene: int,
    random_state: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(int(random_state))
    gene_names = np.asarray([f"gene_{i:04d}" for i in range(num_genes)])
    rows = []

    for cell_index in range(num_cells):
        active_genes = rng.choice(
            gene_names,
            size=min(active_genes_per_cell, num_genes),
            replace=False,
        )
        center = rng.uniform(0, 10_000, size=2)
        for gene in active_genes:
            coordinates = center + rng.normal(
                0,
                3,
                size=(transcripts_per_gene, 2),
            )
            for x, y in coordinates:
                rows.append(
                    {
                        "gene": str(gene),
                        "absX": float(x),
                        "absY": float(y),
                        "uID": f"cell_{cell_index:06d}",
                    }
                )

    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark the isolated optimized PP/CPB implementation."
    )
    parser.add_argument("--cells", type=int, default=1_000)
    parser.add_argument("--genes", type=int, default=469)
    parser.add_argument("--active-genes-per-cell", type=int, default=4)
    parser.add_argument("--transcripts-per-gene", type=int, default=12)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    input_df = _synthetic_input(
        num_cells=args.cells,
        num_genes=args.genes,
        active_genes_per_cell=args.active_genes_per_cell,
        transcripts_per_gene=args.transcripts_per_gene,
        random_state=args.random_state,
    )
    started = time.perf_counter()
    pp_results = run_proximal_pairs(
        input_df,
        distance_threshold=4.0,
        min_genecount=20,
        threads=args.threads,
        use_3d=False,
        precision_mode="high",
        profile=True,
    )
    cpb_results = run_conditional_global_colocalization(
        pp_results,
        alpha_cellwise=0.001,
        threads=args.threads,
        precision_mode="high",
        profile=True,
    )
    elapsed = time.perf_counter() - started
    max_rss_gib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2

    print(
        "[instant-optimized benchmark] "
        f"cells={pp_results.num_cells}, genes={pp_results.num_genes}, "
        f"transcripts={len(input_df)}, stored_pair_values={len(pp_results.pvalues)}, "
        f"total_elapsed={elapsed:.2f}s, parent_max_rss={max_rss_gib:.3f} GiB, "
        f"finite_cpb={int(np.isfinite(cpb_results.cpb_pvalues).sum())}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
