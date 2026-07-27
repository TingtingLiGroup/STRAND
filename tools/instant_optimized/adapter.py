from __future__ import annotations

import pandas as pd

from tools.engines.instant_adapter import (
    _build_significant_pairs_table,
    bundle_to_instant_dataframe,
)
from tools.instant_optimized.core import (
    CPBResults,
    SparsePPResults,
    run_conditional_global_colocalization,
    run_proximal_pairs,
)


def _add_pair_cell_stats(
    pairs: pd.DataFrame,
    *,
    cpb_results: CPBResults,
    genes: list[str],
) -> pd.DataFrame:
    out = pairs.copy()
    if out.empty:
        out["n_cells_pp_significant"] = []
        return out

    gene_to_index = {gene: index for index, gene in enumerate(genes)}
    out["n_cells_pp_significant"] = [
        int(
            cpb_results.observed_significant_cells[
                gene_to_index[str(row["gene_1"])],
                gene_to_index[str(row["gene_2"])],
            ]
        )
        for _, row in out.iterrows()
    ]
    return out


def run_optimized_colocalization_from_bundle(
    bundle: dict,
    *,
    distance_threshold: float = 4.0,
    pp_alpha: float = 0.001,
    cpb_alpha: float = 0.0001,
    min_genecount: int = 20,
    threads: int = 1,
    use_3d: bool = True,
    precision_mode: str = "high",
    profile: bool = False,
    cell_id_col: str = "cell",
    gene_col: str = "gene",
    x_col: str = "x",
    y_col: str = "y",
    z_col: str = "z",
) -> dict:
    """Run isolated sparse PP and CPB on an already-prefiltered bundle.

    Core thresholds, dimensionality, precision and worker count match the
    original adapter. The returned dictionary keeps the original matrix/output
    keys and additionally exposes ``pp_results`` and ``cpb_results`` for
    sparse access without materializing the dense PP tensor.

    Parameters
    ----------
    bundle
        Standard toolbox bundle containing ``data_df``.
    distance_threshold, pp_alpha, cpb_alpha, min_genecount
        Original InSTAnT PP/CPB thresholds.
    threads
        Number of PP and CPB worker processes.
    use_3d
        Use XYZ distances when true, otherwise XY.
    precision_mode
        ``high`` for float64 PP values or ``low`` for float16.
    profile
        Print core timing and storage summaries.

    Returns
    -------
    dict
        Gene list, converted input, sparse PP result, gene counts, CPB
        matrices and the significant-pair table.
    """
    instant_input_df = bundle_to_instant_dataframe(
        bundle,
        cell_id_col=cell_id_col,
        gene_col=gene_col,
        x_col=x_col,
        y_col=y_col,
        z_col=z_col,
        use_3d=use_3d,
    )
    pp_results = run_proximal_pairs(
        instant_input_df,
        distance_threshold=distance_threshold,
        min_genecount=min_genecount,
        threads=threads,
        use_3d=use_3d,
        precision_mode=precision_mode,
        profile=profile,
    )
    cpb_results = run_conditional_global_colocalization(
        pp_results,
        alpha_cellwise=pp_alpha,
        min_transcript=0,
        threads=threads,
        precision_mode=precision_mode,
        profile=profile,
    )
    genes = list(pp_results.genes)
    significant_pairs = _build_significant_pairs_table(
        cpb_pvals=cpb_results.cpb_pvalues,
        expected_coloc=cpb_results.expected_colocalization,
        genes=genes,
        cpb_alpha=cpb_alpha,
        distance_threshold=distance_threshold,
        pp_alpha=pp_alpha,
    )
    significant_pairs = _add_pair_cell_stats(
        significant_pairs,
        cpb_results=cpb_results,
        genes=genes,
    )

    return {
        "gene_list": genes,
        "instant_input_df": instant_input_df,
        "pp_results": pp_results,
        "pp_pvals": pp_results,
        "gene_counts": pp_results.gene_counts.astype(
            pp_results.dtype,
            copy=False,
        ),
        "cpb_pvals": cpb_results.cpb_pvalues,
        "expected_coloc": cpb_results.expected_colocalization,
        "significant_pairs": significant_pairs,
        "cpb_results": cpb_results,
    }
