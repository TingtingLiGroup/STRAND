from __future__ import annotations

from pathlib import Path

import pandas as pd

from tools.engines.coloc_visualization import (
    plot_coloc_outputs,
    plot_top_pair_cells,
)
from tools.instant_optimized.core import SparsePPResults


def plot_optimized_coloc_outputs(
    *,
    out_dir: str | Path,
    pkl_path: str | Path | None,
    gene_list: list[str],
    instant_input_df: pd.DataFrame,
    significant_pairs: pd.DataFrame,
    region_annotated_pairs: pd.DataFrame | None,
    cpb_pvals,
    expected_coloc,
    pp_results: SparsePPResults,
    top_n_genes: int | None = 80,
    max_edges: int = 80,
    max_cells: int = 6,
    top_pairs: int = 10,
    gene_1: str | None = None,
    gene_2: str | None = None,
    distance_threshold: float = 4.0,
    use_3d: bool = False,
    save_all_pairs: bool = False,
    cell_selection: str = "pp",
) -> dict[str, object]:
    """Reuse legacy plots while providing PP values one selected pair at a time."""
    paths = plot_coloc_outputs(
        out_dir=out_dir,
        pkl_path=pkl_path,
        gene_list=gene_list,
        instant_input_df=instant_input_df,
        significant_pairs=significant_pairs,
        region_annotated_pairs=region_annotated_pairs,
        cpb_pvals=cpb_pvals,
        expected_coloc=expected_coloc,
        pp_pvals=None,
        top_n_genes=top_n_genes,
        max_edges=max_edges,
        max_cells=max_cells,
        top_pairs=top_pairs,
        gene_1=gene_1,
        gene_2=gene_2,
        distance_threshold=distance_threshold,
        use_3d=use_3d,
        save_all_pairs=save_all_pairs,
        cell_selection=cell_selection,
        make_top_pairs=False,
    )

    if significant_pairs is None or significant_pairs.empty:
        return paths

    if gene_1 is not None and gene_2 is not None:
        selected_pairs = [(1, str(gene_1), str(gene_2), distance_threshold)]
    else:
        sort_column = (
            "cpb_pvalue_raw"
            if "cpb_pvalue_raw" in significant_pairs.columns
            else "cpb_pvalue"
        )
        top = (
            significant_pairs.sort_values(sort_column, ascending=True)
            .head(int(top_pairs))
            .reset_index(drop=True)
        )
        selected_pairs = [
            (
                rank,
                str(row["gene_1"]),
                str(row["gene_2"]),
                float(row.get("distance_threshold", distance_threshold)),
            )
            for rank, (_, row) in enumerate(top.iterrows(), start=1)
        ]

    top_paths = []
    for rank, selected_gene_1, selected_gene_2, selected_distance in selected_pairs:
        pair_pp_view = (
            pp_results.broadcast_values_for_pair(
                selected_gene_1,
                selected_gene_2,
            )
            if cell_selection == "pp"
            else None
        )
        path = plot_top_pair_cells(
            out_dir,
            pkl_path=pkl_path,
            instant_input_df=instant_input_df,
            significant_pairs=significant_pairs,
            gene_list=gene_list,
            pp_pvals=pair_pp_view,
            gene_1=selected_gene_1,
            gene_2=selected_gene_2,
            max_cells=max_cells,
            distance_threshold=selected_distance,
            use_3d=use_3d,
            cell_selection=cell_selection,
            rank=rank,
        )
        if path is not None:
            top_paths.append(str(path))

    if top_paths:
        paths["top_pair_cells"] = top_paths
    return paths
