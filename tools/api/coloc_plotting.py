from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np

from tools.engines.coloc_visualization import plot_coloc_outputs


def run_coloc_plots_from_outdir(
    *,
    out_dir: str | Path,
    pkl_path: str | Path | None = None,
    use_3d: bool = False,
    plot_heatmap: bool = True,
    plot_network: bool = True,
    plot_top_pairs: bool = True,
    top_n_genes: int | None = 80,
    max_edges: int = 80,
    max_cells: int = 6,
    top_pairs: int = 10,
    gene_1: str | None = None,
    gene_2: str | None = None,
    cell_selection: str = 'pp',
    distance_threshold: float | None = None,
    save_all_pairs: bool = False,
) -> dict:
    """Rebuild colocalization plots from existing output files only.

    This does not rerun PP/CPB or region annotation. It only reads files from
    an existing colocalization result directory.
    """
    out_dir = Path(out_dir)
    if not out_dir.exists():
        raise FileNotFoundError(f'Cannot find output directory: {out_dir}')

    pairs_path = out_dir / 'instant_significant_pairs.csv'
    if not pairs_path.exists():
        raise FileNotFoundError(f'Cannot find required file: {pairs_path}')
    significant_pairs = pd.read_csv(pairs_path)

    region_path = out_dir / 'instant_region_annotated_pairs.csv'
    region_df = pd.read_csv(region_path) if region_path.exists() else None

    gene_list_path = out_dir / 'instant_gene_list.csv'
    gene_list = pd.read_csv(gene_list_path)['gene'].astype(str).tolist() if gene_list_path.exists() else None

    instant_input_path = out_dir / 'instant_input_after_prefilter.csv'
    instant_input_df = pd.read_csv(instant_input_path) if instant_input_path.exists() else None

    cpb_pvals = None
    cpb_path = out_dir / 'instant_cpb_pvals.npy'
    if cpb_path.exists():
        cpb_pvals = np.load(cpb_path)

    expected_coloc = None
    expected_path = out_dir / 'instant_expected_coloc.npy'
    if expected_path.exists():
        expected_coloc = np.load(expected_path)

    pp_pvals = None
    pp_path = out_dir / 'instant_pp_pvals.npy'
    if pp_path.exists():
        pp_pvals = np.load(pp_path)

    if distance_threshold is None:
        if not significant_pairs.empty and 'distance_threshold' in significant_pairs.columns:
            distance_threshold = float(significant_pairs['distance_threshold'].iloc[0])
        else:
            distance_threshold = 4.0

    paths = {}

    # heatmap uses the original logic and still needs cpb matrix + gene list
    if plot_heatmap:
        if cpb_pvals is None or gene_list is None:
            raise FileNotFoundError(
                'Heatmap requires instant_cpb_pvals.npy and instant_gene_list.csv. '
                'These files are missing in the output directory.'
            )

    if save_all_pairs and (cpb_pvals is None or expected_coloc is None or gene_list is None):
        raise FileNotFoundError(
            'save_all_pairs requires instant_cpb_pvals.npy, instant_expected_coloc.npy, and instant_gene_list.csv.'
        )

    if plot_top_pairs and (pkl_path is None or instant_input_df is None):
        raise FileNotFoundError(
            'Top-pair spatial plots require both --pkl and instant_input_after_prefilter.csv.'
        )

    raw_paths = plot_coloc_outputs(
        out_dir=out_dir,
        pkl_path=pkl_path,
        gene_list=gene_list,
        instant_input_df=instant_input_df,
        significant_pairs=significant_pairs,
        region_annotated_pairs=region_df,
        cpb_pvals=cpb_pvals,
        expected_coloc=expected_coloc,
        pp_pvals=pp_pvals,
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
        make_heatmap=plot_heatmap,
        make_network=plot_network,
        make_top_pairs=plot_top_pairs,
    )

    return {
        'out_dir': str(out_dir),
        'pairs_path': str(pairs_path),
        'region_annotated_pairs_path': str(region_path) if region_path.exists() else None,
        'viz_paths': raw_paths,
    }
