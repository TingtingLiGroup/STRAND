from __future__ import annotations

import argparse
from pathlib import Path

from tools.api.coloc_plotting import run_coloc_plots_from_outdir


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog='subcellfeat-coloc-plot',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            'Generate colocalization visualizations from an existing output directory '
            'without rerunning PP/CPB or region annotation.'
        ),
    )
    p.add_argument('--out-dir', required=True, help='Existing colocalization output directory.')
    p.add_argument('--pkl', default=None, help='Original PKL bundle path. Required for top-pair spatial plots.')

    dim = p.add_mutually_exclusive_group()
    dim.add_argument('--use-2d', action='store_true', help='Use 2D coordinates for top-pair plots.')
    dim.add_argument('--use-3d', action='store_true', help='Use 3D coordinates for top-pair plots when absZ is present.')

    p.add_argument('--no-heatmap', action='store_true', help='Skip heatmap generation.')
    p.add_argument('--no-network', action='store_true', help='Skip network generation.')
    p.add_argument('--no-top-pairs', action='store_true', help='Skip top-pair spatial plots.')

    p.add_argument('--viz-top-n-genes', type=int, default=80, help='Heatmap parameter. Keep original heatmap logic unchanged.')
    p.add_argument('--viz-max-edges', type=int, default=80, help='Maximum number of edges shown in the network.')
    p.add_argument('--viz-max-cells', type=int, default=6, help='Maximum number of cells shown for each top-pair spatial plot.')
    p.add_argument('--viz-top-pairs', type=int, default=10, help='Number of top significant pairs to plot spatially.')
    p.add_argument('--plot-gene-1', default=None, help='Optional specific first gene for one spatial plot.')
    p.add_argument('--plot-gene-2', default=None, help='Optional specific second gene for one spatial plot.')
    p.add_argument('--distance', type=float, default=None, help='Override distance threshold for spatial plots. Default: read from pair file.')
    p.add_argument('--cell-selection', choices=['pp', 'balanced', 'expression'], default='pp', help='Cell selection rule for top-pair spatial plots.')
    p.add_argument('--save-all-pairs', action='store_true', help='Also export instant_all_pairs.csv when matrices are available.')
    return p


def main() -> None:
    args = build_parser().parse_args()
    use_3d = bool(args.use_3d)

    res = run_coloc_plots_from_outdir(
        out_dir=args.out_dir,
        pkl_path=args.pkl,
        use_3d=use_3d,
        plot_heatmap=not args.no_heatmap,
        plot_network=not args.no_network,
        plot_top_pairs=not args.no_top_pairs,
        top_n_genes=args.viz_top_n_genes,
        max_edges=args.viz_max_edges,
        max_cells=args.viz_max_cells,
        top_pairs=args.viz_top_pairs,
        gene_1=args.plot_gene_1,
        gene_2=args.plot_gene_2,
        cell_selection=args.cell_selection,
        distance_threshold=args.distance,
        save_all_pairs=args.save_all_pairs,
    )

    print('[subcellfeat-coloc-plot] output dir:', res['out_dir'])
    for name, value in res.get('viz_paths', {}).items():
        print(f'[subcellfeat-coloc-plot] saved {name}: {value}')


if __name__ == '__main__':
    main()
