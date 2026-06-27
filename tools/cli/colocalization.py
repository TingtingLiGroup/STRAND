# tools/cli/colocalization.py

from __future__ import annotations

import argparse
from pathlib import Path

from tools.api.colocalization import run_colocalization_from_pkl


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="subcellfeat-coloc",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "Run InSTAnT colocalization analysis from a standard PKL bundle. "
            "By default, the command applies the toolbox prefilter, runs PP/CPB, "
            "saves significant global d-colocalized gene pairs, writes a compact "
            "prefilter summary, and generates standard visualizations."
        ),
    )

    parser.add_argument("--pkl", required=True, help="Path to PKL bundle containing data_df/cell_boundary/nuclear_boundary.")
    parser.add_argument("--out-dir", required=True, help="Output directory for colocalization results.")

    # InSTAnT core parameters.
    parser.add_argument("--distance", type=float, default=4.0, help="InSTAnT distance threshold d.")
    parser.add_argument("--pp-alpha", type=float, default=0.001, help="Cell-wise PP p-value threshold used by CPB.")
    parser.add_argument("--cpb-alpha", type=float, default=0.0001, help="Global CPB p-value threshold for significant gene pairs.")
    parser.add_argument("--min-genecount", type=int, default=20, help="InSTAnT internal minimum total transcript count per cell for PP.")
    parser.add_argument("--threads", type=int, default=8, help="Threads/processes for InSTAnT.")
    parser.add_argument(
        "--use-2d",
        action="store_true",
        help="Run InSTAnT 2D PP test instead of the default 3D PP test.",
    )
    parser.add_argument("--precision-mode", choices=["high", "low"], default="high", help="CPB precision mode.")

    # Runtime / debug.
    parser.add_argument("--profile", action="store_true", help="Print timing and prefilter statistics.")
    parser.add_argument("--max-cells", type=int, default=None, help="Limit number of cells after prefiltering for smoke test.")
    parser.add_argument("--keep-tmp", action="store_true", help="Keep temporary InSTAnT input directory.")

    # Prefilter.
    parser.add_argument("--no-prefilter", action="store_true", help="Disable toolbox standard prefilter. Useful for paper-like InSTAnT reproduction.")
    parser.add_argument("--nc-ratio-q-low", type=float, default=0.025, help="Lower quantile for nc ratio filtering.")
    parser.add_argument("--nc-ratio-q-high", type=float, default=0.975, help="Upper quantile for nc ratio filtering.")
    parser.add_argument("--nc-ratio-mean-low", type=float, default=0.4, help="Target lower bound for filtered nc ratio mean.")
    parser.add_argument("--nc-ratio-mean-high", type=float, default=0.6, help="Target upper bound for filtered nc ratio mean.")
    parser.add_argument("--cellgene-filter-min-transcripts", type=int, default=6, help="Minimum transcript count for each cell-gene sample.")
    parser.add_argument("--gene-filter-min-cells", type=int, default=10, help="Minimum cells required for each gene after cell-gene filtering.")

    # Output control.
    parser.add_argument("--save-matrices", action="store_true", help="Save full matrix outputs: CPB p-values, expected colocalization, and gene counts.")
    parser.add_argument("--save-pp-pvals", action="store_true", help="Save full PP p-value tensor. This can be very large; implies --save-matrices.")
    parser.add_argument("--save-intermediate", action="store_true", help="Save intermediate InSTAnT tables: instant_input_after_prefilter.csv, instant_gene_list.csv, and instant_all_pairs.csv.")
    parser.add_argument("--save-qc-details", action="store_true", help="Save detailed prefilter tables: nc ratio, cell-gene counts, and gene support tables.")

    # Visualization.
    parser.add_argument("--no-plot", action="store_true", help="Do not generate standard colocalization visualizations.")
    parser.add_argument("--viz-top-n-genes", type=int, default=80, help="Top N genes shown in CPB heatmap.")
    parser.add_argument("--viz-max-edges", type=int, default=80, help="Maximum significant edges shown in network plot.")
    parser.add_argument("--viz-max-cells", type=int, default=6, help="Maximum cells shown in each top-pair spatial plot.")
    parser.add_argument("--viz-top-pairs", type=int, default=10, help="Number of top significant gene pairs to visualize spatially.")
    parser.add_argument(
        "--viz-cell-selection",
        choices=["pp", "balanced", "expression"],
        default="pp",
        help=(
            "How to choose representative cells for top-pair spatial plots. "
            "'pp' selects cells with lowest cell-wise PP p-value; "
            "'balanced' selects cells with high reciprocal neighbor coverage; "
            "'expression' uses the old total-expression ranking."
        ),
    )
    parser.add_argument("--plot-gene-1", default=None, help="Gene 1 for spatial pair plot. Default: top significant pair.")
    parser.add_argument("--plot-gene-2", default=None, help="Gene 2 for spatial pair plot. Default: top significant pair.")
    parser.add_argument(
        "--no-region-annotation",
        action="store_true",
        help="Disable default four-region annotation of significant colocalized gene pairs.",
    )
    parser.add_argument(
        "--perinuclear-width",
        type=float,
        default=None,
        help=(
            "Deprecated/ignored for region annotation. The streaming annotator uses "
            "InSTAnT source defaults: nucleus=2.5, cyto_nuclear=2.5, cyto_peri=4.0."
        ),
    )
    parser.add_argument(
        "--peripheral-width",
        type=float,
        default=None,
        help=(
            "Deprecated/ignored for region annotation. The streaming annotator uses "
            "InSTAnT source defaults: nucleus=2.5, cyto_nuclear=2.5, cyto_peri=4.0."
        ),
    )
    parser.add_argument(
        "--region-max-pairs",
        type=int,
        default=None,
        help="Annotate only the first N significant pairs. Default: all significant pairs.",
    )

    parser.add_argument(
        "--region-threshold-mode",
        choices=["absolute", "cell_scaled"],
        default="cell_scaled",
        help=(
            "Region threshold mode for enrichment-based region annotation. "
            "Default 'cell_scaled' scales InSTAnT source thresholds by each cell/nucleus size; "
            "'absolute' uses fixed InSTAnT source defaults 2.5/2.5/4.0."
        ),
    )
    parser.add_argument(
        "--region-workers",
        type=int,
        default=1,
        help=(
            "Workers for region annotation only. Default 1 is safest and avoids extra processes. "
            "This does not affect InSTAnT PP/CPB --threads."
        ),
    )

    parser.add_argument(
        "--sample-cells",
        type=int,
        default=None,
        help=(
            "Optional number of cells to randomly sample before InSTAnT PP/CPB. "
            "This keeps the full gene panel but reduces the cell x gene x gene memory cost."
        ),
    )
    parser.add_argument(
        "--sample-cell-frac",
        type=float,
        default=None,
        help=(
            "Optional fraction of cells to randomly sample before InSTAnT PP/CPB. "
            "Use either --sample-cells or --sample-cell-frac, not both."
        ),
    )
    parser.add_argument(
        "--sample-random-state",
        type=int,
        default=42,
        help="Random seed for reproducible cell sampling.",
    )
    parser.add_argument(
        "--sample-stratify-by",
        default=None,
        help=(
            "Optional column for stratified cell sampling, e.g. fov or batch. "
            "The column may be in data_df or in coordinates and will be merged by cell."
        ),
    )

    parser.add_argument(
        "--groupby",
        default=None,
        help=(
            "Run colocalization separately for each group. The column may be in data_df "
            "or in coordinates and will be merged by cell, e.g. --groupby celltype."
        ),
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    do_prefilter = not args.no_prefilter
    prefilter_kwargs = {
        "filter_cells_by_nc_ratio": True,
        "nc_ratio_q_low": args.nc_ratio_q_low,
        "nc_ratio_q_high": args.nc_ratio_q_high,
        "nc_ratio_mean_low": args.nc_ratio_mean_low,
        "nc_ratio_mean_high": args.nc_ratio_mean_high,
        "cellgene_filter_min_transcripts": args.cellgene_filter_min_transcripts,
        "gene_filter_min_cells": args.gene_filter_min_cells,
    }

    save_matrices = bool(args.save_matrices or args.save_pp_pvals)

    # Auto-detect 2D/3D: if user didn't pass --use-2d but PKL has no z column, auto-switch
    use_3d = not args.use_2d
    if use_3d:
        import pickle
        with open(args.pkl, "rb") as _f:
            _bundle = pickle.load(_f)
        if "z" not in _bundle["data_df"].columns:
            print("[coloc] WARNING: z column not found in data_df, automatically switching to 2D mode.")
            print("[coloc] To suppress this warning, pass --use-2d explicitly.")
            use_3d = False
        del _bundle

    out = run_colocalization_from_pkl(
        pkl_path=args.pkl,
        out_dir=args.out_dir,
        distance_threshold=args.distance,
        pp_alpha=args.pp_alpha,
        cpb_alpha=args.cpb_alpha,
        min_genecount=args.min_genecount,
        threads=args.threads,
        use_3d=use_3d,
        precision_mode=args.precision_mode,
        profile=args.profile,
        max_cells=args.max_cells,
        do_prefilter=not args.no_prefilter,
        prefilter_kwargs=prefilter_kwargs,
        save_matrices=save_matrices,
        save_pp_pvals=args.save_pp_pvals,
        keep_tmp=args.keep_tmp,
        save_intermediate=args.save_intermediate,
        save_qc_details=args.save_qc_details,
        plot=not args.no_plot,
        viz_top_n_genes=args.viz_top_n_genes,
        viz_max_edges=args.viz_max_edges,
        viz_max_cells=args.viz_max_cells,
        viz_top_pairs=args.viz_top_pairs,
        viz_cell_selection=args.viz_cell_selection,
        plot_gene_1=args.plot_gene_1,
        plot_gene_2=args.plot_gene_2,
        region_annotation=not args.no_region_annotation,
        perinuclear_width=args.perinuclear_width,
        peripheral_width=args.peripheral_width,
        region_max_pairs=args.region_max_pairs,
        region_threshold_mode=args.region_threshold_mode,
        region_workers=args.region_workers,
        sample_cells=args.sample_cells,
        sample_cell_frac=args.sample_cell_frac,
        sample_random_state=args.sample_random_state,
        sample_stratify_by=args.sample_stratify_by,
        groupby=args.groupby,
    )

    if out.get("groupby") is not None:
        print(f"[subcellfeat-coloc] saved groupby summary: {out['groupby_summary_path']}")
        return 0

    print(f"[subcellfeat-coloc] saved significant pairs: {out['pairs_path']}")
    print(f"[subcellfeat-coloc] saved prefilter summary: {out['prefilter_summary_path']}")
    if out.get("region_annotated_pairs_path") is not None:
        print(
            "[subcellfeat-coloc] saved region annotated pairs: "
            f"{out['region_annotated_pairs_path']}"
        )
    if out.get("cell_sampling_summary_path") is not None:
        print(f"[subcellfeat-coloc] saved cell sampling summary: {out['cell_sampling_summary_path']}")
    if out.get("sampled_cells_path") is not None:
        print(f"[subcellfeat-coloc] saved sampled cells: {out['sampled_cells_path']}")
    if out.get("gene_list_path") is not None:
        print(f"[subcellfeat-coloc] saved gene list: {out['gene_list_path']}")
    if out.get("instant_input_path") is not None:
        print(f"[subcellfeat-coloc] saved converted InSTAnT input: {out['instant_input_path']}")
    if out.get("viz_paths"):
        print("[subcellfeat-coloc] visualization outputs:")
        for name, path in out["viz_paths"].items():
            print(f"  - {name}: {path}")
    print("[subcellfeat-coloc] top significant pairs:")
    print(out["significant_pairs"].head())
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
