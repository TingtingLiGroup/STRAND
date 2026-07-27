from __future__ import annotations

from tools.cli.colocalization import build_parser
from tools.instant_optimized.api import run_colocalization_from_pkl


def main() -> int:
    parser = build_parser()
    parser.prog = "python -m tools.instant_optimized.cli"
    parser.description = (
        "Run the isolated sparse, result-equivalent InSTAnT optimization. "
        "Use an output directory separate from the original pipeline."
    )
    args = parser.parse_args()
    prefilter_kwargs = {
        "filter_cells_by_nc_ratio": True,
        "nc_ratio_q_low": args.nc_ratio_q_low,
        "nc_ratio_q_high": args.nc_ratio_q_high,
        "nc_ratio_mean_low": args.nc_ratio_mean_low,
        "nc_ratio_mean_high": args.nc_ratio_mean_high,
        "cellgene_filter_min_transcripts": (
            args.cellgene_filter_min_transcripts
        ),
        "gene_filter_min_cells": args.gene_filter_min_cells,
    }
    save_matrices = bool(args.save_matrices or args.save_pp_pvals)

    result = run_colocalization_from_pkl(
        pkl_path=args.pkl,
        out_dir=args.out_dir,
        distance_threshold=args.distance,
        pp_alpha=args.pp_alpha,
        cpb_alpha=args.cpb_alpha,
        min_genecount=args.min_genecount,
        threads=args.threads,
        use_3d=not args.use_2d,
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
        group_values=args.groups,
        skip_existing=args.skip_existing,
    )

    if result.get("groupby") is not None:
        print(
            "[instant-optimized] saved groupby summary: "
            f"{result['groupby_summary_path']}"
        )
        return 0

    print(
        "[instant-optimized] saved significant pairs: "
        f"{result['pairs_path']}"
    )
    print("[instant-optimized] top significant pairs:")
    print(result["significant_pairs"].head())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
