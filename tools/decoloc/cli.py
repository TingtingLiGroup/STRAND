from __future__ import annotations

import argparse

from tools.decoloc.api import run_decoloc_from_pkl


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tools.decoloc.cli",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "DeCoLoc prototype: single-pass cKDTree colocalization engine with "
            "whole-cell and compartment-conditioned closed-form null models."
        ),
    )
    parser.add_argument("--pkl", required=True, help="Path to input PKL bundle.")
    parser.add_argument("--out-dir", required=True, help="Output directory for CSV tables.")
    parser.add_argument(
        "--distance",
        type=float,
        default=None,
        help="Neighbor radius d. Default None uses auto_distance_radius_frac * median_cell_radius.",
    )
    parser.add_argument(
        "--distance-mode",
        choices=["cell_scaled", "physical"],
        default="cell_scaled",
        help="Scale radius per cell equivalent radius or keep a fixed physical distance.",
    )
    parser.add_argument(
        "--auto-distance-radius-frac",
        type=float,
        default=0.10,
        help="Fraction of median cell radius used when --distance is omitted.",
    )
    parser.add_argument(
        "--region-threshold-mode",
        choices=["proportional", "cell_scaled", "absolute"],
        default="proportional",
        help="Subcellular 4-region boundary width mode.",
    )
    parser.add_argument(
        "--kernel",
        choices=["gaussian", "uniform"],
        default="gaussian",
        help="Continuous Gaussian decay weight vs uniform 0/1 step ball.",
    )
    parser.add_argument("--kernel-sigma-frac", type=float, default=0.5, help="Gaussian sigma as fraction of radius.")
    parser.add_argument("--use-3d", action="store_true", help="Use 3D XYZ coordinates instead of 2D XY.")
    parser.add_argument("--min-coexpr-cells", type=int, default=15, help="Minimum co-expressing cells to test a pair.")
    parser.add_argument("--fdr-alpha", type=float, default=0.01, help="BH q-value cutoff.")
    parser.add_argument("--min-fe", type=float, default=1.15, help="Minimum fold enrichment O/E cutoff.")
    parser.add_argument("--grid-tile-size", type=float, default=20.0, help="Spatial tile size when cell boundaries are absent.")
    parser.add_argument("--no-prefilter", action="store_true", help="Skip N/C ratio and global gene support prefiltering.")
    parser.add_argument("--min-gene-transcripts", type=int, default=5, help="Minimum global transcripts per gene.")
    parser.add_argument("--min-gene-cells", type=int, default=10, help="Minimum expressing cells per gene.")
    parser.add_argument("--sample-cells", type=int, default=None, help="Randomly sample N cells before testing.")
    parser.add_argument("--sample-cell-frac", type=float, default=None, help="Randomly sample fraction of cells.")
    parser.add_argument("--sample-random-state", type=int, default=42, help="Cell sampling random seed.")
    parser.add_argument("--sample-stratify-by", default=None, help="Column for stratified cell sampling.")
    parser.add_argument("--groupby", default=None, help="Run independently per group (e.g. celltype).")
    parser.add_argument("--groups", nargs="+", default=None, help="Subset of group values when --groupby is set.")
    parser.add_argument("--save-all-pairs", action="store_true", help="Save full table of active tested pairs.")
    parser.add_argument("--threads", type=int, default=8, help="Worker processes for cell scanning.")
    parser.add_argument("--profile", action="store_true", help="Print stage timings and summary counts.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_decoloc_from_pkl(
        args.pkl,
        out_dir=args.out_dir,
        distance=args.distance,
        distance_mode=args.distance_mode,
        auto_distance_radius_frac=args.auto_distance_radius_frac,
        region_threshold_mode=args.region_threshold_mode,
        kernel=args.kernel,
        kernel_sigma_frac=args.kernel_sigma_frac,
        use_3d=args.use_3d,
        min_coexpr_cells=args.min_coexpr_cells,
        fdr_alpha=args.fdr_alpha,
        min_fe=args.min_fe,
        grid_tile_size=args.grid_tile_size,
        do_prefilter=not args.no_prefilter,
        min_gene_total_transcripts=args.min_gene_transcripts,
        min_gene_cells=args.min_gene_cells,
        sample_cells=args.sample_cells,
        sample_cell_frac=args.sample_cell_frac,
        sample_random_state=args.sample_random_state,
        sample_stratify_by=args.sample_stratify_by,
        groupby=args.groupby,
        group_values=args.groups,
        save_all_pairs=args.save_all_pairs,
        threads=args.threads,
        profile=args.profile,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
