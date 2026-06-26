# tools/cli/coloc_summary.py

from __future__ import annotations

import argparse

from tools.api.coloc_summary import summarize_one_groupby_dir, summarize_from_metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="subcellfeat-coloc-summary",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "Summarize colocalization result directories into shared, specific, "
            "ubiquitous, region-consistent, region-shifted, hub-gene, overlap, "
            "and cross-dataset conservation tables. Can be run with python -m "
            "tools.cli.coloc_summary if no console entry point is installed."
        ),
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--input", help="One colocalization result directory or groupby output directory.")
    mode.add_argument("--metadata", help="CSV with dataset,result_dir and optional species/tissue/technology columns.")
    parser.add_argument("--out", required=True, help="Output summary directory.")
    parser.add_argument("--dataset", default=None, help="Dataset name for --input mode. Default: input directory name.")
    parser.add_argument("--min-shared-groups", type=int, default=2, help="Minimum number of groups required for shared_pairs.csv in --input mode.")
    parser.add_argument("--min-conserved-datasets", type=int, default=2, help="Minimum number of datasets required for conserved_colocalized_pairs.csv in --metadata mode.")
    parser.add_argument("--region-consistency-threshold", type=float, default=0.8, help="Threshold for region-conserved pairs.")
    parser.add_argument("--top-n", type=int, default=50, help="Number of top pairs to save per group/context.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.metadata:
        paths = summarize_from_metadata(
            args.metadata,
            args.out,
            min_conserved_datasets=args.min_conserved_datasets,
            region_consistency_threshold=args.region_consistency_threshold,
            top_n=args.top_n,
        )
    else:
        paths = summarize_one_groupby_dir(
            args.input,
            args.out,
            dataset=args.dataset,
            min_shared_groups=args.min_shared_groups,
            region_consistency_threshold=args.region_consistency_threshold,
            top_n=args.top_n,
        )
    for name, path in paths.items():
        print(f"[subcellfeat-coloc-summary] saved {name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
