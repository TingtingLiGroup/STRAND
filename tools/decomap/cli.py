"""Command-line interface for DeCoMap subcellular compartment detection."""

from __future__ import annotations

import argparse
import sys

from tools.decomap.api import run_decomap_from_pkl


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m tools.decomap.cli",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "DeCoMap: scale-invariant dual-scale subcellular compartment discovery "
            "with closed-form CSR calibration and organelle marker naming."
        ),
    )
    p.add_argument("--pkl", required=True, help="Input standardized STRAND PKL file.")
    p.add_argument("--out-prefix", required=True, help="Output file prefix.")
    p.add_argument("--group-col", default="batch", help="Column to split dataset into computation groups.")
    p.add_argument("--only-groups", default=None, help="Comma-separated group values to run.")
    p.add_argument("--display-col", default=None, help="Optional column to split outputs after global fitting.")
    p.add_argument("--n-clusters", default="auto", help="Number of compartments or 'auto'.")
    p.add_argument("--cluster-range-min", type=int, default=4, help="Min K when --n-clusters auto.")
    p.add_argument("--cluster-range-max", type=int, default=7, help="Max K when --n-clusters auto.")
    p.add_argument("--n-components", type=int, default=10, help="Shared MiniBatchNMF dictionary size.")
    p.add_argument("--n-micro-clusters", type=int, default=30, help="Over-clustering micro-clusters before cosine merge.")
    p.add_argument("--no-csr-gate", action="store_true", help="Disable closed-form CSR unstructured filtering.")
    p.add_argument("--fdr-alpha", type=float, default=0.05, help="BH FDR threshold for cell CSR gate.")
    p.add_argument(
        "--marker-csv",
        default=None,
        help="Custom naming reference CSV (columns gene, compartment, optional weight); default built-in Fazal2019 + Xia2019.",
    )
    p.add_argument("--figsize", default="auto", help='Plot figure size ("auto" or "W,H").')
    p.add_argument("--dpi", type=int, default=300, help="PNG DPI.")
    p.add_argument("--no-plots", action="store_true", help="Skip PNG rendering.")
    p.add_argument("--plot-points", action="store_true", help="Overlay transcript points on the compartment region plot.")
    p.add_argument("--threads", type=int, default=0, help="Worker threads (0 = auto up to 16).")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    p.add_argument("--profile", action="store_true", help="Print progress and runtime summary.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    n_clusters = args.n_clusters
    if isinstance(n_clusters, str) and n_clusters.lower() != "auto":
        n_clusters = int(n_clusters)

    figsize = "auto"
    if args.figsize is not None and str(args.figsize).lower() != "auto":
        w, h = str(args.figsize).split(",")
        figsize = (float(w), float(h))

    result = run_decomap_from_pkl(
        args.pkl,
        args.out_prefix,
        batch_col=args.group_col,
        only_batches=args.only_groups,
        display_col=args.display_col,
        n_clusters=n_clusters,
        cluster_range_min=args.cluster_range_min,
        cluster_range_max=args.cluster_range_max,
        n_components=args.n_components,
        n_micro_clusters=args.n_micro_clusters,
        calibrate_csr=not args.no_csr_gate,
        fdr_alpha=args.fdr_alpha,
        marker_csv=args.marker_csv,
        figsize=figsize,
        dpi=args.dpi,
        write_plots=not args.no_plots,
        subdomain_plot_points=args.plot_points,
        random_state=args.seed,
        threads=args.threads,
        profile=args.profile,
    )
    batches = result.get("batches", [])
    failed = [b for b in batches if b.get("status") == "failed"]
    if failed and len(failed) == len(batches):
        print(f"[decomap] ERROR: all {len(failed)} groups failed.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
