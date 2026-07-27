from __future__ import annotations

import argparse
import sys

from tools.api.compartments import detect_compartments_from_pkl


def _parse_radius(value):
    """解析自动半径或正数绝对半径。

    Parameters
    ----------
    value
        命令行输入；``auto``表示按当前计算组细胞边界计算平均半径的三分之一。

    Returns
    -------
    str or float
        ``auto``或正浮点数半径。

    Raises
    ------
    argparse.ArgumentTypeError
        输入既不是``auto``也不是正数时抛出。
    """

    radius_text = str(value).strip()
    if radius_text.lower() == "auto":
        return "auto"
    try:
        radius_value = float(radius_text)
    except ValueError as error:
        raise argparse.ArgumentTypeError("--radius must be 'auto' or a positive number") from error
    if radius_value <= 0:
        raise argparse.ArgumentTypeError("--radius must be greater than 0")
    return radius_value


def main(argv=None):
    p = argparse.ArgumentParser(
        description=(
            "Sampled RNAflux/SOM compartment detection from unified PKL. "
            "This backend follows subdomain_test.ipynb logic."
        )
    )
    p.add_argument("--pkl", required=True, help="Input unified PKL file")
    p.add_argument("--out-prefix", required=True, help="Output prefix")
    # p.add_argument("--batch-col", default="batch", help="Batch column name. If absent in data_df, read from coordinates by cell.")
    # p.add_argument("--only-batches", default=None, help="Comma-separated batch values to run, e.g. 1 or 1,3,8")
        # General grouping options.
    # Default is still batch, but users can also run by fov, sample_id, celltype, etc.
    p.add_argument(
        "--group-col",
        default=None,
        help=(
            "Column used to split cells into groups. "
            "Default: batch. Examples: batch, fov, sample_id, celltype. "
            "If absent in data_df, it will be read from coordinates by cell."
        ),
    )
    p.add_argument(
        "--only-groups",
        default=None,
        help=(
            "Comma-separated group values to run, e.g. 1 or 1,3,8. "
            "Works with --group-col."
        ),
    )
    p.add_argument(
        "--display-col",
        default=None,
        help=(
            "After one global computation, split point-level CSV files and plots by this column. "
            "Example: use --group-col batch --display-col tile for one dataset-wide model shown by tile."
        ),
    )
    # Backward-compatible old names.
    # These are kept so old commands using --batch-col / --only-batches still work.
    p.add_argument(
        "--batch-col",
        default=None,
        help="Deprecated alias of --group-col. Kept for backward compatibility.",
    )
    p.add_argument(
        "--only-batches",
        default=None,
        help="Deprecated alias of --only-groups. Kept for backward compatibility.",
    )
    p.add_argument(
        "--figsize",
        default="auto",
        help='Figure size for output plots. Use "auto" or width,height. Example: auto or 20,20'
    )
    p.add_argument(
        "--embedding-point-size",
        type=float,
        default=10,
        help="Point size for RNAflux embedding plot. Notebook default: 10."
    )
    p.add_argument(
        "--subdomain-point-size",
        type=float,
        default=1,
        help="Point size for SOM subdomain plot. Notebook default: 1."
    )

    # Main notebook parameters
    p.add_argument("--frac", type=float, default=0.01, help="Notebook sampling fraction: df.sample(frac=frac, random_state=42). Default: 0.01")
    p.add_argument(
        "--radius",
        type=_parse_radius,
        default="auto",
        help=(
            "Local radius. Default: auto, calculated as one third of the mean cell radius "
            "from valid boundaries in the current computation group. A positive absolute value is also accepted."
        ),
    )

    # SOM / clustering
    p.add_argument("--n-clusters", default="auto", help="SOM clusters. Use 'auto' for KneeLocator over cluster range, or an integer like 4.")
    p.add_argument("--cluster-range-min", type=int, default=4, help="Minimum k for auto SOM search. Default: 4")
    p.add_argument("--cluster-range-max", type=int, default=7, help="Maximum k for auto SOM search. Default: 7")
    p.add_argument("--som-iterations", type=int, default=1000, help="MiniSom train iterations. Default: 1000")
    p.add_argument("--som-sigma", type=float, default=1.0, help="MiniSom sigma. Default: 1")
    p.add_argument("--som-learning-rate", type=float, default=0.5, help="MiniSom learning rate. Default: 0.5")
    p.add_argument(
        "--som-chunk-size",
        type=int,
        default=1_000_000,
        help="Rows per exact-equivalent SOM quantization/winner chunk. Default: 1000000.",
    )
    p.add_argument(
        "--output-csv-chunk-size",
        type=int,
        default=250_000,
        help="Rows per point-level CSV write chunk. Default: 250000.",
    )

    # Exact notebook loop by default; cardiomyocytes vectorized version optional
    p.add_argument("--embedding-mode", choices=["loop", "vectorized"], default="loop", help="loop = notebook exact; vectorized = cardiomyocytes accelerated implementation")
    p.add_argument("--device", default="auto", help="Torch device: auto, cpu, cuda:0, etc. Default uses cuda:0 if available, otherwise CPU.")

    p.add_argument("--dpi", type=int, default=300, help="Output PNG dpi. Default: 300")
    p.add_argument("--export-csv", action="store_true", help="Export full transcript CSV files like notebook. Can be huge.")
    p.add_argument("--profile", action="store_true", help="Print progress information")

    p.add_argument(
        "--same-cell-neighborhood",
        action="store_true",
        default=True,
        help=(
            "Deprecated compatibility flag. Local RNAflux neighborhoods and interpolation "
            "are now always restricted to the same cell."
        ),
    )

    args = p.parse_args(argv)

    group_col = args.group_col or args.batch_col or "batch"
    only_groups = args.only_groups if args.only_groups is not None else args.only_batches

    # figsize = None
    # if args.figsize is not None:
    #     w, h = args.figsize.split(",")
    #     figsize = (float(w), float(h))
    figsize = "auto"
    if args.figsize is not None:
        if str(args.figsize).lower() == "auto":
            figsize = "auto"
        else:
            w, h = args.figsize.split(",")
            figsize = (float(w), float(h))

    n_clusters = args.n_clusters
    if isinstance(n_clusters, str) and n_clusters.lower() != "auto":
        try:
            n_clusters = int(n_clusters)
        except ValueError:
            raise ValueError("--n-clusters must be 'auto' or an integer")

    result = detect_compartments_from_pkl(
        pkl_path=args.pkl,
        out_prefix=args.out_prefix,
        batch_col=group_col,
        only_batches=only_groups,
        frac=args.frac,
        radius=args.radius,
        n_clusters=n_clusters,
        cluster_range_min=args.cluster_range_min,
        cluster_range_max=args.cluster_range_max,
        embedding_mode=args.embedding_mode,
        som_iterations=args.som_iterations,
        som_sigma=args.som_sigma,
        som_learning_rate=args.som_learning_rate,
        device=args.device,
        dpi=args.dpi,
        export_csv=args.export_csv,
        profile=args.profile,
        figsize=figsize,
        embedding_point_size=args.embedding_point_size,
        subdomain_point_size=args.subdomain_point_size,
        same_cell_neighborhood=args.same_cell_neighborhood,
        display_col=args.display_col,
        som_chunk_size=args.som_chunk_size,
        output_csv_chunk_size=args.output_csv_chunk_size,
    )

    batches = result.get("batches", [])
    failed = [batch for batch in batches if batch.get("status") == "failed"]
    if failed and len(failed) == len(batches):
        print(f"[compartment] ERROR: all {len(failed)} batches failed.", file=sys.stderr)
        return 1
    if failed:
        print(f"[compartment] WARNING: {len(failed)}/{len(batches)} batches failed.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
