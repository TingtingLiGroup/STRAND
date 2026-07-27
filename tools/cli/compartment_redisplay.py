from __future__ import annotations

import argparse
from pathlib import Path

from tools.engines.compartment_redisplay import redisplay_existing_compartment


def build_parser() -> argparse.ArgumentParser:
    """创建已有区室结果重新拆分展示的命令行解析器。

    Returns
    -------
    argparse.ArgumentParser
        仅包含重新拆分和绘图所需参数的解析器。
    """
    parser = argparse.ArgumentParser(
        description=(
            "Reuse existing point-level compartment labels and RGB values, then "
            "redisplay them by another metadata column without rerunning RNAflux, SVD, or SOM."
        )
    )
    parser.add_argument("--pkl", required=True, type=Path, help="Original compartment data_dict.pkl")
    parser.add_argument(
        "--source-meta",
        required=True,
        type=Path,
        help="Completed source *_sampled_meta.json containing display_outputs",
    )
    parser.add_argument("--out-prefix", required=True, type=Path, help="New output prefix")
    parser.add_argument("--target-display-col", default="fov", help="New display column. Default: fov")
    parser.add_argument("--batch-col", default="batch", help="Global computation group column. Default: batch")
    parser.add_argument("--read-chunk-size", type=int, default=250_000, help="Source CSV rows per chunk")
    parser.add_argument("--dpi", type=int, default=300, help="Output PNG DPI. Default: 300")
    parser.add_argument("--embedding-point-size", type=float, default=10, help="RNAflux plot point size")
    parser.add_argument("--subdomain-point-size", type=float, default=1, help="Subdomain plot point size")
    parser.add_argument(
        "--only-target-values",
        nargs="+",
        default=None,
        help="Optional target values for a smoke test, for example: --only-target-values 0",
    )
    parser.add_argument("--profile", action="store_true", help="Print split and plot progress")
    return parser


def main() -> None:
    """解析命令行并执行无重算的区室结果重新展示。"""
    arguments = build_parser().parse_args()
    metadata = redisplay_existing_compartment(
        pkl_path=arguments.pkl,
        source_meta_path=arguments.source_meta,
        out_prefix=arguments.out_prefix,
        target_display_col=arguments.target_display_col,
        batch_col=arguments.batch_col,
        read_chunk_size=arguments.read_chunk_size,
        dpi=arguments.dpi,
        embedding_point_size=arguments.embedding_point_size,
        subdomain_point_size=arguments.subdomain_point_size,
        only_target_values=arguments.only_target_values,
        profile=arguments.profile,
    )
    print(
        f"[REDISPLAY] completed: best_k={metadata['best_k']}, "
        f"display_groups={len(metadata['display_outputs'])}, "
        f"transcripts={metadata['n_output_transcripts']:,}",
        flush=True,
    )


if __name__ == "__main__":
    main()
