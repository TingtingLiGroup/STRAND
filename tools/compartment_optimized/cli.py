"""独立优化区室探测命令：``python -m tools.compartment_optimized.cli``。"""

from __future__ import annotations

import argparse
import json

from .core import OptimizedCompartmentConfig, run_optimized_compartment


def build_parser() -> argparse.ArgumentParser:
    """创建命令行参数解析器，参数名称尽量与旧区室命令保持一致。"""

    parser = argparse.ArgumentParser(description="Low-memory same-cell RNAflux compartment detection")
    parser.add_argument("--pkl", required=True, help="统一PKL输入文件。")
    parser.add_argument("--out-dir", required=True, help="独立输出目录。")
    parser.add_argument("--work-dir", default=None, help="大型临时文件目录；Dataset4建议使用本地 /tmp。")
    parser.add_argument("--group-col", default="batch", help="全局建模分组列，默认 batch。")
    parser.add_argument("--only-group", default=None, help="只运行指定分组；留空表示全部行共同建模。")
    parser.add_argument("--display-col", default="tile", help="全局建模完成后按此列拆分CSV；传入 none 表示不拆分。")
    parser.add_argument("--frac", type=float, default=0.01)
    parser.add_argument("--radius", type=float, default=5.0)
    parser.add_argument("--n-clusters", default="auto")
    parser.add_argument("--cluster-range-min", type=int, default=4)
    parser.add_argument("--cluster-range-max", type=int, default=7)
    parser.add_argument("--som-iterations", type=int, default=100)
    parser.add_argument("--som-chunk-size", type=int, default=1_000_000)
    parser.add_argument("--output-chunk-size", type=int, default=250_000)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=16, help="细胞级RNAflux和插值线程数。")
    parser.add_argument("--cell-batch-size", type=int, default=256)
    parser.add_argument("--som-workers", type=int, default=4)
    parser.add_argument("--output-workers", type=int, default=4, help="并行写出tile兼容CSV的线程数。")
    parser.add_argument("--profile", action="store_true")
    return parser


def main() -> None:
    """解析参数并启动独立优化流程。"""

    args = build_parser().parse_args()
    config = OptimizedCompartmentConfig(
        frac=args.frac,
        radius=args.radius,
        n_clusters=args.n_clusters,
        cluster_range_min=args.cluster_range_min,
        cluster_range_max=args.cluster_range_max,
        som_iterations=args.som_iterations,
        som_chunk_size=args.som_chunk_size,
        output_chunk_size=args.output_chunk_size,
        random_seed=args.random_seed,
        workers=args.workers,
        cell_batch_size=args.cell_batch_size,
        som_workers=args.som_workers,
        output_workers=args.output_workers,
    )
    metadata = run_optimized_compartment(
        args.pkl,
        args.out_dir,
        work_dir=args.work_dir,
        group_col=args.group_col,
        only_group=args.only_group,
        display_col=None if str(args.display_col).lower() == "none" else args.display_col,
        config=config,
        profile=args.profile,
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
