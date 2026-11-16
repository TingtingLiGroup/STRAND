from __future__ import annotations

import argparse
import os
from pathlib import Path

from tools.api.compute_all import compute_all_from_pkl, run_all_with_patterns
from tools.api.patterns import DEFAULT_LOW_ABUNDANCE_CUT
from tools.models.pattern_classifier import (
    DEFAULT_MODEL_V2_FILENAME,
    LEGACY_MODEL_V1_FILENAME,
    default_model_path,
)


def build_parser():
    p = argparse.ArgumentParser(
        prog="subcellfeat",
        description=(
            "从 PKL bundle 计算 28 维尺度不变与同细胞 CSR Z-score 特征（可通过 --feature-set v1_17d 切换旧版 17 维），"
            "并进行 8 类 RNA 亚细胞定位模式预测。默认主模型为 "
            f"{DEFAULT_MODEL_V2_FILENAME}（当 --feature-set v1_17d 且未指定模型时自动切换为 {LEGACY_MODEL_V1_FILENAME}）。"
            "默认自动执行输入预处理：nc ratio 极端细胞过滤、cell-gene 低转录本过滤、gene 支持过滤和 cell 同步。"
        ),
    )

    p.add_argument(
        "--pkl",
        required=True,
        help="Path to PKL bundle (data_df/cell_boundary/nuclear_boundary).",
    )
    p.add_argument(
        "--out",
        required=True,
        help="Final output path (.parquet or .csv).",
    )

    p.add_argument(
        "--fast",
        action="store_true",
        help="Fast sanity run: SPRAWL only peripheral+central (debug, requires --features-only).",
    )

    default_proc = max(1, min(8, (os.cpu_count() or 1) - 1))
    p.add_argument(
        "--sprawl-processes",
        type=int,
        default=default_proc,
        help="Processes for SPRAWL metrics.",
    )
    p.add_argument(
        "--sprawl-iterations",
        type=int,
        default=200,
        help="num_iterations for punctate/radial.",
    )
    p.add_argument(
        "--sprawl-pairs",
        type=int,
        default=4,
        help="num_pairs for punctate/radial.",
    )

    p.add_argument(
        "--profile",
        action="store_true",
        help="Print timing and prefilter statistics for each stage.",
    )
    p.add_argument(
        "--max-cells",
        type=int,
        default=None,
        help="Limit number of cells for smoke test.",
    )

    p.add_argument(
        "--feature-set",
        default="auto",
        choices=("auto", "v2_28d", "v2_scaled_17d", "v1_17d"),
        help=(
            "特征体系版本：默认 auto（启用新版 28 维尺度不变特征 v2_28d；"
            "传 v1_17d 可回退至旧版 17 维特征与旧版默认模型）。"
        ),
    )
    p.add_argument(
        "--pattern-model",
        default=None,
        help=(
            f"primary 分类器 joblib 路径；默认按特征版本自动选用 models/{DEFAULT_MODEL_V2_FILENAME} "
            f"（或 v1_17d 下的 models/{LEGACY_MODEL_V1_FILENAME}）。"
        ),
    )
    p.add_argument(
        "--fallback-pattern-model",
        default=None,
        help="历史兼容参数；当前 CLI 默认不再启用 Foci fallback。",
    )
    p.add_argument(
        "--foci-fallback-threshold",
        type=float,
        default=0.5,
        help="历史兼容参数；当前 CLI 默认不再按 Foci 占比自动切换 fallback。",
    )
    p.add_argument(
        "--no-foci-fallback",
        action="store_true",
        help="历史兼容参数；当前 CLI 已默认关闭 Foci fallback。",
    )

    p.add_argument(
        "--features-only",
        action="store_true",
        help="只计算特征表（默认 28 维，可用 --feature-set v1_17d 切换 17 维），不进行定位模式预测。",
    )

    p.add_argument(
        "--no-prefilter",
        action="store_true",
        help="关闭默认输入预处理。",
    )

    p.add_argument(
        "--nc-ratio-q-low",
        type=float,
        default=0.025,
        help="nc ratio 分位数过滤的下分位数，默认 0.025。",
    )
    p.add_argument(
        "--nc-ratio-q-high",
        type=float,
        default=0.975,
        help="nc ratio 分位数过滤的上分位数，默认 0.975。",
    )
    p.add_argument(
        "--nc-ratio-mean-low",
        type=float,
        default=0.4,
        help="nc ratio 过滤后均值报告下限（仅报告，不强制过滤），默认 0.4。",
    )
    p.add_argument(
        "--nc-ratio-mean-high",
        type=float,
        default=0.6,
        help="nc ratio 过滤后均值报告上限（仅报告，不强制过滤），默认 0.6。",
    )
    p.add_argument(
        "--cellgene-filter-min-transcripts",
        type=int,
        default=6,
        help="过滤低转录本 cell-gene 样本的最小 transcript 数阈值，默认 6。",
    )
    p.add_argument(
        "--gene-filter-min-cells",
        type=int,
        default=10,
        help="经过 cell-gene 过滤后，保留 gene 所需的最小细胞数，默认 10。",
    )
    p.add_argument(
        "--low-abundance-cut",
        type=int,
        default=DEFAULT_LOW_ABUNDANCE_CUT,
        help=(
            "标记 low_abundance 的转录本数切点，默认 10。该标记是描述性的，"
            "只说明分类所依据的分子数处于受支持范围低端，不代表置信度低。"
        ),
    )

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()

    # Fast mode intentionally skips two SPRAWL features, so the pattern
    # classifier cannot be used on its output.
    if args.fast and not args.features_only:
        parser.error("--fast can only be used with --features-only; remove --fast for pattern prediction.")

    if args.fast and args.max_cells is None:
        args.max_cells = 50

    if args.fast:
        metrics = ("peripheral", "central")
        sprawl_kwargs = {
            "metrics": metrics,
            "processes": args.sprawl_processes,
        }
    else:
        metrics = ("peripheral", "central", "punctate", "radial")
        sprawl_kwargs = {
            "metrics": metrics,
            "processes": args.sprawl_processes,
            "num_iterations": args.sprawl_iterations,
            "num_pairs": args.sprawl_pairs,
        }

    if args.no_prefilter:
        prefilter_kwargs = None
    else:
        prefilter_kwargs = {
            "filter_cells_by_nc_ratio": True,
            "nc_ratio_q_low": args.nc_ratio_q_low,
            "nc_ratio_q_high": args.nc_ratio_q_high,
            "nc_ratio_mean_low": args.nc_ratio_mean_low,
            "nc_ratio_mean_high": args.nc_ratio_mean_high,
            "cellgene_filter_min_transcripts": args.cellgene_filter_min_transcripts,
            "gene_filter_min_cells": args.gene_filter_min_cells,
        }

    if args.features_only:
        fset = "v2_28d" if args.feature_set == "auto" else args.feature_set
        compute_all_from_pkl(
            args.pkl,
            feature_set=fset,
            sprawl_kwargs=sprawl_kwargs,
            out_path=args.out,
            profile=args.profile,
            max_cells=args.max_cells,
            prefilter_kwargs=prefilter_kwargs,
        )
    else:
        if args.pattern_model is not None:
            model_path = Path(args.pattern_model)
        elif args.feature_set in ("v1_17d", "v2_scaled_17d"):
            model_path = default_model_path(LEGACY_MODEL_V1_FILENAME)
        else:
            model_path = default_model_path(DEFAULT_MODEL_V2_FILENAME)

        fallback_model_path = (
            Path(args.fallback_pattern_model) if args.fallback_pattern_model is not None
            else None
        )

        run_all_with_patterns(
            pkl_path=args.pkl,
            model_path=str(model_path),
            feature_set=args.feature_set,
            sprawl_kwargs=sprawl_kwargs,
            pattern_output_path=args.out,
            profile=args.profile,
            max_cells=args.max_cells,
            prefilter_kwargs=prefilter_kwargs,
            fallback_model_path=str(fallback_model_path) if fallback_model_path is not None else None,
            foci_fallback_threshold=args.foci_fallback_threshold,
            enable_foci_fallback=False,
            low_abundance_cut=args.low_abundance_cut,
        )


if __name__ == "__main__":
    main()
