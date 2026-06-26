from __future__ import annotations

import argparse
import os
from pathlib import Path

from tools.api.compute_all import compute_all_from_pkl, run_all_with_patterns


def build_parser():
    p = argparse.ArgumentParser(
        prog="subcellfeat",
        description=(
            "从 PKL bundle 计算 17 维特征（Bento13 + SPRAWL4），并进行 "
            "RNA 定位模式预测。默认先使用强 8 类 XGBoost 模型 "
            "multiclass_xgb_8class_prop075_final_from_cv.joblib；"
            "如果预测结果中 Foci 占比 > 0.5，则自动切换为 7 类 no-Foci 模型重新预测。"
            "默认自动执行输入预处理：nc ratio 极端细胞过滤、cell-gene 低转录本过滤、"
            "gene 支持过滤和 cell 同步。"
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
        help="Fast sanity run: SPRAWL only peripheral+central (debug).",
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
        "--pattern-model",
        default=None,
        help="primary 分类器 joblib 路径；默认使用强 8 类 XGBoost 模型models/multiclass_xgb_8class_prop075_final_from_cv.joblib。",
    )
    p.add_argument(
        "--fallback-pattern-model",
        default=None,
        help=(
            "fallback 分类器 joblib 路径；默认使用 7 类 no-Foci XGBoost 模型 "
            "models/multiclass_xgb_7class_no_foci_final_from_cv.joblib。"
            "当 primary 预测中 Foci 占比超过阈值时启用。"
        ),
    )
    p.add_argument(
        "--foci-fallback-threshold",
        type=float,
        default=0.5,
        help="当 primary 预测结果中 Foci 占比 > 该阈值时，自动切换 fallback 模型。默认 0.5。",
    )
    p.add_argument(
        "--no-foci-fallback",
        action="store_true",
        help="关闭 Foci 占比过高时自动切换 7 分类模型的机制。",
    )

    p.add_argument(
        "--features-only",
        action="store_true",
        help="只计算 17 维特征，不进行定位模式预测。",
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
        help="nc ratio 过滤后目标均值下限，默认 0.4。",
    )
    p.add_argument(
        "--nc-ratio-mean-high",
        type=float,
        default=0.6,
        help="nc ratio 过滤后目标均值上限，默认 0.6。",
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

    return p


def main():
    args = build_parser().parse_args()

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
        compute_all_from_pkl(
            args.pkl,
            sprawl_kwargs=sprawl_kwargs,
            out_path=args.out,
            profile=args.profile,
            max_cells=args.max_cells,
            prefilter_kwargs=prefilter_kwargs,
        )
    else:
        root = Path(__file__).resolve().parents[2]
        default_model = root / "models" / "multiclass_xgb_8class_prop075_final_from_cv.joblib"
        default_fallback_model = root / "models" / "multiclass_xgb_7class_no_foci_final_from_cv.joblib"
        model_path = Path(args.pattern_model) if args.pattern_model is not None else default_model
        fallback_model_path = Path(args.fallback_pattern_model) if args.fallback_pattern_model is not None else default_fallback_model

        run_all_with_patterns(
            pkl_path=args.pkl,
            model_path=str(model_path),
            sprawl_kwargs=sprawl_kwargs,
            pattern_output_path=args.out,
            profile=args.profile,
            max_cells=args.max_cells,
            prefilter_kwargs=prefilter_kwargs,
            fallback_model_path=str(fallback_model_path),
            foci_fallback_threshold=args.foci_fallback_threshold,
            enable_foci_fallback=not args.no_foci_fallback,
        )


if __name__ == "__main__":
    main()