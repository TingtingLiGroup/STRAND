from __future__ import annotations

import argparse
from pathlib import Path

from tools.api import predict_patterns_from_parquet
from tools.api.patterns import DEFAULT_LOW_ABUNDANCE_CUT
from tools.models.pattern_classifier import (
    DEFAULT_MODEL_V2_FILENAME,
    LEGACY_MODEL_V1_FILENAME,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Predict RNA localization patterns from a precomputed feature parquet. "
            f"Default ('auto') uses {DEFAULT_MODEL_V2_FILENAME} for 28-feature v2 tables "
            f"and automatically falls back to {LEGACY_MODEL_V1_FILENAME} for legacy 17-feature tables."
        )
    )

    parser.add_argument("--input", required=True, help="输入 feature parquet 路径")
    parser.add_argument("--output", required=True, help="输出 parquet/csv 路径")

    parser.add_argument(
        "--model",
        default="auto",
        help=(
            "primary 分类器 joblib 路径；默认 'auto'（输入含 28 维 v2 特征时选用 "
            f"models/{DEFAULT_MODEL_V2_FILENAME}，旧版 17 维特征表自动回退 "
            f"models/{LEGACY_MODEL_V1_FILENAME}）。"
        ),
    )

    parser.add_argument(
        "--fallback-model",
        default=None,
        help="历史兼容参数；当前 CLI 默认不再启用 Foci fallback。",
    )

    parser.add_argument(
        "--foci-fallback-threshold",
        type=float,
        default=0.5,
        help="历史兼容参数；当前 CLI 默认不再按 Foci 占比自动切换 fallback。",
    )

    parser.add_argument(
        "--no-foci-fallback",
        action="store_true",
        help="历史兼容参数；当前 CLI 已默认关闭 Foci fallback。",
    )

    parser.add_argument(
        "--low-abundance-cut",
        type=int,
        default=DEFAULT_LOW_ABUNDANCE_CUT,
        help=(
            "标记 low_abundance 的转录本数切点，默认 10。仅在输入特征表带 "
            "n_transcripts 列时生效；该标记是描述性的，不代表置信度低。"
        ),
    )

    parser.add_argument(
        "--profile",
        action="store_true",
        help="输出 primary pattern 分布和 fallback 状态。",
    )

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    out = predict_patterns_from_parquet(
        feature_path=Path(args.input),
        model_path=args.model,
        fallback_model_path=Path(args.fallback_model) if args.fallback_model is not None else None,
        foci_fallback_threshold=args.foci_fallback_threshold,
        enable_foci_fallback=False,
        profile=args.profile,
        output_path=Path(args.output),
        low_abundance_cut=args.low_abundance_cut,
    )

    print(f"[subcellfeat-pattern] saved to: {args.output}")
    print(out.head())


if __name__ == "__main__":
    main()
