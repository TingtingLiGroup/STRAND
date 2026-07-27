from __future__ import annotations

import argparse
from pathlib import Path

from tools.api import predict_patterns_from_parquet
from tools.models.pattern_classifier import default_model_path

_DEFAULT_MODEL = default_model_path("multiclass_xgb_8class_prop075_final_from_cv.joblib")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Predict RNA localization patterns from a feature parquet. "
            "Default: primary strong 8-class XGBoost model only; "
            "Foci fallback is disabled."
        )
    )

    parser.add_argument("--input", required=True, help="输入 feature parquet 路径")
    parser.add_argument("--output", required=True, help="输出 parquet/csv 路径")

    parser.add_argument(
        "--model",
        default=str(_DEFAULT_MODEL),
        help=(
            "primary 分类器 joblib 路径；默认使用强 8 类 XGBoost 模型 "
            "models/multiclass_xgb_8class_prop075_final_from_cv.joblib。"
        ),
    )

    parser.add_argument(
        "--fallback-model",
        default=None,
        help=(
            "历史兼容参数；当前 CLI 默认不再启用 Foci fallback。"
        ),
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
        model_path=Path(args.model),
        fallback_model_path=Path(args.fallback_model) if args.fallback_model is not None else None,
        foci_fallback_threshold=args.foci_fallback_threshold,
        enable_foci_fallback=False,
        profile=args.profile,
        output_path=Path(args.output),
    )

    print(f"[subcellfeat-pattern] saved to: {args.output}")
    print(out.head())


if __name__ == "__main__":
    main()
