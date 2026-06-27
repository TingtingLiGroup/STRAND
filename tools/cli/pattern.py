from __future__ import annotations

import argparse
from pathlib import Path

from tools.api import predict_patterns_from_parquet

_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_MODEL = _ROOT / "models" / "multiclass_xgb_8class_prop075_final_from_cv.joblib"
_DEFAULT_FALLBACK = _ROOT / "models" / "multiclass_xgb_7class_no_foci_final_from_cv.joblib"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Predict RNA localization patterns from a feature parquet. "
            "Default: primary strong 8-class XGBoost model; "
            "fallback to 7-class no-Foci model when Foci ratio > threshold."
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
        default=str(_DEFAULT_FALLBACK),
        help=(
            "fallback 分类器 joblib 路径；默认使用 7 类 no-Foci XGBoost 模型 "
            "models/multiclass_xgb_7class_no_foci_final_from_cv.joblib。"
        ),
    )

    parser.add_argument(
        "--foci-fallback-threshold",
        type=float,
        default=0.5,
        help="当 primary 预测中 Foci 占比 > 该阈值时，自动切换 fallback 模型。默认 0.5。",
    )

    parser.add_argument(
        "--no-foci-fallback",
        action="store_true",
        help="关闭 Foci 占比过高时自动切换 7 分类模型的机制。",
    )

    parser.add_argument(
        "--profile",
        action="store_true",
        help="输出 primary / fallback 的 pattern 分布。",
    )

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    out = predict_patterns_from_parquet(
        feature_path=Path(args.input),
        model_path=Path(args.model),
        fallback_model_path=Path(args.fallback_model),
        foci_fallback_threshold=args.foci_fallback_threshold,
        enable_foci_fallback=not args.no_foci_fallback,
        profile=args.profile,
        output_path=Path(args.output),
    )

    print(f"[subcellfeat-pattern] saved to: {args.output}")
    print(out.head())


if __name__ == "__main__":
    main()