from __future__ import annotations

from pathlib import Path
import pandas as pd

from tools.models.pattern_classifier import PatternClassifier


def _get_foci_ratio(pattern_df: pd.DataFrame, pattern_col: str = "pattern") -> float:
    """
    计算预测结果中 Foci 的占比。
    """
    if pattern_df is None or len(pattern_df) == 0:
        return 0.0

    if pattern_col not in pattern_df.columns:
        return 0.0

    return float((pattern_df[pattern_col].astype(str) == "Foci").mean())


def predict_patterns_from_parquet(
    feature_path: str | Path,
    output_path: str | Path,
    model_path: str | Path,
    fallback_model_path: str | Path | None = None,
    foci_fallback_threshold: float = 0.5,
    enable_foci_fallback: bool = True,
    profile: bool = False,
):
    """
    从已计算好的 feature parquet 预测 RNA 定位模式。

    默认逻辑：
      1. 先使用 primary 强 8 分类模型预测。
      2. 如果 Foci 占比 > foci_fallback_threshold，
         则使用 fallback 7 分类 no-Foci 模型重新预测并替代结果。
      3. 最终类别列统一为 pattern。
    """
    df = pd.read_parquet(feature_path)

    clf = PatternClassifier(model_path)
    out = clf.predict(df)

    foci_ratio = _get_foci_ratio(out, pattern_col="pattern")
    used_fallback = False

    if profile:
        print("[subcellfeat-pattern] primary_model =", model_path)
        print("[subcellfeat-pattern] primary_classes =", getattr(clf, "classes", None))
        print("[subcellfeat-pattern] primary_n_rows =", len(out))
        print("[subcellfeat-pattern] primary_foci_ratio =", foci_ratio)
        print("[subcellfeat-pattern] primary_distribution =")
        print(out["pattern"].value_counts(normalize=True))

    if (
        enable_foci_fallback
        and fallback_model_path is not None
        and foci_ratio > foci_fallback_threshold
    ):
        if profile:
            print(
                "[subcellfeat-pattern][fallback] triggered because "
                f"Foci ratio {foci_ratio:.4f} > threshold {foci_fallback_threshold:.4f}"
            )
            print("[subcellfeat-pattern][fallback] fallback_model =", fallback_model_path)

        fallback_clf = PatternClassifier(fallback_model_path)
        out = fallback_clf.predict(df)
        used_fallback = True

        if profile:
            print("[subcellfeat-pattern][fallback] fallback_classes =", getattr(fallback_clf, "classes", None))
            print("[subcellfeat-pattern][fallback] fallback_distribution =")
            print(out["pattern"].value_counts(normalize=True))

    else:
        if profile:
            if not enable_foci_fallback:
                print("[subcellfeat-pattern][fallback] disabled")
            elif fallback_model_path is None:
                print("[subcellfeat-pattern][fallback] skipped because fallback_model_path is None")
            else:
                print(
                    "[subcellfeat-pattern][fallback] not triggered because "
                    f"Foci ratio {foci_ratio:.4f} <= threshold {foci_fallback_threshold:.4f}"
                )

    # out["pattern_model_used"] = "fallback_7class_no_foci" if used_fallback else "primary_8class_prop075"
    # out["primary_foci_ratio"] = foci_ratio
    # out["foci_fallback_threshold"] = float(foci_fallback_threshold)

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if str(out_path).endswith(".csv"):
        out.to_csv(out_path, index=False)
    else:
        out.to_parquet(out_path, index=False)

    return out


# 兼容旧名字
add_patterns_to_parquet = predict_patterns_from_parquet