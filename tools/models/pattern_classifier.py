from __future__ import annotations

from pathlib import Path
import joblib
import pandas as pd
import numpy as np

_MODELS_DIR = Path(__file__).resolve().parents[2] / "models"

DEFAULT_MODEL_V2_FILENAME = "multiclass_xgb_8class_v2_28d_sparse_aug.joblib"
LEGACY_MODEL_V1_FILENAME = "multiclass_xgb_8class_prop075_final_from_cv.joblib"

# Bidirectional alias map between unprefixed feature names and legacy bento_*/sprawl_* names,
# ensuring seamless inference across both v2 (unprefixed) and historical v1 models/tables.
_BENTO_RAW_NAMES = (
    "cell_inner_proximity",
    "nucleus_inner_proximity",
    "nucleus_outer_proximity",
    "cell_inner_asymmetry",
    "nucleus_inner_asymmetry",
    "nucleus_outer_asymmetry",
    "point_dispersion_norm",
    "nucleus_dispersion_norm",
    "l_max",
    "l_max_gradient",
    "l_min_gradient",
    "l_monotony",
    "l_half_radius",
)
_SPRAWL_RAW_NAMES = ("peripheral", "central", "punctate", "radial")

FEATURE_COLUMN_ALIASES: dict[str, str] = {}
for _name in _BENTO_RAW_NAMES:
    FEATURE_COLUMN_ALIASES[_name] = f"bento_{_name}"
    FEATURE_COLUMN_ALIASES[f"bento_{_name}"] = _name
for _name in _SPRAWL_RAW_NAMES:
    FEATURE_COLUMN_ALIASES[_name] = f"sprawl_{_name}"
    FEATURE_COLUMN_ALIASES[f"sprawl_{_name}"] = _name


def strip_legacy_feature_prefixes(col: str) -> str:
    """Strip legacy 'bento_' or 'sprawl_' prefix from a feature column name."""
    if col.startswith("bento_"):
        raw = col[len("bento_"):]
        if raw in _BENTO_RAW_NAMES:
            return raw
    if col.startswith("sprawl_"):
        raw = col[len("sprawl_"):]
        if raw in _SPRAWL_RAW_NAMES:
            return raw
    return col


def default_model_path(filename: str = DEFAULT_MODEL_V2_FILENAME) -> Path:
    """Resolve default model path, works for both editable and wheel installs."""
    p = _MODELS_DIR / filename
    if p.exists():
        return p
    try:
        from importlib.resources import files
        p2 = Path(str(files("models").joinpath(filename)))
        if p2.exists():
            return p2
    except Exception:
        pass
    return p


class PatternClassifier:
    """
    单标签多分类模式预测器。

    预期模型文件是一个 joblib bundle，至少包含：
        - model（或跨版本便携 booster_json）: 已训练好的多分类模型对象
        - feature_cols: 训练时使用的特征列名列表
        - classes: 类别名列表，顺序需与 predict_proba 输出一致

    可选包含：
        - meta: 训练元信息
    """

    def __init__(self, model_path: str | Path):
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {model_path}\n"
                "If installed via git clone, run 'git lfs pull' to download model files."
            )
        # Detect Git LFS pointer (132-byte text stub instead of real binary)
        with open(model_path, "rb") as f:
            header = f.read(40)
        if header.startswith(b"version https://git-lfs"):
            raise RuntimeError(
                f"Model file is a Git LFS pointer, not the actual binary: {model_path}\n"
                "Run 'git lfs pull' in the repository root to download the real model files."
            )
        obj = joblib.load(model_path)

        self.model_path = model_path

        if not isinstance(obj, dict):
            raise ValueError(
                f"模型文件格式错误：{model_path}。\n"
                "当前 PatternClassifier 期望 joblib 文件保存为 dict，至少包含："
                "'model'（或 'booster_json'）、'feature_cols'、'classes'。"
            )

        self.model = obj.get("model", obj.get("clf", None))
        self.booster_json = obj.get("booster_json", None)
        self.feature_cols = obj.get("feature_cols", None)
        self.classes = obj.get("classes", None)
        self.meta = obj.get("meta", {})

        if self.model is None and self.booster_json is not None:
            import xgboost as xgb
            bst = xgb.Booster()
            bst.load_model(bytearray(self.booster_json))
            self.model = bst

        if self.model is None:
            raise ValueError(
                f"模型文件 {model_path} 中未找到 'model'（或 'booster_json' / 兼容别名 'clf'）。"
            )

        if self.feature_cols is None:
            raise ValueError(
                f"模型文件 {model_path} 中未找到 'feature_cols'。"
            )

        if self.classes is None:
            raise ValueError(
                f"模型文件 {model_path} 中未找到 'classes'。"
            )

        self.feature_cols = list(self.feature_cols)
        self.classes = list(self.classes)

    def _normalize_input(self, feature_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        df = feature_df.copy()

        if "cell" not in df.columns or "gene" not in df.columns:
            raise ValueError("输入特征表必须包含 'cell' 和 'gene' 列。")

        df["cell"] = df["cell"].astype(str).str.strip()
        df["gene"] = df["gene"].astype(str).str.strip()

        missing = []
        x_cols = {}
        for c in self.feature_cols:
            if c in df.columns:
                x_cols[c] = df[c]
            else:
                alias = FEATURE_COLUMN_ALIASES.get(c)
                if alias is not None and alias in df.columns:
                    x_cols[c] = df[alias]
                else:
                    missing.append(c)

        if missing:
            raise ValueError(f"输入特征表缺少这些特征列: {missing}")

        X = pd.DataFrame(x_cols, index=df.index)
        # Canonicalize output table column names by stripping legacy bento_/sprawl_ prefixes
        rename_out = {
            col: strip_legacy_feature_prefixes(col)
            for col in df.columns
            if strip_legacy_feature_prefixes(col) != col
            and strip_legacy_feature_prefixes(col) not in df.columns
        }
        if rename_out:
            df = df.rename(columns=rename_out)

        return df, X

    def predict_proba(self, feature_df: pd.DataFrame) -> pd.DataFrame:
        """
        返回每个样本的多分类概率表。
        列顺序与 self.classes 一致。
        """
        df, X = self._normalize_input(feature_df)

        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(X)
        else:
            import xgboost as xgb
            dmat = xgb.DMatrix(X, feature_names=self.feature_cols)
            probs = self.model.predict(dmat)

        if probs.shape[1] != len(self.classes):
            raise ValueError(
                "模型 predict_proba 输出列数与 classes 数量不一致："
                f"{probs.shape[1]} vs {len(self.classes)}"
            )

        prob_df = pd.DataFrame(probs, columns=self.classes, index=df.index)
        return prob_df

    def infer_labels(self, prob_df: pd.DataFrame):
        """
        从多分类概率表中得到最终单标签：
            - pattern
        """
        top1 = prob_df.idxmax(axis=1).tolist()
        return top1

    def predict(self, feature_df: pd.DataFrame) -> pd.DataFrame:
        """
        输入包含 cell/gene + 特征列的表，
        输出原表（统一去掉旧版 bento_/sprawl_ 前缀）+ 概率列 + pattern + pattern_prob。

        pattern_prob 是被选中类别的后验概率，即 max_k P(y=k|x)。
        """
        df, _ = self._normalize_input(feature_df)
        prob_df = self.predict_proba(feature_df)
        top1 = self.infer_labels(prob_df)

        out = df.copy()

        for c in self.classes:
            out[f"p_{c}"] = prob_df[c].values

        out["pattern"] = top1
        out["pattern_prob"] = prob_df.max(axis=1).values

        return out
