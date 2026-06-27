from __future__ import annotations

from pathlib import Path
import joblib
import pandas as pd
import numpy as np


class PatternClassifier:
    """
    单标签多分类模式预测器。

    预期模型文件是一个 joblib bundle，至少包含：
        - model: 已训练好的多分类模型对象（如 XGBClassifier）
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
                "'model', 'feature_cols', 'classes'。"
            )

        self.model = obj.get("model", obj.get("clf", None))
        self.feature_cols = obj.get("feature_cols", None)
        self.classes = obj.get("classes", None)
        self.meta = obj.get("meta", {})

        if self.model is None:
            raise ValueError(
                f"模型文件 {model_path} 中未找到 'model'（或兼容别名 'clf'）。"
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

        # print("[PatternClassifier] model_path =", model_path)
        # print("[PatternClassifier] meta =", self.meta)
        # print("[PatternClassifier] model_type =", type(self.model))
        # print("[PatternClassifier] classes =", self.classes)

    def _normalize_input(self, feature_df: pd.DataFrame) -> pd.DataFrame:
        df = feature_df.copy()

        if "cell" not in df.columns or "gene" not in df.columns:
            raise ValueError("输入特征表必须包含 'cell' 和 'gene' 列。")

        df["cell"] = df["cell"].astype(str).str.strip()
        df["gene"] = df["gene"].astype(str).str.strip()

        missing = [c for c in self.feature_cols if c not in df.columns]
        if missing:
            raise ValueError(f"输入特征表缺少这些特征列: {missing}")

        return df

    def predict_proba(self, feature_df: pd.DataFrame) -> pd.DataFrame:
        """
        返回每个样本的多分类概率表。
        列顺序与 self.classes 一致。
        """
        df = self._normalize_input(feature_df)
        X = df[self.feature_cols].copy()

        probs = self.model.predict_proba(X)

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
        输入包含 cell/gene + 17维特征 的表，
        输出原表 + 概率列 + pattern。
        """
        df = self._normalize_input(feature_df)
        prob_df = self.predict_proba(df)
        top1 = self.infer_labels(prob_df)

        out = df.copy()

        for c in self.classes:
            out[f"p_{c}"] = prob_df[c].values

        out["pattern"] = top1

        return out

