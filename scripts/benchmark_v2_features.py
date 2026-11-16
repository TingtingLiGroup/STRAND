from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    log_loss,
)
from sklearn.model_selection import GroupKFold
from xgboost import XGBClassifier

from tools.api.compute_all import compute_all_from_pkl
from tools.engines.csr_features import (
    FEATURE_COLS_V1_17D,
    FEATURE_COLS_V2_23D,
    FEATURE_COLS_V2_28D,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
CACHE_DIR = DATA_DIR / "v2_cache"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "benchmarks" / "pattern_v2_28d"

CLASSES = [
    "Foci",
    "Nuclear",
    "Cytoplasmic",
    "Nuclear edge",
    "Cell edge",
    "Protrusion",
    "Radial",
    "Random",
]
CLASS_TO_ID = {c: i for i, c in enumerate(CLASSES)}
ID_TO_CLASS = {i: c for c, i in CLASS_TO_ID.items()}
STRONG_PATTERN_MIN = 0.75


def make_multiclass_xgb(random_state: int = 42) -> XGBClassifier:
    return XGBClassifier(
        objective="multi:softprob",
        num_class=len(CLASSES),
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.0,
        reg_lambda=1.0,
        eval_metric="mlogloss",
        random_state=random_state,
        n_jobs=-1,
        tree_method="hist",
    )


def build_strong_cellgene_meta(sim_bundle: dict) -> pd.DataFrame:
    spot = sim_bundle["data_df"].copy()
    for c in ["cell", "gene"]:
        spot[c] = spot[c].astype(str).str.strip()
    spot["groundtruth"] = spot["groundtruth"].astype(str).str.strip()
    spot["proportion_pattern"] = pd.to_numeric(spot["proportion_pattern"], errors="coerce")

    meta = (
        spot.groupby(["cell", "gene"])
        .agg(
            y_true=("groundtruth", lambda s: s.value_counts().idxmax()),
            proportion_pattern=("proportion_pattern", "mean"),
            n_transcripts=("gene", "size"),
        )
        .reset_index()
    )
    strong = meta[meta["proportion_pattern"] >= STRONG_PATTERN_MIN].copy()
    strong = strong.sort_values(["cell", "gene"]).reset_index(drop=True)
    return strong


def scale_bundle(bundle: dict, scale: float) -> dict:
    df = bundle["data_df"].copy()
    df["x"] = df["x"].astype(np.float64) * scale
    df["y"] = df["y"].astype(np.float64) * scale

    def _scale_b(bdict):
        if bdict is None:
            return None
        out = {}
        for k, v in bdict.items():
            if isinstance(v, pd.DataFrame):
                v2 = v.copy()
                v2["x"] = v2["x"].astype(np.float64) * scale
                v2["y"] = v2["y"].astype(np.float64) * scale
                out[k] = v2
            else:
                arr = np.asarray(v, dtype=np.float64).copy()
                arr[:, 0] *= scale
                arr[:, 1] *= scale
                out[k] = arr
        return out

    return {
        "data_df": df,
        "cell_boundary": _scale_b(bundle.get("cell_boundary")),
        "nuclear_boundary": _scale_b(bundle.get("nuclear_boundary")),
    }


def subsample_bundle_spots(
    bundle: dict,
    min_spots: int,
    max_spots: int,
    seed: int,
    drop_below_min: bool = True,
) -> dict:
    """
    Deterministically subsample each (cell, gene) transcript cloud to a random count
    in [min_spots, max_spots] without replacement to simulate sparse detection depth.
    """
    rng = np.random.default_rng(seed)
    df = bundle["data_df"].copy()
    df["cell"] = df["cell"].astype(str).str.strip()
    df["gene"] = df["gene"].astype(str).str.strip()

    kept_indices = []
    for _, grp in df.groupby(["cell", "gene"], sort=True):
        n = len(grp)
        if n < min_spots:
            if not drop_below_min:
                kept_indices.append(grp.index.to_numpy())
            continue
        target_n = int(rng.integers(min_spots, max_spots + 1))
        if n <= target_n:
            kept_indices.append(grp.index.to_numpy())
        else:
            chosen = rng.choice(grp.index.to_numpy(), size=target_n, replace=False)
            kept_indices.append(np.sort(chosen))

    all_idx = np.concatenate(kept_indices) if kept_indices else np.array([], dtype=np.int64)
    return {
        "data_df": df.loc[all_idx].reset_index(drop=True),
        "cell_boundary": bundle.get("cell_boundary"),
        "nuclear_boundary": bundle.get("nuclear_boundary"),
    }


def compute_metrics_dict(y_true: np.ndarray, probs: np.ndarray) -> dict:
    pred_id = np.argmax(probs, axis=1)
    pred_label = np.array([ID_TO_CLASS[i] for i in pred_id])
    y_true_id = np.array([CLASS_TO_ID[c] for c in y_true])

    rep = classification_report(
        y_true,
        pred_label,
        labels=CLASSES,
        output_dict=True,
        zero_division=0,
    )
    out = {
        "n_samples": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, pred_label)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred_label)),
        "macro_precision": float(rep["macro avg"]["precision"]),
        "macro_recall": float(rep["macro avg"]["recall"]),
        "macro_f1": float(rep["macro avg"]["f1-score"]),
        "mlogloss": float(log_loss(y_true_id, probs, labels=list(range(len(CLASSES))))),
    }
    for c in CLASSES:
        out[f"f1_{c}"] = float(rep[c]["f1-score"])
        out[f"recall_{c}"] = float(rep[c]["recall"])
        out[f"precision_{c}"] = float(rep[c]["precision"])
    return out


def ensure_cached_features(
    cache_path: Path,
    bundle: dict,
    feature_set: str,
    processes: int = 6,
) -> pd.DataFrame:
    if cache_path.exists():
        print(f"[cache hit] {cache_path.name}")
        return pd.read_parquet(cache_path)

    t0 = time.time()
    print(f"[computing] {cache_path.name} (feature_set={feature_set}, processes={processes}) ...")
    feat_df = compute_all_from_pkl(
        bundle,
        feature_set=feature_set,
        sprawl_kwargs={"processes": processes, "num_iterations": 200, "num_pairs": 4},
        profile=False,
    ).reset_index()
    feat_df["cell"] = feat_df["cell"].astype(str).str.strip()
    feat_df["gene"] = feat_df["gene"].astype(str).str.strip()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    feat_df.to_parquet(cache_path, index=False)
    print(f"[done] {cache_path.name} shape={feat_df.shape} in {time.time() - t0:.1f}s")
    return feat_df


def align_with_meta(meta_df: pd.DataFrame, feat_df: pd.DataFrame) -> pd.DataFrame:
    f = feat_df.copy()
    f["cell"] = f["cell"].astype(str).str.strip()
    f["gene"] = f["gene"].astype(str).str.strip()
    merged = meta_df[["cell", "gene", "y_true", "n_transcripts"]].merge(
        f,
        on=["cell", "gene"],
        how="inner",
    )
    return merged


def run_ablation_stage(sim_bundle: dict, processes: int = 6) -> tuple[pd.DataFrame, pd.DataFrame]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Align strong meta in the exact order of original simulated_features.parquet merge
    orig_v1_feat = ensure_cached_features(
        CACHE_DIR / "simulated_features.parquet",
        sim_bundle,
        feature_set="v1_17d",
        processes=processes,
    )
    for c in ["cell", "gene"]:
        orig_v1_feat[c] = orig_v1_feat[c].astype(str).str.strip()

    spot = sim_bundle["data_df"].copy()
    for c in ["cell", "gene"]:
        spot[c] = spot[c].astype(str).str.strip()
    spot["groundtruth"] = spot["groundtruth"].astype(str).str.strip()
    spot["proportion_pattern"] = pd.to_numeric(spot["proportion_pattern"], errors="coerce")
    cellgene_meta = (
        spot.groupby(["cell", "gene"])
        .agg(
            y_true=("groundtruth", lambda s: s.value_counts().idxmax()),
            proportion_pattern=("proportion_pattern", "mean"),
            n_transcripts=("gene", "size"),
        )
        .reset_index()
    )
    strong_meta = cellgene_meta[cellgene_meta["proportion_pattern"] >= STRONG_PATTERN_MIN].copy()

    # Exact merge order used by XGBoost_8class_proportion075.ipynb
    base_v1_data = orig_v1_feat.merge(
        strong_meta[["cell", "gene", "y_true"]],
        on=["cell", "gene"],
        how="inner",
    ).reset_index(drop=True)

    v2_feat_full = ensure_cached_features(
        CACHE_DIR / "simulated_features_v2_28d.parquet",
        sim_bundle,
        feature_set="v2_28d",
        processes=processes,
    )
    base_v2_data = base_v1_data[["cell", "gene", "y_true"]].merge(
        v2_feat_full,
        on=["cell", "gene"],
        how="inner",
    ).reset_index(drop=True)

    assert len(base_v1_data) == len(base_v2_data) == 12000
    assert (base_v1_data["cell"] == base_v2_data["cell"]).all()
    assert (base_v1_data["gene"] == base_v2_data["gene"]).all()

    configs = [
        ("1_v1_raw_17d", base_v1_data, FEATURE_COLS_V1_17D),
        ("2_v2_scaled_17d", base_v2_data, FEATURE_COLS_V1_17D),
        ("3_v2_23d_plus_csr", base_v2_data, FEATURE_COLS_V2_23D),
        ("4_v2_full_28d", base_v2_data, FEATURE_COLS_V2_28D),
    ]

    groups = base_v1_data["cell"].astype(str).values
    y_str = base_v1_data["y_true"].astype(str).values
    y_id = np.array([CLASS_TO_ID[c] for c in y_str], dtype=int)
    gkf = GroupKFold(n_splits=5)
    splits = list(gkf.split(base_v1_data, y_id, groups=groups))

    ablation_rows = []
    fold_detail_rows = []

    for cfg_name, df_src, fcols in configs:
        X = df_src[fcols].copy()
        oof_probs = np.zeros((len(df_src), len(CLASSES)), dtype=np.float64)

        for fold_id, (tr_idx, te_idx) in enumerate(splits, start=1):
            clf = make_multiclass_xgb(random_state=42 + fold_id)
            clf.fit(X.iloc[tr_idx], y_id[tr_idx])
            probs = clf.predict_proba(X.iloc[te_idx])
            oof_probs[te_idx] = probs

            f_m = compute_metrics_dict(y_str[te_idx], probs)
            f_m["config"] = cfg_name
            f_m["fold"] = fold_id
            fold_detail_rows.append(f_m)

        overall = compute_metrics_dict(y_str, oof_probs)
        overall["config"] = cfg_name
        overall["n_features"] = len(fcols)
        ablation_rows.append(overall)

        if cfg_name in ("1_v1_raw_17d", "4_v2_full_28d"):
            pred_top1 = [ID_TO_CLASS[i] for i in np.argmax(oof_probs, axis=1)]
            cm = confusion_matrix(y_str, pred_top1, labels=CLASSES)
            pd.DataFrame(
                cm,
                index=[f"true:{c}" for c in CLASSES],
                columns=[f"pred:{c}" for c in CLASSES],
            ).to_csv(RESULTS_DIR / f"cm_oof_{cfg_name}.csv")

    ablation_df = pd.DataFrame(ablation_rows)
    cols_order = ["config", "n_features", "accuracy", "macro_f1", "mlogloss"] + [
        f"f1_{c}" for c in CLASSES
    ]
    ablation_df = ablation_df[cols_order]
    ablation_df.to_csv(RESULTS_DIR / "exp1_simfish_ablation_oof.csv", index=False)
    pd.DataFrame(fold_detail_rows).to_csv(
        RESULTS_DIR / "exp1_simfish_ablation_folds.csv", index=False
    )

    # Train full 50-cell v2_28d model & export feature importances
    final_v2 = make_multiclass_xgb(random_state=42)
    final_v2.fit(base_v2_data[FEATURE_COLS_V2_28D], y_id)

    fi_df = pd.DataFrame({
        "feature": FEATURE_COLS_V2_28D,
        "gain_importance": final_v2.feature_importances_,
    }).sort_values("gain_importance", ascending=False).reset_index(drop=True)
    fi_df.to_csv(RESULTS_DIR / "exp1_v2_28d_feature_importances.csv", index=False)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    booster_json = bytes(final_v2.get_booster().save_raw(raw_format="json"))
    v2_model_path = CACHE_DIR / "multiclass_xgb_8class_v2_28d_portable.joblib"
    joblib.dump(
        {
            "model": None,
            "booster_json": booster_json,
            "feature_cols": FEATURE_COLS_V2_28D,
            "classes": CLASSES,
            "class_to_id": CLASS_TO_ID,
            "id_to_class": ID_TO_CLASS,
            "meta": {
                "trained_on": "simulated_data_dict.pkl strong patterns (proportion_pattern >= 0.75)",
                "feature_set": "v2_28d",
                "n_samples": int(len(base_v2_data)),
                "n_cells": 50,
                "xgboost_version": xgb.__version__,
                "oof_metrics": ablation_rows[-1],
            },
        },
        v2_model_path,
    )
    print("\n=== Experiment 1: Same-Fold 5-Fold GroupKFold Ablation ===")
    print(ablation_df.to_string(index=False))
    print("\n=== Top 15 Features in v2_full_28d (Gain) ===")
    print(fi_df.head(15).to_string(index=False))
    print(f"\nSaved model: {v2_model_path}")

    return base_v1_data, base_v2_data


def run_stress_stage(
    sim_bundle: dict,
    base_v1_data: pd.DataFrame,
    base_v2_data: pd.DataFrame,
    processes: int = 6,
) -> pd.DataFrame:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    key_meta = base_v1_data[["cell", "gene", "y_true"]].copy()

    # 1. Micron scale (1/15x, cell span ~328 -> ~21.8 um)
    micron_bundle = scale_bundle(sim_bundle, scale=1.0 / 15.0)
    micron_v1_feat = ensure_cached_features(
        CACHE_DIR / "sim_micron_div15_v1_17d.parquet",
        micron_bundle,
        feature_set="v1_17d",
        processes=processes,
    )
    micron_v2_feat = ensure_cached_features(
        CACHE_DIR / "sim_micron_div15_v2_28d.parquet",
        micron_bundle,
        feature_set="v2_28d",
        processes=processes,
    )

    # 2. Sparse test bundles: Tier A [6, 10] and Tier B [11, 20]
    sparse_6_10_bundle = subsample_bundle_spots(sim_bundle, min_spots=6, max_spots=10, seed=101)
    sparse_11_20_bundle = subsample_bundle_spots(sim_bundle, min_spots=11, max_spots=20, seed=202)

    sp6_v1 = ensure_cached_features(
        CACHE_DIR / "sim_sparse_6_10_v1_17d.parquet",
        sparse_6_10_bundle,
        feature_set="v1_17d",
        processes=processes,
    )
    sp6_v2 = ensure_cached_features(
        CACHE_DIR / "sim_sparse_6_10_v2_28d.parquet",
        sparse_6_10_bundle,
        feature_set="v2_28d",
        processes=processes,
    )
    sp11_v1 = ensure_cached_features(
        CACHE_DIR / "sim_sparse_11_20_v1_17d.parquet",
        sparse_11_20_bundle,
        feature_set="v1_17d",
        processes=processes,
    )
    sp11_v2 = ensure_cached_features(
        CACHE_DIR / "sim_sparse_11_20_v2_28d.parquet",
        sparse_11_20_bundle,
        feature_set="v2_28d",
        processes=processes,
    )

    # 3. Independent training sparse-augmentation copy [6, 20] (seed=2026)
    sparse_train_aug_bundle = subsample_bundle_spots(sim_bundle, min_spots=6, max_spots=20, seed=2026)
    sp_aug_v2 = ensure_cached_features(
        CACHE_DIR / "sim_sparse_aug_train_6_20_v2_28d.parquet",
        sparse_train_aug_bundle,
        feature_set="v2_28d",
        processes=processes,
    )
    aug_v2_data = key_meta.merge(sp_aug_v2, on=["cell", "gene"], how="inner").reset_index(drop=True)

    # Align evaluation tables
    eval_sets = {
        "orig_full_1x": (base_v1_data, base_v2_data),
        "micron_scale_div15": (
            key_meta.merge(micron_v1_feat, on=["cell", "gene"], how="inner").reset_index(drop=True),
            key_meta.merge(micron_v2_feat, on=["cell", "gene"], how="inner").reset_index(drop=True),
        ),
        "sparse_11_20_spots": (
            key_meta.merge(sp11_v1, on=["cell", "gene"], how="inner").reset_index(drop=True),
            key_meta.merge(sp11_v2, on=["cell", "gene"], how="inner").reset_index(drop=True),
        ),
        "sparse_6_10_spots": (
            key_meta.merge(sp6_v1, on=["cell", "gene"], how="inner").reset_index(drop=True),
            key_meta.merge(sp6_v2, on=["cell", "gene"], how="inner").reset_index(drop=True),
        ),
    }

    # Same 5-fold cell split
    groups = base_v1_data["cell"].astype(str).values
    y_id_base = np.array([CLASS_TO_ID[c] for c in base_v1_data["y_true"]], dtype=int)
    gkf = GroupKFold(n_splits=5)
    splits = list(gkf.split(base_v1_data, y_id_base, groups=groups))

    # Collect OOF predictions per (scenario, model_variant)
    oof_store: dict[tuple[str, str], tuple[list[str], list[np.ndarray]]] = {}
    for scen in eval_sets:
        for mname in ("v1_raw_17d", "v2_full_28d", "v2_full_28d_sparse_aug"):
            oof_store[(scen, mname)] = ([], [])

    for fold_id, (tr_idx, te_idx) in enumerate(splits, start=1):
        test_cells = set(base_v1_data["cell"].iloc[te_idx].astype(str))
        train_cells = set(base_v1_data["cell"].iloc[tr_idx].astype(str))
        assert train_cells.isdisjoint(test_cells)

        # Fit 1: v1_raw_17d on train_cells
        m_v1 = make_multiclass_xgb(random_state=42 + fold_id)
        m_v1.fit(base_v1_data[FEATURE_COLS_V1_17D].iloc[tr_idx], y_id_base[tr_idx])

        # Fit 2: v2_full_28d on train_cells
        m_v2 = make_multiclass_xgb(random_state=42 + fold_id)
        m_v2.fit(base_v2_data[FEATURE_COLS_V2_28D].iloc[tr_idx], y_id_base[tr_idx])

        # Fit 3: v2_full_28d_sparse_aug on train_cells (full + sparse [6, 20] copy of train_cells ONLY)
        aug_tr = aug_v2_data[aug_v2_data["cell"].isin(train_cells)]
        X_tr_aug = pd.concat(
            [base_v2_data[FEATURE_COLS_V2_28D].iloc[tr_idx], aug_tr[FEATURE_COLS_V2_28D]],
            axis=0,
            ignore_index=True,
        )
        y_tr_aug = np.concatenate([
            y_id_base[tr_idx],
            np.array([CLASS_TO_ID[c] for c in aug_tr["y_true"]], dtype=int),
        ])
        m_v2_aug = make_multiclass_xgb(random_state=42 + fold_id)
        m_v2_aug.fit(X_tr_aug, y_tr_aug)

        # Evaluate all 3 models on test_cells across all 4 scenarios
        for scen, (df_v1_scen, df_v2_scen) in eval_sets.items():
            sub_v1 = df_v1_scen[df_v1_scen["cell"].isin(test_cells)]
            sub_v2 = df_v2_scen[df_v2_scen["cell"].isin(test_cells)]

            p_v1 = m_v1.predict_proba(sub_v1[FEATURE_COLS_V1_17D])
            oof_store[(scen, "v1_raw_17d")][0].extend(sub_v1["y_true"].tolist())
            oof_store[(scen, "v1_raw_17d")][1].append(p_v1)

            p_v2 = m_v2.predict_proba(sub_v2[FEATURE_COLS_V2_28D])
            oof_store[(scen, "v2_full_28d")][0].extend(sub_v2["y_true"].tolist())
            oof_store[(scen, "v2_full_28d")][1].append(p_v2)

            p_v2_aug = m_v2_aug.predict_proba(sub_v2[FEATURE_COLS_V2_28D])
            oof_store[(scen, "v2_full_28d_sparse_aug")][0].extend(sub_v2["y_true"].tolist())
            oof_store[(scen, "v2_full_28d_sparse_aug")][1].append(p_v2_aug)

    stress_rows = []
    for (scen, mname), (y_list, p_chunks) in oof_store.items():
        y_arr = np.array(y_list)
        p_arr = np.vstack(p_chunks)
        row = compute_metrics_dict(y_arr, p_arr)
        row["scenario"] = scen
        row["model"] = mname
        stress_rows.append(row)

    stress_df = pd.DataFrame(stress_rows)
    cols = [
        "scenario",
        "model",
        "n_samples",
        "accuracy",
        "macro_f1",
        "mlogloss",
    ] + [f"f1_{c}" for c in CLASSES] + [f"recall_{c}" for c in ("Foci", "Radial", "Protrusion")]
    stress_df = stress_df[cols]
    stress_df.to_csv(RESULTS_DIR / "exp2_exp3_scale_and_sparse_stress.csv", index=False)

    # Train final full-dataset v2_full_28d_sparse_aug model
    X_all_aug = pd.concat(
        [base_v2_data[FEATURE_COLS_V2_28D], aug_v2_data[FEATURE_COLS_V2_28D]],
        axis=0,
        ignore_index=True,
    )
    y_all_aug = np.concatenate([
        y_id_base,
        np.array([CLASS_TO_ID[c] for c in aug_v2_data["y_true"]], dtype=int),
    ])
    final_v2_aug = make_multiclass_xgb(random_state=42)
    final_v2_aug.fit(X_all_aug, y_all_aug)

    fi_aug_df = pd.DataFrame({
        "feature": FEATURE_COLS_V2_28D,
        "gain_importance": final_v2_aug.feature_importances_,
    }).sort_values("gain_importance", ascending=False).reset_index(drop=True)
    fi_aug_df.to_csv(RESULTS_DIR / "exp3_v2_28d_sparse_aug_feature_importances.csv", index=False)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    booster_aug_json = bytes(final_v2_aug.get_booster().save_raw(raw_format="json"))
    aug_model_path = MODELS_DIR / "multiclass_xgb_8class_v2_28d_sparse_aug.joblib"
    joblib.dump(
        {
            "model": None,
            "booster_json": booster_aug_json,
            "feature_cols": FEATURE_COLS_V2_28D,
            "classes": CLASSES,
            "class_to_id": CLASS_TO_ID,
            "id_to_class": ID_TO_CLASS,
            "meta": {
                "trained_on": "simulated_data_dict.pkl strong patterns (full + sparse [6, 20] downsampling augmentation)",
                "feature_set": "v2_28d",
                "n_samples": int(len(X_all_aug)),
                "n_cells": 50,
                "xgboost_version": xgb.__version__,
            },
        },
        aug_model_path,
    )

    print("\n=== Experiment 2 & 3: Micron Scale (1/15x) & Low-Abundance Stress Tests ===")
    print(stress_df[["scenario", "model", "n_samples", "accuracy", "macro_f1", "mlogloss", "f1_Foci", "f1_Cytoplasmic", "f1_Random", "recall_Foci", "recall_Radial", "recall_Protrusion"]].to_string(index=False))
    print(f"\nSaved sparse-augmented model: {aug_model_path}")
    return stress_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("ablation", "stress", "all"), default="all")
    parser.add_argument("--processes", type=int, default=6)
    args = parser.parse_args()

    print("Loading simulated_data_dict.pkl ...")
    with open(DATA_DIR / "simulated_data_dict.pkl", "rb") as f:
        sim_bundle = pickle.load(f)

    base_v1, base_v2 = run_ablation_stage(sim_bundle, processes=args.processes)
    if args.stage in ("stress", "all"):
        run_stress_stage(sim_bundle, base_v1, base_v2, processes=args.processes)


if __name__ == "__main__":
    main()
