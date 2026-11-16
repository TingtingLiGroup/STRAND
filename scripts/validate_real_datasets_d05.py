from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from tools.api.compute_all import (
    _prefilter_bundle_before_features,
    compute_all_from_pkl,
)
from tools.models.pattern_classifier import PatternClassifier

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


def sample_cells_deterministic(bundle: dict, max_cells: int = 500, seed: int = 42) -> dict:
    df = bundle["data_df"].copy()
    df["cell"] = df["cell"].astype(str)
    unique_cells = sorted(df["cell"].unique())
    if len(unique_cells) > max_cells:
        rng = np.random.default_rng(seed)
        chosen = set(rng.choice(unique_cells, size=max_cells, replace=False).tolist())
    else:
        chosen = set(unique_cells)

    out = {"data_df": df[df["cell"].isin(chosen)].copy()}
    if bundle.get("cell_boundary") is not None:
        out["cell_boundary"] = {
            str(k): v for k, v in bundle["cell_boundary"].items() if str(k) in chosen
        }
    if bundle.get("nuclear_boundary") is not None:
        out["nuclear_boundary"] = {
            str(k): v for k, v in bundle["nuclear_boundary"].items() if str(k) in chosen
        }
    return out


def gene_pattern_matrix(pred_df: pd.DataFrame) -> pd.DataFrame:
    ct = pd.crosstab(pred_df["gene"], pred_df["pattern"], normalize="index")
    for c in CLASSES:
        if c not in ct.columns:
            ct[c] = 0.0
    return ct[CLASSES]


def main():
    exp_root = Path("/home/lixiangyu/xwq/Subcellular_Feature/tmp/v2_28d_real_validation")
    out_dir = exp_root / "results"
    feat_cache_dir = exp_root / "feat_cache"
    out_dir.mkdir(parents=True, exist_ok=True)
    feat_cache_dir.mkdir(parents=True, exist_ok=True)

    models_dir = exp_root / "models"
    clf_v1 = PatternClassifier("/home/lixiangyu/xwq/Subcellular_Feature/models/multiclass_xgb_8class_prop075_final_from_cv.joblib")
    clf_v2 = PatternClassifier(models_dir / "multiclass_xgb_8class_v2_28d_portable.joblib")
    clf_v2_aug = PatternClassifier(models_dir / "multiclass_xgb_8class_v2_28d_sparse_aug_portable.joblib")

    datasets = [
        (
            "Dataset10_1_Xenium_MouseBrain_micron",
            "/home/lixiangyu/xwq/SubCellular_Datasets/SubCellular/DataSet-Final/Dataset10_1_Xenium_MouseBrain_data_dict.pkl",
            "/home/lixiangyu/xwq/Subcellular_Feature/RNApatterns/Dataset10/Dataset10_1_pattern.parquet",
        ),
        (
            "Dataset33_1_MERFISH_U2OS_micron",
            "/home/lixiangyu/xwq/SubCellular_Datasets/SubCellular/DataSet-Final/Dataset33_1_merfish_u2os_data_dict.pkl",
            "/home/lixiangyu/xwq/Subcellular_Feature/RNApatterns/Dataset33/Dataset33_1_pattern.parquet",
        ),
        (
            "Dataset5_MERFISH_U2OS_pixel",
            "/home/lixiangyu/xwq/SubCellular_Datasets/SubCellular/DataSet-Final/Dataset5_merfish_u2os_data_dict.pkl",
            "/home/lixiangyu/xwq/Subcellular_Feature/RNApatterns/Dataset5/Dataset5_pattern.parquet",
        ),
    ]

    prefilter_cfg = {
        "filter_cells_by_nc_ratio": True,
        "nc_ratio_q_low": 0.025,
        "nc_ratio_q_high": 0.975,
        "nc_ratio_mean_low": 0.4,
        "nc_ratio_mean_high": 0.6,
        "cellgene_filter_min_transcripts": 6,
        "gene_filter_min_cells": 10,
    }

    dist_rows = []
    anchor_rows = []
    gene_mats: dict[tuple[str, str], pd.DataFrame] = {}

    u2os_anchors = [
        "MALAT1", "FBN2", "COL5A1", "FBN1", "TLN1", "EGFR",
        "AFAP1", "MYH10", "DYNC1H1", "KIF13B", "PRKDC", "SPTBN1",
    ]
    brain_anchors = [
        "Rmst", "Carmn", "Mbp", "Mobp", "Plp1", "Gfap", "Aqp4",
        "Col1a1", "Fn1", "Apoe", "Clu", "Cst3", "Sst", "Vip", "Slc17a7",
    ]

    for ds_name, pkl_path, precomp_v1_parquet in datasets:
        t0 = time.time()
        print(f"\n=== Processing 500 cells from {ds_name} ===")
        v2_cache = feat_cache_dir / f"{ds_name}_v2_28d_500cells.parquet"

        if v2_cache.exists():
            f_v2 = pd.read_parquet(v2_cache)
            sampled_cells = set(f_v2["cell"].astype(str).unique())
            n_cells = len(sampled_cells)
            print(f"  [cache hit] v2_28d features: {f_v2.shape} ({n_cells} cells)")
        else:
            raw_bundle = pd.read_pickle(pkl_path)
            filt_bundle, _ = _prefilter_bundle_before_features(raw_bundle, **prefilter_cfg)
            sub_bundle = sample_cells_deterministic(filt_bundle, max_cells=500, seed=42)
            del raw_bundle, filt_bundle
            sampled_cells = set(sub_bundle["data_df"]["cell"].astype(str).unique())
            n_cells = len(sampled_cells)
            print(f"  sampled cells={n_cells} (prep {time.time()-t0:.1f}s)")

            t2 = time.time()
            f_v2 = compute_all_from_pkl(
                sub_bundle,
                feature_set="v2_28d",
                sprawl_kwargs={"processes": 8, "num_iterations": 200, "num_pairs": 4},
            ).reset_index()
            f_v2["cell"] = f_v2["cell"].astype(str)
            f_v2["gene"] = f_v2["gene"].astype(str)
            f_v2.to_parquet(v2_cache, index=False)
            print(f"  v2_28d features computed & cached: {f_v2.shape} in {time.time()-t2:.1f}s")

        # Load exact precomputed v1_17d features for the same sampled cells
        v1_cols = ["cell", "gene"] + clf_v1.feature_cols
        pre_df = pd.read_parquet(precomp_v1_parquet, columns=v1_cols)
        pre_df["cell"] = pre_df["cell"].astype(str)
        pre_df["gene"] = pre_df["gene"].astype(str)
        f_v1 = f_v2[["cell", "gene"]].merge(pre_df, on=["cell", "gene"], how="inner")
        print(
            f"  matched v1_17d rows={len(f_v1)}/{len(f_v2)} | "
            f"bento_l_max median: v1_raw={f_v1['bento_l_max'].median():.3f} -> v2_scaled={f_v2['bento_l_max'].median():.3f}"
        )

        preds = {
            "v1_raw_17d": clf_v1.predict(f_v1),
            "v2_full_28d": clf_v2.predict(f_v2),
            "v2_full_28d_sparse_aug": clf_v2_aug.predict(f_v2),
        }

        anchors = brain_anchors if "MouseBrain" in ds_name else u2os_anchors

        for mname, pdf in preds.items():
            vc = pdf["pattern"].value_counts(normalize=True)
            row = {
                "dataset": ds_name,
                "model": mname,
                "n_cells": int(n_cells),
                "n_cellgenes": int(len(pdf)),
                "bento_l_max_median": float(
                    f_v1["bento_l_max"].median() if mname == "v1_raw_17d" else f_v2["bento_l_max"].median()
                ),
            }
            for c in CLASSES:
                row[f"prop_{c}"] = float(vc.get(c, 0.0))
            dist_rows.append(row)

            gmat = gene_pattern_matrix(pdf)
            gene_mats[(ds_name, mname)] = gmat
            gmat.to_csv(out_dir / f"gene_pattern_matrix_{ds_name}_{mname}.csv")

            for g in anchors:
                if g in gmat.index:
                    sub_g = pdf[pdf["gene"] == g]
                    arow = {
                        "dataset": ds_name,
                        "model": mname,
                        "gene": g,
                        "n_cells_expr": int(len(sub_g)),
                        "dominant_pattern": str(gmat.loc[g].idxmax()),
                    }
                    for c in CLASSES:
                        arow[c] = float(gmat.loc[g, c])
                    anchor_rows.append(arow)

    dist_df = pd.DataFrame(dist_rows)
    dist_df.to_csv(out_dir / "real_datasets_8class_distributions.csv", index=False)

    anchor_df = pd.DataFrame(anchor_rows)
    anchor_df.to_csv(out_dir / "real_datasets_anchor_genes.csv", index=False)

    concord_rows = []
    for mname in ("v1_raw_17d", "v2_full_28d", "v2_full_28d_sparse_aug"):
        m5 = gene_mats[("Dataset5_MERFISH_U2OS_pixel", mname)]
        m33 = gene_mats[("Dataset33_1_MERFISH_U2OS_micron", mname)]
        shared = sorted(set(m5.index) & set(m33.index))
        A = m5.loc[shared].to_numpy(dtype=np.float64)
        B = m33.loc[shared].to_numpy(dtype=np.float64)

        cos_num = np.sum(A * B, axis=1)
        cos_den = np.maximum(np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1), 1e-12)
        cos_sims = cos_num / cos_den
        tv_dists = 0.5 * np.sum(np.abs(A - B), axis=1)
        dom_agree = (np.argmax(A, axis=1) == np.argmax(B, axis=1)).mean()
        flat_r = float(np.corrcoef(A.ravel(), B.ravel())[0, 1])

        concord_rows.append({
            "model": mname,
            "n_shared_genes": len(shared),
            "mean_gene_cosine_sim": float(np.mean(cos_sims)),
            "median_gene_cosine_sim": float(np.median(cos_sims)),
            "mean_total_variation_dist": float(np.mean(tv_dists)),
            "dominant_pattern_agreement": float(dom_agree),
            "flattened_matrix_pearson_r": flat_r,
        })

    concord_df = pd.DataFrame(concord_rows)
    concord_df.to_csv(out_dir / "u2os_pixel_vs_micron_concordance.csv", index=False)

    print("\n=== Real Datasets 8-Class Proportions ===")
    print(dist_df.round(4).to_string(index=False))
    print("\n=== U2OS Cross-Unit Concordance (#5 Pixel vs #33_1 Micron) ===")
    print(concord_df.round(4).to_string(index=False))
    print("\n=== Anchor Gene Profiles ===")
    print(anchor_df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
