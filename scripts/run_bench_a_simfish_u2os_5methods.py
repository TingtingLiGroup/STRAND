"""Task A: 5-Method Benchmark on SimFISH 6-Regime Mechanisms & Real U2OS Two-Tier Nulls.

Compares official packages for all 5 method families:
  1. DeCoLoc (direct_micro & whole_cell)
  2. InSTAnT (PP+CPB)
  3. SCRIN (official scrin v1.1.0 cooccurrence CLB)
  4. Bento (official bento-tools v2.1.4 CLQ & RNAcoloc tensor decomposition)
  5. pyTrance (official pytrance v0.2.0 1-layer DGI GNN embeddings)

Outputs pure CSV metric tables to benchmarks/decoloc_v2_metrics/:
  - exp1_simfish_5methods_overall.csv
  - exp1_simfish_5methods_by_regime.csv
  - exp2_u2os_nulls_5methods_summary.csv
  - exp2_u2os_nulls_5methods_topK.csv
"""

from __future__ import annotations

from pathlib import Path
import pickle
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score

STRAND_ROOT = Path(__file__).resolve().parents[1]
if str(STRAND_ROOT) not in sys.path:
    sys.path.insert(0, str(STRAND_ROOT))

from scripts.benchmark_decoloc_vs_instant_scrin import (
    _permute_u2os_labels,
    build_simfish_mechanism_dataset,
)
from scripts.coloc_competitor_runners import run_all_five_methods, run_method_scrin
from tools.decoloc.api import prefilter_bundle_for_decoloc
from tools.engines.coloc_region_annotation import (
    _equivalent_radius,
    _load_cell_geometries,
    _safe_area,
)

OUT_DIR = STRAND_ROOT / "benchmarks" / "decoloc_v2_metrics"
THREADS = 16


def run_exp1_simfish_5methods() -> tuple[pd.DataFrame, pd.DataFrame]:
    print("=== [Task A - Exp 1] SimFISH 6-Regime Mechanism Benchmark across 5 Method Families ===")
    bundle, gt_df, radius = build_simfish_mechanism_dataset(seed=2026)
    print(
        f"Dataset: cells={bundle['data_df']['cell'].nunique()}, "
        f"genes={bundle['data_df']['gene'].nunique()}, "
        f"spots={len(bundle['data_df'])}, radius={radius:.2f}"
    )

    method_tables = run_all_five_methods(
        bundle,
        radius=radius,
        distance_mode="physical",
        bento_min_points=5,
        pytrance_epochs=25,
        threads=THREADS,
    )
    # Also include official SCRIN without the 100x expression ratio filter as an explicit ablation.
    method_tables["SCRIN (no expr cut)"] = run_method_scrin(
        bundle, radius=radius, max_expr_ratio=1e9, threads=THREADS
    )

    gt_keys = list(zip(gt_df["gene_1"], gt_df["gene_2"]))
    y_true = gt_df["is_true_micro"].to_numpy(dtype=int)
    n_pos = int(y_true.sum())  # 75 true micro-colocalized pairs out of 150 target pairs

    has_cutoff_methods = {
        "DeCoLoc (direct_micro)",
        "DeCoLoc (whole_cell)",
        "InSTAnT (PP+CPB)",
        "SCRIN (cooccurrence)",
        "SCRIN (no expr cut)",
    }

    overall_rows = []
    regime_rows = []

    for m_name, df_res in method_tables.items():
        if df_res.empty:
            score_map = {}
            sig_map = {}
            elapsed = np.nan
            rss_gb = np.nan
        else:
            pair_tuples = list(zip(df_res["gene_1"], df_res["gene_2"]))
            score_map = dict(zip(pair_tuples, df_res["score"].to_numpy(dtype=float)))
            sig_map = dict(zip(pair_tuples, df_res["significant"].to_numpy(dtype=bool)))
            elapsed = float(df_res["elapsed_seconds"].iloc[0])
            rss_gb = float(df_res["peak_rss_gb"].iloc[0])

        min_score = (min(score_map.values()) - 1.0) if score_map else 0.0
        y_score = np.array([score_map.get(k, min_score) for k in gt_keys], dtype=float)

        auprc = float(average_precision_score(y_true, y_score))
        auroc = float(roc_auc_score(y_true, y_score))

        # Threshold-free Precision@75 (exact number of ground-truth positives) for ALL 5 methods!
        top75_idx = np.argsort(-y_score, kind="mergesort")[:n_pos]
        y_top75 = np.zeros_like(y_true)
        y_top75[top75_idx] = 1
        prec_at_75 = float(precision_score(y_true, y_top75, zero_division=0))

        if m_name in has_cutoff_methods:
            y_pred = np.array([1 if sig_map.get(k, False) else 0 for k in gt_keys], dtype=int)
            prec = float(precision_score(y_true, y_pred, zero_division=0))
            rec = float(recall_score(y_true, y_pred, zero_division=0))
            f1 = float(f1_score(y_true, y_pred, zero_division=0))
            n_called = int(y_pred.sum())
        else:
            # Threshold-free methods (Bento, pyTrance): report native threshold metrics as NaN
            # and evaluate regime selection under oracle Top-75 ranking.
            y_pred = y_top75
            prec = np.nan
            rec = np.nan
            f1 = np.nan
            n_called = -1

        overall_rows.append(
            {
                "method": m_name,
                "has_statistical_cutoff": m_name in has_cutoff_methods,
                "auprc": auprc,
                "auroc": auroc,
                "precision_at_75": prec_at_75,
                "threshold_precision": prec,
                "threshold_recall": rec,
                "threshold_f1": f1,
                "threshold_n_called": n_called,
                "elapsed_seconds": elapsed,
                "peak_rss_gb": rss_gb,
            }
        )

        for cat, sub in gt_df.groupby("category", sort=False):
            sub_idx = sub.index.to_numpy(dtype=int)
            call_rate_thresh = (
                float(y_pred[sub_idx].mean()) if m_name in has_cutoff_methods else np.nan
            )
            call_rate_top75 = float(y_top75[sub_idx].mean())
            regime_rows.append(
                {
                    "method": m_name,
                    "category": cat,
                    "is_positive_regime": int(sub["is_true_micro"].iloc[0]),
                    "threshold_call_rate": call_rate_thresh,
                    "oracle_top75_call_rate": call_rate_top75,
                    "mean_rank_score": float(y_score[sub_idx].mean()),
                    "n_pairs_in_regime": len(sub_idx),
                }
            )

    overall_df = pd.DataFrame(overall_rows)
    regime_df = pd.DataFrame(regime_rows)
    overall_df.to_csv(OUT_DIR / "exp1_simfish_5methods_overall.csv", index=False)
    regime_df.to_csv(OUT_DIR / "exp1_simfish_5methods_by_regime.csv", index=False)

    print("\n--- [Exp 1 Summary] SimFISH 5-Method Metrics ---")
    print(overall_df.round(4).to_string(index=False))
    return overall_df, regime_df


def _combined_ranking_fpr_at_k(orig_df: pd.DataFrame, perm_df: pd.DataFrame, k_list: list[int]) -> dict[int, float]:
    """SCRIN paper combined-ranking empirical FPR(K) = N_perm(K) / K."""
    if orig_df.empty and perm_df.empty:
        return {k: np.nan for k in k_list}
    o = pd.DataFrame({"score": orig_df["score"].to_numpy(float), "is_perm": 0})
    p = pd.DataFrame({"score": perm_df["score"].to_numpy(float), "is_perm": 1})
    comb = pd.concat([o, p], ignore_index=True)
    # Break exact score ties conservatively (0.5 expected if completely tied at 0).
    comb = comb.sort_values(["score", "is_perm"], ascending=[False, True], kind="mergesort").reset_index(drop=True)
    out = {}
    for k in k_list:
        kk = min(k, len(comb))
        # Only count pairs with strictly positive evidence if score == 0 floor is reached
        top_slice = comb.iloc[:kk]
        pos_slice = top_slice[top_slice["score"] > 0]
        if len(pos_slice) == 0:
            out[k] = 0.0
        else:
            out[k] = float(pos_slice["is_perm"].sum()) / float(len(pos_slice))
    return out


def _top_k_overlap_rate(orig_df: pd.DataFrame, perm_df: pd.DataFrame, k_list: list[int]) -> dict[int, float]:
    """SCRIN paper Fig. S2 overlap rate OR(K) = |TopK(orig) intersect TopK(perm)| / K."""
    out = {}
    orig_pairs = list(zip(orig_df["gene_1"], orig_df["gene_2"]))
    perm_pairs = list(zip(perm_df["gene_1"], perm_df["gene_2"]))
    for k in k_list:
        s_o = set(orig_pairs[:k])
        s_p = set(perm_pairs[:k])
        denom = max(min(k, len(s_o)), 1)
        out[k] = len(s_o & s_p) / float(denom)
    return out


def run_exp2_u2os_two_tier_nulls_5methods(n_cells_eval: int = 300) -> tuple[pd.DataFrame, pd.DataFrame]:
    print(f"\n=== [Task A - Exp 2] Real U2OS Two-Tier Null Benchmark ({n_cells_eval} cells) across 5 Methods ===")
    with open(STRAND_ROOT / "data" / "Dataset1_merfish_u2os_data_dict.pkl", "rb") as f:
        u2os_raw = pickle.load(f)

    u2os_clean, _ = prefilter_bundle_for_decoloc(u2os_raw)
    rng = np.random.default_rng(42)
    all_cids = sorted(u2os_clean["data_df"]["cell"].unique())
    chosen_cids = set(rng.choice(all_cids, size=min(n_cells_eval, len(all_cids)), replace=False))

    base_df = u2os_clean["data_df"][u2os_clean["data_df"]["cell"].isin(chosen_cids)].reset_index(drop=True)
    cb = {k: v for k, v in u2os_clean["cell_boundary"].items() if str(k) in chosen_cids}
    nb = {k: v for k, v in u2os_clean["nuclear_boundary"].items() if str(k) in chosen_cids}
    orig_bundle = {"data_df": base_df, "cell_boundary": cb, "nuclear_boundary": nb}
    geoms = _load_cell_geometries(orig_bundle)

    radii = [_equivalent_radius(_safe_area(g.cell)) for g in geoms.values()]
    radius = float(0.10 * np.median(radii))

    tier1_df = _permute_u2os_labels(base_df, geoms, within_compartment=False, seed=101)
    tier2_df = _permute_u2os_labels(base_df, geoms, within_compartment=True, seed=202)

    cond_bundles = {
        "Original_U2OS": orig_bundle,
        "Tier1_WholeCell_Shuffle": {"data_df": tier1_df, "cell_boundary": cb, "nuclear_boundary": nb},
        "Tier2_WithinRegion_Shuffle": {"data_df": tier2_df, "cell_boundary": cb, "nuclear_boundary": nb},
    }

    all_cond_res: dict[str, dict[str, pd.DataFrame]] = {}
    summary_rows = []
    total_pairs = len(base_df["gene"].unique()) * (len(base_df["gene"].unique()) - 1) // 2

    for cond_name, bnd in cond_bundles.items():
        print(f"\n--- Running 5 method families on {cond_name} ---")
        res_map = run_all_five_methods(
            bnd,
            radius=radius,
            distance_mode="cell_scaled",
            bento_min_points=10,  # Bento default min_points=10 on dense U2OS
            pytrance_epochs=25,
            pytrance_max_spots=1_500_000,
            threads=THREADS,
        )
        all_cond_res[cond_name] = res_map

        for m_name, df_m in res_map.items():
            n_sig = int(df_m["significant"].sum()) if not df_m.empty else 0
            top50_med = float(df_m["score"].iloc[:50].median()) if len(df_m) >= 1 else np.nan
            elapsed = float(df_m["elapsed_seconds"].iloc[0]) if not df_m.empty else np.nan
            rss_gb = float(df_m["peak_rss_gb"].iloc[0]) if not df_m.empty else np.nan
            summary_rows.append(
                {
                    "condition": cond_name,
                    "method": m_name,
                    "n_tested_or_scored": len(df_m),
                    "n_significant": n_sig,
                    "sig_fraction_of_panel": n_sig / max(total_pairs, 1),
                    "top50_median_score": top50_med,
                    "elapsed_seconds": elapsed,
                    "peak_rss_gb": rss_gb,
                }
            )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUT_DIR / "exp2_u2os_nulls_5methods_summary.csv", index=False)

    # Combined-ranking FPR@K and Top-K overlap OR@K (universal across thresholded & unthresholded methods).
    k_list = [50, 100, 200, 500]
    topk_rows = []
    orig_map = all_cond_res["Original_U2OS"]
    for null_name in ["Tier1_WholeCell_Shuffle", "Tier2_WithinRegion_Shuffle"]:
        perm_map = all_cond_res[null_name]
        for m_name in orig_map:
            fpr_dict = _combined_ranking_fpr_at_k(orig_map[m_name], perm_map[m_name], k_list)
            or_dict = _top_k_overlap_rate(orig_map[m_name], perm_map[m_name], k_list)
            orig_med = float(orig_map[m_name]["score"].iloc[:100].median()) if len(orig_map[m_name]) else np.nan
            perm_med = float(perm_map[m_name]["score"].iloc[:100].median()) if len(perm_map[m_name]) else np.nan
            for k in k_list:
                topk_rows.append(
                    {
                        "null_type": null_name,
                        "method": m_name,
                        "K": k,
                        "combined_ranking_FPR_at_K": fpr_dict[k],
                        "topK_orig_perm_overlap_rate": or_dict[k],
                        "top100_orig_median_score": orig_med,
                        "top100_perm_median_score": perm_med,
                    }
                )

    topk_df = pd.DataFrame(topk_rows)
    topk_df.to_csv(OUT_DIR / "exp2_u2os_nulls_5methods_topK.csv", index=False)

    print("\n--- [Exp 2 Summary] U2OS Two-Tier Nulls (Thresholded Calls & Runtime) ---")
    print(summary_df.round(4).to_string(index=False))
    print("\n--- [Exp 2 Top-K] Combined-Ranking Empirical FPR@K (K=100 & 200) ---")
    sub_tk = topk_df[topk_df["K"].isin([100, 200])]
    print(sub_tk.round(4).to_string(index=False))
    return summary_df, topk_df


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["exp1", "exp2", "both"], default="both")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.part in {"exp1", "both"}:
        run_exp1_simfish_5methods()
    if args.part in {"exp2", "both"}:
        run_exp2_u2os_two_tier_nulls_5methods(n_cells_eval=300)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
