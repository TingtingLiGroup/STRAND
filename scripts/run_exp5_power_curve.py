"""Exp 5: Continuous Power & Sensitivity Curve across Coupling Strength (10%->60%) x Cell Count (20->200).

Evaluates statistical recall (power), false positive rate (on matched co-compartment & foci controls),
and AUPRC across:
  - Coupling fraction alpha in [0.10, 0.20, 0.30, 0.45, 0.60]
  - Number of cells C in [20, 50, 100, 200]
  - Two abundance regimes simultaneously present in every simulation:
      * Low-copy regime (2-4 transcripts/cell, typical imaging ST / lncRNA)
      * Moderate same-compartment regime (~12 transcripts/cell inside nucleus/cytoplasm)
      * Matched negative controls: N1 co-compartment independent + N2 homotypic foci
  - 3 independent random seeds per (C, alpha) grid point for mean +/- SD curves.

Compares:
  1. DeCoLoc (Gaussian)  [default continuous decay kernel]
  2. DeCoLoc (Uniform)   [ablation: 0/1 step ball]
  3. InSTAnT (PP+CPB)    [raw InSTAnT]
  4. InSTAnT (STRAND>=6) [with STRAND's default cell-gene >= 6 prefilter]
  5. SCRIN (cooccurrence)
  6. Bento (CLQ)         [threshold-free AUPRC / Top-K precision]
"""

from __future__ import annotations

from pathlib import Path
import pickle
import sys
import time

import numpy as np
import pandas as pd
import shapely
from shapely.affinity import scale as shp_scale, translate as shp_translate
from sklearn.metrics import average_precision_score

STRAND_ROOT = Path(__file__).resolve().parents[1]
if str(STRAND_ROOT) not in sys.path:
    sys.path.insert(0, str(STRAND_ROOT))

from scripts.benchmark_decoloc_vs_instant_scrin import _clamp_to_cell, _sample_points_in_polygon
from scripts.coloc_competitor_runners import (
    run_method_bento,
    run_method_decoloc,
    run_method_instant,
    run_method_scrin,
)
from tools.decoloc.api import run_decoloc_from_bundle
from tools.engines.coloc_region_annotation import (
    _equivalent_radius,
    _load_cell_geometries,
    _safe_area,
    _safe_buffer,
    _safe_difference,
    _safe_intersection,
)

OUT_DIR = STRAND_ROOT / "benchmarks" / "decoloc_v2_metrics"
THREADS = 16


def _load_template_geoms() -> list:
    with open(STRAND_ROOT / "data" / "simulated_data_dict.pkl", "rb") as f:
        sim_raw = pickle.load(f)
    return list(_load_cell_geometries(sim_raw).values())


def simulate_power_grid_bundle(
    base_geoms: list,
    *,
    n_cells: int,
    coupling_frac: float,
    n_pairs_per_group: int = 20,
    seed: int = 42,
) -> tuple[dict, pd.DataFrame, float]:
    """Generate controlled dataset for one (n_cells, coupling_frac, seed) point."""
    rng = np.random.default_rng(seed)
    cell_boundary = {}
    nuclear_boundary = {}
    cell_regions = {}
    radii_list = []

    for c_idx in range(n_cells):
        tmpl = base_geoms[c_idx % len(base_geoms)]
        scale_f = float(rng.uniform(0.85, 1.20))
        cx, cy = tmpl.cell.centroid.x, tmpl.cell.centroid.y
        c_poly = shp_translate(
            shp_scale(tmpl.cell, xfact=scale_f, yfact=scale_f, origin=(cx, cy)),
            xoff=c_idx * 400.0,
        )
        n_poly = shp_translate(
            shp_scale(tmpl.nuclear, xfact=scale_f, yfact=scale_f, origin=(cx, cy)),
            xoff=c_idx * 400.0,
        )
        n_poly = _safe_intersection(n_poly, c_poly) or n_poly
        cid = f"c_{c_idx:03d}"
        cell_boundary[cid] = np.asarray(c_poly.exterior.coords, dtype=np.float64)
        nuclear_boundary[cid] = np.asarray(n_poly.exterior.coords, dtype=np.float64)

        r_c = _equivalent_radius(_safe_area(c_poly))
        r_n = _equivalent_radius(_safe_area(n_poly))
        radii_list.append(r_c)

        cyto = _safe_difference(c_poly, n_poly)
        inner_nuc = _safe_buffer(n_poly, -0.25 * r_n) or n_poly
        peri_outer = _safe_intersection(cyto, _safe_difference(_safe_buffer(n_poly, 0.25 * r_n), n_poly))
        peri_inner = _safe_difference(n_poly, inner_nuc)
        peri_band = shapely.union_all([g for g in [peri_inner, peri_outer] if g is not None and not g.is_empty])

        for g_prep in (c_poly, inner_nuc, peri_band, cyto):
            if g_prep is not None:
                shapely.prepare(g_prep)

        cell_regions[cid] = {
            "cell": c_poly,
            "nuc": inner_nuc,
            "peri": peri_band if (peri_band is not None and peri_band.area > 0) else cyto,
            "cyto": cyto,
        }

    med_r = float(np.median(radii_list))
    # Search radius ~12% of cell radius (~15.3 px ~ 1.5 um), realistic slightly wider window where
    # Gaussian distance decay vs Uniform step ball shows clear signal-to-noise separation.
    eval_radius = round(0.12 * med_r, 2)
    complex_sigma = 0.16 * eval_radius  # tight ~2.4 px physical complex distance

    cids = [f"c_{i:03d}" for i in range(n_cells)]
    spot_chunks = []
    gt_rows = []

    def _emit(cid: str, gene: str, pts: np.ndarray) -> None:
        if len(pts):
            spot_chunks.append(pd.DataFrame({"cell": cid, "gene": gene, "x": pts[:, 0], "y": pts[:, 1]}))

    # Background transcriptome (~900 spots/cell across 25 background genes).
    for b_idx in range(25):
        g_bg = f"BG_{b_idx:02d}"
        for cid in cids:
            n_sp = int(rng.poisson(36))
            _emit(cid, g_bg, _sample_points_in_polygon(cell_regions[cid]["cell"], n_sp, rng))

    # Group 1: Low-copy true micro-colocalized pairs (2-4 copies/cell).
    for k in range(n_pairs_per_group):
        ga, gb = f"P_lowA_{k:02d}", f"P_lowB_{k:02d}"
        gt_rows.append({"gene_1": min(ga, gb), "gene_2": max(ga, gb), "regime": "low_copy_pos", "is_true_micro": 1})
        for cid in cids:
            na = int(rng.integers(2, 5))
            nb = int(rng.integers(2, 5))
            c_poly = cell_regions[cid]["cell"]
            pts_a = _sample_points_in_polygon(c_poly, na, rng)
            n_cpl = int(rng.binomial(nb, coupling_frac))
            parts = []
            if n_cpl > 0:
                anch = pts_a[rng.integers(0, len(pts_a), size=n_cpl)]
                parts.append(_clamp_to_cell(anch + rng.normal(0.0, complex_sigma, size=(n_cpl, 2)), c_poly, rng))
            if nb - n_cpl > 0:
                parts.append(_sample_points_in_polygon(c_poly, nb - n_cpl, rng))
            _emit(cid, ga, pts_a)
            _emit(cid, gb, np.vstack(parts))

    # Group 2: Moderate-copy intra-compartment true micro-colocalized pairs (~12 copies/cell in nuc or peri).
    for k in range(n_pairs_per_group):
        ga, gb = f"P_modA_{k:02d}", f"P_modB_{k:02d}"
        gt_rows.append({"gene_1": min(ga, gb), "gene_2": max(ga, gb), "regime": "mod_comp_pos", "is_true_micro": 1})
        comp_key = "nuc" if (k % 2 == 0) else "peri"
        for cid in cids:
            na = max(1, int(rng.poisson(12)))
            nb = max(1, int(rng.poisson(12)))
            r_poly = cell_regions[cid][comp_key]
            c_poly = cell_regions[cid]["cell"]
            pts_a = _sample_points_in_polygon(r_poly, na, rng)
            n_cpl = int(rng.binomial(nb, coupling_frac))
            parts = []
            if n_cpl > 0:
                anch = pts_a[rng.integers(0, len(pts_a), size=n_cpl)]
                parts.append(_clamp_to_cell(anch + rng.normal(0.0, complex_sigma, size=(n_cpl, 2)), c_poly, rng))
            if nb - n_cpl > 0:
                parts.append(_sample_points_in_polygon(r_poly, nb - n_cpl, rng))
            _emit(cid, ga, pts_a)
            _emit(cid, gb, np.vstack(parts))

    # Group 3 (Negative): Co-compartment independent pairs (~12 copies/cell in nuc or peri, 0% coupling).
    for k in range(n_pairs_per_group):
        ga, gb = f"N_compA_{k:02d}", f"N_compB_{k:02d}"
        gt_rows.append({"gene_1": min(ga, gb), "gene_2": max(ga, gb), "regime": "neg_cocomp", "is_true_micro": 0})
        comp_key = "peri" if (k % 2 == 0) else "nuc"
        for cid in cids:
            na = max(1, int(rng.poisson(12)))
            nb = max(1, int(rng.poisson(12)))
            r_poly = cell_regions[cid][comp_key]
            _emit(cid, ga, _sample_points_in_polygon(r_poly, na, rng))
            _emit(cid, gb, _sample_points_in_polygon(r_poly, nb, rng))

    # Group 4 (Negative): Independent homotypic foci bursts (0% cross coupling).
    for k in range(n_pairs_per_group):
        ga, gb = f"N_fociA_{k:02d}", f"N_fociB_{k:02d}"
        gt_rows.append({"gene_1": min(ga, gb), "gene_2": max(ga, gb), "regime": "neg_foci", "is_true_micro": 0})
        for cid in cids:
            c_poly = cell_regions[cid]["cell"]
            for gf in (ga, gb):
                ctrs = _sample_points_in_polygon(c_poly, 2, rng)
                burst = np.vstack([c + rng.normal(0.0, 2.0, size=(6, 2)) for c in ctrs])
                _emit(cid, gf, _clamp_to_cell(burst, c_poly, rng))

    bundle = {
        "data_df": pd.concat(spot_chunks, ignore_index=True),
        "cell_boundary": cell_boundary,
        "nuclear_boundary": nuclear_boundary,
    }
    return bundle, pd.DataFrame(gt_rows), eval_radius


def _extract_metrics_for_table(
    df_res: pd.DataFrame,
    gt_df: pd.DataFrame,
    *,
    has_cutoff: bool,
) -> dict:
    gt_keys = list(zip(gt_df["gene_1"], gt_df["gene_2"]))
    y_true = gt_df["is_true_micro"].to_numpy(dtype=int)
    n_pos = int(y_true.sum())

    if df_res.empty:
        score_map, sig_map = {}, {}
    else:
        p_keys = list(zip(df_res["gene_1"], df_res["gene_2"]))
        score_map = dict(zip(p_keys, df_res["score"].to_numpy(float)))
        sig_map = dict(zip(p_keys, df_res["significant"].to_numpy(bool)))

    floor_s = (min(score_map.values()) - 1.0) if score_map else 0.0
    y_score = np.array([score_map.get(k, floor_s) for k in gt_keys], dtype=float)
    auprc = float(average_precision_score(y_true, y_score))

    top_idx = np.argsort(-y_score, kind="mergesort")[:n_pos]
    y_topk = np.zeros_like(y_true)
    y_topk[top_idx] = 1
    prec_at_pos = float((y_true[top_idx] == 1).mean())

    if has_cutoff:
        y_pred = np.array([1 if sig_map.get(k, False) else 0 for k in gt_keys], dtype=int)
        recall_all = float(y_pred[y_true == 1].mean())
        recall_low = float(y_pred[(gt_df["regime"] == "low_copy_pos").to_numpy()].mean())
        recall_mod = float(y_pred[(gt_df["regime"] == "mod_comp_pos").to_numpy()].mean())
        fpr_all = float(y_pred[y_true == 0].mean())
        fpr_cocomp = float(y_pred[(gt_df["regime"] == "neg_cocomp").to_numpy()].mean())
    else:
        recall_all = np.nan
        recall_low = np.nan
        recall_mod = np.nan
        fpr_all = np.nan
        fpr_cocomp = np.nan

    return {
        "auprc": auprc,
        "precision_at_Kpos": prec_at_pos,
        "power_recall_all": recall_all,
        "power_recall_low_copy": recall_low,
        "power_recall_mod_comp": recall_mod,
        "fpr_all_neg": fpr_all,
        "fpr_cocomp": fpr_cocomp,
    }


def run_decoloc_variant(bundle: dict, radius: float, kernel: str, min_coexpr: int) -> pd.DataFrame:
    res, _ = run_decoloc_from_bundle(
        bundle,
        distance=radius,
        distance_mode="physical",
        region_threshold_mode="proportional",
        kernel=kernel,
        fdr_alpha=0.01,
        min_fe=1.15,
        min_coexpr_cells=min_coexpr,
        do_prefilter=False,
        threads=THREADS,
    )
    ap = res.all_pairs.copy()
    if ap.empty:
        return pd.DataFrame(columns=["gene_1", "gene_2", "score", "significant"])
    score = np.where(
        ap["t_cell"] > 0,
        -np.log10(np.clip(ap["p_comp"], 1e-300, 1.0)) + 1e-4 * np.maximum(ap["t_comp"], 0.0),
        0.0,
    )
    return pd.DataFrame(
        {
            "gene_1": ap["gene_1"],
            "gene_2": ap["gene_2"],
            "score": score,
            "significant": (ap["coloc_class"] == "direct_micro"),
        }
    )


def run_instant_with_strand_prefilter(bundle: dict, radius: float) -> pd.DataFrame:
    """InSTAnT preceded by STRAND's legacy cell-gene >= 6 transcript prefilter."""
    df = bundle["data_df"]
    cg_counts = df.groupby(["cell", "gene"], observed=True).size()
    keep_cg = set(cg_counts.index[cg_counts >= 6])
    mask = pd.MultiIndex.from_frame(df[["cell", "gene"]]).isin(keep_cg)
    df_filt = df[mask].reset_index(drop=True)
    if df_filt.empty:
        return pd.DataFrame(columns=["gene_1", "gene_2", "score", "significant"])
    sub_b = {
        "data_df": df_filt,
        "cell_boundary": bundle["cell_boundary"],
        "nuclear_boundary": bundle["nuclear_boundary"],
    }
    return run_method_instant(sub_b, radius=radius, threads=THREADS)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base_geoms = _load_template_geoms()

    cell_grid = [20, 50, 100, 200]
    coupling_grid = [0.10, 0.20, 0.30, 0.45, 0.60]
    seeds = [101, 202, 303]

    raw_rows = []
    t_start = time.perf_counter()
    total_runs = len(cell_grid) * len(coupling_grid) * len(seeds)
    done = 0

    for n_cells in cell_grid:
        min_coexpr = min(15, max(8, n_cells // 2))
        for alpha in coupling_grid:
            for rep_idx, sd_val in enumerate(seeds):
                done += 1
                bnd, gt_df, radius = simulate_power_grid_bundle(
                    base_geoms,
                    n_cells=n_cells,
                    coupling_frac=alpha,
                    n_pairs_per_group=20,
                    seed=sd_val + int(n_cells * 10 + alpha * 1000),
                )

                df_dec_g = run_decoloc_variant(bnd, radius, kernel="gaussian", min_coexpr=min_coexpr)
                df_dec_u = run_decoloc_variant(bnd, radius, kernel="uniform", min_coexpr=min_coexpr)
                df_inst_raw = run_method_instant(bnd, radius=radius, threads=THREADS)
                df_inst_pre6 = run_instant_with_strand_prefilter(bnd, radius=radius)
                df_scrin = run_method_scrin(bnd, radius=radius, threads=THREADS)
                df_bento = run_method_bento(bnd, radius=radius, min_points=2, threads=THREADS)["Bento (CLQ)"]

                for m_name, df_m, has_cut in [
                    ("DeCoLoc (Gaussian)", df_dec_g, True),
                    ("DeCoLoc (Uniform)", df_dec_u, True),
                    ("InSTAnT (raw PP+CPB)", df_inst_raw, True),
                    ("InSTAnT (STRAND >=6)", df_inst_pre6, True),
                    ("SCRIN (cooccurrence)", df_scrin, True),
                    ("Bento (CLQ)", df_bento, False),
                ]:
                    met = _extract_metrics_for_table(df_m, gt_df, has_cutoff=has_cut)
                    raw_rows.append(
                        {
                            "n_cells": n_cells,
                            "coupling_frac": alpha,
                            "replicate": rep_idx,
                            "seed": sd_val,
                            "method": m_name,
                            **met,
                        }
                    )

                if rep_idx == len(seeds) - 1:
                    print(
                        f"[{done:2d}/{total_runs}] cells={n_cells:3d}, coupling={alpha:.2f} done "
                        f"(elapsed={time.perf_counter()-t_start:.1f}s)"
                    )

    raw_df = pd.DataFrame(raw_rows)
    raw_df.to_csv(OUT_DIR / "exp5_power_curve_replicates.csv", index=False)

    metric_cols = [
        "auprc",
        "precision_at_Kpos",
        "power_recall_all",
        "power_recall_low_copy",
        "power_recall_mod_comp",
        "fpr_all_neg",
        "fpr_cocomp",
    ]
    agg_funcs = {c: ["mean", "std"] for c in metric_cols}
    grid_summary = raw_df.groupby(["n_cells", "coupling_frac", "method"], observed=True).agg(agg_funcs)
    grid_summary.columns = [f"{col}_{stat}" for col, stat in grid_summary.columns]
    grid_summary = grid_summary.reset_index()
    grid_summary.to_csv(OUT_DIR / "exp5_power_curve_grid.csv", index=False)

    print("\n=== [Exp 5 Summary] Power (Recall_all) across Coupling x Cells ===")
    piv_rec = grid_summary[grid_summary["method"] != "Bento (CLQ)"].pivot(
        index=["n_cells", "coupling_frac"], columns="method", values="power_recall_all_mean"
    )
    print(piv_rec.round(3).to_string())

    print("\n=== [Exp 5 Summary] Threshold-Free AUPRC across Coupling x Cells ===")
    piv_ap = grid_summary.pivot(
        index=["n_cells", "coupling_frac"], columns="method", values="auprc_mean"
    )
    print(piv_ap.round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
