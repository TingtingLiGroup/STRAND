"""Comprehensive 3-Method Subcellular Compartment Benchmark:
DeCoMap vs Official Bento (RNAflux + fluxmap) vs Official Points2Regions.

All methods are evaluated under a strictly blind-nuclear protocol (nuclear_boundary={}):
  - Exp 1A: SimFISH realistic cell-template 5-compartment benchmark (homogeneous & 2-celltype regimes,
            including rare nucleolar and polarized protrusion microdomains).
  - Exp 1B: Pure intracellular CSR calibration benchmark (synthetic CSR inside real cell templates +
            within-cell label-permuted real MERFISH U2OS cells).
  - Exp 2:  Blind DAPI nuclear mask recovery + runtime/memory across 3 imaging platforms:
            MERFISH U2OS, Bruker CosMx HEK293T, and 10x Xenium HEK293T.
  - Exp 3:  MERFISH U2OS 3-replicate signature reproducibility (Dataset33_2..4) and orthogonal
            organelle enrichment against Fazal 2019 APEX-seq & Xia 2019 MERFISH ER/Nucleus references.

Outputs CSV metric tables to STRAND/benchmarks/decomap_v1_metrics/.
"""

from __future__ import annotations

import argparse
from itertools import combinations
import math
from pathlib import Path
import pickle
import sys
import time
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr
import shapely
from shapely.affinity import translate as shp_translate
from shapely.geometry import Point
from sklearn.metrics import (
    adjusted_rand_score,
    f1_score,
    jaccard_score,
    normalized_mutual_info_score,
    roc_auc_score,
)

STRAND_ROOT = Path(__file__).resolve().parents[1]
if str(STRAND_ROOT) not in sys.path:
    sys.path.insert(0, str(STRAND_ROOT))

from scripts.compartment_competitor_runners import run_all_three_compartment_methods
from tools.decomap.api import _load_decomap_geometries
from tools.engines.coloc_region_annotation import (
    _equivalent_radius,
    _safe_area,
    _safe_buffer,
    _safe_difference,
    _safe_intersection,
)

OUT_DIR = STRAND_ROOT / "benchmarks" / "decomap_v1_metrics"
SIM_TEMPLATE_PKL = STRAND_ROOT / "data" / "simulated_data_dict.pkl"
U2OS_D1_PKL = STRAND_ROOT / "data" / "Dataset1_merfish_u2os_data_dict.pkl"
COSMX_HEK_PKL = Path("/home/wzx/rnp/RNA_spatial/CosMx data_HEK/CosMx_HEK293T_data_dict.pkl")
XENIUM_HEK_CACHE_PKL = STRAND_ROOT / "benchmarks" / "decoloc_v2_metrics" / "_cache_xenium_hek293t_1500cells.pkl"
U2OS_REP_PATHS = {
    "Rep2": Path("/home/wzx/thesis/GRASP/u2os_data/spatial/Dataset33_merfish_u2os/Dataset33_2_merfish_u2os_data_dict.pkl"),
    "Rep3": Path("/home/wzx/thesis/GRASP/u2os_data/spatial/Dataset33_merfish_u2os/Dataset33_3_merfish_u2os_data_dict.pkl"),
    "Rep4": Path("/home/wzx/thesis/GRASP/u2os_data/spatial/Dataset33_merfish_u2os/Dataset33_4_merfish_u2os_data_dict.pkl"),
}
FAZAL_CSV = Path("/home/wzx/.conda/envs/coloc_bench/lib/python3.11/site-packages/bento/tools/gene_sets/fazal2019.csv")
XIA_CSV = Path("/home/wzx/.conda/envs/coloc_bench/lib/python3.11/site-packages/bento/tools/gene_sets/xia2019.csv")
EXP2_SEEDS = (42, 7, 2026)

CONTROL_PREFIXES = (
    "negprb",
    "falsecode",
    "negcontrol",
    "unassigned",
    "blank",
    "systemcontrol",
)


def _sample_points_in_poly(poly, n: int, rng: np.random.Generator) -> np.ndarray:
    """Uniform rejection sampling inside a valid Shapely polygon."""
    if n <= 0 or poly is None or poly.is_empty or poly.area <= 0:
        return np.zeros((0, 2), dtype=np.float64)
    shapely.prepare(poly)
    minx, miny, maxx, maxy = poly.bounds
    collected = []
    total = 0
    batch = max(64, n * 4)
    for _ in range(80):
        xs = rng.uniform(minx, maxx, size=batch)
        ys = rng.uniform(miny, maxy, size=batch)
        pts = shapely.points(xs, ys)
        ok = shapely.contains(poly, pts)
        if np.any(ok):
            xy = np.column_stack([xs[ok], ys[ok]])
            collected.append(xy)
            total += len(xy)
            if total >= n:
                break
        batch = min(batch * 2, 50000)
    if not collected:
        cx, cy = poly.centroid.x, poly.centroid.y
        return np.column_stack([np.full(n, cx), np.full(n, cy)])
    out = np.vstack(collected)[:n]
    if len(out) < n:
        pad = out[rng.choice(len(out), size=n - len(out), replace=True)]
        out = np.vstack([out, pad])
    return out


def _hungarian_macro_f1_and_per_class(y_true: np.ndarray, y_pred: np.ndarray, n_true: int) -> tuple[float, dict[int, float]]:
    """Compute optimal 1-to-1 Hungarian cluster-matched F1 across ground-truth compartments."""
    k_pred = int(max(y_pred.max() + 1, n_true))
    f1_mat = np.zeros((n_true, k_pred), dtype=np.float64)
    for t_i in range(n_true):
        t_bin = (y_true == t_i)
        n_t = int(t_bin.sum())
        if n_t == 0:
            continue
        for p_j in range(k_pred):
            p_bin = (y_pred == p_j)
            n_p = int(p_bin.sum())
            if n_p == 0:
                continue
            inter = int(np.sum(t_bin & p_bin))
            if inter > 0:
                f1_mat[t_i, p_j] = 2.0 * inter / float(n_t + n_p)
    row_ind, col_ind = linear_sum_assignment(-f1_mat)
    per_cls = {int(r): float(f1_mat[r, c]) for r, c in zip(row_ind, col_ind)}
    for t_i in range(n_true):
        per_cls.setdefault(t_i, 0.0)
    return float(np.mean(list(per_cls.values()))), per_cls


# ===========================================================================
# Experiment 1: SimFISH Realistic Cell-Template 5-Compartment + CSR Benchmark
# ===========================================================================
def build_simfish_5compartment_bundle(
    template_geoms: list[Any],
    *,
    n_cells: int = 80,
    two_celltypes: bool = False,
    seed: int = 2026,
) -> tuple[dict[str, Any], np.ndarray]:
    """Synthesize realistic 5-compartment cells on real cell/nuclear morphology templates.

    Ground-truth compartments (spatially disjoint):
      0 = Nuclear interior (~24% spots)
      1 = Nucleolus / Nuclear body (~5% spots, rare dense sub-nuclear domain)
      2 = Perinuclear ER ring (~24% spots)
      3 = Bulk Cytosol (~40% spots)
      4 = Polarized Protrusion pocket (~7% spots, asymmetric peripheral microdomain)

    12% background cross-compartment diffusion mimics biological transcript leakage.
    When ``two_celltypes=True``, CellType A and CellType B have 4x differential baseline
    expression across gene subsets while sharing the same 5 subcellular compartments.
    """
    rng = np.random.default_rng(seed)
    # 60 genes assigned to the 5 compartment programs (15 nuc, 6 nucleolar, 14 ER, 18 cyto, 7 protrusion)
    prog_sizes = [15, 6, 14, 18, 7]
    genes_by_comp: list[list[str]] = []
    g_counter = 0
    for k_idx, sz in enumerate(prog_sizes):
        g_list = [f"G{g_counter + j:03d}" for j in range(sz)]
        genes_by_comp.append(g_list)
        g_counter += sz
    all_genes = [g for sub in genes_by_comp for g in sub]

    # Cell-type baseline multipliers on genes (tests robustness to intercellular expression shifts)
    ct_weights = []
    for ct in range(2):
        w = np.ones(len(all_genes), dtype=np.float64)
        if two_celltypes:
            for idx in range(len(all_genes)):
                if idx % 2 == ct:
                    w[idx] = 3.5
                else:
                    w[idx] = 0.5
        ct_weights.append(w)

    rows = []
    gt_labels = []
    cell_bounds: dict[str, pd.DataFrame] = {}
    nuc_bounds: dict[str, pd.DataFrame] = {}

    n_cols = int(math.ceil(math.sqrt(n_cells)))
    for c_i in range(n_cells):
        tmpl = template_geoms[c_i % len(template_geoms)]
        cid = f"simcell_{c_i:03d}"
        ct = c_i % 2

        b_minx, b_miny, b_maxx, b_maxy = tmpl.cell.bounds
        ox = (c_i % n_cols) * 900.0 - b_minx + 50.0
        oy = (c_i // n_cols) * 900.0 - b_miny + 50.0
        c_poly = shp_translate(tmpl.cell, xoff=ox, yoff=oy)
        n_poly = shp_translate(tmpl.nuclear, xoff=ox, yoff=oy)

        r_cell = _equivalent_radius(_safe_area(c_poly))
        r_nuc = _equivalent_radius(_safe_area(n_poly))

        # Carve 5 disjoint spatial regions inside c_poly:
        # 1) Nucleolus: small disk inside nucleus
        n_cx, n_cy = n_poly.centroid.x, n_poly.centroid.y
        ang_n = rng.uniform(0, 2 * np.pi)
        nucl_center = Point(n_cx + 0.25 * r_nuc * math.cos(ang_n), n_cy + 0.25 * r_nuc * math.sin(ang_n))
        poly_nucleolus = _safe_intersection(nucl_center.buffer(0.24 * r_nuc, quad_segs=24), n_poly)
        poly_nuc_main = _safe_difference(n_poly, nucl_center.buffer(0.28 * r_nuc, quad_segs=24))

        # 2) Perinuclear ER ring: outer buffer around n_poly clipped to cytoplasm
        cyto_full = _safe_difference(c_poly, n_poly)
        er_outer = _safe_buffer(n_poly, 0.38 * r_nuc)
        poly_er = _safe_intersection(cyto_full, er_outer)
        cyto_outer = _safe_difference(cyto_full, er_outer)

        # 3) Polarized Protrusion pocket: disk centered near the farthest boundary vertex of c_poly
        bx, by = np.asarray(c_poly.exterior.xy[0]), np.asarray(c_poly.exterior.xy[1])
        dists = np.hypot(bx - c_poly.centroid.x, by - c_poly.centroid.y)
        far_idx = int(np.argmax(dists))
        prot_pt = Point(
            0.22 * c_poly.centroid.x + 0.78 * bx[far_idx],
            0.22 * c_poly.centroid.y + 0.78 * by[far_idx],
        )
        poly_prot = _safe_intersection(cyto_outer, prot_pt.buffer(0.28 * r_cell, quad_segs=24))
        poly_cyto = _safe_difference(cyto_outer, prot_pt.buffer(0.30 * r_cell, quad_segs=24))
        if poly_cyto is None or _safe_area(poly_cyto) <= 0:
            poly_cyto = cyto_outer

        cx_v, cy_v = c_poly.exterior.xy
        nx_v, ny_v = n_poly.exterior.xy
        cell_bounds[cid] = pd.DataFrame({"x": list(cx_v), "y": list(cy_v)})
        nuc_bounds[cid] = pd.DataFrame({"x": list(nx_v), "y": list(ny_v)})

        comp_polys = [poly_nuc_main, poly_nucleolus, poly_er, poly_cyto, poly_prot]
        spot_quotas = [130, 30, 130, 220, 40]  # 550 spots/cell (~5.5% nucleolus, ~7.3% protrusion)

        # Build gene sampling distributions per compartment (88% compartment-specific + 12% cell background)
        bg_w = ct_weights[ct].copy()
        bg_p = bg_w / bg_w.sum()

        g_offset = 0
        comp_gene_probs = []
        for k_idx, g_list in enumerate(genes_by_comp):
            spec_w = np.zeros(len(all_genes), dtype=np.float64)
            spec_w[g_offset : g_offset + len(g_list)] = ct_weights[ct][g_offset : g_offset + len(g_list)]
            spec_p = spec_w / spec_w.sum()
            comp_gene_probs.append(0.88 * spec_p + 0.12 * bg_p)
            g_offset += len(g_list)

        for k_idx, (reg_poly, quota) in enumerate(zip(comp_polys, spot_quotas)):
            if reg_poly is None or _safe_area(reg_poly) <= 0:
                continue
            pts_xy = _sample_points_in_poly(reg_poly, quota, rng)
            sampled_genes = rng.choice(all_genes, size=len(pts_xy), p=comp_gene_probs[k_idx])
            for (px, py), g_name in zip(pts_xy, sampled_genes):
                rows.append((cid, g_name, float(px), float(py)))
                gt_labels.append(k_idx)

    df = pd.DataFrame(rows, columns=["cell", "gene", "x", "y"])
    bundle = {"data_df": df, "cell_boundary": cell_bounds, "nuclear_boundary": nuc_bounds}
    return bundle, np.asarray(gt_labels, dtype=np.int64)


def _run_methods_subset(bundle, *, n_clusters, cluster_range=(4, 7), threads=8, random_state=42, decomap_only=False):
    if decomap_only:
        from scripts.compartment_competitor_runners import run_method_decomap

        return {
            "DeCoMap": run_method_decomap(
                bundle,
                n_clusters=n_clusters,
                cluster_range=cluster_range,
                calibrate_csr=True,
                threads=threads,
                random_state=random_state,
            )
        }
    return run_all_three_compartment_methods(
        bundle, n_clusters=n_clusters, cluster_range=cluster_range, threads=threads, random_state=random_state
    )


def _upsert_decomap_rows(csv_path: Path, new_df: pd.DataFrame, decomap_only: bool) -> pd.DataFrame:
    if decomap_only and csv_path.exists():
        old_df = pd.read_csv(csv_path)
        kept = old_df[old_df["method"] != "DeCoMap"]
        merged = pd.concat([new_df, kept], ignore_index=True)
        merged.to_csv(csv_path, index=False)
        return merged
    new_df.to_csv(csv_path, index=False)
    return new_df


def run_exp1_simfish(threads: int = 8, decomap_only: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    print("=== [Exp 1] SimFISH 5-Compartment Recovery & CSR Null Calibration ===", flush=True)
    with open(SIM_TEMPLATE_PKL, "rb") as fh:
        sim_b = pickle.load(fh)
    tmpl_map = _load_decomap_geometries(sim_b, include_posthoc_nuclear_qc=True)
    templates = [
        g for g in tmpl_map.values()
        if g.cell is not None and g.nuclear is not None and 0.10 < (_safe_area(g.nuclear) / _safe_area(g.cell)) < 0.70
    ]

    regimes = [
        ("Homogeneous_SingleType", False),
        ("Heterogeneous_TwoCellTypes", True),
    ]
    exp1a_rows = []
    for reg_name, two_ct in regimes:
        for rep_seed in [2026, 2027, 2028]:
            bundle, gt = build_simfish_5compartment_bundle(
                templates, n_cells=80, two_celltypes=two_ct, seed=rep_seed
            )
            outs = _run_methods_subset(
                bundle, n_clusters=5, threads=threads, random_state=rep_seed, decomap_only=decomap_only
            )
            for m_name, r in outs.items():
                raw = r["raw_labels"]
                ari = float(adjusted_rand_score(gt, raw))
                nmi = float(normalized_mutual_info_score(gt, raw))
                macro_f1, cls_f1 = _hungarian_macro_f1_and_per_class(gt, raw, 5)
                exp1a_rows.append(
                    {
                        "regime": reg_name,
                        "seed": rep_seed,
                        "method": m_name,
                        "best_k": r["best_k"],
                        "transcript_ARI": round(ari, 4),
                        "transcript_NMI": round(nmi, 4),
                        "macro_F1": round(macro_f1, 4),
                        "nuclear_F1": round(cls_f1[0], 4),
                        "rare_nucleolus_F1": round(cls_f1[1], 4),
                        "perinuclear_ER_F1": round(cls_f1[2], 4),
                        "cytosol_F1": round(cls_f1[3], 4),
                        "rare_protrusion_F1": round(cls_f1[4], 4),
                        "structured_spot_fraction": round(r["structured_spot_fraction"], 4),
                        "elapsed_seconds": round(r["elapsed_seconds"], 3),
                        "peak_rss_gb": round(r["peak_rss_gb"], 3),
                    }
                )
        print(f"  [Exp 1A] finished regime={reg_name}", flush=True)

    df_1a = pd.DataFrame(exp1a_rows)

    # Exp 1B: Pure CSR negative controls (1: Uniform spatial CSR inside real cell templates;
    #                                     2: Within-cell gene-label permutation on real MERFISH U2OS cells)
    exp1b_rows = []
    rng_csr = np.random.default_rng(999)
    sim_bundle, _ = build_simfish_5compartment_bundle(templates, n_cells=80, two_celltypes=False, seed=2026)
    # 1) Uniform spatial coordinates inside each real cell polygon preserving exact cell gene counts
    csr_rows = []
    sim_geoms = _load_decomap_geometries(sim_bundle, include_posthoc_nuclear_qc=False)
    for cid, grp in sim_bundle["data_df"].groupby("cell", sort=False):
        pts_u = _sample_points_in_poly(sim_geoms[cid].cell, len(grp), rng_csr)
        for (ux, uy), g_name in zip(pts_u, grp["gene"].to_numpy()):
            csr_rows.append((cid, g_name, float(ux), float(uy)))
    csr_sim_bundle = {
        "data_df": pd.DataFrame(csr_rows, columns=["cell", "gene", "x", "y"]),
        "cell_boundary": sim_bundle["cell_boundary"],
        "nuclear_boundary": {},
    }

    # 2) Real U2OS coordinates (60 cells) with within-cell randomly permuted gene labels
    with open(U2OS_D1_PKL, "rb") as fh:
        u2os_b = pickle.load(fh)
    u_df = u2os_b["data_df"].copy()
    u_df["cell"] = u_df["cell"].astype(str)
    u_cids = set(sorted(u_df["cell"].unique())[:60])
    u_sub = u_df[u_df["cell"].isin(u_cids)].reset_index(drop=True)
    for cid, idx in u_sub.groupby("cell").indices.items():
        u_sub.iloc[idx, u_sub.columns.get_loc("gene")] = rng_csr.permutation(
            u_sub.iloc[idx]["gene"].to_numpy()
        )
    csr_u2os_bundle = {
        "data_df": u_sub,
        "cell_boundary": {k: v for k, v in u2os_b["cell_boundary"].items() if str(k) in u_cids},
        "nuclear_boundary": {},
    }

    for null_name, nbun in [
        ("SimFISH_SpatialUniform_CSR", csr_sim_bundle),
        ("U2OS_WithinCell_Permuted_CSR", csr_u2os_bundle),
    ]:
        outs_null = _run_methods_subset(
            nbun, n_clusters="auto", cluster_range=(4, 7), threads=threads, random_state=42, decomap_only=decomap_only
        )
        c_groups = nbun["data_df"].groupby("cell").indices
        for m_name, r in outs_null.items():
            lbl = r["labels"]
            false_spot_rate = float(np.mean(lbl >= 0))
            false_cell_rate = float(
                np.mean([np.mean(lbl[idx] >= 0) >= 0.50 for idx in c_groups.values()])
            )
            exp1b_rows.append(
                {
                    "null_benchmark": null_name,
                    "method": m_name,
                    "n_cells": len(c_groups),
                    "n_transcripts": len(lbl),
                    "false_structured_spot_rate": round(false_spot_rate, 4),
                    "false_structured_cell_rate": round(false_cell_rate, 4),
                    "elapsed_seconds": round(r["elapsed_seconds"], 3),
                }
            )
        print(f"  [Exp 1B] finished null={null_name}", flush=True)

    df_1b = pd.DataFrame(exp1b_rows)
    df_1a = _upsert_decomap_rows(OUT_DIR / "exp1a_simfish_5compartments.csv", df_1a, decomap_only)
    df_1b = _upsert_decomap_rows(OUT_DIR / "exp1b_csr_null_calibration.csv", df_1b, decomap_only)
    return df_1a, df_1b


# ===========================================================================
# Experiment 2: Blind DAPI Nuclear Mask Recovery Across 3 Real Platforms
# ===========================================================================
def _load_real_dapi_dataset(name: str, seed: int = 2026) -> tuple[dict[str, Any], np.ndarray]:
    """Load QC-valid real cells with true DAPI nuclear ground truth (kept strictly outside input bundle)."""
    rng = np.random.default_rng(seed)
    if name == "MERFISH_U2OS":
        with open(U2OS_D1_PKL, "rb") as fh:
            b = pickle.load(fh)
        n_pick = 150
        max_genes = None
    elif name == "CosMx_HEK293T":
        with open(COSMX_HEK_PKL, "rb") as fh:
            b = pickle.load(fh)
        n_pick = 300
        max_genes = 500
    elif name == "Xenium_HEK293T":
        with open(XENIUM_HEK_CACHE_PKL, "rb") as fh:
            b = pickle.load(fh)
        n_pick = 300
        max_genes = None
    else:
        raise ValueError(name)

    df = b["data_df"][["cell", "gene", "x", "y"]].copy()
    df["cell"] = df["cell"].astype(str)
    df["gene"] = df["gene"].astype(str)
    df = df[~df["gene"].str.lower().str.startswith(CONTROL_PREFIXES)].reset_index(drop=True)

    cb_str = {str(k): v for k, v in b["cell_boundary"].items()}
    nb_str = {str(k): v for k, v in b["nuclear_boundary"].items()}
    common_cids = sorted(set(df["cell"].unique()) & set(cb_str.keys()) & set(nb_str.keys()))

    # Keep cells with >= 80 transcripts and plausible N/C area ratio in [0.15, 0.85]
    geoms_all = _load_decomap_geometries(
        {"data_df": df, "cell_boundary": cb_str, "nuclear_boundary": nb_str},
        include_posthoc_nuclear_qc=True,
    )
    spot_cnts = df.groupby("cell", observed=True).size()
    valid_cids = []
    for cid in common_cids:
        g = geoms_all.get(cid)
        if g is None or g.cell is None or g.nuclear is None:
            continue
        ac = _safe_area(g.cell)
        an = _safe_area(g.nuclear)
        if ac <= 0 or an <= 0:
            continue
        nc = an / ac
        if 0.15 <= nc <= 0.85 and int(spot_cnts.get(cid, 0)) >= 80:
            valid_cids.append(cid)

    chosen = sorted(rng.choice(valid_cids, size=min(n_pick, len(valid_cids)), replace=False).tolist())
    cset = set(chosen)
    sub_df = df[df["cell"].isin(cset)].reset_index(drop=True)

    if max_genes is not None and sub_df["gene"].nunique() > max_genes:
        top_g = set(sub_df["gene"].value_counts().head(max_genes).index.tolist())
        sub_df = sub_df[sub_df["gene"].isin(top_g)].reset_index(drop=True)

    sub_cb = {cid: cb_str[cid] for cid in chosen}
    # Compute ground-truth DAPI nuclear indicator with shapely.prepare
    in_dapi = np.zeros(len(sub_df), dtype=np.int64)
    xy = sub_df[["x", "y"]].to_numpy(dtype=np.float64)
    for cid, rows in sub_df.groupby("cell", sort=False).indices.items():
        nuc_poly = geoms_all[cid].nuclear
        shapely.prepare(nuc_poly)
        pts = shapely.points(xy[rows, 0], xy[rows, 1])
        in_dapi[rows] = (shapely.contains(nuc_poly, pts) | shapely.touches(nuc_poly, pts)).astype(np.int64)

    blind_bundle = {"data_df": sub_df, "cell_boundary": sub_cb, "nuclear_boundary": {}}
    return blind_bundle, in_dapi


def run_exp2_blind_dapi_recovery(threads: int = 8, decomap_only: bool = False) -> pd.DataFrame:
    print("=== [Exp 2] Blind DAPI Nuclear Mask Recovery Across 3 Real Platforms ===", flush=True)
    datasets = ["MERFISH_U2OS", "CosMx_HEK293T", "Xenium_HEK293T"]
    rows = []

    for ds_name in datasets:
        if ds_name == "Xenium_HEK293T" and not XENIUM_HEK_CACHE_PKL.exists():
            continue
        blind_bundle, in_dapi = _load_real_dapi_dataset(ds_name, seed=2026)
        df = blind_bundle["data_df"]
        n_cells = int(df["cell"].nunique())
        n_spots = len(df)
        n_genes = int(df["gene"].nunique())
        print(
            f"  [Exp 2] {ds_name}: cells={n_cells}, spots={n_spots:,}, genes={n_genes}, dapi_frac={in_dapi.mean():.3f}",
            flush=True,
        )

        cell_idx_map = df.groupby("cell", sort=False).indices
        # Nuclear IoU on sparse panels moves by ~0.03-0.04 between seeds, so every method is run per seed.
        method_runs = []
        for seed in EXP2_SEEDS:
            outs = _run_methods_subset(
                blind_bundle, n_clusters=5, threads=threads, random_state=seed, decomap_only=decomap_only
            )
            method_runs.extend((seed, m_name, r) for m_name, r in outs.items())

        for seed, m_name, r in method_runs:
            raw = r["raw_labels"]
            k = r["best_k"]

            # 1) Best single cluster IoU & F1
            single_ious = []
            single_f1s = []
            for c_id in range(k):
                p_bin = (raw == c_id).astype(np.int64)
                single_ious.append(float(jaccard_score(in_dapi, p_bin)))
                single_f1s.append(float(f1_score(in_dapi, p_bin)))
            best_single_idx = int(np.argmax(single_ious))
            best_single_iou = single_ious[best_single_idx]
            best_single_f1 = single_f1s[best_single_idx]

            # 2) Standard nuclear compartment set (clusters with nuclear precision >= 0.50, fallback best cluster)
            nuc_clusters = [
                c_id for c_id in range(k)
                if np.any(raw == c_id) and float(in_dapi[raw == c_id].mean()) >= 0.50
            ]
            if not nuc_clusters:
                nuc_clusters = [best_single_idx]
            pred_nuc = np.isin(raw, nuc_clusters).astype(np.int64)

            union_iou = float(jaccard_score(in_dapi, pred_nuc))
            union_f1 = float(f1_score(in_dapi, pred_nuc))
            bin_ari = float(adjusted_rand_score(in_dapi, pred_nuc))

            # 3) Per-cell nuclear IoU distribution (measures cell-to-cell consistency)
            cell_ious = []
            for rows_c in cell_idx_map.values():
                yt_c = in_dapi[rows_c]
                yp_c = pred_nuc[rows_c]
                denom_u = int(np.sum((yt_c == 1) | (yp_c == 1)))
                if denom_u > 0:
                    cell_ious.append(float(np.sum((yt_c == 1) & (yp_c == 1)) / denom_u))
            med_cell_iou = float(np.median(cell_ious)) if cell_ious else 0.0
            iqr_cell_iou = float(np.percentile(cell_ious, 75) - np.percentile(cell_ious, 25)) if cell_ious else 0.0

            rows.append(
                {
                    "dataset": ds_name,
                    "method": m_name,
                    "seed": seed,
                    "n_cells": n_cells,
                    "n_transcripts": n_spots,
                    "n_genes": n_genes,
                    "best_k": k,
                    "single_cluster_IoU": round(best_single_iou, 4),
                    "single_cluster_F1": round(best_single_f1, 4),
                    "nuclear_IoU": round(union_iou, 4),
                    "nuclear_F1": round(union_f1, 4),
                    "nuclear_binary_ARI": round(bin_ari, 4),
                    "per_cell_median_IoU": round(med_cell_iou, 4),
                    "per_cell_IQR_IoU": round(iqr_cell_iou, 4),
                    "elapsed_seconds": round(r["elapsed_seconds"], 3),
                    "peak_rss_gb": round(r["peak_rss_gb"], 3),
                }
            )
            print(
                f"    {m_name:18s} seed={seed} | single_IoU={best_single_iou:.4f} | nuc_IoU={union_iou:.4f} | "
                f"F1={union_f1:.4f} | ARI={bin_ari:.4f} | med_cell_IoU={med_cell_iou:.4f} | time={r['elapsed_seconds']:.2f}s",
                flush=True,
            )

    df_2 = pd.DataFrame(rows)
    df_2 = _upsert_decomap_rows(OUT_DIR / "exp2_real_blind_dapi_recovery.csv", df_2, decomap_only)
    return df_2


# ===========================================================================
# Experiment 3: U2OS 3-Replicate Concordance & Wet-Lab Organelle Enrichment
# ===========================================================================
def _load_wetlab_reference_sets(shared_genes: tuple[str, ...]) -> dict[str, set[str]]:
    """Build orthogonal wet-lab positive gene sets on the U2OS gene universe."""
    univ = {g.upper(): g for g in shared_genes}
    ref_sets: dict[str, set[str]] = {}

    if XIA_CSV.exists():
        xia = pd.read_csv(XIA_CSV)
        piv_x = xia.pivot_table(index="target", columns="source", values="weight", aggfunc="mean")
        # Top quartile truly enriched genes in Xia 2019 MERFISH 10k U2OS (with ER staining)
        er_pos = {univ[g.upper()] for g in piv_x.index[piv_x["ER"] >= 0.50] if g.upper() in univ}
        nuc_pos = {univ[g.upper()] for g in piv_x.index[piv_x["Nucleus"] >= 1.00] if g.upper() in univ}
        if len(er_pos) >= 5:
            ref_sets["Xia2019_ER"] = er_pos
        if len(nuc_pos) >= 5:
            ref_sets["Xia2019_Nucleus"] = nuc_pos

    if FAZAL_CSV.exists():
        faz = pd.read_csv(FAZAL_CSV)
        piv_f = faz.pivot_table(index="target", columns="source", values="weight", aggfunc="mean")
        piv_fc = piv_f.sub(piv_f.mean(axis=1), axis=0)
        erm_pos = {univ[g.upper()] for g in piv_fc.index[piv_fc["ERM"] >= 0.45] if g.upper() in univ}
        nuc_f_pos = {univ[g.upper()] for g in piv_fc.index[piv_fc["Nucleus"] >= 0.45] if g.upper() in univ}
        if len(erm_pos) >= 5:
            ref_sets["Fazal2019_APEX_ERM"] = erm_pos
        if len(nuc_f_pos) >= 5:
            ref_sets["Fazal2019_APEX_Nucleus"] = nuc_f_pos

    return ref_sets


def run_exp3_u2os_replicates(threads: int = 8, decomap_only: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    print("=== [Exp 3] MERFISH U2OS 3-Replicate Concordance & Organelle Enrichment ===", flush=True)
    rng = np.random.default_rng(2026)

    # Load 3 replicates and intersect shared non-control genes
    rep_raw = {}
    gene_sets = []
    for rep_name, pkl_path in U2OS_REP_PATHS.items():
        with open(pkl_path, "rb") as fh:
            b = pickle.load(fh)
        df = b["data_df"][["cell", "gene", "x", "y"]].copy()
        df["cell"] = df["cell"].astype(str)
        df["gene"] = df["gene"].astype(str)
        df = df[~df["gene"].str.lower().str.startswith(CONTROL_PREFIXES)]
        gene_sets.append(set(df["gene"].unique()))
        rep_raw[rep_name] = (df, b["cell_boundary"])

    shared_genes = tuple(sorted(set.intersection(*gene_sets)))
    print(f"  [Exp 3] Shared U2OS panel genes across 3 replicates: {len(shared_genes)}", flush=True)

    rep_results: dict[str, dict[str, dict[str, Any]]] = {}
    for rep_name, (df, cb) in rep_raw.items():
        df_sh = df[df["gene"].isin(set(shared_genes))].copy()
        cb_str = {str(k): v for k, v in cb.items()}
        valid_cids = sorted(set(df_sh["cell"].unique()) & set(cb_str.keys()))
        chosen = sorted(rng.choice(valid_cids, size=min(300, len(valid_cids)), replace=False).tolist())
        cset = set(chosen)
        sub_df = df_sh[df_sh["cell"].isin(cset)].reset_index(drop=True)
        sub_cb = {cid: cb_str[cid] for cid in chosen}

        bun = {"data_df": sub_df, "cell_boundary": sub_cb, "nuclear_boundary": {}}
        rep_results[rep_name] = _run_methods_subset(
            bun, n_clusters=5, threads=threads, random_state=42, decomap_only=decomap_only
        )
        print(f"  [Exp 3] Completed {rep_name} ({len(chosen)} cells, {len(sub_df):,} spots)", flush=True)

    methods = ["DeCoMap"] if decomap_only else ["DeCoMap", "Bento (RNAflux)", "Points2Regions"]
    repro_rows = []
    for m_name in methods:
        pair_cos = []
        pair_spear = []
        pair_gene_ari = []
        for r1, r2 in combinations(["Rep2", "Rep3", "Rep4"], 2):
            P1 = rep_results[r1][m_name]["gene_profiles"]  # (K, G)
            P2 = rep_results[r2][m_name]["gene_profiles"]  # (K, G)
            # L2 normalize rows for cosine matching
            P1_n = P1 / np.maximum(np.linalg.norm(P1, axis=1, keepdims=True), 1e-8)
            P2_n = P2 / np.maximum(np.linalg.norm(P2, axis=1, keepdims=True), 1e-8)
            cos_mat = P1_n @ P2_n.T
            ri, ci = linear_sum_assignment(-cos_mat)
            mean_cos = float(np.mean(cos_mat[ri, ci]))
            spears = [float(spearmanr(P1[i], P2[j]).statistic) for i, j in zip(ri, ci)]
            mean_sp = float(np.mean(spears))

            # Dominant compartment assignment per gene (G,) across replicates
            dom1 = np.argmax(P1, axis=0)
            dom2 = np.argmax(P2, axis=0)
            g_ari = float(adjusted_rand_score(dom1, dom2))

            pair_cos.append(mean_cos)
            pair_spear.append(mean_sp)
            pair_gene_ari.append(g_ari)

        repro_rows.append(
            {
                "method": m_name,
                "mean_profile_cosine": round(float(np.mean(pair_cos)), 4),
                "min_profile_cosine": round(float(np.min(pair_cos)), 4),
                "mean_profile_spearman": round(float(np.mean(pair_spear)), 4),
                "mean_gene_compartment_ARI": round(float(np.mean(pair_gene_ari)), 4),
                "mean_elapsed_seconds": round(
                    float(np.mean([rep_results[rn][m_name]["elapsed_seconds"] for rn in ["Rep2", "Rep3", "Rep4"]])),
                    3,
                ),
            }
        )

    df_3a = pd.DataFrame(repro_rows)

    # Orthogonal organelle marker recovery AUROC across the 3 replicates
    ref_sets = _load_wetlab_reference_sets(shared_genes)
    enrich_rows = []
    for ref_name, pos_genes in ref_sets.items():
        y_true = np.array([1 if g in pos_genes else 0 for g in shared_genes], dtype=int)
        if y_true.sum() < 3 or y_true.sum() == len(y_true):
            continue
        for m_name in methods:
            rep_aurocs = []
            for rn in ["Rep2", "Rep3", "Rep4"]:
                P = rep_results[rn][m_name]["gene_profiles"]  # (K, G)
                best_auroc = max(float(roc_auc_score(y_true, P[k])) for k in range(P.shape[0]))
                rep_aurocs.append(best_auroc)
            enrich_rows.append(
                {
                    "reference_set": ref_name,
                    "n_positive_genes": int(y_true.sum()),
                    "n_panel_genes": len(shared_genes),
                    "method": m_name,
                    "mean_best_domain_AUROC": round(float(np.mean(rep_aurocs)), 4),
                    "sd_AUROC": round(float(np.std(rep_aurocs)), 4),
                }
            )

    df_3b = pd.DataFrame(enrich_rows)
    df_3a = _upsert_decomap_rows(OUT_DIR / "exp3a_u2os_replicate_concordance.csv", df_3a, decomap_only)
    df_3b = _upsert_decomap_rows(OUT_DIR / "exp3b_u2os_organelle_enrichment.csv", df_3b, decomap_only)
    return df_3a, df_3b


def build_master_summary() -> pd.DataFrame:
    """Aggregate headline metrics across Exp 1, Exp 2, and Exp 3 into one master table."""
    methods = ["DeCoMap", "Bento (RNAflux)", "Points2Regions"]
    df1a = pd.read_csv(OUT_DIR / "exp1a_simfish_5compartments.csv")
    df1b = pd.read_csv(OUT_DIR / "exp1b_csr_null_calibration.csv")
    df2 = pd.read_csv(OUT_DIR / "exp2_real_blind_dapi_recovery.csv")
    df3a = pd.read_csv(OUT_DIR / "exp3a_u2os_replicate_concordance.csv")
    df3b = pd.read_csv(OUT_DIR / "exp3b_u2os_organelle_enrichment.csv")

    rows = []
    for m in methods:
        s1a_hom = df1a[(df1a["method"] == m) & (df1a["regime"] == "Homogeneous_SingleType")]
        s1a_het = df1a[(df1a["method"] == m) & (df1a["regime"] == "Heterogeneous_TwoCellTypes")]
        s1b = df1b[df1b["method"] == m]
        s2 = df2[df2["method"] == m]
        s3a = df3a[df3a["method"] == m].iloc[0]
        s3b = df3b[df3b["method"] == m]

        rec = {
            "method": m,
            "SimFISH_Homogeneous_ARI": round(float(s1a_hom["transcript_ARI"].mean()), 4),
            "SimFISH_TwoCellTypes_ARI": round(float(s1a_het["transcript_ARI"].mean()), 4),
            "SimFISH_RareNucleolus_F1": round(float(df1a[df1a["method"] == m]["rare_nucleolus_F1"].mean()), 4),
            "SimFISH_RareProtrusion_F1": round(float(df1a[df1a["method"] == m]["rare_protrusion_F1"].mean()), 4),
            "CSR_Specificity_1_minus_FPR": round(1.0 - float(s1b["false_structured_spot_rate"].mean()), 4),
            "Mean_Real_Blind_DAPI_IoU": round(float(s2["nuclear_IoU"].mean()), 4),
            "Mean_Real_Blind_DAPI_ARI": round(float(s2["nuclear_binary_ARI"].mean()), 4),
            "U2OS_3Rep_Profile_Cosine": float(s3a["mean_profile_cosine"]),
            "U2OS_3Rep_GeneDomain_ARI": float(s3a["mean_gene_compartment_ARI"]),
            "WetLab_Organelle_Mean_AUROC": round(float(s3b["mean_best_domain_AUROC"].mean()), 4),
            "Mean_Runtime_Seconds": round(float(s2["elapsed_seconds"].mean()), 2),
        }
        rows.append(rec)

    master = pd.DataFrame(rows)
    master.to_csv(OUT_DIR / "master_3methods_compartment_summary.csv", index=False)
    return master


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", choices=["all", "exp1", "exp2", "exp3", "master"], default="all")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--decomap-only", action="store_true")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.part in ("all", "exp1"):
        run_exp1_simfish(threads=args.threads, decomap_only=args.decomap_only)
    if args.part in ("all", "exp2"):
        run_exp2_blind_dapi_recovery(threads=args.threads, decomap_only=args.decomap_only)
    if args.part in ("all", "exp3"):
        run_exp3_u2os_replicates(threads=args.threads, decomap_only=args.decomap_only)
    if args.part in ("all", "master"):
        master = build_master_summary()
        print("\n=== MASTER 3-METHOD COMPARTMENT BENCHMARK SUMMARY ===", flush=True)
        print(master.to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
