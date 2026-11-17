"""Comprehensive benchmark of DeCoLoc vs InSTAnT (PP+CPB) and SCRIN (cooccurrence hypergeometric).

Experiment 1: SimFISH controlled mechanism benchmark (built on real cell/nuclear shapes from
              data/simulated_data_dict.pkl) testing:
              - P1: Sparse micro-colocalization (1-5 copies/cell)
              - P2: Moderate intra-compartment micro-colocalization
              - P3: >100x expression-imbalanced true lncRNA-mRNA interaction
              - N1: Co-compartment independent localization (ER/perinuclear & nuclear)
              - N2: Independent homotypic transcriptional burst foci
              - N3: Subpopulation density / co-expression confounder (Simpson's paradox)

Experiment 2: Real U2OS MERFISH (data/Dataset1_merfish_u2os_data_dict.pkl) with two-tier
              intracellular null permutations:
              - Original unpermuted U2OS data
              - Tier-1 null: whole-cell gene-label permutation (destroys all spatial preference)
              - Tier-2 null: within-compartment gene-label permutation (preserves 4-region
                compartment localization while destroying intra-compartment micro-interactions)
              Plus end-to-end wall-clock runtime comparison (including 4-region annotation).
"""

from __future__ import annotations

from pathlib import Path
import pickle
import sys
import time

import numpy as np
import pandas as pd
from scipy import sparse, stats
from scipy.spatial import cKDTree
import shapely
from shapely.affinity import scale as shp_scale, translate as shp_translate
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score

STRAND_ROOT = Path(__file__).resolve().parents[1]
if str(STRAND_ROOT) not in sys.path:
    sys.path.insert(0, str(STRAND_ROOT))

from tools.decoloc.api import prefilter_bundle_for_decoloc, run_decoloc_from_bundle
from tools.decoloc.core import assign_disjoint_transcript_regions, compute_cell_region_thresholds
from tools.engines.coloc_region_annotation import (
    CellGeometry,
    _equivalent_radius,
    _load_cell_geometries,
    _safe_area,
    _safe_buffer,
    _safe_difference,
    _safe_intersection,
    annotate_colocalized_pairs_regions,
)
from tools.instant_optimized.adapter import run_optimized_colocalization_from_bundle

OUT_DIR = STRAND_ROOT / "benchmarks" / "decoloc_v1"
THREADS = 16


# ---------------------------------------------------------------------------
# Faithful SCRIN (cooccurrence hypergeometric test) implementation
# ---------------------------------------------------------------------------
_SCRIN_COORDS: np.ndarray | None = None
_SCRIN_GENES: np.ndarray | None = None
_SCRIN_OFFSETS: np.ndarray | None = None
_SCRIN_RADIUS: float = 10.0


def _init_scrin_worker(coords: np.ndarray, genes: np.ndarray, offsets: np.ndarray, radius: float) -> None:
    global _SCRIN_COORDS, _SCRIN_GENES, _SCRIN_OFFSETS, _SCRIN_RADIUS
    _SCRIN_COORDS = coords
    _SCRIN_GENES = genes
    _SCRIN_OFFSETS = offsets
    _SCRIN_RADIUS = radius


def _scrin_cell_worker(c_idx: int):
    s0, s1 = int(_SCRIN_OFFSETS[c_idx]), int(_SCRIN_OFFSETS[c_idx + 1])
    n_spots = s1 - s0
    if n_spots < 2:
        if n_spots == 1:
            return (
                c_idx,
                np.array([int(_SCRIN_GENES[s0])], dtype=np.int32),
                np.array([1], dtype=np.int64),
                np.array([0], dtype=np.int64),
                np.empty(0, dtype=np.int32),
                np.empty(0, dtype=np.int32),
                np.empty(0, dtype=np.int64),
            )
        return (
            c_idx,
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int64),
        )

    coords = _SCRIN_COORDS[s0:s1]
    genes = _SCRIN_GENES[s0:s1]
    ug, lg = np.unique(genes, return_inverse=True)
    g_cnt = np.bincount(lg, minlength=len(ug)).astype(np.int64)
    g_deg = np.zeros(len(ug), dtype=np.int64)

    pairs = cKDTree(coords).query_pairs(r=_SCRIN_RADIUS, output_type="ndarray")
    if len(pairs) == 0:
        return (
            c_idx,
            ug.astype(np.int32),
            g_cnt,
            g_deg,
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int64),
        )

    lp = lg[pairs[:, 0]]
    lq = lg[pairs[:, 1]]
    # In SCRIN, every neighbor molecule within r around A increments gene_around(A).
    np.add.at(g_deg, lp, 1)
    np.add.at(g_deg, lq, 1)

    het = lp != lq
    if not np.any(het):
        return (
            c_idx,
            ug.astype(np.int32),
            g_cnt,
            g_deg,
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int64),
        )

    u_min = np.minimum(lp[het], lq[het])
    u_max = np.maximum(lp[het], lq[het])
    pkey = u_min.astype(np.int64) * len(ug) + u_max.astype(np.int64)
    uk, inv = np.unique(pkey, return_inverse=True)
    k_vals = np.bincount(inv, minlength=len(uk)).astype(np.int64)
    gi = ug[(uk // len(ug)).astype(np.int32)].astype(np.int32)
    gj = ug[(uk % len(ug)).astype(np.int32)].astype(np.int32)
    return (c_idx, ug.astype(np.int32), g_cnt, g_deg, gi, gj, k_vals)


def run_scrin_cooccurrence(
    df: pd.DataFrame,
    *,
    radius: float,
    q_threshold: float = 1e-5,
    max_expr_ratio: float | None = 100.0,
    pair_keep: str = "last",
    min_gene_number: int = 5,
    threads: int = 16,
) -> tuple[pd.DataFrame, float]:
    """Run SCRIN's cooccurrence-background hypergeometric test + BH correction + deduplication."""
    import multiprocessing as mp

    t0 = time.perf_counter()
    work = df[["cell", "gene", "x", "y"]].copy()
    work["cell"] = work["cell"].astype(str)
    work["gene"] = work["gene"].astype(str)

    g_totals = work.groupby("gene", observed=True).size()
    kept_genes = sorted(g_totals.index[g_totals >= int(min_gene_number)])
    work = work[work["gene"].isin(set(kept_genes))].reset_index(drop=True)

    cell_cat = pd.Categorical(work["cell"], categories=sorted(work["cell"].unique()))
    gene_cat = pd.Categorical(work["gene"], categories=kept_genes)
    c_codes = np.asarray(cell_cat.codes, dtype=np.int32)
    g_codes = np.asarray(gene_cat.codes, dtype=np.int32)
    genes = tuple(gene_cat.categories)
    num_cells = len(cell_cat.categories)
    num_genes = len(genes)

    order = np.argsort(c_codes, kind="mergesort")
    c_codes = c_codes[order]
    g_codes = g_codes[order]
    coords = work[["x", "y"]].to_numpy(dtype=np.float64)[order]

    offsets = np.zeros(num_cells + 1, dtype=np.int64)
    np.cumsum(np.bincount(c_codes, minlength=num_cells), out=offsets[1:])

    cg_c, cg_g, cg_cnt, cg_deg = [], [], [], []
    edge_i, edge_j, edge_k = [], [], []

    init_args = (coords, g_codes, offsets, float(radius))
    if threads <= 1 or num_cells < 8:
        _init_scrin_worker(*init_args)
        results = [_scrin_cell_worker(c) for c in range(num_cells)]
    else:
        ctx = mp.get_context("fork")
        with ctx.Pool(processes=threads, initializer=_init_scrin_worker, initargs=init_args) as pool:
            results = list(pool.imap_unordered(_scrin_cell_worker, range(num_cells), chunksize=16))

    for c_idx, ug, g_cnt, g_deg, gi, gj, kv in results:
        if len(ug):
            cg_c.append(np.full(len(ug), c_idx, dtype=np.int32))
            cg_g.append(ug)
            cg_cnt.append(g_cnt)
            cg_deg.append(g_deg)
        if len(gi):
            edge_i.append(gi)
            edge_j.append(gj)
            edge_k.append(kv)

    all_c = np.concatenate(cg_c)
    all_g = np.concatenate(cg_g)
    all_cnt = np.concatenate(cg_cnt).astype(np.float64)
    all_deg = np.concatenate(cg_deg).astype(np.float64)
    cell_N = np.bincount(all_c, weights=all_cnt, minlength=num_cells)

    shape = (num_cells, num_genes)
    mat_pres = sparse.csr_matrix((np.ones(len(all_c), dtype=np.float64), (all_c, all_g)), shape=shape)
    mat_cnt = sparse.csr_matrix((all_cnt, (all_c, all_g)), shape=shape)
    mat_deg = sparse.csr_matrix((all_deg, (all_c, all_g)), shape=shape)
    mat_cellN = sparse.csr_matrix((cell_N[all_c], (all_c, all_g)), shape=shape)

    # Directed statistics over co-expressing cells C_{AB}:
    # n_{A->B} = sum_{c in C_AB} deg_{cA} = (mat_deg.T @ mat_pres)[A, B]
    # M_{A->B} = sum_{c in C_AB} cnt_{cB} = (mat_pres.T @ mat_cnt)[A, B]
    # N_{A->B} = sum_{c in C_AB} N_c      = (mat_cellN.T @ mat_pres)[A, B]
    n_mat = (mat_deg.T @ mat_pres).toarray()
    M_mat = (mat_pres.T @ mat_cnt).toarray()
    N_mat = (mat_cellN.T @ mat_pres).toarray()

    ei = np.concatenate(edge_i)
    ej = np.concatenate(edge_j)
    ek = np.concatenate(edge_k)
    pid = ei.astype(np.int64) * num_genes + ej.astype(np.int64)
    upid, inv = np.unique(pid, return_inverse=True)
    k_undir = np.bincount(inv, weights=ek, minlength=len(upid)).astype(np.int64)
    ui = (upid // num_genes).astype(np.int32)
    uj = (upid % num_genes).astype(np.int32)

    # Build directed pairs (A->B and B->A) for pairs with k >= 1 (--min_neighbor_number 1).
    dir_A = np.concatenate([ui, uj])
    dir_B = np.concatenate([uj, ui])
    dir_k = np.concatenate([k_undir, k_undir])

    global_gene_tot = np.asarray(mat_cnt.sum(axis=0)).ravel()
    if max_expr_ratio is not None:
        ratio = np.maximum(global_gene_tot[dir_A], 1.0) / np.maximum(global_gene_tot[dir_B], 1.0)
        expr_ok = (ratio >= (1.0 / max_expr_ratio)) & (ratio <= max_expr_ratio)
        dir_A, dir_B, dir_k = dir_A[expr_ok], dir_B[expr_ok], dir_k[expr_ok]

    dir_n = np.round(n_mat[dir_A, dir_B]).astype(np.int64)
    dir_M = np.round(M_mat[dir_A, dir_B]).astype(np.int64)
    dir_N = np.round(N_mat[dir_A, dir_B]).astype(np.int64)

    valid = (dir_k >= 1) & (dir_n >= dir_k) & (dir_M >= dir_k) & (dir_N >= dir_n) & (dir_N >= dir_M)
    dir_A, dir_B, dir_k = dir_A[valid], dir_B[valid], dir_k[valid]
    dir_n, dir_M, dir_N = dir_n[valid], dir_M[valid], dir_N[valid]

    pvals = stats.hypergeom.sf(dir_k - 1, dir_N, dir_M, dir_n)
    pvals = np.clip(pvals, 0.0, 1.0)

    # Total tested directed pairs with co-expression (matching SCRIN's pair evaluation scope).
    coexpr_dense = (mat_pres.T @ mat_pres).toarray()
    np.fill_diagonal(coexpr_dense, 0.0)
    if max_expr_ratio is not None:
        r_mat = global_gene_tot[:, None] / np.maximum(global_gene_tot[None, :], 1.0)
        valid_scope = (coexpr_dense > 0) & (r_mat >= 1.0 / max_expr_ratio) & (r_mat <= max_expr_ratio)
    else:
        valid_scope = coexpr_dense > 0
    total_dir_tests = max(int(valid_scope.sum()), len(pvals))

    # BH adjustment over directed tests.
    order_p = np.argsort(pvals, kind="mergesort")
    ranks = np.arange(1, len(pvals) + 1, dtype=np.float64)
    q_sorted = np.minimum.accumulate(((total_dir_tests * pvals[order_p]) / ranks)[::-1])[::-1]
    qvals = np.empty_like(pvals)
    qvals[order_p] = np.clip(q_sorted, 0.0, 1.0)

    enrich = (dir_k.astype(np.float64) * dir_N.astype(np.float64)) / np.maximum(
        dir_n.astype(np.float64) * dir_M.astype(np.float64), 1.0
    )

    gene_arr = np.asarray(genes, dtype=object)
    g1 = np.minimum(gene_arr[dir_A], gene_arr[dir_B])
    g2 = np.maximum(gene_arr[dir_A], gene_arr[dir_B])
    dir_df = pd.DataFrame(
        {
            "gene_1": g1,
            "gene_2": g2,
            "pvalue": pvals,
            "qvalue_BH": qvals,
            "enrichment_ratio": enrich,
            "k": dir_k,
        }
    ).sort_values("qvalue_BH", ascending=True)

    dedup = dir_df.drop_duplicates(subset=["gene_1", "gene_2"], keep=pair_keep).reset_index(drop=True)
    dedup["significant"] = dedup["qvalue_BH"] < float(q_threshold)
    return dedup, time.perf_counter() - t0


# ---------------------------------------------------------------------------
# Experiment 1: SimFISH Controlled Mechanism Benchmark
# ---------------------------------------------------------------------------
def _sample_points_in_polygon(poly, n: int, rng: np.random.Generator) -> np.ndarray:
    """Rejection-sample n uniform 2D points inside a Shapely polygon."""
    if n <= 0 or poly is None or poly.is_empty or poly.area <= 0:
        return np.empty((0, 2), dtype=np.float64)
    shapely.prepare(poly)
    minx, miny, maxx, maxy = poly.bounds
    collected = []
    needed = n
    batch = max(64, n * 4)
    for _ in range(50):
        cand = np.column_stack(
            [rng.uniform(minx, maxx, size=batch), rng.uniform(miny, maxy, size=batch)]
        )
        pts = shapely.points(cand)
        inside = np.asarray(shapely.contains(poly, pts), dtype=bool)
        if np.any(inside):
            collected.append(cand[inside])
            needed -= int(inside.sum())
            if needed <= 0:
                break
        batch = max(batch, needed * 6)
    if not collected:
        c = poly.representative_point()
        return np.tile([c.x, c.y], (n, 1))
    out = np.vstack(collected)
    if len(out) < n:
        extra = out[rng.integers(0, len(out), size=n - len(out))]
        out = np.vstack([out, extra])
    return out[:n]


def _clamp_to_cell(pts: np.ndarray, cell_poly, rng: np.random.Generator) -> np.ndarray:
    """Keep perturbed companion points inside cell_poly."""
    if len(pts) == 0:
        return pts
    shapely.prepare(cell_poly)
    inside = np.asarray(shapely.contains(cell_poly, shapely.points(pts)), dtype=bool)
    if np.all(inside):
        return pts
    out = pts.copy()
    out[~inside] = _sample_points_in_polygon(cell_poly, int((~inside).sum()), rng)
    return out


def build_simfish_mechanism_dataset(seed: int = 2026) -> tuple[dict, pd.DataFrame, float]:
    """Construct 200-cell realistic morphology dataset with 6 ground-truth pair regimes."""
    rng = np.random.default_rng(seed)
    with open(STRAND_ROOT / "data" / "simulated_data_dict.pkl", "rb") as f:
        sim_raw = pickle.load(f)
    base_geoms = list(_load_cell_geometries(sim_raw).values())

    n_cells = 200
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
        cid = f"simcell_{c_idx:03d}"
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
        cell_regions[cid] = {
            "cell": c_poly,
            "nuc": inner_nuc,
            "peri": peri_band if (peri_band is not None and peri_band.area > 0) else cyto,
            "cyto": cyto,
        }

    med_r = float(np.median(radii_list))
    eval_radius = round(0.10 * med_r, 2)  # ~12.8 px (~1.3 um equivalent)
    pair_sigma = 0.22 * eval_radius       # ~2.8 px tight molecular complex offset

    n_per_cat = 25
    gt_rows = []
    spot_chunks = []

    cids = [f"simcell_{i:03d}" for i in range(n_cells)]
    activated_cells = set(cids[:35])  # 17.5% high-expression / high-density subpopulation

    def _emit(cid: str, gene: str, pts: np.ndarray) -> None:
        if len(pts):
            spot_chunks.append(
                pd.DataFrame(
                    {
                        "cell": cid,
                        "gene": gene,
                        "x": pts[:, 0],
                        "y": pts[:, 1],
                    }
                )
            )

    # 1. Background transcriptome (30 housekeepers so per-cell total N_c is realistic ~1200 spots,
    #    with 3x higher background density in activated_cells).
    for b_idx in range(30):
        g_bg = f"BG_{b_idx:02d}"
        for cid in cids:
            lam = 65 if cid in activated_cells else 25
            n_sp = int(rng.poisson(lam))
            _emit(cid, g_bg, _sample_points_in_polygon(cell_regions[cid]["cell"], n_sp, rng))

    # P1: Sparse micro-colocalization (1-2 copies/cell, mostly single-copy, 55% coupled — Xenium/lncRNA low-copy regime).
    for k in range(n_per_cat):
        ga, gb = f"P1_spA_{k:02d}", f"P1_spB_{k:02d}"
        gt_rows.append({"gene_1": min(ga, gb), "gene_2": max(ga, gb), "category": "P1_sparse_micro", "is_true_micro": 1})
        for cid in cids:
            if rng.random() < 0.20:
                continue
            na = int(rng.choice([1, 2], p=[0.82, 0.18]))
            nb = int(rng.choice([1, 2], p=[0.82, 0.18]))
            c_poly = cell_regions[cid]["cell"]
            pts_a = _sample_points_in_polygon(c_poly, na, rng)
            n_coupled = int(rng.binomial(nb, 0.55))
            pts_b_parts = []
            if n_coupled > 0:
                anchors = pts_a[rng.integers(0, len(pts_a), size=n_coupled)]
                pts_b_parts.append(
                    _clamp_to_cell(anchors + rng.normal(0.0, pair_sigma, size=(n_coupled, 2)), c_poly, rng)
                )
            if nb - n_coupled > 0:
                pts_b_parts.append(_sample_points_in_polygon(c_poly, nb - n_coupled, rng))
            _emit(cid, ga, pts_a)
            _emit(cid, gb, np.vstack(pts_b_parts))

    # P2: Moderate intra-compartment micro-colocalization (inside nucleus or perinuclear, 40% coupled).
    for k in range(n_per_cat):
        ga, gb = f"P2_modA_{k:02d}", f"P2_modB_{k:02d}"
        gt_rows.append({"gene_1": min(ga, gb), "gene_2": max(ga, gb), "category": "P2_moderate_micro", "is_true_micro": 1})
        comp_key = "nuc" if (k % 2 == 0) else "cyto"
        for cid in cids:
            na = max(1, int(rng.poisson(14)))
            nb = max(1, int(rng.poisson(14)))
            reg_poly = cell_regions[cid][comp_key]
            c_poly = cell_regions[cid]["cell"]
            pts_a = _sample_points_in_polygon(reg_poly, na, rng)
            n_coupled = max(1, int(round(0.45 * nb)))
            anchors = pts_a[rng.integers(0, len(pts_a), size=n_coupled)]
            pts_b_cpl = _clamp_to_cell(anchors + rng.normal(0.0, pair_sigma, size=(n_coupled, 2)), c_poly, rng)
            pts_b_ind = _sample_points_in_polygon(reg_poly, nb - n_coupled, rng)
            _emit(cid, ga, pts_a)
            _emit(cid, gb, np.vstack([pts_b_cpl, pts_b_ind]) if len(pts_b_ind) else pts_b_cpl)

    # P3: >100x abundance-imbalanced true interaction (~2.5 lncRNA copies vs ~360 mRNA copies -> ~145x).
    for k in range(n_per_cat):
        ga, gb = f"P3_lnc_{k:02d}", f"P3_mrna_{k:02d}"
        gt_rows.append({"gene_1": min(ga, gb), "gene_2": max(ga, gb), "category": "P3_imbalanced_150x", "is_true_micro": 1})
        for cid in cids:
            na = int(rng.integers(2, 4))
            nb = int(rng.poisson(360))
            c_poly = cell_regions[cid]["cell"]
            pts_a = _sample_points_in_polygon(c_poly, na, rng)
            # Each low-copy lncRNA recruits 3 partner mRNA transcripts locally.
            n_recruited = min(nb, na * 3)
            anchors = np.repeat(pts_a, 3, axis=0)[:n_recruited]
            pts_b_cpl = _clamp_to_cell(anchors + rng.normal(0.0, pair_sigma * 0.85, size=(n_recruited, 2)), c_poly, rng)
            pts_b_bg = _sample_points_in_polygon(c_poly, nb - n_recruited, rng)
            _emit(cid, ga, pts_a)
            _emit(cid, gb, np.vstack([pts_b_cpl, pts_b_bg]))

    # N1: Co-compartment independent pairs (both in perinuclear/ER rim or both in inner nucleus, 0% coupling!).
    for k in range(n_per_cat):
        ga, gb = f"N1_compA_{k:02d}", f"N1_compB_{k:02d}"
        gt_rows.append({"gene_1": min(ga, gb), "gene_2": max(ga, gb), "category": "N1_co_compartment", "is_true_micro": 0})
        comp_key = "peri" if (k < 13) else "nuc"
        for cid in cids:
            na = max(1, int(rng.poisson(18)))
            nb = max(1, int(rng.poisson(18)))
            reg_poly = cell_regions[cid][comp_key]
            _emit(cid, ga, _sample_points_in_polygon(reg_poly, na, rng))
            _emit(cid, gb, _sample_points_in_polygon(reg_poly, nb, rng))

    # N2: Independent homotypic foci pairs (each gene forms 2 tight self-bursts of 9 transcripts, 0% cross coupling!).
    for k in range(n_per_cat):
        ga, gb = f"N2_fociA_{k:02d}", f"N2_fociB_{k:02d}"
        gt_rows.append({"gene_1": min(ga, gb), "gene_2": max(ga, gb), "category": "N2_homotypic_foci", "is_true_micro": 0})
        for cid in cids:
            c_poly = cell_regions[cid]["cell"]
            for g_foci in (ga, gb):
                centers = _sample_points_in_polygon(c_poly, 2, rng)
                burst = np.vstack([c + rng.normal(0.0, 2.0, size=(9, 2)) for c in centers])
                _emit(cid, g_foci, _clamp_to_cell(burst, c_poly, rng))

    # N3: Subpopulation co-expression + doublet outlier confounder (0% true regular coupling!).
    for k in range(n_per_cat):
        ga, gb = f"N3_confA_{k:02d}", f"N3_confB_{k:02d}"
        gt_rows.append({"gene_1": min(ga, gb), "gene_2": max(ga, gb), "category": "N3_subpop_confounder", "is_true_micro": 0})
        outlier_cell = cids[k % len(cids)]
        for cid in cids:
            c_poly = cell_regions[cid]["cell"]
            if cid == outlier_cell:
                # Single segmented doublet / sticky debris artifact in 1 out of 200 cells!
                anchor = _sample_points_in_polygon(c_poly, 1, rng)[0]
                _emit(cid, ga, _clamp_to_cell(anchor + rng.normal(0.0, 2.5, size=(35, 2)), c_poly, rng))
                _emit(cid, gb, _clamp_to_cell(anchor + rng.normal(0.0, 2.5, size=(35, 2)), c_poly, rng))
            elif cid in activated_cells:
                # High co-expression in activated subpopulation, completely uniform random inside cell!
                _emit(cid, ga, _sample_points_in_polygon(c_poly, int(rng.poisson(42)), rng))
                _emit(cid, gb, _sample_points_in_polygon(c_poly, int(rng.poisson(42)), rng))
            else:
                if rng.random() < 0.5:
                    _emit(cid, ga, _sample_points_in_polygon(c_poly, 1, rng))
                    _emit(cid, gb, _sample_points_in_polygon(c_poly, 1, rng))

    data_df = pd.concat(spot_chunks, ignore_index=True)
    bundle = {
        "data_df": data_df,
        "cell_boundary": cell_boundary,
        "nuclear_boundary": nuclear_boundary,
    }
    gt_df = pd.DataFrame(gt_rows)
    return bundle, gt_df, eval_radius


def evaluate_simfish_benchmark() -> tuple[pd.DataFrame, pd.DataFrame]:
    print("=== [Exp 1] Building SimFISH 6-regime mechanism dataset (200 cells, 150 target pairs) ===")
    bundle, gt_df, radius = build_simfish_mechanism_dataset(seed=2026)
    print(
        f"SimFISH spots={len(bundle['data_df'])}, cells={bundle['data_df']['cell'].nunique()}, "
        f"genes={bundle['data_df']['gene'].nunique()}, radius={radius}"
    )

    # 1. DeCoLoc (Gaussian kernel, default)
    res_dec_gauss, _ = run_decoloc_from_bundle(
        bundle,
        distance=radius,
        distance_mode="physical",
        region_threshold_mode="proportional",
        kernel="gaussian",
        fdr_alpha=0.01,
        min_fe=1.15,
        do_prefilter=False,
        threads=THREADS,
        profile=True,
    )

    # 2. DeCoLoc (Uniform kernel ablation)
    res_dec_unif, _ = run_decoloc_from_bundle(
        bundle,
        distance=radius,
        distance_mode="physical",
        region_threshold_mode="proportional",
        kernel="uniform",
        fdr_alpha=0.01,
        min_fe=1.15,
        do_prefilter=False,
        threads=THREADS,
    )

    # 3. InSTAnT (PP < 0.001, CPB < 0.0001)
    t_inst = time.perf_counter()
    res_inst = run_optimized_colocalization_from_bundle(
        bundle,
        distance_threshold=radius,
        pp_alpha=0.001,
        cpb_alpha=0.0001,
        use_3d=False,
        threads=THREADS,
    )
    inst_elapsed = time.perf_counter() - t_inst

    # 4. SCRIN standard (cooccurrence, expr_ratio <= 100, q < 1e-5)
    scrin_std_df, scrin_elapsed = run_scrin_cooccurrence(
        bundle["data_df"],
        radius=radius,
        q_threshold=1e-5,
        max_expr_ratio=100.0,
        threads=THREADS,
    )

    # 5. SCRIN without 100x expression filter (q < 1e-5)
    scrin_nofilt_df, _ = run_scrin_cooccurrence(
        bundle["data_df"],
        radius=radius,
        q_threshold=1e-5,
        max_expr_ratio=None,
        threads=THREADS,
    )

    # Map predictions and continuous scores onto the 150 ground-truth pairs.
    genes_inst = list(res_inst["gene_list"])
    g2idx = {g: idx for idx, g in enumerate(genes_inst)}
    cpb_mat = res_inst["cpb_pvals"]
    inst_sig_set = set(
        zip(
            np.minimum(res_inst["significant_pairs"]["gene_1"], res_inst["significant_pairs"]["gene_2"]),
            np.maximum(res_inst["significant_pairs"]["gene_1"], res_inst["significant_pairs"]["gene_2"]),
        )
    )

    def _lookup_table(df_in: pd.DataFrame, score_col: str, pred_mask: pd.Series) -> tuple[dict, dict]:
        keys = list(zip(df_in["gene_1"], df_in["gene_2"]))
        return dict(zip(keys, pred_mask.astype(int))), dict(zip(keys, df_in[score_col]))

    dec_g_df = res_dec_gauss.all_pairs.copy()
    dec_g_df["score_micro"] = -np.log10(np.clip(dec_g_df["p_comp"], 1e-300, 1.0))
    dec_g_df["score_cell"] = -np.log10(np.clip(dec_g_df["p_cell"], 1e-300, 1.0))
    dec_g_pred, dec_g_score = _lookup_table(
        dec_g_df, "score_micro", dec_g_df["coloc_class"] == "direct_micro"
    )
    dec_cell_pred, dec_cell_score = _lookup_table(
        dec_g_df, "score_cell", dec_g_df["coloc_class"] != "non_significant"
    )

    dec_u_df = res_dec_unif.all_pairs.copy()
    dec_u_df["score_micro"] = -np.log10(np.clip(dec_u_df["p_comp"], 1e-300, 1.0))
    dec_u_pred, dec_u_score = _lookup_table(
        dec_u_df, "score_micro", dec_u_df["coloc_class"] == "direct_micro"
    )

    scrin_std_df["score"] = -np.log10(np.clip(scrin_std_df["qvalue_BH"], 1e-300, 1.0))
    scrin_std_pred, scrin_std_score = _lookup_table(
        scrin_std_df, "score", scrin_std_df["significant"]
    )

    scrin_nofilt_df["score"] = -np.log10(np.clip(scrin_nofilt_df["qvalue_BH"], 1e-300, 1.0))
    scrin_nf_pred, scrin_nf_score = _lookup_table(
        scrin_nofilt_df, "score", scrin_nofilt_df["significant"]
    )

    inst_pred, inst_score = {}, {}
    for _, row in gt_df.iterrows():
        k = (row["gene_1"], row["gene_2"])
        inst_pred[k] = 1 if k in inst_sig_set else 0
        if k[0] in g2idx and k[1] in g2idx:
            pv = float(np.clip(cpb_mat[g2idx[k[0]], g2idx[k[1]]], 1e-300, 1.0))
            inst_score[k] = -np.log10(pv) if np.isfinite(pv) else 0.0
        else:
            inst_score[k] = 0.0

    methods = [
        ("InSTAnT (PP+CPB)", inst_pred, inst_score, inst_elapsed),
        ("SCRIN (std <=100x)", scrin_std_pred, scrin_std_score, scrin_elapsed),
        ("SCRIN (no expr cut)", scrin_nf_pred, scrin_nf_score, scrin_elapsed),
        ("DeCoLoc (whole-cell)", dec_cell_pred, dec_cell_score, res_dec_gauss.elapsed_seconds),
        ("DeCoLoc (Uniform)", dec_u_pred, dec_u_score, res_dec_unif.elapsed_seconds),
        ("DeCoLoc (Gaussian)", dec_g_pred, dec_g_score, res_dec_gauss.elapsed_seconds),
    ]

    cat_rows = []
    overall_rows = []
    y_true = gt_df["is_true_micro"].to_numpy(int)

    for m_name, pred_map, score_map, elapsed in methods:
        keys = list(zip(gt_df["gene_1"], gt_df["gene_2"]))
        y_pred = np.array([pred_map.get(k, 0) for k in keys], dtype=int)
        y_sc = np.array([score_map.get(k, 0.0) for k in keys], dtype=float)

        overall_rows.append(
            {
                "method": m_name,
                "precision": precision_score(y_true, y_pred, zero_division=0),
                "recall": recall_score(y_true, y_pred, zero_division=0),
                "f1": f1_score(y_true, y_pred, zero_division=0),
                "auprc": average_precision_score(y_true, y_sc),
                "n_called": int(y_pred.sum()),
                "elapsed_seconds": elapsed,
            }
        )

        for cat, sub in gt_df.groupby("category", sort=False):
            sub_keys = list(zip(sub["gene_1"], sub["gene_2"]))
            sub_pred = np.array([pred_map.get(k, 0) for k in sub_keys], dtype=int)
            rate = float(sub_pred.mean())
            cat_rows.append(
                {
                    "method": m_name,
                    "category": cat,
                    "is_positive_regime": int(sub["is_true_micro"].iloc[0]),
                    "positive_call_rate": rate,  # Recall for P1-P3, FPR for N1-N3
                    "n_called": int(sub_pred.sum()),
                    "n_total": len(sub),
                }
            )

    overall_df = pd.DataFrame(overall_rows)
    cat_df = pd.DataFrame(cat_rows)
    print("\n--- [Exp 1] SimFISH Overall Micro-Colocalization Detection ---")
    print(overall_df.to_string(index=False))
    print("\n--- [Exp 1] SimFISH Detection Rate by Regime (Recall on P1-P3, FPR on N1-N3) ---")
    piv = cat_df.pivot(index="category", columns="method", values="positive_call_rate")
    print(piv.round(3).to_string())
    return overall_df, cat_df


# ---------------------------------------------------------------------------
# Experiment 2: Real U2OS MERFISH + Two-Tier Intracellular Null Permutations
# ---------------------------------------------------------------------------
def _permute_u2os_labels(
    df: pd.DataFrame,
    geoms: dict[str, CellGeometry],
    *,
    within_compartment: bool,
    seed: int,
) -> pd.DataFrame:
    """Shuffle transcript gene IDs within each cell (Tier 1) or within each cell x region (Tier 2)."""
    rng = np.random.default_rng(seed)
    out = df.copy()
    genes_arr = out["gene"].to_numpy(dtype=object).copy()
    xy_all = out[["x", "y"]].to_numpy(dtype=np.float64)

    for cid, idx in out.groupby("cell", sort=False).indices.items():
        idx_arr = np.asarray(idx, dtype=np.int64)
        if len(idx_arr) < 2:
            continue
        if not within_compartment:
            genes_arr[idx_arr] = rng.permutation(genes_arr[idx_arr])
        else:
            geom = geoms[str(cid)]
            tn, tcn, tcp = compute_cell_region_thresholds(geom, mode="proportional")
            regs = assign_disjoint_transcript_regions(
                xy_all[idx_arr], geom, t_nucleus=tn, t_cyto_nuclear=tcn, t_cyto_peri=tcp
            )
            for v in range(4):
                sub_idx = idx_arr[regs == v]
                if len(sub_idx) > 1:
                    genes_arr[sub_idx] = rng.permutation(genes_arr[sub_idx])

    out["gene"] = genes_arr
    return out


def evaluate_u2os_two_tier_benchmark(n_cells_eval: int = 300) -> tuple[pd.DataFrame, pd.DataFrame]:
    print(f"\n=== [Exp 2] Loading U2OS MERFISH and sampling {n_cells_eval} cells ===")
    with open(STRAND_ROOT / "data" / "Dataset1_merfish_u2os_data_dict.pkl", "rb") as f:
        u2os_raw = pickle.load(f)

    u2os_clean, _ = prefilter_bundle_for_decoloc(u2os_raw)
    rng = np.random.default_rng(42)
    all_cids = sorted(u2os_clean["data_df"]["cell"].unique())
    chosen_cids = set(rng.choice(all_cids, size=min(n_cells_eval, len(all_cids)), replace=False))

    base_df = u2os_clean["data_df"][u2os_clean["data_df"]["cell"].isin(chosen_cids)].reset_index(drop=True)
    cb = {k: v for k, v in u2os_clean["cell_boundary"].items() if str(k) in chosen_cids}
    nb = {k: v for k, v in u2os_clean["nuclear_boundary"].items() if str(k) in chosen_cids}
    base_bundle = {"data_df": base_df, "cell_boundary": cb, "nuclear_boundary": nb}
    geoms = _load_cell_geometries(base_bundle)

    radii = [_equivalent_radius(_safe_area(g.cell)) for g in geoms.values()]
    radius = float(0.10 * np.median(radii))  # ~20.0 px (~1.8 um in U2OS)
    total_pairs = len(base_df["gene"].unique()) * (len(base_df["gene"].unique()) - 1) // 2
    print(
        f"U2OS subset: cells={len(chosen_cids)}, transcripts={len(base_df)}, "
        f"genes={base_df['gene'].nunique()}, total_pairs={total_pairs}, radius={radius:.2f} px"
    )

    tier1_df = _permute_u2os_labels(base_df, geoms, within_compartment=False, seed=101)
    tier2_df = _permute_u2os_labels(base_df, geoms, within_compartment=True, seed=202)

    conditions = [
        ("Original U2OS", base_bundle),
        ("Tier-1 Null (Whole-cell shuffle)", {"data_df": tier1_df, "cell_boundary": cb, "nuclear_boundary": nb}),
        ("Tier-2 Null (Within-region shuffle)", {"data_df": tier2_df, "cell_boundary": cb, "nuclear_boundary": nb}),
    ]

    rows = []
    u2os_orig_decoloc = None

    for cond_name, bnd in conditions:
        print(f"\n--- Running condition: {cond_name} ---")
        # 1. DeCoLoc (single pass includes 4-region enrichment)
        t0 = time.perf_counter()
        res_dec, _ = run_decoloc_from_bundle(
            bnd,
            distance=radius,
            distance_mode="cell_scaled",
            region_threshold_mode="proportional",
            kernel="gaussian",
            fdr_alpha=0.01,
            min_fe=1.15,
            do_prefilter=False,
            threads=THREADS,
            profile=True,
        )
        t_dec = time.perf_counter() - t0
        if cond_name == "Original U2OS":
            u2os_orig_decoloc = res_dec.significant_pairs.copy()

        # 2. InSTAnT PP+CPB (+ region annotation time measured on Original U2OS for fair full-pipeline comparison)
        t1 = time.perf_counter()
        res_inst = run_optimized_colocalization_from_bundle(
            bnd,
            distance_threshold=radius,
            pp_alpha=0.001,
            cpb_alpha=0.0001,
            use_3d=False,
            threads=THREADS,
        )
        t_inst_core = time.perf_counter() - t1
        t_inst_reg = 0.0
        if cond_name == "Original U2OS" and not res_inst["significant_pairs"].empty:
            tr0 = time.perf_counter()
            annotate_colocalized_pairs_regions(
                bundle=bnd,
                instant_input_df=res_inst["instant_input_df"],
                significant_pairs=res_inst["significant_pairs"],
                distance_threshold=radius,
                use_3d=False,
                region_threshold_mode="cell_scaled",
                region_workers=THREADS,
                region_executor="process",
            )
            t_inst_reg = time.perf_counter() - tr0

        # 3. SCRIN standard (cooccurrence, q < 1e-5, expr_ratio <= 100)
        scrin_df, t_scrin = run_scrin_cooccurrence(
            bnd["data_df"],
            radius=radius,
            q_threshold=1e-5,
            max_expr_ratio=100.0,
            threads=THREADS,
        )

        n_inst = len(res_inst["significant_pairs"])
        n_scrin = int(scrin_df["significant"].sum())
        n_dec_total = len(res_dec.significant_pairs)
        n_dec_micro = int((res_dec.significant_pairs["coloc_class"] == "direct_micro").sum()) if n_dec_total else 0
        n_dec_cocomp = int((res_dec.significant_pairs["coloc_class"] == "co_compartment").sum()) if n_dec_total else 0

        for m_label, n_sig, elapsed in [
            ("InSTAnT (PP+CPB)", n_inst, t_inst_core + t_inst_reg),
            ("SCRIN (cooccurrence)", n_scrin, t_scrin),
            ("DeCoLoc (Total: micro+comp)", n_dec_total, t_dec),
            ("DeCoLoc (direct_micro)", n_dec_micro, t_dec),
        ]:
            rows.append(
                {
                    "condition": cond_name,
                    "method": m_label,
                    "n_significant": n_sig,
                    "fraction_of_all_pairs": n_sig / max(total_pairs, 1),
                    "n_co_compartment_split": n_dec_cocomp if "DeCoLoc" in m_label else np.nan,
                    "elapsed_seconds": elapsed,
                    "instant_core_seconds": t_inst_core if m_label.startswith("InSTAnT") else np.nan,
                    "instant_region_seconds": t_inst_reg if m_label.startswith("InSTAnT") else np.nan,
                }
            )

    u2os_summary = pd.DataFrame(rows)
    print("\n--- [Exp 2] U2OS Two-Tier Null Benchmark Summary ---")
    print(u2os_summary.to_string(index=False))
    return u2os_summary, u2os_orig_decoloc


# ---------------------------------------------------------------------------
# Plotting (Bioinformatics Plotting Standards — Thesis Mode)
# ---------------------------------------------------------------------------
def plot_benchmark_results(
    sim_overall: pd.DataFrame,
    sim_cat: pd.DataFrame,
    u2os_summary: pd.DataFrame,
) -> Path:
    skill_scripts = Path("/home/wzx/.claude/skills/bioinformatics-plotting-standards/scripts")
    if str(skill_scripts) not in sys.path:
        sys.path.insert(0, str(skill_scripts))
    import matplotlib.pyplot as plt
    import seaborn as sns
    from thesis_style import add_panel_labels, apply_thesis_style, save_thesis_figure

    apply_thesis_style()

    # Colorblind-safe journal palette matching fig_results.
    C_DECOLOC = "#0173B2"      # Nature blue (DeCoLoc direct_micro)
    C_DECOLOC_SUB = "#56B4E9"  # Sky blue (DeCoLoc Uniform / Total)
    C_SCRIN = "#DE8F05"        # Nature orange (SCRIN)
    C_INSTANT = "#7F7F7F"      # Neutral grey (InSTAnT)

    fig = plt.figure(figsize=(6.02, 4.55), constrained_layout=True)
    fig.get_layout_engine().set(w_pad=0.05, h_pad=0.05, wspace=0.08, hspace=0.10, rect=(0, 0, 1, 0.955))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.15, 1.0], height_ratios=[1.05, 1.0])

    # Panel A: SimFISH True Positive Recall across P1 (Sparse), P2 (Moderate), P3 (>100x Imbalanced)
    ax_a = fig.add_subplot(gs[0, 0])
    pos_cats = ["P1_sparse_micro", "P2_moderate_micro", "P3_imbalanced_150x"]
    pos_labels = ["Sparse\n(1-2 spots/cell)", "Moderate\nsame-region", ">100x expr.\nimbalance"]
    show_methods = [
        ("InSTAnT (PP+CPB)", "InSTAnT", C_INSTANT),
        ("SCRIN (std <=100x)", "SCRIN", C_SCRIN),
        ("DeCoLoc (Gaussian)", "DeCoLoc", C_DECOLOC),
    ]
    x = np.arange(len(pos_cats))
    width = 0.25
    for idx, (m_key, m_short, col) in enumerate(show_methods):
        vals = [
            float(
                sim_cat.loc[
                    (sim_cat["method"] == m_key) & (sim_cat["category"] == c),
                    "positive_call_rate",
                ].iloc[0]
            )
            for c in pos_cats
        ]
        bars = ax_a.bar(x + (idx - 1) * width, vals, width=width * 0.92, color=col, label=m_short)
        for b, v in zip(bars, vals):
            ax_a.text(
                b.get_x() + b.get_width() / 2,
                v + 0.02,
                f"{v:.2f}",
                ha="center",
                va="bottom",
                fontsize=6.2,
            )
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(pos_labels)
    ax_a.set_ylim(0, 1.32)
    ax_a.set_yticks([0.0, 0.25, 0.50, 0.75, 1.00])
    ax_a.set_ylabel("True micro-coloc recall")
    ax_a.legend(frameon=False, loc="upper center", ncol=3, columnspacing=0.8, handlelength=1.0)
    sns.despine(ax=ax_a)

    # Panel B: SimFISH False Positive Rate on N1 (Co-compartment), N2 (Foci), N3 (Subpop/Doublet)
    ax_b = fig.add_subplot(gs[0, 1])
    neg_cats = ["N1_co_compartment", "N2_homotypic_foci", "N3_subpop_confounder"]
    neg_labels = ["Co-compartment\nindependent", "Homotypic\nfoci bursts", "Subpop. &\ndoublet conf."]
    x_b = np.arange(len(neg_cats))
    for idx, (m_key, m_short, col) in enumerate(show_methods):
        vals = [
            float(
                sim_cat.loc[
                    (sim_cat["method"] == m_key) & (sim_cat["category"] == c),
                    "positive_call_rate",
                ].iloc[0]
            )
            for c in neg_cats
        ]
        bars = ax_b.bar(x_b + (idx - 1) * width, vals, width=width * 0.92, color=col, label=m_short)
        for b, v in zip(bars, vals):
            ax_b.text(
                b.get_x() + b.get_width() / 2,
                v + 0.02,
                f"{v:.2f}",
                ha="center",
                va="bottom",
                fontsize=6.2,
            )
    ax_b.axhline(0.01, ls="--", lw=0.6, color="#444444")
    ax_b.set_xticks(x_b)
    ax_b.set_xticklabels(neg_labels)
    ax_b.set_ylim(0, 1.25)
    ax_b.set_yticks([0.0, 0.25, 0.50, 0.75, 1.00])
    ax_b.set_ylabel("False positive rate (FPR)")
    sns.despine(ax=ax_b)

    # Panel C: Real U2OS MERFISH Two-Tier Permutation Benchmark (Significant Pairs Called)
    ax_c = fig.add_subplot(gs[1, 0])
    cond_order = [
        "Original U2OS",
        "Tier-1 Null (Whole-cell shuffle)",
        "Tier-2 Null (Within-region shuffle)",
    ]
    cond_labels = ["Original\nU2OS", "Whole-cell\nshuffle", "Within-region\nshuffle"]
    u2os_methods = [
        ("InSTAnT (PP+CPB)", "InSTAnT", C_INSTANT),
        ("SCRIN (cooccurrence)", "SCRIN", C_SCRIN),
        ("DeCoLoc (direct_micro)", "DeCoLoc (micro)", C_DECOLOC),
    ]
    x_c = np.arange(len(cond_order))
    max_sig = 1.0
    for idx, (m_key, m_short, col) in enumerate(u2os_methods):
        vals = [
            int(
                u2os_summary.loc[
                    (u2os_summary["method"] == m_key) & (u2os_summary["condition"] == c),
                    "n_significant",
                ].iloc[0]
            )
            for c in cond_order
        ]
        max_sig = max(max_sig, max(vals))
        bars = ax_c.bar(x_c + (idx - 1) * width, vals, width=width * 0.92, color=col, label=m_short)
        for b, v in zip(bars, vals):
            ax_c.text(
                b.get_x() + b.get_width() / 2,
                v + max_sig * 0.02,
                f"{v}",
                ha="center",
                va="bottom",
                fontsize=6.2,
            )
    ax_c.set_xticks(x_c)
    ax_c.set_xticklabels(cond_labels)
    ax_c.set_ylim(0, max_sig * 1.22)
    ax_c.set_ylabel("Significant gene pairs (U2OS)")
    ax_c.legend(frameon=False, loc="upper right", handlelength=1.0)
    sns.despine(ax=ax_c)

    # Panel D: Overall SimFISH F1 / AUPRC & Full-Pipeline U2OS Runtime (with 4-region annotation)
    ax_d = fig.add_subplot(gs[1, 1])
    summary_m = [
        ("InSTAnT (PP+CPB)", "InSTAnT\n(PP+CPB)", C_INSTANT),
        ("SCRIN (std <=100x)", "SCRIN\n(<=100x)", C_SCRIN),
        ("SCRIN (no expr cut)", "SCRIN\n(unfiltered)", "#E6AB02"),
        ("DeCoLoc (Uniform)", "DeCoLoc\n(Uniform)", C_DECOLOC_SUB),
        ("DeCoLoc (Gaussian)", "DeCoLoc\n(Gaussian)", C_DECOLOC),
    ]
    y_pos = np.arange(len(summary_m))[::-1]
    f1_vals = [
        float(sim_overall.loc[sim_overall["method"] == mk, "f1"].iloc[0])
        for mk, _, _ in summary_m
    ]
    ap_vals = [
        float(sim_overall.loc[sim_overall["method"] == mk, "auprc"].iloc[0])
        for mk, _, _ in summary_m
    ]
    h = 0.34
    ax_d.barh(y_pos + h / 2, f1_vals, height=h * 0.9, color=[c for _, _, c in summary_m], label="F1 score")
    ax_d.barh(
        y_pos - h / 2,
        ap_vals,
        height=h * 0.9,
        color="white",
        edgecolor=[c for _, _, c in summary_m],
         linewidth=1.0,
        label="AUPRC",
    )
    for yp, vf, va in zip(y_pos, f1_vals, ap_vals):
        ax_d.text(vf + 0.015, yp + h / 2, f"{vf:.2f}", va="center", fontsize=6.0)
        ax_d.text(va + 0.015, yp - h / 2, f"{va:.2f}", va="center", fontsize=6.0, color="#555555")

    ax_d.set_yticks(y_pos)
    ax_d.set_yticklabels([lbl.replace("\n", " ") for _, lbl, _ in summary_m])
    ax_d.set_xlim(0, 1.16)
    ax_d.set_xlabel("SimFISH micro-coloc F1 (solid) & AUPRC (open)")
    sns.despine(ax=ax_d)

    add_panel_labels(fig, [ax_a, ax_b, ax_c, ax_d])
    out_base = OUT_DIR / "fig_decoloc_benchmark"
    save_thesis_figure(fig, str(out_base))
    plt.close(fig)
    return out_base.with_suffix(".png")


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--reuse-u2os", action="store_true", help="Reuse cached U2OS CSV if already present.")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    sim_overall, sim_cat = evaluate_simfish_benchmark()
    sim_overall.to_csv(OUT_DIR / "simfish_overall_metrics.csv", index=False)
    sim_cat.to_csv(OUT_DIR / "simfish_regime_rates.csv", index=False)

    u2os_csv = OUT_DIR / "u2os_two_tier_null_summary.csv"
    if args.reuse_u2os and u2os_csv.exists():
        u2os_summary = pd.read_csv(u2os_csv)
    else:
        u2os_summary, u2os_orig_pairs = evaluate_u2os_two_tier_benchmark(n_cells_eval=300)
        u2os_summary.to_csv(u2os_csv, index=False)
        if u2os_orig_pairs is not None:
            u2os_orig_pairs.to_csv(OUT_DIR / "u2os_decoloc_significant_pairs.csv", index=False)

    fig_png = plot_benchmark_results(sim_overall, sim_cat, u2os_summary)
    print(f"\nSaved benchmark tables and figure to: {fig_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
