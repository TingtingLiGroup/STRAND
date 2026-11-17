from __future__ import annotations

from dataclasses import dataclass
import multiprocessing as mp
import time
from typing import Literal

import numpy as np
import pandas as pd
from scipy import sparse, stats
from scipy.spatial import cKDTree
import shapely
from shapely.geometry import Point

from tools.engines.coloc_region_annotation import (
    CellGeometry,
    INSTANT_DISTANCE_THRESHOLD_CYTO_NUCLEAR,
    INSTANT_DISTANCE_THRESHOLD_CYTO_PERI,
    INSTANT_DISTANCE_THRESHOLD_NUCLEUS,
    _equivalent_radius,
    _load_cell_geometries,
    _safe_area,
    _safe_buffer,
    _safe_difference,
    _safe_intersection,
    _safe_union,
)

REGION_NAMES: tuple[str, ...] = (
    "nuclear",
    "perinuclear",
    "cytosolic",
    "peripheral",
)

# Worker globals for Linux fork multiprocessing without pickling large arrays per task.
_W_COORDS: np.ndarray | None = None
_W_GENES: np.ndarray | None = None
_W_CELL_OFFSETS: np.ndarray | None = None
_W_RADII: np.ndarray | None = None
_W_GEOM_LIST: list[CellGeometry] | None = None
_W_THRESHOLDS: np.ndarray | None = None
_W_NUM_REGIONS: int = 1
_W_KERNEL: str = "gaussian"
_W_SIGMA_FRAC: float = 0.5
_W_SHORT_FRAC: float = 0.5


@dataclass
class DeCoLocResults:
    """Output container for the DeCoLoc colocalization engine."""

    all_pairs: pd.DataFrame
    significant_pairs: pd.DataFrame
    genes: tuple[str, ...]
    cell_ids: tuple[str, ...]
    has_compartment_null: bool
    n_testable_pairs: int
    median_cell_radius: float
    median_nucleus_radius: float
    effective_median_distance: float
    elapsed_seconds: float


def compute_cell_region_thresholds(
    geom: CellGeometry,
    *,
    mode: Literal["proportional", "cell_scaled", "absolute"] = "proportional",
    median_cell_radius: float = 0.0,
    median_nucleus_radius: float = 0.0,
) -> tuple[float, float, float]:
    """Compute (t_nucleus, t_cyto_nuclear, t_cyto_peri) for one cell.

    ``proportional`` makes region widths invariant to coordinate units (pixels vs um):
    for a typical mammalian cell (r_nuc=10 um, r_cell=20 um), 0.25*r_nuc = 2.5 um and
    0.20*r_cell = 4.0 um, matching InSTAnT's physical defaults while automatically
    scaling to pixel-coordinate datasets (e.g. r_cell ~ 200 px).
    """
    r_cell = _equivalent_radius(_safe_area(geom.cell))
    r_nuc = _equivalent_radius(_safe_area(geom.nuclear))

    if mode == "proportional":
        return (0.25 * r_nuc, 0.25 * r_nuc, 0.20 * r_cell)
    if mode == "cell_scaled":
        nuc_scale = (
            r_nuc / median_nucleus_radius
            if median_nucleus_radius > 0 and r_nuc > 0
            else 1.0
        )
        cell_scale = (
            r_cell / median_cell_radius
            if median_cell_radius > 0 and r_cell > 0
            else 1.0
        )
        return (
            INSTANT_DISTANCE_THRESHOLD_NUCLEUS * nuc_scale,
            INSTANT_DISTANCE_THRESHOLD_CYTO_NUCLEAR * nuc_scale,
            INSTANT_DISTANCE_THRESHOLD_CYTO_PERI * cell_scale,
        )
    if mode == "absolute":
        return (
            INSTANT_DISTANCE_THRESHOLD_NUCLEUS,
            INSTANT_DISTANCE_THRESHOLD_CYTO_NUCLEAR,
            INSTANT_DISTANCE_THRESHOLD_CYTO_PERI,
        )
    raise ValueError(f"Unsupported region_threshold_mode: {mode!r}")


def assign_disjoint_transcript_regions(
    xy: np.ndarray,
    geom: CellGeometry,
    *,
    t_nucleus: float,
    t_cyto_nuclear: float,
    t_cyto_peri: float,
) -> np.ndarray:
    """Assign each transcript to a strictly disjoint region id in {0, 1, 2, 3}.

    0 = nuclear, 1 = perinuclear, 2 = cytosolic, 3 = peripheral.
    When narrow cytoplasm causes the outer perinuclear band and peripheral band
    to overlap, transcripts are assigned to the closer boundary in normalized units.
    """
    n = len(xy)
    region_id = np.full(n, 2, dtype=np.int8)
    if n == 0:
        return region_id

    try:
        pts = shapely.points(xy)
        in_nuc = np.asarray(shapely.covers(geom.nuclear, pts), dtype=bool)
        nuc_dist = np.asarray(shapely.distance(pts, geom.nuclear.boundary), dtype=np.float64)
        cell_dist = np.asarray(shapely.distance(pts, geom.cell.boundary), dtype=np.float64)
    except Exception:
        in_nuc = np.zeros(n, dtype=bool)
        nuc_dist = np.full(n, np.inf, dtype=np.float64)
        cell_dist = np.full(n, np.inf, dtype=np.float64)
        for idx, (x_val, y_val) in enumerate(xy):
            pt = Point(float(x_val), float(y_val))
            try:
                in_nuc[idx] = bool(geom.nuclear.covers(pt))
                nuc_dist[idx] = float(pt.distance(geom.nuclear.boundary))
                cell_dist[idx] = float(pt.distance(geom.cell.boundary))
            except Exception:
                pass

    # Inside nucleus: inner nuclear (0) vs inner perinuclear rim (1).
    region_id[in_nuc & (nuc_dist > t_nucleus)] = 0
    region_id[in_nuc & (nuc_dist <= t_nucleus)] = 1

    # Cytoplasm: resolve outer perinuclear (1), peripheral (3), and cytosolic (2) disjointly.
    cyto = ~in_nuc
    near_nuc = cyto & (nuc_dist <= t_cyto_nuclear) & (t_cyto_nuclear > 0)
    near_mem = cyto & (cell_dist <= t_cyto_peri) & (t_cyto_peri > 0)
    both = near_nuc & near_mem

    region_id[near_nuc & ~near_mem] = 1
    region_id[near_mem & ~near_nuc] = 3
    if np.any(both):
        rel_nuc = nuc_dist[both] / max(t_cyto_nuclear, 1e-12)
        rel_mem = cell_dist[both] / max(t_cyto_peri, 1e-12)
        region_id_both = np.where(rel_nuc <= rel_mem, 1, 3).astype(np.int8)
        region_id[both] = region_id_both

    return region_id


def compute_disjoint_region_areas(
    geom: CellGeometry,
    *,
    t_nucleus: float,
    t_cyto_nuclear: float,
    t_cyto_peri: float,
) -> np.ndarray:
    """Compute disjoint region area fractions [nuclear, perinuclear, cytosolic, peripheral]."""
    cell = geom.cell
    nuc = _safe_intersection(geom.nuclear, cell) or geom.nuclear
    cytoplasm = _safe_difference(cell, nuc)

    inner_nuclear = _safe_buffer(nuc, -float(t_nucleus))
    peri_inner = _safe_difference(nuc, inner_nuclear)

    nuc_outer = _safe_buffer(nuc, float(t_cyto_nuclear))
    peri_outer_raw = _safe_intersection(cytoplasm, _safe_difference(nuc_outer, nuc))

    cell_core = _safe_buffer(cell, -float(t_cyto_peri))
    peri_mem_raw = _safe_intersection(cytoplasm, _safe_difference(cell, cell_core))

    overlap = _safe_intersection(peri_outer_raw, peri_mem_raw)
    overlap_area = _safe_area(overlap)

    cytosolic = _safe_difference(
        cytoplasm,
        _safe_union([peri_outer_raw, peri_mem_raw]),
    )

    areas = np.array(
        [
            _safe_area(inner_nuclear),
            _safe_area(peri_inner) + max(0.0, _safe_area(peri_outer_raw) - 0.5 * overlap_area),
            _safe_area(cytosolic),
            max(0.0, _safe_area(peri_mem_raw) - 0.5 * overlap_area),
        ],
        dtype=np.float64,
    )
    total = float(areas.sum())
    if total <= 0:
        return np.full(4, 0.25, dtype=np.float64)
    return areas / total


def _init_scan_worker(
    coords: np.ndarray,
    genes: np.ndarray,
    cell_offsets: np.ndarray,
    radii: np.ndarray,
    geom_list: list[CellGeometry] | None,
    thresholds: np.ndarray | None,
    num_regions: int,
    kernel: str,
    sigma_frac: float,
    short_frac: float,
) -> None:
    global _W_COORDS, _W_GENES, _W_CELL_OFFSETS, _W_RADII
    global _W_GEOM_LIST, _W_THRESHOLDS, _W_NUM_REGIONS, _W_KERNEL, _W_SIGMA_FRAC, _W_SHORT_FRAC
    _W_COORDS = coords
    _W_GENES = genes
    _W_CELL_OFFSETS = cell_offsets
    _W_RADII = radii
    _W_GEOM_LIST = geom_list
    _W_THRESHOLDS = thresholds
    _W_NUM_REGIONS = num_regions
    _W_KERNEL = kernel
    _W_SIGMA_FRAC = sigma_frac
    _W_SHORT_FRAC = short_frac


def _scan_one_cell(cell_idx: int):
    """Single-pass parallel worker: disjoint 4-region labeling + cKDTree neighbor scan."""
    start = int(_W_CELL_OFFSETS[cell_idx])
    stop = int(_W_CELL_OFFSETS[cell_idx + 1])
    n_spots = stop - start
    radius = float(_W_RADII[cell_idx])
    u_reg = int(_W_NUM_REGIONS)

    if _W_GEOM_LIST is not None and _W_THRESHOLDS is not None:
        geom = _W_GEOM_LIST[cell_idx]
        t_nuc, t_cn, t_cp = _W_THRESHOLDS[cell_idx]
        area_frac = compute_disjoint_region_areas(
            geom, t_nucleus=t_nuc, t_cyto_nuclear=t_cn, t_cyto_peri=t_cp
        )
        if n_spots > 0:
            regs = assign_disjoint_transcript_regions(
                _W_COORDS[start:stop, :2],
                geom,
                t_nucleus=t_nuc,
                t_cyto_nuclear=t_cn,
                t_cyto_peri=t_cp,
            )
        else:
            regs = np.empty(0, dtype=np.int8)
    else:
        area_frac = np.ones(1, dtype=np.float64)
        regs = np.zeros(n_spots, dtype=np.int8)

    if n_spots < 2 or radius <= 0:
        if n_spots == 1:
            g0 = int(_W_GENES[start])
            r0 = int(regs[0])
            n_reg = np.zeros((1, u_reg), dtype=np.int32)
            n_reg[0, r0] = 1
            return (
                cell_idx,
                area_frac,
                np.array([g0], dtype=np.int32),
                n_reg,
                np.zeros((1, u_reg), dtype=np.float64),
                np.empty(0, dtype=np.int32),
                np.empty(0, dtype=np.int32),
                np.empty(0, dtype=np.float64),
                np.empty(0, dtype=np.int32),
                np.empty(0, dtype=np.int32),
                np.empty((0, u_reg), dtype=np.float64),
            )
        return (
            cell_idx,
            area_frac,
            np.empty(0, dtype=np.int32),
            np.empty((0, u_reg), dtype=np.int32),
            np.empty((0, u_reg), dtype=np.float64),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int32),
            np.empty((0, u_reg), dtype=np.float64),
        )

    coords = _W_COORDS[start:stop]
    genes = _W_GENES[start:stop]

    uniq_genes, local_g = np.unique(genes, return_inverse=True)
    g_active = len(uniq_genes)
    n_reg = np.zeros((g_active, u_reg), dtype=np.int32)
    np.add.at(n_reg, (local_g, regs), 1)
    d_ext = np.zeros((g_active, u_reg), dtype=np.float64)

    if g_active < 2:
        return (
            cell_idx,
            area_frac,
            uniq_genes.astype(np.int32, copy=False),
            n_reg,
            d_ext,
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int32),
            np.empty((0, u_reg), dtype=np.float64),
        )

    tree = cKDTree(coords)
    pairs = tree.query_pairs(r=radius, output_type="ndarray")
    if len(pairs) == 0:
        return (
            cell_idx,
            area_frac,
            uniq_genes.astype(np.int32, copy=False),
            n_reg,
            d_ext,
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int32),
            np.empty((0, u_reg), dtype=np.float64),
        )

    p_idx = pairs[:, 0]
    q_idx = pairs[:, 1]
    lg_p = local_g[p_idx]
    lg_q = local_g[q_idx]

    # Discard homotypic edges (g_p == g_q) so transcriptional burst foci never
    # inflate background neighbor degrees or violate pairing assumptions.
    hetero = lg_p != lg_q
    if not np.any(hetero):
        return (
            cell_idx,
            area_frac,
            uniq_genes.astype(np.int32, copy=False),
            n_reg,
            d_ext,
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int32),
            np.empty((0, u_reg), dtype=np.float64),
        )

    p_idx = p_idx[hetero]
    q_idx = q_idx[hetero]
    lg_p = lg_p[hetero]
    lg_q = lg_q[hetero]
    reg_p = regs[p_idx]
    reg_q = regs[q_idx]

    diff = coords[p_idx] - coords[q_idx]
    dists = np.sqrt(np.einsum("ij,ij->i", diff, diff))

    if _W_KERNEL == "gaussian":
        sigma = max(_W_SIGMA_FRAC * radius, 1e-12)
        weights = np.exp(-0.5 * (dists / sigma) ** 2)
    elif _W_KERNEL == "uniform":
        weights = np.ones(len(dists), dtype=np.float64)
    else:
        raise ValueError(f"Unknown kernel: {_W_KERNEL!r}")

    # Accumulate external weighted neighbor degrees per (local_gene, target_region).
    np.add.at(d_ext, (lg_p, reg_q), weights)
    np.add.at(d_ext, (lg_q, reg_p), weights)

    # Canonicalize unordered heterotypic gene pairs (u_min < u_max in local gene index).
    u_min = np.minimum(lg_p, lg_q)
    u_max = np.maximum(lg_p, lg_q)
    pair_key = u_min.astype(np.int64) * g_active + u_max.astype(np.int64)
    uniq_keys, inv = np.unique(pair_key, return_inverse=True)
    n_active_pairs = len(uniq_keys)

    obs_w = np.bincount(inv, weights=weights, minlength=n_active_pairs).astype(np.float64)
    raw_edges = np.bincount(inv, minlength=n_active_pairs).astype(np.int32)
    short_mask = (dists <= (_W_SHORT_FRAC * radius)).astype(np.int32)
    short_edges = np.bincount(inv, weights=short_mask, minlength=n_active_pairs).astype(np.int32)

    reg_contrib = np.zeros((n_active_pairs, u_reg), dtype=np.float64)
    half_w = 0.5 * weights
    np.add.at(reg_contrib, (inv, reg_p), half_w)
    np.add.at(reg_contrib, (inv, reg_q), half_w)

    local_i = (uniq_keys // g_active).astype(np.int32)
    local_j = (uniq_keys % g_active).astype(np.int32)
    global_i = uniq_genes[local_i].astype(np.int32, copy=False)
    global_j = uniq_genes[local_j].astype(np.int32, copy=False)

    return (
        cell_idx,
        area_frac,
        uniq_genes.astype(np.int32, copy=False),
        n_reg,
        d_ext,
        global_i,
        global_j,
        obs_w,
        raw_edges,
        short_edges,
        reg_contrib,
    )


def bh_adjust_with_total_tests(pvalues: np.ndarray, total_tests: int) -> np.ndarray:
    """Benjamini-Hochberg adjustment over ``total_tests`` hypotheses.

    Tested pairs absent from ``pvalues`` have zero observed edges (t <= 0, p >= 0.5),
    so every pair with p < 0.5 holds its exact global rank among ``total_tests``.
    """
    m = len(pvalues)
    if m == 0:
        return np.empty(0, dtype=np.float64)
    total = max(int(total_tests), m)
    p = np.clip(np.asarray(pvalues, dtype=np.float64), 0.0, 1.0)
    order = np.argsort(p, kind="mergesort")
    p_sorted = p[order]
    ranks = np.arange(1, m + 1, dtype=np.float64)
    raw_q = (total * p_sorted) / ranks
    # Account for the remaining (total - m) zero-edge pairs whose p >= 0.5 -> q <= 1.0.
    q_sorted = np.minimum.accumulate(raw_q[::-1])[::-1]
    q_sorted = np.clip(q_sorted, 0.0, 1.0)
    q = np.empty(m, dtype=np.float64)
    q[order] = q_sorted
    return q


def _extract_pair_sums_from_factors(
    x_factors: list[sparse.csr_matrix],
    y_factors: list[sparse.csr_matrix],
    pair_i: np.ndarray,
    pair_j: np.ndarray,
    *,
    gene_chunk: int = 1024,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute exact sum_c E_cij and sum_c E_cij^2 for E_cij = 0.5 * sum_k X_{ci,k} Y_{cj,k}.

    Uses chunked sparse matrix products so memory stays O(gene_chunk * G) even for
    whole-transcriptome panels while preserving exact double-precision algebra.
    """
    n_pairs = len(pair_i)
    if n_pairs == 0:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)

    num_genes = x_factors[0].shape[1]
    k_terms = len(x_factors)
    sum_e = np.zeros(n_pairs, dtype=np.float64)
    sum_e2 = np.zeros(n_pairs, dtype=np.float64)

    # Group queried pairs by row block [g0, g1) to avoid materializing full G x G matrices.
    order = np.argsort(pair_i, kind="mergesort")
    pi_sorted = pair_i[order]
    pj_sorted = pair_j[order]

    # Pre-transpose X factors to CSC (so slicing columns g0:g1 and transposing is fast).
    x_csc = [xf.tocsc() for xf in x_factors]
    y_csr = [yf.tocsr() for yf in y_factors]

    # Precompute pairwise elementwise products for second-moment cross terms.
    xx_csc: list[tuple[int, int, float, sparse.csc_matrix, sparse.csr_matrix]] = []
    for k in range(k_terms):
        xx_csc.append((k, k, 0.25, x_csc[k].multiply(x_csc[k]), y_csr[k].multiply(y_csr[k])))
        for l in range(k + 1, k_terms):
            xx_csc.append((k, l, 0.50, x_csc[k].multiply(x_csc[l]), y_csr[k].multiply(y_csr[l])))

    ptr = 0
    while ptr < n_pairs:
        g0 = int(pi_sorted[ptr])
        g1 = min(num_genes, g0 + gene_chunk)
        end_ptr = int(np.searchsorted(pi_sorted, g1, side="left"))
        local_rows = pi_sorted[ptr:end_ptr] - g0
        local_cols = pj_sorted[ptr:end_ptr]
        out_idx = order[ptr:end_ptr]

        block_e = np.zeros((g1 - g0, num_genes), dtype=np.float64)
        for k in range(k_terms):
            prod = (x_csc[k][:, g0:g1].T @ y_csr[k]).toarray()
            block_e += 0.5 * prod
        sum_e[out_idx] = block_e[local_rows, local_cols]

        block_e2 = np.zeros((g1 - g0, num_genes), dtype=np.float64)
        for _, _, coeff, xkl, ykl in xx_csc:
            prod2 = (xkl[:, g0:g1].T @ ykl).toarray()
            block_e2 += coeff * prod2
        sum_e2[out_idx] = block_e2[local_rows, local_cols]

        ptr = end_ptr

    return sum_e, sum_e2


def run_decoloc_core(
    df: pd.DataFrame,
    *,
    geometries: dict[str, CellGeometry] | None = None,
    distance: float | None = None,
    distance_mode: Literal["physical", "cell_scaled"] = "cell_scaled",
    auto_distance_radius_frac: float = 0.10,
    region_threshold_mode: Literal["proportional", "cell_scaled", "absolute"] = "proportional",
    kernel: Literal["gaussian", "uniform"] = "gaussian",
    kernel_sigma_frac: float = 0.5,
    short_range_frac_cutoff: float = 0.5,
    use_3d: bool = False,
    min_coexpr_cells: int = 15,
    fdr_alpha: float = 0.01,
    min_fe: float = 1.15,
    threads: int = 8,
    cell_col: str = "cell",
    gene_col: str = "gene",
    x_col: str = "x",
    y_col: str = "y",
    z_col: str = "z",
    profile: bool = False,
) -> DeCoLocResults:
    """Run the 3-step DeCoLoc colocalization test in a single cKDTree pass.

    Step 1: Single-pass per-cell cKDTree scan for heterotypic neighbors + disjoint 4-region tags.
    Step 2: Exact closed-form degree-preserving expectations E_cell and E_comp (0 permutations).
    Step 3: Self-normalized cross-cell paired t-test on (O_c - E_c) + BH FDR + mechanism split.
    """
    t0 = time.perf_counter()

    work = df[[cell_col, gene_col, x_col, y_col] + ([z_col] if use_3d else [])].copy()
    work[cell_col] = work[cell_col].astype(str)
    work[gene_col] = work[gene_col].astype(str)

    has_geoms = geometries is not None and len(geometries) > 0
    if has_geoms:
        valid_cids = set(geometries.keys())
        work = work[work[cell_col].isin(valid_cids)].reset_index(drop=True)

    if work.empty:
        empty_df = pd.DataFrame()
        return DeCoLocResults(
            all_pairs=empty_df,
            significant_pairs=empty_df,
            genes=(),
            cell_ids=(),
            has_compartment_null=False,
            n_testable_pairs=0,
            median_cell_radius=0.0,
            median_nucleus_radius=0.0,
            effective_median_distance=0.0,
            elapsed_seconds=time.perf_counter() - t0,
        )

    cell_cat = pd.Categorical(work[cell_col], categories=sorted(work[cell_col].unique()))
    gene_cat = pd.Categorical(work[gene_col], categories=sorted(work[gene_col].unique()))
    cell_codes = np.asarray(cell_cat.codes, dtype=np.int32)
    gene_codes = np.asarray(gene_cat.codes, dtype=np.int32)
    cell_ids = tuple(str(x) for x in cell_cat.categories)
    genes = tuple(str(x) for x in gene_cat.categories)
    num_cells = len(cell_ids)
    num_genes = len(genes)

    order = np.argsort(cell_codes, kind="mergesort")
    cell_codes = cell_codes[order]
    gene_codes = gene_codes[order]
    coord_cols = [x_col, y_col, z_col] if use_3d else [x_col, y_col]
    coords = work[coord_cols].to_numpy(dtype=np.float64)[order]
    xy_coords = coords[:, :2]

    counts_per_cell = np.bincount(cell_codes, minlength=num_cells)
    cell_offsets = np.zeros(num_cells + 1, dtype=np.int64)
    np.cumsum(counts_per_cell, out=cell_offsets[1:])

    # Step 1a: Precompute per-cell geometry radii, region widths, disjoint transcript regions, and area priors.
    if has_geoms:
        num_regions = 4
        cell_radii = np.array(
            [_equivalent_radius(_safe_area(geometries[cid].cell)) for cid in cell_ids],
            dtype=np.float64,
        )
        nuc_radii = np.array(
            [_equivalent_radius(_safe_area(geometries[cid].nuclear)) for cid in cell_ids],
            dtype=np.float64,
        )
        valid_rc = cell_radii[cell_radii > 0]
        valid_rn = nuc_radii[nuc_radii > 0]
        med_r_cell = float(np.median(valid_rc)) if len(valid_rc) else 1.0
        med_r_nuc = float(np.median(valid_rn)) if len(valid_rn) else 1.0

        base_dist = (
            float(auto_distance_radius_frac * med_r_cell)
            if distance is None
            else float(distance)
        )
        if distance_mode == "cell_scaled" and med_r_cell > 0:
            radii = base_dist * np.where(cell_radii > 0, cell_radii / med_r_cell, 1.0)
        elif distance_mode == "physical":
            radii = np.full(num_cells, base_dist, dtype=np.float64)
        else:
            raise ValueError(f"Unsupported distance_mode: {distance_mode!r}")

        geom_list: list[CellGeometry] | None = [geometries[cid] for cid in cell_ids]
        thresholds = np.array(
            [
                compute_cell_region_thresholds(
                    geom_list[c_idx],
                    mode=region_threshold_mode,
                    median_cell_radius=med_r_cell,
                    median_nucleus_radius=med_r_nuc,
                )
                for c_idx in range(num_cells)
            ],
            dtype=np.float64,
        )
    else:
        num_regions = 1
        med_r_cell = 0.0
        med_r_nuc = 0.0
        base_dist = 2.0 if distance is None else float(distance)
        radii = np.full(num_cells, base_dist, dtype=np.float64)
        geom_list = None
        thresholds = None

    t_geom = time.perf_counter()

    # Step 1b: Single-pass parallel region tagging + cKDTree scan across cells.
    init_args = (
        coords,
        gene_codes,
        cell_offsets,
        radii,
        geom_list,
        thresholds,
        num_regions,
        kernel,
        float(kernel_sigma_frac),
        float(short_range_frac_cutoff),
    )

    cell_area_fracs = np.zeros((num_cells, num_regions), dtype=np.float64)
    cg_cells: list[np.ndarray] = []
    cg_genes: list[np.ndarray] = []
    cg_n_reg: list[np.ndarray] = []
    cg_d_ext: list[np.ndarray] = []

    nz_cells: list[np.ndarray] = []
    nz_gi: list[np.ndarray] = []
    nz_gj: list[np.ndarray] = []
    nz_obs_w: list[np.ndarray] = []
    nz_raw_edges: list[np.ndarray] = []
    nz_short_edges: list[np.ndarray] = []
    nz_reg_contrib: list[np.ndarray] = []

    def _collect(res) -> None:
        c_idx, area_f, u_genes, n_reg, d_ext, gi, gj, obs_w, raw_e, short_e, reg_c = res
        cell_area_fracs[c_idx] = area_f
        if len(u_genes):
            cg_cells.append(np.full(len(u_genes), c_idx, dtype=np.int32))
            cg_genes.append(u_genes)
            cg_n_reg.append(n_reg)
            cg_d_ext.append(d_ext)
        if len(gi):
            nz_cells.append(np.full(len(gi), c_idx, dtype=np.int32))
            nz_gi.append(gi)
            nz_gj.append(gj)
            nz_obs_w.append(obs_w)
            nz_raw_edges.append(raw_e)
            nz_short_edges.append(short_e)
            nz_reg_contrib.append(reg_c)

    n_workers = max(1, int(threads))
    if n_workers == 1 or num_cells < 8:
        _init_scan_worker(*init_args)
        for c_idx in range(num_cells):
            _collect(_scan_one_cell(c_idx))
    else:
        chunk = max(1, min(32, num_cells // max(n_workers * 4, 1)))
        ctx = mp.get_context("fork")
        with ctx.Pool(processes=n_workers, initializer=_init_scan_worker, initargs=init_args) as pool:
            for res in pool.imap_unordered(_scan_one_cell, range(num_cells), chunksize=chunk):
                _collect(res)

    t_scan = time.perf_counter()

    # Step 2: Build sparse C x G cell-gene factor matrices for exact closed-form expectations.
    all_cg_c = np.concatenate(cg_cells) if cg_cells else np.empty(0, dtype=np.int32)
    all_cg_g = np.concatenate(cg_genes) if cg_genes else np.empty(0, dtype=np.int32)
    all_n_reg = (
        np.concatenate(cg_n_reg, axis=0).astype(np.float64)
        if cg_n_reg
        else np.empty((0, num_regions), dtype=np.float64)
    )
    all_d_ext = (
        np.concatenate(cg_d_ext, axis=0)
        if cg_d_ext
        else np.empty((0, num_regions), dtype=np.float64)
    )

    n_tot_cg = all_n_reg.sum(axis=1)
    d_tot_cg = all_d_ext.sum(axis=1)

    # Total transcripts per (cell, region) and per cell.
    cell_reg_totals = np.zeros((num_cells, num_regions), dtype=np.float64)
    if len(all_cg_c):
        np.add.at(cell_reg_totals, all_cg_c, all_n_reg)
    cell_totals = cell_reg_totals.sum(axis=1)

    denom_cell = cell_totals[all_cg_c] - n_tot_cg
    a_cell_vals = np.where(denom_cell > 0, d_tot_cg / np.maximum(denom_cell, 1.0), 0.0)

    shape_cg = (num_cells, num_genes)
    mat_present = sparse.csr_matrix(
        (np.ones(len(all_cg_c), dtype=np.int32), (all_cg_c, all_cg_g)),
        shape=shape_cg,
    )
    mat_n_cell = sparse.csr_matrix((n_tot_cg, (all_cg_c, all_cg_g)), shape=shape_cg)
    mat_a_cell = sparse.csr_matrix((a_cell_vals, (all_cg_c, all_cg_g)), shape=shape_cg)

    mat_n_regs: list[sparse.csr_matrix] = []
    mat_a_regs: list[sparse.csr_matrix] = []
    for v in range(num_regions):
        nv = all_n_reg[:, v]
        dv = all_d_ext[:, v]
        denom_v = cell_reg_totals[all_cg_c, v] - nv
        av = np.where(denom_v > 0, dv / np.maximum(denom_v, 1.0), 0.0)
        mat_n_regs.append(sparse.csr_matrix((nv, (all_cg_c, all_cg_g)), shape=shape_cg))
        mat_a_regs.append(sparse.csr_matrix((av, (all_cg_c, all_cg_g)), shape=shape_cg))

    # Co-expression cell count M_ij for all gene pairs (sparse upper triangle).
    coexpr_mat = (mat_present.T @ mat_present).tocsr()
    coexpr_upper = sparse.triu(coexpr_mat, k=1, format="coo")
    n_testable_pairs = int(np.count_nonzero(coexpr_upper.data >= int(min_coexpr_cells)))

    if not nz_gi or n_testable_pairs == 0:
        empty_df = pd.DataFrame()
        return DeCoLocResults(
            all_pairs=empty_df,
            significant_pairs=empty_df,
            genes=genes,
            cell_ids=cell_ids,
            has_compartment_null=(num_regions > 1),
            n_testable_pairs=n_testable_pairs,
            median_cell_radius=med_r_cell,
            median_nucleus_radius=med_r_nuc,
            effective_median_distance=base_dist,
            elapsed_seconds=time.perf_counter() - t0,
        )

    # Aggregate non-zero proximal records (cells where O_cij > 0).
    rec_c = np.concatenate(nz_cells)
    rec_i = np.concatenate(nz_gi)
    rec_j = np.concatenate(nz_gj)
    rec_obs = np.concatenate(nz_obs_w)
    rec_raw = np.concatenate(nz_raw_edges)
    rec_short = np.concatenate(nz_short_edges)
    rec_reg = np.concatenate(nz_reg_contrib, axis=0)

    # Exact E_cell and E_comp on those (c, i, j) records with O_cij > 0:
    n_ci = np.asarray(mat_n_cell[rec_c, rec_i]).ravel()
    n_cj = np.asarray(mat_n_cell[rec_c, rec_j]).ravel()
    a_ci = np.asarray(mat_a_cell[rec_c, rec_i]).ravel()
    a_cj = np.asarray(mat_a_cell[rec_c, rec_j]).ravel()
    rec_e_cell = 0.5 * (a_ci * n_cj + a_cj * n_ci)

    if num_regions > 1:
        rec_e_comp = np.zeros(len(rec_c), dtype=np.float64)
        for v in range(num_regions):
            nv_i = np.asarray(mat_n_regs[v][rec_c, rec_i]).ravel()
            nv_j = np.asarray(mat_n_regs[v][rec_c, rec_j]).ravel()
            av_i = np.asarray(mat_a_regs[v][rec_c, rec_i]).ravel()
            av_j = np.asarray(mat_a_regs[v][rec_c, rec_j]).ravel()
            rec_e_comp += 0.5 * (av_i * nv_j + av_j * nv_i)
    else:
        rec_e_comp = rec_e_cell.copy()

    pair_id = rec_i.astype(np.int64) * num_genes + rec_j.astype(np.int64)
    uniq_pids, inv_pid = np.unique(pair_id, return_inverse=True)
    n_cand = len(uniq_pids)
    cand_i = (uniq_pids // num_genes).astype(np.int32)
    cand_j = (uniq_pids % num_genes).astype(np.int32)

    m_coexpr = np.asarray(coexpr_mat[cand_i, cand_j]).ravel().astype(np.int32)
    keep = m_coexpr >= int(min_coexpr_cells)

    if not np.any(keep):
        empty_df = pd.DataFrame()
        return DeCoLocResults(
            all_pairs=empty_df,
            significant_pairs=empty_df,
            genes=genes,
            cell_ids=cell_ids,
            has_compartment_null=(num_regions > 1),
            n_testable_pairs=n_testable_pairs,
            median_cell_radius=med_r_cell,
            median_nucleus_radius=med_r_nuc,
            effective_median_distance=base_dist,
            elapsed_seconds=time.perf_counter() - t0,
        )

    sum_obs = np.bincount(inv_pid, weights=rec_obs, minlength=n_cand)[keep]
    sum_obs2 = np.bincount(inv_pid, weights=rec_obs * rec_obs, minlength=n_cand)[keep]
    sum_oe_cell = np.bincount(inv_pid, weights=rec_obs * rec_e_cell, minlength=n_cand)[keep]
    sum_oe_comp = np.bincount(inv_pid, weights=rec_obs * rec_e_comp, minlength=n_cand)[keep]
    n_prox_cells = np.bincount(inv_pid, minlength=n_cand).astype(np.int32)[keep]
    n_support_cells = np.bincount(
        inv_pid,
        weights=(rec_obs > rec_e_cell).astype(np.int32),
        minlength=n_cand,
    ).astype(np.int32)[keep]
    sum_raw_edges = np.bincount(inv_pid, weights=rec_raw, minlength=n_cand).astype(np.int64)[keep]
    sum_short_edges = np.bincount(inv_pid, weights=rec_short, minlength=n_cand).astype(np.int64)[keep]

    # Region enrichment accumulators (observed weighted endpoints vs cell area expectation).
    obs_reg_pair = np.zeros((n_cand, num_regions), dtype=np.float64)
    exp_reg_pair = np.zeros((n_cand, num_regions), dtype=np.float64)
    np.add.at(obs_reg_pair, inv_pid, rec_reg)
    rec_tot_reg = rec_reg.sum(axis=1, keepdims=True)
    np.add.at(exp_reg_pair, inv_pid, rec_tot_reg * cell_area_fracs[rec_c])
    obs_reg_pair = obs_reg_pair[keep]
    exp_reg_pair = exp_reg_pair[keep]

    cand_i = cand_i[keep]
    cand_j = cand_j[keep]
    m_coexpr = m_coexpr[keep]

    # Exact global sum_c E_cij and sum_c E_cij^2 across ALL co-expressing cells.
    sum_e_cell, sum_e2_cell = _extract_pair_sums_from_factors(
        [mat_a_cell, mat_n_cell],
        [mat_n_cell, mat_a_cell],
        cand_i,
        cand_j,
    )
    if num_regions > 1:
        sum_e_comp, sum_e2_comp = _extract_pair_sums_from_factors(
            mat_a_regs + mat_n_regs,
            mat_n_regs + mat_a_regs,
            cand_i,
            cand_j,
        )
    else:
        sum_e_comp = sum_e_cell.copy()
        sum_e2_comp = sum_e2_cell.copy()

    # Step 3: Cross-cell paired t-test on r_cij = O_cij - E_cij.
    def _paired_t_and_p(
        s_o: np.ndarray,
        s_o2: np.ndarray,
        s_oe: np.ndarray,
        s_e: np.ndarray,
        s_e2: np.ndarray,
        m_vec: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        diff_sum = s_o - s_e
        ss_res = np.maximum(0.0, s_o2 - 2.0 * s_oe + s_e2)
        m_f = m_vec.astype(np.float64)
        mean_r = diff_sum / m_f
        var_num = np.maximum(0.0, ss_res - m_f * mean_r * mean_r)
        sample_var = var_num / np.maximum(m_f - 1.0, 1.0)
        se = np.sqrt(sample_var / m_f)
        t_stat = np.divide(mean_r, se, out=np.zeros_like(mean_r), where=se > 1e-15)
        p_val = stats.t.sf(t_stat, df=np.maximum(m_vec - 1, 1))
        return t_stat, p_val

    t_cell, p_cell = _paired_t_and_p(
        sum_obs, sum_obs2, sum_oe_cell, sum_e_cell, sum_e2_cell, m_coexpr
    )
    t_comp, p_comp = _paired_t_and_p(
        sum_obs, sum_obs2, sum_oe_comp, sum_e_comp, sum_e2_comp, m_coexpr
    )

    q_cell = bh_adjust_with_total_tests(p_cell, n_testable_pairs)
    q_comp = bh_adjust_with_total_tests(p_comp, n_testable_pairs)

    fe_cell = sum_obs / np.maximum(sum_e_cell, 1e-12)
    fe_comp = sum_obs / np.maximum(sum_e_comp, 1e-12)
    support_rate = n_support_cells.astype(np.float64) / m_coexpr.astype(np.float64)
    short_frac_arr = sum_short_edges.astype(np.float64) / np.maximum(sum_raw_edges, 1)

    sig_cell = (q_cell < fdr_alpha) & (fe_cell >= min_fe)
    if num_regions > 1:
        sig_comp = sig_cell & (q_comp < fdr_alpha) & (fe_comp >= min_fe)
        coloc_class = np.where(
            sig_comp,
            "direct_micro",
            np.where(sig_cell, "co_compartment", "non_significant"),
        )
    else:
        sig_comp = sig_cell.copy()
        coloc_class = np.where(sig_cell, "colocalized_unstratified", "non_significant")

    gene_arr = np.asarray(genes, dtype=object)
    out = pd.DataFrame(
        {
            "gene_1": gene_arr[cand_i],
            "gene_2": gene_arr[cand_j],
            "coloc_class": coloc_class,
            "q_cell": q_cell,
            "q_comp": q_comp,
            "p_cell": p_cell,
            "p_comp": p_comp,
            "t_cell": t_cell,
            "t_comp": t_comp,
            "fe_cell": fe_cell,
            "fe_comp": fe_comp,
            "obs_score": sum_obs,
            "exp_cell": sum_e_cell,
            "exp_comp": sum_e_comp,
            "n_proximal_pairs": sum_raw_edges,
            "short_range_frac": short_frac_arr,
            "n_coexpr_cells": m_coexpr,
            "n_proximal_cells": n_prox_cells,
            "n_support_cells": n_support_cells,
            "cell_support_rate": support_rate,
        }
    )

    if num_regions == 4:
        reg_enrich = obs_reg_pair / (exp_reg_pair + 1e-12)
        for r_idx, r_name in enumerate(REGION_NAMES):
            out[f"{r_name}_enrichment"] = reg_enrich[:, r_idx]
        rank_order = np.argsort(-reg_enrich, axis=1)
        dom_idx = rank_order[:, 0]
        sec_idx = rank_order[:, 1]
        reg_label_arr = np.asarray(REGION_NAMES, dtype=object)
        out["dominant_region"] = reg_label_arr[dom_idx]
        out["dominant_enrichment_score"] = reg_enrich[np.arange(len(out)), dom_idx]
        out["second_region"] = reg_label_arr[sec_idx]
        out["second_enrichment_score"] = reg_enrich[np.arange(len(out)), sec_idx]
    else:
        out["dominant_region"] = "unannotated"
        out["dominant_enrichment_score"] = fe_cell

    out = out.sort_values(
        ["q_comp", "q_cell", "t_comp", "t_cell"],
        ascending=[True, True, False, False],
    ).reset_index(drop=True)

    sig_df = out[out["coloc_class"] != "non_significant"].reset_index(drop=True)
    elapsed = time.perf_counter() - t0

    if profile:
        print(
            f"[decoloc] cells={num_cells}, genes={num_genes}, testable_pairs={n_testable_pairs}, "
            f"active_pairs={len(out)}, sig_total={len(sig_df)}, "
            f"direct_micro={int((sig_df['coloc_class'] == 'direct_micro').sum())}, "
            f"co_compartment={int((sig_df['coloc_class'] == 'co_compartment').sum())}, "
            f"geom={t_geom - t0:.2f}s, scan={t_scan - t_geom:.2f}s, total={elapsed:.2f}s"
        )

    return DeCoLocResults(
        all_pairs=out,
        significant_pairs=sig_df,
        genes=genes,
        cell_ids=cell_ids,
        has_compartment_null=(num_regions > 1),
        n_testable_pairs=n_testable_pairs,
        median_cell_radius=med_r_cell,
        median_nucleus_radius=med_r_nuc,
        effective_median_distance=base_dist,
        elapsed_seconds=elapsed,
    )
