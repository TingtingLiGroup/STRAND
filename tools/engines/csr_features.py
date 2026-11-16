from __future__ import annotations

import zlib
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import shapely
from scipy.spatial import cKDTree

from tools.io.data_loader import load_and_process_data
from tools.engines.bento_adapter import _to_single_polygon
from tools.bento_feature.bento.bento_tools import (
    scale_normalize_cell_geometry,
    _poly_aspect_ratio,
)
from tools.utils.timing import timed, TimerReport

BENTO_13_COLS = [
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
]

SPRAWL_4_COLS = [
    "peripheral",
    "central",
    "punctate",
    "radial",
]

CSR_6_COLS = [
    "csr_z_nuclear_frac",
    "csr_z_cell_edge_dist",
    "csr_z_perinuc_cyto_dist",
    "csr_z_centroid_shift",
    "csr_z_nn_clustering",
    "csr_z_rayleigh_radial",
]

COV_5_COLS = [
    "feat_log1p_n_spots",
    "feat_nuclear_fraction",
    "feat_cell_nc_ratio",
    "feat_cell_aspect_ratio",
    "feat_cell_solidity",
]

FEATURE_COLS_V1_17D = BENTO_13_COLS + SPRAWL_4_COLS
FEATURE_COLS_V2_23D = FEATURE_COLS_V1_17D + CSR_6_COLS
FEATURE_COLS_V2_28D = FEATURE_COLS_V1_17D + CSR_6_COLS + COV_5_COLS

_EPS = 1e-8


def _make_rng(seed: int, tag: str) -> np.random.Generator:
    seed_u32 = zlib.crc32(f"{seed}:{tag}".encode("utf-8")) & 0xFFFFFFFF
    return np.random.default_rng(seed_u32)


def _sample_continuous_csr_in_polygon(
    poly_norm,
    n_points: int,
    rng: np.random.Generator,
    target_span: float = 100.0,
) -> np.ndarray:
    """
    Exact continuous Complete Spatial Randomness (CSR) rejection sampling inside
    a centroid-centered normalized polygon (`span == target_span`).

    Sampling from the fixed canonical square [-target_span, target_span]^2 ensures
    that candidate draws depend ONLY on the RNG seed, never on floating-point bbox jitter.
    """
    accepted = []
    n_acc = 0
    batch_size = max(16384, n_points * 4)
    half_box = float(target_span)

    for _ in range(64):
        cand = rng.uniform(-half_box, half_box, size=(batch_size, 2))
        mask = shapely.contains_xy(poly_norm, cand[:, 0], cand[:, 1])
        pts_in = cand[mask]
        if pts_in.shape[0] > 0:
            accepted.append(pts_in)
            n_acc += pts_in.shape[0]
            if n_acc >= n_points:
                break

    if n_acc == 0:
        return np.zeros((n_points, 2), dtype=np.float64)

    out = np.vstack(accepted)
    if out.shape[0] >= n_points:
        return out[:n_points]

    # Rare ultra-thin sliver polygon fallback: repeat with replacement
    extra_idx = rng.integers(0, out.shape[0], size=n_points - out.shape[0])
    return np.vstack([out, out[extra_idx]])


def _mean_1nn_single(pts: np.ndarray) -> float:
    n = pts.shape[0]
    if n < 2:
        return np.nan
    if n <= 160:
        diff = pts[:, None, :] - pts[None, :, :]
        sq = np.sum(diff * diff, axis=-1)
        np.fill_diagonal(sq, np.inf)
        return float(np.sqrt(sq.min(axis=1)).mean())
    dists, _ = cKDTree(pts).query(pts, k=2)
    return float(dists[:, 1].mean())


def _batch_mean_1nn(samples: np.ndarray) -> np.ndarray:
    """
    Compute mean 1st-nearest-neighbor distance for a batch of point sets of shape (B, n, 2).
    """
    b_total, n, _ = samples.shape
    if n < 2:
        return np.full(b_total, np.nan, dtype=np.float64)

    if n <= 160:
        out = np.empty(b_total, dtype=np.float64)
        diag_idx = np.arange(n)
        chunk_size = 32
        for start in range(0, b_total, chunk_size):
            end = min(start + chunk_size, b_total)
            chunk = samples[start:end]
            diff = chunk[:, :, None, :] - chunk[:, None, :, :]
            sq = np.sum(diff * diff, axis=-1)
            sq[:, diag_idx, diag_idx] = np.inf
            out[start:end] = np.sqrt(sq.min(axis=2)).mean(axis=1)
        return out

    # Single batched 3D cKDTree separating each Monte Carlo replicate along z by 1e4 (> target_span)
    z_sep = (np.arange(b_total, dtype=np.float64) * 10000.0)[:, None, None]
    z_col = np.broadcast_to(z_sep, (b_total, n, 1))
    pts3d = np.concatenate([samples, z_col], axis=2).reshape(b_total * n, 3)
    dists, _ = cKDTree(pts3d).query(pts3d, k=2)
    return dists[:, 1].reshape(b_total, n).mean(axis=1)


class _CellCSRContext:
    """
    Per-cell precomputed continuous CSR background and per-n Monte Carlo null cache.
    """

    def __init__(
        self,
        cid: str,
        cell_poly_norm,
        nuc_poly_norm,
        *,
        n_bg: int = 8192,
        n_mc: int = 128,
        random_seed: int = 42,
        target_span: float = 100.0,
    ):
        self.cid = cid
        self.cell_poly = cell_poly_norm
        self.nuc_poly = nuc_poly_norm
        self.n_mc = int(n_mc)
        self.random_seed = int(random_seed)

        # Cell morphology covariates (scale-invariant)
        cell_area = float(cell_poly_norm.area)
        ch_area = float(cell_poly_norm.convex_hull.area)
        self.aspect_ratio = float(_poly_aspect_ratio(cell_poly_norm))
        self.solidity = cell_area / ch_area if ch_area > 0 else np.nan

        if nuc_poly_norm is not None and nuc_poly_norm.area > 0 and cell_area > 0:
            raw_nc = float(nuc_poly_norm.area / cell_area)
            self.nc_ratio = raw_nc
            self.p0_nuc = float(np.clip(raw_nc, 1e-6, 1.0 - 1e-6))
        else:
            self.nc_ratio = np.nan
            self.p0_nuc = np.nan

        # Continuous CSR pool inside normalized cell polygon
        rng_bg = _make_rng(self.random_seed, f"csr_bg:{cid}")
        self.bg_pts = _sample_continuous_csr_in_polygon(
            cell_poly_norm,
            n_points=n_bg,
            rng=rng_bg,
            target_span=target_span,
        )
        bg_geom = shapely.points(self.bg_pts)

        # Analytical 1st-moment null for cell boundary distance
        bg_edge_dists = shapely.distance(cell_poly_norm.boundary, bg_geom)
        self.mu_edge = float(np.mean(bg_edge_dists))
        self.sd_edge = float(np.std(bg_edge_dists, ddof=1))

        # Unit vectors relative to cell centroid (0, 0) for Rayleigh radial R
        bg_norms = np.linalg.norm(self.bg_pts, axis=1, keepdims=True)
        self.bg_units = self.bg_pts / np.maximum(bg_norms, 1e-12)

        # Analytical 1st-moment null for cytoplasmic perinuclear distance
        if nuc_poly_norm is not None:
            bg_in_nuc = shapely.contains_xy(
                nuc_poly_norm, self.bg_pts[:, 0], self.bg_pts[:, 1]
            )
            bg_cyto_geom = bg_geom[~bg_in_nuc]
            if bg_cyto_geom.shape[0] >= 20:
                bg_perinuc_dists = shapely.distance(nuc_poly_norm.boundary, bg_cyto_geom)
                self.mu_perinuc = float(np.mean(bg_perinuc_dists))
                self.sd_perinuc = float(np.std(bg_perinuc_dists, ddof=1))
            else:
                self.mu_perinuc = np.nan
                self.sd_perinuc = np.nan
        else:
            self.mu_perinuc = np.nan
            self.sd_perinuc = np.nan

        # Cache: n -> (mu_shift, sd_shift, mu_nn, sd_nn, mu_rayleigh, sd_rayleigh)
        self._null_cache: Dict[int, Tuple[float, float, float, float, float, float]] = {}

    def get_mc_null_stats(self, n: int) -> Tuple[float, float, float, float, float, float]:
        if n in self._null_cache:
            return self._null_cache[n]

        m_pool = self.bg_pts.shape[0]
        rng_n = _make_rng(self.random_seed, f"csr_mc:{self.cid}:{n}")

        # Sample without replacement within each draw to avoid zero-distance collisions
        if n <= m_pool:
            idx = np.empty((self.n_mc, n), dtype=np.int64)
            for b in range(self.n_mc):
                idx[b] = rng_n.choice(m_pool, size=n, replace=False)
            samples = self.bg_pts[idx]
            unit_samples = self.bg_units[idx]
        else:
            # Extreme high-count gene (>8192 transcripts in one cell): subsample cap n_eff=512
            n_eff = 512
            idx = np.empty((self.n_mc, n_eff), dtype=np.int64)
            for b in range(self.n_mc):
                idx[b] = rng_n.choice(m_pool, size=n_eff, replace=False)
            samples = self.bg_pts[idx]
            unit_samples = self.bg_units[idx]

        null_shifts = np.linalg.norm(samples.mean(axis=1), axis=1)
        null_rayleigh = np.linalg.norm(unit_samples.mean(axis=1), axis=1)
        null_nn = _batch_mean_1nn(samples) if n >= 2 else np.full(self.n_mc, np.nan)

        stats = (
            float(np.mean(null_shifts)),
            float(np.std(null_shifts, ddof=1)),
            float(np.nanmean(null_nn)) if n >= 2 else np.nan,
            float(np.nanstd(null_nn, ddof=1)) if n >= 2 else np.nan,
            float(np.mean(null_rayleigh)),
            float(np.std(null_rayleigh, ddof=1)),
        )
        self._null_cache[n] = stats
        return stats

    def extract_gene_features(
        self,
        pts_norm: np.ndarray,
        pts_geom: np.ndarray,
        in_nuc_mask: Optional[np.ndarray],
        precomp_cell_edge_dists: Optional[np.ndarray] = None,
        precomp_nuc_edge_dists: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        n = int(pts_norm.shape[0])
        log1p_n = float(np.log1p(n))

        # 1. Nuclear fraction & binomial CSR Z-score
        if self.nuc_poly is not None and in_nuc_mask is not None and np.isfinite(self.p0_nuc):
            k_nuc = int(np.sum(in_nuc_mask))
            nuc_frac = k_nuc / float(n)
            se_nuc = np.sqrt(self.p0_nuc * (1.0 - self.p0_nuc) / float(n))
            csr_z_nuclear_frac = float((nuc_frac - self.p0_nuc) / (se_nuc + _EPS))
        else:
            nuc_frac = np.nan
            csr_z_nuclear_frac = np.nan

        # 2. Cell membrane proximity CSR Z-score (positive = closer to cell edge)
        obs_edge_mean = float(
            np.mean(precomp_cell_edge_dists)
            if precomp_cell_edge_dists is not None
            else np.mean(shapely.distance(self.cell_poly.boundary, pts_geom))
        )
        csr_z_cell_edge_dist = float(
            (self.mu_edge - obs_edge_mean) / (self.sd_edge / np.sqrt(n) + _EPS)
        )

        # 3. Perinuclear proximity CSR Z-score conditioned on cytoplasmic spots
        if (
            self.nuc_poly is not None
            and in_nuc_mask is not None
            and np.isfinite(self.mu_perinuc)
        ):
            cyto_mask = ~in_nuc_mask
            n_cyto = int(np.sum(cyto_mask))
            if n_cyto >= 1:
                obs_perinuc_mean = float(
                    np.mean(precomp_nuc_edge_dists[cyto_mask])
                    if precomp_nuc_edge_dists is not None
                    else np.mean(shapely.distance(self.nuc_poly.boundary, pts_geom[cyto_mask]))
                )
                csr_z_perinuc_cyto_dist = float(
                    (self.mu_perinuc - obs_perinuc_mean)
                    / (self.sd_perinuc / np.sqrt(n_cyto) + _EPS)
                )
            else:
                csr_z_perinuc_cyto_dist = np.nan
        else:
            csr_z_perinuc_cyto_dist = np.nan

        # 4-6. Monte Carlo CSR Z-scores (centroid shift, 1-NN clustering, Rayleigh radial R)
        # If n > 8192, deterministically subsample 512 points to match null cap
        if n > self.bg_pts.shape[0]:
            rng_sub = _make_rng(self.random_seed, f"obs_cap:{self.cid}:{n}")
            sub_idx = rng_sub.choice(n, size=512, replace=False)
            pts_eval = pts_norm[sub_idx]
        else:
            pts_eval = pts_norm

        mu_shift, sd_shift, mu_nn, sd_nn, mu_ray, sd_ray = self.get_mc_null_stats(
            int(pts_eval.shape[0])
        )

        obs_shift = float(np.linalg.norm(pts_eval.mean(axis=0)))
        csr_z_centroid_shift = float((obs_shift - mu_shift) / (sd_shift + _EPS))

        if pts_eval.shape[0] >= 2 and np.isfinite(mu_nn):
            obs_nn = _mean_1nn_single(pts_eval)
            csr_z_nn_clustering = float((mu_nn - obs_nn) / (sd_nn + _EPS))
        else:
            csr_z_nn_clustering = np.nan

        pt_norms = np.linalg.norm(pts_eval, axis=1, keepdims=True)
        pt_units = pts_eval / np.maximum(pt_norms, 1e-12)
        obs_ray = float(np.linalg.norm(pt_units.mean(axis=0)))
        csr_z_rayleigh_radial = float((obs_ray - mu_ray) / (sd_ray + _EPS))

        return {
            "csr_z_nuclear_frac": csr_z_nuclear_frac,
            "csr_z_cell_edge_dist": csr_z_cell_edge_dist,
            "csr_z_perinuc_cyto_dist": csr_z_perinuc_cyto_dist,
            "csr_z_centroid_shift": csr_z_centroid_shift,
            "csr_z_nn_clustering": csr_z_nn_clustering,
            "csr_z_rayleigh_radial": csr_z_rayleigh_radial,
            "feat_log1p_n_spots": log1p_n,
            "feat_nuclear_fraction": nuc_frac,
            "feat_cell_nc_ratio": self.nc_ratio,
            "feat_cell_aspect_ratio": self.aspect_ratio,
            "feat_cell_solidity": self.solidity,
        }


def compute_csr_and_cov_from_dict(
    pkl_file_path: str | dict,
    *,
    cell_id_col: str = "cell",
    gene_col: str = "gene",
    x_col: str = "x",
    y_col: str = "y",
    target_span: float = 100.0,
    n_bg: int = 8192,
    n_mc: int = 128,
    random_seed: int = 42,
    profile: bool = False,
    max_genes_per_cell: int | None = None,
    max_spots_per_gene: int | None = None,
) -> pd.DataFrame:
    """
    Compute Block B (6 same-cell CSR null Z-scores) + Block C (5 morphology & abundance covariates)
    for every (cell, gene) pair in a PKL path or in-memory bundle dict.
    """
    rep = TimerReport() if profile else None

    with timed("csr:load_data", rep, print_each=profile):
        data_df, cell_boundary_dict, nuclear_boundary_dict = load_and_process_data(
            pkl_file_path,
            build_geometry=False,
        )

    data_df = data_df.copy()
    data_df[cell_id_col] = data_df[cell_id_col].astype(str)
    data_df[gene_col] = data_df[gene_col].astype(str)

    cell_boundary_dict = {str(k): v for k, v in (cell_boundary_dict or {}).items()}
    nuclear_boundary_dict = {str(k): v for k, v in (nuclear_boundary_dict or {}).items()}

    rows: list[dict] = []
    idx: list[tuple[str, str]] = []

    with timed("csr:loop_cells", rep, print_each=profile):
        for cid_raw, cell_df in data_df.groupby(cell_id_col, sort=False):
            cid = str(cid_raw)
            cell_poly = _to_single_polygon(cell_boundary_dict.get(cid))
            if cell_poly is None:
                continue
            nuc_poly = _to_single_polygon(nuclear_boundary_dict.get(cid)) if nuclear_boundary_dict else None

            cell_xy_all = cell_df[[x_col, y_col]].to_numpy(dtype=np.float64)
            try:
                cell_poly_norm, nuc_poly_norm, cell_xy_norm, _ = scale_normalize_cell_geometry(
                    cell_poly,
                    nuc_poly=nuc_poly,
                    pts_xy=cell_xy_all,
                    target_span=target_span,
                )
            except Exception:
                continue

            ctx = _CellCSRContext(
                cid=cid,
                cell_poly_norm=cell_poly_norm,
                nuc_poly_norm=nuc_poly_norm,
                n_bg=n_bg,
                n_mc=n_mc,
                random_seed=random_seed,
                target_span=target_span,
            )

            cell_pts_geom = shapely.points(cell_xy_norm)
            cell_edge_dists_all = shapely.distance(cell_poly_norm.boundary, cell_pts_geom)
            if nuc_poly_norm is not None:
                cell_in_nuc = shapely.contains_xy(
                    nuc_poly_norm, cell_xy_norm[:, 0], cell_xy_norm[:, 1]
                )
                nuc_edge_dists_all = shapely.distance(nuc_poly_norm.boundary, cell_pts_geom)
            else:
                cell_in_nuc = None
                nuc_edge_dists_all = None

            genes_arr = cell_df[gene_col].to_numpy()
            pos_series = pd.Series(np.arange(len(cell_df)), index=genes_arr)
            gene_pos_groups = pos_series.groupby(level=0, sort=False)

            if max_genes_per_cell is not None:
                gene_sizes = gene_pos_groups.size().sort_values(ascending=False)
                top_genes = set(gene_sizes.head(max_genes_per_cell).index.astype(str))
            else:
                top_genes = None

            for gene_raw, pos_s in gene_pos_groups:
                gene = str(gene_raw)
                if top_genes is not None and gene not in top_genes:
                    continue

                pos_idx = pos_s.to_numpy()
                if max_spots_per_gene is not None and pos_idx.size > max_spots_per_gene:
                    pos_idx = pos_idx[:max_spots_per_gene]
                if pos_idx.size == 0:
                    continue

                feats = ctx.extract_gene_features(
                    pts_norm=cell_xy_norm[pos_idx],
                    pts_geom=cell_pts_geom[pos_idx],
                    in_nuc_mask=cell_in_nuc[pos_idx] if cell_in_nuc is not None else None,
                    precomp_cell_edge_dists=cell_edge_dists_all[pos_idx],
                    precomp_nuc_edge_dists=nuc_edge_dists_all[pos_idx] if nuc_edge_dists_all is not None else None,
                )
                idx.append((cid, gene))
                rows.append(feats)

    out = pd.DataFrame(
        rows,
        columns=CSR_6_COLS + COV_5_COLS,
        index=pd.MultiIndex.from_tuples(idx, names=[cell_id_col, gene_col]),
    )
    if profile and rep is not None:
        print(rep.summary())
    return out
