"""Unified 3-method runner for subcellular compartment discovery benchmarks.

Wraps official / primary implementations under a strictly blind-nuclear protocol
(ZERO ``nuclear_boundary`` input is ever passed to any method):
  1. DeCoMap (ours: dual-scale bilateral shrinkage + closed-form CSR gate + MiniBatchNMF + cosine merge)
  2. Bento (official bento-tools v2.1.4 ``bt.tl.flux`` + ``bt.tl.fluxmap``)
  3. Points2Regions (official points2regions v0.0.6 pyramidal multi-scale composition + cosine merge)

Every runner returns a standardized dictionary aligned row-for-row with ``bundle["data_df"]``:
  - method: str
  - labels: (N,) int64 array in {-1, 0, ..., K-1} (-1 = unstructured / unassigned)
  - raw_labels: (N,) int64 array in {0, ..., K-1}
  - best_k: int
  - structured_spot_fraction: float
  - gene_profiles: (K, G) float64 empirical log2 fold-enrichment over global gene composition
  - genes: tuple[str, ...]
  - elapsed_seconds: float
  - peak_rss_gb: float
"""

from __future__ import annotations

import math
import os
from pathlib import Path
import resource
import sys
import time
from typing import Any, Union
import warnings

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from shapely.affinity import scale as shp_scale, translate as shp_translate

STRAND_ROOT = Path(__file__).resolve().parents[1]
if str(STRAND_ROOT) not in sys.path:
    sys.path.insert(0, str(STRAND_ROOT))

from tools.decomap.api import _load_decomap_geometries, run_decomap_from_bundle
from tools.engines.coloc_region_annotation import _equivalent_radius, _safe_area


def _rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 ** 2)


def compute_empirical_compartment_gene_profiles(
    raw_labels: np.ndarray,
    genes_col: np.ndarray,
    sorted_genes: tuple[str, ...],
    k_count: int,
) -> np.ndarray:
    """Compute standardized (K, G) log2 enrichment matrix directly from assigned transcripts.

    Using the exact same empirical definition across DeCoMap, Bento, and Points2Regions
    ensures 100% apples-to-apples cross-replicate correlation and APEX-seq enrichment.
    """
    n_genes = len(sorted_genes)
    valid = raw_labels >= 0
    if not np.any(valid) or k_count <= 0:
        return np.zeros((max(k_count, 1), n_genes), dtype=np.float64)

    g_codes = pd.Categorical(genes_col[valid], categories=list(sorted_genes)).codes.astype(np.int64)
    lbl_v = raw_labels[valid].astype(np.int64)
    counts = np.zeros((k_count, n_genes), dtype=np.float64)
    np.add.at(counts, (lbl_v, g_codes), 1.0)

    comp_p = (counts + 0.5) / np.maximum(counts.sum(axis=1, keepdims=True) + 0.5 * n_genes, 1.0)
    glob_p = (counts.sum(axis=0, keepdims=True) + 0.5) / max(float(counts.sum() + 0.5 * n_genes), 1.0)
    return np.log2(comp_p / glob_p)


def _strip_nuclear_boundary(bundle: dict[str, Any]) -> dict[str, Any]:
    """Guarantee zero nuclear_boundary access across all benchmarked methods."""
    return {
        "data_df": bundle["data_df"][["cell", "gene", "x", "y"]].copy(),
        "cell_boundary": bundle.get("cell_boundary", {}),
        "nuclear_boundary": {},
    }


def _prepare_non_overlapping_scaled_cells(
    df: pd.DataFrame,
    cell_geoms: dict[str, Any],
    *,
    target_median_radius: float = 100.0,
) -> tuple[pd.DataFrame, list[Any], list[str], np.ndarray, float]:
    """Arrange cells on a non-overlapping layout and normalize micron datasets so Bento's int cast is lossless."""
    all_cids = [c for c in sorted(df["cell"].astype(str).unique()) if c in cell_geoms]
    radii = np.array(
        [_equivalent_radius(_safe_area(cell_geoms[c].cell)) for c in all_cids],
        dtype=np.float64,
    )
    med_r = float(np.median(radii[radii > 0])) if np.any(radii > 0) else 50.0
    # Bento's fluxmap casts raster coordinates to int before contouring; if R_med < 40 (e.g. microns),
    # scaling up to R_med = 100 preserves sub-micron precision cleanly.
    scale_fac = (target_median_radius / med_r) if med_r < 40.0 else 1.0

    bounds = np.array([cell_geoms[c].cell.bounds for c in all_cids], dtype=np.float64) * scale_fac
    spans = np.maximum(bounds[:, 2] - bounds[:, 0], bounds[:, 3] - bounds[:, 1])
    max_span = float(np.max(spans)) if len(spans) else 200.0

    centers = np.column_stack([(bounds[:, 0] + bounds[:, 2]) * 0.5, (bounds[:, 1] + bounds[:, 3]) * 0.5])
    needs_shift = False
    if len(centers) >= 2:
        sample_c = centers[: min(40, len(centers))]
        dmat = np.linalg.norm(sample_c[:, None, :] - sample_c[None, :, :], axis=2)
        np.fill_diagonal(dmat, np.inf)
        if float(np.min(dmat)) < 0.25 * max_span:
            needs_shift = True

    c_set = set(all_cids)
    keep_mask = df["cell"].astype(str).isin(c_set).to_numpy()
    kept_positions = np.flatnonzero(keep_mask)
    sub_df = df.iloc[kept_positions].copy()
    sub_df["x"] = sub_df["x"].to_numpy(dtype=np.float64) * scale_fac
    sub_df["y"] = sub_df["y"].to_numpy(dtype=np.float64) * scale_fac

    if not needs_shift:
        # Ensure all coordinates are strictly positive (Bento rasterization prefers positive coordinates)
        min_x = float(min(sub_df["x"].min(), bounds[:, 0].min()))
        min_y = float(min(sub_df["y"].min(), bounds[:, 1].min()))
        shift_x = (50.0 - min_x) if min_x < 10.0 else 0.0
        shift_y = (50.0 - min_y) if min_y < 10.0 else 0.0
        c_polys = []
        for cid in all_cids:
            p = cell_geoms[cid].cell
            if scale_fac != 1.0:
                p = shp_scale(p, xfact=scale_fac, yfact=scale_fac, origin=(0.0, 0.0))
            if shift_x != 0.0 or shift_y != 0.0:
                p = shp_translate(p, xoff=shift_x, yoff=shift_y)
            c_polys.append(p)
        sub_df["x"] += shift_x
        sub_df["y"] += shift_y
        return sub_df, c_polys, all_cids, kept_positions, med_r * scale_fac

    step = max_span * 2.2 + 50.0
    n_cols = max(1, int(math.ceil(math.sqrt(len(all_cids)))))
    dx_map, dy_map = {}, {}
    c_polys = []
    for idx, cid in enumerate(all_cids):
        p = cell_geoms[cid].cell
        if scale_fac != 1.0:
            p = shp_scale(p, xfact=scale_fac, yfact=scale_fac, origin=(0.0, 0.0))
        b = p.bounds
        gx, gy = idx % n_cols, idx // n_cols
        ox = 50.0 + gx * step - b[0]
        oy = 50.0 + gy * step - b[1]
        dx_map[cid] = ox
        dy_map[cid] = oy
        c_polys.append(shp_translate(p, xoff=ox, yoff=oy))

    c_str = sub_df["cell"].astype(str)
    sub_df["x"] = sub_df["x"].to_numpy(dtype=np.float64) + c_str.map(dx_map).to_numpy(dtype=np.float64)
    sub_df["y"] = sub_df["y"].to_numpy(dtype=np.float64) + c_str.map(dy_map).to_numpy(dtype=np.float64)
    return sub_df, c_polys, all_cids, kept_positions, med_r * scale_fac


# ---------------------------------------------------------------------------
# 1. DeCoMap (ours)
# ---------------------------------------------------------------------------
def run_method_decomap(
    bundle: dict[str, Any],
    *,
    n_clusters: Union[str, int] = "auto",
    cluster_range: tuple[int, int] = (4, 7),
    calibrate_csr: bool = True,
    threads: int = 16,
    random_state: int = 42,
) -> dict[str, Any]:
    clean_bundle = _strip_nuclear_boundary(bundle)
    t0 = time.perf_counter()
    res = run_decomap_from_bundle(
        clean_bundle,
        n_clusters=n_clusters,
        cluster_range_min=cluster_range[0],
        cluster_range_max=cluster_range[1],
        calibrate_csr=calibrate_csr,
        include_posthoc_nuclear_qc=False,
        random_state=random_state,
        threads=threads,
    )
    elapsed = time.perf_counter() - t0
    peak_gb = _rss_gb()

    labels = res.point_df["subdomain"].to_numpy(dtype=np.int64)
    raw_labels = res.point_df["raw_subdomain"].to_numpy(dtype=np.int64)
    genes = res.genes
    emp_profiles = compute_empirical_compartment_gene_profiles(
        raw_labels,
        res.point_df["gene"].astype(str).to_numpy(),
        genes,
        res.best_k,
    )
    return {
        "method": "DeCoMap",
        "labels": labels,
        "raw_labels": raw_labels,
        "best_k": int(res.best_k),
        "structured_spot_fraction": float(np.mean(labels >= 0)),
        "gene_profiles": emp_profiles,
        "compartment_summary": res.compartment_summary,
        "genes": genes,
        "elapsed_seconds": float(elapsed),
        "peak_rss_gb": float(peak_gb),
    }


# ---------------------------------------------------------------------------
# 2. Official Bento (bt.tl.flux + bt.tl.fluxmap)
# ---------------------------------------------------------------------------
def run_method_bento_fluxmap(
    bundle: dict[str, Any],
    *,
    n_clusters: Union[str, int] = "auto",
    cluster_range: tuple[int, int] = (4, 7),
    threads: int = 8,
    random_state: int = 42,
) -> dict[str, Any]:
    import dask
    import geopandas as gpd
    import spatialdata as sd
    from spatialdata.models import PointsModel, ShapesModel
    import bento as bt

    warnings.filterwarnings("ignore")
    clean_bundle = _strip_nuclear_boundary(bundle)
    df = clean_bundle["data_df"]
    df["cell"] = df["cell"].astype(str)
    df["gene"] = df["gene"].astype(str)
    sorted_genes = tuple(sorted(df["gene"].unique().tolist()))

    t0 = time.perf_counter()
    geoms = _load_decomap_geometries(clean_bundle, include_posthoc_nuclear_qc=False)
    df_shifted, c_polys, cids, kept_positions, scaled_med_r = _prepare_non_overlapping_scaled_cells(
        df, geoms, target_median_radius=100.0
    )

    # Match Bento's U2OS paper grid density (~10 units for R_med ~ 100 -> ~314 raster pixels/cell)
    grid_step = max(2.0, round(scaled_med_r / 10.0, 1))
    res_val = 1.0 / grid_step

    # Adaptive min_count so Bento does not blank out 100% of pixels on moderate-density cells
    spots_per_cell = float(df_shifted.groupby("cell").size().median())
    min_count = min(50, max(5, int(round(0.03 * spots_per_cell))))

    orig_genes = list(sorted_genes)
    g2tok = {g: f"GENE{idx:05d}" for idx, g in enumerate(orig_genes)}
    pts_df = pd.DataFrame(
        {
            "x": df_shifted["x"].to_numpy(dtype=np.float64),
            "y": df_shifted["y"].to_numpy(dtype=np.float64),
            "feature_name": pd.Categorical(df_shifted["gene"].map(g2tok)),
        }
    )
    sdata = sd.SpatialData(
        shapes={
            "cell_boundaries": ShapesModel.parse(gpd.GeoDataFrame(geometry=c_polys, index=cids)),
        },
        points={
            "transcripts": PointsModel.parse(
                pts_df, coordinates={"x": "x", "y": "y"}, feature_key="feature_name"
            )
        },
    )

    sdata = bt.io.prep(sdata, shape_keys=["cell_boundaries"])
    with dask.config.set(scheduler="threads"):
        bt.tl.flux(
            sdata,
            instance_key="cell_boundaries",
            method="radius",
            radius=None,  # official default: 1/3 of mean cell radius
            res=res_val,
            random_state=random_state,
            recompute=True,
            num_workers=max(1, min(int(threads), 8)),
        )

    if isinstance(n_clusters, str) and n_clusters.strip().lower() == "auto":
        k_arg: Any = range(int(cluster_range[0]), int(cluster_range[1]) + 1)
    else:
        k_arg = int(n_clusters)

    bt.tl.fluxmap(
        sdata,
        instance_key="cell_boundaries",
        n_clusters=k_arg,
        min_count=min_count,
        res=res_val,
        random_state=random_state,
    )

    raster_df = sdata.points["cell_boundaries_raster"]
    if hasattr(raster_df, "compute"):
        raster_df = raster_df.compute()
    # If KneeLocator failed in official fluxmap (`best_k is None`), rerun with fallback k=5 (matching Bento U2OS paper)
    if "fluxmap" not in raster_df.columns:
        fallback_k = max(cluster_range[0], min(cluster_range[1], 5)) if isinstance(k_arg, range) else int(k_arg)
        bt.tl.fluxmap(
            sdata,
            instance_key="cell_boundaries",
            n_clusters=fallback_k,
            min_count=min_count,
            res=res_val,
            random_state=random_state,
        )
        raster_df = sdata.points["cell_boundaries_raster"]
        if hasattr(raster_df, "compute"):
            raster_df = raster_df.compute()

    # Map each transcript to its nearest same-cell raster pixel's fluxmap label:
    # fluxmap == 0 means unassigned (< min_count); 1..K map to 0..K-1.
    labels_kept = np.full(len(df_shifted), -1, dtype=np.int64)
    raw_kept = np.zeros(len(df_shifted), dtype=np.int64)

    r_groups = raster_df.groupby("cell_boundaries", observed=True)
    t_groups = df_shifted.groupby("cell", sort=False, observed=True).indices
    t_xy = df_shifted[["x", "y"]].to_numpy(dtype=np.float64)

    for cid, t_idx in t_groups.items():
        if cid not in r_groups.groups:
            continue
        sub_r = r_groups.get_group(cid)
        r_xy = sub_r[["x", "y"]].to_numpy(dtype=np.float64)
        r_fmap = sub_r["fluxmap"].to_numpy(dtype=np.int64)
        if len(r_xy) == 0:
            continue
        _, nn = cKDTree(r_xy).query(t_xy[t_idx], k=1)
        fvals = r_fmap[nn]
        labels_kept[t_idx] = np.where(fvals > 0, fvals - 1, -1)
        # For raw_labels, assign any 0-pixel transcript to the nearest positive-fluxmap pixel in that cell
        pos_mask = r_fmap > 0
        if np.any(pos_mask):
            _, nn_pos = cKDTree(r_xy[pos_mask]).query(t_xy[t_idx], k=1)
            raw_kept[t_idx] = r_fmap[pos_mask][nn_pos] - 1

    n_total = len(df)
    labels_full = np.full(n_total, -1, dtype=np.int64)
    raw_full = np.zeros(n_total, dtype=np.int64)
    labels_full[kept_positions] = labels_kept
    raw_full[kept_positions] = raw_kept

    best_k = int(max(raw_full.max() + 1, 1)) if len(raw_full) else 1
    elapsed = time.perf_counter() - t0
    peak_gb = _rss_gb()

    emp_profiles = compute_empirical_compartment_gene_profiles(
        raw_full, df["gene"].to_numpy(), sorted_genes, best_k
    )
    return {
        "method": "Bento (RNAflux)",
        "labels": labels_full,
        "raw_labels": raw_full,
        "best_k": best_k,
        "structured_spot_fraction": float(np.mean(labels_full >= 0)),
        "gene_profiles": emp_profiles,
        "genes": sorted_genes,
        "elapsed_seconds": float(elapsed),
        "peak_rss_gb": float(peak_gb),
    }


# ---------------------------------------------------------------------------
# 3. Official Points2Regions (subcellular pyramidal composition + cosine merge)
# ---------------------------------------------------------------------------
def run_method_points2regions(
    bundle: dict[str, Any],
    *,
    n_clusters: Union[str, int] = "auto",
    cluster_range: tuple[int, int] = (4, 7),
    per_cell_datasetids: bool = True,
    random_state: int = 42,
) -> dict[str, Any]:
    from points2regions import Points2Regions

    warnings.filterwarnings("ignore")
    clean_bundle = _strip_nuclear_boundary(bundle)
    df = clean_bundle["data_df"]
    df["cell"] = df["cell"].astype(str)
    df["gene"] = df["gene"].astype(str)
    sorted_genes = tuple(sorted(df["gene"].unique().tolist()))

    t0 = time.perf_counter()
    geoms = _load_decomap_geometries(clean_bundle, include_posthoc_nuclear_qc=False)
    radii = [
        _equivalent_radius(_safe_area(g.cell))
        for g in geoms.values()
        if g.cell is not None and _safe_area(g.cell) > 0
    ]
    if radii:
        r_med = float(np.median(radii))
    else:
        # Fallback from empirical per-cell spread
        sample_c = df.groupby("cell", observed=True)[["x", "y"]].std().median().mean()
        r_med = max(float(sample_c) * 2.0, 10.0)

    # Subcellular pyramidal grid: finest bin 0.08 * R_med, 4-level smoothing multiplier 4.0 (-> 0.32 * R_med)
    pixel_width = max(0.08 * r_med, 1e-3)
    pixel_smoothing = 4.0

    xy = df[["x", "y"]].to_numpy(dtype=np.float32)
    genes_arr = df["gene"].to_numpy()
    if per_cell_datasetids:
        cell_codes = pd.Categorical(df["cell"]).codes.astype(np.int32)
    else:
        cell_codes = None

    k_target = (
        5
        if (isinstance(n_clusters, str) and n_clusters.strip().lower() == "auto")
        else int(n_clusters)
    )
    p2r = Points2Regions(
        xy,
        genes_arr,
        pixel_width=pixel_width,
        pixel_smoothing=pixel_smoothing,
        min_num_pts_per_pixel=0.0,
        datasetids=cell_codes,
    )
    pred = p2r.fit_predict(num_clusters=k_target, output="marker", seed=random_state)
    labels = np.asarray(pred, dtype=np.int64)
    raw_labels = np.where(labels >= 0, labels, 0).astype(np.int64)
    best_k = int(max(raw_labels.max() + 1, k_target))

    elapsed = time.perf_counter() - t0
    peak_gb = _rss_gb()
    emp_profiles = compute_empirical_compartment_gene_profiles(
        raw_labels, genes_arr, sorted_genes, best_k
    )
    return {
        "method": "Points2Regions",
        "labels": labels,
        "raw_labels": raw_labels,
        "best_k": best_k,
        "structured_spot_fraction": float(np.mean(labels >= 0)),
        "gene_profiles": emp_profiles,
        "genes": sorted_genes,
        "elapsed_seconds": float(elapsed),
        "peak_rss_gb": float(peak_gb),
    }


def run_all_three_compartment_methods(
    bundle: dict[str, Any],
    *,
    n_clusters: Union[str, int] = "auto",
    cluster_range: tuple[int, int] = (4, 7),
    threads: int = 8,
    random_state: int = 42,
) -> dict[str, dict[str, Any]]:
    """Run DeCoMap, Bento (RNAflux), and Points2Regions under identical blind-nuclear inputs."""
    out: dict[str, dict[str, Any]] = {}
    out["DeCoMap"] = run_method_decomap(
        bundle,
        n_clusters=n_clusters,
        cluster_range=cluster_range,
        calibrate_csr=True,
        threads=threads,
        random_state=random_state,
    )
    out["Bento (RNAflux)"] = run_method_bento_fluxmap(
        bundle,
        n_clusters=n_clusters,
        cluster_range=cluster_range,
        threads=threads,
        random_state=random_state,
    )
    out["Points2Regions"] = run_method_points2regions(
        bundle,
        n_clusters=n_clusters,
        cluster_range=cluster_range,
        random_state=random_state,
    )
    return out
