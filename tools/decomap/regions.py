"""Area-level compartment regions rasterized from DeCoMap per-transcript labels.

Each cell polygon is tiled with a grid of ``grid_frac * R_c`` pixels. A pixel takes the compartment
with the largest Gaussian-kernel (``sigma_frac * R_c``) confidence-weighted vote of nearby
transcripts. Pixels without enough nearby transcripts, or whose vote is nearly tied, stay
unassigned instead of being extrapolated. Islands smaller than the kernel footprint are absorbed
by their surrounding majority, then each compartment is polygonized and clipped to the cell.
Transcript labels are never changed.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import math
import os

import numpy as np
import pandas as pd
from scipy import ndimage, sparse
from scipy.spatial import cKDTree
import shapely

from tools.engines.coloc_region_annotation import CellGeometry

REGION_COLUMNS = ["cell", "subdomain", "compartment_name", "n_pixels", "area", "area_fraction", "geometry_wkt"]


def _cell_regions(
    cid: str,
    cell_poly,
    xy: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    n_labels: int,
    *,
    grid_frac: float,
    sigma_frac: float,
    min_support: float,
    min_vote_margin: float,
) -> list[dict]:
    cell_area = float(cell_poly.area)
    r_c = math.sqrt(cell_area / math.pi)
    step = grid_frac * r_c
    sigma = sigma_frac * r_c
    x0, y0, x1, y1 = cell_poly.bounds
    # Pixel boxes are built from these shared edges so neighbours meet exactly and union cleanly.
    edge_x = x0 + step * np.arange(max(1, math.ceil((x1 - x0) / step)) + 1)
    edge_y = y0 + step * np.arange(max(1, math.ceil((y1 - y0) / step)) + 1)
    gx = 0.5 * (edge_x[:-1] + edge_x[1:])
    gy = 0.5 * (edge_y[:-1] + edge_y[1:])
    if len(xy) == 0:
        return []
    mx, my = np.meshgrid(gx, gy)
    inside = shapely.contains_xy(cell_poly, mx, my)
    pix_rc = np.argwhere(inside)
    if len(pix_rc) == 0:
        return []
    pix_xy = np.column_stack([gx[pix_rc[:, 1]], gy[pix_rc[:, 0]]])

    dist = cKDTree(pix_xy).sparse_distance_matrix(cKDTree(xy), 3.0 * sigma, output_type="coo_matrix")
    kern = np.exp(-0.5 * (dist.data / sigma) ** 2)
    support = np.bincount(dist.row, weights=kern, minlength=len(pix_xy))
    vote_w = sparse.csr_matrix((kern * weights[dist.col], (dist.row, dist.col)), shape=dist.shape)
    onehot = sparse.csr_matrix(
        (np.ones(len(labels)), (np.arange(len(labels)), labels)), shape=(len(labels), n_labels)
    )
    votes = np.asarray((vote_w @ onehot).todense())
    total = votes.sum(axis=1)
    share = np.divide(votes, total[:, None], out=np.zeros_like(votes), where=total[:, None] > 0)
    share_sorted = np.sort(np.pad(share, ((0, 0), (1, 0))), axis=1)
    margin = share_sorted[:, -1] - share_sorted[:, -2]
    pix_label = np.argmax(votes, axis=1)
    pix_label[(support < min_support) | (margin < min_vote_margin)] = -1

    outside = -2
    grid = np.full(mx.shape, outside, dtype=np.int64)
    grid[pix_rc[:, 0], pix_rc[:, 1]] = pix_label

    # Islands and unassigned holes below the kernel footprint (pi * sigma^2) are beneath the
    # method's resolution; larger unassigned areas stay unassigned.
    min_island_px = max(1, int(round(math.pi * (sigma / step) ** 2)))
    for k in [-1, *range(n_labels)]:
        comp, n_comp = ndimage.label(grid == k)
        if n_comp == 0:
            continue
        sizes = np.bincount(comp.ravel())
        for c_idx in np.flatnonzero(sizes[1:] < min_island_px) + 1:
            island = comp == c_idx
            ring = grid[ndimage.binary_dilation(island) & ~island]
            ring = ring[(ring >= 0) & (ring != k)]
            grid[island] = np.bincount(ring).argmax() if len(ring) else -1

    rows = []
    for k in np.unique(grid[grid >= 0]):
        rr, cc = np.nonzero(grid == k)
        boxes = shapely.box(edge_x[cc], edge_y[rr], edge_x[cc + 1], edge_y[rr + 1])
        region = shapely.union_all(boxes).simplify(0.5 * step).intersection(cell_poly)
        if region.is_empty:
            continue
        rows.append(
            {
                "cell": cid,
                "subdomain": int(k),
                "n_pixels": int(len(rr)),
                "area": float(region.area),
                "area_fraction": float(region.area / cell_area),
                "geometry_wkt": region.wkt,
            }
        )
    return rows


def build_compartment_regions(
    point_df: pd.DataFrame,
    geometries: dict[str, CellGeometry],
    compartment_summary: pd.DataFrame,
    *,
    grid_frac: float = 0.05,
    sigma_frac: float = 0.07,
    min_support: float = 2.0,
    min_vote_margin: float = 0.10,
    threads: int = 0,
) -> pd.DataFrame:
    """Rasterize DeCoMap transcript labels into per-cell compartment polygons.

    ``min_support`` is the kernel-weighted transcript count a pixel needs to be assigned. Cells that
    failed the CSR gate (all ``subdomain`` == -1) or lack a boundary polygon produce no regions.
    The unassigned area of a cell is ``1 - sum(area_fraction)``.
    """
    n_labels = max(int(len(compartment_summary)), 1)
    struct = point_df[point_df["subdomain"] >= 0]
    xy_all = struct[["x", "y"]].to_numpy(dtype=np.float64)
    lab_all = struct["subdomain"].to_numpy(dtype=np.int64)
    w_all = struct["confidence"].to_numpy(dtype=np.float64) if "confidence" in struct else np.ones(len(struct))

    tasks = []
    for cid, idx in struct.groupby(struct["cell"].astype(str), sort=False).indices.items():
        geom = geometries.get(cid)
        if geom is None or geom.cell is None or geom.cell.is_empty or geom.cell.area <= 0:
            continue
        tasks.append((cid, geom.cell, xy_all[idx], lab_all[idx], w_all[idx]))

    def _run(task):
        return _cell_regions(
            *task,
            n_labels,
            grid_frac=grid_frac,
            sigma_frac=sigma_frac,
            min_support=min_support,
            min_vote_margin=min_vote_margin,
        )

    n_workers = threads if threads > 0 else max(1, min(16, os.cpu_count() or 1))
    if n_workers <= 1 or len(tasks) < 8:
        per_cell = [_run(t) for t in tasks]
    else:
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            per_cell = list(ex.map(_run, tasks))

    regions = pd.DataFrame([r for rows in per_cell for r in rows], columns=[c for c in REGION_COLUMNS if c != "compartment_name"])
    names = compartment_summary.set_index("subdomain")["compartment_name"].to_dict()
    regions.insert(2, "compartment_name", regions["subdomain"].map(names))
    return regions[REGION_COLUMNS]
