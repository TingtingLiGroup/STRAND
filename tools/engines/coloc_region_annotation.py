from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import math

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

try:
    import shapely
    from shapely.geometry import Polygon, MultiPolygon, GeometryCollection, Point
    from shapely.geometry.base import BaseGeometry
    from shapely.validation import make_valid
except Exception as e:  # pragma: no cover
    raise ImportError(
        "colocalization region annotation requires shapely. "
        "Please install shapely in the current environment."
    ) from e


# InSTAnT source defaults in InSTAnT.py::_spatial_category().
INSTANT_DISTANCE_THRESHOLD_NUCLEUS = 2.5
INSTANT_DISTANCE_THRESHOLD_CYTO_NUCLEAR = 2.5
INSTANT_DISTANCE_THRESHOLD_CYTO_PERI = 4.0
_EPS = 1e-12


@dataclass
class CellGeometry:
    cell_id: str
    cell: BaseGeometry
    nuclear: BaseGeometry


def _fix_geometry(geom: BaseGeometry | None) -> BaseGeometry | None:
    """Repair invalid/empty geometries and keep polygonal components only."""
    if geom is None:
        return None
    try:
        if geom.is_empty:
            return None
    except Exception:
        return None

    try:
        geom = make_valid(geom)
    except Exception:
        try:
            geom = geom.buffer(0)
        except Exception:
            return None

    if geom is None:
        return None
    try:
        if geom.is_empty:
            return None
    except Exception:
        return None

    if isinstance(geom, GeometryCollection):
        polys = []
        for g in geom.geoms:
            if isinstance(g, Polygon):
                polys.append(g)
            elif isinstance(g, MultiPolygon):
                polys.extend(list(g.geoms))
        polys = [p for p in polys if not p.is_empty and p.area > 0]
        if not polys:
            return None
        if len(polys) == 1:
            geom = polys[0]
        else:
            try:
                geom = shapely.union_all(polys)
            except Exception:
                geom = max(polys, key=lambda p: p.area)

    if isinstance(geom, MultiPolygon):
        polys = [p for p in geom.geoms if not p.is_empty and p.area > 0]
        if not polys:
            return None
        geom = max(polys, key=lambda p: p.area)

    try:
        if geom.area <= 0:
            return None
    except Exception:
        return None
    return geom


def _boundary_to_polygon(boundary) -> BaseGeometry | None:
    """Convert common boundary formats into a shapely polygon."""
    if boundary is None:
        return None
    if hasattr(boundary, "geom_type"):
        return _fix_geometry(boundary)

    if isinstance(boundary, pd.DataFrame):
        if {"x", "y"}.issubset(boundary.columns):
            xy = boundary[["x", "y"]].to_numpy(float)
        else:
            return None
    else:
        try:
            xy = np.asarray(boundary, dtype=float)
        except Exception:
            return None

    if xy.ndim != 2 or xy.shape[1] < 2 or len(xy) < 3:
        return None
    xy = xy[:, :2]
    xy = xy[np.isfinite(xy).all(axis=1)]
    if len(xy) < 3:
        return None
    try:
        return _fix_geometry(Polygon(xy))
    except Exception:
        return None


def _safe_area(geom) -> float:
    if geom is None:
        return 0.0
    try:
        if geom.is_empty:
            return 0.0
        return float(max(geom.area, 0.0))
    except Exception:
        return 0.0


def _safe_buffer(geom, distance: float):
    if geom is None:
        return None
    try:
        out = geom.buffer(float(distance))
    except Exception:
        try:
            out = geom.buffer(float(distance), join_style=2)
        except Exception:
            return None
    return _fix_geometry(out)


def _safe_difference(a, b):
    if a is None:
        return None
    if b is None:
        return a
    try:
        return _fix_geometry(a.difference(b))
    except Exception:
        try:
            return _fix_geometry(a.buffer(0).difference(b.buffer(0)))
        except Exception:
            return None


def _safe_intersection(a, b):
    if a is None or b is None:
        return None
    try:
        return _fix_geometry(a.intersection(b))
    except Exception:
        try:
            return _fix_geometry(a.buffer(0).intersection(b.buffer(0)))
        except Exception:
            return None


def _safe_union(parts):
    parts = [g for g in parts if g is not None and not getattr(g, "is_empty", True)]
    if not parts:
        return None
    try:
        return _fix_geometry(shapely.union_all(parts))
    except Exception:
        return _fix_geometry(parts[0])


def _load_cell_geometries(bundle: dict) -> dict[str, CellGeometry]:
    """Load cell and nucleus polygons from a toolbox PKL bundle."""
    cell_boundary = bundle.get("cell_boundary", {}) or {}
    nuclear_boundary = bundle.get("nuclear_boundary", {}) or {}

    if not isinstance(cell_boundary, dict) or not isinstance(nuclear_boundary, dict):
        return {}

    cb_str = {str(k): v for k, v in cell_boundary.items()}
    nb_str = {str(k): v for k, v in nuclear_boundary.items()}
    common = set(cb_str).intersection(set(nb_str))

    geoms: dict[str, CellGeometry] = {}
    for cid in common:
        cell = _boundary_to_polygon(cb_str.get(cid))
        nuc = _boundary_to_polygon(nb_str.get(cid))
        if cell is None or nuc is None:
            continue
        nuc_inside = _safe_intersection(nuc, cell)
        if nuc_inside is None:
            nuc_inside = nuc
        geoms[cid] = CellGeometry(cell_id=cid, cell=cell, nuclear=nuc_inside)
    return geoms


def _equivalent_radius(area: float) -> float:
    area = float(area)
    if area <= 0:
        return 0.0
    return math.sqrt(area / math.pi)


def _cell_scaled_thresholds(
    geom: CellGeometry,
    *,
    median_cell_radius: float,
    median_nucleus_radius: float,
) -> tuple[float, float, float]:
    """
    Optional threshold scaling by cell/nucleus size.

    The median-sized cell keeps InSTAnT source defaults. Larger cells receive
    wider bands, smaller cells receive narrower bands. This is optional and is
    disabled by default to preserve the InSTAnT source defaults.
    """
    cell_radius = _equivalent_radius(_safe_area(geom.cell))
    nucleus_radius = _equivalent_radius(_safe_area(geom.nuclear))

    nucleus_scale = nucleus_radius / median_nucleus_radius if median_nucleus_radius > 0 and nucleus_radius > 0 else 1.0
    cell_scale = cell_radius / median_cell_radius if median_cell_radius > 0 and cell_radius > 0 else 1.0

    return (
        INSTANT_DISTANCE_THRESHOLD_NUCLEUS * nucleus_scale,
        INSTANT_DISTANCE_THRESHOLD_CYTO_NUCLEAR * nucleus_scale,
        INSTANT_DISTANCE_THRESHOLD_CYTO_PERI * cell_scale,
    )


def _region_areas_for_cell(
    geom: CellGeometry,
    *,
    t_nucleus: float,
    t_cyto_nuclear: float,
    t_cyto_peri: float,
) -> dict[str, float]:
    """Area background matching InSTAnT-style transcript region definitions."""
    cell = geom.cell
    nuc = _safe_intersection(geom.nuclear, cell) or geom.nuclear
    cytoplasm = _safe_difference(cell, nuc)

    # Inner nuclear: nucleus eroded by nuclear threshold.
    inner_nuclear = _safe_buffer(nuc, -float(t_nucleus))

    # Peri-nuclear: inner nuclear band + outer cytoplasmic band near nucleus.
    peri_inner = _safe_difference(nuc, inner_nuclear)
    nuc_outer = _safe_buffer(nuc, float(t_cyto_nuclear))
    peri_outer = _safe_intersection(cytoplasm, _safe_difference(nuc_outer, nuc))
    peri_nuclear = _safe_union([peri_inner, peri_outer])

    # Peri-membrane: cytoplasmic band near cell boundary.
    cell_core = _safe_buffer(cell, -float(t_cyto_peri))
    cell_boundary_band = _safe_difference(cell, cell_core)
    peri_membrane = _safe_intersection(cytoplasm, cell_boundary_band)

    # Cytosolic: cytoplasm excluding peri-nuclear outer band and peri-membrane band.
    cytosolic = cytoplasm
    cytosolic = _safe_difference(cytosolic, peri_outer)
    cytosolic = _safe_difference(cytosolic, peri_membrane)

    return {
        "inner_nuclear": _safe_area(inner_nuclear),
        "peri_nuclear": _safe_area(peri_nuclear),
        "cytosolic": _safe_area(cytosolic),
        "peri_membrane": _safe_area(peri_membrane),
    }


def _instant_transcript_region_labels(
    xy: np.ndarray,
    geom: CellGeometry,
    *,
    t_nucleus: float = INSTANT_DISTANCE_THRESHOLD_NUCLEUS,
    t_cyto_nuclear: float = INSTANT_DISTANCE_THRESHOLD_CYTO_NUCLEAR,
    t_cyto_peri: float = INSTANT_DISTANCE_THRESHOLD_CYTO_PERI,
) -> np.ndarray:
    """
    Transcript-level InSTAnT region labels.

    Columns:
        0 inner_nuclear
        1 peri_nuclear
        2 cytosolic
        3 peri_membrane
    """
    n = int(len(xy))
    labels = np.zeros((n, 4), dtype=np.float64)
    if n == 0:
        return labels

    in_nucleus = np.zeros(n, dtype=np.int8)
    nuc_distance = np.full(n, np.nan, dtype=np.float64)
    cyt_distance = np.full(n, np.nan, dtype=np.float64)

    for k, (x, y) in enumerate(xy):
        pt = Point(float(x), float(y))
        try:
            in_nucleus[k] = int(geom.nuclear.covers(pt))
        except Exception:
            in_nucleus[k] = 0
        try:
            nuc_distance[k] = float(pt.distance(geom.nuclear.boundary))
        except Exception:
            nuc_distance[k] = np.nan
        try:
            cyt_distance[k] = float(pt.distance(geom.cell.boundary))
        except Exception:
            cyt_distance[k] = np.nan

    valid_nuc = np.isfinite(nuc_distance)
    valid_cyt = np.isfinite(cyt_distance)

    labels[(in_nucleus == 1) & valid_nuc & (nuc_distance > t_nucleus), 0] = 1.0

    labels[(in_nucleus == 1) & valid_nuc & (nuc_distance <= t_nucleus), 1] = 1.0
    labels[(in_nucleus == 0) & valid_nuc & (nuc_distance <= t_cyto_nuclear), 1] = 1.0

    labels[
        (in_nucleus == 0)
        & valid_nuc
        & valid_cyt
        & (nuc_distance > t_cyto_nuclear)
        & (cyt_distance > t_cyto_peri),
        2,
    ] = 1.0

    labels[(in_nucleus == 0) & valid_cyt & (cyt_distance <= t_cyto_peri), 3] = 1.0
    return labels


def _normalize_significant_pairs(pairs: pd.DataFrame) -> pd.DataFrame:
    out = pairs.copy()
    if "cpb_pvalue" in out.columns:
        if "cpb_pvalue_raw" not in out.columns:
            out["cpb_pvalue_raw"] = pd.to_numeric(out["cpb_pvalue"], errors="coerce")
        else:
            out["cpb_pvalue_raw"] = pd.to_numeric(out["cpb_pvalue_raw"], errors="coerce")
        out["cpb_pvalue"] = (
            pd.to_numeric(out["cpb_pvalue"], errors="coerce")
            .fillna(1.0)
            .clip(lower=0.0, upper=1.0)
        )
        out = out.sort_values("cpb_pvalue_raw", ascending=True).reset_index(drop=True)
    return out


def _pair_key(g1: str, g2: str) -> tuple[str, str]:
    return (str(g1), str(g2)) if str(g1) <= str(g2) else (str(g2), str(g1))



def _empty_region_table(significant_pairs: pd.DataFrame | None) -> pd.DataFrame:
    base_cols = list(significant_pairs.columns) if significant_pairs is not None else []
    extra = [
        "n_proximal_pairs",
        "nuclear_enrichment",
        "perinuclear_enrichment",
        "cytosolic_enrichment",
        "peripheral_enrichment",
        "dominant_region",
        "dominant_enrichment_score",
        "second_region",
        "second_enrichment_score",
    ]
    return pd.DataFrame(columns=base_cols + [c for c in extra if c not in base_cols])


def annotate_colocalized_pairs_regions(
    *,
    bundle: dict,
    instant_input_df: pd.DataFrame,
    significant_pairs: pd.DataFrame,
    distance_threshold: float = 4.0,
    use_3d: bool = False,
    max_pairs: Optional[int] = None,
    region_threshold_mode: str = "cell_scaled",
    region_workers: int = 1,
) -> pd.DataFrame:
    """
    Enrichment-only InSTAnT-style regional annotation for significant pairs.

    This function is a post-processing step and does not change PP/CPB
    colocalization results. It annotates only already-significant pairs.

    The reported dominant_region is based on area-normalized enrichment:
        enrichment = observed_endpoint_average_score / expected_score_by_area

    Output deliberately omits raw regional counts/fractions and threshold columns
    to keep the result table focused on the corrected regional interpretation.

    region_threshold_mode:
        cell_scaled : default. Scale InSTAnT source thresholds by each
                      cell/nucleus size; median-sized cell keeps 2.5/2.5/4.0.
        absolute    : use fixed InSTAnT source thresholds 2.5/2.5/4.0.

    region_workers:
        Default 1. Uses deterministic serial execution and is safest for large
        datasets. Values >1 use threads only, not extra processes, to avoid
        zombie/defunct worker processes and excessive memory duplication.
    """
    if significant_pairs is None or significant_pairs.empty:
        return _empty_region_table(significant_pairs)

    if region_threshold_mode not in {"absolute", "cell_scaled"}:
        raise ValueError("region_threshold_mode must be 'absolute' or 'cell_scaled'.")

    required = {"gene", "uID", "absX", "absY"}
    missing = required - set(instant_input_df.columns)
    if missing:
        raise ValueError(f"instant_input_df missing required columns for region annotation: {missing}")

    geoms = _load_cell_geometries(bundle)
    if not geoms:
        print("[region_annotation] warning: no valid cell/nuclear geometries were found; region annotation skipped.")
        return _empty_region_table(significant_pairs)

    cell_radii = [_equivalent_radius(_safe_area(g.cell)) for g in geoms.values() if _safe_area(g.cell) > 0]
    nuc_radii = [_equivalent_radius(_safe_area(g.nuclear)) for g in geoms.values() if _safe_area(g.nuclear) > 0]
    median_cell_radius = float(np.median(cell_radii)) if cell_radii else 0.0
    median_nucleus_radius = float(np.median(nuc_radii)) if nuc_radii else 0.0

    pairs = _normalize_significant_pairs(significant_pairs)
    if max_pairs is not None:
        pairs = pairs.head(int(max_pairs)).copy()

    pair_keys: list[tuple[str, str]] = []
    pair_index_by_key: dict[tuple[str, str], int] = {}
    for idx, row in pairs.iterrows():
        key = _pair_key(str(row["gene_1"]), str(row["gene_2"]))
        pair_keys.append(key)
        pair_index_by_key[key] = int(idx)

    genes_needed = set()
    pairs_by_gene: dict[str, list[tuple[str, str]]] = {}
    for g1, g2 in pair_keys:
        genes_needed.add(g1)
        genes_needed.add(g2)
        pairs_by_gene.setdefault(g1, []).append((g1, g2))
        if g2 != g1:
            pairs_by_gene.setdefault(g2, []).append((g1, g2))

    df = instant_input_df.copy()
    df["uID"] = df["uID"].astype(str)
    df["gene"] = df["gene"].astype(str)
    df = df[df["gene"].isin(genes_needed)].copy()
    for col in ["absX", "absY"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if use_3d and "absZ" in df.columns:
        df["absZ"] = pd.to_numeric(df["absZ"], errors="coerce")

    score_template = {
        int(idx): {
            "obs": np.zeros(4, dtype=np.float64),
            "exp": np.zeros(4, dtype=np.float64),
            "n_proximal_pairs": 0,
        }
        for idx in pairs.index
    }

    coord_cols_dist = ["absX", "absY"]
    if use_3d and "absZ" in df.columns:
        coord_cols_dist = ["absX", "absY", "absZ"]

    cell_items = [(str(cid), cdf.copy()) for cid, cdf in df.groupby("uID", sort=False)]

    def _process_one_cell(item: tuple[str, pd.DataFrame]) -> dict[int, dict[str, object]]:
        cid, cdf = item
        geom = geoms.get(str(cid))
        if geom is None or cdf.empty:
            return {}
        cdf = cdf.dropna(subset=["absX", "absY"]).copy()
        if cdf.empty:
            return {}

        if region_threshold_mode == "cell_scaled":
            t_nucleus, t_cyto_nuclear, t_cyto_peri = _cell_scaled_thresholds(
                geom,
                median_cell_radius=median_cell_radius,
                median_nucleus_radius=median_nucleus_radius,
            )
        else:
            t_nucleus = INSTANT_DISTANCE_THRESHOLD_NUCLEUS
            t_cyto_nuclear = INSTANT_DISTANCE_THRESHOLD_CYTO_NUCLEAR
            t_cyto_peri = INSTANT_DISTANCE_THRESHOLD_CYTO_PERI

        xy_all = cdf[["absX", "absY"]].to_numpy(dtype=np.float64)
        labels_all = _instant_transcript_region_labels(
            xy_all,
            geom,
            t_nucleus=t_nucleus,
            t_cyto_nuclear=t_cyto_nuclear,
            t_cyto_peri=t_cyto_peri,
        )

        areas = _region_areas_for_cell(
            geom,
            t_nucleus=t_nucleus,
            t_cyto_nuclear=t_cyto_nuclear,
            t_cyto_peri=t_cyto_peri,
        )
        area_vec = np.asarray([
            areas["inner_nuclear"],
            areas["peri_nuclear"],
            areas["cytosolic"],
            areas["peri_membrane"],
        ], dtype=np.float64)
        area_total = float(np.nansum(area_vec))
        bg_frac = area_vec / area_total if area_total > 0 else np.zeros(4, dtype=np.float64)

        gene_to_pos = {str(g): np.asarray(pos, dtype=int) for g, pos in cdf.groupby("gene", sort=False).indices.items()}
        present = set(gene_to_pos.keys())
        candidate_keys = set()
        for g in present:
            for key in pairs_by_gene.get(g, []):
                if key[0] in present and key[1] in present:
                    candidate_keys.add(key)
        if not candidate_keys:
            return {}

        coords_all = cdf[coord_cols_dist].to_numpy(dtype=np.float64)
        partial: dict[int, dict[str, object]] = {}

        for g1, g2 in candidate_keys:
            idx1 = gene_to_pos.get(g1)
            idx2 = gene_to_pos.get(g2)
            if idx1 is None or idx2 is None or len(idx1) == 0 or len(idx2) == 0:
                continue

            coords1 = coords_all[idx1]
            coords2 = coords_all[idx2]
            labels1 = labels_all[idx1]
            labels2 = labels_all[idx2]

            if g1 == g2:
                tree = cKDTree(coords1)
                raw_pairs = list(tree.query_pairs(r=distance_threshold))
                if not raw_pairs:
                    continue
                i_idx = np.asarray([p[0] for p in raw_pairs], dtype=int)
                j_idx = np.asarray([p[1] for p in raw_pairs], dtype=int)
            else:
                tree2 = cKDTree(coords2)
                hits = tree2.query_ball_point(coords1, r=distance_threshold)
                if not any(hits):
                    continue
                i_list = []
                j_list = []
                for i, js in enumerate(hits):
                    if js:
                        i_list.extend([i] * len(js))
                        j_list.extend(js)
                if not i_list:
                    continue
                i_idx = np.asarray(i_list, dtype=int)
                j_idx = np.asarray(j_list, dtype=int)

            contrib = (labels1[i_idx] + labels2[j_idx]) * 0.5
            sums = np.nansum(contrib, axis=0)
            obs_total = float(np.nansum(sums))
            if obs_total <= 0:
                continue

            pair_idx = pair_index_by_key[(g1, g2)]
            rec = partial.setdefault(pair_idx, {"obs": np.zeros(4, dtype=np.float64), "exp": np.zeros(4, dtype=np.float64), "n_proximal_pairs": 0})
            rec["obs"] = rec["obs"] + sums
            rec["exp"] = rec["exp"] + obs_total * bg_frac
            rec["n_proximal_pairs"] = int(rec["n_proximal_pairs"]) + int(len(i_idx))

        return partial

    # Deterministic merge order. region_workers=1 avoids extra worker overhead and
    # is the recommended default for large boundary-heavy datasets.
    if int(region_workers or 1) <= 1 or len(cell_items) <= 1:
        partial_results = map(_process_one_cell, cell_items)
    else:
        from concurrent.futures import ThreadPoolExecutor
        max_workers = max(1, int(region_workers))
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            partial_results = list(ex.map(_process_one_cell, cell_items))

    for partial in partial_results:
        for idx, rec in partial.items():
            score_template[idx]["obs"] = score_template[idx]["obs"] + rec["obs"]
            score_template[idx]["exp"] = score_template[idx]["exp"] + rec["exp"]
            score_template[idx]["n_proximal_pairs"] = int(score_template[idx]["n_proximal_pairs"]) + int(rec["n_proximal_pairs"])

    rows = []
    for idx, row in pairs.iterrows():
        sm = score_template[int(idx)]
        obs = np.asarray(sm["obs"], dtype=np.float64)
        exp = np.asarray(sm["exp"], dtype=np.float64)
        enrich = obs / (exp + _EPS)
        region_names = ["nuclear", "perinuclear", "cytosolic", "peripheral"]
        ranked = sorted(zip(region_names, enrich), key=lambda x: x[1], reverse=True)

        out = row.to_dict()
        out.update({
            "n_proximal_pairs": int(sm["n_proximal_pairs"]),
            "nuclear_enrichment": float(enrich[0]),
            "perinuclear_enrichment": float(enrich[1]),
            "cytosolic_enrichment": float(enrich[2]),
            "peripheral_enrichment": float(enrich[3]),
            "dominant_region": ranked[0][0] if int(sm["n_proximal_pairs"]) > 0 else "none",
            "dominant_enrichment_score": float(ranked[0][1]) if int(sm["n_proximal_pairs"]) > 0 else 0.0,
            "second_region": ranked[1][0] if int(sm["n_proximal_pairs"]) > 0 else "none",
            "second_enrichment_score": float(ranked[1][1]) if int(sm["n_proximal_pairs"]) > 0 else 0.0,
        })
        rows.append(out)

    return pd.DataFrame(rows)
