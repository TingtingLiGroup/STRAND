# tools/engines/instant_adapter.py

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from dataclasses import dataclass
import shapely
from shapely.geometry import Polygon, MultiPolygon, GeometryCollection, Point
from shapely.geometry.base import BaseGeometry
from shapely.validation import make_valid


def _import_instant_classes():
    """
    Import bundled InSTAnT core from this toolbox.

    InSTAnT source files should be placed in:
        tools/instant_core/
    """
    try:
        from tools.instant_core.InSTAnT import (
            Instant,
            ConditionalGlobalColocalization,
        )

        return Instant, ConditionalGlobalColocalization

    except Exception as e:
        raise ImportError(
            "Cannot import bundled InSTAnT core.\n"
            "Please check that the following files exist:\n"
            "  tools/instant_core/InSTAnT.py\n"
            "  tools/instant_core/poibin.py\n"
            "  tools/instant_core/poisson_binomial.py\n"
            "  tools/instant_core/gspan_mining/\n"
        ) from e

def _fix_geometry(geom: BaseGeometry | None) -> BaseGeometry | None:
    """Repair invalid geometry and keep polygonal components only."""
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
    """Convert boundary object to shapely polygon."""
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


def _add_instant_region_input_columns(
    instant_input_df: pd.DataFrame,
    bundle: dict,
) -> pd.DataFrame:
    """
    Add columns required by InSTAnT.annotate_ProximalPairs():

        inNucleus
        distNucleus
        distPeriphery

    InSTAnT's annotate_ProximalPairs() itself does not compute these columns.
    It only consumes them in _spatial_category().

    Here they are computed from the toolbox PKL boundaries:
        inNucleus     = transcript point covered by nuclear polygon
        distNucleus   = distance from transcript point to nuclear boundary
        distPeriphery = distance from transcript point to cell boundary
    """
    out = instant_input_df.copy()

    cell_boundary = {
        str(k): _boundary_to_polygon(v)
        for k, v in bundle.get("cell_boundary", {}).items()
    }
    nuclear_boundary = {
        str(k): _boundary_to_polygon(v)
        for k, v in bundle.get("nuclear_boundary", {}).items()
    }

    in_nucleus = np.zeros(len(out), dtype=np.int8)
    dist_nucleus = np.full(len(out), np.nan, dtype=float)
    dist_periphery = np.full(len(out), np.nan, dtype=float)

    for cid, idx in out.groupby("uID", sort=False).groups.items():
        cid = str(cid)
        idx = np.asarray(list(idx), dtype=int)

        cell_poly = cell_boundary.get(cid)
        nuc_poly = nuclear_boundary.get(cid)

        if cell_poly is None or nuc_poly is None:
            continue

        # Keep nuclear geometry inside cell, matching region logic.
        try:
            nuc_poly = _fix_geometry(nuc_poly.intersection(cell_poly))
        except Exception:
            nuc_poly = _fix_geometry(nuc_poly)

        if nuc_poly is None:
            continue

        xy = out.loc[idx, ["absX", "absY"]].to_numpy(float)

        for local_k, row_i in enumerate(idx):
            x, y = xy[local_k]
            pt = Point(float(x), float(y))

            try:
                in_nucleus[row_i] = int(nuc_poly.covers(pt))
            except Exception:
                in_nucleus[row_i] = 0

            try:
                dist_nucleus[row_i] = float(pt.distance(nuc_poly.boundary))
            except Exception:
                dist_nucleus[row_i] = np.nan

            try:
                dist_periphery[row_i] = float(pt.distance(cell_poly.boundary))
            except Exception:
                dist_periphery[row_i] = np.nan

    out["inNucleus"] = in_nucleus
    out["distNucleus"] = dist_nucleus
    out["distPeriphery"] = dist_periphery

    return out


def _summarize_instant_builtin_region_annotation(
    *,
    significant_pairs: pd.DataFrame,
    genes: list[str],
    inner_nuc: np.ndarray,
    peri_nuc: np.ndarray,
    cytosolic: np.ndarray,
    perimem: np.ndarray,
    max_pairs: int | None = None,
) -> pd.DataFrame:
    """
    Convert InSTAnT.annotate_ProximalPairs() cell-level matrices into
    global pair-level annotation.

    No cell-level file is written.
    """
    if significant_pairs is None or significant_pairs.empty:
        return pd.DataFrame(columns=list(significant_pairs.columns) if significant_pairs is not None else [])

    pairs = significant_pairs.copy()

    if max_pairs is not None:
        pairs = pairs.head(int(max_pairs)).copy()

    gene_to_idx = {str(g): i for i, g in enumerate(genes)}

    rows = []

    for _, row in pairs.iterrows():
        g1 = str(row["gene_1"])
        g2 = str(row["gene_2"])

        if g1 not in gene_to_idx or g2 not in gene_to_idx:
            continue

        i = gene_to_idx[g1]
        j = gene_to_idx[g2]

        inner_score = float(np.nansum(inner_nuc[:, i, j]))
        peri_score = float(np.nansum(peri_nuc[:, i, j]))
        cyto_score = float(np.nansum(cytosolic[:, i, j]))
        mem_score = float(np.nansum(perimem[:, i, j]))

        total_score = inner_score + peri_score + cyto_score + mem_score

        if total_score > 0:
            nuclear_fraction = inner_score / total_score
            perinuclear_fraction = peri_score / total_score
            cytosolic_fraction = cyto_score / total_score
            peripheral_fraction = mem_score / total_score
        else:
            nuclear_fraction = 0.0
            perinuclear_fraction = 0.0
            cytosolic_fraction = 0.0
            peripheral_fraction = 0.0

        region_scores = {
            "nuclear": inner_score,
            "perinuclear": peri_score,
            "cytosolic": cyto_score,
            "peripheral": mem_score,
        }

        ranked_regions = sorted(region_scores.items(), key=lambda x: x[1], reverse=True)

        dominant_region = ranked_regions[0][0] if total_score > 0 else "none"
        dominant_fraction = (
            ranked_regions[0][1] / total_score if total_score > 0 else 0.0
        )

        second_region = ranked_regions[1][0] if total_score > 0 else "none"
        second_fraction = (
            ranked_regions[1][1] / total_score if total_score > 0 else 0.0
        )

        out = row.to_dict()
        out.update(
            {
                "inner_nuclear_score": inner_score,
                "peri_nuclear_score": peri_score,
                "cytosolic_score": cyto_score,
                "peri_membrane_score": mem_score,
                "nuclear_fraction": float(nuclear_fraction),
                "perinuclear_fraction": float(perinuclear_fraction),
                "cytosolic_fraction": float(cytosolic_fraction),
                "peripheral_fraction": float(peripheral_fraction),
                "dominant_region": dominant_region,
                "dominant_fraction": float(dominant_fraction),
                "second_region": second_region,
                "second_fraction": float(second_fraction),
            }
        )

        rows.append(out)

    return pd.DataFrame(rows)

def bundle_to_instant_dataframe(
    bundle: dict,
    *,
    cell_id_col: str = "cell",
    gene_col: str = "gene",
    x_col: str = "x",
    y_col: str = "y",
    z_col: str = "z",
    use_3d: bool = True,
) -> pd.DataFrame:
    """
    Convert STRAND Tools standard PKL bundle to InSTAnT input table.

    Your toolbox input format:
        bundle["data_df"] columns:
            cell, gene, x, y, z ...

    InSTAnT required format:
        gene, absX, absY, uID

    If use_3d=True:
        gene, absX, absY, absZ, uID

    Mapping:
        cell -> uID
        gene -> gene
        x    -> absX
        y    -> absY
        z    -> absZ
    """
    if "data_df" not in bundle:
        raise KeyError("bundle must contain key 'data_df'.")

    df = bundle["data_df"].copy()

    required_cols = [cell_id_col, gene_col, x_col, y_col]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"data_df is missing required columns for InSTAnT conversion: {missing}"
        )

    if use_3d and z_col not in df.columns:
        raise ValueError(
            f"use_3d=True, but z column '{z_col}' was not found in data_df."
        )

    out = pd.DataFrame(
        {
            "gene": df[gene_col].astype(str).values,
            "absX": df[x_col].astype(float).values,
            "absY": df[y_col].astype(float).values,
            "uID": df[cell_id_col].astype(str).values,
        }
    )

    if use_3d:
        out["absZ"] = df[z_col].astype(float).values

    return out


def _build_significant_pairs_table(
    *,
    cpb_pvals: np.ndarray,
    expected_coloc: np.ndarray,
    genes: list[str],
    cpb_alpha: float,
    distance_threshold: float,
    pp_alpha: float,
) -> pd.DataFrame:
    """
    Convert CPB p-value matrix to a long significant gene-pair table.
    """
    rows = []
    n = len(genes)

    for i in range(n):
        for j in range(i + 1, n):
            pval = float(cpb_pvals[i, j])

            if pval < cpb_alpha:
                rows.append(
                    {
                        "gene_1": genes[i],
                        "gene_2": genes[j],
                        "cpb_pvalue": pval,
                        "expected_coloc": float(expected_coloc[i, j]),
                        "distance_threshold": float(distance_threshold),
                        "pp_alpha": float(pp_alpha),
                        "cpb_alpha": float(cpb_alpha),
                    }
                )

    columns = [
        "gene_1",
        "gene_2",
        "cpb_pvalue",
        "expected_coloc",
        "distance_threshold",
        "pp_alpha",
        "cpb_alpha",
    ]

    if not rows:
        return pd.DataFrame(columns=columns)

    return (
        pd.DataFrame(rows, columns=columns)
        .sort_values("cpb_pvalue", ascending=True)
        .reset_index(drop=True)
    )

def _add_pair_cell_stats(
    pairs: pd.DataFrame,
    *,
    pp_pvals: np.ndarray,
    genes: list[str],
    pp_alpha: float,
) -> pd.DataFrame:
    """
    Add only one interpretation column: n_cells_pp_significant.

    This does not change InSTAnT PP/CPB results. It only counts, for each
    CPB-significant gene pair, how many cells have cell-wise PP p-value < pp_alpha.
    No extra dense matrix is created; the function only slices the already existing
    InSTAnT all_pval array and writes one integer per significant pair.
    """
    out = pairs.copy()

    if out.empty:
        out["n_cells_pp_significant"] = []
        return out

    gene_to_idx = {str(g): i for i, g in enumerate(genes)}
    pp = np.asarray(pp_pvals)

    values = []
    for _, row in out.iterrows():
        g1 = str(row["gene_1"])
        g2 = str(row["gene_2"])

        if g1 not in gene_to_idx or g2 not in gene_to_idx:
            values.append(0)
            continue

        i = gene_to_idx[g1]
        j = gene_to_idx[g2]
        pvals = pp[:, i, j]
        n_sig = int(np.sum(np.isfinite(pvals) & (pvals < pp_alpha)))
        values.append(n_sig)

    out["n_cells_pp_significant"] = values
    return out

def run_instant_colocalization_from_bundle(
    bundle: dict,
    *,
    distance_threshold: float = 4.0,
    pp_alpha: float = 0.001,
    cpb_alpha: float = 0.0001,
    min_genecount: int = 20,
    threads: int = 1,
    use_3d: bool = True,
    precision_mode: str = "high",
    tmp_dir: str | Path | None = None,
    keep_tmp: bool = False,
    cell_id_col: str = "cell",
    gene_col: str = "gene",
    x_col: str = "x",
    y_col: str = "y",
    z_col: str = "z",
    region_annotation: bool = False,
    perinuclear_width: float | None = None,
    peripheral_width: float | None = None,
    region_max_pairs: int | None = None,
) -> dict:
    """
    Run InSTAnT PP test + CPB test on a prefiltered toolbox bundle.

    Important:
    This function assumes that bundle has already passed the toolbox standard
    prefiltering step:
        1. nc ratio filtering
        2. cell-gene transcript filtering + gene support filtering
        3. data_df / cell_boundary / nuclear_boundary synchronization

    Returns
    -------
    dict with:
        gene_list
        instant_input_df
        pp_pvals
        gene_counts
        cpb_pvals
        expected_coloc
        significant_pairs
    """
    Instant, ConditionalGlobalColocalization = _import_instant_classes()

    if threads is None or threads < 1:
        threads = 1

    if precision_mode not in {"high", "low"}:
        raise ValueError("precision_mode must be 'high' or 'low'.")

    tmp_dir = Path(tmp_dir or "_tmp_instant_coloc")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    instant_input_df = bundle_to_instant_dataframe(
        bundle,
        cell_id_col=cell_id_col,
        gene_col=gene_col,
        x_col=x_col,
        y_col=y_col,
        z_col=z_col,
        use_3d=use_3d,
    )
    if region_annotation:
        instant_input_df = _add_instant_region_input_columns(
            instant_input_df=instant_input_df,
            bundle=bundle,
        )

    input_csv = tmp_dir / "instant_input.csv"
    instant_input_df.to_csv(input_csv, index=False)

    # 注意：
    # InSTAnT.py 里 Instant.__init__ 对 precision_mode 的判断是 bool 逻辑。
    # 如果直接传 "low"，字符串仍然为 True，所以这里手动转成 bool。
    instant_precision_flag = True if precision_mode == "high" else False

    instant_model = Instant(
        distance_threshold=distance_threshold,
        threads=threads,
        precision_mode=instant_precision_flag,
    )

    instant_model.load_preprocessed_data(
        str(input_csv),
        force_csv=True,
    )

    if use_3d:
        instant_model.run_ProximalPairs3D(
            distance_threshold=distance_threshold,
            min_genecount=min_genecount,
        )
    else:
        instant_model.run_ProximalPairs(
            distance_threshold=distance_threshold,
            min_genecount=min_genecount,
        )

    cpb_model = ConditionalGlobalColocalization(
        all_pvals=instant_model.all_pval,
        transcript_count=instant_model.all_gene_count,
        alpha_cellwise=pp_alpha,
        min_transcript=0,
        threads=threads,
        precision_mode=precision_mode,
    )

    if threads > 1:
        cpb_pvals, expected_coloc = cpb_model.global_colocalization()
    else:
        cpb_pvals, expected_coloc = cpb_model.global_colocalization_serial()

    genes = [str(g) for g in list(instant_model.geneList)]

    significant_pairs = _build_significant_pairs_table(
        cpb_pvals=cpb_pvals,
        expected_coloc=expected_coloc,
        genes=genes,
        cpb_alpha=cpb_alpha,
        distance_threshold=distance_threshold,
        pp_alpha=pp_alpha,
    )
    significant_pairs = _add_pair_cell_stats(
        significant_pairs,
        pp_pvals=instant_model.all_pval,
        genes=genes,
        pp_alpha=pp_alpha,
    )

    # Region annotation is intentionally NOT computed here.
    # The API layer saves instant_significant_pairs.csv first, then runs the
    # streaming InSTAnT-style annotator only for significant pairs. This avoids
    # InSTAnT.annotate_ProximalPairs() generating large cell x gene x gene
    # matrices and re-annotating all possible gene pairs.
    region_annotated_pairs = None

    result = {
        "gene_list": genes,
        "instant_input_df": instant_input_df,
        "pp_pvals": instant_model.all_pval,
        "gene_counts": instant_model.all_gene_count,
        "cpb_pvals": cpb_pvals,
        "expected_coloc": expected_coloc,
        "significant_pairs": significant_pairs,
        "region_annotated_pairs": region_annotated_pairs,
    }

    if not keep_tmp:
        try:
            shutil.rmtree(tmp_dir)
        except Exception:
            pass

    return result