from __future__ import annotations

from pathlib import Path
import os
import pickle
import tempfile

import numpy as np
import pandas as pd
from shapely.geometry import Polygon

from tools.engines.bento_adapter import compute_bento13_from_dict
from tools.engines.sprawl_adapter import compute_sprawl_scores_from_pkl
from tools.models.pattern_classifier import PatternClassifier
from tools.api.patterns import _get_foci_ratio
from tools.utils.timing import timed, TimerReport


def _normalize_index(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure MultiIndex levels are strings for reliable joins."""
    if not isinstance(df.index, pd.MultiIndex) or df.index.nlevels != 2:
        return df

    df.index = df.index.set_levels(
        [df.index.levels[0].astype(str), df.index.levels[1].astype(str)]
    )
    return df


def _save_df(df: pd.DataFrame, out_path: str) -> str:
    """Write output to parquet/csv; return actual path."""
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    if out_path.endswith(".parquet"):
        df.reset_index().to_parquet(out_path, index=False)
        return out_path

    if out_path.endswith(".csv"):
        df.reset_index().to_csv(out_path, index=False)
        return out_path

    out_path = out_path + ".parquet"
    df.reset_index().to_parquet(out_path, index=False)
    return out_path


def _polygon_area(coords) -> float:
    """Robust polygon area from boundary coordinates."""
    try:
        poly = Polygon(coords)
        if not poly.is_valid:
            poly = poly.buffer(0)
        return float(poly.area) if poly.is_valid else np.nan
    except Exception:
        return np.nan


def _bundle_basic_stats(bundle: dict) -> dict:
    """
    Return basic data size statistics for the current bundle.

    Statistics:
      - n_cells: number of cells in data_df
      - n_genes: number of genes in data_df
      - n_rows: number of transcript rows
      - n_cellgenes: number of unique (cell, gene) pairs
      - n_cell_boundary: number of cell boundaries
      - n_nuclear_boundary: number of nuclear boundaries
    """
    df = bundle["data_df"].copy()
    df["cell"] = df["cell"].astype(str)
    df["gene"] = df["gene"].astype(str)

    return {
        "n_cells": int(df["cell"].nunique()),
        "n_genes": int(df["gene"].nunique()),
        "n_rows": int(len(df)),
        "n_cellgenes": int(df[["cell", "gene"]].drop_duplicates().shape[0]),
        "n_cell_boundary": int(len(bundle.get("cell_boundary", {}))),
        "n_nuclear_boundary": int(len(bundle.get("nuclear_boundary", {}))),
    }

def _compute_nc_ratio_df(bundle: dict) -> pd.DataFrame:
    """
    Compute per-cell nc ratio:
        nc_ratio = nuclear_area / cell_area

    Compatibility:
      - If nuclear_boundary is missing or empty, return an empty DataFrame.
        This means nc_ratio filtering will be skipped.
      - If only part of cells have nuclear boundaries, compute nc_ratio only
        for cells with both cell_boundary and nuclear_boundary.
    """
    cell_boundary = bundle.get("cell_boundary", {})
    nuclear_boundary = bundle.get("nuclear_boundary", None)

    # 有些数据集没有核边界，直接跳过 nc_ratio 过滤
    if nuclear_boundary is None or len(nuclear_boundary) == 0:
        return pd.DataFrame(
            columns=["cell", "cell_area", "nuclear_area", "nc_ratio"]
        )

    if cell_boundary is None or len(cell_boundary) == 0:
        return pd.DataFrame(
            columns=["cell", "cell_area", "nuclear_area", "nc_ratio"]
        )

    cell_boundary_str = {str(k): v for k, v in cell_boundary.items()}
    nuclear_boundary_str = {str(k): v for k, v in nuclear_boundary.items()}

    common_cells = set(cell_boundary_str.keys()) & set(nuclear_boundary_str.keys())

    rows = []

    for cell_id in common_cells:
        cell_coords = cell_boundary_str[cell_id]
        nuc_coords = nuclear_boundary_str[cell_id]

        cell_area = _polygon_area(cell_coords)
        nuclear_area = _polygon_area(nuc_coords)

        if pd.isna(cell_area) or pd.isna(nuclear_area) or cell_area <= 0:
            nc_ratio = np.nan
        else:
            nc_ratio = nuclear_area / cell_area

        rows.append(
            {
                "cell": str(cell_id),
                "cell_area": cell_area,
                "nuclear_area": nuclear_area,
                "nc_ratio": nc_ratio,
            }
        )

    return pd.DataFrame(rows)

def _synchronize_cells_across_bundle(bundle: dict) -> tuple[dict, set[str]]:
    """
    Synchronize cells across:
      - data_df
      - cell_boundary
      - nuclear_boundary

    Keep only the intersection of cell IDs across the available objects.

    Compatibility:
      - If nuclear_boundary exists and is non-empty:
          synchronize data_df / cell_boundary / nuclear_boundary.
      - If nuclear_boundary is missing or empty:
          synchronize only data_df / cell_boundary.
        This avoids deleting all cells for datasets without nuclear boundary.
    """
    bundle_new = dict(bundle)

    df = bundle_new["data_df"].copy()
    df["cell"] = df["cell"].astype(str)

    cells_df = set(df["cell"].unique())

    cell_boundary = bundle_new.get("cell_boundary", {})
    nuclear_boundary = bundle_new.get("nuclear_boundary", None)

    cells_cell_boundary = set(map(str, cell_boundary.keys())) if cell_boundary else set()

    has_cell_boundary = cell_boundary is not None and len(cell_boundary) > 0
    has_nuclear_boundary = nuclear_boundary is not None and len(nuclear_boundary) > 0

    if has_cell_boundary and has_nuclear_boundary:
        cells_nuclear_boundary = set(map(str, nuclear_boundary.keys()))
        common_cells = cells_df & cells_cell_boundary & cells_nuclear_boundary

    elif has_cell_boundary and not has_nuclear_boundary:
        common_cells = cells_df & cells_cell_boundary

    elif (not has_cell_boundary) and has_nuclear_boundary:
        cells_nuclear_boundary = set(map(str, nuclear_boundary.keys()))
        common_cells = cells_df & cells_nuclear_boundary

    else:
        common_cells = cells_df

    bundle_new["data_df"] = df[df["cell"].isin(common_cells)].copy()

    if has_cell_boundary:
        bundle_new["cell_boundary"] = {
            str(k): v
            for k, v in cell_boundary.items()
            if str(k) in common_cells
        }

    if has_nuclear_boundary:
        bundle_new["nuclear_boundary"] = {
            str(k): v
            for k, v in nuclear_boundary.items()
            if str(k) in common_cells
        }

    return bundle_new, common_cells

def _filter_cells_by_nc_ratio(
    bundle: dict,
    q_low: float = 0.025,
    q_high: float = 0.975,
    mean_low: float = 0.4,
    mean_high: float = 0.6,
    enforce_mean_range: bool = False,
    min_cells_remaining: int = 10,
) -> tuple[dict, pd.DataFrame]:
    """
    Filter cells by nc_ratio.

    Strategy:
      1. Compute nc_ratio = nuclear_area / cell_area.
      2. If nuclear_boundary is missing, skip this filter.
      3. Remove invalid ratios.
      4. Remove only extreme nc_ratio cells using quantiles.
      5. Do NOT force the retained-cell mean into [mean_low, mean_high].

    Important:
      mean_low / mean_high are now used only for reporting.
      They are not used as hard filtering targets.

    This avoids the problem where a dataset naturally has nc_ratio mean
    outside 0.4-0.6 and the filter deletes almost all cells.
    """
    df_nc_all = _compute_nc_ratio_df(bundle).copy()

    # 没有核边界，或者无法计算 nc_ratio：直接跳过过滤
    if df_nc_all.empty:
        df_empty = pd.DataFrame(
            columns=["cell", "cell_area", "nuclear_area", "nc_ratio"]
        )
        df_empty.attrs["skip_reason"] = "missing_or_empty_nuclear_boundary"
        df_empty.attrs["mean_status"] = "SKIPPED_NO_NUCLEAR_BOUNDARY"
        return bundle, df_empty

    df_nc = df_nc_all.dropna(subset=["nc_ratio"]).copy()
    df_nc["cell"] = df_nc["cell"].astype(str)

    # 只保留有限值和物理上合理的比例
    df_nc = df_nc[np.isfinite(df_nc["nc_ratio"])].copy()
    df_nc = df_nc[(df_nc["nc_ratio"] > 0) & (df_nc["nc_ratio"] < 1)].copy()

    if df_nc.empty:
        df_nc.attrs["skip_reason"] = "no_valid_nc_ratio"
        df_nc.attrs["mean_status"] = "SKIPPED_NO_VALID_RATIO"
        return bundle, df_nc

    n_before = int(df_nc["cell"].nunique())
    mean_before = float(df_nc["nc_ratio"].mean())
    median_before = float(df_nc["nc_ratio"].median())

    # 分位数极端值过滤
    low_value = float(df_nc["nc_ratio"].quantile(q_low))
    high_value = float(df_nc["nc_ratio"].quantile(q_high))

    df_keep = df_nc[
        (df_nc["nc_ratio"] >= low_value)
        & (df_nc["nc_ratio"] <= high_value)
    ].copy()

    # 保护：如果分位数过滤后细胞过少，则不执行 nc_ratio 过滤
    if df_keep["cell"].nunique() < min_cells_remaining:
        df_nc.attrs["skip_reason"] = "too_few_cells_after_quantile_filter"
        df_nc.attrs["mean_status"] = "SKIPPED_TOO_FEW_CELLS"
        df_nc.attrs["n_cells_before"] = n_before
        df_nc.attrs["n_cells_after"] = n_before
        df_nc.attrs["mean_before"] = mean_before
        df_nc.attrs["mean_after"] = mean_before
        df_nc.attrs["median_before"] = median_before
        df_nc.attrs["median_after"] = median_before
        df_nc.attrs["q_low"] = float(q_low)
        df_nc.attrs["q_high"] = float(q_high)
        df_nc.attrs["q_low_value"] = low_value
        df_nc.attrs["q_high_value"] = high_value
        df_nc.attrs["mean_low"] = float(mean_low)
        df_nc.attrs["mean_high"] = float(mean_high)
        return bundle, df_nc

    n_after = int(df_keep["cell"].nunique())
    mean_after = float(df_keep["nc_ratio"].mean())
    median_after = float(df_keep["nc_ratio"].median())

    if mean_low <= mean_after <= mean_high:
        mean_status = "PASS"
    else:
        mean_status = "WARN_NOT_FORCED"

    df_keep.attrs["filter_mode"] = "quantile_extreme_only"
    df_keep.attrs["n_cells_before"] = n_before
    df_keep.attrs["n_cells_after"] = n_after
    df_keep.attrs["n_cells_removed"] = n_before - n_after
    df_keep.attrs["mean_before"] = mean_before
    df_keep.attrs["mean_after"] = mean_after
    df_keep.attrs["median_before"] = median_before
    df_keep.attrs["median_after"] = median_after
    df_keep.attrs["q_low"] = float(q_low)
    df_keep.attrs["q_high"] = float(q_high)
    df_keep.attrs["q_low_value"] = low_value
    df_keep.attrs["q_high_value"] = high_value
    df_keep.attrs["mean_low"] = float(mean_low)
    df_keep.attrs["mean_high"] = float(mean_high)
    df_keep.attrs["mean_status"] = mean_status

    keep_cells = set(df_keep["cell"].astype(str))

    bundle_new = dict(bundle)

    df = bundle_new["data_df"].copy()
    df["cell"] = df["cell"].astype(str)
    bundle_new["data_df"] = df[df["cell"].isin(keep_cells)].copy()

    if "cell_boundary" in bundle_new and bundle_new["cell_boundary"] is not None:
        bundle_new["cell_boundary"] = {
            str(k): v
            for k, v in bundle_new["cell_boundary"].items()
            if str(k) in keep_cells
        }

    if "nuclear_boundary" in bundle_new and bundle_new["nuclear_boundary"] is not None:
        bundle_new["nuclear_boundary"] = {
            str(k): v
            for k, v in bundle_new["nuclear_boundary"].items()
            if str(k) in keep_cells
        }

    return bundle_new, df_keep


def _filter_cellgenes_by_min_transcripts(
    bundle: dict,
    min_transcripts: int = 6,
) -> tuple[dict, pd.DataFrame]:
    """
    Filter low-count cell-gene samples.

    For each (cell, gene), count transcript rows.
    Keep only cell-gene pairs with transcript count >= min_transcripts.

    This directly removes unreliable low-transcript cell-gene samples before
    Bento/SPRAWL feature computation and downstream pattern classification.

    Example:
        cell1 - geneA: 20 transcripts -> keep
        cell2 - geneA: 3 transcripts  -> remove this cell-gene only
        cell3 - geneA: 8 transcripts  -> keep
    """
    bundle_new = dict(bundle)
    df = bundle_new["data_df"].copy()

    df["cell"] = df["cell"].astype(str)
    df["gene"] = df["gene"].astype(str)

    cellgene_counts = (
        df.groupby(["cell", "gene"])
        .size()
        .rename("n_transcripts")
        .reset_index()
    )

    keep_cellgenes = cellgene_counts[
        cellgene_counts["n_transcripts"] >= min_transcripts
    ][["cell", "gene"]].copy()

    df_filtered = df.merge(
        keep_cellgenes,
        on=["cell", "gene"],
        how="inner",
    )

    bundle_new["data_df"] = df_filtered

    return bundle_new, cellgene_counts


def _filter_genes_by_min_cells_after_cellgene_filter(
    bundle: dict,
    min_cells: int = 10,
) -> tuple[dict, pd.DataFrame]:
    """
    After cell-gene transcript filtering, keep only genes that remain
    in at least `min_cells` cells.

    This assumes low-transcript cell-gene samples have already been removed.

    Rule:
        keep gene g if:
            number of unique cells containing g after cell-gene filtering >= min_cells
    """
    bundle_new = dict(bundle)
    df = bundle_new["data_df"].copy()

    df["cell"] = df["cell"].astype(str)
    df["gene"] = df["gene"].astype(str)

    gene_support = (
        df.groupby("gene")["cell"]
        .nunique()
        .rename("n_cells_after_cellgene_filter")
        .reset_index()
    )

    keep_genes = set(
        gene_support.loc[
            gene_support["n_cells_after_cellgene_filter"] >= min_cells,
            "gene",
        ].astype(str)
    )

    df_filtered = df[df["gene"].isin(keep_genes)].copy()
    bundle_new["data_df"] = df_filtered

    return bundle_new, gene_support


def _limit_cells_in_bundle(bundle: dict, max_cells: int) -> tuple[dict, list[str]]:
    """
    Keep only the first `max_cells` cells in data_df, and synchronize
    data_df / cell_boundary / nuclear_boundary accordingly.
    """
    bundle_new = dict(bundle)
    df = bundle_new["data_df"].copy()
    df["cell"] = df["cell"].astype(str)

    cells = list(df["cell"].unique()[:max_cells])
    keep_cells = set(cells)

    bundle_new["data_df"] = df[df["cell"].isin(keep_cells)].copy()

    if "cell_boundary" in bundle_new:
        bundle_new["cell_boundary"] = {
            str(k): v
            for k, v in bundle_new["cell_boundary"].items()
            if str(k) in keep_cells
        }

    if "nuclear_boundary" in bundle_new:
        bundle_new["nuclear_boundary"] = {
            str(k): v
            for k, v in bundle_new["nuclear_boundary"].items()
            if str(k) in keep_cells
        }

    return bundle_new, cells


def _prefilter_bundle_before_features(
    bundle: dict,
    *,
    filter_cells_by_nc_ratio: bool = False,
    nc_ratio_q_low: float = 0.025,
    nc_ratio_q_high: float = 0.975,
    nc_ratio_mean_low: float = 0.4,
    nc_ratio_mean_high: float = 0.6,
    cellgene_filter_min_transcripts: int | None = None,
    gene_filter_min_cells: int | None = None,
) -> tuple[dict, dict]:
    """
    Unified prefilter before feature computation.

    Order:
      1. Optional nc-ratio based cell filtering
         - removes quantile outliers (q_low / q_high)
         - reports whether mean nc_ratio falls in [nc_ratio_mean_low, nc_ratio_mean_high]
           but does NOT enforce the mean range (enforce_mean_range=False by default).
      2. Optional cell-gene minimum transcript filtering.
         - each (cell, gene) must have at least min_transcripts transcript rows.
      3. Optional gene support filtering after cell-gene filtering.
         - each gene must remain in at least min_cells cells.
      4. Synchronize cells across data_df / cell_boundary / nuclear_boundary.

    The final synchronized bundle is the one passed to Bento and SPRAWL.
    """
    stats: dict = {}
    bundle_new = dict(bundle)

    stats["raw"] = _bundle_basic_stats(bundle_new)

    if filter_cells_by_nc_ratio:
        bundle_new, df_nc = _filter_cells_by_nc_ratio(
            bundle_new,
            q_low=nc_ratio_q_low,
            q_high=nc_ratio_q_high,
            mean_low=nc_ratio_mean_low,
            mean_high=nc_ratio_mean_high,
            enforce_mean_range=False,
        )

        stats["nc_ratio_df"] = df_nc
        stats["n_cells_after_nc_ratio_filter"] = (
            int(df_nc["cell"].nunique()) if not df_nc.empty else stats["raw"]["n_cells"]
        )

        if not df_nc.empty and "nc_ratio" in df_nc.columns:
            stats["nc_ratio_mean_after_filter"] = float(df_nc["nc_ratio"].mean())
        else:
            stats["nc_ratio_mean_after_filter"] = np.nan

        stats["nc_ratio_mean_status"] = df_nc.attrs.get(
            "mean_status",
            "SKIPPED",
        )
        stats["after_nc_ratio_filter"] = _bundle_basic_stats(bundle_new)

    if cellgene_filter_min_transcripts is not None:
        bundle_new, cellgene_counts = _filter_cellgenes_by_min_transcripts(
            bundle_new,
            min_transcripts=cellgene_filter_min_transcripts,
        )

        n_before = int(len(cellgene_counts))
        n_after = int(
            (cellgene_counts["n_transcripts"] >= cellgene_filter_min_transcripts).sum()
        )

        stats["cellgene_counts_df"] = cellgene_counts
        stats["cellgene_filter_min_transcripts"] = int(cellgene_filter_min_transcripts)
        stats["n_cellgenes_before_cellgene_filter"] = n_before
        stats["n_cellgenes_after_cellgene_filter"] = n_after
        stats["n_cellgenes_removed_by_cellgene_filter"] = n_before - n_after
        stats["after_cellgene_filter"] = _bundle_basic_stats(bundle_new)

    if gene_filter_min_cells is not None:
        bundle_new, gene_support = _filter_genes_by_min_cells_after_cellgene_filter(
            bundle_new,
            min_cells=gene_filter_min_cells,
        )

        n_genes_before = int(len(gene_support))
        n_genes_after = int(bundle_new["data_df"]["gene"].astype(str).nunique())

        stats["gene_support_df"] = gene_support
        stats["gene_filter_min_cells"] = int(gene_filter_min_cells)
        stats["n_genes_before_gene_support_filter"] = n_genes_before
        stats["n_genes_after_gene_support_filter"] = n_genes_after
        stats["n_genes_removed_by_gene_support_filter"] = n_genes_before - n_genes_after
        stats["after_gene_support_filter"] = _bundle_basic_stats(bundle_new)

    bundle_new, common_cells = _synchronize_cells_across_bundle(bundle_new)
    stats["common_cells"] = common_cells
    stats["n_common_cells"] = len(common_cells)
    stats["final"] = _bundle_basic_stats(bundle_new)

    return bundle_new, stats


def _print_prefilter_stats(filter_stats: dict) -> None:
    """
    Pretty-print prefilter statistics for --profile mode.
    """
    print("[compute_all] prefilter stats keys:", list(filter_stats.keys()))

    raw = filter_stats.get("raw", {})
    if raw:
        print("[prefilter][raw] n_cells_in_data_df =", raw.get("n_cells"))
        print("[prefilter][raw] n_genes_in_data_df =", raw.get("n_genes"))
        print("[prefilter][raw] n_cellgenes_in_data_df =", raw.get("n_cellgenes"))
        print("[prefilter][raw] n_rows_in_data_df =", raw.get("n_rows"))
        print("[prefilter][raw] n_cell_boundary =", raw.get("n_cell_boundary"))
        print("[prefilter][raw] n_nuclear_boundary =", raw.get("n_nuclear_boundary"))

    df_nc = filter_stats.get("nc_ratio_df", None)
    if df_nc is not None:
        if df_nc.empty:
            print("[prefilter][nc_ratio] skipped")
            print(
                "[prefilter][nc_ratio] reason =",
                df_nc.attrs.get("skip_reason", "unknown"),
            )
            print(
                "[prefilter][nc_ratio] mean_status =",
                df_nc.attrs.get("mean_status", "SKIPPED"),
            )
        else:
            print("[prefilter][nc_ratio] filter_mode =", df_nc.attrs.get("filter_mode", "unknown"))
            print("[prefilter][nc_ratio] n_cells_before =", df_nc.attrs.get("n_cells_before"))
            print("[prefilter][nc_ratio] n_cells_after_filter =", df_nc.attrs.get("n_cells_after"))
            print("[prefilter][nc_ratio] n_cells_removed =", df_nc.attrs.get("n_cells_removed"))
            print("[prefilter][nc_ratio] mean_before =", df_nc.attrs.get("mean_before"))
            print("[prefilter][nc_ratio] mean_after =", df_nc.attrs.get("mean_after"))
            print("[prefilter][nc_ratio] median_before =", df_nc.attrs.get("median_before"))
            print("[prefilter][nc_ratio] median_after =", df_nc.attrs.get("median_after"))
            print(
                "[prefilter][nc_ratio] mean_target_range =",
                f"{df_nc.attrs.get('mean_low', 0.4)}-{df_nc.attrs.get('mean_high', 0.6)}",
            )
            print("[prefilter][nc_ratio] mean_status =", df_nc.attrs.get("mean_status"))
            print("[prefilter][nc_ratio] q_low =", df_nc.attrs.get("q_low"))
            print("[prefilter][nc_ratio] q_high =", df_nc.attrs.get("q_high"))
            print("[prefilter][nc_ratio] q_low_value =", df_nc.attrs.get("q_low_value"))
            print("[prefilter][nc_ratio] q_high_value =", df_nc.attrs.get("q_high_value"))
            print("[prefilter][nc_ratio] quantiles =")
            print(
                df_nc["nc_ratio"].quantile(
                    [0.01, 0.025, 0.05, 0.25, 0.5, 0.75, 0.95, 0.975, 0.99]
                )
            )
        
       
    

    after_nc = filter_stats.get("after_nc_ratio_filter", {})
    if after_nc:
        print("[prefilter][after_nc_ratio] n_cells_in_data_df =", after_nc.get("n_cells"))
        print("[prefilter][after_nc_ratio] n_genes_in_data_df =", after_nc.get("n_genes"))
        print("[prefilter][after_nc_ratio] n_cellgenes_in_data_df =", after_nc.get("n_cellgenes"))
        print("[prefilter][after_nc_ratio] n_rows_in_data_df =", after_nc.get("n_rows"))
        print("[prefilter][after_nc_ratio] n_cell_boundary =", after_nc.get("n_cell_boundary"))
        print("[prefilter][after_nc_ratio] n_nuclear_boundary =", after_nc.get("n_nuclear_boundary"))

    cellgene_counts = filter_stats.get("cellgene_counts_df", None)
    if cellgene_counts is not None and not cellgene_counts.empty:
        print("[prefilter][cellgene] min_transcripts =", filter_stats.get("cellgene_filter_min_transcripts"))
        print("[prefilter][cellgene] n_cellgenes_before_filter =", filter_stats.get("n_cellgenes_before_cellgene_filter"))
        print("[prefilter][cellgene] n_cellgenes_after_filter =", filter_stats.get("n_cellgenes_after_cellgene_filter"))
        print("[prefilter][cellgene] n_cellgenes_removed =", filter_stats.get("n_cellgenes_removed_by_cellgene_filter"))
        print("[prefilter][cellgene] transcript count summary before filter =")
        print(cellgene_counts["n_transcripts"].describe())

    after_cellgene = filter_stats.get("after_cellgene_filter", {})
    if after_cellgene:
        print("[prefilter][after_cellgene] n_cells_in_data_df =", after_cellgene.get("n_cells"))
        print("[prefilter][after_cellgene] n_genes_in_data_df =", after_cellgene.get("n_genes"))
        print("[prefilter][after_cellgene] n_cellgenes_in_data_df =", after_cellgene.get("n_cellgenes"))
        print("[prefilter][after_cellgene] n_rows_in_data_df =", after_cellgene.get("n_rows"))
        print("[prefilter][after_cellgene] n_cell_boundary =", after_cellgene.get("n_cell_boundary"))
        print("[prefilter][after_cellgene] n_nuclear_boundary =", after_cellgene.get("n_nuclear_boundary"))

    gene_support = filter_stats.get("gene_support_df", None)
    if gene_support is not None and not gene_support.empty:
        print("[prefilter][gene_support] min_cells =", filter_stats.get("gene_filter_min_cells"))
        print("[prefilter][gene_support] n_genes_before_filter =", filter_stats.get("n_genes_before_gene_support_filter"))
        print("[prefilter][gene_support] n_genes_after_filter =", filter_stats.get("n_genes_after_gene_support_filter"))
        print("[prefilter][gene_support] n_genes_removed =", filter_stats.get("n_genes_removed_by_gene_support_filter"))
        print("[prefilter][gene_support] support summary after cell-gene filter =")
        print(gene_support["n_cells_after_cellgene_filter"].describe())

    after_gene = filter_stats.get("after_gene_support_filter", {})
    if after_gene:
        print("[prefilter][after_gene_support] n_cells_in_data_df =", after_gene.get("n_cells"))
        print("[prefilter][after_gene_support] n_genes_in_data_df =", after_gene.get("n_genes"))
        print("[prefilter][after_gene_support] n_cellgenes_in_data_df =", after_gene.get("n_cellgenes"))
        print("[prefilter][after_gene_support] n_rows_in_data_df =", after_gene.get("n_rows"))
        print("[prefilter][after_gene_support] n_cell_boundary =", after_gene.get("n_cell_boundary"))
        print("[prefilter][after_gene_support] n_nuclear_boundary =", after_gene.get("n_nuclear_boundary"))

    print("[prefilter][sync] n_common_cells =", filter_stats.get("n_common_cells"))

    final = filter_stats.get("final", {})
    if final:
        print("[prefilter][final] n_cells_in_data_df =", final.get("n_cells"))
        print("[prefilter][final] n_genes_in_data_df =", final.get("n_genes"))
        print("[prefilter][final] n_cellgenes_in_data_df =", final.get("n_cellgenes"))
        print("[prefilter][final] n_rows_in_data_df =", final.get("n_rows"))
        print("[prefilter][final] n_cell_boundary =", final.get("n_cell_boundary"))
        print("[prefilter][final] n_nuclear_boundary =", final.get("n_nuclear_boundary"))


def compute_all_from_pkl(
    pkl_path: str,
    *,
    bento_kwargs: dict | None = None,
    sprawl_kwargs: dict | None = None,
    out_path: str | None = None,
    how: str = "outer",
    profile: bool = False,
    max_cells: int | None = None,
    prefilter_kwargs: dict | None = None,
) -> pd.DataFrame:
    """
    一键计算并合并：
      - Bento 13 features（13列，bento_*）
      - SPRAWL scores（默认4列，sprawl_*）

    Default prefiltering can be controlled by prefilter_kwargs:
      - nc ratio cell filtering
      - cell-gene-level min transcript filtering
      - gene support filtering after cell-gene filtering
      - data_df / cell_boundary / nuclear_boundary cell synchronization
    """
    rep = TimerReport() if profile else None
    bento_kwargs = dict(bento_kwargs or {})
    sprawl_kwargs = dict(sprawl_kwargs or {})
    prefilter_kwargs = dict(prefilter_kwargs or {})

    bento_kwargs.setdefault("profile", profile)
    sprawl_kwargs.setdefault("profile", profile)

    tmp_path = None

    try:
        with timed("ALL:load_bundle", rep, print_each=profile):
            with open(pkl_path, "rb") as f:
                bundle = pickle.load(f)

        if prefilter_kwargs:
            with timed("ALL:prefilter_bundle", rep, print_each=profile):
                bundle, filter_stats = _prefilter_bundle_before_features(
                    bundle,
                    **prefilter_kwargs,
                )

            if profile:
                _print_prefilter_stats(filter_stats)

        if max_cells is not None:
            with timed(f"ALL:limit_cells({max_cells})", rep, print_each=profile):
                bundle, cells = _limit_cells_in_bundle(bundle, max_cells=max_cells)
            if profile:
                print(f"[compute_all] kept first {len(cells)} cells")

        with timed("ALL:dump_tmp_bundle", rep, print_each=profile):
            tmp = tempfile.NamedTemporaryFile(suffix=".pkl", delete=False)
            tmp_path = tmp.name
            tmp.close()
            with open(tmp_path, "wb") as f:
                pickle.dump(bundle, f)

        pkl_to_use = tmp_path

        with timed("ALL:bento_total", rep, print_each=profile):
            bento_df = compute_bento13_from_dict(pkl_to_use, **bento_kwargs)

        with timed("ALL:sprawl_total", rep, print_each=profile):
            sprawl_df = compute_sprawl_scores_from_pkl(pkl_to_use, **sprawl_kwargs)

        with timed("ALL:normalize_index", rep, print_each=profile):
            bento_df = _normalize_index(bento_df)
            sprawl_df = _normalize_index(sprawl_df)

        with timed("ALL:join", rep, print_each=profile):
            all_df = bento_df.join(sprawl_df, how=how)

        if out_path:
            with timed("ALL:write_out", rep, print_each=profile):
                saved = _save_df(all_df, out_path)
            print(f"[compute_all] saved to: {saved}")

        if profile and rep is not None:
            print(rep.summary())

        return all_df

    finally:
        if tmp_path is not None:
            try:
                os.remove(tmp_path)
            except Exception:
                pass



def run_all_with_patterns(
    pkl_path: str,
    model_path: str,
    *,
    bento_kwargs: dict | None = None,
    sprawl_kwargs: dict | None = None,
    feature_output_path: str | None = None,
    pattern_output_path: str | None = None,
    how: str = "outer",
    profile: bool = False,
    max_cells: int | None = None,
    prefilter_kwargs: dict | None = None,
    fallback_model_path: str | None = None,
    foci_fallback_threshold: float = 0.5,
    enable_foci_fallback: bool = True,
):
    """
    从 pkl 一步完成：
      1. 预过滤
      2. 计算 17 维特征
      3. 默认使用强 8 分类 XGBoost 模型预测定位模式
      4. 如果 8 分类结果中 Foci 占比 > foci_fallback_threshold，
         则使用 7 分类 no-Foci 模型重新预测，并用 7 分类结果替代原结果。

    注意：
      - Bento/SPRAWL 特征只计算一次。
      - fallback 只重新执行 PatternClassifier.predict()。
      - 最终类别列统一为 pattern。
    """
    feat_df = compute_all_from_pkl(
        pkl_path,
        bento_kwargs=bento_kwargs,
        sprawl_kwargs=sprawl_kwargs,
        out_path=None,
        how=how,
        profile=profile,
        max_cells=max_cells,
        prefilter_kwargs=prefilter_kwargs,
    )

    if feature_output_path is not None:
        Path(feature_output_path).parent.mkdir(parents=True, exist_ok=True)
        if str(feature_output_path).endswith(".csv"):
            feat_df.reset_index().to_csv(feature_output_path, index=False)
        else:
            feat_df.reset_index().to_parquet(feature_output_path, index=False)

    # ------------------------------------------------------------
    # Step 1. Primary prediction: strong 8-class model
    # ------------------------------------------------------------
    clf = PatternClassifier(model_path)
    out_df = clf.predict(feat_df.reset_index())

    foci_ratio = _get_foci_ratio(out_df, pattern_col="pattern")

    if profile:
        print("[pattern] primary_model =", model_path)
        print("[pattern] primary_classes =", getattr(clf, "classes", None))
        print("[pattern] primary_n_rows =", len(out_df))
        print("[pattern] primary_foci_ratio =", foci_ratio)

        if "pattern" in out_df.columns:
            print("[pattern] primary_distribution =")
            print(out_df["pattern"].value_counts(normalize=True))

    # ------------------------------------------------------------
    # Step 2. Foci fallback:
    # If primary 8-class prediction assigns > 50% rows to Foci,
    # rerun prediction with 7-class no-Foci model.
    # ------------------------------------------------------------
    used_fallback = False

    if (
        enable_foci_fallback
        and fallback_model_path is not None
        and foci_ratio > foci_fallback_threshold
    ):
        if profile:
            print(
                "[pattern][fallback] triggered because "
                f"Foci ratio {foci_ratio:.4f} > threshold {foci_fallback_threshold:.4f}"
            )
            print("[pattern][fallback] fallback_model =", fallback_model_path)

        fallback_clf = PatternClassifier(fallback_model_path)
        out_df = fallback_clf.predict(feat_df.reset_index())
        used_fallback = True

        if profile:
            print("[pattern][fallback] fallback_classes =", getattr(fallback_clf, "classes", None))
            print("[pattern][fallback] fallback_n_rows =", len(out_df))

            if "pattern" in out_df.columns:
                print("[pattern][fallback] fallback_distribution =")
                print(out_df["pattern"].value_counts(normalize=True))

    else:
        if profile:
            if not enable_foci_fallback:
                print("[pattern][fallback] disabled")
            elif fallback_model_path is None:
                print("[pattern][fallback] skipped because fallback_model_path is None")
            else:
                print(
                    "[pattern][fallback] not triggered because "
                    f"Foci ratio {foci_ratio:.4f} <= threshold {foci_fallback_threshold:.4f}"
                )

    # 建议保留这三个字段，方便之后知道到底用了哪个模型。
    # 如果你想让结果文件更干净，可以删掉这三行。
    # out_df["pattern_model_used"] = "fallback_7class_no_foci" if used_fallback else "primary_8class_prop075"
    # out_df["primary_foci_ratio"] = foci_ratio
    # out_df["foci_fallback_threshold"] = float(foci_fallback_threshold)

    if pattern_output_path is not None:
        Path(pattern_output_path).parent.mkdir(parents=True, exist_ok=True)
        if str(pattern_output_path).endswith(".csv"):
            out_df.to_csv(pattern_output_path, index=False)
        else:
            out_df.to_parquet(pattern_output_path, index=False)

    return out_df

    # clf = PatternClassifier(model_path)
    # out_df = clf.predict(feat_df.reset_index())

    # if pattern_output_path is not None:
    #     Path(pattern_output_path).parent.mkdir(parents=True, exist_ok=True)
    #     if str(pattern_output_path).endswith(".csv"):
    #         out_df.to_csv(pattern_output_path, index=False)
    #     else:
    #         out_df.to_parquet(pattern_output_path, index=False)

    # return out_df