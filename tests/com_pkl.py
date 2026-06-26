from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from shapely.geometry import Polygon, MultiPolygon
from shapely.validation import make_valid


def polygon_area(coords) -> float:
    """Safely compute polygon area from boundary coordinates."""
    if coords is None:
        return np.nan

    try:
        # 你的 cell_boundary / nuclear_boundary 可能是 DataFrame，也可能是 array/list
        if isinstance(coords, pd.DataFrame):
            if {"x", "y"}.issubset(coords.columns):
                arr = coords[["x", "y"]].to_numpy(dtype=float)
            else:
                arr = coords.iloc[:, :2].to_numpy(dtype=float)
        else:
            arr = np.asarray(coords, dtype=float)

        if arr.ndim != 2 or arr.shape[1] < 2 or arr.shape[0] < 3:
            return np.nan

        poly = Polygon(arr[:, :2])
        if not poly.is_valid:
            poly = make_valid(poly)

        if poly.is_empty:
            return np.nan

        if isinstance(poly, MultiPolygon):
            return float(sum(p.area for p in poly.geoms))

        return float(poly.area)

    except Exception:
        return np.nan


def attach_batch_from_coordinates(data_df, coordinates, batch_col="batch"):
    """
    Add batch column to data_df.

    Priority:
    1. if data_df already has batch_col, use it.
    2. else if coordinates has cell + batch_col, merge by cell.
    3. else create batch0.
    """
    data_df = data_df.copy()
    data_df["cell"] = data_df["cell"].astype(str)

    if batch_col in data_df.columns:
        data_df[batch_col] = data_df[batch_col].astype(str)
        print(f"[INFO] Use batch column from data_df['{batch_col}']")
        return data_df

    if coordinates is not None and isinstance(coordinates, pd.DataFrame):
        coords = coordinates.copy()

        if "cell" in coords.columns and batch_col in coords.columns:
            coords["cell"] = coords["cell"].astype(str)
            coords[batch_col] = coords[batch_col].astype(str)

            batch_map = coords[["cell", batch_col]].drop_duplicates("cell")

            data_df = data_df.merge(batch_map, on="cell", how="left")

            missing = data_df[batch_col].isna().sum()
            if missing > 0:
                print(
                    f"[WARN] {missing} transcripts could not find batch from coordinates. "
                    f"Set them to batch0."
                )
                data_df[batch_col] = data_df[batch_col].fillna("batch0")

            print(f"[INFO] Attach batch from coordinates[['cell', '{batch_col}']]")
            return data_df

    print(f"[WARN] No batch column found in data_df or coordinates. Use batch0.")
    data_df[batch_col] = "batch0"
    return data_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkl", required=True, help="Input unified PKL file")
    parser.add_argument("--out", default=None, help="Output CSV path")
    parser.add_argument("--batch-col", default="batch")
    args = parser.parse_args()

    pkl_path = Path(args.pkl)

    with open(pkl_path, "rb") as f:
        obj = pickle.load(f)

    data_df = obj["data_df"].copy()
    cell_boundary = obj.get("cell_boundary", {})
    nuclear_boundary = obj.get("nuclear_boundary", {})
    coordinates = obj.get("coordinates", None)
    expression = obj.get("expression", None)
    metadata = obj.get("metadata", None)

    required = {"cell", "gene", "x", "y"}
    missing = required - set(data_df.columns)
    if missing:
        raise ValueError(f"data_df missing required columns: {missing}")

    data_df["cell"] = data_df["cell"].astype(str)
    data_df["gene"] = data_df["gene"].astype(str)

    # 关键修改：batch 从 coordinates 表合并回来
    data_df = attach_batch_from_coordinates(
        data_df,
        coordinates,
        batch_col=args.batch_col,
    )

    # ---------- cell-level transcript count ----------
    cell_tx = (
        data_df.groupby([args.batch_col, "cell"])
        .size()
        .rename("cell_transcripts")
        .reset_index()
    )

    # ---------- cell-level gene count ----------
    cell_gene = (
        data_df.groupby([args.batch_col, "cell"])["gene"]
        .nunique()
        .rename("cell_genes")
        .reset_index()
    )

    cell_stats = cell_tx.merge(cell_gene, on=[args.batch_col, "cell"], how="left")

    # ---------- cell area ----------
    nuclear_keys = {str(k) for k in nuclear_boundary.keys()} if isinstance(nuclear_boundary, dict) else set()

    area_records = []
    if isinstance(cell_boundary, dict):
        for cell_id, coords in cell_boundary.items():
            cid = str(cell_id)
            area_records.append(
                {
                    "cell": cid,
                    "cell_area": polygon_area(coords),
                    "has_cell_boundary": True,
                    "has_nuclear_boundary": cid in nuclear_keys,
                }
            )

    area_df = pd.DataFrame(area_records)
    if len(area_df) == 0:
        area_df = pd.DataFrame(
            columns=[
                "cell",
                "cell_area",
                "has_cell_boundary",
                "has_nuclear_boundary",
            ]
        )

    cell_stats = cell_stats.merge(area_df, on="cell", how="left")
    cell_stats["has_cell_boundary"] = cell_stats["has_cell_boundary"].fillna(False)
    cell_stats["has_nuclear_boundary"] = cell_stats["has_nuclear_boundary"].fillna(False)

    # ---------- batch-level basic stats ----------
    batch_basic = (
        data_df.groupby(args.batch_col)
        .agg(
            n_transcripts=("gene", "size"),
            n_genes=("gene", "nunique"),
            n_cells_from_transcripts=("cell", "nunique"),
        )
        .reset_index()
        .rename(columns={args.batch_col: "batch"})
    )

    # ---------- batch-level cell stats ----------
    batch_cell = (
        cell_stats.groupby(args.batch_col)
        .agg(
            n_cells=("cell", "nunique"),
            mean_tx_per_cell=("cell_transcripts", "mean"),
            median_tx_per_cell=("cell_transcripts", "median"),
            max_tx_per_cell=("cell_transcripts", "max"),
            mean_genes_per_cell=("cell_genes", "mean"),
            median_genes_per_cell=("cell_genes", "median"),
            max_genes_per_cell=("cell_genes", "max"),
            total_cell_area=("cell_area", "sum"),
            mean_cell_area=("cell_area", "mean"),
            max_cell_area=("cell_area", "max"),
            n_cells_with_boundary=("has_cell_boundary", "sum"),
            n_cells_with_nucleus=("has_nuclear_boundary", "sum"),
        )
        .reset_index()
        .rename(columns={args.batch_col: "batch"})
    )

    summary = batch_basic.merge(batch_cell, on="batch", how="outer")

    # ---------- risk indicators ----------
    summary["risk_area_x_genes"] = summary["total_cell_area"] * summary["n_genes"]
    summary["risk_tx_x_genes"] = summary["n_transcripts"] * summary["n_genes"]

    summary["risk_rank_score"] = (
        summary["risk_area_x_genes"].rank(ascending=False, method="min")
        + summary["risk_tx_x_genes"].rank(ascending=False, method="min")
    )

    summary = summary.sort_values(
        ["risk_rank_score", "risk_area_x_genes", "risk_tx_x_genes"],
        ascending=[True, False, False],
    )

    # ---------- overview ----------
    print("\n========== PKL keys ==========")
    print(list(obj.keys()))

    print("\n========== Dataset overview ==========")
    print(f"PKL: {pkl_path}")
    print(f"Total transcripts in data_df: {len(data_df):,}")
    print(f"Total cells in data_df: {data_df['cell'].nunique():,}")
    print(f"Total genes in data_df: {data_df['gene'].nunique():,}")
    print(f"Total cell boundaries: {len(cell_boundary):,}")
    print(f"Total nuclear boundaries: {len(nuclear_boundary):,}")
    print(f"Batch column: {args.batch_col}")
    print(f"Number of batches: {summary['batch'].nunique():,}")

    if isinstance(coordinates, pd.DataFrame):
        print("\n========== coordinates overview ==========")
        print(f"coordinates shape: {coordinates.shape}")
        print(f"coordinates columns: {list(coordinates.columns)}")
        if args.batch_col in coordinates.columns:
            print(f"batches from coordinates: {sorted(coordinates[args.batch_col].astype(str).unique().tolist())}")

    if isinstance(expression, pd.DataFrame):
        print("\n========== expression overview ==========")
        print(f"expression shape: {expression.shape}")

    print("\n========== Top risky batches ==========")
    cols = [
        "batch",
        "n_cells",
        "n_transcripts",
        "n_genes",
        "mean_tx_per_cell",
        "max_tx_per_cell",
        "mean_genes_per_cell",
        "max_genes_per_cell",
        "total_cell_area",
        "mean_cell_area",
        "max_cell_area",
        "risk_area_x_genes",
        "risk_tx_x_genes",
    ]
    print(summary[cols].head(30).to_string(index=False))

    out = args.out
    if out is None:
        out = str(pkl_path.with_suffix("")) + "_batch_compartment_risk_summary.csv"

    summary.to_csv(out, index=False)
    print(f"\n[OK] Saved batch summary to: {out}")


if __name__ == "__main__":
    main()