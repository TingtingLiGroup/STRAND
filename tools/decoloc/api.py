from __future__ import annotations

from pathlib import Path
import pickle
from typing import Literal

import numpy as np
import pandas as pd

from tools.api.colocalization import (
    _apply_cell_sampling_to_bundle,
    _attach_groupby_from_coordinates,
    _safe_group_name,
    _subset_bundle_by_cells,
)
from tools.api.compute_all import _compute_nc_ratio_df
from tools.decoloc.core import DeCoLocResults, run_decoloc_core
from tools.engines.coloc_region_annotation import _load_cell_geometries

DEFAULT_NEG_PREFIXES: tuple[str, ...] = (
    "negprb",
    "negcontrol",
    "blank",
    "unassigned",
)


def prefilter_bundle_for_decoloc(
    bundle: dict,
    *,
    nc_ratio_q_low: float = 0.025,
    nc_ratio_q_high: float = 0.975,
    min_gene_total_transcripts: int = 5,
    min_gene_cells: int = 10,
) -> tuple[dict, dict]:
    """Lightweight prefilter preserving 1-5 copy cell-gene observations.

    Unlike single-gene 28D pattern classification (which requires >= 6 spots per cell),
    colocalization borrows strength across co-expressing cells; keeping 1-5 copy cells
    recovers sparse colocalization without sacrificing specificity.
    """
    df = bundle["data_df"].copy()
    df["cell"] = df["cell"].astype(str)
    df["gene"] = df["gene"].astype(str)

    n_cells_in = int(df["cell"].nunique())
    n_genes_in = int(df["gene"].nunique())
    n_rows_in = int(len(df))

    # 1. Cell-level extreme N/C ratio filter (only when nuclear boundaries exist).
    nc_df = _compute_nc_ratio_df(bundle)
    if not nc_df.empty:
        valid_nc = nc_df[
            np.isfinite(nc_df["nc_ratio"])
            & (nc_df["nc_ratio"] > 0.0)
            & (nc_df["nc_ratio"] < 1.0)
        ]
        if len(valid_nc) >= 10:
            low_v = float(valid_nc["nc_ratio"].quantile(nc_ratio_q_low))
            high_v = float(valid_nc["nc_ratio"].quantile(nc_ratio_q_high))
            kept_cells = set(
                valid_nc.loc[
                    (valid_nc["nc_ratio"] >= low_v) & (valid_nc["nc_ratio"] <= high_v),
                    "cell",
                ].astype(str)
            )
            if len(kept_cells) >= 10:
                df = df[df["cell"].isin(kept_cells)]

    # 2. Gene-level global support filter.
    gene_stats = df.groupby("gene", observed=True).agg(
        total_spots=("cell", "size"),
        n_cells=("cell", "nunique"),
    )
    kept_genes = set(
        gene_stats.index[
            (gene_stats["total_spots"] >= int(min_gene_total_transcripts))
            & (gene_stats["n_cells"] >= int(min_gene_cells))
        ]
    )
    df = df[df["gene"].isin(kept_genes)].reset_index(drop=True)

    # 3. Synchronize cell/nuclear boundary dicts if present.
    active_cells = set(df["cell"].unique())
    cb = bundle.get("cell_boundary", None)
    nb = bundle.get("nuclear_boundary", None)

    out_bundle = dict(bundle)
    if isinstance(cb, dict) and len(cb) > 0:
        cb_str = {str(k): v for k, v in cb.items() if str(k) in active_cells}
        if isinstance(nb, dict) and len(nb) > 0:
            nb_str = {str(k): v for k, v in nb.items() if str(k) in active_cells}
            common = set(cb_str.keys()) & set(nb_str.keys())
            df = df[df["cell"].isin(common)].reset_index(drop=True)
            out_bundle["cell_boundary"] = {k: cb_str[k] for k in common}
            out_bundle["nuclear_boundary"] = {k: nb_str[k] for k in common}
        else:
            common = set(cb_str.keys())
            df = df[df["cell"].isin(common)].reset_index(drop=True)
            out_bundle["cell_boundary"] = {k: cb_str[k] for k in common}

    out_bundle["data_df"] = df
    stats_dict = {
        "n_cells_in": n_cells_in,
        "n_genes_in": n_genes_in,
        "n_rows_in": n_rows_in,
        "n_cells_out": int(df["cell"].nunique()),
        "n_genes_out": int(df["gene"].nunique()),
        "n_rows_out": int(len(df)),
    }
    return out_bundle, stats_dict


def assign_spatial_grid_tiles(
    df: pd.DataFrame,
    *,
    tile_size: float = 20.0,
    x_col: str = "x",
    y_col: str = "y",
    out_col: str = "cell",
) -> pd.DataFrame:
    """Partition boundary-less transcript coordinates into square pseudo-cell tiles."""
    out = df.copy()
    x = out[x_col].to_numpy(dtype=np.float64)
    y = out[y_col].to_numpy(dtype=np.float64)
    gx = np.floor((x - np.nanmin(x)) / float(tile_size)).astype(np.int64)
    gy = np.floor((y - np.nanmin(y)) / float(tile_size)).astype(np.int64)
    out[out_col] = [f"tile_{ix}_{iy}" for ix, iy in zip(gx, gy)]
    return out


def summarize_negative_controls(
    all_pairs: pd.DataFrame,
    *,
    genes: tuple[str, ...],
    control_prefixes: tuple[str, ...] = DEFAULT_NEG_PREFIXES,
    q_thresholds: tuple[float, ...] = (0.05, 0.01, 0.001, 1e-4),
) -> pd.DataFrame:
    """Empirical false-positive check using platform negative-control probes if present."""
    lower_prefixes = tuple(p.lower() for p in control_prefixes)
    ctrl_genes = {g for g in genes if str(g).lower().startswith(lower_prefixes)}
    if not ctrl_genes or all_pairs.empty:
        return pd.DataFrame()

    is_ctrl = all_pairs["gene_1"].isin(ctrl_genes) | all_pairs["gene_2"].isin(ctrl_genes)
    rows = []
    for q_cut in q_thresholds:
        for layer, q_col in [("whole_cell", "q_cell"), ("within_compartment", "q_comp")]:
            sig = all_pairs[q_col] < q_cut
            n_sig = int(sig.sum())
            n_ctrl_sig = int((sig & is_ctrl).sum())
            rows.append(
                {
                    "null_layer": layer,
                    "q_threshold": q_cut,
                    "n_control_genes_in_panel": len(ctrl_genes),
                    "n_significant_pairs": n_sig,
                    "n_control_associated_pairs": n_ctrl_sig,
                    "control_pair_fraction": (n_ctrl_sig / n_sig) if n_sig > 0 else 0.0,
                }
            )
    return pd.DataFrame(rows)


def run_decoloc_from_bundle(
    bundle: dict,
    *,
    distance: float | None = None,
    distance_mode: Literal["physical", "cell_scaled"] = "cell_scaled",
    auto_distance_radius_frac: float = 0.10,
    region_threshold_mode: Literal["proportional", "cell_scaled", "absolute"] = "proportional",
    kernel: Literal["gaussian", "uniform"] = "gaussian",
    kernel_sigma_frac: float = 0.5,
    use_3d: bool = False,
    min_coexpr_cells: int = 15,
    fdr_alpha: float = 0.01,
    min_fe: float = 1.15,
    grid_tile_size: float = 20.0,
    do_prefilter: bool = True,
    min_gene_total_transcripts: int = 5,
    min_gene_cells: int = 10,
    threads: int = 8,
    profile: bool = False,
) -> tuple[DeCoLocResults, dict]:
    """Run DeCoLoc on an in-memory standard PKL bundle."""
    work_bundle = dict(bundle)
    df = work_bundle["data_df"]

    has_cell_col = "cell" in df.columns and df["cell"].notna().any()
    has_cb = isinstance(work_bundle.get("cell_boundary"), dict) and len(work_bundle["cell_boundary"]) > 0
    has_nb = isinstance(work_bundle.get("nuclear_boundary"), dict) and len(work_bundle["nuclear_boundary"]) > 0

    if not has_cell_col:
        work_bundle["data_df"] = assign_spatial_grid_tiles(df, tile_size=grid_tile_size)

    filter_stats = {}
    if do_prefilter:
        work_bundle, filter_stats = prefilter_bundle_for_decoloc(
            work_bundle,
            min_gene_total_transcripts=min_gene_total_transcripts,
            min_gene_cells=min_gene_cells,
        )

    geoms = _load_cell_geometries(work_bundle) if (has_cb and has_nb) else None
    results = run_decoloc_core(
        work_bundle["data_df"],
        geometries=geoms,
        distance=distance,
        distance_mode=distance_mode if geoms else "physical",
        auto_distance_radius_frac=auto_distance_radius_frac,
        region_threshold_mode=region_threshold_mode,
        kernel=kernel,
        kernel_sigma_frac=kernel_sigma_frac,
        use_3d=use_3d,
        min_coexpr_cells=min_coexpr_cells,
        fdr_alpha=fdr_alpha,
        min_fe=min_fe,
        threads=threads,
        profile=profile,
    )
    return results, filter_stats


def run_decoloc_from_pkl(
    pkl_path: str | Path,
    *,
    out_dir: str | Path,
    distance: float | None = None,
    distance_mode: Literal["physical", "cell_scaled"] = "cell_scaled",
    auto_distance_radius_frac: float = 0.10,
    region_threshold_mode: Literal["proportional", "cell_scaled", "absolute"] = "proportional",
    kernel: Literal["gaussian", "uniform"] = "gaussian",
    kernel_sigma_frac: float = 0.5,
    use_3d: bool = False,
    min_coexpr_cells: int = 15,
    fdr_alpha: float = 0.01,
    min_fe: float = 1.15,
    grid_tile_size: float = 20.0,
    do_prefilter: bool = True,
    min_gene_total_transcripts: int = 5,
    min_gene_cells: int = 10,
    sample_cells: int | None = None,
    sample_cell_frac: float | None = None,
    sample_random_state: int = 42,
    sample_stratify_by: str | None = None,
    groupby: str | None = None,
    group_values: list[str] | None = None,
    save_all_pairs: bool = False,
    threads: int = 8,
    profile: bool = False,
) -> dict:
    """Load a PKL bundle, run DeCoLoc (optionally grouped by celltype), and write CSV outputs."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(pkl_path, "rb") as f:
        bundle = pickle.load(f)

    filter_stats = {}
    if do_prefilter:
        bundle, filter_stats = prefilter_bundle_for_decoloc(
            bundle,
            min_gene_total_transcripts=min_gene_total_transcripts,
            min_gene_cells=min_gene_cells,
        )

    if sample_cells is not None or sample_cell_frac is not None:
        bundle, _, _, _ = _apply_cell_sampling_to_bundle(
            bundle,
            sample_cells=sample_cells,
            sample_cell_frac=sample_cell_frac,
            random_state=sample_random_state,
            stratify_by=sample_stratify_by,
            out_dir=out_dir,
        )

    if groupby is not None:
        bundle = _attach_groupby_from_coordinates(bundle, groupby)
        data_df = bundle["data_df"].copy()
        data_df["cell"] = data_df["cell"].astype(str)
        group_cells = (
            data_df[["cell", groupby]]
            .dropna(subset=[groupby])
            .drop_duplicates()
            .groupby(groupby)["cell"]
            .apply(lambda s: set(s.astype(str)))
        )
        if group_values is not None:
            wanted = {str(g) for g in group_values}
            group_cells = group_cells[group_cells.index.astype(str).isin(wanted)]

        summary_rows = []
        group_outputs = {}
        for grp_val, cell_set in group_cells.items():
            sub_dir = out_dir / _safe_group_name(grp_val)
            sub_dir.mkdir(parents=True, exist_ok=True)
            sub_bundle = _subset_bundle_by_cells(bundle, cell_set)
            res, _ = run_decoloc_from_bundle(
                sub_bundle,
                distance=distance,
                distance_mode=distance_mode,
                auto_distance_radius_frac=auto_distance_radius_frac,
                region_threshold_mode=region_threshold_mode,
                kernel=kernel,
                kernel_sigma_frac=kernel_sigma_frac,
                use_3d=use_3d,
                min_coexpr_cells=min_coexpr_cells,
                fdr_alpha=fdr_alpha,
                min_fe=min_fe,
                grid_tile_size=grid_tile_size,
                do_prefilter=False,
                threads=threads,
                profile=profile,
            )
            _write_single_group_outputs(
                res,
                out_dir=sub_dir,
                filter_stats={"group": str(grp_val), "n_cells": len(res.cell_ids)},
                save_all_pairs=save_all_pairs,
            )
            n_direct = int((res.significant_pairs["coloc_class"] == "direct_micro").sum()) if not res.significant_pairs.empty else 0
            n_cocomp = int((res.significant_pairs["coloc_class"] == "co_compartment").sum()) if not res.significant_pairs.empty else 0
            summary_rows.append(
                {
                    "group": str(grp_val),
                    "n_cells": len(res.cell_ids),
                    "n_genes": len(res.genes),
                    "n_significant_total": len(res.significant_pairs),
                    "n_direct_micro": n_direct,
                    "n_co_compartment": n_cocomp,
                    "elapsed_seconds": res.elapsed_seconds,
                }
            )
            group_outputs[str(grp_val)] = res

        grp_summary_df = pd.DataFrame(summary_rows)
        grp_summary_path = out_dir / f"decoloc_groupby_{_safe_group_name(groupby)}_summary.csv"
        grp_summary_df.to_csv(grp_summary_path, index=False)
        return {"group_results": group_outputs, "summary_path": grp_summary_path}

    res, _ = run_decoloc_from_bundle(
        bundle,
        distance=distance,
        distance_mode=distance_mode,
        auto_distance_radius_frac=auto_distance_radius_frac,
        region_threshold_mode=region_threshold_mode,
        kernel=kernel,
        kernel_sigma_frac=kernel_sigma_frac,
        use_3d=use_3d,
        min_coexpr_cells=min_coexpr_cells,
        fdr_alpha=fdr_alpha,
        min_fe=min_fe,
        grid_tile_size=grid_tile_size,
        do_prefilter=False,
        threads=threads,
        profile=profile,
    )
    paths = _write_single_group_outputs(
        res,
        out_dir=out_dir,
        filter_stats=filter_stats,
        save_all_pairs=save_all_pairs,
    )
    return {"results": res, **paths}


def _write_single_group_outputs(
    res: DeCoLocResults,
    *,
    out_dir: Path,
    filter_stats: dict,
    save_all_pairs: bool,
) -> dict:
    sig_path = out_dir / "decoloc_significant_pairs.csv"
    direct_path = out_dir / "decoloc_direct_micro_pairs.csv"
    summary_path = out_dir / "decoloc_summary.csv"

    res.significant_pairs.to_csv(sig_path, index=False)
    if not res.significant_pairs.empty and "coloc_class" in res.significant_pairs.columns:
        direct_df = res.significant_pairs[res.significant_pairs["coloc_class"] == "direct_micro"]
    else:
        direct_df = res.significant_pairs
    direct_df.to_csv(direct_path, index=False)

    all_path = None
    if save_all_pairs:
        all_path = out_dir / "decoloc_all_active_pairs.csv"
        res.all_pairs.to_csv(all_path, index=False)

    neg_df = summarize_negative_controls(res.all_pairs, genes=res.genes)
    neg_path = None
    if not neg_df.empty:
        neg_path = out_dir / "neg_control_summary.csv"
        neg_df.to_csv(neg_path, index=False)

    n_direct = int((res.significant_pairs["coloc_class"] == "direct_micro").sum()) if not res.significant_pairs.empty else 0
    n_cocomp = int((res.significant_pairs["coloc_class"] == "co_compartment").sum()) if not res.significant_pairs.empty else 0
    summary_row = {
        **filter_stats,
        "n_analyzed_cells": len(res.cell_ids),
        "n_analyzed_genes": len(res.genes),
        "n_testable_pairs": res.n_testable_pairs,
        "n_active_pairs": len(res.all_pairs),
        "n_significant_total": len(res.significant_pairs),
        "n_direct_micro": n_direct,
        "n_co_compartment": n_cocomp,
        "median_cell_radius": res.median_cell_radius,
        "effective_median_distance": res.effective_median_distance,
        "elapsed_seconds": res.elapsed_seconds,
    }
    pd.DataFrame([summary_row]).to_csv(summary_path, index=False)

    return {
        "significant_pairs_path": sig_path,
        "direct_micro_pairs_path": direct_path,
        "all_pairs_path": all_path,
        "summary_path": summary_path,
        "neg_control_path": neg_path,
    }
