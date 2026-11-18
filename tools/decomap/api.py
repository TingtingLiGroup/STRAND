"""Python orchestration API for DeCoMap subcellular compartment detection."""

from __future__ import annotations

import json
from pathlib import Path
import pickle
from typing import Any, Sequence, Union

from matplotlib.cm import ScalarMappable
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize
import numpy as np
import pandas as pd
from shapely.plotting import patch_from_polygon
from shapely.wkt import loads

from tools.decomap.core import DeCoMapResults, run_decomap_core
from tools.decomap.regions import build_compartment_regions
from tools.engines.coloc_region_annotation import (
    CellGeometry,
    _boundary_to_polygon,
    _safe_intersection,
)
from tools.engines.compartment_adapter import (
    _attach_batch_from_coordinates,
    _filter_boundary_dict,
    _make_shape_df,
    _parse_only_batches,
    _plot_embedding_notebook,
    _resolve_figsize_and_limits,
    _safe_output_value,
)


def _decomap_output_columns(df: pd.DataFrame, batch_col: str) -> list[str]:
    """Stable column order backward-compatible with compartment_adapter plus DeCoMap columns."""
    cols: list[str] = []
    for c in [
        "cell",
        "gene",
        "x",
        "y",
        "z",
        "umi",
        batch_col,
        "batch",
        "tile",
        "fov",
        "sample_id",
        "celltype",
        "annotation",
        "subdomain",
        "raw_subdomain",
        "compartment_name",
        "confidence",
        "local_csr_z",
        "is_structured",
        "rnaflux_r",
        "rnaflux_g",
        "rnaflux_b",
    ]:
        if c in df.columns and c not in cols:
            cols.append(c)
    return cols


def _plot_decomap_subdomains(
    df: pd.DataFrame,
    cell_shape: pd.DataFrame,
    nucleus_shape: pd.DataFrame,
    summary_df: pd.DataFrame,
    out_png: Path,
    *,
    regions: pd.DataFrame | None = None,
    plot_points: bool = False,
    figsize: Any = "auto",
    dpi: int = 300,
    point_size: float = 1.0,
) -> None:
    """Plot DeCoMap compartments as filled regions (or transcript points) with cell/nuclear outlines.

    Unassigned area inside a cell is left white; cells failing the CSR gate are filled light grey.
    """
    resolved_figsize, xlim, ylim = _resolve_figsize_and_limits(df, figsize)
    fig, ax = plt.subplots(figsize=resolved_figsize)

    k_total = max(len(summary_df), 1)
    cmap_colors = plt.cm.tab10(np.linspace(0, min(0.9, (k_total - 1) * 0.1), k_total))
    cmap = ListedColormap(cmap_colors)
    norm = Normalize(vmin=-0.5, vmax=k_total - 0.5)
    draw_points = plot_points or regions is None
    unstruct_mask = df["subdomain"] < 0

    if regions is not None:
        unstruct_cells = set(df.loc[unstruct_mask, "cell"].astype(str)) - set(df.loc[~unstruct_mask, "cell"].astype(str))
        for row in cell_shape.itertuples(index=False):
            if str(row.cell) in unstruct_cells:
                ax.add_patch(patch_from_polygon(loads(row.cell_shape), facecolor="#E6E6E6", edgecolor="none"))
        for row in regions.itertuples(index=False):
            ax.add_patch(
                patch_from_polygon(
                    loads(row.geometry_wkt), facecolor=cmap(norm(row.subdomain)), edgecolor="none", alpha=0.85
                )
            )

    if draw_points and bool(unstruct_mask.any()):
        sub_u = df.loc[unstruct_mask]
        ax.scatter(
            sub_u["x"],
            sub_u["y"],
            c="#CFCFCF",
            s=point_size * 0.8,
            alpha=0.35,
            linewidths=0,
            label="Unstructured (CSR)",
        )

    struct_df = df.loc[~unstruct_mask]

    if not struct_df.empty:
        if draw_points:
            ax.scatter(
                struct_df["x"],
                struct_df["y"],
                c=struct_df["subdomain"],
                cmap=cmap,
                norm=norm,
                s=point_size,
                alpha=0.75,
                linewidths=0,
            )
        cbar = plt.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, label="Compartment")
        ticks = list(range(k_total))
        labels = [
            f"{row.subdomain}: {row.compartment_name}"
            for row in summary_df.itertuples(index=False)
        ]
        cbar.set_ticks(ticks)
        cbar.set_ticklabels(labels)
        cbar.ax.invert_yaxis()

    for _, row in cell_shape.iterrows():
        poly = loads(row["cell_shape"])
        if not poly.is_empty:
            bx, by = poly.exterior.xy
            ax.plot(bx, by, linestyle="-", color="black", linewidth=0.8)

    for _, row in nucleus_shape.iterrows():
        poly = loads(row["nucleus_shape"])
        if not poly.is_empty:
            nx, ny = poly.exterior.xy
            ax.plot(nx, ny, linestyle="-", color="red", linewidth=0.7)

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.set_title("DeCoMap Subcellular Compartments")
    plt.tight_layout()
    plt.savefig(out_png, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _load_decomap_geometries(
    bundle: dict[str, Any],
    *,
    include_posthoc_nuclear_qc: bool = True,
) -> dict[str, CellGeometry]:
    """Load all valid cell polygons independently of whether nuclear boundaries exist."""
    cell_boundary = bundle.get("cell_boundary") or {}
    nuclear_boundary = (bundle.get("nuclear_boundary") or {}) if include_posthoc_nuclear_qc else {}
    if not isinstance(cell_boundary, dict):
        return {}

    nb_str = {str(k): v for k, v in nuclear_boundary.items()} if isinstance(nuclear_boundary, dict) else {}
    geoms: dict[str, CellGeometry] = {}
    for raw_cid, raw_cpoly in cell_boundary.items():
        cid = str(raw_cid)
        cell = _boundary_to_polygon(raw_cpoly)
        if cell is None:
            continue
        nuc = _boundary_to_polygon(nb_str.get(cid)) if cid in nb_str else None
        if nuc is not None:
            nuc = _safe_intersection(nuc, cell) or nuc
        geoms[cid] = CellGeometry(cell_id=cid, cell=cell, nuclear=nuc)
    return geoms


def run_decomap_from_bundle(
    bundle: dict[str, Any],
    *,
    n_clusters: Union[str, int] = "auto",
    cluster_range_min: int = 4,
    cluster_range_max: int = 7,
    n_components: int = 10,
    n_micro_clusters: int = 30,
    sigma_fine_frac: float = 0.07,
    sigma_broad_frac: float = 0.21,
    target_gene_neighbors: float | None = 0.088,
    prior_fine: float = 3.0,
    prior_broad: float = 6.0,
    spatial_smooth_alpha: float = 0.40,
    calibrate_csr: bool = True,
    fdr_alpha: float = 0.05,
    include_posthoc_nuclear_qc: bool = True,
    marker_csv: Union[str, Path, None] = None,
    random_state: int = 42,
    threads: int = 0,
) -> DeCoMapResults:
    """Run DeCoMap directly on an in-memory STRAND PKL bundle dictionary."""
    geoms = _load_decomap_geometries(
        bundle, include_posthoc_nuclear_qc=include_posthoc_nuclear_qc
    )
    return run_decomap_core(
        bundle["data_df"],
        geometries=geoms,
        n_clusters=n_clusters,
        cluster_range_min=cluster_range_min,
        cluster_range_max=cluster_range_max,
        n_components=n_components,
        n_micro_clusters=n_micro_clusters,
        sigma_fine_frac=sigma_fine_frac,
        sigma_broad_frac=sigma_broad_frac,
        target_gene_neighbors=target_gene_neighbors,
        prior_fine=prior_fine,
        prior_broad=prior_broad,
        spatial_smooth_alpha=spatial_smooth_alpha,
        calibrate_csr=calibrate_csr,
        fdr_alpha=fdr_alpha,
        marker_csv=marker_csv,
        random_state=random_state,
        threads=threads,
    )


def run_decomap_from_pkl(
    pkl_path: Union[str, Path],
    out_prefix: Union[str, Path],
    *,
    batch_col: str = "batch",
    only_batches: Sequence[str] | str | None = None,
    display_col: str | None = None,
    n_clusters: Union[str, int] = "auto",
    cluster_range_min: int = 4,
    cluster_range_max: int = 7,
    n_components: int = 10,
    n_micro_clusters: int = 30,
    calibrate_csr: bool = True,
    fdr_alpha: float = 0.05,
    marker_csv: Union[str, Path, None] = None,
    figsize: Any = "auto",
    dpi: int = 300,
    embedding_point_size: float = 10.0,
    subdomain_point_size: float = 1.0,
    subdomain_plot_points: bool = False,
    write_plots: bool = True,
    output_csv_chunk_size: int = 250_000,
    random_state: int = 42,
    threads: int = 0,
    profile: bool = False,
) -> dict[str, Any]:
    """Run DeCoMap per group from a standardized STRAND PKL bundle and export compatible files."""
    pkl_path = Path(pkl_path)
    out_prefix = Path(out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    with pkl_path.open("rb") as fh:
        obj = pickle.load(fh)

    data_df = obj["data_df"]
    cell_boundary = obj.get("cell_boundary", {})
    nuclear_boundary = obj.get("nuclear_boundary", {})
    coordinates = obj.get("coordinates", None)
    del obj

    data_df["cell"] = data_df["cell"].astype(str)
    data_df["gene"] = data_df["gene"].astype(str)
    data_df = _attach_batch_from_coordinates(data_df, coordinates, batch_col=batch_col)
    if display_col is not None and display_col != batch_col:
        data_df = _attach_batch_from_coordinates(data_df, coordinates, batch_col=display_col)

    all_batches = [str(x) for x in pd.unique(data_df[batch_col].astype(str))]
    selected = _parse_only_batches(only_batches)
    if selected is not None:
        run_batches = [b for b in all_batches if str(b) in set(selected)]
        missing = sorted(set(selected) - set(run_batches))
        if missing:
            raise ValueError(f"Requested groups not found: {missing}. Available: {all_batches}")
    else:
        run_batches = all_batches

    group_name = _safe_output_value(batch_col)
    rows = []
    for batch in run_batches:
        safe_batch = _safe_output_value(batch)
        batch_prefix = Path(f"{out_prefix}_{group_name}-{safe_batch}")
        try:
            mask = data_df[batch_col].astype(str) == str(batch)
            sub_df = data_df if bool(mask.all()) else data_df.loc[mask].copy()
            cids = set(sub_df["cell"].astype(str).unique().tolist())
            sub_cb = _filter_boundary_dict(cell_boundary, cids)
            sub_nb = _filter_boundary_dict(nuclear_boundary, cids)

            res = run_decomap_from_bundle(
                {"data_df": sub_df, "cell_boundary": sub_cb, "nuclear_boundary": sub_nb},
                n_clusters=n_clusters,
                cluster_range_min=cluster_range_min,
                cluster_range_max=cluster_range_max,
                n_components=n_components,
                n_micro_clusters=n_micro_clusters,
                calibrate_csr=calibrate_csr,
                fdr_alpha=fdr_alpha,
                marker_csv=marker_csv,
                random_state=random_state,
                threads=threads,
            )

            regions = build_compartment_regions(
                res.point_df,
                _load_decomap_geometries({"cell_boundary": sub_cb}, include_posthoc_nuclear_qc=False),
                res.compartment_summary,
                sigma_frac=res.sigma_fine_frac,
                threads=threads,
            )

            summary_csv = Path(f"{batch_prefix}_compartment_summary.csv")
            cell_stats_csv = Path(f"{batch_prefix}_cell_csr_stats.csv")
            regions_csv = Path(f"{batch_prefix}_compartment_regions.csv")
            meta_json = Path(f"{batch_prefix}_sampled_meta.json")
            res.compartment_summary.to_csv(summary_csv, index=False)
            res.cell_stats.to_csv(cell_stats_csv, index=False)
            regions.to_csv(regions_csv, index=False)

            out_cols = _decomap_output_columns(res.point_df, batch_col)
            cell_shape = _make_shape_df(sub_cb, "cell_shape")
            nucleus_shape = _make_shape_df(sub_nb, "nucleus_shape")

            display_outputs = []
            embedding_png = Path(f"{batch_prefix}_sampled_rnaflux_embedding.png")
            subdomains_png = Path(f"{batch_prefix}_sampled_subdomains.png")
            points_csv = Path(f"{batch_prefix}_sampled_subdomain_points.csv")

            if display_col is not None:
                if display_col not in res.point_df.columns:
                    raise ValueError(f"Display column not found: {display_col}")
                safe_disp_col = _safe_output_value(display_col)
                for disp_val, idx_pos in res.point_df.groupby(display_col, sort=False, observed=False).indices.items():
                    gdf = res.point_df.iloc[np.asarray(idx_pos, dtype=np.int64)]
                    g_prefix = Path(f"{batch_prefix}_{safe_disp_col}-{_safe_output_value(disp_val)}")
                    g_csv = Path(f"{g_prefix}_sampled_subdomain_points.csv")
                    g_emb = Path(f"{g_prefix}_sampled_rnaflux_embedding.png")
                    g_sub = Path(f"{g_prefix}_sampled_subdomains.png")
                    gdf[out_cols].to_csv(g_csv, index=False, chunksize=output_csv_chunk_size)
                    if write_plots:
                        gcids = set(gdf["cell"].astype(str).unique())
                        _plot_embedding_notebook(
                            gdf,
                            gdf[["rnaflux_r", "rnaflux_g", "rnaflux_b"]].to_numpy(),
                            g_emb,
                            figsize=figsize,
                            dpi=dpi,
                            point_size=embedding_point_size,
                        )
                        _plot_decomap_subdomains(
                            gdf,
                            cell_shape[cell_shape["cell"].isin(gcids)],
                            nucleus_shape[nucleus_shape["cell"].isin(gcids)],
                            res.compartment_summary,
                            g_sub,
                            regions=regions[regions["cell"].isin(gcids)],
                            plot_points=subdomain_plot_points,
                            figsize=figsize,
                            dpi=dpi,
                            point_size=subdomain_point_size,
                        )
                    display_outputs.append(
                        {
                            "display_value": str(disp_val),
                            "n_transcripts": int(len(gdf)),
                            "embedding_png": str(g_emb) if write_plots else None,
                            "subdomains_png": str(g_sub) if write_plots else None,
                            "subdomain_points_csv": str(g_csv),
                        }
                    )
            else:
                res.point_df[out_cols].to_csv(points_csv, index=False, chunksize=output_csv_chunk_size)
                if write_plots:
                    _plot_embedding_notebook(
                        res.point_df,
                        res.point_df[["rnaflux_r", "rnaflux_g", "rnaflux_b"]].to_numpy(),
                        embedding_png,
                        figsize=figsize,
                        dpi=dpi,
                        point_size=embedding_point_size,
                    )
                    _plot_decomap_subdomains(
                        res.point_df,
                        cell_shape,
                        nucleus_shape,
                        res.compartment_summary,
                        subdomains_png,
                        regions=regions,
                        plot_points=subdomain_plot_points,
                        figsize=figsize,
                        dpi=dpi,
                        point_size=subdomain_point_size,
                    )

            meta = {
                "backend": "decomap_v1",
                "batch": str(batch),
                "best_k": int(res.best_k),
                "n_transcripts": int(len(res.point_df)),
                "n_cells": int(res.n_total_cells),
                "n_structured_cells": int(res.n_structured_cells),
                "structured_spot_fraction": float(res.structured_spot_fraction),
                "n_genes": int(len(res.genes)),
                "median_cell_radius": float(res.median_cell_radius),
                "scale_factor": float(res.scale_factor),
                "run_time_min": float(res.elapsed_seconds / 60.0),
                "outputs": {
                    "subdomain_points_csv": str(points_csv) if display_col is None else None,
                    "compartment_summary_csv": str(summary_csv),
                    "cell_csr_stats_csv": str(cell_stats_csv),
                    "compartment_regions_csv": str(regions_csv),
                    "embedding_png": str(embedding_png) if (display_col is None and write_plots) else None,
                    "subdomains_png": str(subdomains_png) if (display_col is None and write_plots) else None,
                    "display_outputs": display_outputs,
                    "meta_json": str(meta_json),
                },
            }
            with meta_json.open("w") as fh:
                json.dump(meta, fh, indent=2)
            rows.append({"batch": batch, "status": "done", **meta})
            if profile:
                print(
                    f"[DeCoMap] group={batch} done in {res.elapsed_seconds:.2f}s | "
                    f"K={res.best_k} | structured_cells={res.n_structured_cells}/{res.n_total_cells}",
                    flush=True,
                )
        except Exception as exc:
            rows.append({"batch": batch, "status": "failed", "error": repr(exc)})
            if profile:
                print(f"[DeCoMap] group={batch} FAILED: {exc!r}", flush=True)

    summary_csv_all = Path(f"{out_prefix}_sampled_batches.csv")
    summary_json_all = Path(f"{out_prefix}_sampled_batches_meta.json")
    pd.DataFrame(rows).to_csv(summary_csv_all, index=False)
    with summary_json_all.open("w") as fh:
        json.dump({"pkl": str(pkl_path), "out_prefix": str(out_prefix), "batches": rows}, fh, indent=2)

    return {"summary_csv": str(summary_csv_all), "summary_json": str(summary_json_all), "batches": rows}
