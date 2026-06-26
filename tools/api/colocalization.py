# tools/api/colocalization.py

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from tools.api.compute_all import (
    _prefilter_bundle_before_features,
    _print_prefilter_stats,
    _limit_cells_in_bundle,
)
from tools.engines.instant_adapter import run_instant_colocalization_from_bundle
from tools.engines.coloc_visualization import plot_coloc_outputs
from tools.engines.coloc_region_annotation import annotate_colocalized_pairs_regions
from tools.utils.timing import timed, TimerReport


DEFAULT_PREFILTER_KWARGS = {
    "filter_cells_by_nc_ratio": True,
    "nc_ratio_q_low": 0.025,
    "nc_ratio_q_high": 0.975,
    "nc_ratio_mean_low": 0.4,
    "nc_ratio_mean_high": 0.6,
    "cellgene_filter_min_transcripts": 6,
    "gene_filter_min_cells": 10,
}


_GENERATED_OPTIONAL_FILES = [
    "instant_gene_list.csv",
    "instant_input_after_prefilter.csv",
    "instant_all_pairs.csv",
    "instant_cpb_pvals.npy",
    "instant_expected_coloc.npy",
    "instant_gene_counts.npy",
    "instant_pp_pvals.npy",
    "prefilter_nc_ratio_cells.csv",
    "prefilter_cellgene_counts.csv",
    "prefilter_gene_support.csv",
    "instant_region_annotated_pairs.csv",
    "cell_sampling_summary.csv",
    "sampled_cells.txt",
]


def _cleanup_previous_optional_outputs(out_dir: Path) -> None:
    """Remove stale optional outputs from older runs so default output stays clean."""
    for name in _GENERATED_OPTIONAL_FILES:
        p = out_dir / name
        if p.exists():
            try:
                p.unlink()
            except Exception:
                pass
    for p in out_dir.glob("viz_*.png"):
        try:
            p.unlink()
        except Exception:
            pass


def _filter_stats_to_summary_df(filter_stats: dict, *, do_prefilter: bool = True) -> pd.DataFrame:
    """Convert prefilter stats to a compact summary table."""
    rows = []

    if not do_prefilter:
        return pd.DataFrame([{"stage": "prefilter", "metric": "enabled", "value": False}])

    if not filter_stats:
        return pd.DataFrame([{"stage": "prefilter", "metric": "stats_available", "value": False}])

    for stage in [
        "raw",
        "after_nc_ratio_filter",
        "after_cellgene_filter",
        "after_gene_support_filter",
        "final",
    ]:
        d = filter_stats.get(stage)
        if isinstance(d, dict):
            row = {"stage": stage}
            row.update(d)
            rows.append(row)

    extra_keys = [
        "n_cells_after_nc_ratio_filter",
        "nc_ratio_mean_after_filter",
        "nc_ratio_mean_status",
        "cellgene_filter_min_transcripts",
        "n_cellgenes_before_cellgene_filter",
        "n_cellgenes_after_cellgene_filter",
        "n_cellgenes_removed_by_cellgene_filter",
        "gene_filter_min_cells",
        "n_genes_before_gene_support_filter",
        "n_genes_after_gene_support_filter",
        "n_genes_removed_by_gene_support_filter",
        "n_common_cells",
    ]

    for key in extra_keys:
        if key in filter_stats:
            rows.append({"stage": "summary", "metric": key, "value": filter_stats[key]})

    if not rows:
        for k, v in filter_stats.items():
            if isinstance(v, (int, float, str, bool)):
                rows.append({"stage": "summary", "metric": k, "value": v})

    return pd.DataFrame(rows)


def _save_prefilter_tables(
    *,
    out_dir: Path,
    filter_stats: dict,
    do_prefilter: bool,
    save_details: bool = False,
) -> Path:
    """Save compact summary by default, detailed QC only when requested."""
    summary_df = _filter_stats_to_summary_df(filter_stats, do_prefilter=do_prefilter)
    summary_path = out_dir / "prefilter_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    if not save_details:
        return summary_path

    detail_map = {
        "nc_ratio_df": "prefilter_nc_ratio_cells.csv",
        "cellgene_counts_df": "prefilter_cellgene_counts.csv",
        "gene_support_df": "prefilter_gene_support.csv",
    }
    for key, filename in detail_map.items():
        df = filter_stats.get(key)
        if isinstance(df, pd.DataFrame) and not df.empty:
            df.to_csv(out_dir / filename, index=False)

    return summary_path


def _save_matrix_outputs(
    *,
    out_dir: Path,
    result: dict,
    save_matrices: bool,
    save_pp_pvals: bool,
) -> None:
    if not save_matrices:
        return

    np.save(out_dir / "instant_cpb_pvals.npy", result["cpb_pvals"])
    np.save(out_dir / "instant_expected_coloc.npy", result["expected_coloc"])
    np.save(out_dir / "instant_gene_counts.npy", result["gene_counts"])

    if save_pp_pvals:
        np.save(out_dir / "instant_pp_pvals.npy", result["pp_pvals"])

def _save_intermediate_outputs(*, out_dir: Path, result: dict, save_intermediate: bool) -> tuple[Path | None, Path | None]:
    if not save_intermediate:
        return None, None

    gene_list_path = out_dir / "instant_gene_list.csv"
    pd.Series(result["gene_list"], name="gene").to_csv(gene_list_path, index=False)

    instant_input_path = out_dir / "instant_input_after_prefilter.csv"
    result["instant_input_df"].to_csv(instant_input_path, index=False)

    return gene_list_path, instant_input_path


def _safe_group_name(x) -> str:
    import re
    x = str(x)
    x = re.sub(r"[^\w\-.]+", "_", x)
    return x.strip("_") or "unknown"


def _attach_groupby_from_coordinates(bundle: dict, groupby: str) -> dict:
    """Ensure data_df contains groupby. If absent, merge it from coordinates by cell."""
    data_df = bundle["data_df"].copy()
    if groupby in data_df.columns:
        new_bundle = dict(bundle)
        new_bundle["data_df"] = data_df
        return new_bundle

    coords = bundle.get("coordinates")
    if coords is None or not isinstance(coords, pd.DataFrame):
        raise ValueError(f"--groupby {groupby!r} not found in data_df, and bundle['coordinates'] is missing.")
    if groupby not in coords.columns:
        raise ValueError(f"--groupby {groupby!r} not found in data_df or coordinates.")
    if "cell" not in data_df.columns:
        raise ValueError("data_df must contain column 'cell' for --groupby merge.")
    if "cell" not in coords.columns:
        raise ValueError("coordinates must contain column 'cell' for --groupby merge.")

    mapper = coords[["cell", groupby]].drop_duplicates("cell").copy()
    mapper["cell"] = mapper["cell"].astype(str)
    data_df["cell"] = data_df["cell"].astype(str)
    data_df = data_df.merge(mapper, on="cell", how="left")

    missing_cells = data_df.loc[data_df[groupby].isna(), "cell"].nunique()
    if missing_cells:
        print(f"[subcellfeat-coloc] warning: {missing_cells} cells have no {groupby!r} annotation after coordinates merge.")

    new_bundle = dict(bundle)
    new_bundle["data_df"] = data_df
    return new_bundle


def _subset_bundle_by_cells(bundle: dict, cells: set[str]) -> dict:
    """Create a sub-bundle containing only selected cells."""
    cells = {str(c) for c in cells}
    out = dict(bundle)

    data_df = bundle["data_df"].copy()
    data_df["cell"] = data_df["cell"].astype(str)
    out["data_df"] = data_df[data_df["cell"].isin(cells)].copy()

    if isinstance(bundle.get("coordinates"), pd.DataFrame) and "cell" in bundle["coordinates"].columns:
        coords = bundle["coordinates"].copy()
        coords["cell"] = coords["cell"].astype(str)
        out["coordinates"] = coords[coords["cell"].isin(cells)].copy()

    for key in ["cell_boundary", "nuclear_boundary"]:
        if isinstance(bundle.get(key), dict):
            out[key] = {k: v for k, v in bundle[key].items() if str(k) in cells}

    return out


def _attach_sampling_strata_from_coordinates(bundle: dict, stratify_by: str) -> pd.DataFrame:
    """Return a unique cell table with an optional stratification column.

    The stratification column may be present in data_df or in coordinates.  This
    helper does not change the bundle; it only builds metadata used for sampling.
    """
    data_df = bundle.get("data_df")
    if not isinstance(data_df, pd.DataFrame) or "cell" not in data_df.columns:
        raise ValueError("bundle['data_df'] must contain column 'cell' for cell sampling.")

    cells = pd.DataFrame({"cell": data_df["cell"].astype(str).unique()})

    if stratify_by in data_df.columns:
        mapper = (
            data_df[["cell", stratify_by]]
            .copy()
            .assign(cell=lambda x: x["cell"].astype(str))
            .dropna(subset=[stratify_by])
            .drop_duplicates("cell")
        )
        return cells.merge(mapper, on="cell", how="left")

    coords = bundle.get("coordinates")
    if isinstance(coords, pd.DataFrame) and "cell" in coords.columns and stratify_by in coords.columns:
        mapper = (
            coords[["cell", stratify_by]]
            .copy()
            .assign(cell=lambda x: x["cell"].astype(str))
            .dropna(subset=[stratify_by])
            .drop_duplicates("cell")
        )
        return cells.merge(mapper, on="cell", how="left")

    raise ValueError(f"--sample-stratify-by {stratify_by!r} not found in data_df or coordinates.")


def _allocate_stratified_counts(group_sizes: pd.Series, target_n: int) -> dict:
    """Allocate an exact total sample size across strata proportionally."""
    group_sizes = group_sizes.astype(int)
    total = int(group_sizes.sum())
    target_n = int(min(max(target_n, 0), total))
    if target_n <= 0 or total <= 0:
        return {k: 0 for k in group_sizes.index}

    raw = group_sizes / total * target_n
    base = np.floor(raw).astype(int)

    # Give at least one sample to non-empty strata when possible.
    non_empty = group_sizes[group_sizes > 0].index.tolist()
    if target_n >= len(non_empty):
        for k in non_empty:
            base.loc[k] = max(int(base.loc[k]), 1)

    # Do not exceed group size.
    base = pd.Series({k: min(int(base.loc[k]), int(group_sizes.loc[k])) for k in group_sizes.index})

    current = int(base.sum())
    remainder = (raw - np.floor(raw)).sort_values(ascending=False)

    # Add samples until exact target is reached.
    while current < target_n:
        progressed = False
        for k in remainder.index:
            if base.loc[k] < group_sizes.loc[k]:
                base.loc[k] += 1
                current += 1
                progressed = True
                if current >= target_n:
                    break
        if not progressed:
            break

    # Remove samples if minimum-one rule overshot the target.
    while current > target_n:
        removable = [k for k in base.index if base.loc[k] > 0]
        if not removable:
            break
        # Prefer removing from the largest currently allocated strata.
        k = max(removable, key=lambda x: base.loc[x])
        base.loc[k] -= 1
        current -= 1

    return {k: int(v) for k, v in base.items()}


def _apply_cell_sampling_to_bundle(
    bundle: dict,
    *,
    sample_cells: int | None = None,
    sample_cell_frac: float | None = None,
    random_state: int = 42,
    stratify_by: str | None = None,
    out_dir: Path | None = None,
) -> tuple[dict, Path | None, Path | None, pd.DataFrame | None]:
    """Randomly sample cells before InSTAnT PP/CPB while keeping the full gene panel.

    This step does not alter InSTAnT PP/CPB statistics. It only changes the input
    cell set, which reduces the cell x gene x gene memory burden for large data.
    """
    if sample_cells is None and sample_cell_frac is None:
        return bundle, None, None, None
    if sample_cells is not None and sample_cell_frac is not None:
        raise ValueError("Use only one of --sample-cells or --sample-cell-frac, not both.")

    data_df = bundle.get("data_df")
    if not isinstance(data_df, pd.DataFrame) or "cell" not in data_df.columns:
        raise ValueError("bundle['data_df'] must contain column 'cell' for cell sampling.")

    data_df = data_df.copy()
    data_df["cell"] = data_df["cell"].astype(str)
    cells_before = sorted(data_df["cell"].unique().tolist())
    n_cells_before = len(cells_before)
    rows_before = int(len(data_df))
    genes_before = int(data_df["gene"].nunique()) if "gene" in data_df.columns else None

    if n_cells_before == 0:
        raise ValueError("Cell sampling cannot run because data_df contains zero cells.")

    rng = np.random.default_rng(int(random_state))

    if sample_cells is not None:
        target_n = int(sample_cells)
        if target_n <= 0:
            raise ValueError("--sample-cells must be positive.")
        target_n = min(target_n, n_cells_before)
        sampling_method = f"sample_cells:{target_n}"
    else:
        frac = float(sample_cell_frac)
        if not (0 < frac <= 1):
            raise ValueError("--sample-cell-frac must be in the interval (0, 1].")
        target_n = int(round(n_cells_before * frac))
        target_n = max(1, min(target_n, n_cells_before))
        sampling_method = f"sample_cell_frac:{frac}"

    if stratify_by is None:
        selected_cells = sorted(rng.choice(cells_before, size=target_n, replace=False).astype(str).tolist())
    else:
        cell_meta = _attach_sampling_strata_from_coordinates(bundle, stratify_by)
        cell_meta["cell"] = cell_meta["cell"].astype(str)
        cell_meta[stratify_by] = cell_meta[stratify_by].fillna("__missing__").astype(str)
        group_sizes = cell_meta.groupby(stratify_by)["cell"].nunique()
        allocations = _allocate_stratified_counts(group_sizes, target_n)
        picked = []
        for group_value, n_take in allocations.items():
            if n_take <= 0:
                continue
            group_cells = sorted(cell_meta.loc[cell_meta[stratify_by] == group_value, "cell"].unique().tolist())
            picked.extend(rng.choice(group_cells, size=min(n_take, len(group_cells)), replace=False).astype(str).tolist())
        selected_cells = sorted(set(picked))
        # Exact-size correction in rare rounding/duplicate edge cases.
        if len(selected_cells) < target_n:
            remaining = sorted(set(cells_before) - set(selected_cells))
            add_n = min(target_n - len(selected_cells), len(remaining))
            if add_n > 0:
                selected_cells.extend(rng.choice(remaining, size=add_n, replace=False).astype(str).tolist())
                selected_cells = sorted(set(selected_cells))
        elif len(selected_cells) > target_n:
            selected_cells = sorted(rng.choice(selected_cells, size=target_n, replace=False).astype(str).tolist())

    sampled_bundle = _subset_bundle_by_cells(bundle, set(selected_cells))
    sampled_df = sampled_bundle["data_df"].copy()
    sampled_df["cell"] = sampled_df["cell"].astype(str)

    n_cells_after = int(sampled_df["cell"].nunique())
    rows_after = int(len(sampled_df))
    genes_after = int(sampled_df["gene"].nunique()) if "gene" in sampled_df.columns else None

    summary = pd.DataFrame([
        {
            "sampling_method": sampling_method,
            "sample_cells_requested": sample_cells,
            "sample_cell_frac_requested": sample_cell_frac,
            "sample_random_state": int(random_state),
            "sample_stratify_by": stratify_by,
            "n_cells_before_sampling": n_cells_before,
            "n_cells_after_sampling": n_cells_after,
            "n_rows_before_sampling": rows_before,
            "n_rows_after_sampling": rows_after,
            "n_genes_before_sampling": genes_before,
            "n_genes_after_sampling": genes_after,
        }
    ])

    summary_path = None
    cells_path = None
    if out_dir is not None:
        summary_path = Path(out_dir) / "cell_sampling_summary.csv"
        cells_path = Path(out_dir) / "sampled_cells.txt"
        summary.to_csv(summary_path, index=False)
        with open(cells_path, "w", encoding="utf-8") as f:
            for c in selected_cells:
                f.write(str(c) + "\n")

    print(
        "[subcellfeat-coloc] cell sampling: "
        f"{n_cells_before} -> {n_cells_after} cells, "
        f"{rows_before} -> {rows_after} transcripts, "
        f"genes {genes_before} -> {genes_after}, "
        f"method={sampling_method}, stratify_by={stratify_by}, random_state={random_state}"
    )

    return sampled_bundle, summary_path, cells_path, summary


def run_colocalization_from_pkl(
    pkl_path: str | Path,
    *,
    out_dir: str | Path,
    distance_threshold: float = 4.0,
    pp_alpha: float = 0.001,
    cpb_alpha: float = 0.0001,
    min_genecount: int = 20,
    threads: int = 8,
    use_3d: bool = True,
    precision_mode: str = "high",
    profile: bool = False,
    max_cells: int | None = None,
    do_prefilter: bool = True,
    prefilter_kwargs: dict | None = None,
    save_matrices: bool = False,
    save_pp_pvals: bool = False,
    keep_tmp: bool = False,
    save_intermediate: bool = False,
    save_qc_details: bool = False,
    plot: bool = True,
    viz_top_n_genes: int | None = 80,
    viz_max_edges: int = 80,
    viz_max_cells: int = 6,
    viz_top_pairs: int = 10,
    plot_gene_1: str | None = None,
    plot_gene_2: str | None = None,
    viz_cell_selection: str = "pp",
    region_annotation: bool = True,
    perinuclear_width: float | None = None,
    peripheral_width: float | None = None,
    region_max_pairs: int | None = None,
    region_threshold_mode: str = "cell_scaled",
    region_workers: int = 1,
    sample_cells: int | None = None,
    sample_cell_frac: float | None = None,
    sample_random_state: int = 42,
    sample_stratify_by: str | None = None,
    groupby: str | None = None,
) -> dict:
    """
    Full colocalization pipeline:
        PKL bundle -> optional toolbox prefilter -> InSTAnT PP/CPB
        -> significant global d-colocalized gene pairs -> visualizations.

    Default outputs are intentionally clean:
        - instant_significant_pairs.csv
        - prefilter_summary.csv
        - viz_*.png if plot=True

    Optional debug files are saved only with:
        --save-intermediate, --save-qc-details, --save-matrices, --save-pp-pvals
    """
    rep = TimerReport() if profile else None

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _cleanup_previous_optional_outputs(out_dir)

    pkl_path = Path(pkl_path)

    with timed("COLOC:load_bundle", rep, print_each=profile):
        with open(pkl_path, "rb") as f:
            bundle = pickle.load(f)

    filter_stats = {}
    cell_sampling_summary_path = None
    sampled_cells_path = None
    cell_sampling_summary = None

    if do_prefilter:
        kwargs = dict(DEFAULT_PREFILTER_KWARGS)
        if prefilter_kwargs is not None:
            kwargs.update(prefilter_kwargs)

        with timed("COLOC:prefilter_bundle", rep, print_each=profile):
            bundle, filter_stats = _prefilter_bundle_before_features(bundle, **kwargs)

        if profile:
            _print_prefilter_stats(filter_stats)

    if max_cells is not None:
        with timed(f"COLOC:limit_cells({max_cells})", rep, print_each=profile):
            bundle, cells = _limit_cells_in_bundle(bundle, max_cells=max_cells)
        if profile:
            print(f"[colocalization] kept first {len(cells)} cells")

    # Optional scale-control step for large panels. This is applied BEFORE
    # InSTAnT and only changes the input cell set; it keeps the full gene panel.
    if sample_cells is not None or sample_cell_frac is not None:
        with timed("COLOC:cell_sampling", rep, print_each=profile):
            bundle, cell_sampling_summary_path, sampled_cells_path, cell_sampling_summary = _apply_cell_sampling_to_bundle(
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
            .apply(lambda x: sorted(set(x.astype(str))))
        )
        group_results = {}
        summary_rows = []
        for group_value, cells in group_cells.items():
            group_name = _safe_group_name(group_value)
            sub_out = out_dir / group_name
            sub_bundle = _subset_bundle_by_cells(bundle, set(cells))
            tmp_pkl = sub_out / "_tmp_group_bundle.pkl"
            sub_out.mkdir(parents=True, exist_ok=True)
            with open(tmp_pkl, "wb") as f:
                pickle.dump(sub_bundle, f)
            try:
                sub_result = run_colocalization_from_pkl(
                    tmp_pkl,
                    out_dir=sub_out,
                    distance_threshold=distance_threshold,
                    pp_alpha=pp_alpha,
                    cpb_alpha=cpb_alpha,
                    min_genecount=min_genecount,
                    threads=threads,
                    use_3d=use_3d,
                    precision_mode=precision_mode,
                    profile=profile,
                    max_cells=None,
                    do_prefilter=False,
                    prefilter_kwargs=None,
                    save_matrices=save_matrices,
                    save_pp_pvals=save_pp_pvals,
                    keep_tmp=keep_tmp,
                    save_intermediate=save_intermediate,
                    save_qc_details=save_qc_details,
                    plot=plot,
                    viz_top_n_genes=viz_top_n_genes,
                    viz_max_edges=viz_max_edges,
                    viz_max_cells=viz_max_cells,
                    viz_top_pairs=viz_top_pairs,
                    plot_gene_1=plot_gene_1,
                    plot_gene_2=plot_gene_2,
                    viz_cell_selection=viz_cell_selection,
                    region_annotation=region_annotation,
                    perinuclear_width=perinuclear_width,
                    peripheral_width=peripheral_width,
                    region_max_pairs=region_max_pairs,
                    region_threshold_mode=region_threshold_mode,
                    region_workers=region_workers,
                    sample_cells=None,
                    sample_cell_frac=None,
                    sample_random_state=sample_random_state,
                    sample_stratify_by=None,
                    groupby=None,
                )
            finally:
                if tmp_pkl.exists():
                    try:
                        tmp_pkl.unlink()
                    except Exception:
                        pass
            group_results[str(group_value)] = sub_result
            n_pairs = len(sub_result.get("significant_pairs", []))
            summary_rows.append({"groupby": groupby, "group": group_value, "n_cells": len(cells), "n_significant_pairs": n_pairs, "out_dir": str(sub_out)})

        summary = pd.DataFrame(summary_rows)
        summary_path = out_dir / f"groupby_{_safe_group_name(groupby)}_summary.csv"
        summary.to_csv(summary_path, index=False)
        return {
            "out_dir": out_dir,
            "groupby": groupby,
            "groupby_summary_path": summary_path,
            "groupby_results": group_results,
            "pairs_path": None,
            "prefilter_summary_path": None,
            "gene_list_path": None,
            "instant_input_path": None,
            "region_annotated_pairs_path": None,
            "cell_sampling_summary_path": cell_sampling_summary_path,
            "sampled_cells_path": sampled_cells_path,
            "cell_sampling_summary": cell_sampling_summary,
            "significant_pairs": pd.DataFrame(),
            "filter_stats": filter_stats,
            "viz_paths": {},
        }


    with timed("COLOC:run_instant_pp_cpb", rep, print_each=profile):
        result = run_instant_colocalization_from_bundle(
            bundle,
            distance_threshold=distance_threshold,
            pp_alpha=pp_alpha,
            cpb_alpha=cpb_alpha,
            min_genecount=min_genecount,
            threads=threads,
            use_3d=use_3d,
            precision_mode=precision_mode,
            tmp_dir=out_dir / "_tmp_instant_coloc",
            keep_tmp=keep_tmp,
            region_annotation=False,
            region_max_pairs=None,
        )

    with timed("COLOC:write_outputs", rep, print_each=profile):
        # pairs_path = out_dir / "instant_significant_pairs.csv"
        # if "cpb_pvalue" in result["significant_pairs"].columns:
        #     result["significant_pairs"]["cpb_pvalue"] = (
        #         pd.to_numeric(result["significant_pairs"]["cpb_pvalue"], errors="coerce")
        #         .fillna(1.0)
        #         .clip(lower=0.0, upper=1.0)
        #     )
        # result["significant_pairs"].to_csv(pairs_path, index=False)
        pairs_path = out_dir / "instant_significant_pairs.csv"
        pairs = result["significant_pairs"].copy()

        if "cpb_pvalue" in pairs.columns:
            # Keep original InSTAnT output for sorting/debugging.
            # Tiny negative values may appear because of floating-point approximation.
            pairs["cpb_pvalue_raw"] = pd.to_numeric(
                pairs["cpb_pvalue"],
                errors="coerce",
            )
            # Formal p-value for output/display must be within [0, 1].
            pairs["cpb_pvalue"] = (
                pairs["cpb_pvalue_raw"]
                .fillna(1.0)
                .clip(lower=0.0, upper=1.0)
            )
            # Preserve original ranking information.
            # This avoids ties caused by clipping many tiny negative values to 0.
            pairs = pairs.sort_values("cpb_pvalue_raw", ascending=True).reset_index(drop=True)

        result["significant_pairs"] = pairs
        pairs.to_csv(pairs_path, index=False)

        prefilter_summary_path = _save_prefilter_tables(
            out_dir=out_dir,
            filter_stats=filter_stats,
            do_prefilter=do_prefilter,
            save_details=save_qc_details,
        )

        gene_list_path, instant_input_path = _save_intermediate_outputs(
            out_dir=out_dir,
            result=result,
            save_intermediate=save_intermediate,
        )
        _save_matrix_outputs(
            out_dir=out_dir,
            result=result,
            save_matrices=save_matrices,
            save_pp_pvals=save_pp_pvals,
        )

    # region_path = None
    # region_df = None

    # if region_annotation:
    #     with timed("COLOC:region_annotation", rep, print_each=profile):
    #         region_df = annotate_colocalized_pairs_regions(
    #             bundle=bundle,
    #             instant_input_df=result["instant_input_df"],
    #             significant_pairs=result["significant_pairs"],
    #             distance_threshold=distance_threshold,
    #             use_3d=use_3d,
    #             max_pairs=region_max_pairs,
    #         )
    #         region_path = out_dir / "instant_region_annotated_pairs.csv"
    #         region_df.to_csv(region_path, index=False)
    #         print(f"[subcellfeat-coloc] saved region annotated pairs: {region_path}")
    region_path = None
    region_df = None

    if region_annotation:
        nuclear_boundary = bundle.get("nuclear_boundary", None)

        has_nuclear_boundary = (
            nuclear_boundary is not None
            and isinstance(nuclear_boundary, dict)
            and len(nuclear_boundary) > 0
        )

        if not has_nuclear_boundary:
            print(
                "[subcellfeat-coloc] warning: region annotation skipped because "
                "nuclear_boundary is missing or empty."
            )
        else:
            with timed("COLOC:region_annotation", rep, print_each=profile):
                try:
                    region_df = annotate_colocalized_pairs_regions(
                        bundle=bundle,
                        instant_input_df=result["instant_input_df"],
                        significant_pairs=result["significant_pairs"],
                        distance_threshold=distance_threshold,
                        use_3d=use_3d,
                        max_pairs=region_max_pairs,
                        region_threshold_mode=region_threshold_mode,
                        region_workers=region_workers,
                    )

                    region_path = out_dir / "instant_region_annotated_pairs.csv"
                    region_df.to_csv(region_path, index=False)

                    print(
                        "[subcellfeat-coloc] saved region annotated pairs: "
                        f"{region_path}"
                    )

                except ValueError as e:
                    msg = str(e)
                    if "No valid cell/nuclear geometries" in msg:
                        print(
                            "[subcellfeat-coloc] warning: region annotation skipped: "
                            f"{msg}"
                        )
                        region_df = None
                        region_path = None
                    else:
                        raise

    viz_paths = {}
    if plot:
        with timed("COLOC:plot_outputs", rep, print_each=profile):
            viz_paths = plot_coloc_outputs(
                out_dir=out_dir,
                pkl_path=pkl_path,
                gene_list=result["gene_list"],
                instant_input_df=result["instant_input_df"],
                significant_pairs=result["significant_pairs"],
                region_annotated_pairs=region_df,
                cpb_pvals=result["cpb_pvals"],
                expected_coloc=result["expected_coloc"],
                pp_pvals=result.get("pp_pvals"),
                top_n_genes=viz_top_n_genes,
                max_edges=viz_max_edges,
                max_cells=viz_max_cells,
                top_pairs=viz_top_pairs,
                gene_1=plot_gene_1,
                gene_2=plot_gene_2,
                cell_selection=viz_cell_selection,
                distance_threshold=distance_threshold,
                use_3d=use_3d,
                save_all_pairs=save_intermediate,
            )
            for name, path in viz_paths.items():
                print(f"[subcellfeat-coloc] saved visualization ({name}): {path}")

    if profile and rep is not None:
        print(rep.summary())

    return {
        "out_dir": out_dir,
        "pairs_path": pairs_path,
        "prefilter_summary_path": prefilter_summary_path,
        "gene_list_path": gene_list_path,
        "instant_input_path": instant_input_path,
        "region_annotated_pairs_path": region_path,
        "region_annotated_pairs": region_df,
        "cell_sampling_summary_path": cell_sampling_summary_path,
        "sampled_cells_path": sampled_cells_path,
        "cell_sampling_summary": cell_sampling_summary,
        "significant_pairs": result["significant_pairs"],
        "filter_stats": filter_stats,
        "viz_paths": viz_paths,
    }
