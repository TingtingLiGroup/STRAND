from __future__ import annotations

import pickle
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from tools.api.colocalization import (
    DEFAULT_PREFILTER_KWARGS,
    _apply_cell_sampling_to_bundle,
    _attach_groupby_from_coordinates,
    _cleanup_previous_optional_outputs,
    _safe_group_name,
    _save_intermediate_outputs,
    _save_prefilter_tables,
    _subset_bundle_by_cells,
)
from tools.api.compute_all import (
    _limit_cells_in_bundle,
    _prefilter_bundle_before_features,
    _print_prefilter_stats,
)
from tools.engines.coloc_region_annotation import (
    annotate_colocalized_pairs_regions,
)
from tools.instant_optimized.adapter import (
    run_optimized_colocalization_from_bundle,
)
from tools.instant_optimized.visualization import (
    plot_optimized_coloc_outputs,
)
from tools.utils.timing import TimerReport, timed


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
    np.save(
        out_dir / "instant_expected_coloc.npy",
        result["expected_coloc"],
    )
    np.save(out_dir / "instant_gene_counts.npy", result["gene_counts"])
    if save_pp_pvals:
        result["pp_results"].save_dense_npy(
            out_dir / "instant_pp_pvals.npy"
        )


def _run_single_bundle(
    bundle: dict,
    *,
    pkl_path: Path,
    out_dir: Path,
    distance_threshold: float,
    pp_alpha: float,
    cpb_alpha: float,
    min_genecount: int,
    threads: int,
    use_3d: bool,
    precision_mode: str,
    profile: bool,
    filter_stats: dict,
    do_prefilter: bool,
    save_matrices: bool,
    save_pp_pvals: bool,
    keep_tmp: bool,
    save_intermediate: bool,
    save_qc_details: bool,
    plot: bool,
    viz_top_n_genes: int | None,
    viz_max_edges: int,
    viz_max_cells: int,
    viz_top_pairs: int,
    viz_cell_selection: str,
    plot_gene_1: str | None,
    plot_gene_2: str | None,
    region_annotation: bool,
    region_max_pairs: int | None,
    region_threshold_mode: str,
    region_workers: int,
) -> dict:
    report = TimerReport() if profile else None
    out_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = out_dir / "_tmp_instant_coloc"
    if not keep_tmp and temp_dir.exists():
        shutil.rmtree(temp_dir)

    with timed(
        "COLOC_OPT:run_sparse_pp_cpb",
        report,
        print_each=profile,
    ):
        result = run_optimized_colocalization_from_bundle(
            bundle,
            distance_threshold=distance_threshold,
            pp_alpha=pp_alpha,
            cpb_alpha=cpb_alpha,
            min_genecount=min_genecount,
            threads=threads,
            use_3d=use_3d,
            precision_mode=precision_mode,
            profile=profile,
        )

    if keep_tmp:
        temp_dir.mkdir(parents=True, exist_ok=True)
        result["instant_input_df"].to_csv(
            temp_dir / "instant_input.csv",
            index=False,
        )

    with timed("COLOC_OPT:write_outputs", report, print_each=profile):
        pairs = result["significant_pairs"].copy()
        if "cpb_pvalue" in pairs.columns:
            pairs["cpb_pvalue_raw"] = pd.to_numeric(
                pairs["cpb_pvalue"],
                errors="coerce",
            )
            pairs["cpb_pvalue"] = (
                pairs["cpb_pvalue_raw"]
                .fillna(1.0)
                .clip(lower=0.0, upper=1.0)
            )
            pairs = pairs.sort_values(
                "cpb_pvalue_raw",
                ascending=True,
            ).reset_index(drop=True)
        result["significant_pairs"] = pairs

        pairs_path = out_dir / "instant_significant_pairs.csv"
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

    region_path = None
    region_df = None
    nuclear_boundary = bundle.get("nuclear_boundary")
    if (
        region_annotation
        and isinstance(nuclear_boundary, dict)
        and len(nuclear_boundary) > 0
    ):
        with timed(
            "COLOC_OPT:region_annotation",
            report,
            print_each=profile,
        ):
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
            except ValueError as exc:
                if "No valid cell/nuclear geometries" not in str(exc):
                    raise
                print(
                    "[instant-optimized] region annotation skipped: "
                    f"{exc}"
                )

    viz_paths = {}
    if plot:
        with timed(
            "COLOC_OPT:plot_outputs",
            report,
            print_each=profile,
        ):
            viz_paths = plot_optimized_coloc_outputs(
                out_dir=out_dir,
                pkl_path=pkl_path,
                gene_list=result["gene_list"],
                instant_input_df=result["instant_input_df"],
                significant_pairs=result["significant_pairs"],
                region_annotated_pairs=region_df,
                cpb_pvals=result["cpb_pvals"],
                expected_coloc=result["expected_coloc"],
                pp_results=result["pp_results"],
                top_n_genes=viz_top_n_genes,
                max_edges=viz_max_edges,
                max_cells=viz_max_cells,
                top_pairs=viz_top_pairs,
                gene_1=plot_gene_1,
                gene_2=plot_gene_2,
                distance_threshold=distance_threshold,
                use_3d=use_3d,
                save_all_pairs=save_intermediate,
                cell_selection=viz_cell_selection,
            )

    if profile and report is not None:
        print(report.summary())

    return {
        "out_dir": out_dir,
        "pairs_path": pairs_path,
        "prefilter_summary_path": prefilter_summary_path,
        "gene_list_path": gene_list_path,
        "instant_input_path": instant_input_path,
        "region_annotated_pairs_path": region_path,
        "region_annotated_pairs": region_df,
        "significant_pairs": result["significant_pairs"],
        "filter_stats": filter_stats,
        "viz_paths": viz_paths,
        "optimized_result": result,
    }


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
    group_values: list[str] | None = None,
    skip_existing: bool = False,
) -> dict:
    """Run the isolated optimized pipeline without modifying the original API."""
    del perinuclear_width, peripheral_width
    pkl_path = Path(pkl_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _cleanup_previous_optional_outputs(out_dir)
    report = TimerReport() if profile else None

    with timed("COLOC_OPT:load_bundle", report, print_each=profile):
        with pkl_path.open("rb") as handle:
            bundle = pickle.load(handle)

    filter_stats = {}
    if do_prefilter:
        kwargs = dict(DEFAULT_PREFILTER_KWARGS)
        if prefilter_kwargs is not None:
            kwargs.update(prefilter_kwargs)
        with timed(
            "COLOC_OPT:prefilter_bundle",
            report,
            print_each=profile,
        ):
            bundle, filter_stats = _prefilter_bundle_before_features(
                bundle,
                **kwargs,
            )
        if profile:
            _print_prefilter_stats(filter_stats)

    if max_cells is not None:
        bundle, _ = _limit_cells_in_bundle(bundle, max_cells=max_cells)

    cell_sampling_summary_path = None
    sampled_cells_path = None
    cell_sampling_summary = None
    if sample_cells is not None or sample_cell_frac is not None:
        (
            bundle,
            cell_sampling_summary_path,
            sampled_cells_path,
            cell_sampling_summary,
        ) = _apply_cell_sampling_to_bundle(
            bundle,
            sample_cells=sample_cells,
            sample_cell_frac=sample_cell_frac,
            random_state=sample_random_state,
            stratify_by=sample_stratify_by,
            out_dir=out_dir,
        )

    common_kwargs = {
        "pkl_path": pkl_path,
        "distance_threshold": distance_threshold,
        "pp_alpha": pp_alpha,
        "cpb_alpha": cpb_alpha,
        "min_genecount": min_genecount,
        "threads": threads,
        "use_3d": use_3d,
        "precision_mode": precision_mode,
        "profile": profile,
        "save_matrices": save_matrices,
        "save_pp_pvals": save_pp_pvals,
        "keep_tmp": keep_tmp,
        "save_intermediate": save_intermediate,
        "save_qc_details": save_qc_details,
        "plot": plot,
        "viz_top_n_genes": viz_top_n_genes,
        "viz_max_edges": viz_max_edges,
        "viz_max_cells": viz_max_cells,
        "viz_top_pairs": viz_top_pairs,
        "viz_cell_selection": viz_cell_selection,
        "plot_gene_1": plot_gene_1,
        "plot_gene_2": plot_gene_2,
        "region_annotation": region_annotation,
        "region_max_pairs": region_max_pairs,
        "region_threshold_mode": region_threshold_mode,
        "region_workers": region_workers,
    }

    if groupby is None:
        result = _run_single_bundle(
            bundle,
            out_dir=out_dir,
            filter_stats=filter_stats,
            do_prefilter=do_prefilter,
            **common_kwargs,
        )
        result.update(
            {
                "cell_sampling_summary_path": cell_sampling_summary_path,
                "sampled_cells_path": sampled_cells_path,
                "cell_sampling_summary": cell_sampling_summary,
            }
        )
        return result

    bundle = _attach_groupby_from_coordinates(bundle, groupby)
    data_df = bundle["data_df"].copy()
    data_df["cell"] = data_df["cell"].astype(str)
    group_cells = (
        data_df[["cell", groupby]]
        .dropna(subset=[groupby])
        .drop_duplicates()
        .groupby(groupby)["cell"]
        .apply(lambda values: sorted(set(values.astype(str))))
    )
    if group_values is not None:
        requested = {str(value) for value in group_values}
        group_cells = group_cells[
            group_cells.index.astype(str).isin(requested)
        ]

    group_names = {
        str(group_value): _safe_group_name(group_value)
        for group_value in group_cells.index
    }
    safe_name_to_group = {}
    for group_value, safe_name in group_names.items():
        previous_group = safe_name_to_group.get(safe_name)
        if previous_group is not None and previous_group != group_value:
            raise ValueError(
                "Different group values map to the same output directory: "
                f"{previous_group!r} and {group_value!r} -> {safe_name!r}."
            )
        safe_name_to_group[safe_name] = group_value

    group_results = {}
    summary_rows = []
    for group_value, cells in group_cells.items():
        group_name = group_names[str(group_value)]
        group_out_dir = out_dir / group_name
        group_out_dir.mkdir(parents=True, exist_ok=True)
        pairs_path = group_out_dir / "instant_significant_pairs.csv"
        if skip_existing and pairs_path.exists():
            try:
                num_pairs = len(pd.read_csv(pairs_path))
            except Exception:
                num_pairs = -1
            group_results[str(group_value)] = {
                "out_dir": group_out_dir,
                "pairs_path": pairs_path,
                "skipped_existing": True,
            }
            summary_rows.append(
                {
                    "groupby": groupby,
                    "group": group_value,
                    "n_cells": len(cells),
                    "n_significant_pairs": num_pairs,
                    "out_dir": str(group_out_dir),
                    "status": "skipped_existing",
                }
            )
            continue

        _cleanup_previous_optional_outputs(group_out_dir)
        group_bundle = _subset_bundle_by_cells(bundle, set(cells))
        group_result = _run_single_bundle(
            group_bundle,
            out_dir=group_out_dir,
            filter_stats={},
            do_prefilter=False,
            **common_kwargs,
        )
        group_results[str(group_value)] = group_result
        summary_rows.append(
            {
                "groupby": groupby,
                "group": group_value,
                "n_cells": len(cells),
                "n_significant_pairs": len(
                    group_result["significant_pairs"]
                ),
                "out_dir": str(group_out_dir),
                "status": "done",
            }
        )

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
