from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import pandas as pd

from tools.api.compute_all import _prefilter_bundle_before_features
from tools.engines.instant_adapter import bundle_to_instant_dataframe
from tools.engines.coloc_region_annotation import annotate_colocalized_pairs_regions


DEFAULT_PREFILTER_KWARGS = {
    "filter_cells_by_nc_ratio": True,
    "nc_ratio_q_low": 0.025,
    "nc_ratio_q_high": 0.975,
    "nc_ratio_mean_low": 0.4,
    "nc_ratio_mean_high": 0.6,
    "cellgene_filter_min_transcripts": 6,
    "gene_filter_min_cells": 10,
}


def _subset_bundle_by_cells(bundle: dict, cells: set[str]) -> dict:
    cells = {str(c) for c in cells}
    out = dict(bundle)

    data_df = bundle["data_df"].copy()
    data_df["cell"] = data_df["cell"].astype(str)
    out["data_df"] = data_df[data_df["cell"].isin(cells)].copy()

    coords = bundle.get("coordinates")
    if isinstance(coords, pd.DataFrame) and "cell" in coords.columns:
        coords = coords.copy()
        coords["cell"] = coords["cell"].astype(str)
        out["coordinates"] = coords[coords["cell"].isin(cells)].copy()

    for key in ["cell_boundary", "nuclear_boundary"]:
        if isinstance(bundle.get(key), dict):
            out[key] = {k: v for k, v in bundle[key].items() if str(k) in cells}

    return out


def _read_cells(path: str | Path) -> set[str]:
    p = Path(path)
    cells = set()
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            x = line.strip()
            if x:
                cells.add(str(x))
    if not cells:
        raise ValueError(f"No cells found in sampled-cells file: {p}")
    return cells


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_region_annotation_only.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Only run enrichment-based region annotation from an existing instant_significant_pairs.csv.",
    )
    p.add_argument("--pkl", required=True, help="Original PKL bundle used for colocalization.")
    p.add_argument("--pairs", required=True, help="Existing instant_significant_pairs.csv from completed PP/CPB.")
    p.add_argument("--out", required=True, help="Output CSV path for instant_region_annotated_pairs.csv.")
    p.add_argument("--sampled-cells", default=None, help="sampled_cells.txt from the PP/CPB run. Required if first step used --sample-cells/--sample-cell-frac.")
    p.add_argument("--instant-input", default=None, help="Optional saved instant_input_after_prefilter.csv. If provided, skip pkl-to-InSTAnT conversion for coordinates but still use pkl boundaries.")

    p.add_argument("--distance", type=float, default=4.0, help="Same distance threshold used in PP/CPB.")
    p.add_argument("--use-2d", action="store_true", help="Use 2D distance. Default is 3D if absZ/z exists.")
    p.add_argument("--region-threshold-mode", choices=["absolute", "cell_scaled"], default="cell_scaled")
    p.add_argument("--region-workers", type=int, default=1, help="Workers for region annotation only. 1 is safest.")
    p.add_argument("--region-max-pairs", type=int, default=None, help="Annotate only the first N significant pairs for testing.")

    p.add_argument("--no-prefilter", action="store_true", help="Do not reapply toolbox prefilter before region annotation.")
    p.add_argument("--nc-ratio-q-low", type=float, default=0.025)
    p.add_argument("--nc-ratio-q-high", type=float, default=0.975)
    p.add_argument("--nc-ratio-mean-low", type=float, default=0.4)
    p.add_argument("--nc-ratio-mean-high", type=float, default=0.6)
    p.add_argument("--cellgene-filter-min-transcripts", type=int, default=6)
    p.add_argument("--gene-filter-min-cells", type=int, default=10)
    return p


def main() -> int:
    args = build_parser().parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[region-only] loading PKL: {args.pkl}")
    with open(args.pkl, "rb") as f:
        bundle = pickle.load(f)

    if not args.no_prefilter:
        kwargs = dict(DEFAULT_PREFILTER_KWARGS)
        kwargs.update({
            "nc_ratio_q_low": args.nc_ratio_q_low,
            "nc_ratio_q_high": args.nc_ratio_q_high,
            "nc_ratio_mean_low": args.nc_ratio_mean_low,
            "nc_ratio_mean_high": args.nc_ratio_mean_high,
            "cellgene_filter_min_transcripts": args.cellgene_filter_min_transcripts,
            "gene_filter_min_cells": args.gene_filter_min_cells,
        })
        print(f"[region-only] reapplying prefilter: {kwargs}")
        bundle, _stats = _prefilter_bundle_before_features(bundle, **kwargs)
    else:
        print("[region-only] prefilter disabled by --no-prefilter")

    if args.sampled_cells:
        cells = _read_cells(args.sampled_cells)
        print(f"[region-only] subsetting to sampled cells: {len(cells)} cells from {args.sampled_cells}")
        bundle = _subset_bundle_by_cells(bundle, cells)

    print(f"[region-only] loading significant pairs: {args.pairs}")
    pairs = pd.read_csv(args.pairs)
    if pairs.empty:
        print("[region-only] warning: significant pairs file is empty; writing empty output.")
        pairs.to_csv(out_path, index=False)
        return 0

    if args.instant_input:
        print(f"[region-only] using saved instant input: {args.instant_input}")
        instant_input_df = pd.read_csv(args.instant_input)
        if args.sampled_cells and "uID" in instant_input_df.columns:
            cells = _read_cells(args.sampled_cells)
            instant_input_df["uID"] = instant_input_df["uID"].astype(str)
            instant_input_df = instant_input_df[instant_input_df["uID"].isin(cells)].copy()
    else:
        use_3d = not args.use_2d
        print(f"[region-only] converting bundle to InSTAnT dataframe, use_3d={use_3d}")
        instant_input_df = bundle_to_instant_dataframe(bundle, use_3d=use_3d)

    use_3d = (not args.use_2d) and ("absZ" in instant_input_df.columns)
    print(
        "[region-only] input for annotation: "
        f"cells={instant_input_df['uID'].astype(str).nunique()}, "
        f"genes={instant_input_df['gene'].astype(str).nunique()}, "
        f"transcripts={len(instant_input_df)}, "
        f"pairs={len(pairs)}, use_3d={use_3d}"
    )

    region_df = annotate_colocalized_pairs_regions(
        bundle=bundle,
        instant_input_df=instant_input_df,
        significant_pairs=pairs,
        distance_threshold=args.distance,
        use_3d=use_3d,
        max_pairs=args.region_max_pairs,
        region_threshold_mode=args.region_threshold_mode,
        region_workers=args.region_workers,
    )

    region_df.to_csv(out_path, index=False)
    print(f"[region-only] saved region annotated pairs: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
