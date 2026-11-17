"""Task B: Empirical Negative-Control Probe Benchmark on Bruker CosMx & 10x Xenium HEK293T.

Extracts full-panel cells (including all native negative control probes) from:
  1. 10x Xenium HEK293T H5AD (290 target genes + NegControlProbe / NegControlCodeword / UnassignedCodeword)
  2. Bruker CosMx HEK293T PKL (shared HEK293T target genes + 50 NegPrb + top 50 FalseCode probes)

Runs all 5 colocalization method families, caches full pairwise score tables for Task C2
(cross-platform concordance & ComplexPortal / RASP RRI enrichment), and writes CSV metrics:
  - exp3a_negprobes_thresholded_summary.csv
  - exp3b_negprobes_topK_all5methods.csv
"""

from __future__ import annotations

from pathlib import Path
import pickle
import sys
import time

import cv2
import h5py
import numpy as np
import pandas as pd

STRAND_ROOT = Path(__file__).resolve().parents[1]
if str(STRAND_ROOT) not in sys.path:
    sys.path.insert(0, str(STRAND_ROOT))

from scripts.coloc_competitor_runners import run_all_five_methods
from tools.engines.coloc_region_annotation import (
    _equivalent_radius,
    _load_cell_geometries,
    _safe_area,
)

OUT_DIR = STRAND_ROOT / "benchmarks" / "decoloc_v2_metrics"
PAIR_CACHE_DIR = OUT_DIR / "_pair_tables"
THREADS = 16

XENIUM_H5AD = Path("/home/wzx/rnp/RNA_spatial/Xenium_data/hek293T_Xenium_adata_subcell_including_randomized_embeddings.h5ad")
XENIUM_CELL_QC = Path("/home/wzx/thesis/GRASP/data4_Xenium_HEK293T/step0_qc/cell_qc.csv")
COSMX_PKL = Path("/home/wzx/rnp/RNA_spatial/CosMx data_HEK/CosMx_HEK293T_data_dict.pkl")

CONTROL_PREFIXES = (
    "negprb",
    "falsecode",
    "negcontrolprobe_",
    "negcontrolcodeword_",
    "unassignedcodeword_",
    "blank",
    "systemcontrol",
)


def is_control_gene(gene_name: str) -> bool:
    g = str(gene_name).lower()
    return g.startswith(CONTROL_PREFIXES)


def _extract_main_instance(flat_mask: np.ndarray, mask_side: int = 100) -> np.ndarray:
    uint16_max = np.iinfo(np.uint16).max
    label_img = np.rint(flat_mask.reshape(mask_side, mask_side) * uint16_max).astype(np.uint16)
    nz = label_img[label_img > 0]
    if nz.size == 0:
        return np.zeros((mask_side, mask_side), dtype=bool)
    labels, counts = np.unique(nz, return_counts=True)
    return label_img == int(labels[np.argmax(counts)])


def _contour_coords(mask_bool: np.ndarray) -> np.ndarray | None:
    contours, _ = cv2.findContours(
        mask_bool.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea).reshape(-1, 2)
    if len(cnt) < 3:
        return None
    return cnt.astype(np.float64)


def load_or_build_xenium_hek293t_bundle(n_cells: int = 1500, seed: int = 2026) -> tuple[dict, set[str]]:
    """Extract 1,500 complete cells (all target + negative control transcripts & contours) from Xenium H5AD."""
    cache_pkl = OUT_DIR / f"_cache_xenium_hek293t_{n_cells}cells.pkl"
    if cache_pkl.exists():
        print(f"[Xenium] Loading cached bundle: {cache_pkl}")
        with open(cache_pkl, "rb") as f:
            bundle = pickle.load(f)
        target_genes = {
            str(g) for g in bundle["data_df"]["gene"].unique() if not is_control_gene(str(g))
        }
        return bundle, target_genes

    print(f"[Xenium] Extracting {n_cells} full-panel cells (targets + negative probes) from H5AD ...")
    t0 = time.perf_counter()
    qc = pd.read_csv(XENIUM_CELL_QC)
    nc_pix_ratio = qc["nucleus_mask_pixels"] / np.maximum(qc["cell_mask_pixels"], 1)
    eligible = qc[
        (qc["eligible_cell"] == True)  # noqa: E712
        & (nc_pix_ratio >= 0.15)
        & (nc_pix_ratio <= 0.85)
        & (qc["target_transcripts"] >= 200)
    ]
    rng = np.random.default_rng(seed)
    chosen_indices = np.sort(
        rng.choice(eligible.index.to_numpy(dtype=np.int64), size=min(n_cells, len(eligible)), replace=False)
    )

    cell_boundary = {}
    nuclear_boundary = {}
    spot_dfs = []

    with h5py.File(XENIUM_H5AD, "r") as handle:
        raw_cids = handle["obs/cell_id"][:]
        cell_ids_all = [
            x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in raw_cids
        ]
        raw_vars = handle["var/gene_name"][:]
        target_genes = {
            (x.decode("utf-8") if isinstance(x, bytes) else str(x))
            for x in raw_vars
            if not is_control_gene(x.decode("utf-8") if isinstance(x, bytes) else str(x))
        }

        tg = handle["obsm/detected_transcripts"]
        t_offsets = tg["node1-offsets"][:]
        s_offsets_ds = tg["node4-offsets"]
        s_bytes_ds = tg["node5-data"]
        x_ds = tg["node30-data"]
        y_ds = tg["node28-data"]
        cmask_ds = handle["obsm/cell_mask"]
        nmask_ds = handle["obsm/dapi_mask"]

        for rank_i, c_idx in enumerate(chosen_indices):
            cid = cell_ids_all[int(c_idx)]
            c_cnt = _contour_coords(_extract_main_instance(cmask_ds[int(c_idx)]))
            n_cnt = _contour_coords(_extract_main_instance(nmask_ds[int(c_idx)]))
            if c_cnt is None or n_cnt is None:
                continue

            # Place each local 100x100 cell crop on a clean non-overlapping coordinate grid.
            ox = float((rank_i % 40) * 250.0)
            oy = float((rank_i // 40) * 250.0)
            cell_boundary[cid] = c_cnt + np.array([ox, oy])
            nuclear_boundary[cid] = n_cnt + np.array([ox, oy])

            r0 = int(t_offsets[int(c_idx)])
            r1 = int(t_offsets[int(c_idx) + 1])
            if r1 <= r0:
                continue

            so = s_offsets_ds[r0 : r1 + 1]
            b0, b1 = int(so[0]), int(so[-1])
            raw_b = bytes(s_bytes_ds[b0:b1])
            genes_cell = [
                raw_b[int(so[k]) - b0 : int(so[k + 1]) - b0].decode("utf-8")
                for k in range(r1 - r0)
            ]
            xs = np.clip(x_ds[r0:r1].astype(np.float64), 0.0, 99.0) + ox
            ys = np.clip(y_ds[r0:r1].astype(np.float64), 0.0, 99.0) + oy
            spot_dfs.append(
                pd.DataFrame({"cell": cid, "gene": genes_cell, "x": xs, "y": ys})
            )

    data_df = pd.concat(spot_dfs, ignore_index=True)
    # Keep genes with >= 3 total transcripts so ultra-singleton barcode typos don't add empty columns,
    # while retaining active NegControlProbe / NegControlCodeword / UnassignedCodeword features!
    g_tot = data_df.groupby("gene", observed=True).size()
    kept_g = set(g_tot.index[g_tot >= 3])
    data_df = data_df[data_df["gene"].isin(kept_g)].reset_index(drop=True)

    bundle = {
        "data_df": data_df,
        "cell_boundary": cell_boundary,
        "nuclear_boundary": nuclear_boundary,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(cache_pkl, "wb") as f:
        pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)

    n_ctrl_g = sum(1 for g in kept_g if is_control_gene(g))
    print(
        f"[Xenium] Extracted {len(cell_boundary)} cells, {len(data_df):,} spots, "
        f"{len(kept_g)} features ({n_ctrl_g} negative controls) in {time.perf_counter()-t0:.1f}s"
    )
    return bundle, target_genes


def build_cosmx_hek293t_negprobe_bundle(
    shared_targets: set[str],
    *,
    n_cells: int = 1000,
    seed: int = 2026,
) -> dict:
    """Build 1,000-cell Bruker CosMx HEK293T bundle with shared targets + 50 NegPrb + top 50 FalseCode."""
    print(f"[CosMx] Loading CosMx_HEK293T_data_dict.pkl and selecting {n_cells} cells ...")
    t0 = time.perf_counter()
    with open(COSMX_PKL, "rb") as f:
        cosmx = pickle.load(f)

    df = cosmx["data_df"][["cell", "gene", "x", "y"]].copy()
    df["cell"] = df["cell"].astype(str)
    df["gene"] = df["gene"].astype(str)

    cb_all = {str(k): v for k, v in cosmx["cell_boundary"].items()}
    nb_all = {str(k): v for k, v in cosmx["nuclear_boundary"].items()}
    valid_cids = sorted(set(df["cell"].unique()) & set(cb_all.keys()) & set(nb_all.keys()))

    rng = np.random.default_rng(seed)
    chosen_cids = set(rng.choice(valid_cids, size=min(n_cells, len(valid_cids)), replace=False))
    df = df[df["cell"].isin(chosen_cids)].reset_index(drop=True)

    g_counts = df.groupby("gene", observed=True).size().sort_values(ascending=False)
    all_genes = list(g_counts.index)

    negprbs = [g for g in all_genes if g.lower().startswith("negprb") and g_counts[g] >= 3]
    falsecodes = [g for g in all_genes if g.lower().startswith("falsecode") and g_counts[g] >= 3][:50]
    overlap_targets = [g for g in all_genes if (g in shared_targets) and (not is_control_gene(g))]
    # Top up with highest-expressed HEK293T targets to ~290 biological targets (matching Xenium panel size).
    extra_targets = [
        g for g in all_genes if (g not in shared_targets) and (not is_control_gene(g))
    ][: max(0, 290 - len(overlap_targets))]

    selected_panel = set(overlap_targets + extra_targets + negprbs + falsecodes)
    df = df[df["gene"].isin(selected_panel)].reset_index(drop=True)
    cb = {k: cb_all[k] for k in chosen_cids}
    nb = {k: nb_all[k] for k in chosen_cids}

    n_ctrl = len(negprbs) + len(falsecodes)
    print(
        f"[CosMx] Prepared {len(chosen_cids)} cells, {len(df):,} spots, "
        f"{df['gene'].nunique()} features ({len(overlap_targets)} Xenium-shared targets, "
        f"{n_ctrl} NegPrb/FalseCode controls) in {time.perf_counter()-t0:.1f}s"
    )
    return {"data_df": df, "cell_boundary": cb, "nuclear_boundary": nb}


def evaluate_negprobe_metrics(
    dataset_name: str,
    bundle: dict,
    method_tables: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    genes = sorted(bundle["data_df"]["gene"].astype(str).unique())
    ctrl_set = {g for g in genes if is_control_gene(g)}
    n_genes = len(genes)
    n_ctrl = len(ctrl_set)
    n_bio = n_genes - n_ctrl
    total_pairs = n_genes * (n_genes - 1) // 2
    ctrl_pairs_total = total_pairs - (n_bio * (n_bio - 1) // 2)
    prior_ctrl_frac = ctrl_pairs_total / max(total_pairs, 1)

    # 1. Thresholded metrics across cutoffs for statistical-testing methods.
    thresh_rows = []
    q_grid = [0.05, 0.01, 1e-3, 1e-4, 1e-5]

    for m_name, df_m in method_tables.items():
        if df_m.empty:
            continue
        is_c = df_m["gene_1"].isin(ctrl_set) | df_m["gene_2"].isin(ctrl_set)

        # Native default cutoff first:
        if m_name in {"DeCoLoc (direct_micro)", "DeCoLoc (whole_cell)", "InSTAnT (PP+CPB)", "SCRIN (cooccurrence)"}:
            sig = df_m["significant"].astype(bool)
            n_sig = int(sig.sum())
            n_csig = int((sig & is_c).sum())
            c_frac = (n_csig / n_sig) if n_sig > 0 else 0.0
            thresh_rows.append(
                {
                    "dataset": dataset_name,
                    "method": m_name,
                    "cutoff_type": "default_cutoff",
                    "cutoff_value": np.nan,
                    "n_panel_genes": n_genes,
                    "n_control_genes": n_ctrl,
                    "panel_control_pair_prior": prior_ctrl_frac,
                    "n_significant_pairs": n_sig,
                    "n_control_associated_pairs": n_csig,
                    "control_pair_fraction": c_frac,
                    "fold_vs_random_prior": c_frac / max(prior_ctrl_frac, 1e-12),
                }
            )

            p_col = "qval" if "qval" in df_m.columns else ("pval" if "pval" in df_m.columns else None)
            if p_col is not None:
                for cut in q_grid:
                    sig_q = df_m[p_col] < cut
                    if m_name == "DeCoLoc (direct_micro)" and "coloc_class" in df_m.columns:
                        # For DeCoLoc direct_micro sweep q_comp while keeping positive enrichment fe >= 1.15
                        sig_q = sig_q & (df_m["fe"] >= 1.15) & (df_m["score"] > 0)
                    elif m_name == "DeCoLoc (whole_cell)":
                        sig_q = sig_q & (df_m["fe"] >= 1.15)
                    ns = int(sig_q.sum())
                    ncs = int((sig_q & is_c).sum())
                    cf = (ncs / ns) if ns > 0 else 0.0
                    thresh_rows.append(
                        {
                            "dataset": dataset_name,
                            "method": m_name,
                            "cutoff_type": p_col,
                            "cutoff_value": cut,
                            "n_panel_genes": n_genes,
                            "n_control_genes": n_ctrl,
                            "panel_control_pair_prior": prior_ctrl_frac,
                            "n_significant_pairs": ns,
                            "n_control_associated_pairs": ncs,
                            "control_pair_fraction": cf,
                            "fold_vs_random_prior": cf / max(prior_ctrl_frac, 1e-12),
                        }
                    )

    # 2. Universal Top-K control contamination across ALL 5 method families.
    topk_rows = []
    for m_name, df_m in method_tables.items():
        is_c = (df_m["gene_1"].isin(ctrl_set) | df_m["gene_2"].isin(ctrl_set)).to_numpy(bool) if len(df_m) else np.empty(0, dtype=bool)
        elapsed = float(df_m["elapsed_seconds"].iloc[0]) if len(df_m) else np.nan
        for k in [50, 100, 200, 500, 1000]:
            pos_df = df_m[df_m["score"] > 0] if m_name.startswith("DeCoLoc") else df_m
            kk = min(k, len(pos_df))
            if kk == 0:
                n_ck, frac_k = 0, 0.0
            else:
                is_c_sub = (pos_df["gene_1"].iloc[:kk].isin(ctrl_set) | pos_df["gene_2"].iloc[:kk].isin(ctrl_set))
                n_ck = int(is_c_sub.sum())
                frac_k = n_ck / float(kk)
            topk_rows.append(
                {
                    "dataset": dataset_name,
                    "method": m_name,
                    "K": k,
                    "effective_K": kk,
                    "n_control_in_topK": n_ck,
                    "control_fraction_at_K": frac_k,
                    "panel_control_pair_prior": prior_ctrl_frac,
                    "fold_vs_random_prior": frac_k / max(prior_ctrl_frac, 1e-12),
                    "elapsed_seconds": elapsed,
                }
            )

    return pd.DataFrame(thresh_rows), pd.DataFrame(topk_rows)


def _safe_method_slug(m_name: str) -> str:
    return (
        m_name.lower()
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("+", "plus")
    )


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PAIR_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    xen_bundle, xen_targets = load_or_build_xenium_hek293t_bundle(n_cells=1500, seed=2026)
    cos_bundle = build_cosmx_hek293t_negprobe_bundle(xen_targets, n_cells=1000, seed=2026)

    all_thresh, all_topk = [], []
    for ds_name, bnd, b_min_pts in [
        ("Xenium_HEK293T", xen_bundle, 5),
        ("CosMx_HEK293T", cos_bundle, 5),
    ]:
        geoms = _load_cell_geometries(bnd)
        med_r = float(np.median([_equivalent_radius(_safe_area(g.cell)) for g in geoms.values()]))
        radius = float(0.10 * med_r)
        print(f"\n=== [Task B] Running 5 method families on {ds_name} (median_r={med_r:.2f}, radius={radius:.2f}) ===")

        res_map = run_all_five_methods(
            bnd,
            radius=radius,
            distance_mode="cell_scaled",
            bento_min_points=b_min_pts,
            pytrance_epochs=25,
            pytrance_max_spots=1_800_000,
            threads=THREADS,
        )

        # Cache full pair score tables so Task C2 can evaluate cross-platform concordance & biology immediately.
        for m_name, df_m in res_map.items():
            out_pq = PAIR_CACHE_DIR / f"{ds_name}__{_safe_method_slug(m_name)}.parquet"
            df_m.to_parquet(out_pq, index=False)

        t_df, tk_df = evaluate_negprobe_metrics(ds_name, bnd, res_map)
        all_thresh.append(t_df)
        all_topk.append(tk_df)

    thresh_full = pd.concat(all_thresh, ignore_index=True)
    topk_full = pd.concat(all_topk, ignore_index=True)
    thresh_full.to_csv(OUT_DIR / "exp3a_negprobes_thresholded_summary.csv", index=False)
    topk_full.to_csv(OUT_DIR / "exp3b_negprobes_topK_all5methods.csv", index=False)

    print("\n--- [Task B Summary] Default Cutoff Negative-Control Contamination ---")
    print(
        thresh_full[thresh_full["cutoff_type"] == "default_cutoff"][
            ["dataset", "method", "n_significant_pairs", "n_control_associated_pairs", "control_pair_fraction", "fold_vs_random_prior"]
        ].round(4).to_string(index=False)
    )
    print("\n--- [Task B Top-K] Control Fraction @ K=100 & K=200 across All 5 Methods ---")
    print(
        topk_full[topk_full["K"].isin([100, 200])][
            ["dataset", "method", "K", "n_control_in_topK", "control_fraction_at_K", "fold_vs_random_prior", "elapsed_seconds"]
        ].round(4).to_string(index=False)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
