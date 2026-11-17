"""Task C: Multi-Replicate Reproducibility (U2OS) & Cross-Platform Concordance + Wet-Lab Enrichment (HEK293T).

Sub-benchmark C1 (--part c1):
  Runs all 5 method families independently on 3 biological replicates of U2OS MERFISH
  (Dataset33_2, Dataset33_3, Dataset33_4; 127 shared genes, 600 cells/replicate) and
  computes pairwise Top-K Overlap Coefficient (OC@K), Jaccard@K, Spearman rank correlation,
  and significant-set reproducibility.

Sub-benchmark C2 (--part c2):
  Loads Task B's cached full pairwise tables on Bruker CosMx HEK293T and 10x Xenium HEK293T,
  restricts to the exact intersection of shared biological target genes (~265 genes), and evaluates:
    1) Cross-platform Top-K Overlap Coefficient (OC@K), fold over random expectation, hypergeometric p-value,
       and Spearman correlation across platforms;
    2) Orthogonal wet-lab enrichment against HEK293/HEK293T experimental RNA-RNA interactions (RASP database:
       KARR-seq / PARIS2 / RIC-seq) and ComplexPortal human RNP co-complex memberships.
"""

from __future__ import annotations

import argparse
import gzip
from itertools import combinations
from pathlib import Path
import pickle
import sys

import numpy as np
import pandas as pd
from scipy import stats

STRAND_ROOT = Path(__file__).resolve().parents[1]
if str(STRAND_ROOT) not in sys.path:
    sys.path.insert(0, str(STRAND_ROOT))

from scripts.coloc_competitor_runners import canonical_pair_cols, run_all_five_methods
from scripts.run_bench_b_negprobes import _safe_method_slug, is_control_gene
from tools.decoloc.api import prefilter_bundle_for_decoloc
from tools.engines.coloc_region_annotation import (
    _equivalent_radius,
    _load_cell_geometries,
    _safe_area,
)

OUT_DIR = STRAND_ROOT / "benchmarks" / "decoloc_v2_metrics"
PAIR_CACHE_DIR = OUT_DIR / "_pair_tables"
THREADS = 16

U2OS_REP_PATHS = {
    "Rep2": Path("/home/wzx/thesis/GRASP/u2os_data/spatial/Dataset33_merfish_u2os/Dataset33_2_merfish_u2os_data_dict.pkl"),
    "Rep3": Path("/home/wzx/thesis/GRASP/u2os_data/spatial/Dataset33_merfish_u2os/Dataset33_3_merfish_u2os_data_dict.pkl"),
    "Rep4": Path("/home/wzx/thesis/GRASP/u2os_data/spatial/Dataset33_merfish_u2os/Dataset33_4_merfish_u2os_data_dict.pkl"),
}

RASP_HEK293_PATH = Path("/home/wzx/rnp/RNA_RRI/rasp/rri_rasp_hek293.tsv.gz")
COMPLEX_MEMBERS_PATH = Path("/home/wzx/rnp/RNP_complex_verification/ComplexPortal_human_RNP/core_human_rnp_members.tsv")


def _concordance_between_two_tables(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    universe_pairs: list[tuple[str, str]],
    k_list: list[int],
) -> tuple[dict, list[dict]]:
    """Compute rank correlation, threshold concordance, and Top-K OC/Jaccard on a shared gene-pair universe."""
    univ_set = set(universe_pairs)
    n_univ = len(universe_pairs)

    da = df_a[df_a.apply(lambda r: (r["gene_1"], r["gene_2"]) in univ_set, axis=1)].copy() if len(df_a) else df_a
    db = df_b[df_b.apply(lambda r: (r["gene_1"], r["gene_2"]) in univ_set, axis=1)].copy() if len(df_b) else df_b
    da = da.sort_values("score", ascending=False).reset_index(drop=True)
    db = db.sort_values("score", ascending=False).reset_index(drop=True)

    map_a = dict(zip(zip(da["gene_1"], da["gene_2"]), da["score"].to_numpy(float)))
    map_b = dict(zip(zip(db["gene_1"], db["gene_2"]), db["score"].to_numpy(float)))
    floor_a = (min(map_a.values()) - 1.0) if map_a else 0.0
    floor_b = (min(map_b.values()) - 1.0) if map_b else 0.0
    vec_a = np.array([map_a.get(p, floor_a) for p in universe_pairs], dtype=float)
    vec_b = np.array([map_b.get(p, floor_b) for p in universe_pairs], dtype=float)

    if np.std(vec_a) > 0 and np.std(vec_b) > 0:
        spearman_r, spearman_p = stats.spearmanr(vec_a, vec_b)
    else:
        spearman_r, spearman_p = 0.0, 1.0

    sig_a = set(zip(da.loc[da["significant"].astype(bool), "gene_1"], da.loc[da["significant"].astype(bool), "gene_2"])) if len(da) else set()
    sig_b = set(zip(db.loc[db["significant"].astype(bool), "gene_1"], db.loc[db["significant"].astype(bool), "gene_2"])) if len(db) else set()
    inter_sig = len(sig_a & sig_b)
    min_sig = min(len(sig_a), len(sig_b))
    union_sig = len(sig_a | sig_b)
    sig_oc = (inter_sig / min_sig) if min_sig > 0 else np.nan
    sig_jaccard = (inter_sig / union_sig) if union_sig > 0 else np.nan

    global_metrics = {
        "n_universe_pairs": n_univ,
        "spearman_r": float(spearman_r),
        "spearman_p": float(spearman_p),
        "n_sig_A": len(sig_a),
        "n_sig_B": len(sig_b),
        "n_sig_overlap": inter_sig,
        "sig_overlap_coeff": sig_oc,
        "sig_jaccard": sig_jaccard,
    }

    pairs_a = list(zip(da["gene_1"], da["gene_2"]))
    pairs_b = list(zip(db["gene_1"], db["gene_2"]))
    topk_metrics = []
    for k in k_list:
        ka = min(k, len(pairs_a))
        kb = min(k, len(pairs_b))
        sa = set(pairs_a[:ka])
        sb = set(pairs_b[:kb])
        inter = len(sa & sb)
        denom = min(ka, kb)
        uni = len(sa | sb)
        oc = (inter / float(denom)) if denom > 0 else 0.0
        jac = (inter / float(uni)) if uni > 0 else 0.0
        expected_inter = (ka * kb) / max(float(n_univ), 1.0)
        fold_rand = inter / max(expected_inter, 1e-12)
        pval_hg = float(stats.hypergeom.sf(inter - 1, n_univ, ka, kb)) if inter > 0 else 1.0
        topk_metrics.append(
            {
                "K": k,
                "n_overlap_at_K": inter,
                "overlap_coeff_at_K": oc,
                "jaccard_at_K": jac,
                "expected_random_overlap": expected_inter,
                "fold_over_random": fold_rand,
                "hypergeom_p": pval_hg,
            }
        )

    return global_metrics, topk_metrics


def run_part_c1_u2os_replicates(n_cells_per_rep: int = 600) -> pd.DataFrame:
    print(f"=== [Task C1] U2OS 3-Replicate Reproducibility ({n_cells_per_rep} cells/rep) ===")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PAIR_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    rep_bundles = {}
    gene_sets = []
    for rep_idx, (rep_name, pkl_path) in enumerate(U2OS_REP_PATHS.items()):
        with open(pkl_path, "rb") as f:
            raw_b = pickle.load(f)
        clean_b, _ = prefilter_bundle_for_decoloc(raw_b)
        rng = np.random.default_rng(1000 + rep_idx)
        cids = sorted(clean_b["data_df"]["cell"].unique())
        chosen = set(rng.choice(cids, size=min(n_cells_per_rep, len(cids)), replace=False))
        df_sub = clean_b["data_df"][clean_b["data_df"]["cell"].isin(chosen)].reset_index(drop=True)
        cb = {k: v for k, v in clean_b["cell_boundary"].items() if str(k) in chosen}
        nb = {k: v for k, v in clean_b["nuclear_boundary"].items() if str(k) in chosen}
        rep_bundles[rep_name] = {"data_df": df_sub, "cell_boundary": cb, "nuclear_boundary": nb}
        gene_sets.append(set(df_sub["gene"].astype(str).unique()))

    shared_genes = sorted(set.intersection(*gene_sets))
    shared_set = set(shared_genes)
    universe_pairs = list(combinations(shared_genes, 2))
    print(f"Shared U2OS genes across Rep2/Rep3/Rep4: {len(shared_genes)} ({len(universe_pairs)} pairs)")

    rep_results: dict[str, dict[str, pd.DataFrame]] = {}
    for rep_name, bnd in rep_bundles.items():
        bnd["data_df"] = bnd["data_df"][bnd["data_df"]["gene"].isin(shared_set)].reset_index(drop=True)
        geoms = _load_cell_geometries(bnd)
        med_r = float(np.median([_equivalent_radius(_safe_area(g.cell)) for g in geoms.values()]))
        radius = float(0.10 * med_r)
        print(f"\n--- Running 5 method families on U2OS {rep_name} (spots={len(bnd['data_df']):,}, radius={radius:.2f}) ---")
        res_map = run_all_five_methods(
            bnd,
            radius=radius,
            distance_mode="cell_scaled",
            bento_min_points=10,
            pytrance_epochs=25,
            pytrance_max_spots=1_800_000,
            threads=THREADS,
        )
        rep_results[rep_name] = res_map
        for m_name, df_m in res_map.items():
            df_m.to_parquet(PAIR_CACHE_DIR / f"U2OS_{rep_name}__{_safe_method_slug(m_name)}.parquet", index=False)

    rows = []
    k_list = [50, 100, 200, 300]
    method_names = list(rep_results["Rep2"].keys())
    for rep_a, rep_b in [("Rep2", "Rep3"), ("Rep2", "Rep4"), ("Rep3", "Rep4")]:
        for m_name in method_names:
            glob_m, tk_list = _concordance_between_two_tables(
                rep_results[rep_a][m_name],
                rep_results[rep_b][m_name],
                universe_pairs,
                k_list,
            )
            for tk in tk_list:
                rows.append({"rep_pair": f"{rep_a}_vs_{rep_b}", "method": m_name, **glob_m, **tk})

    rep_df = pd.DataFrame(rows)
    rep_df.to_csv(OUT_DIR / "exp4a_u2os_replicate_reproducibility.csv", index=False)

    mean_df = (
        rep_df.groupby(["method", "K"], observed=True)[
            ["overlap_coeff_at_K", "jaccard_at_K", "fold_over_random", "spearman_r", "sig_overlap_coeff", "sig_jaccard"]
        ]
        .mean()
        .reset_index()
    )
    mean_df.to_csv(OUT_DIR / "exp4a_u2os_replicate_mean_summary.csv", index=False)

    print("\n--- [Task C1 Summary] Mean Replicate Reproducibility across Rep2/3/4 (K=100 & 200) ---")
    print(mean_df[mean_df["K"].isin([100, 200])].round(4).to_string(index=False))
    return mean_df


def _load_orthogonal_gold_standards(shared_genes: set[str]) -> dict[str, set[tuple[str, str]]]:
    """Load RASP HEK293 experimental RRI pairs and ComplexPortal RNP co-complex pairs within shared_genes."""
    rasp_pairs: set[tuple[str, str]] = set()
    if RASP_HEK293_PATH.exists():
        with gzip.open(RASP_HEK293_PATH, "rt") as f:
            header = f.readline().rstrip("\n").split("\t")
            i1 = header.index("gene_name1")
            i2 = header.index("gene_name2")
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) <= max(i1, i2):
                    continue
                g1, g2 = parts[i1], parts[i2]
                if g1 != g2 and (g1 in shared_genes) and (g2 in shared_genes):
                    rasp_pairs.add((min(g1, g2), max(g1, g2)))

    cpx_pairs: set[tuple[str, str]] = set()
    if COMPLEX_MEMBERS_PATH.exists():
        cm = pd.read_csv(COMPLEX_MEMBERS_PATH, sep="\t")
        cm = cm[cm["name"].astype(str).isin(shared_genes)]
        for _, grp in cm.groupby("complex_ac"):
            members = sorted(set(grp["name"].astype(str)))
            for a, b in combinations(members, 2):
                cpx_pairs.add((a, b))

    combined = rasp_pairs | cpx_pairs
    return {
        "RASP_HEK293_RRI": rasp_pairs,
        "ComplexPortal_RNP": cpx_pairs,
        "Union_RASP_or_Complex": combined,
    }


def run_part_c2_hek293t_cross_platform_and_biology() -> tuple[pd.DataFrame, pd.DataFrame]:
    print("\n=== [Task C2] HEK293T Cross-Platform Concordance (CosMx vs Xenium) & Wet-Lab Enrichment ===")
    method_names = [
        "DeCoLoc (direct_micro)",
        "DeCoLoc (whole_cell)",
        "InSTAnT (PP+CPB)",
        "SCRIN (cooccurrence)",
        "Bento (CLQ)",
        "Bento (RNAcoloc)",
        "pyTrance (GNN)",
    ]

    xen_tables, cos_tables = {}, {}
    xen_bio_genes: set[str] = set()
    cos_bio_genes: set[str] = set()

    for m_name in method_names:
        slug = _safe_method_slug(m_name)
        px = PAIR_CACHE_DIR / f"Xenium_HEK293T__{slug}.parquet"
        pc = PAIR_CACHE_DIR / f"CosMx_HEK293T__{slug}.parquet"
        if not px.exists() or not pc.exists():
            raise FileNotFoundError(f"Missing cached pair table for {m_name}: {px} or {pc}. Run Task B first.")
        dx = canonical_pair_cols(pd.read_parquet(px))
        dc = canonical_pair_cols(pd.read_parquet(pc))
        xen_tables[m_name] = dx
        cos_tables[m_name] = dc
        if m_name == "DeCoLoc (whole_cell)":
            xen_bio_genes = {g for g in set(dx["gene_1"]) | set(dx["gene_2"]) if not is_control_gene(g)}
            cos_bio_genes = {g for g in set(dc["gene_1"]) | set(dc["gene_2"]) if not is_control_gene(g)}

    shared_genes = sorted(xen_bio_genes & cos_bio_genes)
    shared_set = set(shared_genes)
    universe_pairs = list(combinations(shared_genes, 2))
    n_univ = len(universe_pairs)
    print(f"Shared HEK293T biological target genes (CosMx ∩ Xenium): {len(shared_genes)} ({n_univ:,} pairs)")

    # 1. Cross-platform concordance on shared biological pairs.
    cp_rows = []
    k_list = [50, 100, 200, 500]
    for m_name in method_names:
        dx = xen_tables[m_name]
        dc = cos_tables[m_name]
        dx_bio = dx[dx["gene_1"].isin(shared_set) & dx["gene_2"].isin(shared_set)].reset_index(drop=True)
        dc_bio = dc[dc["gene_1"].isin(shared_set) & dc["gene_2"].isin(shared_set)].reset_index(drop=True)
        # Keep strictly positive-scored candidates for DeCoLoc so non-colocalized zero-floor pairs aren't randomly ranked
        if m_name.startswith("DeCoLoc"):
            dx_bio = dx_bio[dx_bio["score"] > 0].reset_index(drop=True)
            dc_bio = dc_bio[dc_bio["score"] > 0].reset_index(drop=True)

        glob_m, tk_list = _concordance_between_two_tables(dx_bio, dc_bio, universe_pairs, k_list)
        for tk in tk_list:
            cp_rows.append({"comparison": "CosMx_vs_Xenium_HEK293T", "method": m_name, **glob_m, **tk})

    cp_df = pd.DataFrame(cp_rows)
    cp_df.to_csv(OUT_DIR / "exp4b_hek293t_cross_platform_concordance.csv", index=False)

    # 2. Wet-lab RASP RRI & ComplexPortal RNP enrichment on shared gene universe.
    gold_sets = _load_orthogonal_gold_standards(shared_set)
    for gs_name, gs_pairs in gold_sets.items():
        print(f"  Gold standard {gs_name}: {len(gs_pairs):,} positive pairs in {n_univ:,} shared pairs ({len(gs_pairs)/max(n_univ,1):.4%})")

    bio_rows = []
    for platform_name, tbl_map in [("Xenium_HEK293T", xen_tables), ("CosMx_HEK293T", cos_tables)]:
        for m_name, df_m in tbl_map.items():
            sub = df_m[df_m["gene_1"].isin(shared_set) & df_m["gene_2"].isin(shared_set)].copy()
            sub = sub.sort_values("score", ascending=False).reset_index(drop=True)
            if m_name.startswith("DeCoLoc"):
                ranked_pos = sub[sub["score"] > 0]
            else:
                ranked_pos = sub
            ranked_pairs = list(zip(ranked_pos["gene_1"], ranked_pos["gene_2"]))

            eval_slices: list[tuple[str, list[tuple[str, str]]]] = [
                ("Top100", ranked_pairs[:100]),
                ("Top200", ranked_pairs[:200]),
            ]
            if m_name in {"DeCoLoc (direct_micro)", "DeCoLoc (whole_cell)", "InSTAnT (PP+CPB)", "SCRIN (cooccurrence)"}:
                sig_sub = sub[sub["significant"].astype(bool)]
                eval_slices.append(("Significant_Set", list(zip(sig_sub["gene_1"], sig_sub["gene_2"]))))

            for gs_name, gs_pairs in gold_sets.items():
                if not gs_pairs:
                    continue
                bg_rate = len(gs_pairs) / float(n_univ)
                for slice_name, sel_pairs in eval_slices:
                    n_sel = len(sel_pairs)
                    if n_sel == 0:
                        hits, prec_val, fe_val, p_hg = 0, 0.0, 0.0, 1.0
                    else:
                        hits = sum(1 for p in sel_pairs if p in gs_pairs)
                        prec_val = hits / float(n_sel)
                        fe_val = prec_val / max(bg_rate, 1e-12)
                        p_hg = float(stats.hypergeom.sf(hits - 1, n_univ, len(gs_pairs), n_sel)) if hits > 0 else 1.0
                    bio_rows.append(
                        {
                            "platform": platform_name,
                            "gold_standard": gs_name,
                            "n_gold_in_universe": len(gs_pairs),
                            "universe_prior_rate": bg_rate,
                            "method": m_name,
                            "selection": slice_name,
                            "n_selected": n_sel,
                            "n_hits": hits,
                            "hit_rate_precision": prec_val,
                            "fold_enrichment": fe_val,
                            "hypergeom_p": p_hg,
                        }
                    )

    bio_df = pd.DataFrame(bio_rows)
    bio_df.to_csv(OUT_DIR / "exp4c_hek293t_orthogonal_biology_enrichment.csv", index=False)

    print("\n--- [Task C2 Summary] Cross-Platform Concordance CosMx vs Xenium (K=100 & 200) ---")
    print(
        cp_df[cp_df["K"].isin([100, 200])][
            ["method", "K", "n_overlap_at_K", "overlap_coeff_at_K", "fold_over_random", "hypergeom_p", "spearman_r"]
        ].round(4).to_string(index=False)
    )
    print("\n--- [Task C2 Biology] Wet-Lab Union (RASP RRI + RNP Complex) Enrichment (Top100 & Significant_Set) ---")
    sub_b = bio_df[
        (bio_df["gold_standard"] == "Union_RASP_or_Complex")
        & (bio_df["selection"].isin(["Top100", "Significant_Set"]))
    ]
    print(
        sub_b[
            ["platform", "method", "selection", "n_selected", "n_hits", "hit_rate_precision", "fold_enrichment", "hypergeom_p"]
        ].round(4).to_string(index=False)
    )
    return cp_df, bio_df


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", choices=["c1", "c2", "both"], default="both")
    parser.add_argument("--cells-per-rep", type=int, default=600)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.part in {"c1", "both"}:
        run_part_c1_u2os_replicates(n_cells_per_rep=args.cells_per_rep)
    if args.part in {"c2", "both"}:
        run_part_c2_hek293t_cross_platform_and_biology()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
