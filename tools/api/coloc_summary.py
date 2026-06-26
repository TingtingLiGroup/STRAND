# tools/api/coloc_summary.py

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Any
import json

import numpy as np
import pandas as pd


REGION_COLS = [
    "nuclear_enrichment",
    "perinuclear_enrichment",
    "cytosolic_enrichment",
    "peripheral_enrichment",
]
REGION_NAMES = ["nuclear", "perinuclear", "cytosolic", "peripheral"]


def _safe_to_csv(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def _pair_key_df(df: pd.DataFrame) -> pd.Series:
    if not {"gene_1", "gene_2"}.issubset(df.columns):
        raise ValueError("DataFrame must contain gene_1 and gene_2 columns.")
    g1 = df["gene_1"].astype(str)
    g2 = df["gene_2"].astype(str)
    a = np.where(g1 <= g2, g1, g2)
    b = np.where(g1 <= g2, g2, g1)
    return pd.Series(a, index=df.index) + "--" + pd.Series(b, index=df.index)


def _split_pair(pair: str) -> tuple[str, str]:
    parts = str(pair).split("--", 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    return str(pair), ""


def _join_unique(values: Iterable[Any]) -> str:
    vals = []
    for v in values:
        if pd.isna(v):
            continue
        s = str(v)
        if s and s.lower() != "nan":
            vals.append(s)
    return ";".join(sorted(set(vals)))


def _n_unique(values: Iterable[Any]) -> int:
    vals = set()
    for v in values:
        if pd.isna(v):
            continue
        s = str(v)
        if s and s.lower() != "nan":
            vals.add(s)
    return len(vals)


def _first_existing(cols: list[str], candidates: list[str]) -> str | None:
    for c in candidates:
        if c in cols:
            return c
    return None


def _normalize_numeric(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def _read_one_result_dir(
    result_dir: Path,
    *,
    dataset: str | None = None,
    group: str | None = None,
    metadata: dict | None = None,
) -> pd.DataFrame:
    """Read one colocalization output directory.

    Expected files:
        instant_significant_pairs.csv
        instant_region_annotated_pairs.csv, optional
    """
    result_dir = Path(result_dir)
    pairs_path = result_dir / "instant_significant_pairs.csv"
    if not pairs_path.exists():
        return pd.DataFrame()

    pairs = pd.read_csv(pairs_path)
    if pairs.empty:
        return pd.DataFrame()
    if not {"gene_1", "gene_2"}.issubset(pairs.columns):
        raise ValueError(f"{pairs_path} must contain gene_1 and gene_2 columns.")

    pairs["gene_1"] = pairs["gene_1"].astype(str)
    pairs["gene_2"] = pairs["gene_2"].astype(str)
    pairs["pair"] = _pair_key_df(pairs)
    pairs = _normalize_numeric(
        pairs,
        [
            "cpb_pvalue",
            "cpb_pvalue_raw",
            "expected_coloc",
            "distance_threshold",
            "pp_alpha",
            "cpb_alpha",
            "n_cells_pp_significant",
        ],
    )

    region_path = result_dir / "instant_region_annotated_pairs.csv"
    if region_path.exists():
        region = pd.read_csv(region_path)
        if not region.empty and {"gene_1", "gene_2"}.issubset(region.columns):
            region["gene_1"] = region["gene_1"].astype(str)
            region["gene_2"] = region["gene_2"].astype(str)
            region["pair"] = _pair_key_df(region)
            region = _normalize_numeric(
                region,
                [
                    "n_proximal_pairs",
                    "dominant_enrichment_score",
                    "second_enrichment_score",
                    *REGION_COLS,
                ],
            )
            keep_region_cols = [
                c
                for c in [
                    "pair",
                    "n_proximal_pairs",
                    *REGION_COLS,
                    "dominant_region",
                    "dominant_enrichment_score",
                    "second_region",
                    "second_enrichment_score",
                ]
                if c in region.columns
            ]
            region = region[keep_region_cols].drop_duplicates("pair")
            # Avoid duplicate columns if older significant-pair table already has region columns.
            drop_cols = [c for c in region.columns if c != "pair" and c in pairs.columns]
            pairs = pairs.drop(columns=drop_cols, errors="ignore")
            pairs = pairs.merge(region, on="pair", how="left")

    pairs.insert(0, "result_dir", str(result_dir))
    pairs.insert(0, "group", str(group) if group is not None else "global")
    pairs.insert(0, "dataset", str(dataset) if dataset is not None else result_dir.name)

    if metadata:
        for k, v in metadata.items():
            if k not in {"result_dir", "dataset"}:
                pairs[k] = v

    return pairs


def _discover_group_dirs(input_dir: Path) -> list[tuple[str, Path]]:
    """Return (group_name, dir) for either one result dir or subdirs from groupby runs."""
    input_dir = Path(input_dir)
    if (input_dir / "instant_significant_pairs.csv").exists():
        return [("global", input_dir)]
    out = []
    if not input_dir.exists():
        return out
    for p in sorted(input_dir.iterdir()):
        if p.is_dir() and (p / "instant_significant_pairs.csv").exists():
            out.append((p.name, p))
    return out


def _presence_matrix(master: pd.DataFrame, context_col: str) -> pd.DataFrame:
    if master.empty:
        return pd.DataFrame()
    presence = (
        master.assign(present=1)
        .pivot_table(index="pair", columns=context_col, values="present", aggfunc="max", fill_value=0)
        .reset_index()
    )
    gene_cols = master.drop_duplicates("pair")[["pair", "gene_1", "gene_2"]]
    return gene_cols.merge(presence, on="pair", how="right")


def _overlap_and_jaccard(master: pd.DataFrame, context_col: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if master.empty or context_col not in master.columns:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    contexts = sorted(master[context_col].dropna().astype(str).unique())
    pair_sets = {
        c: set(master.loc[master[context_col].astype(str) == c, "pair"].astype(str))
        for c in contexts
    }
    overlap = pd.DataFrame(index=contexts, columns=contexts, dtype=int)
    jaccard = pd.DataFrame(index=contexts, columns=contexts, dtype=float)
    long_rows = []
    for a in contexts:
        for b in contexts:
            inter = len(pair_sets[a] & pair_sets[b])
            union = len(pair_sets[a] | pair_sets[b])
            jac = float(inter / union) if union else np.nan
            overlap.loc[a, b] = inter
            jaccard.loc[a, b] = jac
            if a <= b:
                long_rows.append({
                    f"{context_col}_1": a,
                    f"{context_col}_2": b,
                    "n_pairs_1": len(pair_sets[a]),
                    "n_pairs_2": len(pair_sets[b]),
                    "n_shared_pairs": inter,
                    "n_union_pairs": union,
                    "jaccard": jac,
                })
    overlap = overlap.reset_index().rename(columns={"index": context_col})
    jaccard = jaccard.reset_index().rename(columns={"index": context_col})
    return overlap, jaccard, pd.DataFrame(long_rows)


def _region_distribution(master: pd.DataFrame, context_col: str) -> pd.DataFrame:
    if master.empty or "dominant_region" not in master.columns:
        return pd.DataFrame()
    sub = master.copy()
    sub["dominant_region"] = sub["dominant_region"].fillna("unannotated").astype(str)
    count = (
        sub.groupby([context_col, "dominant_region"], dropna=False)
        .size()
        .reset_index(name="n_pairs")
    )
    total = count.groupby(context_col)["n_pairs"].transform("sum")
    count["fraction"] = count["n_pairs"] / total.replace(0, np.nan)
    return count.sort_values([context_col, "n_pairs"], ascending=[True, False])


def _region_enrichment_long(master: pd.DataFrame) -> pd.DataFrame:
    if master.empty:
        return pd.DataFrame()
    cols = [c for c in REGION_COLS if c in master.columns]
    if not cols:
        return pd.DataFrame()
    id_cols = [c for c in ["dataset", "group", "pair", "gene_1", "gene_2", "dominant_region"] if c in master.columns]
    long = master[id_cols + cols].melt(
        id_vars=id_cols,
        value_vars=cols,
        var_name="region_metric",
        value_name="enrichment",
    )
    long["region"] = long["region_metric"].str.replace("_enrichment", "", regex=False)
    return long.drop(columns=["region_metric"])


def _gene_hubs(master: pd.DataFrame, context_col: str | None = None) -> pd.DataFrame:
    if master.empty:
        return pd.DataFrame()
    rows = []
    group_iter = [("all", master)] if context_col is None else master.groupby(context_col, dropna=False)
    for ctx, sub in group_iter:
        edges = sub[["gene_1", "gene_2", "pair"]].drop_duplicates("pair")
        genes = pd.concat([
            edges[["gene_1", "pair"]].rename(columns={"gene_1": "gene"}),
            edges[["gene_2", "pair"]].rename(columns={"gene_2": "gene"}),
        ], ignore_index=True)
        deg = genes.groupby("gene")["pair"].nunique().reset_index(name="degree")
        if "dominant_region" in sub.columns:
            reg_rows = []
            for gene, gsub in genes.groupby("gene"):
                related_pairs = set(gsub["pair"])
                regs = sub[sub["pair"].isin(related_pairs)]["dominant_region"].dropna().astype(str)
                regs = regs[regs != "none"]
                reg_rows.append({
                    "gene": gene,
                    "top_region": regs.value_counts().index[0] if len(regs) else "none",
                    "region_counts": ";".join(f"{r}:{int(c)}" for r, c in regs.value_counts().items()) if len(regs) else "",
                })
            deg = deg.merge(pd.DataFrame(reg_rows), on="gene", how="left")
        if "cpb_pvalue" in sub.columns:
            # Best significance among pairs involving this gene.
            p_rows = []
            for gene in deg["gene"]:
                vals = sub[(sub["gene_1"] == gene) | (sub["gene_2"] == gene)]["cpb_pvalue"]
                p_rows.append({"gene": gene, "best_cpb_pvalue": pd.to_numeric(vals, errors="coerce").min()})
            deg = deg.merge(pd.DataFrame(p_rows), on="gene", how="left")
        if context_col is not None:
            deg.insert(0, context_col, ctx)
        rows.append(deg.sort_values("degree", ascending=False))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _pair_context_summary(master: pd.DataFrame, context_col: str) -> pd.DataFrame:
    if master.empty:
        return pd.DataFrame()
    rows = []
    all_contexts = sorted(master[context_col].dropna().astype(str).unique())
    n_all = len(all_contexts)
    for pair, sub in master.groupby("pair", sort=False):
        contexts = sorted(set(sub[context_col].dropna().astype(str)))
        regs = sub["dominant_region"].dropna().astype(str) if "dominant_region" in sub.columns else pd.Series(dtype=str)
        regs = regs[regs != "none"]
        if len(regs):
            region_counts = regs.value_counts()
            consensus_region = str(region_counts.index[0])
            region_consistency = float(region_counts.iloc[0] / len(regs))
            n_regions = int(region_counts.size)
            region_counts_str = ";".join(f"{r}:{int(c)}" for r, c in region_counts.items())
        else:
            consensus_region = "none"
            region_consistency = np.nan
            n_regions = 0
            region_counts_str = ""
        first = sub.iloc[0]
        row = {
            "pair": pair,
            "gene_1": first["gene_1"],
            "gene_2": first["gene_2"],
            f"n_{context_col}s_detected": len(contexts),
            f"{context_col}s_detected": ";".join(contexts),
            f"presence_fraction_across_{context_col}s": float(len(contexts) / n_all) if n_all else np.nan,
            "is_specific": len(contexts) == 1,
            "specific_context": contexts[0] if len(contexts) == 1 else "",
            "is_shared": len(contexts) >= 2,
            "is_ubiquitous": len(contexts) == n_all and n_all > 0,
            "consensus_region": consensus_region,
            "region_consistency": region_consistency,
            "n_dominant_regions": n_regions,
            "region_counts": region_counts_str,
            "is_region_consistent": bool(region_consistency == 1.0) if pd.notna(region_consistency) else False,
            "is_region_shifted": n_regions > 1,
        }
        if "cpb_pvalue" in sub.columns:
            vals = pd.to_numeric(sub["cpb_pvalue"], errors="coerce")
            row.update({
                "min_cpb_pvalue": vals.min(),
                "median_cpb_pvalue": vals.median(),
                "max_neglog10_cpb": float((-np.log10(vals.replace(0, np.nan))).max()) if vals.notna().any() else np.nan,
            })
        if "n_cells_pp_significant" in sub.columns:
            vals = pd.to_numeric(sub["n_cells_pp_significant"], errors="coerce")
            row.update({
                "max_n_cells_pp_significant": vals.max(),
                "sum_n_cells_pp_significant": vals.sum(),
            })
        if "n_proximal_pairs" in sub.columns:
            vals = pd.to_numeric(sub["n_proximal_pairs"], errors="coerce")
            row.update({
                "max_n_proximal_pairs": vals.max(),
                "sum_n_proximal_pairs": vals.sum(),
            })
        if "dominant_enrichment_score" in sub.columns:
            vals = pd.to_numeric(sub["dominant_enrichment_score"], errors="coerce")
            row.update({
                "max_dominant_enrichment_score": vals.max(),
                "mean_dominant_enrichment_score": vals.mean(),
            })
        rows.append(row)
    out = pd.DataFrame(rows)
    sort_cols = [c for c in [f"n_{context_col}s_detected", "region_consistency", "max_dominant_enrichment_score", "min_cpb_pvalue"] if c in out.columns]
    if sort_cols:
        ascending = [False if c != "min_cpb_pvalue" else True for c in sort_cols]
        out = out.sort_values(sort_cols, ascending=ascending)
    return out


def _context_overview(master: pd.DataFrame, context_col: str) -> pd.DataFrame:
    if master.empty:
        return pd.DataFrame()
    rows = []
    for ctx, sub in master.groupby(context_col, dropna=False):
        pairs = sub.drop_duplicates("pair")
        genes = set(pairs["gene_1"].astype(str)) | set(pairs["gene_2"].astype(str))
        row = {
            context_col: ctx,
            "n_pairs": int(pairs["pair"].nunique()),
            "n_genes_in_pairs": int(len(genes)),
        }
        if "cpb_pvalue" in pairs.columns:
            vals = pd.to_numeric(pairs["cpb_pvalue"], errors="coerce")
            row.update({
                "min_cpb_pvalue": vals.min(),
                "median_cpb_pvalue": vals.median(),
                "n_pairs_cpb_lt_1e_4": int((vals < 1e-4).sum()),
                "n_pairs_cpb_lt_1e_6": int((vals < 1e-6).sum()),
            })
        if "n_cells_pp_significant" in pairs.columns:
            vals = pd.to_numeric(pairs["n_cells_pp_significant"], errors="coerce")
            row.update({
                "median_n_cells_pp_significant": vals.median(),
                "max_n_cells_pp_significant": vals.max(),
            })
        if "dominant_region" in pairs.columns:
            regs = pairs["dominant_region"].fillna("unannotated").astype(str).value_counts()
            row["top_dominant_region"] = str(regs.index[0]) if len(regs) else "none"
            row["dominant_region_counts"] = ";".join(f"{r}:{int(c)}" for r, c in regs.items())
        rows.append(row)
    return pd.DataFrame(rows).sort_values("n_pairs", ascending=False)


def _region_specific_pairs(master: pd.DataFrame) -> pd.DataFrame:
    if master.empty or "dominant_region" not in master.columns:
        return pd.DataFrame()
    rows = []
    for region, sub in master.dropna(subset=["dominant_region"]).groupby("dominant_region", sort=False):
        for _, r in sub.iterrows():
            rows.append({
                "dominant_region": region,
                "dataset": r.get("dataset", ""),
                "group": r.get("group", ""),
                "pair": r.get("pair", ""),
                "gene_1": r.get("gene_1", ""),
                "gene_2": r.get("gene_2", ""),
                "dominant_enrichment_score": r.get("dominant_enrichment_score", np.nan),
                "cpb_pvalue": r.get("cpb_pvalue", np.nan),
                "n_proximal_pairs": r.get("n_proximal_pairs", np.nan),
            })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["dominant_region", "dominant_enrichment_score"], ascending=[True, False])
    return out


def _top_pairs_per_context(master: pd.DataFrame, context_col: str, top_n: int = 50) -> pd.DataFrame:
    if master.empty:
        return pd.DataFrame()
    sort_col = _first_existing(list(master.columns), ["cpb_pvalue", "dominant_enrichment_score", "n_cells_pp_significant"])
    rows = []
    for ctx, sub in master.groupby(context_col, dropna=False):
        sub = sub.drop_duplicates("pair").copy()
        if sort_col == "cpb_pvalue":
            sub = sub.sort_values(sort_col, ascending=True)
        elif sort_col:
            sub = sub.sort_values(sort_col, ascending=False)
        sub = sub.head(top_n).copy()
        sub.insert(0, "rank_in_context", range(1, len(sub) + 1))
        rows.append(sub)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _write_catalog(out_dir: Path, paths: dict[str, Path], mode: str) -> Path:
    descriptions = {
        "master": "Long-format table containing every significant pair from every group/context, merged with region annotation when available.",
        "presence_matrix": "Pair-by-group binary matrix. 1 means the pair is significant in that group.",
        "pair_summary": "Per-pair summary across groups, including shared/specific flags, presence fraction, CPB summaries, and region consistency.",
        "shared_pairs": "Pairs detected in at least the requested number of groups/contexts.",
        "specific_pairs": "Pairs detected in exactly one group/context.",
        "ubiquitous_pairs": "Pairs detected in all available groups/contexts.",
        "region_consistency": "Per-pair dominant-region consistency across groups/contexts.",
        "region_shifted_pairs": "Pairs whose dominant region differs across groups/contexts.",
        "region_distribution": "Counts and fractions of dominant regions within each group/context.",
        "region_enrichment_long": "Long-format enrichment table for nuclear/perinuclear/cytosolic/peripheral values.",
        "region_specific_pairs": "All annotated pairs organized by dominant region.",
        "group_overview": "Per-group summary, including number of pairs, genes, p-value summaries, and dominant-region counts.",
        "group_overlap_matrix": "Number of shared pairs between every two groups/contexts.",
        "group_jaccard_matrix": "Jaccard similarity of pair sets between every two groups/contexts.",
        "group_similarity_long": "Long-format overlap/Jaccard table for group/context pairs.",
        "hub_genes_overall": "Gene-level degree table over all groups/contexts.",
        "hub_genes_by_group": "Gene-level degree table within each group/context.",
        "top_pairs_per_group": "Top-ranked significant pairs within each group/context.",
        "multi_context_pair_summary": "Per-pair summary across datasets/groups for metadata mode.",
        "cross_dataset_presence_matrix": "Pair-by-dataset binary matrix.",
        "dataset_specific_pairs": "Pairs detected in exactly one dataset.",
        "conserved_pairs": "Pairs detected in at least the requested number of datasets.",
        "region_conserved_pairs": "Conserved pairs whose dominant region is consistent above the threshold.",
        "region_shifted_across_contexts": "Pairs with different dominant regions across datasets/groups.",
        "species_conserved_pairs": "Pairs detected in more than one species, when species metadata are provided.",
        "tissue_conserved_pairs": "Pairs detected in more than one tissue, when tissue metadata are provided.",
        "dataset_overlap_matrix": "Number of shared pairs between every two datasets.",
        "dataset_jaccard_matrix": "Jaccard similarity of pair sets between every two datasets.",
        "dataset_similarity_long": "Long-format dataset overlap/Jaccard table.",
        "gene_hubs_multi_dataset": "Gene-level degree table across all datasets/groups.",
        "summary_json": "Machine-readable summary of high-level counts and output paths.",
    }
    rows = []
    for name, path in paths.items():
        rows.append({
            "output_key": name,
            "file": str(path),
            "mode": mode,
            "description": descriptions.get(name, "Generated summary output."),
        })
    catalog = pd.DataFrame(rows)
    catalog_path = out_dir / "analysis_catalog.csv"
    catalog.to_csv(catalog_path, index=False)
    return catalog_path


def summarize_one_groupby_dir(
    input_dir: str | Path,
    out_dir: str | Path,
    *,
    dataset: str | None = None,
    min_shared_groups: int = 2,
    region_consistency_threshold: float = 0.8,
    top_n: int = 50,
) -> dict[str, Path]:
    input_dir = Path(input_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    for group, d in _discover_group_dirs(input_dir):
        frames.append(_read_one_result_dir(d, dataset=dataset or input_dir.name, group=group))
    master = pd.concat([f for f in frames if f is not None and not f.empty], ignore_index=True) if frames else pd.DataFrame()

    paths: dict[str, Path] = {}
    paths["master"] = _safe_to_csv(master, out_dir / "coloc_master_table.csv")

    if master.empty:
        for key, fname in {
            "presence_matrix": "pair_presence_matrix.csv",
            "pair_summary": "pair_summary.csv",
            "shared_pairs": "shared_pairs.csv",
            "specific_pairs": "specific_pairs.csv",
            "ubiquitous_pairs": "ubiquitous_pairs.csv",
            "region_consistency": "region_consistency.csv",
            "region_shifted_pairs": "region_shifted_pairs.csv",
            "region_distribution": "region_distribution_by_group.csv",
            "region_enrichment_long": "region_enrichment_long.csv",
            "region_specific_pairs": "region_specific_pairs.csv",
            "group_overview": "group_overview.csv",
            "group_overlap_matrix": "group_overlap_matrix.csv",
            "group_jaccard_matrix": "group_jaccard_matrix.csv",
            "group_similarity_long": "group_similarity_long.csv",
            "hub_genes_overall": "hub_genes_overall.csv",
            "hub_genes_by_group": "hub_genes_by_group.csv",
            "top_pairs_per_group": "top_pairs_per_group.csv",
        }.items():
            paths[key] = _safe_to_csv(pd.DataFrame(), out_dir / fname)
        summary = {"mode": "groupby", "n_input_rows": 0, "outputs": {k: str(v) for k, v in paths.items()}}
        paths["summary_json"] = out_dir / "summary_report.json"
        paths["summary_json"].write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        paths["catalog"] = _write_catalog(out_dir, paths, "groupby")
        return paths

    # Core tables.
    presence = _presence_matrix(master, "group")
    paths["presence_matrix"] = _safe_to_csv(presence, out_dir / "pair_presence_matrix.csv")

    pair_summary = _pair_context_summary(master, "group")
    paths["pair_summary"] = _safe_to_csv(pair_summary, out_dir / "pair_summary.csv")

    n_groups_total = master["group"].dropna().astype(str).nunique()
    n_col = "n_groups_detected"
    shared = pair_summary[pair_summary[n_col] >= int(min_shared_groups)].copy() if n_col in pair_summary.columns else pd.DataFrame()
    specific = pair_summary[pair_summary["is_specific"] == True].copy() if "is_specific" in pair_summary.columns else pd.DataFrame()
    ubiquitous = pair_summary[pair_summary["is_ubiquitous"] == True].copy() if "is_ubiquitous" in pair_summary.columns else pd.DataFrame()
    paths["shared_pairs"] = _safe_to_csv(shared, out_dir / "shared_pairs.csv")
    paths["specific_pairs"] = _safe_to_csv(specific, out_dir / "specific_pairs.csv")
    paths["ubiquitous_pairs"] = _safe_to_csv(ubiquitous, out_dir / "ubiquitous_pairs.csv")

    region_consistency = pair_summary[[c for c in [
        "pair", "gene_1", "gene_2", n_col, "groups_detected", "consensus_region",
        "region_consistency", "n_dominant_regions", "region_counts", "is_region_consistent", "is_region_shifted",
    ] if c in pair_summary.columns]].copy()
    paths["region_consistency"] = _safe_to_csv(region_consistency, out_dir / "region_consistency.csv")
    shifted = region_consistency[region_consistency.get("is_region_shifted", pd.Series(False, index=region_consistency.index)) == True].copy() if not region_consistency.empty else pd.DataFrame()
    paths["region_shifted_pairs"] = _safe_to_csv(shifted, out_dir / "region_shifted_pairs.csv")

    paths["region_distribution"] = _safe_to_csv(_region_distribution(master, "group"), out_dir / "region_distribution_by_group.csv")
    paths["region_enrichment_long"] = _safe_to_csv(_region_enrichment_long(master), out_dir / "region_enrichment_long.csv")
    paths["region_specific_pairs"] = _safe_to_csv(_region_specific_pairs(master), out_dir / "region_specific_pairs.csv")
    paths["group_overview"] = _safe_to_csv(_context_overview(master, "group"), out_dir / "group_overview.csv")

    overlap, jaccard, sim_long = _overlap_and_jaccard(master, "group")
    paths["group_overlap_matrix"] = _safe_to_csv(overlap, out_dir / "group_overlap_matrix.csv")
    paths["group_jaccard_matrix"] = _safe_to_csv(jaccard, out_dir / "group_jaccard_matrix.csv")
    paths["group_similarity_long"] = _safe_to_csv(sim_long, out_dir / "group_similarity_long.csv")

    paths["hub_genes_overall"] = _safe_to_csv(_gene_hubs(master), out_dir / "hub_genes_overall.csv")
    paths["hub_genes_by_group"] = _safe_to_csv(_gene_hubs(master, "group"), out_dir / "hub_genes_by_group.csv")
    paths["top_pairs_per_group"] = _safe_to_csv(_top_pairs_per_context(master, "group", top_n=top_n), out_dir / "top_pairs_per_group.csv")

    summary = {
        "mode": "groupby",
        "input_dir": str(input_dir),
        "dataset": dataset or input_dir.name,
        "n_groups_total": int(n_groups_total),
        "n_rows_master": int(len(master)),
        "n_unique_pairs": int(master["pair"].nunique()),
        "n_shared_pairs": int(len(shared)),
        "n_specific_pairs": int(len(specific)),
        "n_ubiquitous_pairs": int(len(ubiquitous)),
        "region_consistency_threshold": float(region_consistency_threshold),
        "n_region_shifted_pairs": int(len(shifted)),
        "outputs": {k: str(v) for k, v in paths.items()},
    }
    paths["summary_json"] = out_dir / "summary_report.json"
    paths["summary_json"].write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    paths["catalog"] = _write_catalog(out_dir, paths, "groupby")
    return paths


def summarize_from_metadata(
    metadata_csv: str | Path,
    out_dir: str | Path,
    *,
    min_conserved_datasets: int = 2,
    region_consistency_threshold: float = 0.8,
    top_n: int = 50,
) -> dict[str, Path]:
    """Summarize multiple result directories.

    Required metadata columns:
        dataset,result_dir
    Optional columns are preserved, such as species,tissue,technology,group_level.
    """
    meta = pd.read_csv(metadata_csv)
    if not {"dataset", "result_dir"}.issubset(meta.columns):
        raise ValueError("metadata CSV must contain columns: dataset,result_dir")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    for _, row in meta.iterrows():
        dataset = str(row["dataset"])
        result_dir = Path(str(row["result_dir"]))
        md = row.to_dict()
        for group, d in _discover_group_dirs(result_dir):
            frames.append(_read_one_result_dir(d, dataset=dataset, group=group, metadata=md))

    master = pd.concat([f for f in frames if f is not None and not f.empty], ignore_index=True) if frames else pd.DataFrame()
    paths: dict[str, Path] = {}
    paths["master"] = _safe_to_csv(master, out_dir / "multi_dataset_coloc_master_table.csv")

    if master.empty:
        paths["summary_json"] = out_dir / "summary_report.json"
        paths["summary_json"].write_text(json.dumps({"mode": "metadata", "n_input_rows": 0}, indent=2), encoding="utf-8")
        paths["catalog"] = _write_catalog(out_dir, paths, "metadata")
        return paths

    # Context summary across all dataset+group combinations.
    master["context"] = master["dataset"].astype(str) + "::" + master["group"].astype(str)
    context_summary = _pair_context_summary(master, "context")
    paths["multi_context_pair_summary"] = _safe_to_csv(context_summary, out_dir / "multi_context_pair_summary.csv")

    # Dataset-level conservation.
    dataset_summary = _pair_context_summary(master.drop_duplicates(["dataset", "pair"]), "dataset")
    if "n_datasets_detected" in dataset_summary.columns:
        conserved = dataset_summary[dataset_summary["n_datasets_detected"] >= int(min_conserved_datasets)].copy()
        dataset_specific = dataset_summary[dataset_summary["is_specific"] == True].copy()
    else:
        conserved = pd.DataFrame()
        dataset_specific = pd.DataFrame()
    paths["conserved_pairs"] = _safe_to_csv(conserved, out_dir / "conserved_colocalized_pairs.csv")
    paths["dataset_specific_pairs"] = _safe_to_csv(dataset_specific, out_dir / "dataset_specific_pairs.csv")

    region_conserved = conserved.copy()
    if "region_consistency" in region_conserved.columns:
        region_conserved = region_conserved[region_conserved["region_consistency"] >= float(region_consistency_threshold)]
    paths["region_conserved_pairs"] = _safe_to_csv(region_conserved, out_dir / "region_conserved_pairs.csv")

    shifted = context_summary[context_summary.get("is_region_shifted", pd.Series(False, index=context_summary.index)) == True].copy() if not context_summary.empty else pd.DataFrame()
    paths["region_shifted_across_contexts"] = _safe_to_csv(shifted, out_dir / "region_shifted_across_contexts.csv")

    paths["cross_dataset_presence_matrix"] = _safe_to_csv(_presence_matrix(master.drop_duplicates(["dataset", "pair"]), "dataset"), out_dir / "cross_dataset_presence_matrix.csv")

    # Optional species/tissue conservation.
    if "species" in master.columns:
        species_summary = _pair_context_summary(master.dropna(subset=["species"]).drop_duplicates(["species", "pair"]), "species")
        species_cons = species_summary[species_summary.get("n_speciess_detected", pd.Series(0, index=species_summary.index)) >= 2].copy() if not species_summary.empty else pd.DataFrame()
        # Cleaner fallback if plural column name is awkward: detect any n_* column.
        ncols = [c for c in species_summary.columns if c.startswith("n_") and c.endswith("s_detected")]
        if ncols:
            species_cons = species_summary[species_summary[ncols[0]] >= 2].copy()
        paths["species_conserved_pairs"] = _safe_to_csv(species_cons, out_dir / "species_conserved_pairs.csv")
    if "tissue" in master.columns:
        tissue_summary = _pair_context_summary(master.dropna(subset=["tissue"]).drop_duplicates(["tissue", "pair"]), "tissue")
        ncols = [c for c in tissue_summary.columns if c.startswith("n_") and c.endswith("s_detected")]
        tissue_cons = tissue_summary[tissue_summary[ncols[0]] >= 2].copy() if ncols else pd.DataFrame()
        paths["tissue_conserved_pairs"] = _safe_to_csv(tissue_cons, out_dir / "tissue_conserved_pairs.csv")

    paths["region_distribution"] = _safe_to_csv(_region_distribution(master, "context"), out_dir / "region_distribution_by_context.csv")
    paths["region_enrichment_long"] = _safe_to_csv(_region_enrichment_long(master), out_dir / "region_enrichment_long.csv")
    paths["region_specific_pairs"] = _safe_to_csv(_region_specific_pairs(master), out_dir / "region_specific_pairs.csv")
    paths["dataset_overview"] = _safe_to_csv(_context_overview(master.drop_duplicates(["dataset", "pair"]), "dataset"), out_dir / "dataset_overview.csv")
    paths["context_overview"] = _safe_to_csv(_context_overview(master, "context"), out_dir / "context_overview.csv")

    overlap, jaccard, sim_long = _overlap_and_jaccard(master.drop_duplicates(["dataset", "pair"]), "dataset")
    paths["dataset_overlap_matrix"] = _safe_to_csv(overlap, out_dir / "dataset_overlap_matrix.csv")
    paths["dataset_jaccard_matrix"] = _safe_to_csv(jaccard, out_dir / "dataset_jaccard_matrix.csv")
    paths["dataset_similarity_long"] = _safe_to_csv(sim_long, out_dir / "dataset_similarity_long.csv")

    paths["gene_hubs_multi_dataset"] = _safe_to_csv(_gene_hubs(master), out_dir / "gene_hubs_multi_dataset.csv")
    paths["hub_genes_by_context"] = _safe_to_csv(_gene_hubs(master, "context"), out_dir / "hub_genes_by_context.csv")
    paths["top_pairs_per_context"] = _safe_to_csv(_top_pairs_per_context(master, "context", top_n=top_n), out_dir / "top_pairs_per_context.csv")

    summary = {
        "mode": "metadata",
        "metadata_csv": str(metadata_csv),
        "n_datasets_total": int(master["dataset"].dropna().astype(str).nunique()),
        "n_contexts_total": int(master["context"].dropna().astype(str).nunique()),
        "n_rows_master": int(len(master)),
        "n_unique_pairs": int(master["pair"].nunique()),
        "n_conserved_pairs": int(len(conserved)),
        "n_dataset_specific_pairs": int(len(dataset_specific)),
        "n_region_conserved_pairs": int(len(region_conserved)),
        "n_region_shifted_pairs": int(len(shifted)),
        "min_conserved_datasets": int(min_conserved_datasets),
        "region_consistency_threshold": float(region_consistency_threshold),
        "outputs": {k: str(v) for k, v in paths.items()},
    }
    paths["summary_json"] = out_dir / "summary_report.json"
    paths["summary_json"].write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    paths["catalog"] = _write_catalog(out_dir, paths, "metadata")
    return paths
