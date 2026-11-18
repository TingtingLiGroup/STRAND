"""Organelle marker gene-set enrichment and automatic semantic compartment naming."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


_DEFAULT_MARKER_CSV = Path(__file__).resolve().parent / "gene_sets" / "organelle_markers.csv"

# Nested reference sets share many genes, so a narrow win of a sub-compartment over its sibling is
# reported at the parent level instead of over-claiming the finer organelle.
_PARENT_ORGANELLE = {
    "Nucleolar": "Nuclear",
    "Nuclear-Edge": "Nuclear",
    "Peripheral-Cytosolic": "Cytosolic",
}


def load_organelle_markers(marker_csv: Path | str | None = None) -> pd.DataFrame:
    """Load marker weights: the built-in Fazal2019 + Xia2019 + canonical table, or a user CSV.

    A user CSV needs ``gene`` and ``compartment`` columns; ``weight`` defaults to 1.0 when absent.
    """
    path = Path(marker_csv) if marker_csv is not None else _DEFAULT_MARKER_CSV
    if marker_csv is not None and not path.exists():
        raise FileNotFoundError(f"Marker CSV not found: {path}")
    if not path.exists():
        return pd.DataFrame(columns=["gene", "compartment", "weight", "reference"])
    df = pd.read_csv(path)
    missing = {"gene", "compartment"} - set(df.columns)
    if missing:
        raise ValueError(f"Marker CSV {path} lacks required columns: {sorted(missing)}")
    if "weight" not in df.columns:
        df["weight"] = 1.0
    df["gene"] = df["gene"].astype(str).str.upper()
    df["compartment"] = df["compartment"].astype(str)
    df["weight"] = pd.to_numeric(df["weight"], errors="coerce").fillna(0.0)
    return df


def _radial_domain_name(radial_frac: float) -> str:
    """Purely cell-boundary radial name used when marker evidence is absent or ambiguous."""
    if radial_frac < 0.38:
        return "Central-Domain"
    if radial_frac < 0.68:
        return "Intermediate-Domain"
    return "Peripheral-Domain"


def annotate_compartments(
    compartment_gene_scores: np.ndarray,
    gene_names: Sequence[str],
    *,
    cluster_RadialFrac: np.ndarray | None = None,
    posthoc_nuclear_frac: np.ndarray | None = None,
    marker_df: pd.DataFrame | None = None,
    min_enrichment_z: float = 2.0,
    min_marker_genes: int = 5,
    min_name_margin: float = 1.0,
    top_n_genes: int = 8,
) -> pd.DataFrame:
    """Assign interpretable organelle names and top marker genes to K compartments.

    An organelle name is given only when its reference set has >= ``min_marker_genes`` genes on the
    panel, its enrichment Z >= ``min_enrichment_z``, and it leads the runner-up by >= ``min_name_margin``.
    A narrow lead over a sibling set falls back to the parent organelle (e.g. Nucleolar -> Nuclear);
    otherwise the compartment keeps a neutral radial name and the scores stay in the table.

    Note: ``posthoc_nuclear_frac`` is strictly recorded as a post-hoc QC column when
    nuclear boundaries exist in the bundle; it NEVER alters cluster assignments or
    marker-based semantic naming when marker genes overlap the panel.
    """
    k_count, n_genes = compartment_gene_scores.shape
    genes_arr = np.asarray([str(g) for g in gene_names])
    genes_upper = np.asarray([g.upper() for g in genes_arr])

    if marker_df is None:
        marker_df = load_organelle_markers()

    # Build (n_genes, n_organelles) reference weight matrix on overlapping genes.
    organelles: list[str] = []
    W_ref = np.zeros((n_genes, 0), dtype=np.float64)
    overlap_counts: dict[str, int] = {}

    if not marker_df.empty and n_genes > 0:
        gene_to_idx: dict[str, list[int]] = {}
        for idx, g_up in enumerate(genes_upper):
            gene_to_idx.setdefault(g_up, []).append(idx)

        sub = marker_df[marker_df["gene"].isin(gene_to_idx)]
        if not sub.empty:
            organelles = sorted(sub["compartment"].unique().tolist())
            org_to_col = {o: c for c, o in enumerate(organelles)}
            W_ref = np.zeros((n_genes, len(organelles)), dtype=np.float64)
            for row in sub.itertuples(index=False):
                c_idx = org_to_col[row.compartment]
                for g_idx in gene_to_idx[row.gene]:
                    W_ref[g_idx, c_idx] = max(W_ref[g_idx, c_idx], float(row.weight))
            for c_idx, org in enumerate(organelles):
                overlap_counts[org] = int(np.sum(W_ref[:, c_idx] > 0))

    # Row-standardize compartment gene specificity across genes.
    S = np.asarray(compartment_gene_scores, dtype=np.float64)
    row_mean = S.mean(axis=1, keepdims=True)
    row_std = np.maximum(S.std(axis=1, keepdims=True), 1e-8)
    S_z = (S - row_mean) / row_std

    # Analytical permutation Z-score of weighted sum S_z @ w_org under random gene order.
    # Sets with too few panel genes stay NaN: a couple of genes can give a large but meaningless Z.
    enrichment_z = np.full((k_count, len(organelles)), np.nan, dtype=np.float64)
    for c_idx in range(len(organelles)):
        w = W_ref[:, c_idx]
        n_pos = int(np.sum(w > 0))
        if n_pos < max(2, min_marker_genes) or n_genes < 3:
            continue
        w_centered = w - w.mean()
        denom = np.linalg.norm(w_centered) * np.sqrt(n_genes / max(n_genes - 1, 1))
        if denom > 1e-12:
            # Because each row of S_z has mean 0 and unit variance, E=0 and SD=||w - mean(w)||.
            enrichment_z[:, c_idx] = (S_z @ w_centered) / np.linalg.norm(w_centered)

    rows = []
    used_names: dict[str, int] = {}

    for k in range(k_count):
        order = np.argsort(-S[k])
        top_idx = [int(i) for i in order[: min(top_n_genes, n_genes)] if S[k, i] > 0]
        if not top_idx:
            top_idx = [int(i) for i in order[: min(top_n_genes, n_genes)]]
        top_genes_str = ",".join(genes_arr[i] for i in top_idx)

        best_org = f"Domain-{k}"
        if cluster_RadialFrac is not None and np.isfinite(cluster_RadialFrac[k]):
            best_org = _radial_domain_name(float(cluster_RadialFrac[k]))
        best_z = 0.0
        best_n_markers = 0
        second_org = ""
        second_z = 0.0

        scored = np.flatnonzero(np.isfinite(enrichment_z[k]))
        if scored.size > 0:
            z_vec = enrichment_z[k]
            z_ord = scored[np.argsort(-z_vec[scored])]
            best_idx = int(z_ord[0])
            best_z = float(z_vec[best_idx])
            best_n_markers = overlap_counts[organelles[best_idx]]
            if len(z_ord) > 1:
                second_org = organelles[int(z_ord[1])]
                second_z = float(z_vec[int(z_ord[1])])
            if best_z >= min_enrichment_z:
                top_org = organelles[best_idx]
                if not second_org or best_z - second_z >= min_name_margin:
                    best_org = top_org
                else:
                    top_parent = _PARENT_ORGANELLE.get(top_org, top_org)
                    if top_parent == _PARENT_ORGANELLE.get(second_org, second_org):
                        best_org = top_parent

        cnt = used_names.get(best_org, 0) + 1
        used_names[best_org] = cnt
        unique_label = best_org if cnt == 1 else f"{best_org}-{cnt}"

        rec: dict[str, object] = {
            "subdomain": int(k),
            "compartment_name": unique_label,
            "base_organelle": best_org,
            "organelle_enrichment_z": round(best_z, 4),
            "organelle_n_markers": int(best_n_markers),
            "runner_up_organelle": second_org,
            "runner_up_z": round(second_z, 4),
            "top_enriched_genes": top_genes_str,
        }
        if cluster_RadialFrac is not None:
            rec["mean_cell_radial_pos"] = round(float(cluster_RadialFrac[k]), 4)
        if posthoc_nuclear_frac is not None:
            rec["posthoc_nuclear_fraction"] = (
                round(float(posthoc_nuclear_frac[k]), 4)
                if np.isfinite(posthoc_nuclear_frac[k])
                else np.nan
            )
        for c_idx, org in enumerate(organelles):
            col_key = f"z_{org.lower().replace('-', '_').replace(' ', '_')}"
            rec[col_key] = round(float(enrichment_z[k, c_idx]), 4)
        rows.append(rec)

    return pd.DataFrame(rows)
