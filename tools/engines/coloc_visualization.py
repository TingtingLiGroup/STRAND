# tools/engines/coloc_visualization.py

from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Patch
from matplotlib.lines import Line2D

import networkx as nx
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------


def _safe_filename(x: str) -> str:
    x = str(x)
    x = re.sub(r"[^\w\-.]+", "_", x)
    return x.strip("_") or "gene"


# def _safe_neglog10(p):
#     p = np.asarray(p, dtype=float)
#     finite_pos = p[np.isfinite(p) & (p > 0)]
#     floor = np.min(finite_pos) if finite_pos.size else 1e-300
#     p = np.where((p <= 0) | (~np.isfinite(p)), floor, p)
#     return -np.log10(p)
def _safe_neglog10(p):
    p = np.asarray(p, dtype=float)
    p = np.clip(p, 0.0, 1.0)

    finite_pos = p[np.isfinite(p) & (p > 0)]
    floor = np.min(finite_pos) if finite_pos.size else 1e-300

    p = np.where((p <= 0) | (~np.isfinite(p)), floor, p)
    return -np.log10(p)


def _load_gene_list(out_dir: Path) -> list[str]:
    path = out_dir / "instant_gene_list.csv"
    if not path.exists():
        raise FileNotFoundError(f"Cannot find gene list: {path}")
    return pd.read_csv(path)["gene"].astype(str).tolist()


def _get_gene_list(out_dir: Path, gene_list: Optional[list[str]] = None) -> list[str]:
    if gene_list is not None:
        return [str(g) for g in gene_list]
    return _load_gene_list(out_dir)


def _get_array(out_dir: Path, value, filename: str):
    if value is not None:
        return np.asarray(value)
    path = out_dir / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Cannot find {path}. Pass arrays directly or save matrix outputs."
        )
    return np.load(path)


# -----------------------------------------------------------------------------
# Tables
# -----------------------------------------------------------------------------


def export_all_pairs_table(
    out_dir: str | Path,
    *,
    gene_list: Optional[list[str]] = None,
    cpb_pvals=None,
    expected_coloc=None,
) -> Path:
    """Export all CPB-tested gene pairs, not only significant ones."""
    out_dir = Path(out_dir)
    genes = _get_gene_list(out_dir, gene_list)
    cpb = _get_array(out_dir, cpb_pvals, "instant_cpb_pvals.npy")
    expected = _get_array(out_dir, expected_coloc, "instant_expected_coloc.npy")

    rows = []
    for i in range(len(genes)):
        for j in range(i + 1, len(genes)):
            p = float(cpb[i, j])
            rows.append(
                {
                    "gene_1": genes[i],
                    "gene_2": genes[j],
                    "cpb_pvalue": p,
                    "neglog10_cpb_pvalue": float(_safe_neglog10([p])[0]),
                    "expected_coloc": float(expected[i, j]),
                }
            )

    df = pd.DataFrame(rows).sort_values("cpb_pvalue").reset_index(drop=True)
    save_path = out_dir / "instant_all_pairs.csv"
    df.to_csv(save_path, index=False)
    return save_path


# -----------------------------------------------------------------------------
# Heatmap
# -----------------------------------------------------------------------------

def plot_cpb_heatmap(
    out_dir: str | Path,
    *,
    gene_list: Optional[list[str]] = None,
    cpb_pvals=None,
    top_n_genes: int | None = 80,
    vmax: float = 2.2,
    cmap: str = "YlGnBu",
) -> Path:
    """Plot a CPB heatmap in an InSTAnT-paper-like style."""
    out_dir = Path(out_dir)
    genes = _get_gene_list(out_dir, gene_list)
    cpb = _get_array(out_dir, cpb_pvals, "instant_cpb_pvals.npy")

    if len(genes) == 0:
        raise ValueError("No genes available for heatmap.")

    mat = _safe_neglog10(cpb)
    n = len(genes)

    if top_n_genes is not None and top_n_genes > 0 and top_n_genes < n:
        tmp = mat.copy()
        np.fill_diagonal(tmp, np.nan)
        gene_score = np.nanmax(tmp, axis=1)
        idx = np.argsort(gene_score)[::-1][:top_n_genes]
    else:
        idx = np.arange(n)

    idx = np.array(sorted(idx, key=lambda i: genes[i]))
    mat_plot = mat[np.ix_(idx, idx)].copy()
    genes_plot = [genes[i] for i in idx]

    mat_plot = np.clip(mat_plot, 0, vmax)
    np.fill_diagonal(mat_plot, vmax)

    fig_w = max(8.0, min(13.0, len(genes_plot) * 0.16))
    fig_h = max(7.0, min(12.0, len(genes_plot) * 0.16))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(mat_plot, aspect="equal", interpolation="nearest", cmap=cmap, vmin=0, vmax=vmax)
    ax.set_title("-log10(Global P-values) Co-localization", fontsize=14, pad=10)
    ax.set_xlabel("Genes", fontsize=11)
    ax.set_ylabel("Genes", fontsize=11)
    ax.set_xticks(range(len(genes_plot)))
    ax.set_yticks(range(len(genes_plot)))
    ax.set_xticklabels(genes_plot, rotation=90, fontsize=6)
    ax.set_yticklabels(genes_plot, fontsize=7)
    ax.tick_params(length=0)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("-log10(Global P-values)", fontsize=10)

    fig.tight_layout()
    save_path = out_dir / f"viz_cpb_heatmap_top{len(genes_plot)}.png"
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    return save_path


# -----------------------------------------------------------------------------
# Network
# -----------------------------------------------------------------------------
# def plot_coloc_network(
#     out_dir: str | Path,
#     *,
#     significant_pairs: Optional[pd.DataFrame] = None,
#     region_annotated_pairs: Optional[pd.DataFrame] = None,
#     max_edges: int = 80,
#     seed: int = 7,
#     keep_largest_component: bool = False,
# ) -> Path | None:
#     """Plot the final global colocalization network with ellipse nodes.

#     Final visual encoding:
#     - node = gene
#     - edge = significant colocalized gene pair
#     - edge width / alpha = old rank-based significance logic
#     - edge color = dominant subcellular region when region annotation is available
#     """
#     out_dir = Path(out_dir)

#     def _load_table(path: Path) -> pd.DataFrame:
#         if not path.exists():
#             raise FileNotFoundError(f"Cannot find: {path}")
#         return pd.read_csv(path)

#     # Prefer region-annotated pairs because they contain dominant_region.
#     if region_annotated_pairs is None:
#         region_path = out_dir / "instant_region_annotated_pairs.csv"
#         region_df = _load_table(region_path) if region_path.exists() else None
#     else:
#         region_df = region_annotated_pairs.copy()

#     if significant_pairs is None:
#         pairs_path = out_dir / "instant_significant_pairs.csv"
#         sig_df = _load_table(pairs_path) if pairs_path.exists() else pd.DataFrame()
#     else:
#         sig_df = significant_pairs.copy()

#     df = region_df.copy() if isinstance(region_df, pd.DataFrame) and not region_df.empty else sig_df.copy()

#     if df.empty:
#         print("[viz] significant pair table is empty. Network skipped.")
#         return None

#     required = {"gene_1", "gene_2", "cpb_pvalue"}
#     missing = required - set(df.columns)
#     if missing:
#         raise ValueError(f"network pair table missing columns: {missing}")

#     # ------------------------------------------------------------
#     # 1) Ranking and p-value handling
#     # ------------------------------------------------------------
#     if "cpb_pvalue_raw" not in df.columns:
#         df["cpb_pvalue_raw"] = pd.to_numeric(df["cpb_pvalue"], errors="coerce")

#     df["cpb_pvalue_raw"] = pd.to_numeric(df["cpb_pvalue_raw"], errors="coerce")

#     df["cpb_pvalue"] = (
#         pd.to_numeric(df["cpb_pvalue"], errors="coerce")
#         .fillna(1.0)
#         .clip(lower=0.0, upper=1.0)
#     )

#     if "dominant_region" not in df.columns:
#         df["dominant_region"] = "unannotated"

#     df["dominant_region"] = (
#         df["dominant_region"]
#         .fillna("unannotated")
#         .astype(str)
#         .str.lower()
#     )

#     # Use raw values for ranking, because many clipped p-values may become 0.
#     df = df.sort_values("cpb_pvalue_raw", ascending=True).head(int(max_edges)).copy()
#     df = df.reset_index(drop=True)

#     # Valid -log10 value for metadata/debugging.
#     df["neglog10_p"] = _safe_neglog10(df["cpb_pvalue"].values)

#     # OLD VERSION LOGIC:
#     # Rank-based visual strength. 1.0 = strongest among displayed top edges.
#     if len(df) > 1:
#         df["edge_strength"] = 1.0 - np.arange(len(df), dtype=float) / (len(df) - 1)
#     else:
#         df["edge_strength"] = 1.0

#     # Layout weight: stronger edge has stronger attraction in layout.
#     df["layout_weight"] = 0.8 + 3.2 * df["edge_strength"]

#     # Low-saturation region palette, close to the InSTAnT-paper style.
#     region_color_map = {
#         "nuclear": "#9ecae1",
#         "perinuclear": "#a1d99b",
#         "cytosolic": "#f4a6b7",
#         "peripheral": "#c7b9e6",
#         "unassigned": "#bdbdbd",
#         "unannotated": "#bdbdbd",
#     }

#     # ------------------------------------------------------------
#     # 2) Build graph
#     # ------------------------------------------------------------
#     G = nx.Graph()

#     for _, row in df.iterrows():
#         g1 = str(row["gene_1"])
#         g2 = str(row["gene_2"])

#         raw_val = row["cpb_pvalue_raw"]
#         raw_val = float(raw_val) if pd.notna(raw_val) else float(row["cpb_pvalue"])

#         region = str(row.get("dominant_region", "unannotated")).lower()
#         color = region_color_map.get(region, "#bdbdbd")

#         G.add_edge(
#             g1,
#             g2,
#             cpb_pvalue=float(row["cpb_pvalue"]),
#             cpb_pvalue_raw=raw_val,
#             neglog10_p=float(row["neglog10_p"]),
#             edge_strength=float(row["edge_strength"]),
#             layout_weight=float(row["layout_weight"]),
#             dominant_region=region,
#             edge_color=color,
#         )

#     if G.number_of_nodes() == 0:
#         print("[viz] network has no nodes. Network skipped.")
#         return None

#     if keep_largest_component:
#         comps = list(nx.connected_components(G))
#         if comps:
#             largest = max(comps, key=len)
#             G = G.subgraph(largest).copy()

#     degrees = dict(G.degree())

#     # ------------------------------------------------------------
#     # 3) Graphviz sfdp layout
#     # ------------------------------------------------------------
#     try:
#         from networkx.drawing.nx_pydot import graphviz_layout

#         H = G.copy()
#         for _, _, data in H.edges(data=True):
#             data["weight"] = float(data.get("layout_weight", 1.0))

#         pos = graphviz_layout(H, prog="sfdp")
#         pos = {
#             k: np.array([float(v[0]), float(v[1])], dtype=float)
#             for k, v in pos.items()
#         }

#         print("[viz] network layout: Graphviz sfdp")

#     except Exception as e:
#         raise RuntimeError(
#             "Graphviz sfdp layout failed. Install graphviz and pydot first:\n"
#             "conda install -c conda-forge graphviz pydot -y\n"
#             "Then confirm with:\n"
#             "which sfdp\n"
#             "which neato"
#         ) from e

#     # ------------------------------------------------------------
#     # 4) Normalize coordinates
#     # ------------------------------------------------------------
#     nodes = list(G.nodes())
#     coords = np.array([pos[n] for n in nodes], dtype=float)

#     rng = np.random.default_rng(seed)
#     coords += rng.normal(scale=0.02, size=coords.shape)

#     coords -= coords.mean(axis=0, keepdims=True)

#     max_abs = np.abs(coords).max()
#     if max_abs > 0:
#         coords = coords / max_abs * 8.0

#     pos = {n: coords[i] for i, n in enumerate(nodes)}

#     # ------------------------------------------------------------
#     # 5) Node ellipse sizes
#     # ------------------------------------------------------------
#     node_w = {}
#     node_h = {}

#     for node in nodes:
#         label = str(node)
#         deg = degrees.get(node, 1)

#         node_w[node] = max(
#             0.78,
#             0.085 * len(label) + 0.34 + 0.015 * min(deg, 10),
#         )
#         node_h[node] = 0.32 + 0.014 * min(deg, 8)

#     # ------------------------------------------------------------
#     # 6) Soft overlap reduction
#     # ------------------------------------------------------------
#     pad_x = 0.08
#     pad_y = 0.05

#     for _ in range(120):
#         moved = False

#         for i in range(len(nodes)):
#             for j in range(i + 1, len(nodes)):
#                 ni = nodes[i]
#                 nj = nodes[j]

#                 dx = coords[j, 0] - coords[i, 0]
#                 dy = coords[j, 1] - coords[i, 1]

#                 min_dx = (node_w[ni] + node_w[nj]) / 2.0 + pad_x
#                 min_dy = (node_h[ni] + node_h[nj]) / 2.0 + pad_y

#                 ox = min_dx - abs(dx)
#                 oy = min_dy - abs(dy)

#                 if ox > 0 and oy > 0:
#                     moved = True

#                     if abs(dx) < 1e-9 and abs(dy) < 1e-9:
#                         direction = rng.normal(size=2)
#                     else:
#                         direction = np.array([dx, dy], dtype=float)

#                     direction = direction / (np.linalg.norm(direction) + 1e-9)

#                     push = 0.08 * max(ox, oy)
#                     coords[i] -= direction * push
#                     coords[j] += direction * push

#         if not moved:
#             break

#     pos = {n: coords[i] for i, n in enumerate(nodes)}

#     # ------------------------------------------------------------
#     # 7) Edge appearance: OLD rank-based width + NEW region color
#     # ------------------------------------------------------------
#     edge_widths = []
#     edge_alphas = []

#     for _, _, data in G.edges(data=True):
#         s = float(data.get("edge_strength", 0.5))
#         s = float(np.clip(s, 0.0, 1.0))

#         # Exactly the old subtle width/alpha logic.
#         edge_widths.append(0.45 + 1.15 * s)
#         edge_alphas.append(0.22 + 0.30 * s)

#     # ------------------------------------------------------------
#     # 8) Draw
#     # ------------------------------------------------------------
#     fig, ax = plt.subplots(figsize=(13.5, 8.2))

#     for ((u, v, data), width, alpha) in zip(G.edges(data=True), edge_widths, edge_alphas):
#         x1, y1 = pos[u]
#         x2, y2 = pos[v]

#         ax.plot(
#             [x1, x2],
#             [y1, y2],
#             color=data.get("edge_color", "#5f5f5f"),
#             linewidth=width,
#             alpha=alpha,
#             zorder=1,
#         )

#     node_face = "#a9c7df"
#     node_edge = "#666666"

#     for node in G.nodes():
#         x, y = pos[node]
#         label = str(node)

#         ax.add_patch(
#             Ellipse(
#                 (x, y),
#                 width=node_w[node],
#                 height=node_h[node],
#                 facecolor=node_face,
#                 edgecolor=node_edge,
#                 linewidth=0.75,
#                 alpha=0.96,
#                 zorder=2,
#             )
#         )

#         ax.text(
#             x,
#             y,
#             label,
#             ha="center",
#             va="center",
#             fontsize=6.2,
#             color="#333333",
#             zorder=3,
#         )

#     ax.set_title(
#         f"Global d-colocalized gene-pair network\n"
#         f"Top {G.number_of_edges()} CPB-significant edges",
#         fontsize=15,
#         pad=14,
#     )

#     ax.axis("off")
#     ax.set_aspect("equal", adjustable="box")

#     xs = np.array([pos[n][0] for n in G.nodes()])
#     ys = np.array([pos[n][1] for n in G.nodes()])

#     ax.set_xlim(xs.min() - 0.8, xs.max() + 0.8)
#     ax.set_ylim(ys.min() - 0.8, ys.max() + 0.8)

#     # Region color legend only. Edge width follows the old subtle rank style.
#     present_regions = []
#     for region in ["nuclear", "perinuclear", "cytosolic", "peripheral", "unassigned", "unannotated"]:
#         if any(str(d.get("dominant_region", "")) == region for _, _, d in G.edges(data=True)):
#             label = "unannotated/unassigned" if region in {"unassigned", "unannotated"} else region
#             present_regions.append(
#                 Patch(
#                     facecolor=region_color_map.get(region, "#bdbdbd"),
#                     edgecolor="none",
#                     label=label,
#                 )
#             )

#     uniq = {}
#     for h in present_regions:
#         uniq[h.get_label()] = h
#     present_regions = list(uniq.values())

#     if present_regions:
#         ax.legend(
#             handles=present_regions,
#             title="Edge color: region",
#             loc="upper left",
#             bbox_to_anchor=(1.01, 1.00),
#             frameon=False,
#             fontsize=9,
#             title_fontsize=10,
#         )

#     fig.tight_layout()

#     save_path = out_dir / "viz_coloc_network_styled.png"
#     fig.savefig(save_path, dpi=300, bbox_inches="tight")
#     plt.close(fig)

#     return save_path
def plot_coloc_network(
    out_dir: str | Path,
    *,
    significant_pairs: Optional[pd.DataFrame] = None,
    region_annotated_pairs: Optional[pd.DataFrame] = None,
    max_edges: int = 80,
    seed: int = 7,
    keep_largest_component: bool = False,
) -> Path | None:
    """Plot the final global colocalization network.

    Visual encoding:
    - node = gene
    - edge = significant colocalized gene pair
    - edge width = CPB significance rank
    - edge color = dominant subcellular region
    """
    out_dir = Path(out_dir)

    def _load_table(path: Path) -> pd.DataFrame:
        if not path.exists():
            raise FileNotFoundError(f"Cannot find: {path}")
        return pd.read_csv(path)

    # Prefer region-annotated pairs because they contain dominant_region.
    if region_annotated_pairs is None:
        region_path = out_dir / "instant_region_annotated_pairs.csv"
        if region_path.exists():
            region_df = _load_table(region_path)
        else:
            region_df = None
    else:
        region_df = region_annotated_pairs.copy()

    if significant_pairs is None:
        pairs_path = out_dir / "instant_significant_pairs.csv"
        if pairs_path.exists():
            sig_df = _load_table(pairs_path)
        else:
            sig_df = pd.DataFrame()
    else:
        sig_df = significant_pairs.copy()

    df = region_df.copy() if isinstance(region_df, pd.DataFrame) and not region_df.empty else sig_df.copy()

    if df.empty:
        print("[viz] significant pair table is empty. Network skipped.")
        return None

    required = {"gene_1", "gene_2", "cpb_pvalue"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"network pair table missing columns: {missing}")

    # ------------------------------------------------------------
    # 1) P-value and region handling
    # ------------------------------------------------------------
    if "cpb_pvalue_raw" not in df.columns:
        df["cpb_pvalue_raw"] = pd.to_numeric(df["cpb_pvalue"], errors="coerce")

    df["cpb_pvalue_raw"] = pd.to_numeric(df["cpb_pvalue_raw"], errors="coerce")

    df["cpb_pvalue"] = (
        pd.to_numeric(df["cpb_pvalue"], errors="coerce")
        .fillna(1.0)
        .clip(lower=0.0, upper=1.0)
    )

    if "dominant_region" not in df.columns:
        df["dominant_region"] = "unannotated"

    df["dominant_region"] = (
        df["dominant_region"]
        .fillna("unannotated")
        .astype(str)
        .str.lower()
    )

    # Use raw p-value for ranking, because clipped p-values may tie.
    df = df.sort_values("cpb_pvalue_raw", ascending=True).head(int(max_edges)).reset_index(drop=True)

    # Formal -log10 value, mainly for record/debugging.
    df["neglog10_p"] = _safe_neglog10(df["cpb_pvalue"].values)

    # Old rank-based edge strength logic.
    # Strongest displayed edge = 1.0; weakest displayed edge = 0.0.
    if len(df) > 1:
        df["edge_strength"] = 1.0 - np.arange(len(df), dtype=float) / (len(df) - 1)
    else:
        df["edge_strength"] = 1.0

    # Layout weight follows the previous version.
    df["layout_weight"] = 0.8 + 3.2 * df["edge_strength"]

    # ------------------------------------------------------------
    # 2) Region palette
    # ------------------------------------------------------------
    region_color_map = {
        "nuclear": "#9ecae1",
        "perinuclear": "#a1d99b",
        "cytosolic": "#f4a6b7",
        "peripheral": "#c7b9e6",
        "unassigned": "#bdbdbd",
        "unannotated": "#bdbdbd",
    }

    # ------------------------------------------------------------
    # 3) Build graph
    # ------------------------------------------------------------
    G = nx.Graph()

    for _, row in df.iterrows():
        g1 = str(row["gene_1"])
        g2 = str(row["gene_2"])

        raw_val = row["cpb_pvalue_raw"]
        raw_val = float(raw_val) if pd.notna(raw_val) else float(row["cpb_pvalue"])

        region = str(row.get("dominant_region", "unannotated")).lower()
        color = region_color_map.get(region, "#bdbdbd")

        G.add_edge(
            g1,
            g2,
            cpb_pvalue=float(row["cpb_pvalue"]),
            cpb_pvalue_raw=raw_val,
            neglog10_p=float(row["neglog10_p"]),
            edge_strength=float(row["edge_strength"]),
            layout_weight=float(row["layout_weight"]),
            dominant_region=region,
            edge_color=color,
        )

    if G.number_of_nodes() == 0:
        print("[viz] network has no nodes. Network skipped.")
        return None

    if keep_largest_component:
        comps = list(nx.connected_components(G))
        if comps:
            largest = max(comps, key=len)
            G = G.subgraph(largest).copy()

    degrees = dict(G.degree())

    # ------------------------------------------------------------
    # 4) Layout
    # ------------------------------------------------------------
    try:
        from networkx.drawing.nx_pydot import graphviz_layout

        H = G.copy()
        for _, _, data in H.edges(data=True):
            data["weight"] = float(data.get("layout_weight", 1.0))

        pos = graphviz_layout(H, prog="sfdp")
        pos = {
            k: np.array([float(v[0]), float(v[1])], dtype=float)
            for k, v in pos.items()
        }

        print("[viz] network layout: Graphviz sfdp")

    except Exception:
        print(
            "[viz] WARNING: Graphviz sfdp not available, falling back to spring_layout.\n"
            "For better layout, install: conda install -c conda-forge graphviz pydot -y"
        )
        import networkx as nx
        pos = nx.spring_layout(G, seed=42, k=2.0 / max(1, len(G.nodes())) ** 0.5)
        pos = {
            k: np.array([float(v[0]), float(v[1])], dtype=float)
            for k, v in pos.items()
        }

    # ------------------------------------------------------------
    # 5) Normalize coordinates
    # ------------------------------------------------------------
    nodes = list(G.nodes())
    coords = np.array([pos[n] for n in nodes], dtype=float)

    rng = np.random.default_rng(seed)
    coords += rng.normal(scale=0.02, size=coords.shape)

    coords -= coords.mean(axis=0, keepdims=True)

    max_abs = np.abs(coords).max()
    if max_abs > 0:
        coords = coords / max_abs * 8.0

    pos = {n: coords[i] for i, n in enumerate(nodes)}

    # ------------------------------------------------------------
    # 6) Node ellipse sizes
    # ------------------------------------------------------------
    node_w = {}
    node_h = {}

    for node in nodes:
        label = str(node)
        deg = degrees.get(node, 1)

        node_w[node] = max(
            0.78,
            0.085 * len(label) + 0.34 + 0.015 * min(deg, 10),
        )
        node_h[node] = 0.32 + 0.014 * min(deg, 8)

    # ------------------------------------------------------------
    # 7) Soft overlap reduction
    # ------------------------------------------------------------
    pad_x = 0.08
    pad_y = 0.05

    for _ in range(120):
        moved = False

        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                ni = nodes[i]
                nj = nodes[j]

                dx = coords[j, 0] - coords[i, 0]
                dy = coords[j, 1] - coords[i, 1]

                min_dx = (node_w[ni] + node_w[nj]) / 2.0 + pad_x
                min_dy = (node_h[ni] + node_h[nj]) / 2.0 + pad_y

                ox = min_dx - abs(dx)
                oy = min_dy - abs(dy)

                if ox > 0 and oy > 0:
                    moved = True

                    if abs(dx) < 1e-9 and abs(dy) < 1e-9:
                        direction = rng.normal(size=2)
                    else:
                        direction = np.array([dx, dy], dtype=float)

                    direction = direction / (np.linalg.norm(direction) + 1e-9)

                    push = 0.08 * max(ox, oy)
                    coords[i] -= direction * push
                    coords[j] += direction * push

        if not moved:
            break

    pos = {n: coords[i] for i, n in enumerate(nodes)}

    # ------------------------------------------------------------
    # 8) Edge appearance: old rank-based width + region color
    # ------------------------------------------------------------
    edge_draw_info = []

    for u, v, data in G.edges(data=True):
        s = float(data.get("edge_strength", 0.5))
        s = float(np.clip(s, 0.0, 1.0))

        # Old subtle width logic.
        width = 0.45 + 1.15 * s
        alpha = 0.22 + 0.30 * s

        edge_draw_info.append(
            (
                u,
                v,
                data.get("edge_color", "#5f5f5f"),
                width,
                alpha,
            )
        )

    # ------------------------------------------------------------
    # 9) Draw
    # ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(13.5, 8.2))

    for u, v, color, width, alpha in edge_draw_info:
        x1, y1 = pos[u]
        x2, y2 = pos[v]

        ax.plot(
            [x1, x2],
            [y1, y2],
            color=color,
            linewidth=width,
            alpha=alpha,
            zorder=1,
        )

    node_face = "#a9c7df"
    node_edge = "#666666"

    for node in G.nodes():
        x, y = pos[node]
        label = str(node)

        ax.add_patch(
            Ellipse(
                (x, y),
                width=node_w[node],
                height=node_h[node],
                facecolor=node_face,
                edgecolor=node_edge,
                linewidth=0.75,
                alpha=0.96,
                zorder=2,
            )
        )

        ax.text(
            x,
            y,
            label,
            ha="center",
            va="center",
            fontsize=6.2,
            color="#333333",
            zorder=3,
        )

    ax.set_title(
        f"Global d-colocalized gene-pair network\n"
        f"Top {G.number_of_edges()} CPB-significant edges",
        fontsize=15,
        pad=14,
    )

    ax.axis("off")
    ax.set_aspect("equal", adjustable="box")

    xs = np.array([pos[n][0] for n in G.nodes()])
    ys = np.array([pos[n][1] for n in G.nodes()])

    ax.set_xlim(xs.min() - 0.8, xs.max() + 0.8)
    ax.set_ylim(ys.min() - 0.8, ys.max() + 0.8)

    # ------------------------------------------------------------
    # 10) Legends
    # ------------------------------------------------------------

    # Edge color legend
    present_regions = []

    for region in [
        "nuclear",
        "perinuclear",
        "cytosolic",
        "peripheral",
        "unassigned",
        "unannotated",
    ]:
        if any(
            str(d.get("dominant_region", "")) == region
            for _, _, d in G.edges(data=True)
        ):
            label = (
                "unannotated/unassigned"
                if region in {"unassigned", "unannotated"}
                else region
            )
            present_regions.append(
                Patch(
                    facecolor=region_color_map.get(region, "#bdbdbd"),
                    edgecolor="none",
                    label=label,
                )
            )

    uniq = {}
    for h in present_regions:
        uniq[h.get_label()] = h
    present_regions = list(uniq.values())

    if present_regions:
        leg1 = ax.legend(
            handles=present_regions,
            title="Edge color: region",
            loc="upper left",
            bbox_to_anchor=(1.01, 1.00),
            frameon=False,
            fontsize=9,
            title_fontsize=10,
        )
        ax.add_artist(leg1)

        # Edge width meaning text only
        ax.text(
            1.01,
            0.80,
            "Edge width: significance",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            color="#222222",
        )

    fig.tight_layout()

    save_path = out_dir / "viz_coloc_network_styled.png"
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return save_path

# -----------------------------------------------------------------------------
# Top pair spatial plot with boundaries
# -----------------------------------------------------------------------------


def _boundary_to_xy(boundary):
    """Convert common boundary formats to an N x 2 xy array."""
    if boundary is None:
        return None

    try:
        if hasattr(boundary, "exterior") and hasattr(boundary.exterior, "coords"):
            xy = np.asarray(boundary.exterior.coords)
            return xy[:, :2] if xy.ndim == 2 and xy.shape[1] >= 2 else None
    except Exception:
        pass

    if isinstance(boundary, pd.DataFrame):
        if {"x", "y"}.issubset(boundary.columns):
            xy = boundary[["x", "y"]].to_numpy()
        else:
            return None
    else:
        xy = np.asarray(boundary)

    if xy.ndim != 2 or xy.shape[1] < 2 or len(xy) < 3:
        return None
    return xy[:, :2].astype(float)


def _load_boundaries_from_pkl(pkl_path: str | Path | None):
    if pkl_path is None:
        return {}, {}
    with open(pkl_path, "rb") as f:
        bundle = pickle.load(f)
    cell_boundary = {str(k): v for k, v in bundle.get("cell_boundary", {}).items()}
    nuclear_boundary = {str(k): v for k, v in bundle.get("nuclear_boundary", {}).items()}
    return cell_boundary, nuclear_boundary


def _proximal_link_indices(
    g1: pd.DataFrame,
    g2: pd.DataFrame,
    *,
    distance_threshold: float,
    use_3d: bool = False,
):
    if g1.empty or g2.empty:
        return []

    if use_3d and "absZ" in g1.columns and "absZ" in g2.columns:
        coords1 = g1[["absX", "absY", "absZ"]].to_numpy(float)
        coords2 = g2[["absX", "absY", "absZ"]].to_numpy(float)
    else:
        coords1 = g1[["absX", "absY"]].to_numpy(float)
        coords2 = g2[["absX", "absY"]].to_numpy(float)

    D = cdist(coords1, coords2)
    ii, jj = np.where(D <= distance_threshold)
    return list(zip(ii.tolist(), jj.tolist()))

def _rank_cells_for_pair_by_balanced_hits(
    sub: pd.DataFrame,
    *,
    gene_1: str,
    gene_2: str,
    distance_threshold: float,
    use_3d: bool = False,
    min_gene_count: int = 3,
) -> pd.DataFrame:
    """
    Rank representative cells by reciprocal transcript-neighbor coverage.

    balanced_hit_fraction is the smaller of two fractions:
        - fraction of gene_1 transcripts with at least one gene_2 neighbor within d
        - fraction of gene_2 transcripts with at least one gene_1 neighbor within d

    This is only used to choose visually representative cells. It does not
    change InSTAnT PP/CPB statistics or the significant pair table.
    """
    coord_cols = ["absX", "absY"]
    if use_3d and "absZ" in sub.columns:
        coord_cols = ["absX", "absY", "absZ"]

    rows = []
    for cid, cdf in sub.groupby("uID", sort=False):
        g1 = cdf[cdf["gene"].astype(str) == str(gene_1)].copy()
        g2 = cdf[cdf["gene"].astype(str) == str(gene_2)].copy()

        n1 = len(g1)
        n2 = len(g2)
        if n1 < min_gene_count or n2 < min_gene_count:
            continue

        coords1 = g1[coord_cols].to_numpy(float)
        coords2 = g2[coord_cols].to_numpy(float)

        tree2 = cKDTree(coords2)
        hits12 = tree2.query_ball_point(coords1, r=distance_threshold)
        n_links = int(sum(len(x) for x in hits12))
        if n_links == 0:
            continue

        g1_hit_count = int(sum(len(x) > 0 for x in hits12))
        g1_hit_fraction = float(g1_hit_count / n1) if n1 > 0 else 0.0

        tree1 = cKDTree(coords1)
        hits21 = tree1.query_ball_point(coords2, r=distance_threshold)
        g2_hit_count = int(sum(len(x) > 0 for x in hits21))
        g2_hit_fraction = float(g2_hit_count / n2) if n2 > 0 else 0.0

        balanced = float(min(g1_hit_fraction, g2_hit_fraction))
        link_density = float(n_links / np.sqrt(n1 * n2)) if n1 > 0 and n2 > 0 else 0.0
        links_per_min_count = float(n_links / max(min(n1, n2), 1))

        rows.append(
            {
                "uID": str(cid),
                "gene_1_count": int(n1),
                "gene_2_count": int(n2),
                "links": n_links,
                "gene_1_hit_fraction": g1_hit_fraction,
                "gene_2_hit_fraction": g2_hit_fraction,
                "balanced_hit_fraction": balanced,
                "link_density": link_density,
                "links_per_min_count": links_per_min_count,
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    return out.sort_values(
        ["balanced_hit_fraction", "link_density", "links"],
        ascending=False,
    ).reset_index(drop=True)


def _rank_cells_for_pair_by_pp_pvalue(
    df: pd.DataFrame,
    *,
    gene_1: str,
    gene_2: str,
    gene_list: Optional[list[str]],
    pp_pvals,
) -> pd.DataFrame:
    """Rank cells by cell-wise PP p-value for a given gene pair."""
    if pp_pvals is None or gene_list is None:
        return pd.DataFrame()

    genes = [str(g) for g in gene_list]
    gene_to_idx = {g: i for i, g in enumerate(genes)}
    if str(gene_1) not in gene_to_idx or str(gene_2) not in gene_to_idx:
        return pd.DataFrame()

    pp = np.asarray(pp_pvals)
    if pp.ndim != 3:
        return pd.DataFrame()

    i = gene_to_idx[str(gene_1)]
    j = gene_to_idx[str(gene_2)]

    cell_ids = np.unique(df["uID"].astype(str).values)
    if len(cell_ids) != pp.shape[0]:
        # Cell order cannot be safely matched.
        return pd.DataFrame()

    cell_gene_counts = df[df["gene"].astype(str).isin([gene_1, gene_2])].groupby(["uID", "gene"]).size().unstack(fill_value=0)
    if gene_1 not in cell_gene_counts.columns or gene_2 not in cell_gene_counts.columns:
        return pd.DataFrame()
    both_cells = set(cell_gene_counts[(cell_gene_counts[gene_1] > 0) & (cell_gene_counts[gene_2] > 0)].index.astype(str))

    rows = []
    for cell_idx, cid in enumerate(cell_ids):
        cid = str(cid)
        if cid not in both_cells:
            continue
        pval = float(pp[cell_idx, i, j])
        if not np.isfinite(pval):
            continue
        rows.append({"uID": cid, "pp_pvalue": pval})

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["pp_pvalue", "uID"], ascending=[True, True]).reset_index(drop=True)


def plot_top_pair_cells(
    out_dir: str | Path,
    *,
    pkl_path: str | Path | None = None,
    instant_input_df: Optional[pd.DataFrame] = None,
    significant_pairs: Optional[pd.DataFrame] = None,
    gene_list: Optional[list[str]] = None,
    pp_pvals=None,
    gene_1: str | None = None,
    gene_2: str | None = None,
    max_cells: int = 6,
    distance_threshold: float = 4.0,
    use_3d: bool = False,
    cell_selection: str = "pp",
    rank: int | None = None,
) -> Path | None:
    """Plot spatial transcript examples for one colocalized gene pair."""
    out_dir = Path(out_dir)

    if instant_input_df is None:
        input_path = out_dir / "instant_input_after_prefilter.csv"
        if not input_path.exists():
            raise FileNotFoundError(f"Cannot find: {input_path}")
        df = pd.read_csv(input_path)
    else:
        df = instant_input_df.copy()

    required = {"gene", "absX", "absY", "uID"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"instant input missing columns: {missing}")

    if significant_pairs is None:
        pairs_path = out_dir / "instant_significant_pairs.csv"
        if not pairs_path.exists():
            raise FileNotFoundError(f"Cannot find: {pairs_path}")
        pairs = pd.read_csv(pairs_path)
    else:
        pairs = significant_pairs.copy()

    if pairs.empty:
        print("[viz] significant pair table is empty. Top-pair spatial plot skipped.")
        return None

    if gene_1 is None or gene_2 is None:
        sort_col = "cpb_pvalue_raw" if "cpb_pvalue_raw" in pairs.columns else "cpb_pvalue"
        top = pairs.sort_values(sort_col, ascending=True).iloc[0]
        gene_1 = str(top["gene_1"])
        gene_2 = str(top["gene_2"])
        if "distance_threshold" in pairs.columns:
            distance_threshold = float(top["distance_threshold"])

    gene_1 = str(gene_1)
    gene_2 = str(gene_2)
    cell_boundary, nuclear_boundary = _load_boundaries_from_pkl(pkl_path)

    sub = df[df["gene"].astype(str).isin([gene_1, gene_2])].copy()
    if sub.empty:
        print(f"[viz] no transcripts found for pair: {gene_1}, {gene_2}")
        return None

    cell_gene_counts = sub.groupby(["uID", "gene"]).size().unstack(fill_value=0)
    if gene_1 not in cell_gene_counts.columns or gene_2 not in cell_gene_counts.columns:
        print(f"[viz] selected pair not found in same cells: {gene_1}, {gene_2}")
        return None

    both = cell_gene_counts[(cell_gene_counts[gene_1] > 0) & (cell_gene_counts[gene_2] > 0)].copy()
    if both.empty:
        print(f"[viz] no cells contain both genes: {gene_1}, {gene_2}")
        return None

    cell_metrics = {}
    selected_cells = []

    if cell_selection == "pp":
        ranked = _rank_cells_for_pair_by_pp_pvalue(
            df,
            gene_1=gene_1,
            gene_2=gene_2,
            gene_list=gene_list,
            pp_pvals=pp_pvals,
        )
        if not ranked.empty:
            selected_cells = ranked.head(max_cells)["uID"].astype(str).tolist()
            cell_metrics = {str(r["uID"]): r for _, r in ranked.iterrows()}
        else:
            print(f"[viz] PP-pvalue cell selection unavailable for {gene_1}-{gene_2}; falling back to balanced selection.")
            cell_selection = "balanced"

    if cell_selection == "balanced":
        ranked = _rank_cells_for_pair_by_balanced_hits(
            sub,
            gene_1=gene_1,
            gene_2=gene_2,
            distance_threshold=distance_threshold,
            use_3d=use_3d,
        )
        if not ranked.empty:
            selected_cells = ranked.head(max_cells)["uID"].astype(str).tolist()
            cell_metrics = {str(r["uID"]): r for _, r in ranked.iterrows()}
        else:
            print(f"[viz] balanced cell selection found no linked cells for {gene_1}-{gene_2}; falling back to expression-total selection.")
            cell_selection = "expression"

    if cell_selection == "expression":
        both["total"] = both[gene_1] + both[gene_2]
        selected_cells = both.sort_values("total", ascending=False).head(max_cells).index.astype(str).tolist()
    elif cell_selection not in {"pp", "balanced"}:
        raise ValueError("cell_selection must be 'pp', 'balanced', or 'expression'.")

    n = len(selected_cells)
    if n == 0:
        print(f"[viz] no representative cells selected for {gene_1}-{gene_2}")
        return None

    fig, axes = plt.subplots(1, n, figsize=(4.0 * n, 4.1), squeeze=False)

    for ax, cid in zip(axes[0], selected_cells):
        cdf = sub[sub["uID"].astype(str) == str(cid)].copy()
        g1 = cdf[cdf["gene"].astype(str) == gene_1].reset_index(drop=True)
        g2 = cdf[cdf["gene"].astype(str) == gene_2].reset_index(drop=True)

        cxy = _boundary_to_xy(cell_boundary.get(str(cid)))
        if cxy is not None:
            ax.plot(cxy[:, 0], cxy[:, 1], color="#444444", linewidth=1.25, alpha=0.9, zorder=1)

        nxy = _boundary_to_xy(nuclear_boundary.get(str(cid)))
        if nxy is not None:
            ax.plot(nxy[:, 0], nxy[:, 1], color="#7e7e7e", linewidth=1.05, linestyle="--", alpha=0.9, zorder=1)

        links = _proximal_link_indices(g1, g2, distance_threshold=distance_threshold, use_3d=use_3d)
        g1_xy = g1[["absX", "absY"]].to_numpy(float)
        g2_xy = g2[["absX", "absY"]].to_numpy(float)

        for i, j in links[:800]:
            ax.plot([g1_xy[i, 0], g2_xy[j, 0]], [g1_xy[i, 1], g2_xy[j, 1]], color="#d75f5f", linewidth=0.35, alpha=0.20, zorder=2)

        ax.scatter(g1["absX"], g1["absY"], s=9, c="#4C78A8", label=gene_1, alpha=0.86, zorder=3)
        ax.scatter(g2["absX"], g2["absY"], s=9, c="#F58518", label=gene_2, alpha=0.80, zorder=3)

        title_extra = ""
        if str(cid) in cell_metrics:
            m = cell_metrics[str(cid)]
            if "pp_pvalue" in m:
                title_extra = f", PP p={float(m['pp_pvalue']):.2e}"
            elif "balanced_hit_fraction" in m:
                title_extra = f", bal={float(m['balanced_hit_fraction']):.2f}"
        ax.set_title(f"cell {cid}\n{gene_1}={len(g1)}, {gene_2}={len(g2)}, links={len(links)}{title_extra}", fontsize=9)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal", adjustable="box")

    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right", fontsize=9)

    prefix = f"Top {rank:02d}" if rank is not None else "Top"
    fig.suptitle(f"{prefix} global d-colocalized pair: {gene_1} - {gene_2} | proximal links d ≤ {distance_threshold:g}", fontsize=13)
    fig.tight_layout()

    if rank is None:
        save_name = f"viz_top_pair_cells_with_boundaries_{_safe_filename(gene_1)}_{_safe_filename(gene_2)}.png"
    else:
        save_name = f"viz_top{rank:02d}_pair_cells_with_boundaries_{_safe_filename(gene_1)}_{_safe_filename(gene_2)}.png"
    save_path = out_dir / save_name
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    return save_path


# -----------------------------------------------------------------------------
# One-call visualization wrapper
# -----------------------------------------------------------------------------


def plot_coloc_outputs(
    *,
    out_dir: str | Path,
    pkl_path: str | Path | None = None,
    gene_list: Optional[list[str]] = None,
    instant_input_df: Optional[pd.DataFrame] = None,
    significant_pairs: Optional[pd.DataFrame] = None,
    region_annotated_pairs: Optional[pd.DataFrame] = None,
    cpb_pvals=None,
    expected_coloc=None,
    pp_pvals=None,
    top_n_genes: int | None = 80,
    max_edges: int = 80,
    max_cells: int = 6,
    top_pairs: int = 10,
    gene_1: str | None = None,
    gene_2: str | None = None,
    distance_threshold: float = 4.0,
    use_3d: bool = False,
    save_all_pairs: bool = False,
    cell_selection: str = "pp",
    make_heatmap: bool = True,
    make_network: bool = True,
    make_top_pairs: bool = True,
) -> dict[str, object]:
    """Generate standard colocalization visualizations.

    This wrapper supports independent plotting:
    - make_heatmap=False skips heatmap
    - make_network=False skips network
    - make_top_pairs=False skips top-pair spatial plots
    """
    out_dir = Path(out_dir)
    paths: dict[str, object] = {}

    if save_all_pairs:
        all_pairs = export_all_pairs_table(
            out_dir,
            gene_list=gene_list,
            cpb_pvals=cpb_pvals,
            expected_coloc=expected_coloc,
        )
        paths["all_pairs"] = str(all_pairs)

    if make_heatmap:
        heatmap = plot_cpb_heatmap(
            out_dir,
            gene_list=gene_list,
            cpb_pvals=cpb_pvals,
            top_n_genes=top_n_genes,
        )
        paths["heatmap"] = str(heatmap)

    if make_network:
        # 新版 plot_coloc_network 支持 region_annotated_pairs；
        # 如果你本地暂时还是旧版签名，则自动 fallback，不让独立画图命令崩。
        try:
            network = plot_coloc_network(
                out_dir,
                significant_pairs=significant_pairs,
                region_annotated_pairs=region_annotated_pairs,
                max_edges=max_edges,
            )
        except TypeError as e:
            if "region_annotated_pairs" not in str(e):
                raise
            network = plot_coloc_network(
                out_dir,
                significant_pairs=(
                    region_annotated_pairs
                    if region_annotated_pairs is not None
                    else significant_pairs
                ),
                max_edges=max_edges,
            )

        if network is not None:
            paths["network"] = str(network)

    if significant_pairs is None:
        pairs_path = out_dir / "instant_significant_pairs.csv"
        pairs = pd.read_csv(pairs_path) if pairs_path.exists() else pd.DataFrame()
    else:
        pairs = significant_pairs.copy()

    top_paths = []

    if make_top_pairs and gene_1 is not None and gene_2 is not None:
        p = plot_top_pair_cells(
            out_dir,
            pkl_path=pkl_path,
            instant_input_df=instant_input_df,
            significant_pairs=pairs,
            gene_list=gene_list,
            pp_pvals=pp_pvals,
            gene_1=gene_1,
            gene_2=gene_2,
            max_cells=max_cells,
            distance_threshold=distance_threshold,
            use_3d=use_3d,
            cell_selection=cell_selection,
            rank=1,
        )
        if p is not None:
            top_paths.append(str(p))

    elif make_top_pairs and not pairs.empty:
        sort_col = "cpb_pvalue_raw" if "cpb_pvalue_raw" in pairs.columns else "cpb_pvalue"
        pairs = pairs.sort_values(sort_col, ascending=True).reset_index(drop=True).head(int(top_pairs))

        for rank, (_, row) in enumerate(pairs.iterrows(), start=1):
            p = plot_top_pair_cells(
                out_dir,
                pkl_path=pkl_path,
                instant_input_df=instant_input_df,
                significant_pairs=pairs,
                gene_list=gene_list,
                pp_pvals=pp_pvals,
                gene_1=str(row["gene_1"]),
                gene_2=str(row["gene_2"]),
                max_cells=max_cells,
                distance_threshold=float(row.get("distance_threshold", distance_threshold)),
                use_3d=use_3d,
                cell_selection=cell_selection,
                rank=rank,
            )
            if p is not None:
                top_paths.append(str(p))

    if top_paths:
        paths["top_pair_cells"] = top_paths

    return paths

# def plot_coloc_outputs(
#     *,
#     out_dir: str | Path,
#     pkl_path: str | Path | None = None,
#     gene_list: Optional[list[str]] = None,
#     instant_input_df: Optional[pd.DataFrame] = None,
#     significant_pairs: Optional[pd.DataFrame] = None,
#     cpb_pvals=None,
#     expected_coloc=None,
#     top_n_genes: int | None = 80,
#     max_edges: int = 80,
#     max_cells: int = 6,
#     gene_1: str | None = None,
#     gene_2: str | None = None,
#     distance_threshold: float = 4.0,
#     use_3d: bool = False,
#     save_all_pairs: bool = False,
# ) -> dict[str, str]:
#     """Generate all standard colocalization visualizations."""
#     out_dir = Path(out_dir)
#     paths: dict[str, str] = {}

#     if save_all_pairs:
#         all_pairs = export_all_pairs_table(
#             out_dir,
#             gene_list=gene_list,
#             cpb_pvals=cpb_pvals,
#             expected_coloc=expected_coloc,
#         )
#         paths["all_pairs"] = str(all_pairs)

#     heatmap = plot_cpb_heatmap(
#         out_dir,
#         # gene_list=gene_list,
#         # cpb_pvals=cpb_pvals,
#         top_n_genes=top_n_genes,
#     )
#     paths["heatmap"] = str(heatmap)

#     network = plot_coloc_network(
#         out_dir,
#         significant_pairs=significant_pairs,
#         max_edges=max_edges,
#     )
#     if network is not None:
#         paths["network"] = str(network)

#     top_cells = plot_top_pair_cells(
#         out_dir,
#         pkl_path=pkl_path,
#         instant_input_df=instant_input_df,
#         significant_pairs=significant_pairs,
#         gene_1=gene_1,
#         gene_2=gene_2,
#         max_cells=max_cells,
#         distance_threshold=distance_threshold,
#         use_3d=use_3d,
#     )
#     if top_cells is not None:
#         paths["top_pair_cells"] = str(top_cells)

#     return paths
