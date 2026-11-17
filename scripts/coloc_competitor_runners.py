"""Unified 5-method runner for subcellular RNA colocalization benchmarks.

Wraps official implementations of:
  1. DeCoLoc (STRAND prototype: direct_micro & whole_cell)
  2. InSTAnT (PP + CPB, Nature Communications 2024)
  3. SCRIN (cooccurrence hypergeometric test + BH FDR, Research Square 2025)
  4. Bento (bt.tl.coloc_quotient CLQ & bt.tl.colocation tensor factor loadings, Genome Biology 2024)
  5. pyTrance (self-supervised 1-layer DGI GCN gene embeddings + Leiden co-clustering, bioRxiv 2026)

Every runner returns standardized undirected pair tables with columns:
  gene_1, gene_2, score, significant, elapsed_seconds, peak_rss_gb
where gene_1 < gene_2 lexicographically.
"""

from __future__ import annotations

import math
from pathlib import Path
import resource
import sys
import time
import warnings

import numpy as np
import pandas as pd

STRAND_ROOT = Path(__file__).resolve().parents[1]
if str(STRAND_ROOT) not in sys.path:
    sys.path.insert(0, str(STRAND_ROOT))

from tools.decoloc.api import run_decoloc_from_bundle
from tools.engines.coloc_region_annotation import (
    CellGeometry,
    _load_cell_geometries,
)
from tools.instant_optimized.adapter import run_optimized_colocalization_from_bundle


def _rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 ** 2)


def canonical_pair_cols(df: pd.DataFrame, g1_col: str = "gene_1", g2_col: str = "gene_2") -> pd.DataFrame:
    out = df.copy()
    a = out[g1_col].astype(str).to_numpy()
    b = out[g2_col].astype(str).to_numpy()
    out["gene_1"] = np.minimum(a, b)
    out["gene_2"] = np.maximum(a, b)
    return out[out["gene_1"] != out["gene_2"]].reset_index(drop=True)


# ---------------------------------------------------------------------------
# 1. DeCoLoc (direct_micro & whole_cell)
# ---------------------------------------------------------------------------
def run_method_decoloc(
    bundle: dict,
    *,
    radius: float,
    distance_mode: str = "cell_scaled",
    region_threshold_mode: str = "proportional",
    kernel: str = "gaussian",
    fdr_alpha: float = 0.01,
    min_fe: float = 1.15,
    min_coexpr_cells: int = 15,
    threads: int = 16,
) -> dict[str, pd.DataFrame]:
    t0 = time.perf_counter()
    res, _ = run_decoloc_from_bundle(
        bundle,
        distance=radius,
        distance_mode=distance_mode,
        region_threshold_mode=region_threshold_mode,
        kernel=kernel,
        fdr_alpha=fdr_alpha,
        min_fe=min_fe,
        min_coexpr_cells=min_coexpr_cells,
        do_prefilter=False,
        threads=threads,
    )
    elapsed = time.perf_counter() - t0
    peak_gb = _rss_gb()

    ap = res.all_pairs.copy()
    if ap.empty:
        empty = pd.DataFrame(columns=["gene_1", "gene_2", "score", "significant", "qval", "fe", "elapsed_seconds", "peak_rss_gb"])
        return {"DeCoLoc (direct_micro)": empty, "DeCoLoc (whole_cell)": empty.copy()}

    ap = canonical_pair_cols(ap)

    # Micro-colocalization score: -log10(p_comp) gated by positive t_cell (so co-compartment-depleted
    # pairs never outrank true colocalized pairs), with tie-breaking via t_comp.
    score_micro = np.where(
        ap["t_cell"] > 0,
        -np.log10(np.clip(ap["p_comp"], 1e-300, 1.0)) + 1e-4 * np.maximum(ap["t_comp"], 0.0),
        0.0,
    )
    df_micro = pd.DataFrame(
        {
            "gene_1": ap["gene_1"],
            "gene_2": ap["gene_2"],
            "score": score_micro,
            "significant": (ap["coloc_class"] == "direct_micro"),
            "qval": ap["q_comp"],
            "fe": ap["fe_comp"],
            "coloc_class": ap["coloc_class"],
            "dominant_region": ap.get("dominant_region", "unannotated"),
            "elapsed_seconds": elapsed,
            "peak_rss_gb": peak_gb,
        }
    ).sort_values("score", ascending=False).reset_index(drop=True)

    score_cell = -np.log10(np.clip(ap["p_cell"], 1e-300, 1.0)) + 1e-4 * np.maximum(ap["t_cell"], 0.0)
    df_cell = pd.DataFrame(
        {
            "gene_1": ap["gene_1"],
            "gene_2": ap["gene_2"],
            "score": score_cell,
            "significant": (ap["coloc_class"] != "non_significant"),
            "qval": ap["q_cell"],
            "fe": ap["fe_cell"],
            "coloc_class": ap["coloc_class"],
            "dominant_region": ap.get("dominant_region", "unannotated"),
            "elapsed_seconds": elapsed,
            "peak_rss_gb": peak_gb,
        }
    ).sort_values("score", ascending=False).reset_index(drop=True)

    return {
        "DeCoLoc (direct_micro)": df_micro,
        "DeCoLoc (whole_cell)": df_cell,
    }


# ---------------------------------------------------------------------------
# 2. InSTAnT (PP + CPB)
# ---------------------------------------------------------------------------
def run_method_instant(
    bundle: dict,
    *,
    radius: float,
    pp_alpha: float = 0.001,
    cpb_alpha: float = 0.0001,
    min_genecount: int = 20,
    threads: int = 16,
) -> pd.DataFrame:
    t0 = time.perf_counter()
    res = run_optimized_colocalization_from_bundle(
        bundle,
        distance_threshold=float(radius),
        pp_alpha=pp_alpha,
        cpb_alpha=cpb_alpha,
        min_genecount=min_genecount,
        use_3d=False,
        threads=threads,
    )
    elapsed = time.perf_counter() - t0
    peak_gb = _rss_gb()

    genes = list(res["gene_list"])
    cpb = np.asarray(res["cpb_pvals"], dtype=np.float64)
    exp_mat = np.asarray(res["expected_coloc"], dtype=np.float64)
    obs_mat = np.asarray(res["cpb_results"].observed_significant_cells, dtype=np.int32)

    ii, jj = np.triu_indices(len(genes), k=1)
    raw_p = cpb[ii, jj]
    finite = np.isfinite(raw_p)
    ii, jj, raw_p = ii[finite], jj[finite], raw_p[finite]

    clipped_p = np.clip(raw_p, 1e-300, 1.0)
    obs_c = obs_mat[ii, jj]
    exp_c = exp_mat[ii, jj]
    # Break p=0 numerical underflow ties using observed significant cells (matching STRAND viz ranker).
    score = -np.log10(clipped_p) + 1e-4 * obs_c

    gene_arr = np.asarray(genes, dtype=object)
    out = pd.DataFrame(
        {
            "gene_1": np.minimum(gene_arr[ii], gene_arr[jj]),
            "gene_2": np.maximum(gene_arr[ii], gene_arr[jj]),
            "score": score,
            "significant": (raw_p < cpb_alpha),
            "pval": np.clip(raw_p, 0.0, 1.0),
            "obs_sig_cells": obs_c,
            "exp_sig_cells": exp_c,
            "elapsed_seconds": elapsed,
            "peak_rss_gb": peak_gb,
        }
    ).sort_values("score", ascending=False).reset_index(drop=True)
    return out


# ---------------------------------------------------------------------------
# 3. Official SCRIN (scrin v1.1.0 core CLB cooccurrence hypergeometric)
# ---------------------------------------------------------------------------
_SCRIN_TASKS = None
_SCRIN_R_CHECK = 1.0


def _init_official_scrin_pool(tasks, r_check: float) -> None:
    global _SCRIN_TASKS, _SCRIN_R_CHECK
    _SCRIN_TASKS = tasks
    _SCRIN_R_CHECK = float(r_check)


def _official_scrin_worker(cell_idx: int):
    from scrin.core.clb_compact import region_function_compact_radius_kr

    return region_function_compact_radius_kr(
        _SCRIN_TASKS[cell_idx], r_check=_SCRIN_R_CHECK, z_continuous=False
    )


def run_method_scrin(
    bundle: dict,
    *,
    radius: float,
    q_threshold: float = 1e-5,
    max_expr_ratio: float = 100.0,
    min_gene_number: int = 5,
    min_neighbor_number: int = 1,
    pair_keep: str = "last",
    threads: int = 16,
) -> pd.DataFrame:
    """Run official scrin v1.1.0 CLB cooccurrence kernels + hyper_test + pair deduplication."""
    import multiprocessing as mp

    from scrin.core.clb_compact import CompactCellKRTasks, build_compact_transcript_data
    from scrin.core.clb_kr import (
        aggregate_k_records,
        build_background_arrays,
        reconstruct_a_statistics,
        select_gene_records,
        sort_r_records,
    )
    from scrin.hyper_test_clb_distribution import hyper_test
    from scrin.tools.stream_results import prepare_hyper_result

    t0 = time.perf_counter()
    raw = bundle["data_df"]
    scrin_df = pd.DataFrame(
        {
            "x": raw["x"].to_numpy(dtype=np.float64),
            "y": raw["y"].to_numpy(dtype=np.float64),
            "z": np.zeros(len(raw), dtype=np.float64),
            "geneID": raw["gene"].astype(str).to_numpy(),
            "cell": raw["cell"].astype(str).to_numpy(),
        }
    )
    g_counts = scrin_df.groupby("geneID", observed=True).size()
    kept_genes = set(g_counts.index[g_counts >= int(min_gene_number)])
    scrin_df = scrin_df[scrin_df["geneID"].isin(kept_genes)].reset_index(drop=True)

    if scrin_df.empty:
        return pd.DataFrame(columns=["gene_1", "gene_2", "score", "significant", "qval", "fe", "elapsed_seconds", "peak_rss_gb"])

    compact = build_compact_transcript_data(scrin_df)
    bg = build_background_arrays(compact, nine_grid=False)
    tasks = CompactCellKRTasks(compact)
    gene_names = compact.gene_names
    gene_count = len(gene_names)

    n_cells = len(tasks)
    if threads <= 1 or n_cells < 16:
        _init_official_scrin_pool(tasks, float(radius))
        kr_out = [_official_scrin_worker(i) for i in range(n_cells)]
    else:
        ctx = mp.get_context("fork")
        with ctx.Pool(
            processes=int(threads),
            initializer=_init_official_scrin_pool,
            initargs=(tasks, float(radius)),
        ) as pool:
            kr_out = list(pool.imap_unordered(_official_scrin_worker, range(n_cells), chunksize=16))

    k_all = aggregate_k_records([item[0] for item in kr_out], gene_count)
    r_all = sort_r_records([item[1] for item in kr_out])

    dfs = []
    for gene_a in range(gene_count):
        k_rec = select_gene_records(k_all, gene_a)
        if not len(k_rec):
            continue
        r_rec = select_gene_records(r_all, gene_a)
        stats_arr = reconstruct_a_statistics(gene_a, k_rec, r_rec, bg, gene_count)
        gene_a_vals = {gene_names[int(row[0])]: row[1:].tolist() for row in stats_arr}
        res_df = hyper_test(
            (gene_names[int(gene_a)], gene_a_vals),
            min_gene_number=int(min_neighbor_number),
            expression_level=float(max_expr_ratio),
        )
        if res_df is not None and not res_df.empty:
            dfs.append(prepare_hyper_result(res_df))

    elapsed = time.perf_counter() - t0
    peak_gb = _rss_gb()
    if not dfs:
        return pd.DataFrame(columns=["gene_1", "gene_2", "score", "significant", "qval", "fe", "elapsed_seconds", "peak_rss_gb"])

    full_df = pd.concat(dfs, ignore_index=True)
    # Official SCRIN post-processing: sort by qvalue_BH ascending, deduplicate undirected 'pair' keeping `pair_keep`.
    full_df = full_df.sort_values("qvalue_BH", ascending=True, kind="mergesort")
    dedup = full_df.drop_duplicates(subset=["pair"], keep=pair_keep).reset_index(drop=True)
    dedup = canonical_pair_cols(dedup, g1_col="gene_A", g2_col="gene_B")

    q_clip = np.clip(dedup["qvalue_BH"].to_numpy(dtype=np.float64), 1e-300, 1.0)
    fe = dedup["enrichment_ratio"].to_numpy(dtype=np.float64)
    score = -np.log10(q_clip) + 1e-3 * np.log1p(np.maximum(fe, 0.0))

    return pd.DataFrame(
        {
            "gene_1": dedup["gene_1"],
            "gene_2": dedup["gene_2"],
            "score": score,
            "significant": (dedup["qvalue_BH"] < float(q_threshold)),
            "qval": dedup["qvalue_BH"],
            "fe": fe,
            "support_ratio": dedup["support_ratio"],
            "elapsed_seconds": elapsed,
            "peak_rss_gb": peak_gb,
        }
    ).sort_values("score", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 4. Official Bento (bt.tl.coloc_quotient & bt.tl.colocation)
# ---------------------------------------------------------------------------
def _ensure_non_overlapping_geoms(
    df: pd.DataFrame,
    geoms: dict[str, CellGeometry],
) -> tuple[pd.DataFrame, list, list, list[str]]:
    """Arrange cells on a non-overlapping grid if cell bounding boxes overlap (e.g. local crops)."""
    from shapely.affinity import translate as shp_translate

    cids = [c for c in sorted(df["cell"].astype(str).unique()) if c in geoms]
    bounds = np.array([geoms[c].cell.bounds for c in cids], dtype=np.float64)
    spans = np.maximum(bounds[:, 2] - bounds[:, 0], bounds[:, 3] - bounds[:, 1])
    max_span = float(np.max(spans)) if len(spans) else 200.0

    # Quick overlap check on first 25 cells: if centers are all near (0, 0), shift onto a grid.
    centers = np.column_stack([(bounds[:, 0] + bounds[:, 2]) * 0.5, (bounds[:, 1] + bounds[:, 3]) * 0.5])
    needs_shift = False
    if len(centers) >= 2:
        sample_c = centers[: min(40, len(centers))]
        dmat = np.linalg.norm(sample_c[:, None, :] - sample_c[None, :, :], axis=2)
        np.fill_diagonal(dmat, np.inf)
        if float(np.min(dmat)) < 0.25 * max_span:
            needs_shift = True

    if not needs_shift:
        sub_df = df[df["cell"].astype(str).isin(set(cids))].copy()
        c_polys = [geoms[c].cell for c in cids]
        n_polys = [geoms[c].nuclear for c in cids]
        return sub_df, c_polys, n_polys, cids

    step = max_span * 2.5 + 50.0
    n_cols = max(1, int(math.ceil(math.sqrt(len(cids)))))
    dx_map, dy_map = {}, {}
    c_polys, n_polys = [], []
    for idx, cid in enumerate(cids):
        gx, gy = idx % n_cols, idx // n_cols
        ox = gx * step - bounds[idx, 0]
        oy = gy * step - bounds[idx, 1]
        dx_map[cid] = ox
        dy_map[cid] = oy
        c_polys.append(shp_translate(geoms[cid].cell, xoff=ox, yoff=oy))
        n_polys.append(shp_translate(geoms[cid].nuclear, xoff=ox, yoff=oy))

    sub_df = df[df["cell"].astype(str).isin(set(cids))].copy()
    c_str = sub_df["cell"].astype(str)
    sub_df["x"] = sub_df["x"].to_numpy(dtype=np.float64) + c_str.map(dx_map).to_numpy(dtype=np.float64)
    sub_df["y"] = sub_df["y"].to_numpy(dtype=np.float64) + c_str.map(dy_map).to_numpy(dtype=np.float64)
    return sub_df, c_polys, n_polys, cids


def run_method_bento(
    bundle: dict,
    *,
    radius: float,
    min_points: int = 5,
    rank: int = 5,
    threads: int = 16,
) -> dict[str, pd.DataFrame]:
    """Run official bento-tools coloc_quotient + colocation tensor decomposition."""
    import dask
    import geopandas as gpd
    import spatialdata as sd
    from spatialdata.models import PointsModel, ShapesModel
    import bento as bt

    warnings.filterwarnings("ignore")
    t0 = time.perf_counter()

    df = bundle["data_df"][["cell", "gene", "x", "y"]].copy()
    df["cell"] = df["cell"].astype(str)
    df["gene"] = df["gene"].astype(str)
    geoms = _load_cell_geometries(bundle)

    df_shifted, c_polys, n_polys, cids = _ensure_non_overlapping_geoms(df, geoms)

    # Map gene names to underscore-free tokens so bento's internal f"{g1}_{g2}" join is 100% unambiguous.
    orig_genes = sorted(df_shifted["gene"].unique())
    g2tok = {g: f"GENE{idx:05d}" for idx, g in enumerate(orig_genes)}
    tok2g = {tok: g for g, tok in g2tok.items()}

    pts_df = pd.DataFrame(
        {
            "x": df_shifted["x"].to_numpy(dtype=np.float64),
            "y": df_shifted["y"].to_numpy(dtype=np.float64),
            "feature_name": pd.Categorical(df_shifted["gene"].map(g2tok)),
        }
    )
    sdata = sd.SpatialData(
        shapes={
            "cell_boundaries": ShapesModel.parse(gpd.GeoDataFrame(geometry=c_polys, index=cids)),
            "nucleus_boundaries": ShapesModel.parse(gpd.GeoDataFrame(geometry=n_polys, index=cids)),
        },
        points={
            "transcripts": PointsModel.parse(
                pts_df, coordinates={"x": "x", "y": "y"}, feature_key="feature_name"
            )
        },
    )

    sdata = bt.io.prep(sdata)
    with dask.config.set(scheduler="threads"):
        bt.tl.coloc_quotient(
            sdata,
            radius=float(radius),
            min_points=int(min_points),
            num_workers=max(1, int(threads)),
        )
    t_clq = time.perf_counter() - t0

    clq_raw = sdata.tables["table"].uns["clq"]["cell_boundaries"].copy()
    if clq_raw.empty:
        empty = pd.DataFrame(columns=["gene_1", "gene_2", "score", "significant", "elapsed_seconds", "peak_rss_gb"])
        return {"Bento (CLQ)": empty, "Bento (RNAcoloc)": empty.copy()}

    clq_raw["g_src"] = clq_raw["feature_name"].astype(str).map(tok2g)
    clq_raw["g_dst"] = clq_raw["neighbor"].astype(str).map(tok2g)
    clq_raw = clq_raw[clq_raw["g_src"] != clq_raw["g_dst"]].copy()
    clq_raw["gene_1"] = np.minimum(clq_raw["g_src"].to_numpy(), clq_raw["g_dst"].to_numpy())
    clq_raw["gene_2"] = np.maximum(clq_raw["g_src"].to_numpy(), clq_raw["g_dst"].to_numpy())

    # Aggregate cell-level CLQ across cells (mean finite log_clq weighted by cell support, plus mean clq).
    clq_valid = clq_raw[np.isfinite(clq_raw["log_clq"])]
    n_total_cells = max(len(cids), 1)
    agg_clq = (
        clq_valid.groupby(["gene_1", "gene_2"], observed=True)
        .agg(
            mean_log_clq=("log_clq", "mean"),
            mean_clq=("clq", "mean"),
            n_obs=("log_clq", "size"),
        )
        .reset_index()
    )
    # Bento CLQ ranking score: mean positive log2(CLQ) across observed cells.
    agg_clq["score"] = agg_clq["mean_log_clq"]

    df_bento_clq = pd.DataFrame(
        {
            "gene_1": agg_clq["gene_1"],
            "gene_2": agg_clq["gene_2"],
            "score": agg_clq["score"],
            "significant": False,  # Bento has no hypothesis-test p/q cutoff
            "mean_clq": agg_clq["mean_clq"],
            "n_obs": agg_clq["n_obs"],
            "elapsed_seconds": t_clq,
            "peak_rss_gb": _rss_gb(),
        }
    ).sort_values("score", ascending=False).reset_index(drop=True)

    # Step 2 of Bento: Tensor decomposition (bt.tl.colocation).
    # Guard memory: dense tensor size is 1 x n_cells x n_directed_pairs x 8 bytes.
    n_unique_dir_pairs = int(clq_raw[["g_src", "g_dst"]].drop_duplicates().shape[0])
    est_tensor_gb = (len(cids) * n_unique_dir_pairs * 8) / (1024.0 ** 3)
    if est_tensor_gb <= 12.0 and n_unique_dir_pairs >= rank:
        bt.tl.colocation(sdata, ranks=[int(rank)], iterations=2, plot_error=False)
        t_total = time.perf_counter() - t0
        facs = sdata.tables["table"].uns["factors"][int(rank)]
        pair_labels = sdata.tables["table"].uns["tensor_labels"]["pair"]
        # facs[2] has shape (n_directed_pairs, rank); pair loading score = L2 norm across factors (matching SCRIN benchmark).
        pair_loadings = np.linalg.norm(facs[2], axis=1)
        tok_split = [str(p).split("_", 1) for p in pair_labels]
        g1_list = [tok2g[a] for a, _ in tok_split]
        g2_list = [tok2g[b] for _, b in tok_split]
        fac_df = pd.DataFrame({"g_src": g1_list, "g_dst": g2_list, "loading": pair_loadings})
        fac_df = fac_df[fac_df["g_src"] != fac_df["g_dst"]].copy()
        fac_df["gene_1"] = np.minimum(fac_df["g_src"].to_numpy(), fac_df["g_dst"].to_numpy())
        fac_df["gene_2"] = np.maximum(fac_df["g_src"].to_numpy(), fac_df["g_dst"].to_numpy())
        fac_undir = (
            fac_df.groupby(["gene_1", "gene_2"], observed=True)["loading"]
            .mean()
            .reset_index()
        )
        df_bento_tensor = pd.DataFrame(
            {
                "gene_1": fac_undir["gene_1"],
                "gene_2": fac_undir["gene_2"],
                "score": fac_undir["loading"],
                "significant": False,
                "elapsed_seconds": t_total,
                "peak_rss_gb": _rss_gb(),
            }
        ).sort_values("score", ascending=False).reset_index(drop=True)
    else:
        df_bento_tensor = df_bento_clq.copy()

    return {
        "Bento (CLQ)": df_bento_clq,
        "Bento (RNAcoloc)": df_bento_tensor,
    }


# ---------------------------------------------------------------------------
# 5. Official pyTrance (1-layer DGI GCN + gene embedding similarity & Leiden)
# ---------------------------------------------------------------------------
def run_method_pytrance(
    bundle: dict,
    *,
    radius: float,
    hidden_dim: int = 32,
    n_epochs: int = 25,
    batch_size: int = 16,
    lr: float = 1e-3,
    max_spots_cap: int = 1_800_000,
    seed: int = 808,
) -> pd.DataFrame:
    """Run official pyTrance DGI GNN workflow on GPU and extract pairwise gene embedding correlations."""
    from anndata import AnnData
    import pytrance
    from scipy import sparse as sp
    import torch
    from torch_geometric.loader import DataLoader as PyGLoader

    t0 = time.perf_counter()
    df = bundle["data_df"][["cell", "gene", "x", "y"]].copy()
    df["cell"] = df["cell"].astype(str)
    df["gene"] = df["gene"].astype(str)

    # Note: pyTrance's CellData materializes dense one-hot (n_spots_cell x n_genes) tensors per cell;
    # if total spots exceed max_spots_cap (e.g., 4.6M spots in 300 U2OS cells), uniformly subsample
    # cells so pyTrance fits in RAM without crashing, while keeping all genes.
    if len(df) > max_spots_cap:
        rng = np.random.default_rng(seed)
        cids_all = sorted(df["cell"].unique())
        frac = max_spots_cap / float(len(df))
        keep_n = max(30, int(round(len(cids_all) * frac)))
        chosen = set(rng.choice(cids_all, size=min(keep_n, len(cids_all)), replace=False))
        df = df[df["cell"].isin(chosen)].reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    genes = sorted(df["gene"].unique())
    cids = sorted(df["cell"].unique())
    g2i = {g: i for i, g in enumerate(genes)}
    c2i = {c: i for i, c in enumerate(cids)}

    df["id"] = np.arange(len(df), dtype=np.int64)
    df["cell_encoded"] = df["cell"].map(c2i).astype(np.int32)

    rows_idx = np.arange(len(df), dtype=np.int64)
    cols_idx = df["gene"].map(g2i).to_numpy(dtype=np.int64)
    x_mat = sp.csr_matrix(
        (np.ones(len(df), dtype=np.float32), (rows_idx, cols_idx)),
        shape=(len(df), len(genes)),
    )
    adata = AnnData(X=x_mat, obs=df)
    adata.var_names = genes

    cell_indices = {
        cid: grp.index.to_numpy(dtype=np.int64)
        for cid, grp in adata.obs.groupby("cell", sort=False)
        if len(grp) >= 2
    }

    ds = pytrance.data.CellData(
        adata,
        cell_indices,
        graph_kwargs={"radius": float(radius), "x_key": "x", "y_key": "y"},
        seed=seed,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Relax strict deterministic flag for sparse CUDA matmul speed while fixing seeds.
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(8)
    torch.use_deterministic_algorithms(False)

    avg_spots_per_cell = len(df) / max(len(cell_indices), 1)
    eff_batch_size = max(2, min(int(batch_size), int(round(32000.0 / max(avg_spots_per_cell, 1.0)))))
    train_loader = PyGLoader(ds, batch_size=eff_batch_size, shuffle=True)
    model = pytrance.DGI(n_in=len(genes), n_h=[int(hidden_dim)]).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)

    for ep in range(int(n_epochs)):
        pytrance.train_epoch(model, train_loader, optimiser, device=device, seed=seed + ep)

    eval_loader = PyGLoader(ds, batch_size=eff_batch_size, shuffle=False)
    spot_embeds = pytrance.gnn.compute_embeddings(model, eval_loader, (adata.n_obs, int(hidden_dim)))
    gene_embeds_df = pytrance.tl.aggregate_transcript_embeddings(spot_embeds, adata, gene_key="gene")
    gene_embeds_num = gene_embeds_df.astype(np.float64)

    # Official Leiden clustering on gene embeddings to mark co-clustered pairs.
    try:
        clusters = pytrance.tl.cluster_gene_embeddings_leiden(
            gene_embeds_num,
            resolution=1.0,
            n_neighbors=min(10, max(2, len(genes) - 1)),
            return_labels=True,
            verbose=0,
            seed=seed,
        )
        cluster_arr = np.asarray(clusters, dtype=int)
    except Exception:
        cluster_arr = np.zeros(len(genes), dtype=int)

    # Pairwise Pearson correlation across latent dimensions (pyTrance Fig. 2d & Fig. 4c).
    emb_mat = gene_embeds_num.to_numpy(dtype=np.float64)
    corr_mat = np.corrcoef(emb_mat)
    corr_mat = np.nan_to_num(corr_mat, nan=0.0)

    ii, jj = np.triu_indices(len(genes), k=1)
    gene_arr = np.asarray(genes, dtype=object)
    same_cluster = cluster_arr[ii] == cluster_arr[jj]

    elapsed = time.perf_counter() - t0
    peak_gb = _rss_gb()

    out = pd.DataFrame(
        {
            "gene_1": np.minimum(gene_arr[ii], gene_arr[jj]),
            "gene_2": np.maximum(gene_arr[ii], gene_arr[jj]),
            "score": corr_mat[ii, jj],
            "significant": False,  # Embedding clustering has no pair-level p/q cutoff
            "same_leiden_cluster": same_cluster,
            "elapsed_seconds": elapsed,
            "peak_rss_gb": peak_gb,
        }
    ).sort_values("score", ascending=False).reset_index(drop=True)
    return out


def run_all_five_methods(
    bundle: dict,
    *,
    radius: float,
    distance_mode: str = "cell_scaled",
    bento_min_points: int = 5,
    pytrance_epochs: int = 25,
    pytrance_max_spots: int = 1_800_000,
    threads: int = 16,
) -> dict[str, pd.DataFrame]:
    """Execute all 5 method families on the same standardized bundle and return pair score tables."""
    results: dict[str, pd.DataFrame] = {}

    print(f"  -> [1/5] Running DeCoLoc (radius={radius:.2f}, mode={distance_mode}) ...")
    dec_dict = run_method_decoloc(
        bundle, radius=radius, distance_mode=distance_mode, threads=threads
    )
    results.update(dec_dict)

    print(f"  -> [2/5] Running InSTAnT (PP+CPB, radius={radius:.2f}) ...")
    results["InSTAnT (PP+CPB)"] = run_method_instant(
        bundle, radius=radius, threads=threads
    )

    print(f"  -> [3/5] Running SCRIN (cooccurrence, radius={radius:.2f}) ...")
    results["SCRIN (cooccurrence)"] = run_method_scrin(
        bundle, radius=radius, threads=threads
    )

    print(f"  -> [4/5] Running official Bento (CLQ & RNAcoloc, radius={radius:.2f}) ...")
    bento_dict = run_method_bento(
        bundle, radius=radius, min_points=bento_min_points, threads=threads
    )
    results.update(bento_dict)

    print(f"  -> [5/5] Running official pyTrance (DGI GNN, epochs={pytrance_epochs}, radius={radius:.2f}) ...")
    results["pyTrance (GNN)"] = run_method_pytrance(
        bundle,
        radius=radius,
        n_epochs=pytrance_epochs,
        max_spots_cap=pytrance_max_spots,
    )

    return results
