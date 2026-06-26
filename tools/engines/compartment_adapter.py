from __future__ import annotations

import json
import os
import pickle
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.interpolate import griddata
from scipy.spatial import cKDTree
from sklearn.decomposition import TruncatedSVD
from kneed import KneeLocator
from minisom import MiniSom
from shapely.geometry import Polygon
from shapely.wkt import loads
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


def _device_from_arg(device: Optional[str] = None):
    if device is not None and device != "auto":
        return torch.device(device)
    return torch.device("cuda:3" if torch.cuda.is_available() else "cpu")


def _as_str_series(s: pd.Series) -> pd.Series:
    return s.astype(str)


def _attach_batch_from_coordinates(
    data_df: pd.DataFrame,
    coordinates: Optional[pd.DataFrame],
    batch_col: str = "batch",
) -> pd.DataFrame:
    """Attach batch information to data_df.

    This is the only intentional adaptation to the notebook logic, because the
    user's unified PKL stores batch in coordinates rather than data_df.
    """
    df = data_df.copy()
    df["cell"] = df["cell"].astype(str)

    if batch_col in df.columns:
        df[batch_col] = df[batch_col].astype(str)
        return df

    if coordinates is not None and isinstance(coordinates, pd.DataFrame):
        coords = coordinates.copy()
        if "cell" in coords.columns and batch_col in coords.columns:
            coords["cell"] = coords["cell"].astype(str)
            coords[batch_col] = coords[batch_col].astype(str)
            batch_map = coords[["cell", batch_col]].drop_duplicates("cell")
            df = df.merge(batch_map, on="cell", how="left")
            df[batch_col] = df[batch_col].fillna("batch0").astype(str)
            return df

    df[batch_col] = "batch0"
    return df


def _make_shape_df(boundary: Dict[Any, pd.DataFrame], value_col: str) -> pd.DataFrame:
    """Notebook-exact boundary conversion: boundary df x/y -> WKT string."""
    data = []
    if boundary is None:
        return pd.DataFrame(columns=["cell", value_col])
    for cell_id, bdf in boundary.items():
        try:
            coords = list(zip(bdf["x"], bdf["y"]))
            if len(coords) >= 3:
                poly = Polygon(coords)
                wkt_polygon = poly.wkt
            else:
                wkt_polygon = "POLYGON EMPTY"
        except Exception:
            wkt_polygon = "POLYGON EMPTY"
        data.append({"cell": str(cell_id), value_col: wkt_polygon})
    out = pd.DataFrame(data)
    if len(out) > 0:
        out["cell"] = out["cell"].astype(str)
    return out


def _quantile_normalize_notebook(arr, lower_bound=0.1, upper_bound=0.9):
    """Exact function from notebook."""
    q_min = np.percentile(arr, 1, axis=0)
    q_max = np.percentile(arr, 99, axis=0)
    arr_normalized = (arr - q_min) / (q_max - q_min)
    arr_normalized = np.clip(arr_normalized, 0, 1)
    arr_scaled = arr_normalized * (lower_bound - upper_bound) + upper_bound
    return arr_scaled

def _interpolate_rgb_same_cell(
    df: pd.DataFrame,
    query_coords: List[np.ndarray],
    cell_ids: List[str],
    normalized_embeddings: np.ndarray,
) -> np.ndarray:
    """
    Interpolate RGB embeddings from sampled query points to all transcript points
    within each cell separately.

    This avoids cross-cell interpolation in dense FOV/tissue data.
    """
    df_work = df.copy()
    df_work["_row_index"] = np.arange(len(df_work))
    df_work["cell"] = df_work["cell"].astype(str)

    query_df = pd.DataFrame({
        "cell": [str(c) for c in cell_ids],
        "qx": np.asarray(query_coords)[:, 0],
        "qy": np.asarray(query_coords)[:, 1],
        "r": normalized_embeddings[:, 0],
        "g": normalized_embeddings[:, 1],
        "b": normalized_embeddings[:, 2],
    })

    rgb_image = np.full((len(df_work), 3), np.nan, dtype=float)

    global_mean = np.nanmean(normalized_embeddings, axis=0)

    for cell_id, cell_points in df_work.groupby("cell", sort=False):
        idx = cell_points["_row_index"].to_numpy()
        target_xy = cell_points[["x", "y"]].to_numpy()

        q = query_df[query_df["cell"] == str(cell_id)]

        if len(q) == 0:
            rgb_image[idx, :] = global_mean
            continue

        source_xy = q[["qx", "qy"]].to_numpy()
        source_rgb = q[["r", "g", "b"]].to_numpy()

        # 如果一个 cell 内 query 点太少，不能做 linear interpolation，直接用 nearest。
        if len(q) < 3:
            for ch in range(3):
                rgb_image[idx, ch] = griddata(
                    source_xy,
                    source_rgb[:, ch],
                    target_xy,
                    method="nearest",
                )
            continue

        # 先 linear，cell 边缘或凸包外 nan 再 nearest 补
        for ch in range(3):
            linear_vals = griddata(
                source_xy,
                source_rgb[:, ch],
                target_xy,
                method="linear",
            )
            nearest_vals = griddata(
                source_xy,
                source_rgb[:, ch],
                target_xy,
                method="nearest",
            )
            vals = np.where(np.isnan(linear_vals), nearest_vals, linear_vals)
            rgb_image[idx, ch] = vals

    rgb_image = np.nan_to_num(
        rgb_image,
        nan=np.nanmean(normalized_embeddings),
        posinf=np.nanmean(normalized_embeddings),
        neginf=np.nanmean(normalized_embeddings),
    )

    rgb_image = np.clip(rgb_image, 0, 1)
    return rgb_image

def _compute_local_expression_vectors_notebook(
    df: pd.DataFrame,
    sampled_df: pd.DataFrame,
    unique_genes: np.ndarray,
    radius: float,
    device,
    profile: bool = False,
    same_cell_neighborhood: bool = False,
):
    """Notebook-exact local composition loop using value_counts + reindex."""
    num_genes = len(unique_genes)
    tree = cKDTree(df[["x", "y"]].values)
    query_coords = []
    expression_vectors = []
    cell_ids = []
    i = 1
    for idx, row in sampled_df.iterrows():
        point = row[["x", "y"]].values
        cell_id = row["cell"]
        cell_ids.append(cell_id)
        indices = tree.query_ball_point(point, radius)
        nearby_cells = df.iloc[indices]
        
        if same_cell_neighborhood:
            nearby_cells = nearby_cells[
                nearby_cells["cell"].astype(str) == str(cell_id)
            ]

        if nearby_cells.empty:
            expression_vector = torch.zeros(num_genes, device=device)
        else:
            gene_count = nearby_cells["gene"].value_counts()
            expression_vector = torch.tensor(
                gene_count.reindex(unique_genes, fill_value=0).values,
                device=device,
                dtype=torch.float32,
            )
            total = expression_vector.sum()
            if total > 0:
                expression_vector = expression_vector / total
            else:
                expression_vector = torch.zeros(num_genes, device=device)
        query_coords.append(point)
        expression_vectors.append(expression_vector)
        if profile and i % 10000 == 0:
            print(f"The grid_points is {i}/{len(sampled_df)}.", flush=True)
        i += 1
    return query_coords, expression_vectors, cell_ids


def _compute_cell_vectors_notebook(df: pd.DataFrame, unique_genes: np.ndarray, device):
    """Notebook-exact cell-level profile + torch.std."""
    cell_vectors = []
    for _, group in df.groupby("cell"):
        gene_count = group["gene"].value_counts()
        cell_vector = gene_count.reindex(unique_genes, fill_value=0).values
        cell_vector = cell_vector / cell_vector.sum()
        cell_vectors.append(cell_vector)
    cell_vectors = torch.tensor(cell_vectors, device=device)
    std_dev = torch.std(cell_vectors, dim=0)
    return cell_vectors, std_dev


def _compute_rnaflux_loop_notebook(
    df: pd.DataFrame,
    unique_genes: np.ndarray,
    query_coords: List[np.ndarray],
    expression_vectors: List[torch.Tensor],
    cell_ids: List[str],
    std_dev: torch.Tensor,
    device,
    profile: bool = False,
):
    """Notebook-exact loop RNAflux embedding."""
    rnaflux_embeddings = []
    num = 1
    for i, query_coord in enumerate(query_coords):
        nearest_cell_vector = (
            df[df["cell"] == cell_ids[i]]["gene"]
            .value_counts()
            .reindex(unique_genes, fill_value=0)
            .values
        )
        nearest_cell_vector = nearest_cell_vector / nearest_cell_vector.sum()
        rnaflux_embedding = (
            torch.tensor(expression_vectors[i], device=device).clone().detach()
            - torch.tensor(nearest_cell_vector, device=device).clone().detach()
        ) / std_dev
        rnaflux_embeddings.append(rnaflux_embedding.cpu().numpy())
        if profile and num % 100 == 0:
            print(f"The grid_points is {num}/{len(query_coords)}.", flush=True)
        num += 1
    rnaflux_embeddings = np.array(rnaflux_embeddings)
    return rnaflux_embeddings


def _compute_rnaflux_vectorized_cardiomyocytes(
    df: pd.DataFrame,
    unique_genes: np.ndarray,
    expression_vectors: List[torch.Tensor],
    cell_ids: List[str],
    std_dev: torch.Tensor,
):
    """Cardiomyocytes notebook vectorized implementation.

    This is optional and may not be byte-for-byte identical to the loop path.
    """
    cell_gene_counts = pd.crosstab(df["cell"], df["gene"]).reindex(columns=unique_genes, fill_value=0)
    cell_gene_profiles = cell_gene_counts.div(cell_gene_counts.sum(axis=1), axis=0)
    nearest_cell_matrix = cell_gene_profiles.loc[cell_ids].values
    expression_vectors_np = np.array(expression_vectors)
    rnaflux_embeddings = (expression_vectors_np - nearest_cell_matrix) / std_dev
    return np.array(rnaflux_embeddings)


def _run_som_notebook(
    rgb_image: np.ndarray,
    n_clusters: Union[str, int] = "auto",
    cluster_range_min: int = 2,
    cluster_range_max: int = 12,
    som_iterations: int = 1000,
    som_sigma: float = 1.0,
    som_learning_rate: float = 0.5,
):
    """Notebook-exact SOM logic.

    Uses np.random.seed(42), MiniSom without random_seed, range(2,13),
    KneeLocator(curve='convex', direction='decreasing'), and raw cluster_numbers.
    """
    np.random.seed(42)

    if str(n_clusters).lower() == "auto":
        som_models = {}
        quantization_errors = []
        k_values = list(range(cluster_range_min, cluster_range_max + 1))
        for k in k_values:
            som = MiniSom(x=1, y=k, input_len=3, sigma=som_sigma, learning_rate=som_learning_rate)
            som.train(rgb_image, som_iterations)
            som_models[k] = som
            quantization_error = som.quantization_error(rgb_image)
            quantization_errors.append(quantization_error)
        kl = KneeLocator(k_values, quantization_errors, curve="convex", direction="decreasing")
        best_k = kl.elbow
        if best_k is None:
            raise RuntimeError("KneeLocator did not find an elbow for n_clusters auto, same as notebook would fail here. Please set --n-clusters manually.")
        som = som_models[best_k]
    else:
        best_k = int(n_clusters)
        som = MiniSom(x=1, y=best_k, input_len=3, sigma=som_sigma, learning_rate=som_learning_rate)
        som.train(rgb_image, som_iterations)
        quantization_errors = []

    cluster_assignments = np.array([som.winner(e) for e in rgb_image])
    cluster_numbers = np.array([i * best_k + j for i, j in cluster_assignments])
    return cluster_numbers, int(best_k), quantization_errors

def _auto_figsize_from_xy(
    df: pd.DataFrame,
    x_col: str = "x",
    y_col: str = "y",
    base: float = 10.0,
    min_size: float = 5.0,
    max_size: float = 24.0,
    pad_ratio: float = 0.03,
):
    """
    Automatically determine figure size and plot limits from x/y coordinate range.

    This keeps the spatial aspect ratio correct for each batch/FOV.
    """
    x = pd.to_numeric(df[x_col], errors="coerce")
    y = pd.to_numeric(df[y_col], errors="coerce")

    x_min, x_max = np.nanmin(x), np.nanmax(x)
    y_min, y_max = np.nanmin(y), np.nanmax(y)

    x_range = x_max - x_min
    y_range = y_max - y_min

    if not np.isfinite(x_range) or not np.isfinite(y_range) or x_range <= 0 or y_range <= 0:
        return (base, base), None, None

    aspect = x_range / y_range

    if aspect >= 1:
        width = base * aspect
        height = base
    else:
        width = base
        height = base / aspect

    width = min(max(width, min_size), max_size)
    height = min(max(height, min_size), max_size)

    x_pad = x_range * pad_ratio
    y_pad = y_range * pad_ratio

    xlim = (x_min - x_pad, x_max + x_pad)
    ylim = (y_min - y_pad, y_max + y_pad)

    return (width, height), xlim, ylim


def _resolve_figsize_and_limits(df: pd.DataFrame, figsize):
    """
    figsize can be:
    - "auto"
    - None
    - (width, height)
    """
    if figsize is None or str(figsize).lower() == "auto":
        return _auto_figsize_from_xy(df)

    return figsize, None, None
# def _plot_embedding_notebook(
#     df: pd.DataFrame,
#     rgb_image: np.ndarray,
#     out_png: Path,
#     figsize=(8, 6),
#     dpi=300,
#     point_size: float = 10,
# ):
#     plt.figure(figsize=figsize)
#     plt.scatter(df["x"], df["y"], c=rgb_image, s=point_size, marker="o")
#     plt.title("RNAflux Embeddings Visualization")
#     plt.xlabel("X Coordinate")
#     plt.ylabel("Y Coordinate")
#     plt.savefig(out_png, format="png", dpi=dpi)
#     plt.close()
def _plot_embedding_notebook(
    df: pd.DataFrame,
    rgb_image: np.ndarray,
    out_png: Path,
    figsize="auto",
    dpi=300,
    point_size: float = 10,
):
    figsize, xlim, ylim = _resolve_figsize_and_limits(df, figsize)

    fig, ax = plt.subplots(figsize=figsize)

    ax.scatter(
        df["x"],
        df["y"],
        c=rgb_image,
        s=point_size,
        marker="o",
        linewidths=0,
    )

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    ax.set_aspect("equal", adjustable="box")
    ax.set_title("RNAflux Embeddings Visualization")
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")

    plt.tight_layout()
    plt.savefig(out_png, format="png", dpi=dpi, bbox_inches="tight")
    plt.close()

# def _plot_subdomains_notebook(
#     df: pd.DataFrame,
#     cell_shape: pd.DataFrame,
#     nucleus_shape: pd.DataFrame,
#     out_png: Path,
#     figsize=(8, 6),
#     dpi=300,
#     point_size: float = 1,
# ):
#     num_clusters = len(np.unique(df["cluster"]))
#     cmap = ListedColormap(plt.cm.viridis(np.linspace(0, 1, num_clusters)))

#     plt.figure(figsize=figsize)
#     scatter = plt.scatter(
#         df["x"],
#         df["y"],
#         c=df["cluster"],
#         cmap=cmap,
#         s=point_size,
#         alpha=0.7,
#     )
#     cbar = plt.colorbar(scatter, label="Cluster")
#     cbar.set_ticks(np.arange(num_clusters))
#     cbar.set_ticklabels(np.arange(num_clusters))
#     cbar.ax.invert_yaxis()

#     plt.xlabel("X Coordinate")
#     plt.ylabel("Y Coordinate")
#     plt.title("SOM-based Clustering of RNAflux Embeddings")

#     for index, row in cell_shape.iterrows():
#         polygon = loads(row["cell_shape"])
#         if not polygon.is_empty:
#             x, y = polygon.exterior.xy
#             plt.plot(x, y, linestyle="-", color="black")

#     for index, row in nucleus_shape.iterrows():
#         polygon = loads(row["nucleus_shape"])
#         if not polygon.is_empty:
#             x, y = polygon.exterior.xy
#             plt.plot(x, y, linestyle="-", color="red")

#     plt.savefig(out_png, format="png", dpi=dpi)
#     plt.close()
def _plot_subdomains_notebook(
    df: pd.DataFrame,
    cell_shape: pd.DataFrame,
    nucleus_shape: pd.DataFrame,
    out_png: Path,
    figsize="auto",
    dpi=300,
    point_size: float = 1,
):
    figsize, xlim, ylim = _resolve_figsize_and_limits(df, figsize)

    clusters_sorted = sorted(pd.unique(df["cluster"]))
    num_clusters = len(clusters_sorted)
    cmap = ListedColormap(plt.cm.viridis(np.linspace(0, 1, num_clusters)))

    fig, ax = plt.subplots(figsize=figsize)

    scatter = ax.scatter(
        df["x"],
        df["y"],
        c=df["cluster"],
        cmap=cmap,
        s=point_size,
        alpha=0.7,
        linewidths=0,
    )

    cbar = plt.colorbar(scatter, ax=ax, label="Cluster")
    cbar.set_ticks(np.arange(num_clusters))
    cbar.set_ticklabels(np.arange(num_clusters))
    cbar.ax.invert_yaxis()

    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.set_title("SOM-based Clustering of RNAflux Embeddings")

    for index, row in cell_shape.iterrows():
        polygon = loads(row["cell_shape"])
        if not polygon.is_empty:
            x, y = polygon.exterior.xy
            ax.plot(x, y, linestyle="-", color="black", linewidth=0.8)

    for index, row in nucleus_shape.iterrows():
        polygon = loads(row["nucleus_shape"])
        if not polygon.is_empty:
            x, y = polygon.exterior.xy
            ax.plot(x, y, linestyle="-", color="red", linewidth=0.7)

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    ax.set_aspect("equal", adjustable="box")

    plt.tight_layout()
    plt.savefig(out_png, format="png", dpi=dpi, bbox_inches="tight")
    plt.close()


def run_sampled_compartment_for_batch(
    data_df: pd.DataFrame,
    cell_boundary: Dict[Any, pd.DataFrame],
    nuclear_boundary: Dict[Any, pd.DataFrame],
    *,
    batch: str,
    batch_col: str,
    dataset_name: str,
    out_prefix: Union[str, Path],
    frac: float = 0.01,
    radius: float = 40,
    n_clusters: Union[str, int] = "auto",
    cluster_range_min: int = 2,
    cluster_range_max: int = 12,
    embedding_mode: str = "loop",
    som_iterations: int = 1000,
    som_sigma: float = 1.0,
    som_learning_rate: float = 0.5,
    device: Optional[str] = "auto",
    dpi: int = 300,
    export_csv: bool = True,
    profile: bool = False,
    figsize=None,
    embedding_point_size=10,
    subdomain_point_size=1,
    same_cell_neighborhood: bool = False,
) -> Dict[str, Any]:
    start_time = time.time()
    out_prefix = Path(out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    torch_device = _device_from_arg(device)

    df = data_df[data_df[batch_col].astype(str) == str(batch)].copy()
    if len(df) == 0:
        raise ValueError(f"No transcripts found for batch={batch!r}")

    # Notebook exact after filtering batch
    cell_shape = _make_shape_df(cell_boundary, "cell_shape")
    nucleus_shape = _make_shape_df(nuclear_boundary, "nucleus_shape")

    cell_list = df["cell"].unique().tolist()
    cell_shape = cell_shape[cell_shape["cell"].isin(cell_list)]
    nucleus_shape = nucleus_shape[nucleus_shape["cell"].isin(cell_list)]

    df["cell"] = df["cell"].astype(str)
    df["gene"] = df["gene"].astype(str)

    # EXACT notebook sampling
    sampled_df = df.sample(frac=frac, random_state=42)
    print(len(sampled_df), flush=True)
    if len(sampled_df) == 0:
        raise ValueError("sampled_df is empty. This follows notebook df.sample(frac=frac); increase --frac.")

    unique_genes = df["gene"].unique()
    if profile:
        print(f"[SAMPLED] batch={batch}, cells={df['cell'].nunique()}, transcripts={len(df)}, genes={len(unique_genes)}, sampled={len(sampled_df)}", flush=True)
        print(f"[SAMPLED] frac={frac}, radius={radius}, embedding_mode={embedding_mode}", flush=True)
        print(f"[SAMPLED] same_cell_neighborhood={same_cell_neighborhood}")
    if same_cell_neighborhood and embedding_mode != "loop":
        print(
            "[SAMPLED][WARN] same_cell_neighborhood=True requires loop mode. "
            "Forcing embedding_mode='loop'.",
            flush=True,
        )
        embedding_mode = "loop"

    # query_coords, expression_vectors, cell_ids = _compute_local_expression_vectors_notebook(
    #     df, sampled_df, unique_genes, radius, torch_device, profile=profile
    # )
    query_coords, expression_vectors, cell_ids = _compute_local_expression_vectors_notebook(
        df,
        sampled_df,
        unique_genes,
        radius,
        torch_device,
        profile=profile,
        same_cell_neighborhood=same_cell_neighborhood,
    )
    cell_vectors, std_dev = _compute_cell_vectors_notebook(df, unique_genes, torch_device)

    # print(cell_vectors) and print(std_dev) from notebook are intentionally not forced by default
    # because they can be huge; enable profile if you need progress messages.

    if embedding_mode == "loop":
        rnaflux_embeddings = _compute_rnaflux_loop_notebook(
            df, unique_genes, query_coords, expression_vectors, cell_ids, std_dev, torch_device, profile=profile
        )
    elif embedding_mode == "vectorized":
        rnaflux_embeddings = _compute_rnaflux_vectorized_cardiomyocytes(
            df, unique_genes, expression_vectors, cell_ids, std_dev
        )
    else:
        raise ValueError("embedding_mode must be 'loop' or 'vectorized'")

    query_coords_np = np.array(query_coords)
    result_df = pd.DataFrame({
        "x": query_coords_np[:, 0],
        "y": query_coords_np[:, 1],
        "rnaflux_embedding": list(rnaflux_embeddings),
    })

    pca = TruncatedSVD(n_components=3)
    pca_embeddings = pca.fit_transform(rnaflux_embeddings)
    normalized_embeddings = _quantile_normalize_notebook(pca_embeddings)

    # grid_x, grid_y = df[["x", "y"]].values.T
    # r_values = normalized_embeddings[:, 0]
    # g_values = normalized_embeddings[:, 1]
    # b_values = normalized_embeddings[:, 2]

    # r_interp = griddata(query_coords, r_values, (grid_x, grid_y), method="linear")
    # g_interp = griddata(query_coords, g_values, (grid_x, grid_y), method="linear")
    # b_interp = griddata(query_coords, b_values, (grid_x, grid_y), method="linear")

    # r_interp = np.nan_to_num(r_interp, nan=np.nanmean(r_values))
    # g_interp = np.nan_to_num(g_interp, nan=np.nanmean(g_values))
    # b_interp = np.nan_to_num(b_interp, nan=np.nanmean(b_values))

    # rgb_image = np.stack([r_interp, g_interp, b_interp], axis=-1)
    # df["rgb_image"] = list(rgb_image)
    if same_cell_neighborhood:
        rgb_image = _interpolate_rgb_same_cell(
            df=df,
            query_coords=query_coords,
            cell_ids=cell_ids,
            normalized_embeddings=normalized_embeddings,
        )
    else:
        grid_x, grid_y = df[["x", "y"]].values.T
        r_values = normalized_embeddings[:, 0]
        g_values = normalized_embeddings[:, 1]
        b_values = normalized_embeddings[:, 2]

        r_interp = griddata(query_coords, r_values, (grid_x, grid_y), method="linear")
        g_interp = griddata(query_coords, g_values, (grid_x, grid_y), method="linear")
        b_interp = griddata(query_coords, b_values, (grid_x, grid_y), method="linear")

        r_interp = np.nan_to_num(r_interp, nan=np.nanmean(r_values))
        g_interp = np.nan_to_num(g_interp, nan=np.nanmean(g_values))
        b_interp = np.nan_to_num(b_interp, nan=np.nanmean(b_values))

        rgb_image = np.stack([r_interp, g_interp, b_interp], axis=-1)

    df["rgb_image"] = list(rgb_image)

    embedding_png = Path(f"{out_prefix}_sampled_rnaflux_embedding.png")
    subdomains_png = Path(f"{out_prefix}_sampled_subdomains.png")
    rnaembedding_csv = Path(f"{out_prefix}_sampled_rnaembedding.csv")
    allpoints_csv = Path(f"{out_prefix}_sampled_df.csv")
    # Lightweight point-level compartment table for downstream visualization.
    # This is exported by default and is much smaller/easier to consume than sampled_df.csv.
    subdomain_points_csv = Path(f"{out_prefix}_sampled_subdomain_points.csv")
    query_csv = Path(f"{out_prefix}_sampled_query_embeddings.csv")
    meta_json = Path(f"{out_prefix}_sampled_meta.json")

    # _plot_embedding_notebook(df, rgb_image, embedding_png, figsize=(8, 6), dpi=dpi)
    _plot_embedding_notebook(
        df,
        rgb_image,
        embedding_png,
        figsize=figsize,
        dpi=dpi,
        point_size=embedding_point_size,
    )
    # Notebook saves rnaembedding before SOM, but writing full CSV can be expensive.
    # Here it is controlled by export_csv to avoid unintentional huge files.
    if export_csv:
        pd.DataFrame(df).to_csv(rnaembedding_csv)

    cluster_numbers, best_k, quantization_errors = _run_som_notebook(
        rgb_image,
        n_clusters=n_clusters,
        cluster_range_min=cluster_range_min,
        cluster_range_max=cluster_range_max,
        som_iterations=som_iterations,
        som_sigma=som_sigma,
        som_learning_rate=som_learning_rate,
    )
    print("The best cluster number is: ", best_k, flush=True)
    df["cluster"] = cluster_numbers

    # Standardized name for downstream visualization modules.
    # cluster is kept for backward compatibility; subdomain is the semantic label.
    df["subdomain"] = df["cluster"].astype(int)

    # Store interpolated RNAflux RGB values as numeric columns instead of a list-like object.
    rgb_arr = np.asarray(rgb_image)
    df["rnaflux_r"] = rgb_arr[:, 0]
    df["rnaflux_g"] = rgb_arr[:, 1]
    df["rnaflux_b"] = rgb_arr[:, 2]

    # Always export a lightweight point-level compartment table for visualization.
    # Keep available metadata columns if present, and avoid duplicate group column names.
    subdomain_cols = []
    for col in [
        "cell",
        "gene",
        "x",
        "y",
        "z",
        "umi",
        batch_col,
        "batch",
        "fov",
        "sample_id",
        "celltype",
        "annotation",
    ]:
        if col in df.columns and col not in subdomain_cols:
            subdomain_cols.append(col)

    for col in ["subdomain", "rnaflux_r", "rnaflux_g", "rnaflux_b"]:
        if col in df.columns and col not in subdomain_cols:
            subdomain_cols.append(col)

    df[subdomain_cols].to_csv(subdomain_points_csv, index=False)

    # _plot_subdomains_notebook(df, cell_shape, nucleus_shape, subdomains_png, figsize=(8, 6), dpi=dpi)
    _plot_subdomains_notebook(
        df,
        cell_shape,
        nucleus_shape,
        subdomains_png,
        figsize=figsize,
        dpi=dpi,
        point_size=subdomain_point_size,
    )

    # Notebook saves df again after clusters; keep optional because full transcript CSV is large.
    if export_csv:
        pd.DataFrame(df).to_csv(allpoints_csv)

    result_df.to_csv(query_csv, index=False)

    run_time = (time.time() - start_time) / 60
    print(f"Dataset {dataset_name} - Batch {batch} running time ：{run_time} mins", flush=True)

    meta = {
        "backend": "sampled_rnaflux_som_notebook_exact",
        "dataset": dataset_name,
        "batch": str(batch),
        "frac": float(frac),
        "radius": float(radius),
        "n_clusters": str(n_clusters),
        "best_k": int(best_k),
        "embedding_mode": embedding_mode,
        "same_cell_neighborhood": bool(same_cell_neighborhood),
        "som_iterations": int(som_iterations),
        "som_sigma": float(som_sigma),
        "som_learning_rate": float(som_learning_rate),
        "n_transcripts": int(len(df)),
        "n_sampled": int(len(sampled_df)),
        "n_cells": int(df["cell"].nunique()),
        "n_genes": int(len(unique_genes)),
        "run_time_min": float(run_time),
        "outputs": {
            "embedding_png": str(embedding_png),
            "subdomains_png": str(subdomains_png),
            "subdomain_points_csv": str(subdomain_points_csv),
            "query_embeddings_csv": str(query_csv),
            "rnaembedding_csv": str(rnaembedding_csv) if export_csv else None,
            "allpoints_csv": str(allpoints_csv) if export_csv else None,
            "meta_json": str(meta_json),
        },
    }
    with open(meta_json, "w") as f:
        json.dump(meta, f, indent=2)
    return meta


def _parse_only_batches(only_batches: Optional[Union[str, Sequence[str]]]) -> Optional[List[str]]:
    if only_batches is None:
        return None
    if isinstance(only_batches, str):
        return [x.strip() for x in only_batches.split(",") if x.strip()]
    return [str(x) for x in only_batches]


def run_sampled_compartment_from_pkl(
    pkl_path: Union[str, Path],
    out_prefix: Union[str, Path],
    *,
    batch_col: str = "batch",
    only_batches: Optional[Union[str, Sequence[str]]] = None,
    frac: float = 0.01,
    radius: float = 40,
    n_clusters: Union[str, int] = "auto",
    cluster_range_min: int = 2,
    cluster_range_max: int = 12,
    embedding_mode: str = "loop",
    som_iterations: int = 1000,
    som_sigma: float = 1.0,
    som_learning_rate: float = 0.5,
    device: Optional[str] = "auto",
    dpi: int = 300,
    export_csv: bool = False,
    profile: bool = False,
    figsize=None,
    embedding_point_size=10,
    subdomain_point_size=1,
    same_cell_neighborhood: bool = False,
) -> Dict[str, Any]:
    pkl_path = Path(pkl_path)
    out_prefix = Path(out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    with open(pkl_path, "rb") as f:
        obj = pickle.load(f)

    data_df = obj["data_df"].copy()
    cell_boundary = obj.get("cell_boundary", {})
    nuclear_boundary = obj.get("nuclear_boundary", {})
    coordinates = obj.get("coordinates", None)

    data_df["cell"] = data_df["cell"].astype(str)
    data_df["gene"] = data_df["gene"].astype(str)
    data_df = _attach_batch_from_coordinates(data_df, coordinates, batch_col=batch_col)

    all_batches = [str(x) for x in pd.unique(data_df[batch_col].astype(str))]
    selected = _parse_only_batches(only_batches)
    if selected is not None:
        run_batches = [b for b in all_batches if str(b) in set(selected)]
        missing = sorted(set(selected) - set(run_batches))
        if missing:
            raise ValueError(f"Requested batches not found: {missing}. Available: {all_batches}")
    else:
        run_batches = all_batches

    if profile:
        print("[SAMPLED] sampled RNAflux/SOM notebook-exact backend", flush=True)
        print(f"[SAMPLED] pkl          : {pkl_path}", flush=True)
        print(f"[SAMPLED] out_prefix   : {out_prefix}", flush=True)
        # print(f"[SAMPLED] batches      : {run_batches}", flush=True)
        print(f"[SAMPLED] group_col    : {batch_col}", flush=True)
        print(f"[SAMPLED] groups       : {run_batches}", flush=True)
        print(f"[SAMPLED] frac         : {frac}", flush=True)
        print(f"[SAMPLED] radius       : {radius}", flush=True)
        print(f"[SAMPLED] n_clusters   : {n_clusters}", flush=True)
        print(f"[SAMPLED] embedding    : {embedding_mode}", flush=True)

    rows = []
    metas = []
    dataset_name = pkl_path.stem.replace("_data_dict", "")

    # for batch in run_batches:
    #     safe_batch = str(batch).replace("/", "_").replace(" ", "_")
    #     batch_prefix = Path(f"{out_prefix}_batch-{safe_batch}")
    group_name = str(batch_col).replace("/", "_").replace(" ", "_")
    for batch in run_batches:
        safe_batch = str(batch).replace("/", "_").replace(" ", "_")
        batch_prefix = Path(f"{out_prefix}_{group_name}-{safe_batch}")
        try:
            meta = run_sampled_compartment_for_batch(
                data_df,
                cell_boundary,
                nuclear_boundary,
                batch=batch,
                batch_col=batch_col,
                dataset_name=dataset_name,
                out_prefix=batch_prefix,
                frac=frac,
                radius=radius,
                n_clusters=n_clusters,
                cluster_range_min=cluster_range_min,
                cluster_range_max=cluster_range_max,
                embedding_mode=embedding_mode,
                som_iterations=som_iterations,
                som_sigma=som_sigma,
                som_learning_rate=som_learning_rate,
                device=device,
                dpi=dpi,
                export_csv=export_csv,
                profile=profile,
                figsize=figsize,
                embedding_point_size=embedding_point_size,
                subdomain_point_size=subdomain_point_size,
                same_cell_neighborhood=same_cell_neighborhood,
            )
            rows.append({"batch": batch, "status": "done", **meta})
            metas.append(meta)
        except Exception as e:
            rows.append({"batch": batch, "status": "failed", "error": repr(e)})
            if profile:
                print(f"[SAMPLED] Batch {batch} FAILED: {repr(e)}", flush=True)

    summary_csv = Path(f"{out_prefix}_sampled_batches.csv")
    summary_json = Path(f"{out_prefix}_sampled_batches_meta.json")
    pd.DataFrame(rows).to_csv(summary_csv, index=False)
    with open(summary_json, "w") as f:
        json.dump({"pkl": str(pkl_path), "out_prefix": str(out_prefix), "batches": rows}, f, indent=2)

    return {"summary_csv": str(summary_csv), "summary_json": str(summary_json), "batches": rows}
