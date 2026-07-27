from __future__ import annotations

import json
import os
import pickle
import tempfile
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

from tools.engines.compartment_same_cell_fast import (
    compute_same_cell_rnaflux_fast,
    interpolate_same_cell_rgb_fast,
)

warnings.filterwarnings("ignore", category=UserWarning)


def _device_from_arg(device: Optional[str] = None):
    if device is not None and device != "auto":
        return torch.device(device)
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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
    # 该函数只在当前运行内部使用，直接复用输入表可避免超大数据集产生整表副本。
    df = data_df
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
    if len(out) == 0:
        # 空边界仍然保留列名，避免无核/无细胞边界的小批次在后续取cell列时失败。
        return pd.DataFrame(columns=["cell", value_col])
    out["cell"] = out["cell"].astype(str)
    return out


def _filter_boundary_dict(boundary: Dict[Any, pd.DataFrame], cell_ids: set[str]) -> Dict[Any, pd.DataFrame]:
    """按细胞集合筛选边界字典，同时保持原字典迭代顺序。

    Parameters
    ----------
    boundary
        原始细胞或细胞核边界字典。
    cell_ids
        当前计算组包含的字符串细胞编号。

    Returns
    -------
    dict
        仅包含当前组细胞的边界字典；键和值均保持原对象和原顺序。
    """

    if boundary is None:
        return {}
    return {cell_id: boundary_df for cell_id, boundary_df in boundary.items() if str(cell_id) in cell_ids}


def _cell_radius_from_boundary(boundary_df: pd.DataFrame) -> Optional[float]:
    """计算一个细胞的边界点到Polygon质心的平均二维距离。

    Parameters
    ----------
    boundary_df
        至少包含``x``和``y``列的单细胞边界表。

    Returns
    -------
    float or None
        有效正半径；边界无效时返回``None``。
    """

    if not isinstance(boundary_df, pd.DataFrame):
        return None
    if "x" not in boundary_df.columns or "y" not in boundary_df.columns:
        return None
    x_values = pd.to_numeric(boundary_df["x"], errors="coerce").to_numpy(dtype=np.float64)
    y_values = pd.to_numeric(boundary_df["y"], errors="coerce").to_numpy(dtype=np.float64)
    finite_mask = np.isfinite(x_values) & np.isfinite(y_values)
    boundary_xy = np.column_stack((x_values[finite_mask], y_values[finite_mask]))
    if len(boundary_xy) < 3:
        return None
    try:
        polygon = Polygon(boundary_xy)
        if polygon.is_empty:
            return None
        centroid = polygon.centroid
        point_distances = np.sqrt(
            (boundary_xy[:, 0] - centroid.x) ** 2 +
            (boundary_xy[:, 1] - centroid.y) ** 2
        )
        cell_radius = float(np.mean(point_distances, dtype=np.float64))
    except Exception:
        return None
    if not np.isfinite(cell_radius) or cell_radius <= 0:
        return None
    return cell_radius


def _calculate_mean_cell_radius(boundary: Dict[Any, pd.DataFrame], cell_ids: set[str]) -> float:
    """根据当前计算组有效细胞边界计算平均细胞半径。

    每个细胞先使用边界点构建Polygon并计算质心，再计算全部有限边界点到质心的
    欧氏距离均值作为该细胞代表半径，最后对全部有效细胞代表半径取平均。

    Parameters
    ----------
    boundary
        细胞边界字典，每个值应至少包含``x``和``y``列。
    cell_ids
        当前计算组包含的字符串细胞编号。

    Returns
    -------
    float
        当前计算组的平均细胞半径。

    Raises
    ------
    ValueError
        当前计算组没有可用于计算半径的有效细胞边界时抛出。
    """

    cell_radii = []
    for cell_id, boundary_df in (boundary or {}).items():
        if str(cell_id) not in cell_ids:
            continue
        cell_radius = _cell_radius_from_boundary(boundary_df)
        if cell_radius is not None:
            cell_radii.append(cell_radius)
    if not cell_radii:
        raise ValueError(
            "radius='auto' requires at least one valid cell boundary in the current computation group. "
            "Provide valid cell_boundary data or pass an explicit positive --radius value."
        )
    return float(np.mean(cell_radii, dtype=np.float64))


def _resolve_compartment_radius(
    radius: Union[str, float],
    boundary: Dict[Any, pd.DataFrame],
    cell_ids: set[str],
) -> float:
    """把自动或显式半径解析为当前组实际使用的正浮点数半径。

    Parameters
    ----------
    radius
        ``auto``表示使用当前组平均细胞半径的三分之一；数字表示绝对坐标半径。
    boundary
        用于自动半径计算的细胞边界字典。
    cell_ids
        当前计算组包含的字符串细胞编号。

    Returns
    -------
    float
        实际传给KDTree邻域统计的正半径。

    Raises
    ------
    ValueError
        半径输入非法或自动模式缺少有效边界时抛出。
    """

    if isinstance(radius, str) and radius.strip().lower() == "auto":
        return _calculate_mean_cell_radius(boundary, cell_ids) / 3.0
    try:
        resolved_radius = float(radius)
    except (TypeError, ValueError) as error:
        raise ValueError("radius must be 'auto' or a positive number") from error
    if not np.isfinite(resolved_radius) or resolved_radius <= 0:
        raise ValueError("radius must be a finite number greater than 0")
    return resolved_radius


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


def _som_quantization_error_chunked(
    som: MiniSom,
    rgb_image: np.ndarray,
    *,
    chunk_size: int,
    profile: bool = False,
    progress_label: str = "SOM quantization",
) -> float:
    """按原始行顺序分块计算 MiniSom 量化误差，避免创建全量 ``N × k`` 距离矩阵。

    Parameters
    ----------
    som
        已完成训练的 MiniSom 模型。
    rgb_image
        与原流程相同的全量三维 RNAflux RGB 数组。
    chunk_size
        每个分块包含的转录本数量，必须为正整数。
    profile
        是否输出完成数量、百分比、速度和预计剩余时间。
    progress_label
        进度日志前缀，用于区分不同候选聚类数。

    Returns
    -------
    float
        所有转录本到其最佳匹配 SOM 节点距离的平均值。

    Raises
    ------
    ValueError
        当 ``chunk_size`` 非正或输入为空时抛出。
    """
    if chunk_size <= 0:
        raise ValueError("som_chunk_size must be a positive integer")
    total_points = len(rgb_image)
    if total_points == 0:
        raise ValueError("rgb_image is empty")

    distance_sum = 0.0
    completed_points = 0
    progress_start = time.time()
    for chunk_start in range(0, total_points, chunk_size):
        chunk_end = min(chunk_start + chunk_size, total_points)
        rgb_chunk = np.asarray(rgb_image[chunk_start:chunk_end])
        quantized_chunk = som.quantization(rgb_chunk)
        point_distances = np.linalg.norm(rgb_chunk - quantized_chunk, axis=1)
        distance_sum += float(point_distances.sum(dtype=np.float64))
        completed_points = chunk_end

        if profile:
            elapsed_seconds = max(time.time() - progress_start, 1e-9)
            points_per_second = completed_points / elapsed_seconds
            remaining_seconds = (total_points - completed_points) / max(points_per_second, 1e-9)
            progress_percent = completed_points / total_points * 100
            print(
                f"[SAMPLED] {progress_label}: {completed_points}/{total_points} "
                f"({progress_percent:.1f}%), {points_per_second:,.0f} points/s, "
                f"ETA {remaining_seconds / 60:.1f} min",
                flush=True,
            )

    return distance_sum / total_points


def _som_winners_chunked(
    som: MiniSom,
    rgb_image: np.ndarray,
    *,
    chunk_size: int,
    profile: bool = False,
) -> np.ndarray:
    """按原始行顺序分块计算每个转录本的 SOM winner，保持原始标签定义不变。

    Parameters
    ----------
    som
        已完成训练的 MiniSom 模型。
    rgb_image
        与原流程相同的全量三维 RNAflux RGB 数组。
    chunk_size
        每个分块包含的转录本数量，必须为正整数。
    profile
        是否输出完成数量、百分比、速度和预计剩余时间。

    Returns
    -------
    numpy.ndarray
        与输入行一一对应的整数型 SOM 节点编号。

    Raises
    ------
    ValueError
        当 ``chunk_size`` 非正时抛出。
    """
    if chunk_size <= 0:
        raise ValueError("som_chunk_size must be a positive integer")

    total_points = len(rgb_image)
    som_weights = np.asarray(som.get_weights())
    flattened_weights = som_weights.reshape(-1, som_weights.shape[-1])
    cluster_numbers = np.empty(total_points, dtype=np.int64)
    progress_start = time.time()

    for chunk_start in range(0, total_points, chunk_size):
        chunk_end = min(chunk_start + chunk_size, total_points)
        rgb_chunk = np.asarray(rgb_image[chunk_start:chunk_end])
        distance_matrix = np.linalg.norm(
            rgb_chunk[:, np.newaxis, :] - flattened_weights[np.newaxis, :, :],
            axis=2,
        )
        cluster_numbers[chunk_start:chunk_end] = np.argmin(distance_matrix, axis=1)

        if profile:
            completed_points = chunk_end
            elapsed_seconds = max(time.time() - progress_start, 1e-9)
            points_per_second = completed_points / elapsed_seconds
            remaining_seconds = (total_points - completed_points) / max(points_per_second, 1e-9)
            progress_percent = completed_points / total_points * 100
            print(
                f"[SAMPLED] SOM winner: {completed_points}/{total_points} "
                f"({progress_percent:.1f}%), {points_per_second:,.0f} points/s, "
                f"ETA {remaining_seconds / 60:.1f} min",
                flush=True,
            )

    return cluster_numbers


def _run_som_notebook(
    rgb_image: np.ndarray,
    n_clusters: Union[str, int] = "auto",
    cluster_range_min: int = 4,
    cluster_range_max: int = 7,
    som_iterations: int = 1000,
    som_sigma: float = 1.0,
    som_learning_rate: float = 0.5,
    som_chunk_size: int = 1_000_000,
    profile: bool = False,
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
            quantization_error = _som_quantization_error_chunked(
                som,
                rgb_image,
                chunk_size=som_chunk_size,
                profile=profile,
                progress_label=f"SOM k={k} quantization",
            )
            quantization_errors.append(quantization_error)
        # kl = KneeLocator(k_values, quantization_errors, curve="convex", direction="decreasing")
        # best_k = kl.elbow
        # if best_k is None:
        #     raise RuntimeError("KneeLocator did not find an elbow for n_clusters auto, same as notebook would fail here. Please set --n-clusters manually.")
        # som = som_models[best_k]
        kl = KneeLocator(k_values, quantization_errors, curve="convex", direction="decreasing")
        best_k = kl.elbow

        if best_k is None:
            fallback_k = 5
            fallback_k = max(cluster_range_min, min(cluster_range_max, fallback_k))

            print(
                f"[SAMPLED] KneeLocator did not find an elbow for n_clusters auto. "
                f"Fallback to n_clusters={fallback_k}.",
                flush=True
            )

            best_k = fallback_k

        som = som_models[best_k]
    else:
        best_k = int(n_clusters)
        som = MiniSom(x=1, y=best_k, input_len=3, sigma=som_sigma, learning_rate=som_learning_rate)
        som.train(rgb_image, som_iterations)
        quantization_errors = []

    cluster_numbers = _som_winners_chunked(
        som,
        rgb_image,
        chunk_size=som_chunk_size,
        profile=profile,
    )
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


def _safe_output_value(value: Any) -> str:
    """将分组值转换为安全文件名片段，同时保留原始文本含义。"""
    return str(value).replace("/", "_").replace(" ", "_")


def _subdomain_output_columns(df: pd.DataFrame, batch_col: str) -> List[str]:
    """返回轻量逐点区室结果的稳定字段顺序。"""
    output_columns: List[str] = []
    for column_name in [
        "cell",
        "gene",
        "x",
        "y",
        "z",
        "umi",
        batch_col,
        "batch",
        "tile",
        "fov",
        "sample_id",
        "celltype",
        "annotation",
    ]:
        if column_name in df.columns and column_name not in output_columns:
            output_columns.append(column_name)
    for column_name in ["subdomain", "rnaflux_r", "rnaflux_g", "rnaflux_b"]:
        if column_name in df.columns and column_name not in output_columns:
            output_columns.append(column_name)
    return output_columns


def _write_query_embeddings_chunked(
    query_coords: np.ndarray,
    rnaflux_embeddings: np.ndarray,
    out_csv: Path,
    *,
    chunk_size: int,
) -> None:
    """按原始查询点顺序分块写出高维 RNAflux 向量，避免构建巨型对象列。

    Parameters
    ----------
    query_coords
        查询点二维坐标，行顺序与 ``rnaflux_embeddings`` 完全一致。
    rnaflux_embeddings
        每个查询点的高维 RNAflux 向量。
    out_csv
        输出 CSV 路径。
    chunk_size
        每次写出的查询点数量，必须为正整数。

    Raises
    ------
    ValueError
        当分块大小非正或两个输入行数不一致时抛出。
    """
    if chunk_size <= 0:
        raise ValueError("output_csv_chunk_size must be a positive integer")
    if len(query_coords) != len(rnaflux_embeddings):
        raise ValueError("query_coords and rnaflux_embeddings must have equal row counts")

    wrote_header = False
    for chunk_start in range(0, len(query_coords), chunk_size):
        chunk_end = min(chunk_start + chunk_size, len(query_coords))
        query_chunk = pd.DataFrame(
            {
                "x": query_coords[chunk_start:chunk_end, 0],
                "y": query_coords[chunk_start:chunk_end, 1],
                "rnaflux_embedding": list(rnaflux_embeddings[chunk_start:chunk_end]),
            }
        )
        query_chunk.to_csv(
            out_csv,
            mode="w" if not wrote_header else "a",
            header=not wrote_header,
            index=False,
        )
        wrote_header = True


def _write_display_group_outputs(
    df: pd.DataFrame,
    rgb_image: np.ndarray,
    cluster_numbers: np.ndarray,
    cell_shape: pd.DataFrame,
    nucleus_shape: pd.DataFrame,
    *,
    display_col: str,
    batch_col: str,
    out_prefix: Path,
    figsize,
    dpi: int,
    embedding_point_size: float,
    subdomain_point_size: float,
    output_csv_chunk_size: int,
    profile: bool,
) -> List[Dict[str, Any]]:
    """使用全数据集统一标签按展示字段拆分 CSV 和图，不重新拟合任何模型。

    Parameters
    ----------
    df, rgb_image, cluster_numbers
        全数据集转录本表、全局插值 RGB 和全局 SOM 标签，三者行顺序必须一致。
    cell_shape, nucleus_shape
        全数据集细胞及细胞核边界，仅用于每个展示分组绘图。
    display_col
        输出展示分组字段，例如 ``tile``。
    batch_col
        全局计算分组字段，用于保持输出字段兼容。
    out_prefix
        当前全局计算任务的输出前缀。
    figsize, dpi, embedding_point_size, subdomain_point_size
        与原绘图函数相同的参数。
    output_csv_chunk_size
        每次写入逐点 CSV 的行数。
    profile
        是否输出各展示分组的进度。

    Returns
    -------
    list of dict
        每个展示分组的名称、点数及输出路径。

    Raises
    ------
    ValueError
        当展示字段不存在或三个逐点输入行数不一致时抛出。
    """
    if display_col not in df.columns:
        raise ValueError(f"Display column not found: {display_col}")
    if len(df) != len(rgb_image) or len(df) != len(cluster_numbers):
        raise ValueError("df, rgb_image and cluster_numbers must have equal row counts")
    if output_csv_chunk_size <= 0:
        raise ValueError("output_csv_chunk_size must be a positive integer")

    display_outputs: List[Dict[str, Any]] = []
    display_groups = df.groupby(display_col, sort=False, observed=False).indices
    total_groups = len(display_groups)
    safe_display_col = _safe_output_value(display_col)

    for group_number, (display_value, row_positions) in enumerate(display_groups.items(), start=1):
        row_positions = np.asarray(row_positions, dtype=np.int64)
        group_df = df.iloc[row_positions].copy()
        group_rgb = np.asarray(rgb_image[row_positions])
        group_clusters = np.asarray(cluster_numbers[row_positions], dtype=np.int64)
        group_df["cluster"] = group_clusters
        group_df["subdomain"] = group_clusters
        group_df["rnaflux_r"] = group_rgb[:, 0]
        group_df["rnaflux_g"] = group_rgb[:, 1]
        group_df["rnaflux_b"] = group_rgb[:, 2]

        safe_display_value = _safe_output_value(display_value)
        group_prefix = Path(f"{out_prefix}_{safe_display_col}-{safe_display_value}")
        embedding_png = Path(f"{group_prefix}_sampled_rnaflux_embedding.png")
        subdomains_png = Path(f"{group_prefix}_sampled_subdomains.png")
        subdomain_points_csv = Path(f"{group_prefix}_sampled_subdomain_points.csv")

        output_columns = _subdomain_output_columns(group_df, batch_col)
        group_df[output_columns].to_csv(
            subdomain_points_csv,
            index=False,
            chunksize=output_csv_chunk_size,
        )

        group_cells = set(group_df["cell"].astype(str).unique())
        group_cell_shape = cell_shape[cell_shape["cell"].isin(group_cells)]
        group_nucleus_shape = nucleus_shape[nucleus_shape["cell"].isin(group_cells)]
        _plot_embedding_notebook(
            group_df,
            group_rgb,
            embedding_png,
            figsize=figsize,
            dpi=dpi,
            point_size=embedding_point_size,
        )
        _plot_subdomains_notebook(
            group_df,
            group_cell_shape,
            group_nucleus_shape,
            subdomains_png,
            figsize=figsize,
            dpi=dpi,
            point_size=subdomain_point_size,
        )

        display_outputs.append(
            {
                "display_value": str(display_value),
                "n_transcripts": int(len(group_df)),
                "embedding_png": str(embedding_png),
                "subdomains_png": str(subdomains_png),
                "subdomain_points_csv": str(subdomain_points_csv),
            }
        )
        if profile:
            print(
                f"[SAMPLED] display {display_col}: {group_number}/{total_groups} "
                f"value={display_value}, transcripts={len(group_df)}",
                flush=True,
            )

    return display_outputs


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
    radius: Union[str, float] = "auto",
    n_clusters: Union[str, int] = "auto",
    cluster_range_min: int = 4,
    cluster_range_max: int = 7,
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
    same_cell_neighborhood: bool = True,
    display_col: Optional[str] = None,
    som_chunk_size: int = 1_000_000,
    output_csv_chunk_size: int = 250_000,
) -> Dict[str, Any]:
    start_time = time.time()
    out_prefix = Path(out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    torch_device = _device_from_arg(device)

    # 正式区室定义始终使用同细胞邻域和同细胞插值；参数仅为兼容既有调用保留。
    same_cell_neighborhood = True

    batch_mask = data_df[batch_col].astype(str) == str(batch)
    if bool(batch_mask.all()):
        # 全数据集只有一个计算分组时复用原表，避免复制数亿行数据。
        df = data_df
    else:
        df = data_df.loc[batch_mask].copy()
    if len(df) == 0:
        raise ValueError(f"No transcripts found for batch={batch!r}")

    # 同细胞大数据路径先筛选当前组边界再转WKT，避免为无关细胞创建大量Polygon和字符串。
    cell_list = df["cell"].astype(str).unique().tolist()
    selected_cell_ids = set(cell_list)
    selected_cell_boundary = _filter_boundary_dict(cell_boundary, selected_cell_ids)
    selected_nuclear_boundary = _filter_boundary_dict(nuclear_boundary, selected_cell_ids)
    resolved_radius = _resolve_compartment_radius(radius, selected_cell_boundary, selected_cell_ids)
    cell_shape = _make_shape_df(selected_cell_boundary, "cell_shape")
    nucleus_shape = _make_shape_df(selected_nuclear_boundary, "nucleus_shape")
    cell_shape = cell_shape[cell_shape["cell"].isin(cell_list)]
    nucleus_shape = nucleus_shape[nucleus_shape["cell"].isin(cell_list)]
    radius = resolved_radius

    df["cell"] = df["cell"].astype(str)
    df["gene"] = df["gene"].astype(str)

    # EXACT notebook sampling
    sampled_df = df.sample(frac=frac, random_state=42)
    print(len(sampled_df), flush=True)
    if len(sampled_df) == 0:
        raise ValueError("sampled_df is empty. This follows notebook df.sample(frac=frac); increase --frac.")
    n_sampled = int(len(sampled_df))

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

    same_cell_temp_dir = None
    same_cell_fast_state = None
    if same_cell_neighborhood:
        # 大数据同细胞路径使用本地临时磁盘保存RNAflux和索引，避免数百万Python/Torch对象驻留RAM。
        same_cell_temp_dir = tempfile.TemporaryDirectory(
            prefix="compartment_same_cell_",
            dir=os.environ.get("TMPDIR"),
        )
        query_coords, cell_ids, rnaflux_embeddings, same_cell_fast_state = compute_same_cell_rnaflux_fast(
            df,
            sampled_df,
            unique_genes,
            radius,
            torch_device,
            Path(same_cell_temp_dir.name),
            profile=profile,
        )
        # 保持后续统一释放语句和输出结构不变；快速路径不再创建这三个巨型中间对象。
        expression_vectors = None
        cell_vectors = None
        std_dev = None
    else:
        query_coords, expression_vectors, cell_ids = _compute_local_expression_vectors_notebook(
            df,
            sampled_df,
            unique_genes,
            radius,
            torch_device,
            profile=profile,
            same_cell_neighborhood=False,
        )
        cell_vectors, std_dev = _compute_cell_vectors_notebook(df, unique_genes, torch_device)
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
    # 查询点坐标和RNAflux已经生成，后续不再需要完整 sampled_df。
    del sampled_df

    query_coords_np = np.array(query_coords)

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
        rgb_image = interpolate_same_cell_rgb_fast(
            df=df,
            query_coords=query_coords,
            normalized_embeddings=normalized_embeddings,
            state=same_cell_fast_state,
            output_path=Path(same_cell_temp_dir.name) / "same_cell_rgb.npy",
            profile=profile,
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
        # RGB 已合并完成，三个独立全量通道和坐标矩阵可立即释放。
        del r_interp, g_interp, b_interp, grid_x, grid_y
        del r_values, g_values, b_values

    embedding_png = Path(f"{out_prefix}_sampled_rnaflux_embedding.png")
    subdomains_png = Path(f"{out_prefix}_sampled_subdomains.png")
    rnaembedding_csv = Path(f"{out_prefix}_sampled_rnaembedding.csv")
    allpoints_csv = Path(f"{out_prefix}_sampled_df.csv")
    # Lightweight point-level compartment table for downstream visualization.
    # This is exported by default and is much smaller/easier to consume than sampled_df.csv.
    subdomain_points_csv = Path(f"{out_prefix}_sampled_subdomain_points.csv")
    query_csv = Path(f"{out_prefix}_sampled_query_embeddings.csv")
    meta_json = Path(f"{out_prefix}_sampled_meta.json")

    # 分块写出与原流程相同的查询点嵌入，避免保留巨型 pandas 对象列。
    _write_query_embeddings_chunked(
        query_coords_np,
        rnaflux_embeddings,
        query_csv,
        chunk_size=output_csv_chunk_size,
    )
    # 查询嵌入和全量 RGB 已生成，释放高维 RNAflux 与查询阶段中间量。
    del expression_vectors, cell_vectors, std_dev, rnaflux_embeddings, pca_embeddings
    del normalized_embeddings, query_coords, query_coords_np, cell_ids

    # 未指定展示字段时保持原来的全组绘图；指定后等待全局标签完成再按 tile/FOV 绘图。
    if display_col is None:
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
        df["rgb_image"] = list(rgb_image)
        pd.DataFrame(df).to_csv(rnaembedding_csv)

    cluster_numbers, best_k, quantization_errors = _run_som_notebook(
        rgb_image,
        n_clusters=n_clusters,
        cluster_range_min=cluster_range_min,
        cluster_range_max=cluster_range_max,
        som_iterations=som_iterations,
        som_sigma=som_sigma,
        som_learning_rate=som_learning_rate,
        som_chunk_size=som_chunk_size,
        profile=profile,
    )
    print("The best cluster number is: ", best_k, flush=True)
    display_outputs: List[Dict[str, Any]] = []
    if display_col is not None:
        display_outputs = _write_display_group_outputs(
            df,
            rgb_image,
            cluster_numbers,
            cell_shape,
            nucleus_shape,
            display_col=display_col,
            batch_col=batch_col,
            out_prefix=out_prefix,
            figsize=figsize,
            dpi=dpi,
            embedding_point_size=embedding_point_size,
            subdomain_point_size=subdomain_point_size,
            output_csv_chunk_size=output_csv_chunk_size,
            profile=profile,
        )
    else:
        df["cluster"] = cluster_numbers
        df["subdomain"] = df["cluster"].astype(int)
        rgb_arr = np.asarray(rgb_image)
        df["rnaflux_r"] = rgb_arr[:, 0]
        df["rnaflux_g"] = rgb_arr[:, 1]
        df["rnaflux_b"] = rgb_arr[:, 2]
        subdomain_cols = _subdomain_output_columns(df, batch_col)
        df[subdomain_cols].to_csv(
            subdomain_points_csv,
            index=False,
            chunksize=output_csv_chunk_size,
        )
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
        if display_col is not None:
            df["cluster"] = cluster_numbers
            df["subdomain"] = df["cluster"].astype(int)
            rgb_arr = np.asarray(rgb_image)
            df["rnaflux_r"] = rgb_arr[:, 0]
            df["rnaflux_g"] = rgb_arr[:, 1]
            df["rnaflux_b"] = rgb_arr[:, 2]
        pd.DataFrame(df).to_csv(allpoints_csv)

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
        "display_col": display_col,
        "som_chunk_size": int(som_chunk_size),
        "output_csv_chunk_size": int(output_csv_chunk_size),
        "som_iterations": int(som_iterations),
        "som_sigma": float(som_sigma),
        "som_learning_rate": float(som_learning_rate),
        "n_transcripts": int(len(df)),
        "n_sampled": n_sampled,
        "n_cells": int(df["cell"].nunique()),
        "n_genes": int(len(unique_genes)),
        "run_time_min": float(run_time),
        "outputs": {
            "embedding_png": str(embedding_png) if display_col is None else None,
            "subdomains_png": str(subdomains_png) if display_col is None else None,
            "subdomain_points_csv": str(subdomain_points_csv) if display_col is None else None,
            "query_embeddings_csv": str(query_csv),
            "rnaembedding_csv": str(rnaembedding_csv) if export_csv else None,
            "allpoints_csv": str(allpoints_csv) if export_csv else None,
            "display_outputs": display_outputs,
            "meta_json": str(meta_json),
        },
    }
    with open(meta_json, "w") as f:
        json.dump(meta, f, indent=2)
    # 所有最终文件已经写出后删除本地磁盘映射；输出目录结构与原模块保持不变。
    if same_cell_temp_dir is not None:
        del rgb_image
        same_cell_temp_dir.cleanup()
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
    radius: Union[str, float] = "auto",
    n_clusters: Union[str, int] = "auto",
    cluster_range_min: int = 4,
    cluster_range_max: int = 7,
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
    same_cell_neighborhood: bool = True,
    display_col: Optional[str] = None,
    som_chunk_size: int = 1_000_000,
    output_csv_chunk_size: int = 250_000,
) -> Dict[str, Any]:
    pkl_path = Path(pkl_path)
    out_prefix = Path(out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    # 所有正式入口均固定使用同细胞邻域；保留参数仅避免旧API调用报错。
    same_cell_neighborhood = True

    with open(pkl_path, "rb") as f:
        obj = pickle.load(f)

    # 当前任务只读入一个统一数据包，直接复用内部 DataFrame，避免加载后立即整表复制。
    data_df = obj["data_df"]
    cell_boundary = obj.get("cell_boundary", {})
    nuclear_boundary = obj.get("nuclear_boundary", {})
    coordinates = obj.get("coordinates", None)
    del obj

    data_df["cell"] = data_df["cell"].astype(str)
    data_df["gene"] = data_df["gene"].astype(str)
    data_df = _attach_batch_from_coordinates(data_df, coordinates, batch_col=batch_col)
    if display_col is not None and display_col != batch_col:
        data_df = _attach_batch_from_coordinates(data_df, coordinates, batch_col=display_col)

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
        print(f"[SAMPLED] display_col  : {display_col}", flush=True)
        print(f"[SAMPLED] som_chunk    : {som_chunk_size}", flush=True)

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
                display_col=display_col,
                som_chunk_size=som_chunk_size,
                output_csv_chunk_size=output_csv_chunk_size,
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
