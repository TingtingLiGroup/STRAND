"""为优化版区室探测生成与旧模块一致的逐 tile PNG 图像。"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from shapely.geometry import Polygon


DEFAULT_FIGURE_SIZE = "auto"
DEFAULT_DPI = 300
DEFAULT_EMBEDDING_POINT_SIZE = 10.0
DEFAULT_SUBDOMAIN_POINT_SIZE = 1.0


def _resolve_figure_geometry(
    x_values: np.ndarray,
    y_values: np.ndarray,
    figure_size: str | Tuple[float, float] | None,
    *,
    base_size: float = 10.0,
    minimum_size: float = 5.0,
    maximum_size: float = 24.0,
    padding_ratio: float = 0.03,
) -> Tuple[Tuple[float, float], Optional[Tuple[float, float]], Optional[Tuple[float, float]]]:
    """按旧模块规则计算画布尺寸和坐标范围。

    Parameters
    ----------
    x_values, y_values
        当前 tile 全部转录本的二维坐标。
    figure_size
        ``"auto"`` 或 ``None`` 表示按空间长宽比自动计算；也可直接传入宽高元组。
    base_size, minimum_size, maximum_size, padding_ratio
        旧模块自动画布算法使用的基础尺寸、尺寸上下限和边距比例。

    Returns
    -------
    tuple
        ``(figsize, xlim, ylim)``；固定画布时坐标范围返回 ``None``。

    Raises
    ------
    ValueError
        当前 tile 没有可绘制点时抛出。
    """

    # 图像必须至少包含一个点，否则 NumPy 无法计算坐标范围。
    if len(x_values) == 0:
        raise ValueError("当前 tile 没有可绘制的转录本。")
    if figure_size is not None and str(figure_size).lower() != DEFAULT_FIGURE_SIZE:
        return tuple(figure_size), None, None

    # 直接从磁盘映射字段计算范围，避免为了绘图构造完整 DataFrame。
    x_min = float(np.nanmin(x_values))
    x_max = float(np.nanmax(x_values))
    y_min = float(np.nanmin(y_values))
    y_max = float(np.nanmax(y_values))
    x_range = x_max - x_min
    y_range = y_max - y_min
    if not np.isfinite(x_range) or not np.isfinite(y_range) or x_range <= 0 or y_range <= 0:
        return (base_size, base_size), None, None

    # 保持旧模块的空间长宽比和最大画布限制。
    aspect_ratio = x_range / y_range
    if aspect_ratio >= 1:
        width = base_size * aspect_ratio
        height = base_size
    else:
        width = base_size
        height = base_size / aspect_ratio
    width = min(max(width, minimum_size), maximum_size)
    height = min(max(height, minimum_size), maximum_size)
    x_padding = x_range * padding_ratio
    y_padding = y_range * padding_ratio
    return (width, height), (x_min - x_padding, x_max + x_padding), (y_min - y_padding, y_max + y_padding)


def _plot_boundaries(
    axis,
    cell_ids: Iterable[str],
    boundary_by_cell: Mapping[object, pd.DataFrame],
    *,
    color: str,
    line_width: float,
) -> None:
    """在当前坐标轴上绘制指定细胞的边界。

    Parameters
    ----------
    axis
        Matplotlib 坐标轴对象。
    cell_ids
        当前 tile 实际包含的细胞编号。
    boundary_by_cell
        已将细胞编号统一为字符串键、值为含 ``x/y`` 列 DataFrame 的边界字典。
    color, line_width
        与旧模块一致的边界颜色和线宽。

    Returns
    -------
    None
        边界直接绘制到传入坐标轴。
    """

    for cell_id in cell_ids:
        boundary = boundary_by_cell.get(str(cell_id))
        if boundary is None or not {"x", "y"}.issubset(boundary.columns):
            continue
        try:
            # 使用与旧模块相同的 Polygon 外环，保证边界闭合和异常边界处理一致。
            coordinates = list(zip(boundary["x"], boundary["y"]))
            polygon = Polygon(coordinates) if len(coordinates) >= 3 else Polygon()
            if polygon.is_empty:
                continue
            boundary_x, boundary_y = polygon.exterior.xy
            axis.plot(boundary_x, boundary_y, linestyle="-", color=color, linewidth=line_width)
        except Exception:
            # 单个损坏边界不应阻断整个 tile 的结果图生成，这与旧模块的空 Polygon 回退一致。
            continue


def plot_tile_outputs(
    tile_points_path: str | Path,
    output_prefix: str | Path,
    *,
    cell_ids: Iterable[str] = (),
    cell_boundary: Optional[Dict[object, pd.DataFrame]] = None,
    nuclear_boundary: Optional[Dict[object, pd.DataFrame]] = None,
    figure_size: str | Tuple[float, float] | None = DEFAULT_FIGURE_SIZE,
    dpi: int = DEFAULT_DPI,
    embedding_point_size: float = DEFAULT_EMBEDDING_POINT_SIZE,
    subdomain_point_size: float = DEFAULT_SUBDOMAIN_POINT_SIZE,
) -> Dict[str, str]:
    """从 tile 磁盘暂存数组生成旧版同名 RNAflux 和 subdomain PNG。

    Parameters
    ----------
    tile_points_path
        ``write_tile_coordinate_outputs`` 生成的结构化 ``.npy`` 文件。
    output_prefix
        不含后缀的输出前缀，例如 ``results/tile_0_0``。
    cell_ids
        当前 tile 包含的细胞编号，仅用于筛选边界。
    cell_boundary, nuclear_boundary
        PKL 中的细胞边界和细胞核边界字典；缺失时仍正常输出散点图。
    figure_size, dpi, embedding_point_size, subdomain_point_size
        与旧区室模块相同的绘图参数，默认分别为自动画布、300 DPI、10 和 1。

    Returns
    -------
    dict
        两张实际生成 PNG 的路径。

    Raises
    ------
    ValueError
        暂存数组为空时抛出。
    """

    # 使用 mmap 读取当前 tile，不读取大型 CSV，也不复制其他 tile 的数据。
    tile_points = np.load(Path(tile_points_path), mmap_mode="r")
    if len(tile_points) == 0:
        raise ValueError("不能为没有转录本的 tile 生成图像。")
    x_values = tile_points["x"]
    y_values = tile_points["y"]
    figure_dimensions, x_limits, y_limits = _resolve_figure_geometry(x_values, y_values, figure_size)
    output_prefix = Path(output_prefix)
    embedding_path = Path(f"{output_prefix}_sampled_rnaflux_embedding.png")
    subdomain_path = Path(f"{output_prefix}_sampled_subdomains.png")

    # 第一张图复刻旧模块的 RNAflux RGB 空间散点图。
    rgb_values = np.column_stack((tile_points["rnaflux_r"], tile_points["rnaflux_g"], tile_points["rnaflux_b"]))
    figure, axis = plt.subplots(figsize=figure_dimensions)
    axis.scatter(x_values, y_values, c=rgb_values, s=embedding_point_size, marker="o", linewidths=0)
    if x_limits is not None:
        axis.set_xlim(x_limits)
    if y_limits is not None:
        axis.set_ylim(y_limits)
    axis.set_aspect("equal", adjustable="box")
    axis.set_title("RNAflux Embeddings Visualization")
    axis.set_xlabel("X Coordinate")
    axis.set_ylabel("Y Coordinate")
    figure.tight_layout()
    figure.savefig(embedding_path, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    del rgb_values

    # 第二张图复刻旧模块的 SOM subdomain 配色、色条和边界叠加。
    subdomain_values = tile_points["subdomain"]
    cluster_values = sorted(np.unique(subdomain_values))
    color_map = ListedColormap(plt.cm.viridis(np.linspace(0, 1, len(cluster_values))))
    figure, axis = plt.subplots(figsize=figure_dimensions)
    scatter = axis.scatter(x_values, y_values, c=subdomain_values, cmap=color_map, s=subdomain_point_size, alpha=0.7, linewidths=0)
    color_bar = figure.colorbar(scatter, ax=axis, label="Cluster")
    color_bar.set_ticks(np.arange(len(cluster_values)))
    color_bar.set_ticklabels(np.arange(len(cluster_values)))
    color_bar.ax.invert_yaxis()
    axis.set_xlabel("X Coordinate")
    axis.set_ylabel("Y Coordinate")
    axis.set_title("SOM-based Clustering of RNAflux Embeddings")
    _plot_boundaries(axis, cell_ids, cell_boundary or {}, color="black", line_width=0.8)
    _plot_boundaries(axis, cell_ids, nuclear_boundary or {}, color="red", line_width=0.7)
    if x_limits is not None:
        axis.set_xlim(x_limits)
    if y_limits is not None:
        axis.set_ylim(y_limits)
    axis.set_aspect("equal", adjustable="box")
    figure.tight_layout()
    figure.savefig(subdomain_path, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return {"embedding_png": str(embedding_path), "subdomains_png": str(subdomain_path)}
