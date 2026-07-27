from __future__ import annotations

import gc
import json
import pickle
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
from shapely.wkt import loads

from tools.engines.compartment_adapter import (
    _make_shape_df,
    _plot_embedding_notebook,
    _resolve_figsize_and_limits,
    _safe_output_value,
    _subdomain_output_columns,
)


REQUIRED_RESULT_COLUMNS = {
    "cell",
    "x",
    "y",
    "subdomain",
    "rnaflux_r",
    "rnaflux_g",
    "rnaflux_b",
}


def _load_redisplay_inputs(
    pkl_path: Path,
    target_display_col: str,
) -> tuple[pd.DataFrame, Dict[Any, pd.DataFrame], Dict[Any, pd.DataFrame]]:
    """读取重新展示所需的细胞分组和边界，不重新使用转录本表达数据。

    Parameters
    ----------
    pkl_path
        原区室探测使用的统一数据字典PKL。
    target_display_col
        新展示分组字段，例如``fov``。

    Returns
    -------
    tuple
        唯一的``cell → target_display_col``映射、细胞边界字典和细胞核边界字典。

    Raises
    ------
    ValueError
        当PKL结构错误、展示字段缺失、细胞分组缺失或同一细胞对应多个展示值时抛出。
    """
    with pkl_path.open("rb") as handle:
        data_bundle = pickle.load(handle)
    if not isinstance(data_bundle, dict):
        raise ValueError("Compartment PKL must contain a dictionary")

    coordinates = data_bundle.get("coordinates")
    if not isinstance(coordinates, pd.DataFrame):
        raise ValueError("Compartment PKL does not contain a coordinates DataFrame")
    if "cell" not in coordinates.columns or target_display_col not in coordinates.columns:
        raise ValueError(
            f"coordinates must contain 'cell' and '{target_display_col}' columns"
        )

    # 只保留重新拆分必需的细胞映射，立即释放超大转录本表和表达矩阵引用。
    cell_display_map = coordinates[["cell", target_display_col]].copy()
    cell_display_map["cell"] = cell_display_map["cell"].astype(str)
    if cell_display_map[target_display_col].isna().any():
        raise ValueError(f"coordinates contains missing values in '{target_display_col}'")
    conflicting_counts = cell_display_map.groupby("cell", sort=False)[target_display_col].nunique()
    conflicting_cells = conflicting_counts[conflicting_counts > 1]
    if not conflicting_cells.empty:
        raise ValueError(
            f"{len(conflicting_cells)} cells map to multiple '{target_display_col}' values"
        )
    cell_display_map = cell_display_map.drop_duplicates("cell", keep="first")

    cell_boundary = data_bundle.get("cell_boundary") or {}
    nuclear_boundary = data_bundle.get("nuclear_boundary") or {}
    del coordinates, data_bundle
    gc.collect()
    return cell_display_map, cell_boundary, nuclear_boundary


def _source_display_csvs(source_meta_path: Path) -> List[Path]:
    """从已有批次元数据中读取全部展示CSV路径并验证文件存在。

    Parameters
    ----------
    source_meta_path
        已完成区室结果的``*_sampled_meta.json``。

    Returns
    -------
    list of Path
        按原元数据顺序排列的逐点结果CSV。

    Raises
    ------
    ValueError
        当元数据没有展示输出或记录缺少CSV路径时抛出。
    FileNotFoundError
        当任一源CSV不存在时抛出。
    """
    with source_meta_path.open() as handle:
        source_meta = json.load(handle)
    display_outputs = source_meta.get("display_outputs")
    if display_outputs is None:
        display_outputs = source_meta.get("outputs", {}).get("display_outputs")
    if not display_outputs:
        raise ValueError("Source metadata does not contain display_outputs")

    source_csvs: List[Path] = []
    for display_output in display_outputs:
        csv_value = display_output.get("subdomain_points_csv")
        if not csv_value:
            raise ValueError("A source display output is missing subdomain_points_csv")
        csv_path = Path(csv_value)
        if not csv_path.exists():
            fallback_path = source_meta_path.parent / csv_path.name
            csv_path = fallback_path if fallback_path.exists() else csv_path
        if not csv_path.exists():
            raise FileNotFoundError(f"Source compartment CSV not found: {csv_path}")
        source_csvs.append(csv_path)
    return source_csvs


def _plot_subdomains_with_global_palette(
    point_table: pd.DataFrame,
    cell_shape: pd.DataFrame,
    nucleus_shape: pd.DataFrame,
    out_png: Path,
    global_cluster_values: Sequence[int],
    *,
    figsize: Any,
    dpi: int,
    point_size: float,
) -> None:
    """使用全局统一标签色表绘制一个展示分组，避免不同FOV颜色含义变化。

    Parameters
    ----------
    point_table
        当前FOV逐点结果，必须包含``x``、``y``和``subdomain``。
    cell_shape, nucleus_shape
        当前FOV涉及细胞的细胞及细胞核WKT边界表。
    out_png
        输出PNG路径。
    global_cluster_values
        整个数据集统一的区室编号列表。
    figsize, dpi, point_size
        与正式区室绘图一致的画布、分辨率和点大小参数。

    Raises
    ------
    ValueError
        当全局标签为空或不是从0开始的连续整数时抛出。
    """
    cluster_values = [int(value) for value in global_cluster_values]
    expected_values = list(range(len(cluster_values)))
    if cluster_values != expected_values:
        raise ValueError("Global subdomain labels must be consecutive integers starting at 0")

    resolved_figsize, x_limits, y_limits = _resolve_figsize_and_limits(point_table, figsize)
    color_map = ListedColormap(plt.cm.viridis(np.linspace(0, 1, len(cluster_values))))
    figure, axis = plt.subplots(figsize=resolved_figsize)
    scatter = axis.scatter(
        point_table["x"],
        point_table["y"],
        c=point_table["subdomain"],
        cmap=color_map,
        vmin=-0.5,
        vmax=len(cluster_values) - 0.5,
        s=point_size,
        alpha=0.7,
        linewidths=0,
    )
    color_bar = plt.colorbar(scatter, ax=axis, label="Cluster")
    color_bar.set_ticks(cluster_values)
    color_bar.set_ticklabels(cluster_values)
    color_bar.ax.invert_yaxis()

    axis.set_xlabel("X Coordinate")
    axis.set_ylabel("Y Coordinate")
    axis.set_title("SOM-based Clustering of RNAflux Embeddings")
    for _, boundary_row in cell_shape.iterrows():
        polygon = loads(boundary_row["cell_shape"])
        if not polygon.is_empty:
            boundary_x, boundary_y = polygon.exterior.xy
            axis.plot(boundary_x, boundary_y, linestyle="-", color="black", linewidth=0.8)
    for _, boundary_row in nucleus_shape.iterrows():
        polygon = loads(boundary_row["nucleus_shape"])
        if not polygon.is_empty:
            boundary_x, boundary_y = polygon.exterior.xy
            axis.plot(boundary_x, boundary_y, linestyle="-", color="red", linewidth=0.7)

    if x_limits is not None:
        axis.set_xlim(x_limits)
    if y_limits is not None:
        axis.set_ylim(y_limits)
    axis.set_aspect("equal", adjustable="box")
    plt.tight_layout()
    plt.savefig(out_png, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def redisplay_existing_compartment(
    *,
    pkl_path: Path,
    source_meta_path: Path,
    out_prefix: Path,
    target_display_col: str = "fov",
    batch_col: str = "batch",
    read_chunk_size: int = 250_000,
    figsize: Any = "auto",
    dpi: int = 300,
    embedding_point_size: float = 10,
    subdomain_point_size: float = 1,
    only_target_values: Optional[Iterable[str]] = None,
    write_plots: bool = True,
    profile: bool = False,
) -> Dict[str, Any]:
    """保留已有逐点区室结果，仅按新的展示字段重新拆分CSV并绘图。

    该函数不会计算RNAflux、TruncatedSVD或MiniSom，因此每个点的``subdomain``和
    RGB值严格来自已有结果。源CSV按块读取，避免把全数据集载入内存。

    Parameters
    ----------
    pkl_path
        原始统一数据字典PKL，用于读取``cell → fov``和边界。
    source_meta_path
        已完成区室结果的批次级``*_sampled_meta.json``。
    out_prefix
        新输出前缀，例如``Compartment/Dataset33_1_fov/Dataset33_1_batch-rep1``。
    target_display_col
        新展示字段，当前需求为``fov``。
    batch_col
        输出中保留的全局计算分组字段。
    read_chunk_size
        每次读取源CSV的行数，必须为正整数。
    figsize, dpi, embedding_point_size, subdomain_point_size
        与正式区室模块一致的绘图参数。
    only_target_values
        可选的展示值白名单，用于小型测试；完整运行时保持``None``。
    write_plots
        是否生成两类PNG；测试可关闭，正式运行应保持``True``。
    profile
        是否打印拆分和绘图进度。

    Returns
    -------
    dict
        输出元数据，包括源结果、全局标签、总点数和每个FOV文件路径。

    Raises
    ------
    ValueError
        当分块参数无效、源字段缺失、细胞无法映射或标签不合法时抛出。
    """
    if read_chunk_size <= 0:
        raise ValueError("read_chunk_size must be a positive integer")
    selected_values: Optional[Set[str]] = None
    if only_target_values is not None:
        selected_values = {str(value) for value in only_target_values}
        if not selected_values:
            raise ValueError("only_target_values must not be empty")

    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    source_csvs = _source_display_csvs(source_meta_path)
    cell_display_map, cell_boundary, nuclear_boundary = _load_redisplay_inputs(
        pkl_path,
        target_display_col,
    )
    display_lookup = cell_display_map.set_index("cell")[target_display_col]

    # 第一次遇到目标文件时使用写模式，之后追加；这样可安全覆盖同名旧错误结果。
    initialized_outputs: Set[Path] = set()
    output_row_counts: Dict[str, int] = {}
    global_cluster_values: Set[int] = set()
    total_source_rows = 0
    total_written_rows = 0
    for source_number, source_csv in enumerate(source_csvs, start=1):
        for source_chunk in pd.read_csv(source_csv, chunksize=read_chunk_size):
            missing_columns = REQUIRED_RESULT_COLUMNS.difference(source_chunk.columns)
            if missing_columns:
                raise ValueError(
                    f"Source CSV {source_csv} is missing columns: {sorted(missing_columns)}"
                )
            source_chunk["cell"] = source_chunk["cell"].astype(str)
            mapped_values = source_chunk["cell"].map(display_lookup)
            if mapped_values.isna().any():
                missing_cell_count = int(source_chunk.loc[mapped_values.isna(), "cell"].nunique())
                raise ValueError(
                    f"{missing_cell_count} cells from {source_csv} have no '{target_display_col}' mapping"
                )
            source_chunk[target_display_col] = mapped_values
            source_chunk["subdomain"] = pd.to_numeric(
                source_chunk["subdomain"], errors="raise"
            ).astype(np.int64)
            global_cluster_values.update(source_chunk["subdomain"].unique().tolist())
            total_source_rows += len(source_chunk)

            if selected_values is not None:
                selected_mask = source_chunk[target_display_col].astype(str).isin(selected_values)
                source_chunk = source_chunk.loc[selected_mask]
            if source_chunk.empty:
                continue

            for display_value, display_chunk in source_chunk.groupby(
                target_display_col,
                sort=False,
                observed=False,
            ):
                display_text = str(display_value)
                safe_display_value = _safe_output_value(display_value)
                display_prefix = Path(
                    f"{out_prefix}_{_safe_output_value(target_display_col)}-{safe_display_value}"
                )
                output_csv = Path(f"{display_prefix}_sampled_subdomain_points.csv")
                output_columns = _subdomain_output_columns(display_chunk, batch_col)
                display_chunk[output_columns].to_csv(
                    output_csv,
                    mode="a" if output_csv in initialized_outputs else "w",
                    header=output_csv not in initialized_outputs,
                    index=False,
                )
                initialized_outputs.add(output_csv)
                output_row_counts[display_text] = output_row_counts.get(display_text, 0) + len(display_chunk)
                total_written_rows += len(display_chunk)
        if profile:
            print(
                f"[REDISPLAY] source CSV: {source_number}/{len(source_csvs)} "
                f"rows={total_source_rows:,}",
                flush=True,
            )

    sorted_cluster_values = sorted(global_cluster_values)
    expected_cluster_values = list(range(len(sorted_cluster_values)))
    if sorted_cluster_values != expected_cluster_values:
        raise ValueError(
            "Existing subdomain labels must be consecutive integers starting at 0"
        )

    cell_shape = _make_shape_df(cell_boundary, "cell_shape")
    nucleus_shape = _make_shape_df(nuclear_boundary, "nucleus_shape")
    display_outputs: List[Dict[str, Any]] = []
    sorted_display_values = sorted(
        output_row_counts,
        key=lambda value: (not value.isdigit(), int(value) if value.isdigit() else value),
    )
    for display_number, display_text in enumerate(sorted_display_values, start=1):
        safe_display_value = _safe_output_value(display_text)
        display_prefix = Path(
            f"{out_prefix}_{_safe_output_value(target_display_col)}-{safe_display_value}"
        )
        output_csv = Path(f"{display_prefix}_sampled_subdomain_points.csv")
        embedding_png = Path(f"{display_prefix}_sampled_rnaflux_embedding.png")
        subdomains_png = Path(f"{display_prefix}_sampled_subdomains.png")

        if write_plots:
            display_table = pd.read_csv(output_csv)
            display_rgb = display_table[["rnaflux_r", "rnaflux_g", "rnaflux_b"]].to_numpy()
            display_cells = set(display_table["cell"].astype(str).unique())
            display_cell_shape = cell_shape[cell_shape["cell"].isin(display_cells)]
            display_nucleus_shape = nucleus_shape[nucleus_shape["cell"].isin(display_cells)]
            _plot_embedding_notebook(
                display_table,
                display_rgb,
                embedding_png,
                figsize=figsize,
                dpi=dpi,
                point_size=embedding_point_size,
            )
            _plot_subdomains_with_global_palette(
                display_table,
                display_cell_shape,
                display_nucleus_shape,
                subdomains_png,
                sorted_cluster_values,
                figsize=figsize,
                dpi=dpi,
                point_size=subdomain_point_size,
            )
            del display_table, display_rgb

        display_outputs.append(
            {
                "display_value": display_text,
                "n_transcripts": int(output_row_counts[display_text]),
                "embedding_png": str(embedding_png) if write_plots else None,
                "subdomains_png": str(subdomains_png) if write_plots else None,
                "subdomain_points_csv": str(output_csv),
            }
        )
        if profile:
            print(
                f"[REDISPLAY] {target_display_col}: {display_number}/{len(sorted_display_values)} "
                f"value={display_text}, transcripts={output_row_counts[display_text]:,}",
                flush=True,
            )

    metadata = {
        "status": "done",
        "backend": "existing_compartment_redisplay",
        "source_meta_json": str(source_meta_path),
        "target_display_col": target_display_col,
        "global_subdomains": sorted_cluster_values,
        "best_k": len(sorted_cluster_values),
        "n_source_csvs": len(source_csvs),
        "n_source_transcripts": int(total_source_rows),
        "n_output_transcripts": int(total_written_rows),
        "selected_display_values": sorted(selected_values) if selected_values is not None else None,
        "display_outputs": display_outputs,
    }
    metadata_path = Path(f"{out_prefix}_sampled_meta.json")
    with metadata_path.open("w") as handle:
        json.dump(metadata, handle, indent=2)
    metadata["meta_json"] = str(metadata_path)
    return metadata
