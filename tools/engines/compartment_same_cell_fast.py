"""原区室模块的同细胞低内存快速计算助手。

本文件只替换 ``--same-cell-neighborhood`` 下最慢的数据访问方式，不改变查询点抽样、
RNAflux公式、基因顺序、SciPy ``griddata`` 插值或后续SVD/SOM调用。
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from numba import njit
from scipy.interpolate import griddata
from scipy.spatial import cKDTree


DEFAULT_SAME_CELL_WORKERS = max(1, min(16, os.cpu_count() or 1))
DEFAULT_CELL_BATCH_SIZE = 256


@dataclass
class FastSameCellState:
    """保存RNAflux和同细胞插值共同使用的磁盘索引。

    Parameters
    ----------
    row_order_path
        按细胞稳定排列的原始DataFrame行号文件。
    cell_offsets_path
        每个细胞在行号文件中的起止位置。
    query_cell_codes
        每个查询点对应的细胞编码，顺序与 ``sampled_df`` 完全一致。
    query_order, query_starts, query_stops
        按细胞定位查询点的稳定查找表。
    num_cells
        当前计算组中的细胞数。
    """

    row_order_path: Path
    cell_offsets_path: Path
    query_cell_codes: np.ndarray
    query_order: np.ndarray
    query_starts: np.ndarray
    query_stops: np.ndarray
    num_cells: int


@njit(cache=True)
def _fill_stable_row_order(cell_codes: np.ndarray, cell_offsets: np.ndarray, row_order: np.ndarray) -> None:
    """按原始行顺序稳定填充每个细胞的行号，避免全量 ``argsort`` 临时内存。"""

    write_positions = cell_offsets[:-1].copy()
    for row_index in range(len(cell_codes)):
        cell_code = cell_codes[row_index]
        target_position = write_positions[cell_code]
        row_order[target_position] = row_index
        write_positions[cell_code] += 1


def _build_stable_cell_index(cell_codes: np.ndarray, work_dir: Path) -> Tuple[Path, Path]:
    """建立稳定的磁盘型cell行号索引。

    Parameters
    ----------
    cell_codes
        每条转录本的连续非负细胞编码。
    work_dir
        当前运行的本地临时目录。

    Returns
    -------
    tuple
        ``(row_order_path, cell_offsets_path)``。

    Raises
    ------
    ValueError
        行数超过uint32范围或存在缺失细胞编码时抛出。
    """

    if len(cell_codes) > np.iinfo(np.uint32).max:
        raise ValueError("同细胞快速路径最多支持4,294,967,295条转录本。")
    if np.any(cell_codes < 0):
        raise ValueError("cell列包含缺失值，无法建立同细胞索引。")
    num_cells = int(cell_codes.max()) + 1
    cell_counts = np.bincount(cell_codes, minlength=num_cells).astype(np.int64, copy=False)
    cell_offsets = np.empty(num_cells + 1, dtype=np.int64)
    cell_offsets[0] = 0
    np.cumsum(cell_counts, out=cell_offsets[1:])
    row_order_path = work_dir / "same_cell_row_order.npy"
    cell_offsets_path = work_dir / "same_cell_offsets.npy"
    row_order = np.lib.format.open_memmap(row_order_path, mode="w+", dtype=np.uint32, shape=(len(cell_codes),))
    _fill_stable_row_order(cell_codes, cell_offsets, row_order)
    row_order.flush()
    np.save(cell_offsets_path, cell_offsets)
    return row_order_path, cell_offsets_path


def _build_query_lookup(query_cell_codes: np.ndarray, num_cells: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """按细胞稳定组织查询点，并保留每个细胞内部的原采样顺序。"""

    query_order = np.argsort(query_cell_codes, kind="stable")
    sorted_codes = query_cell_codes[query_order]
    query_starts = np.full(num_cells, -1, dtype=np.int64)
    query_stops = np.full(num_cells, -1, dtype=np.int64)
    if len(sorted_codes) == 0:
        return query_order, query_starts, query_stops
    change_positions = np.flatnonzero(sorted_codes[1:] != sorted_codes[:-1]) + 1
    starts = np.concatenate(([0], change_positions))
    stops = np.concatenate((change_positions, [len(sorted_codes)]))
    codes = sorted_codes[starts]
    query_starts[codes] = starts
    query_stops[codes] = stops
    return query_order, query_starts, query_stops


def _resolve_sample_positions(df: pd.DataFrame, sampled_df: pd.DataFrame) -> np.ndarray:
    """把原Notebook抽样结果映射回DataFrame位置，不改变抽样顺序。"""

    if not df.index.is_unique:
        raise ValueError("同细胞快速路径要求DataFrame索引唯一。")
    sample_positions = df.index.get_indexer(sampled_df.index)
    if np.any(sample_positions < 0):
        raise ValueError("抽样行无法映射回原DataFrame。")
    return sample_positions.astype(np.int64, copy=False)


def _build_cell_profiles(
    gene_codes: np.ndarray,
    state: FastSameCellState,
    num_genes: int,
    output_path: Path,
    *,
    profile: bool,
    workers: int,
) -> np.memmap:
    """按原groupby细胞顺序构建cell×gene比例磁盘矩阵。"""

    cell_profiles = np.lib.format.open_memmap(output_path, mode="w+", dtype=np.float64, shape=(state.num_cells, num_genes))
    row_order = np.load(state.row_order_path, mmap_mode="r")
    cell_offsets = np.load(state.cell_offsets_path, mmap_mode="r")

    def process_batch(batch_start: int) -> int:
        """计算一批互不重叠细胞的整体基因比例。"""

        batch_stop = min(batch_start + DEFAULT_CELL_BATCH_SIZE, state.num_cells)
        for cell_code in range(batch_start, batch_stop):
            cell_rows = row_order[int(cell_offsets[cell_code]):int(cell_offsets[cell_code + 1])]
            gene_counts = np.bincount(gene_codes[cell_rows], minlength=num_genes).astype(np.float64, copy=False)
            cell_profiles[cell_code] = gene_counts / gene_counts.sum()
        return batch_stop - batch_start

    completed_cells = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for completed_batch in executor.map(process_batch, range(0, state.num_cells, DEFAULT_CELL_BATCH_SIZE)):
            completed_cells += completed_batch
            if profile and completed_cells % 10_000 < completed_batch:
                print(f"[SAMPLED][FAST-SAME-CELL] cell profiles: {completed_cells:,}/{state.num_cells:,}", flush=True)
    cell_profiles.flush()
    return cell_profiles


def compute_same_cell_rnaflux_fast(
    df: pd.DataFrame,
    sampled_df: pd.DataFrame,
    unique_genes: np.ndarray,
    radius: float,
    device: torch.device,
    work_dir: Path,
    *,
    profile: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.memmap, FastSameCellState]:
    """批量计算与原同细胞loop定义一致的RNAflux。

    Parameters
    ----------
    df, sampled_df, unique_genes, radius, device
        与原同细胞Notebook路径相同的输入、抽样结果、基因顺序、半径和Torch设备。
    work_dir
        存放行号索引、细胞比例矩阵和RNAflux磁盘映射的临时目录。
    profile
        是否输出阶段进度。

    Returns
    -------
    tuple
        查询坐标、查询细胞编号、RNAflux磁盘映射和复用索引状态。
    """

    work_dir.mkdir(parents=True, exist_ok=True)
    sample_positions = _resolve_sample_positions(df, sampled_df)
    query_coords = sampled_df[["x", "y"]].to_numpy(copy=True)
    cell_ids = sampled_df["cell"].astype(str).to_numpy(copy=True)

    # 原groupby默认按cell名称排序，因此显式使用排序类别保持cell profile行顺序一致。
    cell_labels = np.sort(pd.unique(df["cell"].astype(str)))
    cell_codes = pd.Categorical(df["cell"].astype(str), categories=cell_labels).codes.astype(np.int32, copy=False)
    gene_codes = pd.Categorical(df["gene"].astype(str), categories=unique_genes).codes.astype(np.int32, copy=False)
    if np.any(gene_codes < 0):
        raise ValueError("gene列包含不在unique_genes中的值。")
    row_order_path, cell_offsets_path = _build_stable_cell_index(cell_codes, work_dir)
    query_cell_codes = cell_codes[sample_positions]
    query_order, query_starts, query_stops = _build_query_lookup(query_cell_codes, len(cell_labels))
    state = FastSameCellState(row_order_path, cell_offsets_path, query_cell_codes, query_order, query_starts, query_stops, len(cell_labels))

    workers = 1 if device.type == "cuda" else DEFAULT_SAME_CELL_WORKERS
    cell_profiles = _build_cell_profiles(gene_codes, state, len(unique_genes), work_dir / "same_cell_profiles.npy", profile=profile, workers=workers)

    # 使用原Torch std实现和同一设备，避免改变跨细胞标准差定义与数据类型提升规则。
    profile_tensor = torch.from_numpy(cell_profiles)
    if device.type != "cpu":
        profile_tensor = profile_tensor.to(device)
    std_dev = torch.std(profile_tensor, dim=0)
    del profile_tensor

    rnaflux = np.lib.format.open_memmap(work_dir / "same_cell_rnaflux.npy", mode="w+", dtype=np.float64, shape=(len(sampled_df), len(unique_genes)))
    row_order = np.load(state.row_order_path, mmap_mode="r")
    cell_offsets = np.load(state.cell_offsets_path, mmap_mode="r")
    x_values = df["x"].to_numpy(copy=False)
    y_values = df["y"].to_numpy(copy=False)

    def process_batch(batch_start: int) -> int:
        """按细胞批量统计局部基因比例，并直接写入原采样顺序位置。"""

        batch_stop = min(batch_start + DEFAULT_CELL_BATCH_SIZE, state.num_cells)
        for cell_code in range(batch_start, batch_stop):
            lookup_start = state.query_starts[cell_code]
            if lookup_start < 0:
                continue
            cell_rows = row_order[int(cell_offsets[cell_code]):int(cell_offsets[cell_code + 1])]
            query_ids = state.query_order[lookup_start:state.query_stops[cell_code]]
            transcript_xy = np.column_stack((x_values[cell_rows], y_values[cell_rows]))
            source_query_xy = query_coords[query_ids]
            neighbor_pairs = cKDTree(source_query_xy).sparse_distance_matrix(cKDTree(transcript_xy), radius, output_type="coo_matrix")
            flat_positions = neighbor_pairs.row.astype(np.int64) * len(unique_genes) + gene_codes[cell_rows][neighbor_pairs.col]
            local_counts = np.bincount(flat_positions, minlength=len(query_ids) * len(unique_genes)).reshape(len(query_ids), len(unique_genes)).astype(np.float32)
            local_totals = local_counts.sum(axis=1, dtype=np.float32)
            local_profiles = np.divide(local_counts, local_totals[:, None], out=np.zeros_like(local_counts), where=local_totals[:, None] > 0)

            # 与原loop相同：float32局部比例减float64细胞比例，再除以Torch样本标准差。
            local_tensor = torch.from_numpy(local_profiles)
            cell_tensor = torch.from_numpy(np.asarray(cell_profiles[cell_code]))
            if device.type != "cpu":
                local_tensor = local_tensor.to(device)
                cell_tensor = cell_tensor.to(device)
            cell_rnaflux = (local_tensor - cell_tensor) / std_dev
            rnaflux[query_ids] = cell_rnaflux.cpu().numpy()
        return batch_stop - batch_start

    completed_cells = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for completed_batch in executor.map(process_batch, range(0, state.num_cells, DEFAULT_CELL_BATCH_SIZE)):
            completed_cells += completed_batch
            if profile and completed_cells % 10_000 < completed_batch:
                print(f"[SAMPLED][FAST-SAME-CELL] RNAflux: {completed_cells:,}/{state.num_cells:,}", flush=True)
    rnaflux.flush()
    del std_dev, cell_profiles
    return query_coords, cell_ids, rnaflux, state


def interpolate_same_cell_rgb_fast(
    df: pd.DataFrame,
    query_coords: np.ndarray,
    normalized_embeddings: np.ndarray,
    state: FastSameCellState,
    output_path: Path,
    *,
    profile: bool = False,
) -> np.memmap:
    """使用原逐通道 ``griddata`` 规则完成低内存同细胞RGB插值。

    无查询点细胞、少于三个查询点细胞、线性插值、最近邻补值及最终裁剪规则均与
    ``_interpolate_rgb_same_cell`` 保持一致；仅把反复扫描query DataFrame改为索引查找。
    """

    rgb_image = np.lib.format.open_memmap(output_path, mode="w+", dtype=np.float64, shape=(len(df), 3))
    row_order = np.load(state.row_order_path, mmap_mode="r")
    cell_offsets = np.load(state.cell_offsets_path, mmap_mode="r")
    x_values = df["x"].to_numpy(copy=False)
    y_values = df["y"].to_numpy(copy=False)
    global_mean = np.nanmean(normalized_embeddings, axis=0)
    fallback_mean = float(np.nanmean(normalized_embeddings))

    def process_batch(batch_start: int) -> int:
        """对一批细胞执行与原函数相同的逐通道griddata。"""

        batch_stop = min(batch_start + DEFAULT_CELL_BATCH_SIZE, state.num_cells)
        for cell_code in range(batch_start, batch_stop):
            cell_rows = row_order[int(cell_offsets[cell_code]):int(cell_offsets[cell_code + 1])]
            target_xy = np.column_stack((x_values[cell_rows], y_values[cell_rows]))
            lookup_start = state.query_starts[cell_code]
            if lookup_start < 0:
                rgb_image[cell_rows] = global_mean
                continue
            query_ids = state.query_order[lookup_start:state.query_stops[cell_code]]
            source_xy = query_coords[query_ids]
            source_rgb = normalized_embeddings[query_ids]
            if len(query_ids) < 3:
                for channel_index in range(3):
                    rgb_image[cell_rows, channel_index] = griddata(source_xy, source_rgb[:, channel_index], target_xy, method="nearest")
                continue
            for channel_index in range(3):
                linear_values = griddata(source_xy, source_rgb[:, channel_index], target_xy, method="linear")
                nearest_values = griddata(source_xy, source_rgb[:, channel_index], target_xy, method="nearest")
                rgb_image[cell_rows, channel_index] = np.where(np.isnan(linear_values), nearest_values, linear_values)
            rgb_image[cell_rows] = np.clip(rgb_image[cell_rows], 0, 1)
        return batch_stop - batch_start

    completed_cells = 0
    with ThreadPoolExecutor(max_workers=DEFAULT_SAME_CELL_WORKERS) as executor:
        for completed_batch in executor.map(process_batch, range(0, state.num_cells, DEFAULT_CELL_BATCH_SIZE)):
            completed_cells += completed_batch
            if profile and completed_cells % 10_000 < completed_batch:
                print(f"[SAMPLED][FAST-SAME-CELL] interpolation: {completed_cells:,}/{state.num_cells:,}", flush=True)
    # 原函数在全部细胞处理结束后统一替换nan/inf，再裁剪三个通道；这里保持相同顺序和标量回填值。
    np.nan_to_num(rgb_image, copy=False, nan=fallback_mean, posinf=fallback_mean, neginf=fallback_mean)
    np.clip(rgb_image, 0, 1, out=rgb_image)
    rgb_image.flush()
    return rgb_image
