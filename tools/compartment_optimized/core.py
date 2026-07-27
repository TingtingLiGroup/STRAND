"""面向超大转录本数据集的细胞内 RNAflux 区室探测。

本模块不依赖也不修改旧区室实现。空间邻域和插值严格限制在所属细胞内，
SVD 与 SOM 在全部细胞上统一训练；高维 RNAflux、全量 RGB 和标签使用磁盘映射。
"""

from __future__ import annotations

import json
import pickle
import re
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterator, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as arrow_csv
from kneed import KneeLocator
from minisom import MiniSom
from numba import njit
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from scipy.spatial import QhullError, cKDTree
from sklearn.decomposition import TruncatedSVD

from .plotting import plot_tile_outputs


@dataclass(frozen=True)
class OptimizedCompartmentConfig:
    """保存优化流程参数，所有随机步骤共用 ``random_seed`` 以保证可复现。"""

    frac: float = 0.01
    radius: float = 5.0
    n_clusters: str = "auto"
    cluster_range_min: int = 4
    cluster_range_max: int = 7
    som_iterations: int = 100
    som_sigma: float = 0.5
    som_learning_rate: float = 0.5
    random_seed: int = 42
    som_chunk_size: int = 1_000_000
    output_chunk_size: int = 250_000
    progress_interval_cells: int = 10_000
    workers: int = 16
    cell_batch_size: int = 256
    som_workers: int = 4
    output_workers: int = 4


@dataclass(frozen=True)
class CellRowIndex:
    """使用磁盘行号数组描述每个细胞对应的原始DataFrame行。"""

    row_order_path: Path
    offsets_path: Path
    num_cells: int


@njit(cache=True)
def _fill_stable_cell_row_order(cell_codes: np.ndarray, cell_offsets: np.ndarray, row_order: np.ndarray) -> None:
    """按原始行顺序执行稳定计数分组，避免3.67亿行全量argsort。"""

    write_positions = cell_offsets[:-1].copy()
    for row_index in range(len(cell_codes)):
        cell_code = cell_codes[row_index]
        target_position = write_positions[cell_code]
        row_order[target_position] = row_index
        write_positions[cell_code] += 1


def _safe_name(value: object) -> str:
    """将分组值转换为安全文件名，避免路径分隔符进入输出路径。"""

    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("_")
    return cleaned or "group"


def _attach_cell_metadata_category(df: pd.DataFrame, coordinates: pd.DataFrame, column: str) -> None:
    """先在细胞级映射元数据，再用紧凑类别码展开到转录本，避免全量object字符串列。"""

    if not {"cell", column}.issubset(coordinates.columns):
        raise ValueError(f"coordinates 缺少 cell 或 {column!r}。")
    coordinate_values = coordinates[["cell", column]].drop_duplicates("cell").copy()
    coordinate_values["cell"] = coordinate_values["cell"].astype(str)
    metadata_map = coordinate_values.set_index("cell")[column]
    cell_labels = pd.Index(df["cell"].cat.categories.astype(str))
    cell_metadata = metadata_map.reindex(cell_labels).fillna("unknown").astype(str).to_numpy()
    metadata_categories = pd.unique(cell_metadata)
    cell_metadata_codes = pd.Categorical(cell_metadata, categories=metadata_categories).codes
    point_metadata_codes = cell_metadata_codes[df["cell"].cat.codes.to_numpy(copy=False)]
    df[column] = pd.Categorical.from_codes(point_metadata_codes, categories=metadata_categories)


def _sample_positions(num_rows: int, frac: float, random_seed: int) -> np.ndarray:
    """复现 ``DataFrame.sample(frac=..., random_state=...)`` 的无放回位置抽样。"""

    if not 0 < frac <= 1:
        raise ValueError("frac 必须位于 (0, 1]。")
    sample_size = int(round(frac * num_rows))
    if sample_size <= 0:
        raise ValueError("采样结果为空，请提高 frac。")
    random_state = np.random.RandomState(random_seed)
    return random_state.choice(num_rows, size=sample_size, replace=False).astype(np.int64, copy=False)


def _iter_contiguous_cell_blocks(cell_codes: np.ndarray) -> Iterator[Tuple[int, int, int]]:
    """按输入顺序返回连续细胞区间，并拒绝同一细胞分散在多个区间的高内存输入。"""

    if len(cell_codes) == 0:
        return
    # 分块寻找边界，避免为3.67亿行一次创建约367 MB的全量布尔数组。
    scan_chunk_size = 5_000_000
    seen_codes = set()
    block_start = 0
    current_code = int(cell_codes[0])
    for scan_start in range(1, len(cell_codes), scan_chunk_size):
        scan_stop = min(scan_start + scan_chunk_size, len(cell_codes))
        previous_codes = cell_codes[scan_start - 1:scan_stop - 1]
        current_codes = cell_codes[scan_start:scan_stop]
        change_positions = np.flatnonzero(current_codes != previous_codes) + scan_start
        for block_stop in change_positions:
            if current_code in seen_codes:
                raise ValueError(
                    "同一 cell 的转录本不连续。为避免给超大数据创建全量排序索引，"
                    "请先按 cell 稳定排序后再运行优化版本。"
                )
            seen_codes.add(current_code)
            yield block_start, int(block_stop), current_code
            block_start = int(block_stop)
            current_code = int(cell_codes[block_start])
    if current_code in seen_codes:
        raise ValueError(
            "同一 cell 的转录本不连续。为避免给超大数据创建全量排序索引，"
            "请先按 cell 稳定排序后再运行优化版本。"
        )
    yield block_start, len(cell_codes), current_code


def build_cell_row_index(cell_codes: np.ndarray, work_dir: Path, *, profile: bool = False) -> CellRowIndex:
    """稳定按cell编码排列原始行号，并将最终uint32索引保存到磁盘映射。

    Dataset4的转录本按基因交错排列，不能依赖cell连续性。这里仅排序行号，不复制或
    重排16.7 GiB DataFrame；稳定排序保证每个细胞内部仍保持原始转录本顺序。
    """

    if len(cell_codes) > np.iinfo(np.uint32).max:
        raise ValueError("当前uint32行号索引最多支持4,294,967,295行。")
    if np.any(cell_codes < 0):
        raise ValueError("cell包含缺失类别编码。")
    num_cells = int(cell_codes.max()) + 1
    work_dir.mkdir(parents=True, exist_ok=True)
    row_order_path = work_dir / "cell_row_order.npy"
    offsets_path = work_dir / "cell_row_offsets.npy"
    cell_counts = np.bincount(cell_codes, minlength=num_cells).astype(np.int64, copy=False)
    offsets = np.empty(num_cells + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(cell_counts, out=offsets[1:])
    np.save(offsets_path, offsets)
    if profile:
        print("[OPT-COMPARTMENT] building stable linear cell index with Numba", flush=True)
    row_order = np.lib.format.open_memmap(row_order_path, mode="w+", dtype=np.uint32, shape=(len(cell_codes),))
    _fill_stable_cell_row_order(cell_codes, offsets, row_order)
    row_order.flush()
    if profile:
        print(f"[OPT-COMPARTMENT] stable cell index ready: {num_cells:,} cells, {len(cell_codes):,} rows", flush=True)
    return CellRowIndex(row_order_path=row_order_path, offsets_path=offsets_path, num_cells=num_cells)


def _iter_indexed_cells(cell_index: CellRowIndex) -> Iterator[Tuple[int, np.ndarray]]:
    """按cell编码顺序返回该细胞的原始行号视图，不复制全量索引。"""

    row_order = np.load(cell_index.row_order_path, mmap_mode="r")
    offsets = np.load(cell_index.offsets_path, mmap_mode="r")
    for cell_code in range(cell_index.num_cells):
        start = int(offsets[cell_code])
        stop = int(offsets[cell_code + 1])
        if stop > start:
            yield cell_code, row_order[start:stop]


def _prepare_dataframe(bundle: dict, group_col: str, only_group: Optional[str]) -> pd.DataFrame:
    """读取并压缩必要列；只有确实筛选子分组时才创建 DataFrame 副本。"""

    data_df = bundle.get("data_df")
    if not isinstance(data_df, pd.DataFrame):
        raise ValueError("PKL 中缺少 DataFrame 类型的 data_df。")
    required_columns = {"cell", "gene", "x", "y"}
    missing_columns = sorted(required_columns.difference(data_df.columns))
    if missing_columns:
        raise ValueError(f"data_df 缺少列：{missing_columns}")
    if data_df[["cell", "gene", "x", "y"]].isna().any().any():
        raise ValueError("cell、gene、x、y 不允许包含缺失值。")

    df = data_df
    # gene 类别顺序严格使用首次出现顺序，以保持 RNAflux 列定义不变。
    unique_genes = pd.unique(df["gene"])
    df["gene"] = pd.Categorical(df["gene"], categories=unique_genes, ordered=True)
    df["cell"] = df["cell"].astype("category")
    if group_col not in df.columns:
        coordinates = bundle.get("coordinates")
        if not isinstance(coordinates, pd.DataFrame) or not {"cell", group_col}.issubset(coordinates.columns):
            if only_group is not None:
                raise ValueError(f"data_df 和 coordinates 均缺少分组列 {group_col!r}。")
            df[group_col] = pd.Categorical.from_codes(np.zeros(len(df), dtype=np.int8), categories=["all"])
        else:
            _attach_cell_metadata_category(df, coordinates, group_col)
    else:
        df[group_col] = df[group_col].astype("category")

    if only_group is not None:
        category_labels = df[group_col].cat.categories.astype(str).to_numpy()
        matching_codes = np.flatnonzero(category_labels == str(only_group))
        if len(matching_codes) != 1:
            raise ValueError(f"分组列 {group_col!r} 中找不到 {only_group!r}。")
        group_mask = df[group_col].cat.codes.to_numpy(copy=False) == int(matching_codes[0])
        if not np.all(group_mask):
            df = df.loc[group_mask].copy()
    df["x"] = pd.to_numeric(df["x"], errors="raise")
    df["y"] = pd.to_numeric(df["y"], errors="raise")
    if not np.isfinite(df[["x", "y"]].to_numpy()).all():
        raise ValueError("x、y 必须全部为有限数值。")
    return df


def _build_query_lookup(query_cell_codes: np.ndarray, num_cells: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """建立紧凑的查询点排序索引，避免为30万个细胞创建大量小数组。"""

    query_order = np.argsort(query_cell_codes, kind="stable")
    sorted_codes = query_cell_codes[query_order]
    query_starts = np.full(num_cells, -1, dtype=np.int64)
    query_stops = np.full(num_cells, -1, dtype=np.int64)
    changes = np.flatnonzero(np.r_[True, sorted_codes[1:] != sorted_codes[:-1]])
    stops = np.r_[changes[1:], len(sorted_codes)]
    for start, stop in zip(changes, stops):
        cell_code = int(sorted_codes[start])
        query_starts[cell_code] = int(start)
        query_stops[cell_code] = int(stop)
    return query_order, query_starts, query_stops


def _compute_cell_std(cell_index: CellRowIndex, gene_codes: np.ndarray, num_genes: int, profile: bool, interval: int) -> np.ndarray:
    """用Welford流式算法计算跨细胞基因比例样本标准差，避免细胞×基因稠密矩阵。"""

    mean_profile = np.zeros(num_genes, dtype=np.float64)
    squared_deviation_sum = np.zeros(num_genes, dtype=np.float64)
    num_cells = 0
    for _, cell_rows in _iter_indexed_cells(cell_index):
        gene_counts = np.bincount(gene_codes[cell_rows], minlength=num_genes).astype(np.float64, copy=False)
        cell_profile = gene_counts / gene_counts.sum()
        num_cells += 1
        profile_delta = cell_profile - mean_profile
        mean_profile += profile_delta / num_cells
        squared_deviation_sum += profile_delta * (cell_profile - mean_profile)
        if profile and num_cells % interval == 0:
            print(f"[OPT-COMPARTMENT] cell std: {num_cells:,} cells", flush=True)
    if num_cells < 2:
        raise ValueError("至少需要两个细胞计算跨细胞标准差。")
    return np.sqrt(squared_deviation_sum / (num_cells - 1))


def _check_work_disk(work_dir: Path, num_queries: int, num_genes: int, num_points: int) -> None:
    """在长任务开始前估算主要磁盘映射文件，避免运行数小时后因磁盘不足退出。"""

    required_bytes = num_queries * num_genes * 8 + num_points * 3 * 8 + num_points * 4 + num_points * 4
    free_bytes = shutil.disk_usage(work_dir).free
    if free_bytes < required_bytes * 1.15:
        required_gib = required_bytes / 1024**3
        free_gib = free_bytes / 1024**3
        raise OSError(f"工作目录磁盘不足：预计至少需要 {required_gib:.1f} GiB，当前可用 {free_gib:.1f} GiB。")


def compute_same_cell_rnaflux(
    df: pd.DataFrame,
    sample_positions: np.ndarray,
    radius: float,
    output_path: Path,
    *,
    cell_index: Optional[CellRowIndex] = None,
    profile: bool = False,
    progress_interval_cells: int = 10_000,
    workers: int = 1,
    cell_batch_size: int = 256,
) -> Tuple[np.ndarray, np.ndarray]:
    """逐细胞批量计算同细胞邻域RNAflux，并将高维结果直接写入 ``.npy`` 磁盘映射。"""

    if radius < 0:
        raise ValueError("radius 不能为负数。")
    cell_codes = df["cell"].cat.codes.to_numpy(copy=False)
    gene_codes = df["gene"].cat.codes.to_numpy(copy=False)
    num_cells = len(df["cell"].cat.categories)
    num_genes = len(df["gene"].cat.categories)
    cell_index = cell_index or build_cell_row_index(cell_codes, output_path.parent, profile=profile)
    query_cell_codes = cell_codes[sample_positions]
    query_order, query_starts, query_stops = _build_query_lookup(query_cell_codes, num_cells)
    cell_std = _compute_cell_std(cell_index, gene_codes, num_genes, profile, progress_interval_cells)
    np.save(output_path.with_name("cell_gene_std.npy"), cell_std)

    x_values = df["x"].to_numpy(copy=False)
    y_values = df["y"].to_numpy(copy=False)
    rnaflux = np.lib.format.open_memmap(output_path, mode="w+", dtype=np.float64, shape=(len(sample_positions), num_genes))
    row_order = np.load(cell_index.row_order_path, mmap_mode="r")
    offsets = np.load(cell_index.offsets_path, mmap_mode="r")

    def process_batch(batch_start: int) -> int:
        """处理一批互不重叠的细胞，线程只写自身查询点行。"""

        batch_stop = min(batch_start + cell_batch_size, cell_index.num_cells)
        for cell_code in range(batch_start, batch_stop):
            lookup_start = query_starts[cell_code]
            if lookup_start < 0:
                continue
            cell_rows = row_order[int(offsets[cell_code]):int(offsets[cell_code + 1])]
            query_ids = query_order[lookup_start:query_stops[cell_code]]
            transcript_xy = np.column_stack((x_values[cell_rows], y_values[cell_rows]))
            query_xy = np.column_stack((x_values[sample_positions[query_ids]], y_values[sample_positions[query_ids]]))
            neighbor_pairs = cKDTree(query_xy).sparse_distance_matrix(cKDTree(transcript_xy), radius, output_type="coo_matrix")
            flat_gene_positions = neighbor_pairs.row.astype(np.int64) * num_genes + gene_codes[cell_rows][neighbor_pairs.col]
            local_counts = np.bincount(flat_gene_positions, minlength=len(query_ids) * num_genes).reshape(len(query_ids), num_genes).astype(np.float32)
            local_totals = local_counts.sum(axis=1, dtype=np.float32)
            local_profiles = np.divide(local_counts, local_totals[:, None], out=np.zeros_like(local_counts), where=local_totals[:, None] > 0)
            cell_gene_counts = np.bincount(gene_codes[cell_rows], minlength=num_genes).astype(np.float64, copy=False)
            cell_profile = cell_gene_counts / cell_gene_counts.sum()
            with np.errstate(divide="ignore", invalid="ignore"):
                rnaflux[query_ids] = (local_profiles.astype(np.float64) - cell_profile) / cell_std
        return batch_stop - batch_start

    batch_starts = range(0, cell_index.num_cells, cell_batch_size)
    completed_cells = 0
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        for completed_batch in executor.map(process_batch, batch_starts):
            completed_cells += completed_batch
            if profile and completed_cells // progress_interval_cells != (completed_cells - completed_batch) // progress_interval_cells:
                print(f"[OPT-COMPARTMENT] RNAflux: {completed_cells:,}/{cell_index.num_cells:,} cells", flush=True)
    rnaflux.flush()
    return cell_std, query_cell_codes


def _quantile_normalize(embeddings: np.ndarray, lower_bound: float = 0.1, upper_bound: float = 0.9) -> np.ndarray:
    """保持旧区室流程的1%/99%分位数归一化公式。"""

    q_min = np.percentile(embeddings, 1, axis=0)
    q_max = np.percentile(embeddings, 99, axis=0)
    normalized = (embeddings - q_min) / (q_max - q_min)
    normalized = np.clip(normalized, 0, 1)
    return normalized * (lower_bound - upper_bound) + upper_bound


def fit_global_svd(rnaflux_path: Path, random_seed: int, output_path: Path) -> Tuple[TruncatedSVD, np.ndarray]:
    """在磁盘映射RNAflux上统一拟合三维TruncatedSVD，并保存归一化查询点embedding。"""

    rnaflux = np.load(rnaflux_path, mmap_mode="r")
    if min(rnaflux.shape) < 3:
        raise ValueError("RNAflux 至少需要3个查询点和3个基因才能拟合三维SVD。")
    svd = TruncatedSVD(n_components=3, random_state=random_seed)
    normalized_embeddings = _quantile_normalize(svd.fit_transform(rnaflux))
    np.save(output_path, normalized_embeddings)
    return svd, normalized_embeddings


def interpolate_same_cell_rgb(
    df: pd.DataFrame,
    sample_positions: np.ndarray,
    query_cell_codes: np.ndarray,
    normalized_embeddings: np.ndarray,
    output_path: Path,
    *,
    cell_index: Optional[CellRowIndex] = None,
    profile: bool = False,
    progress_interval_cells: int = 10_000,
    workers: int = 1,
    cell_batch_size: int = 256,
) -> np.memmap:
    """逐细胞完成线性插值并以最近邻补凸包外点，任何空间值都不会跨越细胞边界。"""

    cell_codes = df["cell"].cat.codes.to_numpy(copy=False)
    num_cells = len(df["cell"].cat.categories)
    cell_index = cell_index or build_cell_row_index(cell_codes, output_path.parent, profile=profile)
    query_order, query_starts, query_stops = _build_query_lookup(query_cell_codes, num_cells)
    x_values = df["x"].to_numpy(copy=False)
    y_values = df["y"].to_numpy(copy=False)
    global_mean = np.nanmean(normalized_embeddings, axis=0)
    rgb = np.lib.format.open_memmap(output_path, mode="w+", dtype=np.float64, shape=(len(df), 3))

    row_order = np.load(cell_index.row_order_path, mmap_mode="r")
    offsets = np.load(cell_index.offsets_path, mmap_mode="r")
    fallback_value = float(np.nanmean(normalized_embeddings))

    def process_batch(batch_start: int) -> int:
        """对一批细胞独立插值，线程写入互不重叠的原始点行。"""

        batch_stop = min(batch_start + cell_batch_size, cell_index.num_cells)
        for cell_code in range(batch_start, batch_stop):
            cell_rows = row_order[int(offsets[cell_code]):int(offsets[cell_code + 1])]
            target_xy = np.column_stack((x_values[cell_rows], y_values[cell_rows]))
            lookup_start = query_starts[cell_code]
            if lookup_start < 0:
                rgb[cell_rows] = global_mean
                continue
            query_ids = query_order[lookup_start:query_stops[cell_code]]
            source_xy = np.column_stack((x_values[sample_positions[query_ids]], y_values[sample_positions[query_ids]]))
            source_rgb = normalized_embeddings[query_ids]
            nearest_values = NearestNDInterpolator(source_xy, source_rgb)(target_xy)
            if len(query_ids) < 3:
                cell_rgb = nearest_values
            else:
                try:
                    linear_values = LinearNDInterpolator(source_xy, source_rgb)(target_xy)
                    cell_rgb = np.where(np.isnan(linear_values), nearest_values, linear_values)
                except QhullError:
                    cell_rgb = nearest_values
            rgb[cell_rows] = np.clip(np.nan_to_num(cell_rgb, nan=fallback_value), 0, 1)
        return batch_stop - batch_start

    completed_cells = 0
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        for completed_batch in executor.map(process_batch, range(0, cell_index.num_cells, cell_batch_size)):
            completed_cells += completed_batch
            if profile and completed_cells // progress_interval_cells != (completed_cells - completed_batch) // progress_interval_cells:
                print(f"[OPT-COMPARTMENT] interpolation: {completed_cells:,}/{cell_index.num_cells:,} cells", flush=True)
    rgb.flush()
    return rgb


def _som_quantization_error_chunked(som: MiniSom, rgb: np.ndarray, chunk_size: int) -> float:
    """按原顺序分块计算SOM量化误差，避免创建全量点×节点距离矩阵。"""

    distance_sum = 0.0
    for start in range(0, len(rgb), chunk_size):
        chunk = np.asarray(rgb[start:start + chunk_size])
        distance_sum += float(np.linalg.norm(chunk - som.quantization(chunk), axis=1).sum(dtype=np.float64))
    return distance_sum / len(rgb)


def _som_winners_chunked(som: MiniSom, rgb: np.ndarray, output_path: Path, chunk_size: int) -> np.memmap:
    """按原始点顺序分块写出SOM winner，避免数亿次Python ``som.winner`` 调用。"""

    labels = np.lib.format.open_memmap(output_path, mode="w+", dtype=np.int32, shape=(len(rgb),))
    weights = som.get_weights()[0]
    for start in range(0, len(rgb), chunk_size):
        chunk = np.asarray(rgb[start:start + chunk_size])
        labels[start:start + len(chunk)] = np.argmin(np.linalg.norm(chunk[:, None, :] - weights[None, :, :], axis=2), axis=1)
    labels.flush()
    return labels


def fit_global_som(rgb_path: Path, config: OptimizedCompartmentConfig, labels_path: Path) -> Tuple[int, np.memmap, list]:
    """在全数据集RGB上统一选择并训练SOM，再将每点winner分块写入磁盘。"""

    rgb = np.load(rgb_path, mmap_mode="r")
    if str(config.n_clusters).lower() == "auto":
        k_values = list(range(config.cluster_range_min, config.cluster_range_max + 1))
        models: Dict[int, MiniSom] = {}
        quantization_errors = []
        def train_candidate(cluster_count: int):
            som = MiniSom(1, cluster_count, 3, sigma=config.som_sigma, learning_rate=config.som_learning_rate, random_seed=config.random_seed)
            som.train(rgb, config.som_iterations)
            return cluster_count, som, _som_quantization_error_chunked(som, rgb, config.som_chunk_size)

        with ThreadPoolExecutor(max_workers=min(max(1, config.som_workers), len(k_values))) as executor:
            candidate_results = list(executor.map(train_candidate, k_values))
        for cluster_count, som_model, quantization_error in candidate_results:
            models[cluster_count] = som_model
            quantization_errors.append(quantization_error)
        elbow = KneeLocator(k_values, quantization_errors, curve="convex", direction="decreasing").elbow
        best_k = int(elbow if elbow is not None else max(config.cluster_range_min, min(config.cluster_range_max, 5)))
        som = models[best_k]
    else:
        best_k = int(config.n_clusters)
        quantization_errors = []
        som = MiniSom(1, best_k, 3, sigma=config.som_sigma, learning_rate=config.som_learning_rate, random_seed=config.random_seed)
        som.train(rgb, config.som_iterations)

    labels = _som_winners_chunked(som, rgb, labels_path, config.som_chunk_size)
    return best_k, labels, quantization_errors


def write_point_outputs(df: pd.DataFrame, labels_path: Path, output_dir: Path, display_col: Optional[str], chunk_size: int) -> Dict[str, str]:
    """单次扫描DataFrame并按tile追加CSV，避免为每个tile重复扫描3.67亿行。"""

    output_dir.mkdir(parents=True, exist_ok=True)
    labels = np.load(labels_path, mmap_mode="r")
    output_paths: Dict[str, str] = {}
    initialized_paths = set()
    for start in range(0, len(df), chunk_size):
        stop = min(start + chunk_size, len(df))
        chunk = pd.DataFrame({
            "point_index": np.arange(start, stop, dtype=np.int64),
            "cell": df["cell"].iloc[start:stop].astype(str).to_numpy(),
            "gene": df["gene"].iloc[start:stop].astype(str).to_numpy(),
            "x": df["x"].iloc[start:stop].to_numpy(),
            "y": df["y"].iloc[start:stop].to_numpy(),
            "subdomain": np.asarray(labels[start:stop]),
        })
        if display_col is None:
            groups = [("all", chunk)]
        else:
            display_values = df[display_col].iloc[start:stop].astype(str).to_numpy()
            chunk[display_col] = display_values
            groups = chunk.groupby(display_col, sort=False)
        for display_value, group_df in groups:
            safe_value = _safe_name(display_value)
            output_path = output_dir / f"subdomains_{safe_value}.csv"
            # 第一次遇到该tile时先删除旧文件，防止重复运行把历史结果追加到本次结果中。
            if output_path not in initialized_paths and output_path.exists():
                output_path.unlink()
            group_df.to_csv(output_path, mode="a", index=False, header=output_path not in initialized_paths)
            initialized_paths.add(output_path)
            output_paths[str(display_value)] = str(output_path)
    return output_paths


def write_tile_coordinate_outputs(df: pd.DataFrame, labels_path: Path, rgb_path: Path, output_dir: Path, display_col: str, chunk_size: int) -> Dict[str, str]:
    """在全局标签完成后单次扫描，按tile暂存坐标、统一subdomain和全局RNAflux RGB。"""

    if display_col not in df.columns or not isinstance(df[display_col].dtype, pd.CategoricalDtype):
        raise ValueError(f"展示列 {display_col!r} 必须是分类列。")
    output_dir.mkdir(parents=True, exist_ok=True)
    display_codes = df[display_col].cat.codes.to_numpy(copy=False)
    display_labels = df[display_col].cat.categories.astype(str).to_numpy()
    tile_counts = np.bincount(display_codes, minlength=len(display_labels)).astype(np.int64, copy=False)
    point_dtype = np.dtype([("point_index", "<u4"), ("x", "<f8"), ("y", "<f8"), ("subdomain", "<i4"), ("rnaflux_r", "<f8"), ("rnaflux_g", "<f8"), ("rnaflux_b", "<f8")])
    tile_arrays = []
    output_paths: Dict[str, str] = {}
    for tile_code, tile_label in enumerate(display_labels):
        # 分类列可能保留筛选前的空类别；空 tile 不应创建文件或出现在元数据中。
        if tile_counts[tile_code] == 0:
            tile_arrays.append(None)
            continue
        output_path = output_dir / f"subdomains_{_safe_name(tile_label)}.npy"
        tile_arrays.append(np.lib.format.open_memmap(output_path, mode="w+", dtype=point_dtype, shape=(int(tile_counts[tile_code]),)))
        output_paths[str(tile_label)] = str(output_path)

    labels = np.load(labels_path, mmap_mode="r")
    rgb = np.load(rgb_path, mmap_mode="r")
    x_values = df["x"].to_numpy(copy=False)
    y_values = df["y"].to_numpy(copy=False)
    write_offsets = np.zeros(len(display_labels), dtype=np.int64)
    for start in range(0, len(df), chunk_size):
        stop = min(start + chunk_size, len(df))
        chunk_codes = display_codes[start:stop]
        for tile_code in np.unique(chunk_codes):
            local_rows = np.flatnonzero(chunk_codes == tile_code)
            output_start = int(write_offsets[tile_code])
            output_stop = output_start + len(local_rows)
            tile_output = tile_arrays[tile_code][output_start:output_stop]
            source_rows = start + local_rows
            tile_output["point_index"] = source_rows.astype(np.uint32, copy=False)
            tile_output["x"] = x_values[source_rows]
            tile_output["y"] = y_values[source_rows]
            tile_output["subdomain"] = labels[source_rows]
            tile_output["rnaflux_r"] = rgb[source_rows, 0]
            tile_output["rnaflux_g"] = rgb[source_rows, 1]
            tile_output["rnaflux_b"] = rgb[source_rows, 2]
            write_offsets[tile_code] = output_stop
    for tile_array in tile_arrays:
        if tile_array is not None:
            tile_array.flush()
    if not np.array_equal(write_offsets, tile_counts):
        raise RuntimeError("分tile输出数量校验失败。")
    return output_paths


def write_compatible_tile_csv_outputs(
    df: pd.DataFrame,
    tile_staging_paths: Dict[str, str],
    output_dir: Path,
    display_col: str,
    chunk_size: int,
    output_workers: int,
) -> Dict[str, str]:
    """并行把每tile暂存文件转换为旧版兼容CSV，保持逐点字段和原始tile内行序。"""

    output_dir.mkdir(parents=True, exist_ok=True)
    base_columns = [column for column in ["cell", "gene", "x", "y", "z", "umi"] if column in df.columns]
    metadata_columns = [column for column in ["fov", display_col] if column in df.columns and column not in base_columns]
    output_columns = base_columns + metadata_columns + ["subdomain", "rnaflux_r", "rnaflux_g", "rnaflux_b"]

    def convert_tile(item: Tuple[str, str]) -> Tuple[str, str]:
        """转换单个tile；不同线程只写各自CSV。"""

        tile_label, staging_path = item
        tile_points = np.load(staging_path, mmap_mode="r")
        output_path = output_dir / f"{_safe_name(tile_label)}_sampled_subdomain_points.csv"
        wrote_header = False
        for start in range(0, len(tile_points), chunk_size):
            stop = min(start + chunk_size, len(tile_points))
            point_chunk = tile_points[start:stop]
            source_rows = point_chunk["point_index"].astype(np.int64, copy=False)
            output_data = {}
            for column in base_columns:
                if column == "x":
                    output_data[column] = point_chunk["x"]
                elif column == "y":
                    output_data[column] = point_chunk["y"]
                else:
                    output_data[column] = df[column].iloc[source_rows].astype(str).to_numpy() if isinstance(df[column].dtype, pd.CategoricalDtype) else df[column].iloc[source_rows].to_numpy()
            for column in metadata_columns:
                output_data[column] = df[column].iloc[source_rows].astype(str).to_numpy()
            output_data["subdomain"] = point_chunk["subdomain"]
            output_data["rnaflux_r"] = point_chunk["rnaflux_r"]
            output_data["rnaflux_g"] = point_chunk["rnaflux_g"]
            output_data["rnaflux_b"] = point_chunk["rnaflux_b"]
            table = pa.table(output_data)
            if not wrote_header:
                write_options = arrow_csv.WriteOptions(include_header=True, batch_size=65_536, quoting_style="needed", quoting_header="none")
                writer = arrow_csv.CSVWriter(str(output_path), table.schema, write_options=write_options)
                wrote_header = True
            writer.write_table(table)
        if wrote_header:
            writer.close()
        return tile_label, str(output_path)

    with ThreadPoolExecutor(max_workers=max(1, output_workers)) as executor:
        converted = list(executor.map(convert_tile, tile_staging_paths.items()))
    return dict(converted)


def write_compatible_tile_plot_outputs(
    df: pd.DataFrame,
    tile_staging_paths: Dict[str, str],
    output_dir: Path,
    *,
    cell_boundary: Optional[Dict[object, pd.DataFrame]] = None,
    nuclear_boundary: Optional[Dict[object, pd.DataFrame]] = None,
    profile: bool = False,
) -> Dict[str, Dict[str, str]]:
    """按旧版命名为每个非空 tile 生成 RNAflux 和 subdomain PNG。

    Parameters
    ----------
    df
        与暂存数组保持同一原始行顺序的全局转录本表。
    tile_staging_paths
        每个非空 tile 对应的结构化磁盘数组路径。
    output_dir
        与兼容 CSV 相同的结果目录。
    cell_boundary, nuclear_boundary
        PKL 中的细胞和细胞核边界，仅用于图像叠加，不参与任何标签计算。
    profile
        是否逐 tile 打印绘图进度。

    Returns
    -------
    dict
        每个 tile 对应的两张 PNG 路径。
    """

    plot_paths: Dict[str, Dict[str, str]] = {}
    total_tiles = len(tile_staging_paths)
    # 边界字典键可能是整数或字符串；在所有 tile 绘图前只统一一次，避免重复扫描三十万细胞。
    normalized_cell_boundary = {str(cell_id): boundary for cell_id, boundary in (cell_boundary or {}).items()}
    normalized_nuclear_boundary = {str(cell_id): boundary for cell_id, boundary in (nuclear_boundary or {}).items()}
    for tile_number, (tile_label, staging_path) in enumerate(tile_staging_paths.items(), start=1):
        # 暂存数组中的 point_index 可直接定位当前 tile 的细胞，不需要扫描大型 CSV。
        tile_points = np.load(staging_path, mmap_mode="r")
        source_rows = tile_points["point_index"].astype(np.int64, copy=False)
        tile_cell_ids = df["cell"].iloc[source_rows].unique().astype(str)
        output_prefix = output_dir / _safe_name(tile_label)
        plot_paths[tile_label] = plot_tile_outputs(
            staging_path,
            output_prefix,
            cell_ids=tile_cell_ids,
            cell_boundary=normalized_cell_boundary,
            nuclear_boundary=normalized_nuclear_boundary,
        )
        if profile:
            print(f"[OPT-COMPARTMENT] plots: {tile_number}/{total_tiles} tile={tile_label}", flush=True)
    return plot_paths


def run_optimized_compartment(
    pkl_path: str | Path,
    out_dir: str | Path,
    *,
    group_col: str = "batch",
    only_group: Optional[str] = None,
    display_col: Optional[str] = "tile",
    work_dir: Optional[str | Path] = None,
    config: Optional[OptimizedCompartmentConfig] = None,
    profile: bool = False,
) -> dict:
    """执行完整优化流程，并返回包含文件路径、规模、参数和耗时的元数据。"""

    config = config or OptimizedCompartmentConfig()
    started_at = time.time()
    out_dir = Path(out_dir)
    work_dir = Path(work_dir) if work_dir is not None else out_dir / "work"
    result_dir = out_dir / "results"
    work_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    with Path(pkl_path).open("rb") as input_file:
        bundle = pickle.load(input_file)
    df = _prepare_dataframe(bundle, group_col, only_group)
    if display_col is not None and display_col not in df.columns:
        coordinates = bundle.get("coordinates")
        if not isinstance(coordinates, pd.DataFrame) or not {"cell", display_col}.issubset(coordinates.columns):
            raise ValueError(f"找不到展示列 {display_col!r}。")
        _attach_cell_metadata_category(df, coordinates, display_col)
    if "fov" not in df.columns:
        coordinates = bundle.get("coordinates")
        if isinstance(coordinates, pd.DataFrame) and {"cell", "fov"}.issubset(coordinates.columns):
            _attach_cell_metadata_category(df, coordinates, "fov")

    cell_codes = df["cell"].cat.codes.to_numpy(copy=False)
    num_cells = len(df["cell"].cat.categories)
    sample_positions = _sample_positions(len(df), config.frac, config.random_seed)
    np.save(work_dir / "sample_positions.npy", sample_positions)
    num_genes = len(df["gene"].cat.categories)
    _check_work_disk(work_dir, len(sample_positions), num_genes, len(df))
    cell_index = build_cell_row_index(cell_codes, work_dir, profile=profile)

    if profile:
        print(f"[OPT-COMPARTMENT] points={len(df):,}, cells={num_cells:,}, genes={num_genes:,}, queries={len(sample_positions):,}", flush=True)
    rnaflux_path = work_dir / "rnaflux.npy"
    _, query_cell_codes = compute_same_cell_rnaflux(df, sample_positions, config.radius, rnaflux_path, cell_index=cell_index, profile=profile, progress_interval_cells=config.progress_interval_cells, workers=config.workers, cell_batch_size=config.cell_batch_size)
    _, normalized_embeddings = fit_global_svd(rnaflux_path, config.random_seed, work_dir / "query_rgb.npy")
    rgb_path = work_dir / "all_point_rgb.npy"
    interpolate_same_cell_rgb(df, sample_positions, query_cell_codes, normalized_embeddings, rgb_path, cell_index=cell_index, profile=profile, progress_interval_cells=config.progress_interval_cells, workers=config.workers, cell_batch_size=config.cell_batch_size)
    labels_path = work_dir / "subdomain_labels.npy"
    best_k, _, quantization_errors = fit_global_som(rgb_path, config, labels_path)
    plot_paths: Dict[str, Dict[str, str]] = {}
    if display_col is None:
        output_paths = write_point_outputs(df, labels_path, result_dir, display_col, config.output_chunk_size)
    else:
        staging_dir = work_dir / "tile_staging"
        staging_paths = write_tile_coordinate_outputs(df, labels_path, rgb_path, staging_dir, display_col, config.output_chunk_size)
        output_paths = write_compatible_tile_csv_outputs(df, staging_paths, result_dir, display_col, config.output_chunk_size, config.output_workers)
        plot_paths = write_compatible_tile_plot_outputs(
            df,
            staging_paths,
            result_dir,
            cell_boundary=bundle.get("cell_boundary"),
            nuclear_boundary=bundle.get("nuclear_boundary"),
            profile=profile,
        )

    metadata = {
        "pkl_path": str(pkl_path),
        "out_dir": str(out_dir),
        "work_dir": str(work_dir),
        "num_points": len(df),
        "num_cells": num_cells,
        "num_genes": num_genes,
        "num_queries": len(sample_positions),
        "best_k": best_k,
        "quantization_errors": quantization_errors,
        "config": asdict(config),
        "same_cell_neighborhood": True,
        "same_cell_interpolation": True,
        "global_svd_som": True,
        "output_paths": output_paths,
        "plot_paths": plot_paths,
        "elapsed_seconds": time.time() - started_at,
    }
    with (out_dir / "metadata.json").open("w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, ensure_ascii=False, indent=2)
    return metadata
