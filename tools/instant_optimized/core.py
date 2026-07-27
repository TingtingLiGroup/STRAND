from __future__ import annotations

import multiprocessing as mp
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.signal import fftconvolve
from scipy.stats import binom

from tools.instant_core.poibin import PoiBin


_pp_coordinates: np.ndarray | None = None
_pp_gene_codes: np.ndarray | None = None
_pp_cell_offsets: np.ndarray | None = None
_pp_num_genes = 0
_pp_distance_threshold = 0.0
_pp_min_genecount = 0
_pp_dtype = np.dtype(np.float64)

_cpb_sorted_cell_indices: np.ndarray | None = None
_cpb_pair_offsets: np.ndarray | None = None
_cpb_active_pair_ids: np.ndarray | None = None
_cpb_gene_pair_weight: np.ndarray | None = None
_cpb_denom_per_cell: np.ndarray | None = None
_cpb_n_edges_per_cell: np.ndarray | None = None
_cpb_obs: np.ndarray | None = None
_cpb_present: np.ndarray | None = None
_cpb_num_genes = 0


def _poisson_binomial_right_tail_fft(
    probabilities: np.ndarray,
    observed: int,
    *,
    block_size: int = 256,
) -> float:
    """使用分块 FFT 卷积计算 Poisson-binomial 右尾概率。

    参数
    ----------
    probabilities
        每个独立 Bernoulli trial 的成功概率。
    observed
        需要计算的成功次数下界，返回 ``P(X >= observed)``。
    block_size
        每个局部 PMF 的最大 trial 数。较小值降低局部卷积的平方级开销，
        较大值减少中间数组数量；256 是当前大规模空间转录组数据的折中值。

    返回
    -------
    float
        Poisson-binomial 分布的右尾概率。该实现不改变统计定义，只替换
        原始 PoiBin 在大 trial 数下产生平方级中间矩阵的数值计算路径。
    """
    probabilities = np.asarray(probabilities, dtype=np.float64)
    n_trials = len(probabilities)
    if observed <= 0:
        return 1.0
    if observed > n_trials or n_trials == 0:
        return 0.0

    # 第一步：在小块内用一维递推构建局部 PMF，内存只随 block_size 增长。
    block_pmfs = []
    for start in range(0, n_trials, int(block_size)):
        block_pmf = np.array([1.0], dtype=np.float64)
        for probability in probabilities[start:start + int(block_size)]:
            block_pmf = np.convolve(
                block_pmf,
                np.array([1.0 - probability, probability]),
            )
        block_pmfs.append(block_pmf)

    # 第二步：两两合并局部 PMF，避免构造 n_trials x n_trials 的矩阵。
    while len(block_pmfs) > 1:
        merged_pmfs = []
        for index in range(0, len(block_pmfs), 2):
            if index + 1 == len(block_pmfs):
                merged_pmfs.append(block_pmfs[index])
                continue

            merged_pmf = fftconvolve(
                block_pmfs[index],
                block_pmfs[index + 1],
            )
            # FFT 舍入可能产生极小负数；截断后重新归一化，避免尾概率越界。
            merged_pmf[np.abs(merged_pmf) < 1e-15] = 0.0
            merged_pmf = np.clip(merged_pmf, 0.0, None)
            total_probability = merged_pmf.sum()
            if total_probability > 0.0:
                merged_pmf /= total_probability
            merged_pmfs.append(merged_pmf)
        block_pmfs = merged_pmfs

    # 第三步：直接累加右尾，避免用 1-CDF 在极小 p 值处发生消减误差。
    return float(np.clip(block_pmfs[0][int(observed):].sum(), 0.0, 1.0))


@dataclass(frozen=True)
class SparsePPResults:
    """Exact PP values with implicit p=1 for absent gene pairs."""

    genes: tuple[str, ...]
    cell_ids: np.ndarray
    gene_counts: np.ndarray
    cell_indices: np.ndarray
    pair_ids: np.ndarray
    pvalues: np.ndarray
    precision_mode: str
    elapsed_seconds: float

    @property
    def num_cells(self) -> int:
        return len(self.cell_ids)

    @property
    def num_genes(self) -> int:
        return len(self.genes)

    @property
    def dtype(self) -> np.dtype:
        return np.dtype(np.float64 if self.precision_mode == "high" else np.float16)

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.num_cells, self.num_genes, self.num_genes)

    def __getitem__(self, key):
        """Support the common legacy access pattern pp[:, gene_i, gene_j]."""
        if not isinstance(key, tuple) or len(key) != 3:
            raise TypeError(
                "Sparse PP values support three-dimensional indexing only."
            )
        cell_key, gene_1, gene_2 = key
        if not isinstance(gene_1, (int, np.integer)) or not isinstance(
            gene_2,
            (int, np.integer),
        ):
            raise TypeError("Gene indices must be integers.")
        return self.values_for_pair(int(gene_1), int(gene_2))[cell_key]

    def pair_id(self, gene_1: str | int, gene_2: str | int) -> int:
        if isinstance(gene_1, str):
            try:
                i = self.genes.index(gene_1)
            except ValueError as exc:
                raise KeyError(f"Unknown gene: {gene_1}") from exc
        else:
            i = int(gene_1)

        if isinstance(gene_2, str):
            try:
                j = self.genes.index(gene_2)
            except ValueError as exc:
                raise KeyError(f"Unknown gene: {gene_2}") from exc
        else:
            j = int(gene_2)

        if i > j:
            i, j = j, i
        return i * self.num_genes + j

    def values_for_pair(self, gene_1: str | int, gene_2: str | int) -> np.ndarray:
        """Return cell-ordered PP values for one pair without dense expansion."""
        pair_id = self.pair_id(gene_1, gene_2)
        values = np.ones(self.num_cells, dtype=self.dtype)
        matching = self.pair_ids == pair_id
        values[self.cell_indices[matching]] = self.pvalues[matching]
        return values

    def broadcast_values_for_pair(
        self,
        gene_1: str | int,
        gene_2: str | int,
    ) -> np.ndarray:
        """Provide a zero-copy 3D view for the legacy single-pair plotter."""
        values = self.values_for_pair(gene_1, gene_2)
        return np.broadcast_to(
            values[:, np.newaxis, np.newaxis],
            (self.num_cells, self.num_genes, self.num_genes),
        )

    def to_dense(self) -> np.ndarray:
        """Materialize the legacy dense PP tensor; intended for small tests."""
        dense = np.ones(
            (self.num_cells, self.num_genes, self.num_genes),
            dtype=self.dtype,
        )
        if len(self.pair_ids) == 0:
            return dense

        rows = self.pair_ids // self.num_genes
        cols = self.pair_ids % self.num_genes
        dense[self.cell_indices, rows, cols] = self.pvalues
        dense[self.cell_indices, cols, rows] = self.pvalues
        return dense

    def save_dense_npy(self, path: str | Path, *, cell_chunk_size: int = 128) -> Path:
        """Write the legacy dense tensor as a file-backed NPY without large RAM use."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        dense = np.lib.format.open_memmap(
            path,
            mode="w+",
            dtype=self.dtype,
            shape=(self.num_cells, self.num_genes, self.num_genes),
        )
        for start in range(0, self.num_cells, int(cell_chunk_size)):
            stop = min(start + int(cell_chunk_size), self.num_cells)
            dense[start:stop] = 1

        if len(self.pair_ids):
            rows = self.pair_ids // self.num_genes
            cols = self.pair_ids % self.num_genes
            dense[self.cell_indices, rows, cols] = self.pvalues
            dense[self.cell_indices, cols, rows] = self.pvalues

        dense.flush()
        del dense
        return path


@dataclass(frozen=True)
class CPBResults:
    cpb_pvalues: np.ndarray
    expected_colocalization: np.ndarray
    observed_significant_cells: np.ndarray
    n_edges_per_cell: np.ndarray
    elapsed_seconds: float


def _prepare_arrays(
    instant_input_df: pd.DataFrame,
    *,
    use_3d: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[str, ...]]:
    required = {"gene", "uID", "absX", "absY"}
    if use_3d:
        required.add("absZ")
    missing = required - set(instant_input_df.columns)
    if missing:
        raise ValueError(f"InSTAnT input is missing columns: {sorted(missing)}")

    genes_raw = instant_input_df["gene"].astype(str).to_numpy()
    cells_raw = instant_input_df["uID"].astype(str).to_numpy()
    genes = tuple(str(g) for g in np.unique(genes_raw))
    cell_ids = np.unique(cells_raw)

    gene_codes = np.searchsorted(np.asarray(genes, dtype=object), genes_raw).astype(np.int32)
    cell_codes = np.searchsorted(cell_ids, cells_raw).astype(np.int32)
    coordinate_columns = ["absX", "absY", "absZ"] if use_3d else ["absX", "absY"]
    coordinates = instant_input_df[coordinate_columns].to_numpy(dtype=np.float64)

    order = np.argsort(cell_codes, kind="stable")
    sorted_cell_codes = cell_codes[order]
    cell_counts = np.bincount(sorted_cell_codes, minlength=len(cell_ids))
    cell_offsets = np.empty(len(cell_ids) + 1, dtype=np.int64)
    cell_offsets[0] = 0
    np.cumsum(cell_counts, out=cell_offsets[1:])

    return (
        np.ascontiguousarray(coordinates[order]),
        np.ascontiguousarray(gene_codes[order]),
        cell_offsets,
        cell_ids,
        genes,
    )


def _initialize_pp_worker(
    coordinates: np.ndarray,
    gene_codes: np.ndarray,
    cell_offsets: np.ndarray,
    num_genes: int,
    distance_threshold: float,
    min_genecount: int,
    dtype_string: str,
) -> None:
    global _pp_coordinates, _pp_gene_codes, _pp_cell_offsets
    global _pp_num_genes, _pp_distance_threshold, _pp_min_genecount, _pp_dtype
    _pp_coordinates = coordinates
    _pp_gene_codes = gene_codes
    _pp_cell_offsets = cell_offsets
    _pp_num_genes = int(num_genes)
    _pp_distance_threshold = float(distance_threshold)
    _pp_min_genecount = int(min_genecount)
    _pp_dtype = np.dtype(dtype_string)


def _calculate_cell_pp(cell_index: int) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    if _pp_coordinates is None or _pp_gene_codes is None or _pp_cell_offsets is None:
        raise RuntimeError("PP worker was not initialized.")

    start = int(_pp_cell_offsets[cell_index])
    stop = int(_pp_cell_offsets[cell_index + 1])
    num_transcripts = stop - start

    if num_transcripts <= _pp_min_genecount:
        return (
            int(cell_index),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=_pp_dtype),
            np.zeros(_pp_num_genes, dtype=np.int32),
        )

    coordinates = _pp_coordinates[start:stop]
    gene_codes = _pp_gene_codes[start:stop]
    gene_counts = np.bincount(gene_codes, minlength=_pp_num_genes).astype(np.int32)
    active_genes = np.flatnonzero(gene_counts)

    local_rows, local_cols = np.triu_indices(len(active_genes))
    gene_rows = active_genes[local_rows]
    gene_cols = active_genes[local_cols]
    pair_ids = (gene_rows * _pp_num_genes + gene_cols).astype(np.int32)
    num_trials = (
        gene_counts[gene_rows].astype(np.int64)
        * gene_counts[gene_cols].astype(np.int64)
    )

    neighbor_pairs = cKDTree(coordinates).query_pairs(
        _pp_distance_threshold,
        output_type="ndarray",
    )
    observed = np.zeros(len(pair_ids), dtype=np.int64)

    if len(neighbor_pairs):
        first_genes = gene_codes[neighbor_pairs[:, 0]]
        second_genes = gene_codes[neighbor_pairs[:, 1]]
        near_rows = np.minimum(first_genes, second_genes)
        near_cols = np.maximum(first_genes, second_genes)
        near_pair_ids = near_rows * _pp_num_genes + near_cols
        observed_pair_ids, observed_counts = np.unique(
            near_pair_ids,
            return_counts=True,
        )
        observed_positions = np.searchsorted(pair_ids, observed_pair_ids)
        observed[observed_positions] = observed_counts
        null_probability = (
            len(neighbor_pairs) * 2
            / (num_transcripts * (num_transcripts - 1))
        )
    else:
        null_probability = 1.0

    pvalues = binom.sf(observed - 1, num_trials, null_probability).astype(
        _pp_dtype,
        copy=False,
    )
    return int(cell_index), pair_ids, pvalues, gene_counts


def run_proximal_pairs(
    instant_input_df: pd.DataFrame,
    *,
    distance_threshold: float = 4.0,
    min_genecount: int = 20,
    threads: int = 1,
    use_3d: bool = False,
    precision_mode: str = "high",
    profile: bool = False,
) -> SparsePPResults:
    """Run the original PP statistic with sparse exact storage.

    Parameters
    ----------
    instant_input_df
        Transcript table containing gene, cell ID and 2D/3D coordinates.
    distance_threshold
        Maximum transcript distance counted as proximal.
    min_genecount
        A cell is analyzed only when its total transcript count is strictly
        greater than this value, matching the original implementation.
    threads
        Number of worker processes. One runs serially.
    use_3d
        Include ``absZ`` in KDTree distances when true.
    precision_mode
        ``high`` stores float64 PP values; ``low`` stores float16 values.
    profile
        Print elapsed time and storage statistics.

    Returns
    -------
    SparsePPResults
        Exact cell-wise PP values and gene counts. Unstored pairs have p=1.

    Raises
    ------
    ValueError
        If precision mode or required input columns are invalid.
    """
    if precision_mode not in {"high", "low"}:
        raise ValueError("precision_mode must be 'high' or 'low'.")
    if threads is None or int(threads) < 1:
        threads = 1

    started = time.perf_counter()
    coordinates, gene_codes, cell_offsets, cell_ids, genes = _prepare_arrays(
        instant_input_df,
        use_3d=use_3d,
    )
    dtype = np.dtype(np.float64 if precision_mode == "high" else np.float16)
    num_cells = len(cell_ids)
    num_genes = len(genes)
    gene_counts = np.zeros((num_cells, num_genes), dtype=np.int32)
    pair_ids_by_cell: list[np.ndarray | None] = [None] * num_cells
    pvalues_by_cell: list[np.ndarray | None] = [None] * num_cells

    initializer_args = (
        coordinates,
        gene_codes,
        cell_offsets,
        num_genes,
        float(distance_threshold),
        int(min_genecount),
        dtype.str,
    )

    def store_result(result: tuple[int, np.ndarray, np.ndarray, np.ndarray]) -> None:
        cell_index, pair_ids, pvalues, cell_gene_counts = result
        pair_ids_by_cell[cell_index] = pair_ids
        pvalues_by_cell[cell_index] = pvalues
        gene_counts[cell_index] = cell_gene_counts

    if int(threads) == 1:
        _initialize_pp_worker(*initializer_args)
        for cell_index in range(num_cells):
            store_result(_calculate_cell_pp(cell_index))
    else:
        chunksize = max(1, min(64, num_cells // max(int(threads) * 16, 1)))
        with mp.Pool(
            processes=int(threads),
            initializer=_initialize_pp_worker,
            initargs=initializer_args,
        ) as pool:
            for result in pool.imap_unordered(
                _calculate_cell_pp,
                range(num_cells),
                chunksize=chunksize,
            ):
                store_result(result)

    lengths = np.fromiter(
        (len(values) if values is not None else 0 for values in pair_ids_by_cell),
        dtype=np.int64,
        count=num_cells,
    )
    cell_indices = np.repeat(np.arange(num_cells, dtype=np.int32), lengths)
    nonempty_pair_ids = [values for values in pair_ids_by_cell if values is not None and len(values)]
    nonempty_pvalues = [values for values in pvalues_by_cell if values is not None and len(values)]
    all_pair_ids = (
        np.concatenate(nonempty_pair_ids).astype(np.int32, copy=False)
        if nonempty_pair_ids
        else np.empty(0, dtype=np.int32)
    )
    all_pvalues = (
        np.concatenate(nonempty_pvalues).astype(dtype, copy=False)
        if nonempty_pvalues
        else np.empty(0, dtype=dtype)
    )
    elapsed = time.perf_counter() - started

    if profile:
        valid_cells = int(np.count_nonzero(np.diff(cell_offsets) > int(min_genecount)))
        dense_gib = num_cells * num_genes * num_genes * dtype.itemsize / 1024**3
        sparse_mib = (
            all_pair_ids.nbytes + all_pvalues.nbytes + cell_indices.nbytes
        ) / 1024**2
        print(
            "[instant-optimized] PP: "
            f"cells={num_cells}, valid_cells={valid_cells}, genes={num_genes}, "
            f"active_pairs={len(all_pair_ids)}, elapsed={elapsed:.2f}s"
        )
        print(
            "[instant-optimized] PP storage: "
            f"sparse={sparse_mib:.2f} MiB vs dense={dense_gib:.2f} GiB"
        )

    return SparsePPResults(
        genes=genes,
        cell_ids=cell_ids,
        gene_counts=gene_counts,
        cell_indices=cell_indices,
        pair_ids=all_pair_ids,
        pvalues=all_pvalues,
        precision_mode=precision_mode,
        elapsed_seconds=elapsed,
    )


def _initialize_cpb_worker(
    sorted_cell_indices: np.ndarray,
    pair_offsets: np.ndarray,
    active_pair_ids: np.ndarray,
    gene_pair_weight: np.ndarray,
    denom_per_cell: np.ndarray,
    n_edges_per_cell: np.ndarray,
    obs: np.ndarray,
    present: np.ndarray,
    num_genes: int,
) -> None:
    global _cpb_sorted_cell_indices, _cpb_pair_offsets, _cpb_active_pair_ids
    global _cpb_gene_pair_weight, _cpb_denom_per_cell, _cpb_n_edges_per_cell
    global _cpb_obs, _cpb_present, _cpb_num_genes
    _cpb_sorted_cell_indices = sorted_cell_indices
    _cpb_pair_offsets = pair_offsets
    _cpb_active_pair_ids = active_pair_ids
    _cpb_gene_pair_weight = gene_pair_weight
    _cpb_denom_per_cell = denom_per_cell
    _cpb_n_edges_per_cell = n_edges_per_cell
    _cpb_obs = obs
    _cpb_present = present
    _cpb_num_genes = int(num_genes)


def _calculate_pair_cpb(active_pair_position: int) -> tuple[int, float, float]:
    if (
        _cpb_sorted_cell_indices is None
        or _cpb_pair_offsets is None
        or _cpb_active_pair_ids is None
        or _cpb_gene_pair_weight is None
        or _cpb_denom_per_cell is None
        or _cpb_n_edges_per_cell is None
        or _cpb_obs is None
        or _cpb_present is None
    ):
        raise RuntimeError("CPB worker was not initialized.")

    pair_id = int(_cpb_active_pair_ids[active_pair_position])
    start = int(_cpb_pair_offsets[active_pair_position])
    stop = int(_cpb_pair_offsets[active_pair_position + 1])
    cells = _cpb_sorted_cell_indices[start:stop]
    i, j = divmod(pair_id, _cpb_num_genes)
    cells = cells[_cpb_present[cells, i] & _cpb_present[cells, j]]
    weight = _cpb_gene_pair_weight[i, j] / _cpb_denom_per_cell[cells]
    probabilities = 1 - np.power(
        1 - weight,
        _cpb_n_edges_per_cell[cells],
    )
    observed = int(_cpb_obs[i, j])
    # trial 数较小时保留原实现，确保既有原版逐元素等价测试不变；
    # 大规模 pair 改用线性内存的分块 FFT，避免 PoiBin 的平方级矩阵。
    if len(probabilities) <= 4096:
        pvalue = float(PoiBin(probabilities).pval(observed))
    else:
        pvalue = _poisson_binomial_right_tail_fft(probabilities, observed)
    return pair_id, pvalue, float(np.sum(probabilities))


def run_conditional_global_colocalization(
    pp_results: SparsePPResults,
    *,
    alpha_cellwise: float = 0.001,
    min_transcript: int = 0,
    threads: int = 1,
    precision_mode: str = "high",
    profile: bool = False,
) -> CPBResults:
    """Run the original CPB statistic from sparse PP sufficient statistics.

    Parameters
    ----------
    pp_results
        Exact output from :func:`run_proximal_pairs`.
    alpha_cellwise
        Strict PP p-value threshold used to define a significant cell edge.
    min_transcript
        Both genes must have counts strictly above this value in a cell before
        that cell contributes a Bernoulli probability for the pair.
    threads
        Number of worker processes. One runs serially.
    precision_mode
        Selects the same CPB denominator floor as the original implementation.
    profile
        Print elapsed time and active-pair statistics.

    Returns
    -------
    CPBResults
        Symmetric CPB p-values, expected counts and observed edge summaries.

    Raises
    ------
    ValueError
        If ``precision_mode`` is not ``high`` or ``low``.
    """
    if precision_mode not in {"high", "low"}:
        raise ValueError("precision_mode must be 'high' or 'low'.")
    if threads is None or int(threads) < 1:
        threads = 1

    started = time.perf_counter()
    num_cells = pp_results.num_cells
    num_genes = pp_results.num_genes

    # 第一步：提前报告 CPB 输入规模，让大型任务在数组准备期间也不是黑盒状态。
    if profile:
        print(
            "[instant-optimized] CPB prepare: "
            f"cells={num_cells}, genes={num_genes}, "
            f"stored_pp_pairs={len(pp_results.pair_ids)}",
            flush=True,
        )

    significant = pp_results.pvalues < float(alpha_cellwise)
    significant_pair_ids = pp_results.pair_ids[significant]
    significant_cells = pp_results.cell_indices[significant]

    upper_obs = np.bincount(
        significant_pair_ids,
        minlength=num_genes * num_genes,
    ).reshape(num_genes, num_genes)
    observed = upper_obs + upper_obs.T - np.diag(np.diag(upper_obs))
    observed = observed.astype(np.int64, copy=False)
    n_edges_per_cell = np.bincount(
        significant_cells,
        minlength=num_cells,
    ).astype(np.float64)

    global_gene_count = np.zeros(num_genes, dtype=np.float64)
    if len(significant_pair_ids):
        rows = significant_pair_ids // num_genes
        cols = significant_pair_ids % num_genes
        np.add.at(global_gene_count, rows, 1)
        off_diagonal = rows != cols
        np.add.at(global_gene_count, cols[off_diagonal], 1)

    gene_pair_weight = np.outer(global_gene_count, global_gene_count)
    row_upper_weight = np.fromiter(
        (gene_pair_weight[i, i:].sum() for i in range(num_genes)),
        dtype=np.float64,
        count=num_genes,
    )
    present = pp_results.gene_counts > int(min_transcript)
    p_mode = 1e-64 if precision_mode == "high" else 1e-16
    denom_per_cell = present.astype(np.float64) @ row_upper_weight + p_mode

    pair_order = np.argsort(pp_results.pair_ids, kind="stable")
    sorted_pair_ids = pp_results.pair_ids[pair_order]
    sorted_cell_indices = pp_results.cell_indices[pair_order]
    active_pair_ids, pair_counts = np.unique(sorted_pair_ids, return_counts=True)
    pair_offsets = np.empty(len(active_pair_ids) + 1, dtype=np.int64)
    pair_offsets[0] = 0
    np.cumsum(pair_counts, out=pair_offsets[1:])

    # 第二步：索引构建完成后报告真正需要计算的基因对数量和准备耗时。
    if profile:
        preparation_elapsed = time.perf_counter() - started
        print(
            "[instant-optimized] CPB index ready: "
            f"active_gene_pairs={len(active_pair_ids)}, "
            f"preparation_elapsed={preparation_elapsed:.2f}s",
            flush=True,
        )

    cpb_pvalues = np.ones((num_genes, num_genes), dtype=np.float64)
    expected_colocalization = np.zeros((num_genes, num_genes), dtype=np.float64)
    initializer_args = (
        sorted_cell_indices,
        pair_offsets,
        active_pair_ids,
        gene_pair_weight,
        denom_per_cell,
        n_edges_per_cell,
        observed,
        present,
        num_genes,
    )

    def store_result(result: tuple[int, float, float]) -> None:
        pair_id, pvalue, expected = result
        i, j = divmod(int(pair_id), num_genes)
        cpb_pvalues[i, j] = pvalue
        cpb_pvalues[j, i] = pvalue
        expected_colocalization[i, j] = expected
        expected_colocalization[j, i] = expected

    # 第三步：统一维护串行和并行计算的实时进度、速度与剩余时间估计。
    total_pairs = len(active_pair_ids)
    completed_pairs = 0
    calculation_started = time.perf_counter()
    last_progress_time = calculation_started
    progress_pair_interval = max(1, total_pairs // 100)

    def report_progress(*, force: bool = False) -> None:
        """在开启 profile 时按时间或完成比例输出 CPB 实时进度。"""
        nonlocal last_progress_time
        if not profile or total_pairs == 0:
            return

        current_time = time.perf_counter()
        reached_pair_interval = completed_pairs % progress_pair_interval == 0
        reached_time_interval = current_time - last_progress_time >= 30.0
        if not force and not reached_pair_interval and not reached_time_interval:
            return

        elapsed = max(current_time - calculation_started, 1e-12)
        speed = completed_pairs / elapsed
        remaining_pairs = max(total_pairs - completed_pairs, 0)
        eta_seconds = remaining_pairs / speed if speed > 0 else float("inf")
        progress_percent = completed_pairs / total_pairs * 100.0
        print(
            "[instant-optimized] CPB progress: "
            f"{completed_pairs}/{total_pairs} pairs "
            f"({progress_percent:.1f}%), elapsed={elapsed / 60:.1f} min, "
            f"speed={speed:.2f} pairs/s, ETA={eta_seconds / 60:.1f} min",
            flush=True,
        )
        last_progress_time = current_time

    if int(threads) == 1:
        _initialize_cpb_worker(*initializer_args)
        for active_pair_position in range(len(active_pair_ids)):
            store_result(_calculate_pair_cpb(active_pair_position))
            completed_pairs += 1
            report_progress(force=completed_pairs == total_pairs)
    else:
        chunksize = max(
            1,
            min(512, len(active_pair_ids) // max(int(threads) * 32, 1)),
        )
        # 开启 profile 时使用较小任务块，使主进程更快收到结果并刷新进度。
        if profile:
            chunksize = min(chunksize, 8)
        with mp.Pool(
            processes=int(threads),
            initializer=_initialize_cpb_worker,
            initargs=initializer_args,
        ) as pool:
            for result in pool.imap_unordered(
                _calculate_pair_cpb,
                range(len(active_pair_ids)),
                chunksize=chunksize,
            ):
                store_result(result)
                completed_pairs += 1
                report_progress(force=completed_pairs == total_pairs)

    elapsed = time.perf_counter() - started
    if profile:
        print(
            "[instant-optimized] CPB: "
            f"active_gene_pairs={len(active_pair_ids)}, elapsed={elapsed:.2f}s"
        )

    return CPBResults(
        cpb_pvalues=cpb_pvalues,
        expected_colocalization=expected_colocalization,
        observed_significant_cells=observed,
        n_edges_per_cell=n_edges_per_cell,
        elapsed_seconds=elapsed,
    )
