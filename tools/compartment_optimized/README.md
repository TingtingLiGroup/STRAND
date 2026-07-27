# 低内存优化区室探测

该目录是 STRAND 最终优化版中的低内存全局区室后端。公开入口为：

- `subcellfeat-compartment-optimized`
- `python -m tools.compartment_optimized.cli`

主入口 `subcellfeat-compartment` 也已经同步 same-cell 默认语义、`radius=auto`
和自动聚类范围 `4..7`，用于保持公开 CLI 与最终优化口径一致。

## 计算语义

1. 查询点保持原来的 `DataFrame.sample(frac, random_state)` 抽样顺序。
2. 半径邻域只统计查询点所属细胞的转录本。
3. 局部组成减去所属细胞整体组成，再除以跨细胞基因比例标准差。
4. 全部细胞的 RNAflux 共同拟合一个三维 `TruncatedSVD`。
5. RGB 只在所属细胞内部插值，不跨细胞边界传播。
6. 全部转录本共同训练一个 SOM，随后按 `tile` 拆分逐点结果。

## 内存优化

- 细胞整体标准差使用 Welford 流式算法，不创建“细胞 × 基因”稠密矩阵。
- 同一细胞全部查询点通过 `cKDTree.sparse_distance_matrix` 批量计数。
- 交错cell行号使用Numba稳定计数分组，不再创建3.67亿行int64 argsort临时数组。
- 高维 RNAflux、全量 RGB 和每点标签使用 `.npy` 磁盘映射。
- 插值逐细胞完成并立即释放临时数组。
- SOM量化误差和winner按块计算。
- CSV只扫描全表一次，并按tile追加写出。
- 兼容CSV使用PyArrow C++流式写出，避免Pandas逐值字符串格式化瓶颈。

## 小型测试

```bash
python -m unittest discover \
  -s tools/compartment_optimized/tests \
  -p 'test_*.py' -v
```

## Dataset4命令模板

正式运行前必须先通过小型测试。输入转录本允许像Dataset4一样按基因交错排列；程序会建立一次稳定的磁盘型cell行号索引。

```bash
python -m tools.compartment_optimized.cli \
  --pkl Dataset/Dataset4_merscope_MouseLiver_data_dict.pkl \
  --out-dir Compartment_Optimized/Dataset4_global \
  --work-dir /tmp/Dataset4_compartment_work \
  --group-col batch \
  --display-col tile \
  --frac 0.01 \
  --radius 5 \
  --n-clusters auto \
  --cluster-range-min 4 \
  --cluster-range-max 7 \
  --som-chunk-size 1000000 \
  --output-chunk-size 250000 \
  --random-seed 42 \
  --workers 16 \
  --cell-batch-size 256 \
  --som-workers 4 \
  --output-workers 4 \
  --profile
```

## 输出

- `work/rnaflux.npy`：查询点高维RNAflux磁盘映射。
- `work/cell_row_order.npy`：按cell稳定排列的原始行号索引，Dataset4约1.37 GiB。
- `work/cell_row_offsets.npy`：每个cell在行号索引中的起止位置。
- `work/query_rgb.npy`：查询点三维归一化embedding。
- `work/all_point_rgb.npy`：全部转录本的细胞内插值RGB。
- `work/subdomain_labels.npy`：全部转录本的全局SOM标签。
- `results/<tile>_sampled_subdomain_points.csv`：兼容旧模块的逐点记录，包含原始字段、全局 `subdomain` 和全局RNAflux RGB。
- `results/<tile>_sampled_rnaflux_embedding.png`：兼容旧模块的 RNAflux RGB 空间散点图。
- `results/<tile>_sampled_subdomains.png`：兼容旧模块的 SOM subdomain 空间图，并叠加可用的细胞和细胞核边界。
- `metadata.json`：参数、规模、最佳聚类数、耗时和输出路径。

注意：PKL本身无法流式读取，因此加载 `data_df` 仍是任务的最低内存成本。首次建立cell索引时，`np.argsort` 会临时使用约“行数 × 8字节”的RAM；Dataset4约2.74 GiB，随后立即压缩为uint32磁盘映射并释放。
