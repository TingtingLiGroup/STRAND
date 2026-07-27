# InSTAnT 稀疏优化版

本目录包含 STRAND colocalization 的稀疏优化后端。公开命令
`subcellfeat-coloc` 已通过 `tools/api/colocalization.py` 默认调用该后端；
同时仍保留 `python -m tools.instant_optimized.cli` 独立入口，方便开发者做对照
和性能测试。

兼容边界：

- `tools/instant_core/` 仍保留原 InSTAnT 实现，仅包含通用 chunksize 内存修复；
- `tools/engines/instant_adapter.py` 保留原版适配路径；
- 优化版不改变 PP、CPB 或 expected colocalization 的统计定义。

## 主要优化

### 1. 转录本按细胞一次性索引

原版处理每个细胞时都会在全部转录本中执行一次 `np.where`。当前真实组
包含 18,283 个有效细胞和 2,074,554 个转录本，相当于约 379 亿次 cell ID
比较。

优化版只稳定排序一次，并通过 cell offsets 直接切片读取每个细胞。

### 2. 只检验细胞内实际存在的基因对

原版每个细胞都遍历全基因面板的上三角矩阵。当前真实组的 469 个基因会
产生约 20.15 亿次 Python pair 检查。

过滤后每个细胞实际基因数的中位数只有 3，真正需要检验的 pair 总数为
471,418。优化版只计算这些 pair，并用精确的
`scipy.stats.binom.sf` 批量计算二项分布右尾概率。

### 3. 避免多进程序列化整个模型

原版向进程池提交绑定方法，可能反复序列化包含 37.19 GiB PP 张量的
`self`，并触发额外的 Copy-on-Write 内存。

优化版使用模块级 worker，任务只传 cell index。坐标、基因编码和 cell
offsets 只在 worker 初始化时注册。

### 4. 精确稀疏存储 PP 结果

原版保存完整的 `cell × gene × gene` float 张量；当前真实组仅该张量约
37.19 GiB。

优化版只保存：

- cell index；
- 上三角 gene-pair ID；
- 精确 PP p-value；
- gene counts。

当前真实组的稀疏 PP 数据约 7.19 MiB。未存储的 pair 按原版语义精确表示
为 `p=1`，不是统计近似。只有显式使用 `--save-pp-pvals` 时，才通过磁盘
memmap 分块生成原版格式的完整 `.npy` 文件。

### 5. CPB 只遍历共同表达细胞

优化版在 PP 阶段流式累计 CPB 所需的充分统计量，并建立
gene-pair →共同表达细胞索引。CPB 不再对每个 gene pair 扫描所有细胞，
但仍使用原版的 `PoiBin` 实现。

### 6. 稀疏 PP 绘图兼容

代表细胞绘图只为当前 gene pair 构造零复制广播视图，不会为绘图重新分配
完整三维 PP 张量。

## 统计定义是否改变

没有改变。优化版保持：

- 相同的 `cKDTree.query_pairs(distance)` 邻近规则；
- 相同的 `n_i * n_j` 二项检验次数定义，包括原版对角线行为；
- 相同的精确二项分布右尾概率；
- 相同的严格 `PP p-value < alpha` 判定；
- 相同的 high/low precision 转换顺序；
- 相同的 CPB 权重、denominator 和 `PoiBin` 实现；
- 相同的 `counts > min_genecount` 细胞筛选规则。

## 已完成测试

测试使用 Python 标准库 `unittest`，不需要安装 `pytest`。

当前自动化测试覆盖：

- high/low precision；
- 2D/3D PP；
- 串行/多进程一致性；
- 同基因对角线、跳过细胞和距离阈值；
- 非零 `min_transcript` CPB；
- 稀疏 PP 到完整 `.npy` 的无损恢复；
- API 保存文件与原版兼容；
- groupby、cell sampling、临时文件清理和 group 名冲突检查。

运行全部测试：

```bash
python -m unittest discover \
  -s tools/instant_optimized/tests \
  -p 'test_*.py'
```

测试结果：

```text
10 passed
```

此外还完成了以下差分验证：

- 小型合成数据：原版与优化版 PP、gene counts、CPB、expected
  colocalization 逐元素完全一致；
- 真实 30 细胞子集：PP、CPB、expected colocalization 逐元素完全一致；
- 完整 `.npy` 输出：shape、dtype、对称性和数值与原版完全一致；
- 原版源码聚合 SHA256 在优化前后保持不变。

## 真实数据性能

测试数据：

```text
Transcripts: 2,074,554
Cells:       22,693
Valid cells: 18,283
Genes:       469
Mode:        2D, high precision, 4 workers
```

优化版完整 PP + CPB 实测：

```text
PP:                 9.07 s
CPB:               20.16 s
Total:             29.23 s
Parent peak RSS:    0.43 GiB
Sparse PP storage:  7.19 MiB
```

时间会受服务器负载影响。相同组的原版任务此前运行超过 36 小时，进程组
PSS 约 187 GiB。

## 跨数据集验证

除 `Dataset22_2` 外，又在 3 个不同平台/组织/规模的已导出
`instant_input.csv` 上做了对照。每个数据集均包含两层验证：

1. **60 细胞随机子集**：原版串行 PP + CPB 与优化版逐元素对比；
2. **全量数据**：仅跑优化版，记录耗时、峰值内存和稀疏 PP 体积。

参数与生产一致：`distance=4`、`min_genecount=20`、`pp_alpha=0.001`、
`2D`、`high precision`、`4 threads`。

这些跨数据集结果来自发布前的内部验证记录。路径相关的批量验证脚本没有随
公开包发布，避免在公共仓库中保留本地数据路径；公开包内保留的是可复用的
优化核心、CLI、API 和自动化等价性测试。

### 结果等价性（60 细胞子集）

| 数据集 | 转录本 | 细胞 | 基因 | PP | CPB | expected | 显著 gene pair |
|--------|--------|------|------|----|-----|----------|----------------|
| Dataset7 Enterocyte | 2,834 | 60 | 39 | ✓ | ✓ | ✓ | 1,521 |
| Dataset17 B cell | 2,624 | 60 | 39 | ✓ | ✓ | ✓ | 1,521 |
| Dataset13 lung malignant | 11,914 | 60 | 114 | ✓ | ✓ | ✓ | 12,996 |
| Dataset22 ovarian malignant | 4,668 | 60 | 62 | ✓ | ✓ | ✓ | 3,844 |

四个数据集 `all_equal=True`，说明优化版在不同基因面板密度和转录本分布下
与原版统计定义一致。

### 全量性能（优化版）

| 数据集 | 转录本 | 有效细胞 | 基因 | 总耗时 | 峰值 RSS | 稀疏 PP | 等价稠密 PP |
|--------|--------|----------|------|--------|----------|---------|------------|
| Dataset7 Enterocyte | 1.50M | 16,647 | 147 | 92 s | 0.72 GiB | 31 MiB | 4.4 GiB |
| Dataset17 B cell | 1.89M | 26,837 | 309 | 122 s | 0.89 GiB | 57 MiB | 32 GiB |
| Dataset13 lung malignant | 7.33M | 28,596 | 480 | 470 s (~8 min) | 1.49 GiB | 140 MiB | 55 GiB |
| Dataset22 ovarian malignant | 2.07M | 18,283 | 469 | 61 s | 1.49 GiB | 48 MiB | 37 GiB |

结论：

- **结果正确**：多数据集、不同基因数（147–480）和转录本规模（1.5M–7.3M）
  下，子集差分全部通过；
- **性能稳定**：全量均在数分钟内完成，峰值内存始终 < 1.5 GiB；
- **内存节省显著**：若按原版方式保存完整 PP 张量，上述四组分别需要约
  4–55 GiB；优化版稀疏存储仅 31–140 MiB；
- **原版对照**：`Dataset22_2` 同组原版此前 >36 h / ~187 GiB PSS，与优化版
  61 s / 1.5 GiB 形成同一数据上的直接对照。

因此可以认为：**优化在统计等价前提下已成功**，且优势随基因面板变大、
转录本变多而更明显（如 Dataset13 的 480 基因组，稠密 PP 约 55 GiB，
优化版 8 分钟内完成）。

## 使用方法

从项目根目录运行，并使用与原版不同的输出目录：

```bash
python -m tools.instant_optimized.cli \
  --pkl Dataset/Dataset22_2_Xenium_HumanOvarianCancer_data_dict.pkl \
  --out-dir Colocalization_optimized/Dataset22_2 \
  --profile \
  --groupby celltype \
  --threads 4 \
  --skip-existing \
  --cellgene-filter-min-transcripts 12 \
  --use-2d \
  --no-plot
```

当前服务器缺少原版网络图所需的 Graphviz `sfdp` 可执行文件，因此示例使用
`--no-plot`。这不是优化版引入的问题；如需网络图，应先安装 Graphviz。

## 性能基准

```bash
python -m tools.instant_optimized.benchmark \
  --cells 1000 \
  --genes 469 \
  --threads 4
```
