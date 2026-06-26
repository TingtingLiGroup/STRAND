# Colocalization Analysis Module

The Colocalization module identifies RNA-RNA colocalization relationships within cells.

Command:

```bash
subcellfeat-coloc
```

---

## 1. Purpose

This module analyzes whether pairs of genes show significant spatial colocalization at the transcript level. It uses InSTAnT-based PP and CPB calculations, followed by filtering, summary generation, and visualization.

---

## 2. Main workflow

```text
PKL bundle
   ↓
Read transcript coordinates and boundaries
   ↓
Standard prefiltering
   ↓
Generate InSTAnT-compatible input
   ↓
Run cell-wise PP test
   ↓
Run global CPB test
   ↓
Filter significant colocalized gene pairs
   ↓
Generate heatmap, network, pair plots, and region annotation
```

---

## 3. Basic usage

```bash
subcellfeat-coloc \
  --pkl data/simulated_data_dict.pkl \
  --out-dir results/simulated_coloc \
  --use-2d \
  --profile
```

For large datasets:

```bash
subcellfeat-coloc \
  --pkl data/simulated_data_dict.pkl \
  --out-dir Colocalization/simulateds \
  --use-2d \
  --threads 8 \
  --sample-cells 3000 \
  --profile
```

---

## 4. Important parameters

### `--distance`

Distance threshold for colocalization.

Default:

```text
4.0
```

### `--pp-alpha`

Cell-wise PP p-value threshold used by CPB.

Default:

```text
0.001
```

### `--cpb-alpha`

Global CPB p-value threshold for significant gene pairs.

Default:

```text
0.0001
```

### `--use-2d`

Run 2D PP test. Recommended when the dataset is effectively 2D or when z coordinates are not meaningful.

### `--threads`

Number of threads or processes used by InSTAnT.

### `--sample-cells` and `--sample-cell-frac`

Reduce memory cost by sampling cells before PP/CPB.

Use only one of them:

```bash
--sample-cells 3000
```

or:

```bash
--sample-cell-frac 0.1
```

### `--groupby`

Run colocalization separately for each group.

Example:

```bash
--groupby celltype
```

The group column can be in `data_df` or `coordinates`.

---

## 5. Output files

Typical outputs include:

```text
instant_significant_pairs.csv
prefilter_summary.csv
viz_cpb_heatmap.png
viz_coloc_network.png
top-pair spatial plots
region annotation files
```

Optional large outputs:

```text
full CPB p-value matrices
full PP p-value tensor
intermediate InSTAnT input tables
QC details
```

These are controlled by:

```bash
--save-matrices
--save-pp-pvals
--save-intermediate
--save-qc-details
```

---

## 6. Visualization controls

```bash
--viz-top-n-genes 80
--viz-max-edges 80
--viz-max-cells 6
--viz-top-pairs 10
```

To skip plots:

```bash
--no-plot
```

---

## 7. Region annotation

By default, significant colocalized pairs can be annotated by subcellular region. Region annotation can be disabled:

```bash
--no-region-annotation
```

The default region threshold mode is:

```text
cell_scaled
```

This scales region thresholds by cell and nucleus size.

---

## 8. Notes

1. PP calculation can be expensive for large datasets.
2. CPB memory cost grows with the number of genes and cells.
3. Cell sampling is recommended for exploratory analysis on very large datasets.
4. Use `--save-pp-pvals` only when necessary because the output can be very large.
