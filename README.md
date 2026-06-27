# STRAND Tools

STRAND (Subcellular Transcript RNA Architecture and Navigation Database) Tools is a command-line toolbox for subcellular spatial transcriptomics analysis. It provides four modules for standardized PKL input data:

1. RNA localization pattern classification (`subcellfeat`)
2. Pattern prediction from pre-computed features (`subcellfeat-pattern`)
3. Subcellular compartment detection (`subcellfeat-compartment`)
4. RNA colocalization analysis (`subcellfeat-coloc`)

The toolbox is designed for molecule-resolved spatial transcriptomics datasets such as MERFISH, Xenium, CosMx, seqFISH, and other transcript-level spatial data.

---

## 1. Main Functions

### 1.1 Pattern Classification

Command:

```bash
subcellfeat
```

This module predicts RNA localization patterns for each cell-gene pair. It computes Bento and SPRAWL spatial features, then applies a trained XGBoost classifier.

The final output column is:

```text
pattern
```

Supported pattern classes:

```text
Nuclear
Nuclear edge
Cytoplasmic
Cell edge
Protrusion
Radial
Random
Foci
```

The default prediction strategy is:

```text
Step 1: Use the primary 8-class XGBoost model.
Step 2: Calculate the proportion of predictions classified as Foci.
Step 3: If Foci ratio > 0.5, rerun prediction with the 7-class no-Foci model.
Step 4: Save the final result with the unified column name pattern.
```

Default primary model:

```text
models/multiclass_xgb_8class_prop075_final_from_cv.joblib
```

Default fallback model:

```text
models/multiclass_xgb_7class_no_foci_final_from_cv.joblib
```

---

### 1.2 Compartment Detection

Command:

```bash
subcellfeat-compartment
```

This module detects subcellular transcriptomic compartments using RNA spatial distribution and embedding-based analysis. It supports sampled mode for large datasets.

Typical outputs include:

```text
sampled_batches.csv
sampled_batches_meta.json
fluxmap plot
embedding plot
subdomain / compartment visualization
```

---

### 1.3 Colocalization Analysis

Command:

```bash
subcellfeat-coloc
```

This module identifies significant RNA-RNA colocalization relationships within cells. It includes preprocessing, transcript filtering, pairwise colocalization scoring, significance filtering, and visualization.

Typical outputs include:

```text
instant_significant_pairs.csv
viz_cpb_heatmap.png
viz_coloc_network.png
prefilter_summary.csv
top pair cell visualizations
```

---

## 2. Installation

**Important:** This repository uses [Git LFS](https://git-lfs.com/) for model and data files. After cloning, you must run:

```bash
git lfs pull
```

### 2.1 Recommended Conda/Mamba Installation

```bash
conda env create -f environment.yml
conda activate subcellular
pip install -e .
```

### 2.2 Pip Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

### 2.3 GPU Acceleration (Optional)

The compartment detection module supports GPU acceleration via PyTorch. To enable it:

```bash
pip install -e ".[gpu]"
```

Without PyTorch, compartment detection runs on CPU with NumPy (functionally identical, slower on large datasets).

After installation, check whether the command-line tools are available:

```bash
subcellfeat --help
subcellfeat-pattern --help
subcellfeat-compartment --help
subcellfeat-coloc --help
```

---

## 3. Input Data Format

The toolbox uses a standardized PKL bundle as input.

The PKL file should contain a Python dictionary:

```python
{
    "data_df": pandas.DataFrame,
    "cell_boundary": dict,
    "nuclear_boundary": dict,
    "coordinates": pandas.DataFrame  # optional
}
```

### 3.1 data_df

Required columns:

```text
cell
gene
x
y
```

Optional columns:

```text
z
fov
batch
sample_id
umi
annotation
```

Each row represents one transcript molecule.

Example:

```text
cell    gene    x        y        z      fov
cell_1  ACTB    123.5    456.2    0      FOV1
cell_1  MALAT1  130.1    440.8    0      FOV1
```

### 3.2 cell_boundary

The cell boundary should be a dictionary:

```python
cell_boundary[cell_id] = pandas.DataFrame({
    "x": [...],
    "y": [...]
})
```

Each key is a cell ID. Each value is the polygon boundary of that cell.

### 3.3 nuclear_boundary

The nuclear boundary should follow the same format:

```python
nuclear_boundary[cell_id] = pandas.DataFrame({
    "x": [...],
    "y": [...]
})
```

If nuclear boundary is unavailable, it can be empty or missing. Some nucleus-related features may be affected, but the pipeline should remain compatible.

---

## 4. Quick Start

### 4.1 Pattern Classification

```bash
subcellfeat \
  --pkl data/simulated_data_dict.pkl \
  --out results/simulated_pattern.parquet \
  --profile
```

For large datasets, SPRAWL punctate and radial parameters can be reduced:

```bash
subcellfeat \
  --pkl data/simulated_data_dict.pkl \
  --out Pattern/simulated_pattern.parquet \
  --profile \
  --sprawl-iterations 20 \
  --sprawl-pairs 2
```

### 4.2 Compartment Detection

```bash
subcellfeat-compartment \
  --pkl data/simulated_data_dict.pkl \
  --out-prefix results/simulated_compartment \
  --frac 0.01 \
  --radius 40 \
  --n-clusters auto \
  --profile
```

### 4.3 Colocalization Analysis

```bash
subcellfeat-coloc \
  --pkl data/simulated_data_dict.pkl \
  --out-dir results/simulated_coloc \
  --use-2d \
  --profile
```

---

## 5. Pattern Classification Details

The Pattern module computes 17 features for each cell-gene pair.

### 5.1 Bento Features

```text
bento_cell_inner_proximity
bento_nucleus_inner_proximity
bento_nucleus_outer_proximity
bento_cell_inner_asymmetry
bento_nucleus_inner_asymmetry
bento_nucleus_outer_asymmetry
bento_point_dispersion_norm
bento_nucleus_dispersion_norm
bento_l_max
bento_l_max_gradient
bento_l_min_gradient
bento_l_monotony
bento_l_half_radius
```

### 5.2 SPRAWL Features

```text
sprawl_peripheral
sprawl_central
sprawl_punctate
sprawl_radial
```

The two slowest SPRAWL features are:

```text
sprawl_punctate
sprawl_radial
```

Their approximation parameters can be controlled by:

```bash
--sprawl-iterations
--sprawl-pairs
```

Example:

```bash
--sprawl-iterations 20 --sprawl-pairs 2
```

This reduces runtime but may introduce more randomness in SPRAWL punctate and radial scores.

---

## 6. Output Files

### 6.1 Pattern Output

The main output is a parquet file:

```text
pattern.parquet
```

It contains:

```text
cell
gene
Bento features
SPRAWL features
class probability columns
pattern
```

The final pattern label is stored in:

```text
pattern
```

### 6.2 Compartment Output

Typical outputs:

```text
sampled_batches.csv
sampled_batches_meta.json
fluxmap.png
embedding.png
subdomain visualization
```

### 6.3 Colocalization Output

Typical outputs:

```text
instant_significant_pairs.csv
prefilter_summary.csv
viz_cpb_heatmap.png
viz_coloc_network.png
top pair cell plots
```

---


## 7. Notes

1. The toolbox expects standardized PKL input.
2. The main command-line tools use user-provided input and output paths.
3. Large datasets should be processed with background execution.
4. Pattern classification depends on trained XGBoost models stored in the `models/` directory.
5. For reproducibility, keep the model files and the code version synchronized.

