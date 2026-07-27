# STRAND Tools

STRAND (Subcellular-Resolved Spatial Transcriptome RNA Architecture and Navigation Database) Tools is a command-line toolbox for subcellular spatial transcriptomics analysis. It provides four commands for standardized PKL input data:

| Command | Function |
|---------|----------|
| `subcellfeat` | Full pipeline: compute 17 spatial features (Bento + SPRAWL) and predict RNA localization patterns |
| `subcellfeat-pattern` | Predict patterns from pre-computed feature parquet (skip feature computation) |
| `subcellfeat-compartment` | Detect subcellular transcriptomic compartments via RNAflux embedding + SOM clustering |
| `subcellfeat-compartment-optimized` | Run the low-memory global same-cell RNAflux/SOM compartment backend |
| `subcellfeat-compartment-redisplay` | Redraw compartment plots from existing point-level compartment CSV files |
| `subcellfeat-coloc` | Identify significant RNA-RNA colocalization pairs via InSTAnT PP + CPB |
| `subcellfeat-coloc-plot` | Redraw colocalization visualizations from an existing output directory |
| `subcellfeat-coloc-summary` | Summarize colocalization outputs across groups or datasets |

The toolbox is designed for molecule-resolved spatial transcriptomics datasets such as MERFISH, Xenium, CosMx, seqFISH, and other transcript-level spatial data.

---

## 1. Main Functions

### 1.1 Pattern Classification (`subcellfeat`)

The main command. It computes 17 spatial features (13 Bento + 4 SPRAWL) for each cell-gene pair from a PKL bundle, then predicts RNA localization patterns using a trained XGBoost classifier.

Supported pattern classes:

```text
Nuclear, Nuclear edge, Cytoplasmic, Cell edge, Protrusion, Radial, Random, Foci
```

Default prediction strategy:

1. Use the primary 8-class XGBoost model.
2. Keep the primary 8-class output, including `Foci`.
3. Save the final result with the unified column name `pattern`.

The historical 7-class no-Foci model is retained only for explicit compatibility use in the Python API. The command-line defaults do not switch models based on Foci ratio.

Use `--features-only` to compute features without pattern prediction.

### 1.2 Pattern Prediction from Features (`subcellfeat-pattern`)

A lightweight command for when features have already been computed. Takes a feature parquet file as input and applies the XGBoost classifier directly, skipping the expensive Bento/SPRAWL computation.

### 1.3 Compartment Detection (`subcellfeat-compartment`)

Detects subcellular transcriptomic compartments using RNA spatial distribution and embedding-based analysis. The default workflow uses same-cell local RNAflux neighborhoods, `radius=auto` based on the current group's cell size, and SOM clustering over the default auto range `4..7`.

For very large datasets, `subcellfeat-compartment-optimized` provides a low-memory global same-cell backend with Numba cell indexing, memmapped RNAflux/RGB arrays, chunked SOM assignment, and display-column output splitting.

Outputs (prefixed with `--out-prefix`):

```text
{out_prefix}_sampled_batches.csv
{out_prefix}_sampled_batches_meta.json
{out_prefix}_{batch}_embedding.png
{out_prefix}_{batch}_subdomain.png
```

### 1.4 Colocalization Analysis (`subcellfeat-coloc`)

Identifies significant RNA-RNA colocalization relationships within cells. It runs InSTAnT Proximal Pairs (PP) and Conditional Probability of Barcodes (CPB) tests, filters for significance, and generates visualizations.

If the input data has no `z` column, the command automatically switches to 2D mode.

Outputs:

```text
instant_significant_pairs.csv
prefilter_summary.csv
viz_cpb_heatmap_topN.png
viz_coloc_network_styled.png
```

---

## 2. Installation

### 2.1 Clone and pull data

This repository uses [Git LFS](https://git-lfs.com/) for model and data files.

```bash
git clone https://github.com/TingtingLiGroup/STRAND.git
cd STRAND
git lfs pull
```

### 2.2 Recommended Conda/Mamba Installation

```bash
conda env create -f environment.yml
conda activate subcellular
pip install -e .
```

### 2.3 Pip Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

### 2.4 Compartment Runtime

The compartment detection module uses PyTorch and Numba. `--device auto` uses `cuda:0` when a CUDA device is available, otherwise CPU.

```bash
subcellfeat-compartment --help
subcellfeat-compartment-optimized --help
```

### 2.5 Verify Installation

```bash
subcellfeat --help
subcellfeat-pattern --help
subcellfeat-compartment --help
subcellfeat-compartment-optimized --help
subcellfeat-compartment-redisplay --help
subcellfeat-coloc --help
subcellfeat-coloc-plot --help
subcellfeat-coloc-summary --help
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

### 4.1 Pattern Classification (full pipeline)

```bash
subcellfeat \
  --pkl data/simulated_data_dict.pkl \
  --out results/simulated_pattern.parquet \
  --profile
```

For large datasets, use `--fast` to skip the two slowest SPRAWL features (punctate + radial):

```bash
subcellfeat \
  --pkl data/simulated_data_dict.pkl \
  --out results/simulated_pattern.parquet \
  --fast --profile
```

Or reduce SPRAWL approximation parameters:

```bash
subcellfeat \
  --pkl data/simulated_data_dict.pkl \
  --out results/simulated_pattern.parquet \
  --sprawl-iterations 20 --sprawl-pairs 2 --profile
```

To compute features only (no pattern prediction):

```bash
subcellfeat \
  --pkl data/simulated_data_dict.pkl \
  --out results/simulated_features.parquet \
  --features-only --profile
```

### 4.2 Pattern Prediction from Features

If you already have a feature parquet from step 4.1 `--features-only`:

```bash
subcellfeat-pattern \
  --input results/simulated_features.parquet \
  --output results/simulated_pattern.parquet \
  --profile
```

### 4.3 Compartment Detection

```bash
subcellfeat-compartment \
  --pkl data/simulated_data_dict.pkl \
  --out-prefix results/simulated_compartment \
  --frac 0.01 \
  --radius auto \
  --n-clusters auto \
  --cluster-range-min 4 \
  --cluster-range-max 7 \
  --profile
```

Large low-memory backend:

```bash
subcellfeat-compartment-optimized \
  --pkl data/simulated_data_dict.pkl \
  --out-dir results/simulated_compartment_optimized \
  --group-col batch \
  --display-col fov \
  --frac 0.01 \
  --radius 5 \
  --n-clusters auto \
  --profile
```

### 4.4 Colocalization Analysis

```bash
subcellfeat-coloc \
  --pkl data/simulated_data_dict.pkl \
  --out-dir results/simulated_coloc \
  --use-2d \
  --profile
```

Note: if the input PKL has no `z` column, `--use-2d` is applied automatically.

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

Typical outputs (prefixed with `--out-prefix`):

```text
{out_prefix}_sampled_batches.csv
{out_prefix}_sampled_batches_meta.json
{out_prefix}_{batch}_embedding.png
{out_prefix}_{batch}_subdomain.png
```

The optimized backend writes `metadata.json`, memmapped work arrays under `work/`, and display-level CSV/PNG outputs under `results/`.

### 6.3 Colocalization Output

Typical outputs:

```text
instant_significant_pairs.csv
prefilter_summary.csv
viz_cpb_heatmap_topN.png
viz_coloc_network_styled.png
top pair cell plots
```

---


## 7. Security

The toolbox uses Python `pickle` to load PKL input bundles. Pickle deserialization can execute arbitrary code. **Only load PKL files that you or your collaborators produced from trusted data sources.** Do not load PKL files from untrusted or unknown origins.

---

## 8. Notes

1. The toolbox expects standardized PKL input.
2. The main command-line tools use user-provided input and output paths.
3. Large datasets should be processed with background execution.
4. Pattern classification depends on trained XGBoost models stored in the `models/` directory.
5. For reproducibility, keep the model files and the code version synchronized.

---

## 9. Citation

If you use STRAND Tools in your research, please cite:

> STRAND Tools: A command-line toolbox for subcellular spatial transcriptomics analysis.
> https://github.com/TingtingLiGroup/STRAND

---

## 10. Contributing

Contributions are welcome. Please open an issue or submit a pull request on [GitHub](https://github.com/TingtingLiGroup/STRAND/issues).

---

## 11. License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
