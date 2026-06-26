# Pattern Classification Module

The Pattern Classification module predicts RNA subcellular localization patterns for each `cell-gene` pair.

Command:

```bash
subcellfeat
```

A feature-only prediction command is also available:

```bash
subcellfeat-pattern
```

---

## 1. Purpose

Spatial transcriptomics data contain both gene identity and RNA molecule coordinates. The same gene may show different spatial distributions across cells. The Pattern module converts the spatial distribution of each `cell-gene` pair into an interpretable localization pattern label.

The final output column is:

```text
pattern
```

---

## 2. Supported classes

The primary 8-class model supports:

```text
Foci
Nuclear
Cytoplasmic
Nuclear edge
Cell edge
Protrusion
Radial
Random
```

The fallback 7-class model supports:

```text
Nuclear
Cytoplasmic
Nuclear edge
Cell edge
Protrusion
Radial
Random
```

---

## 3. Main workflow

```text
PKL bundle
   ↓
Read data_df / cell_boundary / nuclear_boundary
   ↓
Standard prefiltering and cell synchronization
   ↓
Bento RNAforest13 feature extraction
   ↓
SPRAWL4 feature extraction
   ↓
Merge into 17-dimensional feature table
   ↓
Primary 8-class XGBoost prediction
   ↓
Calculate Foci ratio
   ↓
If Foci ratio > 0.5, use fallback 7-class no-Foci model
   ↓
Save final pattern table
```

---

## 4. Feature system

The classifier uses 17 spatial features.

### 4.1 Bento RNAforest13 features

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

### 4.2 SPRAWL features

```text
sprawl_peripheral
sprawl_central
sprawl_punctate
sprawl_radial
```

---

## 5. Default models

Primary model:

```text
models/multiclass_xgb_8class_prop075_final_from_cv.joblib
```

Fallback model:

```text
models/multiclass_xgb_7class_no_foci_final_from_cv.joblib
```

Fallback condition:

```text
Foci ratio > 0.5
```

The fallback mechanism is designed to reduce Foci absorption in real datasets where Foci-sensitive features are shifted from the simulated training distribution.

---

## 6. Basic usage

```bash
subcellfeat \
  --pkl data/simulated_data_dict.pkl \
  --out results/simulated_pattern.parquet \
  --profile
```

For large datasets, SPRAWL punctate and radial can be approximated by reducing the sampling parameters:

```bash
subcellfeat \
  --pkl data/simulated_data_dict.pkl \
  --out Pattern/simulated_pattern.parquet \
  --profile \
  --sprawl-iterations 20 \
  --sprawl-pairs 2
```

---

## 7. Feature-only mode

To compute only the 17-dimensional feature table:

```bash
subcellfeat \
  --pkl data/simulated_data_dict.pkl \
  --out results/simulated_features.parquet \
  --features-only \
  --profile
```

Then predict patterns from a feature table:

```bash
subcellfeat-pattern \
  --input results/simulated_features.parquet \
  --output results/simulated_pattern.parquet \
  --profile
```

---

## 8. Output columns

Typical output includes:

```text
cell
gene
17 feature columns
p_Foci
p_Nuclear
p_Cytoplasmic
p_Nuclear edge
p_Cell edge
p_Protrusion
p_Radial
p_Random
pattern
```

If fallback is triggered, `p_Foci` will not appear because the 7-class fallback model does not contain the Foci class.

---

## 9. Notes

1. The official final class column is `pattern`.
2. `pattern_top1` is not used as the final output column.
3. The default workflow includes prefiltering.
4. Use `--no-prefilter` only for debugging or reproducing external workflows.
5. Use `--no-foci-fallback` only when you explicitly want raw 8-class predictions.
