# Compartment Detection Module

The Compartment Detection module identifies transcriptomic subcellular compartments from transcript coordinates.

Command:

```bash
subcellfeat-compartment
```

---

## 1. Purpose

This module detects spatial subdomains or transcriptomic compartments by analyzing local RNA neighborhoods and embedding patterns. It is designed to support large molecule-resolved spatial transcriptomics datasets.

---

## 2. Main workflow

```text
PKL bundle
   ↓
Read transcript coordinates and cell metadata
   ↓
Split by group, such as batch or fov
   ↓
Sample transcripts by fraction
   ↓
Construct local neighborhoods using radius search
   ↓
Compute RNAflux-like local embedding
   ↓
Train SOM or clustering model
   ↓
Assign subdomain / compartment labels
   ↓
Export summary tables and plots
```

---

## 3. Basic usage

```bash
subcellfeat-compartment \
  --pkl data/simulated_data_dict.pkl \
  --out-prefix results/simulated_compartment/simulateds \
  --frac 0.01 \
  --radius 40 \
  --n-clusters auto \
  --profile
```

---

## 4. Important parameters

### `--pkl`

Input standardized PKL bundle.

### `--out-prefix`

Output file prefix.

### `--group-col`

Column used to split the dataset into groups.

Common values:

```text
batch
fov
sample_id
celltype
```

Example:

```bash
--group-col fov --only-groups B9
```

### `--frac`

Sampling fraction for transcripts.

Example:

```bash
--frac 0.01
```

For very large datasets, start with a small fraction such as `0.01`.

### `--radius`

Radius for local neighborhood search.

Example:

```bash
--radius 40
```

The value must match the coordinate unit. If coordinates are in pixels, radius is in pixels. If coordinates are in microns, radius is in microns.

### `--n-clusters`

Number of SOM clusters.

Use automatic elbow selection:

```bash
--n-clusters auto
```

Or manually specify:

```bash
--n-clusters 7
```

If automatic clustering fails with a KneeLocator error, manually specify a cluster number.

### `--cluster-range-min` and `--cluster-range-max`

Range used when `--n-clusters auto`.

```bash
--cluster-range-min 2 --cluster-range-max 12
```

### `--embedding-mode`

```text
loop       notebook-aligned implementation
vectorized accelerated implementation
```

For reproducibility with notebook results, use:

```bash
--embedding-mode loop
```

For speed, use:

```bash
--embedding-mode vectorized
```

---

## 5. Typical output

The module may generate:

```text
*_sampled_batches.csv
*_sampled_batches_meta.json
embedding plots
subdomain plots
cluster assignment tables
```

When `--export-csv` is used, full transcript-level CSV files may be exported. These files can be very large.

---

## 6. Common issues

### KneeLocator does not find an elbow

Error example:

```text
KneeLocator did not find an elbow for n_clusters auto
```

Solution:

```bash
--n-clusters 7
```

or another manually selected number.

### Figure size is too large

Use:

```bash
--figsize auto
```

or specify a smaller size:

```bash
--figsize 12,12
```

### Radius is not appropriate

If the radius is too small, neighborhoods may be empty. If it is too large, local patterns may be over-smoothed. Choose radius according to dataset scale and coordinate unit.
