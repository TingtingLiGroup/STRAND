# Troubleshooting

This document lists common installation and runtime issues.

---

## 1. `pip install -e .` tries to install unexpected dependency versions

If pip upgrades packages such as `numpy`, `scipy`, or `matplotlib`, use a clean environment and install with version-limited dependencies.

Recommended:

```bash
conda env create -f environment.yml
conda activate subcellular
pip install -e . --no-deps
```

If dependencies are already installed in the current environment:

```bash
pip install -e . --no-deps --no-build-isolation
```

---

## 2. Proxy error during pip or conda installation

Error example:

```text
ProxyError: Cannot connect to proxy
```

Check proxy variables:

```bash
env | grep -i proxy
```

Temporarily unset invalid proxy variables:

```bash
unset http_proxy
unset https_proxy
unset HTTP_PROXY
unset HTTPS_PROXY
unset all_proxy
unset ALL_PROXY
```

If your server must use a proxy, configure the correct proxy address before installation.

---

## 3. `conda-libmamba-solver` reports missing `libarchive.so.13`

Error example:

```text
conda-libmamba-solver (libarchive.so.13: cannot open shared object file)
```

Use the classic conda solver:

```bash
CONDA_NO_PLUGINS=true conda config --set solver classic
CONDA_NO_PLUGINS=true conda env create -f environment.yml
```

---

## 4. Command not found after installation

Error example:

```text
subcellfeat: command not found
```

Check that the correct environment is activated:

```bash
conda activate subcellular
which python
which pip
which subcellfeat
```

Reinstall the toolbox in the active environment:

```bash
pip install -e . --no-deps --no-build-isolation
```

---

## 5. Model file not found

Error example:

```text
FileNotFoundError: models/multiclass_xgb_8class_prop075_final_from_cv.joblib
```

Make sure the `models/` directory exists in the project root and contains:

```text
multiclass_xgb_8class_prop075_final_from_cv.joblib
multiclass_xgb_7class_no_foci_final_from_cv.joblib
```

If running from another directory, use absolute or relative model parameters:

```bash
--pattern-model /path/to/models/multiclass_xgb_8class_prop075_final_from_cv.joblib
--fallback-pattern-model /path/to/models/multiclass_xgb_7class_no_foci_final_from_cv.joblib
```

---

## 6. SPRAWL reports zero valid cells

Error example:

```text
[SPRAWL] START metric=peripheral cells=0
ValueError: No objects to concatenate
```

Possible causes:

1. Too few transcripts per cell-gene pair.
2. Cell IDs do not match between `data_df` and boundaries.
3. Filtering is too strict.
4. Boundary data are missing or malformed.

Try lowering the cell-gene filter threshold:

```bash
--cellgene-filter-min-transcripts 3
```

For debugging:

```bash
--no-prefilter
```

---

## 7. Foci dominates the pattern output

If more than half of the predictions are `Foci`, the default workflow automatically triggers the 7-class fallback model.

To inspect raw 8-class predictions:

```bash
--no-foci-fallback
```

To change the threshold:

```bash
--foci-fallback-threshold 0.6
```

---

## 8. KneeLocator fails in compartment detection

Error example:

```text
KneeLocator did not find an elbow for n_clusters auto
```

Manually specify the number of clusters:

```bash
--n-clusters 7
```

or adjust the search range:

```bash
--cluster-range-min 2 --cluster-range-max 10
```

---

## 9. Colocalization runs very slowly or uses too much memory

Use cell sampling:

```bash
--sample-cells 3000
```

or:

```bash
--sample-cell-frac 0.1
```

Avoid saving full matrices unless needed:

```bash
# Do not use these unless debugging
--save-matrices
--save-pp-pvals
```

---

## 10. Large files should not be committed

Do not commit these folders:

```text
Dataset/
Pattern/
Compartment/
Colocalization/
logs/
```

Use `.gitignore` to exclude runtime data and generated results.
