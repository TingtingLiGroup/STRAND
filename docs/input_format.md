# Input Format

Subcellular ToolBox uses a standardized PKL bundle as input. The PKL file should be a Python dictionary containing transcript coordinates, cell boundaries, nuclear boundaries, and optional cell metadata.

## 1. Required PKL structure

```python
{
    "data_df": pandas.DataFrame,
    "cell_boundary": dict,
    "nuclear_boundary": dict,
    "coordinates": pandas.DataFrame  # optional
}
```

The minimum required keys are:

```text
data_df
cell_boundary
```

`nuclear_boundary` is strongly recommended. If nuclear boundaries are unavailable, the toolbox should still run, but nucleus-related features and nc-ratio filtering may be skipped or less reliable.

---

## 2. `data_df`

`data_df` is the transcript-level table. Each row represents one RNA transcript molecule.

### Required columns

```text
cell
gene
x
y
```

### Optional columns

```text
z
fov
batch
sample_id
umi
annotation
celltype
```

### Example

```text
cell       gene      x        y        z    
cell_001   ACTB      123.5    456.2    0     
cell_001   MALAT1    130.1    440.8    0     
cell_002   GAPDH     210.3    508.6    0     
```

### Important notes

1. `cell` is the cell identifier.
2. `gene` is the gene name or gene ID.
3. `x` and `y` must use the same coordinate system as the cell and nuclear boundaries.
4. `z` is optional. If present, it can be used by 3D-aware modules or preserved for downstream analysis.
5. The same `cell` ID should be shared across `data_df`, `cell_boundary`, `nuclear_boundary`, and `coordinates`.

---

## 3. `cell_boundary`

`cell_boundary` should be a dictionary:

```python
cell_boundary[cell_id] = pandas.DataFrame({
    "x": [...],
    "y": [...]
})
```

Each key is a cell ID. Each value describes the polygon boundary of that cell.

### Example

```python
cell_boundary["cell_001"] = pd.DataFrame({
    "x": [10, 20, 20, 10, 10],
    "y": [10, 10, 20, 20, 10]
})
```

The first and last coordinates may be the same. If the polygon is not explicitly closed, the toolbox should handle closure internally when possible.

---

## 4. `nuclear_boundary`

`nuclear_boundary` should follow the same structure:

```python
nuclear_boundary[cell_id] = pandas.DataFrame({
    "x": [...],
    "y": [...]
})
```

If no nuclear boundary is available, use an empty dictionary:

```python
nuclear_boundary = {}
```

When nuclear boundaries are missing:

1. nc-ratio filtering should be skipped.
2. nucleus-related Bento features may be unavailable or less reliable.
3. localization classes such as `Nuclear` and `Nuclear edge` should be interpreted with caution.

---

## 5. `coordinates` metadata table

`coordinates` is optional but recommended. It stores cell-level metadata.

Common columns:

```text
cell
centerX
centerY
batch
fov
sample_id
celltype
```

This table is especially useful for grouped analysis, such as:

```bash
subcellfeat-compartment --group-col fov
subcellfeat-coloc --groupby celltype
```

---

## 6. Coordinate consistency

All coordinate-related inputs must use the same coordinate system:

```text
data_df x/y
cell_boundary x/y
nuclear_boundary x/y
coordinates centerX/centerY
```

Do not mix pixel coordinates and micron coordinates unless they have been converted beforehand.

---

## 7. Minimal validation checklist

Before running the toolbox, check:

```python
import pickle

with open("your_data_dict.pkl", "rb") as f:
    data = pickle.load(f)

df = data["data_df"]
print(df.shape)
print(df.columns)
print(df["cell"].nunique())
print(df["gene"].nunique())
print(len(data.get("cell_boundary", {})))
print(len(data.get("nuclear_boundary", {})))
```

The number of cells in `data_df` and `cell_boundary` should have a large intersection. If the intersection is zero, check whether cell IDs are stored as different types, such as integer in one table and string in another.
