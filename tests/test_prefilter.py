import pickle
import pandas as pd

from tools.api.compute_all import (
    _compute_nc_ratio_df,
    _prefilter_bundle_before_features,
)


PKL_PATH = "Dataset/Dataset3_merfish_intestine_data_dict.pkl"


with open(PKL_PATH, "rb") as f:
    bundle = pickle.load(f)

print("=== 原始数据 ===")
df0 = bundle["data_df"].copy()
df0["cell"] = df0["cell"].astype(str)
df0["gene"] = df0["gene"].astype(str)

print("n_cells in data_df:", df0["cell"].nunique())
print("n_genes in data_df:", df0["gene"].nunique())
print("n_rows in data_df:", len(df0))
print("n_cell_boundary:", len(bundle.get("cell_boundary", {})))
print("n_nuclear_boundary:", len(bundle.get("nuclear_boundary", {})))

df_nc_before = _compute_nc_ratio_df(bundle)
df_nc_before = df_nc_before.dropna(subset=["nc_ratio"]).copy()
df_nc_before = df_nc_before[(df_nc_before["nc_ratio"] > 0) & (df_nc_before["nc_ratio"] < 1)].copy()

print("nc_ratio mean before:", df_nc_before["nc_ratio"].mean())
print("nc_ratio quantiles before:")
print(df_nc_before["nc_ratio"].quantile([0.01, 0.025, 0.05, 0.5, 0.95, 0.975, 0.99]))

bundle_filtered, stats = _prefilter_bundle_before_features(
    bundle,
    filter_cells_by_nc_ratio=True,
    nc_ratio_q_low=0.025,
    nc_ratio_q_high=0.975,
    gene_filter_min_cells=10,
    gene_filter_min_transcripts=6,
)

print("\n=== 过滤后 ===")
df1 = bundle_filtered["data_df"].copy()
df1["cell"] = df1["cell"].astype(str)
df1["gene"] = df1["gene"].astype(str)

print("n_cells in data_df:", df1["cell"].nunique())
print("n_genes in data_df:", df1["gene"].nunique())
print("n_rows in data_df:", len(df1))
print("n_cell_boundary:", len(bundle_filtered.get("cell_boundary", {})))
print("n_nuclear_boundary:", len(bundle_filtered.get("nuclear_boundary", {})))

df_nc_after = _compute_nc_ratio_df(bundle_filtered)
df_nc_after = df_nc_after.dropna(subset=["nc_ratio"]).copy()
df_nc_after = df_nc_after[(df_nc_after["nc_ratio"] > 0) & (df_nc_after["nc_ratio"] < 1)].copy()

print("nc_ratio mean after:", df_nc_after["nc_ratio"].mean())
print("nc_ratio quantiles after:")
print(df_nc_after["nc_ratio"].quantile([0.01, 0.025, 0.05, 0.5, 0.95, 0.975, 0.99]))

cells_df = set(df1["cell"].unique())
cells_cb = set(map(str, bundle_filtered.get("cell_boundary", {}).keys()))
cells_nb = set(map(str, bundle_filtered.get("nuclear_boundary", {}).keys()))

print("\n=== 三方同步检查 ===")
print("data_df == cell_boundary:", cells_df == cells_cb)
print("data_df == nuclear_boundary:", cells_df == cells_nb)
print("cell_boundary == nuclear_boundary:", cells_cb == cells_nb)

print("\n=== 统计信息 ===")
print("stats keys:", list(stats.keys()))
print("n_cells_after_nc_ratio_filter:", stats.get("n_cells_after_nc_ratio_filter"))
print("n_genes_after_gene_filter:", stats.get("n_genes_after_gene_filter"))
print("n_common_cells:", stats.get("n_common_cells"))