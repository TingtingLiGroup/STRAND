import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

# 假设你的.pkl文件路径是 'your_file.pkl'
pkl_file_path = './data/Dataset1_merfish_u2os_data_dict.pkl'

# 加载.pkl文件
data = pd.read_pickle(pkl_file_path)

# 获取 data_df 和 cell_boundary, nuclear_boundary
data_df = data.get('data_df')  # 转录本数据
cell_boundary = data.get('cell_boundary')  # 细胞边界
nuclear_boundary = data.get('nuclear_boundary')  # 细胞核边界

# 检查 data_df 是否正确加载
print(f"data_df: {type(data_df)} rows: {data_df.shape[0]}")
print(f"cell_boundary: {type(cell_boundary)} len: {len(cell_boundary)}")
print(f"nuclear_boundary: {type(nuclear_boundary)} len: {len(nuclear_boundary)}")

# 确保 'geometry' 列是 GeoSeries 类型
if not isinstance(data_df["geometry"], gpd.GeoSeries):
    data_df["geometry"] = gpd.GeoSeries(data_df["geometry"])

# 检查 'geometry' 列是否是 Point 类型
data_df["geometry_is_point"] = data_df["geometry"].apply(lambda x: isinstance(x, Point))

# 打印结果，检查是否正确
print(data_df[["geometry", "geometry_is_point"]].head())