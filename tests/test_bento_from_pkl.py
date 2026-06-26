# import pickle
# import pandas as pd
# import sys
# import os
# from pathlib import Path
# ROOT = Path(__file__).resolve().parents[1]   # 指向 Subcellular_Feature/
# sys.path.insert(0, str(ROOT))                # ✅ 加项目根目录（tools 的父目录）
# from tools.engines.bento_adapter import compute_bento13_from_dict
# def main():
#     # 载入 pkl 文件
#     pkl_file_path = "data/Dataset1_merfish_u2os_data_dict.pkl"
#     data = pd.read_pickle(pkl_file_path)

#     data_df = data["data_df"]  # 读取转录本数据
#     cell_boundary = data["cell_boundary"]  # 读取细胞边界数据
#     nuclear_boundary = data["nuclear_boundary"]  # 读取细胞核边界数据

#     # 检查数据格式
#     print(f"data_df: {type(data_df)} rows: {len(data_df)}")
#     print(f"cell_boundary: {type(cell_boundary)} len: {len(cell_boundary)}")
#     print(f"nuclear_boundary: {type(nuclear_boundary)} len: {len(nuclear_boundary)}")

#     # 调用 bento_adapter.py 计算 Bento13 特征
#     out = compute_bento13_from_dict(
#         transcripts=data_df,
#         cell_boundary_dict=cell_boundary,
#         nucleus_boundary_dict=nuclear_boundary,
#         cell_id_col="cell",
#         gene_col="gene",
#         x_col="x",
#         y_col="y",
#     )

#     # 输出结果，查看计算的特征
#     print(out.head())

# if __name__ == "__main__":
#     main()

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.engines.bento_adapter import compute_bento13_from_dict

def main():
    pkl_file_path = ROOT / "data" / "Dataset1_merfish_u2os_data_dict.pkl"

    out = compute_bento13_from_dict(
        str(pkl_file_path),
        cell_id_col="cell",
        gene_col="gene",
        x_col="x",
        y_col="y",
        instance_key="cell",
        nucleus_key="nucleus",
        raster_step=1,
    )

    print(out.shape)
    print(out.columns.tolist())
    print(out.head())

if __name__ == "__main__":
    main()