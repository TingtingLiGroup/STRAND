import pickle

import geopandas as gpd
import numpy as np
import pandas as pd

from shapely.geometry import Point, Polygon, MultiPolygon, GeometryCollection


def load_pkl_file(file_path):
    """加载 pkl 文件并返回数据。"""
    with open(file_path, "rb") as file:
        data = pickle.load(file)
    return data


def preprocess_transcripts(data):
    """
    处理转录本数据。

    输入的 pkl bundle 中应包含 data_df，至少包含：
      - cell
      - gene
      - x
      - y

    输出：
      - 添加 geometry 列后的 DataFrame
    """
    data_df = data.get("data_df")

    if data_df is None:
        raise ValueError("PKL bundle 中缺少 data_df")

    data_df = data_df.copy()

    if "x" not in data_df.columns or "y" not in data_df.columns:
        raise ValueError("data_df 必须包含 x 和 y 列")

    if "cell" in data_df.columns:
        data_df["cell"] = data_df["cell"].astype(str)

    if "gene" in data_df.columns:
        data_df["gene"] = data_df["gene"].astype(str)

    data_df["geometry"] = [
        Point(xy) for xy in zip(data_df["x"], data_df["y"])
    ]

    return data_df


def _boundary_to_xy(v):
    """
    将单个 boundary 对象转为 Nx2 坐标数组。

    兼容：
      - DataFrame，包含 x/y 列
      - numpy array/list，形状为 Nx2
      - dict-like，包含 x/y
    """
    if v is None:
        return None

    # DataFrame: boundary[["x", "y"]]
    if isinstance(v, pd.DataFrame):
        if "x" not in v.columns or "y" not in v.columns:
            return None
        pts = v[["x", "y"]].to_numpy()

    # dict-like: {"x": ..., "y": ...}
    elif isinstance(v, dict) and ("x" in v) and ("y" in v):
        pts = np.column_stack([v["x"], v["y"]])

    # array/list: [[x, y], ...]
    else:
        pts = np.asarray(v)

    if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 3:
        return None

    pts = pts.astype(float)

    # 去除 NaN / inf
    mask = np.isfinite(pts).all(axis=1)
    pts = pts[mask]

    if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 3:
        return None

    # 闭合 polygon
    if not np.allclose(pts[0], pts[-1]):
        pts = np.vstack([pts, pts[0]])

    return pts


def _to_single_polygon(geom):
    """
    将 shapely geometry 统一转成单个 Polygon。

    规则：
      - Polygon：直接返回
      - MultiPolygon：取面积最大的 Polygon
      - GeometryCollection：取其中面积最大的 Polygon
      - invalid geometry：先 buffer(0) 修复，再递归处理
      - None / empty / 无有效 polygon：返回 None

    目的：
      Bento 的 shape_features.py 里会访问 poly.exterior，
      因此传入对象必须是 Polygon，不能是 MultiPolygon。
    """
    if geom is None:
        return None

    try:
        if geom.is_empty:
            return None
    except Exception:
        return None

    # 修复无效 geometry
    try:
        if not geom.is_valid:
            geom = geom.buffer(0)
    except Exception:
        return None

    if geom is None:
        return None

    try:
        if geom.is_empty:
            return None
    except Exception:
        return None

    if isinstance(geom, Polygon):
        if geom.area <= 0:
            return None
        return geom

    if isinstance(geom, MultiPolygon):
        polys = [
            p for p in geom.geoms
            if isinstance(p, Polygon) and (not p.is_empty) and p.area > 0
        ]
        if not polys:
            return None
        return max(polys, key=lambda p: p.area)

    if isinstance(geom, GeometryCollection):
        polys = []

        for g in geom.geoms:
            if isinstance(g, Polygon) and (not g.is_empty) and g.area > 0:
                polys.append(g)

            elif isinstance(g, MultiPolygon):
                polys.extend([
                    p for p in g.geoms
                    if isinstance(p, Polygon) and (not p.is_empty) and p.area > 0
                ])

        if not polys:
            return None

        return max(polys, key=lambda p: p.area)

    return None


def boundary_dict_to_polygons(boundary_dict):
    """
    将边界字典转化为 shapely Polygon 字典。

    兼容：
      - boundary_dict is None
      - key 为 int / str
      - value 为 DataFrame / ndarray / list
      - invalid Polygon
      - buffer(0) 后变成 MultiPolygon
      - buffer(0) 后变成 GeometryCollection

    输出：
      - dict[str(cell_id)] -> Polygon or None

    注意：
      - MultiPolygon 会被压缩为最大面积 Polygon。
      - 这样可以避免 Bento 后续调用 poly.exterior 时报错。
    """
    out = {}

    if boundary_dict is None:
        return out

    for cid, v in boundary_dict.items():
        cid = str(cid)

        pts = _boundary_to_xy(v)
        if pts is None:
            out[cid] = None
            continue

        try:
            geom = Polygon(pts)
            poly = _to_single_polygon(geom)
            out[cid] = poly
        except Exception:
            out[cid] = None

    return out


def preprocess_boundaries(data):
    """
    处理细胞边界和细胞核边界。

    如果 nuclear_boundary 不存在或为 None，则返回空 dict。
    """
    cell_boundary = data.get("cell_boundary", None)
    nuclear_boundary = data.get("nuclear_boundary", None)

    cell_boundary_dict = boundary_dict_to_polygons(cell_boundary)
    nuclear_boundary_dict = boundary_dict_to_polygons(nuclear_boundary)

    return cell_boundary_dict, nuclear_boundary_dict


def load_and_process_data(pkl_file_path):
    """
    从 pkl 文件加载并处理数据。

    返回：
      - data_df: 带 geometry 列的 transcript DataFrame
      - cell_boundary_dict: dict[str(cell_id)] -> Polygon or None
      - nuclear_boundary_dict: dict[str(cell_id)] -> Polygon or None
    """
    data = load_pkl_file(pkl_file_path)
    data_df = preprocess_transcripts(data)
    cell_boundary_dict, nuclear_boundary_dict = preprocess_boundaries(data)
    return data_df, cell_boundary_dict, nuclear_boundary_dict