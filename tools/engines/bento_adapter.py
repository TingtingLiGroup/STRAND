from tools.io.data_loader import load_and_process_data
from tools.bento_feature.bento.bento_tools import (
    compute_rnaforest13_features,
    compute_cell_prereqs,
)

import pandas as pd
import geopandas as gpd

from shapely.geometry import Point, Polygon, MultiPolygon, GeometryCollection
from shapely.prepared import prep

from tools.utils.timing import timed, TimerReport


def _to_single_polygon(geom):
    """
    Ensure geometry passed to Bento is a single Polygon.

    Bento shape functions such as polygon_radius() call poly.exterior,
    which only exists for Polygon, not MultiPolygon.

    Rules:
      - Polygon: return directly if valid and non-empty.
      - MultiPolygon: keep the largest polygon by area.
      - GeometryCollection: keep the largest polygon inside it.
      - invalid geometry: try buffer(0), then reduce again.
      - None / empty / unusable: return None.
    """
    if geom is None:
        return None

    try:
        if geom.is_empty:
            return None
    except Exception:
        return None

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


def compute_bento13_from_dict(
    pkl_file_path: str,
    *,
    cell_id_col: str = "cell",
    gene_col: str = "gene",
    x_col: str = "x",
    y_col: str = "y",
    instance_key: str = "cell",
    nucleus_key: str = "nucleus",
    raster_step: int = 1,
    profile: bool = False,
    max_genes_per_cell: int | None = None,
    max_spots_per_gene: int | None = None,
) -> pd.DataFrame:
    """
    输入：pkl 文件路径

    输出：
      index = (cell_id, gene)
      columns = 13 个 bento_ 前缀特征

    Robustness:
      - cell id is normalized to str.
      - Polygon / MultiPolygon / GeometryCollection are reduced to a single Polygon.
      - bad cell polygons are skipped instead of interrupting the whole run.
      - missing nuclear boundary is allowed.
    """

    rep = TimerReport() if profile else None

    with timed("bento:load_and_process_data", rep, print_each=profile):
        data_df, cell_boundary_dict, nuclear_boundary_dict = load_and_process_data(
            pkl_file_path
        )

    # 保证 cell/gene 类型统一
    data_df = data_df.copy()
    data_df[cell_id_col] = data_df[cell_id_col].astype(str)
    data_df[gene_col] = data_df[gene_col].astype(str)

    # boundary dict 的 key 统一成 str，防止 int / str 不一致
    cell_boundary_dict = {
        str(k): v for k, v in (cell_boundary_dict or {}).items()
    }

    nuclear_boundary_dict = {
        str(k): v for k, v in (nuclear_boundary_dict or {}).items()
    }

    rows: list[dict] = []
    idx: list[tuple[str, str]] = []

    prereq_cache: dict[str, dict] = {}

    n_skip_missing_cell_boundary = 0
    n_skip_bad_cell_polygon = 0
    n_skip_feature_error = 0

    with timed("bento:groupby_setup", rep, print_each=profile):
        cell_groups = data_df.groupby(cell_id_col, sort=False)

    with timed("bento:loop_total", rep, print_each=profile):
        for cid_raw, cell_df in cell_groups:
            cid = str(cid_raw)

            # 这里必须用 cid，不要用 cid_raw
            cell_poly = cell_boundary_dict.get(cid)
            if cell_poly is None:
                n_skip_missing_cell_boundary += 1
                continue

            nuc_poly = nuclear_boundary_dict.get(cid) if nuclear_boundary_dict else None

            # 再做一次 Polygon 清洗，防止 MultiPolygon 进入 Bento
            cell_poly = _to_single_polygon(cell_poly)
            nuc_poly = _to_single_polygon(nuc_poly)

            if cell_poly is None:
                n_skip_bad_cell_polygon += 1
                continue

            try:
                nuc_prepped = prep(nuc_poly) if nuc_poly is not None else None
            except Exception:
                nuc_poly = None
                nuc_prepped = None

            # prereqs：每个 cell 只算一次
            with timed("bento:cell_prereqs_total", rep, print_each=False):
                if cid in prereq_cache:
                    pr = prereq_cache[cid]
                else:
                    try:
                        pr = compute_cell_prereqs(cell_poly, step=raster_step)
                    except Exception as e:
                        n_skip_bad_cell_polygon += 1
                        if profile:
                            print(
                                f"[bento][skip_cell] cell={cid} "
                                f"compute_cell_prereqs failed: {repr(e)}"
                            )
                        continue

                    prereq_cache[cid] = pr

            gene_groups = cell_df.groupby(gene_col, sort=False)

            # 可选：限制每个 cell 只算前 N 个 gene（按 spot 数最多）
            if max_genes_per_cell is not None:
                gene_sizes = gene_groups.size().sort_values(ascending=False)
                top_genes = set(gene_sizes.head(max_genes_per_cell).index.astype(str))
            else:
                top_genes = None

            for gene_raw, df0 in gene_groups:
                gene = str(gene_raw)

                if top_genes is not None and gene not in top_genes:
                    continue

                # 可选：限制每个 cell-gene 的 spot 数
                if max_spots_per_gene is not None and len(df0) > max_spots_per_gene:
                    df0 = df0.iloc[:max_spots_per_gene]

                pts = df0[[x_col, y_col]].to_numpy()
                if pts.shape[0] == 0:
                    continue

                # build geometry
                with timed("bento:build_geodataframe_total", rep, print_each=False):
                    base = pd.DataFrame(pts, columns=["x", "y"])
                    geom = gpd.GeoSeries(
                        [Point(xy) for xy in pts],
                        index=base.index,
                        name="geometry",
                    )
                    df = gpd.GeoDataFrame(base, geometry=geom)

                    df[instance_key] = cell_poly
                    df[nucleus_key] = nuc_poly

                    # 核内判断：prepared contains
                    if nuc_prepped is not None:
                        df[f"{nucleus_key}_index"] = [
                            "1" if nuc_prepped.contains(p) else ""
                            for p in geom
                        ]
                    else:
                        df[f"{nucleus_key}_index"] = ""

                    # prereqs
                    df[f"{instance_key}_radius"] = pr["radius"]
                    df[f"{instance_key}_raster"] = [pr["raster"]] * len(df)
                    df[f"{instance_key}_span"] = pr["span"]
                    df[f"{instance_key}_minx"] = pr["minx"]
                    df[f"{instance_key}_miny"] = pr["miny"]
                    df[f"{instance_key}_maxx"] = pr["maxx"]
                    df[f"{instance_key}_maxy"] = pr["maxy"]
                    df[f"{instance_key}_area"] = pr["area"]

                with timed("bento:features13_total", rep, print_each=False):
                    try:
                        feats = compute_rnaforest13_features(
                            df,
                            instance_key=instance_key,
                            nucleus_key=nucleus_key,
                        )
                    except Exception as e:
                        n_skip_feature_error += 1
                        if profile:
                            print(
                                f"[bento][skip_cellgene] cell={cid} gene={gene} "
                                f"compute_rnaforest13_features failed: {repr(e)}"
                            )
                        continue

                feats = {f"bento_{k}": v for k, v in feats.items()}
                idx.append((cid, gene))
                rows.append(feats)

    with timed("bento:build_output_df", rep, print_each=profile):
        out = pd.DataFrame(
            rows,
            index=pd.MultiIndex.from_tuples(
                idx,
                names=[cell_id_col, gene_col],
            ),
        )

    if profile:
        print("[bento] n_output_rows =", len(out))
        print("[bento] skipped cells missing boundary =", n_skip_missing_cell_boundary)
        print("[bento] skipped bad cell polygons =", n_skip_bad_cell_polygon)
        print("[bento] skipped cell-gene feature errors =", n_skip_feature_error)

    if profile and rep is not None:
        print(rep.summary())

    return out