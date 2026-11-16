from tools.io.data_loader import load_and_process_data
from tools.bento_feature.bento.bento_tools import (
    compute_rnaforest13_features,
    compute_rnaforest13_features_fast,
    compute_cell_prereqs,
    scale_normalize_cell_geometry,
)
from tools.bento_feature.bento.bento_tools.point_features import _second_moment

import numpy as np
import pandas as pd
import geopandas as gpd
import shapely
from astropy.stats.spatial import RipleysKEstimator

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
    pkl_file_path: str | dict,
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
    scale_normalize: bool = False,
    target_span: float = 100.0,
    preserve_empty_nan: bool = False,
    use_fast_vectorized: bool = True,
) -> pd.DataFrame:
    """
    输入：pkl 文件路径或内存 bundle dict

    输出：
      index = (cell_id, gene)
      columns = 13 个 Bento 特征（无 bento_ 前缀）

    v2 模式参数：
      - scale_normalize=True: 每个细胞平移到自身质心并按最大直径缩放到 target_span=100.0，
        彻底消除 pixel 与 micron 单位差异对 Ripley 特征的影响。
      - preserve_empty_nan=True: 核内或核外点数为 0 时保留 NaN，不再硬填为 0。
    """

    rep = TimerReport() if profile else None

    with timed("bento:load_and_process_data", rep, print_each=profile):
        data_df, cell_boundary_dict, nuclear_boundary_dict = load_and_process_data(
            pkl_file_path,
            build_geometry=False,
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

            cell_poly = cell_boundary_dict.get(cid)
            if cell_poly is None:
                n_skip_missing_cell_boundary += 1
                continue

            nuc_poly = nuclear_boundary_dict.get(cid) if nuclear_boundary_dict else None

            cell_poly = _to_single_polygon(cell_poly)
            nuc_poly = _to_single_polygon(nuc_poly)

            if cell_poly is None:
                n_skip_bad_cell_polygon += 1
                continue

            cell_xy_all = cell_df[[x_col, y_col]].to_numpy(dtype=np.float64)

            if scale_normalize:
                try:
                    cell_poly, nuc_poly, cell_xy_all, _ = scale_normalize_cell_geometry(
                        cell_poly,
                        nuc_poly=nuc_poly,
                        pts_xy=cell_xy_all,
                        target_span=target_span,
                    )
                except Exception as e:
                    n_skip_bad_cell_polygon += 1
                    if profile:
                        print(f"[bento][skip_cell] cell={cid} scale_normalize failed: {repr(e)}")
                    continue

            try:
                nuc_prepped = prep(nuc_poly) if (nuc_poly is not None and not use_fast_vectorized) else None
            except Exception:
                nuc_poly = None
                nuc_prepped = None

            # prereqs：每个 cell 只算一次
            with timed("bento:cell_prereqs_total", rep, print_each=False):
                if cid in prereq_cache:
                    pr = prereq_cache[cid]
                else:
                    try:
                        pr = compute_cell_prereqs(
                            cell_poly,
                            step=raster_step,
                            canonical_grid=scale_normalize,
                            fixed_span=target_span if scale_normalize else None,
                        )
                    except Exception as e:
                        n_skip_bad_cell_polygon += 1
                        if profile:
                            print(
                                f"[bento][skip_cell] cell={cid} "
                                f"compute_cell_prereqs failed: {repr(e)}"
                            )
                        continue

                    prereq_cache[cid] = pr

            # Cell-level precomputations for fast vectorized path
            if use_fast_vectorized:
                cell_pts_geom_all = shapely.points(cell_xy_all)
                cell_b_dists_all = shapely.distance(cell_poly.boundary, cell_pts_geom_all)
                cell_c_dists_all = shapely.distance(cell_poly.centroid, cell_pts_geom_all)
                if nuc_poly is not None:
                    cell_in_nuc_all = shapely.contains_xy(
                        nuc_poly, cell_xy_all[:, 0], cell_xy_all[:, 1]
                    )
                    nuc_b_dists_all = shapely.distance(nuc_poly.boundary, cell_pts_geom_all)
                    nuc_c_dists_all = shapely.distance(nuc_poly.centroid, cell_pts_geom_all)
                    if pr["raster"] is not None and len(pr["raster"]) > 0:
                        nuc_c_xy = np.array(
                            [[nuc_poly.centroid.x, nuc_poly.centroid.y]], dtype=np.float64
                        )
                        cell_raster_nuc_moment = _second_moment(nuc_c_xy, pr["raster"])
                    else:
                        cell_raster_nuc_moment = None
                else:
                    cell_in_nuc_all = None
                    nuc_b_dists_all = None
                    nuc_c_dists_all = None
                    cell_raster_nuc_moment = None

                ripley_est = RipleysKEstimator(
                    area=float(pr["area"]),
                    x_min=float(pr["minx"]),
                    y_min=float(pr["miny"]),
                    x_max=float(pr["maxx"]),
                    y_max=float(pr["maxy"]),
                )
            else:
                cell_pts_geom_all = None
                cell_in_nuc_all = None
                cell_raster_nuc_moment = None
                ripley_est = None

            # Group genes within cell using positional indices into cell_xy_all
            genes_arr = cell_df[gene_col].to_numpy()
            pos_series = pd.Series(np.arange(len(cell_df)), index=genes_arr)
            gene_pos_groups = pos_series.groupby(level=0, sort=False)

            if max_genes_per_cell is not None:
                gene_sizes = gene_pos_groups.size().sort_values(ascending=False)
                top_genes = set(gene_sizes.head(max_genes_per_cell).index.astype(str))
            else:
                top_genes = None

            for gene_raw, pos_s in gene_pos_groups:
                gene = str(gene_raw)

                if top_genes is not None and gene not in top_genes:
                    continue

                pos_idx = pos_s.to_numpy()
                if max_spots_per_gene is not None and pos_idx.size > max_spots_per_gene:
                    pos_idx = pos_idx[:max_spots_per_gene]

                if pos_idx.size == 0:
                    continue

                pts = cell_xy_all[pos_idx]

                if use_fast_vectorized:
                    with timed("bento:features13_total", rep, print_each=False):
                        try:
                            feats = compute_rnaforest13_features_fast(
                                pts_xy=pts,
                                pts_geom=cell_pts_geom_all[pos_idx],
                                cell_poly=cell_poly,
                                nuc_poly=nuc_poly,
                                in_nuc_mask=cell_in_nuc_all[pos_idx] if cell_in_nuc_all is not None else None,
                                pr=pr,
                                instance_key=instance_key,
                                nucleus_key=nucleus_key,
                                preserve_empty_nan=preserve_empty_nan,
                                ripley_estimator=ripley_est,
                                cell_raster_nuc_moment=cell_raster_nuc_moment,
                                precomp_cell_b_dists=cell_b_dists_all[pos_idx],
                                precomp_cell_c_dists=cell_c_dists_all[pos_idx],
                                precomp_nuc_b_dists=nuc_b_dists_all[pos_idx] if nuc_b_dists_all is not None else None,
                                precomp_nuc_c_dists=nuc_c_dists_all[pos_idx] if nuc_c_dists_all is not None else None,
                            )
                        except Exception as e:
                            n_skip_feature_error += 1
                            if profile:
                                print(
                                    f"[bento][skip_cellgene] cell={cid} gene={gene} "
                                    f"compute_rnaforest13_features_fast failed: {repr(e)}"
                                )
                            continue
                else:
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

                        if nuc_prepped is not None:
                            df[f"{nucleus_key}_index"] = [
                                "1" if nuc_prepped.contains(p) else ""
                                for p in geom
                            ]
                        else:
                            df[f"{nucleus_key}_index"] = ""

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
                                preserve_empty_nan=preserve_empty_nan,
                            )
                        except Exception as e:
                            n_skip_feature_error += 1
                            if profile:
                                print(
                                    f"[bento][skip_cellgene] cell={cid} gene={gene} "
                                    f"compute_rnaforest13_features failed: {repr(e)}"
                                )
                            continue

                idx.append((cid, gene))
                rows.append(dict(feats))

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
