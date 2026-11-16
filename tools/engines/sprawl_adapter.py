import pickle
import numpy as np
import pandas as pd
from scipy.spatial import distance

from tools.sprawl_score.sprawl.cells import Cell
from tools.sprawl_score.sprawl.scoring import iter_scores

from tools.utils.timing import timed, TimerReport


def _boundary_to_xy(boundary):
    """dict value can be DataFrame(x,y) or ndarray/list -> ndarray(N,2)."""
    if boundary is None:
        return None
    if isinstance(boundary, pd.DataFrame):
        return boundary[["x", "y"]].to_numpy(dtype=np.float64)
    return np.asarray(boundary, dtype=np.float64)


def compute_sprawl_scores_from_pkl(
    pkl_file_path: str | dict,
    *,
    metrics=("peripheral", "central", "punctate", "radial"),
    processes: int = 1,
    num_iterations: int = 200,
    num_pairs: int = 4,
    cell_id_col: str = "cell",
    gene_col: str = "gene",
    x_col: str = "x",
    y_col: str = "y",
    profile: bool = False,
    random_seed: int | None = None,
    scale_normalize: bool = False,
    target_span: float = 100.0,
) -> pd.DataFrame:
    """
    从 pkl 或内存 bundle dict 计算 SPRAWL scores（默认4项）。
    单 z-slice 模式：忽略 z（边界只有2D）。
    输出：index=(cell,gene)，columns=sprawl_<metric>
    """
    rep = TimerReport() if profile else None

    with timed("sprawl:load_pkl", rep, print_each=profile):
        data = (
            pickle.load(open(pkl_file_path, "rb"))
            if not isinstance(pkl_file_path, dict)
            else pkl_file_path
        )
        df = data["data_df"].copy()
        cell_boundary = data["cell_boundary"]

    cell_boundary = {str(k): v for k, v in (cell_boundary or {}).items()}
    df[cell_id_col] = df[cell_id_col].astype(str)

    cells = []
    with timed("sprawl:build_cells_total", rep, print_each=profile):
        for cid, cell_df in df.groupby(cell_id_col, sort=False):
            bxy = _boundary_to_xy(cell_boundary.get(str(cid)))
            if bxy is None or len(bxy) < 3:
                continue

            spot_xy = cell_df[[x_col, y_col]].to_numpy(dtype=np.float64)
            spot_genes = cell_df[gene_col].astype(str).to_numpy()

            if scale_normalize:
                bxy_valid = bxy[np.isfinite(bxy).all(axis=1)]
                if len(bxy_valid) < 3:
                    continue
                raw_span = float(distance.pdist(bxy_valid).max())
                if not np.isfinite(raw_span) or raw_span <= 1e-12:
                    continue
                center = bxy_valid.mean(axis=0, keepdims=True)
                scale = float(target_span) / raw_span
                bxy = np.round((bxy_valid - center) * scale, 8)
                spot_xy = np.round((spot_xy - center) * scale, 8)

            cells.append(
                Cell(
                    cell_id=str(cid),
                    boundaries={0: bxy},
                    spot_coords={0: spot_xy},
                    spot_genes={0: spot_genes},
                    annotation="NA",
                )
            )

    if not cells:
        empty = pd.DataFrame(columns=[f"sprawl_{m}" for m in metrics])
        empty.index = pd.MultiIndex.from_tuples([], names=[cell_id_col, gene_col])
        return empty

    wide = None
    with timed("sprawl:all_metrics_total", rep, print_each=profile):
        for m in metrics:
            kwargs = {"processes": processes}
            if m in ("peripheral", "central"):
                kwargs["compute_variance"] = False
            if m in ("radial", "punctate"):
                kwargs.update({
                    "num_iterations": num_iterations,
                    "num_pairs": num_pairs,
                    "random_seed": random_seed,
                })

            if profile:
                print(f"[sprawl] START metric={m} cells={len(cells)} kwargs={kwargs}")

            with timed(f"sprawl:metric:{m}", rep, print_each=profile):
                score_df = iter_scores(cells, metric=m, **kwargs)

            if profile:
                print(f"[sprawl] END   metric={m} rows={len(score_df)}")

            with timed(f"sprawl:reshape:{m}", rep, print_each=False):
                s = score_df[["cell_id", "gene", "score"]].copy()
                s["cell_id"] = s["cell_id"].astype(str)
                s["gene"] = s["gene"].astype(str)
                s = s.rename(columns={"score": f"sprawl_{m}"}).set_index(["cell_id", "gene"])
                wide = s if wide is None else wide.join(s, how="outer")

    wide.index.set_names([cell_id_col, gene_col], inplace=True)

    if profile and rep is not None:
        print(rep.summary())

    return wide
