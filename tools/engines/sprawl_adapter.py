import pickle
import numpy as np
import pandas as pd

from tools.sprawl_score.sprawl.cells import Cell
from tools.sprawl_score.sprawl.scoring import iter_scores

from tools.utils.timing import timed, TimerReport


def _boundary_to_xy(boundary):
    """dict value can be DataFrame(x,y) or ndarray/list -> ndarray(N,2)."""
    if boundary is None:
        return None
    if isinstance(boundary, pd.DataFrame):
        return boundary[["x", "y"]].to_numpy()
    return np.asarray(boundary)


def compute_sprawl_scores_from_pkl(
    pkl_file_path: str,
    *,
    metrics=("peripheral", "central", "punctate", "radial"),
    processes: int = 1,
    num_iterations: int = 200,
    num_pairs: int = 4,
    cell_id_col: str = "cell",
    gene_col: str = "gene",
    x_col: str = "x",
    y_col: str = "y",
    profile: bool = False
) -> pd.DataFrame:
    """
    从 pkl 计算 SPRAWL scores（默认4项）。
    单 z-slice 模式：忽略 z（边界只有2D）。
    输出：index=(cell,gene)，columns=sprawl_<metric>
    """
    rep = TimerReport() if profile else None

    with timed("sprawl:load_pkl", rep, print_each=profile):
        data = pickle.load(open(pkl_file_path, "rb"))
        df = data["data_df"]
        cell_boundary = data["cell_boundary"]

    cells = []
    with timed("sprawl:build_cells_total", rep, print_each=profile):
        for cid, cell_df in df.groupby(cell_id_col, sort=False):
            bxy = _boundary_to_xy(cell_boundary.get(cid))
            if bxy is None or len(bxy) < 3:
                continue

            spot_xy = cell_df[[x_col, y_col]].to_numpy()
            spot_genes = cell_df[gene_col].astype(str).to_numpy()

            cells.append(
                Cell(
                    cell_id=str(cid),
                    boundaries={0: bxy},
                    spot_coords={0: spot_xy},
                    spot_genes={0: spot_genes},
                    annotation="NA",
                )
            )

    wide = None
    with timed("sprawl:all_metrics_total", rep, print_each=profile):
        for m in metrics:
            kwargs = {"processes": processes}
            if m in ("peripheral","central"):
                kwargs["compute_variance"] = False
            if m in ("radial", "punctate"):
                kwargs.update({"num_iterations": num_iterations, "num_pairs": num_pairs})

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