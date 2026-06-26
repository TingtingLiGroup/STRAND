from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Union

from tools.engines.compartment_adapter import run_sampled_compartment_from_pkl


def detect_compartments_from_pkl(
    pkl_path: Union[str, Path],
    out_prefix: Union[str, Path],
    *,
    batch_col: str = "batch",
    only_batches: Optional[Union[str, Sequence[str]]] = None,
    frac: float = 0.01,
    radius: float = 40,
    n_clusters: Union[str, int] = "auto",
    cluster_range_min: int = 2,
    cluster_range_max: int = 12,
    embedding_mode: str = "loop",
    som_iterations: int = 1000,
    som_sigma: float = 1.0,
    som_learning_rate: float = 0.5,
    device: str = "auto",
    dpi: int = 300,
    export_csv: bool = False,
    profile: bool = False,
    figsize=None,
    embedding_point_size=10,
    subdomain_point_size=1,
    same_cell_neighborhood: bool = False,
    **kwargs,
):
    return run_sampled_compartment_from_pkl(
        pkl_path=pkl_path,
        out_prefix=out_prefix,
        batch_col=batch_col,
        only_batches=only_batches,
        frac=frac,
        radius=radius,
        n_clusters=n_clusters,
        cluster_range_min=cluster_range_min,
        cluster_range_max=cluster_range_max,
        embedding_mode=embedding_mode,
        som_iterations=som_iterations,
        som_sigma=som_sigma,
        som_learning_rate=som_learning_rate,
        device=device,
        dpi=dpi,
        export_csv=export_csv,
        profile=profile,
        figsize=figsize,
        embedding_point_size=embedding_point_size,
        subdomain_point_size=subdomain_point_size,
        same_cell_neighborhood=same_cell_neighborhood,
    )
