"""Integration test: Bento13 feature computation from PKL.

Requires LFS data. Skipped by default in CI.
"""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PKL_PATH = ROOT / "data" / "Dataset1_merfish_u2os_data_dict.pkl"

needs_lfs = pytest.mark.skipif(
    not PKL_PATH.exists() or PKL_PATH.stat().st_size < 1000,
    reason="LFS data not pulled (run 'git lfs pull')",
)


@needs_lfs
def test_bento13_from_pkl():
    from tools.engines.bento_adapter import compute_bento13_from_dict

    out = compute_bento13_from_dict(
        str(PKL_PATH),
        cell_id_col="cell",
        gene_col="gene",
        x_col="x",
        y_col="y",
        instance_key="cell",
        nucleus_key="nucleus",
        raster_step=1,
    )
    assert out.shape[0] > 0, "Should produce at least one row"
    assert out.shape[1] >= 13, "Should have at least 13 Bento features"
    assert "cell" in out.columns
    assert "gene" in out.columns
