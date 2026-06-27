"""Integration test: full pipeline (Bento + SPRAWL) from PKL.

Requires LFS data. Run with: pytest tests/test_all_from_pkl.py -s
Skipped by default in CI (no LFS data).
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
def test_compute_all_from_pkl():
    from tools.api.compute_all import compute_all_from_pkl

    all_df = compute_all_from_pkl(
        str(PKL_PATH),
        sprawl_kwargs={
            "metrics": ("peripheral", "central", "punctate", "radial"),
            "processes": 1,
            "num_iterations": 200,
            "num_pairs": 4,
        },
        out_path=None,
    )
    # Bento 13 features + SPRAWL 4 features + cell + gene = at least 19 columns
    assert all_df.shape[0] > 0, "Should produce at least one row"
    assert all_df.shape[1] >= 13, "Should contain at least Bento 13 features"
    assert "cell" in all_df.columns
    assert "gene" in all_df.columns
