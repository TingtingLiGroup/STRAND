"""Integration test: SPRAWL feature computation from PKL.

Requires LFS data. Skipped by default in CI.
"""
from pathlib import Path
import os

import pytest

ROOT = Path(__file__).resolve().parents[1]
PKL_PATH = ROOT / "data" / "Dataset1_merfish_u2os_data_dict.pkl"
RUN_DATA_TESTS = os.environ.get("STRAND_RUN_DATA_TESTS") == "1"

needs_lfs = pytest.mark.skipif(
    not RUN_DATA_TESTS or not PKL_PATH.exists() or PKL_PATH.stat().st_size < 1000,
    reason="data integration test disabled; set STRAND_RUN_DATA_TESTS=1 after 'git lfs pull'",
)


@needs_lfs
def test_sprawl_scores_from_pkl():
    from tools.engines.sprawl_adapter import compute_sprawl_scores_from_pkl

    out = compute_sprawl_scores_from_pkl(
        str(PKL_PATH),
        metrics=("peripheral", "central", "punctate", "radial"),
        processes=1,
        num_iterations=200,
        num_pairs=4,
    )
    assert out.shape[0] > 0, "Should produce at least one row"
    assert out.shape[1] >= 4, "Should have at least 4 SPRAWL features"
    assert "cell" in out.columns
    assert "gene" in out.columns
