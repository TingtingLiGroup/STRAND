"""Unit tests for the scale-invariant 28-feature (v2_28d) localization pipeline.

Synthetic geometry tests run unconditionally in CI without Git LFS data.
Model inference tests run automatically whenever LFS model binaries are pulled.
"""
from __future__ import annotations

import pickle
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tools.api.compute_all import compute_all_from_pkl, run_all_with_patterns, _limit_cells_in_bundle
from tools.api.patterns import predict_patterns_from_parquet
from tools.engines.bento_adapter import compute_bento13_from_dict
from tools.engines.csr_features import FEATURE_COLS_V1_17D, FEATURE_COLS_V2_28D
from tools.models.pattern_classifier import (
    DEFAULT_MODEL_V2_FILENAME,
    LEGACY_MODEL_V1_FILENAME,
    default_model_path,
)

ROOT = Path(__file__).resolve().parents[1]
SIM_PKL = ROOT / "data" / "simulated_data_dict.pkl"


def _has_real_binary(path: Path) -> bool:
    if not path.exists() or path.stat().st_size < 1000:
        return False
    with open(path, "rb") as f:
        return not f.read(40).startswith(b"version https://git-lfs")


def _make_synthetic_bundle() -> dict:
    """Create a multi-gene synthetic cell bundle with asymmetric morphology."""
    rng = np.random.default_rng(123)
    angles = np.linspace(0, 2 * np.pi, 36, endpoint=False)
    radii = 60.0 + 15.0 * np.cos(2 * angles) + 8.0 * np.sin(3 * angles)
    cell_xy = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    cell_xy = np.vstack([cell_xy, cell_xy[0]])

    nuc_xy = np.column_stack([22.0 * np.cos(angles), 18.0 * np.sin(angles)]) + np.array([[5.0, -3.0]])
    nuc_xy = np.vstack([nuc_xy, nuc_xy[0]])

    rows = []
    # 1. Purely nuclear gene
    for x, y in rng.normal(loc=[5.0, -3.0], scale=4.0, size=(12, 2)):
        rows.append({"cell": "cell_A", "gene": "GENE_NUC", "x": float(x), "y": float(y)})
    # 2. Purely cytoplasmic peripheral gene
    ang_edge = rng.uniform(-0.4, 0.4, size=14)
    for a in ang_edge:
        r = 0.90 * (60.0 + 15.0 * np.cos(2 * a) + 8.0 * np.sin(3 * a))
        rows.append({"cell": "cell_A", "gene": "GENE_EDGE", "x": float(r * np.cos(a)), "y": float(r * np.sin(a))})
    # 3. Tight cytoplasmic foci gene
    for x, y in rng.normal(loc=[-32.0, 10.0], scale=0.8, size=(15, 2)):
        rows.append({"cell": "cell_A", "gene": "GENE_FOCI", "x": float(x), "y": float(y)})
    # 4. Mixed nuclear + cytoplasmic gene
    for x, y in rng.uniform(-25.0, 25.0, size=(20, 2)):
        rows.append({"cell": "cell_A", "gene": "GENE_MIX", "x": float(x), "y": float(y)})

    return {
        "data_df": pd.DataFrame(rows),
        "cell_boundary": {"cell_A": cell_xy},
        "nuclear_boundary": {"cell_A": nuc_xy},
    }


def _scale_and_shift_bundle(bundle: dict, scale: float, dx: float = 0.0, dy: float = 0.0) -> dict:
    df = bundle["data_df"].copy()
    df["x"] = df["x"].astype(np.float64) * scale + dx
    df["y"] = df["y"].astype(np.float64) * scale + dy

    def _transform_bounds(bdict):
        if bdict is None:
            return None
        out = {}
        for k, v in bdict.items():
            if isinstance(v, pd.DataFrame):
                v2 = v.copy()
                v2["x"] = v2["x"].astype(np.float64) * scale + dx
                v2["y"] = v2["y"].astype(np.float64) * scale + dy
                out[k] = v2
            else:
                arr = np.asarray(v, dtype=np.float64).copy()
                arr[:, 0] = arr[:, 0] * scale + dx
                arr[:, 1] = arr[:, 1] * scale + dy
                out[k] = arr
        return out

    return {
        "data_df": df,
        "cell_boundary": _transform_bounds(bundle.get("cell_boundary")),
        "nuclear_boundary": _transform_bounds(bundle.get("nuclear_boundary")),
    }


def test_fast_bento13_matches_legacy_reference():
    bundle = _make_synthetic_bundle()
    df_fast = compute_bento13_from_dict(bundle, use_fast_vectorized=True)
    df_legacy = compute_bento13_from_dict(bundle, use_fast_vectorized=False)

    assert list(df_fast.columns) == FEATURE_COLS_V1_17D[:13]
    assert df_fast.index.equals(df_legacy.index)
    for col in df_fast.columns:
        np.testing.assert_allclose(
            df_fast[col].to_numpy(),
            df_legacy[col].to_numpy(),
            rtol=1e-10,
            atol=1e-10,
            equal_nan=True,
            err_msg=f"Fast vs legacy mismatch in {col}",
        )


def test_v2_28d_scale_and_translation_invariance():
    sub_1x = _make_synthetic_bundle()
    sub_005x = _scale_and_shift_bundle(sub_1x, scale=0.05, dx=1234.5, dy=-987.25)
    sub_20x = _scale_and_shift_bundle(sub_1x, scale=20.0, dx=-5000.0, dy=3333.0)

    f_1x = compute_all_from_pkl(sub_1x, feature_set="v2_28d", sprawl_kwargs={"processes": 1})
    f_005x = compute_all_from_pkl(sub_005x, feature_set="v2_28d", sprawl_kwargs={"processes": 1})
    f_20x = compute_all_from_pkl(sub_20x, feature_set="v2_28d", sprawl_kwargs={"processes": 1})

    expected_cols = FEATURE_COLS_V2_28D + ["n_transcripts"]
    assert list(f_1x.columns) == expected_cols
    assert f_1x["n_transcripts"].dtype == np.int64
    assert f_1x.index.equals(f_005x.index)
    assert f_1x.index.equals(f_20x.index)
    assert not np.isinf(f_1x[FEATURE_COLS_V2_28D].to_numpy(dtype=np.float64)).any()

    for col in expected_cols:
        np.testing.assert_allclose(
            f_1x[col].to_numpy(),
            f_005x[col].to_numpy(),
            rtol=1e-6,
            atol=1e-6,
            equal_nan=True,
            err_msg=f"1x vs 0.05x mismatch in {col}",
        )
        np.testing.assert_allclose(
            f_1x[col].to_numpy(),
            f_20x[col].to_numpy(),
            rtol=1e-6,
            atol=1e-6,
            equal_nan=True,
            err_msg=f"1x vs 20x mismatch in {col}",
        )


def test_v2_empty_compartment_nan_and_missing_nucleus():
    bundle = _make_synthetic_bundle()
    f_nuc = compute_all_from_pkl(bundle, feature_set="v2_28d", sprawl_kwargs={"processes": 1})

    row_pn = f_nuc.loc[("cell_A", "GENE_NUC")]
    assert np.isnan(row_pn["bento_nucleus_outer_proximity"])
    assert np.isnan(row_pn["bento_nucleus_outer_asymmetry"])
    assert np.isnan(row_pn["csr_z_perinuc_cyto_dist"])
    assert row_pn["feat_nuclear_fraction"] == 1.0
    assert row_pn["csr_z_nuclear_frac"] > 3.0

    row_pe = f_nuc.loc[("cell_A", "GENE_EDGE")]
    assert np.isnan(row_pe["bento_nucleus_inner_proximity"])
    assert np.isnan(row_pe["bento_nucleus_inner_asymmetry"])
    assert row_pe["feat_nuclear_fraction"] == 0.0
    assert row_pe["csr_z_cell_edge_dist"] > 1.5

    bundle_no_nuc = {
        "data_df": bundle["data_df"],
        "cell_boundary": bundle["cell_boundary"],
        "nuclear_boundary": None,
    }
    f_nonuc = compute_all_from_pkl(bundle_no_nuc, feature_set="v2_28d", sprawl_kwargs={"processes": 1})
    assert len(f_nonuc) == 4
    assert np.isnan(f_nonuc["csr_z_nuclear_frac"]).all()
    assert np.isnan(f_nonuc["feat_cell_nc_ratio"]).all()
    assert np.isfinite(f_nonuc["csr_z_cell_edge_dist"]).all()


@pytest.mark.skipif(
    not (
        _has_real_binary(default_model_path(DEFAULT_MODEL_V2_FILENAME))
        and _has_real_binary(default_model_path(LEGACY_MODEL_V1_FILENAME))
    ),
    reason="LFS model binaries not pulled",
)
def test_v2_and_v1_end_to_end_inference_and_parquet_fallback():
    bundle = _make_synthetic_bundle()
    v2_model = default_model_path(DEFAULT_MODEL_V2_FILENAME)
    v1_model = default_model_path(LEGACY_MODEL_V1_FILENAME)

    # 1. Default v2_28d end-to-end prediction + scale invariance of probabilities
    pred_1x = run_all_with_patterns(
        bundle,
        model_path=str(v2_model),
        sprawl_kwargs={"processes": 1},
    )
    pred_micron = run_all_with_patterns(
        _scale_and_shift_bundle(bundle, scale=1.0 / 15.0),
        model_path=str(v2_model),
        sprawl_kwargs={"processes": 1},
    )
    assert {"pattern", "pattern_prob", "n_transcripts", "low_abundance"}.issubset(pred_1x.columns)
    assert list(pred_1x["pattern"]) == list(pred_micron["pattern"])
    np.testing.assert_allclose(pred_1x["pattern_prob"], pred_micron["pattern_prob"], rtol=1e-6)

    # 2. Legacy v1_17d model & auto-detection in predict_patterns_from_parquet
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        f_v2_path = tmp / "feat_v2.parquet"
        f_v1_path = tmp / "feat_v1.parquet"
        compute_all_from_pkl(bundle, feature_set="v2_28d", out_path=str(f_v2_path), sprawl_kwargs={"processes": 1})
        compute_all_from_pkl(bundle, feature_set="v1_17d", out_path=str(f_v1_path), sprawl_kwargs={"processes": 1})

        out_v2 = predict_patterns_from_parquet(f_v2_path, tmp / "pred_v2.parquet", model_path="auto")
        out_v1 = predict_patterns_from_parquet(f_v1_path, tmp / "pred_v1.parquet", model_path="auto")
        assert len(out_v2) == len(out_v1) == 4
        assert "low_abundance" in out_v2.columns and "low_abundance" in out_v1.columns
