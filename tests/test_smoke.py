"""Smoke tests: import + CLI --help for public entry points.

No data files needed, runs in seconds, safe for CI.
"""
import subprocess
import sys


def test_import_tools():
    import tools
    assert hasattr(tools, "__path__")


def test_import_api():
    from tools.api import compute_all, patterns, colocalization, compartments


def test_import_engines():
    from tools.engines import (
        bento_adapter,
        compartment_redisplay,
        compartment_same_cell_fast,
        csr_features,
        instant_adapter,
        sprawl_adapter,
    )


def _run_help(cmd: str):
    result = subprocess.run(
        [sys.executable, "-m", cmd, "--help"],
        capture_output=True, text=True, timeout=180,
    )
    assert result.returncode == 0, f"{cmd} --help failed:\n{result.stderr}"
    assert "usage" in result.stdout.lower() or "--help" in result.stdout.lower()


def test_cli_subcellfeat_help():
    _run_help("tools.cli.run")


def test_cli_subcellfeat_fast_requires_features_only():
    """Fast mode should fail before loading data when pattern prediction is requested."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "tools.cli.run",
            "--pkl",
            "missing.pkl",
            "--out",
            "missing.parquet",
            "--fast",
        ],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode != 0
    assert "--fast can only be used with --features-only" in result.stderr
    assert "missing.pkl" not in result.stderr


def test_cli_pattern_help():
    _run_help("tools.cli.pattern")


def test_cli_compartment_help():
    _run_help("tools.cli.compartments")


def test_cli_compartment_redisplay_help():
    _run_help("tools.cli.compartment_redisplay")


def test_cli_coloc_help():
    _run_help("tools.cli.colocalization")


def test_cli_coloc_plot_help():
    _run_help("tools.cli.coloc_plot")


def test_cli_coloc_summary_help():
    _run_help("tools.cli.coloc_summary")


def test_lfs_detection():
    """PatternClassifier should detect LFS pointer files."""
    import tempfile
    from pathlib import Path
    from tools.models.pattern_classifier import PatternClassifier

    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as f:
        f.write(b"version https://git-lfs.github.com/spec/v1\n"
                b"oid sha256:abc123\nsize 12345\n")
        f.flush()
        try:
            PatternClassifier(f.name)
            assert False, "Should have raised RuntimeError for LFS pointer"
        except RuntimeError as e:
            assert "git lfs pull" in str(e).lower()
        finally:
            Path(f.name).unlink()


def test_compartment_device_cpu():
    """Compartment adapter should resolve an explicit CPU device."""
    from tools.engines.compartment_adapter import _device_from_arg
    device = _device_from_arg("cpu")
    assert str(device) == "cpu"
