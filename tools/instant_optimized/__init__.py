"""Isolated, result-equivalent optimization of the bundled InSTAnT workflow."""

from .core import SparsePPResults, run_conditional_global_colocalization, run_proximal_pairs

__all__ = [
    "SparsePPResults",
    "run_conditional_global_colocalization",
    "run_proximal_pairs",
]
