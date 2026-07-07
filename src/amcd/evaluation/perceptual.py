"""Perceptual metrics stub (for controlled listening study)."""
import torch

from .metric_row import MetricTriple


def compute_perceptual_metrics(
    pred_energy: torch.Tensor,
    high_ref: torch.Tensor,
    low_ref: torch.Tensor,
) -> dict[str, MetricTriple]:
    """
    Perceptual metrics for preview rendering (future listening study).
    Stubbed — returns empty dict.
    """
    return {}
