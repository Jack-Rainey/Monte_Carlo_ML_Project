"""Perceptual metrics stub (for controlled listening study)."""
import torch

from .metric_row import MetricTriple


def compute_perceptual_metrics(
    pred_energy: torch.Tensor,
    high_ref: torch.Tensor,
    low_ref: torch.Tensor,
) -> tuple[dict[str, MetricTriple], dict[tuple[str, str], str]]:
    """
    Perceptual metrics for preview rendering (future listening study).
    Stubbed — returns no metrics. Producer contract: (triples with a declared
    improvement `kind` each, nan_reasons keyed (metric, leg)) — see metric_row.
    """
    return {}, {}
