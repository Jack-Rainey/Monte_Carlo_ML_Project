"""Spatial metrics stub (HOA-decoded directional energy ratio)."""
import torch

from .metric_row import MetricTriple


def compute_spatial_metrics(
    pred_energy: torch.Tensor,
    high_ref: torch.Tensor,
    low_ref: torch.Tensor,
) -> tuple[dict[str, MetricTriple], dict[tuple[str, str], str]]:
    """
    Inter-channel directional energy ratio and HOA-decoded spatial metrics.
    Stubbed — returns no metrics (known gap in v1). Producer contract: (triples
    with a declared improvement `kind` each, nan_reasons keyed (metric, leg)) —
    see metric_row (F-20/F-21).
    """
    return {}, {}
