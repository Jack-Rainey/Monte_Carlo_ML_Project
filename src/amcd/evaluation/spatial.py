"""Spatial metrics stub (HOA-decoded directional energy ratio)."""
import torch

from .metric_row import MetricTriple


def compute_spatial_metrics(
    pred_energy: torch.Tensor,
    high_ref: torch.Tensor,
    low_ref: torch.Tensor,
) -> dict[str, MetricTriple]:
    """
    Inter-channel directional energy ratio and HOA-decoded spatial metrics.
    Stubbed — returns empty dict (known gap in v1).
    """
    return {}
