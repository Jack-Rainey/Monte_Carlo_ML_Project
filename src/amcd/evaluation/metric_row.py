"""Per-scene metric-row primitives (design_spec §6).

Every eval metric is expressed as a `(low_val, pred_val, high_ref)` triple — the
baseline (low-ray) value, the model's value, and the high-ray reference. The eval
stage derives the two report quantities uniformly from that triple, so the
"improved" flag is ALWAYS per-metric (it compares each metric's own pred/low
distance to its own high reference) and can never be borrowed from another
metric — the F-07 misattribution this replaces.

`improved` is `None` (not `False`) whenever any leg of the triple is NaN, i.e.
improvement is *undefined* for that scene/metric — the room-acoustic
artifacts-absent case, and diagnostic-only metrics (e.g. energy SNR) that carry a
pred value but no baseline comparison. `stats` counts `n_improved` and the pct
denominator over the SAME non-None population (fixes F-08's >100% pct).
"""
from __future__ import annotations

import math
from typing import NamedTuple


class MetricTriple(NamedTuple):
    """One metric's per-scene values against the high-ray reference.

    low: baseline (low-ray) metric value; pred: model's metric value;
    high: high-ray reference. NaN in any leg means that leg is unavailable
    (e.g. missing decoded/reference IR) → improvement is undefined for the row.
    For a diagnostic-only metric with no baseline comparison, set low=high=NaN.
    """

    low: float
    pred: float
    high: float


def metric_improvement(triple: MetricTriple) -> tuple[bool | None, float]:
    """Derive (improved, baseline_rel_ratio) from a metric triple.

    improved = |pred − high| < |low − high|  — the prediction is closer to the
    high-ray reference than the low-ray baseline is. Returns None when any leg is
    NaN (improvement undefined). baseline_rel_ratio = low_err / (pred_err + eps)
    (> 1 ⟺ improved); NaN when improvement is undefined.
    """
    low, pred, high = triple.low, triple.pred, triple.high
    if math.isnan(low) or math.isnan(pred) or math.isnan(high):
        return None, float("nan")
    low_err = abs(low - high)
    pred_err = abs(pred - high)
    improved = pred_err < low_err
    baseline_rel_ratio = low_err / (pred_err + 1e-10)
    return improved, baseline_rel_ratio
