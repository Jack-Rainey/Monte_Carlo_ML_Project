"""Per-scene metric-row primitives (design_spec §6).

Every eval metric is expressed as a `MetricTriple` — the baseline (low-ray)
value, the model's value, the high-ray reference — plus the metric's declared
improvement `kind`. The eval stage derives the two report quantities uniformly
from the triple, so the "improved" flag is ALWAYS per-metric (it compares each
metric's own legs) and can never be borrowed from another metric.

`kind` is the explicit contract the eval/stats spine branches on. The
spine previously ASSUMED every metric was match-reference, which silently
unscored any metric whose reference leg is unreachable under that framing
(SNR of the high-ray reference against itself is +∞):

- ``match_reference`` — the prediction should land closer to the high-ray
  reference than the baseline does; consumes all three legs.
  improved ⟺ |pred − high| < |low − high|.
- ``maximize`` — higher is better; consumes only (low, pred), the high leg is
  structurally absent (NaN) and is never counted as a drop.
  improved ⟺ pred > low.
- ``minimize`` — lower is better; mirror of maximize. improved ⟺ pred < low.

``maximize`` currently has one filler (`energy_snr_db`); ``minimize`` none yet.
Both are the seam for the roadmap's higher-is-better perceptual and
lower-is-better spatial-error metrics (research_I_paper §6) — forward-looking
design, not dead generality.

`improved` is `None` (not `False`) whenever a leg the kind relies on is NaN,
i.e. improvement is *undefined* for that scene/metric — e.g. the room-acoustic
artifacts-absent case. `stats` counts `n_improved` and the pct denominator over
the SAME non-None population, so the percentage cannot exceed 100. Every such
NaN leg is logged with a reason to `metrics/drops.csv` by the evaluator — no
silent exclusion.
"""
from __future__ import annotations

import math
from typing import NamedTuple

METRIC_KINDS = ("match_reference", "maximize", "minimize")

# Legs a kind's improvement actually consumes. The evaluator uses this to tell a
# structurally-absent leg (e.g. `high` under maximize — never a drop) from a
# dropped one (NaN in a consumed leg — must be logged with a reason).
KIND_LEGS: dict[str, tuple[str, ...]] = {
    "match_reference": ("low", "pred", "high"),
    "maximize": ("low", "pred"),
    "minimize": ("low", "pred"),
}


class MetricTriple(NamedTuple):
    """One metric's per-scene values against the low/high references, plus its
    declared improvement kind.

    low: baseline (low-ray) metric value; pred: model's value; high: high-ray
    reference (NaN when the kind does not consume it). kind: one of
    `METRIC_KINDS`, declared HERE at the metric's definition site — required, no
    default, so a producer can never fall back to an implicit match-reference
    assumption. `unit` is declared at the same place and for the same
    reason. NaN in a consumed leg means that leg is unavailable →
    improvement is undefined for the row and the evaluator logs the drop.
    """

    low: float
    pred: float
    high: float
    kind: str
    #: The metric's physical unit, declared HERE at the producer for the same
    #: reason `kind` is: a second declaration in the reporting layer would assert
    #: what another module's numbers mean with nothing binding the two, so a
    #: metric could change domain while the Unit column kept printing the old one.
    #: `""` is legal and means dimensionless; a unit that depends on the
    #: representation's operand domain (an MSE) is rendered by the reporting layer
    #: from the stamped `value_domain`, which only that layer knows.
    unit: str


class MetricDrop(NamedTuple):
    """One unscored (or partially computed) metric leg: which metric, which leg,
    why. The evaluator adds (scene, split) and writes the full record to
    `metrics/drops.csv` — nothing leaves a result silently."""

    metric: str
    leg: str
    reason: str


def _consumed_legs(triple: MetricTriple) -> tuple[float, ...] | None:
    """Values of the legs `triple.kind` consumes, or None if any is NaN.
    Raises ValueError on an undeclared kind — fail loud, never guess."""
    if triple.kind not in METRIC_KINDS:
        raise ValueError(
            f"Unknown metric kind {triple.kind!r}; expected one of {METRIC_KINDS}."
        )
    values = tuple(getattr(triple, leg) for leg in KIND_LEGS[triple.kind])
    if any(math.isnan(v) for v in values):
        return None
    return values


def metric_improvement(triple: MetricTriple) -> tuple[bool | None, float]:
    """Derive (improved, baseline_rel_ratio) from a metric triple, per its kind.

    improved (see kind semantics in the module docstring) is None when any
    consumed leg is NaN — improvement undefined for that scene/metric.
    baseline_rel_ratio = low_err / (pred_err + eps) (> 1 ⟺ improved) is a
    match-reference-only diagnostic — the other kinds have no reference-error
    pair — and is NaN for them and wherever improved is None.
    """
    if _consumed_legs(triple) is None:
        return None, float("nan")
    low, pred, high, kind = triple.low, triple.pred, triple.high, triple.kind
    if kind == "match_reference":
        low_err = abs(low - high)
        pred_err = abs(pred - high)
        return pred_err < low_err, low_err / (pred_err + 1e-10)
    improved = (pred > low) if kind == "maximize" else (pred < low)
    return improved, float("nan")


def paired_improvement(triple: MetricTriple) -> float:
    """The design_spec §9 per-scene PAIRED improvement — the quantity `stats`
    runs CI/MDES on. Positive ⟺ improved (sign-consistency with
    `metric_improvement` is pinned in tests); NaN whenever improvement is
    undefined (same consumed-legs rule).

    match_reference → |low − high| − |pred − high|;
    maximize → pred − low;  minimize → low − pred.
    """
    if _consumed_legs(triple) is None:
        return float("nan")
    low, pred, high, kind = triple.low, triple.pred, triple.high, triple.kind
    if kind == "match_reference":
        return abs(low - high) - abs(pred - high)
    return (pred - low) if kind == "maximize" else (low - pred)
