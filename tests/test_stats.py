"""Stats-stage numerics (ledger F-09): MDES must use the UNBIASED sample std.

MDES (minimum detectable effect size) is the load-bearing "could we even have seen
an effect this small?" number at this design's small per-split n. Feeding it a
population std (ddof=0) biases σ down by √((n−1)/n) — ≈0.82 at n=3 — which
overstates detectability. These pin the sample-std (ddof=1) path.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.integrate
import scipy.stats

from amcd.stats.aggregate import bootstrap_ci, mdes, run_stats

from tests.conftest import tiny_config


def _achieved_power(ncp: float, df: int, t_crit: float) -> float:
    """Independent oracle: two-sided noncentral-t power at `ncp`, NOT reusing the
    module's bisection. Prefer scipy's `nct` where it is finite (a genuinely
    independent implementation); fall back to the chi-square-mixture integral in the
    small-df hole zone where `nct.cdf` returns NaN (this is exactly the F-16 regime
    the raw scipy recompute could not check)."""
    p = scipy.stats.nct.sf(t_crit, df, ncp) + scipy.stats.nct.cdf(-t_crit, df, ncp)
    if np.isfinite(p):
        return float(p)
    integrand = lambda v: (  # noqa: E731
        scipy.stats.norm.cdf(ncp - t_crit * np.sqrt(v / df))
        + scipy.stats.norm.cdf(-t_crit * np.sqrt(v / df) - ncp)
    ) * scipy.stats.chi2.pdf(v, df)
    return float(scipy.integrate.quad(integrand, 0.0, np.inf)[0])


def test_bootstrap_ci_reports_sample_std_not_population() -> None:
    vals = np.array([1.0, 2.0, 4.0])
    ci = bootstrap_ci(vals, n_resamples=200, alpha=0.05, rng=np.random.default_rng(0))
    assert ci["std"] == pytest.approx(float(np.std(vals, ddof=1)))
    # The pre-fix population std (ddof=0) is smaller — guard against its return.
    assert ci["std"] != pytest.approx(float(np.std(vals, ddof=0)))


@pytest.mark.parametrize(
    "n, power, alpha",
    [
        (3, 0.8, 0.05),   # smallest shipped split, base config
        (4, 0.8, 0.05),
        (5, 0.8, 0.05),
        (8, 0.8, 0.05),
        # F-16 hole zone: at low alpha / high power the root sits where scipy's
        # nct.cdf is NaN, so the old nudge-then-bisect solver truncated the bracket
        # below the root and understated MDES. Plausible as a multiple-comparison
        # correction across the 6 splits. The hole-free evaluator must nail it.
        (3, 0.95, 0.01),
        (3, 0.99, 0.001),
    ],
)
def test_mdes_achieves_target_power_via_noncentral_t(n: int, power: float, alpha: float) -> None:
    """F-13/F-16: for n≥3 (df≥2) MDES is the EXACT small-sample effect — plugging its
    noncentrality back into the two-sided noncentral-t power reproduces the target
    power. Non-tautological: `_achieved_power` recomputes power from the returned
    effect independently of the bisection that found it (scipy `nct` where finite, a
    separate integral in the hole zone)."""
    std_sample = 1.5
    d = mdes(std_sample, n, power, alpha)
    assert np.isfinite(d) and d > 0, f"mdes non-finite/≤0 at n={n}, power={power}, alpha={alpha}"

    df = n - 1
    ncp = d * np.sqrt(n) / std_sample
    t_crit = scipy.stats.t.ppf(1 - alpha / 2, df)
    achieved = _achieved_power(ncp, df, t_crit)
    assert achieved == pytest.approx(power, abs=1e-4), (
        f"n={n}, power={power}, alpha={alpha}: MDES effect achieves power {achieved:.5f} ≠ target"
    )


def test_mdes_returns_nan_for_n2_df1() -> None:
    """F-14: at n=2 (df=1) the earlier clamp-to-1.0 solver returned a finite,
    ~2×-too-small MDES (overstating detectability). MDES must instead be N/A there —
    an unpowered 1-df test — never a finite value that misses target power."""
    assert np.isnan(mdes(1.5, 2, 0.8, 0.05))
    assert np.isnan(mdes(1.5, 1, 0.8, 0.05))  # n≤1 unchanged guard


def test_mdes_exceeds_large_sample_z_form_at_small_n() -> None:
    """The old large-sample z-form understated MDES ~2× at n=3; the noncentral-t
    fix must exceed it clearly, so the approximation is no longer what we report."""
    std_sample, n, power, alpha = 1.5, 3, 0.8, 0.05
    d = mdes(std_sample, n, power, alpha)
    z_form = (scipy.stats.norm.ppf(1 - alpha / 2) + scipy.stats.norm.ppf(power)) * std_sample / np.sqrt(n)
    assert d > 1.9 * z_form


def test_mdes_converges_to_normal_form_at_large_n() -> None:
    """Sanity: as n grows the noncentral-t MDES approaches the large-sample z-form
    (the approximation is only wrong at small n)."""
    std_sample, n, power, alpha = 1.0, 500, 0.8, 0.05
    d = mdes(std_sample, n, power, alpha)
    z_form = (scipy.stats.norm.ppf(1 - alpha / 2) + scipy.stats.norm.ppf(power)) * std_sample / np.sqrt(n)
    assert d == pytest.approx(z_form, rel=0.02)


def test_run_stats_mdes_uses_sample_std_and_noncentral_t() -> None:
    """End-to-end: run_stats reports std = the group SAMPLE std (ddof=1, F-09) and
    mdes_80pct = the exact noncentral-t MDES (F-13) on that std, not the z-form."""
    cfg = tiny_config()
    vals = [0.20, 0.35, 0.55]
    rows = [
        {"scene_id": f"s_{i}", "split": "test_id", "metric": "T30",
         "low_val": 0.3, "pred_val": v, "high_ref": 0.4,
         "baseline_rel_ratio": 1.0, "improved": True}
        for i, v in enumerate(vals)
    ]
    with tempfile.TemporaryDirectory() as d:
        run_dir = Path(d)
        (run_dir / "metrics").mkdir()
        pd.DataFrame(rows).to_parquet(run_dir / "metrics" / "metrics.parquet", index=False)
        run_stats(cfg, run_dir)
        ci = pd.read_csv(run_dir / "stats" / "ci_table.csv")

    row = ci[ci["metric"] == "T30"].iloc[0]
    sample_std = float(np.std(vals, ddof=1))
    assert row["std"] == pytest.approx(sample_std)
    assert row["mdes_80pct"] == pytest.approx(
        mdes(sample_std, len(vals), cfg.bootstrap_power, cfg.bootstrap_alpha)
    )
