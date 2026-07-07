"""stats stage: bootstrap percentile CI over per-scene metrics."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.integrate
import scipy.stats

from ..config import Config


def bootstrap_ci(
    values: np.ndarray,
    n_resamples: int,
    alpha: float,
    rng: np.random.Generator,
) -> dict[str, float]:
    """Bootstrap percentile CI (robust to non-normal metric distributions).

    All parameters are explicit — no defaults — so the caller (run_stats) always
    supplies config-declared values and a seeded RNG.
    """
    n = len(values)
    if n == 0:
        return {"mean": float("nan"), "ci_lower": float("nan"), "ci_upper": float("nan"), "std": float("nan")}
    if n == 1:
        v = float(values[0])
        return {"mean": v, "ci_lower": v, "ci_upper": v, "std": 0.0}

    boot_means = np.array([
        values[rng.integers(0, n, n)].mean()
        for _ in range(n_resamples)
    ])
    return {
        "mean": float(values.mean()),
        "ci_lower": float(np.percentile(boot_means, 100 * alpha / 2)),
        "ci_upper": float(np.percentile(boot_means, 100 * (1 - alpha / 2))),
        # Sample std (ddof=1): this feeds mdes() as the population-σ estimate, and the
        # unbiased estimator is required at this design's small per-split n — ddof=0
        # biases σ down by √((n−1)/n) (≈ 0.82 at n=3), overstating detectability (F-09).
        "std": float(values.std(ddof=1)),
    }


def _two_sided_power(ncp: float, t_crit: float, df: int) -> float:
    """Two-sided one-sample t-test power at noncentrality `ncp` — computed
    HOLE-FREE so it is safe to bisect over.

    The test statistic under an effect is noncentral-t: T = (Z + ncp)/√(V/df) with
    Z~N(0,1) ⟂ V~χ²_df. scipy's `nct.cdf` returns ISOLATED NaN holes near the root
    at small df, and those holes are dense enough at df≤4 that a fixed nudge cannot
    reliably escape them — a NaN reaching the bisection reads as `nan < power == False`
    and truncates the bracket below the true root, understating MDES (ledger F-14/F-16).

    So instead of `nct.cdf`, integrate the exact rejection probability over the
    chi-square scale mixture: conditional on V=v the statistic is normal, so
    P(reject | v) = Φ(ncp − t_crit·s) + Φ(−t_crit·s − ncp) with s = √(v/df) (both
    tails; the far tail is negligible for ncp>0 but included for exactness). The
    integrand is smooth in v and the result is smooth and monotone increasing in
    ncp (from alpha at 0 to 1), with no NaN — no nudging or clamping needed.
    """
    def integrand(v: float) -> float:
        s = np.sqrt(v / df)
        return (
            scipy.stats.norm.cdf(ncp - t_crit * s)
            + scipy.stats.norm.cdf(-t_crit * s - ncp)
        ) * scipy.stats.chi2.pdf(v, df)

    # Integrate over the chi-square's actual support, not [0, ∞): at large df the
    # density is a narrow spike around v=df that quad's infinite-interval transform
    # samples right past (→ NaN). Bounding by extreme quantiles guarantees quad sees
    # the mass; the truncated tails carry <2e-12 of the probability, negligible here.
    lo_v = float(scipy.stats.chi2.ppf(1e-12, df))
    hi_v = float(scipy.stats.chi2.ppf(1 - 1e-12, df))
    value, _ = scipy.integrate.quad(integrand, lo_v, hi_v)
    return float(value)


def mdes(std: float, n: int, power: float, alpha: float) -> float:
    """Minimum detectable effect size for a two-sided one-sample t-test.

    The MDES is the mean effect δ whose test achieves `power` at significance
    `alpha` with n samples (ν = n−1 df). Solved EXACTLY via the noncentral-t: the
    t-statistic under an effect δ is noncentral-t with noncentrality λ = δ√n/σ, so
    we find the λ whose two-sided power equals `power` and return δ = λσ/√n. Power
    is evaluated hole-free by `_two_sided_power` (see there re ledger F-14/F-16).

    Why not the large-sample normal form `(z_{1-α/2}+z_power)·σ/√n`: at this design's
    per-split n (test splits are as small as 3) the normal critical value understates
    the small-sample one ~2.2× at n=3, making MDES ~2.2× too small — overstating
    detectability (ledger F-13). The noncentral-t is correct at every n and converges
    to the normal form as n grows.

    Returns NaN for n ≤ 2 (df ≤ 1): a one-sample t-test with a single degree of
    freedom is essentially unpowered (MDES ≈ 11σ), so MDES is reported N/A there
    rather than as a misleadingly precise huge number (ledger F-14)."""
    if n <= 2 or std == 0:
        return float("nan")
    df = n - 1
    t_crit = scipy.stats.t.ppf(1 - alpha / 2, df)

    # Grow the bracket until power ≥ target (monotone ⇒ this brackets the root).
    hi = 8.0
    while _two_sided_power(hi, t_crit, df) < power:
        hi *= 2.0
        if hi > 1e6:
            return float("nan")  # target power unreachable (degenerate n/σ or power≤alpha)
    # Bisection over the smooth monotone power (no NaN to defend against now).
    lo = 0.0
    for _ in range(200):
        if hi - lo < 1e-9:
            break
        mid = 0.5 * (lo + hi)
        if _two_sided_power(mid, t_crit, df) < power:
            lo = mid
        else:
            hi = mid
    ncp = 0.5 * (lo + hi)
    return float(ncp * std / np.sqrt(n))


def run_stats(config: Config, run_dir: Path) -> None:
    metrics_path = run_dir / "metrics" / "metrics.parquet"
    if not metrics_path.exists():
        raise FileNotFoundError(f"No metrics found at {metrics_path}. Run eval first.")

    stats_dir = run_dir / "stats"
    stats_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(metrics_path)
    rng = np.random.default_rng(config.seed("bootstrap"))

    # Compute per-(split, metric) stats — never pool test splits (invariant #9).
    summary: list[dict] = []
    for (split_name, metric_name), group in df.groupby(["split", "metric"]):
        values = group["pred_val"].dropna().values.astype(float)
        ci = bootstrap_ci(
            values,
            n_resamples=config.bootstrap_n_resamples,
            alpha=config.bootstrap_alpha,
            rng=rng,
        )
        mdes_val = mdes(ci["std"], len(values), power=config.bootstrap_power, alpha=config.bootstrap_alpha)

        # Improvement pct: numerator and denominator over the SAME population —
        # scenes where `improved` is defined (not None/NaN). Counting n_improved over
        # rows dropped from the denominator produced pct > 100 (F-08). `n_scored` may
        # differ from `n` (the CI/MDES pred_val count) when a metric's improvement is
        # undefined but its pred value is not, so it is reported as its own column.
        improved = group["improved"]
        scored = improved.notna()
        n_scored = int(scored.sum())
        n_improved = int((improved[scored] == True).sum())  # noqa: E712

        row = {
            "split": split_name,
            "metric": metric_name,
            "n": len(values),
            "mean": ci["mean"],
            "ci_lower": ci["ci_lower"],
            "ci_upper": ci["ci_upper"],
            "std": ci["std"],
            "mdes_80pct": mdes_val,
            "n_scored": n_scored,
            "n_improved": n_improved,
            "pct_improved": float(n_improved) / n_scored * 100 if n_scored > 0 else float("nan"),
        }
        summary.append(row)

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(stats_dir / "ci_table.csv", index=False)

    # JSON-serialisable summary
    (stats_dir / "summary.json").write_text(
        json.dumps(summary_df.to_dict(orient="records"), indent=2)
    )

    n_splits = summary_df["split"].nunique() if not summary_df.empty else 0
    n_metrics = summary_df["metric"].nunique() if not summary_df.empty else 0
    print(f"  Stats for {n_metrics} metrics × {n_splits} splits → {stats_dir}")
