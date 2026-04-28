from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


EPS = 1e-20


@dataclass(frozen=True)
class DecayFitResult:
    slope_db_per_s: float
    intercept_db: float
    r2: float
    sample_count: int


def hoa_w_channel(hoa: np.ndarray, channel_index: int = 0) -> np.ndarray:
    """
    Extract the channel used for scalar room-acoustic metrics.

    Expected HOA layout is channel-first: (C, T).
    For the current pipeline, channel 0 is treated as the omnidirectional/W-like channel.
    """
    if hoa.ndim != 2:
        raise ValueError(f"Expected HOA array of shape (C, T), got {hoa.shape}")
    if not (0 <= channel_index < hoa.shape[0]):
        raise ValueError(f"Invalid channel_index={channel_index} for HOA shape {hoa.shape}")
    return hoa[channel_index].astype(np.float64, copy=False)


def energy_decay_curve_db(ir: np.ndarray) -> np.ndarray:
    """
    Compute normalized Schroeder-style reverse integrated energy decay curve in dB.
    """
    ir = np.asarray(ir, dtype=np.float64)
    energy = ir * ir
    edc = np.cumsum(energy[::-1])[::-1]
    edc = edc / max(float(edc[0]), EPS)
    return 10.0 * np.log10(np.maximum(edc, EPS))


def _linear_fit(x: np.ndarray, y: np.ndarray) -> DecayFitResult | None:
    if x.size < 2:
        return None

    slope, intercept = np.polyfit(x, y, deg=1)
    y_pred = slope * x + intercept

    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > EPS else math.nan

    return DecayFitResult(
        slope_db_per_s=float(slope),
        intercept_db=float(intercept),
        r2=float(r2),
        sample_count=int(x.size),
    )


def fit_decay_range(
    edc_db: np.ndarray,
    sample_rate_hz: int,
    upper_db: float,
    lower_db: float,
) -> DecayFitResult | None:
    """
    Fit EDC decay over a dB range, e.g. -5 to -25 dB for T20.
    """
    if upper_db <= lower_db:
        raise ValueError("upper_db should be less negative than lower_db")

    mask = (edc_db <= upper_db) & (edc_db >= lower_db)
    indices = np.nonzero(mask)[0]

    if indices.size < 2:
        return None

    times = indices.astype(np.float64) / float(sample_rate_hz)
    values = edc_db[indices]
    return _linear_fit(times, values)


def decay_time_from_fit(fit: DecayFitResult | None) -> float:
    """
    Convert decay slope to extrapolated T60.

    Since slope is in dB/s and should be negative, T60 = -60 / slope.
    """
    if fit is None:
        return math.nan
    if fit.slope_db_per_s >= 0.0:
        return math.nan
    return float(-60.0 / fit.slope_db_per_s)


def edt_seconds(edc_db: np.ndarray, sample_rate_hz: int) -> float:
    """
    EDT is estimated from the initial -0 to -10 dB decay range,
    extrapolated to 60 dB.
    """
    fit = fit_decay_range(edc_db, sample_rate_hz, upper_db=0.0, lower_db=-10.0)
    return decay_time_from_fit(fit)


def t20_seconds(edc_db: np.ndarray, sample_rate_hz: int) -> float:
    """
    T20 uses the -5 to -25 dB decay range, extrapolated to 60 dB.
    """
    fit = fit_decay_range(edc_db, sample_rate_hz, upper_db=-5.0, lower_db=-25.0)
    return decay_time_from_fit(fit)


def t30_seconds(edc_db: np.ndarray, sample_rate_hz: int) -> float:
    """
    T30 uses the -5 to -35 dB decay range, extrapolated to 60 dB.
    """
    fit = fit_decay_range(edc_db, sample_rate_hz, upper_db=-5.0, lower_db=-35.0)
    return decay_time_from_fit(fit)


def _energy_before_after(ir: np.ndarray, sample_rate_hz: int, boundary_s: float) -> tuple[float, float]:
    boundary = int(round(boundary_s * sample_rate_hz))
    boundary = max(0, min(boundary, ir.shape[0]))

    energy = ir * ir
    early = float(np.sum(energy[:boundary], dtype=np.float64))
    late = float(np.sum(energy[boundary:], dtype=np.float64))
    return early, late


def clarity_db(ir: np.ndarray, sample_rate_hz: int, boundary_s: float) -> float:
    early, late = _energy_before_after(ir, sample_rate_hz, boundary_s)
    return float(10.0 * np.log10((early + EPS) / (late + EPS)))


def definition_percent(ir: np.ndarray, sample_rate_hz: int, boundary_s: float = 0.050) -> float:
    early, late = _energy_before_after(ir, sample_rate_hz, boundary_s)
    return float(100.0 * early / (early + late + EPS))


def center_time_seconds(ir: np.ndarray, sample_rate_hz: int) -> float:
    energy = ir * ir
    times = np.arange(ir.shape[0], dtype=np.float64) / float(sample_rate_hz)
    return float(np.sum(times * energy, dtype=np.float64) / (np.sum(energy, dtype=np.float64) + EPS))


def compute_scalar_acoustic_metrics(
    hoa: np.ndarray,
    *,
    sample_rate_hz: int,
    channel_index: int = 0,
) -> dict[str, float]:
    """
    Compute first-pass scalar acoustic metrics from one HOA channel.

    This intentionally avoids spatial metrics until the project defines a spatial decoding
    or multichannel analysis procedure.
    """
    ir = hoa_w_channel(hoa, channel_index=channel_index)
    edc_db = energy_decay_curve_db(ir)

    t20_fit = fit_decay_range(edc_db, sample_rate_hz, upper_db=-5.0, lower_db=-25.0)
    t30_fit = fit_decay_range(edc_db, sample_rate_hz, upper_db=-5.0, lower_db=-35.0)

    return {
        "edt_s": edt_seconds(edc_db, sample_rate_hz),
        "t20_s": t20_seconds(edc_db, sample_rate_hz),
        "t30_s": t30_seconds(edc_db, sample_rate_hz),
        "c50_db": clarity_db(ir, sample_rate_hz, boundary_s=0.050),
        "c80_db": clarity_db(ir, sample_rate_hz, boundary_s=0.080),
        "d50_percent": definition_percent(ir, sample_rate_hz, boundary_s=0.050),
        "ts_s": center_time_seconds(ir, sample_rate_hz),
        "t20_fit_r2": math.nan if t20_fit is None else t20_fit.r2,
        "t30_fit_r2": math.nan if t30_fit is None else t30_fit.r2,
        "t20_fit_sample_count": 0 if t20_fit is None else t20_fit.sample_count,
        "t30_fit_sample_count": 0 if t30_fit is None else t30_fit.sample_count,
    }


def metric_errors(
    *,
    low_metrics: dict[str, float],
    pred_metrics: dict[str, float],
    high_metrics: dict[str, float],
) -> dict[str, float]:
    rows: dict[str, float] = {}

    diagnostic_metric_names = {
        "t20_fit_r2",
        "t30_fit_r2",
        "t20_fit_sample_count",
        "t30_fit_sample_count",
    }

    for name, high_value in high_metrics.items():
        low_value = low_metrics[name]
        pred_value = pred_metrics[name]

        rows[f"{name}_low"] = low_value
        rows[f"{name}_pred"] = pred_value
        rows[f"{name}_high"] = high_value

        # These are quality-control diagnostics for the decay fit, not acoustic
        # metrics whose errors should be interpreted as denoising performance.
        if name in diagnostic_metric_names:
            continue

        rows[f"{name}_abs_error_low_vs_high"] = abs(low_value - high_value)
        rows[f"{name}_abs_error_pred_vs_high"] = abs(pred_value - high_value)
        rows[f"{name}_improvement"] = (
            abs(low_value - high_value) - abs(pred_value - high_value)
        )

    return rows