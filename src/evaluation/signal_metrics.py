from __future__ import annotations

import numpy as np


EPS = 1e-12


def _check_same_shape(reference: np.ndarray, estimate: np.ndarray) -> None:
    if reference.shape != estimate.shape:
        raise ValueError(
            f"Shape mismatch: reference has shape {reference.shape}, "
            f"but estimate has shape {estimate.shape}"
        )


def mse(reference: np.ndarray, estimate: np.ndarray) -> float:
    _check_same_shape(reference, estimate)
    diff = estimate - reference
    return float(np.mean(diff * diff, dtype=np.float64))


def mae(reference: np.ndarray, estimate: np.ndarray) -> float:
    _check_same_shape(reference, estimate)
    return float(np.mean(np.abs(estimate - reference), dtype=np.float64))


def rmse(reference: np.ndarray, estimate: np.ndarray) -> float:
    return float(np.sqrt(mse(reference, estimate)))


def relative_l2(reference: np.ndarray, estimate: np.ndarray) -> float:
    _check_same_shape(reference, estimate)
    numerator = np.linalg.norm((estimate - reference).ravel())
    denominator = np.linalg.norm(reference.ravel()) + EPS
    return float(numerator / denominator)


def snr_db(reference: np.ndarray, estimate: np.ndarray) -> float:
    _check_same_shape(reference, estimate)
    signal_power = np.sum(reference * reference, dtype=np.float64)
    noise_power = np.sum((estimate - reference) ** 2, dtype=np.float64)
    return float(10.0 * np.log10((signal_power + EPS) / (noise_power + EPS)))


def peak_abs_error(reference: np.ndarray, estimate: np.ndarray) -> float:
    _check_same_shape(reference, estimate)
    return float(np.max(np.abs(estimate - reference)))


def compute_signal_metrics(
    *,
    low: np.ndarray,
    pred: np.ndarray,
    high: np.ndarray,
) -> dict[str, float]:
    low_mse = mse(high, low)
    pred_mse = mse(high, pred)

    low_mae = mae(high, low)
    pred_mae = mae(high, pred)

    low_rmse = rmse(high, low)
    pred_rmse = rmse(high, pred)

    low_rel_l2 = relative_l2(high, low)
    pred_rel_l2 = relative_l2(high, pred)

    low_snr = snr_db(high, low)
    pred_snr = snr_db(high, pred)

    low_peak_err = peak_abs_error(high, low)
    pred_peak_err = peak_abs_error(high, pred)

    return {
        "mse_low_vs_high": low_mse,
        "mse_pred_vs_high": pred_mse,
        "mse_improvement_ratio": low_mse / (pred_mse + EPS),

        "mae_low_vs_high": low_mae,
        "mae_pred_vs_high": pred_mae,
        "mae_improvement_ratio": low_mae / (pred_mae + EPS),

        "rmse_low_vs_high": low_rmse,
        "rmse_pred_vs_high": pred_rmse,
        "rmse_improvement_ratio": low_rmse / (pred_rmse + EPS),

        "relative_l2_low_vs_high": low_rel_l2,
        "relative_l2_pred_vs_high": pred_rel_l2,
        "relative_l2_improvement_ratio": low_rel_l2 / (pred_rel_l2 + EPS),

        "snr_low_vs_high_db": low_snr,
        "snr_pred_vs_high_db": pred_snr,
        "snr_improvement_db": pred_snr - low_snr,

        "peak_abs_error_low_vs_high": low_peak_err,
        "peak_abs_error_pred_vs_high": pred_peak_err,
        "peak_abs_error_improvement_ratio": low_peak_err / (pred_peak_err + EPS),
    }