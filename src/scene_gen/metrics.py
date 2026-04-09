from __future__ import annotations

from pathlib import Path
import json

import numpy as np


def detect_onset_sample(signal: np.ndarray) -> int:
    abs_signal = np.abs(signal)
    peak = float(abs_signal.max())
    if peak <= 0.0:
        return 0
    threshold = max(peak * 0.05, 1e-12)
    indices = np.flatnonzero(abs_signal >= threshold)
    return int(indices[0]) if indices.size else 0


def total_energy(signal: np.ndarray) -> float:
    return float(np.sum(np.square(signal, dtype=np.float64), dtype=np.float64))


def early_late_energy(signal: np.ndarray, sample_rate_hz: int, split_ms: float = 80.0) -> tuple[float, float]:
    split_idx = int(round(sample_rate_hz * split_ms / 1000.0))
    early = float(np.sum(np.square(signal[..., :split_idx], dtype=np.float64), dtype=np.float64))
    late = float(np.sum(np.square(signal[..., split_idx:], dtype=np.float64), dtype=np.float64))
    return early, late


def drr_like_db(signal: np.ndarray, sample_rate_hz: int) -> float:
    onset = detect_onset_sample(signal)
    direct_window = min(signal.shape[-1], onset + int(round(sample_rate_hz * 0.005)))
    direct = float(np.sum(np.square(signal[..., onset:direct_window], dtype=np.float64), dtype=np.float64))
    reverberant = float(np.sum(np.square(signal[..., direct_window:], dtype=np.float64), dtype=np.float64))
    if reverberant <= 0.0:
        return float("inf")
    if direct <= 0.0:
        return float("-inf")
    return float(10.0 * np.log10(direct / reverberant))


def write_metrics(metrics: dict, path: str | Path) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
