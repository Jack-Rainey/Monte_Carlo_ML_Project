from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import math

import numpy as np

from .config_schema import QcConfig
from .metrics import detect_onset_sample, drr_like_db, early_late_energy, total_energy
from .scene_spec import SceneSpec


@dataclass(frozen=True)
class QcResult:
    passed: bool
    issues: list[str]
    metrics: dict[str, float | int | str | bool]

    def write_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(
                {"passed": self.passed, "issues": self.issues, "metrics": self.metrics},
                handle,
                indent=2,
                sort_keys=True,
            )


def _expected_onset_sample(scene: SceneSpec) -> int:
    speed_of_sound_mps = 343.0
    seconds = scene.placement.source_receiver_distance_m / speed_of_sound_mps
    return int(round(seconds * scene.simulation.sample_rate_hz))


def run_qc(scene: SceneSpec, low_hoa: np.ndarray, high_hoa: np.ndarray, paths_file: Path, cfg: QcConfig) -> QcResult:
    issues: list[str] = []
    expected_channels = scene.simulation.expected_num_channels
    expected_samples = scene.simulation.expected_num_samples

    for label, array in (("low", low_hoa), ("high", high_hoa)):
        if array.ndim != 2:
            issues.append(f"{label}_hoa must be rank-2, got shape={array.shape}")
            continue
        if array.shape != (expected_channels, expected_samples):
            issues.append(
                f"{label}_hoa shape mismatch: expected {(expected_channels, expected_samples)} got {array.shape}"
            )
        if not np.isfinite(array).all():
            issues.append(f"{label}_hoa contains NaN or inf")
        if total_energy(array) < cfg.min_total_energy:
            issues.append(f"{label}_hoa total energy below floor")

    expected_onset = _expected_onset_sample(scene)
    observed_low = detect_onset_sample(low_hoa[0])
    observed_high = detect_onset_sample(high_hoa[0])
    onset_error_low_ms = abs(observed_low - expected_onset) / scene.simulation.sample_rate_hz * 1000.0
    onset_error_high_ms = abs(observed_high - expected_onset) / scene.simulation.sample_rate_hz * 1000.0
    if onset_error_low_ms > cfg.max_onset_error_ms:
        issues.append(f"low onset error too large: {onset_error_low_ms:.3f} ms")
    if onset_error_high_ms > cfg.max_onset_error_ms:
        issues.append(f"high onset error too large: {onset_error_high_ms:.3f} ms")

    if cfg.require_non_empty_paths:
        if (not paths_file.exists()) or paths_file.stat().st_size <= 0:
            issues.append("paths file missing or empty")

    if paths_file.exists():
        size_mb = paths_file.stat().st_size / (1024.0 * 1024.0)
        if size_mb > cfg.max_retained_paths_file_size_mb:
            issues.append(f"paths file too large: {size_mb:.3f} MB")
    else:
        size_mb = math.nan

    low_early, low_late = early_late_energy(low_hoa, scene.simulation.sample_rate_hz)
    high_early, high_late = early_late_energy(high_hoa, scene.simulation.sample_rate_hz)
    metrics = {
        "expected_onset_sample": expected_onset,
        "observed_low_onset_sample": observed_low,
        "observed_high_onset_sample": observed_high,
        "low_onset_error_ms": onset_error_low_ms,
        "high_onset_error_ms": onset_error_high_ms,
        "low_total_energy": total_energy(low_hoa),
        "high_total_energy": total_energy(high_hoa),
        "low_early_energy": low_early,
        "low_late_energy": low_late,
        "high_early_energy": high_early,
        "high_late_energy": high_late,
        "low_drr_like_db": drr_like_db(low_hoa[0], scene.simulation.sample_rate_hz),
        "high_drr_like_db": drr_like_db(high_hoa[0], scene.simulation.sample_rate_hz),
        "paths_file_size_mb": size_mb,
    }
    return QcResult(passed=not issues, issues=issues, metrics=metrics)
