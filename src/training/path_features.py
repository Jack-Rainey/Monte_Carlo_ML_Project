from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np

PATH_FEATURE_COLUMNS = [
    "listener_direction_x",
    "listener_direction_y",
    "listener_direction_z",
    "distance",
    "arrival_time_s",
    "intensity_band_0",
    "intensity_band_1",
    "intensity_band_2",
    "intensity_band_3",
    "intensity_band_4",
    "intensity_band_5",
    "intensity_band_6",
    "intensity_band_7",
    "energy_sum",
]

LOG1P_COLUMNS = {
    "distance",
    "arrival_time_s",
    "intensity_band_0",
    "intensity_band_1",
    "intensity_band_2",
    "intensity_band_3",
    "intensity_band_4",
    "intensity_band_5",
    "intensity_band_6",
    "intensity_band_7",
    "energy_sum",
}

@dataclass(frozen=True)
class PathFeatureStats:
    mean: np.ndarray
    std: np.ndarray

    def normalize(self, array_k_f: np.ndarray) -> np.ndarray:
        return (array_k_f - self.mean[None, :]) / self.std[None, :]


def _log1p_selected(array_n_f: np.ndarray) -> np.ndarray:
    out = array_n_f.astype(np.float32, copy=True)
    for idx, col in enumerate(PATH_FEATURE_COLUMNS):
        if col in LOG1P_COLUMNS:
            out[:, idx] = np.log1p(np.maximum(out[:, idx], 0.0))
    return out


def load_path_feature_matrix(csv_path: str | Path, *, top_k: int) -> np.ndarray:
    csv_path = Path(csv_path)
    data = np.genfromtxt(csv_path, delimiter=",", names=True, dtype=np.float32, encoding="utf-8")
    if data.shape == ():
        data = np.array([data], dtype=data.dtype)
    n = min(int(data.shape[0]), top_k)
    features = np.zeros((top_k, len(PATH_FEATURE_COLUMNS)), dtype=np.float32)
    for feature_idx, col in enumerate(PATH_FEATURE_COLUMNS):
        features[:n, feature_idx] = np.asarray(data[col][:n], dtype=np.float32)
    return _log1p_selected(features)


def compute_path_feature_stats(csv_paths: list[Path], *, top_k: int, epsilon: float = 1e-6) -> PathFeatureStats:
    running_sum = np.zeros(len(PATH_FEATURE_COLUMNS), dtype=np.float64)
    running_sumsq = np.zeros(len(PATH_FEATURE_COLUMNS), dtype=np.float64)
    total_count = 0
    for csv_path in csv_paths:
        array_k_f = load_path_feature_matrix(csv_path, top_k=top_k).astype(np.float64, copy=False)
        running_sum += array_k_f.sum(axis=0, dtype=np.float64)
        running_sumsq += np.square(array_k_f, dtype=np.float64).sum(axis=0, dtype=np.float64)
        total_count += array_k_f.shape[0]
    mean = running_sum / total_count
    variance = running_sumsq / total_count - np.square(mean)
    variance = np.maximum(variance, epsilon**2)
    std = np.sqrt(variance)
    return PathFeatureStats(mean=mean.astype(np.float32), std=std.astype(np.float32))
