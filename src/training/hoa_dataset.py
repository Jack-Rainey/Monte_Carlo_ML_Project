from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import math
import random

import numpy as np
import tensorflow as tf


@dataclass(frozen=True)
class ScenePairRecord:
    scene_id: str
    split_name: str
    low_path: Path
    high_path: Path


@dataclass(frozen=True)
class ChannelStats:
    mean: np.ndarray
    std: np.ndarray

    def normalize(self, array_ch_t: np.ndarray) -> np.ndarray:
        return (array_ch_t - self.mean[:, None]) / self.std[:, None]

    def denormalize(self, array_ch_t: np.ndarray) -> np.ndarray:
        return array_ch_t * self.std[:, None] + self.mean[:, None]


@dataclass(frozen=True)
class DatasetSpec:
    dataset_name: str
    metadata_root_dir: Path
    raw_data_root_dir: Path
    sample_rate_hz: int
    ir_duration_s: float
    hoa_order: int
    dtype: str

    @property
    def expected_num_channels(self) -> int:
        return (self.hoa_order + 1) ** 2

    @property
    def expected_num_samples(self) -> int:
        return int(round(self.sample_rate_hz * self.ir_duration_s))


DEFAULT_EPSILON = 1e-6


def load_dataset_spec(project_root: str | Path, dataset_spec_path: str | Path) -> DatasetSpec:
    project_root = Path(project_root)
    dataset_spec_path = Path(dataset_spec_path)
    with dataset_spec_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    simulation = config["simulation"]
    paths = config["paths"]
    return DatasetSpec(
        dataset_name=config["dataset_name"],
        metadata_root_dir=project_root / paths["metadata_root_dir"],
        raw_data_root_dir=project_root / paths["raw_data_root_dir"],
        sample_rate_hz=int(simulation["sample_rate_hz"]),
        ir_duration_s=float(simulation["ir_duration_s"]),
        hoa_order=int(simulation["hoa_order"]),
        dtype=str(simulation.get("dtype", "float32")),
    )


def discover_records(
    project_root: str | Path,
    dataset_spec_path: str | Path,
    split_name: str,
    *,
    limit: int | None = None,
) -> list[ScenePairRecord]:
    project_root = Path(project_root)
    spec = load_dataset_spec(project_root, dataset_spec_path)
    split_dir = spec.metadata_root_dir / split_name
    if not split_dir.exists():
        raise FileNotFoundError(f"Metadata directory does not exist for split '{split_name}': {split_dir}")

    records: list[ScenePairRecord] = []
    for render_record_path in sorted(split_dir.glob("*/render_record.json")):
        with render_record_path.open("r", encoding="utf-8") as handle:
            render_record = json.load(handle)

        low_rel = render_record["artifacts"]["low_hoa_path"]
        high_rel = render_record["artifacts"]["high_hoa_path"]
        low_path = project_root / low_rel
        high_path = project_root / high_rel
        scene_id = render_record["scene_id"]

        if not low_path.exists():
            raise FileNotFoundError(f"Missing low HOA artifact for {scene_id}: {low_path}")
        if not high_path.exists():
            raise FileNotFoundError(f"Missing high HOA artifact for {scene_id}: {high_path}")

        records.append(
            ScenePairRecord(
                scene_id=scene_id,
                split_name=split_name,
                low_path=low_path,
                high_path=high_path,
            )
        )

    if limit is not None:
        records = records[:limit]
    return records


def validate_record_shapes(records: list[ScenePairRecord], spec: DatasetSpec) -> None:
    expected_shape = (spec.expected_num_channels, spec.expected_num_samples)
    for record in records:
        low = np.load(record.low_path, mmap_mode="r")
        high = np.load(record.high_path, mmap_mode="r")
        if tuple(low.shape) != expected_shape:
            raise ValueError(
                f"Low HOA shape mismatch for {record.scene_id}: expected {expected_shape}, got {tuple(low.shape)}"
            )
        if tuple(high.shape) != expected_shape:
            raise ValueError(
                f"High HOA shape mismatch for {record.scene_id}: expected {expected_shape}, got {tuple(high.shape)}"
            )


def compute_channel_stats(
    records: list[ScenePairRecord],
    *,
    array_selector: str,
    epsilon: float = DEFAULT_EPSILON,
) -> ChannelStats:
    if array_selector not in {"low", "high"}:
        raise ValueError(f"Unsupported array_selector: {array_selector}")
    if not records:
        raise ValueError("Cannot compute channel statistics from an empty record list")

    running_sum: np.ndarray | None = None
    running_sumsq: np.ndarray | None = None
    total_count = 0

    for record in records:
        path = record.low_path if array_selector == "low" else record.high_path
        array = np.load(path, mmap_mode="r").astype(np.float64, copy=False)
        if array.ndim != 2:
            raise ValueError(f"Expected 2D HOA array for {record.scene_id}, got shape {tuple(array.shape)}")

        if running_sum is None:
            running_sum = np.zeros(array.shape[0], dtype=np.float64)
            running_sumsq = np.zeros(array.shape[0], dtype=np.float64)

        running_sum += array.sum(axis=1, dtype=np.float64)
        running_sumsq += np.square(array, dtype=np.float64).sum(axis=1, dtype=np.float64)
        total_count += array.shape[1]

    assert running_sum is not None
    assert running_sumsq is not None

    mean = running_sum / total_count
    variance = running_sumsq / total_count - np.square(mean)
    variance = np.maximum(variance, epsilon**2)
    std = np.sqrt(variance)
    return ChannelStats(mean=mean.astype(np.float32), std=std.astype(np.float32))


def save_stats(output_path: str | Path, *, input_stats: ChannelStats, target_stats: ChannelStats) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        input_mean=input_stats.mean,
        input_std=input_stats.std,
        target_mean=target_stats.mean,
        target_std=target_stats.std,
    )


def load_stats(stats_path: str | Path) -> tuple[ChannelStats, ChannelStats]:
    stats_path = Path(stats_path)
    with np.load(stats_path) as stats:
        input_stats = ChannelStats(mean=stats["input_mean"], std=stats["input_std"])
        target_stats = ChannelStats(mean=stats["target_mean"], std=stats["target_std"])
    return input_stats, target_stats


class HOAFullIRSequence(tf.keras.utils.Sequence):
    def __init__(
        self,
        records: list[ScenePairRecord],
        *,
        input_stats: ChannelStats,
        target_stats: ChannelStats,
        expected_num_channels: int,
        expected_num_samples: int,
        batch_size: int = 1,
        shuffle: bool = True,
        seed: int = 42,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if not records:
            raise ValueError("records must not be empty")

        self.records = list(records)
        self.input_stats = input_stats
        self.target_stats = target_stats
        self.expected_num_channels = expected_num_channels
        self.expected_num_samples = expected_num_samples
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self._rng = random.Random(seed)
        self.indices = list(range(len(self.records)))
        self.on_epoch_end()

    def __len__(self) -> int:
        return math.ceil(len(self.indices) / self.batch_size)

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        start = index * self.batch_size
        stop = min(start + self.batch_size, len(self.indices))
        batch_indices = self.indices[start:stop]
        actual_batch_size = len(batch_indices)

        x_batch = np.empty(
            (actual_batch_size, self.expected_num_samples, self.expected_num_channels),
            dtype=np.float32,
        )
        y_batch = np.empty_like(x_batch)

        for batch_row, record_idx in enumerate(batch_indices):
            record = self.records[record_idx]
            low = np.load(record.low_path, mmap_mode="r").astype(np.float32, copy=False)
            high = np.load(record.high_path, mmap_mode="r").astype(np.float32, copy=False)

            if tuple(low.shape) != (self.expected_num_channels, self.expected_num_samples):
                raise ValueError(
                    f"Low HOA shape mismatch for {record.scene_id}: expected "
                    f"{(self.expected_num_channels, self.expected_num_samples)}, got {tuple(low.shape)}"
                )
            if tuple(high.shape) != (self.expected_num_channels, self.expected_num_samples):
                raise ValueError(
                    f"High HOA shape mismatch for {record.scene_id}: expected "
                    f"{(self.expected_num_channels, self.expected_num_samples)}, got {tuple(high.shape)}"
                )

            low_norm = self.input_stats.normalize(low)
            high_norm = self.target_stats.normalize(high)
            x_batch[batch_row] = low_norm.T
            y_batch[batch_row] = high_norm.T

        return x_batch, y_batch

    def on_epoch_end(self) -> None:
        if self.shuffle:
            self._rng.shuffle(self.indices)