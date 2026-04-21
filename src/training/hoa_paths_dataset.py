from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import random
import numpy as np
import tensorflow as tf

from training.hoa_dataset import ChannelStats, DatasetSpec, discover_records
from training.path_features import PathFeatureStats, compute_path_feature_stats, load_path_feature_matrix

@dataclass(frozen=True)
class ScenePathRecord:
    scene_id: str
    split_name: str
    low_path: Path
    high_path: Path
    paths_csv_path: Path


def discover_records_with_paths(project_root: str | Path, dataset_spec_path: str | Path, split_name: str, *, limit: int | None = None) -> list[ScenePathRecord]:
    base_records = discover_records(project_root, dataset_spec_path, split_name, limit=limit)
    out = []
    for record in base_records:
        paths_csv_path = record.low_path.parent / "paths_top.csv"
        if not paths_csv_path.exists():
            raise FileNotFoundError(f"Missing paths_top.csv for {record.scene_id}: {paths_csv_path}")
        out.append(ScenePathRecord(record.scene_id, record.split_name, record.low_path, record.high_path, paths_csv_path))
    return out


def validate_record_shapes_with_paths(records: list[ScenePathRecord], spec: DatasetSpec, *, path_top_k: int) -> None:
    expected_shape = (spec.expected_num_channels, spec.expected_num_samples)
    for record in records:
        low = np.load(record.low_path, mmap_mode="r")
        high = np.load(record.high_path, mmap_mode="r")
        if tuple(low.shape) != expected_shape or tuple(high.shape) != expected_shape:
            raise ValueError(f"HOA shape mismatch for {record.scene_id}")
        load_path_feature_matrix(record.paths_csv_path, top_k=path_top_k)


class HOAWithPathsSequence(tf.keras.utils.Sequence):
    def __init__(self, records: list[ScenePathRecord], *, input_stats: ChannelStats, target_stats: ChannelStats, path_stats: PathFeatureStats, expected_num_channels: int, expected_num_samples: int, path_top_k: int, path_num_features: int, batch_size: int = 1, shuffle: bool = True, seed: int = 42, **kwargs):
        super().__init__(**kwargs)
        self.records = list(records)
        self.input_stats = input_stats
        self.target_stats = target_stats
        self.path_stats = path_stats
        self.expected_num_channels = expected_num_channels
        self.expected_num_samples = expected_num_samples
        self.path_top_k = path_top_k
        self.path_num_features = path_num_features
        self.batch_size = batch_size
        self.shuffle = shuffle
        self._rng = random.Random(seed)
        self.indices = list(range(len(self.records)))
        self.on_epoch_end()

    def __len__(self) -> int:
        return math.ceil(len(self.indices) / self.batch_size)

    def __getitem__(self, index: int):
        start = index * self.batch_size
        stop = min(start + self.batch_size, len(self.indices))
        batch_indices = self.indices[start:stop]
        b = len(batch_indices)
        low_batch = np.empty((b, self.expected_num_samples, self.expected_num_channels), dtype=np.float32)
        path_batch = np.empty((b, self.path_top_k, self.path_num_features), dtype=np.float32)
        y_batch = np.empty_like(low_batch)
        for row, record_idx in enumerate(batch_indices):
            record = self.records[record_idx]
            low = np.load(record.low_path, mmap_mode="r").astype(np.float32, copy=False)
            high = np.load(record.high_path, mmap_mode="r").astype(np.float32, copy=False)
            paths = load_path_feature_matrix(record.paths_csv_path, top_k=self.path_top_k)
            low_batch[row] = self.input_stats.normalize(low).T
            path_batch[row] = self.path_stats.normalize(paths)
            y_batch[row] = self.target_stats.normalize(high).T
        return {"low_hoa_input": low_batch, "path_features_input": path_batch}, y_batch

    def on_epoch_end(self) -> None:
        if self.shuffle:
            self._rng.shuffle(self.indices)


def compute_path_stats_from_records(records: list[ScenePathRecord], *, top_k: int) -> PathFeatureStats:
    return compute_path_feature_stats([record.paths_csv_path for record in records], top_k=top_k)
