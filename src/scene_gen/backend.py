from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
import csv
import wave

import numpy as np

from .scene_spec import SceneSpec


@dataclass(frozen=True)
class RenderArtifacts:
    low_hoa_path: Path
    high_hoa_path: Path
    paths_path: Path
    preview_wav_path: Path


class SimulationBackend(ABC):
    @abstractmethod
    def render(self, scene: SceneSpec, output_dir: Path) -> RenderArtifacts:
        raise NotImplementedError


class DryRunBackend(SimulationBackend):
    """
    End-to-end testing backend.

    This backend does not attempt physically accurate simulation. It produces deterministic,
    geometry-conditioned tensors with a direct-path impulse, exponentially decaying tail,
    and low/high ray noise differences so that the dataset pipeline, QC logic, and storage
    layout can be exercised before wiring in GSound-SIR.
    """

    def render(self, scene: SceneSpec, output_dir: Path) -> RenderArtifacts:
        output_dir.mkdir(parents=True, exist_ok=True)
        num_channels = scene.simulation.expected_num_channels
        num_samples = scene.simulation.expected_num_samples
        rng = np.random.default_rng(scene.global_seed)
        onset = int(round(scene.placement.source_receiver_distance_m / 343.0 * scene.simulation.sample_rate_hz))
        onset = max(0, min(num_samples - 1, onset))

        high = np.zeros((num_channels, num_samples), dtype=np.float32)
        low = np.zeros((num_channels, num_samples), dtype=np.float32)
        tail_length = num_samples - onset
        time = np.arange(tail_length, dtype=np.float32) / scene.simulation.sample_rate_hz
        base_decay = np.exp(-time * (1.8 + 2.2 * scene.materials.descriptors["mean_absorption"]))

        for ch in range(num_channels):
            direct_gain = 1.0 / (1.0 + scene.placement.source_receiver_distance_m)
            directional_scale = 1.0 / (1.0 + 0.15 * ch)
            high[ch, onset] += direct_gain * directional_scale
            low[ch, onset] += direct_gain * directional_scale

            smooth_noise = rng.normal(0.0, 0.03, size=tail_length).astype(np.float32)
            noisy_tail = rng.normal(0.0, 0.08, size=tail_length).astype(np.float32)
            high[ch, onset:] += directional_scale * base_decay * smooth_noise
            low[ch, onset:] += directional_scale * base_decay * noisy_tail

        low_hoa_path = output_dir / "low_hoa.npy"
        high_hoa_path = output_dir / "high_hoa.npy"
        np.save(low_hoa_path, low)
        np.save(high_hoa_path, high)

        paths_path = output_dir / "paths_top.csv"
        with paths_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["path_rank", "time_s", "energy", "order_hint"])
            retained = min(scene.simulation.retained_path_value, 64)
            for rank in range(retained):
                writer.writerow([
                    rank,
                    float((onset + rank * 37) / scene.simulation.sample_rate_hz),
                    float(1.0 / (1.0 + rank)),
                    int(rank % max(1, num_channels)),
                ])

        preview_wav_path = output_dir / "preview_binaural.wav"
        self._write_preview(high[0], preview_wav_path, scene.simulation.sample_rate_hz)
        return RenderArtifacts(
            low_hoa_path=low_hoa_path,
            high_hoa_path=high_hoa_path,
            paths_path=paths_path,
            preview_wav_path=preview_wav_path,
        )

    @staticmethod
    def _write_preview(w_channel: np.ndarray, out_path: Path, sample_rate_hz: int) -> None:
        signal = np.clip(w_channel, -1.0, 1.0)
        stereo = np.stack([signal, signal], axis=1)
        pcm16 = (stereo * 32767.0).astype(np.int16)
        with wave.open(str(out_path), "wb") as wav_file:
            wav_file.setnchannels(2)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate_hz)
            wav_file.writeframes(pcm16.tobytes())


class GSoundSIRBackend(SimulationBackend):
    """
    Integration seam for the real simulator.

    Replace this stub with code that either:
      1. imports your existing GSound-SIR Python bridge, or
      2. shells out to a dedicated script under src/gsound_tests or src/scripts.

    The pipeline around this backend is already ready for real renders.
    """

    def render(self, scene: SceneSpec, output_dir: Path) -> RenderArtifacts:
        raise NotImplementedError(
            "Wire this class to your actual GSound-SIR render entry point. "
            "Start by adapting your existing tests in src/gsound_tests."
        )
