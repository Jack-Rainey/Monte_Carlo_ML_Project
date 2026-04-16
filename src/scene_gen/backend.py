from __future__ import annotations

from .gsoundsir_bridge import (
    build_rectangular_scene,
    make_context,
    require_supported_family,
    save_retained_paths,
    synthesize_hoa_ir,
    to_retained_paths_dataframe,
    write_preview_wav_from_hoa,
    preview_stereo_from_hoa,
    write_multichannel_hoa_wav,
)

from .preview_render import render_preview_set

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
import csv
import time
import wave

import numpy as np

from .scene_spec import SceneSpec

import pygsound  # noqa: F401
import spherical_harmonics_rt  # noqa: F401


@dataclass(frozen=True)
class RenderArtifacts:
    low_hoa_path: Path
    high_hoa_path: Path
    paths_path: Path
    preview_wav_path: Path
    low_hoa_wav_path: Path | None = None
    high_hoa_wav_path: Path | None = None
    preview_renders_dir: Path | None = None
    preview_manifest_path: Path | None = None


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

    def __init__(
        self,
        *,
        preview_sources_dir: str | Path = "assets/audio_preview_sources",
        preview_channel_index: int = 0,
        preview_peak_normalization: float = 0.95,
    ) -> None:
        self.preview_sources_dir = Path(preview_sources_dir)
        self.preview_channel_index = preview_channel_index
        self.preview_peak_normalization = preview_peak_normalization

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
        time_axis = np.arange(tail_length, dtype=np.float32) / scene.simulation.sample_rate_hz
        base_decay = np.exp(-time_axis * (1.8 + 2.2 * scene.materials.descriptors["mean_absorption"]))

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

        low_hoa_wav_path = output_dir / "low_hoa_16ch.wav"
        write_multichannel_hoa_wav(
            hoa_ir=low,
            out_path=low_hoa_wav_path,
            sample_rate_hz=scene.simulation.sample_rate_hz,
        )

        high_hoa_wav_path = output_dir / "high_hoa_16ch.wav"
        write_multichannel_hoa_wav(
            hoa_ir=high,
            out_path=high_hoa_wav_path,
            sample_rate_hz=scene.simulation.sample_rate_hz,
        )

        paths_path = output_dir / "paths_top.csv"
        with paths_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["path_rank", "time_s", "energy", "order_hint"])
            retained = min(scene.simulation.retained_path_value, 64)
            for rank in range(retained):
                writer.writerow(
                    [
                        rank,
                        float((onset + rank * 37) / scene.simulation.sample_rate_hz),
                        float(1.0 / (1.0 + rank)),
                        int(rank % max(1, num_channels)),
                    ]
                )

        preview_wav_path = output_dir / "preview_binaural.wav"
        self._write_preview(high[self.preview_channel_index], preview_wav_path, scene.simulation.sample_rate_hz)

        low_preview_stereo = preview_stereo_from_hoa(
            hoa_ir=low,
            channel_index=self.preview_channel_index,
            normalize=True,
            peak_scale=self.preview_peak_normalization,
        )
        high_preview_stereo = preview_stereo_from_hoa(
            hoa_ir=high,
            channel_index=self.preview_channel_index,
            normalize=True,
            peak_scale=self.preview_peak_normalization,
        )

        preview_renders_dir = output_dir / "preview_renders"
        preview_manifest_path = render_preview_set(
            preview_sources_dir=self.preview_sources_dir,
            output_dir=preview_renders_dir,
            stereo_ir_low=low_preview_stereo,
            stereo_ir_high=high_preview_stereo,
            sample_rate_hz=scene.simulation.sample_rate_hz,
            normalize_peak=self.preview_peak_normalization,
        )

        if preview_manifest_path is None:
            preview_renders_dir = None

        return RenderArtifacts(
            low_hoa_path=low_hoa_path,
            high_hoa_path=high_hoa_path,
            paths_path=paths_path,
            preview_wav_path=preview_wav_path,
            low_hoa_wav_path=low_hoa_wav_path,
            high_hoa_wav_path=high_hoa_wav_path,
            preview_renders_dir=preview_renders_dir,
            preview_manifest_path=preview_manifest_path,
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
    Real GSound-SIR-backed renderer for procedural dataset generation.

    Current scope:
      * Supported room families: shoebox, corridor
      * HOA synthesis route: scene.getPathData(...) -> spherical_harmonics_rt.generate_ambisonic_ir(...)
      * Material handling: a global box-material approximation based on mean absorption,
        with a fixed low scattering coefficient until per-surface materials are wired.

    Unsupported families (l_room, alcove) fail loudly rather than silently degrading to
    incorrect geometry. Use the rectangular-only config for immediate real runs.
    """

    def __init__(
        self,
        *,
        max_threads: int = 8,
        source_radius_m: float = 0.01,
        listener_radius_m: float = 0.01,
        source_power: float = 1.0,
        fixed_scattering_coefficient: float = 0.10,
        precise_early_reflections: bool = False,
        normalize_hoa: bool = True,
        early_reflection_threshold_s: float = 0.01,
        preview_channel_index: int = 0,
        preview_sources_dir: str | Path = "assets/audio_preview_sources",
        preview_peak_normalization: float = 0.95,
    ) -> None:
        self.max_threads = max_threads
        self.source_radius_m = source_radius_m
        self.listener_radius_m = listener_radius_m
        self.source_power = source_power
        self.fixed_scattering_coefficient = fixed_scattering_coefficient
        self.precise_early_reflections = precise_early_reflections
        self.normalize_hoa = normalize_hoa
        self.early_reflection_threshold_s = early_reflection_threshold_s
        self.preview_channel_index = preview_channel_index
        self.preview_sources_dir = Path(preview_sources_dir)
        self.preview_peak_normalization = preview_peak_normalization

    def render(self, scene: SceneSpec, output_dir: Path) -> RenderArtifacts:
        from .gsoundsir_bridge import _fit_hoa_length

        output_dir.mkdir(parents=True, exist_ok=True)
        require_supported_family(scene)

        low_ctx = make_context(
            total_ray_count=scene.simulation.low_ray_count,
            sample_rate_hz=scene.simulation.sample_rate_hz,
            max_threads=self.max_threads,
        )

        low_scene_obj, low_source, low_listener, _low_mesh = build_rectangular_scene(
            scene=scene,
            source_radius_m=self.source_radius_m,
            listener_radius_m=self.listener_radius_m,
            source_power=self.source_power,
            fixed_scattering_coefficient=self.fixed_scattering_coefficient,
        )

        scene_t0 = time.perf_counter()
        low_t0 = time.perf_counter()
        low_result = low_scene_obj.getPathData([low_source], [low_listener], low_ctx)
        low_path_s = time.perf_counter() - low_t0
        low_t1 = time.perf_counter()
        low_path_data = low_result["path_data"][0]

        low_hoa = synthesize_hoa_ir(
            path_data=low_path_data,
            hoa_order=scene.simulation.hoa_order,
            output_sample_rate_hz=scene.simulation.sample_rate_hz,
            precise_early_reflections=self.precise_early_reflections,
            normalize=self.normalize_hoa,
            early_reflection_threshold_s=self.early_reflection_threshold_s,
        ).astype(np.dtype(scene.simulation.dtype), copy=False)
        low_hoa = _fit_hoa_length(low_hoa, scene.simulation.expected_num_samples)
        low_hoa_s = time.perf_counter() - low_t1

        low_hoa_path = output_dir / "low_hoa.npy"
        np.save(low_hoa_path, low_hoa)

        low_hoa_wav_path = output_dir / "low_hoa_16ch.wav"
        write_multichannel_hoa_wav(
            hoa_ir=low_hoa,
            out_path=low_hoa_wav_path,
            sample_rate_hz=scene.simulation.sample_rate_hz,
        )

        del low_scene_obj, low_source, low_listener, low_ctx

        high_ctx = make_context(
            total_ray_count=scene.simulation.high_ray_count,
            sample_rate_hz=scene.simulation.sample_rate_hz,
            max_threads=self.max_threads,
        )

        high_scene_obj, high_source, high_listener, _high_mesh = build_rectangular_scene(
            scene=scene,
            source_radius_m=self.source_radius_m,
            listener_radius_m=self.listener_radius_m,
            source_power=self.source_power,
            fixed_scattering_coefficient=self.fixed_scattering_coefficient,
        )

        high_t0 = time.perf_counter()
        high_result = high_scene_obj.getPathData([high_source], [high_listener], high_ctx)
        high_path_s = time.perf_counter() - high_t0
        high_t1 = time.perf_counter()
        high_path_data = high_result["path_data"][0]

        high_hoa = synthesize_hoa_ir(
            path_data=high_path_data,
            hoa_order=scene.simulation.hoa_order,
            output_sample_rate_hz=scene.simulation.sample_rate_hz,
            precise_early_reflections=self.precise_early_reflections,
            normalize=self.normalize_hoa,
            early_reflection_threshold_s=self.early_reflection_threshold_s,
        ).astype(np.dtype(scene.simulation.dtype), copy=False)
        high_hoa = _fit_hoa_length(high_hoa, scene.simulation.expected_num_samples)
        high_hoa_s = time.perf_counter() - high_t1
        scene_total_s = time.perf_counter() - scene_t0

        high_hoa_path = output_dir / "high_hoa.npy"
        np.save(high_hoa_path, high_hoa)

        high_hoa_wav_path = output_dir / "high_hoa_16ch.wav"
        write_multichannel_hoa_wav(
            hoa_ir=high_hoa,
            out_path=high_hoa_wav_path,
            sample_rate_hz=scene.simulation.sample_rate_hz,
        )

        retained_paths_df = to_retained_paths_dataframe(
            path_data=high_path_data,
            policy=scene.simulation.retained_path_policy,
            value=scene.simulation.retained_path_value,
        )
        paths_path = save_retained_paths(retained_paths_df, output_dir / "paths_top.parquet")

        preview_wav_path = output_dir / "preview_binaural.wav"
        write_preview_wav_from_hoa(
            hoa_ir=high_hoa,
            out_path=preview_wav_path,
            sample_rate_hz=scene.simulation.sample_rate_hz,
            channel_index=self.preview_channel_index,
        )

        low_preview_stereo = preview_stereo_from_hoa(
            hoa_ir=low_hoa,
            channel_index=self.preview_channel_index,
            normalize=False,
            peak_scale=self.preview_peak_normalization,
        )
        high_preview_stereo = preview_stereo_from_hoa(
            hoa_ir=high_hoa,
            channel_index=self.preview_channel_index,
            normalize=False,
            peak_scale=self.preview_peak_normalization,
        )

        preview_renders_dir = output_dir / "preview_renders"
        preview_manifest_path = render_preview_set(
            preview_sources_dir=self.preview_sources_dir,
            output_dir=preview_renders_dir,
            stereo_ir_low=low_preview_stereo,
            stereo_ir_high=high_preview_stereo,
            sample_rate_hz=scene.simulation.sample_rate_hz,
            normalize_peak=self.preview_peak_normalization,
        )

        if preview_manifest_path is None:
            preview_renders_dir = None

        """
        print(
            {
                "scene_id": scene.scene_id,
                "timing_s": {
                    "low_paths": round(low_path_s, 3),
                    "low_hoa": round(low_hoa_s, 3),
                    "high_paths": round(high_path_s, 3),
                    "high_hoa": round(high_hoa_s, 3),
                    "scene_total": round(scene_total_s, 3),
                },
            },
            flush=True,
        )
        """

        return RenderArtifacts(
            low_hoa_path=low_hoa_path,
            high_hoa_path=high_hoa_path,
            paths_path=paths_path,
            preview_wav_path=preview_wav_path,
            low_hoa_wav_path=low_hoa_wav_path,
            high_hoa_wav_path=high_hoa_wav_path,
            preview_renders_dir=preview_renders_dir,
            preview_manifest_path=preview_manifest_path,
        )