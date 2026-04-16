from __future__ import annotations

from pathlib import Path
import json
import math

import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve, resample_poly


def _ensure_stereo(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        return np.stack([audio, audio], axis=1)
    if audio.ndim == 2 and audio.shape[1] == 1:
        return np.repeat(audio, 2, axis=1)
    if audio.ndim == 2 and audio.shape[1] == 2:
        return audio
    raise ValueError(f"Unsupported audio shape: {audio.shape}")


def _normalize_peak(audio: np.ndarray, peak: float = 0.95) -> np.ndarray:
    max_abs = float(np.max(np.abs(audio)))
    if max_abs <= 0.0:
        return audio.astype(np.float32, copy=False)
    return (audio * (peak / max_abs)).astype(np.float32, copy=False)


def _collect_preview_sources(preview_sources_dir: Path) -> list[Path]:
    if not preview_sources_dir.exists():
        return []

    allowed_suffixes = {".wav", ".flac", ".ogg"}
    return sorted(
        path
        for path in preview_sources_dir.iterdir()
        if path.is_file() and path.suffix.lower() in allowed_suffixes
    )


def _resample_audio(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr:
        return audio.astype(np.float32, copy=False)

    audio = np.asarray(audio, dtype=np.float32)
    gcd = math.gcd(int(src_sr), int(dst_sr))
    up = int(dst_sr // gcd)
    down = int(src_sr // gcd)

    if audio.ndim == 1:
        return resample_poly(audio, up, down).astype(np.float32, copy=False)

    if audio.ndim == 2:
        channels = [
            resample_poly(audio[:, ch], up, down).astype(np.float32, copy=False)
            for ch in range(audio.shape[1])
        ]
        return np.stack(channels, axis=1)

    raise ValueError(f"Unsupported audio shape for resampling: {audio.shape}")


def convolve_stereo_with_stereo_ir(source_audio: np.ndarray, stereo_ir: np.ndarray) -> np.ndarray:
    source_audio = _ensure_stereo(np.asarray(source_audio, dtype=np.float32))
    stereo_ir = np.asarray(stereo_ir, dtype=np.float32)

    if stereo_ir.ndim != 2 or stereo_ir.shape[1] != 2:
        raise ValueError(f"Expected stereo IR of shape [num_samples, 2], got {stereo_ir.shape}")

    left = fftconvolve(source_audio[:, 0], stereo_ir[:, 0], mode="full")
    right = fftconvolve(source_audio[:, 1], stereo_ir[:, 1], mode="full")
    return np.stack([left, right], axis=1).astype(np.float32)


def render_preview_set(
    *,
    preview_sources_dir: Path,
    output_dir: Path,
    stereo_ir_low: np.ndarray | None,
    stereo_ir_high: np.ndarray | None,
    sample_rate_hz: int,
    normalize_peak: float = 0.95,
) -> Path | None:
    source_paths = _collect_preview_sources(preview_sources_dir)
    if not source_paths:
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, dict[str, str | int]] = {}

    for source_path in source_paths:
        source_audio, sr = sf.read(source_path, always_2d=False)
        source_audio = np.asarray(source_audio, dtype=np.float32)
        original_sr = int(sr)
        source_audio = _resample_audio(source_audio, original_sr, int(sample_rate_hz))

        source_key = source_path.stem
        entry: dict[str, str | int] = {
            "source_path": str(source_path),
            "original_sample_rate_hz": original_sr,
            "render_sample_rate_hz": int(sample_rate_hz),
        }

        if stereo_ir_low is not None:
            low_render = convolve_stereo_with_stereo_ir(source_audio, stereo_ir_low)
            low_render = low_render.astype(np.float32, copy=False) # low_render = _normalize_peak(low_render, normalize_peak)
            low_out_path = output_dir / f"low_{source_key}.wav"
            sf.write(str(low_out_path), low_render, sample_rate_hz)
            entry["low_render_path"] = str(low_out_path)

        if stereo_ir_high is not None:
            high_render = convolve_stereo_with_stereo_ir(source_audio, stereo_ir_high)
            high_render - high_render.astype(np.float32, copy=False) # high_render = _normalize_peak(high_render, normalize_peak)
            high_out_path = output_dir / f"high_{source_key}.wav"
            sf.write(str(high_out_path), high_render, sample_rate_hz)
            entry["high_render_path"] = str(high_out_path)

        manifest[source_key] = entry

    manifest_path = output_dir / "preview_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)

    return manifest_path