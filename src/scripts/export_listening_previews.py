from __future__ import annotations

import argparse
from pathlib import Path
import json
import wave

import numpy as np

from scene_gen.gsoundsir_bridge import (
    preview_stereo_from_hoa,
    write_multichannel_hoa_wav,
)
from training.hoa_dataset import load_dataset_spec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export listening previews for low / predicted / high HOA IRs."
    )
    parser.add_argument("--project-root", default=".")
    parser.add_argument(
        "--dataset-spec",
        default="configs/scenes/procedural_rir_dataset_real_backend_full_v1.json",
    )
    parser.add_argument(
        "--prediction-root",
        required=True,
        help="Directory containing prediction_manifest.json from export_hoa_predictions.py",
    )
    parser.add_argument(
        "--preview-sources-dir",
        default="assets/audio_preview_sources",
        help="Directory of dry source WAVs used for convolution renders.",
    )
    parser.add_argument(
        "--preview-channel-index",
        type=int,
        default=0,
        help="HOA channel index used by preview_stereo_from_hoa.",
    )
    parser.add_argument(
        "--peak-scale",
        type=float,
        default=0.95,
        help="Shared peak target used when writing comparison WAVs.",
    )
    parser.add_argument(
        "--write-multichannel-hoawav",
        action="store_true",
        help="Also export pred_high_16ch.wav for each scene.",
    )
    parser.add_argument(
        "--predicted-subdir-name",
        default="predicted",
        help="Subdirectory name used under listening/ for prediction renders.",
    )
    return parser.parse_args()


def write_stereo_wav(stereo_signal: np.ndarray, out_path: Path, sample_rate_hz: int) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stereo_signal = np.asarray(stereo_signal, dtype=np.float32)
    if stereo_signal.ndim != 2 or stereo_signal.shape[1] != 2:
        raise ValueError(f"Expected stereo array of shape (num_samples, 2), got {stereo_signal.shape}")

    clipped = np.clip(stereo_signal, -1.0, 1.0)
    pcm16 = (clipped * 32767.0).astype(np.int16)

    with wave.open(str(out_path), "wb") as wav_file:
        wav_file.setnchannels(2)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate_hz)
        wav_file.writeframes(pcm16.tobytes())


def read_wav_as_mono_float32(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wav_file:
        sample_rate_hz = wav_file.getframerate()
        num_channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        num_frames = wav_file.getnframes()
        raw = wav_file.readframes(num_frames)

    bytes_per_frame = sample_width * num_channels
    if len(raw) % bytes_per_frame != 0:
        raise ValueError(f"Malformed WAV data in {path}")

    if sample_width == 1:
        data = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        data = (data - 128.0) / 128.0

    elif sample_width == 2:
        data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

    elif sample_width == 3:
        raw_u8 = np.frombuffer(raw, dtype=np.uint8)
        triples = raw_u8.reshape(-1, 3)

        data_i32 = (
            triples[:, 0].astype(np.int32)
            | (triples[:, 1].astype(np.int32) << 8)
            | (triples[:, 2].astype(np.int32) << 16)
        )

        sign_mask = 1 << 23
        data_i32 = np.where(data_i32 & sign_mask, data_i32 - (1 << 24), data_i32)
        data = data_i32.astype(np.float32) / 8388608.0

    elif sample_width == 4:
        data = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0

    else:
        raise ValueError(f"Unsupported WAV sample width {sample_width} for {path}")

    data = data.reshape(-1, num_channels)

    if num_channels == 1:
        mono = data[:, 0]
    else:
        mono = data.mean(axis=1, dtype=np.float32)

    return mono.astype(np.float32, copy=False), sample_rate_hz


def resample_mono_linear(signal: np.ndarray, src_rate_hz: int, dst_rate_hz: int) -> np.ndarray:
    if src_rate_hz == dst_rate_hz:
        return signal.astype(np.float32, copy=False)

    if signal.size == 0:
        return signal.astype(np.float32, copy=False)

    src_times = np.arange(signal.shape[0], dtype=np.float64) / float(src_rate_hz)
    dst_length = int(round(signal.shape[0] * float(dst_rate_hz) / float(src_rate_hz)))
    dst_length = max(dst_length, 1)
    dst_times = np.arange(dst_length, dtype=np.float64) / float(dst_rate_hz)

    resampled = np.interp(dst_times, src_times, signal.astype(np.float64, copy=False))
    return resampled.astype(np.float32, copy=False)


def fft_convolve_1d(signal: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    signal = np.asarray(signal, dtype=np.float32)
    kernel = np.asarray(kernel, dtype=np.float32)

    out_len = signal.shape[0] + kernel.shape[0] - 1
    fft_len = 1 << (out_len - 1).bit_length()

    signal_fft = np.fft.rfft(signal, n=fft_len)
    kernel_fft = np.fft.rfft(kernel, n=fft_len)
    out = np.fft.irfft(signal_fft * kernel_fft, n=fft_len)[:out_len]
    return out.astype(np.float32, copy=False)


def render_mono_source_with_stereo_ir(mono_source: np.ndarray, stereo_ir: np.ndarray) -> np.ndarray:
    if stereo_ir.ndim != 2 or stereo_ir.shape[1] != 2:
        raise ValueError(f"Expected stereo IR of shape (num_samples, 2), got {stereo_ir.shape}")

    left = fft_convolve_1d(mono_source, stereo_ir[:, 0])
    right = fft_convolve_1d(mono_source, stereo_ir[:, 1])
    return np.stack([left, right], axis=1).astype(np.float32, copy=False)


def shared_peak_scale(*signals: np.ndarray, peak_scale: float) -> float:
    peak = max(float(np.max(np.abs(sig))) for sig in signals)
    return peak_scale / max(peak, 1e-8)


def rms(signal: np.ndarray) -> float:
    signal = np.asarray(signal, dtype=np.float32)
    return float(np.sqrt(np.mean(np.square(signal), dtype=np.float64)))


def percentile_peak(signal: np.ndarray, q: float = 0.999) -> float:
    signal = np.asarray(signal, dtype=np.float32)
    return float(np.quantile(np.abs(signal), q))


def shared_comparison_gain(
    *signals: np.ndarray,
    peak_scale: float,
    target_rms: float = 0.10,
    peak_quantile: float = 0.999,
) -> float:
    peak_ref = max(percentile_peak(sig, q=peak_quantile) for sig in signals)
    rms_ref = max(rms(sig) for sig in signals)

    peak_gain = peak_scale / max(peak_ref, 1e-8)
    rms_gain = target_rms / max(rms_ref, 1e-8)

    return min(peak_gain, rms_gain)


def peak(signal: np.ndarray) -> float:
    signal = np.asarray(signal, dtype=np.float32)
    return float(np.max(np.abs(signal)))


def list_preview_source_wavs(preview_sources_dir: Path) -> list[Path]:
    wavs = sorted(preview_sources_dir.rglob("*.wav"))
    if not wavs:
        raise FileNotFoundError(f"No .wav files found under preview sources dir: {preview_sources_dir}")
    return wavs


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    dataset_spec_path = (project_root / args.dataset_spec).resolve()
    prediction_root = Path(args.prediction_root).resolve()
    preview_sources_dir = (project_root / args.preview_sources_dir).resolve()

    manifest_path = prediction_root / "prediction_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing prediction manifest: {manifest_path}")

    preview_source_paths = list_preview_source_wavs(preview_sources_dir)
    dataset_spec = load_dataset_spec(project_root, dataset_spec_path)

    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    exported = []

    for item in manifest:
        scene_id = item["scene_id"]
        split_name = item["split"]

        low_path = project_root / item["source_low_path"]
        high_path = project_root / item["source_high_path"]
        pred_path = project_root / item["prediction_path"]

        low = np.load(low_path).astype(np.float32, copy=False)
        high = np.load(high_path).astype(np.float32, copy=False)
        pred = np.load(pred_path).astype(np.float32, copy=False)

        scene_output_dir = pred_path.parent / "listening"
        low_dir = scene_output_dir / "low"
        pred_dir = scene_output_dir / args.predicted_subdir_name
        high_dir = scene_output_dir / "high"
        ir_dir = scene_output_dir / "ir_previews"

        for out_dir in (low_dir, pred_dir, high_dir, ir_dir):
            out_dir.mkdir(parents=True, exist_ok=True)

        low_preview = preview_stereo_from_hoa(
            hoa_ir=low,
            channel_index=args.preview_channel_index,
            normalize=False,
            peak_scale=1.0,
        ).astype(np.float32, copy=False)
        pred_preview = preview_stereo_from_hoa(
            hoa_ir=pred,
            channel_index=args.preview_channel_index,
            normalize=False,
            peak_scale=1.0,
        ).astype(np.float32, copy=False)
        high_preview = preview_stereo_from_hoa(
            hoa_ir=high,
            channel_index=args.preview_channel_index,
            normalize=False,
            peak_scale=1.0,
        ).astype(np.float32, copy=False)

        ir_scale = shared_comparison_gain(
            low_preview,
            pred_preview,
            high_preview,
            peak_scale=args.peak_scale,
            target_rms=0.10,
            peak_quantile=0.999,
        )

        low_ir_wav = ir_dir / "low_preview_ir.wav"
        pred_ir_wav = ir_dir / "predicted_preview_ir.wav"
        high_ir_wav = ir_dir / "high_preview_ir.wav"

        write_stereo_wav(low_preview * ir_scale, low_ir_wav, dataset_spec.sample_rate_hz)
        write_stereo_wav(pred_preview * ir_scale, pred_ir_wav, dataset_spec.sample_rate_hz)
        write_stereo_wav(high_preview * ir_scale, high_ir_wav, dataset_spec.sample_rate_hz)

        if args.write_multichannel_hoawav:
            write_multichannel_hoa_wav(
                hoa_ir=pred,
                out_path=scene_output_dir / "pred_high_16ch.wav",
                sample_rate_hz=dataset_spec.sample_rate_hz,
            )

        scene_manifest: list[dict] = []
        level_report = {
            "scene_id": scene_id,
            "split": split_name,
            "ir_previews": {
                "low": {"peak": peak(low_preview), "rms": rms(low_preview)},
                "predicted": {"peak": peak(pred_preview), "rms": rms(pred_preview)},
                "high": {"peak": peak(high_preview), "rms": rms(high_preview)},
                "shared_scale_written": ir_scale,
            },
            "rendered_assets": [],
        }

        for src_path in preview_source_paths:
            mono_source, source_sr = read_wav_as_mono_float32(src_path)
            mono_source = resample_mono_linear(mono_source, source_sr, dataset_spec.sample_rate_hz)

            low_render = render_mono_source_with_stereo_ir(mono_source, low_preview)
            pred_render = render_mono_source_with_stereo_ir(mono_source, pred_preview)
            high_render = render_mono_source_with_stereo_ir(mono_source, high_preview)

            render_scale = shared_comparison_gain(
                low_render,
                pred_render,
                high_render,
                peak_scale=args.peak_scale,
                target_rms=0.10,
                peak_quantile=0.999,
            )

            rel_src = src_path.relative_to(preview_sources_dir)
            low_out = low_dir / rel_src
            pred_out = pred_dir / rel_src
            high_out = high_dir / rel_src

            write_stereo_wav(low_render * render_scale, low_out, dataset_spec.sample_rate_hz)
            write_stereo_wav(pred_render * render_scale, pred_out, dataset_spec.sample_rate_hz)
            write_stereo_wav(high_render * render_scale, high_out, dataset_spec.sample_rate_hz)

            scene_manifest.append(
                {
                    "asset": str(rel_src),
                    "low_wav": str(low_out.relative_to(project_root)),
                    "predicted_wav": str(pred_out.relative_to(project_root)),
                    "high_wav": str(high_out.relative_to(project_root)),
                    "shared_scale_written": render_scale,
                }
            )

            level_report["rendered_assets"].append(
                {
                    "asset": str(rel_src),
                    "low": {"peak": peak(low_render), "rms": rms(low_render)},
                    "predicted": {"peak": peak(pred_render), "rms": rms(pred_render)},
                    "high": {"peak": peak(high_render), "rms": rms(high_render)},
                    "shared_scale_written": render_scale,
                }
            )

        scene_manifest_path = scene_output_dir / "listening_manifest.json"
        with scene_manifest_path.open("w", encoding="utf-8") as handle:
            json.dump(scene_manifest, handle, indent=2)

        level_report_path = scene_output_dir / "level_report.json"
        with level_report_path.open("w", encoding="utf-8") as handle:
            json.dump(level_report, handle, indent=2)

        exported.append(
            {
                "scene_id": scene_id,
                "split": split_name,
                "scene_output_dir": str(scene_output_dir.relative_to(project_root)),
                "scene_listening_manifest": str(scene_manifest_path.relative_to(project_root)),
                "scene_level_report": str(level_report_path.relative_to(project_root)),
                "low_dir": str(low_dir.relative_to(project_root)),
                "predicted_dir": str(pred_dir.relative_to(project_root)),
                "high_dir": str(high_dir.relative_to(project_root)),
                "ir_preview_dir": str(ir_dir.relative_to(project_root)),
            }
        )

    out_manifest = prediction_root / "listening_manifest.json"
    with out_manifest.open("w", encoding="utf-8") as handle:
        json.dump(exported, handle, indent=2)

    print(
        {
            "scene_count": len(exported),
            "prediction_root": str(prediction_root),
            "listening_manifest": str(out_manifest),
        },
        flush=True,
    )


if __name__ == "__main__":
    main()