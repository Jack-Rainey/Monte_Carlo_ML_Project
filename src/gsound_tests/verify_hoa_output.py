from __future__ import annotations

from pathlib import Path
import math
import json

import numpy as np
import soundfile as sf
import pandas as pd


def infer_hoa_order_from_channels(n_channels: int) -> int | None:
    """
    For full-sphere HOA, channels = (N + 1)^2.
    Return inferred N if exact, else None.
    """
    root = int(math.isqrt(n_channels))
    if root * root != n_channels:
        return None
    order = root - 1
    if (order + 1) ** 2 == n_channels and order >= 0:
        return order
    return None


def describe_npy(npy_path: Path, sample_rate_hint: int | None = None) -> dict:
    arr = np.load(npy_path)

    info: dict = {
        "path": str(npy_path),
        "dtype": str(arr.dtype),
        "shape": tuple(arr.shape),
        "ndim": arr.ndim,
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
    }

    if arr.ndim != 2:
        info["interpretation"] = "Expected 2D array for HOA IR, but got non-2D array."
        return info

    dim0, dim1 = arr.shape
    order0 = infer_hoa_order_from_channels(dim0)
    order1 = infer_hoa_order_from_channels(dim1)

    if order0 is not None and order1 is None:
        axis_layout = "channels_first"
        n_channels = dim0
        n_samples = dim1
        inferred_order = order0
        channel_energies = np.sum(arr ** 2, axis=1)
    elif order1 is not None and order0 is None:
        axis_layout = "time_first"
        n_channels = dim1
        n_samples = dim0
        inferred_order = order1
        channel_energies = np.sum(arr ** 2, axis=0)
    elif order0 is not None and order1 is not None:
        axis_layout = "ambiguous_both_axes_match_hoa_pattern"
        n_channels = None
        n_samples = None
        inferred_order = None
        channel_energies = None
    else:
        axis_layout = "unknown"
        n_channels = None
        n_samples = None
        inferred_order = None
        channel_energies = None

    info["axis_layout"] = axis_layout
    info["n_channels"] = n_channels
    info["n_samples"] = n_samples
    info["inferred_hoa_order"] = inferred_order

    if sample_rate_hint is not None and n_samples is not None:
        info["sample_rate_hint"] = sample_rate_hint
        info["duration_seconds_est"] = n_samples / sample_rate_hint

    if channel_energies is not None:
        info["channel_energy_min"] = float(np.min(channel_energies))
        info["channel_energy_max"] = float(np.max(channel_energies))
        info["channel_energy_mean"] = float(np.mean(channel_energies))
        info["near_zero_channels"] = int(np.sum(channel_energies < 1e-12))

    return info


def describe_wav(wav_path: Path) -> dict:
    data, sample_rate = sf.read(wav_path, always_2d=True)

    info = {
        "path": str(wav_path),
        "sample_rate": int(sample_rate),
        "shape": tuple(data.shape),
        "dtype": str(data.dtype),
        "n_samples": int(data.shape[0]),
        "n_channels": int(data.shape[1]),
        "duration_seconds": float(data.shape[0] / sample_rate),
        "min": float(np.min(data)),
        "max": float(np.max(data)),
        "mean": float(np.mean(data)),
        "std": float(np.std(data)),
    }

    if info["n_channels"] == 1:
        info["interpretation"] = "mono"
    elif info["n_channels"] == 2:
        info["interpretation"] = "stereo_or_binaural_render"
    else:
        info["interpretation"] = "multichannel_audio"

    return info


def describe_parquet(parquet_path: Path) -> dict:
    df = pd.read_parquet(parquet_path)

    info = {
        "path": str(parquet_path),
        "n_rows": int(len(df)),
        "n_columns": int(len(df.columns)),
        "columns": list(map(str, df.columns)),
    }

    return info


def main() -> None:
    output_dir = Path("src/gsound_tests/outputs")

    npy_path = output_dir / "shoebox_hoa_ir.npy"
    wav_path = output_dir / "shoebox_ir.wav"
    parquet_path = output_dir / "shoebox_paths.parquet"

    report: dict = {
        "detected_files": sorted([p.name for p in output_dir.iterdir() if p.is_file()]),
    }

    wav_sample_rate = None
    if wav_path.exists():
        wav_info = describe_wav(wav_path)
        report["wav"] = wav_info
        wav_sample_rate = wav_info["sample_rate"]
    else:
        report["wav"] = {"missing": True}

    if npy_path.exists():
        report["npy"] = describe_npy(npy_path, sample_rate_hint=wav_sample_rate)
    else:
        report["npy"] = {"missing": True}

    if parquet_path.exists():
        report["parquet"] = describe_parquet(parquet_path)
    else:
        report["parquet"] = {"missing": True}

    # High-level conclusions
    conclusions = []

    npy_info = report.get("npy", {})
    wav_info = report.get("wav", {})

    if npy_info.get("inferred_hoa_order") == 3 and npy_info.get("n_channels") == 16:
        conclusions.append("NPY appears structurally consistent with third-order HOA (16 channels).")
    elif npy_info.get("missing"):
        conclusions.append("NPY file missing; cannot verify canonical HOA representation.")
    else:
        conclusions.append("NPY does not yet cleanly verify as third-order HOA; inspect shape/order manually.")

    if wav_info.get("n_channels") == 2:
        conclusions.append("WAV appears to be stereo/binaural-style render, not the canonical HOA tensor.")
    elif wav_info.get("n_channels", 0) > 2:
        conclusions.append("WAV is multichannel; inspect whether it preserves HOA channels directly.")
    elif wav_info.get("missing"):
        conclusions.append("WAV file missing.")
    else:
        conclusions.append("WAV exists but is not obviously stereo; inspect manually.")

    if npy_info.get("n_channels") and wav_info.get("n_channels"):
        if npy_info["n_channels"] != wav_info["n_channels"]:
            conclusions.append("NPY and WAV channel counts differ, suggesting they represent different output forms.")

    report["conclusions"] = conclusions

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()