from __future__ import annotations

from pathlib import Path
import importlib
import math
import multiprocessing
import wave

import numpy as np
import soundfile as sf

from .scene_spec import SceneSpec

_SUPPORTED_FAMILIES = frozenset({"shoebox", "corridor"})
_STANDARD_FREQUENCY_POINTS_HZ = np.array([125, 250, 500, 1000, 2000, 4000, 8000], dtype=np.float32)


def _import_runtime_dependencies():
    try:
        pd = importlib.import_module("pandas")
        ps = importlib.import_module("pygsound")
        sh = importlib.import_module("spherical_harmonics_rt")
    except ImportError as exc:  # pragma: no cover - depends on runtime environment
        raise RuntimeError(
            "Real GSound-SIR rendering requires pandas, pygsound, and spherical_harmonics_rt. "
            "Activate the gsoundsir environment and verify the working tests under src/gsound_tests."
        ) from exc
    return pd, ps, sh


def supported_families() -> frozenset[str]:
    return _SUPPORTED_FAMILIES


def require_supported_family(scene: SceneSpec) -> None:
    family = scene.geometry.family
    if family in _SUPPORTED_FAMILIES:
        return
    raise NotImplementedError(
        "GSoundSIRBackend currently supports only rectangular rooms built through pygsound.createbox(): "
        f"{sorted(_SUPPORTED_FAMILIES)}. "
        f"Received family={family!r}. "
        "Use the rectangular-only config for immediate real runs, or add a custom mesh builder for polygonal footprints."
    )


def _clip01(value: float, lo: float = 1e-3, hi: float = 0.99) -> float:
    return float(min(hi, max(lo, value)))


def global_box_absorption(scene: SceneSpec) -> float:
    """
    Map the richer material profile to pygsound.createbox's single absorption coefficient.

    Until per-surface materials are wired through custom meshes, the least misleading
    approximation is the scene-wide mean absorption that the generator already records.
    """
    return _clip01(scene.materials.descriptors.get("mean_absorption", 0.30))


def _assert_box_dimensions(name, p, length, width, height):
    if not (0.0 < p.x < length):
        raise ValueError(f"{name}.x={p.x} outside (0, {length})")
    if not (0.0 < p.y < width):
        raise ValueError(f"{name}.y={p.y} outside (0, {width})")
    if not (0.0 < p.z < height):
        raise ValueError(f"{name}.z={p.z} outside (0, {height})")


def box_dimensions(scene: SceneSpec) -> tuple[float, float, float]:
    params = scene.geometry.parameters
    family = scene.geometry.family
    if family in {"shoebox", "corridor"}:
        length = float(params["length_m"])
        width = float(params["width_m"])
        height = float(scene.geometry.height_m)

        _assert_box_dimensions("source", scene.placement.source, length, width, height)
        _assert_box_dimensions("receiver", scene.placement.receiver, length, width, height)

        return length, width, height
    require_supported_family(scene)
    raise AssertionError("unreachable")


def _split_total_ray_count(total_ray_count: int) -> tuple[int, int]:
    if total_ray_count <= 0:
        raise ValueError("total_ray_count must be positive")
    # Preserve the working test's approximate 10:1 diffuse:specular ratio.
    specular_count = max(1, int(round(total_ray_count / 11.0)))
    diffuse_count = max(1, int(total_ray_count - specular_count))
    return diffuse_count, specular_count


def make_context(*, total_ray_count: int, sample_rate_hz: int, max_threads: int = 8):
    _, ps, _ = _import_runtime_dependencies()
    diffuse_count, specular_count = _split_total_ray_count(total_ray_count)
    ctx = ps.Context()
    ctx.diffuse_count = int(diffuse_count)
    ctx.specular_count = int(specular_count)
    ctx.threads_count = int(min(multiprocessing.cpu_count(), max_threads))
    ctx.channel_type = ps.ChannelLayoutType.mono
    ctx.sample_rate = int(sample_rate_hz)
    return ctx


def build_rectangular_scene(
    *,
    scene: SceneSpec,
    source_radius_m: float = 0.01,
    listener_radius_m: float = 0.01,
    source_power: float = 1.0,
    fixed_scattering_coefficient: float = 0.10,
):
    _, ps, _ = _import_runtime_dependencies()
    require_supported_family(scene)

    length_m, width_m, height_m = box_dimensions(scene)
    absorption = float(global_box_absorption(scene))
    scattering = float(fixed_scattering_coefficient)

    sx = float(scene.placement.source.x)
    sy = float(scene.placement.source.y)
    sz = float(scene.placement.source.z)

    rx = float(scene.placement.receiver.x)
    ry = float(scene.placement.receiver.y)
    rz = float(scene.placement.receiver.z)

    mesh = ps.createbox(
        float(length_m),
        float(width_m),
        float(height_m),
        absorption,
        scattering,
    )

    scene_obj = ps.Scene()
    scene_obj.setMesh(mesh)

    source = ps.Source([sx, sy, sz])
    source.radius = float(source_radius_m)
    source.power = float(source_power)

    listener = ps.Listener([rx, ry, rz])
    listener.radius = float(listener_radius_m)

    return scene_obj, source, listener, mesh


def _validate_path_data(path_data: dict) -> None:
    num_paths = int(path_data["num_paths"])
    num_bands = int(path_data["num_bands"])
    if num_paths <= 0:
        raise RuntimeError("GSound-SIR returned zero propagation paths")
    if num_bands <= 1:
        raise RuntimeError(f"Expected more than one frequency band, got num_bands={num_bands}")
    if num_bands != 8:
        raise RuntimeError(
            f"Expected num_bands=8 based on the current HOA synthesis binding test, got {num_bands}. "
            "Audit the binding and crossover frequencies before changing this assumption."
        )


def _frequency_points_hz_for_path_data(path_data: dict) -> np.ndarray:
    _validate_path_data(path_data)
    expected = int(path_data["num_bands"]) - 1
    if len(_STANDARD_FREQUENCY_POINTS_HZ) != expected:
        raise RuntimeError(
            f"Expected {expected} crossover frequencies, got {len(_STANDARD_FREQUENCY_POINTS_HZ)}"
        )
    return _STANDARD_FREQUENCY_POINTS_HZ.copy()


def synthesize_hoa_ir(
    *,
    path_data: dict,
    hoa_order: int,
    output_sample_rate_hz: int,
    precise_early_reflections: bool,
    normalize: bool,
    early_reflection_threshold_s: float,
) -> np.ndarray:
    _, _, sh = _import_runtime_dependencies()
    freq_points = _frequency_points_hz_for_path_data(path_data)
    hoa_ir = sh.generate_ambisonic_ir(
        int(hoa_order),
        np.asarray(path_data["listener_directions"], dtype=np.float32),
        np.asarray(path_data["intensities"], dtype=np.float32),
        np.asarray(path_data["distances"], dtype=np.float32),
        np.asarray(path_data["speeds_of_sound"], dtype=np.float32),
        freq_points,
        float(output_sample_rate_hz),
        bool(precise_early_reflections),
        bool(normalize),
        float(early_reflection_threshold_s),
    )
    hoa_ir = np.asarray(hoa_ir)
    expected_channels = (hoa_order + 1) ** 2
    if hoa_ir.ndim != 2:
        raise RuntimeError(f"Expected rank-2 HOA IR, got shape={hoa_ir.shape}")
    if hoa_ir.shape[0] != expected_channels:
        raise RuntimeError(
            f"Expected {(hoa_order + 1) ** 2} HOA channels for order={hoa_order}, got {hoa_ir.shape[0]}"
        )
    if not np.isfinite(hoa_ir).all():
        raise RuntimeError("Generated HOA IR contains NaN or inf")
    if float(np.max(np.abs(hoa_ir))) <= 0.0:
        raise RuntimeError("Generated HOA IR is all zeros")
    return hoa_ir


def _fit_hoa_length(hoa: np.ndarray, expected_num_samples: int) -> np.ndarray:
    if hoa.ndim != 2:
        raise ValueError(f"Expected rank-2 HOA array, got shape={hoa.shape}")

    channels, samples = hoa.shape
    if samples == expected_num_samples:
        return hoa

    if samples > expected_num_samples:
        return hoa[:, :expected_num_samples]

    out = np.zeros((channels, expected_num_samples), dtype=hoa.dtype)
    out[:, :samples] = hoa
    return out


def paths_to_dataframe(path_data: dict):
    pd, _, _ = _import_runtime_dependencies()
    _validate_path_data(path_data)
    df = pd.DataFrame(
        {
            "listener_direction_x": path_data["listener_directions"][:, 0],
            "listener_direction_y": path_data["listener_directions"][:, 1],
            "listener_direction_z": path_data["listener_directions"][:, 2],
            "distance": path_data["distances"],
            "speed_of_sound": path_data["speeds_of_sound"],
        }
    )
    intensities = np.asarray(path_data["intensities"], dtype=np.float32)
    for band_idx in range(int(path_data["num_bands"])):
        df[f"intensity_band_{band_idx}"] = intensities[:, band_idx]
    df["energy_sum"] = intensities.sum(axis=1, dtype=np.float64)
    df["arrival_time_s"] = (
        np.asarray(path_data["distances"], dtype=np.float64)
        / np.asarray(path_data["speeds_of_sound"], dtype=np.float64)
    )
    df["original_path_index"] = np.arange(len(df), dtype=np.int64)
    return df


def retain_paths_dataframe(df, *, policy: str, value: int):
    normalized_policy = policy.strip().lower()
    if normalized_policy == "top_k_energy":
        retained = df.nlargest(int(value), columns="energy_sum", keep="first").copy()
    elif normalized_policy == "top_percent_energy":
        fraction = float(value)
        if fraction > 1.0:
            fraction = fraction / 100.0
        if not (0.0 < fraction <= 1.0):
            raise ValueError(f"Invalid top_percent_energy value: {value}")
        keep_n = max(1, int(math.ceil(len(df) * fraction)))
        retained = df.nlargest(keep_n, columns="energy_sum", keep="first").copy()
    else:
        raise ValueError(f"Unsupported retained_path_policy: {policy}")
    retained.sort_values(by=["energy_sum", "arrival_time_s"], ascending=[False, True], inplace=True)
    retained.reset_index(drop=True, inplace=True)
    retained.insert(0, "path_rank", np.arange(len(retained), dtype=np.int64))
    return retained


def to_retained_paths_dataframe(*, path_data: dict, policy: str, value: int):
    df = paths_to_dataframe(path_data)
    return retain_paths_dataframe(df, policy=policy, value=value)


def save_retained_paths(df, target_path: str | Path) -> Path:
    out_path = Path(target_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = out_path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        try:
            df.to_parquet(out_path, index=False)
            return out_path
        except Exception:
            fallback = out_path.with_suffix(".csv")
            df.to_csv(fallback, index=False)
            return fallback
    if suffix == ".csv":
        df.to_csv(out_path, index=False)
        return out_path
    raise ValueError(f"Unsupported retained-path file extension: {out_path.suffix}")


def preview_stereo_from_hoa(
    *,
    hoa_ir: np.ndarray,
    channel_index: int = 0,  # retained for compatibility, no longer the main control
    normalize: bool = True,
    peak_scale: float = 0.95,
) -> np.ndarray:
    if hoa_ir.ndim != 2:
        raise ValueError(f"Expected rank-2 HOA IR, got shape={hoa_ir.shape}")

    num_channels = hoa_ir.shape[0]
    if num_channels < 1:
        raise ValueError("HOA IR must have at least one channel")

    # Simple multi-channel fold-down for debugging/listening.
    # Assumes ACN-like channel ordering enough that low-order directional channels
    # are more informative than using W alone. This is not a true binaural decode.
    w = np.asarray(hoa_ir[0], dtype=np.float32)

    if num_channels >= 4:
        x = np.asarray(hoa_ir[3], dtype=np.float32)  # common ACN index for first-order X-ish term
        y = np.asarray(hoa_ir[1], dtype=np.float32)  # common ACN index for first-order Y-ish term
        z = np.asarray(hoa_ir[2], dtype=np.float32)  # common ACN index for first-order Z-ish term

        left = w + 0.5 * x + 0.3 * y
        right = w - 0.5 * x + 0.3 * y
    else:
        left = w.copy()
        right = w.copy()

    stereo = np.stack([left, right], axis=1).astype(np.float32, copy=False)

    if normalize:
        peak = float(np.max(np.abs(stereo)))
        if peak > 0.0:
            stereo = stereo / peak * float(peak_scale)

    stereo = np.clip(stereo, -1.0, 1.0)
    return stereo.astype(np.float32, copy=False)


def write_preview_wav_from_hoa(
    *,
    hoa_ir: np.ndarray,
    out_path: str | Path,
    sample_rate_hz: int,
    channel_index: int = 0,
) -> Path:
    stereo = preview_stereo_from_hoa(
        hoa_ir=hoa_ir,
        channel_index=channel_index,
        normalize=True,
        peak_scale=0.95,
    )
    pcm16 = (stereo * 32767.0).astype(np.int16)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(out_path), "wb") as wav_file:
        wav_file.setnchannels(2)
        wav_file.setsampwidth(2)
        wav_file.setframerate(int(sample_rate_hz))
        wav_file.writeframes(pcm16.tobytes())
    return out_path


def write_multichannel_hoa_wav(
    hoa_ir: np.ndarray,
    out_path: Path,
    sample_rate_hz: int,
) -> None:
    # hoa_ir shape: (channels, samples)
    sf.write(str(out_path), hoa_ir.T, sample_rate_hz, subtype="FLOAT")