"""Shape and round-trip invariants (design_spec §10 invariant #4)."""
import numpy as np
import pytest
import torch

from amcd.config import Config
from amcd.representations import build_representation
from amcd.representations.spectrogram import ThirdOctaveSpectrogram
from amcd.simulators.base import SceneSpec

from tests.conftest import dry_run_simulator


def make_scene(seed: int = 42) -> SceneSpec:
    return SceneSpec(
        scene_id="test_scene",
        seed=seed,
        geometry_family="shoebox",
        dims=(5.0, 4.0, 3.0),
        material_absorption=0.3,
        source_pos=(1.0, 1.0, 1.5),
        receiver_pos=(4.0, 3.0, 1.5),
    )


class TestIRShape:
    """IR storage is channel-first (C, T)."""

    def test_dry_run_output_shape(self, dry_run_config: Config) -> None:
        sim = dry_run_simulator(
            n_channels=dry_run_config.n_channels,
            n_samples=dry_run_config.n_samples,
            sample_rate=dry_run_config.sample_rate,
        )
        result = sim.render(make_scene(), ray_budget=100)
        assert result.ir.shape == (dry_run_config.n_channels, dry_run_config.n_samples)
        assert result.ir.dtype == np.float32

    def test_channel_first_storage(self, dry_run_config: Config) -> None:
        """IR shape must be (C, T), not (T, C)."""
        sim = dry_run_simulator(
            n_channels=dry_run_config.n_channels,
            n_samples=dry_run_config.n_samples,
            sample_rate=dry_run_config.sample_rate,
        )
        ir = sim.render(make_scene(), ray_budget=1000).ir
        C, T = ir.shape
        assert C == dry_run_config.n_channels
        assert T == dry_run_config.n_samples
        assert C < T, "Sanity: C should be much smaller than T"

    def test_ir_numpy_roundtrip(self, tmp_path, dry_run_config: Config) -> None:
        """(C, T) ↔ numpy save/load is lossless."""
        sim = dry_run_simulator(
            n_channels=dry_run_config.n_channels,
            n_samples=dry_run_config.n_samples,
            sample_rate=dry_run_config.sample_rate,
        )
        ir = sim.render(make_scene(), ray_budget=1000).ir
        path = tmp_path / "ir.npy"
        np.save(path, ir)
        ir_loaded = np.load(path)
        assert np.array_equal(ir, ir_loaded), "IR numpy round-trip is not lossless"

    def test_channel_time_transpose_recoverable(self, dry_run_config: Config) -> None:
        """Transposing to (T, C) and back to (C, T) is lossless."""
        sim = dry_run_simulator(
            n_channels=dry_run_config.n_channels,
            n_samples=dry_run_config.n_samples,
            sample_rate=dry_run_config.sample_rate,
        )
        ir = sim.render(make_scene(), ray_budget=1000).ir  # (C, T)
        ir_time_major = ir.T                               # (T, C)
        ir_back = ir_time_major.T                          # (C, T)
        assert np.array_equal(ir, ir_back)


class TestEnergyTensorShape:
    """Energy tensors are (C, n_bands, n_frames)."""

    def _make_rep(self, config: Config) -> ThirdOctaveSpectrogram:
        return build_representation(
            config.representation.name, config.representation.params,
            sample_rate=config.sample_rate,
        )

    def test_encode_output_shape(self, dry_run_config: Config, sample_ir: np.ndarray) -> None:
        rep = self._make_rep(dry_run_config)
        energy = rep.encode(sample_ir)
        assert energy.ndim == 3
        C, n_bands, n_frames = energy.shape
        assert C == dry_run_config.n_channels
        assert n_bands == rep.n_bands
        assert n_frames > 0

    def test_encode_output_dtype(self, dry_run_config: Config, sample_ir: np.ndarray) -> None:
        rep = self._make_rep(dry_run_config)
        energy = rep.encode(sample_ir)
        assert energy.dtype == torch.float32

    def test_n_bands_positive(self, dry_run_config: Config) -> None:
        rep = self._make_rep(dry_run_config)
        assert rep.n_bands > 0

    def test_energy_values_finite(self, dry_run_config: Config, sample_ir: np.ndarray) -> None:
        rep = self._make_rep(dry_run_config)
        energy = rep.encode(sample_ir)
        assert torch.isfinite(energy).all(), "Energy tensor contains non-finite values"

    def test_energy_values_above_floor(self, dry_run_config: Config, sample_ir: np.ndarray) -> None:
        rep = self._make_rep(dry_run_config)
        energy = rep.encode(sample_ir)
        assert (energy >= rep.min_db).all(), "Energy tensor has values below min_db floor"

    def test_consistent_frames_across_scenes(self, dry_run_config: Config) -> None:
        """All scenes with same config produce energy tensors of the same shape."""
        rep = self._make_rep(dry_run_config)
        sim = dry_run_simulator(
            n_channels=dry_run_config.n_channels,
            n_samples=dry_run_config.n_samples,
            sample_rate=dry_run_config.sample_rate,
        )
        shapes = []
        for seed in range(5):
            ir = sim.render(make_scene(seed), ray_budget=1000).ir
            shapes.append(rep.encode(ir).shape)
        assert len(set(shapes)) == 1, f"Inconsistent energy tensor shapes: {shapes}"
