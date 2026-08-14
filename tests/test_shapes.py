"""Shape and round-trip invariants (design_spec §10 invariant #4)."""
import numpy as np
import pytest
import torch

from amcd.config import Config
from amcd.representations import build_representation
from amcd.representations.spectrogram import ThirdOctaveSpectrogram
from amcd.simulators.base import SceneSpec

import json
from pathlib import Path

from tests.conftest import CANONICAL_DRY_RUN, EVAL_FREQS, QUIET, dry_run_simulator


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
            sample_rate=config.sample_rate, eval_freqs_hz=EVAL_FREQS,
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


class TestTheSpectralSlopeIsRecordedAndTheRefusalIsNotSwallowed:
    """Narrowing the headroom guard removed a false rejection AND the
    disclosure that went with it.

    The guard originally took a minimum across ALL bands, which is a spectral-
    FLATNESS constraint: a steeply sloped render failed it even when both REPORTED
    metric bands had ample headroom. Narrowing the operand to those bands fixed the
    false rejection — and a 2nd-order 4 kHz lowpass now passes SILENTLY where
    it previously failed loudly. The slope is a real property of the render, so it is
    recorded rather than gated.

    The other half is why "err toward rejecting" is not itself a selection effect: a
    refusal aborts the run rather than dropping a scene. That rests on the ABSENCE of
    an except, which nothing asserted.
    """

    def test_preprocess_does_not_swallow_an_encode_refusal(self) -> None:
        """The absence of a try/except is what makes the guard safe.

        Swallowing a refusal per scene would silently drop scenes whose spectra sit
        near `min_db` — a population that correlates with absorption, i.e. with
        `test_material_shift`'s own declared axis — and shrink the split it came
        from with nothing recording it.

        Asserted over the module's AST rather than by grep so a comment mentioning
        `except` cannot satisfy it and a nested handler cannot hide from it.
        """
        import ast
        import inspect

        import amcd.data.preprocess as preprocess

        tree = ast.parse(inspect.getsource(preprocess))
        handlers = [n for n in ast.walk(tree) if isinstance(n, ast.ExceptHandler)]
        assert handlers == [], (
            f"preprocess.py now catches exceptions (line(s) "
            f"{[h.lineno for h in handlers]}). If `rep.encode`'s headroom refusal is "
            f"inside one, scenes near the min_db floor are dropped instead of "
            f"aborting the run — and which scenes those are correlates with "
            f"absorption."
        )

    def test_every_scene_and_leg_carries_a_slope(self, tmp_path: Path) -> None:
        from amcd.pipeline import Pipeline

        cfg = Config.load(*CANONICAL_DRY_RUN)
        pipe = Pipeline(cfg, tmp_path, QUIET)
        for stage in ("gen-scenes", "render", "preprocess"):
            pipe.run_stage(stage)
        meta = json.loads((tmp_path / "preprocessed" / "meta.json").read_text())

        slopes = meta["spectral_slope_db_per_decade"]
        assert slopes, "no scene recorded a spectral slope"
        for sid, legs in slopes.items():
            assert set(legs) == {"low", "high"}, (sid, legs)
            for leg, value in legs.items():
                assert value == value, f"{sid}/{leg} recorded a NaN slope"

    def test_the_slope_MOVES_when_the_spectrum_is_sloped(self) -> None:
        """The measurement, not just the key: a lowpassed IR must read a steeper
        slope than the same IR unfiltered, or this records a constant."""
        import numpy as np
        from scipy.signal import butter, sosfilt

        from amcd.data.preprocess import _spectral_slope_db_per_decade
        from amcd.representations.base import build_representation

        cfg = Config.load(*CANONICAL_DRY_RUN)
        rep = build_representation(
            cfg.representation.name, cfg.representation.params,
            sample_rate=cfg.sample_rate, eval_freqs_hz=EVAL_FREQS,
        )
        rng = np.random.default_rng(4)
        n = int(0.3 * cfg.sample_rate)
        t = np.arange(n) / cfg.sample_rate
        flat = (rng.standard_normal(n) * np.exp(-6.9 * t / 0.5)).astype(np.float32)
        sloped = sosfilt(
            butter(2, 4000.0, btype="low", fs=cfg.sample_rate, output="sos"), flat
        ).astype(np.float32)

        def slope(x):
            return _spectral_slope_db_per_decade(
                rep.encode(x[None, :]), rep.center_freqs
            )

        assert slope(sloped) < slope(flat) - 5.0, (
            f"a 2nd-order 4 kHz lowpass moved the recorded slope from "
            f"{slope(flat):.1f} to only {slope(sloped):.1f} dB/decade — this is not "
            f"measuring the spectral tilt the narrowed guard stopped seeing"
        )
