"""The simulator seam, end to end: config block, `build_simulator`, provenance,
the retained-path artifact, and the GSound-SIR render worker.

CONTENTS
  1. Config seam       TestSimulatorBlock … TestRayBudgetsStayTopLevel
                       the `{name, params}` grammar, sweeps, per-backend schemas
  2. Stage cache       TestStageFingerprint
  3. Scaffold physics  TestDryRunTailIsUnbiased, TestPlacementAxisIsAcousticallyLive
                       — dry_run.py's RENDERED physics, read back through
                       evaluation/room_acoustic; a different subject from the seam
  4. Path artifact     TestPathDataIsSelfDescribing, TestIRResultCarriesPaths
  5. Render stage      TestWrittenArtifactsCarryAnIntegrityRecord — digests, both-leg
                       shape checks, batch refusal (the STAGE, not the backend)
  6. gsound backend    TestTruncationDisclosure, TestGsoundProvenanceFill,
                       TestHostScopedParamsStayOutOfProvenance
  7. Render worker     stub pygsound / spherical_harmonics_rt, then
                       TestRenderWorkerContract — compiles the worker, checks its
                       imports, and RUNS it under a venv against those stubs

No section needs a GSound-SIR install: section 7 runs the real worker source
against stubs, so the whole file is runnable off the render host. Two tests do
need something extra and SKIP without it — the known-answer ambisonic measurement
(the render env) and the band-identity anchor (the pinned upstream checkout).
"""
import dataclasses
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pytest
from unittest import mock
import yaml

from amcd.config import Config, _BASE_YAML, _merge_layer
from amcd.pipeline import STAGE_FINGERPRINT, STAGES, Pipeline, _sentinel
from amcd.registry import simulator_registry
from amcd.simulators.base import (
    PATH_ARRAY_DTYPES,
    REQUIRED_PATH_DESCRIPTOR_KEYS,
    REQUIRED_PROVENANCE_KEYS,
    IRResult,
    PathData,
    SceneSpec,
    build_simulator,
    validate_path_descriptor,
    validate_provenance,
)

from tests.conftest import (
    CANONICAL_DRY_RUN,
    QUIET,
    TEST_TINY,
    dry_run_simulator,
    tiny_config,
)


def _merged(*layers: dict) -> dict:
    """base.yaml + test_tiny.yaml + extra layers, as a declared (unresolved) tree."""
    merged: dict = {}
    for path in (_BASE_YAML, TEST_TINY):
        with open(path) as f:
            _merge_layer(merged, yaml.safe_load(f) or {})
    for layer in layers:
        _merge_layer(merged, layer)
    return merged


class TestSimulatorBlock:
    """`simulator` is a {name, params} plugin block like model/representation."""

    def test_params_load_from_simulators_dir(self) -> None:
        cfg = Config.load(Path("configs/base.yaml"))
        params = cfg.simulator.params
        # Values come from configs/simulators/gsound_sir.yaml, never from base.yaml.
        assert params["commit_sha"] == "608ea30f6dc4cda149c18947f9cae48bd379fa27"
        assert params["normalize_ir"] is False
        assert params["path_retention"] == {"mode": "top_k", "value": 5000}
        assert len(params["frequency_points"]) == 7

    def test_inline_params_override_file(self) -> None:
        cfg = Config._from_merged(
            _merged({"simulator": {"name": "gsound_sir", "params": {"specular_count": 99}}}),
            None,
        )
        assert cfg.simulator.params["specular_count"] == 99
        assert cfg.simulator.params["commit_sha"]  # untouched keys still present

    def test_name_change_drops_prior_params(self) -> None:
        """Gsound_sir's params must not bleed onto dry_run, whose schema
        would reject every one of them."""
        cfg = Config.load(*CANONICAL_DRY_RUN)
        assert cfg.simulator.name == "dry_run"
        # dry_run gets its OWN params file; none of gsound's keys survive the switch
        # (its schema forbids extras, so any that did would fail loudly here).
        assert set(cfg.simulator.params) == {
            "speed_of_sound_m_s", "min_source_receiver_distance_m"}

    def test_unknown_simulator_fails_loud(self) -> None:
        with pytest.raises(ValueError, match="Unknown simulator"):
            Config._from_merged(_merged({"simulator": {"name": "no_such_sim"}}), None)

    def test_bare_string_simulator_rejected(self) -> None:
        """The pre-block-form spelling must fail loudly, not silently half-work."""
        with pytest.raises(ValueError, match="must be a mapping with a `name` key"):
            Config._from_merged(_merged({"simulator": "dry_run"}), None)


class TestSimulatorSweep:
    """Simulator params must reach the role grammar.

    `_PLUGIN_BLOCKS` membership only scopes params across a name change; without
    the `_from_merged` attach, a `sweep:` inside simulator params never reaches
    `_resolve_roles` and the roadmap's retained-path-count sweep is foreclosed.
    """

    def test_sweep_in_simulator_params_expands(self) -> None:
        cfg = Config._from_merged(
            _merged({
                "simulator": {
                    "name": "gsound_sir",
                    "params": {"path_retention": {"mode": "top_k",
                                                  "value": {"sweep": [1000, 5000, 20000]}}},
                }
            }),
            None,
        )
        siblings = cfg.expand_sweeps()
        assert len(siblings) == 3
        assert [s.simulator.params["path_retention"]["value"] for s in siblings] == [
            1000, 5000, 20000
        ]

    def test_sweep_is_recorded_as_a_role(self) -> None:
        cfg = Config._from_merged(
            _merged({
                "simulator": {"name": "gsound_sir",
                              "params": {"specular_count": {"sweep": [500, 2000]}}},
            }),
            None,
        )
        assert cfg.resolved_roles["simulator.params.specular_count"]["role"] == "swept"


class TestBuildSimulator:
    def test_builds_dry_run(self, dry_run_config: Config) -> None:
        sim = build_simulator(
            "dry_run", dry_run_config.simulator.params,
            n_channels=dry_run_config.n_channels,
            n_samples=dry_run_config.n_samples,
            sample_rate=dry_run_config.sample_rate,
        )
        assert type(sim) is simulator_registry.get("dry_run")

    def test_rejects_unknown_param(self) -> None:
        with pytest.raises(Exception, match="extra_forbidden|Extra inputs"):
            build_simulator("dry_run", {"speed_of_sound_m_s": 343.0, "bogus": 1},
                            n_channels=4, n_samples=800, sample_rate=8000)

    def test_gsound_rejects_normalization(self) -> None:
        """normalize_ir: true would destroy low↔high energy comparability."""
        params = {**Config.load(Path("configs/base.yaml")).simulator.params,
                  "normalize_ir": True}
        with pytest.raises(Exception, match="normalize_ir must be false"):
            build_simulator("gsound_sir", params,
                            n_channels=16, n_samples=144000, sample_rate=48000)

    def test_gsound_rejects_band_centres_as_edges(self) -> None:
        """The upstream test.py trap: 8 centres where 7 crossovers are required."""
        params = {**Config.load(Path("configs/base.yaml")).simulator.params,
                  "frequency_points": [63, 125, 250, 500, 1000, 2000, 4000, 8000]}
        with pytest.raises(Exception, match="band EDGES"):
            build_simulator("gsound_sir", params,
                            n_channels=16, n_samples=144000, sample_rate=48000)


class TestProvenanceContract:
    """Every simulator must declare a fixed provenance key set."""

    def test_dry_run_declares_every_required_key(self, dry_run_config: Config,
                                                 sample_scene) -> None:
        sim = build_simulator(
            "dry_run", dry_run_config.simulator.params,
            n_channels=dry_run_config.n_channels,
            n_samples=dry_run_config.n_samples,
            sample_rate=dry_run_config.sample_rate,
        )
        meta = sim.render(sample_scene, 50).meta
        assert set(REQUIRED_PROVENANCE_KEYS) <= set(meta)

    def test_missing_key_raises_naming_it(self) -> None:
        partial = {k: "x" for k in REQUIRED_PROVENANCE_KEYS if k != "speed_of_sound_m_s"}
        with pytest.raises(ValueError, match="speed_of_sound_m_s"):
            validate_provenance(partial, simulator_name="toy", scene_id="s0", leg="low")

    def test_render_stage_writes_canonical_meta_at_save_zero(self, tmp_path: Path) -> None:
        """Provenance is no longer verbosity-gated."""
        import json
        from amcd.scenes.generator import run_gen_scenes
        from amcd.simulators.render import run_render

        cfg = tiny_config(scenes={"n_id": 4})
        run_gen_scenes(cfg, tmp_path, QUIET)   # QUIET is save=0
        run_render(cfg, tmp_path, QUIET)

        metas = sorted(tmp_path.glob("renders/*/meta.json"))
        assert metas, "canonical render provenance missing at save=0"
        rec = json.loads(metas[0].read_text())
        assert rec["simulator"]["name"] == "dry_run"
        assert rec["low_ray_budget"] == cfg.low_ray_budget
        assert rec["high_ray_budget"] == cfg.high_ray_budget
        for leg in ("low", "high"):
            assert set(REQUIRED_PROVENANCE_KEYS) <= set(rec[leg])
        assert rec["low"]["ray_budget"] != rec["high"]["ray_budget"]


class TestRayBudgetsStayTopLevel:
    """The swept research axis must survive a simulator name change."""

    def test_budgets_survive_simulator_switch(self) -> None:
        cfg = Config.load(*CANONICAL_DRY_RUN)
        assert cfg.simulator.name == "dry_run"
        assert cfg.low_ray_budget == 5000
        assert cfg.high_ray_budget == 200000

    def test_budgets_are_not_simulator_params(self) -> None:
        params = Config.load(Path("configs/base.yaml")).simulator.params
        assert "low_ray_budget" not in params
        assert "high_ray_budget" not in params


class TestStageFingerprint:
    """A cached stage must belong to the CURRENT config."""

    def _pipeline(self, cfg: Config, run_dir: Path, force: bool = False) -> Pipeline:
        return Pipeline(cfg, run_dir, QUIET, force=force)

    def test_unchanged_config_reuses_cache(self, tmp_path: Path) -> None:
        cfg = tiny_config(scenes={"n_id": 4})
        self._pipeline(cfg, tmp_path).run_stage("gen-scenes")
        mtimes = {p: p.stat().st_mtime_ns for p in (tmp_path / "scenes").glob("scene_*.json")}
        self._pipeline(cfg, tmp_path).run_stage("gen-scenes")  # must skip, not rerun
        assert {p: p.stat().st_mtime_ns for p in (tmp_path / "scenes").glob("scene_*.json")} == mtimes

    def test_changed_simulator_param_fails_loudly(self, tmp_path: Path) -> None:
        cfg = tiny_config(scenes={"n_id": 4})
        self._pipeline(cfg, tmp_path).run_stage("gen-scenes")
        self._pipeline(cfg, tmp_path).run_stage("render")

        changed = tiny_config(scenes={"n_id": 4}, low_ray_budget=77)
        with pytest.raises(RuntimeError, match="cached under a DIFFERENT config"):
            self._pipeline(changed, tmp_path).run_stage("render")

    def test_mismatch_error_names_the_changed_field(self, tmp_path: Path) -> None:
        """A bare sha cannot tell the operator whether an expensive
        renders/ dir is salvageable."""
        cfg = tiny_config(scenes={"n_id": 4})
        self._pipeline(cfg, tmp_path).run_stage("gen-scenes")
        self._pipeline(cfg, tmp_path).run_stage("render")

        changed = tiny_config(scenes={"n_id": 4}, low_ray_budget=77)
        with pytest.raises(RuntimeError) as exc:
            self._pipeline(changed, tmp_path).run_stage("render")
        assert "low_ray_budget" in str(exc.value)
        assert "50" in str(exc.value) and "77" in str(exc.value)

    def test_render_chains_upstream_gen_scenes(self, tmp_path: Path) -> None:
        """Renders are per-scene, so changing the SCENES makes them stale
        even when every simulator parameter is untouched."""
        cfg = tiny_config(scenes={"n_id": 4})
        self._pipeline(cfg, tmp_path).run_stage("gen-scenes")
        self._pipeline(cfg, tmp_path).run_stage("render")

        moved = tiny_config(scenes={"n_id": 4, "margins": {"wall": 0.75}})
        # Caught via the UPSTREAM sentinel: render's own inputs are unchanged, so
        # only the chain can reveal that the scenes it rendered are now stale.
        with pytest.raises(RuntimeError, match="upstream stage 'gen-scenes'"):
            self._pipeline(moved, tmp_path).run_stage("render")

    def test_force_bypasses_the_check(self, tmp_path: Path) -> None:
        cfg = tiny_config(scenes={"n_id": 4})
        self._pipeline(cfg, tmp_path).run_stage("gen-scenes")
        changed = tiny_config(scenes={"n_id": 6})
        self._pipeline(changed, tmp_path, force=True).run_stage("gen-scenes")
        assert len(list((tmp_path / "scenes").glob("scene_*.json"))) == 6 + 6  # id + shifts

    def test_legacy_sentinel_raises_rather_than_guessing(self, tmp_path: Path) -> None:
        cfg = tiny_config(scenes={"n_id": 4})
        self._pipeline(cfg, tmp_path).run_stage("gen-scenes")
        _sentinel(tmp_path, "gen-scenes").write_text("1752438000.0")  # pre-fingerprint
        with pytest.raises(RuntimeError, match="no fingerprint"):
            self._pipeline(cfg, tmp_path).run_stage("gen-scenes")

    def test_an_unwired_stage_would_still_cache_on_a_bare_sentinel(
        self, tmp_path: Path
    ) -> None:
        """Every stage is DECLARED in `STAGE_FINGERPRINT`, so an unwired one is a
        stated gap rather than a silent absence.

        There is no unwired stage left to use as the example — `diagnostics` was
        the last, and citing it here is what made this test read as sanctioning the
        hole rather than recording it. The `None` path is exercised through a
        stubbed entry instead, so the behaviour stays pinned without needing a real
        stage to be broken for it.
        """
        assert set(STAGE_FINGERPRINT) == set(STAGES)

        cfg = tiny_config(scenes={"n_id": 4})
        pipe = self._pipeline(cfg, tmp_path)
        with mock.patch.dict(STAGE_FINGERPRINT, {"diagnostics": None}):
            pipe._mark_done("diagnostics")
            assert pipe._is_done("diagnostics")


class TestDryRunTailIsUnbiased:
    """The scaffold's diffuse tail must not BE the noise.

    It previously read `diffuse = decay * noise * noise_scale` with
    `noise_scale = 1/sqrt(N)`, so E[tail energy] scaled as 1/N. Measured, the low
    leg's late-window energy was 39.7x the high leg's — 16.0 dB, exactly
    200000/5000 — which makes low→high a deterministic level shift (trivially
    learnable) whose converged limit is an IR with no reverberant tail at all.
    That is the mechanism behind the dry_run D0b "CARRIER BOTTLENECK" verdict
    being a plumbing artifact rather than a result.

    A ray budget controls VARIANCE, not level: the late-window energy must agree
    between legs while the per-sample variance still falls as 1/N.
    """

    def _late_window(self, ir, sample_rate):
        lo = int(0.20 * sample_rate)
        hi = int(1.00 * sample_rate)
        return ir[0, lo:hi].astype(float)

    def test_tail_level_is_budget_independent_but_variance_is_not(self) -> None:
        cfg = tiny_config(sample_rate=48000, ir_duration=1.5)
        sim = dry_run_simulator(
            n_channels=cfg.n_channels, n_samples=cfg.n_samples,
            sample_rate=cfg.sample_rate,
        )
        scene = SceneSpec(
            scene_id="scene_ac18", seed=4242,
            dims=(10.0, 8.0, 3.5), material_absorption=0.15, geometry_family="shoebox",
            source_pos=(2.0, 2.0, 1.5), receiver_pos=(7.0, 6.0, 1.5),
            split_regime="id", regime_axes={},
        )
        low = self._late_window(sim.render(scene, 5000).ir, cfg.sample_rate)
        high = self._late_window(sim.render(scene, 200000).ir, cfg.sample_rate)

        energy_ratio = float((low ** 2).sum() / (high ** 2).sum())
        assert 0.9 < energy_ratio < 1.1, (
            f"late-window energy ratio low/high = {energy_ratio:.2f}; the tail LEVEL "
            f"still moves with the ray budget (pre-fix this was 39.7 = 200000/5000, "
            f"a deterministic -16 dB shift the model can learn trivially)."
        )

        # The tail must remain a WAVEFORM, not an envelope. `decay*(1 + n·σ)` would
        # also equalise the energy, but is strictly positive — and a positive tail
        # carries almost no energy in the 500/1000 Hz eval bands after octave
        # filtering, hollowing out the ISO metrics this scaffold exists to exercise.
        assert (np.diff(np.sign(low)) != 0).sum() > len(low) // 10, (
            "the diffuse tail does not oscillate about zero — it is an envelope, not "
            "an impulse response, and will not survive octave-band filtering."
        )

        # The budget must still do something: the legs differ by MC estimation noise
        # that shrinks as 1/sqrt(N). Known answer — with the converged response shared,
        # std(low - high) / rms(high) ≈ sqrt(1/5000 + 1/200000) = 0.0143.
        rel_noise = float(np.std(low - high) / np.sqrt((high ** 2).mean()))
        expected = (1 / 5000 + 1 / 200000) ** 0.5
        assert 0.5 * expected < rel_noise < 2.0 * expected, (
            f"low-vs-high difference is {rel_noise:.5f} of the reference rms, expected "
            f"~{expected:.5f} (1/sqrt(N) Monte-Carlo convergence). Too small means the "
            f"ray budget no longer controls anything and there is nothing to denoise; "
            f"too large means the legs differ by more than estimation noise."
        )


class TestPlacementAxisIsAcousticallyLive:
    """The scaffold's "direct sound" was not a direct sound.

    `direct = direct_gain * exp(-t/0.02)` is a one-pole envelope with a 7.96 Hz
    corner, so only 6.06e-7 of its energy reached the 500 Hz octave band. MEASURED
    in a 10x8x3.5 m room at alpha 0.2 (r_c = 1.19 m): C50 read 1.966 / 1.957 /
    1.953 / 1.951 / 1.950 dB at d = 0.5, 1, 2, 4 and 8 m — flat to 0.02 dB across a
    16x distance range — while the closed-form DRR the scene report publishes swung
    +7.55 to -16.53 dB. `test_placement_shift` therefore carried NO acoustic
    difference from the id baseline in any reported ISO-3382 metric.

    SCOPE OF THESE TESTS. The DRR agreement below is a SCAFFOLD
    SELF-CONSISTENCY check: the diffuse tail is scaled by
    sqrt(16*pi / (R * sum(decay^2))) precisely so the rendered DRR equals the
    closed form, so it verifies that the two share one formula — it does
    NOT validate the ISO path against independent physics. The independent check is
    the Step-6 probe against a real gsound render. What IS non-circular
    here is the C50 SHAPE: nothing in the construction forces C50, an quantity computed through octave filtering and Schroeder integration, to track
    distance at all.
    """

    _DIMS = (10.0, 8.0, 3.5)
    _ALPHA = 0.2
    _SURFACE = 2.0 * (10.0 * 8.0 + 8.0 * 3.5 + 10.0 * 3.5)

    def _render(self, distance: float):
        from amcd.simulators.dry_run import DryRunSimulator

        sim = DryRunSimulator(
            n_channels=1, n_samples=48000, sample_rate=48000,
            speed_of_sound_m_s=343.0, min_source_receiver_distance_m=0.3,
        )
        scene = SceneSpec(
            scene_id="s", seed=1, geometry_family="shoebox", dims=self._DIMS,
            material_absorption=self._ALPHA, source_pos=(1.0, 1.0, 1.5),
            receiver_pos=(1.0 + distance, 1.0, 1.5), sim_params={},
            split_regime="id", regime_axes={},
        )
        return sim.render(scene, 200000)

    def _c50(self, distance: float) -> float:
        from amcd.evaluation.room_acoustic import channel_band_avg_metrics

        values, _ = channel_band_avg_metrics(
            self._render(distance).ir[0], sample_rate=48000,
            iso_eval_freqs=[500.0, 1000.0], onset_rel_db=-20.0,
            band_resolvability_margin=0.0,
            decay_range_fit=_decay_fit(), min_decay_range_db={"T30": 0.0, "EDT": 0.0},
            octave_filter_order=Config.load(
                Path("configs/base.yaml")
            ).metric_octave_filter.order,
        )
        return values["C50"]

    def test_c50_moves_with_distance(self) -> None:
        """The kill assertion. Pre-fix the spread over this range was 0.016 dB."""
        c50 = [self._c50(d) for d in (0.5, 1.0, 2.0, 4.0, 8.0)]
        assert c50 == sorted(c50, reverse=True), f"C50 not monotone in distance: {c50}"
        assert c50[0] - c50[-1] > 6.0, (
            f"C50 spans only {c50[0] - c50[-1]:.3f} dB over a 16x distance range "
            f"({c50}) — the placement axis is acoustically inert again"
        )

    def test_the_direct_arrival_is_the_loudest_sample(self) -> None:
        """`find_onset` documents this as an assumption. Pre-fix the global
        peak sat 300-550 samples INTO the diffuse tail, violating it — inert only
        because the whole response starts at d/c."""
        from amcd.evaluation.room_acoustic import find_onset

        for d in (0.5, 2.0, 8.0):
            ir = self._render(d).ir[0]
            assert int(np.argmax(np.abs(ir))) == find_onset(ir, -20.0)[0]

    def test_realized_drr_matches_the_published_closed_form(self) -> None:
        """SCAFFOLD SELF-CONSISTENCY, not metric validation — see the class
        docstring. The residual grows with distance because the direct sample and
        the tail's first sample are superposed, which matters more as the direct
        term shrinks."""
        from amcd.acoustics import diffuse_field_drr_db
        from amcd.evaluation.room_acoustic import find_onset

        for d in (0.5, 1.0, 2.0, 4.0, 8.0):
            ir = self._render(d).ir[0]
            onset, _ = find_onset(ir, -20.0)
            direct = float(ir[onset]) ** 2
            reverberant = float(np.sum(ir[onset + 1:].astype(np.float64) ** 2))
            realized = 10.0 * np.log10(direct / reverberant)
            expected = diffuse_field_drr_db(self._SURFACE, self._ALPHA, d)
            assert realized == pytest.approx(expected, abs=1.0), (
                f"rendered DRR {realized:.2f} dB vs published {expected:.2f} dB at "
                f"d={d} m — the scaffold and scenes/placement_report.json have "
                f"stopped sharing one formula"
            )

    def test_the_realized_snr_is_stamped(self) -> None:
        """Caveat needs a magnitude, and the Step-6 probe needs a
        number to put a real gsound render beside."""
        meta = self._render(2.0).meta
        assert meta["realized_snr_db"] == pytest.approx(10.0 * np.log10(200000), abs=0.01)
        assert "modelling assumption" in meta["noise_scale_basis"]


def _fake_paths(n: int = 6, n_bands: int = 8, **descriptor_overrides) -> PathData:
    """A PathData with the upstream shapes, for tests that need no render env.

    Values are arbitrary; SHAPES and DTYPES are not — they are the contract
    `PATH_ARRAY_DTYPES` pins against the real `getPathData` output.
    """
    descriptor = {
        "simulator": "gsound_sir",
        "commit_sha": "608ea30f6dc4cda149c18947f9cae48bd379fa27",
        "band_edges_hz": [88.7412, 176.7767, 353.5534, 707.1068,
                          1414.2136, 2828.4271, 5656.8542],
        "band_centres_hz": [63.0, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0],
        "sample_rate": 48000,
        "speed_of_sound_m_s": 344.0,
        "path_retention": {"mode": "top_k", "value": 5000},
        "ray_budget": 5000,
        "leg": "low",
        "realization_index": 0,
    }
    descriptor.update(descriptor_overrides)
    return PathData(
        distances=np.linspace(1.0, 9.0, n).astype("float32"),
        intensities=np.arange(n * n_bands, dtype="float32").reshape(n, n_bands),
        listener_directions=np.zeros((n, 3), dtype="float32"),
        source_directions=np.ones((n, 3), dtype="float32"),
        path_types=np.arange(n, dtype="uint32"),
        speeds_of_sound=np.full(n, 344.0, dtype="float32"),
        relative_speeds=np.zeros(n, dtype="float32"),
        source_indices=np.zeros(n, dtype="uint64"),
        num_paths=n,
        num_bands=n_bands,
        total_energy=1.25,
        kept_energy_percentage=98.5,
        descriptor=descriptor,
    )


class TestPathDataIsSelfDescribing:
    """A retained-path file must be interpretable WITHOUT its config.

    `intensities` is (N, 8) and the band meaning of those 8 columns lives only in
    the simulator config that produced them. A path file from a second raytracer —
    which the roadmap wants — would therefore be unreadable the moment it was
    separated from that config, so the descriptor travels inside the parquet.
    """

    def test_round_trip_preserves_arrays_and_descriptor(self, tmp_path: Path) -> None:
        paths = _fake_paths()
        target = tmp_path / "paths_low.parquet"
        paths.to_parquet(target)
        back = PathData.from_parquet(target)

        assert back.descriptor == paths.descriptor
        for name, dtype in PATH_ARRAY_DTYPES.items():
            original, reloaded = getattr(paths, name), getattr(back, name)
            assert reloaded.shape == original.shape, name
            assert reloaded.dtype == np.dtype(dtype), name
            assert np.array_equal(reloaded, original), name
        assert (back.num_paths, back.num_bands) == (paths.num_paths, paths.num_bands)
        assert back.kept_energy_percentage == paths.kept_energy_percentage

    def test_band_axis_survives_as_a_shape(self, tmp_path: Path) -> None:
        """Not 8 positionally-named columns whose meaning a reader must rebuild."""
        target = tmp_path / "p.parquet"
        _fake_paths(n=4).to_parquet(target)
        assert PathData.from_parquet(target).intensities.shape == (4, 8)

    def test_the_descriptor_names_each_intensity_column(self, tmp_path: Path) -> None:
        """The point of the row: num_bands columns, num_bands named centres."""
        target = tmp_path / "p.parquet"
        _fake_paths().to_parquet(target)
        back = PathData.from_parquet(target)
        assert len(back.descriptor["band_centres_hz"]) == back.num_bands
        assert len(back.descriptor["band_edges_hz"]) == back.num_bands - 1

    def test_a_file_without_its_descriptor_is_rejected(self, tmp_path: Path) -> None:
        """A path file that lost its metadata must fail loudly, not load headless."""
        import pyarrow as pa
        import pyarrow.parquet as pq

        bare = tmp_path / "bare.parquet"
        pq.write_table(pa.table({"distances": pa.array([1.0, 2.0])}), bare)
        with pytest.raises(ValueError, match="uninterpretable by design"):
            PathData.from_parquet(bare)

    def test_missing_descriptor_key_is_named(self) -> None:
        paths = _fake_paths()
        del paths.descriptor["band_centres_hz"]
        with pytest.raises(ValueError, match="band_centres_hz"):
            validate_path_descriptor(paths, simulator_name="gsound_sir", scene_id="s0")

    def test_intensities_must_match_the_declared_band_count(self) -> None:
        # `replace` re-runs __post_init__, which is the validation under test.
        with pytest.raises(ValueError, match="describe bands it does not contain"):
            dataclasses.replace(_fake_paths(n_bands=8), num_bands=4)

    def test_the_descriptor_must_name_as_many_bands_as_there_are_columns(self) -> None:
        """Presence of the band keys is not interpretability.

        `__post_init__` compares `num_bands` against `intensities` — two numbers
        from the SAME producer, self-consistent by construction. Nothing checked
        that the descriptor NAMES that many bands, so a file could declare 8 centres
        over 9 intensity columns and pass every existing guard.
        """
        paths = _fake_paths(n_bands=8)
        paths.descriptor["band_centres_hz"] = [63.0, 125.0, 250.0]
        with pytest.raises(ValueError, match="band centres .* for 8 intensity columns"):
            validate_path_descriptor(paths, simulator_name="gsound_sir", scene_id="s0")

        paths = _fake_paths(n_bands=8)
        paths.descriptor["band_edges_hz"] = [88.7412, 176.7767]
        with pytest.raises(ValueError, match="band edges"):
            validate_path_descriptor(paths, simulator_name="gsound_sir", scene_id="s0")

    def test_a_wider_array_is_accepted_when_the_cast_is_exact(self, tmp_path: Path) -> None:
        """The round trip cast on READ only, so float64 in gave float32 back
        with no error. The declared dtype is the contract, enforced at construction.

        A wider input whose values ARE representable is narrowed and round-trips
        unchanged — the check is value-based, so a Python list of ints or an
        incidental float64 is not rejected for its dtype alone.
        """
        wide = dataclasses.replace(
            _fake_paths(),
            distances=np.array([1.0, 2.5, 4.25, 5.5, 7.0, 9.0], dtype="float64"),
            intensities=np.arange(48, dtype="float64").reshape(6, 8),
        )
        assert wide.distances.dtype == np.dtype(PATH_ARRAY_DTYPES["distances"])
        assert wide.intensities.dtype == np.dtype(PATH_ARRAY_DTYPES["intensities"])

        target = tmp_path / "p.parquet"
        wide.to_parquet(target)
        back = PathData.from_parquet(target)
        for name, dtype in PATH_ARRAY_DTYPES.items():
            assert getattr(back, name).dtype == np.dtype(dtype), name
            np.testing.assert_array_equal(getattr(back, name), getattr(wide, name))

    def test_a_lossy_cast_raises_instead_of_narrowing_in_silence(self) -> None:
        """`np.asarray(x, dtype=...)` quietly turns float64 into
        float32, so a second raytracer's higher-precision distances would be
        truncated on the way in with nothing recorded — the silent exclusion the
        drop log exists to prevent, one layer down. 2.6 is not representable in
        float32, so this input cannot survive the declared dtype."""
        with pytest.raises(ValueError, match="LOSSY"):
            dataclasses.replace(
                _fake_paths(),
                distances=np.linspace(1.0, 9.0, 6).astype("float64"),
            )

    def test_an_undefined_kept_share_round_trips_as_none_not_zero(self, tmp_path: Path) -> None:
        """0.0 would read as 'we retained almost nothing' for a subset that in
        fact holds every path. An unscored quantity is not rendered as a number."""
        undefined = dataclasses.replace(
            _fake_paths(), total_energy=0.0, kept_energy_percentage=None
        )
        target = tmp_path / "p.parquet"
        undefined.to_parquet(target)
        back = PathData.from_parquet(target)
        assert back.kept_energy_percentage is None
        assert back.total_energy == 0.0

    def test_the_file_identifies_its_render_without_the_filename(self, tmp_path: Path) -> None:
        """`paths_{low,high}.parquet` encodes two legs and one
        realization. The artifact layout must not foreclose a realization index, so
        the identity lives in the file's own metadata, not in its name."""
        target = tmp_path / "renamed_by_someone.parquet"
        _fake_paths().to_parquet(target)
        back = PathData.from_parquet(target)
        assert back.descriptor["ray_budget"] == 5000
        assert back.descriptor["leg"] == "low"
        assert back.descriptor["realization_index"] == 0


class TestIRResultCarriesPaths:
    """`IRResult.paths` is the producer half of the path-conditioned seam.

    Design_spec §8 shows `IRResult` carrying `paths: PathData`; the field lands with
    its producer, never speculatively before it.
    """

    def test_scaffold_leg_has_no_paths(self, dry_run_config: Config, sample_scene) -> None:
        sim = build_simulator(
            "dry_run", dry_run_config.simulator.params,
            n_channels=dry_run_config.n_channels,
            n_samples=dry_run_config.n_samples,
            sample_rate=dry_run_config.sample_rate,
        )
        assert sim.render(sample_scene, 50).paths is None

    def test_the_stage_writes_no_path_file_for_a_backend_without_paths(
        self, tmp_path: Path
    ) -> None:
        """The scaffolding rule: the stage keys on the FIELD, so a backend that
        exports no paths needs no downstream edit and no isinstance check."""
        from amcd.scenes.generator import run_gen_scenes
        from amcd.simulators.render import run_render

        cfg = tiny_config(scenes={"n_id": 2})
        run_gen_scenes(cfg, tmp_path, QUIET)
        run_render(cfg, tmp_path, QUIET)

        assert not list(tmp_path.glob("renders/*/paths_*.parquet"))
        # …while the canonical artifacts are all still there.
        assert list(tmp_path.glob("renders/*/low.npy"))
        assert list(tmp_path.glob("renders/*/meta.json"))

    def test_the_stage_writes_and_stamps_the_leg_when_paths_exist(
        self, tmp_path: Path
    ) -> None:
        """The producer knows its ray budget; the STAGE owns the leg's label."""
        from amcd.scenes.generator import run_gen_scenes
        from amcd.simulators.render import run_render

        cfg = tiny_config(scenes={"n_id": 2})

        # A backend that exports paths, registered for this test only. Registering a
        # double is how the stage's paths branch is exercised without a render host;
        # the entry is removed again so no other test sees it.
        class _PathyDryRun(simulator_registry.get("dry_run")):
            def render(self, scene, ray_budget):
                result = super().render(scene, ray_budget)
                paths = _fake_paths(n=3)
                paths.descriptor["ray_budget"] = ray_budget
                paths.descriptor["leg"] = None      # unset until the stage stamps it
                return IRResult(ir=result.ir, meta=result.meta, paths=paths)

        simulator_registry.register("pathy_dry_run")(_PathyDryRun)
        try:
            cfg = tiny_config(
                scenes={"n_id": 2},
                simulator={"name": "pathy_dry_run",
                           "params": cfg.simulator.params},
            )
            run_gen_scenes(cfg, tmp_path, QUIET)
            run_render(cfg, tmp_path, QUIET)

            written = sorted(p.name for p in tmp_path.glob("renders/scene_0000/paths_*"))
            assert written == ["paths_high.parquet", "paths_low.parquet"]
            low = PathData.from_parquet(tmp_path / "renders/scene_0000/paths_low.parquet")
            high = PathData.from_parquet(tmp_path / "renders/scene_0000/paths_high.parquet")
            assert low.descriptor["leg"] == "low"
            assert high.descriptor["leg"] == "high"
            assert low.descriptor["ray_budget"] == cfg.low_ray_budget
            assert high.descriptor["ray_budget"] == cfg.high_ray_budget
            assert set(REQUIRED_PATH_DESCRIPTOR_KEYS) <= set(low.descriptor)

            # every artifact this scene produced is digested into meta.json,
            # including the path files.
            meta = json.loads((tmp_path / "renders/scene_0000/meta.json").read_text())
            assert set(meta["artifact_sha256"]) == {
                "low.npy", "high.npy", "paths_low.parquet", "paths_high.parquet",
            }
        finally:
            simulator_registry._entries.pop("pathy_dry_run", None)


class TestWrittenArtifactsCarryAnIntegrityRecord:
    """`rng_seeded: false` puts reproducibility on the cached artifacts.

    Those artifacts carried no digest, so two physically different datasets had
    byte-identical provenance and a truncated or half-written IR was undetectable.
    """

    def _render(self, tmp_path: Path):
        from amcd.scenes.generator import run_gen_scenes
        from amcd.simulators.render import run_render

        cfg = tiny_config(scenes={"n_id": 1})
        run_gen_scenes(cfg, tmp_path, QUIET)
        run_render(cfg, tmp_path, QUIET)
        return tmp_path / "renders/scene_0000"

    def test_the_digest_matches_the_bytes_on_disk(self, tmp_path: Path) -> None:
        import hashlib

        out = self._render(tmp_path)
        meta = json.loads((out / "meta.json").read_text())

        assert set(meta["artifact_sha256"]) == {"low.npy", "high.npy"}
        for name, digest in meta["artifact_sha256"].items():
            assert hashlib.sha256((out / name).read_bytes()).hexdigest() == digest

    def test_only_what_the_stage_wrote_is_digested(self, tmp_path: Path) -> None:
        """A directory scan would digest whatever the HOST left in the run dir.

        Found by the dry run itself: on macOS over a non-native filesystem the OS
        writes AppleDouble `._low.npy` sidecars beside each artifact, which a scan
        picked up — putting a host fact into canonical provenance and making the same
        render's meta.json differ between the two supported hosts.
        """
        from amcd.scenes.generator import run_gen_scenes
        from amcd.simulators.render import run_render

        cfg = tiny_config(scenes={"n_id": 1})
        run_gen_scenes(cfg, tmp_path, QUIET)

        renders = tmp_path / "renders" / "scene_0000"
        renders.mkdir(parents=True, exist_ok=True)
        (renders / "._low.npy").write_bytes(b"AppleDouble sidecar")
        (renders / ".DS_Store").write_bytes(b"host clutter")

        run_render(cfg, tmp_path, QUIET)

        meta = json.loads((renders / "meta.json").read_text())
        assert set(meta["artifact_sha256"]) == {"low.npy", "high.npy"}

    def test_a_corrupted_artifact_no_longer_matches_its_provenance(
        self, tmp_path: Path
    ) -> None:
        """The point of the digest: the failure it makes visible."""
        import hashlib

        out = self._render(tmp_path)
        meta = json.loads((out / "meta.json").read_text())

        ir = np.load(out / "low.npy")
        ir[0, 0] += 1.0
        np.save(out / "low.npy", ir)

        recomputed = hashlib.sha256((out / "low.npy").read_bytes()).hexdigest()
        assert recomputed != meta["artifact_sha256"]["low.npy"]

    def test_a_refused_scene_does_not_abort_the_scenes_after_it(
        self, tmp_path: Path
    ) -> None:
        """A backend refusal at scene 500 of 720 used to abort mid-loop.

        `_preflight_separations` collects every offender for exactly this reason —
        an emulated batch aborted hours in has to be redone from scratch. Every
        scene is now attempted, the refused one is EXCLUDED from the manifest
        (RI §B.4), and the scenes after it are rendered and admitted.
        """
        import json

        from amcd.scenes.generator import run_gen_scenes
        from amcd.simulators.base import SceneRefused
        from amcd.simulators.render import run_render

        cfg = tiny_config(scenes={"n_id": 3})
        run_gen_scenes(cfg, tmp_path, QUIET)

        class _RefusesOneScene(simulator_registry.get("dry_run")):
            def render(self, scene, ray_budget):
                if scene.scene_id == "scene_0001":
                    raise SceneRefused("zero total energy in the window (simulated)")
                return super().render(scene, ray_budget)

        simulator_registry.register("refuses_one")(_RefusesOneScene)
        try:
            bad = tiny_config(
                scenes={"n_id": 3},
                simulator={"name": "refuses_one", "params": cfg.simulator.params},
                # One refusal in three is 33 %; the bounds exist to stop a broken
                # backend, and this test is about the per-scene path.
                max_excluded_frac=1.0,
                max_refused_frac=1.0,
            )
            run_render(bad, tmp_path, QUIET)

            manifest = json.loads((tmp_path / "renders/manifest.json").read_text())
            assert "scene_0001" not in manifest["admitted"]
            assert len(manifest["admitted"]) == manifest["generated"] - 1
            excluded = manifest["excluded"]
            assert [e["scene_id"] for e in excluded] == ["scene_0001"]
            assert excluded[0]["category"] == "refused"
            assert "zero total energy" in excluded[0]["reason"]
            # The scenes after the refusal were still rendered.
            assert (tmp_path / "renders/scene_0002/low.npy").exists()
            assert (tmp_path / "renders/scene_0000/low.npy").exists()
        finally:
            simulator_registry._entries.pop("refuses_one", None)

    def test_a_backend_contract_failure_aborts_instead_of_excluding(
        self, tmp_path: Path
    ) -> None:
        """The other half of the rule, and the one exclusion put at risk.

        `SceneRefused` says THIS SCENE is bad. Every other error class says the
        BACKEND is bad, which means every remaining scene fails identically — so
        catching it per scene would "exclude" all 720 one at a time and hand back
        an empty dataset instead of an error.
        """
        from amcd.scenes.generator import run_gen_scenes
        from amcd.simulators.render import run_render

        cfg = tiny_config(scenes={"n_id": 3})
        run_gen_scenes(cfg, tmp_path, QUIET)

        class _BrokenBackend(simulator_registry.get("dry_run")):
            def render(self, scene, ray_budget):
                raise ValueError("worker returned 4 channels, expected 16")

        simulator_registry.register("broken_backend")(_BrokenBackend)
        try:
            bad = tiny_config(
                scenes={"n_id": 3},
                simulator={"name": "broken_backend", "params": cfg.simulator.params},
                max_excluded_frac=1.0,
                max_refused_frac=1.0,
            )
            with pytest.raises(ValueError, match="expected 16"):
                run_render(bad, tmp_path, QUIET)
            assert not (tmp_path / "renders/manifest.json").exists(), (
                "a contract failure wrote a manifest, so the run produced a "
                "'dataset' from a backend that rendered nothing"
            )
        finally:
            simulator_registry._entries.pop("broken_backend", None)

    def test_both_legs_shapes_are_checked_and_it_raises(self, tmp_path: Path) -> None:
        """The guard covered the LOW leg only and was a bare `assert`, which
        `python -O` strips — leaving the canonical artifact's only shape check
        conditionally absent."""
        from amcd.scenes.generator import run_gen_scenes
        from amcd.simulators.render import run_render

        cfg = tiny_config(scenes={"n_id": 1})
        run_gen_scenes(cfg, tmp_path, QUIET)

        class _BadHighLeg(simulator_registry.get("dry_run")):
            def render(self, scene, ray_budget):
                result = super().render(scene, ray_budget)
                if ray_budget == cfg.high_ray_budget:
                    result.ir = result.ir[:, :-1]      # only the HIGH leg is wrong
                return result

        simulator_registry.register("bad_high_leg")(_BadHighLeg)
        try:
            bad = tiny_config(
                scenes={"n_id": 1},
                simulator={"name": "bad_high_leg", "params": cfg.simulator.params},
            )
            with pytest.raises(ValueError, match="leg 'high'.*returned an IR of shape"):
                run_render(bad, tmp_path, QUIET)
        finally:
            simulator_registry._entries.pop("bad_high_leg", None)


def _gsound_sim(**param_overrides):
    """Build gsound_sir through the seam, with config params optionally overridden.

    Params come from configs/simulators/gsound_sir.yaml — never hardcoded here — so
    these tests keep exercising the same `build_simulator` path the stage uses.
    """
    cfg = Config.load(Path("configs/base.yaml"))
    params = {**cfg.simulator.params, **param_overrides}
    return build_simulator(
        "gsound_sir", params,
        n_channels=cfg.n_channels, n_samples=cfg.n_samples, sample_rate=cfg.sample_rate,
    )


class TestTruncationDisclosure:
    """Trimming gsound's natural IR to `ir_duration` discards tail energy.

    On the most reverberant scenes that trim can silently invalidate T30/EDT, so
    what was discarded is MEASURED per (scene, leg) rather than assumed negligible.
    """

    def test_a_short_native_ir_is_padded_and_not_flagged(self) -> None:
        sim = _gsound_sim()
        native = np.ones((16, 1000), dtype=np.float32)
        ir, disclosure = sim._fit_to_window(native)

        assert ir.shape == (16, sim.n_samples)
        assert np.array_equal(ir[:, :1000], native)
        assert not ir[:, 1000:].any()
        assert disclosure["truncated"] is False
        assert disclosure["native_ir_samples"] == 1000
        # Nothing was discarded, so there is no number to report — and an unmeasured
        # quantity must never be rendered as one (not 0.0, not -inf).
        assert disclosure["discarded_tail_db"] is None
        assert disclosure["truncation_qc_flag"] is False

    def test_a_long_native_ir_is_trimmed_and_the_loss_measured(self) -> None:
        sim = _gsound_sim()
        native = np.zeros((16, sim.n_samples + 100), dtype=np.float32)
        # Known answer: equal energy inside and outside the window → half the total
        # is discarded → 10*log10(0.5) = -3.0103 dB.
        native[0, 0] = 1.0
        native[0, sim.n_samples] = 1.0
        ir, disclosure = sim._fit_to_window(native)

        assert ir.shape == (16, sim.n_samples)
        assert disclosure["truncated"] is True
        assert disclosure["native_ir_samples"] == sim.n_samples + 100
        assert disclosure["discarded_tail_db"] == pytest.approx(-3.0103, abs=1e-3)

    def test_the_qc_threshold_is_config_declared_and_flags_a_breach(self) -> None:
        """The threshold is a config value, not a literal: a different declared
        level must move the flag with no code change."""
        native = np.zeros((16, 300000), dtype=np.float32)
        native[0, 0] = 1.0
        native[0, 250000] = 0.001            # discarded tail at -60.0 dB of total

        lenient = _gsound_sim(max_discarded_tail_db=-20.0)._fit_to_window(native)[1]
        strict = _gsound_sim(max_discarded_tail_db=-90.0)._fit_to_window(native)[1]

        assert lenient["discarded_tail_db"] == pytest.approx(strict["discarded_tail_db"])
        assert lenient["truncation_qc_flag"] is False
        assert strict["truncation_qc_flag"] is True
        assert strict["max_discarded_tail_db"] == -90.0

    def test_a_silent_leg_is_distinguishable_from_a_healthy_one(self) -> None:
        """`total_energy == 0` used to be folded into 'nothing was discarded',
        so a zero-energy render produced the SAME all-clear disclosure as a good one
        and first surfaced as a NaN in a metric, hours downstream."""
        sim = _gsound_sim()
        healthy = np.zeros((16, 1000), dtype=np.float32)
        healthy[0, 0] = 1.0

        assert sim._fit_to_window(healthy)[1]["fitted_ir_total_energy"] > 0.0
        assert sim._fit_to_window(np.zeros((16, 1000), dtype=np.float32))[1][
            "fitted_ir_total_energy"
        ] == 0.0

    def test_the_two_energies_name_the_arrays_they_describe(self) -> None:
        """The guard must read the array that SHIPS.

        A native IR whose energy lies entirely beyond the window is trimmed to
        silence: native energy is non-zero, the stored array is all zeros. Guarding
        the native energy would pass exactly the leg exists to refuse.
        """
        sim = _gsound_sim()
        native = np.zeros((16, sim.n_samples + 1000), dtype=np.float32)
        native[0, sim.n_samples + 500] = 1.0      # the only arrival is past the window

        ir, disclosure = sim._fit_to_window(native)

        assert disclosure["native_ir_total_energy"] == 1.0
        assert disclosure["fitted_ir_total_energy"] == 0.0
        assert not ir.any()
        assert disclosure["fitted_ir_samples"] == sim.n_samples

    def test_retention_values_are_range_checked_per_mode(self) -> None:
        """Untyped, an out-of-domain value was not rejected but silently
        REINTERPRETED — `top_k: 0` and `top_k: -3` both meant 'keep everything',
        `top_k: 5000.7` truncated, `top_percent: 150` meant 'all'."""
        for bad in (0, -3, 5000.7):
            with pytest.raises(Exception, match="whole number of paths"):
                _gsound_sim(path_retention={"mode": "top_k", "value": bad})
        for bad in (0.0, -1.0, 150.0):
            with pytest.raises(Exception, match=r"share in \(0, 100\]"):
                _gsound_sim(path_retention={"mode": "top_percent", "value": bad})

        # …and the canonical values still build.
        assert _gsound_sim(path_retention={"mode": "top_k", "value": 5000})
        assert _gsound_sim(path_retention={"mode": "top_percent", "value": 100.0})
        assert _gsound_sim(path_retention={"mode": "all", "value": None})


class TestGsoundProvenanceFill:
    """What the gsound leg must be able to state about itself."""

    def test_the_ambisonic_stamp_is_measured_against_upstream_not_asserted(self) -> None:
        """KNOWN-ANSWER test of the encoding this project stamps.

        The stamp used to be checked as `_AMBISONIC_CONVENTION == "acn_n3d"` — the
        constant against itself, which cannot fail. It is the one upstream-compiled
        fact here taken from a source COMMENT, while its sibling `speed_of_sound_m_s`
        is cross-checked against the paths; this closes that asymmetry by
        pushing one synthetic path through the real synthesizer and reading the
        encoding off the output.

        Requires the render env (x86 + spherical_harmonics_rt) and SKIPS without it,
        which is the honest state on an Apple-Silicon pipeline host. The measured
        values it asserts are recorded at `_SH_CONDON_SHORTLEY_PHASE`.
        """
        sh = pytest.importorskip(
            "spherical_harmonics_rt",
            reason="render env absent — this asserts against the real synthesizer",
        )
        from amcd.simulators.gsound_sir import (
            _AMBISONIC_CONVENTION,
            _SH_CONDON_SHORTLEY_PHASE,
        )

        # ORDER 3, which is what production renders (`configs/base.yaml`
        # `ambisonics_order: 3`, 16 channels). The earlier version of this test ran
        # at order 1, so `calculate_sh_normalization` was validated only at l = 0
        # and l = 1 and the Condon-Shortley phase only at |m| = 1 — while a
        # per-degree normalization error at l = 2 or 3 would have been invisible.
        # Every number below is MEASURED against the real synthesizer, not
        # predicted.
        order, n_channels = 3, 16
        fs, n_bands = 48000.0, 8
        edges = np.asarray(
            Config.load(Path("configs/base.yaml")).simulator.params["frequency_points"],
            dtype=np.float32,
        )
        axes = {
            "+x": (1.0, 0.0, 0.0),
            "+y": (0.0, 1.0, 0.0),
            "+z": (0.0, 0.0, 1.0),
        }
        peaks = {}
        for name, direction in axes.items():
            ir = np.asarray(sh.generate_ambisonic_ir(
                order,
                np.asarray([direction], dtype=np.float32),
                np.ones((1, n_bands), dtype=np.float32),
                np.asarray([1.0], dtype=np.float32),
                np.asarray([344.0], dtype=np.float32),
                edges,
                fs,
                normalize=False,
            ), dtype=np.float64)
            # One path, one arrival: read the encoding at the peak of channel 0.
            t = int(np.argmax(np.abs(ir[0])))
            peaks[name] = ir[:, t]

        w = peaks["+x"][0]
        assert w != 0.0, "W is silent; the synthesizer produced no arrival to read"
        assert len(w_axis := peaks["+x"]) == n_channels, (
            f"order {order} must give (order+1)^2 = {n_channels} channels; "
            f"got {len(w_axis)}"
        )

        # N3D AT EVERY DEGREE, which is what order 3 buys over order 1. On the +z
        # axis only the m = 0 channels are excited, and N3D's normalization makes
        # each one exactly sqrt(2l+1) — 1, sqrt(3), sqrt(5), sqrt(7) at ACN 0, 2, 6,
        # 12. SN3D would give 1.0 at every degree, so this separates the two
        # conventions three times rather than once.
        z_ratios = peaks["+z"] / peaks["+z"][0]
        for degree, channel in enumerate((0, 2, 6, 12)):
            assert abs(abs(z_ratios[channel]) - np.sqrt(2 * degree + 1)) < 1e-4, (
                f"+z ACN {channel} (l={degree}): |ratio| = "
                f"{abs(z_ratios[channel]):.6f}; N3D predicts "
                f"{np.sqrt(2 * degree + 1):.6f} and SN3D predicts 1.0"
            )
        # …and nothing else is excited on that axis: an m != 0 channel with energy
        # would mean the ordering is not ACN.
        m0 = {0, 2, 6, 12}
        assert not [i for i in range(n_channels)
                    if i not in m0 and abs(z_ratios[i]) > 1e-6], z_ratios

        # ACN ordering (W, Y, Z, X): each axis excites exactly one first-order channel.
        for name, channel in (("+x", 3), ("+y", 1), ("+z", 2)):
            ratios = peaks[name] / peaks[name][0]
            live = [i for i in (1, 2, 3) if abs(ratios[i]) > 1e-6]
            assert live == [channel], (
                f"{name} should excite only ACN channel {channel}; got ratios {ratios}"
            )
            # N3D, not SN3D: the magnitude is sqrt(3), not 1.0.
            assert abs(abs(ratios[channel]) - np.sqrt(3.0)) < 1e-4, (
                f"{name}: |ratio| = {abs(ratios[channel]):.6f}; N3D predicts "
                f"{np.sqrt(3.0):.6f} and SN3D predicts 1.0"
            )
        assert _AMBISONIC_CONVENTION == "acn_n3d"

        # Condon-Shortley (-1)^|m|: X and Y negated relative to W, Z not. Negating
        # X and Y together is a 180-degree yaw, which is why it is stamped.
        signs = {n: np.sign(peaks[n] / peaks[n][0]) for n in axes}
        measured_phase = signs["+x"][3] < 0 and signs["+y"][1] < 0 and signs["+z"][2] > 0
        assert bool(measured_phase) == _SH_CONDON_SHORTLEY_PHASE

        # NO absolute-scale assertion here. The late field is
        # `result[c, t] = normalized_sh[c] * carrier[t]`, so W is Y_00 TIMES the
        # noise-carrier sample at the arrival bin — not Y_00 alone, and not a
        # property of the encoding at all. Every assertion above is a RATIO,
        # in which that per-bin scalar cancels exactly; that is what makes them
        # known answers. Pinning |W| would pin the synthesis carrier instead.

    def test_the_ambisonic_stamp_reaches_provenance(self) -> None:
        """The seam half of the row, which needs no render env: whatever the
        measurement above establishes must actually be stamped per leg."""
        from amcd.simulators.base import REQUIRED_PROVENANCE_KEYS

        assert "ambisonic_convention" in REQUIRED_PROVENANCE_KEYS

    def _worker_speed_check(self):
        """The live cross-check, which runs inside the worker.

        The parent used to keep a copy over the RETAINED subset; it could not fail
        once the worker's had passed, so it was removed rather than described as
        defence in depth. These tests follow the check to where it actually runs.
        """
        from amcd.simulators.gsound_sir import _WORKER_SRC

        namespace: dict = {}
        exec(compile(_WORKER_SRC, "<gsound worker>", "exec"), namespace)
        return namespace["_check_declared_speed"]

    def test_declared_speed_of_sound_is_falsified_by_the_paths(self) -> None:
        """Gsound's 344 m/s is compiled into C++ and can only be DECLARED.
        The paths' own `speeds_of_sound` is the free empirical check that keeps the
        declaration honest instead of letting it go stale as a comment."""
        check = self._worker_speed_check()
        assert check(np.full(5, 344.0, dtype="float32"), 344.0) == 5

        with pytest.raises(SystemExit, match="compiled in and can only be declared"):
            check(np.full(5, 343.0, dtype="float32"), 344.0)

    def test_a_render_with_no_paths_refuses_the_SCENE_not_the_batch(self) -> None:
        """A geometry the tracer resolves no path for is the most likely per-scene
        failure of this backend, at the absorptive end where the energy floor also
        bites. It leaves as the distinguished exit code, so the parent excludes the
        scene and keeps going instead of aborting 720 renders.

        A CODE, not a message: the parent reads an exit status across a process
        boundary and cannot classify prose."""
        from amcd.simulators.gsound_sir import _SCENE_REFUSED_EXIT

        check = self._worker_speed_check()
        with pytest.raises(SystemExit) as excinfo:
            check(np.zeros(0, dtype="float32"), 344.0)
        assert excinfo.value.code == _SCENE_REFUSED_EXIT

    def test_the_two_halves_of_the_refusal_code_agree(self) -> None:
        """The worker is read as TEXT and cannot import the parent's constant, so
        the number exists twice. If they drift, a per-scene refusal is read as a
        backend fault and aborts the batch — the exact failure the code prevents."""
        from amcd.simulators.gsound_sir import _SCENE_REFUSED_EXIT

        from amcd.simulators.gsound_sir import _WORKER_SRC

        namespace: dict = {}
        exec(compile(_WORKER_SRC, "<gsound worker>", "exec"), namespace)
        assert namespace["SCENE_REFUSED_EXIT"] == _SCENE_REFUSED_EXIT
        assert _SCENE_REFUSED_EXIT != 0, "a refusal must not look like success"

    def test_the_declared_centres_are_upstreams_compiled_set(self) -> None:
        """Counting bands is not identifying them.

        `model_post_init` only requires the edges to be the geometric means of the
        centres, so a whole filterbank shifted an octave is SELF-CONSISTENT and
        passes every guard — while `frequency_points` is handed to
        `generate_ambisonic_ir`, so it changes the IR rather than merely mislabelling
        a column. Nothing anchored the declaration to the compiled set, and pygsound
        exposes no accessor for it, so the anchor has to be the pinned source.

        Skips when the upstream checkout is absent (the pattern).
        """
        # Derived from the repo root, never hardcoded: a `/Volumes/...`
        # literal skips on the project's declared second host and in every lane
        # worktree — i.e. exactly where band identity could silently differ, so the
        # test would be absent precisely where it is needed.
        source = (
            Path(__file__).resolve().parent.parent
            / "external" / "GSound-SIR" / "ray_generator" / "src" / "pygsound"
            / "src" / "Context.cpp"
        )
        if not source.exists():
            pytest.skip(f"pinned upstream checkout absent at {source}")

        # `const gs::Float f[] = { 63.0f, 125.0f, … };` — the compiled octave set.
        import re

        match = re.search(r"const gs::Float f\[\]\s*=\s*\{([^}]*)\}", source.read_text())
        assert match, f"could not find the band-centre array in {source}"
        compiled = [float(v) for v in re.findall(r"([\d.]+)f", match.group(1))]

        declared = Config.load(Path("configs/base.yaml")).simulator.params
        assert declared["band_centres_hz"] == compiled, (
            "configs/simulators/gsound_sir.yaml declares band centres that are not "
            f"pygsound's compiled set {compiled}; frequency_points is passed to the "
            "synthesizer, so this changes the IR, not just the column labels."
        )
        # …and the edges the synthesizer actually receives are their geometric means.
        expected_edges = [
            (a * b) ** 0.5 for a, b in zip(compiled, compiled[1:])
        ]
        for got, want in zip(declared["frequency_points"], expected_edges):
            assert abs(got - want) < 1e-3 * want

    def test_band_edges_and_centres_must_describe_one_filterbank(self) -> None:
        """`band_centres_hz` is a DECLARATION of a compiled-in fact, not a second
        tunable: each edge must be the geometric mean of its adjacent centres, as
        `gs::FrequencyBands` derives them. This is the failure mode (88.4 vs
        88.7412) one level up — two band definitions that can silently disagree."""
        with pytest.raises(Exception, match="different .*filterbanks"):
            _gsound_sim(band_centres_hz=[62.5, 125.0, 250.0, 500.0,
                                         1000.0, 2000.0, 4000.0, 8000.0])

    def test_the_retention_policy_maps_onto_the_workers_own_arguments(self) -> None:
        """`path_retention` maps onto the (energy_percentage, max_rays) pair the
        WORKER's `_retain` applies after synthesis — not onto `getPathData`, which is
        always called unfiltered so the IR sees every path."""
        assert _gsound_sim(path_retention={"mode": "all", "value": None}
                           )._retention_args() == (100.0, 0)
        assert _gsound_sim(path_retention={"mode": "top_percent", "value": 90.0}
                           )._retention_args() == (90.0, 0)
        assert _gsound_sim(path_retention={"mode": "top_k", "value": 5000}
                           )._retention_args() == (100.0, 5000)

    def test_the_channel_count_must_be_a_whole_ambisonic_order(self) -> None:
        cfg = Config.load(Path("configs/base.yaml"))
        sim = build_simulator("gsound_sir", cfg.simulator.params,
                              n_channels=5, n_samples=1000, sample_rate=48000)
        with pytest.raises(ValueError, match="not a whole ambisonic order"):
            sim._ambisonics_order


class TestHostScopedParamsStayOutOfProvenance:
    """`render_python` is a HOST fact, not a dataset fact.

    Stamped into canonical provenance it would make the same render carry different
    metadata on the Apple-Silicon and the native-x86_64 host this project must both
    support, and leak a user home path into every scene's meta.json.
    """

    def test_render_python_is_redacted_from_canonical_meta(self, tmp_path: Path) -> None:
        """The scaffold config has no `render_python` at all, so a dry_run render
        would pass this whether or not the redaction exists. `_canonical_meta` is
        therefore called against a params block that DOES carry the key, and with a
        value that would be unmistakable in the output if it leaked."""
        from amcd.simulators.render import _canonical_meta

        cfg = tiny_config(scenes={"n_id": 2})
        cfg.simulator.name = "gsound_sir"
        cfg.simulator.params = {
            **Config.load(Path("configs/base.yaml")).simulator.params,
            "render_python": "/Users/SOMEONE/envs/amcd-render-x86/bin/python",
        }
        scene = SceneSpec(
            scene_id="scene_0000", seed=1, geometry_family="shoebox",
            dims=(6.0, 5.0, 3.0), material_absorption=0.2,
            source_pos=(1.0, 1.0, 1.5), receiver_pos=(4.0, 3.0, 1.5),
        )
        leg = IRResult(ir=np.zeros((16, 8), dtype=np.float32), meta={})
        recorded = _canonical_meta(
            cfg, scene, leg, leg,
            artifact_sha256={"low.npy": "0" * 64},
            artifact_fingerprint="deadbeef",
            wall_clock_s={"low": 1.0, "high": 2.0},
        )

        params = recorded["simulator"]["params"]
        assert "render_python" not in params
        assert "SOMEONE" not in json.dumps(recorded)
        # …while the experiment-governing params are all still there.
        assert params["commit_sha"] and params["specular_count"] == 2000
        assert params["speed_of_sound_m_s"] == 344.0

    def test_the_backend_declares_the_redaction_not_the_stage(self) -> None:
        """The list is named by the BACKEND and asked for by the stage, so it
        is still a declaration rather than a filter that could quietly widen — and a
        backend that declares nothing gets the full echo, never a silent redaction."""
        from amcd.simulators.base import simulator_host_scoped_params
        from amcd.simulators.gsound_sir import GsoundSirSimulator

        assert GsoundSirSimulator.host_scoped_params() == ("render_python",)

        cfg = tiny_config(scenes={"n_id": 2})
        cfg.simulator.name = "gsound_sir"
        assert simulator_host_scoped_params(cfg) == ("render_python",)

        # The scaffold declares none, and gets everything echoed.
        cfg.simulator.name = "dry_run"
        assert simulator_host_scoped_params(cfg) == ()


#: The stub backends' fixed sizes. Declared HERE and injected into both stub
#: sources, so an assertion in a test and the value the stub produces cannot drift
#: apart — they used to appear as bare 5 / 8 / 777 / 16 literals in both.
_STUB_N_PATHS, _STUB_N_BANDS, _STUB_NATIVE_SAMPLES = 5, 8, 777

#: Channels the worker returns for the ambisonics_order 3 the requests declare:
#: (order + 1)**2, derived rather than written as a bare 16.
_ORDER3_CHANNELS = (3 + 1) ** 2
#: A SOURCE FRAGMENT, not a container: prepended to each stub below so the sizes
#: above become module-level names inside it. Because it is PREPENDED, each stub's
#: own leading `"""..."""` is a plain string expression rather than a module
#: docstring, though it still reads as one.
_STUB_CONSTANTS_SRC = (
    f"N_PATHS, N_BANDS = {_STUB_N_PATHS}, {_STUB_N_BANDS}\n"
    f"NATIVE_SAMPLES = {_STUB_NATIVE_SAMPLES}\n"
)

_STUB_PYGSOUND = _STUB_CONSTANTS_SRC + '''
"""Stand-in for pygsound: records what the worker asked for, returns fixed paths."""
import json
import os

import numpy as np


class Context:
    pass


class _Mesh:
    def __init__(self, args):
        self.args = args


def createbox(w, l, h, absorp, scatter):
    return _Mesh([w, l, h, absorp, scatter])


class Source:
    def __init__(self, coord):
        self.coord, self.radius, self.power = coord, None, None


class Listener:
    def __init__(self, coord):
        self.coord, self.radius = coord, None


class Scene:
    def setMesh(self, mesh):
        self.mesh = mesh

    def getPathData(self, sources, listeners, ctx, energy_percentage, max_rays, use_gpu):
        # The worker's request, as the stub actually received it — the assertion
        # surface for "did the config value reach upstream's argument?".
        with open(os.environ["AMCD_STUB_CALLS"], "w") as f:
            json.dump({
                "mesh": self.mesh.args,
                "source": {"coord": sources[0].coord, "radius": sources[0].radius,
                           "power": sources[0].power},
                "listener": {"coord": listeners[0].coord, "radius": listeners[0].radius},
                "diffuse_count": ctx.diffuse_count,
                "specular_count": ctx.specular_count,
                "diffuse_depth": ctx.diffuse_depth,
                "specular_depth": ctx.specular_depth,
                "sample_rate": ctx.sample_rate,
                "normalize": ctx.normalize,
                "energy_percentage": energy_percentage,
                "max_rays": max_rays,
                "use_gpu": use_gpu,
            }, f)
        # Per-path energy ascends with index (8, 16, 24, 32, 40), so which paths a
        # retention policy keeps is observable: upstream keeps the HIGHEST-energy
        # ones, i.e. the LAST indices here.
        return {"path_data": [{
            "distances": np.linspace(1.0, 5.0, N_PATHS).astype("float32"),
            "intensities": (np.arange(1, N_PATHS + 1, dtype="float32")[:, None]
                            * np.ones(N_BANDS, dtype="float32")),
            "listener_directions": np.zeros((N_PATHS, 3), dtype="float32"),
            "source_directions": np.ones((N_PATHS, 3), dtype="float32"),
            "path_types": np.arange(N_PATHS, dtype="uint32"),
            "speeds_of_sound": np.full(N_PATHS, 344.0, dtype="float32"),
            "relative_speeds": np.zeros(N_PATHS, dtype="float32"),
            "source_indices": np.zeros(N_PATHS, dtype="uint64"),
            "num_paths": N_PATHS,
            "num_bands": N_BANDS,
            "total_energy": 2.5,
            "kept_energy_percentage": 99.0,
        }]}
'''

_STUB_AURALIZER = _STUB_CONSTANTS_SRC + '''
"""Stand-in for spherical_harmonics_rt. Records how many paths synthesis received."""
import json
import os

import numpy as np


def generate_ambisonic_ir(order, listener_directions, intensities, distances, speeds,
                          frequency_points, sample_rate, precise_early_reflections=False,
                          normalize=True, early_reflection_threshold=0.01):
    assert len(frequency_points) == intensities.shape[1] - 1, "edges must be n_bands-1"
    with open(os.environ["AMCD_STUB_SYNTHESIS"], "w") as f:
        json.dump({"synthesis_paths": int(intensities.shape[0]),
                   "normalize": bool(normalize)}, f)
    # Two switches, both only reachable from the tests, for legs the stub could not
    # otherwise produce: AMCD_STUB_SILENT returns a wholly zero-energy IR;
    # AMCD_STUB_TAIL_ONLY puts all the energy in the LAST sample, so a window
    # shorter than NATIVE_SAMPLES trims the leg to silence.
    n_channels = (order + 1) ** 2
    if os.environ.get("AMCD_STUB_SILENT"):
        return np.zeros((n_channels, NATIVE_SAMPLES), dtype=np.float32)
    ir = np.ones((n_channels, NATIVE_SAMPLES), dtype=np.float32)
    if os.environ.get("AMCD_STUB_TAIL_ONLY"):
        ir[:] = 0.0
        ir[:, -1] = 1.0
    return ir
'''


class TestRenderWorkerContract:
    """The worker needs a regression surface that is NOT a real render.

    `_WORKER_SRC` runs under an interpreter where `amcd` does not exist, and its only
    other exercise is a render that costs a scene from a standing ≤4-scene grant. Left
    untested, every future edit to the cycle's headline deliverable would need render
    permission to verify. These tests run it under the PIPELINE interpreter against
    stub backends: a test double, not scaffold coupling — nothing in `gsound_sir.py`
    knows they exist.
    """

    _SHA = "608ea30f6dc4cda149c18947f9cae48bd379fa27"

    def test_the_worker_source_compiles(self) -> None:
        """A syntax error in the worker would otherwise surface only on the render
        host, mid-render, after the expensive part."""
        from amcd.simulators.gsound_sir import _WORKER_SRC

        compile(_WORKER_SRC, "<gsound worker>", "exec")

    def test_the_worker_imports_nothing_from_amcd(self) -> None:
        """The render env has numpy, pygsound and spherical_harmonics_rt — and no
        amcd, so an import of this package would fail at the render host only."""
        import ast

        from amcd.simulators.gsound_sir import _WORKER_SRC

        imported = set()
        for node in ast.walk(ast.parse(_WORKER_SRC)):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        assert "amcd" not in imported
        assert imported <= {"json", "sys", "sysconfig", "pathlib", "numpy",
                            "pygsound", "spherical_harmonics_rt"}, imported

    def _stub_env(self, tmp_path: Path, sha: str | None):
        """A python whose purelib we control, with stub backends importable.

        A venv is used rather than monkeypatching because the worker reads the
        install receipt from its OWN interpreter's site-packages — which is exactly
        the behaviour under test, and cannot be faked from the parent process.
        """
        import subprocess as sp

        venv = tmp_path / "renderenv"
        sp.run([sys.executable, "-m", "venv", "--system-site-packages",
                "--without-pip", str(venv)], check=True, capture_output=True)
        # POSIX puts it in bin/, Windows in Scripts/ — and Windows is a declared
        # supported host (docs/gsound_sir_setup.md), so hardcoding bin/ would make
        # the whole worker-contract suite unrunnable there.
        python = venv / "bin" / "python"
        if not python.exists():
            python = venv / "Scripts" / "python.exe"
        purelib = Path(sp.run([str(python), "-c",
                               "import sysconfig;print(sysconfig.get_paths()['purelib'])"],
                              capture_output=True, text=True, check=True).stdout.strip())
        purelib.mkdir(parents=True, exist_ok=True)
        if sha is not None:
            (purelib / "amcd_gsound_install.json").write_text(
                json.dumps({"commit_sha": sha})
            )

        stubs = tmp_path / "stubs"
        stubs.mkdir()
        (stubs / "pygsound.py").write_text(_STUB_PYGSOUND)
        (stubs / "spherical_harmonics_rt.py").write_text(_STUB_AURALIZER)
        return python, stubs

    def _run(self, tmp_path: Path, python: Path, stubs: Path, request: dict):
        import subprocess as sp

        from amcd.simulators.gsound_sir import _WORKER_SRC

        worker = tmp_path / "worker.py"
        worker.write_text(_WORKER_SRC)
        req_path = tmp_path / "request.json"
        req_path.write_text(json.dumps(request))
        env = {**os.environ, "PYTHONPATH": str(stubs),
               "AMCD_STUB_CALLS": str(tmp_path / "calls.json"),
               "AMCD_STUB_SYNTHESIS": str(tmp_path / "synthesis.json")}
        return sp.run([str(python), str(worker), str(req_path)],
                      capture_output=True, text=True, env=env)

    def _request(self, out_dir: Path, **overrides) -> dict:
        request = {
            "commit_sha": self._SHA,
            # Matches _STUB_PYGSOUND's speeds_of_sound; the worker cross-checks it
            # over the unfiltered path set before synthesis.
            "speed_of_sound_m_s": 344.0,
            "out_dir": str(out_dir),
            "dims": [6.0, 5.0, 3.0],
            "absorption": 0.2,
            "scattering": 0.5,
            "source_pos": [1.0, 1.0, 1.5],
            "receiver_pos": [4.0, 3.0, 1.5],
            "source_radius": 0.1,
            "listener_radius": 0.1,
            "source_power": 1.0,
            "diffuse_count": 5000,
            "specular_count": 2000,
            "diffuse_depth": 100,
            "specular_depth": 50,
            "sample_rate": 48000,
            "normalize_ir": False,
            "receipt_name": "amcd_gsound_install.json",
            "receipt_sha_key": "commit_sha",
            "precise_early_reflections": False,
            "early_reflection_threshold": 0.01,
            "ambisonics_order": 3,
            "frequency_points": [88.7412, 176.7767, 353.5534, 707.1068,
                                 1414.2136, 2828.4271, 5656.8542],
            "energy_percentage": 100.0,
            "max_rays": 5000,
        }
        request.update(overrides)
        return request

    def test_the_request_response_file_contract_holds(self, tmp_path: Path) -> None:
        python, stubs = self._stub_env(tmp_path, sha=self._SHA)
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        proc = self._run(tmp_path, python, stubs, self._request(out_dir))
        assert proc.returncode == 0, proc.stderr

        result = json.loads((out_dir / "result.json").read_text())
        assert result["installed_commit_sha"] == self._SHA
        assert (result["num_paths"], result["num_bands"]) == (_STUB_N_PATHS, _STUB_N_BANDS)
        assert result["native_ir_shape"] == [_ORDER3_CHANNELS, _STUB_NATIVE_SAMPLES]

        ir = np.load(out_dir / "ir.npy")
        assert ir.shape == (_ORDER3_CHANNELS, _STUB_NATIVE_SAMPLES) and ir.dtype == np.float32
        paths = np.load(out_dir / "paths.npz")
        assert set(paths) == set(PATH_ARRAY_DTYPES)
        assert paths["intensities"].shape == (_STUB_N_PATHS, _STUB_N_BANDS)

    def test_every_config_value_reaches_upstreams_own_argument(self, tmp_path: Path) -> None:
        """The failure this catches is silent: a param declared in config, validated
        by the schema, and then never passed to gsound."""
        python, stubs = self._stub_env(tmp_path, sha=self._SHA)
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        proc = self._run(tmp_path, python, stubs, self._request(out_dir))
        assert proc.returncode == 0, proc.stderr

        called = json.loads((tmp_path / "calls.json").read_text())
        assert called["mesh"] == [6.0, 5.0, 3.0, 0.2, 0.5]
        assert called["source"] == {"coord": [1.0, 1.0, 1.5], "radius": 0.1, "power": 1.0}
        assert called["listener"] == {"coord": [4.0, 3.0, 1.5], "radius": 0.1}
        # the swept axis is the DIFFUSE count; specular is held fixed.
        assert called["diffuse_count"] == 5000
        assert called["specular_count"] == 2000
        assert called["diffuse_depth"] == 100 and called["specular_depth"] == 50
        assert called["sample_rate"] == 48000
        # normalize_ir must reach BOTH upstream switches as false.
        assert called["normalize"] is False
        # Upstream is ALWAYS asked for the full path set: `path_retention` scopes the
        # saved artifact, never the IR (see the dedicated retention tests below).
        assert (called["energy_percentage"], called["max_rays"]) == (100.0, 0)

    def test_a_sha_mismatch_stops_before_any_simulation(self, tmp_path: Path) -> None:
        """The check is FIRST: under emulation the render it refuses can cost
        hours, and its artifact would have unknown provenance."""
        python, stubs = self._stub_env(tmp_path, sha="0" * 40)
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        proc = self._run(tmp_path, python, stubs, self._request(out_dir))

        assert proc.returncode != 0
        assert "!=" in proc.stderr and self._SHA in proc.stderr
        assert not (out_dir / "ir.npy").exists()
        assert not (tmp_path / "calls.json").exists()   # upstream was never called

    def test_a_missing_receipt_is_refused_rather_than_guessed(self, tmp_path: Path) -> None:
        python, stubs = self._stub_env(tmp_path, sha=None)
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        proc = self._run(tmp_path, python, stubs, self._request(out_dir))

        assert proc.returncode != 0
        assert "amcd_gsound_install.json" in proc.stderr
        assert not (out_dir / "ir.npy").exists()

    def test_the_parent_surfaces_a_worker_failure_with_its_stderr(self, tmp_path: Path) -> None:
        """The RuntimeError is the SOLE diagnostic for a failed emulated render, so
        its stderr interpolation is what a debugger reads after hours of compute.

        Driven by a REAL interpreter that exits non-zero: pointing
        `render_python` at a nonexistent path raises FileNotFoundError out of
        `subprocess.run` instead, never reaching this branch.

        The interpreter is an ISOLATED stub env with a deliberately mismatched
        install receipt, NOT `sys.executable`. On a native x86_64 host
        `render_python: null` is the documented-correct setting because the render
        env IS the pipeline env — so a test that relies on the pipeline interpreter
        lacking a receipt would, on that host, run a real 4.25 s render inside the
        unit suite and then fail with DID NOT RAISE.
        """
        python, stubs = self._stub_env(tmp_path, sha="0" * 40)
        sim = _gsound_sim(render_python=str(python))
        scene = SceneSpec(
            scene_id="scene_smoke", seed=1, geometry_family="shoebox",
            dims=(6.0, 5.0, 3.0), material_absorption=0.2,
            source_pos=(1.0, 1.0, 1.5), receiver_pos=(4.0, 3.0, 1.5),
        )
        with pytest.raises(RuntimeError) as excinfo:
            sim.render(scene, 5000)

        message = str(excinfo.value)
        assert "GSound-SIR render worker failed" in message
        # The worker's OWN stderr — the SHA-mismatch refusal — must reach the parent.
        assert self._SHA in message and "0" * 40 in message, (
            "the worker's own stderr must reach the parent's error; got:\n" + message
        )
        assert str(python) in message

    def test_a_nonexistent_render_interpreter_is_named(self, tmp_path: Path) -> None:
        """The other half of the same failure: a bad `render_python` path. Kept
        distinct from the test above so neither can absorb the other's failure."""
        sim = _gsound_sim(render_python=str(tmp_path / "no_such_python"))
        scene = SceneSpec(
            scene_id="scene_smoke", seed=1, geometry_family="shoebox",
            dims=(6.0, 5.0, 3.0), material_absorption=0.2,
            source_pos=(1.0, 1.0, 1.5), receiver_pos=(4.0, 3.0, 1.5),
        )
        with pytest.raises(FileNotFoundError, match="no_such_python"):
            sim.render(scene, 5000)

    def test_render_returns_a_full_irresult_through_the_seam(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """STEP 0's contract, exercised end to end without the render host.

        `build_simulator(...).render(...)` must return an `IRResult` carrying the IR,
        the `PathData`, and a provenance block that survives JSON — the same path the
        1-scene smoke render takes, with pygsound and the auralizer stubbed.
        """
        python, stubs = self._stub_env(tmp_path, sha=self._SHA)
        monkeypatch.setenv("PYTHONPATH", str(stubs))
        monkeypatch.setenv("AMCD_STUB_CALLS", str(tmp_path / "calls.json"))
        monkeypatch.setenv("AMCD_STUB_SYNTHESIS", str(tmp_path / "synthesis.json"))

        sim = _gsound_sim(render_python=str(python))
        scene = SceneSpec(
            scene_id="scene_0000", seed=7, geometry_family="shoebox",
            dims=(6.0, 5.0, 3.0), material_absorption=0.2,
            source_pos=(1.0, 1.0, 1.5), receiver_pos=(4.0, 3.0, 1.5),
        )
        result = sim.render(scene, 5000)

        assert isinstance(result, IRResult)
        assert result.ir.shape == (sim.n_channels, sim.n_samples)
        assert result.ir.dtype == np.float32

        # Provenance: the required set, plus gsound-specific fill.
        assert set(REQUIRED_PROVENANCE_KEYS) <= set(result.meta)
        assert result.meta["ambisonic_convention"] == "acn_n3d"
        assert result.meta["rng_seeded"] is False
        assert result.meta["installed_commit_sha"] == self._SHA
        assert result.meta["commit_sha"] == self._SHA
        assert result.meta["diffuse_count"] == 5000        # the swept axis
        assert result.meta["specular_count"] == 2000       # held fixed across legs
        # the native IR was 777 samples, far short of the window, so it was
        # padded and nothing was discarded.
        assert result.meta["native_ir_samples"] == _STUB_NATIVE_SAMPLES
        assert result.meta["truncated"] is False
        assert result.meta["discarded_tail_db"] is None
        assert result.meta["truncation_qc_flag"] is False

        # the leg's total energy is stamped, so a silent leg is visible.
        assert result.meta["fitted_ir_total_energy"] > 0.0
        assert result.meta["native_ir_total_energy"] > 0.0
        # the declared speed was falsified against the FULL simulated set,
        # not the retained subset — the stub simulates 5 paths and retains 5.
        assert result.meta["speed_check_num_paths"] == _STUB_N_PATHS
        # the two RNGs are reported separately, not flattened into one bool.
        assert result.meta["ray_rng_seeded"] is False
        # the VALUE is asserted against pinned upstream, not against itself
        # — see test_the_stamped_carrier_seed_matches_pinned_upstream below. Here
        # only the stamp's presence and type are the stub's business.
        assert isinstance(result.meta["synthesis_carrier_seed"], int)

        # It must survive the canonical meta.json write — numpy scalars would not.
        json.dumps(result.meta)

        # paths populated and self-describing.
        assert result.paths is not None
        validate_path_descriptor(result.paths, simulator_name="gsound_sir",
                                 scene_id=scene.scene_id)
        assert result.paths.num_bands == _STUB_N_BANDS
        assert result.paths.intensities.shape == (_STUB_N_PATHS, _STUB_N_BANDS)
        assert result.paths.descriptor["ray_budget"] == 5000
        assert result.paths.descriptor["band_centres_hz"][0] == 63.0
        round_tripped = tmp_path / "paths.parquet"
        result.paths.to_parquet(round_tripped)
        assert PathData.from_parquet(round_tripped).descriptor == result.paths.descriptor

    def test_a_silent_leg_is_refused_rather_than_shipped(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """A zero-energy render used to be indistinguishable from a healthy one
        — same all-clear disclosure — and first surfaced as a NaN in a metric."""
        python, stubs = self._stub_env(tmp_path, sha=self._SHA)
        monkeypatch.setenv("PYTHONPATH", str(stubs))
        monkeypatch.setenv("AMCD_STUB_CALLS", str(tmp_path / "calls.json"))
        monkeypatch.setenv("AMCD_STUB_SYNTHESIS", str(tmp_path / "synthesis.json"))
        monkeypatch.setenv("AMCD_STUB_SILENT", "1")

        sim = _gsound_sim(render_python=str(python))
        scene = SceneSpec(
            scene_id="scene_0000", seed=7, geometry_family="shoebox",
            dims=(6.0, 5.0, 3.0), material_absorption=0.2,
            source_pos=(1.0, 1.0, 1.5), receiver_pos=(4.0, 3.0, 1.5),
        )
        with pytest.raises(ValueError, match="zero total energy"):
            sim.render(scene, 5000)

    def test_a_leg_trimmed_to_silence_is_refused_too(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """The trim branch is the case the native-energy guard let through.

        The stub's native IR is far SHORTER than the window, so it can only ever
        exercise the pad branch. Here `n_samples` is cut below the stub's native
        length with all the energy past the cut, so the stored array is all zeros
        while the native array is not — and the leg must still be refused.
        """
        python, stubs = self._stub_env(tmp_path, sha=self._SHA)
        monkeypatch.setenv("PYTHONPATH", str(stubs))
        monkeypatch.setenv("AMCD_STUB_CALLS", str(tmp_path / "calls.json"))
        monkeypatch.setenv("AMCD_STUB_SYNTHESIS", str(tmp_path / "synthesis.json"))
        monkeypatch.setenv("AMCD_STUB_TAIL_ONLY", "1")

        sim = _gsound_sim(render_python=str(python))
        sim.n_samples = _STUB_NATIVE_SAMPLES // 2
        scene = SceneSpec(
            scene_id="scene_0000", seed=7, geometry_family="shoebox",
            dims=(6.0, 5.0, 3.0), material_absorption=0.2,
            source_pos=(1.0, 1.0, 1.5), receiver_pos=(4.0, 3.0, 1.5),
        )
        with pytest.raises(ValueError, match="zero total energy in the"):
            sim.render(scene, 5000)

    def test_the_declared_speed_is_falsified_before_retention_throws_paths_away(
        self, tmp_path: Path
    ) -> None:
        """The check used to run in the PARENT, over the retained subset — 1%
        of the simulated set under `top_k`. The claim is stated over the paths, so it
        must be checked over the paths, which only the worker holds."""
        python, stubs = self._stub_env(tmp_path, sha=self._SHA)
        out_dir = tmp_path / "out"
        out_dir.mkdir()

        # Retain ONE path of the stub's five; the check must still see all five.
        proc = self._run(tmp_path, python, stubs,
                         self._request(out_dir, max_rays=1))
        assert proc.returncode == 0, proc.stderr
        result = json.loads((out_dir / "result.json").read_text())
        assert result["num_paths"] == 1
        assert result["speed_check_num_paths"] == _STUB_N_PATHS

        # …and a declaration the paths contradict stops the render, in the worker,
        # before anything is written.
        out_dir2 = tmp_path / "out2"
        out_dir2.mkdir()
        proc = self._run(tmp_path, python, stubs,
                         self._request(out_dir2, speed_of_sound_m_s=343.0))
        assert proc.returncode != 0
        assert "compiled in and can only be declared" in proc.stderr
        assert not (out_dir2 / "ir.npy").exists()

    def test_a_zero_energy_path_set_selects_the_way_upstream_does(self) -> None:
        """The `total > 0.0` guard CHANGED THE SELECTION, it did not just
        protect a division. On an all-zero-energy set the cumulative sum is all
        zeros and the target is 0.0, so upstream's `accumulated >= target` is
        satisfied at the first path and it keeps ONE; the guard skipped the branch
        and kept every path. Exercised directly against the worker's own `_retain`,
        because the branch is unreachable through `render()` — a zero-energy path
        set implies a zero IR, and guard raises first.
        """
        from amcd.simulators.gsound_sir import _WORKER_SRC

        namespace: dict = {}
        exec(compile(_WORKER_SRC, "<gsound worker>", "exec"), namespace)
        retain = namespace["_retain"]

        n, n_bands = 4, 8
        zero_paths = {
            name: np.zeros((n, n_bands) if name == "intensities" else n)
            for name in namespace["PATH_ARRAYS"]
        }
        kept, total, kept_pct = retain(zero_paths, 50.0, 0)

        assert total == 0.0
        assert kept["distances"].shape[0] == 1, (
            "upstream keeps exactly one path on an all-zero set; keeping all of them "
            "is what the removed `total > 0.0` guard did"
        )
        # …and the share of a zero total is undefined, not 0.0.
        assert kept_pct is None

    def test_retention_trims_the_artifact_but_never_the_synthesis(
        self, tmp_path: Path
    ) -> None:
        """`path_retention` applies ONLY to the saved path file.

        CAUGHT BY A SMOKE RENDER: the first worker passed the retention
        arguments straight to `getPathData` and then synthesized the IR from what
        came back, so `top_k: 5000` built the IR from 43.1% of the path energy on a
        real scene — deleting more than half the response and confounding the very
        ray-budget axis under study. The config has always said retention is for the
        artifact alone; nothing enforced it. This test does.
        """
        python, stubs = self._stub_env(tmp_path, sha=self._SHA)
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        proc = self._run(tmp_path, python, stubs,
                         self._request(out_dir, energy_percentage=100.0, max_rays=2))
        assert proc.returncode == 0, proc.stderr

        # Upstream was asked for the FULL path set — retention is not its business.
        called = json.loads((tmp_path / "calls.json").read_text())
        assert (called["energy_percentage"], called["max_rays"]) == (100.0, 0)

        # Synthesis saw all 5 paths…
        synthesis = json.loads((tmp_path / "synthesis.json").read_text())
        assert synthesis["synthesis_paths"] == _STUB_N_PATHS
        assert synthesis["normalize"] is False

        # …while the saved artifact holds only the 2 highest-energy ones.
        result = json.loads((out_dir / "result.json").read_text())
        assert result["num_paths"] == 2
        assert result["synthesis_num_paths"] == 5
        # Per-path energies are 8,16,24,32,40 → total 120, kept 32+40=72 → 60%.
        assert result["total_energy"] == pytest.approx(120.0)
        assert result["kept_energy_percentage"] == pytest.approx(60.0)

        paths = np.load(out_dir / "paths.npz")
        assert paths["distances"].shape == (2,)
        # The two kept paths are the last two (highest energy), in descending order.
        assert paths["distances"] == pytest.approx([5.0, 4.0])

    def test_top_percent_retention_matches_upstreams_cumulative_rule(
        self, tmp_path: Path
    ) -> None:
        """Upstream keeps paths until the CUMULATIVE share reaches the target
        (Scene.cpp:210-221), so the boundary path is included, not excluded."""
        python, stubs = self._stub_env(tmp_path, sha=self._SHA)
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        # Energies 40,32,24,16,8 descending; 60% of 120 = 72 → first two paths.
        proc = self._run(tmp_path, python, stubs,
                         self._request(out_dir, energy_percentage=60.0, max_rays=0))
        assert proc.returncode == 0, proc.stderr

        result = json.loads((out_dir / "result.json").read_text())
        assert result["num_paths"] == 2
        assert result["kept_energy_percentage"] == pytest.approx(60.0)


def test_the_stamped_carrier_seed_matches_pinned_upstream() -> None:
    """The stamp must be falsifiable by something other than itself.

    `_SYNTHESIS_CARRIER_SEED` records the seed upstream uses for the noise carrier
    every path's energy is multiplied by. It was only ever asserted against a stub
    that reads the same constant — a literal compared to itself, which would keep
    passing if upstream changed the seed and every rendered IR changed with it.

    Upstream is pinned by commit sha in `configs/simulators/gsound_sir.yaml` and
    vendored under `external/`, so the declaration IS checkable: read the pinned
    source and compare. Same discipline as the `speed_of_sound_m_s` cross-check —
    a value that lives in C++ can only be DECLARED here, so the declaration has to
    be falsified against the thing it describes.

    Path derived from the repo root, never hardcoded: a `/Volumes/...`
    literal would skip on the project's declared second host and in every lane
    worktree, i.e. exactly where band identity could silently differ.
    """
    from amcd.simulators.gsound_sir import _SYNTHESIS_CARRIER_SEED

    binding = (
        Path(__file__).resolve().parent.parent
        / "external" / "GSound-SIR" / "auralizer" / "src" / "cpp" / "binding.cpp"
    )
    if not binding.exists():
        pytest.skip(f"pinned upstream checkout absent at {binding}")

    m = re.search(r"NoiseGenerator\s*\(\s*unsigned\s+int\s+seed\s*=\s*(\d+)\s*\)",
                  binding.read_text())
    assert m, (
        f"could not find NoiseGenerator's default seed in {binding}. If upstream "
        "restructured it, re-derive the seed and update this test — do not delete "
        "it, or the stamp goes back to asserting itself."
    )
    assert int(m.group(1)) == _SYNTHESIS_CARRIER_SEED, (
        f"pinned upstream seeds its synthesis carrier with {m.group(1)}, but "
        f"`_SYNTHESIS_CARRIER_SEED` declares {_SYNTHESIS_CARRIER_SEED}. Every "
        "rendered IR is the path energies times that carrier, so the stamp is "
        "wrong about the signal it describes."
    )


class TestTheAbsorptionConventionIsDeclaredAndApplied:
    """The rendered room must be the declared room.

    This backend's per-bounce ENERGY factor is `sqrt(1-alpha)` where the declared
    physics wants `(1-alpha)`, re-derived from pinned upstream and confirmed by
    render. Left uncorrected it realizes `alpha_eff = 1 - sqrt(1-alpha)`, which is
    1.45-1.98x on T60 across `base.yaml`'s declared support — so every closed form
    in `scenes/generator.py` described a room that was never rendered.

    The convention is declared on the BACKEND, never in the generator:
    the generator defines the dataset's acoustics for every simulator, so
    re-deriving it from one raytracer's domain confusion would make a second
    raytracer render a different room from the same spec.
    """

    def test_pre_compensation_makes_the_realized_alpha_the_declared_one(self) -> None:
        from amcd.simulators.gsound_sir import _createbox_absorption

        for nominal in (0.02, 0.05, 0.10, 0.30, 0.50, 0.80, 0.98):
            passed = _createbox_absorption(nominal, "pre_compensate")
            realized = 1.0 - np.sqrt(1.0 - passed)   # what upstream does to it
            assert realized == pytest.approx(nominal, abs=1e-12), (
                f"pre-compensated alpha {passed} realizes {realized}, not the "
                f"declared {nominal}. The rendered room is not the scene's room."
            )

    def test_as_is_still_reproduces_the_uncorrected_room(self) -> None:
        """`as_is` is not a fallback — it is what every prior measurement used.

        render (5x4x3 m, alpha 0.30, T30 0.5441 s against 0.343 s
        nominal) was taken under it, and those numbers must stay reproducible.
        """
        from amcd.simulators.gsound_sir import _createbox_absorption

        for nominal, expected_ratio in ((0.05, 1.975), (0.30, 1.837), (0.80, 1.447)):
            realized = 1.0 - np.sqrt(1.0 - _createbox_absorption(nominal, "as_is"))
            assert nominal / realized == pytest.approx(expected_ratio, rel=1e-3), (
                f"the uncorrected room at alpha {nominal} no longer realizes "
                f"{expected_ratio}x on T60 — evidence stops reproducing."
            )

    def test_an_undeclared_convention_raises(self) -> None:
        """No default: it decides which room is rendered (CLAUDE.md)."""
        from amcd.simulators.gsound_sir import _createbox_absorption

        with pytest.raises(ValueError, match="neither 'pre_compensate' nor 'as_is'"):
            _createbox_absorption(0.3, "")

    def test_the_declared_realized_alpha_is_not_the_createbox_alpha(self) -> None:
        """The two quantities are DIFFERENT numbers, and the scene report
        needs the second one.

        `_createbox_absorption` answers "what do I hand the renderer"; the backend's
        `realized_absorption` classmethod answers "what does the room then have".
        Under `pre_compensate` a nominal 0.30 is passed as 0.5100 and realized as
        0.30, so a closed form evaluated on the first describes a room 1.8x more
        absorptive than the one rendered. `support_t60_s` in the render provenance
        was computed that way until this separation existed.
        """
        from amcd.simulators.gsound_sir import (
            GsoundSirSimulator, _createbox_absorption,
        )

        for convention, expect in (("pre_compensate", lambda a: a),
                                   ("as_is", lambda a: 1.0 - np.sqrt(1.0 - a))):
            params = {"absorption_convention": convention}
            for nominal in (0.05, 0.30, 0.80, 0.98):
                realized = GsoundSirSimulator.realized_absorption(params, nominal)
                assert realized == pytest.approx(expect(nominal), abs=1e-12)
                if convention == "pre_compensate":
                    assert _createbox_absorption(nominal, convention) != pytest.approx(
                        realized, rel=1e-6
                    ), (
                        f"at alpha {nominal} the createbox value and the realized "
                        f"value coincide, so this test can no longer catch the two "
                        f"being confused"
                    )

    def test_every_backend_declares_what_it_realizes(self) -> None:
        """The scaffold answers too, and answers the identity.

        Left optional, a backend would omit it and the scene report would silently
        describe the declared room as though it were the rendered one — the
        silent-contract shape `realized_support_s` and `min_source_receiver_distance_m`
        are both guarded against.
        """
        from amcd.simulators.base import simulator_realized_absorption

        cfg = tiny_config()
        for nominal in (0.05, 0.30, 0.80):
            assert simulator_realized_absorption(cfg, nominal) == pytest.approx(nominal)

    def test_a_backend_that_does_not_declare_it_is_refused(self) -> None:
        from pydantic import BaseModel

        from amcd.simulators.base import simulator_realized_absorption

        class _Silent:
            class Params(BaseModel):
                # Permissive on purpose: the subject is the MISSING classmethod,
                # so param validation must not be what raises.
                model_config = {"extra": "allow"}

        cfg = tiny_config()
        with mock.patch.object(
            simulator_registry, "get", return_value=_Silent
        ), pytest.raises(TypeError, match="realized_absorption"):
            simulator_realized_absorption(cfg, 0.3)

    def test_the_backend_config_declares_it(self) -> None:
        params = yaml.safe_load(
            (Path(__file__).resolve().parent.parent
             / "configs" / "simulators" / "gsound_sir.yaml").read_text()
        )
        assert params.get("absorption_convention") in ("pre_compensate", "as_is"), (
            "configs/simulators/gsound_sir.yaml must declare "
            "`absorption_convention` — it is experiment-governing and this is the "
            "backend's own config, which is where it belongs."
        )

    def test_the_scene_generator_stays_in_nominal_alpha(self) -> None:
        """Load-bearing half: the generator must NOT learn about alpha_eff.

        If it does, the dataset's declared acoustics become one raytracer's, and a
        second backend renders a different room from the same scene spec with
        nothing saying so.
        """
        src = (Path(__file__).resolve().parent.parent
               / "src" / "amcd" / "scenes" / "generator.py").read_text()
        for marker in ("1 - np.sqrt(1", "1.0 - np.sqrt(1.0", "alpha_eff =",
                       "absorption_convention"):
            assert marker not in src, (
                f"scenes/generator.py contains {marker!r}: the backend's absorption "
                "convention has leaked into the backend-agnostic scene population "
                ". It belongs in configs/simulators/<name>.yaml."
            )


class TestRenderArtifactsHaveAVerifier:
    """Digests that nothing reads, and a stale file nothing reports.

    `rng_seeded: false` puts reproducibility on the CACHED ARTIFACTS rather than on
    re-render bit-identity, so `artifact_sha256` is what stands between a truncated
    IR and a reported number computed from it. It had exactly one writer and no
    reader outside a test that recomputed the digest itself — which checks the
    digest function, not the files.
    """

    @staticmethod
    def _scene_dir(tmp_path: Path) -> Path:
        from amcd.simulators.render import _sha256

        scene = tmp_path / "renders" / "scene_0000"
        scene.mkdir(parents=True)
        (scene / "low.npy").write_bytes(b"low-artifact-bytes")
        (scene / "high.npy").write_bytes(b"high-artifact-bytes")
        digests = {n: _sha256(scene / n) for n in ("low.npy", "high.npy")}
        (scene / "meta.json").write_text(json.dumps({"artifact_sha256": digests}))
        return scene

    def test_intact_artifacts_report_nothing(self, tmp_path: Path) -> None:
        from amcd.simulators.render import verify_render_artifacts

        self._scene_dir(tmp_path)
        assert verify_render_artifacts(tmp_path / "renders") == []

    def test_a_modified_artifact_is_reported(self, tmp_path: Path) -> None:
        from amcd.simulators.render import verify_render_artifacts

        scene = self._scene_dir(tmp_path)
        (scene / "low.npy").write_bytes(b"low-artifact-byteS")   # one bit
        problems = verify_render_artifacts(tmp_path / "renders")
        assert [u for u, _ in problems] == ["scene_0000/low.npy"], problems

    def test_a_missing_artifact_is_reported(self, tmp_path: Path) -> None:
        from amcd.simulators.render import verify_render_artifacts

        scene = self._scene_dir(tmp_path)
        (scene / "high.npy").unlink()
        assert [u for u, _ in verify_render_artifacts(tmp_path / "renders")] == [
            "scene_0000/high.npy"
        ]

    def test_an_empty_integrity_record_is_reported_not_treated_as_passing(
        self, tmp_path: Path
    ) -> None:
        """An absent record used to render as `{}`, and an empty record iterates
        zero times — so it passes every check by having nothing to check."""
        from amcd.simulators.render import verify_render_artifacts

        scene = self._scene_dir(tmp_path)
        (scene / "meta.json").write_text(json.dumps({"artifact_sha256": {}}))
        assert verify_render_artifacts(tmp_path / "renders")


class TestTruncationDisclosureReachabilityPerConfig:
    """`truncation_qc_flag` cannot fire under `base.yaml`, and that is a
    property of the CONFIG PAIR rather than dead code.

    The trim branch fires only when the native record exceeds the window. This
    backend's native record is bounded by the compiled `max_ir_length_s` plus the
    auralizer's tail padding, so under base's `ir_duration: 4.25` the window always
    wins and the flag is constant. Under `research_i.yaml`, whose RI-pinned
    `ir_duration` is 3.0 s — the same length as the cap — it is live.

    Driven through the CONFIGS rather than a hand-built array, so the reachability
    claim in the source is checked rather than asserted: if either `ir_duration` or
    `max_ir_length_s` moves, this says which way.
    """

    @staticmethod
    def _disclosure(cfg) -> dict:
        sim = build_simulator(
            cfg.simulator.name, cfg.simulator.params, n_channels=cfg.n_channels,
            n_samples=cfg.n_samples, sample_rate=cfg.sample_rate,
        )
        cap_samples = int(float(cfg.simulator.params["max_ir_length_s"]) * cfg.sample_rate)
        native = np.zeros((cfg.n_channels, cap_samples + 2048), dtype=np.float32)
        native[0, :] = 0.01          # energy everywhere, so a trim discards some
        return sim._fit_to_window(native)[1]

    def test_base_cannot_reach_the_trim_branch(self) -> None:
        cfg = Config.load(Path("configs/base.yaml"))
        assert cfg.n_samples / cfg.sample_rate > float(
            cfg.simulator.params["max_ir_length_s"]
        ) + 2048 / cfg.sample_rate, (
            "base's window no longer exceeds the backend's realized cap, so the "
            "trim branch has become reachable — the disclosure below is no longer "
            "constant and its docstring is stale"
        )
        d = self._disclosure(cfg)
        assert d["truncated"] is False and d["truncation_qc_flag"] is False

    def test_research_i_does_reach_it(self) -> None:
        """The reason the flag is kept: the reproduction config is where a record
        shorter than the cap makes the discarded tail a real quantity."""
        cfg = Config.load(Path("configs/base.yaml"), Path("configs/research_i.yaml"))
        d = self._disclosure(cfg)
        assert d["truncated"] is True and d["truncation_qc_flag"] is True
        assert d["discarded_tail_db"] is not None


class TestEveryBackendDeclaresWhetherItModelsEarlyReflections:
    """EDT fits the FIRST 10 dB, and in a real room that span IS the
    early-reflection cluster, which is why EDT moves systematically with distance.

    A backend whose diffuse tail begins at the direct arrival has no structure
    there, so its EDT is nearly inert on the placement axis while C50 stays live
    (gave the scaffold a real 1/d direct term against a room-constant tail).
    `test_placement_shift`'s EDT column is then a plumbing result, not an acoustic
    one — an acceptable simplification, but not an invisible one.
    """

    def test_the_scaffold_says_it_does_not_and_gsound_says_it_does(self) -> None:
        from amcd.simulators.base import simulator_models_early_reflections

        assert simulator_models_early_reflections(
            Config.load(*CANONICAL_DRY_RUN[:2])
        ) is False
        assert simulator_models_early_reflections(Config.load(CANONICAL_DRY_RUN[0])) is True

    def test_a_backend_that_does_not_declare_it_is_refused(self) -> None:
        """Left optional, a new backend would omit it and the reported EDT column
        would silently inherit "this axis is live" — the silent-contract shape the
        other three pre-render declarations are guarded against."""
        from pydantic import BaseModel

        from amcd.simulators.base import simulator_models_early_reflections

        class _Silent:
            class Params(BaseModel):
                model_config = {"extra": "allow"}

        with mock.patch.object(
            simulator_registry, "get", return_value=_Silent
        ), pytest.raises(TypeError, match="models_early_reflections"):
            simulator_models_early_reflections(tiny_config())

    def test_the_claim_is_TRUE_of_the_scaffold_it_is_made_about(self) -> None:
        """The declaration has to match the backend's behaviour, or it is just a
        constant. Measured through the reported metric path over a 16x distance
        range: C50 falls monotonically while EDT does not."""
        from amcd.evaluation.room_acoustic import channel_band_avg_metrics
        from amcd.simulators.base import build_simulator

        cfg = Config.load(*CANONICAL_DRY_RUN[:2])
        sim = build_simulator(
            cfg.simulator.name, cfg.simulator.params, n_channels=1,
            n_samples=int(cfg.sample_rate * cfg.ir_duration),
            sample_rate=cfg.sample_rate,
        )
        edt, c50 = [], []
        for d in (1.0, 2.0, 4.0, 8.0):
            scene = SceneSpec(
                scene_id=f"ac43-{d}", seed=7, geometry_family="shoebox",
                dims=(10.0, 8.0, 3.5), material_absorption=0.2,
                source_pos=(1.0, 1.0, 1.5), receiver_pos=(1.0 + d, 1.0, 1.5),
                sim_params={}, split_regime="id", regime_axes={},
            )
            vals, _ = channel_band_avg_metrics(
                sim.render(scene, cfg.high_ray_budget).ir[0],
                sample_rate=cfg.sample_rate,
                iso_eval_freqs=[float(f) for f in cfg.iso_eval_freqs],
                onset_rel_db=cfg.metric_onset_rel_db,
                band_resolvability_margin=cfg.metric_band_resolvability_margin,
                decay_range_fit=_decay_fit(), min_decay_range_db={"T30": 0.0, "EDT": 0.0},
                octave_filter_order=cfg.metric_octave_filter.order,
            )
            edt.append(vals["EDT"])
            c50.append(vals["C50"])

        assert c50 == sorted(c50, reverse=True), (
            f"C50 is no longer monotone in distance ({c50}) — placement "
            f"liveness has regressed, and the contrast rests on it"
        )
        spread = (max(edt) - min(edt)) / float(np.mean(edt))
        assert spread < 0.05, (
            f"EDT now spreads {spread:.1%} over a 16x distance range ({edt}), i.e. it "
            f"HAS become live on the placement axis. If the scaffold gained an "
            f"early-reflection cluster, `models_early_reflections` must stop "
            f"returning False."
        )


def _decay_fit():
    """The shipped decay-range fit, read from the config the pipeline runs under.

    A function rather than a module constant so importing this test module does
    not read a config file at collection time.
    """
    from pathlib import Path as _Path

    from amcd.config import Config as _Config

    return _Config.load(_Path("configs/base.yaml")).metric_decay_range_fit
