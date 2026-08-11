"""The simulator config seam: `{name, params}` block, build_simulator, provenance,
and fingerprinted stage caching (gsound_sir gate, Step 1A).

Covers ledger rows RD-13 (simulator params reach the role grammar), RD-16 /
RD-30 / RD-35 (canonical provenance + cache fingerprint + field-level diff),
RD-31 (required provenance keys), RD-40 (ray budgets stay top-level).

None of these tests need GSound-SIR: they exercise the config contract and the
dry_run path, which is what proves the real backend will be a drop-in.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
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
        """F-11: gsound_sir's params must not bleed onto dry_run, whose schema
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
    """RD-13: simulator params must reach the role grammar.

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
    """RD-31: every simulator must declare a fixed provenance key set."""

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
        """RD-16: provenance is no longer verbosity-gated."""
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
    """RD-40: the swept research axis must survive a simulator name change."""

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
    """RD-16/RD-30/RD-35: a cached stage must belong to the CURRENT config."""

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
        """RD-35: a bare sha cannot tell the operator whether an expensive
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
        """RD-30: renders are per-scene, so changing the SCENES makes them stale
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

    def test_undeclared_stage_still_caches_on_bare_sentinel(self, tmp_path: Path) -> None:
        """The remaining `None` entries are a declared gap, not a silent one: they
        must keep working exactly as before until they are wired (RD-41)."""
        # Every stage is listed, so an unwired one is declared rather than absent.
        assert set(STAGE_FINGERPRINT) == set(STAGES)
        # `eval`/`stats` were promoted out of the gap by RD-54 and `report` by F-63,
        # so `diagnostics` is now the standing example of the unwired path.
        assert STAGE_FINGERPRINT["diagnostics"] is None
        cfg = tiny_config(scenes={"n_id": 4})
        pipe = self._pipeline(cfg, tmp_path)
        pipe._mark_done("diagnostics")
        assert pipe._is_done("diagnostics")


class TestDryRunTailIsUnbiased:
    """AC-18: the scaffold's diffuse tail must not BE the noise.

    It previously read `diffuse = decay * noise * noise_scale` with
    `noise_scale = 1/sqrt(N)`, so E[tail energy] scaled as 1/N. Measured, the low
    leg's late-window energy was 39.7x the high leg's — 16.0 dB, exactly
    200000/5000 — which makes low→high a deterministic level shift (trivially
    learnable) whose converged limit is an IR with no reverberant tail at all.
    That is the mechanism behind the dry_run D0b "CARRIER BOTTLENECK" verdict
    being a plumbing artifact rather than a result (RD-07).

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
            f"a deterministic -16 dB shift the model can learn trivially) — AC-18."
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
    """AC-28: the scaffold's "direct sound" was not a direct sound.

    `direct = direct_gain * exp(-t/0.02)` is a one-pole envelope with a 7.96 Hz
    corner, so only 6.06e-7 of its energy reached the 500 Hz octave band. MEASURED
    in a 10x8x3.5 m room at alpha 0.2 (r_c = 1.19 m): C50 read 1.966 / 1.957 /
    1.953 / 1.951 / 1.950 dB at d = 0.5, 1, 2, 4 and 8 m — flat to 0.02 dB across a
    16x distance range — while the closed-form DRR the scene report publishes swung
    +7.55 to -16.53 dB. `test_placement_shift` therefore carried NO acoustic
    difference from the id baseline in any reported ISO-3382 metric.

    SCOPE OF THESE TESTS (RD-79). The DRR agreement below is a SCAFFOLD
    SELF-CONSISTENCY check: the diffuse tail is scaled by
    sqrt(16*pi / (R * sum(decay^2))) precisely so the rendered DRR equals the
    closed form, so it verifies that the two share one formula (RD-75) — it does
    NOT validate the ISO path against independent physics. The independent check is
    the Step-6 probe against a real gsound render (RD-17). What IS non-circular
    here is the C50 SHAPE: nothing in the construction forces C50, an ISO-3382
    quantity computed through octave filtering and Schroeder integration, to track
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
        )
        return values["C50"]

    def test_c50_moves_with_distance(self) -> None:
        """The kill assertion. Pre-fix the spread over this range was 0.016 dB."""
        c50 = [self._c50(d) for d in (0.5, 1.0, 2.0, 4.0, 8.0)]
        assert c50 == sorted(c50, reverse=True), f"C50 not monotone in distance: {c50}"
        assert c50[0] - c50[-1] > 6.0, (
            f"C50 spans only {c50[0] - c50[-1]:.3f} dB over a 16x distance range "
            f"({c50}) — the placement axis is acoustically inert again (AC-28)"
        )

    def test_the_direct_arrival_is_the_loudest_sample(self) -> None:
        """`_find_onset` documents this as an assumption (AC-07). Pre-fix the global
        peak sat 300-550 samples INTO the diffuse tail, violating it — inert only
        because the whole response starts at d/c."""
        from amcd.evaluation.room_acoustic import _find_onset

        for d in (0.5, 2.0, 8.0):
            ir = self._render(d).ir[0]
            assert int(np.argmax(np.abs(ir))) == _find_onset(ir, -20.0)

    def test_realized_drr_matches_the_published_closed_form(self) -> None:
        """SCAFFOLD SELF-CONSISTENCY, not metric validation — see the class
        docstring. The residual grows with distance because the direct sample and
        the tail's first sample are superposed, which matters more as the direct
        term shrinks."""
        from amcd.acoustics import diffuse_field_drr_db
        from amcd.evaluation.room_acoustic import _find_onset

        for d in (0.5, 1.0, 2.0, 4.0, 8.0):
            ir = self._render(d).ir[0]
            onset = _find_onset(ir, -20.0)
            direct = float(ir[onset]) ** 2
            reverberant = float(np.sum(ir[onset + 1:].astype(np.float64) ** 2))
            realized = 10.0 * np.log10(direct / reverberant)
            expected = diffuse_field_drr_db(self._SURFACE, self._ALPHA, d)
            assert realized == pytest.approx(expected, abs=1.0), (
                f"rendered DRR {realized:.2f} dB vs published {expected:.2f} dB at "
                f"d={d} m — the scaffold and scenes/placement_report.json have "
                f"stopped sharing one formula (RD-75)"
            )

    def test_the_realized_snr_is_stamped(self) -> None:
        """AC-35: RD-07's caveat needs a magnitude, and the Step-6 probe needs a
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
    """RD-24: a retained-path file must be interpretable WITHOUT its config.

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
        with pytest.raises(ValueError, match="describe bands it does not contain"):
            _fake_paths(n_bands=8).__class__(
                **{**_fake_paths().__dict__, "num_bands": 4}
            )

    def test_the_file_identifies_its_render_without_the_filename(self, tmp_path: Path) -> None:
        """RD-117/RD-23: `paths_{low,high}.parquet` encodes two legs and one
        realization. The artifact layout must not foreclose a realization index, so
        the identity lives in the file's own metadata, not in its name."""
        target = tmp_path / "renamed_by_someone.parquet"
        _fake_paths().to_parquet(target)
        back = PathData.from_parquet(target)
        assert back.descriptor["ray_budget"] == 5000
        assert back.descriptor["leg"] == "low"
        assert back.descriptor["realization_index"] == 0


class TestIRResultCarriesPaths:
    """RD-08: `IRResult.paths` is the producer half of the path-conditioned seam.

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
        finally:
            simulator_registry._entries.pop("pathy_dry_run", None)


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
    """RD-21: trimming gsound's natural IR to `ir_duration` discards tail energy.

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


class TestGsoundProvenanceFill:
    """RD-67: what the gsound leg must be able to state about itself."""

    def test_the_ambisonic_convention_is_n3d_not_sn3d(self) -> None:
        """AC-15, verified in the auralizer binding (binding.cpp:18 "normalization
        constant K(l, m) for N3D", :43 "N3D/ACN ordering"). Getting this wrong is a
        per-degree sqrt(2l+1) error — invisible while every live scalar metric reads
        channel 0, load-bearing once evaluation/spatial.py is filled in."""
        from amcd.simulators.gsound_sir import _AMBISONIC_CONVENTION

        assert _AMBISONIC_CONVENTION == "acn_n3d"

    def test_declared_speed_of_sound_is_falsified_by_the_paths(self) -> None:
        """RD-19: gsound's 344 m/s is compiled into C++ and can only be DECLARED.
        The paths' own `speeds_of_sound` is the free empirical check that keeps the
        declaration honest instead of letting it go stale as a comment."""
        sim = _gsound_sim()
        sim._check_declared_speed(np.full(5, 344.0, dtype="float32"), "scene_00000")

        with pytest.raises(ValueError, match="compiled in and can only be declared"):
            sim._check_declared_speed(np.full(5, 343.0, dtype="float32"), "scene_00000")

    def test_a_render_with_no_paths_is_an_error_not_an_empty_ir(self) -> None:
        sim = _gsound_sim()
        with pytest.raises(ValueError, match="returned no paths"):
            sim._check_declared_speed(np.zeros(0, dtype="float32"), "scene_00000")

    def test_band_edges_and_centres_must_describe_one_filterbank(self) -> None:
        """`band_centres_hz` is a DECLARATION of a compiled-in fact, not a second
        tunable: each edge must be the geometric mean of its adjacent centres, as
        `gs::FrequencyBands` derives them. This is the AC-12 failure mode (88.4 vs
        88.7412) one level up — two band definitions that can silently disagree."""
        with pytest.raises(Exception, match="different .*filterbanks"):
            _gsound_sim(band_centres_hz=[62.5, 125.0, 250.0, 500.0,
                                         1000.0, 2000.0, 4000.0, 8000.0])

    def test_the_retention_policy_maps_onto_upstream_arguments(self) -> None:
        """Retention is native upstream, so there is no custom trimming to get
        wrong: `path_retention` maps directly onto (energy_percentage, max_rays)."""
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
    """RD-114: `render_python` is a HOST fact, not a dataset fact.

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
        recorded = _canonical_meta(cfg, scene, leg, leg)

        params = recorded["simulator"]["params"]
        assert "render_python" not in params
        assert "SOMEONE" not in json.dumps(recorded)
        # …while the experiment-governing params are all still there.
        assert params["commit_sha"] and params["specular_count"] == 2000
        assert params["speed_of_sound_m_s"] == 344.0

    def test_every_other_simulator_param_still_reaches_provenance(self) -> None:
        """The redaction is a named list, not a filter that could quietly widen."""
        from amcd.simulators.render import _HOST_SCOPED_PARAMS

        assert _HOST_SCOPED_PARAMS == ("render_python",)


_STUB_PYGSOUND = '''
"""Stand-in for pygsound: records what the worker asked for, returns fixed paths."""
import json
import os

import numpy as np

N_PATHS, N_BANDS = 5, 8


class Context:
    pass


class ChannelLayoutType:
    stereo = "stereo"


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

_STUB_AURALIZER = '''
"""Stand-in for spherical_harmonics_rt. Records how many paths synthesis received."""
import json
import os

import numpy as np

NATIVE_SAMPLES = 777


def generate_ambisonic_ir(order, listener_directions, intensities, distances, speeds,
                          frequency_points, sample_rate, precise_early_reflections=False,
                          normalize=True, early_reflection_threshold=0.01):
    assert len(frequency_points) == intensities.shape[1] - 1, "edges must be n_bands-1"
    with open(os.environ["AMCD_STUB_SYNTHESIS"], "w") as f:
        json.dump({"synthesis_paths": int(intensities.shape[0]),
                   "normalize": bool(normalize)}, f)
    return np.ones(((order + 1) ** 2, NATIVE_SAMPLES), dtype=np.float32)
'''


class TestRenderWorkerContract:
    """RD-116: the worker needs a regression surface that is NOT a real render.

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
        import sysconfig

        venv = tmp_path / "renderenv"
        sp.run([sys.executable, "-m", "venv", "--system-site-packages",
                "--without-pip", str(venv)], check=True, capture_output=True)
        python = venv / "bin" / "python"
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
        assert sysconfig  # keep the import meaningful for readers of the fixture
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
        assert (result["num_paths"], result["num_bands"]) == (5, 8)
        assert result["native_ir_shape"] == [16, 777]

        ir = np.load(out_dir / "ir.npy")
        assert ir.shape == (16, 777) and ir.dtype == np.float32
        paths = np.load(out_dir / "paths.npz")
        assert set(paths) == set(PATH_ARRAY_DTYPES)
        assert paths["intensities"].shape == (5, 8)

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
        # RD-12: the swept axis is the DIFFUSE count; specular is held fixed.
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
        """RD-67. The check is FIRST: under emulation the render it refuses can cost
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
        """A silent subprocess failure would look like a rendering bug."""
        sim = _gsound_sim(render_python=str(tmp_path / "no_such_python"))
        scene = SceneSpec(
            scene_id="scene_smoke", seed=1, geometry_family="shoebox",
            dims=(6.0, 5.0, 3.0), material_absorption=0.2,
            source_pos=(1.0, 1.0, 1.5), receiver_pos=(4.0, 3.0, 1.5),
        )
        with pytest.raises(Exception, match="no_such_python|worker failed"):
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

        # Provenance: the required set, plus RD-67's gsound-specific fill.
        assert set(REQUIRED_PROVENANCE_KEYS) <= set(result.meta)
        assert result.meta["ambisonic_convention"] == "acn_n3d"
        assert result.meta["rng_seeded"] is False
        assert result.meta["installed_commit_sha"] == self._SHA
        assert result.meta["commit_sha"] == self._SHA
        assert result.meta["diffuse_count"] == 5000        # the swept axis (RD-12)
        assert result.meta["specular_count"] == 2000       # held fixed across legs
        # RD-21: the native IR was 777 samples, far short of the window, so it was
        # padded and nothing was discarded.
        assert result.meta["native_ir_samples"] == 777
        assert result.meta["truncated"] is False
        assert result.meta["discarded_tail_db"] is None
        assert result.meta["truncation_qc_flag"] is False

        # It must survive the canonical meta.json write — numpy scalars would not.
        json.dumps(result.meta)

        # RD-08/RD-24: paths populated and self-describing.
        assert result.paths is not None
        validate_path_descriptor(result.paths, simulator_name="gsound_sir",
                                 scene_id=scene.scene_id)
        assert result.paths.num_bands == 8
        assert result.paths.intensities.shape == (5, 8)
        assert result.paths.descriptor["ray_budget"] == 5000
        assert result.paths.descriptor["band_centres_hz"][0] == 63.0
        round_tripped = tmp_path / "paths.parquet"
        result.paths.to_parquet(round_tripped)
        assert PathData.from_parquet(round_tripped).descriptor == result.paths.descriptor

    def test_retention_trims_the_artifact_but_never_the_synthesis(
        self, tmp_path: Path
    ) -> None:
        """`path_retention` applies ONLY to the saved path file.

        CAUGHT BY THE CYCLE-4 SMOKE RENDER: the first worker passed the retention
        arguments straight to `getPathData` and then synthesized the IR from what
        came back, so `top_k: 5000` built the IR from 43.2% of the path energy on a
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
        assert synthesis["synthesis_paths"] == 5
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
