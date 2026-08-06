"""The simulator config seam: `{name, params}` block, build_simulator, provenance,
and fingerprinted stage caching (gsound_sir gate, Step 1A).

Covers ledger rows RD-13 (simulator params reach the role grammar), RD-16 /
RD-30 / RD-35 (canonical provenance + cache fingerprint + field-level diff),
RD-31 (required provenance keys), RD-40 (ray budgets stay top-level).

None of these tests need GSound-SIR: they exercise the config contract and the
dry_run path, which is what proves the real backend will be a drop-in.
"""
from pathlib import Path

import pytest
import yaml

from amcd.config import Config, _BASE_YAML, _merge_layer
from amcd.pipeline import STAGE_FINGERPRINT, STAGES, Pipeline, _sentinel
from amcd.registry import simulator_registry
from amcd.simulators.base import (
    REQUIRED_PROVENANCE_KEYS,
    IRResult,
    build_simulator,
    validate_provenance,
)

from tests.conftest import QUIET, TEST_TINY, tiny_config


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
        cfg = Config.load(Path("configs/base.yaml"), Path("configs/dry_run.yaml"))
        assert cfg.simulator.name == "dry_run"
        assert cfg.simulator.params == {}

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
            "dry_run", {},
            n_channels=dry_run_config.n_channels,
            n_samples=dry_run_config.n_samples,
            sample_rate=dry_run_config.sample_rate,
        )
        assert type(sim) is simulator_registry.get("dry_run")

    def test_rejects_unknown_param(self) -> None:
        with pytest.raises(Exception, match="extra_forbidden|Extra inputs"):
            build_simulator("dry_run", {"bogus": 1},
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
            "dry_run", {},
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
        cfg = Config.load(Path("configs/base.yaml"), Path("configs/dry_run.yaml"))
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
        with pytest.raises(RuntimeError, match="cached under a DIFFERENT config"):
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
        """The eight `None` entries are a declared gap, not a silent one: they must
        keep working exactly as before until they are wired."""
        # Every stage is listed, so an unwired one is declared rather than absent.
        assert set(STAGE_FINGERPRINT) == set(STAGES)
        assert STAGE_FINGERPRINT["preprocess"] is None
        cfg = tiny_config(scenes={"n_id": 4})
        pipe = self._pipeline(cfg, tmp_path)
        pipe._mark_done("preprocess")
        assert pipe._is_done("preprocess")
