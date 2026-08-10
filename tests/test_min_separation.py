"""Minimum source-receiver separation: the pre-render half of the simulator contract.

Ledger rows F-48 / AC-13 / RD-45 / RD-48 / RD-49 / RD-50 / RD-57 / RD-60.

The defect these cover: `configs/base.yaml` declared `distance_range: null` on both
placement regimes, so it generated pairs below every shipped backend's geometric
floor (P(d < 0.3 m) = 0.186 %/scene → ~67 % chance a 600-scene run aborted), and the
only guard fired INSIDE `DryRunSimulator.render` — mid-batch, hours into an emulated
render, with the stage sentinel never written.

Three layers are asserted here, because each catches something the others cannot:
  * config schema      — the constraint is expressible at all (`[lo, null]`)
  * gen-scenes         — a REGIME that could emit unrenderable scenes, before any exist
  * render pre-flight  — a REALIZED scene, for a set generated under another backend
"""
from pathlib import Path

import numpy as np
import pytest
from pydantic import BaseModel, ValidationError

from amcd.config import Config, PlacementRegime
from amcd.registry import simulator_registry
from amcd.scenes.generator import _check_regimes_clear_backend_floor
from amcd.simulators.base import (
    SceneSpec,
    build_simulator,
    simulator_min_separation,
)
from amcd.simulators.render import _preflight_separations

from tests.conftest import tiny_config

_BASE = Path("configs/base.yaml")


def _regime(distance_range):
    return PlacementRegime(
        type="interior", corner_frac=None, height_range=None,
        distance_range=distance_range,
    )


class TestDistanceRangeSpellings:
    """RD-48: a nullable ELEMENT is a new sub-convention, not a second spelling of null."""

    @pytest.mark.parametrize(
        "spelling", [None, [1.0, None], [None, 10.0], [1.0, 10.0]],
        ids=["null", "min-only", "max-only", "both"],
    )
    def test_legal_spellings_load(self, spelling) -> None:
        assert _regime(spelling).distance_range == spelling

    def test_null_null_is_rejected_and_names_the_right_spelling(self) -> None:
        with pytest.raises(ValidationError, match=r"distance_range: null"):
            _regime([None, None])

    def test_lo_must_stay_below_hi_when_both_present(self) -> None:
        with pytest.raises(ValidationError, match="lo < hi"):
            _regime([10.0, 1.0])

    def test_negative_bound_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="non-negative"):
            _regime([-1.0, None])

    def test_height_range_keeps_the_stricter_contract(self) -> None:
        """No backend constrains one side of a height band, so it takes both or neither."""
        with pytest.raises(ValidationError):
            PlacementRegime(
                type="interior", corner_frac=None,
                height_range=[1.2, None], distance_range=None,
            )


class TestBackendDeclaresItsFloor:
    """RD-49/RD-60: required interface member, readable WITHOUT instantiating."""

    def test_shipped_backends_declare_a_floor(self) -> None:
        # gsound_sir DERIVES its floor from the sphere radii it already declares,
        # rather than repeating the same physical fact as a second config value.
        gsound = Config.load(_BASE)
        params = gsound.simulator.params
        assert simulator_min_separation(gsound) == pytest.approx(
            params["source_radius"] + params["listener_radius"]
        )
        # dry_run's floor is a stated policy, so it is config-governed.
        tiny = tiny_config()
        assert simulator_min_separation(tiny) == pytest.approx(
            tiny.simulator.params["min_source_receiver_distance_m"]
        )

    def test_reading_the_floor_never_constructs_the_backend(self) -> None:
        """gen-scenes runs where the render env may not exist (RD-60)."""
        constructed = []

        @simulator_registry.register("_probe_counts_init")
        class _CountsInit:
            class Params(BaseModel):
                model_config = {"extra": "forbid"}

            def __init__(self, n_channels, n_samples, sample_rate):
                constructed.append(1)

            @classmethod
            def min_source_receiver_distance_m(cls, params: dict) -> float:
                return 0.4

            def render(self, scene, ray_budget):
                raise NotImplementedError

        cfg = tiny_config(simulator={"name": "_probe_counts_init", "params": {}})
        assert simulator_min_separation(cfg) == 0.4
        assert constructed == [], "reading the floor must not instantiate the simulator"

    def test_a_backend_omitting_the_declaration_fails_at_build(self) -> None:
        """Not at render: the whole point is to catch it before one happens."""

        @simulator_registry.register("_probe_no_floor")
        class _NoFloor:
            class Params(BaseModel):
                model_config = {"extra": "forbid"}

            def __init__(self, n_channels, n_samples, sample_rate):
                pass

            def render(self, scene, ray_budget):
                raise NotImplementedError

        with pytest.raises(TypeError, match="min_source_receiver_distance_m"):
            build_simulator("_probe_no_floor", {}, n_channels=4, n_samples=100,
                            sample_rate=8000)


class TestGenScenesRejectsUnrenderableRegimes:
    """RD-45: EVERY declared regime, not only the one the id baseline names."""

    def test_shipped_configs_clear_their_backend_floor(self) -> None:
        for cfg in (Config.load(_BASE), tiny_config()):
            _check_regimes_clear_backend_floor(cfg)

    def test_unconstrained_regime_is_rejected(self) -> None:
        cfg = Config.load(_BASE)
        object.__setattr__(cfg.scenes.placement_regimes["interior_random"],
                           "distance_range", None)
        with pytest.raises(ValueError, match="interior_random"):
            _check_regimes_clear_backend_floor(cfg)

    def test_near_corner_is_checked_too(self) -> None:
        """The regime behind test_placement_shift — the half RD-45 caught missing."""
        cfg = Config.load(_BASE)
        object.__setattr__(cfg.scenes.placement_regimes["near_corner"],
                           "distance_range", [0.05, None])
        with pytest.raises(ValueError, match="near_corner"):
            _check_regimes_clear_backend_floor(cfg)

    def test_max_only_range_is_rejected_as_no_minimum(self) -> None:
        cfg = Config.load(_BASE)
        object.__setattr__(cfg.scenes.placement_regimes["interior_random"],
                           "distance_range", [None, 10.0])
        with pytest.raises(ValueError, match="no minimum declared"):
            _check_regimes_clear_backend_floor(cfg)

    def test_error_points_at_the_config_value_not_the_backend_floor(self) -> None:
        """RD-57: the backend floor is a lower limit on a research choice, not its source."""
        cfg = Config.load(_BASE)
        object.__setattr__(cfg.scenes.placement_regimes["interior_random"],
                           "distance_range", None)
        with pytest.raises(ValueError, match="research choice"):
            _check_regimes_clear_backend_floor(cfg)


class TestRenderPreflight:
    """The realized-scene backstop, for a set generated under a different backend."""

    @staticmethod
    def _scene(scene_id: str, separation: float) -> SceneSpec:
        return SceneSpec(
            scene_id=scene_id, seed=1, geometry_family="shoebox",
            dims=(6.0, 5.0, 3.0), material_absorption=0.2,
            source_pos=(2.0, 2.0, 1.5),
            receiver_pos=(2.0 + separation, 2.0, 1.5),
        )

    def test_a_close_pair_fails_naming_the_scene(self) -> None:
        cfg = tiny_config()
        floor = simulator_min_separation(cfg)
        scenes = [self._scene("scene_0000", 3.0),
                  self._scene("scene_0001", 0.15)]
        with pytest.raises(ValueError, match="scene_0001"):
            _preflight_separations(cfg, scenes)
        assert 0.15 < floor, "fixture must sit below the declared floor"

    def test_all_offenders_are_listed_at_once(self) -> None:
        """Failing per-scene mid-loop is what costs an emulated batch."""
        cfg = tiny_config()
        scenes = [self._scene(f"scene_{i:04d}", 0.05) for i in range(3)]
        with pytest.raises(ValueError) as exc:
            _preflight_separations(cfg, scenes)
        for i in range(3):
            assert f"scene_{i:04d}" in str(exc.value)
        assert "none were rendered" in str(exc.value)

    def test_a_clearing_scene_set_passes(self) -> None:
        cfg = tiny_config()
        _preflight_separations(cfg, [self._scene("scene_0000", 2.0)])


class TestBaseConfigCannotEmitBelowItsFloor:
    """F-48's own pass condition, on the shipped config."""

    def test_base_declares_a_minimum_on_every_regime(self) -> None:
        cfg = Config.load(_BASE)
        for name, regime in cfg.scenes.placement_regimes.items():
            assert regime.distance_range is not None, f"{name} declares no distance_range"
            lo = regime.distance_range[0]
            assert lo is not None and lo > 0, f"{name} declares no minimum separation"

    def test_generated_scenes_clear_the_declared_minimum(self, tmp_path: Path) -> None:
        from amcd.scenes.generator import run_gen_scenes
        from tests.conftest import QUIET

        cfg = tiny_config(scenes={"n_id": 40})
        run_gen_scenes(cfg, tmp_path, QUIET)
        floors = {
            name: regime.distance_range[0]
            for name, regime in cfg.scenes.placement_regimes.items()
        }
        seen = 0
        for path in sorted(tmp_path.glob("scenes/scene_*.json")):
            scene = SceneSpec.from_json(path)
            d = float(np.linalg.norm(
                np.subtract(scene.source_pos, scene.receiver_pos)))
            regime = scene.regime_axes["placement"]
            assert d >= floors[regime], f"{scene.scene_id}: {d} m under {regime}"
            seen += 1
        assert seen > 0
