"""Research-I-faithful scene generation and split sizing (gsound_sir gate, Step 1.5).

Covers ledger rows RD-27..RD-29 (the declared RI deviations are actually what the
config says), RD-32 (the RI overlay resolves to pure count mode despite YAML's
inability to delete keys), RD-36 (unconstrained configs keep their exact RNG
stream) and RD-37 (joint resampling + recorded acceptance rates).

The load-bearing property here is that a config CANNOT quietly mean something
other than it says: mixed sizing modes, colliding split seeds, inert overrides
and unreachable placement constraints all raise.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from amcd.config import Config
from amcd.scenes.generator import _generation_plan, run_gen_scenes

from tests.conftest import QUIET, tiny_config

_RI = (Path("configs/base.yaml"), Path("configs/research_i.yaml"))

# Research I Figure 6 (docs/research_I_paper.md l.507-514).
RI_SPLITS = {
    "train": (500, 1001), "valid": (60, 1002), "test_id": (60, 1003),
    "test_material_shift": (40, 1004), "test_placement_shift": (30, 1005),
    "test_geometry_shift": (30, 1006),
}


@pytest.fixture(scope="module")
def ri_config() -> Config:
    return Config.load(*_RI)


class TestResearchIPin:
    """configs/research_i.yaml must resolve to Figure 5 + Figure 6, verbatim."""

    def test_figure5_render_values(self, ri_config: Config) -> None:
        assert ri_config.seeds.master == 42
        assert ri_config.sample_rate == 48000
        assert ri_config.ir_duration == 3.0
        assert ri_config.ambisonics_order == 3
        assert (ri_config.low_ray_budget, ri_config.high_ray_budget) == (5000, 200000)
        assert ri_config.simulator.params["path_retention"] == {"mode": "top_k", "value": 5000}

    def test_figure5_geometry_and_placement(self, ri_config: Config) -> None:
        fam = ri_config.scenes.geometry_families
        assert fam["shoebox"]["dims"] == [[4.0, 14.0], [3.0, 10.0], [2.4, 4.5]]
        assert fam["corridor"]["dims"] == [[8.0, 24.0], [1.8, 4.0], [2.4, 4.0]]
        assert ri_config.scenes.margins.model_dump() == {
            "wall": 0.5, "floor": 0.5, "ceiling": 0.3
        }
        for name in ("interior_random", "near_corner"):
            regime = ri_config.scenes.placement_regimes[name]
            assert regime.height_range == [1.2, 1.8]
            assert regime.distance_range == [1.0, 10.0]

    def test_figure6_counts_and_seeds(self, ri_config: Config) -> None:
        assert {n: (sp.count, sp.seed) for n, sp in ri_config.splits.items()} == RI_SPLITS

    def test_overlay_resolves_to_pure_count_mode(self, ri_config: Config) -> None:
        """RD-32: YAML deep-merge cannot DELETE base.yaml's `frac`/`n_id`, so the
        overlay must null them explicitly or the all-count validator would reject
        the very config it exists to express."""
        assert ri_config.id_pool_is_counted
        assert ri_config.scenes.n_id is None
        assert all(sp.frac is None for sp in ri_config.splits.values())

    def test_geometry_shift_stays_single_axis(self, ri_config: Config) -> None:
        """RD-27: RI's dual-axis geometry shift is deliberately NOT reproduced;
        invariant #10 wins. This test pins the deviation so it cannot drift back
        silently in either direction."""
        assert ri_config.splits["test_geometry_shift"].axes == {"geometry": "corridor"}

    def test_material_shift_omits_asymmetric_walls(self, ri_config: Config) -> None:
        """RD-28: half of RI's material shift is inexpressible with a scalar
        absorption, so the split is weaker than RI's and must stay declared."""
        assert ri_config.splits["test_material_shift"].axes == {
            "material": "ceiling_absorptive"
        }
        assert "asymmetric_walls" not in ri_config.scenes.material_regimes


class TestSizingModeValidation:
    def test_same_split_declaring_both_rejected(self) -> None:
        """Deep-merging a `count` onto base.yaml's `frac: 0.6` leaves both set —
        the likeliest way to reach a mixed config by accident."""
        with pytest.raises(ValueError, match="declare BOTH `frac` and `count`"):
            tiny_config(splits={"train": {"role": "train", "count": 5, "seed": 11}})

    def test_mixed_across_splits_rejected(self) -> None:
        """Some id-pool splits counted, others proportional to a pool they do not
        draw from: no coherent meaning, so it raises rather than being guessed at."""
        with pytest.raises(ValueError, match="all-`count` or all-`frac`"):
            tiny_config(splits={
                "train": {"role": "train", "frac": None, "count": 5, "seed": 11},
                # valid keeps base.yaml's frac: 0.2
            })

    def test_count_mode_requires_null_n_id(self) -> None:
        with pytest.raises(ValueError, match="n_id must be null in count mode"):
            tiny_config(
                scenes={"n_id": 20},
                splits={
                    "train":   {"role": "train", "frac": None, "count": 4, "seed": 11},
                    "valid":   {"role": "valid", "frac": None, "count": 2, "seed": 12},
                    "test_id": {"role": "test",  "frac": None, "count": 2, "seed": 13},
                },
            )

    def test_count_mode_requires_per_split_seed(self) -> None:
        with pytest.raises(ValueError, match="requires an explicit per-split `seed`"):
            tiny_config(
                scenes={"n_id": None},
                splits={
                    "train":   {"role": "train", "frac": None, "count": 4, "seed": 11},
                    "valid":   {"role": "valid", "frac": None, "count": 2, "seed": 12},
                    "test_id": {"role": "test",  "frac": None, "count": 2},  # no seed
                },
            )

    def test_frac_mode_requires_n_id(self) -> None:
        with pytest.raises(ValueError, match="n_id is required in frac mode"):
            tiny_config(scenes={"n_id": None})

    def test_duplicate_split_seeds_rejected(self) -> None:
        """Two splits sharing a seed generate the SAME scenes — an overlap that
        reads as a legitimate dataset."""
        with pytest.raises(ValueError, match="share seed"):
            tiny_config(
                scenes={"n_id": None},
                splits={
                    "train":   {"role": "train", "frac": None, "count": 4, "seed": 11},
                    "valid":   {"role": "valid", "frac": None, "count": 2, "seed": 11},
                    "test_id": {"role": "test",  "frac": None, "count": 2, "seed": 13},
                },
            )

    def test_inert_split_assignment_override_rejected(self) -> None:
        """Count mode does no hash-bucketing, so the override would do nothing."""
        with pytest.raises(ValueError, match="would have no effect"):
            tiny_config(
                scenes={"n_id": None},
                seeds={"master": 0, "split_assignment": 999},
                splits={
                    "train":   {"role": "train", "frac": None, "count": 4, "seed": 11},
                    "valid":   {"role": "valid", "frac": None, "count": 2, "seed": 12},
                    "test_id": {"role": "test",  "frac": None, "count": 2, "seed": 13},
                },
            )


class TestPlacementRegimeSchema:
    def test_typo_is_rejected_not_ignored(self) -> None:
        with pytest.raises(ValueError, match="extra_forbidden|Extra inputs"):
            tiny_config(scenes={"placement_regimes": {"interior_random": {
                "type": "interior", "corner_frac": None,
                "hieght_range": None, "distance_range": None,  # typo
            }}})

    def test_corner_frac_on_interior_rejected(self) -> None:
        with pytest.raises(ValueError, match="meaningless for placement type"):
            tiny_config(scenes={"placement_regimes": {"interior_random": {
                "type": "interior", "corner_frac": 0.2,
                "height_range": None, "distance_range": None,
            }}})

    def test_wall_type_not_supported(self) -> None:
        """RD-34: RI never quantifies `near_wall`, so no `wall` type ships — a
        regime nothing defines must not be silently selectable."""
        with pytest.raises(ValueError, match="must be interior\\|corner"):
            tiny_config(scenes={"placement_regimes": {"interior_random": {
                "type": "wall", "corner_frac": None,
                "height_range": None, "distance_range": None,
            }}})


class TestPlacementConstraints:
    def _gen(self, tmp_path: Path, **overrides) -> list[dict]:
        cfg = tiny_config(**overrides)
        run_gen_scenes(cfg, tmp_path, QUIET)
        return [json.loads(p.read_text())
                for p in sorted((tmp_path / "scenes").glob("scene_*.json"))]

    def test_height_and_distance_constraints_hold(self, tmp_path: Path) -> None:
        specs = self._gen(tmp_path, scenes={
            "n_id": 12,
            "placement_regimes": {"interior_random": {
                "type": "interior", "corner_frac": None,
                "height_range": [1.2, 1.8], "distance_range": [1.0, 4.0],
            }},
        })
        id_specs = [s for s in specs if s["regime_axes"]["placement"] == "interior_random"]
        assert id_specs
        for s in id_specs:
            d = float(np.linalg.norm(np.subtract(s["source_pos"], s["receiver_pos"])))
            assert 1.0 <= d <= 4.0, f"{s['scene_id']}: distance {d}"
            for pos in (s["source_pos"], s["receiver_pos"]):
                assert 1.2 <= pos[2] <= 1.8, f"{s['scene_id']}: height {pos[2]}"

    def test_unreachable_distance_raises_with_the_geometry(self, tmp_path: Path) -> None:
        """No silent fallback: an impossible constraint names the box it could not
        satisfy, rather than emitting scenes that violate it."""
        with pytest.raises(RuntimeError, match="max separation in this box"):
            self._gen(tmp_path, scenes={
                "n_id": 2,
                "placement_regimes": {"interior_random": {
                    "type": "interior", "corner_frac": None,
                    "height_range": None, "distance_range": [500.0, 600.0],
                }},
            })

    def test_height_range_outside_margins_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="does not fit inside the admissible"):
            self._gen(tmp_path, scenes={
                "n_id": 2,
                "placement_regimes": {"interior_random": {
                    "type": "interior", "corner_frac": None,
                    "height_range": [1.2, 90.0], "distance_range": None,
                }},
            })

    def test_placement_report_records_acceptance_and_distances(self, tmp_path: Path) -> None:
        """RD-37/RD-29: rejected draws are accounted for, and the realized
        distance distribution is recorded so the E1 report can quantify what
        stood in for RI's unspecified pairing sub-regimes."""
        self._gen(tmp_path, scenes={
            "n_id": 12,
            "placement_regimes": {"interior_random": {
                "type": "interior", "corner_frac": None,
                "height_range": [1.2, 1.8], "distance_range": [1.0, 4.0],
            }},
        })
        report = json.loads((tmp_path / "scenes" / "placement_report.json").read_text())
        entry = report["id"]
        assert entry["placement_attempts"] >= entry["n_scenes"]
        assert 0.0 < entry["acceptance_rate"] <= 1.0
        assert entry["distance_range_declared"] == [1.0, 4.0]
        dist = entry["source_receiver_distance_m"]
        assert 1.0 <= dist["min"] <= dist["median"] <= dist["max"] <= 4.0


class TestRngStreamPreserved:
    """RD-36: adding the constraint machinery must not perturb configs that
    declare no constraints, or every existing dry-run dataset silently changes."""

    def test_unconstrained_generation_is_unchanged_by_a_null_distance_range(
        self, tmp_path: Path
    ) -> None:
        base = tiny_config(scenes={"n_id": 8})
        run_gen_scenes(base, tmp_path / "a", QUIET)

        # Same config, but the null constraints spelled out rather than inherited.
        explicit = tiny_config(scenes={"n_id": 8, "placement_regimes": {
            "interior_random": {"type": "interior", "corner_frac": None,
                                "height_range": None, "distance_range": None},
        }})
        run_gen_scenes(explicit, tmp_path / "b", QUIET)

        def load(d: Path) -> list[dict]:
            return [json.loads(p.read_text())
                    for p in sorted((d / "scenes").glob("scene_*.json"))]

        assert load(tmp_path / "a") == load(tmp_path / "b")

    def test_single_attempt_when_unconstrained(self, tmp_path: Path) -> None:
        run_gen_scenes(tiny_config(scenes={"n_id": 8}), tmp_path, QUIET)
        report = json.loads((tmp_path / "scenes" / "placement_report.json").read_text())
        assert report["id"]["acceptance_rate"] == 1.0


class TestGenerationPlan:
    def test_frac_mode_generates_one_pooled_entry(self, dry_run_config: Config) -> None:
        regimes = [entry[0] for entry in _generation_plan(dry_run_config)]
        assert regimes[0] == "id"
        assert set(regimes[1:]) == set(dry_run_config.shift_splits)

    def test_count_mode_generates_one_entry_per_split(self, ri_config: Config) -> None:
        plan = _generation_plan(ri_config)
        assert {entry[0] for entry in plan} == set(RI_SPLITS)
        assert {entry[0]: (entry[1], entry[3]) for entry in plan} == RI_SPLITS
