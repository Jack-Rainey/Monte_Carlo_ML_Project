"""Research-I-faithful scene generation and split sizing (gsound_sir gate, Step 1.5).

Covers ledger rows RD-27..RD-29 (the declared RI deviations are actually what the
config says), RD-32 (the RI overlay resolves to pure count mode despite YAML's
inability to delete keys), RD-36 (unconstrained configs keep their exact RNG
stream) and RD-37 (joint resampling + recorded acceptance rates).

The second half of the file covers what the record-length gate SCORES and what it
DISCLOSES: F-71 (uncharacterized scenes leave the gate's denominator), RD-65 (the
per-split over-limit warning), RD-112 (a gate that scored nothing is unscored, not
passed), RD-113 (the derived denominator is pinned to the published one) and AC-30
(the realized shortfall against ISO 3382-1 §5.3's minimum distance).

The load-bearing property here is that a config CANNOT quietly mean something
other than it says: mixed sizing modes, colliding split seeds, inert overrides
and unreachable placement constraints all raise.
"""
import itertools
import json
from pathlib import Path

import numpy as np
import pytest

from amcd.config import Config, PlacementRegime
from amcd.scenes.generator import (
    _disclose_and_gate_record_length,
    _generation_plan,
    _placement_bounds,
    _room_acoustics,
    _sample_positions,
    run_gen_scenes,
)
from amcd.simulators.base import SceneSpec

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
        assert fam["shoebox"].dims == [[4.0, 14.0], [3.0, 10.0], [2.4, 4.5]]
        assert fam["corridor"].dims == [[8.0, 24.0], [1.8, 4.0], [2.4, 4.0]]
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
        # `distance_range` must still clear the backend floor here (F-48), or this
        # would fail on the separation pre-flight and never reach the height check.
        with pytest.raises(ValueError, match="does not fit inside the admissible"):
            self._gen(tmp_path, scenes={
                "n_id": 2,
                "placement_regimes": {"interior_random": {
                    "type": "interior", "corner_frac": None,
                    "height_range": [1.2, 90.0], "distance_range": [1.0, None],
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
    """RD-36: the constraint machinery must not perturb the unconstrained path.

    Asserted directly on `_sample_positions` rather than end-to-end, because
    base.yaml itself is no longer unconstrained — F-48 gave both of its regimes a
    1.0 m minimum, so a whole-pipeline comparison would compare two constrained
    configs and prove nothing about the null path. The property RD-36 needs is
    narrower and is exactly this: with `distance_range: null` the loop body runs
    once and issues the same two 3-vector `uniform` calls, in the same order, as
    before the constraint existed. Switching those to per-axis scalar draws would
    change the stream and silently re-dataset every unconstrained config.
    """

    @staticmethod
    def _regime(distance_range):
        return PlacementRegime(
            type="interior", corner_frac=None,
            height_range=None, distance_range=distance_range,
        )

    def test_null_distance_range_keeps_the_pre_constraint_call_sequence(self) -> None:
        cfg = tiny_config()
        margins, dims = cfg.scenes.margins, (6.0, 5.0, 3.0)
        regimes = {"r": self._regime(None)}

        # The reference stream, as it was before rejection sampling existed:
        # exactly two 3-vector uniform draws over the admissible box.
        lo, hi = _placement_bounds(dims, margins, None)
        ref_rng = np.random.default_rng(7)
        expected_src = tuple(float(v) for v in ref_rng.uniform(lo, hi))
        expected_rcv = tuple(float(v) for v in ref_rng.uniform(lo, hi))
        # A draw AFTER the pair must also line up, or the stream has been consumed
        # at a different rate and every later scene shifts.
        expected_next = float(ref_rng.random())

        rng = np.random.default_rng(7)
        src, rcv, stats = _sample_positions("r", dims, rng, regimes, margins, 1000)
        assert (src, rcv) == (expected_src, expected_rcv)
        assert stats["attempts"] == 1
        assert float(rng.random()) == expected_next

    def test_single_attempt_when_unconstrained(self) -> None:
        """The null path never enters the rejection loop a second time."""
        cfg = tiny_config()
        regimes = {"r": self._regime(None)}
        for seed in range(20):
            _, _, stats = _sample_positions(
                "r", (6.0, 5.0, 3.0), np.random.default_rng(seed),
                regimes, cfg.scenes.margins, 1000,
            )
            assert stats == {"attempts": 1, "below_min": 0, "above_max": 0,
                             "max_reachable_m": stats["max_reachable_m"]}

    def test_the_declared_floor_is_what_does_the_rejecting(self, tmp_path: Path) -> None:
        """F-48's floor is live, not decorative: draws below it ARE discarded, and
        the discard is accounted for on the correct side (AC-14)."""
        run_gen_scenes(tiny_config(scenes={"n_id": 60}), tmp_path, QUIET)
        report = json.loads((tmp_path / "scenes" / "placement_report.json").read_text())
        entry = report["id"]
        assert entry["distance_range_declared"][0] > 0
        assert entry["rejected_below_min"] > 0, "the 1.0 m floor rejected nothing"
        assert entry["rejected_above_max"] == 0, "no maximum is declared"
        assert entry["source_receiver_distance_m"]["min"] >= entry["distance_range_declared"][0]


class TestCornerBiasIsHorizontal:
    """AC-10: `corner_frac` must not be applied to the z axis.

    With a declared height_range, z is an ergonomic band rather than a room
    boundary — 1.2 m is 1.2 m off the floor either way — so biasing it buys no
    boundary proximity while silently narrowing a REPORTED robustness split's
    receiver-height distribution (observed: [1.20, 1.31] instead of [1.2, 1.8]).
    """

    def test_receiver_height_spans_the_declared_band(self, tmp_path: Path) -> None:
        cfg = tiny_config(
            scenes={
                "n_id": 4,
                "placement_regimes": {"near_corner": {
                    "type": "corner", "corner_frac": 0.2,
                    "height_range": [1.2, 1.8], "distance_range": [1.0, None],
                }},
            },
            splits={"test_placement_shift": {
                "role": "test", "count": 40, "axes": {"placement": "near_corner"}}},
        )
        run_gen_scenes(cfg, tmp_path, QUIET)
        specs = [json.loads(p.read_text())
                 for p in sorted((tmp_path / "scenes").glob("scene_*.json"))]
        corner = [s for s in specs if s["regime_axes"]["placement"] == "near_corner"]
        assert corner

        rcv_z = np.array([s["receiver_pos"][2] for s in corner])
        # Horizontal axes ARE confined; z is not.
        for s in corner:
            for i in (0, 1):
                hi = s["dims"][i] - cfg.scenes.margins.wall
                corner_hi = cfg.scenes.margins.wall + 0.2 * (hi - cfg.scenes.margins.wall)
                assert s["receiver_pos"][i] <= corner_hi + 1e-6
        assert rcv_z.min() >= 1.2 and rcv_z.max() <= 1.8
        # Must cover well beyond the collapsed [1.2, 1.32] band the bug produced.
        assert rcv_z.max() - rcv_z.min() > 0.4, (
            f"receiver height band collapsed to [{rcv_z.min():.3f}, {rcv_z.max():.3f}]"
        )

    def test_report_separates_source_and_receiver_heights(self, tmp_path: Path) -> None:
        """A pooled source+receiver statistic hid the collapse."""
        run_gen_scenes(tiny_config(scenes={"n_id": 4}), tmp_path, QUIET)
        report = json.loads((tmp_path / "scenes" / "placement_report.json").read_text())
        entry = next(iter(report.values()))
        assert "source_height_m" in entry and "receiver_height_m" in entry


class TestPropagationDelay:
    """AC-11: the scaffold must obey the speed of sound it declares."""

    def test_onset_matches_distance_over_c(self) -> None:
        from amcd.evaluation.room_acoustic import _find_onset
        from tests.conftest import dry_run_simulator

        cfg = tiny_config()
        c = cfg.simulator.params["speed_of_sound_m_s"]
        sim = dry_run_simulator(n_channels=4, n_samples=48000, sample_rate=48000)

        for distance in (1.0, 5.0, 9.8):
            scene = SceneSpec(
                scene_id="s", seed=1, geometry_family="shoebox", dims=(30.0, 10.0, 3.0),
                material_absorption=0.3,
                source_pos=(1.0, 1.0, 1.5),
                receiver_pos=(1.0 + distance, 1.0, 1.5),
            )
            onset = _find_onset(sim.render(scene, 5000).ir[0], cfg.metric_onset_rel_db)
            expected = round(distance / c * 48000)
            assert abs(onset - expected) <= 1, (
                f"d={distance} m: onset {onset} vs expected {expected} at c={c} m/s"
            )

    def test_declared_speed_is_the_configured_one(self) -> None:
        from tests.conftest import dry_run_simulator

        cfg = tiny_config()
        sim = dry_run_simulator(n_channels=4, n_samples=800, sample_rate=8000)
        scene = SceneSpec(
            scene_id="s", seed=1, geometry_family="shoebox", dims=(5.0, 4.0, 3.0),
            material_absorption=0.3, source_pos=(1.0, 1.0, 1.5), receiver_pos=(3.0, 2.0, 1.5),
        )
        meta = sim.render(scene, 50).meta
        assert meta["speed_of_sound_m_s"] == cfg.simulator.params["speed_of_sound_m_s"]


class TestConfigGuards:
    """Values that would otherwise be silently ignored or silently wrong."""

    def test_negative_margin_rejected(self) -> None:
        """F-34: a negative margin places sources OUTSIDE the room, and the
        emptiness check passes because the box is merely inverted, not empty."""
        with pytest.raises(ValueError, match="must be >= 0"):
            tiny_config(scenes={"margins": {"wall": -1.0}})

    def test_zero_placement_attempts_rejected(self) -> None:
        """F-32: 0 attempts made every scene fail with a message blaming a
        distance constraint the config never declared."""
        with pytest.raises(ValueError, match="max_placement_attempts must be > 0"):
            tiny_config(scenes={"max_placement_attempts": 0})

    def test_frac_on_shift_split_rejected(self) -> None:
        with pytest.raises(ValueError, match="`frac` would be ignored"):
            tiny_config(splits={"test_geometry_shift": {
                "role": "test", "count": 2, "frac": 0.1,
                "axes": {"geometry": "corridor"}}})

    def test_seed_on_id_pool_split_in_frac_mode_rejected(self) -> None:
        """F-33: the researcher believes the split is independently seeded; in
        frac mode it is not, because the pool is generated from one stream."""
        with pytest.raises(ValueError, match="not independently seeded"):
            tiny_config(splits={"train": {"role": "train", "frac": 0.6, "seed": 7777}})

    def test_count_mode_requires_seed_on_shift_splits_too(self) -> None:
        """F-35: count mode's contract is that EVERY split is independently
        seeded, not just the id-pool ones."""
        with pytest.raises(ValueError, match="EVERY split"):
            tiny_config(
                scenes={"n_id": None},
                splits={
                    "train":   {"role": "train", "frac": None, "count": 4, "seed": 11},
                    "valid":   {"role": "valid", "frac": None, "count": 2, "seed": 12},
                    "test_id": {"role": "test",  "frac": None, "count": 2, "seed": 13},
                    "test_geometry_shift": {"role": "test", "count": 2,
                                            "axes": {"geometry": "corridor"}},  # no seed
                },
            )

    def test_unknown_geometry_key_rejected(self) -> None:
        """F-31: an unrecognised key here is a HIDDEN GEOMETRY PARAMETER."""
        with pytest.raises(ValueError, match="extra_forbidden|Extra inputs"):
            tiny_config(scenes={"geometry_families": {
                "shoebox": {"dims": [[3.0, 12.0], [3.0, 10.0], [2.4, 5.0]], "shape": "L"}}})

    def test_typo_in_dims_key_rejected(self) -> None:
        with pytest.raises(ValueError, match="extra_forbidden|Extra inputs|Field required"):
            tiny_config(scenes={"geometry_families": {
                "shoebox": {"dimz": [[3.0, 12.0], [3.0, 10.0], [2.4, 5.0]]}}})

    def test_unknown_material_key_rejected(self) -> None:
        with pytest.raises(ValueError, match="extra_forbidden|Extra inputs"):
            tiny_config(scenes={"material_regimes": {
                "mixed": {"absorption": [0.05, 0.8], "scattering": 0.5}}})


class TestGenerationPlan:
    def test_frac_mode_generates_one_pooled_entry(self, dry_run_config: Config) -> None:
        regimes = [entry[0] for entry in _generation_plan(dry_run_config)]
        assert regimes[0] == "id"
        assert set(regimes[1:]) == set(dry_run_config.shift_splits)

    def test_count_mode_generates_one_entry_per_split(self, ri_config: Config) -> None:
        plan = _generation_plan(ri_config)
        assert {entry[0] for entry in plan} == set(RI_SPLITS)
        assert {entry[0]: (entry[1], entry[3]) for entry in plan} == RI_SPLITS


# ═══════════════════════════════════════════════════════════════════════════════
# The record-length gate: what it scores, and what it discloses
# (F-71 / RD-65 / RD-112 / RD-113), plus the ISO 3382-1 §5.3 distance disclosure
# (AC-30 / AC-50).
#
# Every test below constructs the population in which the defect is VISIBLE — a
# healthy run has no uncharacterized scenes at all, so none of these fire on one.
# ═══════════════════════════════════════════════════════════════════════════════

def _openfield_config(**scene_overrides) -> Config:
    """tiny_config plus a 3-scene split whose geometry declares no enclosure.

    `openfield` is the RD-64 seam exercised: a family declaring
    `characterization: none` gets a recorded reason instead of Sabine/Eyring
    numbers, which is the state F-71 is about.
    """
    return tiny_config(
        scenes={
            "geometry_families": {
                "openfield": {
                    "dims": [[8.0, 12.0], [8.0, 12.0], [3.0, 4.0]],
                    "characterization": "none",
                },
            },
            **scene_overrides,
        },
        splits={
            "test_openfield": {
                "role": "test", "count": 3, "axes": {"geometry": "openfield"},
            },
        },
    )


class TestUncharacterizedScenesLeaveTheRecordLengthGate:
    """F-71: the `characterization: none` branch set `t60_exceeds_ir_duration`
    False rather than omitting it, so an unmodelled geometry entered the gate's
    denominator as PASSING. N of them shrink the over-limit fraction by N/(N+M)."""

    @staticmethod
    def _report(run_dir: Path) -> dict:
        return json.loads((run_dir / "scenes" / "placement_report.json").read_text())

    def test_the_flag_block_excludes_them_instead_of_counting_them_as_passing(
        self, tmp_path: Path
    ) -> None:
        run_gen_scenes(_openfield_config(), tmp_path, QUIET)
        block = self._report(tmp_path)["test_openfield"]["t60_over_ir_duration"]

        assert block["n_scenes"] == 0, (
            "uncharacterized scenes are still in the denominator — the sibling "
            "diffuse_field_validity block has honoured this rule all along (F-71)"
        )
        assert block["n_uncharacterized"] == 3
        assert block["t60_exceeds_ir_duration"]["count"] == 0
        assert block["t60_exceeds_ir_duration"]["fraction"] is None, (
            "an unscored quantity was rendered as a number"
        )

    def test_the_scene_carries_a_reason_and_no_record_length_flag(
        self, tmp_path: Path
    ) -> None:
        room = _room_acoustics(
            (10.0, 10.0, 3.5), 0.3, 4.0,
            alpha_limit=0.5, ir_duration_s=0.1, characterization="none",
        )
        assert "t60_exceeds_ir_duration" not in room, (
            "present-and-False reads as 'measured, and within the record' (F-71)"
        )
        assert "uncharacterized_reason" in room

    def test_non_enclosures_cannot_buy_a_pass_for_a_breaching_dataset(
        self, tmp_path: Path
    ) -> None:
        """The dilution attack, at the tolerance where it decides the outcome.

        Every one of the 26 enclosed scenes breaches the 0.1 s record, so the
        honest fraction is 26/26 = 100 % against a declared tolerance of 0.9.
        Counting the 3 openfield scenes as passing gives 26/29 = 89.7 %, under the
        limit: the dataset would buy its pass with scenes nobody characterized.
        """
        cfg = _openfield_config(max_t60_over_ir_duration_frac=0.9)
        with pytest.raises(ValueError, match="26 of 26 scenes") as exc:
            run_gen_scenes(cfg, tmp_path, QUIET)
        assert "3 of 29 scenes are excluded" in str(exc.value), (
            "the gate must disclose what it did NOT score, not just what it did"
        )

    def test_the_derived_denominator_agrees_with_the_published_one(
        self, tmp_path: Path
    ) -> None:
        """RD-113: the gate derives the characterized count that `_flag_counts`
        already publishes. Two expressions for one number is how AC-24's pair
        drifted apart, so they are pinned to each other here."""
        run_gen_scenes(_openfield_config(), tmp_path, QUIET)
        for name, entry in self._report(tmp_path).items():
            block = entry["t60_over_ir_duration"]
            derived = entry["n_scenes"] - block.get("n_uncharacterized", 0)
            assert derived == block["n_scenes"], name


class TestPerSplitOverLimitWarning:
    """RD-65: the gate is the OVERALL fraction — right, because a per-split gate
    lets the smallest split set the tolerance for train — but an overall pass can
    hide a shift split far over on its own, and the per-shift breakdown IS the
    research result. So every offending split is named, unconditionally."""

    @staticmethod
    def _report(train_over: int, shift_over: int) -> dict:
        return {
            "train": {"n_scenes": 500, "t60_over_ir_duration": {
                "t60_exceeds_ir_duration": {"count": train_over}}},
            "test_placement_shift": {"n_scenes": 30, "t60_over_ir_duration": {
                "t60_exceeds_ir_duration": {"count": shift_over}}},
        }

    def test_a_split_over_its_own_limit_is_named_while_the_gate_passes(
        self, ri_config: Config, capsys
    ) -> None:
        # 1 of 30 is 3.3 % per split but 0.19 % overall: the gate allows it.
        _disclose_and_gate_record_length(ri_config, self._report(0, 1), QUIET)
        warnings = capsys.readouterr().err

        assert "test_placement_shift" in warnings
        assert "1/30" in warnings and "3.333%" in warnings
        assert "train" not in warnings, "a split within its own limit is not warned"

    def test_the_warning_is_emitted_before_the_gate_can_raise(
        self, ri_config: Config, capsys
    ) -> None:
        """Evidence a failing run still names the splits responsible."""
        with pytest.raises(ValueError):
            _disclose_and_gate_record_length(ri_config, self._report(16, 1), QUIET)
        warnings = capsys.readouterr().err

        assert "test_placement_shift" in warnings and "train" in warnings

    def test_warnings_survive_the_quietest_verbosity(
        self, ri_config: Config, capsys
    ) -> None:
        """QUIET is show=0. Warnings bypass the ladder entirely (F-24), which is
        what makes "always-emitted" true rather than aspirational."""
        _disclose_and_gate_record_length(ri_config, self._report(0, 1), QUIET)
        assert "WARNING" in capsys.readouterr().err

    def test_a_split_with_no_characterized_scene_is_named_as_undefined(
        self, tmp_path: Path, capsys
    ) -> None:
        run_gen_scenes(_openfield_config(), tmp_path, QUIET)
        warnings = capsys.readouterr().err

        assert "test_openfield" in warnings
        assert "0 of 3 scenes are characterized" in warnings
        assert "UNDEFINED" in warnings
        # The artifact side of the same claim: null, never a number to average.
        report = json.loads(
            (tmp_path / "scenes" / "placement_report.json").read_text()
        )
        over = report["test_openfield"]["t60_over_ir_duration"]
        assert over["t60_exceeds_ir_duration"]["fraction"] is None

    def test_a_wholly_uncharacterized_config_is_unscored_not_passed(
        self, capsys
    ) -> None:
        """RD-112: with every scene uncharacterized the gate has nothing to measure.
        Falling through would be F-71's own defect one level up — a silent pass at
        exactly the outdoor/partially-open configuration the RD-64 seam enables."""
        report = {
            "test_openfield": {"n_scenes": 3, "t60_over_ir_duration": {
                "n_uncharacterized": 3,
                "t60_exceeds_ir_duration": {"count": 0, "fraction": None}}},
        }
        _disclose_and_gate_record_length(tiny_config(), report, QUIET)
        warnings = capsys.readouterr().err

        assert "UNSCORED, not passed" in warnings
        assert "0 of 3 scenes" in warnings

    def test_the_same_config_survives_the_real_generation_path(
        self, tmp_path: Path, capsys
    ) -> None:
        """The test above builds the report by hand, so it never exercised the
        corner disclosure — which reached `f"{None:.2f}"` and raised TypeError
        before RD-112's warning could be emitted. `Config.worst_case_t60` returns a
        reasoned None for a config with no `sabine` family, which is the very
        config RD-112 is about, so the warning was unreachable on it."""
        cfg = tiny_config(scenes={"geometry_families": {
            "shoebox": {"dims": [[3.0, 12.0], [3.0, 10.0], [2.4, 5.0]],
                        "characterization": "none"},
            "corridor": {"dims": [[15.0, 30.0], [1.5, 3.0], [2.4, 3.5]],
                         "characterization": "none"}}})
        run_gen_scenes(cfg, tmp_path, QUIET)
        warnings = capsys.readouterr().err

        assert "UNSCORED, not passed" in warnings
        report = json.loads(
            (tmp_path / "scenes" / "placement_report.json").read_text()
        )
        assert all(
            e["t60_over_ir_duration"]["n_scenes"] == 0 for e in report.values()
        )


class TestIsoMinimumDistanceDisclosure:
    """AC-30: the config declares ONE global placement floor, but ISO 3382-1 §5.3's
    minimum measurement distance is per scene — it varies with each scene's own
    absorption and surface. The shortfall is measured and reported, not asserted
    away. The per-scene criterion itself stays deferred."""

    @staticmethod
    def _d_min(dims, alpha) -> dict:
        return _room_acoustics(
            dims, alpha, 2.0,
            alpha_limit=0.5, ir_duration_s=10.0, characterization="sabine",
        )

    @staticmethod
    def _declared_support(cfg: Config) -> dict[str, tuple[float, float]]:
        """Sweep every declared geometry x material corner, the way
        `Config.worst_case_t60` sweeps them for the T60 corner.

        DERIVED from the config, never hardcoded (AC-50): AC-30's own [0.41, 5.16] m
        was computed over the `mixed` regime alone, so it missed
        `ceiling_absorptive` (alpha up to 0.98) on the same shoebox family — and a
        test that restated the literals could not see the omission.
        """
        out: dict[str, list[float]] = {"sabine": [], "eyring": []}
        for spec in cfg.scenes.geometry_families.values():
            if spec.characterization != "sabine":
                continue
            for dims in itertools.product(*[(a[0], a[1]) for a in spec.dims]):
                for regime in cfg.scenes.material_regimes.values():
                    for alpha in regime.absorption:
                        room = TestIsoMinimumDistanceDisclosure._d_min(dims, alpha)
                        out["sabine"].append(room["iso_min_distance_sabine_m"])
                        out["eyring"].append(room["iso_min_distance_eyring_m"])
        return {k: (min(v), max(v)) for k, v in out.items()}

    def test_the_declared_support_spans_every_declared_material_regime(self) -> None:
        """base.yaml declares `mixed` [0.05, 0.80] AND `ceiling_absorptive`
        [0.85, 0.98] over the same shoebox family, and `test_material_shift`
        selects the second. The support is the union, not one regime's span."""
        support = self._declared_support(Config.load(Path("configs/base.yaml")))

        assert support["sabine"] == pytest.approx((0.412, 5.712), abs=0.005)
        assert support["eyring"] == pytest.approx((0.417, 11.413), abs=0.005)
        assert support["sabine"][1] > 5.16, (
            "5.16 m is the `mixed` regime's ceiling (alpha 0.80), not the declared "
            "support's — ceiling_absorptive reaches alpha 0.98 (AC-50)"
        )

    def test_the_individual_corners_still_reproduce(self) -> None:
        """AC-30's three hand-checked numbers, which are correct as far as they go —
        they are corners of the `mixed` regime, not of the declared support."""
        assert self._d_min((3.0, 3.0, 2.4), 0.05)["iso_min_distance_sabine_m"] == \
            pytest.approx(0.41, abs=0.005)
        assert self._d_min((12.0, 10.0, 5.0), 0.05)["iso_min_distance_sabine_m"] == \
            pytest.approx(1.29, abs=0.005)
        assert self._d_min((12.0, 10.0, 5.0), 0.80)["iso_min_distance_sabine_m"] == \
            pytest.approx(5.16, abs=0.005)

    def test_the_declared_floor_sits_near_the_bottom_of_that_range(self) -> None:
        """The claim AC-30 refuted: 1.0 m is not "inside the band" — it is near the
        bottom of a support that reaches 5.71 m by Sabine and 11.41 m by Eyring."""
        lo, hi = self._declared_support(Config.load(Path("configs/base.yaml")))["sabine"]
        assert lo < 1.0 < hi
        assert 1.0 < lo + 0.25 * (hi - lo), "1.0 m is not in the band's interior"

    def test_eyring_is_the_stricter_criterion_at_every_absorption(self) -> None:
        """-ln(1-a) > a for all a in (0, 1), so Eyring's shorter T60 always gives the
        larger d_min — by 0.5 % at alpha 0.02 and by ~2x at 0.98. Both are carried
        because that spread is the disclosure: AC-30 measured 25.4 % of id below
        d_min by Sabine against 37.2 % by Eyring."""
        for alpha in (0.02, 0.05, 0.30, 0.80, 0.98):
            room = self._d_min((12.0, 10.0, 5.0), alpha)
            assert room["iso_min_distance_eyring_m"] > \
                room["iso_min_distance_sabine_m"], alpha

    def test_the_report_records_the_realized_fraction_per_split(
        self, tmp_path: Path
    ) -> None:
        run_gen_scenes(tiny_config(), tmp_path, QUIET)
        report = json.loads(
            (tmp_path / "scenes" / "placement_report.json").read_text()
        )
        for name, entry in report.items():
            block = entry["below_iso_min_distance"]
            assert block["n_scenes"] == entry["n_scenes"], name
            for flag in ("below_iso_min_distance_sabine",
                         "below_iso_min_distance_eyring"):
                assert 0.0 <= block[flag]["fraction"] <= 1.0, f"{name}/{flag}"
                assert block[flag]["count"] <= entry["n_scenes"]
            # The floor this fraction is the shortfall OF, carried alongside it.
            assert block["declared_distance_min_m"] is not None, name
            assert entry["iso_min_distance_sabine_m"]["median"] > 0.0, name

    def test_an_uncharacterized_split_reports_no_below_d_min_number(
        self, tmp_path: Path
    ) -> None:
        """A non-enclosure has no surface-and-absorption d_min either, so it is
        excluded here for the same reason it leaves the record-length gate."""
        run_gen_scenes(_openfield_config(), tmp_path, QUIET)
        report = json.loads(
            (tmp_path / "scenes" / "placement_report.json").read_text()
        )
        block = report["test_openfield"]["below_iso_min_distance"]
        assert block["n_scenes"] == 0
        assert block["n_uncharacterized"] == 3
        assert block["below_iso_min_distance_sabine"]["fraction"] is None
        assert report["test_openfield"]["iso_min_distance_sabine_m"] is None
