"""Validity of the scene-characterization estimates, and of the record they assume.

Ledger rows AC-21 (diffuse-field validity) and AC-22 (record length vs T60).

Neither changes a formula. Both are about a number being reported without the one
fact a reader needs to interpret it:

  * AC-21 — `placement_report.json` reports Sabine/Eyring T60, critical distance
    and DRR for `test_material_shift`, a split whose alpha median is 0.894. The
    diffuse-field premise behind all four has failed for 100 % of it. A +4.9 dB DRR
    from a model outside its domain is an extrapolation, and nothing said so.
  * AC-22 — nothing checked that `ir_duration` could support the decay the declared
    ranges admit. A T30 fitted over a truncated record measures the truncation.
"""
import json
from pathlib import Path

import pytest

from amcd.acoustics import critical_distance
from amcd.config import Config
from amcd.scenes.generator import _room_acoustics, run_gen_scenes

from tests.conftest import QUIET, tiny_config

_BASE = Path("configs/base.yaml")
_RI = (Path("configs/base.yaml"), Path("configs/research_i.yaml"))


class TestDeclaredSupportCorner:
    """AC-22's disclosure half: computed, recorded, and never used as a gate."""

    def test_base_corner_is_the_largest_shoebox_at_lowest_absorption(self) -> None:
        corner = Config.load(_BASE).worst_case_t60()
        assert corner["geometry_family"] == "shoebox"
        assert corner["dims_m"] == [12.0, 10.0, 5.0]
        assert corner["absorption"] == 0.05
        # 0.161 * 600 / (0.05 * 460)
        assert corner["t60_sabine_s"] == pytest.approx(4.20, abs=0.01)

    def test_base_record_covers_its_own_support(self) -> None:
        """The user's decision, enforced rather than asserted in a comment."""
        corner = Config.load(_BASE).worst_case_t60()
        assert corner["covered_by_record"] is True
        assert corner["ir_duration_s"] >= corner["t60_sabine_s"]

    def test_research_i_records_that_it_does_NOT_cover_its_support(self) -> None:
        """RI's 3.0 s is RI-pinned. The deviation is disclosed, not fixed."""
        corner = Config.load(*_RI).worst_case_t60()
        assert corner["ir_duration_s"] == 3.0
        assert corner["t60_sabine_s"] == pytest.approx(4.09, abs=0.01)
        assert corner["covered_by_record"] is False

    def test_the_corner_is_stamped_into_run_provenance(self, tmp_path: Path) -> None:
        import yaml

        Config.load(_BASE).stamp(tmp_path)
        resolved = yaml.safe_load((tmp_path / "resolved.yaml").read_text())
        assert resolved["worst_case_t60"]["t60_sabine_s"] == pytest.approx(4.20, abs=0.01)

    def test_an_uncovered_corner_alone_does_not_fail_a_config(self) -> None:
        """RD-56: the corner is two independent extremes with ~0 draw probability.
        Gating on it would reject configs whose realized scenes are all fine."""
        Config.load(*_RI)  # loads despite covered_by_record False


class TestRealizedRecordLengthGate:
    """AC-22's gate half: the population the metrics are actually computed over."""

    def test_base_has_no_scene_over_its_record(self, tmp_path: Path) -> None:
        cfg = Config.load(_BASE)
        assert cfg.scenes.max_t60_over_ir_duration_frac == 0.0
        run_gen_scenes(cfg, tmp_path, QUIET)
        report = json.loads((tmp_path / "scenes" / "placement_report.json").read_text())
        for split, entry in report.items():
            over = entry["t60_over_ir_duration"]["t60_exceeds_ir_duration"]
            assert over["count"] == 0, f"{split}: {over}"

    def test_research_i_discloses_its_realized_over_limit_scenes(
        self, tmp_path: Path
    ) -> None:
        cfg = Config.load(*_RI)
        run_gen_scenes(cfg, tmp_path, QUIET)
        report = json.loads((tmp_path / "scenes" / "placement_report.json").read_text())
        over = sum(e["t60_over_ir_duration"]["t60_exceeds_ir_duration"]["count"]
                   for e in report.values())
        total = sum(e["n_scenes"] for e in report.values())
        assert total == 720
        # Declared tolerance is 0.01, set just above the measured 4/720 = 0.56 %.
        assert 0 < over / total <= cfg.scenes.max_t60_over_ir_duration_frac

    def test_exceeding_the_declared_tolerance_fails_loudly(self, tmp_path: Path) -> None:
        """A 0.1 s record against base's geometry: every scene is over."""
        cfg = tiny_config(scenes={"max_t60_over_ir_duration_frac": 0.0})
        with pytest.raises(ValueError, match="max_t60_over_ir_duration_frac"):
            run_gen_scenes(cfg, tmp_path, QUIET)

    def test_the_failure_names_the_corner_and_the_offending_splits(
        self, tmp_path: Path
    ) -> None:
        cfg = tiny_config(scenes={"max_t60_over_ir_duration_frac": 0.0})
        with pytest.raises(ValueError) as exc:
            run_gen_scenes(cfg, tmp_path, QUIET)
        message = str(exc.value)
        assert "shoebox" in message and "scenes" in message
        assert "ir_duration is 0.1 s" in message

    def test_the_gate_is_overall_not_per_split(self) -> None:
        """A single over-limit scene in a 30-scene split is 3.3 %. Gating per split
        would force a tolerance that then also permits 16 of 500 in train, so the
        smallest split would set the standard for every other one.
        """
        from amcd.scenes.generator import _disclose_and_gate_record_length

        def report(train_over: int, shift_over: int) -> dict:
            return {
                "train": {"n_scenes": 500, "t60_over_ir_duration": {
                    "t60_exceeds_ir_duration": {"count": train_over}}},
                "test_placement_shift": {"n_scenes": 30, "t60_over_ir_duration": {
                    "t60_exceeds_ir_duration": {"count": shift_over}}},
            }

        cfg = Config.load(*_RI)  # tolerance 0.01 over 530 scenes -> at most 5
        # 1 of 30 in the small split is 3.3 % per-split but 0.19 % overall: allowed.
        _disclose_and_gate_record_length(cfg, report(0, 1), QUIET)
        # 16 of 500 is 3.2 % per-split — the rate a per-split gate would have to
        # permit — but 3.0 % overall: refused.
        with pytest.raises(ValueError, match="16 of 530"):
            _disclose_and_gate_record_length(cfg, report(16, 0), QUIET)

    def test_a_plumbing_overlay_declares_that_it_claims_no_fidelity(self) -> None:
        """test_tiny/dry_run declare 1.0 — stated, not silently exempted."""
        assert tiny_config().scenes.max_t60_over_ir_duration_frac == 1.0


class TestDiffuseFieldValidityFlags:
    """AC-21: no formula changes, no scene dropped — the estimates are LABELLED."""

    @staticmethod
    def _acoustics(alpha: float, dims=(10.0, 8.0, 3.5), distance=3.0, **kw):
        return _room_acoustics(
            dims, alpha, distance,
            alpha_limit=kw.get("alpha_limit", 0.3),
            ir_duration_s=kw.get("ir_duration_s", 3.0),
            characterization=kw.get("characterization", "sabine"),
        )

    def test_alpha_above_the_declared_limit_is_flagged(self) -> None:
        assert self._acoustics(0.2)["alpha_above_diffuse_limit"] is False
        assert self._acoustics(0.9)["alpha_above_diffuse_limit"] is True

    def test_the_limit_comes_from_config_not_a_literal(self) -> None:
        assert self._acoustics(0.4, alpha_limit=0.5)["alpha_above_diffuse_limit"] is False
        assert self._acoustics(0.4, alpha_limit=0.3)["alpha_above_diffuse_limit"] is True

    def test_sabine_eyring_ratio_grows_with_absorption(self) -> None:
        """The assumption-free readout of how far the model is being stretched."""
        low = self._acoustics(0.1)["sabine_eyring_ratio"]
        high = self._acoustics(0.95)["sabine_eyring_ratio"]
        assert low == pytest.approx(1.05, abs=0.02)
        assert high > 3.0

    def test_critical_distance_exceeding_the_room_is_flagged(self) -> None:
        """r_c > the room means the reverberant field the DRR divides by does not
        exist inside it — the DRR is an extrapolation, and now says so."""
        assert self._acoustics(0.05)["rc_exceeds_max_dim"] is False
        assert self._acoustics(0.98, dims=(4.0, 3.0, 2.5))["rc_exceeds_max_dim"] is True

    def test_flags_never_change_the_reported_estimates(self) -> None:
        """Labelled, not suppressed and not recomputed."""
        strict = self._acoustics(0.9, alpha_limit=0.3)
        loose = self._acoustics(0.9, alpha_limit=0.95)
        for key in ("t60_sabine_s", "t60_eyring_s", "critical_distance_m", "drr_db"):
            assert strict[key] == loose[key]
        assert strict["alpha_above_diffuse_limit"] != loose["alpha_above_diffuse_limit"]


class TestReceiverInsideCriticalDistance:
    """AC-29: the strictest per-scene condition, already computed but never flagged.

    Inside r_c the receiver sits in the DIRECT field, so the diffuse-field DRR
    being reported has no reverberant field to divide by. `d_over_rc` was already
    summarized as a VALUE while the per-split validity summary omitted it, and the
    shipped flags under-reported by ~2.6x on the split they were built for
    (test_material_shift: 92.5 % vs `rc_exceeds_max_dim`'s 35.0 %).
    """

    @staticmethod
    def _at(d_over_rc: float) -> dict:
        dims, alpha = (10.0, 8.0, 3.5), 0.2
        surface = 2.0 * (10.0 * 8.0 + 8.0 * 3.5 + 10.0 * 3.5)
        r_c = critical_distance(surface, alpha)
        return _room_acoustics(
            dims, alpha, d_over_rc * r_c,
            alpha_limit=0.3, ir_duration_s=3.0, characterization="sabine",
        )

    def test_half_the_critical_distance_flags(self) -> None:
        assert self._at(0.5)["receiver_inside_critical_distance"] is True

    def test_twice_the_critical_distance_does_not(self) -> None:
        assert self._at(2.0)["receiver_inside_critical_distance"] is False

    def test_the_flag_agrees_with_the_d_over_rc_it_already_reported(self) -> None:
        """No new formula — the flag reads the quantity that was already there."""
        for d_over_rc in (0.25, 0.5, 0.99, 1.01, 2.0, 4.0):
            room = self._at(d_over_rc)
            assert room["receiver_inside_critical_distance"] == (room["d_over_rc"] < 1.0)

    def test_the_0_db_drr_crossing_is_the_critical_distance(self) -> None:
        """r_c is defined as where direct and reverberant are equal, so the two
        quantities must agree by construction, not approximately."""
        assert self._at(1.0)["drr_db"] == pytest.approx(0.0, abs=1e-9)


class TestNonEnclosureGeometryIsNotCharacterized:
    """RD-64: the closed-box spine assumed a property that only happens to hold
    today. The roadmap's outdoor / partially-open scenes (paper §6) would have been
    admitted with meaningless Sabine numbers in the canonical report."""

    def test_characterization_has_no_default(self) -> None:
        from amcd.config import GeometryFamily

        assert GeometryFamily.model_fields["characterization"].is_required()

    def test_an_unknown_characterization_is_rejected_at_config_load(self) -> None:
        from pydantic import ValidationError

        from amcd.config import GeometryFamily

        with pytest.raises(ValidationError, match="characterization"):
            GeometryFamily(dims=[[3, 4], [3, 4], [2, 3]], characterization="outdoor")

    def test_a_non_enclosure_gets_a_reason_not_a_number(self) -> None:
        room = _room_acoustics(
            (10.0, 8.0, 3.5), 0.2, 3.0,
            alpha_limit=0.3, ir_duration_s=3.0, characterization="none",
        )
        for key in ("t60_sabine_s", "critical_distance_m", "drr_db", "d_over_rc"):
            assert key not in room, (
                f"{key} was emitted for a non-enclosure — a closed-box number in a "
                f"canonical artifact is exactly what RD-64 is about"
            )
        assert "not a closed enclosure" in room["uncharacterized_reason"]

    def test_the_worst_case_corner_skips_and_names_it(self) -> None:
        cfg = tiny_config(scenes={"geometry_families": {
            "shoebox": {"dims": [[3, 12], [3, 10], [2.4, 5]], "characterization": "sabine"},
            "courtyard": {"dims": [[8, 20], [8, 20], [3, 6]], "characterization": "none"},
        }})
        worst = cfg.worst_case_t60()
        assert worst["geometry_family"] == "shoebox"
        assert worst["skipped_families"] == ["courtyard"]


class TestValidityReachesTheReport:
    """AC-21's numbers, on the config the E1 write-up will actually characterize."""

    @pytest.fixture(scope="class")
    @classmethod
    def ri_report(cls, tmp_path_factory) -> dict:
        run_dir = tmp_path_factory.mktemp("ri_validity")
        run_gen_scenes(Config.load(*_RI), run_dir, QUIET)
        return json.loads((run_dir / "scenes" / "placement_report.json").read_text())

    def test_material_shift_is_entirely_outside_the_diffuse_domain(
        self, ri_report: dict
    ) -> None:
        v = ri_report["test_material_shift"]["diffuse_field_validity"]
        assert v["alpha_above_diffuse_limit"]["fraction"] == 1.0
        assert 0.15 < v["rc_exceeds_max_dim"]["fraction"] < 0.30

    def test_the_id_splits_are_marginal_not_clean(self, ri_report: dict) -> None:
        """Recorded because it is easy to read the flags as a shift-split-only
        problem; two thirds of TRAIN is above the limit too."""
        v = ri_report["train"]["diffuse_field_validity"]
        assert 0.6 < v["alpha_above_diffuse_limit"]["fraction"] < 0.75
        assert v["rc_exceeds_max_dim"]["fraction"] == 0.0

    def test_the_ratio_separates_the_splits(self, ri_report: dict) -> None:
        assert ri_report["train"]["sabine_eyring_ratio"]["median"] < 1.4
        assert ri_report["test_material_shift"]["sabine_eyring_ratio"]["median"] > 2.4

    def test_counts_are_reported_not_just_a_boolean(self, ri_report: dict) -> None:
        """'100 % of the split' and 'one scene' are different disclosures."""
        v = ri_report["test_material_shift"]["diffuse_field_validity"]
        assert v["n_scenes"] == 40
        assert v["alpha_above_diffuse_limit"]["count"] == 40
        assert v["alpha_limit"] == Config.load(*_RI).scenes.diffuse_field_alpha_limit
