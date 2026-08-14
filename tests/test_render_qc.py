"""Render-stage QC: the four Research I admission criteria.

QC decides which rendered scenes are ADMITTED to the dataset, so the risks are
the ones that attach to any admission rule: a criterion that never runs and reads
as a pass, a failure that does not reach the artifact recording it, and — since a
failing scene is excluded rather than fatal (RI §B.4) — attrition that grows
without bound until the dataset is a selected subset nobody chose.

The reuse half of the same design is pinned in
`tests/test_stage_cache.py::TestPerSceneRenderReuse`.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from amcd.pipeline import Pipeline
from amcd.simulators import qc

from tests.conftest import QUIET, tiny_config


def _score(**over):
    """One scene scored against the tiny config's declared criteria."""
    cfg = tiny_config()
    n_ch, n_s = cfg.n_channels, cfg.n_samples
    loud = np.zeros((n_ch, n_s), dtype=np.float32)
    loud[0, 10] = 1.0
    args = dict(
        legs={"low": loud, "high": loud.copy()},
        ray_budgets={"low": cfg.low_ray_budget, "high": cfg.high_ray_budget},
        path_files={"low": None, "high": None},
        path_row_counts={"low": None, "high": None},
        sample_rate=cfg.sample_rate,
        onset_rel_db=cfg.metric_onset_rel_db,
        # Geometry puts both legs' direct arrival at sample 10, where the fixture
        # writes it.
        expected_onset_samples={"low": 10, "high": 10},
        onset_tolerance_samples=int(round(
            cfg.metric_onset_tolerance_ms / 1000.0 * cfg.sample_rate
        )),
        onset_mismatch_tolerance_ms=cfg.onset_mismatch_tolerance_ms,
        min_energy_db=cfg.min_energy_db,
        min_energy_reference=cfg.min_energy_reference,
        max_path_file_mb=cfg.max_path_file_mb,
        require_non_empty_path_file=cfg.require_non_empty_path_file,
    )
    args.update(over)
    return qc.score_scene("scene_0000", **args)


def _by(records, criterion, leg=None):
    return [r for r in records
            if r.criterion == criterion and (leg is None or r.leg == leg)]


class TestEveryCriterionIsScoredOrExplicitlySkipped:
    """A criterion that did not run must not read as one that passed."""

    def test_all_four_criteria_appear_for_a_clean_scene(self) -> None:
        """Plus `w_channel_energy_db`, which is recorded and never gates."""
        records = _score()
        gating = {r.criterion for r in records if r.gating}
        assert gating == {
            "onset_mismatch_ms", "min_energy_db",
            "path_file_mb", "non_empty_path_file",
        }
        disclosures = [r for r in records if not r.gating]
        assert {r.criterion for r in disclosures} == {"total_energy_db"}
        assert all(not r.scored for r in disclosures), (
            "the W-channel energy must be recorded, never gating: the floor RI "
            "declares is over the all-channel total"
        )

    def test_the_pair_level_criterion_is_scored_once_and_carries_no_budget(self) -> None:
        """Onset mismatch is a property of the PAIR, so a per-leg row would
        double-count it and a ray budget would name only half of what it read."""
        rows = _by(_score(), "onset_mismatch_ms")
        assert len(rows) == 1
        assert rows[0].leg == "pair" and rows[0].ray_budget is None

    def test_per_leg_criteria_carry_their_own_leg_and_budget(self) -> None:
        """`min_energy_db` is budget-dependent by construction — fewer diffuse
        rays carry less energy — so a row that cannot say which budget it scored
        conflates the legs today and the swept values at E4."""
        cfg = tiny_config()
        rows = _by(_score(), "min_energy_db")
        assert {(r.leg, r.ray_budget) for r in rows} == {
            ("low", cfg.low_ray_budget), ("high", cfg.high_ray_budget),
        }

    def test_a_backend_without_paths_records_a_skip_not_a_pass(self) -> None:
        """Keyed on whether paths were EXPORTED, never on which backend it is —
        so a backend with no path export says so instead of failing every scene.

        A skip is `passed=None`: neither True nor False. `True` would let a
        criterion nobody measured inflate a "criteria passed" count, which is the
        attempted-vs-scored gap this project does not allow anywhere else.
        """
        for criterion in ("path_file_mb", "non_empty_path_file"):
            rows = _by(_score(), criterion)
            assert rows and all(r.skipped_reason for r in rows), criterion
            assert all("exported no retained paths" in r.skipped_reason for r in rows)
            assert all(r.passed is None and not r.scored for r in rows), criterion
            assert not any(r.failed for r in rows), criterion

    def test_the_onset_criterion_is_skipped_when_geometry_is_unavailable(self) -> None:
        """The bare detector is not evidence either way, so it is not used.

        `find_onset` thresholds below each leg's GLOBAL peak, so a small carrier
        draw at the direct arrival puts it under a bar its own peak set — and the
        miss rate rises with distance, which is `test_placement_shift`'s own axis.
        A criterion that cannot be adjudicated is recorded as unrun.
        """
        rows = _by(_score(expected_onset_samples={"low": 10, "high": None}),
                   "onset_mismatch_ms")
        assert len(rows) == 1
        assert rows[0].passed is None and not rows[0].scored
        assert "geometry cannot adjudicate" in rows[0].skipped_reason

    def test_a_detector_miss_does_not_refuse_a_correctly_aligned_pair(self) -> None:
        """The bare detector latches onto pre-arrival energy; geometry overrules
        it, and the adjudication is RECORDED rather than swallowed.

        Unadjudicated this is a batch refusal — `run_render` raises on any QC
        failure — for a detector artifact whose rate rises with distance, i.e.
        correlated with the placement split's own axis.
        """
        cfg = tiny_config()
        base = np.zeros((cfg.n_channels, cfg.n_samples), dtype=np.float32)
        # Energy well BEFORE the true arrival, further out than the adjudication
        # tolerance, so the detector's first crossing is not the arrival.
        early = base.copy()
        early[0, 100] = 0.5
        early[0, 200] = 1.0
        clean = base.copy()
        clean[0, 200] = 1.0
        row = _by(
            _score(legs={"low": early, "high": clean},
                   expected_onset_samples={"low": 200, "high": 200}),
            "onset_mismatch_ms",
        )[0]
        assert row.scored and row.passed, (
            "geometry puts both legs' arrival at sample 200, so the pair is aligned "
            "and must not be refused for a detector artifact"
        )
        assert row.adjudication and "geometry puts the direct arrival at" in row.adjudication


class TestEachCriterionActuallyBites:
    """A guard nothing can trip is not a guard."""

    def test_the_floor_reads_the_w_channel_not_the_all_channel_total(self) -> None:
        """The gate must bound the quantity every reported ISO metric reads.

        An IR whose W channel is silent while its directional channels carry
        ample energy produces NaN for every reported metric, so admitting it on
        an all-channel total would admit a scene that cannot be scored.
        """
        cfg = tiny_config()
        ir = np.zeros((cfg.n_channels, cfg.n_samples), dtype=np.float32)
        ir[1:, 10] = 1.0                      # degree 1..3 loud, W silent
        rows = _by(_score(legs={"low": ir, "high": ir.copy()}), "min_energy_db", "low")
        assert len(rows) == 1 and rows[0].failed, (
            "a scene with no W-channel energy was admitted; every reported ISO "
            "metric would be NaN on it"
        )
        total = _by(_score(legs={"low": ir, "high": ir.copy()}), "total_energy_db", "low")
        assert total and not total[0].gating and total[0].measured > rows[0].measured

    def test_a_silent_leg_fails_the_energy_floor(self) -> None:
        cfg = tiny_config()
        silent = np.zeros((cfg.n_channels, cfg.n_samples), dtype=np.float32)
        loud = silent.copy()
        loud[0, 10] = 1.0
        rows = _by(_score(legs={"low": silent, "high": loud}), "min_energy_db", "low")
        assert len(rows) == 1 and rows[0].failed
        assert rows[0].measured == float("-inf"), (
            "a silent leg must reach the failure table as a value, not as an error"
        )

    def test_a_pair_whose_legs_are_of_different_rooms_fails(self) -> None:
        """The defect the criterion is for: two legs that are not the same scene.

        Both legs are internally consistent — each leg's arrival is exactly where
        its OWN provenance says — so nothing per-leg is wrong. What is wrong is
        that the two disagree, which is only visible when the expected onset is
        read per leg.
        """
        cfg = tiny_config()
        base = np.zeros((cfg.n_channels, cfg.n_samples), dtype=np.float32)
        offset = 10 + int(cfg.sample_rate * cfg.onset_mismatch_tolerance_ms / 1000.0) + 50
        low = base.copy()
        low[0, 10] = 1.0
        high = base.copy()
        high[0, offset] = 1.0
        row = _by(
            _score(legs={"low": low, "high": high},
                   expected_onset_samples={"low": 10, "high": offset}),
            "onset_mismatch_ms",
        )[0]
        assert row.failed
        assert row.measured > cfg.onset_mismatch_tolerance_ms

    def test_an_oversized_path_file_fails_its_leg(self, tmp_path: Path) -> None:
        big = tmp_path / "paths_low.parquet"
        big.write_bytes(b"\0" * (2 * 1024 * 1024))
        rows = _by(
            _score(path_files={"low": big, "high": None},
                   path_row_counts={"low": 7, "high": None},
                   max_path_file_mb=1.0),
            "path_file_mb", "low",
        )
        assert len(rows) == 1 and rows[0].failed

    def test_an_empty_path_file_fails_only_when_the_switch_demands_it(
        self, tmp_path: Path
    ) -> None:
        empty = tmp_path / "paths_low.parquet"
        empty.write_bytes(b"")
        for require, expect_pass in ((True, False), (False, True)):
            rows = _by(
                _score(path_files={"low": empty, "high": None},
                       path_row_counts={"low": 0, "high": None},
                       require_non_empty_path_file=require),
                "non_empty_path_file", "low",
            )
            assert rows[0].passed is expect_pass, require


class TestTheStageExcludesOffendersAndBoundsTheAttrition:
    """RI §B.4 excludes failing examples from the dataset rather than discarding
    the batch, so one bad scene in 720 must not cost a 14-hour render. Exclusion
    has no natural floor, though, so it is bounded: losing everything is an error,
    not a small dataset."""

    def _run(self, cfg, run_dir: Path):
        Pipeline(cfg, run_dir, QUIET).run_stage("gen-scenes")
        return Pipeline(cfg, run_dir, QUIET).run_stage("render")

    def test_a_clean_batch_records_no_failure_but_does_record_its_skips(
        self, tmp_path: Path
    ) -> None:
        """The table is written even when nothing failed, so "no failures" is a
        recorded fact rather than a missing file — and the criteria that were
        never scored are in it, because a reader judging an admission decision
        needs to know which criteria did not run."""
        self._run(tiny_config(scenes={"n_id": 4}), tmp_path)
        rows = list(csv.DictReader(open(tmp_path / "renders" / "qc_failures.csv")))
        assert all(r["passed"] != "False" for r in rows), "a clean batch failed a criterion"
        # dry_run exports no retained paths, so both path criteria are unrun.
        assert rows and all(r["skipped_reason"] for r in rows)
        assert {r["criterion"] for r in rows} == {
            "path_file_mb", "non_empty_path_file", "total_energy_db",
        }

    def test_a_clean_batch_admits_every_scene(self, tmp_path: Path) -> None:
        """The manifest is what downstream reads as the dataset, so a batch with
        nothing to exclude must list every scene in it."""
        self._run(tiny_config(scenes={"n_id": 4}), tmp_path)
        manifest = json.loads((tmp_path / "renders" / "manifest.json").read_text())
        declared = sorted(p.stem for p in (tmp_path / "scenes").glob("scene_*.json"))
        assert manifest["admitted"] == declared
        assert manifest["excluded"] == []
        assert manifest["generated"] == len(declared)

    def test_one_failing_scene_is_excluded_and_the_rest_are_admitted(
        self, tmp_path: Path
    ) -> None:
        """The whole point of F-308: under batch refusal a single scene failing at
        RI's own thresholds stopped the reproduction, and the only recourse was to
        move a threshold — abandoning the faithfulness `research_i.yaml` exists
        for."""
        from amcd.registry import simulator_registry

        victim = "scene_0002"

        class _SilencesOneScene(simulator_registry.get("dry_run")):
            def render(self, scene, ray_budget):
                result = super().render(scene, ray_budget)
                if scene.scene_id == victim:
                    result.ir = np.zeros_like(result.ir)
                return result

        simulator_registry.register("silences_one")(_SilencesOneScene)
        try:
            base = tiny_config(scenes={"n_id": 6})
            cfg = tiny_config(
                scenes={"n_id": 6},
                simulator={"name": "silences_one", "params": base.simulator.params},
                max_excluded_frac=1.0,
            )
            self._run(cfg, tmp_path)
            manifest = json.loads((tmp_path / "renders" / "manifest.json").read_text())
            assert victim not in manifest["admitted"]
            assert len(manifest["admitted"]) == manifest["generated"] - 1
            entry = next(e for e in manifest["excluded"] if e["scene_id"] == victim)
            assert entry["category"] == "qc_failed"
            assert any(c["criterion"] == "min_energy_db" for c in entry["criteria"])
        finally:
            simulator_registry._entries.pop("silences_one", None)

    def test_an_excluded_scene_keeps_its_artifacts_on_disk(
        self, tmp_path: Path
    ) -> None:
        """Which is why the manifest has to be the authority on membership rather
        than a directory listing: the artifacts are what make the exclusion
        re-derivable, and re-scoring after a threshold change costs seconds."""
        cfg = tiny_config(scenes={"n_id": 4}, min_energy_db=1e6, max_excluded_frac=1.0)
        self._run(cfg, tmp_path)
        declared = {p.stem for p in (tmp_path / "scenes").glob("scene_*.json")}
        rendered = {p.name for p in (tmp_path / "renders").iterdir() if p.is_dir()}
        assert rendered == declared
        manifest = json.loads((tmp_path / "renders" / "manifest.json").read_text())
        assert manifest["admitted"] == []
        assert {e["scene_id"] for e in manifest["excluded"]} == declared

    def test_losing_the_whole_batch_is_an_error_not_a_small_dataset(
        self, tmp_path: Path
    ) -> None:
        """Per-example exclusion has no floor. A backend broken in a way that
        survives the contract checks would otherwise excludeevery scene one at a
        time and hand back a dataset that still trains and still reports."""
        cfg = tiny_config(scenes={"n_id": 4}, min_energy_db=1e6)  # nothing can pass
        with pytest.raises(ValueError, match="more of the batch than the dataset"):
            self._run(cfg, tmp_path)

    def test_a_batch_that_breached_its_bound_writes_no_sentinel(
        self, tmp_path: Path
    ) -> None:
        """Otherwise the next run reports `[skip] (cached)` over a dataset the
        stage itself refused."""
        from amcd.pipeline import _sentinel

        cfg = tiny_config(scenes={"n_id": 4}, min_energy_db=1e6)
        with pytest.raises(ValueError):
            self._run(cfg, tmp_path)
        assert not _sentinel(tmp_path, "render").exists()

    def test_the_per_scene_record_is_diagnostics_gated_and_the_failures_are_not(
        self, tmp_path: Path
    ) -> None:
        """The failure table is the EVIDENCE for the stage's refusal, so it is
        canonical; the pass-and-fail record for every scene is observability."""
        from amcd.runtime import RunContext, Verbosity

        cfg = tiny_config(scenes={"n_id": 4})
        quiet = RunContext(Verbosity(save=1, show=0))
        Pipeline(cfg, tmp_path, quiet).run_stage("gen-scenes")
        Pipeline(cfg, tmp_path, quiet).run_stage("render")
        assert (tmp_path / "renders" / "qc_failures.csv").exists()
        assert not (tmp_path / "renders" / "qc_record.csv").exists()

        loud = RunContext(Verbosity(save=4, show=0))
        Pipeline(cfg, tmp_path, loud, force=True).run_stage("render")
        record = list(csv.DictReader(open(tmp_path / "renders" / "qc_record.csv")))
        assert record and {r["scene_id"] for r in record} == {
            p.name for p in (tmp_path / "renders").iterdir() if p.is_dir()
        }
