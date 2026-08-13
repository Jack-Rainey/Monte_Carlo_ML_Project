"""F-72: the D0 probes must not drop a scene — or a whole split — in silence.

F-45 fixed the probes' SPLIT enumeration: a declared split with no scenes is now
reported as 0 rather than omitted. The per-scene skips underneath it stayed silent.
Both D0a and D0b `continue` past a scene whose preprocessed tensors, carrier or
`renders/<id>/high.npy` are missing, and both then `continue` past a split whose
scenes ALL failed — so the split vanished from the artifact and D0b's `all_clear`
stayed True over a split it never measured.

Every test here constructs the FAILING population by removing inputs from a healthy
run, because that is the only state in which the defect is visible: on a complete
run the drop lists are empty and scored == attempted everywhere.

The probe's F-45 tests live in `tests/test_dataset_integrity.py`
(`TestD0bEnumeratesDeclaredSplits`). Probe coverage split across two files is a
known, temporary cost recorded as RD-83 in `docs/review_ledger.md`; consolidate
here when that row is taken up.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from amcd.config import Config
from amcd.pipeline import Pipeline
from amcd.runtime import RunContext, Verbosity

from tests.conftest import CANONICAL_DRY_RUN, QUIET, tiny_config

#: Shows the `metrics` ladder category, which is where both probes print their
#: verdict tables. Warnings reach stderr at every level (F-24), so QUIET is enough
#: for the drop warnings — this exists only for the D0b verdict line.
SHOW_METRICS = RunContext(Verbosity(save=0, show=3))


def _run_to_diagnostics(cfg: Config, run_dir: Path, ctx: RunContext) -> None:
    for stage in ("gen-scenes", "render", "preprocess", "diagnostics"):
        Pipeline(cfg, run_dir, ctx).run_stage(stage)


def _run_to_preprocess(cfg: Config, run_dir: Path) -> None:
    for stage in ("gen-scenes", "render", "preprocess"):
        Pipeline(cfg, run_dir, QUIET).run_stage(stage)


def _scene_splits(run_dir: Path) -> dict[str, str]:
    return json.loads((run_dir / "preprocessed" / "splits.json").read_text())


def _scenes_in(run_dir: Path, split: str) -> list[str]:
    return sorted(sid for sid, sp in _scene_splits(run_dir).items() if sp == split)


def _d0a(run_dir: Path) -> dict:
    return json.loads((run_dir / "diagnostics" / "d0a_gap.json").read_text())


def _d0b(run_dir: Path) -> dict:
    return json.loads((run_dir / "diagnostics" / "d0b_oracle.json").read_text())


class TestPerSceneDropsCarryAReason:
    """A surviving count with no attempted denominator reads as complete."""

    def test_a_missing_tensor_is_logged_by_both_probes(self, tmp_path: Path) -> None:
        cfg = tiny_config()
        _run_to_preprocess(cfg, tmp_path)

        victim = _scenes_in(tmp_path, "train")[0]
        n_attempted = len(_scenes_in(tmp_path, "train"))
        for suffix in ("low", "high"):
            (tmp_path / "preprocessed" / "train" / f"{victim}_{suffix}.pt").unlink()

        Pipeline(cfg, tmp_path, QUIET).run_stage("diagnostics")

        train = _d0a(tmp_path)["per_split"]["train"]
        assert train["n_attempted"] == n_attempted
        assert train["n_scenes"] == n_attempted - 1, (
            "the scored count moved but the attempted count did not follow it — "
            "headroom over the survivors reads as headroom over the split (F-72)"
        )
        assert [d["scene"] for d in train["dropped"]] == [victim]
        assert "missing" in train["dropped"][0]["reason"]

        train_b = _d0b(tmp_path)["per_split"]["train"]
        assert train_b["n_attempted"] == n_attempted
        assert train_b["n_scenes"] == n_attempted - 1
        assert [d["scene"] for d in train_b["dropped"]] == [victim]

    def test_a_missing_render_is_dropped_by_d0b_and_scored_by_d0a(
        self, tmp_path: Path
    ) -> None:
        """The two probes consume different inputs, so they drop different scenes.

        D0a needs only the preprocessed tensors; D0b additionally needs the carrier
        and the raw high-ray reference. A scene missing only the reference render is
        therefore scored by D0a and must be dropped — with that reason — by D0b.
        """
        cfg = tiny_config()
        _run_to_preprocess(cfg, tmp_path)

        scenes = _scenes_in(tmp_path, "train")
        victim = scenes[0]
        (tmp_path / "renders" / victim / "high.npy").unlink()

        Pipeline(cfg, tmp_path, QUIET).run_stage("diagnostics")

        train_a = _d0a(tmp_path)["per_split"]["train"]
        assert train_a["dropped"] == []
        assert train_a["n_scenes"] == train_a["n_attempted"] == len(scenes)

        train_b = _d0b(tmp_path)["per_split"]["train"]
        assert [d["scene"] for d in train_b["dropped"]] == [victim]
        assert "high.npy" in train_b["dropped"][0]["reason"]
        assert train_b["n_scenes"] == len(scenes) - 1
        assert train_b["n_attempted"] == len(scenes)

    def test_a_complete_run_reports_scored_equal_to_attempted(
        self, tmp_path: Path
    ) -> None:
        """The healthy case the defect hid behind: no drops, and the counts agree."""
        cfg = tiny_config()
        _run_to_diagnostics(cfg, tmp_path, QUIET)

        for artifact in (_d0a(tmp_path), _d0b(tmp_path)):
            for split, entry in artifact["per_split"].items():
                assert entry["dropped"] == [], f"{split}: {entry['dropped']}"
                assert entry["n_scenes"] == entry["n_attempted"], split


class TestAnAllFailedSplitIsUnscoredNeverPassed:
    """The split that vanished, and the verdict that cleared it anyway."""

    @staticmethod
    def _starve_a_split(cfg: Config, run_dir: Path) -> tuple[str, int]:
        """Remove every preprocessed tensor of one shift split. Returns (split, n)."""
        _run_to_preprocess(cfg, run_dir)
        split = "test_geometry_shift"
        scenes = _scenes_in(run_dir, split)
        assert scenes, "fixture is inert — the split must start with scenes"
        for sid in scenes:
            for suffix in ("low", "high"):
                (run_dir / "preprocessed" / split / f"{sid}_{suffix}.pt").unlink()
        return split, len(scenes)

    def test_d0a_records_the_split_with_its_reason(self, tmp_path: Path) -> None:
        cfg = tiny_config()
        split, n = self._starve_a_split(cfg, tmp_path)

        Pipeline(cfg, tmp_path, QUIET).run_stage("diagnostics")

        per_split = _d0a(tmp_path)["per_split"]
        assert split in per_split, (
            "a split whose scenes all failed vanished from d0a_gap.json — "
            "indistinguishable from a split that was never declared (F-72)"
        )
        entry = per_split[split]
        assert entry["n_scenes"] == 0
        assert entry["n_attempted"] == n
        assert len(entry["dropped"]) == n
        assert "unscored_reason" in entry
        assert "mean_gap_db" not in entry, (
            "an unscored split must not carry a number a reader could average"
        )

    def test_d0b_reads_indeterminate_rather_than_clearing_the_ceiling(
        self, tmp_path: Path, capsys
    ) -> None:
        cfg = tiny_config()
        split, n = self._starve_a_split(cfg, tmp_path)

        Pipeline(cfg, tmp_path, SHOW_METRICS).run_stage("diagnostics")
        printed = capsys.readouterr().out

        entry = _d0b(tmp_path)["per_split"][split]
        assert entry["n_scenes"] == 0
        assert entry["n_attempted"] == n
        assert "unscored_reason" in entry
        assert "T30" not in entry, (
            "an unmeasured split must not carry per-metric residuals"
        )

        assert "D0b verdict: INDETERMINATE" in printed, (
            f"D0b cleared a split it never measured — `all_clear` stayed True over "
            f"{split!r}, whose {n} scenes all failed to load (F-72)"
        )
        assert "CARRIER CEILING CLEARS" not in printed

    def test_the_drop_warning_names_the_split_at_every_verbosity(
        self, tmp_path: Path, capsys
    ) -> None:
        """Warnings bypass the ladder (F-24), so QUIET must still show the drops."""
        cfg = tiny_config()
        split, n = self._starve_a_split(cfg, tmp_path)

        Pipeline(cfg, tmp_path, QUIET).run_stage("diagnostics")
        warnings = capsys.readouterr().err

        assert f"{split!r}" in warnings
        assert "D0a scored 0 of" in warnings
        assert "D0b scored 0 of" in warnings


class TestThePerSplitRecordSchema:
    """RR-64: the record shape is built at six sites across both probes and was
    declared at none, so the two consumers disagreed about whether the last key was
    guaranteed — D0a indexed `unscored_reason` while D0b defended with a `.get`
    default. It is now declared once in `probe.py`'s module docstring, and both
    consumers index it. This pins the invariant that makes indexing safe."""

    def test_unscored_reason_is_present_exactly_when_nothing_was_scored(
        self, tmp_path: Path
    ) -> None:
        cfg = tiny_config()
        _run_to_preprocess(cfg, tmp_path)

        # A split scored in full, a split scored in part, and a split scored not at
        # all — all three shapes in one artifact, which a healthy run never has.
        starved = "test_geometry_shift"
        for sid in _scenes_in(tmp_path, starved):
            for suffix in ("low", "high"):
                (tmp_path / "preprocessed" / starved / f"{sid}_{suffix}.pt").unlink()
        partial = _scenes_in(tmp_path, "train")[0]
        for suffix in ("low", "high"):
            (tmp_path / "preprocessed" / "train" / f"{partial}_{suffix}.pt").unlink()

        Pipeline(cfg, tmp_path, QUIET).run_stage("diagnostics")

        seen = {"scored": 0, "unscored": 0}
        for artifact in (_d0a(tmp_path), _d0b(tmp_path)):
            for split, entry in artifact["per_split"].items():
                assert {"n_scenes", "n_attempted", "dropped"} <= set(entry), split
                assert entry["n_scenes"] <= entry["n_attempted"], split
                if entry["n_scenes"] == 0:
                    assert "unscored_reason" in entry, (
                        f"{split}: nothing was scored and no reason was recorded — "
                        f"the key D0a indexes directly (RR-64)"
                    )
                    seen["unscored"] += 1
                else:
                    assert "unscored_reason" not in entry, (
                        f"{split}: a scored split carries an unscored reason, so "
                        f"the key's presence no longer means what it says (RR-64)"
                    )
                    seen["scored"] += 1

        assert seen["unscored"] >= 2 and seen["scored"] >= 2, (
            f"fixture is inert — both shapes must occur: {seen}"
        )


class TestD0bComparesTheSameBands:
    """F-101: the D0b residual must be decided by acoustics, not by which bands
    each leg happened to keep.

    `channel_band_avg_metrics` averages ONE IR over its own surviving bands, which
    is right for a standalone IR and wrong for a comparison. Averaging each leg
    separately let oracle and reference span different band sets, and the
    difference between the sets appeared as a residual with no acoustic cause —
    measured at an identical true T60 with only different noise realizations. The
    asymmetry is directional, because the oracle sits on the noisier low-ray
    carrier and loses bands more often, so it inflates the residual D0b compares
    against a JND. The eval stage already intersects (AC-08).
    """

    FREQS = [500.0, 1000.0]

    class _Cfg:
        sample_rate = 48000
        metric_onset_rel_db = -20.0
        metric_band_resolvability_margin = 2.0
        metric_min_decay_range_db = {"T30": 45.0, "EDT": 20.0}
        metric_octave_filter = Config.load(
            Path("configs/base.yaml")
        ).metric_octave_filter

    @staticmethod
    def _decay(t60: float, seed: int, n: int = 24000) -> np.ndarray:
        t = np.arange(n) / 48000
        rng = np.random.default_rng(seed)
        return (rng.standard_normal(n) * np.exp(-6.907 * t / t60)).astype(np.float32)

    def test_both_legs_average_over_one_band_set(self) -> None:
        """The property, asserted directly: whatever is dropped is dropped from
        BOTH legs, so a surviving residual cannot be composition."""
        from amcd.diagnostics.probe import _band_intersected_pair
        from amcd.evaluation.room_acoustic import channel_per_band_metrics

        oracle = self._decay(0.045, 11)
        reference = self._decay(0.045, 12)
        oracle_vals, ref_vals, _reasons = _band_intersected_pair(
            oracle, reference, config=self._Cfg,
            iso_eval_freqs=self.FREQS, shared_trunc=None,
        )
        # A metric is scored for both legs or for neither — never one.
        for metric in ("T30", "EDT", "C50"):
            assert np.isnan(oracle_vals[metric]) == np.isnan(ref_vals[metric]), (
                f"{metric} is scored for one leg and not the other, so the D0b "
                f"residual for it would compare different band sets (F-101)"
            )

    def test_a_partial_average_says_which_bands_it_lost(self) -> None:
        """A residual over fewer bands is still reported — but never silently."""
        from amcd.diagnostics.probe import _band_intersected_pair

        _o, _r, reasons = _band_intersected_pair(
            self._decay(0.045, 11), self._decay(0.045, 12), config=self._Cfg,
            iso_eval_freqs=self.FREQS, shared_trunc=None,
        )
        assert reasons, "a dropped band left no reason"
        for reason in reasons.values():
            assert "Hz" in reason and ("partial" in reason or "no eval band" in reason)


class TestD0bEnumeratesDeclaredSplits:
    """F-45's D0b half: `sorted(set(splits.values()))` listed only splits that
    RECEIVED a scene, so a declared-but-empty split vanished from d0b_oracle.json
    while d0a_gap.json included it — and the run still printed
    `D0b verdict: CARRIER CEILING CLEARS`, a verdict over a split set that
    silently differed from the declared one. The `if not scene_ids:` branch whose
    message read "declared in config but received no scenes" was DEAD CODE."""

    def test_an_empty_declared_split_is_named_in_the_d0b_artifact(
        self, tmp_path: Path
    ) -> None:
        import json

        from amcd.config import Config
        from amcd.pipeline import Pipeline

        # Same recipe as the stats/report sibling above: split_assignment 104 with
        # n_id 6 starves test_id while train/valid survive, so the run completes and
        # the split's absence from D0b is the only thing under test.
        overlay = tmp_path / "starve.yaml"
        overlay.write_text("seeds:\n  split_assignment: 104\nscenes:\n  n_id: 6\n")
        cfg = Config.load(*CANONICAL_DRY_RUN, overlay)
        tmp_path = tmp_path / "run"
        for stage in ("gen-scenes", "render", "preprocess", "diagnostics"):
            Pipeline(cfg, tmp_path, QUIET).run_stage(stage)

        d0b = json.loads((tmp_path / "diagnostics" / "d0b_oracle.json").read_text())
        assert "test_id" in d0b["per_split"], (
            "a declared split with no scenes is absent from d0b_oracle.json — "
            "indistinguishable from a split that was never declared (F-45)"
        )
        assert d0b["per_split"]["test_id"]["n_scenes"] == 0
        assert "received no scenes" in d0b["per_split"]["test_id"]["unscored_reason"]

    def test_the_declared_split_order_is_config_order_not_alphabetical(
        self, tmp_path: Path
    ) -> None:
        import json

        from amcd.pipeline import Pipeline

        cfg = tiny_config(scenes={"n_id": 8})
        for stage in ("gen-scenes", "render", "preprocess", "diagnostics"):
            Pipeline(cfg, tmp_path, QUIET).run_stage(stage)

        d0b = json.loads((tmp_path / "diagnostics" / "d0b_oracle.json").read_text())
        declared = [s for s in cfg.splits if s in d0b["per_split"]]
        assert list(d0b["per_split"])[: len(declared)] == declared


class TestD0bDisclosesTheAbsoluteLevelMargin:
    """AC-37 (c) — how much level margin the dataset has before `min_db` starts
    injecting an energy floor into the decode.

    The residual D0b reports is measured at each scene's NATIVE level, where the
    defect is inert (worst |dT30| 0.25 % over every declared corner). That inertness
    is a property of THIS render's level convention, not of the pipeline, so the
    artifact has to say how far from inert the dataset actually is — a dataset
    clearing `encode`'s guard by 2 dB and one clearing it by 40 dB are different
    datasets and the residual alone cannot tell them apart.
    """

    def _sweep(self, tmp_path: Path, canonical: bool = False) -> dict:
        cfg = Config.load(*CANONICAL_DRY_RUN) if canonical else tiny_config()
        _run_to_diagnostics(cfg, tmp_path, QUIET)
        return _d0b(tmp_path)["per_scene"]

    def test_every_scored_scene_carries_a_sweep(self, tmp_path: Path) -> None:
        per_scene = self._sweep(tmp_path)
        assert per_scene, "D0b scored no scene, so this proves nothing"
        for sid, rec in per_scene.items():
            assert "level_sweep" in rec, f"{sid} has no level sweep"
            assert "reference_t30_s" in rec["level_sweep"], sid

    def test_the_error_grows_as_the_headroom_falls(self, tmp_path: Path) -> None:
        """The mechanism, asserted rather than assumed: the injected floor is what
        the sweep is measuring, so lowering a scene onto `min_db` must make the
        definitionally-perfect oracle worse."""
        # The CANONICAL config, not tiny: tiny's records are too short for the ISO
        # SNR bound to admit a T30 at all, so every cell is NaN and the assertion
        # below would compare nothing against nothing.
        per_scene = self._sweep(tmp_path, canonical=True)
        rec = next(
            r["level_sweep"] for r in per_scene.values()
            if all(
                v["t30_rel_error"] == v["t30_rel_error"]
                for g, v in r["level_sweep"].items() if g != "reference_t30_s"
            )
        )
        cells = sorted(
            ((float(g), v) for g, v in rec.items() if g != "reference_t30_s"),
            reverse=True,
        )
        headrooms = [v["headroom_db"] for _g, v in cells]
        assert headrooms == sorted(headrooms, reverse=True), (
            f"headroom does not fall with the gain: {headrooms}"
        )
        assert cells[-1][1]["t30_rel_error"] > cells[0][1]["t30_rel_error"], (
            f"the quietest cell is no worse than the loudest, so this sweep is not "
            f"measuring the min_db floor at all: {cells}"
        )

    def test_a_gain_that_cannot_be_scored_is_not_counted_as_passing(
        self, tmp_path: Path
    ) -> None:
        """NaN is not a pass. A cell whose oracle could not be scored says so —
        otherwise an unmeasurable level would read as a cleared one."""
        per_scene = self._sweep(tmp_path)
        for sid, rec in per_scene.items():
            for gain, cell in rec["level_sweep"].items():
                if gain == "reference_t30_s":
                    continue
                err = cell["t30_rel_error"]
                if err != err:                       # NaN
                    assert cell["breaches_jnd"] is None, (sid, gain, cell)
                else:
                    assert cell["breaches_jnd"] in (True, False), (sid, gain, cell)
