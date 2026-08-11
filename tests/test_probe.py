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

from amcd.config import Config
from amcd.pipeline import Pipeline
from amcd.runtime import Verbosity

from tests.conftest import QUIET, tiny_config

#: Shows the `metrics` ladder category, which is where both probes print their
#: verdict tables. Warnings reach stderr at every level (F-24), so QUIET is enough
#: for the drop warnings — this exists only for the D0b verdict line.
SHOW_METRICS = Verbosity(save=0, show=3)


def _run_to_diagnostics(cfg: Config, run_dir: Path, verbosity: Verbosity) -> None:
    for stage in ("gen-scenes", "render", "preprocess", "diagnostics"):
        Pipeline(cfg, run_dir, verbosity).run_stage(stage)


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
