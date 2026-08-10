"""Dataset integrity across RE-RUNS of a run_dir (review pass 1 fixes).

Every test here guards a failure that is invisible in the artifacts: the run
completes, `splits.json` and the printed counts look right, and the reported
numbers are wrong. They are regression tests for defects that were REPRODUCED,
not hypothesised — see docs/review_ledger.md F-25..F-36.

The unifying rule: a run_dir must hold exactly one dataset. Anything left over
from a previous configuration is refused, never silently consumed.
"""
import json
from pathlib import Path

import pytest

from amcd.config import Config
from amcd.data.dataset import EnergyDataset
from amcd.pipeline import Pipeline
from amcd.scenes.generator import run_gen_scenes

from tests.conftest import QUIET, tiny_config


def _pipeline(cfg: Config, run_dir: Path, force: bool = False) -> Pipeline:
    return Pipeline(cfg, run_dir, QUIET, force=force)


def _through_preprocess(cfg: Config, run_dir: Path, force: bool = False) -> None:
    for stage in ("gen-scenes", "render", "preprocess"):
        _pipeline(cfg, run_dir, force=force).run_stage(stage)


class TestSplitResidue:
    """F-25 (blocker): re-preprocessing must not leave a scene in two splits.

    Reproduced end-to-end before the fix: after the `--force` rebuild that the
    fingerprint error itself recommends, 4 of 5 scenes scored as `test_id` were
    in the trainer's dataset, and the report printed test-split CIs over them.
    """

    def test_loader_membership_comes_from_the_manifest(self, tmp_path: Path) -> None:
        cfg = tiny_config(scenes={"n_id": 12})
        _through_preprocess(cfg, tmp_path)

        manifest = json.loads((tmp_path / "preprocessed" / "splits.json").read_text())
        for split in ("train", "valid"):
            declared = {s for s, sp in manifest.items() if sp == split}
            assert set(EnergyDataset(tmp_path / "preprocessed", split).scene_ids) == declared

    def test_reassignment_does_not_leak_across_splits(self, tmp_path: Path) -> None:
        """The decisive one: repin the split seed, rebuild, and no scene the
        manifest calls a test split may appear in the trainer's dataset."""
        first = tiny_config(scenes={"n_id": 12}, seeds={"master": 0, "split_assignment": 111})
        _through_preprocess(first, tmp_path)

        second = tiny_config(scenes={"n_id": 12}, seeds={"master": 0, "split_assignment": 222})
        _through_preprocess(second, tmp_path, force=True)

        manifest = json.loads((tmp_path / "preprocessed" / "splits.json").read_text())
        train_ids = set(EnergyDataset(tmp_path / "preprocessed", "train").scene_ids)
        leaked = sorted(s for s in train_ids if manifest.get(s, "").startswith("test"))
        assert not leaked, f"trained-on scenes are scored as held-out: {leaked}"

    def test_orphan_tensor_is_refused_not_ignored(self, tmp_path: Path) -> None:
        """Residue must fail loudly. Skipping it silently would hide that the
        run_dir holds two datasets."""
        cfg = tiny_config(scenes={"n_id": 12})
        _through_preprocess(cfg, tmp_path)

        train_dir = tmp_path / "preprocessed" / "train"
        sample = next(train_dir.glob("*_low.pt"))
        (train_dir / "scene_9999_low.pt").write_bytes(sample.read_bytes())

        with pytest.raises(RuntimeError, match="absent from the manifest"):
            EnergyDataset(tmp_path / "preprocessed", "train")

    def test_low_and_high_are_paired_by_scene_id(self, tmp_path: Path) -> None:
        """F-36: index-pairing two independently globbed lists mispairs every
        input with the wrong target whenever the sets differ but the counts match."""
        cfg = tiny_config(scenes={"n_id": 12})
        _through_preprocess(cfg, tmp_path)

        ds = EnergyDataset(tmp_path / "preprocessed", "train")
        for sid, low, high in zip(ds.scene_ids, ds.low_paths, ds.high_paths):
            assert low.name == f"{sid}_low.pt"
            assert high.name == f"{sid}_high.pt"


class TestOrphanSceneSpecs:
    """F-27: regenerating fewer scenes must not leave the extras on disk."""

    def test_regeneration_shrinks_the_scene_set(self, tmp_path: Path) -> None:
        run_gen_scenes(tiny_config(scenes={"n_id": 12}), tmp_path, QUIET)
        before = len(list((tmp_path / "scenes").glob("scene_*.json")))

        run_gen_scenes(tiny_config(scenes={"n_id": 4}), tmp_path, QUIET)
        after = sorted((tmp_path / "scenes").glob("scene_*.json"))

        assert len(after) < before
        report = json.loads((tmp_path / "scenes" / "placement_report.json").read_text())
        # The accounting artifact and the directory must agree — they did not
        # before, and render/preprocess glob the directory, not the report.
        assert sum(e["n_scenes"] for e in report.values()) == len(after)


class TestUpstreamChain:
    """F-26: --force must not launder a stale upstream into a valid-looking chain."""

    def test_force_downstream_cannot_hide_stale_upstream(self, tmp_path: Path) -> None:
        cfg = tiny_config(scenes={"n_id": 4})
        _pipeline(cfg, tmp_path).run_stage("gen-scenes")
        _pipeline(cfg, tmp_path).run_stage("render")

        moved = tiny_config(scenes={"n_id": 4, "margins": {"wall": 0.75}})
        # Even with --force, rendering against scene specs generated under the old
        # config must refuse: the renders would not be renders OF those scenes.
        with pytest.raises(RuntimeError, match="upstream stage 'gen-scenes'"):
            _pipeline(moved, tmp_path, force=True).run_stage("render")

    def test_downstream_without_upstream_refuses(self, tmp_path: Path) -> None:
        cfg = tiny_config(scenes={"n_id": 4})
        with pytest.raises(RuntimeError, match="has not completed"):
            _pipeline(cfg, tmp_path).run_stage("render")

    def test_split_seed_change_invalidates_preprocess(self, tmp_path: Path) -> None:
        """F-29: repinning the most leakage-critical value in the project was a
        complete no-op on an existing run_dir — splits.json kept the old
        assignment while config.yaml stamped the new seed."""
        first = tiny_config(scenes={"n_id": 12}, seeds={"master": 0, "split_assignment": 111})
        _through_preprocess(first, tmp_path)

        second = tiny_config(scenes={"n_id": 12}, seeds={"master": 0, "split_assignment": 222})
        with pytest.raises(RuntimeError, match="cached under a DIFFERENT config"):
            _pipeline(second, tmp_path).run_stage("preprocess")


class TestPredictionResidue:
    """F-37: the F-25 residue pattern, one stage downstream.

    `predictions/` was never cleared and eval GLOBS it, so a scene that moved out
    of a test split kept a prediction made by a DIFFERENT model under a DIFFERENT
    split assignment — and it was scored. Reproduced: 2 stale predictions reached
    metrics.parquet under `train`.
    """

    def test_stale_predictions_do_not_survive_a_rerun(self, tmp_path: Path) -> None:
        first = tiny_config(scenes={"n_id": 20}, seeds={"master": 0, "split_assignment": 111})
        Pipeline(first, tmp_path, QUIET).run_all()

        second = tiny_config(scenes={"n_id": 20}, seeds={"master": 0, "split_assignment": 222})
        Pipeline(second, tmp_path, QUIET, force=True).run_all()

        manifest = json.loads((tmp_path / "preprocessed" / "splits.json").read_text())
        stale = [p.stem.replace("_pred", "") for p in (tmp_path / "predictions").glob("*_pred.pt")
                 if manifest.get(p.stem.replace("_pred", "")) not in second.test_split_names]
        assert not stale, f"predictions survive for scenes no longer in a test split: {stale}"

    def test_eval_refuses_a_prediction_for_a_non_test_split(self, tmp_path: Path) -> None:
        """Defence in depth: even if a stale file appears, it must not be scored."""
        from amcd.evaluation.evaluator import run_eval

        cfg = tiny_config(scenes={"n_id": 20})
        Pipeline(cfg, tmp_path, QUIET).run_all()

        manifest = json.loads((tmp_path / "preprocessed" / "splits.json").read_text())
        train_scene = next(s for s, sp in manifest.items() if sp == "train")
        sample = next((tmp_path / "predictions").glob("*_pred.pt"))
        (tmp_path / "predictions" / f"{train_scene}_pred.pt").write_bytes(sample.read_bytes())

        with pytest.raises(RuntimeError, match="not a test split"):
            run_eval(cfg, tmp_path, QUIET)


class TestReservedSplitNames:
    """F-28 / F-38: names that collide with pipeline sentinels or directories."""

    def test_split_named_id_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="reserved"):
            tiny_config(splits={"id": {"role": "test", "count": 2,
                                       "axes": {"geometry": "corridor"}}})

    def test_split_named_carrier_is_rejected(self) -> None:
        """`carrier` is exempted from the stale-split sweep, so a split of that
        name would be permanently exempt from clearing — reinstating F-25 for
        itself (F-38)."""
        with pytest.raises(ValueError, match="reserved"):
            tiny_config(splits={"carrier": {"role": "test", "count": 2,
                                            "axes": {"geometry": "corridor"}}})


class TestOrphanHighTensor:
    """F-39: the orphan backstop must scan both suffixes, as its docstring claims."""

    def test_orphan_high_tensor_is_refused(self, tmp_path: Path) -> None:
        cfg = tiny_config(scenes={"n_id": 12})
        _through_preprocess(cfg, tmp_path)

        train_dir = tmp_path / "preprocessed" / "train"
        sample = next(train_dir.glob("*_high.pt"))
        (train_dir / "scene_9999_high.pt").write_bytes(sample.read_bytes())

        with pytest.raises(RuntimeError, match="absent from the manifest"):
            EnergyDataset(tmp_path / "preprocessed", "train")


class TestEmptySplitVisibility:
    """F-30: a declared split that receives nothing must be reported as 0."""

    def test_declared_splits_all_appear_in_counts(self, tmp_path: Path) -> None:
        cfg = tiny_config(scenes={"n_id": 12})
        _through_preprocess(cfg, tmp_path)
        meta = json.loads((tmp_path / "preprocessed" / "meta.json").read_text())
        assert set(meta["split_counts"]) == set(cfg.splits), (
            "split_counts must be keyed on the CONFIG-declared split set, so a "
            "split receiving zero scenes is visible as 0 rather than absent"
        )
