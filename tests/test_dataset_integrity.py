"""Dataset integrity across RE-RUNS of a run_dir (review pass 1 fixes).

Every test here guards a failure that is invisible in the artifacts: the run
completes, `splits.json` and the printed counts look right, and the reported
numbers are wrong. They are regression tests for defects that were REPRODUCED,
not hypothesised.

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
from amcd.simulators.base import admitted_digest

from tests.conftest import CANONICAL_DRY_RUN, QUIET, tiny_config


def _pipeline(cfg: Config, run_dir: Path, force: bool = False) -> Pipeline:
    return Pipeline(cfg, run_dir, QUIET, force=force)


def _through_preprocess(cfg: Config, run_dir: Path, force: bool = False) -> None:
    for stage in ("gen-scenes", "render", "preprocess"):
        _pipeline(cfg, run_dir, force=force).run_stage(stage)


class TestSplitResidue:
    """Re-preprocessing must not leave a scene in two splits.

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
        """Index-pairing two independently globbed lists mispairs every
        input with the wrong target whenever the sets differ but the counts match."""
        cfg = tiny_config(scenes={"n_id": 12})
        _through_preprocess(cfg, tmp_path)

        ds = EnergyDataset(tmp_path / "preprocessed", "train")
        for sid, low, high in zip(ds.scene_ids, ds.low_paths, ds.high_paths):
            assert low.name == f"{sid}_low.pt"
            assert high.name == f"{sid}_high.pt"


class TestOrphanSceneSpecs:
    """Regenerating fewer scenes must not leave the extras on disk."""

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
    """--force must not launder a stale upstream into a valid-looking chain."""

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
        """Repinning the most leakage-critical value in the project was a
        complete no-op on an existing run_dir — splits.json kept the old
        assignment while config.yaml stamped the new seed."""
        first = tiny_config(scenes={"n_id": 12}, seeds={"master": 0, "split_assignment": 111})
        _through_preprocess(first, tmp_path)

        second = tiny_config(scenes={"n_id": 12}, seeds={"master": 0, "split_assignment": 222})
        with pytest.raises(RuntimeError, match="cached under a DIFFERENT config"):
            _pipeline(second, tmp_path).run_stage("preprocess")


class TestPredictionResidue:
    """The residue pattern, one stage downstream.

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
    """Names that collide with pipeline sentinels or directories."""

    def test_split_named_id_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="reserved"):
            tiny_config(splits={"id": {"role": "test", "count": 2,
                                       "axes": {"geometry": "corridor"}}})

    def test_split_named_carrier_is_rejected(self) -> None:
        """`carrier` is exempted from the stale-split sweep, so a split of that
        name would be permanently exempt from clearing — reinstating the leak for
        itself."""
        with pytest.raises(ValueError, match="reserved"):
            tiny_config(splits={"carrier": {"role": "test", "count": 2,
                                            "axes": {"geometry": "corridor"}}})


class TestOrphanHighTensor:
    """The orphan backstop must scan both suffixes, as its docstring claims."""

    def test_orphan_high_tensor_is_refused(self, tmp_path: Path) -> None:
        cfg = tiny_config(scenes={"n_id": 12})
        _through_preprocess(cfg, tmp_path)

        train_dir = tmp_path / "preprocessed" / "train"
        sample = next(train_dir.glob("*_high.pt"))
        (train_dir / "scene_9999_high.pt").write_bytes(sample.read_bytes())

        with pytest.raises(RuntimeError, match="absent from the manifest"):
            EnergyDataset(tmp_path / "preprocessed", "train")


class TestEmptySplitVisibility:
    """A declared split that receives nothing must be reported as 0."""

    def test_declared_splits_all_appear_in_counts(self, tmp_path: Path) -> None:
        cfg = tiny_config(scenes={"n_id": 12})
        _through_preprocess(cfg, tmp_path)
        meta = json.loads((tmp_path / "preprocessed" / "meta.json").read_text())
        assert set(meta["split_counts"]) == set(cfg.splits), (
            "split_counts must be keyed on the CONFIG-declared split set, so a "
            "split receiving zero scenes is visible as 0 rather than absent"
        )


class TestDeclaredButEmptySplitIsReported:
    """Downstream stages enumerated the splits PRESENT in the data, so a
    declared test split with no scored scenes vanished from stats and the report
    entirely — and an absent split is indistinguishable from one never declared."""

    def test_stats_and_report_name_a_declared_empty_split(self, tmp_path) -> None:
        import csv
        import json

        from amcd.config import Config
        from amcd.pipeline import Pipeline
        from amcd.runtime import RunContext, Verbosity

        # split_assignment 104 with n_id 6 starves test_id while train/valid survive,
        # so the run completes and the split's absence is the only thing under test.
        overlay = tmp_path / "starve.yaml"
        overlay.write_text("seeds:\n  split_assignment: 104\nscenes:\n  n_id: 6\n")
        cfg = Config.load(
            *CANONICAL_DRY_RUN, overlay
        )
        run_dir = tmp_path / "run"
        Pipeline(cfg, run_dir, RunContext(Verbosity(1, 0))).run_all()

        counts = json.loads((run_dir / "preprocessed" / "meta.json").read_text())
        assert counts["split_counts"]["test_id"] == 0, "fixture must actually starve test_id"

        rows = list(csv.DictReader((run_dir / "stats" / "ci_table.csv").open()))
        tid = [r for r in rows if r["split"] == "test_id"]
        assert tid, "declared-but-empty split missing from ci_table.csv"
        assert all(r["n_attempted"] == "0" and r["n_scored"] == "0" for r in tid)

        summary = (run_dir / "report" / "summary.txt").read_text()
        assert "test_id" in summary, "declared-but-empty split missing from summary.txt"
        # And it must be words, not a number a reader could mistake for a result.
        assert "0 scenes — unscored" in summary


class TestTheManifestIsTheDataset:
    """An excluded scene keeps its render on disk — that is what makes the
    exclusion re-derivable and a threshold change cheap. So membership cannot be
    read off the filesystem: `renders/manifest.json` is the authority, and
    anything that discovered scenes by listing would train on renders the render
    stage refused to admit."""

    def test_preprocess_encodes_only_admitted_scenes(self, tmp_path: Path) -> None:
        # The per-split bound is not under test here, and a tiny split breaches it
        # on one scene; `test_attrition_concentrated_on_one_split_is_refused` owns it.
        cfg = tiny_config(scenes={"n_id": 6}, max_excluded_frac_per_split=1.0)
        _pipeline(cfg, tmp_path).run_stage("gen-scenes")
        _pipeline(cfg, tmp_path).run_stage("render")

        manifest_path = tmp_path / "renders" / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        victim = manifest["admitted"].pop()
        manifest["excluded"].append({
            "scene_id": victim, "category": "qc_failed",
            "criteria": [{"leg": "low", "criterion": "min_energy_db",
                          "measured": -999.0, "threshold": -60.0}],
        })
        manifest["admitted_sha256"] = admitted_digest(manifest["admitted"])
        manifest_path.write_text(json.dumps(manifest))

        _pipeline(cfg, tmp_path).run_stage("preprocess")
        splits = json.loads((tmp_path / "preprocessed" / "splits.json").read_text())
        assert victim not in splits, (
            "an excluded scene reached preprocess, so the dataset includes a render "
            "the render stage refused to admit"
        )

    def test_a_missing_manifest_is_refused_not_assumed_complete(
        self, tmp_path: Path
    ) -> None:
        """Assuming "everything was admitted" would silently reinstate the whole
        batch on exactly the run where the difference matters."""
        cfg = tiny_config(scenes={"n_id": 4})
        _pipeline(cfg, tmp_path).run_stage("gen-scenes")
        _pipeline(cfg, tmp_path).run_stage("render")
        (tmp_path / "renders" / "manifest.json").unlink()
        with pytest.raises(RuntimeError, match="No render manifest"):
            _pipeline(cfg, tmp_path).run_stage("preprocess")

    def test_attrition_is_recorded_per_split_with_its_denominator(
        self, tmp_path: Path
    ) -> None:
        """`admitted` alone cannot say whether a small split was SPECIFIED small
        or ARRIVED small, and the scenes QC drops are not a random subset: the
        energy floor bites hardest at high absorption, which is a shift axis."""
        cfg = tiny_config(scenes={"n_id": 6}, max_excluded_frac_per_split=1.0)
        _pipeline(cfg, tmp_path).run_stage("gen-scenes")
        _pipeline(cfg, tmp_path).run_stage("render")

        manifest_path = tmp_path / "renders" / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        victim = manifest["admitted"].pop()
        manifest["excluded"].append(
            {"scene_id": victim, "category": "refused", "reason": "simulated"}
        )
        manifest["admitted_sha256"] = admitted_digest(manifest["admitted"])
        manifest_path.write_text(json.dumps(manifest))

        _pipeline(cfg, tmp_path).run_stage("preprocess")
        attrition = json.loads(
            (tmp_path / "preprocessed" / "meta.json").read_text()
        )["split_attrition"]
        assert set(attrition) == set(cfg.splits)
        hit = [row for row in attrition.values() if row["excluded"]]
        assert len(hit) == 1
        assert hit[0]["refused"] == 1 and hit[0]["qc_failed"] == 0
        assert hit[0]["admitted"] + hit[0]["excluded"] == hit[0]["generated"]
        assert sum(r["generated"] for r in attrition.values()) == manifest["generated"]

    def test_attrition_concentrated_on_one_split_is_refused(
        self, tmp_path: Path
    ) -> None:
        """A global bound is nearly blind to this: 5 lost scenes is 0.7 % of 720
        and 4.2 % of a 120-scene shift split, and it is the per-split comparison
        that is the result."""
        cfg = tiny_config(scenes={"n_id": 6})
        _pipeline(cfg, tmp_path).run_stage("gen-scenes")
        _pipeline(cfg, tmp_path).run_stage("render")

        manifest_path = tmp_path / "renders" / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        # Drop every scene of one shift split, which the global bound would let
        # through when the other splits are untouched.
        from amcd.data.splits import assign_split
        from amcd.simulators.base import SceneSpec

        by_split: dict[str, list[str]] = {}
        for p in sorted((tmp_path / "scenes").glob("scene_*.json")):
            spec = SceneSpec.from_json(p)
            by_split.setdefault(assign_split(spec.to_dict(), cfg), []).append(spec.scene_id)
        target = next(s for s in cfg.test_split_names if by_split.get(s))
        doomed = by_split[target]
        manifest["admitted"] = [s for s in manifest["admitted"] if s not in doomed]
        manifest["excluded"] += [
            {"scene_id": s, "category": "qc_failed", "criteria": []} for s in doomed
        ]
        manifest["admitted_sha256"] = admitted_digest(manifest["admitted"])
        manifest_path.write_text(json.dumps(manifest))

        with pytest.raises(ValueError, match="concentrated on one or more splits"):
            _pipeline(cfg, tmp_path).run_stage("preprocess")

    def test_a_manifest_edited_after_the_render_is_refused(
        self, tmp_path: Path
    ) -> None:
        """`admitted_sha256` has to be CHECKED or it is decoration.

        It was written and read by nothing, while its own docstring claimed it made
        membership drift detectable. A manifest whose admitted list no longer
        matches its digest describes a different dataset than the renders on disk,
        and preprocessing it would train on a membership no stage chose.
        """
        cfg = tiny_config(scenes={"n_id": 4})
        _pipeline(cfg, tmp_path).run_stage("gen-scenes")
        _pipeline(cfg, tmp_path).run_stage("render")

        manifest_path = tmp_path / "renders" / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["admitted"] = manifest["admitted"][:-1]   # digest NOT restamped
        manifest_path.write_text(json.dumps(manifest))

        with pytest.raises(RuntimeError, match="admitted_sha256"):
            _pipeline(cfg, tmp_path).run_stage("preprocess")

    def test_the_membership_digest_reaches_the_preprocess_stamp(
        self, tmp_path: Path
    ) -> None:
        """So a reader of `preprocessed/meta.json` can tell WHICH membership the
        tensors were built from, rather than assuming the manifest on disk now is
        the one that produced them."""
        cfg = tiny_config(scenes={"n_id": 4})
        _through_preprocess(cfg, tmp_path)
        manifest = json.loads((tmp_path / "renders" / "manifest.json").read_text())
        meta = json.loads((tmp_path / "preprocessed" / "meta.json").read_text())
        assert meta["admitted_sha256"] == manifest["admitted_sha256"]
        assert meta["admitted_sha256"] == admitted_digest(manifest["admitted"])

    def test_a_corrupted_admitted_render_is_refused_before_training(
        self, tmp_path: Path
    ) -> None:
        """`verify_render_artifacts` existed with no production caller while its
        docstring said "the caller decides whether a corrupt artifact is fatal".

        Nothing else checks: `_reusable` covers only a re-run of `render`, and only
        without `--force`. A `low.npy` truncated by a full disk between render and
        preprocess was encoded, trained on and evaluated in silence.
        """
        cfg = tiny_config(scenes={"n_id": 4})
        _pipeline(cfg, tmp_path).run_stage("gen-scenes")
        _pipeline(cfg, tmp_path).run_stage("render")

        manifest = json.loads((tmp_path / "renders" / "manifest.json").read_text())
        victim = manifest["admitted"][0]
        target = tmp_path / "renders" / victim / "low.npy"
        target.write_bytes(target.read_bytes()[:-64])   # truncated, as a full disk does

        with pytest.raises(RuntimeError, match="no longer match the digests"):
            _pipeline(cfg, tmp_path).run_stage("preprocess")

    def test_a_corrupted_EXCLUDED_render_does_not_fail_the_run(
        self, tmp_path: Path
    ) -> None:
        """The check runs over the admitted set only. An excluded scene's artifacts
        are not in the dataset, so their integrity cannot invalidate it — and
        failing on them would make a QC exclusion look like corruption."""
        cfg = tiny_config(scenes={"n_id": 6}, max_excluded_frac_per_split=1.0)
        _pipeline(cfg, tmp_path).run_stage("gen-scenes")
        _pipeline(cfg, tmp_path).run_stage("render")

        manifest_path = tmp_path / "renders" / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        victim = manifest["admitted"].pop()
        manifest["excluded"].append(
            {"scene_id": victim, "category": "refused", "reason": "simulated"}
        )
        manifest["admitted_sha256"] = admitted_digest(manifest["admitted"])
        manifest_path.write_text(json.dumps(manifest))

        target = tmp_path / "renders" / victim / "low.npy"
        target.write_bytes(target.read_bytes()[:-64])

        _pipeline(cfg, tmp_path).run_stage("preprocess")
        splits = json.loads((tmp_path / "preprocessed" / "splits.json").read_text())
        assert victim not in splits
