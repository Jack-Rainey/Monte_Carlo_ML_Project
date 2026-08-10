"""Stage-cache fingerprints and artifact residue.

Ledger rows F-46, F-47 (widening F-38), F-49, F-50, RD-54/RD-59.

Two failure families, both about a run_dir quietly disagreeing with its config:
  * FINGERPRINTS — a cached stage must be refused when its inputs changed, and
    NOT refused when they did not (a false refusal costs an emulated re-render).
  * RESIDUE — scene ids are POSITIONAL, so a shrunk scene set leaves orphans that
    a later config silently reuses under different geometry.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from amcd.config import Config
from amcd.pipeline import (
    STAGE_CODE_SCOPE,
    STAGE_FINGERPRINT,
    STAGE_UPSTREAM,
    Pipeline,
    _diff_fingerprints,
    _gen_scenes_fingerprint,
    _preprocess_fingerprint,
    _render_fingerprint,
    _sentinel,
)

from tests.conftest import QUIET, tiny_config


class TestNestedFingerprintDiff:
    """F-49: RD-35's promise was 'the error says WHAT changed'. Top-level only broke it."""

    def test_nested_change_reports_a_dotted_leaf_path(self) -> None:
        old = {"splits": {"train": {"frac": 0.6, "role": "train"}}}
        new = {"splits": {"train": {"frac": 0.55, "role": "train"}}}
        lines = _diff_fingerprints(old, new)
        assert lines == ["    splits.train.frac: 0.6 → 0.55"]

    def test_the_unchanged_siblings_are_not_printed(self) -> None:
        """The observed failure: one changed value buried in two ~700-char blobs."""
        old = {"splits": {f"s{i}": {"frac": 0.1, "count": i} for i in range(6)}}
        new = json.loads(json.dumps(old))
        new["splits"]["s3"]["count"] = 99
        lines = _diff_fingerprints(old, new)
        assert lines == ["    splits.s3.count: 3 → 99"]

    def test_deeply_nested_paths_are_fully_qualified(self) -> None:
        old = {"a": {"b": {"c": {"d": 1}}}}
        new = {"a": {"b": {"c": {"d": 2}}}}
        assert _diff_fingerprints(old, new) == ["    a.b.c.d: 1 → 2"]

    def test_added_and_removed_keys_are_distinguishable(self) -> None:
        lines = _diff_fingerprints({"a": {"x": 1}}, {"a": {"y": 2}})
        assert lines == ["    a.x: 1 → <absent>", "    a.y: <absent> → 2"]

    def test_a_literal_absent_string_is_not_confused_with_a_missing_key(self) -> None:
        """`<absent>` is a rendering, never a sentinel value that can collide."""
        lines = _diff_fingerprints({"a": "<absent>"}, {})
        assert lines == ["    a: '<absent>' → <absent>"]

    def test_dict_replaced_by_scalar_still_reports(self) -> None:
        assert _diff_fingerprints({"a": {"b": 1}}, {"a": 5}) == ["    a: {'b': 1} → 5"]


class TestGenScenesFingerprintScope:
    """F-50: fail safe, but not at the cost of a re-render that cannot be needed."""

    def test_frac_does_not_invalidate_gen_scenes_or_render(self) -> None:
        before = tiny_config()
        after = tiny_config(splits={"train": {"frac": 0.55}, "valid": {"frac": 0.25}})
        assert before.splits["train"].frac != after.splits["train"].frac
        assert _gen_scenes_fingerprint(before) == _gen_scenes_fingerprint(after)
        assert _render_fingerprint(before) == _render_fingerprint(after)

    def test_frac_still_invalidates_preprocess(self) -> None:
        """It moves the split assignment, which is where frac is actually consumed."""
        before = tiny_config()
        after = tiny_config(splits={"train": {"frac": 0.55}, "valid": {"frac": 0.25}})
        assert _preprocess_fingerprint(before) != _preprocess_fingerprint(after)

    @pytest.mark.parametrize(
        "override",
        [
            {"scenes": {"n_id": 7}},
            {"splits": {"test_material_shift": {"count": 5}}},
            {"seeds": {"master": 123}},
        ],
        ids=["n_id", "shift-count", "seed"],
    )
    def test_generation_relevant_changes_still_invalidate(self, override) -> None:
        assert _gen_scenes_fingerprint(tiny_config()) != _gen_scenes_fingerprint(
            tiny_config(**override)
        )


class TestEvalAndStatsFingerprints:
    """RD-54 promoted these out of DEFERRED; RD-59 says config keys alone are not enough."""

    def test_eval_and_stats_are_wired(self) -> None:
        assert STAGE_FINGERPRINT["eval"] is not None
        assert STAGE_FINGERPRINT["stats"] is not None

    def test_every_stage_is_declared_even_when_unwired(self) -> None:
        from amcd.pipeline import STAGES

        assert set(STAGE_FINGERPRINT) == set(STAGES)

    def test_eval_carries_a_code_version(self) -> None:
        """The AC-17 trigger case was a CODE change no config key could see (RD-59)."""
        fp = STAGE_FINGERPRINT["eval"](tiny_config())
        assert "code_version" in fp
        assert isinstance(fp["code_version"], str) and fp["code_version"]

    def test_a_metric_threshold_change_invalidates_eval(self) -> None:
        assert STAGE_FINGERPRINT["eval"](tiny_config()) != STAGE_FINGERPRINT["eval"](
            tiny_config(metric_band_resolvability_margin=0.07)
        )

    def test_stats_declares_eval_as_its_upstream_rather_than_recomputing_it(
        self,
    ) -> None:
        """F-54: `_stats_fingerprint` hand-rolled
        `"upstream_eval": _fingerprint_sha(_eval_fingerprint(config))` — a chain
        RECOMPUTED from the current config, which is what the STAGE_UPSTREAM
        docstring forbids. Recomputing made `--force` a laundering step: stats
        stamped the new config's eval sha while eval.done still held the old one.
        The dependence belongs in STAGE_UPSTREAM, whose sentinel-read + recursive
        leaf diff is the one audited mechanism.
        """
        assert STAGE_UPSTREAM["stats"] == "eval"
        assert "upstream_eval" not in STAGE_FINGERPRINT["stats"](tiny_config())

    def test_bootstrap_settings_invalidate_stats_but_not_eval(self) -> None:
        before, after = tiny_config(), tiny_config(bootstrap_n_resamples=500)
        assert STAGE_FINGERPRINT["stats"](before) != STAGE_FINGERPRINT["stats"](after)
        assert STAGE_FINGERPRINT["eval"](before) == STAGE_FINGERPRINT["eval"](after)


class TestTheChainReachesTheReportedResult:
    """F-53: the cache did not protect the reported result.

    REPRODUCED before the fix: on a complete run_dir, changing five model
    hyperparameters and re-running `amcd all` printed `[skip]` for ALL NINE stages
    and exited 0 — `config.yaml` re-stamped with the new values while
    `checkpoints/best.pt` still held the old model, so `summary.txt` and
    `ci_table.csv` were the old model's numbers under the new model's stamp.
    """

    def test_the_chain_runs_unbroken_from_scenes_to_stats(self) -> None:
        chain = ["gen-scenes", "render", "preprocess", "train", "infer", "eval", "stats"]
        for downstream, upstream in zip(chain[1:], chain[:-1]):
            assert STAGE_UPSTREAM[downstream] == upstream, (
                f"{downstream} is not chained to {upstream}; a change to "
                f"{upstream}'s inputs would not reach the reported result"
            )

    def test_every_chained_stage_can_anchor_a_chain(self) -> None:
        """An upstream with no fingerprint records `null`, which is
        indistinguishable from 'never ran' (F-41)."""
        for stage, upstream in STAGE_UPSTREAM.items():
            if upstream is not None:
                assert STAGE_FINGERPRINT[upstream] is not None, (
                    f"{stage} chains to {upstream}, which declares no fingerprint"
                )

    @pytest.mark.parametrize(
        "override",
        [
            {"model": {"params": {"hidden_channels": 32}}},
            {"model": {"params": {"n_layers": 4}}},
            {"learning_rate": 0.05},
            {"huber_delta": 3.0},
            {"n_epochs": 7},
            {"batch_size": 3},
            {"early_stopping_patience": 9},
            {"seeds": {"weight_init": 4242}},
            {"seeds": {"data_shuffle": 4242}},
        ],
        ids=["hidden_channels", "n_layers", "lr", "huber_delta", "n_epochs",
             "batch_size", "patience", "seed_weight_init", "seed_data_shuffle"],
    )
    def test_a_training_input_change_invalidates_train(self, override) -> None:
        """Every one of these was in the reproduced transcript's `[skip]` run."""
        assert STAGE_FINGERPRINT["train"](tiny_config()) != STAGE_FINGERPRINT[
            "train"
        ](tiny_config(**override))

    def test_train_and_infer_carry_a_code_version(self) -> None:
        for stage in ("train", "infer"):
            fp = STAGE_FINGERPRINT[stage](tiny_config())
            assert isinstance(fp.get("code_version"), str) and fp["code_version"]

    def test_a_model_change_reaches_infer(self) -> None:
        after = tiny_config(model={"params": {"hidden_channels": 32}})
        assert STAGE_FINGERPRINT["infer"](tiny_config()) != STAGE_FINGERPRINT[
            "infer"
        ](after)


class TestCodeVersionSeesTheWorkingTree:
    """F-55 / RD-66: `git rev-parse HEAD` is blind to uncommitted edits — the exact
    state the guard exists for, since this project's loop is edit → run → review →
    commit and AC-17 was a code-only change to `room_acoustic.py`."""

    def test_editing_metric_code_changes_evals_version(self) -> None:
        """The reproduction in the row: edit `evaluation/room_acoustic.py`, re-run
        `amcd eval` on the same run_dir, and get `[skip] eval (cached)` with
        `metrics.parquet` served under the new code."""
        import amcd.provenance as prov
        from amcd.pipeline import _code_version

        target = Path(prov.__file__).resolve().parent / "evaluation" / "room_acoustic.py"
        original = target.read_bytes()
        before = _code_version("eval")
        try:
            target.write_bytes(original + b"\n# F-55 probe\n")
            assert _code_version("eval") != before, (
                "an uncommitted edit to metric code did not move eval's "
                "code_version, so a cached metrics.parquet would be served"
            )
        finally:
            target.write_bytes(original)
        assert _code_version("eval") == before, "restore did not return the version"

    def test_a_core_module_edit_reaches_every_scope(self) -> None:
        """`config.py` parameterizes every stage, so it is in every scope."""
        import amcd.provenance as prov

        target = Path(prov.__file__).resolve().parent / "config.py"
        original = target.read_bytes()
        before = {s: prov.code_version(sc) for s, sc in STAGE_CODE_SCOPE.items()}
        try:
            target.write_bytes(original + b"\n# core probe\n")
            for stage, scope in STAGE_CODE_SCOPE.items():
                assert prov.code_version(scope) != before[stage], stage
        finally:
            target.write_bytes(original)

    def test_an_out_of_scope_edit_does_not_invalidate_a_stage(self) -> None:
        """The reason the scope is declared rather than whole-package: a guard that
        refuses a cached stage for visibly irrelevant reasons teaches the operator
        to reach for `--force`, which disables it entirely."""
        import amcd.provenance as prov
        from amcd.pipeline import _code_version

        target = Path(prov.__file__).resolve().parent / "reporting" / "tables.py"
        original = target.read_bytes()
        before = _code_version("train")
        try:
            target.write_bytes(original + b"\n# out-of-scope probe\n")
            assert _code_version("train") == before
        finally:
            target.write_bytes(original)

    def test_each_stage_declares_the_subpackage_its_own_entry_point_lives_in(
        self,
    ) -> None:
        """A scope that omits a real dependency fails SILENTLY — the stage goes on
        serving a cached artifact under changed code. The minimum checkable claim
        is that a stage's own implementation is in its own scope; the wider
        dependency set is a judgement recorded in STAGE_CODE_SCOPE's comment."""
        from amcd.pipeline import _dispatch

        for stage, scope in STAGE_CODE_SCOPE.items():
            own = _dispatch(stage).__module__          # e.g. amcd.evaluation.evaluator
            subpackage = own.split(".")[1]
            assert subpackage in scope, (
                f"{stage} is implemented in amcd.{subpackage}, which is not in its "
                f"declared code scope {scope} — an edit there would not invalidate it"
            )

    def test_every_stage_carrying_a_code_version_declares_a_scope(self) -> None:
        for stage, fingerprint in STAGE_FINGERPRINT.items():
            if fingerprint is None:
                continue
            if "code_version" in fingerprint(tiny_config()):
                assert stage in STAGE_CODE_SCOPE

    def test_it_does_not_depend_on_git_being_available(self) -> None:
        """A wheel install into site-packages returned "unavailable" permanently,
        and an install inside an UNRELATED repo returned that repo's sha."""
        import amcd.provenance as prov

        version = prov.code_version(("evaluation",))
        assert len(version) == 64 and version != "unavailable"

    def test_a_scope_naming_a_missing_path_is_rejected(self) -> None:
        """Silently hashing nothing would stop protecting the stage."""
        import amcd.provenance as prov

        with pytest.raises(ValueError, match="does not exist"):
            prov.code_version(("no_such_subpackage",))

    def test_the_two_provenance_channels_share_one_helper(self) -> None:
        """F-56: `versions.json` and the eval sentinel must not be able to
        describe different code."""
        import amcd.provenance as prov
        from amcd.pipeline import _code_version

        assert _code_version("eval") == prov.code_version(STAGE_CODE_SCOPE["eval"])

    def test_git_is_resolved_from_the_package_not_the_run_dir(self) -> None:
        """F-56: `cwd=run_dir.parent` stamped "unavailable" into versions.json for
        any run_dir outside the checkout — the normal case for a data volume."""
        import amcd.provenance as prov

        assert prov.git_sha() != "unavailable", (
            "this checkout is a git repo, so the package-resolved sha must exist"
        )


class TestTheRecordLengthGateIsNotBypassableThroughTheCache:
    """F-57: `ir_duration` is what AC-22's record-length gate compares against, but
    it was not in the gen-scenes fingerprint — so the gate could be skipped.

    Reachable in practice, not a contrivance: render's fingerprint DOES move with
    `ir_duration` (through `n_samples`), so the operator forces render and leaves
    gen-scenes cached.
    """

    def test_ir_duration_invalidates_gen_scenes(self) -> None:
        assert _gen_scenes_fingerprint(tiny_config()) != _gen_scenes_fingerprint(
            tiny_config(ir_duration=0.25)
        )

    def test_the_simulator_invalidates_gen_scenes(self) -> None:
        """The backend declares the minimum source-receiver separation that
        gen-scenes pre-flight-checks (AC-13), so it governs admission too."""
        before = tiny_config()
        after = tiny_config(
            simulator={"params": {"min_source_receiver_distance_m": 0.75}}
        )
        assert _gen_scenes_fingerprint(before) != _gen_scenes_fingerprint(after)

    def test_a_cached_run_refuses_gen_scenes_when_only_ir_duration_moved(
        self, tmp_path: Path
    ) -> None:
        cfg = tiny_config(scenes={"n_id": 6})
        Pipeline(cfg, tmp_path, QUIET).run_stage("gen-scenes")

        shortened = tiny_config(scenes={"n_id": 6}, ir_duration=0.25)
        with pytest.raises(RuntimeError, match="ir_duration"):
            Pipeline(shortened, tmp_path, QUIET).run_stage("gen-scenes")


class TestAPartialStageIsNeverServedAsCached:
    """F-58: a stage killed part-way through writing left the PREVIOUS run's
    success sentinel standing over half-new artifacts."""

    def test_a_stage_that_fails_mid_write_does_not_leave_a_valid_sentinel(
        self, tmp_path: Path
    ) -> None:
        import amcd.pipeline as pipeline_mod

        cfg = tiny_config(scenes={"n_id": 6})
        Pipeline(cfg, tmp_path, QUIET).run_stage("gen-scenes")
        assert _sentinel(tmp_path, "gen-scenes").exists()

        def _abort(config, run_dir, verbosity):
            (run_dir / "scenes").mkdir(parents=True, exist_ok=True)
            (run_dir / "scenes" / "half_written.json").write_text("{}")
            raise RuntimeError("killed part-way through writing")

        original = pipeline_mod._dispatch
        pipeline_mod._dispatch = lambda stage: _abort if stage == "gen-scenes" else original(stage)
        try:
            with pytest.raises(RuntimeError, match="killed part-way"):
                Pipeline(cfg, tmp_path, QUIET, force=True).run_stage("gen-scenes")
        finally:
            pipeline_mod._dispatch = original

        assert not _sentinel(tmp_path, "gen-scenes").exists(), (
            "the aborted stage left a success sentinel, so the next run would "
            "print `[skip] gen-scenes (cached)` over half-new artifacts"
        )

    def test_the_next_run_re_runs_rather_than_skipping(self, tmp_path: Path) -> None:
        import amcd.pipeline as pipeline_mod

        cfg = tiny_config(scenes={"n_id": 6})
        Pipeline(cfg, tmp_path, QUIET).run_stage("gen-scenes")

        original = pipeline_mod._dispatch

        def _boom(config, run_dir, verbosity):
            raise RuntimeError("killed")

        pipeline_mod._dispatch = lambda stage: _boom if stage == "gen-scenes" else original(stage)
        try:
            with pytest.raises(RuntimeError):
                Pipeline(cfg, tmp_path, QUIET, force=True).run_stage("gen-scenes")
        finally:
            pipeline_mod._dispatch = original

        calls: list[str] = []

        def _record(config, run_dir, verbosity):
            calls.append("ran")
            original("gen-scenes")(config, run_dir, verbosity)

        pipeline_mod._dispatch = lambda stage: _record if stage == "gen-scenes" else original(stage)
        try:
            Pipeline(cfg, tmp_path, QUIET).run_stage("gen-scenes")
        finally:
            pipeline_mod._dispatch = original

        assert calls == ["ran"], "the stage was skipped as cached after an abort"


class TestZeroCountIsRejected:
    """F-46: a value the schema admits must not be a value the pipeline mishandles."""

    def test_zero_count_on_a_shift_split_fails_at_config_load(self) -> None:
        with pytest.raises(ValueError, match="test_material_shift"):
            tiny_config(splits={"test_material_shift": {"count": 0}})

    def test_the_error_names_the_count(self) -> None:
        with pytest.raises(ValueError, match="count must be > 0"):
            tiny_config(splits={"test_geometry_shift": {"count": -3}})

    def test_room_stats_summary_tolerates_an_empty_split(self) -> None:
        """The second half: the unguarded reduction, matching its three siblings."""
        from amcd.scenes.generator import _summarize

        payload = {
            key: (_summarize([]) if [] else None)
            for key in ("volume_m3", "t60_sabine_s")
        }
        assert payload == {"volume_m3": None, "t60_sabine_s": None}


class TestArtifactResidue:
    """F-47: renders/ and carrier/ are pruned against the CURRENT scene set."""

    def _run(self, cfg: Config, run_dir: Path) -> None:
        from amcd.data.preprocess import run_preprocess
        from amcd.scenes.generator import run_gen_scenes
        from amcd.simulators.render import run_render

        run_gen_scenes(cfg, run_dir, QUIET)
        run_render(cfg, run_dir, QUIET)
        run_preprocess(cfg, run_dir, QUIET)

    def test_shrinking_the_scene_set_leaves_no_orphans(self, tmp_path: Path) -> None:
        big = tiny_config(scenes={"n_id": 20})
        self._run(big, tmp_path)
        n_big = len(list(tmp_path.glob("scenes/scene_*.json")))

        small = tiny_config(scenes={"n_id": 8})
        self._run(small, tmp_path)
        current = {p.stem for p in tmp_path.glob("scenes/scene_*.json")}
        assert 0 < len(current) < n_big, "fixture must actually shrink the set"

        render_ids = {p.name for p in (tmp_path / "renders").iterdir() if p.is_dir()}
        carrier_ids = {p.stem for p in (tmp_path / "preprocessed" / "carrier").glob("*.npy")}
        assert render_ids == current, "renders/ holds scenes the config no longer declares"
        assert carrier_ids == current, "carrier/ holds scenes the config no longer declares"

    def test_carriers_that_survive_are_not_rewritten_from_stale_data(
        self, tmp_path: Path
    ) -> None:
        """Pruning must remove orphans only — a surviving scene keeps its own carrier."""
        cfg = tiny_config(scenes={"n_id": 12})
        self._run(cfg, tmp_path)
        carrier = tmp_path / "preprocessed" / "carrier" / "scene_0000.npy"
        kept = np.load(carrier).copy()
        self._run(cfg, tmp_path)
        assert np.array_equal(np.load(carrier), kept)
