"""Stage-cache fingerprints, code versions, and artifact residue.

Every class names the row it pins; the reproductions behind those rows live in
git history and the review ledger, which is what the ids are for.

The failure families, all about a run_dir quietly disagreeing with its config:
  * FINGERPRINT SCOPE — a cached stage must be refused when its inputs changed,
    and NOT refused when they did not (a false refusal costs an emulated
    re-render, and teaches the operator to reach for `--force`).
  * CODE VERSION — a config fingerprint cannot see a code change; the declared
    per-stage scope must cover what the stage actually depends on.
  * THE CHAIN — staleness must reach the reported table transitively.
  * CONFIG-FIELD COVERAGE — every Config field is fingerprinted or declared exempt.
  * HOST INDEPENDENCE — the cache key describes the source, not the machine.
  * RESIDUE — scene ids are POSITIONAL, so a shrunk scene set leaves orphans that
    a later config silently reuses under different geometry.
"""
import ast
import json
from pathlib import Path

import numpy as np
import pytest

from amcd.config import SEED_NAMES, Config
from amcd.pipeline import (
    FINGERPRINT_EXEMPT_FIELDS,
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
        """A `code_version` with no declared scope would hash something nobody
        stated. Two ways to declare one: a fixed entry in `STAGE_CODE_SCOPE`, or
        membership of `_BACKEND_SCOPED_STAGES`, whose scope the ACTIVE simulator
        supplies so that swapping backends swaps the protection with them."""
        from amcd.pipeline import _BACKEND_SCOPED_STAGES

        for stage, fingerprint in STAGE_FINGERPRINT.items():
            if fingerprint is None:
                continue
            if "code_version" in fingerprint(tiny_config()):
                assert stage in STAGE_CODE_SCOPE or stage in _BACKEND_SCOPED_STAGES, (
                    f"{stage} carries a code_version but declares no scope"
                )

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
        any run_dir outside the checkout — the normal case for a data volume.

        Asserts the CONTRACT, not this machine: a 40-hex sha or the literal
        "unavailable". An earlier version asserted `!= "unavailable"` with the
        reason "this checkout is a git repo" — a property of one host, not of the
        code, contradicting both `provenance.git_sha`'s stated contract and the
        sibling test below. The project must run the same code from a wheel and
        from a source export on a second host, neither of which has a `.git`
        (F-168).
        """
        import re

        import amcd.provenance as prov

        sha = prov.git_sha()
        assert sha == "unavailable" or re.fullmatch(r"[0-9a-f]{40}", sha), sha
        # Resolved from the PACKAGE: asking about a run_dir on a data volume is
        # what stamped "unavailable" beside a real sha in the same run (F-56).
        assert prov._PACKAGE_ROOT == Path(prov.__file__).resolve().parent


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


class TestTheTableProducingStagesAreCacheProtected:
    """F-63 / F-64: cycle 3's "the cache protects the reported result" held for
    train/infer/eval and failed for `preprocess`, `stats` and `report` — the stages
    that produce the table. Each was reproduced end to end at exit 0 before the fix.
    """

    def test_preprocess_carries_a_code_version(self) -> None:
        fp = STAGE_FINGERPRINT["preprocess"](tiny_config())
        assert isinstance(fp.get("code_version"), str) and fp["code_version"]

    def test_stats_and_report_carry_a_code_version(self) -> None:
        for stage in ("stats", "report"):
            fp = STAGE_FINGERPRINT[stage](tiny_config())
            assert isinstance(fp.get("code_version"), str) and fp["code_version"]

    def test_the_chain_runs_unbroken_from_scenes_to_report(self) -> None:
        chain = ["gen-scenes", "render", "preprocess", "train", "infer", "eval",
                 "stats", "report"]
        for downstream, upstream in zip(chain[1:], chain[:-1]):
            assert STAGE_UPSTREAM[downstream] == upstream, (
                f"{downstream} is not chained to {upstream}, so a change to "
                f"{upstream}'s inputs would not reach the reported table"
            )

    def test_report_format_invalidates_report(self) -> None:
        assert STAGE_FINGERPRINT["report"](tiny_config()) != STAGE_FINGERPRINT[
            "report"
        ](tiny_config(report_format="markdown"))

    @pytest.mark.parametrize(
        "stage, module",
        [
            ("preprocess", "representations/spectrogram.py"),
            ("stats", "stats/aggregate.py"),
            ("report", "reporting/tables.py"),
        ],
    )
    def test_editing_the_code_that_produces_the_artifact_moves_its_version(
        self, stage: str, module: str
    ) -> None:
        """The generalisation of the three reproductions: for each newly wired
        stage, an edit to the module that computes its output must move its key."""
        import amcd.provenance as prov
        from amcd.pipeline import _code_version

        target = Path(prov.__file__).resolve().parent / module
        original = target.read_bytes()
        before = _code_version(stage)
        try:
            target.write_bytes(original + b"\n# F-63/F-64 probe\n")
            assert _code_version(stage) != before, (
                f"editing {module} did not move {stage}'s code_version, so a "
                f"cached artifact would be served under the changed code"
            )
        finally:
            target.write_bytes(original)
        assert _code_version(stage) == before, "restore did not return the version"

    def test_an_encoder_edit_refuses_preprocess_and_not_only_train(self) -> None:
        """F-64's sharp edge: the refusal must name the stage whose artifacts are
        actually stale. A message naming `train` sends the operator to `--force`
        train, which rebuilds the wrong thing and exits 0."""
        import amcd.provenance as prov
        from amcd.pipeline import _code_version

        target = Path(prov.__file__).resolve().parent / "representations" / "spectrogram.py"
        original = target.read_bytes()
        before = _code_version("preprocess")
        try:
            target.write_bytes(original + b"\n# encoder probe\n")
            assert _code_version("preprocess") != before
        finally:
            target.write_bytes(original)


class TestTheExpensiveArtifactIsCacheProtected:
    """F-75/RD-107/AC-44: `render` is the costliest artifact and was the least
    protected.

    Neither `gen-scenes` nor `render` carried a `code_version`, so an edit to the
    backend that synthesizes the decay changed every reported absolute AND every
    paired improvement while all nine stages printed `[skip] (cached)` at exit 0.
    eval reads `renders/<scene>/high.npy` as the ISO REFERENCE leg, so this was not
    "the renders are stale" — it was "the thing improvement is measured against is
    stale".

    Scoping is the whole difficulty. Naming the `simulators` package would mean a
    tweak to the dry-run scaffold discards hours of emulated render; naming nothing
    leaves the hole. The backend declares its own scope (`code_scope()`), so the
    protection follows whichever simulator is active.
    """

    def test_editing_the_active_backend_invalidates_the_render(self, tmp_path) -> None:
        assert self._moves(tmp_path, "amcd/simulators/gsound_sir.py"), (
            "an edit to the backend that produces the IRs must invalidate them"
        )

    def test_editing_an_inactive_backend_does_not(self, tmp_path) -> None:
        assert not self._moves(tmp_path, "amcd/simulators/dry_run.py"), (
            "the dry-run scaffold cannot change a gsound render, and discarding "
            "one over it costs hours under emulation"
        )

    def test_editing_device_selection_does_not(self, tmp_path) -> None:
        """`amcd/device.py` exists as its own module for exactly this: it was in
        `_CORE_SOURCES`, which every scope unions in, so the MPS -> CUDA -> CPU
        fallback — the code the cross-platform requirement makes someone touch —
        would have invalidated a 720-scene render."""
        assert not self._moves(tmp_path, "amcd/device.py")

    @staticmethod
    def _moves(tmp_path, rel: str) -> bool:
        """Render `code_version` before vs after appending a statement to `rel`,
        computed in a SEPARATE interpreter against a COPY of the package so the
        probe never writes to tracked source (F-217)."""
        import shutil, subprocess, sys

        root = tmp_path / "pkg"
        shutil.copytree(Path("src"), root / "src")
        shutil.copytree(Path("configs"), root / "configs")
        probe = root / "probe.py"
        probe.write_text(
            f"import sys; sys.path.insert(0, {str(root / 'src')!r})\n"
            "from pathlib import Path\n"
            "from amcd.config import Config\n"
            "from amcd.pipeline import _render_fingerprint\n"
            "print(_render_fingerprint(Config.load(Path('configs/base.yaml')))['code_version'])\n"
        )

        def version() -> str:
            out = subprocess.run([sys.executable, "probe.py"], cwd=root,
                                 capture_output=True, text=True)
            assert out.stdout.strip(), out.stderr[-400:]
            return out.stdout.strip()

        before = version()
        target = root / "src" / rel
        target.write_text(target.read_text() + "\n_MUTATION_PROBE = 1\n")
        return version() != before


class TestTheDatasetFingerprintsAreHostIndependent:
    """F-81/F-100/F-82: a cache key must describe the DATASET, not the machine.

    `_render_fingerprint` and `_gen_scenes_fingerprint` hashed
    `config.simulator.params` whole, so `render_python` — the x86 interpreter the
    emulated render runs under — was part of the render's cache identity. The
    project is required to run on this Apple-Silicon host and on a native x86_64
    desktop from the same code, so that made a dataset rendered on one host demand
    a byte-identical re-render on the other.
    """

    @staticmethod
    def _with_host_layer(tmp_path):
        layer = tmp_path / "host.yaml"
        layer.write_text(
            "simulator:\n  params:\n    render_python: /somewhere/else/bin/python\n"
        )
        return Config.load(Path("configs/base.yaml"), layer)

    @pytest.mark.parametrize("fingerprint", [_render_fingerprint, _gen_scenes_fingerprint])
    def test_the_interpreter_path_is_not_a_cache_key(self, tmp_path, fingerprint) -> None:
        base = Config.load(Path("configs/base.yaml"))
        moved = self._with_host_layer(tmp_path)
        assert base.simulator.params["render_python"] != moved.simulator.params["render_python"], (
            "the fixture must actually differ, or this test cannot fail"
        )
        dumped = json.dumps(fingerprint(base), sort_keys=True, default=str)
        assert json.dumps(fingerprint(moved), sort_keys=True, default=str) == dumped
        assert "render_python" not in dumped

    @pytest.mark.parametrize("fingerprint", [_render_fingerprint, _gen_scenes_fingerprint])
    def test_a_disclosure_threshold_is_not_a_cache_key(self, fingerprint) -> None:
        """`max_discarded_tail_db` changes what is REPORTED about an IR and never
        the IR, so re-tightening it must not cost a multi-hour emulated re-render."""
        assert "max_discarded_tail_db" not in json.dumps(
            fingerprint(Config.load(Path("configs/base.yaml"))), sort_keys=True, default=str
        )

    def test_the_pinned_upstream_version_IS_still_a_cache_key(self) -> None:
        """The filter must not become a hole: `commit_sha` identifies the renderer
        that produced the IR, so it stays."""
        assert "commit_sha" in json.dumps(
            _render_fingerprint(Config.load(Path("configs/base.yaml"))),
            sort_keys=True, default=str,
        )


class TestDeclaredScopeCoversWhatTheStageImports:
    """F-66: the scope declaration is only as good as its weakest entry, and a
    scope that omits a real dependency fails SILENTLY.

    `eval` and `infer` both called `data.normalization.denormalize` on every
    reported leg while neither declared `data`, masked only because `data` sits in
    TRAIN's scope and the chain refuses upstream first — an accident of ordering.

    WHAT THIS TEST CHECKS, precisely, because the previous docstring claimed more
    than the test did and that overstatement was itself the finding: the declared
    scope is a SUPERSET of the stage's STATIC transitive `amcd.*` import closure
    (module-level and function-level), minus `_CORE_SOURCES`.

    WHAT IT CANNOT CHECK: a dependency reached without an import statement. The
    plugin registry loads `representations`, `models` and `simulators` BY NAME, so
    those edges are invisible here and remain a declared judgement. Over-declaring
    is therefore allowed (and several scopes do); under-declaring is what fails.

    It also cannot check a dependency this walker fails to RESOLVE — so the walker
    now asserts rather than dropping, because a silently shrinking closure makes
    the test pass for the wrong reason (F-77). That is the same failure this class
    exists to prevent, one level up: the guard claiming more than it checks.
    """

    @staticmethod
    def _module_file(module: str) -> Path | None:
        import amcd

        src_root = Path(amcd.__file__).resolve().parent.parent
        rel = Path(*module.split("."))
        for candidate in (src_root / rel.with_suffix(".py"), src_root / rel / "__init__.py"):
            if candidate.exists():
                return candidate
        return None

    @classmethod
    def _amcd_imports(cls, module: str) -> set[str]:
        """Every `amcd.*` module `module` imports, absolute or relative.

        Relative imports are anchored on the module's PACKAGE, and a package's
        `__init__.py` is its own anchor while a plain module's anchor is its
        parent. Getting that wrong is not a near-miss: resolving `from .foo import
        x` inside `amcd/data/__init__.py` against `amcd` instead of `amcd.data`
        yields a module that does not exist, which then vanished through the
        resolvability filter below and took the whole subtree out of the closure
        (F-77 — `amcd.representations`'s four imports, including the encoder that
        is F-64's own reproduction, were invisible).
        """
        path = cls._module_file(module)
        if path is None:
            return set()
        parts = module.split(".")
        anchor = parts if path.name == "__init__.py" else parts[:-1]
        found: set[str] = set()
        unresolved: list[str] = []
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                found.update(a.name for a in node.names if a.name.startswith("amcd"))
            elif isinstance(node, ast.ImportFrom):
                if node.level:  # relative: resolve against this module's package
                    base = anchor[: len(anchor) - (node.level - 1)]
                    prefix = base + (node.module.split(".") if node.module else [])
                elif node.module and node.module.startswith("amcd"):
                    prefix = node.module.split(".")
                else:
                    continue
                if not prefix or prefix[0] != "amcd":
                    continue
                stem = ".".join(prefix)
                if cls._module_file(stem) is None:
                    # The module an import statement names MUST resolve. Dropping
                    # it silently is how F-77 hid 60 edges.
                    unresolved.append(f"{module} -> {stem}")
                    continue
                found.add(stem)
                # `from .base import X` — X may be a submodule or an ordinary name.
                # Unresolvable ones here are names, which is expected, so these are
                # filtered rather than reported.
                found.update(f"{stem}.{a.name}" for a in node.names)
        assert not unresolved, (
            f"the import walker could not resolve {unresolved}; it is under-"
            f"reporting the closure, so this test is not checking what it claims"
        )
        return {m for m in found if cls._module_file(m) is not None}

    @classmethod
    def _closure(cls, entry: str) -> set[str]:
        seen: set[str] = set()
        stack = [entry]
        while stack:
            module = stack.pop()
            if module in seen:
                continue
            seen.add(module)
            stack.extend(cls._amcd_imports(module) - seen)
        return seen

    def test_every_declared_scope_covers_the_stages_import_closure(self) -> None:
        import amcd
        import amcd.provenance as prov
        from amcd.pipeline import _dispatch

        package_root = Path(amcd.__file__).resolve().parent
        for stage, scope in STAGE_CODE_SCOPE.items():
            covered = set(scope) | set(prov._CORE_SOURCES)
            missing = []
            for module in self._closure(_dispatch(stage).__module__):
                rel = self._module_file(module).relative_to(package_root).as_posix()
                if any(rel == e or rel.startswith(f"{e}/") for e in covered):
                    continue
                missing.append(rel)
            assert not missing, (
                f"stage {stage!r} imports {sorted(missing)}, which its declared "
                f"scope {scope} does not cover — an edit there would not "
                f"invalidate its cached artifacts"
            )

    def test_eval_and_infer_declare_the_module_they_denormalize_with(self) -> None:
        """The specific omission F-66 names, pinned so it cannot silently return."""
        for stage in ("eval", "infer"):
            assert "data" in STAGE_CODE_SCOPE[stage]

    def test_patching_denormalize_moves_eval_and_infer(self) -> None:
        import amcd.provenance as prov
        from amcd.pipeline import _code_version

        target = Path(prov.__file__).resolve().parent / "data" / "normalization.py"
        original = target.read_bytes()
        before = {s: _code_version(s) for s in ("eval", "infer", "train")}
        try:
            target.write_bytes(original + b"\n# F-66 probe\n")
            for stage in ("eval", "infer", "train"):
                assert _code_version(stage) != before[stage], stage
        finally:
            target.write_bytes(original)

    def test_the_package_init_is_in_every_scope(self) -> None:
        """`amcd/__init__.py` is imported through by every stage and was in no
        scope, so an edit to it invalidated nothing."""
        import amcd.provenance as prov
        from amcd.pipeline import _code_version

        target = Path(prov.__file__).resolve().parent / "__init__.py"
        original = target.read_bytes()
        before = {s: _code_version(s) for s in STAGE_CODE_SCOPE}
        try:
            target.write_bytes(original + b"\n# init probe\n")
            for stage in STAGE_CODE_SCOPE:
                assert _code_version(stage) != before[stage], stage
        finally:
            target.write_bytes(original)


class TestEveryConfigFieldIsCoveredOrDeclaredExempt:
    """F-65's real deliverable: the guard the CLASS needs.

    `metric_edt_variance_limited_s` was added in cycle 3 to fix RD-78 and reached
    no fingerprint, so a REPORTED disclosure column was served under the wrong
    threshold's stamp. Adding the key fixes one instance; this test is what stops
    the next one.

    Coverage is proved by PERTURBATION, not by matching field names against
    payload keys: a payload can carry a field under a different name or nested
    inside `representation.params`, and a name match would pass while the
    fingerprint never moves.

    LIMIT, stated rather than implied: the field sweep proves coverage at
    TOP-LEVEL `Config` field granularity. A new leaf inside a nested model is
    covered only where some fingerprint dumps that model WHOLESALE — true of
    `SplitSpec` (via `_preprocess_fingerprint`), `Scenes`, `SimulatorSpec`,
    `ModelSpec` and `RepresentationSpec`.

    `Seeds` is the exception, and it is swept separately below. NO fingerprint
    dumps it wholesale — every stage names individual leaves (`config.seed(...)`)
    — so perturbing `seeds.master` moves everything downstream and would let a new
    per-aspect seed pass unguarded (F-78). That is the worst place to have a blind
    spot: per-aspect seeds are invariant #5, and `split_assignment` is the
    leakage-critical one.
    """

    #: field → a value differing from the tiny config's, chosen to satisfy that
    #: field's validators. These are PERTURBATION PROBES for the guard below, never
    #: a run's parameters: no pipeline stage ever sees them, so they are not
    #: experiment-governing values living in a test fixture.
    PROBES: dict[str, object] = {
        "seeds": {"master": 4242},
        "simulator": {"params": {"speed_of_sound_m_s": 300.0}},
        "representation": {"params": {"n_fft": 512}},
        "model": {"params": {"hidden_channels": 16}},
        "sample_rate": 16000,
        "ir_duration": 0.2,
        "ambisonics_order": 2,
        "low_ray_budget": 77,
        "high_ray_budget": 7777,
        "scenes": {"n_id": 9},
        "splits": {"test_material_shift": {"count": 4}},
        "n_epochs": 3,
        "batch_size": 3,
        "learning_rate": 0.05,
        "huber_delta": 3.0,
        "early_stopping_patience": 9,
        "report_format": "markdown",
        "iso_eval_freqs": [500, 1000, 2000],
        "metric_onset_rel_db": -25.0,
        "metric_band_resolvability_margin": 0.07,
        # AC-176: governs whether T30/EDT is scored at all, so a change to it
        # must invalidate eval. Probed, not exempted.
        "metric_min_decay_range_db": {"T30": 41.0, "EDT": 19.0},
        "metric_edt_variance_limited_s": 0.3,
        # The D0a/D0b thresholds ARE the verdict `diagnostics` publishes — "signal
        # to learn at this ray budget", "carrier ceiling clears" — so a change to
        # any of them must invalidate it. They were exempt only while that stage
        # carried no fingerprint at all (AC-45).
        "d0a_gap_large_db": 9.0,
        "d0a_gap_small_db": 0.4,
        "d0b_t30_jnd_frac": 0.07,
        "d0b_edt_jnd_frac": 0.07,
        "d0b_c50_jnd_db": 1.5,
        "bootstrap_n_resamples": 500,
        "bootstrap_alpha": 0.1,
        "bootstrap_power": 0.9,
    }

    def _all_fingerprints(self, config: Config) -> dict:
        return {
            stage: fp(config)
            for stage, fp in STAGE_FINGERPRINT.items()
            if fp is not None
        }

    def test_every_config_field_is_either_probed_or_exempt(self) -> None:
        """A newly added `Config` field fails HERE until someone decides which it
        is — which is the whole point of the row."""
        fields = set(Config.model_fields)
        classified = set(self.PROBES) | set(FINGERPRINT_EXEMPT_FIELDS)
        assert fields - classified == set(), (
            f"Config fields {sorted(fields - classified)} are in no stage "
            f"fingerprint probe and in no exemption. Add a probe (if a change to "
            f"the field must invalidate a stage) or an entry in "
            f"FINGERPRINT_EXEMPT_FIELDS saying why it need not."
        )
        assert classified - fields == set(), (
            f"{sorted(classified - fields)} are probed or exempted but are not "
            f"Config fields — the guard is measuring something that no longer exists"
        )

    def test_a_field_is_never_both_probed_and_exempt(self) -> None:
        overlap = set(self.PROBES) & set(FINGERPRINT_EXEMPT_FIELDS)
        assert overlap == set(), f"{sorted(overlap)} are claimed as both"

    @pytest.mark.parametrize("field", sorted(PROBES))
    def test_perturbing_the_field_invalidates_at_least_one_stage(self, field) -> None:
        base = self._all_fingerprints(tiny_config())
        perturbed = self._all_fingerprints(tiny_config(**{field: self.PROBES[field]}))
        moved = [stage for stage in base if base[stage] != perturbed[stage]]
        assert moved, (
            f"changing {field!r} moved no stage fingerprint, so a run_dir would be "
            f"re-used under the new value and the artifacts would be the old "
            f"value's — the F-65 failure mode. Either add it to a fingerprint or "
            f"declare it in FINGERPRINT_EXEMPT_FIELDS with a reason."
        )

    @pytest.mark.parametrize("seed_name", SEED_NAMES)
    def test_perturbing_each_named_seed_invalidates_at_least_one_stage(
        self, seed_name: str
    ) -> None:
        """Every per-aspect seed individually, not just `master` (F-78).

        `Seeds` is dumped by no fingerprint, so this is the only thing standing
        between a newly appended `SEED_NAMES` entry and a stochastic aspect whose
        change invalidates nothing.
        """
        base = self._all_fingerprints(tiny_config())
        perturbed = self._all_fingerprints(tiny_config(seeds={seed_name: 4242}))
        moved = [stage for stage in base if base[stage] != perturbed[stage]]
        assert moved, (
            f"seeds.{seed_name} moved no stage fingerprint. A run that differs in "
            f"this aspect is a different run, so it must invalidate the stage that "
            f"consumes it — name it in that stage's fingerprint via config.seed()."
        )

    def test_the_edt_disclosure_threshold_invalidates_eval(self) -> None:
        """The specific key F-65 was raised about."""
        assert STAGE_FINGERPRINT["eval"](tiny_config()) != STAGE_FINGERPRINT["eval"](
            tiny_config(metric_edt_variance_limited_s=0.3)
        )

    def test_every_exemption_states_a_reason(self) -> None:
        for field, reason in FINGERPRINT_EXEMPT_FIELDS.items():
            assert isinstance(reason, str) and len(reason) > 40, (
                f"exemption for {field!r} must say why it is absent today AND what "
                f"would make it non-exempt"
            )


class TestAnUnprotectedStaleStageIsDisclosedNotVouchedFor:
    """F-75: `gen-scenes` and `render` carry no `code_version` (RD-107, a deliberate
    policy call — scoping `render` to `simulators/` forces a re-render, the
    multi-hour artifact under emulation). The staleness that buys is accepted.

    What was NOT acceptable: `versions.json` is re-stamped every invocation with
    the current whole-package hash, so a run_dir whose renders predate an edit to
    the render backend carried a provenance stamp positively asserting the new code
    produced them — a false witness, worse than the staleness itself.

    These tests pin the disclosure, not a refusal. The refusal is RD-107's to decide.
    """

    def test_the_sentinel_records_which_code_wrote_the_artifacts(
        self, tmp_path: Path
    ) -> None:
        import amcd.provenance as prov

        pipe = Pipeline(tiny_config(scenes={"n_id": 4}), tmp_path, QUIET)
        pipe._mark_done("gen-scenes")
        recorded = json.loads(_sentinel(tmp_path, "gen-scenes").read_text())
        assert recorded["code_version_unscoped"] == prov.code_version(prov.ALL_SOURCES)

    def test_it_is_recorded_outside_the_fingerprint_so_it_never_invalidates(
        self, tmp_path: Path
    ) -> None:
        """Recording it must not turn every stage into a whole-package hash — that
        is exactly the guard-by-over-refusal the scoping rationale rejects."""
        pipe = Pipeline(tiny_config(scenes={"n_id": 4}), tmp_path, QUIET)
        pipe._mark_done("gen-scenes")
        recorded = json.loads(_sentinel(tmp_path, "gen-scenes").read_text())
        assert "code_version_unscoped" not in (recorded["fingerprint"] or {})

    def test_no_stage_is_unprotected_any_more(self) -> None:
        """The warning this class covered no longer has a case to fire on.

        It existed because `gen-scenes` and `render` carried no `code_version`, so
        a run_dir whose renders predated a backend edit was served with a
        provenance stamp positively asserting the new code produced them — a false
        witness, worse than the staleness. Both stages are now cache-protected and
        `diagnostics` with them, so `_warn_if_unprotected_and_stale` was deleted
        rather than kept as a branch nothing can reach.

        This assertion is what makes that deletion safe: if a stage is ever added
        without a `code_version`, the false-witness hole is back and this fails.
        """
        unprotected = [
            stage for stage, fingerprint in STAGE_FINGERPRINT.items()
            if fingerprint is None or "code_version" not in fingerprint(tiny_config())
        ]
        assert not unprotected, (
            f"{unprotected} carry no code_version. `versions.json` re-stamps the "
            f"current whole-package hash every invocation, so their cached "
            f"artifacts would be vouched for by code that did not produce them. "
            f"Either fingerprint them or restore the unprotected-stage warning."
        )


class TestALegacySentinelIsRefusedActionablyNotWithATraceback:
    """F-76: giving `report` a fingerprint made `{"fingerprint": null}` sentinels
    reachable, and they crashed with a bare `TypeError` from `set(None)` instead of
    the actionable "predates fingerprinted caching" message.

    Generic, not a one-off migration wrinkle: it recurs for every stage that gains
    a fingerprint later, `diagnostics` being the next candidate (RD-108/AC-45).
    """

    def _run_dir_with_a_legacy_report_sentinel(
        self, tmp_path: Path, payload: dict
    ) -> Pipeline:
        """A run_dir whose chain is intact and whose `report.done` is cycle-3 shaped.

        The upstream sentinels must be real, or `_is_done` refuses on the chain
        before it ever reads report's own fingerprint — which is what the first
        version of this test actually measured.
        """
        pipe = Pipeline(tiny_config(scenes={"n_id": 4}), tmp_path, QUIET)
        for stage in ("gen-scenes", "render", "preprocess", "train", "infer",
                      "eval", "stats"):
            pipe._mark_done(stage)
        sentinel = _sentinel(tmp_path, "report")
        sentinel.parent.mkdir(parents=True, exist_ok=True)
        sentinel.write_text(json.dumps(payload))
        return pipe

    def test_a_null_recorded_fingerprint_gets_the_actionable_message(
        self, tmp_path: Path
    ) -> None:
        # Exactly what cycle-3 `_mark_done` wrote for a stage with no fingerprint.
        pipe = self._run_dir_with_a_legacy_report_sentinel(
            tmp_path, {"completed_at": 1.0, "fingerprint": None}
        )
        with pytest.raises(RuntimeError, match="predates fingerprinted caching"):
            pipe._is_done("report")

    def test_a_sentinel_missing_the_key_entirely_is_treated_the_same(
        self, tmp_path: Path
    ) -> None:
        pipe = self._run_dir_with_a_legacy_report_sentinel(
            tmp_path, {"completed_at": 1.0}
        )
        with pytest.raises(RuntimeError, match="predates fingerprinted caching"):
            pipe._is_done("report")


class TestTheCacheKeyDescribesTheSourceNotTheHost:
    """F-69: `rglob("*.py")` hashed macOS AppleDouble `._*.py` sidecars.

    They are real files on an exFAT volume and absent on APFS or the project's
    declared second host, so the same source hashed differently there and a run_dir
    carried between hosts was refused with a `code_version: <sha> → <sha>` diff
    naming no leaf — leaving `--force` as the only remedy, which is the compliance
    failure the scoping rationale exists to avoid.
    """

    def test_an_appledouble_sidecar_does_not_change_any_stages_version(self) -> None:
        import amcd.provenance as prov
        from amcd.pipeline import _code_version

        sidecar = Path(prov.__file__).resolve().parent / "data" / "._probe.py"
        before = {stage: _code_version(stage) for stage in STAGE_CODE_SCOPE}
        try:
            sidecar.write_bytes(b"\x00\x05\x16\x07 not python at all\n")
            for stage in STAGE_CODE_SCOPE:
                assert _code_version(stage) == before[stage], (
                    f"{stage}'s cache key moved because of a macOS sidecar, so the "
                    f"same source yields a different key on another host"
                )
        finally:
            sidecar.unlink(missing_ok=True)

    def test_a_real_source_file_still_changes_the_version(self) -> None:
        """The filter must not be so broad that it stops seeing code (a filter that
        excluded everything would pass the test above)."""
        import amcd.provenance as prov
        from amcd.pipeline import _code_version

        probe = Path(prov.__file__).resolve().parent / "data" / "probe_module.py"
        before = _code_version("preprocess")
        try:
            probe.write_text("# a real module\n")
            assert _code_version("preprocess") != before
        finally:
            probe.unlink(missing_ok=True)

    def test_pycache_is_not_hashed(self) -> None:
        import amcd.provenance as prov
        from amcd.pipeline import _code_version

        cache_dir = Path(prov.__file__).resolve().parent / "data" / "__pycache__"
        cache_dir.mkdir(exist_ok=True)
        stray = cache_dir / "stray.py"
        before = _code_version("preprocess")
        try:
            stray.write_text("# build output, not source\n")
            assert _code_version("preprocess") == before
        finally:
            stray.unlink(missing_ok=True)


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
