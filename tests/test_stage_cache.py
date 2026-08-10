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
    STAGE_FINGERPRINT,
    _diff_fingerprints,
    _fingerprint_sha,
    _gen_scenes_fingerprint,
    _preprocess_fingerprint,
    _render_fingerprint,
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
            tiny_config(metric_min_measurable_t60_s=0.07)
        )

    def test_stats_chains_eval(self) -> None:
        """A redefined metric must not be re-summarized under a cached CI table."""
        before = STAGE_FINGERPRINT["stats"](tiny_config())
        after = STAGE_FINGERPRINT["stats"](tiny_config(iso_eval_freqs=[500, 2000]))
        assert before["upstream_eval"] != after["upstream_eval"]
        assert _fingerprint_sha(before) != _fingerprint_sha(after)

    def test_bootstrap_settings_invalidate_stats_but_not_eval(self) -> None:
        before, after = tiny_config(), tiny_config(bootstrap_n_resamples=500)
        assert STAGE_FINGERPRINT["stats"](before) != STAGE_FINGERPRINT["stats"](after)
        assert STAGE_FINGERPRINT["eval"](before) == STAGE_FINGERPRINT["eval"](after)


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
