"""Stage runner: dispatch, timing, and fingerprinted stage caching.

Caching is what makes a long run resumable, and — because renders under x86
emulation are the most expensive artifact this pipeline produces — it is also the
easiest way to end up with a silently mixed dataset. A bare "this stage ran"
sentinel cannot tell whether it ran under the CURRENT config, so changing a
simulator parameter or a scene range and re-running the same run_dir would reuse
stale artifacts and produce a dataset that is part old, part new, with nothing on
disk recording the split (RD-16).

So each stage may declare the config inputs it depends on (`STAGE_FINGERPRINT`).
The sentinel stores that fingerprint, and a mismatch is a LOUD FAILURE, never a
silent re-use and never a silent re-run: only the operator can decide whether the
existing artifacts are salvageable.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Callable

from . import provenance
from .config import Config
from .runtime import Verbosity, emit

STAGES = ["gen-scenes", "render", "preprocess", "diagnostics", "train", "infer", "eval", "stats", "report"]


def _sentinel(run_dir: Path, stage: str) -> Path:
    return run_dir / "stages" / f"{stage.replace('-', '_')}.done"


# ─────────────────────────────────────────────────────────────────────────────
# Stage cache fingerprints
# ─────────────────────────────────────────────────────────────────────────────

def _gen_scenes_fingerprint(config: Config) -> dict:
    """Config inputs that determine the generated scene set — and whether it is
    ADMITTED.

    Covers the sampling ranges, the split set (shift splits carry their own
    counts, and their scenes are generated here) and every seed that feeds scene
    sampling. Change any of them and the scenes on disk are for a different
    experiment.

    Only the GENERATION-relevant split fields are included (F-50). A `SplitSpec`
    also carries `frac` and `role`, which are consumed at preprocess and cannot
    change a single generated scene — dumping the whole spec meant editing
    `train.frac` refused the cached `render` and forced a complete re-render of a
    dataset that provably had not changed. Failing safe, but the cost it imposed is
    precisely the cost RD-16/RD-30 built this machinery to avoid. `frac`/`role`
    still reach `_preprocess_fingerprint`, which carries the full dump.

    `ir_duration` and the SIMULATOR are here for a different reason (F-57): they
    do not change which scenes are sampled, they change whether gen-scenes SUCCEEDS
    and what it discloses. `ir_duration` is what AC-22's record-length gate compares
    against, and the simulator declares the minimum source-receiver separation
    AC-13's pre-flight check enforces. Without them the record-length gate was
    bypassable through the cache, and `placement_report.json` could stay stamped
    with a record length its own `config.yaml` contradicted.
    """
    return {
        "scenes": config.scenes.model_dump(),
        "splits": {
            name: {"count": sp.count, "seed": sp.seed, "axes": sp.axes}
            for name, sp in config.splits.items()
        },
        "seed_scene_generation": config.seed("scene_generation"),
        "ir_duration": config.ir_duration,
        "simulator": {"name": config.simulator.name, "params": config.simulator.params},
    }


def _render_fingerprint(config: Config) -> dict:
    """Config inputs that determine the rendered IR pair for a given scene.

    `simulator.params` carries the pinned upstream `commit_sha`, so a GSound-SIR
    version change invalidates the cache with no extra wiring.
    """
    return {
        "simulator": {"name": config.simulator.name, "params": config.simulator.params},
        "sample_rate": config.sample_rate,
        "n_samples": config.n_samples,
        "ambisonics_order": config.ambisonics_order,
        "low_ray_budget": config.low_ray_budget,
        "high_ray_budget": config.high_ray_budget,
    }


def _preprocess_fingerprint(config: Config) -> dict:
    """Config inputs — AND the code version — that determine the split assignment
    and the encoded tensors.

    `split_assignment` belongs here specifically (F-29): it is the most
    leakage-critical value in the project, it is consumed HERE rather than at
    gen-scenes, and without it in a fingerprint, repinning the split seed on an
    existing run_dir was a complete no-op — splits.json kept the old assignment
    while config.yaml stamped the new seed, a provenance lie.

    `code_version` closes the same hole against a CODE change (F-64): config keys
    alone left the encoder, the normalization statistics and that same
    leakage-critical split assignment reachable through the cache, and the stage
    that DID refuse was `train`, so following the error rebuilt the wrong thing.
    """
    return {
        "splits": {name: sp.model_dump() for name, sp in config.splits.items()},
        "seed_split_assignment": config.seed("split_assignment"),
        "representation": {
            "name": config.representation.name,
            "params": config.representation.params,
        },
        "sample_rate": config.sample_rate,
        "n_samples": config.n_samples,
        "ambisonics_order": config.ambisonics_order,
        "code_version": _code_version("preprocess"),
    }


def _train_fingerprint(config: Config) -> dict:
    """Config inputs that determine the trained weights.

    RD-41 deferred this on COST grounds — "the expensive artifact is the render" —
    which F-53 showed is the wrong test; see STAGE_UPSTREAM, which owns that story.

    Both training seeds are named individually (inv #5): weight init and data
    shuffle are separate aspects and a run that differs in either is a different
    run.
    """
    return {
        "model": {"name": config.model.name, "params": config.model.params},
        "n_epochs": config.n_epochs,
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
        "huber_delta": config.huber_delta,
        "early_stopping_patience": config.early_stopping_patience,
        "seed_weight_init": config.seed("weight_init"),
        "seed_data_shuffle": config.seed("data_shuffle"),
        "representation": {
            "name": config.representation.name,
            "params": config.representation.params,
        },
        "code_version": _code_version("train"),
    }


def _infer_fingerprint(config: Config) -> dict:
    """Config inputs that determine the predictions.

    Inference re-instantiates the model from `model.name`/`params` to load the
    checkpoint, and decodes through the representation, so both are inputs even
    though neither is re-trained here. Everything else it depends on arrives
    through the chain from `train` (F-53).
    """
    return {
        "model": {"name": config.model.name, "params": config.model.params},
        "representation": {
            "name": config.representation.name,
            "params": config.representation.params,
        },
        "code_version": _code_version("infer"),
    }


def _eval_fingerprint(config: Config) -> dict:
    """Config inputs — AND the code version — that determine the reported metrics.

    RD-54 promoted this out of RD-41's DEFERRED set because the metric path is
    actively in flux. RD-59 then established that config keys alone would close
    RD-54 with a mechanism blind to its own trigger case: the AC-17 shared
    Schroeder window was a CODE change in `evaluation/room_acoustic.py` that moved
    no config key (see STAGE_CODE_SCOPE). Hence `code_version` — a cached
    `metrics.parquet` written under a different revision of the metric code is
    refused, not silently reported.

    Erring coarse is right for the one stage whose output IS the research claim,
    and eval is cheap next to render.
    """
    return {
        "iso_eval_freqs": list(config.iso_eval_freqs),
        "metric_onset_rel_db": config.metric_onset_rel_db,
        "metric_band_resolvability_margin": config.metric_band_resolvability_margin,
        # Governs a REPORTED disclosure column (F-65); the class-level guard
        # against the next such key is FINGERPRINT_EXEMPT_FIELDS below.
        "metric_edt_variance_limited_s": config.metric_edt_variance_limited_s,
        "representation": {
            "name": config.representation.name,
            "params": config.representation.params,
        },
        "sample_rate": config.sample_rate,
        "code_version": _code_version("eval"),
    }


def _stats_fingerprint(config: Config) -> dict:
    """Config inputs that determine the CIs and MDES.

    Its dependence on eval is expressed by `STAGE_UPSTREAM["stats"] = "eval"`, NOT
    by a key here. An earlier version carried
    `"upstream_eval": _fingerprint_sha(_eval_fingerprint(config))` — a chain
    RECOMPUTED FROM THE CURRENT CONFIG, which is what the STAGE_UPSTREAM docstring
    below forbids (F-54). Recomputing let `--force` stamp a chain describing
    artifacts that did not exist, and cost the recursive leaf diff F-49 bought.

    `code_version` covers the code that COMPUTES the reported CIs (F-63). `stats`
    is in no other stage's scope, so nothing upstream can notice an edit to
    `bootstrap_ci` on its behalf. Scope is ("stats", "evaluation") because
    `stats/aggregate.py` imports `evaluation.metric_row.paired_improvement` — the
    improvement values the CIs are taken over.
    """
    return {
        "bootstrap_n_resamples": config.bootstrap_n_resamples,
        "bootstrap_alpha": config.bootstrap_alpha,
        "bootstrap_power": config.bootstrap_power,
        "seed_bootstrap": config.seed("bootstrap"),
        "code_version": _code_version("stats"),
    }


def _report_fingerprint(config: Config) -> dict:
    """Config inputs — AND the code version — that determine the reported tables.

    RD-41 left this unwired as terminal, "costing only its own re-use". Its own
    re-use IS `report/summary.txt`, the artifact a reader treats as the result
    (F-63). `STAGE_UPSTREAM["report"] = "stats"` carries the other half: a
    fingerprint over this stage's own inputs cannot see the numbers it formats.
    """
    return {
        "report_format": config.report_format,
        "code_version": _code_version("report"),
    }


#: Per stage, the package sources its OUTPUT is a function of — the code half of a
#: fingerprint (F-55). `provenance.code_version` adds the core modules (config,
#: seeds, registry, shared acoustics, the package `__init__`) to every scope, so
#: these list only what is specific to the stage. An entry is a subpackage
#: (`"evaluation"`), a top-level module (`"config.py"`) or a nested module
#: (`"simulators/base.py"`) when a stage depends on one file of another subtree.
#:
#: WHY A CONFIG FINGERPRINT IS NOT ENOUGH — stated once, here, and cited from the
#: three other places that used to restate it. A fingerprint built from config
#: keys sees CONFIG changes only. A code change — a redefined metric, a different
#: sampling rule — moves nothing in those payloads, and the cached artifact is
#: served under the new code (RD-59). Every stage whose output reaches the
#: reported result therefore carries `code_version()` as well.
#:
#: Why the scope is DECLARED per stage rather than a hash over the whole package:
#: see `provenance.code_version`, which owns that argument.
#:
#: The residual risk of declaring is a scope that omits a real dependency, and it
#: fails SILENTLY. `tests/test_stage_cache.py` therefore asserts each declared
#: scope against the stage's transitive import closure; read that test's docstring
#: for what it does and does not catch (F-66).
#:
#: Six of the nine STAGES are here. `gen-scenes`, `render` and `diagnostics` are
#: absent BY DECISION, not oversight: they carry no `code_version` at all, so they
#: have nothing to scope — see STAGE_FINGERPRINT, RD-99 and RD-100 for the cost
#: that buys and what it leaves exposed.
STAGE_CODE_SCOPE: dict[str, tuple[str, ...]] = {
    # Assigns splits and encodes tensors. `simulators/base.py` is named as a
    # single module rather than the whole subpackage because `data/preprocess.py`
    # depends on exactly one thing there — `SceneSpec`, the render manifest it
    # reads — and scoping the backend implementations would invalidate the
    # encoded dataset whenever an unrelated raytracer detail moved (F-64).
    "preprocess": ("data", "representations", "simulators/base.py"),
    # Trains weights: the model, the loss, the trainer, the dataset it reads, and
    # the representation whose domain the loss is expressed in.
    "train": ("training", "models", "data", "representations"),
    # Loads the checkpoint, decodes through the representation, and denormalizes
    # every predicted leg — hence `data` (F-66).
    "infer": ("training", "models", "data", "representations"),
    # The stage whose output IS the research claim. `data` because it denormalizes
    # every reported leg (F-66) — previously omitted here and from infer's scope,
    # masked only by `data` being in TRAIN's scope so the chain refused upstream
    # first, a coincidence of ordering rather than a guarantee.
    # `representations` is a DECLARED JUDGEMENT, not an import: eval measures the
    # decoded waveform that `infer` wrote, and the representation that produced it
    # is resolved by NAME through the registry, which the closure test cannot see
    # (AC-47 corrected an earlier claim here that "eval decodes before measuring").
    "eval": ("evaluation", "representations", "data"),
    # Computes the reported CIs and MDES over `evaluation.metric_row`'s paired
    # improvements, which is why `evaluation` is in scope and not just `stats`.
    "stats": ("stats", "evaluation"),
    # Formats the reported tables (F-63; see `_report_fingerprint`).
    "report": ("reporting",),
}


def _code_version(stage: str) -> str:
    """The content hash of `stage`'s declared sources — see
    `amcd.provenance.code_version`.

    Shared with `Config.stamp` through that module so the cache key and
    `versions.json` cannot describe different code (F-56).
    """
    return provenance.code_version(STAGE_CODE_SCOPE[stage])


#: Per-stage declaration of the config inputs a cached artifact depends on.
#:
#: `None` means "no fingerprint declared yet" — the stage caches on the bare
#: sentinel, exactly as before. Every stage is listed explicitly, including the
#: `None`s, so an unwired stage is DECLARED rather than silently absent (the same
#: rule docs/verbosity.md applies to verbosity wiring).
#:
#: Which stages carry `code_version()` as well as config keys: every stage from
#: `preprocess` through `report`, i.e. everything whose output reaches the
#: reported table. `gen-scenes` and `render` do not, so the remedy for a code
#: change upstream of preprocess is still a fresh run_dir or `--force`.
#: See STAGE_CODE_SCOPE for why config keys alone are not enough.
#:
#: The one remaining `None` is `diagnostics`, on RD-41's "terminal, so an unwired
#: fingerprint costs only its own re-use" — the same argument that turned out to be
#: wrong for `report` (see `_report_fingerprint`), and which RD-100/AC-45 dispute
#: here too: a stale D0b carrier-ceiling verdict is a false clearance.
STAGE_FINGERPRINT: dict[str, Callable[[Config], dict] | None] = {
    "gen-scenes": _gen_scenes_fingerprint,
    "render": _render_fingerprint,
    "preprocess": _preprocess_fingerprint,
    "diagnostics": None,
    "train": _train_fingerprint,
    "infer": _infer_fingerprint,
    "eval": _eval_fingerprint,
    "stats": _stats_fingerprint,
    "report": _report_fingerprint,
}

#: Which stage's artifacts each stage consumes. The chain is what makes a
#: fingerprint transitive: renders are renders OF scenes, so a render is stale
#: when the SCENES changed even if every simulator parameter is untouched.
#:
#: Chaining reads the upstream SENTINEL's recorded fingerprint, never a
#: recomputation from the current config (F-26). Recomputing made `--force` a
#: laundering step: forcing a downstream stage stamped the CURRENT config's chain
#: value even though the stage had consumed the OLD upstream artifacts, leaving a
#: run_dir whose chain validated while the artifacts disagreed — precisely the
#: silent mixed dataset this machinery exists to prevent.
#: The chain runs unbroken from the scenes to the reported TABLE:
#:
#:   gen-scenes → render → preprocess → train → infer → eval → stats → report
#:
#: Before F-53 it stopped at `preprocess`, so a changed model was reported under a
#: stale checkpoint's numbers. Wiring the chain is what makes a fingerprint
#: transitive: `report` is stale when the SCENES changed, through seven links,
#: without `_report_fingerprint` naming a single scene key.
#:
#: The last link is `stats → report` (F-63; see `_report_fingerprint` for why
#: "terminal" was true of the graph and false of the consequence).
#:
#: `diagnostics` is the only unchained stage left, and consumes nothing a later
#: stage reports.
STAGE_UPSTREAM: dict[str, str | None] = {
    "gen-scenes": None,
    "render": "gen-scenes",
    "preprocess": "render",
    "diagnostics": None,
    "train": "preprocess",
    "infer": "train",
    "eval": "infer",
    "stats": "eval",
    "report": "stats",
}


#: `Config` fields that are deliberately in NO stage fingerprint, each with the
#: condition that would END the exemption.
#:
#: The row this closes (F-65) was not a missing key, it was a missing RULE: a
#: threshold added in one cycle reached no fingerprint, and nothing could tell
#: that from a field that is absent on purpose. `tests/test_stage_cache.py` makes
#: the two distinguishable — every `Config` field must either move a fingerprint
#: when perturbed, or be named here — so the next added key cannot repeat it.
#:
#: Shared reason for the five D0a/D0b thresholds in the table below.
_DIAGNOSTICS_EXEMPTION = (
    "Consumed only by `diagnostics`, which declares no fingerprint at all "
    "(STAGE_FINGERPRINT). The exemption is a pointer to that open hole, not a "
    "judgement that the field is unimportant — the D0a verdict is a research gate "
    "on the ray-count question. Non-exempt as soon as `diagnostics` is wired, and "
    "these belong in its fingerprint."
)

#: An entry states why the field is absent TODAY and what makes it non-exempt,
#: because a present-tense fact ("nothing consumes it") reads to a later editor as
#: permission to delete the field. `tests/test_config.py` enforces that shape: each
#: reason must contain "Non-exempt", or say the field is "fingerprinted through"
#: another key.
FINGERPRINT_EXEMPT_FIELDS: dict[str, str] = {
    "run_id": (
        "Experiment-ledger label with no pipeline consumer (config.py `run_id`). "
        "Fingerprinting it would discard an expensive artifact on a relabel. "
        "Non-exempt if any stage ever reads it to decide what to compute."
    ),
    "resolved_roles": (
        "Not an input: role metadata recorded by `Config.load` and derived from "
        "the same declared tree the resolved scalars come from. The values it "
        "describes are fingerprinted through the fields themselves, and `splits` "
        "is dumped in full by `_preprocess_fingerprint`."
    ),
    "max_onset_ms": (
        "Render-stage QC gate (configs/base.yaml §QC thresholds) that the real "
        "gsound_sir backend will enforce; the dry_run scaffold's synthetic IRs "
        "are clean by construction so nothing reads it yet. Non-exempt the "
        "moment the real backend enforces it — it belongs in "
        "`_render_fingerprint`, because it decides which renders are ADMITTED."
    ),
    "min_energy_db": (
        "Render-stage QC gate like `max_onset_ms`, declared in configs/base.yaml "
        "for the real gsound_sir backend and unread by the dry_run scaffold. "
        "Non-exempt when that backend enforces it, and it belongs in "
        "`_render_fingerprint` for the same reason: it decides which renders are "
        "ADMITTED."
    ),
    "d0a_gap_large_db": _DIAGNOSTICS_EXEMPTION,
    "d0a_gap_small_db": _DIAGNOSTICS_EXEMPTION,
    "d0b_t30_jnd_frac": _DIAGNOSTICS_EXEMPTION + (
        " ALSO: this JND is the calibration criterion behind "
        "`metric_band_resolvability_margin`, which IS an eval fingerprint key — "
        "base.yaml derives the margin from where the 500 Hz T30 estimator's bias "
        "crosses 0.05. No code path, so no cache hole, but moving this without "
        "re-deriving the margin leaves a fingerprinted constant justified by a "
        "number that no longer exists (AC-46)."
    ),
    "d0b_edt_jnd_frac": _DIAGNOSTICS_EXEMPTION,
    "d0b_c50_jnd_db": _DIAGNOSTICS_EXEMPTION,
}


def _normalize(payload: dict) -> dict:
    """Round-trip a fingerprint through JSON so it compares equal to a stored one.

    Not cosmetic: `model_dump()` yields tuples, and JSON has no tuple type, so a
    freshly computed fingerprint would never equal the one read back from a
    sentinel — every cached stage would raise a spurious "config changed". The
    stored form is the canonical form, so compare in it.
    """
    return json.loads(json.dumps(payload, sort_keys=True, default=str))


def _fingerprint_sha(payload: dict) -> str:
    return hashlib.sha256(
        json.dumps(_normalize(payload), sort_keys=True).encode()
    ).hexdigest()


def _diff_fingerprints(old: dict, new: dict, prefix: str = "") -> list[str]:
    """Leaf-level differences, so the error says WHAT changed (RD-35).

    A bare sha mismatch forces the operator to decide blind whether an expensive
    renders/ directory is salvageable; naming the changed field is the difference
    between a five-second judgement and a re-render under emulation.

    RECURSES into nested dicts and reports dotted paths (F-49). A top-level-only
    comparison degenerated on exactly the two fields most likely to change —
    `scenes` and `splits`, the only nested payloads — printing the whole ~700
    character dict twice with the single changed value buried inside, which fails
    the stated purpose above in the one case it was written for.
    """
    lines = []
    for key in sorted(set(old) | set(new)):
        before, after = old.get(key, _ABSENT), new.get(key, _ABSENT)
        if before == after:
            continue
        path = f"{prefix}{key}"
        if isinstance(before, dict) and isinstance(after, dict):
            lines.extend(_diff_fingerprints(before, after, prefix=f"{path}."))
        else:
            lines.append(
                f"    {path}: {_render_leaf(before)} → {_render_leaf(after)}"
            )
    return lines


#: Distinguishes "key absent" from a key whose value happens to be the string
#: "<absent>", so a diff cannot misreport which side a key is missing from.
_ABSENT = object()


def _render_leaf(value) -> str:
    return "<absent>" if value is _ABSENT else repr(value)


def _dispatch(stage: str) -> Callable[[Config, Path, Verbosity], None]:
    if stage == "gen-scenes":
        from .scenes.generator import run_gen_scenes
        return run_gen_scenes
    if stage == "render":
        from .simulators.render import run_render
        return run_render
    if stage == "preprocess":
        from .data.preprocess import run_preprocess
        return run_preprocess
    if stage == "diagnostics":
        from .diagnostics.probe import run_diagnostics
        return run_diagnostics
    if stage == "train":
        from .training.trainer import run_train
        return run_train
    if stage == "infer":
        from .training.infer import run_infer
        return run_infer
    if stage == "eval":
        from .evaluation.evaluator import run_eval
        return run_eval
    if stage == "stats":
        from .stats.aggregate import run_stats
        return run_stats
    if stage == "report":
        from .reporting.tables import run_report
        return run_report
    raise ValueError(f"Unknown stage: {stage!r}")


class Pipeline:
    def __init__(self, config: Config, run_dir: Path, verbosity: Verbosity, force: bool = False) -> None:
        self.config = config
        self.run_dir = run_dir
        self.verbosity = verbosity
        self.force = force

    def _recorded_fingerprint(self, stage: str) -> dict | None:
        """The fingerprint stored in `stage`'s sentinel, or None if it has not run."""
        sentinel = _sentinel(self.run_dir, stage)
        if not sentinel.exists():
            return None
        try:
            return json.loads(sentinel.read_text()).get("fingerprint")
        except (json.JSONDecodeError, AttributeError):
            return None

    def _effective_fingerprint(self, stage: str) -> dict:
        """This stage's own config inputs, plus the upstream sentinel's RECORDED
        fingerprint (F-26).

        Taking the upstream value from disk rather than recomputing it from the
        current config is the whole point: it describes the artifacts this stage
        actually consumed, so a `--force` cannot launder a stale upstream into a
        chain that validates.
        """
        own = _normalize(STAGE_FINGERPRINT[stage](self.config))
        upstream = STAGE_UPSTREAM.get(stage)
        if upstream is None:
            return own
        # An upstream that declares no fingerprint cannot anchor a chain: its
        # sentinel records `null`, which is indistinguishable from "never ran".
        # No STAGE_UPSTREAM value points at such a stage today, but RD-41 exists to
        # wire the remaining seven, and the first chain to cross one would
        # otherwise report "has not completed" for a stage that did (F-41).
        if STAGE_FINGERPRINT[upstream] is None:
            raise RuntimeError(
                f"Stage {stage!r} declares {upstream!r} as its upstream, but "
                f"{upstream!r} has no fingerprint in STAGE_FINGERPRINT, so its "
                f"sentinel cannot say WHICH config its artifacts belong to. Give "
                f"{upstream!r} a fingerprint before chaining to it."
            )
        recorded = self._recorded_fingerprint(upstream)
        if recorded is None:
            raise RuntimeError(
                f"Stage {stage!r} depends on {upstream!r}, which has not completed "
                f"in {self.run_dir} (no readable fingerprinted sentinel). Run "
                f"{upstream!r} first — running {stage!r} now would record a "
                f"provenance chain for artifacts that do not exist."
            )
        # The upstream artifacts must ALSO be current for this config. Recursing
        # here (rather than only comparing our own inputs) is what catches a stale
        # ancestor when a downstream stage is run on its own: `amcd render` after
        # editing a room dimension would otherwise reuse renders that match the
        # OLD scene specs still sitting on disk.
        upstream_expected = self._effective_fingerprint(upstream)
        if recorded != upstream_expected:
            diff = _diff_fingerprints(recorded, upstream_expected) or [
                "    (nested value changed)"
            ]
            raise RuntimeError(
                f"Stage {stage!r} cannot run: its upstream stage {upstream!r} holds "
                f"artifacts built under a DIFFERENT config.\n"
                f"  Changed inputs to {upstream!r}:\n" + "\n".join(diff) + "\n"
                f"  Re-run {upstream!r} (with --force) before {stage!r}, or use a "
                f"fresh --run-dir."
            )
        return {"upstream": {upstream: _fingerprint_sha(recorded)}, **own}

    def _is_done(self, stage: str) -> bool:
        """Whether `stage`'s cached artifacts are valid for the CURRENT config.

        Raises rather than returning False on a fingerprint mismatch: silently
        re-running would overwrite artifacts the operator may still need, and
        silently skipping would mix two experiments in one run_dir. Both failure
        modes are invisible after the fact, so the decision belongs to a human.
        """
        sentinel = _sentinel(self.run_dir, stage)
        if self.force or not sentinel.exists():
            return False

        if STAGE_FINGERPRINT[stage] is None:
            return True  # stage declares no config dependency; bare sentinel

        expected = self._effective_fingerprint(stage)
        try:
            recorded = json.loads(sentinel.read_text())
            found = recorded["fingerprint"]
        except (json.JSONDecodeError, TypeError, KeyError):
            found = None
        # A RECORDED `null` is the realistic legacy shape, not a corrupt file: it is
        # exactly what `_mark_done` wrote for a stage that declared no fingerprint
        # at the time. So it must reach the same actionable message as a sentinel
        # with the key absent — before F-75 it fell through to `_diff_fingerprints`,
        # which did `set(None)` and raised a bare TypeError with a traceback. This
        # recurs for EVERY stage that gains a fingerprint later (`diagnostics` next,
        # RD-100/AC-45), so it is guarded here rather than at the call site.
        if found is None:
            raise RuntimeError(
                f"Stage {stage!r} has a cached sentinel with no fingerprint "
                f"({sentinel}). It predates fingerprinted caching for this stage, so "
                f"whether its artifacts match the current config cannot be "
                f"established. Re-run with --force to rebuild {stage!r} (this "
                f"discards its existing artifacts), or use a fresh --run-dir."
            ) from None

        if found != expected:
            diff = _diff_fingerprints(found, expected) or ["    (nested value changed)"]
            raise RuntimeError(
                f"Stage {stage!r} was cached under a DIFFERENT config; reusing it "
                f"would silently mix two experiments in one run_dir.\n"
                f"  Changed inputs:\n" + "\n".join(diff) + "\n"
                f"  Re-run with --force to rebuild {stage!r} (this discards its "
                f"existing artifacts), or use a fresh --run-dir to keep both."
            )
        return True

    def _mark_done(self, stage: str) -> None:
        # Sentinels are functional (caching input), never verbosity-gated (F-23).
        # The recorded fingerprint chains the UPSTREAM SENTINEL, so it describes the
        # artifacts this run actually consumed — including under --force (F-26).
        payload = {
            "completed_at": time.time(),
            # Full payload, not just the sha: the mismatch error must be able to
            # name the field that changed (RD-35).
            "fingerprint": (
                self._effective_fingerprint(stage)
                if STAGE_FINGERPRINT[stage] else None
            ),
            # OUTSIDE `fingerprint`, so it is recorded and never compared: this is
            # the whole-package hash of the code that actually produced these
            # artifacts. For `gen-scenes` and `render` — which carry no scoped
            # `code_version` — it is the ONLY record on disk of which code made
            # them, and it is what lets `_warn_if_unprotected_and_stale` tell the
            # operator that a cached artifact predates the current source (F-75).
            "code_version_unscoped": provenance.code_version(provenance.ALL_SOURCES),
        }
        s = _sentinel(self.run_dir, stage)
        s.parent.mkdir(parents=True, exist_ok=True)
        s.write_text(json.dumps(payload, indent=2, default=str))

    def _warn_if_unprotected_and_stale(self, stage: str) -> None:
        """Say so when a stage the cache CANNOT protect is served under newer code.

        `gen-scenes` and `render` declare no `code_version` (RD-99), deliberately:
        scoping `render` to `simulators/` would force a full re-render — the
        multi-hour artifact under x86 emulation — on any backend edit. The cost of
        that decision is that a code change to the IR synthesis, the placement
        sampling or the render QC gates leaves these stages cached with no refusal.

        What is NOT acceptable is that being invisible. `versions.json` is
        re-stamped every invocation with the CURRENT whole-package hash, so a
        run_dir whose renders predate an edit to `simulators/dry_run.py` carried a
        provenance stamp positively asserting the new code produced them — a false
        witness, and worse than the staleness itself (F-75).

        So: for stages with no scoped `code_version`, compare the whole-package
        hash recorded when the artifacts were WRITTEN against the current one, and
        warn on stderr when they differ. Deliberately a warning, not a refusal —
        the refusal is the policy call in RD-99, and this must not quietly make it.

        Fingerprinted stages are skipped: a scoped code change already refuses
        them, so a whole-package drift there is expected and would be pure noise.
        """
        fingerprint = STAGE_FINGERPRINT[stage]
        if fingerprint is not None and "code_version" in fingerprint(self.config):
            return
        try:
            recorded = json.loads(_sentinel(self.run_dir, stage).read_text())
        except (json.JSONDecodeError, OSError):
            return
        was_built_with = recorded.get("code_version_unscoped")
        if was_built_with is None:
            # Written before this key existed. Absence is not evidence of a match,
            # so say that rather than implying agreement.
            emit(
                self.verbosity, "warning",
                f"[warn ] {stage} is cached and carries no code_version. Its "
                f"sentinel predates provenance recording for unprotected stages, "
                f"so which code produced its artifacts is UNKNOWN."
            )
            return
        current = provenance.code_version(provenance.ALL_SOURCES)
        if was_built_with != current:
            emit(
                self.verbosity, "warning",
                f"[warn ] {stage} is cached and the package source has CHANGED "
                f"since its artifacts were written "
                f"({was_built_with[:12]} → {current[:12]}), but {stage!r} declares "
                f"no code_version, so nothing refuses it (RD-99). Its artifacts may "
                f"predate the current code; versions.json describes THIS "
                f"invocation, not what produced them. Use --force or a fresh "
                f"--run-dir if the change affects {stage!r}."
            )

    def _record_timing(self, stage: str, seconds: float) -> None:
        # Per-stage wall time is provenance (RD-09): written from save level 1.
        if not self.verbosity.saves("provenance"):
            return
        path = self.run_dir / "timings.json"
        timings = json.loads(path.read_text()) if path.exists() else {}
        timings[stage] = seconds
        path.write_text(json.dumps(timings, indent=2))

    def run_stage(self, stage: str) -> None:
        if stage not in STAGES:
            raise ValueError(f"Unknown stage {stage!r}. Valid: {STAGES}")

        t0 = time.time()
        try:
            # Validate the provenance chain BEFORE doing any work, so a stale
            # upstream is refused up front rather than after an expensive render,
            # and so --force cannot run the stage and only then fail to record its
            # chain. Inside the try so the failure still reaches stderr (F-24).
            if STAGE_FINGERPRINT[stage] is not None:
                self._effective_fingerprint(stage)

            if self._is_done(stage):
                self._warn_if_unprotected_and_stale(stage)
                emit(self.verbosity, "progress", f"[skip] {stage} (cached)")
                return

            # INVALIDATE BEFORE MUTATING (F-58). The stage is about to overwrite
            # its artifacts, so from this moment the old sentinel describes a
            # directory that no longer exists. Leaving it in place until success
            # meant a stage killed PART-WAY THROUGH WRITING left the PREVIOUS run's
            # success sentinel standing over half-new artifacts, and the next
            # invocation printed `[skip] (cached)` against it. The costly instance
            # is render, where a kill mid-run leaves the tail of the scene range
            # holding the previous config's output under a sentinel that validates
            # — the silently mixed dataset RD-16 exists to prevent, and the
            # multi-hour case under emulation.
            _sentinel(self.run_dir, stage).unlink(missing_ok=True)

            emit(self.verbosity, "progress", f"\n[run ] {stage}")
            fn = _dispatch(stage)
            fn(self.config, self.run_dir, self.verbosity)
        except Exception as exc:
            emit(self.verbosity, "error", f"[FAIL] {stage}: {exc}")
            raise

        self._mark_done(stage)
        elapsed = time.time() - t0
        self._record_timing(stage, elapsed)
        emit(self.verbosity, "timing", f"[done] {stage} ({elapsed:.1f}s)")

    def run_all(self) -> None:
        for stage in STAGES:
            self.run_stage(stage)
