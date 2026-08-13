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
from .simulators.base import simulator_code_scope, simulator_host_scoped_params
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
        "code_version": _code_version("gen-scenes", config),
        "simulator": {"name": config.simulator.name,
                      "params": _dataset_simulator_params(config)},
    }


#: Simulator params that change what is REPORTED about an IR and never the IR.
#: Declared here rather than on the backend: a backend's `host_scoped_params()`
#: answers "is this a machine fact", which is a different question, and folding
#: these in would redact them from canonical provenance where they belong (F-82).
_DISCLOSURE_ONLY_PARAMS: frozenset[str] = frozenset({"max_discarded_tail_db"})


def _dataset_simulator_params(config: Config) -> dict:
    """`config.simulator.params` minus the entries that describe the HOST or the
    DISCLOSURE, not the IR.

    Both fingerprints below hashed the params block whole, which made the render
    cache identity depend on things that cannot change a single sample:

    * **Host-scoped params** (`render_python` — the x86 interpreter the emulated
      render runs under). Probe: `base.yaml` alone fingerprints to 6999713b8247…,
      `base.yaml` + a host layer to 51033e7d57c0…, differing only in
      `simulator.params.render_python`. The same dataset rendered on this Mac would
      fail loudly on the x86_64 Linux host demanding a byte-identical re-render —
      a direct violation of the cross-platform requirement. The backend already
      DECLARES which of its params are host-scoped, and `render._canonical_meta`
      already redacts them from provenance; this asks the same declaration.
    * **Disclosure-only thresholds** (`max_discarded_tail_db`). Re-tightening a QC
      threshold changes what is REPORTED about an IR, never the IR, so it must not
      cost a multi-hour emulated re-render. It is deliberately NOT folded into the
      backend's host-scoped declaration: it is not a host fact, and conflating the
      two would redact it from provenance, where it belongs.

    Everything else stays: `commit_sha` in particular, so an upstream version change
    still invalidates the cache with no extra wiring.
    """
    excluded = set(simulator_host_scoped_params(config)) | _DISCLOSURE_ONLY_PARAMS
    return {k: v for k, v in config.simulator.params.items() if k not in excluded}


def _render_fingerprint(config: Config) -> dict:
    """Config inputs that determine the rendered IR pair for a given scene.

    `simulator.params` carries the pinned upstream `commit_sha`, so a GSound-SIR
    version change invalidates the cache with no extra wiring.
    """
    return {
        "code_version": _code_version("render", config),
        "simulator": {"name": config.simulator.name,
                      "params": _dataset_simulator_params(config)},
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


def _diagnostics_fingerprint(config: Config) -> dict:
    """Config inputs — and the code version — that determine the D0a/D0b verdicts.

    D0b's output is not a threshold report, it is a PHYSICAL VERDICT: "CARRIER
    CEILING CLEARS ... proceed to E1" against "CARRIER BOTTLENECK", produced by
    comparing measured T30/EDT/C50 residuals with JND tolerances. With no
    fingerprint, `_is_done` returned True on the bare sentinel, so NOTHING
    invalidated it — measured on a complete run_dir, `diagnostics` was served from
    cache under doubled `ir_duration`, changed `ambisonics_order`, changed
    low/high ray budgets and changed `sample_rate`, every one of which moves the
    residuals being compared. A stale clearance is a false clearance of the
    project's own premise, which is worse than no clearance at all.

    The scope reaches past `diagnostics/` because the probe measures through the
    metric path: a change to the Schroeder window or the octave filter moves the
    verdict without touching this package.
    """
    return {
        "code_version": _code_version("diagnostics"),
        "d0a_gap_large_db": config.d0a_gap_large_db,
        "d0a_gap_small_db": config.d0a_gap_small_db,
        "d0b_t30_jnd_frac": config.d0b_t30_jnd_frac,
        "d0b_edt_jnd_frac": config.d0b_edt_jnd_frac,
        "d0b_c50_jnd_db": config.d0b_c50_jnd_db,
        "d0b_min_scored_frac": config.d0b_min_scored_frac,
        "d0b_level_sweep_db": list(config.d0b_level_sweep_db),
        "iso_eval_freqs": list(config.iso_eval_freqs),
        "metric_onset_rel_db": config.metric_onset_rel_db,
        "metric_onset_tolerance_ms": config.metric_onset_tolerance_ms,
        "metric_band_resolvability_margin": config.metric_band_resolvability_margin,
        "metric_edt_variance_limited_s": config.metric_edt_variance_limited_s,
        "metric_min_decay_range_db": dict(config.metric_min_decay_range_db),
        # THE FILTER IS THE INSTRUMENT (F-143). `order` sets both the out-of-band
        # rejection and the ringing floor every reported ISO metric is measured
        # against, so a run under a different order reports different numbers from
        # the same waveforms. `stopband_rejection_db` is the declared consequence
        # and moves with it.
        "metric_octave_filter": config.metric_octave_filter.model_dump(mode="json"),
        "sample_rate": config.sample_rate,
        "n_samples": config.n_samples,
        "ambisonics_order": config.ambisonics_order,
        "low_ray_budget": config.low_ray_budget,
        "high_ray_budget": config.high_ray_budget,
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
        "metric_onset_tolerance_ms": config.metric_onset_tolerance_ms,
        "metric_band_resolvability_margin": config.metric_band_resolvability_margin,
        # Governs whether T30/EDT is SCORED AT ALL (AC-176), so it decides a
        # reported number and its caveat counts. In the fingerprint from the
        # moment it is declared -- F-65 is the row that exists because a key
        # governing a disclosure column was added and left out of one.
        "metric_min_decay_range_db": dict(sorted(config.metric_min_decay_range_db.items())),
        # Governs a REPORTED disclosure column (F-65); the class-level guard
        # against the next such key is FINGERPRINT_EXEMPT_FIELDS below.
        # THE FILTER IS THE INSTRUMENT (F-143). `order` sets both the out-of-band
        # rejection and the ringing floor every reported ISO metric is measured
        # against, so a run under a different order reports different numbers from
        # the same waveforms. `stopband_rejection_db` is the declared consequence
        # and moves with it.
        "metric_octave_filter": config.metric_octave_filter.model_dump(mode="json"),
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
        # Decides whether a reported interval is labelled uncalibrated, which is a
        # disclosure ON the reported table, so moving it must re-render (F-M7).
        "bootstrap_min_n_for_calibrated_ci": config.bootstrap_min_n_for_calibrated_ci,
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
        # AC-187's verdict is RENDERED here — a per-row caveat, a footer paragraph
        # and a CSV column — so it is a reported quantity and a cached report must
        # not serve a stale one. This is exactly the trigger the field's former
        # FINGERPRINT_EXEMPT_FIELDS entry named ("non-exempt the moment a
        # convergence verdict reaches a reported table").
        "convergence": config.convergence.model_dump(mode="json"),
        "report_format": config.report_format,
        # THE OPERAND DOMAIN THE UNITS ARE RENDERED FROM (F-162). `report` reads
        # `preprocessed/meta.json` `value_domain` to decide whether an
        # operand-domain metric prints as dB^2 or a.u.^2, and declared none of it —
        # so flipping the domain left a cached report rendering the old unit beside
        # unchanged numbers. Taken from the REPRESENTATION's own declaration, which
        # is what preprocess stamps, so this stays a function of the config; the
        # reporting layer cross-checks the stamp against it and refuses on
        # disagreement rather than trusting either alone.
        "value_domain": _representation_value_domain(config),
        "code_version": _code_version("report"),
    }


def _representation_value_domain(config: Config) -> str:
    """The domain the configured representation encodes in, without building it."""
    from .registry import representation_registry

    return str(representation_registry.get(config.representation.name).value_domain)


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
#: have nothing to scope — see STAGE_FINGERPRINT, RD-107 and RD-108 for the cost
#: that buys and what it leaves exposed.
#: Stages whose scope is not a fixed list because it depends on which backend is
#: active. Resolved through `simulator_code_scope(config)`, so swapping simulators
#: swaps the scope with them and a scaffold edit cannot invalidate a real dataset.
_BACKEND_SCOPED_STAGES = frozenset({"gen-scenes", "render"})

STAGE_CODE_SCOPE: dict[str, tuple[str, ...]] = {
    # Assigns splits and encodes tensors. `simulators/base.py` is named as a
    # single module rather than the whole subpackage because `data/preprocess.py`
    # depends on exactly one thing there — `SceneSpec`, the render manifest it
    # reads — and scoping the backend implementations would invalidate the
    # encoded dataset whenever an unrelated raytracer detail moved (F-64).
    "preprocess": ("data", "representations", "simulators/base.py"),
    # Trains weights: the model, the loss, the trainer, the dataset it reads, and
    # the representation whose domain the loss is expressed in.
    "train": ("training", "models", "data", "representations", "device.py"),
    # Loads the checkpoint, decodes through the representation, and denormalizes
    # every predicted leg — hence `data` (F-66).
    "infer": ("training", "models", "data", "representations", "device.py"),
    # The stage whose output IS the research claim. `data` because it denormalizes
    # every reported leg (F-66) — previously omitted here and from infer's scope,
    # masked only by `data` being in TRAIN's scope so the chain refused upstream
    # first, a coincidence of ordering rather than a guarantee.
    # `representations` is a DECLARED JUDGEMENT, not an import: eval measures the
    # decoded waveform that `infer` wrote, and the representation that produced it
    # is resolved by NAME through the registry, which the closure test cannot see
    # (AC-47 corrected an earlier claim here that "eval decodes before measuring").
    "eval": ("evaluation", "representations", "data"),
    # The probe measures through the metric path, so a change to the Schroeder
    # window or the octave filter moves its verdict without touching this package.
    "diagnostics": ("diagnostics", "evaluation", "representations", "data"),
    # Computes the reported CIs and MDES over `evaluation.metric_row`'s paired
    # improvements, which is why `evaluation` is in scope and not just `stats`.
    "stats": ("stats", "evaluation"),
    # Formats the reported tables (F-63; see `_report_fingerprint`).
    "report": ("reporting",),
}


def _code_version(stage: str, config: Config | None = None) -> str:
    """The content hash of `stage`'s declared sources — see
    `amcd.provenance.code_version`.

    Shared with `Config.stamp` through that module so the cache key and
    `versions.json` cannot describe different code (F-56).
    """
    if stage in _BACKEND_SCOPED_STAGES:
        if config is None:
            raise ValueError(
                f"stage {stage!r} scopes itself on the ACTIVE backend, so its "
                f"code_version cannot be computed without a config."
            )
        scope = simulator_code_scope(config)
        if stage == "gen-scenes":
            # The generator's own source decides placement and admission; the
            # backend's decides the realized support that gate measures against.
            scope = ("scenes",) + scope
        return provenance.code_version(scope)
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
#: wrong for `report` (see `_report_fingerprint`), and which RD-108/AC-45 dispute
#: here too: a stale D0b carrier-ceiling verdict is a false clearance.
STAGE_FINGERPRINT: dict[str, Callable[[Config], dict] | None] = {
    "gen-scenes": _gen_scenes_fingerprint,
    "render": _render_fingerprint,
    "preprocess": _preprocess_fingerprint,
    "diagnostics": _diagnostics_fingerprint,
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
    "diagnostics": "preprocess",
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
    **{
        name: (
            "Render-stage QC gate (configs/base.yaml §QC thresholds) that the real "
            "gsound_sir backend will enforce; the dry_run scaffold's synthetic IRs "
            "are clean by construction so nothing reads it yet. Non-exempt the "
            "moment the real backend enforces it — all four belong in "
            "`_render_fingerprint`, because they decide which renders are ADMITTED "
            "(RD-18)."
        )
        for name in (
            "onset_mismatch_tolerance_ms",
            "min_energy_db",
            "min_energy_reference",
            "max_path_file_mb",
            "require_non_empty_path_file",
        )
    },
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

    def _recorded_fingerprint(self, stage: str) -> tuple[str, dict | None]:
        """`stage`'s sentinel state and the fingerprint in it, if any.

        THREE STATES, NOT TWO (F-167). A bare `None` conflated them, and the two
        callers then disagreed about what it meant: `_is_done` told an operator
        their sentinel "predates fingerprinted caching", while `_effective_
        fingerprint` told the same operator about the same run_dir that the stage
        "has not completed ... running would record a provenance chain for artifacts
        that do not exist". Both claims were false — `stats/` was fully populated —
        and only the second was reachable from an upstream leg. Any run_dir
        predating F-63 is in that state.

        * `"absent"`   — no sentinel. The stage genuinely never ran here.
        * `"stale"`    — a sentinel with `fingerprint: null`, or one that will not
                         parse. It RAN; what it ran under cannot be established.
                         `null` is the legacy shape `_mark_done` wrote for a stage
                         that declared no fingerprint at the time, so this recurs
                         for every stage that gains one later.
        * `"present"`  — a fingerprint to compare.
        """
        sentinel = _sentinel(self.run_dir, stage)
        if not sentinel.exists():
            return "absent", None
        try:
            recorded = json.loads(sentinel.read_text())["fingerprint"]
        except (json.JSONDecodeError, TypeError, KeyError):
            recorded = None
        return ("present", recorded) if recorded is not None else ("stale", None)

    def _unestablishable(self, stage: str, *, because: str) -> RuntimeError:
        """The one message for a sentinel that RAN under unknown inputs (F-167).

        Shared by both legs so they cannot drift into describing the same run_dir
        differently — which is the defect, not the wording.
        """
        return RuntimeError(
            f"Stage {stage!r} has a cached sentinel with no fingerprint "
            f"({_sentinel(self.run_dir, stage)}). It predates fingerprinted caching "
            f"for this stage, so whether its artifacts match the current config "
            f"cannot be established{because}. Re-run {stage!r} with --force (this "
            f"discards its existing artifacts), or use a fresh --run-dir."
        )

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
        state, recorded = self._recorded_fingerprint(upstream)
        if state == "absent":
            raise RuntimeError(
                f"Stage {stage!r} depends on {upstream!r}, which has not completed "
                f"in {self.run_dir} (no sentinel). Run {upstream!r} first — running "
                f"{stage!r} now would record a provenance chain for artifacts that "
                f"do not exist."
            )
        if state == "stale":
            # It DID run, and its artifacts are on disk. Saying "has not completed"
            # here sent an operator looking for missing output that was in front of
            # them (F-167).
            raise self._unestablishable(
                upstream,
                because=f", and {stage!r} would chain its provenance to that",
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
        state, found = self._recorded_fingerprint(stage)
        # A RECORDED `null` is the realistic legacy shape, not a corrupt file: it is
        # exactly what `_mark_done` wrote for a stage that declared no fingerprint
        # at the time. Before F-75 it fell through to `_diff_fingerprints`, which did
        # `set(None)` and raised a bare TypeError with a traceback. This recurs for
        # EVERY stage that gains a fingerprint later (`diagnostics` next,
        # RD-108/AC-45), so `_recorded_fingerprint` names the state and both legs
        # read it from there (F-167).
        if state == "stale":
            raise self._unestablishable(stage, because="") from None

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
            # them, and it is what tells the
            # operator that a cached artifact predates the current source (F-75).
            "code_version_unscoped": provenance.code_version(provenance.ALL_SOURCES),
        }
        s = _sentinel(self.run_dir, stage)
        s.parent.mkdir(parents=True, exist_ok=True)
        s.write_text(json.dumps(payload, indent=2, default=str))

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
