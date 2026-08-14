"""Stage runner: dispatch, timing, and fingerprinted stage caching.

Caching is what makes a long run resumable, and — because renders under x86
emulation are the most expensive artifact this pipeline produces — it is also the
easiest way to end up with a silently mixed dataset. A bare "this stage ran"
sentinel cannot tell whether it ran under the CURRENT config, so changing a
simulator parameter or a scene range and re-running the same run_dir would reuse
stale artifacts and produce a dataset that is part old, part new, with nothing on
disk recording the split.

So each stage may declare the config inputs it depends on (`STAGE_FINGERPRINT`).
The sentinel stores that fingerprint, and a mismatch is a LOUD FAILURE, never a
silent re-use and never a silent re-run: only the operator can decide whether the
existing artifacts are salvageable.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Callable

from . import provenance
from .config import Config
from .simulators.base import simulator_code_scope, simulator_host_scoped_params
from .runtime import RunContext, emit

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

    Only the GENERATION-relevant split fields are included. `frac` and `role` are
    consumed at preprocess and cannot change a single generated scene, so dumping
    the whole `SplitSpec` would make editing `train.frac` force a complete
    re-render of a dataset that provably had not changed. Both still reach
    `_preprocess_fingerprint`, which carries the full dump.

    `ir_duration` and the SIMULATOR are here for a different reason: they do not
    change which scenes are sampled, they change whether gen-scenes SUCCEEDS and
    what it discloses. `ir_duration` is what the record-length gate compares
    against, and the simulator declares the minimum source-receiver separation the
    pre-flight check enforces. Without them the gate is bypassable through the
    cache, and `placement_report.json` can stay stamped with a record length its
    own `config.yaml` contradicts.
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
#: these in would redact them from canonical provenance where they belong.
_DISCLOSURE_ONLY_PARAMS: frozenset[str] = frozenset({"max_discarded_tail_db"})


def _dataset_simulator_params(config: Config) -> dict:
    """`config.simulator.params` minus the entries that describe the HOST or the
    DISCLOSURE, not the IR.

    Neither can change a single sample, and either in the render cache identity
    would demand a byte-identical re-render on a different host:

    * **Host-scoped params** (`render_python` — the x86 interpreter the emulated
      render runs under). A dataset rendered on this Mac would otherwise fail
      loudly on the x86_64 Linux host, a direct violation of the cross-platform
      requirement. The backend already DECLARES which of its params are
      host-scoped, and `render._canonical_meta` already redacts them from
      provenance; this asks the same declaration.
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


def _render_artifact_fingerprint(config: Config) -> dict:
    """Config inputs that determine the rendered IR BYTES for a given scene.

    `simulator.params` carries the pinned upstream `commit_sha`, so a GSound-SIR
    version change invalidates the cache with no extra wiring.

    Separate from `_render_fingerprint` because the render stage records this one
    per scene and reuses any scene that still carries it. Everything that decides
    ADMISSION rather than content is kept out of it — the QC thresholds below, and
    the QC CODE via `_RENDER_ADMISSION_SOURCES` — because re-scoring a changed
    admission rule must cost a re-score, not a full emulated re-render.

    `code_version` is therefore the BYTES scope only: this stage's own
    `render.py` plus the active backend's declared sources.
    """
    return {
        "code_version": _render_bytes_code_version(config),
        "simulator": {"name": config.simulator.name,
                      "params": _dataset_simulator_params(config)},
        "sample_rate": config.sample_rate,
        # Both DERIVED values, recorded rather than only the fields they derive
        # from, because it is the derived pair that fixes the array the backend
        # must return (`render.py` checks `(n_channels, n_samples)`). Recording
        # `ambisonics_order` and `ir_duration` alone would leave a change to
        # either DERIVATION invisible here — and this dict has to enumerate the
        # render's inputs COMPLETELY, since `config.py`'s own bytes are not in
        # this scope.
        "n_samples": config.n_samples,
        "n_channels": config.n_channels,
        "ambisonics_order": config.ambisonics_order,
        "low_ray_budget": config.low_ray_budget,
        "high_ray_budget": config.high_ray_budget,
    }


def _render_fingerprint(config: Config) -> dict:
    """Everything that determines which renders reach the dataset: the bytes, and
    the QC criteria that admit them.

    A change to any QC threshold — or to the QC CODE, which
    `_RENDER_ADMISSION_SOURCES` puts in `code_version` here and only here —
    invalidates this stage and everything downstream, which is correct: an
    admission rule change is a different dataset. Per-scene reuse keeps its cost
    to a re-score.
    """
    return {
        **_render_artifact_fingerprint(config),
        "admission_code_version": _code_version("render", config),
        "qc": {
            "onset_rel_db": config.metric_onset_rel_db,
            "onset_mismatch_tolerance_ms": config.onset_mismatch_tolerance_ms,
            "min_energy_db": config.min_energy_db,
            "min_energy_reference": config.min_energy_reference,
            "max_path_file_mb": config.max_path_file_mb,
            "require_non_empty_path_file": config.require_non_empty_path_file,
            # The attrition bounds are admission-side too. Here rather than
            # exempt because the asymmetry bites: a batch that BREACHES a bound
            # raises and writes no sentinel, so loosening one re-runs anyway —
            # but TIGHTENING one on a batch that already completed would never be
            # re-checked, and the dataset would carry attrition above its own
            # declared bound with the sentinel reporting done.
            "max_excluded_frac": config.max_excluded_frac,
            "max_refused_frac": config.max_refused_frac,
            "max_unscored_gating_frac": config.max_unscored_gating_frac,
        },
    }


def _preprocess_fingerprint(config: Config) -> dict:
    """Config inputs — AND the code version — that determine the split assignment
    and the encoded tensors.

    `split_assignment` belongs here specifically: it is the most
    leakage-critical value in the project, it is consumed HERE rather than at
    gen-scenes, and without it in a fingerprint, repinning the split seed on an
    existing run_dir was a complete no-op — splits.json kept the old assignment
    while config.yaml stamped the new seed, a provenance lie.

    `code_version` closes the same hole against a CODE change: config keys
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
        # Enforced here, because this is where split membership is assigned — so
        # it is here that tightening it must re-check rather than be skipped by a
        # sentinel. Cheap either way: preprocess re-runs in seconds.
        "max_excluded_frac_per_split": config.max_excluded_frac_per_split,
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
        # THE FILTER IS THE INSTRUMENT. `order` sets both the out-of-band
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
    through the chain from `train`.
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

    Config keys alone would be blind to the changes that matter most here: the
    shared Schroeder window was a CODE change in `evaluation/room_acoustic.py`
    that moved no config key. Hence `code_version` — a cached `metrics.parquet`
    written under a different revision of the metric code is refused, not silently
    reported.

    Erring coarse is right for the one stage whose output IS the research claim,
    and eval is cheap next to render.
    """
    return {
        "iso_eval_freqs": list(config.iso_eval_freqs),
        "metric_onset_rel_db": config.metric_onset_rel_db,
        "metric_onset_tolerance_ms": config.metric_onset_tolerance_ms,
        "metric_band_resolvability_margin": config.metric_band_resolvability_margin,
        # Governs whether T30/EDT is SCORED AT ALL, so it decides a
        # reported number and its caveat counts. In the fingerprint from the
        # moment it is declared.
        "metric_min_decay_range_db": dict(sorted(config.metric_min_decay_range_db.items())),
        # Governs a REPORTED disclosure column; the class-level guard
        # against the next such key is FINGERPRINT_EXEMPT_FIELDS below.
        # THE FILTER IS THE INSTRUMENT. `order` sets both the out-of-band
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
    below forbids: recomputing lets `--force` stamp a chain describing artifacts
    that do not exist.

    `code_version` covers the code that COMPUTES the reported CIs. `stats`
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
        # disclosure ON the reported table, so moving it must re-render.
        "bootstrap_min_n_for_calibrated_ci": config.bootstrap_min_n_for_calibrated_ci,
        "seed_bootstrap": config.seed("bootstrap"),
        "code_version": _code_version("stats"),
    }


def _report_fingerprint(config: Config) -> dict:
    """Config inputs — AND the code version — that determine the reported tables.

    "Terminal" would be true of the graph and false of the consequence: this
    stage's own re-use IS `report/summary.txt`, the artifact a reader treats as
    the result. `STAGE_UPSTREAM["report"] = "stats"` carries the other half, since
    a fingerprint over this stage's own inputs cannot see the numbers it formats.
    """
    return {
        # The convergence verdict is RENDERED here — a per-row caveat, a footer
        # paragraph and a CSV column — so it is a reported quantity and a cached
        # report must not serve a stale one.
        "convergence": config.convergence.model_dump(mode="json"),
        "report_format": config.report_format,
        # THE OPERAND DOMAIN THE UNITS ARE RENDERED FROM. `report` reads
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


#: Stages whose scope is not a fixed list because it depends on which backend is
#: active. Resolved through `simulator_code_scope(config)`, so swapping simulators
#: swaps the scope with them and a scaffold edit cannot invalidate a real dataset.
_BACKEND_SCOPED_STAGES = frozenset({"gen-scenes", "render"})


#: Per stage, the package sources its OUTPUT is a function of — the code half of a
#: fingerprint. `provenance.code_version` adds the core modules (config,
#: seeds, registry, shared acoustics, the package `__init__`) to every scope, so
#: these list only what is specific to the stage. An entry is a subpackage
#: (`"evaluation"`), a top-level module (`"config.py"`) or a nested module
#: (`"simulators/base.py"`) when a stage depends on one file of another subtree.
#:
#: WHY A CONFIG FINGERPRINT IS NOT ENOUGH — stated once, here, and cited from the
#: three other places that used to restate it. A fingerprint built from config
#: keys sees CONFIG changes only. A code change — a redefined metric, a different
#: sampling rule — moves nothing in those payloads, and the cached artifact is
#: served under the new code. Every stage whose output reaches the
#: reported result therefore carries `code_version()` as well.
#:
#: Why the scope is DECLARED per stage rather than a hash over the whole package:
#: see `provenance.code_version`, which owns that argument.
#:
#: The residual risk of declaring is a scope that omits a real dependency, and it
#: fails SILENTLY. `tests/test_stage_cache.py` therefore asserts each declared
#: scope against the stage's transitive import closure; read that test's docstring
#: for what it does and does not catch.
#:
#: Seven of the nine stages are here. `gen-scenes` and `render` are absent because
#: their scope is not a fixed list — it is resolved per backend in
#: `_code_version`, which for `render` also prepends the stage's own sources
#: (`simulators/render.py`, `simulators/qc.py`, `evaluation/room_acoustic.py`).
#: The resulting file list is written out in `docs/design_spec.md` §11.1, "What
#: DOES discard a persisted dataset".
STAGE_CODE_SCOPE: dict[str, tuple[str, ...]] = {
    # Assigns splits and encodes tensors. `simulators/base.py` is named as a
    # single module rather than the whole subpackage because `data/preprocess.py`
    # depends on exactly one thing there — `SceneSpec`, the render manifest it
    # reads — and scoping the backend implementations would invalidate the
    # encoded dataset whenever an unrelated raytracer detail moved.
    "preprocess": ("data", "representations", "simulators/base.py"),
    # Trains weights: the model, the loss, the trainer, the dataset it reads, and
    # the representation whose domain the loss is expressed in.
    "train": ("training", "models", "data", "representations", "device.py"),
    # Loads the checkpoint, decodes through the representation, and denormalizes
    # every predicted leg — hence `data`.
    "infer": ("training", "models", "data", "representations", "device.py"),
    # The stage whose output IS the research claim. `data` because it denormalizes
    # every reported leg — previously omitted here and from infer's scope,
    # masked only by `data` being in TRAIN's scope so the chain refused upstream
    # first, a coincidence of ordering rather than a guarantee.
    # `representations` is a DECLARED JUDGEMENT, not an import: eval measures the
    # decoded waveform that `infer` wrote, and the representation that produced it
    # is resolved by NAME through the registry, which the closure test cannot see.
    "eval": ("evaluation", "representations", "data"),
    # The probe measures through the metric path, so a change to the Schroeder
    # window or the octave filter moves its verdict without touching this package.
    # `simulators` because both D0 artifacts publish the ACTIVE BACKEND's declared
    # limitations: a backend that stopped declaring "no early reflections"
    # would change what d0a_gap.json and d0b_oracle.json say about their own EDT
    # columns, with no config key moving.
    "diagnostics": (
        "diagnostics", "evaluation", "representations", "data", "simulators",
    ),
    # Computes the reported CIs and MDES over `evaluation.metric_row`'s paired
    # improvements, which is why `evaluation` is in scope and not just `stats`.
    "stats": ("stats", "evaluation"),
    # Formats the reported tables. `simulators` because the footer renders the
    # ACTIVE BACKEND's declared limitations —
    # a backend that changed its early-reflection declaration would change what the
    # table says about its own EDT columns, with no config key moving.
    "report": ("reporting", "simulators"),
}


def _code_version(stage: str, config: Config | None = None) -> str:
    """The content hash of `stage`'s declared sources — see
    `amcd.provenance.code_version`.

    Shared with `Config.stamp` through that module so the cache key and
    `versions.json` cannot describe different code.
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
        elif stage == "render":
            # The STAGE's own sources, not only the backend's — see
            # `_RENDER_BYTES_SOURCES` / `_RENDER_ADMISSION_SOURCES` for why the two
            # halves are separated and which fingerprint each reaches.
            scope = _RENDER_BYTES_SOURCES + _RENDER_ADMISSION_SOURCES + scope
        return provenance.code_version(scope)
    return provenance.code_version(STAGE_CODE_SCOPE[stage])


#: Render-stage sources that decide the IR BYTES. In BOTH render fingerprints, so
#: a change here re-renders every scene.
_RENDER_BYTES_SOURCES: tuple[str, ...] = ("simulators/render.py",)

#: Render-stage sources that decide only which renders are ADMITTED. In the STAGE
#: fingerprint and deliberately NOT in the per-scene artifact one — neither can
#: change a single sample, so a change here must cost a RE-SCORE of the persisted
#: renders, never a re-render. `qc.py` is the criteria; `room_acoustic.py` is the
#: onset detector `qc.py` borrows from it.
#:
#: Without the split the affordance §11.1 is built around fails exactly when it is
#: needed: a QC failure at scene 700 of 720, fixed in `qc.py`, would discard all
#: 700 renders to change a rule that alters no IR.
_RENDER_ADMISSION_SOURCES: tuple[str, ...] = (
    "simulators/qc.py", "evaluation/room_acoustic.py",
)


#: Core sources dropped from the render BYTES scope, and nowhere else.
#:
#: `config.py` is in `_CORE_SOURCES`, so it lands in every scope including this
#: one — which made ANY behavioural edit to it discard 720 persisted renders. It
#: cannot be left there: `config.py` is where every new experiment parameter goes,
#: so E2's loss keys and E3's search keys would each cost ~14 hours of emulation,
#: and the re-rendered bytes would not be the ones the earlier numbers came from
#: (the backend exposes no RNG seed — design_spec §11.1).
#:
#: What replaces it is `_render_artifact_fingerprint` enumerating the resolved
#: config VALUES the render bytes are a function of — simulator name and
#: dataset-scoped params, sample rate, the derived `(n_channels, n_samples)`
#: shape, and both ray budgets. That enumeration is the obligation this constant
#: creates: a config value that reaches the backend and is not in that dict is a
#: silent staleness bug, which is why the shape is recorded DERIVED rather than as
#: the fields it comes from.
_RENDER_BYTES_DROPPED_CORE: tuple[str, ...] = ("config.py",)


def _render_bytes_code_version(config: Config) -> str:
    """`code_version` over the sources that decide the IR bytes only."""
    return provenance.code_version(
        _RENDER_BYTES_SOURCES + simulator_code_scope(config),
        drop_core=_RENDER_BYTES_DROPPED_CORE,
    )


#: Per-stage declaration of the config inputs a cached artifact depends on.
#:
#: `None` means "no fingerprint declared yet" — the stage caches on the bare
#: sentinel, exactly as before. Every stage is listed explicitly, including the
#: `None`s, so an unwired stage is DECLARED rather than silently absent (the same
#: rule docs/verbosity.md applies to verbosity wiring).
#:
#: EVERY stage carries `code_version()` as well as config keys, `gen-scenes` and
#: `render` included — theirs is backend-scoped (see `_code_version`), so a
#: semantic edit to the active backend or to `_CORE_SOURCES` invalidates a
#: persisted dataset while a comment edit does not. See STAGE_CODE_SCOPE for why
#: config keys alone are not enough, and `provenance._semantic_digest` for the
#: comment exemption.
#:
#: The one remaining `None` is `diagnostics`, on the argument that it is terminal
#: so an unwired fingerprint costs only its own re-use — the same argument that
#: turned out to be wrong for `report` (see `_report_fingerprint`), and which is
#: disputed here too: a stale D0b carrier-ceiling verdict is a false clearance.
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

#: Stages that write PER-UNIT artifacts expensive enough to reuse individually,
#: and the subset of their fingerprint that determines those artifacts' content.
#: A stage listed here records the sha beside each unit it writes and skips any
#: unit that still carries it, so a stage that fails part-way through resumes.
#:
#: Absent = the stage re-does all its work whenever it runs, which is right for
#: everything cheap enough that per-unit bookkeeping would cost more than it saves.
STAGE_ARTIFACT_FINGERPRINT: dict[str, Callable[[Config], dict]] = {
    "render": _render_artifact_fingerprint,
}

#: Which stage's artifacts each stage consumes. The chain is what makes a
#: fingerprint transitive: renders are renders OF scenes, so a render is stale
#: when the SCENES changed even if every simulator parameter is untouched.
#:
#: Chaining reads the upstream SENTINEL's recorded fingerprint, never a
#: recomputation from the current config. Recomputing made `--force` a
#: laundering step: forcing a downstream stage stamped the CURRENT config's chain
#: value even though the stage had consumed the OLD upstream artifacts, leaving a
#: run_dir whose chain validated while the artifacts disagreed — precisely the
#: silent mixed dataset this machinery exists to prevent.
#: The chain runs unbroken from the scenes to the reported TABLE:
#:
#:   gen-scenes → render → preprocess → train → infer → eval → stats → report
#:
#: Wiring it unbroken is what makes a fingerprint transitive: `report` is stale
#: when the SCENES changed, through seven links, without `_report_fingerprint`
#: naming a single scene key. A chain stopping at `preprocess` would report a
#: changed model under a stale checkpoint's numbers.
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
#: The rule this enforces: every `Config` field must either move a fingerprint
#: when perturbed or be named here, so a threshold that reaches no fingerprint is
#: distinguishable from one that is absent on purpose. `tests/test_stage_cache.py`
#: checks it.
#:
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
    """Leaf-level differences, so the error says WHAT changed.

    A bare sha mismatch forces the operator to decide blind whether an expensive
    renders/ directory is salvageable; naming the changed field is the difference
    between a five-second judgement and a re-render under emulation.

    RECURSES into nested dicts and reports dotted paths. A top-level-only
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


def _dispatch(stage: str) -> Callable[[Config, Path, RunContext], None]:
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
    def __init__(
        self, config: Config, run_dir: Path, ctx: RunContext, force: bool = False,
        revalidate: bool = False,
    ) -> None:
        self.config = config
        self.run_dir = run_dir
        #: Everything a stage needs that is not an experiment value. Passed
        #: to `_dispatch`'s callables whole, so adding a runtime value later is a
        #: change to `RunContext` rather than to nine stage signatures.
        self.ctx = ctx
        #: Convenience for this class's own `emit` calls; `ctx` is what stages get.
        self.verbosity = ctx.verbosity
        self.force = force
        #: Run past a fingerprint mismatch WITHOUT discarding per-unit artifacts —
        #: see `RunContext.revalidate` for why this is not `--force`.
        self.revalidate = revalidate

    def _recorded_fingerprint(self, stage: str) -> tuple[str, dict | None]:
        """`stage`'s sentinel state and the fingerprint in it, if any.

        THREE STATES, NOT TWO. A bare `None` conflated them, and the two
        callers then disagreed about what it meant: `_is_done` told an operator
        their sentinel "predates fingerprinted caching", while `_effective_
        fingerprint` told the same operator about the same run_dir that the stage
        "has not completed ... running would record a provenance chain for artifacts
        that do not exist". Both claims were false — `stats/` was fully populated —
        and only the second was reachable from an upstream leg.

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
        """The one message for a sentinel that RAN under unknown inputs.

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
        fingerprint.

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
        # No STAGE_UPSTREAM value points at such a stage today; the first chain to
        # cross one would otherwise report "has not completed" for a stage that did.
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
            # them.
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
        if self.force or self.revalidate or not sentinel.exists():
            return False

        if STAGE_FINGERPRINT[stage] is None:
            return True  # stage declares no config dependency; bare sentinel

        expected = self._effective_fingerprint(stage)
        state, found = self._recorded_fingerprint(stage)
        # A RECORDED `null` is the realistic legacy shape, not a corrupt file: it is
        # exactly what `_mark_done` wrote for a stage that declared no fingerprint
        # at the time, and it recurs for EVERY stage that gains a fingerprint
        # later. `_recorded_fingerprint` names the state so both legs read it from
        # there rather than each deriving it from a bare `None`.
        if state == "stale":
            raise self._unestablishable(stage, because="") from None

        if found != expected:
            diff = _diff_fingerprints(found, expected) or ["    (nested value changed)"]
            raise RuntimeError(
                f"Stage {stage!r} was cached under a DIFFERENT config; reusing it "
                f"would silently mix two experiments in one run_dir.\n"
                f"  Changed inputs:\n" + "\n".join(diff) + "\n"
                f"  Re-run with --revalidate to re-run {stage!r} while KEEPING every "
                f"artifact whose own fingerprint still matches (a changed admission "
                f"rule costs a re-score, not a re-render), with --force to rebuild it "
                f"from scratch, or use a fresh --run-dir to keep both."
            )
        return True

    def _mark_done(self, stage: str) -> None:
        # Sentinels are functional (caching input), never verbosity-gated.
        # The recorded fingerprint chains the UPSTREAM SENTINEL, so it describes the
        # artifacts this run actually consumed — including under --force.
        payload = {
            "completed_at": time.time(),
            # Full payload, not just the sha: the mismatch error must be able to
            # name the field that changed.
            "fingerprint": (
                self._effective_fingerprint(stage)
                if STAGE_FINGERPRINT[stage] else None
            ),
            # OUTSIDE `fingerprint`, so it is recorded and never compared: the
            # whole-package hash of the code that actually produced these
            # artifacts. It COMPLEMENTS each stage's scoped `code_version` rather
            # than substituting for one — every stage has a scoped version, those
            # of `gen-scenes` and `render` resolved per backend — and it is what
            # tells an operator that a cached artifact predates the current source
            # even where the scope did not reach the edit.
            "code_version_unscoped": provenance.code_version(provenance.ALL_SOURCES),
        }
        s = _sentinel(self.run_dir, stage)
        s.parent.mkdir(parents=True, exist_ok=True)
        s.write_text(json.dumps(payload, indent=2, default=str))

    def _stage_context(self, stage: str) -> RunContext:
        """`self.ctx` with the artifact fingerprint of the stage about to run.

        Only the ARTIFACT half is passed, never the stage fingerprint: a stage
        reuses a per-unit artifact on the strength of the inputs that produced its
        bytes, and admission thresholds are not among them.
        """
        artifact_fn = STAGE_ARTIFACT_FINGERPRINT.get(stage)
        return replace(
            self.ctx,
            artifact_fingerprint_sha=(
                None if artifact_fn is None else _fingerprint_sha(artifact_fn(self.config))
            ),
            force=self.force,
            revalidate=self.revalidate,
        )

    def _record_timing(self, stage: str, seconds: float) -> None:
        # Per-stage wall time is provenance: written from save level 1.
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
            # chain. Inside the try so the failure still reaches stderr.
            if STAGE_FINGERPRINT[stage] is not None:
                self._effective_fingerprint(stage)

            if self._is_done(stage):
                emit(self.verbosity, "progress", f"[skip] {stage} (cached)")
                return

            # INVALIDATE BEFORE MUTATING. The stage is about to overwrite
            # its artifacts, so from this moment the old sentinel describes a
            # directory that no longer exists. Leaving it in place until success
            # meant a stage killed PART-WAY THROUGH WRITING left the PREVIOUS run's
            # success sentinel standing over half-new artifacts, and the next
            # invocation printed `[skip] (cached)` against it. The costly instance
            # is render, where a kill mid-run leaves the tail of the scene range
            # holding the previous config's output under a sentinel that validates
            # — the silently mixed dataset this machinery exists to prevent, and
            # the multi-hour case under emulation.
            _sentinel(self.run_dir, stage).unlink(missing_ok=True)

            emit(self.verbosity, "progress", f"\n[run ] {stage}")
            fn = _dispatch(stage)
            fn(self.config, self.run_dir, self._stage_context(stage))
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
