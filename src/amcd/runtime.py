"""Runtime output control: the two verbosity axes (save-to-disk, show-live).

`Verbosity` is NOT part of `Config`: verbosity sets only how much a run writes
and prints, never what it produces — it is a runtime output level, not an
experiment-governing value (CLAUDE.md "Output verbosity is not an experiment
value"). It is threaded explicitly cli → Pipeline → every stage function and is
never a module global, riding inside `RunContext` (below) across that boundary;
stage functions are dispatched as `(config, run_dir, ctx)` while helpers below
the entry point still take a bare `Verbosity`. Canonical results, inter-stage
inputs and stage sentinels are written at EVERY save level, including 0 — only
observability artifacts may sit behind `saves()`.

Both axes share one monotonic category ladder: each level adds one
category on top of everything below it. The full per-stage wiring table lives
in docs/verbosity.md; a stage or site absent from that table is declared
unwired there, never silently inert.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass

#: Monotonic category ladder shared by both axes. Level 0 emits/saves none of
#: these. "provenance" and "timing" are the same rung viewed per axis: on the
#: save axis level 1 writes reproducibility metadata (config snapshot, seeds,
#: git SHA, timings.json); on the show axis it prints run identity + durations.
#: "diagnostics" is per-unit QC observability (today: the render stage's
#: meta.json records), not the diagnostics *stage* — that stage's D0a/D0b
#: tables print at "metrics" and its JSON outputs are canonical.
#: "visual" is the reserved roadmap-§6 Blender preview slot — nothing
#: emits it today; the slot and its TTY guard are the seam the §6 front-end
#: will fill.
CATEGORY_LEVELS: dict[str, int] = {
    "provenance": 1,
    "timing": 1,
    "progress": 2,
    "metrics": 3,
    "diagnostics": 4,
    "visual": 5,
}

#: Outside the ladder entirely: failures and warnings always emit, to stderr,
#: regardless of `show`: a suppressed fatal error is never acceptable.
_ALWAYS_STDERR = ("warning", "error")


def _level(category: str) -> int:
    try:
        return CATEGORY_LEVELS[category]
    except KeyError:
        raise ValueError(
            f"Unknown verbosity category {category!r}; "
            f"known: {sorted(CATEGORY_LEVELS)} + {list(_ALWAYS_STDERR)}"
        ) from None


@dataclass(frozen=True)
class Verbosity:
    """Frozen pair of output levels, each in 0..5.

    The CLI already rejects out-of-range values via ``click.IntRange(0, 5)``
    (no silent clamp); the ``__post_init__`` check is defense in depth for
    direct construction (tests, future non-CLI callers). Defaults live ONLY
    at the CLI options — this class deliberately has none.
    """

    save: int
    show: int

    def __post_init__(self) -> None:
        for name, val in (("save", self.save), ("show", self.show)):
            if not 0 <= val <= 5:
                raise ValueError(f"Verbosity.{name} must be in 0..5, got {val}")

    def shows(self, category: str) -> bool:
        """Whether live output of `category` is enabled at this show level.

        The "visual" category additionally requires an interactive stdout: a
        blocking preview must never deadlock a headless run, so when stdout is
        not a TTY it degrades to the save axis (render-and-save or skip),
        decided by the emitting site via `saves("visual")`.
        """
        if category in _ALWAYS_STDERR:
            return True
        if category == "visual":
            return self.show >= CATEGORY_LEVELS["visual"] and sys.stdout.isatty()
        return self.show >= _level(category)

    def saves(self, category: str) -> bool:
        """Whether disk artifacts of `category` are written at this save level.

        May gate ONLY observability/diagnostic artifacts — never canonical
        results, inter-stage inputs, or stage sentinels.
        """
        if category in _ALWAYS_STDERR:
            # Parity with `shows()`: an artifact recording a failure writes at
            # every save level. `renders/qc_failures.csv` is one such artifact.
            return True
        return self.save >= _level(category)


def emit(verbosity: Verbosity, category: str, msg: str) -> None:
    """The single level-gated console emission helper.

    warning/error → stderr, always. Ladder categories → stdout iff enabled
    at `verbosity.show`. All stage console output goes through here so the
    gating rule lives in exactly one place.
    """
    if category in _ALWAYS_STDERR:
        print(msg, file=sys.stderr)
        return
    if verbosity.shows(category):
        print(msg)


@dataclass(frozen=True)
class RunContext:
    """Everything a stage needs that is NOT an experiment value.

    ONE ARGUMENT INSTEAD OF A GROWING LIST. Stage functions are dispatched
    as `(config, run_dir, ctx)`. `config` carries every value that governs what a run
    PRODUCES; this carries what governs how it RUNS. The two are different in kind,
    and keeping the second out of `Config` is the rule `Verbosity` already states —
    an output level is not an experiment-governing value.

    It exists because widening the dispatch signature is a nine-stage-plus-tests
    touch, and `verbosity` was already the second such value after `run_dir`. The
    roadmap makes a third foreseeable: §6's Blender preview needs to know whether it
    has a display, and the render backend's host interpreter is resolved per machine.
    Neither is added here — a field with no consumer would be inventing its shape —
    but adding one is now a change to this class rather than to nine signatures.

    FROZEN, because a stage must not be able to change how a later stage runs. A
    mutable context passed down a nine-stage chain is a channel by which stage 3
    could silence stage 7, and nothing would record it.
    """

    verbosity: Verbosity

    #: sha of the config inputs that determine the ARTIFACT BYTES of the running
    #: stage, set by `Pipeline.run_stage`. A stage that writes expensive per-unit
    #: artifacts records this beside each one and reuses any unit that still
    #: carries it, so a stage that fails part-way through resumes instead of
    #: repeating work the stage-level sentinel cannot describe. None for stages
    #: that declare no fingerprint.
    artifact_fingerprint_sha: str | None = None

    #: Whether the operator passed `--force`. Per-unit reuse must honour it or the
    #: flag would mean "re-run the stage" while the stage reused every unit and
    #: rebuilt nothing — so a run started to escape a suspect artifact would
    #: return that artifact. The sha above is still RECORDED on a forced run, so
    #: the next run can resume from what the forced one produced.
    force: bool = False
