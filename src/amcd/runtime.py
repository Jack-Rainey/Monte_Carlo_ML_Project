"""Runtime output control: the two verbosity axes (save-to-disk, show-live).

`Verbosity` is NOT part of `Config`: verbosity sets only how much a run writes
and prints, never what it produces — it is a runtime output level, not an
experiment-governing value (CLAUDE.md "Output verbosity is not an experiment
value"). It is threaded explicitly cli → Pipeline → every stage function
(F-22); it is never a module global. Canonical results, inter-stage inputs,
and stage sentinels are written at EVERY save level, including 0 — only
observability artifacts may sit behind `saves()` (F-23).

Both axes share one monotonic category ladder (RR-19): each level adds one
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
#: "visual" is the reserved roadmap-§6 Blender preview slot (RD-10) — nothing
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
#: regardless of `show` (F-24 — a suppressed fatal error is never acceptable).
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
    at the CLI options (RD-09) — this class deliberately has none.
    """

    save: int
    show: int

    def __post_init__(self) -> None:
        for name, val in (("save", self.save), ("show", self.show)):
            if not 0 <= val <= 5:
                raise ValueError(f"Verbosity.{name} must be in 0..5, got {val}")

    def shows(self, category: str) -> bool:
        """Whether live output of `category` is enabled at this show level.

        The "visual" category additionally requires an interactive stdout
        (F-24): a blocking preview must never deadlock a headless run — when
        stdout is not a TTY it degrades to the save axis (render-and-save or
        skip), decided by the emitting site via `saves("visual")`.
        """
        if category in _ALWAYS_STDERR:
            return True
        if category == "visual":
            return self.show >= CATEGORY_LEVELS["visual"] and sys.stdout.isatty()
        return self.show >= _level(category)

    def saves(self, category: str) -> bool:
        """Whether disk artifacts of `category` are written at this save level.

        May gate ONLY observability/diagnostic artifacts — never canonical
        results, inter-stage inputs, or stage sentinels (F-23).
        """
        if category in _ALWAYS_STDERR:
            # warning/error are never gated on either axis (F-24 parity with
            # shows()): an artifact recording a failure writes at every save
            # level. No such artifact exists yet; the contract is stated here
            # so a future one cannot be accidentally gated.
            return True
        return self.save >= _level(category)


def emit(verbosity: Verbosity, category: str, msg: str) -> None:
    """The single level-gated console emission helper (RR-19).

    warning/error → stderr, always. Ladder categories → stdout iff enabled
    at `verbosity.show`. All stage console output goes through here so the
    gating rule lives in exactly one place.
    """
    if category in _ALWAYS_STDERR:
        print(msg, file=sys.stderr)
        return
    if verbosity.shows(category):
        print(msg)
