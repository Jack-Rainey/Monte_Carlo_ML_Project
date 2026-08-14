"""Unit tests for the verbosity runtime.

Covers the `Verbosity` value object and the single `emit` helper: range
validation (no silent clamp), the monotonic-superset property of the category
ladder, always-on stderr for warnings/errors, and the TTY guard on the
reserved `visual` category.
"""
from __future__ import annotations

import sys

import pytest

from amcd.runtime import CATEGORY_LEVELS, Verbosity, emit


def test_verbosity_rejects_out_of_range() -> None:
    """Out-of-range levels are an error, never clamped."""
    for bad in (-1, 6, 99):
        with pytest.raises(ValueError):
            Verbosity(save=bad, show=0)
        with pytest.raises(ValueError):
            Verbosity(save=0, show=bad)
    # Bounds themselves are valid.
    Verbosity(save=0, show=0)
    Verbosity(save=5, show=5)


def test_unknown_category_raises() -> None:
    """A typo'd category fails loud, not silently-off."""
    v = Verbosity(save=5, show=5)
    with pytest.raises(ValueError, match="Unknown verbosity category"):
        v.shows("progess")
    with pytest.raises(ValueError, match="Unknown verbosity category"):
        v.saves("provenence")


def test_ladder_is_monotonic_superset(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each level enables a superset of every lower level, on both axes."""
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    for level in range(5):
        lo_show = {c for c in CATEGORY_LEVELS if Verbosity(save=0, show=level).shows(c)}
        hi_show = {c for c in CATEGORY_LEVELS if Verbosity(save=0, show=level + 1).shows(c)}
        assert lo_show <= hi_show
        # Each step up adds at least one category ("provenance"/"timing" share
        # a rung, so steps may add two names but never zero).
        assert lo_show != hi_show
        lo_save = {c for c in CATEGORY_LEVELS if Verbosity(save=level, show=0).saves(c)}
        hi_save = {c for c in CATEGORY_LEVELS if Verbosity(save=level + 1, show=0).saves(c)}
        assert lo_save <= hi_save
        assert lo_save != hi_save


def test_warning_and_error_always_emit_to_stderr(capsys: pytest.CaptureFixture) -> None:
    """Failures are never suppressed, even at show=0, and go to stderr."""
    v = Verbosity(save=0, show=0)
    emit(v, "error", "[FAIL] boom")
    emit(v, "warning", "WARNING: empty split")
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "[FAIL] boom" in captured.err
    assert "WARNING: empty split" in captured.err


def test_ladder_categories_gate_stdout(capsys: pytest.CaptureFixture) -> None:
    v1 = Verbosity(save=0, show=1)
    emit(v1, "progress", "hidden at 1")
    emit(v1, "timing", "shown at 1")
    captured = capsys.readouterr()
    assert "hidden at 1" not in captured.out
    assert "shown at 1" in captured.out
    assert captured.err == ""

    v3 = Verbosity(save=0, show=3)
    emit(v3, "progress", "shown at 3")
    emit(v3, "metrics", "metrics at 3")
    emit(v3, "diagnostics", "hidden at 3")
    captured = capsys.readouterr()
    assert "shown at 3" in captured.out
    assert "metrics at 3" in captured.out
    assert "hidden at 3" not in captured.out


def test_visual_show_requires_tty(monkeypatch: pytest.MonkeyPatch) -> None:
    """A blocking preview must never deadlock a headless run — live
    `visual` needs show>=5 AND an interactive stdout; the save axis is
    TTY-independent (the degrade path is render-and-save)."""
    v = Verbosity(save=5, show=5)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: False)
    assert not v.shows("visual")
    assert v.saves("visual")
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    assert v.shows("visual")
    # Below level 5 it is off even on a TTY.
    assert not Verbosity(save=5, show=4).shows("visual")
