"""One source of truth for "which code produced this artifact".

Two channels record it — `versions.json` (what a human reads) and the eval stage
sentinel (what the cache compares) — and they must not be able to disagree, so
both import from here rather than each shelling out to git (F-56).

The distinction the module exists to keep straight:

* `code_version()` is the CACHE key. It must change whenever the code changes,
  including edits that are not committed, because this project's loop is
  edit → run → review → commit and the review runs against a dirty tree.
* `git_sha()` is HUMAN provenance. It is allowed to be "unavailable", and it is
  never the only thing standing between a config change and a stale artifact.
"""
from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

#: The package root — `src/amcd/`. Resolved from this file so it is correct
#: whether the package is an editable checkout or an installed wheel, and never
#: from the run directory, which may sit on a data volume in a different repo
#: (F-56) or in no repo at all.
_PACKAGE_ROOT = Path(__file__).resolve().parent


#: Modules any stage's output can depend on, so they are in EVERY scope: the
#: config that parameterizes it, the seed derivation, the plugin lookup, and the
#: shared acoustic formulas. `pipeline.py` and `cli.py` are deliberately absent —
#: they orchestrate stages, they do not compute artifact content, so including
#: them would invalidate every cached stage whenever a fingerprint declaration is
#: edited, which is this very file's kind of change.
_CORE_SOURCES = ("config.py", "runtime.py", "registry.py", "acoustics.py", "provenance.py")

#: Every source file, for the human-facing `versions.json` stamp — "which code was
#: this run made with", where over-inclusion costs nothing. Never a cache key: the
#: per-stage scopes are, for the reason stated in `code_version`.
ALL_SOURCES: tuple[str, ...] = (".",)


def code_version(scope: tuple[str, ...]) -> str:
    """A content hash over the `.py` files a stage's output actually depends on.

    `scope` is a tuple of package-relative paths — a module (`"config.py"`) or a
    subpackage (`"evaluation"`) — and `_CORE_SOURCES` is added to every scope.
    Declaring the scope is the point: it states, per stage, which code the artifact
    is a function of, and that claim is auditable.

    Why a content hash and not `git rev-parse HEAD` (F-55 / RD-66). The sha is
    blind to the working tree, which is the exact state the guard exists for: an
    uncommitted edit to `evaluation/room_acoustic.py` left `amcd eval` printing
    `[skip] eval (cached)` and serving `metrics.parquet` under the new code — and
    a code-only edit is how AC-17 was made, so this is the project's normal loop,
    not an edge case. The sha also fails in two directions off a checkout: a wheel
    in site-packages returns "unavailable" permanently, and an install inside an
    UNRELATED git repo returns that repo's sha, which then moves on commits that
    touch nothing here. A content hash has none of those failure modes.

    Why SCOPED rather than whole-package. A hash over every file cannot miss a
    dependency, which is the safer direction — but it also invalidates `train` on
    a docstring edit in `reporting/`, and a guard that refuses a cached stage for
    reasons the operator can see are irrelevant teaches them to reach for
    `--force`, which disables the guard entirely. Precision here buys compliance.
    The residual risk is a scope that omits a real dependency; a stage importing
    outside its declared scope is a coupling defect in its own right, and
    `tests/test_stage_cache.py` asserts each declared scope against the modules
    the stage actually imports.
    """
    digest = hashlib.sha256()
    for entry in sorted(set(scope) | set(_CORE_SOURCES)):
        target = _PACKAGE_ROOT / entry
        paths = sorted(target.rglob("*.py")) if target.is_dir() else [target]
        for path in paths:
            if not path.exists():
                raise ValueError(
                    f"declared code_version scope entry {entry!r} does not exist in "
                    f"{_PACKAGE_ROOT}. A scope that names a missing path would hash "
                    f"to a constant and silently stop protecting the stage."
                )
            # The relative path participates, so moving or renaming a module
            # changes the version even when the bytes are unchanged.
            digest.update(path.relative_to(_PACKAGE_ROOT).as_posix().encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def git_sha() -> str:
    """The commit sha of the checkout containing this package, or "unavailable".

    Resolved from the PACKAGE, never from the run directory: a `--run-dir` on a
    data volume is the normal case, and asking git about it stamped
    `"git_sha": "unavailable"` into `versions.json` while the same run's eval
    sentinel recorded a real sha — two provenance channels disagreeing about one
    run (F-56).

    Human-readable provenance only. `code_version()` is what the cache compares,
    so an "unavailable" sha here weakens nothing.
    """
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=_PACKAGE_ROOT,
            capture_output=True, text=True, timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return "unavailable"
    return out.stdout.strip() if out.returncode == 0 else "unavailable"


def git_is_dirty() -> bool:
    """Whether the checkout has uncommitted changes, or False if git cannot say.

    Reported beside `git_sha()` so a human reading `versions.json` knows the sha
    does not fully describe the code. The cache does not consult this — that is
    `code_version()`'s job — so a False from a git-less install is not a hole.
    """
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=_PACKAGE_ROOT,
            capture_output=True, text=True, timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return out.returncode == 0 and bool(out.stdout.strip())
