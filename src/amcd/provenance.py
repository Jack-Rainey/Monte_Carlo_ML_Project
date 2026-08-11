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

`host_platform()` and `select_device()` sit on the human side too: the same config
and the same `code_version` produce different weights on this Mac (MPS) and on the
x86 box (CUDA/CPU), so the host and the device belong in `versions.json` — and
deliberately in no cache key, since moving machines must not discard a checkpoint
(F-74). `select_device` explains why runtime policy lives in this module.
"""
from __future__ import annotations

import hashlib
import platform
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
#:
#: `__init__.py` is the PACKAGE ROOT's — `amcd/__init__.py`, which every stage
#: imports through and which no per-stage scope named, so an edit to it used to
#: invalidate nothing. Subpackage `__init__.py` files need no entry: they are
#: covered by their own subpackage's scope entry.
_CORE_SOURCES = (
    "__init__.py", "config.py", "runtime.py", "registry.py", "acoustics.py",
    "provenance.py",
)

#: Every source file, for the human-facing `versions.json` stamp — "which code was
#: this run made with", where over-inclusion costs nothing. The `"."` is the
#: package root, resolved like any other scope entry and so hashing the whole tree.
#: Never a cache key: the per-stage scopes are, for the reason stated in
#: `code_version`.
ALL_SOURCES: tuple[str, ...] = (".",)


def _hashable_sources(target: Path) -> list[Path]:
    """The `.py` files under `target` that describe the CODE, in a stable order.

    Two classes are excluded, both because they make the hash describe the HOST
    rather than the source (F-69):

    * `._*` — macOS AppleDouble sidecars. This repo lives on an exFAT volume,
      where they are real files that `rglob("*.py")` matches; 42 `._*.py` sat
      under `src/amcd/` when this was written. They do not exist on APFS or on the
      project's declared second host, so the same source hashed to a different key
      there and a run_dir carried between hosts was refused with a `code_version:
      <sha> → <sha>` diff naming no leaf — leaving `--force` as the only remedy,
      which is exactly the compliance failure the scoping rationale exists to
      avoid. `evaluation/` already filters `._` elsewhere; this glob did not.
    * `__pycache__` — build output, not source. Matched against the path RELATIVE
      to `target`, never the absolute one: an ancestor directory that happens to be
      named `__pycache__` (a checkout under a build tree, a packaging temp dir)
      would otherwise exclude every file and collapse every scope to one constant.

    Raises if a directory yields nothing, for the reason `code_version`'s own
    ValueError gives: a scope entry that hashes to a constant silently stops
    protecting the stage, and "the directory exists but holds no `.py`" reaches
    that outcome without ever naming a missing path (F-79).
    """
    if not target.is_dir():
        return [target]
    found = sorted(
        p for p in target.rglob("*.py")
        if not p.name.startswith("._")
        and "__pycache__" not in p.relative_to(target).parts
    )
    if not found:
        raise ValueError(
            f"code_version scope entry {target} is a directory containing no "
            f"hashable .py source. It would contribute nothing to the hash, so the "
            f"stage would keep serving cached artifacts under changed code. Name a "
            f"module or a subpackage that holds source."
        )
    return found


def code_version(scope: tuple[str, ...]) -> str:
    """A content hash over the `.py` files a stage's output actually depends on.

    `scope` is a tuple of package-relative paths — a module (`"config.py"`) or a
    subpackage (`"evaluation"`) — and `_CORE_SOURCES` is added to every scope.
    Declaring the scope is the point: it states, per stage, which code the artifact
    is a function of, and that claim is auditable.

    Why a content hash and not `git rev-parse HEAD` (F-55 / RD-66): the sha is
    blind to the working tree, which is the exact state the guard exists for —
    this project's loop is edit → run → review → commit, so the code under review
    is uncommitted by definition. The sha also fails in two directions off a
    checkout: a wheel in site-packages returns "unavailable" permanently, and an
    install inside an UNRELATED repo returns that repo's sha, which moves on
    commits touching nothing here. A content hash has none of those failure modes.

    Why SCOPED rather than whole-package — the argument lives HERE, and
    `pipeline.STAGE_CODE_SCOPE` cites it rather than restating it. A hash over
    every file cannot miss a dependency, which is the safer direction, but it also
    invalidates `train` on a docstring edit in `reporting/`; a guard that refuses a
    cached stage for reasons the operator can see are irrelevant teaches them to
    reach for `--force`, which disables the guard entirely. Precision buys
    compliance.

    The residual risk is a scope that omits a real dependency, and it fails
    silently. `tests/test_stage_cache.py` bounds that risk but does not eliminate
    it: it asserts the declared scope is a superset of the stage's STATIC
    transitive `amcd.*` import closure, module-level and function-level, minus
    `_CORE_SOURCES`. What that cannot see is a dependency reached without an
    import statement — above all the plugin registry, through which
    `representations`, `models` and `simulators` are loaded by NAME. Those stay a
    declared judgement, checked by review and not by the test. Stating this
    precisely is deliberate: the previous wording claimed the test checked "the
    modules the stage actually imports", when it only checked that the stage's own
    entry-point subpackage was in scope, and the overstatement was itself the
    finding (F-66).
    """
    digest = hashlib.sha256()
    for entry in sorted(set(scope) | set(_CORE_SOURCES)):
        target = _PACKAGE_ROOT / entry
        paths = _hashable_sources(target)
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


def host_platform() -> str:
    """The machine architecture this run executed on, e.g. `arm64` or `x86_64`.

    HUMAN provenance, beside `git_sha()`. The project is required to run on this
    Apple-Silicon machine and on a native x86_64 desktop from the same code, so
    "which host" is part of describing a run — never a cache key (see
    `select_device`).
    """
    return platform.machine()


def select_device():
    """The torch device this host offers, preferring MPS, then CUDA, then CPU.

    NOT provenance, and not here by design: this is runtime policy, and it lives
    in `provenance.py` because `Config.stamp` must record the chosen device into
    `versions.json` and `config.py` is in `_CORE_SOURCES`. A core module importing
    `amcd.training` would put `training/` in EVERY stage's import closure and make
    every stage's declared scope wrong. The natural home is a top-level
    `amcd/device.py`; lane P's cycle-4 file ownership does not include one, so this
    placement is a constraint, not a claim about where it belongs.

    The cost of living in a core module, recorded so it is a decision and not a
    surprise: an edit HERE invalidates every fingerprinted stage. That is
    tolerable while `gen-scenes`/`render` carry no `code_version`; if they gain
    one, editing this fallback would force a re-render, which under x86 emulation
    is the multi-hour artifact — and this fallback is exactly the code the
    second-host requirement will make someone touch.

    The device is deliberately absent from every fingerprint (F-74). The same
    config on MPS and on CUDA produces different weights, which is a real
    provenance fact and belongs in `versions.json`; making it a cache key would
    discard an expensive checkpoint because the operator moved machines.
    """
    import torch

    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


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
