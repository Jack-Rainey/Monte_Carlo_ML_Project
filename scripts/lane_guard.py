"""PreToolUse hook: refuse an Edit/Write outside this lane's declared files.

Exclusive file ownership is what makes parallel lanes safe
(`docs/parallel_protocol.md`, rules 1 and 3): if no two lanes can write the same
file, textual merge conflicts cannot occur and the shared-authority files
(`docs/review_ledger.md`, `CLAUDE.md`, `docs/design_spec.md`) keep exactly one
writer. Rules a session must remember are rules a session eventually forgets
under a long context, so ownership is enforced by the harness instead: this
script runs before every Edit and Write and denies the ones that are out of
scope.

Wired by `scripts/new_lane.py` into the worktree's own
`.claude/settings.local.json` (gitignored, so it never reaches another
checkout). Its declaration is `.claude/lane.json`, written by the same script at
the same moment.

Contract: reads the hook payload on stdin, writes a PreToolUse decision on
stdout, always exits 0 — a hook that crashes is a hook that stops guarding, and
denying real work because the guard itself broke is worse than the risk it
covers. It fails OPEN, and says so on stderr.

No lane declaration (`.claude/lane.json` absent) means this checkout is the
integrator, which owns everything: allow.
"""
from __future__ import annotations

import fnmatch
import json
import sys
from pathlib import Path

#: This script is tracked at <checkout>/scripts/, so its own location is how the
#: guard finds the checkout it is guarding — never a path baked in at setup time,
#: which would survive a moved or copied worktree and guard the wrong tree.
_CHECKOUT = Path(__file__).resolve().parent.parent
_DECLARATION = _CHECKOUT / ".claude" / "lane.json"


def _allow() -> None:
    """Emit nothing: no decision means the normal permission flow proceeds."""
    sys.exit(0)


def _deny(reason: str) -> None:
    json.dump(
        {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": reason,
            }
        },
        sys.stdout,
    )
    sys.exit(0)


def owns(rel_path: str, patterns: list[str]) -> bool:
    """Does `rel_path` (repo-relative, POSIX) fall under any owned pattern?

    `dir/**` means the directory and everything below it; every other pattern is
    a plain fnmatch. Kept deliberately small — the declaration is written by
    `new_lane.py` from the cycle's partition, not hand-authored, so it needs to
    cover the two shapes that partition actually uses.
    """
    for pattern in patterns:
        if pattern.endswith("/**"):
            prefix = pattern[: -len("/**")]
            if rel_path == prefix or rel_path.startswith(prefix + "/"):
                return True
        elif fnmatch.fnmatchcase(rel_path, pattern):
            return True
    return False


def decide(target: Path, lane: dict) -> str | None:
    """Return a denial reason, or None to allow.

    Three cases, in order: a path inside another checkout of this repo is always
    refused (an absolute path is the one way a lane could reach around its own
    worktree); a path inside this checkout is checked against the owned set; a
    path elsewhere — a scratchpad, a temp file — is not shared state and is
    allowed.
    """
    for root in lane.get("forbidden_roots", []):
        forbidden = Path(root).resolve()
        if target == forbidden or forbidden in target.parents:
            return (
                f"Lane {lane['lane']} may not write into another checkout of this "
                f"repository ({forbidden}). Work in your own worktree "
                f"({_CHECKOUT}); the integrator owns the main tree. See "
                "docs/parallel_protocol.md."
            )

    if _CHECKOUT not in target.parents:
        return None

    rel = target.relative_to(_CHECKOUT).as_posix()
    if owns(rel, lane.get("owns", [])):
        return None

    return (
        f"{rel} is not owned by lane {lane['lane']} this cycle.\n"
        f"Lane {lane['lane']} owns: {', '.join(lane.get('owns', [])) or '(nothing)'}\n"
        "Findings that span two lanes' files are not parallelized — record the "
        f"finding in {lane.get('inbox', 'your inbox')} and the integrator applies "
        "it after the merge (docs/parallel_protocol.md, rule 4)."
    )


def main() -> None:
    if not _DECLARATION.exists():
        _allow()  # integrator: no lane declaration, owns everything

    try:
        payload = json.load(sys.stdin)
        lane = json.loads(_DECLARATION.read_text())
        file_path = payload.get("tool_input", {}).get("file_path")
    except (json.JSONDecodeError, OSError) as exc:
        print(f"lane_guard: failing open, could not read its inputs: {exc}", file=sys.stderr)
        _allow()

    if not file_path:
        _allow()  # a tool call with no path to check

    reason = decide(Path(file_path).resolve(), lane)
    if reason is None:
        _allow()
    _deny(reason)


if __name__ == "__main__":
    main()
