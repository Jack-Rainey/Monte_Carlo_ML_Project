"""Confirm a lane session is in the tree it thinks it is, before it edits anything.

Four signals identify a lane (`docs/parallel_protocol.md`): the directory, the
`LANE.md` in it, the branch, and the source the interpreter actually imports.
Three of those are read from files a session could misread; the fourth is what
its evidence will really measure. This prints all four together so a
disagreement is visible in one glance rather than discovered in a result.

Run it with the same `PYTHONPATH` prefix every other lane command uses — the
point is to exercise the real command form, not an approximation of it:

    PYTHONPATH=<worktree>/src <python> scripts/lane_preflight.py

Exits non-zero if the imported package comes from another checkout, which is the
one failure that would otherwise be silent.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

_CHECKOUT = Path(__file__).resolve().parent.parent


def main() -> int:
    declaration = _CHECKOUT / ".claude" / "lane.json"
    if declaration.exists():
        lane = json.loads(declaration.read_text())
        identity = f"lane {lane['lane']} ({lane['title']}), cycle {lane['cycle']}"
        owns = lane["owns"]
    else:
        identity = "INTEGRATOR (no .claude/lane.json — this is the main checkout)"
        owns = ["everything"]

    branch = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        cwd=_CHECKOUT, capture_output=True, text=True,
    ).stdout.strip()

    try:
        import amcd
        imported = Path(amcd.__file__).resolve().parent.parent.parent
    except ImportError as exc:
        print(f"amcd could not be imported at all: {exc}")
        return 1

    print(f"identity : {identity}")
    print(f"checkout : {_CHECKOUT}")
    print(f"branch   : {branch}")
    print(f"amcd from: {imported}")
    print(f"owns     : {', '.join(owns)}")

    if imported != _CHECKOUT:
        print(
            f"\nSTOP: amcd resolves to {imported}, not this checkout. Any evidence "
            f"gathered now describes that tree.\nRe-run with "
            f"PYTHONPATH={_CHECKOUT / 'src'} prefixed."
        )
        return 1

    print("\nOK — imports, branch and identity all agree.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
