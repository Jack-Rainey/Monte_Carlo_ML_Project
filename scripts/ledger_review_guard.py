#!/usr/bin/env python3
"""Stop-hook guard: `readability-reviewer` must see every ledger edit.

`docs/review_ledger.md` reached 201 KB and ~248 rows across five cycles, roughly
100 lines of it narration of how work was done rather than findings anyone could
act on. The rule that would have prevented it — a reviewer reads the ledger
whenever the ledger changes — is the kind that lives in prose and drifts, which in
this repo it has, three times. So it is enforced here instead.

WHY A `Stop` HOOK AND NOT `PostToolUse`. `Stop` fires when the session is about to
hand control back to the user. A `PostToolUse` hook would trip on the first row of
a planned twenty-five, mid-work, which is both useless and annoying. The unit this
guards is "the user is about to see the result", which is exactly `Stop`.

TWO PROPERTIES THAT KEEP IT FROM WEDGING A SESSION, both deliberate:

  * It blocks AT MOST ONCE per distinct ledger content. If the reviewer genuinely
    cannot run — a five-hour or weekly limit ends the session mid-loop — the
    second stop passes with the warning already delivered. A guard that can strand
    a session would cost more than the sprawl it prevents.
  * It FAILS OPEN on any internal error, and says so on stderr. Same philosophy as
    `lane_guard.py`: a guard that crashes must not also stop real work.

After running the reviewer, record it:

    python scripts/ledger_review_guard.py --record
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LEDGER = REPO / "docs" / "review_ledger.md"
#: Gitignored: which reviewer passes happened in THIS checkout is session state,
#: not a property of the repository, and committing it would make one session's
#: state another session's obligation.
STATE = REPO / ".claude" / "ledger_review_state.json"

_BLOCK_MESSAGE = """\
docs/review_ledger.md changed since the last readability-reviewer pass.

Run it by name over the ledger before finishing:

    Use the readability-reviewer subagent to review docs/review_ledger.md

It checks for what made this file 201 KB: narration of how work was done, rows
that duplicate another row, rows stating no actionable defect, and rows whose
finding column is a pointer rather than a finding. Then record the pass:

    python scripts/ledger_review_guard.py --record

(This blocks once per distinct ledger content. If you cannot run the reviewer,
finishing again will proceed.)"""


def _digest() -> str | None:
    if not LEDGER.exists():
        return None
    return hashlib.sha256(LEDGER.read_bytes()).hexdigest()


def _state() -> dict:
    if not STATE.exists():
        return {}
    try:
        return json.loads(STATE.read_text())
    except (json.JSONDecodeError, OSError):
        # A corrupt marker means "never reviewed", not "crash the session".
        return {}


def _write_state(state: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(state, indent=2) + "\n")


def record() -> int:
    """Mark the current ledger content as reviewed."""
    digest = _digest()
    if digest is None:
        print(f"no ledger at {LEDGER}; nothing to record", file=sys.stderr)
        return 0
    _write_state({"reviewed": digest})
    print(f"recorded readability-reviewer pass over ledger {digest[:12]}")
    return 0


def check() -> int:
    """Block the stop iff the ledger changed and has not been blocked for yet."""
    digest = _digest()
    if digest is None:
        return 0

    state = _state()
    if state.get("reviewed") == digest:
        return 0
    if state.get("blocked_for") == digest:
        # Already told them once about this exact content. Do not strand.
        return 0

    state["blocked_for"] = digest
    _write_state(state)
    # Exit code 2 with stderr is how a Stop hook blocks and feeds its reason back.
    print(_BLOCK_MESSAGE, file=sys.stderr)
    return 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--record",
        action="store_true",
        help="mark the ledger's current content as reviewed by readability-reviewer",
    )
    args = parser.parse_args()
    return record() if args.record else check()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # noqa: BLE001 — failing open is the point
        print(f"ledger_review_guard failed open: {exc!r}", file=sys.stderr)
        sys.exit(0)
