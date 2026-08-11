"""Create the git worktrees a parallel review cycle runs in.

One lane = one worktree = one Claude Code session (`docs/parallel_protocol.md`).
This script turns a cycle's partition declaration into those worktrees: it
creates each checkout and branch, writes the lane's identity and its ownership
declaration into it, and prints the `cd` lines to open sessions with.

It does NOT launch sessions. A session inherits its terminal's working
directory, and that directory is what assigns it a lane — so the assignment is
made here, before any session exists, and nothing has to be claimed or timed at
run time.

Usage
-----
    python scripts/new_lane.py --partition docs/lanes/cycle4.yaml --all
    python scripts/new_lane.py --partition docs/lanes/cycle4.yaml --lane M
    python scripts/new_lane.py --partition docs/lanes/cycle4.yaml --remove --all

The partition file is the single source of truth for who owns what
(`docs/lanes/cycle4.yaml` is the worked example). Ownership is declared there
once and flows to three consumers — `LANE.md` (what the session reads),
`.claude/lane.json` (what `scripts/lane_guard.py` enforces), and the lane brief
(what the session works from) — so the three cannot disagree.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml

#: This script is tracked at <repo>/scripts/, so the main checkout is two levels
#: up. Worktrees are created as its siblings: v3 -> v3-lane-M.
_MAIN = Path(__file__).resolve().parent.parent

#: The interpreter that ran this script also runs the guard hook and appears in
#: the lane's allow-list entries. Taking it from sys.executable rather than a
#: literal keeps the second declared host (native x86_64) working with no edits.
_PYTHON = Path(sys.executable)
_ENV_BIN = _PYTHON.parent


def _git(*args: str, cwd: Path = _MAIN) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    ).stdout.strip()


def worktree_path(lane_id: str) -> Path:
    return _MAIN.parent / f"{_MAIN.name}-lane-{lane_id}"


def branch_name(lane_id: str, cycle: str) -> str:
    return f"lane/{lane_id}-{cycle}"


def _lane_settings(worktree: Path) -> dict:
    """The lane's `.claude/settings.local.json`: inherited allows + lane wiring.

    Two additions on top of whatever the main checkout has already approved:

    - allow-list entries for the `PYTHONPATH=`-prefixed command form. That prefix
      is mandatory in a worktree (the editable install pins the main tree — see
      `tests/test_source_tree_isolation.py`) and it changes the command string,
      so without these every evidence command in every lane would prompt.
    - the PreToolUse ownership hook, which is what actually enforces rule 1.

    Written per worktree and gitignored, so no lane's wiring reaches another.
    """
    inherited: dict = {}
    main_local = _MAIN / ".claude" / "settings.local.json"
    if main_local.exists():
        inherited = json.loads(main_local.read_text())

    allow = list(inherited.get("permissions", {}).get("allow", []))
    src = worktree / "src"
    for tool in ("pytest", "amcd", "python"):
        entry = f"Bash(PYTHONPATH={src} {_ENV_BIN / tool} *)"
        if entry not in allow:
            allow.append(entry)

    settings = dict(inherited)
    settings["permissions"] = {**inherited.get("permissions", {}), "allow": allow}
    settings["hooks"] = {
        **inherited.get("hooks", {}),
        "PreToolUse": [
            {
                "matcher": "Edit|Write",
                "hooks": [
                    {
                        "type": "command",
                        "command": f"{_PYTHON} {worktree / 'scripts' / 'lane_guard.py'}",
                        "timeout": 10,
                        "statusMessage": "lane ownership check",
                    }
                ],
            }
        ],
    }
    return settings


def _lane_md(lane: dict, cycle: str, worktree: Path, base_branch: str) -> str:
    """The file a session reads to learn which lane it is.

    Generated, never hand-edited: it restates the partition, and a hand-edit here
    would silently disagree with the guard, which reads `.claude/lane.json`.
    """
    owns = "\n".join(f"- `{p}`" for p in lane["owns"])
    # Each row carries the fix/test paths it may touch; showing them here is what
    # lets a session notice a boundary BEFORE the ownership hook refuses an edit.
    rows = "\n".join(
        f"- **{row['id']}** — fix: {', '.join(f'`{p}`' for p in row.get('fix', [])) or '—'}"
        f" · test: {', '.join(f'`{p}`' for p in row.get('test', [])) or '—'}"
        for row in lane["rows"]
    )
    note = f"\n> {lane['note'].strip()}\n" if lane.get("note") else ""
    # Rule 6. Without a declared block every lane numbers from the ledger's max at
    # the moment it starts, and cycle 4 proved where that ends: RD-93..RD-100 named
    # FOUR different findings, and one lane's new AC ids landed on live OPEN rows.
    blocks = lane.get("id_block") or {}
    id_block = "\n".join(
        f"- **`{prefix}-{rng}`**" for prefix, rng in sorted(blocks.items())
    ) or "- _none declared — STOP and ask the integrator before allocating any id._"
    src = worktree / "src"
    return f"""# You are lane {lane['id']} — {lane['title']}

GENERATED by `scripts/new_lane.py` from `docs/lanes/{cycle}.yaml`. Do not edit:
`scripts/lane_guard.py` enforces `.claude/lane.json`, written from the same
declaration, so an edit here changes what you believe and not what is allowed.

- **Cycle:** {cycle}
- **Worktree:** `{worktree}`
- **Branch:** `{branch_name(lane['id'], cycle)}` (from `{base_branch}`)
- **Your brief:** `{lane['brief']}` — read it next; it holds the assigned rows in full.
- **Your inbox:** `{lane['inbox']}` — closures and new findings go HERE.
- **Protocol:** `docs/parallel_protocol.md`
{note}
## Assigned rows

{rows}

## Your row-id block — allocate NEW ids only from here (rule 6)

{id_block}

Any finding you or a reviewer raises on this branch takes its id from that block,
in your inbox. Do NOT number from the ledger's current maximum: every lane is
running at once and would pick the same numbers. In cycle 4 they did —
`RD-93`…`RD-100` ended up naming four different findings, and one lane's new
`AC-` ids collided with rows that were live and unrelated. Untangling it needed a
per-lane, per-class remap of the source tree, and the remap then missed a lane's
own citations and had to be caught by a reviewer.

If you exhaust your block, say so in your inbox and stop — do not borrow.

## Files you own this cycle

{owns}

Every other path is refused by the ownership hook, including
`docs/review_ledger.md`, `CLAUDE.md` and `docs/design_spec.md` — those have one
writer, the integrator. A finding that spans another lane's files is not yours
to fix: record it in your inbox and the integrator applies it after the merge.

## Evidence must come from THIS checkout

The project is installed editable against the main tree, so a bare `pytest` or
`amcd` here would import the main checkout's source and report on code you did
not change. Prefix every command:

```
PYTHONPATH={src} {_ENV_BIN}/pytest
PYTHONPATH={src} {_ENV_BIN}/amcd all -c configs/base.yaml -c configs/overlays/simulator_dry_run.yaml -c configs/overlays/dry_run.yaml
```

`tests/test_source_tree_isolation.py` fails loudly if you forget.

## Preflight — run this first, paste the output

```
PYTHONPATH={src} {_PYTHON} scripts/lane_preflight.py
```

It prints the checkout, branch, lane id and the resolved `amcd` path. If any of
the four disagree with this file, stop and say so rather than working.

## Before you report

1. `git merge {base_branch}` — if it brought anything in that you import, re-run
   your pass condition. Evidence is only valid against the tree it was measured on.
2. Commit on your branch. Do not merge into `{base_branch}` yourself.
3. Write closures and any new findings to `{lane['inbox']}`.
"""


def _inbox_header(lane: dict, cycle: str) -> str:
    """Stamped into the lane's inbox so rule 5 has a mechanism, not just a rule.

    Rules 1 and 3 are enforced by `lane_guard.py`, the partition by
    `test_lane_partition.py`, evidence isolation by
    `tests/test_source_tree_isolation.py`. Rule 5 — a lane-branch review never
    counts as a clean pass — had nothing (RD-85), and it is the one protecting
    the definition of done: a lane that runs `falsifier`, gets zero findings and
    writes "clean" here produces exactly what an integrator misreads as a pass.
    """
    return f"""# Lane {lane['id']} inbox — {cycle}

Branch `{branch_name(lane['id'], cycle)}`. Written by lane {lane['id']}, read by
the integrator.

> **Any reviewer run recorded below is a SELF-CHECK on an unintegrated branch —
> NOT a clean pass** (`docs/parallel_protocol.md`, rule 5). A clean pass is the
> reviewer pass over the merged tree, and only the integrator can produce one.
> Say "self-check on {branch_name(lane['id'], cycle)}", never "clean".

Record, in whatever order things happened: rows you closed with the command and
output that shows it; new findings, anchored by concrete file path; anything you
deliberately did not do, and why. An untouched row with no note is
indistinguishable from a row nobody read.

**Two things this file's format has to get right, both learned in cycle 4:**

- **Give every finding a FILE ANCHOR**, as `path` or `path:line`. The integrator's
  fold copies it into the ledger's anchor column, and that column is what assigns
  the row to a lane next cycle AND what the RD-33a gate counts. Cycle 4 shipped
  116 rows anchored "see inbox" and made the gate's own lift condition
  uncomputable.
- **Number new findings from YOUR id block only** (it is in your `LANE.md`). Every
  lane runs at once, so numbering from the ledger's maximum guarantees collisions.

**This file is PERMANENT, not a scratch pad.** The integrator's fold keeps compact
rows that point back here for the measurements, so it is the primary record for
its findings and is never truncated while an OPEN row cites it — see
`docs/ledger_inbox/README.md`. Write it for a reader in a later cycle who has only
this file and the ledger row that names it.

---
"""


def create(lane: dict, cycle: str, base_branch: str) -> Path:
    lane_id = lane["id"]
    worktree = worktree_path(lane_id)
    branch = branch_name(lane_id, cycle)

    if worktree.exists():
        print(f"  lane {lane_id}: {worktree} already exists, leaving it alone")
        return worktree

    _git("worktree", "add", "-b", branch, str(worktree), base_branch)

    declaration = {
        "lane": lane_id,
        "title": lane["title"],
        "cycle": cycle,
        "worktree": str(worktree),
        "branch": branch,
        "inbox": lane["inbox"],
        # The inbox is the one shared-authority path a lane MUST write, so it is
        # owned by construction rather than by remembering to list it.
        "owns": [*lane["owns"], lane["inbox"]],
        # An absolute path is the only way a lane could reach outside its own
        # worktree; naming the other checkouts is what lets the guard refuse it.
        "forbidden_roots": [str(_MAIN)],
    }

    claude_dir = worktree / ".claude"
    claude_dir.mkdir(exist_ok=True)
    (claude_dir / "lane.json").write_text(json.dumps(declaration, indent=2) + "\n")
    (claude_dir / "settings.local.json").write_text(
        json.dumps(_lane_settings(worktree), indent=2) + "\n"
    )
    (worktree / "LANE.md").write_text(_lane_md(lane, cycle, worktree, base_branch))

    inbox = worktree / lane["inbox"]
    inbox.parent.mkdir(parents=True, exist_ok=True)
    if not inbox.exists():
        inbox.write_text(_inbox_header(lane, cycle))

    print(f"  lane {lane_id} ({lane['title']}): {worktree} on {branch}")
    return worktree


def remove(lane: dict, cycle: str) -> None:
    worktree = worktree_path(lane["id"])
    if not worktree.exists():
        print(f"  lane {lane['id']}: nothing at {worktree}")
        return
    _git("worktree", "remove", "--force", str(worktree))
    print(f"  lane {lane['id']}: removed {worktree} (branch {branch_name(lane['id'], cycle)} kept)")


def _resolve_partition(given: Path, parser: argparse.ArgumentParser) -> Path:
    """Find the partition file whether or not the shell is in the repo.

    A terminal opened fresh starts wherever it starts, so a repo-relative
    `--partition docs/lanes/cycle4.yaml` fails with a bare ENOENT on a path the
    user never typed (`//scripts/...`). Everything else here is anchored to the
    checkout via `__file__`; this argument was the one thing that was not.
    """
    if given.exists():
        return given
    from_repo = _MAIN / given
    if from_repo.exists():
        return from_repo
    parser.error(
        f"no partition file at '{given}' (cwd {Path.cwd()}) or '{from_repo}'. "
        f"Available: {', '.join(sorted(p.name for p in (_MAIN / 'docs' / 'lanes').glob('*.yaml') if not p.name.startswith('._'))) or 'none'}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partition", required=True, type=Path)
    parser.add_argument("--lane", action="append", dest="lanes", metavar="ID")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--remove", action="store_true")
    args = parser.parse_args()

    if not args.all and not args.lanes:
        parser.error("choose lanes with --lane ID (repeatable) or --all")

    spec = yaml.safe_load(_resolve_partition(args.partition, parser).read_text())
    cycle, base_branch = spec["cycle"], spec["base_branch"]
    wanted = spec["lanes"] if args.all else [
        lane for lane in spec["lanes"] if lane["id"] in args.lanes
    ]
    if not wanted:
        parser.error(f"no lane in {args.partition} matched {args.lanes}")

    if args.remove:
        print(f"Removing {cycle} worktrees:")
        for lane in wanted:
            remove(lane, cycle)
        return

    print(f"Creating {cycle} worktrees from {base_branch}:")
    for lane in wanted:
        create(lane, cycle, base_branch)

    print("\nOpen one session per directory — order and timing do not matter:\n")
    print(f"  cd {_MAIN}    # no LANE.md -> integrator")
    for lane in wanted:
        print(f"  cd {worktree_path(lane['id'])}")
    print("\nEach session reads LANE.md on its own; you do not have to tell it anything.")


if __name__ == "__main__":
    main()
