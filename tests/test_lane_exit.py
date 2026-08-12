"""A lane does not get to report while its own reviewers are unsatisfied.

`docs/parallel_protocol.md` rule 5 says a lane-branch review never counts toward
a clean pass. That is about AUTHORITY, and it is right. It says nothing about
what a lane must DO, and for two cycles it was read as permission: "a lane MAY
run reviewers on its branch as a cheap self-check."

Cycle 5 is what that bought. All four lanes ran reviewers. All four fixed what
those reviewers raised. **Not one re-ran a reviewer over the fixed tree**, so
roughly sixty fixes arrived at the integrator as claims nobody had re-derived —
against a definition of done whose whole point is that an unverified fix is a
claim. Lane S never invoked `research-director` at all. Lane B's readability
findings had no findings table: eighteen ids in prose, several with no file
anchor, which the fold would have dropped silently.

None of that was a tooling failure — `.claude/agents/` is copied into every
worktree and every lane reached the agents. It was structural. `LANE.md`, the
file CLAUDE.md tells each session to read first, listed three steps under "Before
you report": merge base, commit, write the inbox. Reviewers were not among them.

So the exit gate is generated into `LANE.md` (where the session reads it) and
recorded in a `## LANE EXIT` block in the inbox (where it can be checked), and
this file checks it. Six conditions, in the protocol under "The lane exit gate".

**Deliberately NOT retroactive.** The block is asserted only for partitions
declaring `exit_gate: required`. Cycle 5's four inboxes predate the gate — the
string does not appear in any of them — and backfilling would mean the integrator
authoring reviewer evidence for passes that were never run. That is the exact
thing this machine exists to forbid, so a partition without the declaration is
asserted to be legally missing rather than quietly skipped.
"""
import re
from pathlib import Path

import pytest
import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent
_PARTITIONS = sorted(
    p for p in (_REPO_ROOT / "docs" / "lanes").glob("*.yaml")
    if not p.name.startswith("._")
)

#: Field name -> whether an empty value is legal. `unfixed_in_lane` may be
#: "none"; the rest must say something.
_REQUIRED_FIELDS = (
    "final_commit",
    "reviewers_last_run_on",
    "last_pass_clean",
    "unfixed_in_lane",
    "evidence_remeasured_on_final_commit",
    "gate_lifts",
    "gate_unblocks",
)

_REVIEWERS = (
    "research-director",
    "falsifier",
    "acoustics-reviewer",
    "readability-reviewer",
)

_FIELD_RE = re.compile(r"^\s*[-*]\s*\*\*(?P<key>[a-z_]+):\*\*\s*(?P<value>.*)$")
#: A value still holding its generated placeholder has not been filled in.
_PLACEHOLDER_RE = re.compile(r"<[^>]*>")


def _partitions():
    assert _PARTITIONS, "no partition file in docs/lanes/ — did the directory move?"
    return [(p, yaml.safe_load(p.read_text())) for p in _PARTITIONS]


def _exit_block(inbox: Path) -> dict[str, str] | None:
    """Parse the `## LANE EXIT` block, or None if the inbox has no such block."""
    text = inbox.read_text()
    if "## LANE EXIT" not in text:
        return None
    tail = text.split("## LANE EXIT", 1)[1]
    end = tail.find("\n## ")
    body = tail if end == -1 else tail[:end]
    return {
        m.group("key"): m.group("value").strip()
        for line in body.splitlines()
        if (m := _FIELD_RE.match(line))
    }


def _gated(spec: dict) -> bool:
    return str(spec.get("exit_gate", "")).lower() == "required"


# ── The rules, as pure functions ─────────────────────────────────────────────
#
# Extracted so they can be exercised directly. Every partition in the repo today
# declares `exit_gate: not_required` — correctly, since all of them predate the
# gate — so the parametrized tests below all SKIP, and a check that only ever
# skips is indistinguishable from one that does not work. `TestTheRulesThemselves`
# runs these against synthetic blocks, including the four shapes cycle 5 actually
# produced.


def _missing_reviewers(block: dict[str, str]) -> list[str]:
    ran = block.get("reviewers_last_run_on", "")
    return [r for r in _REVIEWERS if r not in ran]


def _unfilled_fields(block: dict[str, str]) -> list[str]:
    out = []
    for field in _REQUIRED_FIELDS:
        if field not in block or _PLACEHOLDER_RE.search(block[field]):
            out.append(field)
    return out


def _says_yes(block: dict[str, str], field: str) -> bool:
    return block.get(field, "").strip().lower().startswith("yes")


def _reviewers_off_final_commit(block: dict[str, str]) -> list[tuple[str, str]]:
    """Reviewers whose last run is a commit OTHER than the lane's final one.

    `research-director` is exempt: it runs on the PLAN, before a commit exists.
    """
    final = block.get("final_commit", "").strip("` ")
    if not final:
        return []
    stale = []
    for reviewer in _REVIEWERS:
        if reviewer == "research-director":
            continue
        m = re.search(
            rf"{re.escape(reviewer)}\s*=\s*`?([0-9a-f]{{7,40}})`?",
            block.get("reviewers_last_run_on", ""),
        )
        if m and not (
            final.startswith(m.group(1)[:7]) or m.group(1).startswith(final[:7])
        ):
            stale.append((reviewer, m.group(1)[:7]))
    return stale


def _unfixed_without_a_reason(block: dict[str, str]) -> list[str]:
    raw = block.get("unfixed_in_lane", "").strip()
    if raw.lower() in ("none", "-", ""):
        return []
    return [
        e.strip()
        for e in raw.split(";")
        if e.strip() and "blocked by" not in e.lower() and "cluster" not in e.lower()
    ]


@pytest.mark.parametrize("path,spec", _partitions(), ids=lambda v: getattr(v, "name", ""))
def test_a_partition_without_the_gate_says_so_explicitly(path: Path, spec: dict) -> None:
    """`exit_gate` is a declaration, not an absence.

    Skipping silently when the key is missing is how a check quietly stops
    running — RD-148's shape, where a stale `_CURRENT_CYCLE` suspended every
    ledger-coupled test with a green suite. A partition that predates the gate
    must SAY `exit_gate: not_required` with a reason, so "no block" is a recorded
    decision rather than an oversight.
    """
    declared = spec.get("exit_gate")
    assert declared in ("required", "not_required"), (
        f"{path.name}: `exit_gate:` is {declared!r}. Declare 'required' (every "
        "lane fills in a ## LANE EXIT block) or 'not_required' (this partition "
        "predates the gate — say why in a comment). Leaving it unset means this "
        "file's checks skip without anyone deciding that they should."
    )


@pytest.mark.parametrize("path,spec", _partitions(), ids=lambda v: getattr(v, "name", ""))
def test_every_gated_lane_satisfies_the_exit_gate(path: Path, spec: dict) -> None:
    """All six conditions, over every lane of a partition that declares the gate."""
    if not _gated(spec):
        pytest.skip(f"{path.name} declares exit_gate: not_required")

    for lane in spec["lanes"]:
        inbox = _REPO_ROOT / lane["inbox"]
        if not inbox.exists():
            continue  # the lane has not reported yet
        where = lane["inbox"]
        block = _exit_block(inbox)

        assert block is not None, (
            f"{where}: no `## LANE EXIT` block. The exit gate is what stops a lane "
            "reporting while its own reviewers are unsatisfied."
        )
        assert not (unfilled := _unfilled_fields(block)), (
            f"{where}: `## LANE EXIT` fields {unfilled} are missing or still hold "
            "their generated placeholder — the block was emitted, never filled in."
        )
        assert not (missing := _missing_reviewers(block)), (
            f"{where}: `reviewers_last_run_on` does not name {missing}. All four "
            "run, by name — auto-delegation is unreliable, and each covers a risk "
            "the others do not (L1)."
        )
        assert _says_yes(block, "last_pass_clean"), (
            f"{where}: `last_pass_clean` is {block.get('last_pass_clean')!r}. "
            "Re-run the reviewers over the final commit and fix what they raise "
            "until a pass returns zero new in-lane-fixable findings. A pass whose "
            "findings you then fixed is not the last pass (L2)."
        )
        assert not (stale := _reviewers_off_final_commit(block)), (
            f"{where}: {stale} last ran on a commit other than "
            f"{block.get('final_commit')!r}. The reviewed tree is not the tree "
            "that ships (L2)."
        )
        assert _says_yes(block, "evidence_remeasured_on_final_commit"), (
            f"{where}: `evidence_remeasured_on_final_commit` is "
            f"{block.get('evidence_remeasured_on_final_commit')!r}. Evidence is "
            "only valid against the tree it was measured on — re-run the suite and "
            "the ci_table.csv A/B after the last fix (L4)."
        )
        assert not (unreasoned := _unfixed_without_a_reason(block)), (
            f"{where}: unfixed rows {unreasoned} name neither a blocking file "
            "outside the lane's owned set nor a cluster they must close with. "
            "Those are the only two reasons a reachable finding ships unfixed (L3)."
        )


_GOOD = {
    "final_commit": "`4dfe46c`",
    "reviewers_last_run_on": (
        "research-director=`PLAN`, falsifier=`4dfe46c`, "
        "acoustics-reviewer=`4dfe46c`, readability-reviewer=`4dfe46c`"
    ),
    "last_pass_clean": "yes",
    "unfixed_in_lane": "RR-165 — blocked by tests/test_record_length_gate.py (new file, outside owns)",
    "evidence_remeasured_on_final_commit": "yes",
    "gate_lifts": "none",
    "gate_unblocks": "none",
}


class TestTheRulesThemselves:
    """Exercise the six rules directly, on the shapes cycle 5 actually produced.

    Every partition in the repo declares `exit_gate: not_required`, so the
    parametrized test above skips — and a guard that has only ever skipped is
    indistinguishable from one that does not work. These run the same predicates
    the guard runs, against a known-good block and against each real cycle-5
    failure, so the gate is shown to discriminate before any lane depends on it.
    """

    def test_a_complete_block_passes_every_rule(self) -> None:
        assert _unfilled_fields(_GOOD) == []
        assert _missing_reviewers(_GOOD) == []
        assert _says_yes(_GOOD, "last_pass_clean")
        assert _reviewers_off_final_commit(_GOOD) == []
        assert _says_yes(_GOOD, "evidence_remeasured_on_final_commit")
        assert _unfixed_without_a_reason(_GOOD) == []

    def test_lane_s_shape_is_caught_research_director_never_ran(self) -> None:
        """Lane S ran three of four. Nothing was looking, so nothing noticed."""
        block = _GOOD | {
            "reviewers_last_run_on": (
                "falsifier=`c799003`, acoustics-reviewer=`c799003`, "
                "readability-reviewer=`c799003`"
            )
        }
        assert _missing_reviewers(block) == ["research-director"]

    def test_the_cycle_5_shape_is_caught_reviewers_predate_the_final_commit(self) -> None:
        """The defect every lane had: reviewed at X, fixed, shipped at Y.

        Lane B's reviewers ran at `ab1f47e`; it then fixed nine of its own defects
        in `bc03471` and reported. The tree that was reviewed is not the tree that
        shipped, and no field in the old inbox format made that visible.
        """
        block = _GOOD | {
            "final_commit": "`bc03471`",
            "reviewers_last_run_on": (
                "research-director=`PLAN`, falsifier=`ab1f47e`, "
                "acoustics-reviewer=`ab1f47e`, readability-reviewer=`ab1f47e`"
            ),
        }
        stale = _reviewers_off_final_commit(block)
        assert [r for r, _ in stale] == [
            "falsifier", "acoustics-reviewer", "readability-reviewer"
        ]
        assert all(sha == "ab1f47e" for _, sha in stale)

    def test_an_unclean_last_pass_is_caught(self) -> None:
        assert not _says_yes(_GOOD | {"last_pass_clean": "no"}, "last_pass_clean")

    def test_lane_b_shape_is_caught_evidence_predates_the_final_fix(self) -> None:
        """Lane B's suite and A/B numbers predated its final commit by three tests."""
        block = _GOOD | {"evidence_remeasured_on_final_commit": "no"}
        assert not _says_yes(block, "evidence_remeasured_on_final_commit")

    def test_an_unfixed_row_with_no_stated_blocker_is_caught(self) -> None:
        """"Ran out of time" wearing a different word is the case this exists for."""
        assert _unfixed_without_a_reason(_GOOD | {"unfixed_in_lane": "F-123"}) == ["F-123"]
        assert _unfixed_without_a_reason(
            _GOOD | {"unfixed_in_lane": "F-123 — deferred to next cycle"}
        ) == ["F-123 — deferred to next cycle"]
        assert _unfixed_without_a_reason(_GOOD | {"unfixed_in_lane": "none"}) == []
        assert _unfixed_without_a_reason(
            _GOOD | {"unfixed_in_lane": "AC-54 — cluster C6"}
        ) == []

    def test_an_unfilled_placeholder_is_caught(self) -> None:
        """The block is generated pre-filled; emitting it is not completing it."""
        assert _unfilled_fields(_GOOD | {"final_commit": "<sha of your last commit>"}) == [
            "final_commit"
        ]

    def test_the_block_parses_out_of_the_generated_template(self) -> None:
        """The parser and `scripts/new_lane.py`'s emitter must agree on the format.

        A parser that silently returns `{}` for the real template would make every
        field "missing" — or, with a different bug, make the whole gate vacuous.
        """
        import sys

        sys.path.insert(0, str(_REPO_ROOT / "scripts"))
        from new_lane import _inbox_header  # noqa: PLC0415

        lane = {"id": "Z", "inbox": "docs/ledger_inbox/Z.md"}
        tmp = _REPO_ROOT / "docs" / "ledger_inbox" / ".exit_template_probe.md"
        try:
            tmp.write_text(_inbox_header(lane, "cycleN"))
            parsed = _exit_block(tmp)
        finally:
            tmp.unlink(missing_ok=True)
        assert parsed is not None, "the generated inbox has no ## LANE EXIT block"
        assert set(parsed) == set(_REQUIRED_FIELDS), (
            f"emitter and parser disagree: template yields {sorted(parsed)}, "
            f"guard requires {sorted(_REQUIRED_FIELDS)}"
        )
        assert sorted(_unfilled_fields(parsed)) == sorted(_REQUIRED_FIELDS), (
            "a freshly generated block must read as UNFILLED, or a lane that "
            "ignores it entirely would pass the gate"
        )
