"""A lane partition must be a partition — no path owned twice.

Parallel lanes are safe because ownership is exclusive: if no two lanes can
write the same file, textual merge conflicts cannot occur and the
shared-authority files keep exactly one writer (`docs/parallel_protocol.md`,
rules 1 and 3). That guarantee rests entirely on the declaration in
`docs/lanes/<cycle>.yaml` being disjoint, and a duplicated path removes it
silently — the worktrees are still created, the guard still allows both lanes,
and the collision surfaces only as a conflict at the merge, after the parallel
work is already spent.

So the partition is checked here rather than trusted. These tests run over every
partition file in `docs/lanes/`, so a future cycle's declaration is covered the
moment it is written.
"""
from pathlib import Path

import pytest
import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent

#: `._`-prefixed entries are AppleDouble resource forks the host filesystem
#: creates beside every file on this exFAT volume; they are not partitions. The
#: package filters them the same way wherever it globs artifacts (evaluator.py,
#: data/dataset.py) — see F-69, which is the same sidecar reaching a cache key.
_PARTITIONS = sorted(
    p for p in (_REPO_ROOT / "docs" / "lanes").glob("*.yaml")
    if not p.name.startswith("._")
)

#: One writer each, by rule 3. The ledger is working memory for the loop, and
#: CLAUDE.md and the design spec are the authorities a lane's plan cites — a
#: lane editing the authority it is being judged against defeats the review.
SHARED_AUTHORITY = ("docs/review_ledger.md", "CLAUDE.md", "docs/design_spec.md")

#: Which partition is checked against the LIVE ledger. Older cycles' files stay
#: in docs/lanes/ as the record of what was planned, and describe a ledger state
#: that no longer exists — asserting them against today's ledger would fail for
#: being history rather than for being wrong.
_CURRENT_CYCLE = "cycle5"


def _partitions():
    assert _PARTITIONS, "no partition file in docs/lanes/ — did the directory move?"
    return [(p, yaml.safe_load(p.read_text())) for p in _PARTITIONS]


@pytest.mark.parametrize("path,spec", _partitions(), ids=lambda v: getattr(v, "name", ""))
def test_no_path_is_owned_by_two_lanes(path: Path, spec: dict) -> None:
    owners: dict[str, str] = {}
    for lane in spec["lanes"]:
        for owned in lane["owns"]:
            if owned in owners:
                pytest.fail(
                    f"{path.name}: '{owned}' is owned by BOTH lane {owners[owned]} "
                    f"and lane {lane['id']}. Exclusive ownership is what makes the "
                    "merge conflict-free; give the path to one lane and route the "
                    "other's finding to the integrator queue (rule 4)."
                )
            owners[owned] = lane["id"]


@pytest.mark.parametrize("path,spec", _partitions(), ids=lambda v: getattr(v, "name", ""))
def test_no_lane_owns_a_shared_authority_file(path: Path, spec: dict) -> None:
    for lane in spec["lanes"]:
        for owned in lane["owns"]:
            assert owned not in SHARED_AUTHORITY, (
                f"{path.name}: lane {lane['id']} claims '{owned}', which has exactly "
                "one writer — the integrator. Lanes report through their inbox."
            )


@pytest.mark.parametrize("path,spec", _partitions(), ids=lambda v: getattr(v, "name", ""))
def test_lane_ids_and_inboxes_are_distinct(path: Path, spec: dict) -> None:
    ids = [lane["id"] for lane in spec["lanes"]]
    inboxes = [lane["inbox"] for lane in spec["lanes"]]
    assert len(set(ids)) == len(ids), f"{path.name}: duplicate lane id in {ids}"
    assert len(set(inboxes)) == len(inboxes), (
        f"{path.name}: two lanes share an inbox in {inboxes} — distinct filenames "
        "are what let the inboxes merge without conflict resolution."
    )
    for inbox in inboxes:
        assert inbox.startswith("docs/ledger_inbox/"), (
            f"{path.name}: inbox '{inbox}' is outside docs/ledger_inbox/, where the "
            "integrator looks when folding closures into the ledger."
        )


def _owns(rel: str, patterns: list[str]) -> bool:
    """Same matching rule `scripts/lane_guard.py` applies at edit time.

    Kept in step with the guard deliberately: a row that passes this check and
    is then refused by the hook would be the worst of both, so the two must
    agree on what `dir/**` means.
    """
    for pattern in patterns:
        if pattern.endswith("/**"):
            prefix = pattern[: -len("/**")]
            if rel == prefix or rel.startswith(prefix + "/"):
                return True
        elif rel == pattern:
            return True
    return False


@pytest.mark.parametrize("path,spec", _partitions(), ids=lambda v: getattr(v, "name", ""))
def test_every_row_is_fixable_inside_its_own_lane(path: Path, spec: dict) -> None:
    """A row must be finishable without touching another lane's files.

    Non-overlap alone does not give this. A row can be assigned to the lane that
    owns the code it describes while its FIX or its TEST lands somewhere else —
    F-72 was assigned to lane S with its test class in a file lane P owns, and
    the ownership hook would have refused the edit halfway through the session
    (RD-83). Declaring `fix:` and `test:` per row turns that into a failure at
    declaration time, which costs seconds instead of a session.

    A row that genuinely spans two lanes is not a partition bug — it belongs in
    `integrator_queue:` with a reason (rule 4).
    """
    for lane in spec["lanes"]:
        for row in lane["rows"]:
            for kind in ("fix", "test"):
                for target in row.get(kind, []):
                    assert _owns(target, lane["owns"]), (
                        f"{path.name}: row {row['id']} is assigned to lane "
                        f"{lane['id']}, but its {kind} path '{target}' is outside "
                        f"that lane's owned set. Either give the path to lane "
                        f"{lane['id']}, or move the row to integrator_queue: with "
                        "a reason (rule 4)."
                    )


@pytest.mark.parametrize("path,spec", _partitions(), ids=lambda v: getattr(v, "name", ""))
def test_no_row_id_appears_in_two_places(path: Path, spec: dict) -> None:
    """Every row is assigned exactly once, across all four lists.

    The four lists (lane rows, integrator_queue, awaiting_re_review, and any
    raised against the partition itself) are the partition's coverage claim. A
    row in two of them means two different plans for it; a row in none is the
    silent omission RD-73 exists to prevent.
    """
    seen: dict[str, str] = {}
    buckets = [
        *((lane["id"], [row["id"] for row in lane["rows"]]) for lane in spec["lanes"]),
        ("integrator_queue", [row["id"] for row in spec.get("integrator_queue", [])]),
        ("awaiting_re_review", spec.get("awaiting_re_review", [])),
        ("raised_against_this_partition", spec.get("raised_against_this_partition", [])),
    ]
    for bucket, ids in buckets:
        for row_id in ids:
            assert row_id not in seen, (
                f"{path.name}: row {row_id} appears in both '{seen[row_id]}' and "
                f"'{bucket}'. Each row gets exactly one plan."
            )
            seen[row_id] = bucket


def _open_ledger_ids() -> set[str]:
    """Every row id the ledger currently marks OPEN.

    Status is matched as a PREFIX, not an exact cell. `F-45` is `OPEN (narrowed)`
    and `AC-09` is `DEFERRED (gate: E1 report)` — an exact `| OPEN |` match drops
    the first and an exact `| DEFERRED |` match keeps it, so the naive parser
    miscounts by one in the direction that makes a broken partition look sound
    (RD-88).
    """
    return set(_open_ledger_id_list())


def _open_ledger_id_list() -> list[str]:
    """The same ids as a LIST, so duplicates survive to be asserted on.

    `_open_ledger_ids` returns a set, which silently absorbs a duplicated row id —
    and a duplicated id is exactly the failure cycle 4 hit four ways when four
    parallel lanes each numbered from the ledger's maximum (F-103, RD-126).
    """
    ledger = (_REPO_ROOT / "docs" / "review_ledger.md").read_text()
    ids = []
    for line in ledger.splitlines():
        # A row splits to ['', id, agent, sev, status, ...]: the leading pipe
        # yields an empty cell, so status is index 4 and the id index 1.
        cells = [c.strip() for c in line.split("|")]
        if len(cells) > 5 and cells[4].startswith("OPEN"):
            ids.append(cells[1])
    return ids


def test_no_ledger_row_id_is_duplicated() -> None:
    """Two rows may not share an id.

    The coverage identity below compares SETS, so a duplicate is invisible to it —
    the id would be "covered" while two different findings hid behind one entry.
    That is not hypothetical: cycle 4's four lanes each allocated from the ledger's
    max and `RD-93`…`RD-100` ended up naming four different findings apiece
    (RD-126). The remap fixed the instance; this is what stops the class.
    """
    ids = _open_ledger_id_list()
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    assert not dupes, (
        f"duplicate OPEN row ids in docs/review_ledger.md: {dupes}. Two findings "
        "sharing an id means one of them cannot be cited, assigned or deleted "
        "independently — allocate from the lane's declared id_block (rule 6)."
    )


@pytest.mark.parametrize("path,spec", _partitions(), ids=lambda v: getattr(v, "name", ""))
def test_lane_id_blocks_are_disjoint_and_unused(path: Path, spec: dict) -> None:
    """Rule 6: each lane allocates new ids only from its own declared block, and no
    block overlaps another lane's or an id the ledger already uses.

    Cycle 4 had no such rule. Every lane numbered from the ledger's maximum at the
    moment it started, so `RD-93`…`RD-100` named FOUR different findings and one
    lane's new `AC-` ids landed on rows that were live and unrelated. Untangling it
    took a per-lane, per-CLASS remap of the source tree — and the remap then missed
    a lane's own citations, which a reviewer had to catch (F-104).

    A lane with no `id_block` is allowed only if it declares none at all: lane-scoped
    suffixes (`F-M1`, `AC-42-R1`) are the sanctioned alternative and collide by
    construction with nothing.
    """
    used = {i.rsplit("-", 1)[0]: set() for i in ()}
    for row_id in _open_ledger_id_list():
        prefix, _, num = row_id.rpartition("-")
        if prefix and num.isdigit():
            used.setdefault(prefix, set()).add(int(num))

    seen: dict[tuple[str, int], str] = {}
    for lane in spec["lanes"]:
        for prefix, rng in (lane.get("id_block") or {}).items():
            lo, _, hi = str(rng).partition("..")
            assert lo.isdigit() and hi.isdigit(), (
                f"{path.name}: lane {lane['id']} id_block {prefix}: {rng!r} is not "
                "a 'LO..HI' range."
            )
            for n in range(int(lo), int(hi) + 1):
                assert n not in used.get(prefix, ()), (
                    f"{path.name}: lane {lane['id']}'s block {prefix}-{rng} contains "
                    f"{prefix}-{n}, which is ALREADY an OPEN ledger row. A lane that "
                    "allocates it would overwrite a live finding (RD-126)."
                )
                key = (prefix, n)
                assert key not in seen, (
                    f"{path.name}: {prefix}-{n} is in both lane {seen[key]}'s and "
                    f"lane {lane['id']}'s id_block. Blocks must be disjoint — this "
                    "is the four-way collision of cycle 4 (RD-126)."
                )
                seen[key] = lane["id"]


@pytest.mark.parametrize("path,spec", _partitions(), ids=lambda v: getattr(v, "name", ""))
def test_the_partition_covers_exactly_the_ledgers_open_rows(path: Path, spec: dict) -> None:
    """Every OPEN row has a plan, and every planned row is really OPEN.

    Pairwise uniqueness cannot catch a row that is in NO list — which is the
    omission RD-73 was raised for, and precisely what a coverage figure equal to
    its own scope hides. This is the check `cycle4.yaml` and the ledger both
    CLAIMED was here while nothing read the ledger at all (RD-88).

    Only the current cycle's partition is checked against the live ledger; an
    older cycle's file describes a ledger state that no longer exists.
    """
    if path.stem != _CURRENT_CYCLE:
        pytest.skip(f"{path.name} is not the current cycle ({_CURRENT_CYCLE})")

    planned: set[str] = set()
    for lane in spec["lanes"]:
        planned |= {row["id"] for row in lane["rows"]}
    planned |= {row["id"] for row in spec.get("integrator_queue", [])}
    planned |= set(spec.get("awaiting_re_review", []))
    planned |= set(spec.get("raised_against_this_partition", []))

    open_ids = _open_ledger_ids()
    unplanned = open_ids - planned
    assert not unplanned, (
        f"{path.name}: these rows are OPEN in the ledger but appear in NO list — "
        f"neither a lane, the integrator queue, awaiting_re_review, nor raised "
        f"against the partition: {sorted(unplanned)}. A row with no plan is the "
        "silent omission RD-73 exists to prevent."
    )
    stale = planned - open_ids
    assert not stale, (
        f"{path.name}: these rows are planned but are NOT OPEN in the ledger "
        f"(closed, deleted, or still DEFERRED): {sorted(stale)}. Either the row "
        "was resolved and the partition is stale, or it needs re-statusing "
        "before a lane is told to work it (RD-90)."
    )


@pytest.mark.parametrize("path,spec", _partitions(), ids=lambda v: getattr(v, "name", ""))
def test_every_declared_brief_exists(path: Path, spec: dict) -> None:
    """LANE.md sends the session to its brief; a missing one strands it."""
    for lane in spec["lanes"]:
        brief = _REPO_ROOT / lane["brief"]
        assert brief.exists(), (
            f"{path.name}: lane {lane['id']} points at '{lane['brief']}', which does "
            "not exist. The brief is the session's actual instruction set."
        )
