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
import re
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
        ("serial_queue", [row["id"] for row in spec.get("serial_queue", [])]),
        ("integrator_queue", [row["id"] for row in spec.get("integrator_queue", [])]),
        ("awaiting_re_review", spec.get("awaiting_re_review", [])),
        ("raised_against_this_partition", spec.get("raised_against_this_partition", [])),
        ("unassigned", spec.get("unassigned", [])),
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

    # A CONSUMED block is not a collision. Once a lane reports and its findings
    # are folded, ids from its own block are live OPEN rows BY DESIGN — that is
    # rule 6 working, not failing. What RD-126 forbids is a lane allocating an id
    # that names SOMEONE ELSE's live finding, so an id is exempt here only if the
    # lane's own inbox is where it came from.
    own: dict[str, set[str]] = {}
    for lane in spec["lanes"]:
        inbox = _REPO_ROOT / lane["inbox"]
        own[lane["id"]] = set(_ID_RE.findall(inbox.read_text())) if inbox.exists() else set()

    seen: dict[tuple[str, int], str] = {}
    for lane in spec["lanes"]:
        for prefix, rng in (lane.get("id_block") or {}).items():
            lo, _, hi = str(rng).partition("..")
            assert lo.isdigit() and hi.isdigit(), (
                f"{path.name}: lane {lane['id']} id_block {prefix}: {rng!r} is not "
                "a 'LO..HI' range."
            )
            for n in range(int(lo), int(hi) + 1):
                assert (
                    n not in used.get(prefix, ())
                    or f"{prefix}-{n}" in own[lane["id"]]
                ), (
                    f"{path.name}: lane {lane['id']}'s block {prefix}-{rng} contains "
                    f"{prefix}-{n}, which is ALREADY an OPEN ledger row raised "
                    "somewhere other than this lane's own inbox. A lane that "
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

    `unassigned:` is the FIFTH list, added for RD-146. A cycle's fold creates rows
    the partition could not have planned, because they did not exist when it was
    drawn — cycle 5's fold created 142. Without a bucket for them this check goes
    red between every fold and the next partition, and the pressure is then to
    weaken the check rather than to record the rows. `unassigned:` keeps the
    coverage identity total while saying plainly that these await a partition.
    """
    if path.stem != _CURRENT_CYCLE:
        pytest.skip(f"{path.name} is not the current cycle ({_CURRENT_CYCLE})")

    planned: set[str] = set()
    for lane in spec["lanes"]:
        planned |= {row["id"] for row in lane["rows"]}
    planned |= {row["id"] for row in spec.get("serial_queue", [])}
    planned |= {row["id"] for row in spec.get("integrator_queue", [])}
    planned |= set(spec.get("awaiting_re_review", []))
    planned |= set(spec.get("raised_against_this_partition", []))
    planned |= set(spec.get("unassigned", []))

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


def test_the_current_cycle_names_the_newest_partition() -> None:
    """RD-148: `_CURRENT_CYCLE` is a module literal that nothing asserted.

    Every check below that matters — the coverage identity, the id blocks, the
    anchor check — skips when `path.stem != _CURRENT_CYCLE`. If the literal ever
    names a stem with no file, `pytest.skip` fires for EVERY parametrization and
    the whole coverage guarantee is suspended with a green suite. That is RD-88's
    own "asserted by nothing" shape, one level up: the guard reports success
    because it never ran.
    """
    current = _REPO_ROOT / "docs" / "lanes" / f"{_CURRENT_CYCLE}.yaml"
    assert current.exists(), (
        f"_CURRENT_CYCLE is '{_CURRENT_CYCLE}' but {current} does not exist, so "
        "every ledger-coupled test in this file silently skips. Bump the literal "
        "when you open a cycle, or restore the partition file."
    )
    assert current == _PARTITIONS[-1], (
        f"_CURRENT_CYCLE is '{_CURRENT_CYCLE}' but the newest partition in "
        f"docs/lanes/ is '{_PARTITIONS[-1].name}'. The live ledger is checked "
        "against the current cycle only; pointing at an older one checks history."
    )


@pytest.mark.parametrize("path,spec", _partitions(), ids=lambda v: getattr(v, "name", ""))
def test_every_lane_declares_its_expected_ci_table_effect(path: Path, spec: dict) -> None:
    """RD-149: gate step 4's detector only discriminates if every lane declared.

    The step-4 fixed-seed A/B is the cross-lane interference detector, and it can
    tell interference from legitimate change only because each lane said in
    advance what it expected to move. `docs/parallel_protocol.md` stated that as a
    CYCLE-4 FACT ("all three declare none") and nothing asserted that a later
    cycle's lanes re-declare it — so the detector's discriminating power would
    quietly become an assumption again (RD-91 confirmed the durable half holds;
    this is the half that did not).

    Declared in the PARTITION, not only in the brief's prose: RD-147's lesson is
    that when a decision lives in two places, the prose copy is the one that
    drifts, and the machine-readable file is what the next planner reads.
    """
    if path.stem != _CURRENT_CYCLE:
        pytest.skip(f"{path.name} is not the current cycle ({_CURRENT_CYCLE})")

    for lane in spec["lanes"]:
        declared = lane.get("expected_ci_table_effect")
        assert declared, (
            f"{path.name}: lane {lane['id']} declares no "
            "`expected_ci_table_effect:`. Gate step 4 compares a fixed-seed "
            "ci_table.csv against the pre-lane baseline; without a declaration a "
            "moved row cannot be told from interference (RD-91, RD-149)."
        )


@pytest.mark.parametrize("path,spec", _partitions(), ids=lambda v: getattr(v, "name", ""))
def test_the_partition_declares_what_it_moves_on_the_gate(path: Path, spec: dict) -> None:
    """A cycle whose lanes cannot move the gate is a cycle that ends where it began.

    Planning step 1 says to start from the gate and give it a lane FIRST, and step
    1b says to state which conditions the cycle LIFTS and which it only UNBLOCKS.
    Both were prose, and prose is not a check: cycle 5 drew four lanes, and all
    four independently reported in their own inboxes that they lift nothing and
    unblock nothing — lane B "lifts NEITHER condition", lane P "neither LIFTS nor
    UNBLOCKS", lane S "LIFTS NOTHING AND UNBLOCKS NOTHING", lane M that condition
    (i) cannot lift for `evaluation/**` regardless of its execution. Four lanes,
    ~130 new findings, gate unmoved.

    So the partition must SAY, machine-readably, what it moves. A cycle that moves
    nothing is still allowed — some cycles are backlog discharge — but it has to
    be a declared, reasoned choice rather than something discovered afterwards.
    """
    if path.stem != _CURRENT_CYCLE:
        pytest.skip(f"{path.name} is not the current cycle ({_CURRENT_CYCLE})")

    gate = spec.get("gate")
    assert isinstance(gate, dict), (
        f"{path.name}: no `gate:` block. Declare `lifts:`, `unblocks:` and, when "
        "both are empty, `exception:` with a reason (planning steps 1 and 1b)."
    )
    for key in ("lifts", "unblocks"):
        assert key in gate, f"{path.name}: `gate:` declares no `{key}:` list."

    if not gate["lifts"] and not gate["unblocks"]:
        assert gate.get("exception"), (
            f"{path.name}: `gate:` declares it lifts nothing and unblocks nothing, "
            "with no `exception:` reason. That is permitted — a backlog-discharge "
            "cycle is legitimate — but it is a decision, and an undeclared one is "
            "how cycle 5 spent four parallel lanes without moving RD-33a."
        )


#: RD-33a condition (i)'s EXPLICIT path list, as operationalized by RD-76 and
#: scoped by severity in RD-128. Kept here so the check below counts the same
#: paths the gate does; the free text "the metric path" is what RD-76 replaced.
_GATE_PATH_LIST = (
    "src/amcd/scenes/**",
    "src/amcd/evaluation/**",
    "src/amcd/config.py",
    "configs/*.yaml",
)


def _ledger_rows() -> dict[str, list[str]]:
    """Every OPEN row's cells, keyed by id: [_, id, agent, sev, status, anchor, ...]."""
    rows: dict[str, list[str]] = {}
    for line in (_REPO_ROOT / "docs" / "review_ledger.md").read_text().splitlines():
        cells = [c.strip() for c in line.split("|")]
        if len(cells) > 5 and cells[4].startswith("OPEN"):
            rows[cells[1]] = cells
    return rows


#: A repo-relative path written out in full, e.g. `src/amcd/pipeline.py`.
_ANCHOR_PATH_RE = re.compile(
    r"\b((?:src|tests|configs|docs|scripts)/[\w./*-]+?\.(?:py|yaml|yml|md|json))\b"
)
#: A bare module name, e.g. "pipeline.py, runtime.py, cli.py stage dispatch" —
#: RD-20's anchor names its cross-lane file exactly this way, so a full-path-only
#: reader would miss the one row this check exists to catch.
_BARE_MODULE_RE = re.compile(r"\b([\w_]+\.py)\b")


def _unique_basenames() -> dict[str, str]:
    """Basename → repo path, for basenames that resolve to exactly ONE file.

    An ambiguous basename (`base.py` is four different modules) is left
    unresolved rather than guessed: a wrong resolution would flag a row for
    spanning a lane it does not touch, and a false accusation here costs a
    planner more than a miss.
    """
    counts: dict[str, int] = {}
    first: dict[str, str] = {}
    for p in (_REPO_ROOT / "src").rglob("*.py"):
        if p.name.startswith("._"):
            continue
        counts[p.name] = counts.get(p.name, 0) + 1
        first.setdefault(p.name, p.relative_to(_REPO_ROOT).as_posix())
    return {name: rel for name, rel in first.items() if counts[name] == 1}


def _anchor_paths(anchor: str) -> set[str]:
    paths = set(_ANCHOR_PATH_RE.findall(anchor))
    basenames = _unique_basenames()
    for bare in _BARE_MODULE_RE.findall(anchor):
        if bare in basenames and not any(p.endswith("/" + bare) for p in paths):
            paths.add(basenames[bare])
    return paths


@pytest.mark.parametrize("path,spec", _partitions(), ids=lambda v: getattr(v, "name", ""))
def test_no_rows_anchor_lands_in_another_lanes_files(path: Path, spec: dict) -> None:
    """A row's ANCHOR, not only its declared `fix:` paths, must fit its lane.

    `test_every_row_is_fixable_inside_its_own_lane` validates the paths the
    PARTITION declares. Nothing validated the paths the LEDGER declares, and the
    two can disagree — which is how a rule-4 spanning row passes the reachability
    check. RD-20 is the worked example: `cycle5.yaml` declares
    `fix: [src/amcd/pipeline.py]` (lane P owns it, so the check passed), while the
    ledger anchor names `runtime.py`, which lane B owns, and the real remedy
    changes a dispatch signature across nine call sites in three lanes. Lane P
    could not start it and returned it unattempted (RD-208).

    Four reviewers derived this hole independently — RD-111, RD-208, RD-225,
    RD-155 — and they are ONE defect that closes here, together.

    WHAT THIS CANNOT CHECK. The anchor is where a finding LIVES, not where its
    remedy lands. F-60's anchor is `scenes/generator.py`, genuinely lane S's,
    while its resolution needs `evaluation/room_acoustic.py` and
    `configs/base.yaml` — this check passes it, and RD-225 caught it by reading.
    Only a path a HUMAN OR REVIEWER wrote into the anchor is visible here.
    """
    if path.stem != _CURRENT_CYCLE:
        pytest.skip(f"{path.name} is not the current cycle ({_CURRENT_CYCLE})")

    rows = _ledger_rows()
    for lane in spec["lanes"]:
        for row in lane["rows"]:
            cells = rows.get(row["id"])
            if cells is None:  # the coverage test owns this failure
                continue
            declared_spans = set(row.get("spans", []))
            for target in sorted(_anchor_paths(cells[5])):
                if target in declared_spans:
                    continue
                other = next(
                    (l["id"] for l in spec["lanes"]
                     if l["id"] != lane["id"] and _owns(target, l["owns"])),
                    None,
                )
                assert other is None, (
                    f"{path.name}: row {row['id']} is assigned to lane "
                    f"{lane['id']}, but its LEDGER ANCHOR names '{target}', which "
                    f"lane {other} owns. That is a rule-4 spanning row: move it to "
                    "integrator_queue: with a reason, or — if the anchor merely "
                    "cites that file as context — declare it in the row's "
                    "`spans:` list so the acknowledgement is on the record."
                )


@pytest.mark.parametrize("path,spec", _partitions(), ids=lambda v: getattr(v, "name", ""))
def test_a_fix_path_that_does_not_exist_is_declared_as_new(path: Path, spec: dict) -> None:
    """RD-155: the partition had no notion of a fix that must CREATE a file.

    The ownership hook refuses a write to any path outside the lane's owned set,
    and a path that does not exist yet is refused the same way — so a row whose
    remedy is "promote this to its own module" strands mid-session. RD-121 and
    RR-83 both did, and so did lane S's RR-165.

    WHAT THIS CANNOT CHECK: those three rows declared `fix:` paths that DO exist
    (the file the code is being promoted OUT of), so the shortfall was invisible
    at declaration time and is invisible here too. This catches the forward case —
    a partition that names a not-yet-existing path — and makes `creates:` the
    place to say so. The reviewer reading the row's resolution is still the only
    thing that catches the other direction.
    """
    if path.stem != _CURRENT_CYCLE:
        pytest.skip(f"{path.name} is not the current cycle ({_CURRENT_CYCLE})")

    for lane in spec["lanes"]:
        for row in lane["rows"]:
            creates = set(row.get("creates", []))
            for kind in ("fix", "test"):
                for target in row.get(kind, []):
                    if "*" in target or (_REPO_ROOT / target).exists():
                        continue
                    assert target in creates, (
                        f"{path.name}: row {row['id']} declares {kind} path "
                        f"'{target}', which does not exist. If the remedy creates "
                        "it, list it in the row's `creates:`; the ownership hook "
                        "refuses a write to an unowned path whether or not the "
                        "file is there yet (RD-155)."
                    )


#: `| ID |` at the start of a table row, and bare id mentions in prose. Lane B's
#: eighteen readability findings existed ONLY as prose, so a table-row-only reader
#: would have declared them absent and lost them at the fold — which is the exact
#: event this check exists to prevent.
_ID_RE = re.compile(r"\b((?:RD|F|AC|RR)-\d+)\b")
_FOLDED_RE = re.compile(r"\b(?:RD|F|AC|RR)-\d+\b[^\n]{0,80}?folded into\b", re.IGNORECASE)
#: A `- <ID> — NOT A FINDING…` bullet under the ledger's fold-decisions heading.
_FOLD_DECISION_RE = re.compile(r"^- ((?:RD|F|AC|RR)-\d+)\s+—\s+\S", re.MULTILINE)


def _fold_decisions() -> set[str]:
    """Ids an inbox mentions that the fold deliberately did NOT turn into a row.

    A lane's id block is a RESERVATION, not a promise: its bounds get quoted in
    prose (`RD-175..199`), and a lane can leave a dangling citation to a number it
    never raised (`RD-182`). Neither is a finding, and neither can be forced into a
    row without inventing one.

    Every entry needs a stated reason, so that "it was never a finding" cannot
    quietly become the place a genuinely dropped finding goes — which is the whole
    failure mode this check exists for.
    """
    ledger = (_REPO_ROOT / "docs" / "review_ledger.md").read_text()
    heading = "### Fold decisions — ids raised in an inbox that deliberately became NO row"
    if heading not in ledger:
        return set()
    tail = ledger[ledger.index(heading) + len(heading):]
    end = tail.find("\n## ")
    return set(_FOLD_DECISION_RE.findall(tail if end == -1 else tail[:end]))


@pytest.mark.parametrize("path,spec", _partitions(), ids=lambda v: getattr(v, "name", ""))
def test_every_inbox_finding_reaches_the_ledger(path: Path, spec: dict) -> None:
    """The missing direction: inbox → ledger (RD-142, F-160 — ONE defect).

    `test_the_partition_covers_exactly_the_ledgers_open_rows` asserts ledger ↔
    partition, and it is sound, but it cannot see a finding that never became a
    row at all. That is how cycle 4's fold lost five rows including a blocker.

    `docs/parallel_protocol.md` CLAIMED this file already asserted it. Lane P
    measured otherwise — nine tests, none of which opened an inbox — and filed
    F-160 as the same class as F-66/F-77: documentation claiming more than the
    test checks, on the guard that exists to stop silent row loss at the fold.

    SCOPED TO EACH LANE'S OWN id_block. An inbox also cites pre-existing ids it
    merely verified or discussed, and those get DELETED from the ledger when they
    are confirmed fixed — requiring them to stay OPEN would make a correct
    deletion fail this test. A NEW finding, though, comes from the lane's declared
    block by rule 6, so the block is exactly the set the fold must not drop.
    """
    if path.stem != _CURRENT_CYCLE:
        pytest.skip(f"{path.name} is not the current cycle ({_CURRENT_CYCLE})")

    open_ids = _open_ledger_ids()
    for lane in spec["lanes"]:
        inbox = _REPO_ROOT / lane["inbox"]
        if not inbox.exists():
            continue  # lane has not reported yet; nothing to fold
        text = inbox.read_text()
        folded = set(_ID_RE.findall(" ".join(_FOLDED_RE.findall(text))))

        block: set[str] = set()
        for prefix, rng in (lane.get("id_block") or {}).items():
            lo, _, hi = str(rng).partition("..")
            block |= {f"{prefix}-{n}" for n in range(int(lo), int(hi) + 1)}

        raised = {i for i in _ID_RE.findall(text) if i in block}
        missing = sorted(raised - open_ids - folded - _fold_decisions())
        assert not missing, (
            f"{path.name}: lane {lane['id']} raised {missing} in {lane['inbox']} "
            "and they are not OPEN rows in docs/review_ledger.md. The fold must "
            "give every finding a row with a file anchor, record it as 'folded "
            "into <id>' in the inbox, or list it with a reason under the ledger's "
            "'Fold decisions' heading. A finding in none of the three is invisible "
            "to every later cycle (RD-142, F-160)."
        )
