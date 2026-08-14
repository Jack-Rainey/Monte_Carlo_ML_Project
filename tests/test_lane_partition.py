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
import subprocess
from functools import lru_cache
from pathlib import Path

import pytest
import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent

#: `._`-prefixed entries are AppleDouble resource forks the host filesystem
#: creates beside every file on this exFAT volume; they are not partitions. The
#: package filters them the same way wherever it globs artifacts (evaluator.py,
#: data/dataset.py); the same sidecar has reached a cache key before.
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
_CURRENT_CYCLE = "cycle6"


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
    a row assigned to the lane owning the code can have its test class in a file
    another lane owns, and the ownership hook then refuses the edit halfway
    through the session. Declaring `fix:` and `test:` per row turns that into a
    failure at
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
    """Every row is assigned exactly once, across every list.

    The lists together are the partition's coverage claim. A row in two of them
    means two different plans for it; a row in none is the silent omission
    exists to prevent.

    `pre_lane:` and `post_merge:` are the two integrator queues: work that must
    land BEFORE the partition is drawn, and work applied after the merge. There is
    no `awaiting_re_review:` bucket — a fixed row is deleted, and confirmation
    comes from a reviewer re-deriving the defect from code, never from a row.
    """
    seen: dict[str, str] = {}
    buckets = [
        *((lane["id"], [row["id"] for row in lane["rows"]]) for lane in spec["lanes"]),
        ("serial_queue", [row["id"] for row in spec.get("serial_queue", [])]),
        ("pre_lane", [row["id"] for row in spec.get("pre_lane", [])]),
        ("post_merge", [row["id"] for row in spec.get("post_merge", [])]),
        ("integrator_queue", [row["id"] for row in spec.get("integrator_queue", [])]),
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
    miscounts by one in the direction that makes a broken partition look sound.
    """
    return set(_open_ledger_id_list())


def _open_ledger_id_list() -> list[str]:
    """The same ids as a LIST, so duplicates survive to be asserted on.

    `_open_ledger_ids` returns a set, which silently absorbs a duplicated row id —
    and a duplicated id is exactly the failure cycle 4 hit four ways when four
    parallel lanes each numbered from the ledger's maximum.
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
    That is not hypothetical: lanes allocating independently from the ledger's
    max produce one id naming several different findings.
    """
    ids = _open_ledger_id_list()
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    assert not dupes, (
        f"duplicate OPEN row ids in docs/review_ledger.md: {dupes}. Two findings "
        "sharing an id means one of them cannot be cited, assigned or deleted "
        "independently — allocate from the lane's declared id_block (rule 6)."
    )


def _ids_lanes_raised_from_their_own_blocks(spec: dict) -> set[str]:
    """Ids a lane actually raised: inside its OWN id_block AND in its OWN inbox.

    Deriving this from the partition's buckets would be circular — the buckets
    are what it validates — and would widen silently the moment the fold's output
    is split in two.
    """
    raised: set[str] = set()
    for lane in spec["lanes"]:
        inbox = _REPO_ROOT / lane["inbox"]
        if not inbox.exists():
            continue
        block: set[str] = set()
        for prefix, rng in (lane.get("id_block") or {}).items():
            lo, _, hi = str(rng).partition("..")
            block |= {f"{prefix}-{n}" for n in range(int(lo), int(hi) + 1)}
        raised |= {i for i in _ID_RE.findall(inbox.read_text()) if i in block}
    return raised


@lru_cache(maxsize=None)
def _ids_open_before_this_cycle(partition_name: str) -> frozenset[str]:
    """OPEN row ids in the ledger as of the commit that DREW this cycle.

    The non-circular discriminator. What the protocol forbids is a lane
    allocating an id that already names a LIVE finding, so the question is "was
    this id open when the cycle was drawn" — and only history answers it.

    Every lane-side signal is defeated by the offence itself: a lane that
    allocates a colliding id necessarily writes that id into its own inbox, so an
    inbox-derived exemption fires precisely in the case it must catch. That was
    measured on the shipped guard — appending one line naming a live id to a
    lane's inbox turned the collision green.

    The ADDING commit, not the last-modified one: the partition file is edited
    throughout its cycle, including by the integration that folds the lanes' own
    new ids into the ledger.

    Empty set if git cannot answer, which makes the guard STRICTER (a consumed
    block then reads as a collision) rather than looser.
    """
    def _git(*args):
        return subprocess.run(
            ["git", *args], cwd=_REPO_ROOT, capture_output=True, text=True, timeout=60
        ).stdout.strip()

    cycle = partition_name.removesuffix(".yaml")
    try:
        # WHERE THE LANES DIVERGED is when the cycle was drawn — the ledger at
        # that commit is what a lane could have collided with. The partition
        # FILE's creation is not the same instant: cycle 5's was written during
        # cycle 4's integration, before cycle 4's own fold had added the ids
        # cycle 5's lanes would go on to consume.
        base = ""
        for ref in _git("branch", "--list", f"lane/*-{cycle}", "--format=%(refname)").splitlines():
            base = _git("merge-base", "HEAD", ref.strip())
            if base:
                break
        if not base:   # branches retired; fall back to the partition's creation
            drawn = _git("log", "--diff-filter=A", "--format=%H", "-1", "--",
                         f"docs/lanes/{partition_name}")
            base = f"{drawn}^" if drawn else ""
        if not base:
            return frozenset()
        blob = subprocess.run(
            ["git", "show", f"{base}:docs/review_ledger.md"],
            cwd=_REPO_ROOT, capture_output=True, text=True, timeout=60,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return frozenset()
    out = set()
    for line in blob.splitlines():
        cells = [c.strip() for c in line.split("|")]
        if len(cells) > 5 and cells[4].startswith("OPEN"):
            out.add(cells[1])
    return frozenset(out)


@pytest.mark.parametrize("path,spec", _partitions(), ids=lambda v: getattr(v, "name", ""))
def test_lane_id_blocks_are_disjoint_and_unused(path: Path, spec: dict) -> None:
    """Rule 6: each lane allocates new ids only from its own declared block, and no
    block overlaps another lane's or an id the ledger already uses.

Without it, every lane numbers from the ledger's maximum at the moment it
    starts, so one id names several different findings and a lane's new ids land
    on rows that are live and unrelated. Untangling that takes a per-lane,
    per-class remap of the source tree.

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
    # rule 6 working, not failing.
    #
    # The exemption is ATTRIBUTION, not mention. Exempting any id the
    # lane's inbox happens to NAME is blind to the exact failure this guard
    # exists for: a lane that allocates a colliding id necessarily writes that id
    # into its own inbox, so a mention-based exemption fires precisely in the case
    # the protocol forbids: a mention-based rule passes as soon as the lane's
    # own inbox names the colliding id.
    #
    # An id is a COLLISION only if it was already a live finding when this cycle
    # was DRAWN. Consuming your own block and folding the results makes
    # those ids OPEN by design — that is rule 6 working — so "OPEN now" is the
    # wrong test. And every lane-side signal is defeated by the offence itself,
    # which is why this reads history instead.
    was_live = _ids_open_before_this_cycle(path.name)

    seen: dict[tuple[str, int], str] = {}
    for lane in spec["lanes"]:
        for prefix, rng in (lane.get("id_block") or {}).items():
            lo, _, hi = str(rng).partition("..")
            assert lo.isdigit() and hi.isdigit(), (
                f"{path.name}: lane {lane['id']} id_block {prefix}: {rng!r} is not "
                "a 'LO..HI' range."
            )
            for n in range(int(lo), int(hi) + 1):
                assert f"{prefix}-{n}" not in was_live, (
                    f"{path.name}: lane {lane['id']}'s block {prefix}-{rng} contains "
                    f"{prefix}-{n}, which was ALREADY a live OPEN row when this "
                    "cycle was drawn. A lane allocating it would overwrite a live "
                    "finding."
                )
                key = (prefix, n)
                assert key not in seen, (
                    f"{path.name}: {prefix}-{n} is in both lane {seen[key]}'s and "
                    f"lane {lane['id']}'s id_block. Blocks must be disjoint — this "
                    "is the four-way collision of cycle 4."
                )
                seen[key] = lane["id"]


@pytest.mark.parametrize("path,spec", _partitions(), ids=lambda v: getattr(v, "name", ""))
def test_the_partition_covers_exactly_the_ledgers_open_rows(path: Path, spec: dict) -> None:
    """Every OPEN row has a plan, and every planned row is really OPEN.

    Pairwise uniqueness cannot catch a row that is in NO list — which is the
    omission this exists for, and precisely what a coverage figure equal to its
    own scope hides.

    Only the current cycle's partition is checked against the live ledger; an
    older cycle's file describes a ledger state that no longer exists.

    `unassigned:` is the FIFTH list. A cycle's fold creates rows
    the partition could not have planned, because they did not exist when it was
    drawn — cycle 5's fold created 142. Without a bucket for them this check goes
    red between every fold and the next partition, and the pressure is then to
    weaken the check rather than to record the rows. `unassigned:` keeps the
    coverage identity total while saying plainly that these await a partition.

    A SERIAL cycle (`lanes: []`) is exempt, and the exemption is the point rather
    than a loophole: the defect this guards against is a row falling into the gap
    between two lanes' scopes, and a cycle with no lanes has no gaps — every OPEN
    row is the integrator's by construction. Enumerating them anyway makes the
    partition a second copy of the ledger's id list, which then has to be resynced
    after every single deletion, and a list maintained only to satisfy a check is
    the bookkeeping this project keeps having to delete.
    """
    if path.stem != _CURRENT_CYCLE:
        pytest.skip(f"{path.name} is not the current cycle ({_CURRENT_CYCLE})")
    if not spec["lanes"]:
        pytest.skip(
            f"{path.name} is a serial cycle (lanes: []) — every OPEN row is the "
            "integrator's, so there is no inter-lane gap for a row to fall into."
        )

    planned: set[str] = set()
    for lane in spec["lanes"]:
        planned |= {row["id"] for row in lane["rows"]}
    planned |= {row["id"] for row in spec.get("serial_queue", [])}
    planned |= {row["id"] for row in spec.get("pre_lane", [])}
    planned |= {row["id"] for row in spec.get("post_merge", [])}
    planned |= {row["id"] for row in spec.get("integrator_queue", [])}
    planned |= set(spec.get("raised_against_this_partition", []))
    planned |= set(spec.get("unassigned", []))

    open_ids = _open_ledger_ids()
    unplanned = open_ids - planned
    assert not unplanned, (
        f"{path.name}: these rows are OPEN in the ledger but appear in NO list — "
        f"neither a lane, an integrator queue (pre_lane/post_merge), nor raised "
        f"against the partition: {sorted(unplanned)}. A row with no plan is the "
        "silent omission exists to prevent."
    )
    stale = planned - open_ids
    assert not stale, (
        f"{path.name}: these rows are planned but are NOT OPEN in the ledger "
        f"(closed, deleted, or still DEFERRED): {sorted(stale)}. Either the row "
        "was resolved and the partition is stale, or it needs re-statusing "
        "before a lane is told to work it."
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
    """`_CURRENT_CYCLE` is a module literal that nothing asserted.

    Every check below that matters — the coverage identity, the id blocks, the
    anchor check — skips when `path.stem != _CURRENT_CYCLE`. If the literal ever
    names a stem with no file, `pytest.skip` fires for EVERY parametrization and
    the whole coverage guarantee is suspended with a green suite. That is "asserted by nothing" shape, one level up: the guard reports success
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
    """Gate step 4's detector only discriminates if every lane declared.

    The step-4 fixed-seed A/B is the cross-lane interference detector, and it can
    tell interference from legitimate change only because each lane said in
    advance what it expected to move. `docs/parallel_protocol.md` stated that as a
    fact about one cycle rather than a standing requirement, so nothing made a
    later cycle's lanes re-declare it and the detector's discriminating power
    would quietly become an assumption again.

    Declared in the PARTITION, not only in the brief's prose: lesson is
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
            "moved row cannot be told from interference."
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

    It must also name a `deliverable:`: ONE concrete measurable thing, plus the
    evidence that will show it landed. At cycle exit the integrator shows that
    evidence or reports the cycle FAILED. A cycle is judged on its deliverable,
    never on a finding count — two consecutive cycles reported large counts and
    moved the gate by zero.
    """
    if path.stem != _CURRENT_CYCLE:
        pytest.skip(f"{path.name} is not the current cycle ({_CURRENT_CYCLE})")

    gate = spec.get("gate")
    assert isinstance(gate, dict), (
        f"{path.name}: no `gate:` block. Declare `lifts:`, `unblocks:`, "
        "`deliverable:` and, when the first two are empty, `exception:` with a "
        "reason (planning steps 1 and 1b)."
    )
    for key in ("lifts", "unblocks"):
        assert key in gate, f"{path.name}: `gate:` declares no `{key}:` list."

    deliverable = gate.get("deliverable")
    assert isinstance(deliverable, dict) and deliverable.get("what") and deliverable.get("evidence"), (
        f"{path.name}: `gate:` declares no usable `deliverable:`. It needs "
        "`what:` (one concrete measurable outcome) and `evidence:` (what will show "
        "it landed). Without both, the cycle has nothing to be judged against and "
        "gets reported by finding count instead — which is how two cycles in a row "
        "reported success while moving the gate by zero."
    )

    if not gate["lifts"] and not gate["unblocks"]:
        assert gate.get("exception"), (
            f"{path.name}: `gate:` declares it lifts nothing and unblocks nothing, "
            "with no `exception:` reason. That is permitted — a backlog-discharge "
            "cycle is legitimate — but it is a decision, and an undeclared one is "
            "how a cycle spends four parallel lanes without moving its gate."
        )


#: The dataset-render gate's EXPLICIT path list (design_spec §11.1 condition i),
#: kept here so the check below counts the same paths the gate does. The explicit
#: list replaced free text ("the metric path"), which admitted any reading.
#:
#: The GATE itself lifts at zero OPEN rows on these paths, not zero blocker/major
#: (user decision 2026-08-12, superseding severity scoping): severity is a
#: skim aid, and a gate that ignores minors ships with known-open work on its own
#: path list. The blocker/major filter below is a different question — which rows a
#: partition should be judged on SCHEDULING — and it stays.
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
#: anchor names its cross-lane file exactly this way, so a full-path-only
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
    check. The worked example: a partition declares `fix: [src/amcd/pipeline.py]`
    (the lane owns it, so the check passes) while the ledger anchor names
    `runtime.py`, which another lane owns — and the real remedy changes a dispatch
    signature across nine call sites in three lanes, so the lane cannot start it.

    WHAT THIS CANNOT CHECK. The anchor is where a finding LIVES, not where its
    remedy lands. A row anchored on `scenes/generator.py` whose resolution needs
    `evaluation/room_acoustic.py` passes here. Only a path a human or reviewer
    wrote into the anchor is visible.
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
    """The partition had no notion of a fix that must CREATE a file.

    The ownership hook refuses a write to any path outside the lane's owned set,
    and a path that does not exist yet is refused the same way — so a row whose
    remedy is "promote this to its own module" strands mid-session.

    WHAT THIS CANNOT CHECK: such rows declare `fix:` paths that DO exist
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
                        "file is there yet."
                    )


#: `| ID |` at the start of a table row, and bare id mentions in prose. Lane B's
#: eighteen readability findings existed ONLY as prose, so a table-row-only reader
#: would have declared them absent and lost them at the fold — which is the exact
#: event this check exists to prevent.
#: A resolution that already claims a fix. The ledger's own words are the
#: authority — a hand-maintained list of such rows is exactly what drifted.
_FIX_CLAIMED_RE = re.compile(
    r"fix applied|FIXED, awaiting|partial fix applied|CORRECTED, awaiting"
    r"|RE-DERIVED AT THIS|FIXED IN LANE|BOTH HALVES APPLIED"
    r"|CONFIRMED AND CLOSED BY MEASUREMENT|APPLIED at this integration",
    re.IGNORECASE,
)

_ID_RE = re.compile(r"\b((?:RD|F|AC|RR)-\d+)\b")
_FOLDED_RE = re.compile(r"\b(?:RD|F|AC|RR)-\d+\b[^\n]{0,80}?folded into\b", re.IGNORECASE)
#: A `- <ID> — NOT A FINDING…` bullet under the ledger's fold-decisions heading.
_FOLD_DECISION_RE = re.compile(r"^- ((?:RD|F|AC|RR)-\d+)\s+—\s+\S", re.MULTILINE)


@lru_cache(maxsize=1)
def _ids_ever_a_row() -> frozenset[str]:
    """Every id that has EVER been added as a row to the ledger, from git.

    The fold guard below asks "did this finding become a row". A row that was
    folded, re-review-confirmed and then correctly DELETED at gate step 7 answers
    yes — but it is no longer OPEN, and without this the guard would demand its
    resurrection, i.e. punish the loop for completing. That is not hypothetical:
    it fired the moment cycle 5's first 44 confirmed rows were deleted.

    Git history IS this project's audit trail for deleted rows — see the ledger
    header's own `git log -S` instructions — so it is the right oracle.
    One subprocess, cached for the session.

    Degrades to the empty set if git is unavailable — a shallow copy or an
    exported tree — matching `provenance.git_sha`'s contract that the sha is
    allowed to be missing. The guard then falls back to OPEN-only, which is
    stricter, so a missing git cannot make it pass something it should catch.
    """
    try:
        out = subprocess.run(
            ["git", "log", "-p", "--unified=0", "--", "docs/review_ledger.md"],
            cwd=_REPO_ROOT, capture_output=True, text=True, timeout=120,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return frozenset()
    return frozenset(re.findall(r"^\+\|\s*((?:RD|F|AC|RR)-\d+[a-z]?)\s*\|", out, re.M))


def _fold_decisions() -> set[str]:
    """Ids an inbox mentions that the fold deliberately did NOT turn into a row.

    A lane's id block is a RESERVATION, not a promise: its bounds get quoted in
    prose, and a lane can leave a dangling citation to a number it never raised.
    Neither is a finding, and neither can be forced into a
    row without inventing one.

    Every entry needs a stated reason, so that "it was never a finding" cannot
    quietly become the place a genuinely dropped finding goes — which is the whole
    failure mode this check exists for.
    """
    ledger = (_REPO_ROOT / "docs" / "review_ledger.md").read_text()
    heading = "### Fold decisions — ids raised in an inbox that deliberately became NO row"
    # EVERY occurrence, not the first. Each cycle's fold sits under its own
    # `## CYCLE-N FOLD` heading, so a later integrator appending a second
    # decisions section is the natural thing to do — and reading only the first
    # would silently drop it, making this guard demand rows for findings that
    # were correctly recorded as non-findings. Verified: with a first-occurrence
    # reader, an id under a second heading is invisible.
    out: set[str] = set()
    for m in re.finditer(re.escape(heading), ledger):
        tail = ledger[m.end():]
        end = tail.find("\n## ")
        out |= set(_FOLD_DECISION_RE.findall(tail if end == -1 else tail[:end]))
    return out


@pytest.mark.parametrize("path,spec", _partitions(), ids=lambda v: getattr(v, "name", ""))
def test_every_inbox_finding_reaches_the_ledger(path: Path, spec: dict) -> None:
    """The missing direction: inbox → ledger.

    `test_the_partition_covers_exactly_the_ledgers_open_rows` asserts ledger ↔
    partition, and it is sound, but it cannot see a finding that never became a
    row at all. That is how cycle 4's fold lost five rows including a blocker.

    `docs/parallel_protocol.md` once claimed this file asserted it while none of
    its tests opened an inbox — documentation claiming more than the test checks,
    on the guard that exists to stop silent row loss at the fold.

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
        missing = sorted(raised - open_ids - folded - _fold_decisions() - _ids_ever_a_row())
        assert not missing, (
            f"{path.name}: lane {lane['id']} raised {missing} in {lane['inbox']} "
            "and they are not OPEN rows in docs/review_ledger.md. The fold must "
            "give every finding a row with a file anchor, record it as 'folded "
            "into <id>' in the inbox, or list it with a reason under the ledger's "
            "'Fold decisions' heading. A finding in none of the three is invisible "
            "to every later cycle."
        )


@pytest.mark.parametrize("path,spec", _partitions(), ids=lambda v: getattr(v, "name", ""))
def test_every_declared_inbox_exists(path: Path, spec: dict) -> None:
    """A missing inbox must not read as "nothing to check".

    Both fold guards `continue` past an inbox that is not on disk, and nothing
    asserted one should be — `test_every_declared_brief_exists` covers briefs
    only. Measured consequences, both real:

    * with `exit_gate: required` and every inbox path missing,
      `test_every_gated_lane_satisfies_the_exit_gate` PASSES — so a lane that
      reports NOTHING clears the exit gate;
    * archiving an inbox at end of cycle (the protocol's own housekeeping) makes
      `test_every_inbox_finding_reaches_the_ledger` pass VACUOUSLY.

    That is shape one level down, inside the guard the fold rests on. A
    lane that genuinely has not reported yet says so with `reported: false`, which
    is a declaration rather than an absence.
    """
    if path.stem != _CURRENT_CYCLE:
        pytest.skip(f"{path.name} is not the current cycle ({_CURRENT_CYCLE})")

    for lane in spec["lanes"]:
        if lane.get("reported") is False:
            continue
        inbox = _REPO_ROOT / lane["inbox"]
        assert inbox.exists(), (
            f"{path.name}: lane {lane['id']} declares inbox '{lane['inbox']}', "
            "which does not exist. If the lane has not reported yet, declare "
            "`reported: false` — an absent file otherwise makes the fold and "
            "exit-gate checks pass by having nothing to look at."
        )


@pytest.mark.parametrize("path,spec", _partitions(), ids=lambda v: getattr(v, "name", ""))
def test_unassigned_holds_only_rows_this_cycles_fold_created(path: Path, spec: dict) -> None:
    """`unassigned:` must not become a place to park anything.

    The bucket keeps the coverage identity total instead of forcing the check to
    be weakened, which is the right trade. But nothing constrained its
    membership, so the identity would decay from "every OPEN row has a PLAN" to
    "every OPEN row is NAMED in this file" — the coverage-equal-to-its-own-scope
    figure exists to prevent.

    The constraint that makes it honest: an entry must be a finding one of THIS
    cycle's lanes actually raised. A pre-existing row cannot be parked here to
    make it look covered, and — since this same set is what exempts a consumed
    `id_block` — neither can a row be exempted from the collision check
    by listing it.
    """
    if path.stem != _CURRENT_CYCLE:
        pytest.skip(f"{path.name} is not the current cycle ({_CURRENT_CYCLE})")

    unassigned = set(spec.get("unassigned", []))
    if not unassigned:
        return
    # Before a lane reports, `unassigned:` is the INBOUND backlog a cycle is drawn
    # against — carried-forward rows, legitimately. hazard is a FOLD that
    # parks pre-existing rows there to make the coverage identity look total, and
    # a fold only happens after lanes report. So the provenance constraint applies
    # from the moment any inbox exists, and not before.
    if not any((_REPO_ROOT / lane["inbox"]).exists() for lane in spec["lanes"]):
        return

    # `unassigned:` is the fold's unworked output, and it is the bucket with a
    # provenance claim to keep honest.
    strays = sorted(unassigned - _ids_lanes_raised_from_their_own_blocks(spec))
    assert not strays, (
        f"{path.name}: `unassigned:` lists {strays}, which no lane raised from its "
        "own id_block in this cycle. That bucket is for rows THIS cycle's fold "
        "created; anything else parked there makes the coverage identity report "
        "its own scope, and silently exempts the id from the block-collision "
        "check."
    )


@pytest.mark.parametrize("path,spec", _partitions(), ids=lambda v: getattr(v, "name", ""))
def test_some_lane_carries_a_row_that_can_move_the_gate(path: Path, spec: dict) -> None:
    """Accepted resolution, in full this time.

At least one lane's brief must declare rows anchored on a live gate
    condition's path list. Asserting only that a `gate:` block EXISTS moves the
    answer from report time to planning time, which is a real improvement, but it
    is not the same check — and it leaves `_GATE_PATH_LIST` referenced nowhere.

    The count of on-path blocker/major rows a partition SCHEDULES is what it
    should be judged on, before the cycle runs rather than after.
    """
    if path.stem != _CURRENT_CYCLE:
        pytest.skip(f"{path.name} is not the current cycle ({_CURRENT_CYCLE})")

    rows = _ledger_rows()
    per_lane: dict[str, int] = {}
    for lane in spec["lanes"]:
        n = 0
        for row in lane["rows"]:
            cells = rows.get(row["id"])
            if cells is None or cells[3] not in ("blocker", "major"):
                continue
            if any(_owns(p, list(_GATE_PATH_LIST)) for p in _anchor_paths(cells[5])):
                n += 1
        per_lane[lane["id"]] = n

    if sum(per_lane.values()):
        return
    assert (spec.get("gate") or {}).get("exception"), (
        f"{path.name}: no lane carries a blocker/major row anchored on the gate's "
        f"path list — on-path rows scheduled per lane: {per_lane}. A cycle whose "
        "lanes cannot move the gate ends where it started with a bigger ledger "
        ". Give the gate a lane, or declare `gate.exception` saying why "
        "this cycle deliberately does not."
    )


@pytest.mark.parametrize("path,spec", _partitions(), ids=lambda v: getattr(v, "name", ""))
def test_no_lane_is_assigned_a_row_whose_fix_is_already_claimed(path: Path, spec: dict) -> None:
    """Planning step 3, asserted for the first time.

    "Exclude fix-applied rows from every lane; assigning one invites a second fix
    stacked on a first that nobody checked." That has been the rule since cycle 4
    and nothing has ever enforced it — cycle 5 then assigned 35 such rows to lane
    P and 9 to lane S, roughly 40 % of two lanes' apparent workload, and lane P's
    entire deliverable turned out to be verification rather than work.

    Such a row is not work, and it does not go to a lane. Either the fix is real,
    in which case the row should already have been DELETED, or it is not, in which
    case a reviewer re-deriving the defect from CODE raises it fresh. There is no
    third state and no holding bucket — that bucket is what let 22 rows accumulate
    across three cycles.

    Derived from the resolution text rather than a hand-maintained list, because a
    hand-maintained one is what drifted: the ledger says "fix applied … awaiting
    re-review" in the row itself, so that is the authority.
    """
    if path.stem != _CURRENT_CYCLE:
        pytest.skip(f"{path.name} is not the current cycle ({_CURRENT_CYCLE})")

    claimed = {
        cells[1]
        for line in (_REPO_ROOT / "docs" / "review_ledger.md").read_text().splitlines()
        if len(cells := [c.strip() for c in line.split("|")]) > 7
        and cells[4].startswith("OPEN")
        and _FIX_CLAIMED_RE.search(cells[7])
    }
    for lane in spec["lanes"]:
        offending = sorted({row["id"] for row in lane["rows"]} & claimed)
        assert not offending, (
            f"{path.name}: lane {lane['id']} is assigned {offending}, whose "
            "resolutions already claim a fix nobody has re-derived. A second fix "
            "stacked on an unchecked first is what planning step 3 forbids. Either "
            "DELETE the row (the fix is real) or strip the claim from its "
            "resolution so it reads as the open work it is."
        )
