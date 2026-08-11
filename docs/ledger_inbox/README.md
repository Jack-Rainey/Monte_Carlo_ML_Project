# Ledger inbox

One file per lane, written only by that lane, read only by the integrator.

`docs/review_ledger.md` has exactly one writer (`docs/parallel_protocol.md`,
rule 3) — it is working memory for the review loop, and concurrent edits to it
would be both a merge conflict and a race on the project's own record of what is
unresolved. Lanes therefore never touch it. A lane writes here instead, to
`<lane-id>.md`, and the integrator folds these files into the ledger at the
integration gate: closures become deleted rows, new findings become new rows.

Distinct filenames are the whole trick. Three lanes each creating a different
file merge with no conflict resolution at all.

## What a lane writes

Both directions, in whatever order they happened:

- **Closures** — a row you fixed, with the command you ran and its output. The
  integrator deletes the row only after the post-merge reviewer pass confirms it,
  so include enough for that reviewer to check the claim, not just assert it.
- **New findings** — anything a reviewer raised on your branch, or that you found
  and could not fix because it lives in another lane's files (rule 4). Name the
  concrete file path; that is what makes it assignable to a lane next cycle.
- **Anything you deliberately did not do**, and why. A row left untouched with no
  note is indistinguishable from a row nobody read.

## When a lane file may be emptied — READ THIS BEFORE DELETING ANYTHING

**This file used to say "files here are emptied at the start of each cycle". That
instruction is now a DATA-LOSS BUG (RR-80) and has been removed.**

It was safe only while the fold COPIED each finding's substance into the ledger. From
cycle 4 it does not: ~130 ledger rows are compact and say *"substance, measurements and
probes in `docs/ledger_inbox/<lane>.md`"*, because a lane's own write-up is better
evidence than a paraphrase of it. Emptying these files on that schedule would delete the
only readable record of half the OPEN set.

The rule now:

- **A lane file may be emptied only once NO OPEN ledger row cites it.** Check before you
  touch it, not after.
- **Cycle-4 content is permanent.** When cycle 5 opens, move it to
  `docs/ledger_inbox/archive/cycle4-<lane>.md` rather than truncating it, so the next
  cycle's lanes get clean files without destroying the citations.
- **The ids differ between here and the ledger.** Four lanes collided on row ids in cycle
  4 and the ledger renumbered; these files were deliberately left un-renumbered. The
  ledger's header carries the durable remap table, and every compact row names the id AS
  WRITTEN HERE. So `grep` for an id in this directory and in the ledger can legitimately
  land on different findings — always go through the remap table.

Git history remains the audit trail underneath all of this, but a row that points at a
file expects that file to still say something.
