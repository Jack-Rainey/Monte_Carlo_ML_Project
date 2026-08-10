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

Files here are emptied at the start of each cycle, after the integrator has
folded them in. Git history keeps them, which is the same audit trail the ledger
itself relies on.
