# Archived lane inboxes

Per-cycle lane records, moved here when the next cycle opens rather than
truncated (RR-80). They are **not** dead history: ~130 compact rows in
`docs/review_ledger.md` cite these files as the primary record of their finding,
and the ledger's citations point straight at the paths in this directory.

**The ids in these files are the ids the LANE wrote, which are not always the ids
the ledger uses.** Four lanes collided on row ids in cycle 4, so the ledger
renumbered while these files were deliberately left alone — a lane's own words are
better evidence than a paraphrase. `docs/review_ledger.md`'s header carries the
durable remap table, and every compact row names the id as written here.

A file leaves this directory only when no OPEN ledger row cites it.
