"""The imported `amcd` must come from the checkout these tests live in.

The evidence-before-claims rule assumes the code under test is the code just
edited. Under parallel lanes (`docs/parallel_protocol.md`) that assumption is
NOT free: the project is installed with `pip install -e .`, so
`site-packages/__editable__.amcd-0.1.0.pth` contains one absolute path — the
main checkout's `src/`. A lane worktree that runs `pytest` or `amcd` without
`PYTHONPATH=<lane>/src` therefore imports the MAIN tree's modules while editing
its own, and every measurement it reports describes code it did not change.

That failure is silent: the suite passes, the pipeline runs, and the numbers are
about the wrong tree. So this is a test rather than a documented preflight —
it fires on the lane's first `pytest`, before any evidence is gathered.

In the main checkout it passes trivially, which is the point: the guard costs
nothing where it is not needed and refuses to be skipped where it is.
"""
from pathlib import Path

import amcd

#: The checkout this test file belongs to — tests/ sits directly under it.
_TESTS_CHECKOUT = Path(__file__).resolve().parent.parent

#: The checkout the imported package was loaded from: src/amcd/__init__.py, so
#: two levels up from the module file is that checkout's root.
_IMPORT_CHECKOUT = Path(amcd.__file__).resolve().parent.parent.parent


def test_amcd_is_imported_from_the_checkout_under_test() -> None:
    """A lane must not measure another checkout's source.

    The remedy is in the message rather than the docs, because the reader of a
    failure here is a session that has just been handed the wrong tree and has
    no reason yet to suspect the editable install.
    """
    assert _IMPORT_CHECKOUT == _TESTS_CHECKOUT, (
        "amcd was imported from a DIFFERENT checkout than the one under test.\n"
        f"  tests live in:      {_TESTS_CHECKOUT}\n"
        f"  amcd imported from: {_IMPORT_CHECKOUT}\n"
        "The editable install (`__editable__.amcd-0.1.0.pth`) pins one absolute "
        "path, so a worktree must shadow it. Re-run with:\n"
        f"  PYTHONPATH={_TESTS_CHECKOUT / 'src'} <python-env>/bin/pytest\n"
        "Any evidence gathered without this describes the other checkout — see "
        "docs/parallel_protocol.md."
    )
