"""Which torch device this host offers.

A module of its own, and the reason is a cache key. `select_device` is runtime
POLICY, not provenance, but it has to be callable from `Config.stamp` so a run
records the device it used. Living in `provenance.py` put it inside
`_CORE_SOURCES`, which every stage's declared scope unions in — so editing the
MPS -> CUDA -> CPU fallback invalidated every fingerprinted stage.

That was tolerable only while `gen-scenes` and `render` carried no `code_version`.
Once they do, an edit here would force a re-render, which under x86 emulation is
the multi-hour artifact — and this fallback is precisely the code the
cross-platform requirement makes someone touch. Hence a top-level module outside
`_CORE_SOURCES`: a device-selection edit now invalidates nothing, which is correct,
because it changes no dataset.

It cannot live in `training/` for the mirror-image reason: a core module importing
`amcd.training` would drag the whole training package into every stage's import
closure and make every declared scope wrong.
"""
from __future__ import annotations


def select_device() -> "torch.device":  # noqa: F821 — torch imported lazily below
    """The torch device this host offers, preferring MPS, then CUDA, then CPU.

    Falls back rather than assuming: the project is required to run on this
    Apple-Silicon machine and on a native x86_64 desktop from the same code, so
    an unavailable backend is an expected state, not an error.

    The device is deliberately absent from every fingerprint. The same config on
    MPS and on CUDA produces different weights, which is a real provenance fact and
    belongs in `versions.json`; making it a cache key would discard an expensive
    checkpoint because the operator moved machines.
    """
    import torch

    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
