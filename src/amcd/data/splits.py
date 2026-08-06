"""
Config-driven split assignment (design_spec §6.1, invariants #1/#9/#10).

The split *set* lives entirely in `config.splits` — this module hardcodes no
split names. A scene carries its target split name in `split_regime` and is
routed straight there; only a scene tagged `"id"` is hash-bucketed.

Which scenes are tagged `"id"` depends on the id-pool sizing mode the config
declares (see `SplitSpec`), and this module needs no branch for it:

  frac mode  — the generator emits one pooled set tagged `"id"`, and they are
               hash-bucketed here into train/valid/test_id by cumulative `frac`
               (the split with no `frac` is the residual). The hash is stable
               under adding regime bookkeeping fields, and keyed by the dedicated
               `split_assignment` seed so membership shares no entropy with other
               stages.
  count mode — the generator already emitted each id-pool split's exact count
               under that split's own name and seed, so those scenes take the
               direct-routing branch below exactly as shift scenes do. Nothing
               is hash-bucketed, which is why an explicit `seeds.split_assignment`
               is rejected as inert in this mode.

Shift scenes are always routed directly, never into train/valid — controlled-shift
integrity (invariant #10).
"""
from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..config import Config

# Fields excluded from the id-pool hash so that adding regime bookkeeping to a
# scene spec does not change prior assignments (they are constant for id scenes).
_EXCLUDED_FROM_HASH = ("split_regime", "regime_axes")
_HASH_BUCKETS = 10000


def assign_split(scene_spec_dict: dict, config: "Config") -> str:
    """Deterministically assign a scene to one config-declared split.

    Shift scenes (`split_regime` != "id") route to that named split directly.
    id scenes are hash-bucketed into the id-pool splits by cumulative `frac`.
    """
    regime = scene_spec_dict.get("split_regime", "id")
    if regime != "id":
        if regime not in config.splits:
            raise KeyError(
                f"Scene declares split {regime!r} but it is not in config.splits "
                f"({list(config.splits)})"
            )
        return regime

    bucket = _hash_bucket(scene_spec_dict, config.seed("split_assignment"))

    id_pool = config.id_pool_splits
    cum = 0.0
    for name, spec in id_pool.items():
        if spec.frac is None:
            continue  # residual handled below
        cum += spec.frac
        if bucket < cum:
            return name
    # Residual id-pool split (the one without a frac — validated unique in Config).
    return next(name for name, spec in id_pool.items() if spec.frac is None)


def _hash_bucket(scene_spec_dict: dict, seed: int) -> float:
    hash_dict = {k: v for k, v in scene_spec_dict.items() if k not in _EXCLUDED_FROM_HASH}
    h = hashlib.sha256(json.dumps(hash_dict, sort_keys=True).encode())
    h.update((int(seed) % (2**64)).to_bytes(8, "little"))
    return int(h.hexdigest(), 16) % _HASH_BUCKETS / _HASH_BUCKETS
