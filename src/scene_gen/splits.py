from __future__ import annotations

from hashlib import sha256
from random import Random
from typing import Iterable, Sequence, TypeVar

from .config_schema import DatasetConfig, FamilyWeight, MaterialRegimeWeight, PlacementRegimeWeight, SplitSubsetConfig
from .scene_spec import SplitMetadata


T = TypeVar("T", FamilyWeight, MaterialRegimeWeight, PlacementRegimeWeight)


def weighted_choice(rng: Random, items: Sequence[T]) -> str:
    total = sum(item.weight for item in items)
    threshold = rng.uniform(0.0, total)
    cumulative = 0.0
    for item in items:
        cumulative += item.weight
        if threshold <= cumulative:
            return item.name
    return items[-1].name


def subset_seed(base_seed: int, subset_name: str, index: int) -> int:
    token = f"{base_seed}:{subset_name}:{index}".encode("utf-8")
    return int(sha256(token).hexdigest()[:12], 16)


def _top_level_split_name(subset_name: str) -> str:
    if subset_name.startswith("train"):
        return "train"
    if subset_name.startswith("valid"):
        return "valid"
    if subset_name.startswith("test"):
        return "test"
    return subset_name


def iter_split_metadata(cfg: DatasetConfig, subset_name: str) -> Iterable[SplitMetadata]:
    split_cfg = cfg.splits[subset_name]
    for index in range(split_cfg.count):
        yield SplitMetadata(
            split=_top_level_split_name(subset_name),
            subset=subset_name,
            split_seed=subset_seed(split_cfg.seed, subset_name, index),
            scene_index_within_subset=index,
        )


def choose_scene_factors(split_cfg: SplitSubsetConfig, rng: Random) -> tuple[str, str, str]:
    family = weighted_choice(rng, split_cfg.families)
    placement_regime = weighted_choice(rng, split_cfg.placement_regimes)
    material_regime = weighted_choice(rng, split_cfg.material_regimes)
    return family, placement_regime, material_regime
