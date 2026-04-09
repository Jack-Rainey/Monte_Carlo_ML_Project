"""Procedural scene generation package for synthetic HOA room-impulse-response datasets."""

from .config_schema import DatasetConfig, load_dataset_config
from .scene_spec import SceneSpec

__all__ = ["DatasetConfig", "SceneSpec", "load_dataset_config"]
