from typing import TypeVar, Generic

T = TypeVar("T")


class Registry(Generic[T]):
    def __init__(self, name: str) -> None:
        self._name = name
        self._entries: dict[str, type] = {}

    def register(self, name: str):
        def decorator(cls: type) -> type:
            if name in self._entries:
                raise ValueError(f"Duplicate {self._name} registration: {name!r}")
            self._entries[name] = cls
            return cls
        return decorator

    def get(self, name: str) -> type:
        if name not in self._entries:
            available = sorted(self._entries)
            raise KeyError(
                f"Unknown {self._name}: {name!r}. Available: {available}"
            )
        return self._entries[name]

    def __contains__(self, name: str) -> bool:
        return name in self._entries

    def keys(self) -> list[str]:
        return sorted(self._entries)


simulator_registry: Registry = Registry("simulator")
representation_registry: Registry = Registry("representation")
model_registry: Registry = Registry("model")
# Reserved for the design_spec §5/§8 `Metric` plugin seam (metrics as registered,
# swappable plugins). No registrants yet: eval currently calls the metric functions
# in evaluation/ directly; they migrate behind this registry when the Metric
# Protocol lands, so a new metric = drop a file + register, no eval-stage surgery.
metric_registry: Registry = Registry("metric")
# Hyperparameter-search strategies (design_spec §7). The grammar + registry are
# the seam; only `fixed` is implemented for now — grid/factorial/evolutionary are
# registered stubs that raise until the E3 tuning gate (see search.py).
search_strategy_registry: Registry = Registry("search_strategy")
