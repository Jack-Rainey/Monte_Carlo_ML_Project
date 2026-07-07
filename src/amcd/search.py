"""
Hyperparameter-search strategies (design_spec §7 — the `tuned` role).

This is a SEAM, deliberately not an engine yet. The config grammar
(`{tune: {space, scale}, value: ...}`) is parsed, validated, and resolved to a
concrete operating point elsewhere (config.py); a run at the current ledger gate
(D0/E1) needs exactly that concrete point and no search at all.

The strategies below are registered by name so a config can *declare* how a model
would be tuned (`search: {strategy: grid}`), but only `fixed` — "use the declared
operating point, run no search" — is implemented. grid / factorial / evolutionary
raise NotImplementedError until the E3 tuning gate, at which point each becomes a
selection loop over `Config.expand_*` trial configs choosing on the validation
split (never on any test split — invariant #2/#8).
"""
from __future__ import annotations

from typing import Protocol

from .config import Config
from .registry import search_strategy_registry


class SearchStrategy(Protocol):
    def select(self, base: Config) -> Config:
        """Return the single selected concrete Config (selection on valid only)."""
        ...


@search_strategy_registry.register("fixed")
class FixedStrategy:
    """No search: use the declared operating point as-is (the E1 path)."""

    def select(self, base: Config) -> Config:
        return base


class _UnimplementedStrategy:
    strategy_name = "<unimplemented>"

    def select(self, base: Config) -> Config:
        raise NotImplementedError(
            f"Search strategy {self.strategy_name!r} is a registered stub — the "
            f"config grammar accepts it, but the selection engine is gated on the "
            f"E3 tuning ledger step (design_spec §7/§11). Use `fixed` until then."
        )


@search_strategy_registry.register("grid")
class GridStrategy(_UnimplementedStrategy):
    strategy_name = "grid"


@search_strategy_registry.register("factorial")
class FactorialStrategy(_UnimplementedStrategy):
    strategy_name = "factorial"


@search_strategy_registry.register("evolutionary")
class EvolutionaryStrategy(_UnimplementedStrategy):
    strategy_name = "evolutionary"


def get_strategy(name: str) -> SearchStrategy:
    return search_strategy_registry.get(name)()
