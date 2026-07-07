from typing import Protocol, runtime_checkable

import torch


@runtime_checkable
class Model(Protocol):
    def forward(self, x: torch.Tensor, aux: torch.Tensor | None = None) -> torch.Tensor:
        """x: (B, C, bands, frames) → (B, C, bands, frames) residual correction"""
        ...
