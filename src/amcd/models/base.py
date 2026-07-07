from typing import Protocol, runtime_checkable

import torch


@runtime_checkable
class Model(Protocol):
    def forward(self, x: torch.Tensor, aux: torch.Tensor | None = None) -> torch.Tensor:
        """x: (B, C, bands, frames) → (B, C, bands, frames) residual correction.

        `aux` is the forward-looking seam for the path-conditioned variant
        (research_I_paper.md §4.4 / App. C; signature per design_spec §8): per-scene
        path features exported by the gsound_sir backend (see DEFERRED RD-08 —
        IRResult.paths lands with that backend). Unused by vanilla_cnn; None today.
        """
        ...
