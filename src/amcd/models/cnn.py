"""
vanilla_cnn — the Research-I baseline denoising model.

Input/output: (B, C, n_bands, n_frames) energy tensors.
Architecture: stacked Conv2d blocks; the model predicts the *correction* to the
low-ray energy (residual framing, H4 — applied by the trainer as pred = low + model(low)).

Parameters come from configs/models/vanilla_cnn.yaml via `config.model.params`;
this module has no default parameter values. The model validates its own params
through the nested `Params` schema so a generic trainer stays model-agnostic.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from pydantic import BaseModel

from ..registry import model_registry


@model_registry.register("vanilla_cnn")
class CNNDenoisingModel(nn.Module):
    class Params(BaseModel):
        """vanilla_cnn's own config schema (design_spec §8 — models own schema).

        Currently exposes width (hidden_channels) and depth (n_layers). Kernel size
        and dilation are fixed (kernel=3, padding=1) for Research-I; §7 lists them as
        tuned-capable, so they would be added here when promoted to search params."""
        model_config = {"extra": "forbid"}
        hidden_channels: int
        n_layers: int

    def __init__(self, n_channels: int, hidden_channels: int, n_layers: int) -> None:
        super().__init__()
        if n_layers < 2:
            raise ValueError("n_layers must be ≥ 2")

        layers: list[nn.Module] = [
            nn.Conv2d(n_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        ]
        for _ in range(n_layers - 2):
            layers += [
                nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
            ]
        layers.append(nn.Conv2d(hidden_channels, n_channels, kernel_size=3, padding=1))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, aux: torch.Tensor | None = None) -> torch.Tensor:
        # x: (B, C, n_bands, n_frames) normalized low-ray energy
        # returns residual correction of same shape
        # aux: path-conditioning seam (Model protocol, models/base.py) — ignored here
        return self.net(x)


def build_model(name: str, n_channels: int, params: dict) -> nn.Module:
    """Instantiate a registered model, validating `params` against its own schema.

    Keeps trainer/infer model-agnostic: adding a non-CNN model needs no change
    here beyond registering it with its own `Params` schema.
    """
    ModelClass = model_registry.get(name)
    validated = ModelClass.Params(**params).model_dump()
    return ModelClass(n_channels=n_channels, **validated)
