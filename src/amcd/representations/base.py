from typing import Any, Protocol, runtime_checkable

import numpy as np
import torch

from ..registry import representation_registry


@runtime_checkable
class Representation(Protocol):
    """Output-domain plugin (design_spec §8). encode/decode operate in the
    **energy domain**; for banded reps (the spectrogram, future octave/mel/EDR)
    all energy tensors are per-channel log band-energy in **dB** — normalization,
    loss δ, and ISO-3382 decode assume that unit. The band-less identity
    `waveform` rep is the sanctioned exception: it operates in raw amplitude
    (its "bands" axis is a length-1 placeholder) — see WaveformRepresentation.

    Concrete reps own their construction parameters via a nested pydantic
    `Params` schema and are built through `build_representation` (mirrors the
    models' `build_model`), so the master config never bakes in rep-specific
    fields (e.g. spectrogram framing) and swapping reps is drop-in.
    """

    #: Band centre frequencies (Hz) for the encoded bands; interface-level band
    #: metadata written to the preprocess stamp. Empty for band-less reps
    #: (e.g. `waveform`, whose "bands" axis is a length-1 placeholder).
    center_freqs: list[float]

    def encode(self, ir: np.ndarray) -> torch.Tensor:
        """ir: (C, T) float32 → energy tensor (C, bands, frames), dB log energy."""
        ...

    def decode(self, env: torch.Tensor, carrier: np.ndarray) -> np.ndarray:
        """env: (C, bands, frames) dB log energy, carrier: (C, T) → IR (C, T)."""
        ...

    def loss(self, pred: torch.Tensor, target: torch.Tensor, delta: float) -> torch.Tensor:
        """Scalar loss between predicted and target energy tensors.

        `delta` is the Huber knee; it MUST be expressed in the same domain as
        `pred`/`target` (the operand domain), NOT raw dB — the trainer scales a
        dB δ into the z-scored operand domain before calling (see
        training.loss.build_criterion / delta_db_to_norm; cross-ref DEFERRED
        F-06 on unifying the two loss sources of truth)."""
        ...


def build_representation(name: str, params: dict[str, Any], *, sample_rate: int) -> "Representation":
    """Instantiate a registered representation, validating `params` against its
    own nested `Params` schema (mirrors models.cnn.build_model).

    Keeps preprocess/infer/probe rep-agnostic: adding a new output domain needs
    only a registered class with its own `Params`; `sample_rate` is the sole
    cross-cutting arg (the analogue of `n_channels` for models)."""
    RepClass = representation_registry.get(name)
    validated = RepClass.Params(**params).model_dump()
    return RepClass(sample_rate=sample_rate, **validated)
