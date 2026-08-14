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

    #: Value domain of encoded tensors: "db" (log band-energy, the banded-rep
    #: contract above) or "amplitude" (raw samples — the `waveform` identity rep).
    #: Declared, not inferred, and stamped into the preprocess meta so dB-assuming
    #: consumers (e.g. energy SNR's 10**(x/10) undo) key on the declared domain of
    #: the tensors they read — never on isinstance of a concrete rep.
    value_domain: str

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
        training.loss.build_criterion / delta_db_to_norm)."""
        ...


def build_representation(
    name: str,
    params: dict[str, Any],
    *,
    sample_rate: int,
    eval_freqs_hz: list[float],
) -> "Representation":
    """Instantiate a registered representation, validating `params` against its
    own nested `Params` schema (mirrors models.cnn.build_model).

    Keeps preprocess/infer/probe rep-agnostic: adding a new output domain needs
    only a registered class with its own `Params`.

    TWO CROSS-CUTTING ARGS, NOT ONE. With `sample_rate` alone,
    `configs/representations/spectrogram.yaml` had to re-declare the evaluation
    band set as `min_db_headroom_octave_centres_hz` — a second declaration of
    `iso_eval_freqs` held together only by a test asserting the two are equal. A
    representation whose guard is calibrated against the REPORTED metric bands has
    to be told what those are; it cannot see the master config, and inventing its
    own copy is how the described band set and the measured one drift apart.

    `eval_freqs_hz` is `config.iso_eval_freqs`. Passed to every representation
    rather than only to the ones that use it, for the reason `sample_rate` is: an
    argument a rep may ignore costs nothing, while a rep that needs it and cannot
    reach it grows a duplicate declaration.
    """
    RepClass = representation_registry.get(name)
    validated = RepClass.Params(**params).model_dump()
    return RepClass(
        sample_rate=sample_rate, eval_freqs_hz=list(eval_freqs_hz), **validated
    )
