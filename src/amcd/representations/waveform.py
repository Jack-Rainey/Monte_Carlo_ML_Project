"""Waveform representation stub — for E1 faithfulness re-run (reproduce old null result)."""
import numpy as np
import torch
import torch.nn.functional as F
from pydantic import BaseModel

from ..registry import representation_registry


@representation_registry.register("waveform")
class WaveformRepresentation:
    """
    Identity representation: raw waveform samples as target.
    Ill-posed in the diffuse tail (incoherent zero-mean noise → conditional mean ≈ 0
    → identity collapse). Used only for E1 to reproduce the prior negative result.
    """

    class Params(BaseModel):
        """No framing params — the waveform rep is parameter-free (built through the
        same build_representation seam as spectrogram; sample_rate is passed in)."""
        model_config = {"extra": "forbid"}

    # Raw samples, NOT dB log energy: dB-assuming consumers (energy SNR) must see
    # this and report their metric undefined rather than 10**(amplitude/10) (F-19).
    value_domain = "amplitude"

    def __init__(self, sample_rate: int, eval_freqs_hz: list[float]) -> None:
        self.sample_rate = sample_rate
        #: The reported metric bands (`config.iso_eval_freqs`). Accepted and unused:
        #: `build_representation` passes it to every representation so a rep that
        #: NEEDS it never has to grow its own copy of the band set (RD-187).
        self.eval_freqs_hz = list(eval_freqs_hz)

    @property
    def center_freqs(self) -> list[float]:
        # Band-less: the "bands" axis is a length-1 placeholder (see encode).
        return []

    def encode(self, ir: np.ndarray) -> torch.Tensor:
        # Returns (C, 1, T) to match (C, bands, frames) shape convention
        return torch.from_numpy(ir).unsqueeze(1).float()

    def decode(self, env: torch.Tensor, carrier: np.ndarray) -> np.ndarray:
        # env: (C, 1, T) → (C, T)
        return env.squeeze(1).numpy()

    def loss(self, pred: torch.Tensor, target: torch.Tensor, delta: float) -> torch.Tensor:
        # Huber with δ (config.huber_delta) in raw amplitude space: inert when |residual| ≪ δ (H2)
        return F.huber_loss(pred, target, delta=delta, reduction="mean")
