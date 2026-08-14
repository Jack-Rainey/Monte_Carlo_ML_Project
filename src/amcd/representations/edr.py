"""EDR representation stub — alternative to spectrogram for future experiments."""
import numpy as np
import torch
import torch.nn.functional as F
from pydantic import BaseModel

from ..registry import representation_registry


@representation_registry.register("edr")
class EDRRepresentation:
    """
    Energy Decay Relief representation stub.
    EDR[c, f, t] = integral from t to ∞ of |IR_c(f, τ)|² dτ (Schroeder integration).
    Not yet implemented; raises on use.
    """

    class Params(BaseModel):
        """Placeholder schema — real EDR framing/band params land when it is
        implemented (§3/D2). Empty for now so the build_representation seam works."""
        model_config = {"extra": "forbid"}

    # EDR is banded log energy in dB (Schroeder-integrated), like the spectrogram.
    value_domain = "db"

    def __init__(self, sample_rate: int, eval_freqs_hz: list[float]) -> None:
        self.sample_rate = sample_rate
        #: The reported metric bands (`config.iso_eval_freqs`). Accepted and unused:
        #: `build_representation` passes it to every representation so a rep that
        #: NEEDS it never has to grow its own copy of the band set.
        self.eval_freqs_hz = list(eval_freqs_hz)

    @property
    def center_freqs(self) -> list[float]:
        raise NotImplementedError("EDR representation not yet implemented.")

    def encode(self, ir: np.ndarray) -> torch.Tensor:
        raise NotImplementedError("EDR representation not yet implemented.")

    def decode(self, env: torch.Tensor, carrier: np.ndarray) -> np.ndarray:
        raise NotImplementedError("EDR representation not yet implemented.")

    def loss(self, pred: torch.Tensor, target: torch.Tensor, delta: float) -> torch.Tensor:
        raise NotImplementedError("EDR representation not yet implemented.")
