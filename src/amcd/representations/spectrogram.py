"""
Third-octave log power spectrogram representation (v1).

encode: (C, T) → (C, n_bands, n_frames) log energy in dB
decode: impose predicted envelope on low-ray carrier (D3, band-by-band rescaling)
loss:   Huber on log-band-energy (δ is O(1) in dB → resolves H2)
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from pydantic import BaseModel

from ..registry import representation_registry


def _build_third_octave_filters(
    n_fft: int, sample_rate: int
) -> tuple[torch.Tensor, list[float]]:
    """
    Rectangular third-octave filter bank.
    Returns:
      - (n_bands, n_fft//2+1) float32 — each row sums power in one band
      - list of center frequencies (Hz) for the kept bands
    Bands with zero FFT bins are dropped.
    """
    n_freqs = n_fft // 2 + 1
    freqs = np.linspace(0.0, sample_rate / 2.0, n_freqs)

    # ISO 3382 third-octave center frequencies: f_{n} = 1000 * 2^(n/3)
    candidate_freqs: list[float] = []
    f = 1000.0
    while f <= sample_rate / 2.0 * 1.01:
        candidate_freqs.append(f)
        f *= 2.0 ** (1.0 / 3.0)
    f = 1000.0 / 2.0 ** (1.0 / 3.0)
    while f >= 10.0:
        candidate_freqs.insert(0, f)
        f /= 2.0 ** (1.0 / 3.0)

    filters: list[np.ndarray] = []
    kept_freqs: list[float] = []
    for fc in candidate_freqs:
        f_lo = fc / 2.0 ** (1.0 / 6.0)
        f_hi = fc * 2.0 ** (1.0 / 6.0)
        mask = (freqs >= f_lo) & (freqs < f_hi)
        if mask.sum() > 0:
            filters.append(mask.astype(np.float32))
            kept_freqs.append(float(fc))

    return torch.tensor(np.array(filters), dtype=torch.float32), kept_freqs  # (n_bands, n_freqs)


@representation_registry.register("spectrogram")
class ThirdOctaveSpectrogram:
    class Params(BaseModel):
        """spectrogram's own config schema (design_spec §8 — reps own their schema).

        STFT framing (n_fft/hop_length) and the dB floor. §7 lists band resolution
        as a tuned-capable axis; octave/mel alternates (§3/D2) register beside this
        rep with their own Params, so none of these belong in the master config."""
        model_config = {"extra": "forbid"}
        n_fft: int
        hop_length: int
        min_db: float

    def __init__(self, sample_rate: int, n_fft: int, hop_length: int, min_db: float) -> None:
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.min_db = min_db
        self._filter_bank, self.center_freqs = _build_third_octave_filters(n_fft, sample_rate)
        self._window = torch.hann_window(n_fft)

    @property
    def n_bands(self) -> int:
        return self._filter_bank.shape[0]

    def encode(self, ir: np.ndarray) -> torch.Tensor:
        """
        ir: (C, T) float32 channel-first
        returns: (C, n_bands, n_frames) float32 log energy in dB
        """
        C, T = ir.shape
        ir_t = torch.from_numpy(ir)  # (C, T)

        window = self._window
        fb = self._filter_bank  # (n_bands, n_freqs)

        energy_bands: list[torch.Tensor] = []
        for c in range(C):
            stft = torch.stft(
                ir_t[c],
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                window=window,
                return_complex=True,
                center=True,
            )  # (n_freqs, n_frames)

            power = stft.abs().pow(2)  # (n_freqs, n_frames)
            # Apply filter bank: (n_bands, n_frames)
            band_energy = torch.einsum("bf,fn->bn", fb, power)
            energy_bands.append(band_energy)

        energy = torch.stack(energy_bands)  # (C, n_bands, n_frames)

        # Convert to dB, clamp at floor
        energy_db = 10.0 * torch.log10(energy.clamp(min=1e-10))
        energy_db = energy_db.clamp(min=self.min_db)

        return energy_db.float()

    def decode(self, env: torch.Tensor, carrier: np.ndarray) -> np.ndarray:
        """
        D3: impose predicted energy envelope on the low-ray carrier by
        per-band per-frame STFT amplitude rescaling.

        env: (C, n_bands, n_frames) predicted log energy (dB)
        carrier: (C, T) raw low-ray IR float32
        returns: (C, T) reconstructed IR float32

        Algorithm: STFT carrier → for each band b, compute scale[b,t] =
        sqrt(target_power[b,t] / current_power[b,t]) → apply to bin coefficients
        → iSTFT. Bins belonging to no filter-bank band pass through at scale=1.
        Velvet-noise / shaped-noise carrier generation is a later extension.
        """
        C, T = carrier.shape
        carrier_t = torch.from_numpy(carrier)  # (C, T)
        fb = self._filter_bank  # (n_bands, n_freqs)
        window = self._window

        # Mask: which bins have no assigned band (pass through at scale=1)
        no_band_mask = (fb.sum(dim=0) == 0)  # (n_freqs,)

        output_channels: list[np.ndarray] = []
        for c in range(C):
            stft = torch.stft(
                carrier_t[c],
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                window=window,
                return_complex=True,
                center=True,
            )  # (n_freqs, n_frames)

            # Compute current power per band per frame
            power = stft.abs().pow(2)  # (n_freqs, n_frames)
            current_power = torch.einsum("bf,fn->bn", fb, power)  # (n_bands, n_frames)

            # Target power from env (dB → linear)
            env_c = env[c]  # (n_bands, n_frames)
            target_power = torch.pow(10.0, env_c / 10.0)  # (n_bands, n_frames)

            # Scale per band per frame
            scale = torch.sqrt(
                target_power / current_power.clamp(min=1e-30)
            )  # (n_bands, n_frames)
            # Where current power is negligible, keep the carrier as-is (scale=1)
            scale = torch.where(current_power > 1e-30, scale, torch.ones_like(scale))

            # Build per-frequency-bin scale array: (n_freqs, n_frames)
            # Each bin gets the scale of its band; unassigned bins get 1.0
            bin_scale = torch.ones(fb.shape[1], scale.shape[1])  # (n_freqs, n_frames)
            for b in range(fb.shape[0]):
                bin_mask = fb[b].bool()  # (n_freqs,)
                bin_scale[bin_mask] = scale[b]  # broadcast (n_frames,) across selected bins

            stft_out = stft * bin_scale  # (n_freqs, n_frames)

            ir_c = torch.istft(
                stft_out,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                window=window,
                length=T,
            )
            output_channels.append(ir_c.detach().numpy().astype(np.float32))

        return np.stack(output_channels)  # (C, T)

    def loss(self, pred: torch.Tensor, target: torch.Tensor, delta: float) -> torch.Tensor:
        """Huber on log-band-energy. δ is O(1)-meaningful in dB (resolves H2, §2/§3) ONLY
        when `delta` is expressed in the same domain as `pred`/`target`. The production
        path z-scores operands by high_std, so the trainer scales δ there — see
        training.loss.build_criterion / delta_db_to_norm. Passing a raw-dB δ against
        z-scored operands (as some tests do directly) does NOT carry the dB meaning."""
        return F.huber_loss(pred, target, delta=delta, reduction="mean")
