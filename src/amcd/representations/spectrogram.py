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
    n_fft: int,
    sample_rate: int,
    *,
    reference_freq_hz: float,
    bands_per_octave: int,
    min_center_freq_hz: float,
    min_bins_per_band: int,
) -> tuple[torch.Tensor, dict]:
    """Rectangular fractional-octave filter bank, fully declared and self-describing.

    Returns the (n_bands, n_fft//2+1) float32 bank — each row sums power in one
    band — and a DESCRIPTION dict recording what the bank actually is.

    Why the description exists (AC-19). Measured at the production framing
    (48 kHz, n_fft 2048, 23.44 Hz bin spacing) the five lowest bands hold ONE FFT
    bin each, because a third-octave band at 125 Hz is only ~29 Hz wide. Those
    bands therefore measure Hann leakage rather than band content: a 63 Hz tone
    puts 57.7 % of its energy in the 78.7 Hz band and only 35.6 % in the 49.6 Hz
    band — it peaks in the WRONG band. In-band fraction is 99.4 % at 500 Hz and
    above, 93.4 % at 250 Hz, 56.8 % at 125 Hz.

    Two further properties were true but undeclared: dropping empty bands makes
    `center_freqs` an IRREGULAR series rather than a fractional-octave ladder, and
    ~6.1 % of a white-noise IR's STFT power falls above the top band edge or at DC
    and is in no band at all. Energy conservation WITHIN the covered range is
    exact, and the leakage is common-mode across legs, so paired metrics survive
    it; per-band interpretation does not.

    The response is to GUARD AND DECLARE, not to remove bands (AC-19's own
    wording): every constant that shapes the ladder is a config value, the
    resulting bank is described in `preprocessed/meta.json`, and
    `min_bins_per_band` gives the band floor a name. Choosing a value above 1 is a
    research decision with real cost — at production framing 3 bins would drop
    every band below ~315 Hz — so the number stays the researcher's, now with the
    measurement in front of them (ledger: DEFERRED at E2, multi-resolution
    sampling).
    """
    if min_bins_per_band < 1:
        raise ValueError(
            f"min_bins_per_band must be >= 1; got {min_bins_per_band}. A band with "
            f"no FFT bins has no content to sum."
        )
    n_freqs = n_fft // 2 + 1
    freqs = np.linspace(0.0, sample_rate / 2.0, n_freqs)
    nyquist = sample_rate / 2.0
    ratio = 2.0 ** (1.0 / bands_per_octave)
    half_width = 2.0 ** (1.0 / (2.0 * bands_per_octave))

    # Fractional-octave centres about the declared reference: f_n = f_ref * r^n.
    # A band is admitted only if it lies ENTIRELY below Nyquist (upper edge
    # included). This replaces a `sample_rate / 2 * 1.01` fudge factor that
    # admitted the top band by arithmetic accident rather than by a stated rule;
    # it reproduces that ladder exactly at the production framing (27 bands, top
    # edge 22627.4 Hz) while ruling out a band straddling Nyquist, which would be
    # half-empty and read as a real measurement.
    candidate_freqs: list[float] = []
    f = reference_freq_hz
    while f * half_width <= nyquist:
        candidate_freqs.append(f)
        f *= ratio
    f = reference_freq_hz / ratio
    while f >= min_center_freq_hz:
        candidate_freqs.insert(0, f)
        f /= ratio

    filters: list[np.ndarray] = []
    kept: list[dict] = []
    dropped: list[dict] = []
    for fc in candidate_freqs:
        f_lo, f_hi = fc / half_width, fc * half_width
        mask = (freqs >= f_lo) & (freqs < f_hi)
        n_bins = int(mask.sum())
        entry = {"center_hz": float(fc), "lo_hz": float(f_lo),
                 "hi_hz": float(f_hi), "n_bins": n_bins}
        if n_bins >= min_bins_per_band:
            filters.append(mask.astype(np.float32))
            kept.append(entry)
        else:
            dropped.append(entry)

    if not filters:
        raise ValueError(
            f"no band survived min_bins_per_band={min_bins_per_band} at n_fft="
            f"{n_fft}, sample_rate={sample_rate} (bin spacing "
            f"{sample_rate / n_fft:.2f} Hz). Lower min_bins_per_band or raise n_fft."
        )

    bank = torch.tensor(np.array(filters), dtype=torch.float32)
    description = {
        "center_freqs_hz": [b["center_hz"] for b in kept],
        "band_edges_hz": [[b["lo_hz"], b["hi_hz"]] for b in kept],
        "bins_per_band": [b["n_bins"] for b in kept],
        # The ladder is IRREGULAR once bands are dropped, so the covered range is
        # stated rather than inferred from the centres.
        "represented_band_limit_hz": [kept[0]["lo_hz"], kept[-1]["hi_hz"]],
        "fft_bin_spacing_hz": float(sample_rate / n_fft),
        # Power that reaches no band at all: below the lowest edge (incl. DC) or
        # above the highest. Measured 6.1 % for white noise at production framing.
        "uncovered_power_fraction": float(
            (~np.any(np.array(filters, dtype=bool), axis=0)).sum() / n_freqs
        ),
        "dropped_bands": dropped,
        "min_bins_per_band": min_bins_per_band,
    }
    return bank, description


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

        #: The fractional-octave ladder, declared rather than written as literals
        #: in the filter builder (AC-19). `reference_freq_hz` anchors it,
        #: `bands_per_octave` sets the spacing (3 = third-octave), and
        #: `min_center_freq_hz` bounds it below. The upper bound is DERIVED from
        #: Nyquist.
        reference_freq_hz: float
        bands_per_octave: int
        min_center_freq_hz: float

        #: Fewest FFT bins a band must contain to be kept. Below the bin spacing a
        #: "band" measures window leakage, not band content — see
        #: `_build_third_octave_filters`. Config-governed because raising it is a
        #: research decision about low-frequency coverage, not a code detail.
        min_bins_per_band: int

    # Per-channel third-octave log band-energy in dB (the banded-rep contract).
    value_domain = "db"

    def __init__(
        self,
        sample_rate: int,
        n_fft: int,
        hop_length: int,
        min_db: float,
        reference_freq_hz: float,
        bands_per_octave: int,
        min_center_freq_hz: float,
        min_bins_per_band: int,
    ) -> None:
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.min_db = min_db
        self._filter_bank, self.band_description = _build_third_octave_filters(
            n_fft, sample_rate,
            reference_freq_hz=reference_freq_hz,
            bands_per_octave=bands_per_octave,
            min_center_freq_hz=min_center_freq_hz,
            min_bins_per_band=min_bins_per_band,
        )
        self.center_freqs = self.band_description["center_freqs_hz"]
        self._window = torch.hann_window(n_fft)

    @property
    def n_bands(self) -> int:
        return self._filter_bank.shape[0]

    def describe_bands(self) -> dict:
        """The band structure plus a MEASURED in-band energy fraction per band.

        The static description says how wide each band is; this adds what the bank
        actually does with a pure tone at each centre, through the real STFT and
        window. That distinction is the whole of AC-19: the five lowest bands are
        nominally ~29 Hz wide but hold one 23.44 Hz bin, so their in-band fraction
        collapses (56.8 % at 125 Hz) while the geometry alone looks fine.

        Written into preprocessed/meta.json, so a per-band number is never read
        without the evidence of how much of it is really that band.
        """
        out = dict(self.band_description)
        out["in_band_energy_fraction"] = self._measure_in_band_fractions()
        return out

    def _measure_in_band_fractions(self) -> list[float]:
        """Fraction of a centre-frequency tone's STFT power landing in its own band."""
        # A steady tone, with the STFT's edge frames DISCARDED. With center=True
        # the first and last frames window the tone against zero padding, which
        # spreads energy for a reason that has nothing to do with the filter bank
        # — including them understated every band (94.5 % rather than 99.4 % at
        # 500 Hz) and would have made this artifact measure the probe.
        n = self.n_fft * 8
        t = torch.arange(n, dtype=torch.float64) / self.sample_rate
        edge = max(1, self.n_fft // (2 * self.hop_length) + 1)
        fractions: list[float] = []
        for band_idx, fc in enumerate(self.center_freqs):
            tone = torch.sin(2.0 * torch.pi * fc * t).float()
            stft = torch.stft(
                tone, n_fft=self.n_fft, hop_length=self.hop_length,
                window=self._window, return_complex=True, center=True,
            )
            steady = stft[:, edge:-edge] if stft.shape[1] > 2 * edge else stft
            power = steady.abs().pow(2).sum(dim=1)    # (n_freqs,)
            total = float(power.sum())
            in_band = float((self._filter_bank[band_idx] * power).sum())
            fractions.append(in_band / total if total > 0 else float("nan"))
        return fractions

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
