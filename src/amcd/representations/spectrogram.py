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
from pydantic import BaseModel, Field

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
    measurement in front of them (ledger row `AC-19-value`, DEFERRED at E2 —
    the roadmap fix is multi-resolution sampling).
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
    # admitted the top band by arithmetic accident rather than by a stated rule,
    # ruling out a band straddling Nyquist, which would be half-empty and read as
    # a real measurement.
    #
    # It is NOT behaviour-preserving in general — only at 48 kHz (F-59). MEASURED
    # against the old rule at min_bins_per_band=1, as `new (was old)`:
    #     48000/2048 → 27 (was 27), bank bit-identical    48000/512 → 21 (was 21)
    #      8000/256  → 18 (was 19), one band LOST        44100/2048 → 27 (was 28)
    # At 44.1 kHz the change leaves 17959.4-22050 Hz — 18.5 % of the spectrum — in
    # NO band, where `decode()` passes it through unshaped at scale 1.0. That is
    # the correct trade (a straddling band is not a measurement) but it is a real
    # change of coverage at any framing other than 48 kHz, so it is stated rather
    # than implied.
    candidate_freqs: list[float] = []
    f = reference_freq_hz
    while f * half_width <= nyquist:
        candidate_freqs.append(f)
        f *= ratio
    # The rung the Nyquist rule just rejected. Recorded (F-59) because a candidate
    # excluded by the loop CONDITION never becomes a candidate, so it could not
    # appear in `dropped_bands` — meta.json under-reported which bands were removed
    # and why. Only the first is recorded: the ladder above it is infinite.
    nyquist_excluded = {
        "center_hz": float(f),
        "lo_hz": float(f / half_width),
        "hi_hz": float(f * half_width),
        "n_bins": 0,
        "reason": "upper edge above Nyquist",
    }
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
            dropped.append({
                **entry,
                "reason": f"fewer than min_bins_per_band={min_bins_per_band} FFT bins",
            })

    # Kept ascending in frequency, so `dropped_bands` reads as one ordered ladder.
    dropped.append(nyquist_excluded)

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
        # FFT BINS that reach no band at all: below the lowest edge (incl. DC) or
        # above the highest. A property of the BANK, not of any signal — it was
        # named `uncovered_power_fraction`, which a reader takes as "this fraction
        # of the run's power is in no band" (AC-31). The two coincide only for a
        # white spectrum: measured at production framing this figure is 0.0585
        # while the true uncovered POWER is 0.0610 for white noise and 0.7794 for
        # a low-passed signal. Renamed rather than redefined, because the bank
        # property is what belongs in a bank description.
        "uncovered_bin_fraction": float(
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

        #: Lower clamp on the encoded band energy, in **dB re 1.0 in STFT power of
        #: the float32 IR** — an ABSOLUTE floor, not a level below the per-scene
        #: peak (AC-33). The reference has to be stated because the absolute scale
        #: of an IR is set by the backend (`normalize_ir: false`,
        #: `source_power: 1.0 W`, the 1/d direct term), so the effective
        #: dB-below-peak floor is scene-dependent: measured, encoding the SAME IR
        #: at gains 0.01 … 100 moves the clamped fraction 0.760 → 0.513 and the
        #: retained range below peak 91.7 → 171.7 dB.
        #:
        #: Absolute is the DELIBERATE choice, not an oversight. Absolute level
        #: carries the placement axis — DRR and C50 are level ratios against the
        #: 1/d direct term — so a per-scene-peak-relative floor would normalize
        #: away part of the signal `test_placement_shift` exists to test. The cost
        #: is that a second backend, or a different `source_power`, silently moves
        #: the floor; `TestMinDbIsAnAbsoluteFloor` in tests/test_filterbank.py pins
        #: the dependence so it cannot change unnoticed.
        min_db: float

        #: The fractional-octave ladder, declared rather than written as literals
        #: in the filter builder (AC-19). `reference_freq_hz` anchors it,
        #: `bands_per_octave` sets the spacing (3 = third-octave), and
        #: `min_center_freq_hz` bounds it below. The upper bound is DERIVED from
        #: Nyquist.
        #:
        #: The bounds are enforced, not documented (AC-32): all three build the
        #: ladder by repeated multiplication, so three schema-valid settings made
        #: `_build_third_octave_filters` loop FOREVER — a hang or an OOM at
        #: preprocess, i.e. AFTER the render it would waste.
        #:   reference_freq_hz = 0 → `0 * half_width <= nyquist` is true forever
        #:   min_center_freq_hz = 0 → the descending `f /= ratio` underflows to 0
        #:                            and `0 >= 0` stays true
        #:   bands_per_octave < 1 → ratio <= 1, so the ascending loop never
        #:                          reaches Nyquist and appends without bound
        reference_freq_hz: float = Field(
            gt=0.0,
            description="ladder anchor in Hz; must be > 0 or the ascending loop never terminates",
        )
        bands_per_octave: int = Field(
            ge=1,
            description="bands per octave; must be >= 1 or the ladder ratio is <= 1 and never reaches Nyquist",
        )
        min_center_freq_hz: float = Field(
            gt=0.0,
            description="lower bound of the ladder in Hz; must be > 0 or the descending loop underflows to 0 and never terminates",
        )

        #: Fewest FFT bins a band must contain to be kept. Below the bin spacing a
        #: "band" measures window leakage, not band content — see
        #: `_build_third_octave_filters`. Config-governed because raising it is a
        #: research decision about low-frequency coverage, not a code detail.
        min_bins_per_band: int

        #: Smallest dB gap `encode` will accept between a scene's WEAKEST per-band
        #: peak and `min_db` (AC-37). Below it, `encode` raises rather than
        #: returning an envelope that `decode` will turn into an injected energy
        #: floor.
        #:
        #: The guard exists because `min_db` is ABSOLUTE (see above), so a scene
        #: reaches it by LEVEL alone. `decode` rescales the carrier's band power to
        #: `10**(env/10)`, so wherever the clamp is active the decode BOOSTS the
        #: carrier UP to the floor, injecting a non-decaying tail into the
        #: prediction — inside the Schroeder window, which is shared and set by the
        #: PHYSICAL legs (AC-17/RD-43), so the prediction never gets its own Lundeby
        #: cut to truncate the injection away.
        #:
        #: CALIBRATED, not chosen. A definitionally perfect oracle
        #: (`decode(encode(high), low)`) was swept in 1 dB steps of level over six
        #: scenes spanning the declared support, recording the headroom at which
        #: its T30 first breaches `d0b_t30_jnd_frac`:
        #:
        #:     scene            native hr   last OK   first breach
        #:     large   a=0.05     72.6 dB   39.6 dB   38.6 dB (7.6 %)
        #:     large   a=0.16     72.5 dB   46.5 dB   45.5 dB (6.8 %)
        #:     medium  a=0.30     72.2 dB   42.2 dB   41.2 dB (5.6 %)
        #:     small   a=0.50     73.2 dB   39.2 dB   38.2 dB (5.2 %)
        #:     corridor a=0.20    74.1 dB   44.1 dB   43.1 dB (6.3 %)
        #:     absorptive a=0.80  68.7 dB   37.7 dB   36.7 dB (6.1 %)
        #:
        #: The breach point is SCENE-DEPENDENT (36.7-46.5 dB), so no single scalar
        #: is simultaneously tight and safe: admitting every scene that still
        #: measures correctly needs a value <= 46.5, and rejecting every scene that
        #: breaches needs > 45.5. That is a 1 dB window, and it is a real property
        #: of the defect, not of this calibration.
        #:
        #: The shipped 50.0 deliberately errs toward REJECTING, because the two
        #: errors are not symmetric: a false positive fails loudly at preprocess,
        #: while a false negative is a silently wrong reported ISO metric that
        #: surfaces as an apparent model failure. So 50.0 sits 3.5 dB above the
        #: worst measured survivor and 18.7 dB below the tightest scene at its
        #: native level. It ALSO rejects scenes carrying 46.5-50 dB of headroom that
        #: would still have measured within JND — that is the accepted cost, stated
        #: rather than discovered later.
        #:
        #: Physical reading: T30 regresses the EDR over -5 to -35 dB, so 35 dB of
        #: genuine range is the floor of what any declared value may permit; 50 dB
        #: is that span plus 15.
        #:
        #: The calibration is against the SCAFFOLD's level convention
        #: (`direct = 1/d`, room-constant tail). A real gsound render sets absolute
        #: level from `source_power` and `normalize_ir`, so this value must be
        #: re-measured at the dataset-render gate — which is precisely why the guard
        #: RAISES instead of clamping harder: a wrong value fails loudly at
        #: preprocess rather than silently biasing every reported ISO metric.
        min_db_headroom_db: float

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
        min_db_headroom_db: float,
    ) -> None:
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.min_db = min_db
        self.min_db_headroom_db = min_db_headroom_db
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

    def _check_min_db_headroom(self, energy_db: torch.Tensor) -> None:
        """Refuse a scene whose level has slid down onto the absolute `min_db`
        floor (AC-37).

        `min_db` is absolute, so this is reachable by LEVEL alone, and the
        consequence lands in `decode`, not here: the clamp becomes a target power
        the carrier is rescaled UP to, injecting a non-decaying floor into the
        prediction inside a Schroeder window the prediction does not control.

        Checked PER BAND, on the peak over channels and frames, because the
        reported ISO-3382 metrics are per-band quantities — one band on the floor
        corrupts the band average even where the broadband peak looks healthy.
        Measured on a definitionally perfect oracle: a scene 30 dB below its native
        level reads T30 2.1959 s against its target's 0.9681 s — a 126.8 % error —
        at 42.6 dB of headroom, against the shipped 50 dB.

        Raises rather than clamps. A representation cannot know whether a quiet
        scene is a modelling choice or a mis-scaled render, and silently returning
        an envelope that produces a physically wrong T30 is the failure AC-37
        exists to prevent — it would surface on the emulated gsound render as an
        apparent model failure.
        """
        # Peak over FRAMES only, per (channel, band): (C, n_bands).
        #
        # Maxing over channels too would hide the one that matters. The reported
        # ISO-3382 path reads `ir[0]` — the W channel — exclusively
        # (`compute_room_acoustic_metrics`), so a W channel sitting on the floor
        # while a higher-order channel stays loud is precisely the case that
        # corrupts every reported metric, and a channel-max would accept it. Not
        # hypothetical under the real backend: `simulators/base.py` declares
        # `acn_n3d`, and N3D scales degree l by sqrt(2l+1).
        per_channel_band_peak_db = torch.amax(energy_db, dim=2)
        headroom = per_channel_band_peak_db - self.min_db
        flat = int(torch.argmin(headroom))
        worst_c, worst_b = divmod(flat, headroom.shape[1])
        worst_headroom = float(headroom[worst_c, worst_b])
        if worst_headroom < self.min_db_headroom_db:
            raise ValueError(
                f"scene rejected by the min_db headroom guard (AC-37): channel "
                f"{worst_c}'s {self.center_freqs[worst_b]:.1f} Hz band peaks at "
                f"{float(per_channel_band_peak_db[worst_c, worst_b]):.1f} dB, "
                f"only {worst_headroom:.1f} dB "
                f"above min_db={self.min_db:g} dB, against the declared "
                f"min_db_headroom_db={self.min_db_headroom_db:g} dB.\n"
                f"min_db is an ABSOLUTE floor, so this scene's LEVEL has reached it. "
                f"decode() rescales the carrier's band power up to the clamped "
                f"envelope, which would inject a non-decaying energy floor into the "
                f"prediction inside the shared Schroeder window and report a T30 that "
                f"is a property of min_db, not of the room.\n"
                f"Fix the level (source_power / normalize_ir / the backend's gain "
                f"convention) or declare a lower min_db in "
                f"configs/representations/spectrogram.yaml — do not raise "
                f"min_db_headroom_db to silence this."
            )

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
        # Checked BEFORE the clamp: after it, a band sitting entirely on the floor
        # reads exactly min_db and its true distance below is unrecoverable (AC-37).
        self._check_min_db_headroom(energy_db)
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
