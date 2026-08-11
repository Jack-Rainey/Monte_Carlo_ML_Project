"""Dry-run simulator: synthetic IRs for pipeline validation. No x86 required."""
from __future__ import annotations

import numpy as np
from pydantic import BaseModel

from ..acoustics import diffuse_field_drr_db, room_constant, sabine_rt60
from ..registry import simulator_registry
from .base import IRResult, SceneSpec


@simulator_registry.register("dry_run")
class DryRunSimulator:
    """
    Generates realistic-ish IRs without running GSound-SIR.

    The synthetic IR is a function of the *scene geometry/material/placement* (so the
    distribution-shift splits actually differ at the tensor level, not just by label):
      - RT60 follows the Sabine equation from room volume + surface + absorption
        → geometry_shift (corridor) and material_shift (ceiling_absorptive) move it.
      - A BROADBAND direct impulse scaled by 1/distance, against a reverberant tail
        scaled from the room constant → the rendered direct-to-reverberant ratio
        equals the closed form `placement_report.json` publishes, so
        placement_shift (near_corner) genuinely moves C50/DRR. It did not before
        (AC-28): the direct component was an envelope with a 7.96 Hz corner, and
        C50 was flat to 0.02 dB across a 16x distance range.
      - A diffuse reverberant tail carries the Monte-Carlo noise that converges with
        ray budget (σ ∝ 1/√N) — this is the low→high signal to be denoised. The direct
        component is shared (early reflections resolve even at low ray count).
    Everything is deterministic from scene.seed + ray_budget.
    """

    class Params(BaseModel):
        """dry_run's config schema (configs/simulators/dry_run.yaml)."""
        model_config = {"extra": "forbid"}

        #: Propagation speed this backend uses, and declares into canonical render
        #: provenance. Unlike gsound — whose 344 m/s lives in C++ and can only be
        #: DECLARED (RD-19) — the scaffold genuinely obeys this value, so it is
        #: config-governed rather than a Python literal.
        speed_of_sound_m_s: float

        #: Smallest source-receiver separation this backend will render. Below it
        #: the 1/d direct term diverges and a real backend's source and listener
        #: spheres overlap, so the scene is geometrically degenerate rather than
        #: merely extreme. Declared, not hardcoded: the real backend's floor is
        #: source_radius + listener_radius, a different number (AC-13).
        min_source_receiver_distance_m: float

    def __init__(
        self,
        n_channels: int,
        n_samples: int,
        sample_rate: int,
        speed_of_sound_m_s: float,
        min_source_receiver_distance_m: float,
    ) -> None:
        self.n_channels = n_channels
        self.n_samples = n_samples
        self.sample_rate = sample_rate
        self.speed_of_sound_m_s = speed_of_sound_m_s
        self._min_separation_m = min_source_receiver_distance_m

    @classmethod
    def min_source_receiver_distance_m(cls, params: dict) -> float:
        """Required pre-render declaration (`Simulator`). Config-governed here:
        the scaffold's floor is a stated policy, not derivable from anything else
        it declares — unlike gsound_sir, whose floor follows from its sphere radii.
        """
        return float(params["min_source_receiver_distance_m"])

    def render(self, scene: SceneSpec, ray_budget: int) -> IRResult:
        # Separate RNG for scene structure vs noise — structure fixed, noise varies with budget
        rng_scene = np.random.default_rng(scene.seed)
        rng_noise = np.random.default_rng([scene.seed, ray_budget])

        t = np.linspace(0, self.n_samples / self.sample_rate, self.n_samples, endpoint=False)

        # --- Sabine RT60 from geometry + material (couples geometry & material shifts) ---
        lx, ly, lz = scene.dims
        volume = lx * ly * lz
        surface = 2.0 * (lx * ly + ly * lz + lx * lz)
        # One declared Sabine constant, shared with the scene report's own
        # characterization, so the T60 described and the T60 rendered cannot drift
        # apart (AC-24). The clamps below stay — they are numerical guards for the
        # scaffold — but they are now RECORDED in provenance rather than silent.
        alpha = float(np.clip(scene.material_absorption, 0.01, 0.99))
        rt60_native = sabine_rt60(volume, surface, alpha)
        rt60 = float(np.clip(rt60_native, 0.05, 3.0))
        rt60_clipped = rt60 != rt60_native

        # --- Direct-to-reverberant ratio from source↔receiver distance (placement shift) ---
        src = np.asarray(scene.source_pos, dtype=np.float64)
        rcv = np.asarray(scene.receiver_pos, dtype=np.float64)
        distance = float(np.linalg.norm(src - rcv))
        if distance < self._min_separation_m:
            # Was a silent `np.clip(..., 0.3, None)`. That made the scaffold report
            # the onset of a 0.3 m path for any closer pair — contradicting the
            # speed of sound it now declares into canonical provenance — and it
            # masked the fact that a real backend's source and listener spheres
            # would be overlapping at such a separation (F-43 / AC-13).
            raise ValueError(
                f"scene {scene.scene_id!r}: source-receiver separation "
                f"{distance:.4f} m is below {self._min_separation_m} m. At that range the "
                f"direct term 1/d diverges and a real backend's source/listener "
                f"spheres overlap. Declare a placement `distance_range` with a "
                f"lower bound of at least {self._min_separation_m} m."
            )
        direct_gain = 1.0 / distance

        # --- Propagation delay: nothing arrives before the direct sound (AC-11) ---
        # The whole response — direct AND reverberant tail — starts at d/c. Without
        # this the scaffold declared a speed of sound it did not obey (effective c
        # infinite, onset at sample 0 for every scene), so onset alignment, the
        # paired onset QC (Step 4) and the "direct sound is the loudest arrival"
        # assumption were all exercised only on signals a real backend never emits.
        # Delaying the tail too matters: a diffuse floor before the direct arrival
        # is not merely unphysical, it can sit above the -20 dB onset threshold and
        # capture the onset detector.
        delay_samples = int(round(distance / self.speed_of_sound_m_s * self.sample_rate))
        n_active = max(self.n_samples - delay_samples, 0)
        t_active = t[:n_active]

        decay = np.exp(-6.908 * t_active / rt60).astype(np.float32)

        # --- Direct arrival: a BROADBAND impulse, not an envelope (AC-28) ---
        #
        # A physical direct arrival is a broadband impulse scaled by 1/d. As a unit
        # sample it has flat energy in every band, so C50/DRR move with distance in
        # every band rather than in none. It is also the first and largest sample by
        # construction, which is what `_find_onset`'s AC-07 assumption requires.
        #
        # NOT MODELLED: a distinct early-reflection cluster between the direct
        # arrival and the diffuse onset. The tail begins at the direct arrival, so
        # this scaffold has no early-reflection structure — one more reason its D0b
        # verdicts are plumbing evidence, not acoustic results (RD-07). AC-43
        # measures the consequence: EDT is nearly inert on the placement axis
        # (non-monotone, 5.5 % spread over a 16x distance range) even though C50 is
        # live, so `test_placement_shift`'s EDT column is a plumbing result.
        direct = np.zeros(n_active, dtype=np.float32)
        if n_active > 0:
            direct[0] = direct_gain

        # --- Reverberant level from the room constant (AC-28/RD-75) ---
        #
        # The tail is scaled so the RENDERED direct-to-reverberant ratio equals the
        # closed-form DRR that `scenes/placement_report.json` publishes for this
        # same scene. Direct energy is direct_gain^2 = 1/d^2 and the tail carries
        # A^2 * sum(decay^2), so matching 1/(4*pi*d^2) against 4/R gives
        # A = sqrt(16*pi / (R * sum(decay^2))) — independent of d, which is what
        # leaves the whole distance dependence in the direct term (6 dB per
        # doubling) and puts the 0 dB crossing at d = r_c, since r_c = sqrt(R/16pi).
        #
        # Both quantities come from `amcd.acoustics`, for the reason AC-24 gave for
        # the T60: the room the report DESCRIBES and the room the scaffold RENDERS
        # must not be able to drift apart.
        surface_alpha = float(np.clip(scene.material_absorption, 0.01, 0.99))
        r_constant = room_constant(surface, surface_alpha)
        decay_energy = float(np.sum(decay.astype(np.float64) ** 2))
        diffuse_gain = float(
            np.sqrt(16.0 * np.pi / (r_constant * decay_energy)) if decay_energy > 0 else 0.0
        )

        # Per-channel amplitude variation (fixed by scene)
        channel_scales = (0.7 + 0.3 * rng_scene.random(self.n_channels)).astype(np.float32)

        # --- Diffuse tail: a CONVERGED response plus Monte-Carlo estimation noise ---
        #
        # The tail models what ray tracing converges TO, and must not BE the noise
        # (AC-18): a fixed realization of the room's diffuse response (rng_scene,
        # identical in both legs — the signal), plus estimation noise shrinking as
        # 1/sqrt(N) (rng_noise, budget-dependent — what the model must remove).
        # E[energy] is then (1 + 1/N)·decay², budget-independent to within 0.02 %,
        # while low - high is pure noise.
        #
        # The zero-mean form is deliberate. `decay * (1 + noise*noise_scale)` also
        # fixes the energy but produces a STRICTLY POSITIVE tail, and an impulse
        # response is a pressure signal oscillating about zero — a positive envelope
        # carries almost no energy in the 500/1000 Hz eval bands after octave
        # filtering, hollowing out the very metrics this scaffold exercises.
        noise_scale = float(1.0 / np.sqrt(max(ray_budget, 1)))

        ir = np.zeros((self.n_channels, self.n_samples), dtype=np.float32)
        for c in range(self.n_channels):
            converged = rng_scene.standard_normal(n_active).astype(np.float32)
            mc_noise = rng_noise.standard_normal(n_active).astype(np.float32)
            diffuse = diffuse_gain * decay * (converged + mc_noise * noise_scale)
            ir[c, delay_samples:] = channel_scales[c] * (direct + diffuse)

        return IRResult(
            ir=ir,
            meta={
                # Required provenance (REQUIRED_PROVENANCE_KEYS, RD-31).
                "simulator": "dry_run",
                "ray_budget": ray_budget,
                "speed_of_sound_m_s": self.speed_of_sound_m_s,
                # Synthesized directly in the channel basis with no SH encoding, so
                # no ambisonic convention is in play; declared explicitly rather
                # than left absent, so the field is never silently missing.
                "ambisonic_convention": "none_synthetic",
                "rng_seeded": True,  # fully determined by scene.seed + ray_budget
                # Backend-specific extras.
                "rt60_s": rt60,
                # Both recorded: the scene report characterizes the room with the
                # UNCLIPPED Sabine T60, so without these a clipped scene would be
                # described as one room and rendered as another (AC-24).
                "rt60_native_s": rt60_native,
                "rt60_clipped": bool(rt60_clipped),
                "distance_m": distance,
                "noise_scale": noise_scale,
                # ── What `noise_scale` is, and what it is NOT (AC-35) ──────────
                # It is the relative error of a mean formed from N independent
                # samples, 1/sqrt(N), applied PER PRESSURE SAMPLE of the diffuse
                # tail. The scaffold does not model what estimator that would be:
                # a real ray tracer's error depends on ray density per (octave
                # band, time bin), which involves diffuse_depth, the band count and
                # the decay length — none of which this backend consumes. So the
                # MAGNITUDE is an undeclared modelling assumption, not ray-count
                # physics, and it is stamped rather than left implicit.
                "noise_scale_basis": (
                    "1/sqrt(ray_budget) applied per pressure sample of the diffuse "
                    "tail; a modelling assumption, not a derived ray-tracing error"
                ),
                # The realized broadband converged-to-noise ratio of THIS leg,
                # exactly 10*log10(N) under the model above: 37.0 dB at 5,000 and
                # 53.0 dB at 200,000. Recorded so the Step-6 probe (RD-17) can put
                # a real gsound number beside it — under this scaffold the
                # denoising problem is nearly absent, which is what RD-07's caveat
                # says qualitatively and this says in dB.
                "realized_snr_db": float(-20.0 * np.log10(noise_scale)),
                # The reverberant level is set from the room constant so the
                # rendered DRR matches the closed form the scene report publishes
                # (AC-28). Recorded so the two can be compared without re-deriving.
                "room_constant_m2": float(r_constant),
                "diffuse_gain": diffuse_gain,
                "expected_drr_db": float(
                    diffuse_field_drr_db(surface, surface_alpha, distance)
                ),
            },
        )
