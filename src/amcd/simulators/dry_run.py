"""Dry-run simulator: synthetic IRs for pipeline validation. No x86 required."""
from __future__ import annotations

import numpy as np
from pydantic import BaseModel

from ..acoustics import sabine_rt60
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
      - A coherent direct component scaled by 1/distance → placement_shift (near_corner)
        changes the direct-to-reverberant ratio (C50/DRR).
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
        direct = (direct_gain * np.exp(-t_active / 0.02)).astype(np.float32)

        # Per-channel amplitude variation (fixed by scene)
        channel_scales = (0.7 + 0.3 * rng_scene.random(self.n_channels)).astype(np.float32)

        # --- Diffuse tail: a CONVERGED response plus Monte-Carlo estimation noise ---
        #
        # The tail must not BE the noise (AC-18). It previously read
        # `diffuse = decay * noise * noise_scale` with noise_scale = 1/sqrt(N), so
        # E[tail energy] scaled as 1/N: the low leg's late-window energy measured
        # 39.7x the high leg's — 16.0 dB, exactly 200000/5000 — which made low→high a
        # deterministic level shift (trivially learnable) whose converged limit is an
        # IR with no reverberant tail at all. That is the unnamed mechanism behind the
        # dry_run D0b "CARRIER BOTTLENECK" verdict being a plumbing artifact (RD-07).
        #
        # `decay * (1 + noise*noise_scale)` — the form AC-18 proposed — fixes the
        # energy but produces a STRICTLY POSITIVE tail (verified: zero sign changes
        # over a 200 ms window). An impulse response is a pressure signal that
        # oscillates about zero; a positive envelope carries almost no energy in the
        # 500/1000 Hz eval bands after octave filtering, which would hollow out the
        # ISO-3382 metrics this scaffold exists to exercise.
        #
        # So the tail models what ray tracing actually converges TO: a fixed
        # realization of the room's diffuse response (drawn from rng_scene, identical
        # in both legs — this is the signal), plus estimation noise that shrinks as
        # 1/sqrt(N) (drawn from rng_noise, budget-dependent — this is what the model
        # must remove). E[energy] is then (1 + 1/N)·decay², i.e. budget-independent to
        # within 0.02 %, while low - high is pure noise.
        noise_scale = float(1.0 / np.sqrt(max(ray_budget, 1)))

        ir = np.zeros((self.n_channels, self.n_samples), dtype=np.float32)
        for c in range(self.n_channels):
            converged = rng_scene.standard_normal(n_active).astype(np.float32)
            mc_noise = rng_noise.standard_normal(n_active).astype(np.float32)
            diffuse = decay * (converged + mc_noise * noise_scale)
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
            },
        )
