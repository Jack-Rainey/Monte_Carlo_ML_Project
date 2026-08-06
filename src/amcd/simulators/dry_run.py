"""Dry-run simulator: synthetic IRs for pipeline validation. No x86 required."""
from __future__ import annotations

import numpy as np
from pydantic import BaseModel

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

    def __init__(
        self,
        n_channels: int,
        n_samples: int,
        sample_rate: int,
        speed_of_sound_m_s: float,
    ) -> None:
        self.n_channels = n_channels
        self.n_samples = n_samples
        self.sample_rate = sample_rate
        self.speed_of_sound_m_s = speed_of_sound_m_s

    def render(self, scene: SceneSpec, ray_budget: int) -> IRResult:
        # Separate RNG for scene structure vs noise — structure fixed, noise varies with budget
        rng_scene = np.random.default_rng(scene.seed)
        rng_noise = np.random.default_rng([scene.seed, ray_budget])

        t = np.linspace(0, self.n_samples / self.sample_rate, self.n_samples, endpoint=False)

        # --- Sabine RT60 from geometry + material (couples geometry & material shifts) ---
        lx, ly, lz = scene.dims
        volume = lx * ly * lz
        surface = 2.0 * (lx * ly + ly * lz + lx * lz)
        alpha = float(np.clip(scene.material_absorption, 0.01, 0.99))
        rt60 = float(np.clip(0.161 * volume / (alpha * surface), 0.05, 3.0))

        # --- Direct-to-reverberant ratio from source↔receiver distance (placement shift) ---
        src = np.asarray(scene.source_pos, dtype=np.float64)
        rcv = np.asarray(scene.receiver_pos, dtype=np.float64)
        distance = float(np.clip(np.linalg.norm(src - rcv), 0.3, None))
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

        # Monte Carlo noise convergence: σ ∝ 1/√N
        noise_scale = float(1.0 / np.sqrt(max(ray_budget, 1)))

        ir = np.zeros((self.n_channels, self.n_samples), dtype=np.float32)
        for c in range(self.n_channels):
            noise = rng_noise.standard_normal(n_active).astype(np.float32)
            diffuse = decay * noise * noise_scale
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
                "distance_m": distance,
                "noise_scale": noise_scale,
            },
        )
