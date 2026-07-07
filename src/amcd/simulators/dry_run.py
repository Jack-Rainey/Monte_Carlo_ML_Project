"""Dry-run simulator: synthetic IRs for pipeline validation. No x86 required."""
from __future__ import annotations

import numpy as np

from ..registry import simulator_registry
from .base import IRResult, SceneSpec


_SPEED_OF_SOUND = 343.0  # m/s


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

    def __init__(self, n_channels: int, n_samples: int, sample_rate: int) -> None:
        self.n_channels = n_channels
        self.n_samples = n_samples
        self.sample_rate = sample_rate

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
        decay = np.exp(-6.908 * t / rt60).astype(np.float32)

        # --- Direct-to-reverberant ratio from source↔receiver distance (placement shift) ---
        src = np.asarray(scene.source_pos, dtype=np.float64)
        rcv = np.asarray(scene.receiver_pos, dtype=np.float64)
        distance = float(np.clip(np.linalg.norm(src - rcv), 0.3, None))
        direct_gain = 1.0 / distance
        direct = (direct_gain * np.exp(-t / 0.02)).astype(np.float32)  # early coherent part

        # Per-channel amplitude variation (fixed by scene)
        channel_scales = (0.7 + 0.3 * rng_scene.random(self.n_channels)).astype(np.float32)

        # Monte Carlo noise convergence: σ ∝ 1/√N
        noise_scale = float(1.0 / np.sqrt(max(ray_budget, 1)))

        ir = np.empty((self.n_channels, self.n_samples), dtype=np.float32)
        for c in range(self.n_channels):
            noise = rng_noise.standard_normal(self.n_samples).astype(np.float32)
            diffuse = decay * noise * noise_scale
            ir[c] = channel_scales[c] * (direct + diffuse)

        return IRResult(
            ir=ir,
            meta={
                "simulator": "dry_run",
                "rt60_s": rt60,
                "distance_m": distance,
                "ray_budget": ray_budget,
                "noise_scale": noise_scale,
            },
        )
