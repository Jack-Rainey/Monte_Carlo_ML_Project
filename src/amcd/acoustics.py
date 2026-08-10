"""Closed-form room acoustics shared by scene characterization and the scaffold.

Exists because the same physical constant was declared once (with its derivation)
in `scenes/generator.py` and repeated as a bare literal in `simulators/dry_run.py`,
where the two implementations then disagreed in their guards — so the T60 the
scene report described and the T60 the scaffold actually rendered could diverge
with nothing recording it (AC-24).

This is scene/statistical-model physics, NOT the ISO-3382 measurement path: the
metrics computed from rendered waveforms live in `evaluation/room_acoustic.py` and
must never be derived from these estimates (design_spec §3, metric source of truth).
"""
from __future__ import annotations

import math

#: Sabine's constant, 24 ln(10) / c ≈ 0.161 s·m⁻¹ at 20 °C (c = 343 m/s).
#: Verified: 24 * ln(10) / 343 = 0.161114.
SABINE_K = 0.161


def sabine_rt60(volume_m3: float, surface_m2: float, absorption: float) -> float:
    """Sabine reverberation time (s) for a room of `volume_m3` and `surface_m2`
    with a single mean absorption coefficient.

    Raises rather than clamping on a non-physical absorption: a silently clipped
    α produces a T60 that describes a different room than the one declared, which
    is the divergence AC-24 is about. Callers that need a defensive bound apply it
    themselves and record that they did.
    """
    if not 0.0 < absorption < 1.0:
        raise ValueError(
            f"absorption must be in (0, 1) for Sabine's formula; got {absorption!r}. "
            f"α ≤ 0 has no absorption to divide by and α ≥ 1 is a fully anechoic "
            f"surface, for which the diffuse-field model does not apply."
        )
    if volume_m3 <= 0 or surface_m2 <= 0:
        raise ValueError(
            f"volume and surface must be positive; got volume={volume_m3!r}, "
            f"surface={surface_m2!r}."
        )
    return SABINE_K * volume_m3 / (absorption * surface_m2)


def eyring_rt60(volume_m3: float, surface_m2: float, absorption: float) -> float:
    """Eyring reverberation time (s) — `-S·ln(1-α)` in place of Sabine's `S·α`.

    Sabine overestimates as α grows (the two agree only for small α); reporting
    both is how a reader sees that a highly absorptive split is outside the
    diffuse-field model's comfortable domain (AC-21).
    """
    if not 0.0 < absorption < 1.0:
        raise ValueError(
            f"absorption must be in (0, 1) for Eyring's formula; got {absorption!r}."
        )
    if volume_m3 <= 0 or surface_m2 <= 0:
        raise ValueError(
            f"volume and surface must be positive; got volume={volume_m3!r}, "
            f"surface={surface_m2!r}."
        )
    return SABINE_K * volume_m3 / (-surface_m2 * math.log1p(-absorption))
