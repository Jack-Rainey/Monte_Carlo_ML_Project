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


def box_volume_and_surface(dims: tuple[float, float, float]) -> tuple[float, float]:
    """Volume (m³) and total interior surface (m²) of a shoebox.

    Shared for the AC-24 reason the T60 formulas are: `scenes/generator.py`
    characterizes the room from these two numbers and `simulators/gsound_sir.py`
    predicts its realized record length from them (AC-184). Two inlined copies of
    `2*(lx*ly + ly*lz + lx*lz)` is exactly how the described room and the rendered
    room drift apart.
    """
    lx, ly, lz = (float(d) for d in dims)
    if lx <= 0 or ly <= 0 or lz <= 0:
        raise ValueError(f"shoebox dims must all be positive; got {dims!r}.")
    return lx * ly * lz, 2.0 * (lx * ly + ly * lz + lx * lz)


def predicted_support_s(t60_s: float, coefficient_s: float, exponent: float) -> float:
    """How many seconds of record a backend is predicted to produce for a `t60_s`
    decay: `coefficient_s * t60_s ** exponent`.

    The functional form, not the coefficients — those are a backend fact and are
    declared per simulator in config, because a different raytracer trims its
    record differently. An `exponent` below 1 means support grows more slowly than
    the decay it must contain, so the captured fraction falls as rooms get more
    reverberant; that is the AC-184 finding, and it is why a single declared record
    length cannot express this bound.

    Lives here rather than in `simulators/gsound_sir.py` so the gen-scenes gate and
    the render-time falsification evaluate one formula, for the AC-24 reason the
    T60s do.
    """
    if t60_s <= 0.0:
        raise ValueError(f"t60_s must be positive; got {t60_s!r}.")
    if coefficient_s <= 0.0:
        raise ValueError(f"coefficient_s must be positive; got {coefficient_s!r}.")
    return coefficient_s * t60_s**exponent


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


def room_constant(surface_m2: float, absorption: float) -> float:
    """Room constant R = Sα/(1-α) (m²) — the diffuse-field "absorbing power".

    Shared by the scene report and the scaffold for the AC-24 reason: the
    reverberant level the report PUBLISHES and the reverberant level the scaffold
    RENDERS have to come from one formula, or the described room and the rendered
    room diverge with nothing recording it (RD-75).

    CLOSED-ENCLOSURE MODEL. R is defined for a room whose reverberant field is
    sustained by its own boundaries; it is meaningless for the roadmap's outdoor
    and partially-open scenes (paper §6). Callers must not apply it to a geometry
    family that does not declare `characterization: sabine` — see
    `GeometryFamily.characterization` (RD-64).
    """
    if not 0.0 < absorption < 1.0:
        raise ValueError(
            f"absorption must be in (0, 1) for the room constant; got {absorption!r}. "
            f"α ≥ 1 gives an infinite R and α ≤ 0 a non-positive one, neither of "
            f"which describes an enclosure."
        )
    if surface_m2 <= 0:
        raise ValueError(f"surface must be positive; got {surface_m2!r}.")
    return surface_m2 * absorption / (1.0 - absorption)


def critical_distance(surface_m2: float, absorption: float) -> float:
    """Critical distance r_c = sqrt(R/16π) (m): where the direct and reverberant
    fields are equal. Same closed-enclosure caveat as `room_constant`."""
    return math.sqrt(room_constant(surface_m2, absorption) / (16.0 * math.pi))


def diffuse_field_drr_db(surface_m2: float, absorption: float, distance_m: float) -> float:
    """Diffuse-field direct-to-reverberant ratio (dB) at `distance_m`.

    Direct 1/(4πd²) against the reverberant field 4/R, so
    DRR = 10·log10(R / (16π d²)) — which is 0 dB exactly at d = r_c, by
    construction. Same closed-enclosure caveat as `room_constant`.
    """
    if distance_m <= 0:
        raise ValueError(
            f"distance must be > 0 for a DRR; got {distance_m!r}. A coincident "
            f"source/receiver pair has no finite direct-to-reverberant ratio."
        )
    r = room_constant(surface_m2, absorption)
    return 10.0 * math.log10(r / (16.0 * math.pi * distance_m ** 2))


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
