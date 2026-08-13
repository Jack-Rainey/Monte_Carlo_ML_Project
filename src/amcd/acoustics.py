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

#: Speed of sound in air at 20 °C, in m/s. DECLARED HERE and derived from, never
#: alongside, `SABINE_K` — three different speeds used to coexist in this project
#: (343.24 implied by a rounded Sabine constant, 343.0 in the scaffold, 344.0
#: declared by the gsound backend) and only one pair was ever disclosed
#: (AC-109/AC-150).
#:
#: A backend may realize a DIFFERENT speed — gsound computes
#: `getAirSpeedOfSound(20 °C, 101.325 kPa, 50 % RH)` = 344.0 and declares it in its
#: own config, where it is cross-checked against the rendered paths. That is a
#: property of the renderer. This constant is the one the STUDY's closed forms use,
#: and the two are allowed to differ as long as both are stated.
SPEED_OF_SOUND_M_S = 343.0

#: Sabine's constant, exactly 24·ln(10)/c. COMPUTED, not rounded: the shipped
#: literal was 0.161, which implies c = 343.2425 rather than the 343.0 declared
#: above, and that 0.07 % discrepancy then propagated into every d_min corner as a
#: constant −0.035 % offset that had to be documented instead of removed.
SABINE_K = 24.0 * math.log(10.0) / SPEED_OF_SOUND_M_S


#: Absorption is clipped to this OPEN interval before any closed form uses it.
#: Sabine, Eyring, the room constant and the critical distance all divide by alpha
#: or by (1-alpha), so the endpoints are singular rather than merely extreme.
#:
#: ONE interval, shared, because the report and the scaffold used different ones —
#: `1e-6` here and `0.01` in `dry_run` — so above alpha 0.99 the room the report
#: DESCRIBED and the room the scaffold RENDERED were different rooms, with nothing
#: recording it (AC-41). That is the divergence this module exists to prevent, on a
#: third quantity after the T60s and the speed of sound.
ALPHA_CLIP = (1e-6, 1.0 - 1e-6)


def clip_absorption(absorption: float) -> tuple[float, bool]:
    """`absorption` inside `ALPHA_CLIP`, and whether clipping changed it.

    Returns the flag rather than swallowing it: a clipped alpha means the value
    used is not the value declared, and every caller that publishes a number
    derived from it owes the reader that fact — the scaffold stamps
    `alpha_clipped` beside `rt60_clipped` for exactly this reason.
    """
    lo, hi = ALPHA_CLIP
    clipped = min(max(float(absorption), lo), hi)
    return clipped, clipped != float(absorption)


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


def predicted_support_s(
    reflection_depth: float,
    surface_m2: float,
    coefficient_s: float,
    depth_exponent: float,
    surface_exponent: float,
) -> float:
    """How many seconds of record a backend is predicted to produce:
    `coefficient_s * reflection_depth**depth_exponent * surface_m2**surface_exponent`.

    NOT A FUNCTION OF T60, AND THAT IS THE MEASUREMENT (AC-186). The predecessor of
    this formula was `c * T60**k`, fitted on renders whose rooms were all produced
    by scaling one shoebox, so room size and decay time moved together and either
    could appear to drive the record. A crossed probe separates them: holding
    geometry fixed and moving absorption across the declared `mixed` support —
    α 0.05 vs 0.80, a 16x change in T60 — moves realized support by **0.00 %,
    0.49 %, 0.00 %** at three room sizes. An energy trim closes the record when it
    runs out of PATHS, and how fast a room's decay dies does not change how many
    paths there are to trace.

    What does drive it, measured over the same probe:

    * `reflection_depth` — the backend's own bound on reflection order
      (gsound's `diffuse_depth`). At fixed geometry, support runs 0.9022 / 1.2392 /
      2.3403 s over depths 50 / 100 / 200, i.e. `depth**0.688`. The shipped
      predecessor had no depth term at all, so it was silently valid at exactly one
      value of a config knob that moves the answer 2.6x across its plausible range.
    * `surface_m2` — surface area, not volume. Adding volume to a depth+surface fit
      moves its exponent to +0.067 and the residual not at all, while surface alone
      carries +0.464. More surface is more reflecting area for a given order.

    The COEFFICIENTS are a backend fact, declared per simulator in config, because a
    different raytracer trims its record differently — and a backend that genuinely
    trims on decay says so by declaring the exponents its own probe measures.

    Lives here rather than in `simulators/gsound_sir.py` so the gen-scenes gate and
    the render-time falsification evaluate one formula, for the AC-24 reason the
    T60s do.
    """
    if reflection_depth <= 0.0:
        raise ValueError(f"reflection_depth must be positive; got {reflection_depth!r}.")
    if surface_m2 <= 0.0:
        raise ValueError(f"surface_m2 must be positive; got {surface_m2!r}.")
    if coefficient_s <= 0.0:
        raise ValueError(f"coefficient_s must be positive; got {coefficient_s!r}.")
    return coefficient_s * reflection_depth**depth_exponent * surface_m2**surface_exponent


#: c·SABINE_K — the only place either constant appears in the ISO 3382-1 §5.3
#: minimum measurement distance. Substituting either T60 leaves both inside this
#: product and cancels the volume, so d_min needs neither separately. Exactly
#: 24·ln10 by construction, since `SABINE_K` is computed from `SPEED_OF_SOUND_M_S`.
_C_TIMES_SABINE_K = 24.0 * math.log(10.0)


def min_measurement_distance(absorption: float, surface_m2: float,
                             characterization: str) -> float:
    """ISO 3382-1 §5.3 minimum measurement distance, d_min = 2·sqrt(V/(c·T60)) (m).

    VOLUME-INDEPENDENT, which is why it takes surface and absorption instead:
    substituting either T60 puts c·SABINE_K in the denominator and cancels V, so
    d_min reduces to 2·sqrt(αS/(c·K)) for Sabine and the same with −ln(1−α) for
    Eyring. Reported and counted per scene rather than enforced — the criterion
    varies with each scene's own absorption and surface while the config declares
    ONE global placement floor, so no single floor can satisfy it everywhere.

    Lives here rather than in `scenes/generator.py` for the reason `sabine_rt60`
    does: it is scene/statistical-model physics, and a second copy of a formula is
    how the described room and the rendered room drift apart (S-1).
    """
    if characterization == "sabine":
        absorption_area = absorption
    elif characterization == "eyring":
        absorption_area = -math.log(1.0 - absorption)
    else:
        raise ValueError(
            f"characterization must be 'sabine' or 'eyring'; got {characterization!r}. "
            f"Both are carried because Eyring is always the stricter criterion and "
            f"the spread between them is itself the disclosure."
        )
    return 2.0 * math.sqrt(absorption_area * surface_m2 / _C_TIMES_SABINE_K)


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
