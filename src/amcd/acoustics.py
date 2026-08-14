"""Closed-form room acoustics shared by scene characterization and the scaffold.

One declaration each, shared by the scene report and the scaffold, so the T60
the report DESCRIBES and the T60 the scaffold RENDERS cannot diverge.

This is scene/statistical-model physics, NOT the ISO-3382 measurement path: the
metrics computed from rendered waveforms live in `evaluation/room_acoustic.py` and
must never be derived from these estimates (design_spec §3, metric source of truth).
"""
from __future__ import annotations

import math

#: Speed of sound in air at 20 °C, in m/s. `SABINE_K` is derived FROM this rather
#: than declared alongside it, so the two can never imply different speeds.
#:
#: A backend may realize a DIFFERENT speed — gsound computes
#: `getAirSpeedOfSound(20 °C, 101.325 kPa, 50 % RH)` = 344.0 and declares it in its
#: own config, where it is cross-checked against the rendered paths. That is a
#: property of the renderer. This constant is the one the STUDY's closed forms use,
#: and the two are allowed to differ as long as both are stated.
SPEED_OF_SOUND_M_S = 343.0

#: Sabine's constant, exactly 24·ln(10)/c. Computed, not rounded: the textbook
#: 0.161 implies c = 343.2425 rather than the 343.0 declared above.
SABINE_K = 24.0 * math.log(10.0) / SPEED_OF_SOUND_M_S


#: Absorption is clipped to this OPEN interval before any closed form uses it.
#: Sabine, Eyring, the room constant and the critical distance all divide by alpha
#: or by (1-alpha), so the endpoints are singular rather than merely extreme.
#:
#: One interval, shared by the report and the scaffold: two different clips would
#: describe and render different rooms at high alpha.
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

    Shared: `scenes/generator.py` characterizes the room from these two numbers
    and `simulators/gsound_sir.py` predicts its realized record length from them.
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

    NOT A FUNCTION OF T60, and that is a measurement rather than an assumption: a
    crossed probe holding geometry fixed while moving absorption over a 16x change
    in T60 moves realized support by under 0.5 %. An energy trim closes the record
    when it runs out of PATHS, and how fast a room's decay dies does not change how
    many paths there are to trace.

    What drives it instead is the backend's own bound on reflection order
    (gsound's `diffuse_depth`, ~depth^0.688) and the room's SURFACE AREA, not its
    volume (~S^0.464) — more surface is more reflecting area for a given order.
    See `scripts/support_law_probe.py`.

    The COEFFICIENTS are a backend fact, declared per simulator in config: a
    different raytracer trims its record differently, and one that genuinely trims
    on decay says so by declaring the exponents its own probe measures.

    Lives here rather than in `simulators/gsound_sir.py` so the gen-scenes gate and
    the render-time falsification evaluate one formula.
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
    does: a second copy of a formula is how the described room and the rendered
    room drift apart.
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
    α produces a T60 describing a different room than the one declared. Callers
    that need a defensive bound apply it themselves and record that they did.
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

    Shared by the scene report and the scaffold: the reverberant level published
    and the reverberant level rendered have to come from one formula.

    CLOSED-ENCLOSURE MODEL. R is defined for a room whose reverberant field is
    sustained by its own boundaries; it is meaningless for the roadmap's outdoor
    and partially-open scenes (paper §6). Callers must not apply it to a geometry
    family that does not declare `characterization: sabine` — see
    `GeometryFamily.characterization`.
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
    diffuse-field model's comfortable domain.
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
