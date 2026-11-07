"""perception/contour_peak.py -- CONTOUR_PEAK detections from field peaks.

Stage A2 of the front-end replacement. Stage A0/A1 built
`perception.contour_field` and validated it (deterministic, blank-safe,
persistence-passing). This module turns that field's peak list — peaks of the
continuous V1/V2 integration field, NOT discrete shock segments — into
`Detection` objects that downstream consumers will eventually read.

Additive: nothing here mutates the old front end, the old lenses, or the
pipeline, and nothing consumes CONTOUR_PEAK yet. A4 migrates consumers; this
stage only makes the detections exist.

WHAT THIS DOES NOT PROVE
------------------------
* That downstream lenses can read CONTOUR_PEAK. That is A4 work.
* That these signatures are discriminative. That is A3 (the bake-off).
* That the anchor scheme is final. It is caller-supplied; A4 reconciles it.

The anchor interface is deliberately pass-through: the caller owns the allocator
and the anchor_positions dict, because anchor identity is a pipeline-wide
decision (Position detections, border ownership and the spatial binders must
all agree on it) and A2 must not pre-empt it. Every slots bin below is stated
explicitly in this docstring so it can be re-derived or re-tuned without
reading the code.
"""
from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Tuple

import parameters as P
from core.registries import CONTOUR_PEAK_STRENGTH_Z_SCALE
from core.types import Detection, TypedSignature

K_PEAK = "CONTOUR_PEAK"
CLASS_PEAK = "CONTOUR_PEAK"
RELATION_PEAK = "CONTOUR_PROFILE"

# --- slot bin definitions (fixed, documented, tunable) --------------------
# The spans are MEASURED, not tuned: see the constants and their provenance
# in src/parameters/_text.py (CONTOUR_PEAK_STRENGTH_CLIP /
# CONTOUR_PEAK_COHERENCE_CLIP) and the measurement script
# diagnostics/_p0_peak_distributions.py (4200 field peaks over 16 real
# ETH-80 frames, canon 256, the production luminance).
#
# The old numbers (STRENGTH_CLIP = 20 with a claimed p50 of 17.0, and
# COHERENCE_CLIP = 0.15 with a claimed max of 0.147) came from a stale
# measurement (5 images, an older field formula). The measured COHERENCE
# distribution has p50 = 0.1527, so the old clip put 72 % of the peaks in
# the TOP bin: the slot carried one level for most of the population and
# the graded population bump sat on the axis' edge.
#
# The measured distributions (16 frames, 4200 peaks):
#   signed_curvature      min -0.523  p50 -0.001  max 0.521   (2643 distinct)
#   bend_angle            min  0.000  p50  0.189  max 7.218
#   integration_strength  min  0.019  p50  3.238  max 29.838
#   coherence             min  0.096  p50  0.153  max  0.492
# Bin edges are chosen from these observed ranges; anything outside the range
# is clipped to the nearest edge, which keeps the signature total.
N_CURV_BINS = 8
CURV_CLIP = 0.6          # |signed_curvature| observed max 0.521; clip at 0.6
N_BEND_BINS = 8
BEND_CLIP = 1.5          # observed p99 1.182; clip at 1.5 rather than the
                         # outlier max 7.2, so a single wild bend does not
                         # collapse the top bin's meaning
N_STRENGTH_BINS = 8
# [P1-6] the MEASURED span of the strength slot (see _text.py).
STRENGTH_CLIP = float(P.CONTOUR_PEAK_STRENGTH_CLIP)
N_COHERENCE_BINS = 8
# [P1-6] the MEASURED span of the coherence slot (see _text.py).
COHERENCE_CLIP = float(P.CONTOUR_PEAK_COHERENCE_CLIP)

Z_CLAMP_LO, Z_CLAMP_HI = 0.0, 1.0


def _clip_bin(value, clip, n_bins):
    """Map a float to a bin index in [0, n_bins-1] by clipping to a range."""
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        return 0
    v = float(value)
    if v < 0.0:
        v = 0.0
    if v > clip:
        v = clip
    q = int(math.floor(v / (clip / float(n_bins))))
    if q >= n_bins:
        q = n_bins - 1
    return q


def _num(value, default: float = 0.0) -> float:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return default


def _floor_bin(center: float, n_bins: int) -> int:
    q = int(math.floor(center))
    if q < 0:
        return 0
    return n_bins - 1 if q >= n_bins else q


def _unsigned_center(value, clip: float, n_bins: int) -> float:
    """Continuous bin-index centre for a non-negative graded quantity."""
    v = max(0.0, min(clip, _num(value)))
    return (v / clip) * n_bins


def _signed_center(value, clip: float, n_bins: int) -> float:
    """Continuous bin-index centre for a SIGNED graded quantity over
    [-clip, +clip]. The old `_clip_bin(v, clip, n)` mapped every negative
    curvature to bin 0 (sign discarded); this keeps the sign."""
    v = max(-clip, min(clip, _num(value)))
    return ((v + clip) / (2.0 * clip)) * n_bins


def _theta_center(value) -> float:
    th = _num(value) % math.pi
    return (th / math.pi) * 12.0


def _sf_center(pk: Dict) -> float:
    """[P1-8] The GRADED spatial frequency's position on the sf axis.

    The field knows the frequency in CYCLES PER PIXEL (the bank is
    geometrically spaced, so the continuous position of a frequency on the
    bank's own ladder is its position in LOG2 frequency); the slot's index
    (0..4) is only the lookup key. The peak dict carries that continuous
    position ("sf_axis_pos", computed by the field where the ladder is
    known); a band frequency lands exactly on its own bin, and a frequency
    between two bands lands between them.

    Falls back to the index when the peak carries no graded frequency (an
    older field), so the population is never mis-shaped.
    """
    pos = pk.get("sf_axis_pos")
    if isinstance(pos, (int, float)) and math.isfinite(float(pos)):
        return float(pos)
    return float(int(pk.get("sf_idx", 0)))


def _phase_center(value) -> float:
    p = max(-math.pi, min(math.pi, _num(value)))
    return (p + math.pi) / (2.0 * math.pi) * 4.0


def build_peak_population(pk: Dict) -> Tuple[Tuple[float, ...], ...]:
    """The graded population of one field peak: one activity vector per axis
    (core.registries.POPULATION_AXES['CONTOUR_PEAK']).

    Every axis carries the CONTINUOUS quantity the field measured, and the
    bins stay the lookup key:

      theta      the orientation (radians over [0, pi));
      sf         the GRADED spatial frequency's position on the bank's own
                 log-frequency ladder (cycles per pixel; see _sf_center);
      curvature  the SIGNED curvature;
      bend       the bend angle;
      phase      the Gabor phase;
      strength   the integration strength (over the MEASURED span);
      coherence  the coherence (over the MEASURED span).
    """
    from core.population import soft_bump_bins
    from core.registries import population_axis
    # [U22] Every axis is fetched BY NAME.  This class' layout carries one
    # axis the writer does not fill (the end-stop magnitude), so a positional
    # read (`axes[i]`) is not merely fragile: it silently pairs each
    # measurement with whichever axis the registry happens to list there.
    # The honesty gate is tests/perception/test_population_scramble_u22.py.
    _names = ("theta", "sf", "curvature", "bend", "phase", "strength",
              "coherence", "endstop")
    _axes = tuple(population_axis(CLASS_PEAK, n) for n in _names)
    if any(ax is None for ax in _axes):
        return ()
    ax_theta, ax_sf, ax_curv, ax_bend, ax_phase, ax_str, ax_coh, ax_end = _axes
    return (
        soft_bump_bins(_theta_center(pk.get("theta")), ax_theta),
        soft_bump_bins(_sf_center(pk), ax_sf),
        soft_bump_bins(_signed_center(pk.get("signed_curvature"),
                                      CURV_CLIP, N_CURV_BINS), ax_curv),
        soft_bump_bins(_unsigned_center(pk.get("bend_angle"),
                                        BEND_CLIP, N_BEND_BINS), ax_bend),
        soft_bump_bins(_phase_center(pk.get("phase")), ax_phase),
        soft_bump_bins(_unsigned_center(pk.get("integration_strength"),
                                        STRENGTH_CLIP, N_STRENGTH_BINS), ax_str),
        soft_bump_bins(_unsigned_center(pk.get("coherence"),
                                        COHERENCE_CLIP, N_COHERENCE_BINS), ax_coh),
        soft_bump_bins(_unsigned_center(pk.get("endstop_magnitude", 0.0),
                                        1.0, 8), ax_end),
    )


def peaks_to_contour_peak_detections(
    peaks: List[Dict],
    anchor_allocator: Callable[[int, int], Any],
    anchor_positions: Dict,
    extraction_scale: float,
    preserve_input_order: bool = False,
    t: int = 0,
) -> List[Detection]:
    """Field peak dicts -> one CONTOUR_PEAK Detection per peak.

    Anchor is supplied by `anchor_allocator(y, x)`; the caller owns the
    allocator and `anchor_positions`. `anchor_positions` is written here only so
    that later callers (e.g. Position detections) can share the same anchors.

    `extraction_scale` is accepted for signature compatibility with the other
    adapters and is NOT used: CONTOUR_PEAK carries no spatial extent, and the
    bin edges above are absolute rather than scale-relative so that a peak
    means the same thing at every image size.
    """
    # Chain membership refers to the original field-list index, not sorted order.
    ordered = list(peaks) if preserve_input_order else sorted(
        peaks,
        key=lambda p: (int(p.get("y", 0)), int(p.get("x", 0)),
                       float(p.get("theta", 0.0))),
    )

    out: List[Detection] = []
    for pk in ordered:
        y = int(pk.get("y", 0))
        x = int(pk.get("x", 0))
        anchor = anchor_allocator(y, x)
        if anchor is None:
            continue

        anchor_positions[anchor] = (float(x), float(y))

        theta_idx = int(pk.get("theta_idx", 0)) % 12
        sf_idx = int(pk.get("sf_idx", 0))
        phase_bin = int(pk.get("phase_bin", 0)) % 4
        # [P1-7] THE END-STOP SLOT IS AN INDEX, NOT A GRADIENT.
        #
        # The field's end-stop array is GRADED (the normalised |forward -
        # backward| of the terminator's channel); the slot's flag says only
        # "this terminator is above half its channel's asymmetry". The
        # graded magnitude is computed by the field and carried on the peak
        # ("endstop_magnitude"), so the flag is the INDEX of a measurement
        # the system holds, not a stand-in for one.
        #
        # The POPULATION cannot carry it yet: the class' population layout
        # (core.registries.POPULATION_AXES) declares one axis per graded
        # quantity and is owned by core/registries.py, which this mission
        # does not edit. Appending an axis from the emitter would (a) make
        # `population_for_bins("CONTOUR_PEAK", ...)` zip the axes onto the
        # bins in the WRONG order (the bins' order is theta, sf, curvature,
        # bend, phase, end-stop, strength, coherence -- the axes' order is
        # not that), and (b) silently disable the graded comparison for the
        # class (the layout arity would disagree). The measurement is
        # therefore carried on the peak and available to every reader of
        # the peak dict; the flag stays the index.
        endstop_flag = 1 if int(pk.get("endstop_flag", 0)) else 0
        # Curvature is a SIGNED axis over [-CURV_CLIP, +CURV_CLIP]; the centre is
        # continuous and the bin is its floor (population + index agree).
        curv_bin = _floor_bin(
            _signed_center(pk.get("signed_curvature"), CURV_CLIP, N_CURV_BINS),
            N_CURV_BINS)
        bend_bin = _floor_bin(
            _unsigned_center(pk.get("bend_angle"), BEND_CLIP, N_BEND_BINS),
            N_BEND_BINS)
        strength_bin = _floor_bin(
            _unsigned_center(pk.get("integration_strength"), STRENGTH_CLIP,
                             N_STRENGTH_BINS), N_STRENGTH_BINS)
        coherence_bin = _floor_bin(
            _unsigned_center(pk.get("coherence"), COHERENCE_CLIP,
                             N_COHERENCE_BINS), N_COHERENCE_BINS)

        quantized = (
            float(theta_idx),
            float(sf_idx),
            float(curv_bin),
            float(bend_bin),
            float(phase_bin),
            float(endstop_flag),
            float(strength_bin),
            float(coherence_bin),
        )

        sigma = TypedSignature(
            kind="CANONICAL",
            class_=CLASS_PEAK,
            relation=RELATION_PEAK,
            quantized=quantized,
            population=build_peak_population(pk),
        )

        # z is a CONFIDENCE in [0, 1]. The raw field value is NOT normalised:
        # measured over 2643 peaks from 5 real ETH-80 images it runs from
        # 0.029 (min) through 17.015 (p50) to 25.446 (max), so clamping it
        # to [0, 1] reported z = 1.0 for EVERY peak -- a constant, not a
        # confidence. The confidence is now the measured strength normalised
        # by the range the field actually produces:
        # core.registries.CONTOUR_PEAK_STRENGTH_Z_SCALE = 25.0, the measured
        # upper bound rounded down, so a peak at or above the observed
        # maximum saturates at z = 1.0 and the median peak (17.0) reports
        # z ~ 0.68.
        # The strength SLOT's own clip (STRENGTH_CLIP = 20.0) is untouched:
        # the bin edge is a question about the measurement's resolution, the
        # confidence is a question about how much evidence there is.
        strength = pk.get("integration_strength", 0.0)
        if not isinstance(strength, (int, float)) or \
                not math.isfinite(float(strength)):
            strength = 0.0
        z = float(min(max(strength / CONTOUR_PEAK_STRENGTH_Z_SCALE,
                          Z_CLAMP_LO), Z_CLAMP_HI))

        out.append(Detection(
            k=K_PEAK,
            source_layer=1,
            scale=0,
            sigma=sigma,
            alpha=anchor,
            z=z,
            t=t,
            rho=(("entity", anchor, 0, "+"),),
        ))
    return out


def peaks_to_position_detections(
    peaks: List[Dict],
    anchor_allocator: Callable[[int, int], Any],
    anchor_positions: Dict,
) -> List[Detection]:
    """One POSITION detection per peak, sharing the peak's anchor.

    Layout matched from perception/lenses/base.py:1495-1503 -- arity 3,
    `(x_bin, y_bin, z_bin)`, each binned to 16 slots over a [-1, 1] normalised
    frame by core/quantize.py:_q_signed:

        sigma = TypedSignature(kind="CANONICAL", class_="POSITION",
                               relation="OBJECT_CENTRIC_REL",
                               quantized=(float(x_b), float(y_b), float(z_b)))

    z_b is 7.5 -- the centre of the third slot's range, the same
    "information absent" value the pipeline uses for unbinned depth slots
    (see EDGE_OCCLUSION's stack_bin 8.0 convention). A peak carries no depth
    channel, and a fabricated 0.0 would assert flatness the field never
    measured.

    Frame positions are normalised over a 256-px field (centre 127.5), which is
    the size `field_from_luminance` was validated on. Callers working at other
    sizes must normalise before calling.
    """
    ordered = sorted(
        peaks,
        key=lambda p: (int(p.get("y", 0)), int(p.get("x", 0))),
    )

    out: List[Detection] = []
    for pk in ordered:
        y = float(pk.get("y", 0))
        x = float(pk.get("x", 0))
        anchor = anchor_allocator(int(round(y)), int(round(x)))
        if anchor is None:
            continue
        # [P1-9] ONE ANCHOR COORDINATE ORDER: (x, y).
        #
        # The CONTOUR_PEAK writer above (`peaks_to_contour_peak_detections`)
        # writes `anchor_positions[anchor] = (float(x), float(y))` -- the
        # LIVE order, and the one every reader of the anchor map assumes
        # (the position bins are normalised as x/127.5, y/127.5, and the
        # figure-ground tier's flank sampling reads the same map). This
        # writer used to write (y, x), so the two writers of the SAME
        # anchor disagreed on the order: a reader that trusted the map
        # would place the detection on the transposed pixel. The live
        # order wins; this one is corrected to match.
        anchor_positions[anchor] = (x, y)

        # _q_signed(v, 16): (clamp(v,-1,1) + 1) / 2 * 16, capped at 15.
        def _qs(v):
            c = max(-1.0, min(1.0, v))
            q = int(math.floor((c + 1.0) * 0.5 * 16.0))
            return 15 if q >= 16 else q

        sigma = TypedSignature(
            kind="CANONICAL",
            class_="POSITION",
            relation="OBJECT_CENTRIC_REL",
            quantized=(float(_qs(x / 127.5 - 1.0)),
                       float(_qs(y / 127.5 - 1.0)),
                       7.5),
        )
        out.append(Detection(
            k="Position",
            source_layer=0,
            scale=0,
            sigma=sigma,
            alpha=anchor,
            z=1.0,
            t=0,
            rho=(("entity", anchor, 0, "+"),),
        ))
    return out
