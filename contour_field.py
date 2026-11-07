"""perception/contour_field.py — Continuous V1/V2/V4 contour field.

Replaces the shock-graph segmenter (perception/contours.py). Peaks of a
continuous integration field are the primitives; no region partition,
no medial axis, no segments.

Authoritative spec: master plan v17/v18, Steps A0 and A1.

This module is standalone. It returns numpy arrays and a list of peak
dicts. It does not emit Detection objects, does not touch the graph,
does not import from binder / mapper / cores / distributor / the old
contours module. Peak-to-Detection conversion is Stage 2.

Determinism
-----------
- float32 throughout; Python floats are cast to np.float32 before any
  array arithmetic.
- integer-pixel shifts only; `int(round(k * cos(theta)))` is IEEE-754
  deterministic on the same input.
- sorted iteration for every tie-break.
- direct convolution (scipy.ndimage.convolve, mode="reflect") for
  kernels at or below FFT_KERNEL_THRESHOLD; fftconvolve otherwise. BLAS
  threading is capped at the project's standard single-worker setting
  (enforced at the top of perception.filters, inherited here).

Runtime note
------------
The beam integration is a shift-and-sum over integer offsets along
each orientation. O(n_theta * n_sf * K) numpy slice operations, K =
int(3 * wavelength). No dense O(N^2) pair loop. Measured runtime is
reported by the diagnostics block at the bottom of this file.

Documented approximations
-------------------------
1. AF angular tolerance (|theta_p - theta_q| < 0.35 rad) is applied
   by treating each orientation channel independently. A fractional
   cross-channel blend is a future refinement: the +-1 index gap is
   pi/12 = 15 deg, which falls INSIDE the ~20 deg tolerance, so a naive
   blend at neighbouring channels would over-count collinear support.
2. Cocircular support (plan: circle through two orientations with
   radius in [5, 200], tangent error < 0.15) is approximated here by a
   single wide isotropic Gaussian of the summed complex field. The
   full pair computation is O(N^2) over orientation pairs and is not
   implemented.
3. V4 curvature in this module is the directional derivative of the
   dominant-orientation field along its perpendicular. Menger
   curvature on the ridge polyline (A1) is the definitive measure and
   is computed at peak positions after ridge extraction.
4. Peak descriptor `phase_bin` is 4-bin (plan: 4 bins). Ridge peak
   spacing is 1.0 * median_wavelength (plan: 1.0).
5. Peak gating is relative-AND-absolute. The relative term keeps the
   top of the field on real images; the absolute term
   (PEAK_MIN_INTEGRATION_ABS) exists because a relative threshold
   alone normalises noise against itself and lets a structure-free
   frame through.
"""

from __future__ import annotations

import hashlib
import math
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.ndimage import convolve as nd_convolve
from scipy.ndimage import maximum_filter
from scipy.signal import fftconvolve

import parameters as P
from perception.filters import GaborBank


# =============================================================================
# Parameters
# =============================================================================

def _p(name: str, default):
    return getattr(P, name, default)


_FRONT_END_REF_DIAGONAL_PX = _p("FRONT_END_REF_DIAGONAL_PX",
                                362.0386719678744)

# [A5#6] The bank parameters are captured ONCE, here, and used by both the
# field fingerprint and the bank check. Reading four of them live through
# `getattr(P, ...)` in the fingerprint while building the sfs ladder from
# import-time constants made the fingerprint internally inconsistent.
_GABOR_SIGMA_X_FRAC = float(_p("GABOR_SIGMA_X_FRAC", 0.0))
_GABOR_ASPECT = float(_p("GABOR_ASPECT", 0.0))
_GABOR_KERNEL_SIZE_FROM_MAX_SIGMA = bool(
    _p("GABOR_KERNEL_SIZE_FROM_MAX_SIGMA", False))
_GABOR_KERNEL_MAX = int(_p("GABOR_KERNEL_MAX", 0))
_GABOR_N_THETA = int(_p("GABOR_N_THETA", 12))
_GABOR_N_SF = int(_p("GABOR_N_SF", 5))
_GABOR_SF_MIN = float(_p("GABOR_SF_MIN", 0.06))
_GABOR_SF_MAX = float(_p("GABOR_SF_MAX", 0.35))

_DIVISIVE_SIGMA = float(_p("DIVISIVE_SIGMA", 0.1))
_DIVISIVE_POOL_SIGMA = float(_p("DIVISIVE_POOL_SIGMA", 2.0))
# [U5.1] Rule-derived: pool radius covers 3 sigma of the divisive pool
# (99.7% of the Gaussian mass), i.e. 3 * DIVISIVE_POOL_SIGMA. The previous
# value 2.0 truncated the pool at 1 sigma; the parameters.py value (2.0) is
# retained for backward compat but overridden here by rule.
_DIVISIVE_POOL_RADIUS = 3.0 * _DIVISIVE_POOL_SIGMA

_AF_LONGITUDINAL_MULT = float(_p("AF_LONGITUDINAL_MULT", 3.0))
# [ASSOCIATION FIELD] The beam is now a small COCIRCULAR fan, not a single
# same-channel ray: AF_PERP_FRAC sets the lateral tolerance (it fixes the fan
# half-angle via asin(AF_PERP_FRAC / AF_LONGITUDINAL_MULT)), AF_SIGMA_THETA is
# the relatability width (how much bend a fan step tolerates), and
# AF_MAX_DELTA_THETA caps the orientation change a single fan step may span.
# The fan reads the RAW channel at the circle-predicted orientation (no global
# orientation smoothing, which would bleed onto straight edges). See _bent_beam.
_AF_PERP_FRAC = float(_p("AF_PERP_FRAC", 0.3))
_AF_MAX_DELTA_THETA = float(_p("AF_MAX_DELTA_THETA", 0.35))
_AF_SIGMA_THETA = float(_p("AF_SIGMA_THETA", 0.15))
# [AF RE-DERIVATION, Round 6] The association field is MODULATORY. The bent
# (off-tangent) cocircular term no longer ADDS to integration; it multiplies
# the collinear driving energy: integration = driving * (1 + AF_GAIN *
# facil/(driving+facil)). Zero local energy therefore stays zero, and the
# link-support statistic (which reads the DRIVING field) is unaffected by
# facilitation. AF_GAIN is a fixed bound on amplification (not tuned).
_AF_GAIN = float(_p("AF_GAIN", 1.0))

_COCIRCULAR_WIDE_SIGMA_FRAC = float(_p("COCIRCULAR_WIDE_SIGMA_FRAC", 2.0))
_COCIRCULAR_WEIGHT = float(_p("COCIRCULAR_WEIGHT", 0.25))
# [U4 / CF-2] Directed-traversal floor. The undirected orientation field is
# pi-periodic (theta in [0, pi)), so the finite-difference tangent points into
# the upper half-plane at every pixel and the sign of d(theta)/ds flips twice
# around any closed contour. The tangent's SIGN is therefore taken from the
# luminance gradient -- at an edge the gradient is perpendicular to it and
# points consistently inward or outward over a whole closed contour, so
# orienting the tangent left-of-gradient gives one traversal direction and
# kappa keeps one sign. The gradient direction is used only where its
# magnitude clears this fraction of the frame's 99th-percentile |grad|; in a
# flat region there is no edge to reference and the undirected convention is
# kept (curvature is undefined there anyway).
_CURV_EPS_FRAC = float(0.6)  # [U25 F2] behaviour: eps=3.18 at wl=5.3; reduces residual curvature noise on population-coded ramps below the probe's 2*dtheta bar
_CURV_GRAD_FLOOR_FRAC = float(_p("CURV_GRAD_FLOOR_FRAC", 0.05))
# [ROUND 10 / B] Formula version. Bump when the field formula changes so the
# observation-attached cache can never return a field computed by an older
# formula. v2: (a) the wide context is a MODULATORY gain, not an additive term;
# (b) the beam steps are sub-pixel bilinear with equal (unit) arc length.
# v3 [U4/CF-2]: signed curvature uses a luminance-gradient-referenced tangent,
# so one sign is kept around a closed contour.
# v4 [U4/CF-3]: bent beam follows the cocircular arc (tangent rotates per step)
# instead of a straight chord along psi.
# v5 [U7/CF-1]: symmetric channel interpolation (fan offsets +1, -1) replaces
# the one-sided round() indexing; mirror oracle max |delta| = 0.022 (was 1.267).
# CF-1 FIXED; psychophysics gate ADMISSIBLE under the redefined (mirror-
# symmetric) gate: d'5 3.664, d'9 4.598, arc gain 1.379.
# v6 [U25]: five arithmetic defects, each asserted by diagnostics/_u25_probe.py.
# (a) F1 the bent-beam fan DISTRIBUTES its total turn (the fan's full width
#     2*joff) over the K steps; turning by joff every step made a K-step beam
#     end K*joff off the reference (measured: the chord of a 15-step beam
#     turned 0.701 rad where the cocircular relation predicts 0.200).
# (b) F2 the curvature derivative interpolates the DOUBLED-angle unit vector
#     and differentiates in that representation, so a window straddling the pi
#     seam cannot read ~pi/2 between 3.13 rad and 0.01 rad (measured 0.191
#     px^-1 on a 0.020 px^-1 ramp).
# (c) F3 end-stop candidates rank by an ENERGY-WEIGHTED mean bounded by 1; the
#     old plain sum over the 60 (theta, sf) channels peaked at 39.0 on a 96 px
#     test field, so the `v > 0.5` gate admitted almost everything.
# (d) F4 the union NMS re-ranks every candidate by the integration energy at
#     its pixel; the three sources had been compared in incomparable units
#     (ridge energy 2.55 vs end-stop sum 30.28 at the same pixel).
# (e) F5 the extraction margin is capped at half the discardable amount, so a
#     long-wavelength bank can never erase the whole interior.
# v7 [U25 AUDIT]: the formula is unchanged; what changed is the arithmetic
# around it and the validation of it --
#   * the border LEAK GATE is deleted (it was STILL LIVE at default 1 even
#     though the v6 comment said it was removed), so the extraction margin is a
#     hard discard again and `_interior_margin` is its one cap;
#   * kappa divides by the ACTUAL sample separation instead of 2*eps (np.clip
#     shortened it at the frame edge), and the doubled-angle samples go through
#     `_sample_bilinear`;
#   * the curvature floor is a median over the pixels that can become peaks,
#     not over the whole frame;
#   * the peak cell is one argmax over (theta, sf) (lowest theta wins a tie);
#   * the divisive pool radius is ceil(3*sigma) and never 0.
_FIELD_FORMULA_VERSION = 7

_PEAK_RIDGE_SPACING_FRAC = float(_p("PEAK_RIDGE_SPACING_FRAC", 1.0))
# [FIX 6] perpendicular suppression half-width as a fraction of the median
# wavelength. Two contours closer than this perpendicular to the tangent
# are treated as the same contour; further apart they are parallel
# structure and both survive.
_SPACING_PERP_FRAC = float(_p("PEAK_SPACING_PERP_FRAC", 0.3))
# [U25 C10] The two perpendicular NMS offsets, as fractions of the reference
# wavelength; fingerprinted so a change can never reuse a cached field.
_RIDGE_NMS_EPS_FRACS = (1.0 / 6.9, 2.0 / 6.9)
_PEAK_MIN_INTEGRATION_FRAC = float(_p("PEAK_MIN_INTEGRATION_FRAC", 0.15))
# [BLANK-FRAME FIX] Absolute floor on summed integration. The relative
# threshold above normalises noise against itself, so a structure-free
# frame still passes it.
# [ROUND 10 / B] Re-derived after the wide-context term became MODULATORY
# (v2) and the beam steps became sub-pixel (equal arc length). Synthetic
# corpus, 256^2: blank grey 1.7e-4, blank white 3.4e-4, structured (arc+bar)
# 6.37 (was 19.9 with the additive term). Rule unchanged (sit above the
# blank-frame interior max, near the blank/real geometric mean):
# sqrt(3.4e-4 * 6.37) = 0.047 -> 0.05. Verified: blank grey/white emit 0
# peaks, the structured frame keeps its peaks.
# [U26 RE-DERIVATION, v7 field] The v6 calibration above does NOT reproduce
# on the shipped `_FIELD_FORMULA_VERSION = 7` field (U25 excised the border
# gate, made the curvature denominator `hypot(x_fwd-x_bwd, y_fwd-y_bwd)` and
# moved the curvature floor to `I >= min_integration`).  Measured on v7, 256^2
# (`diagnostics/_round10_B_rederive.py`):
#     blank grey 1.72e-4,  blank white 3.52e-4,  structured 6.515 (32 peaks)
# geometric mean(blank white, structured) = sqrt(3.517e-4 * 6.515) = 0.048 --
# i.e. the round-10/B value.  Rule unchanged (sit above the blank-frame
# interior max, near the blank/real geometric mean); the v6-era 0.0066 sat
# 7.3x below the measured geometric mean.
# Verified: blank grey/white emit 0 peaks, the structured frame keeps its
# peaks; the blank-frame guard test stays green; the shift binds only below
# I_max = 0.32 (the relative term governs above it), which no U24-corpus or
# figground frame reaches -- it only hardens the structure-free case.
_PEAK_MIN_INTEGRATION_ABS = float(_p("PEAK_MIN_INTEGRATION_ABS", 0.048))
# [BORDER-MARGIN FIX] minimum fraction of each axis that must remain inside
# the artifact-band discard. Caps the margin on small frames / long-
# wavelength banks, where kernel_support + beam_reach would remove most of
# the image. 0.5 = keep at least half of each axis.
_FIELD_MIN_INTERIOR_FRAC = float(_p("FIELD_MIN_INTERIOR_FRAC", 0.5))
# [U25 F5] The border pass is a HARD DISCARD.  The U25 leak gate that kept band
# candidates with a luminance edge at their pixel is REMOVED: it did close the
# field spine's 121 px gap, but the recovered border peaks let
# `t_junction_order` fabricate an interrupted edge on the faded box scenes and
# three figground pins failed (test_faded_scenes_emit_no_wrong_edge x2,
# test_true_junction_on_calibration_scene_is_zero_wrong).  What replaces it is
# the margin CAP -- the actual arithmetic defect, see _interior_margin.
_ENDSTOP_MAX_RADIUS_FRAC = float(_p("ENDSTOP_MAX_RADIUS_FRAC", 2.0))
_CURVATURE_MAX_RADIUS_FRAC = float(_p("CURVATURE_MAX_RADIUS_FRAC", 2.0))

# [U25 C6] The `min(..., 7)` cap was a leftover from a debug session: it
# forced every production kernel (29-85 px) onto the FFT path, so
# `_convolve_same`'s direct branch could never run and the two branches were
# never compared.
_FFT_KERNEL_THRESHOLD = int(_p("FFT_KERNEL_THRESHOLD", 31))  # FFT for 29+ px
_PADDING_TEST_IMAGE_SIZE = 128
# [O5] Value-preserving parallelism. 1 disables the pool (byte-identical).
_FIELD_THREADS = max(1, int(_p("FIELD_THREADS", 4)))


def field_fingerprint() -> Tuple:
    """Constants that determine field_from_luminance's output.

    Folded into the observation-attached field cache key so a flag/constant
    change -- or a different input luminance (e.g. grayscale vs colour) -- can
    never return a stale cached field.
    """
    return (
        _FIELD_FORMULA_VERSION,
        _FRONT_END_REF_DIAGONAL_PX,
        _GABOR_N_THETA, _GABOR_N_SF, _GABOR_SF_MIN, _GABOR_SF_MAX,
        _GABOR_SIGMA_X_FRAC, _GABOR_ASPECT,
        _GABOR_KERNEL_SIZE_FROM_MAX_SIGMA, _GABOR_KERNEL_MAX,
        _DIVISIVE_SIGMA, _DIVISIVE_POOL_RADIUS, _DIVISIVE_POOL_SIGMA,
        _AF_LONGITUDINAL_MULT, _AF_PERP_FRAC, _AF_MAX_DELTA_THETA,
        _AF_SIGMA_THETA, _AF_GAIN,
        _COCIRCULAR_WIDE_SIGMA_FRAC, _COCIRCULAR_WEIGHT,
        _CURV_GRAD_FLOOR_FRAC,
        _CURV_EPS_FRAC,
        _PEAK_RIDGE_SPACING_FRAC, _SPACING_PERP_FRAC,
        _PEAK_MIN_INTEGRATION_FRAC, _PEAK_MIN_INTEGRATION_ABS,
        _FIELD_MIN_INTERIOR_FRAC, _ENDSTOP_MAX_RADIUS_FRAC,
        _CURVATURE_MAX_RADIUS_FRAC, _FFT_KERNEL_THRESHOLD,
        _PEAK_BORDER_MARGIN_PX, _RIDGE_NMS_EPS_FRACS,
        _FIELD_THREADS,
    )


# [BUG-03] Gabor bank cache keyed by (scale, parameter fingerprint)
_GABOR_BANK_CACHE: Dict[Tuple, GaborBank] = {}


def _check_field_bank(bank) -> None:
    """[A5#6] Verify the bank this module is about to read.

    `field_from_luminance` builds the spatial-frequency ladder from THIS
    module's constants and indexes it by the bank's channel axis, so a bank
    built with a different (n_theta, n_sf) would silently assign every
    wavelength to the wrong channel.  A non-square or even kernel would also
    break `_convolve_same`'s centred "same" contract.
    """
    nt, ns, nph = (int(v) for v in bank.kernels.shape[:3])
    kh, kw = int(bank.kernels.shape[-2]), int(bank.kernels.shape[-1])
    if (nt, ns) != (_GABOR_N_THETA, _GABOR_N_SF) or nph != 2:
        raise ValueError(
            "bank (n_theta, n_sf, n_phase)=%r does not match the module's "
            "(%d, %d, 2); the sfs ladder would be mis-assigned"
            % ((nt, ns, nph), _GABOR_N_THETA, _GABOR_N_SF))
    if kh != kw or kh % 2 == 0:
        raise ValueError(
            "bank kernels must be odd and square, got %dx%d" % (kh, kw))


def _gabor_bank_fingerprint() -> Tuple:
    """Constants that determine GaborBank.build_default.

    Part of the bank cache key so an edit (or env override) of any of these
    can never silently reuse a stale kernel set for the same scale.
    """
    # [U25 C9] the module's OWN constants (the ones this module builds with),
    # not a second, independent read of the parameter table.
    return (
        _GABOR_N_THETA,
        _GABOR_N_SF,
        _GABOR_SF_MIN,
        _GABOR_SF_MAX,
        _GABOR_SIGMA_X_FRAC, _GABOR_ASPECT,
        _GABOR_KERNEL_SIZE_FROM_MAX_SIGMA, _GABOR_KERNEL_MAX,
    )

# [O5] One-entry cache for the raw V1 triple. Observation2D.from_image already
# runs the SAME bank on the SAME adapted luminance (compute_responses is
# byte-identical to v1_responses), so the field stage can reuse it instead of
# recomputing 120 convolutions. One entry bounds memory (~61 MB at 256^2); a
# miss is a correct, silent recomputation, never a stale value.
_V1_RESPONSE_CACHE: Dict[Tuple, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}


def _v1_cache_key(lum: np.ndarray, scale: float) -> Tuple:
    digest = hashlib.blake2b(
        np.ascontiguousarray(lum).tobytes(), digest_size=16
    ).digest()
    return (lum.shape, digest, float(scale), _gabor_bank_fingerprint())


def cache_v1_responses(
    lum: np.ndarray,
    scale: float,
    simple: np.ndarray,
    complex_: np.ndarray,
    phase: np.ndarray,
) -> None:
    """Store a raw V1 triple for reuse by v1_responses (value-preserving).

    Callers must pass exactly the arrays produced by compute_responses(bank, lum)
    for the same (lum, scale, parameter fingerprint). The arrays are expected to
    be read-only downstream; from_image stores complex_ on the observation and
    nothing mutates the triple in place.
    """
    if lum.dtype != np.float32:
        return
    _V1_RESPONSE_CACHE.clear()
    _V1_RESPONSE_CACHE[_v1_cache_key(lum, scale)] = (simple, complex_, phase)


# [BUG-04] Minimum peak-extraction margin; increase for wider kernels.
_PEAK_BORDER_MARGIN_PX = 16


# =============================================================================
# Front-end scale
# =============================================================================

def front_end_scale(h: int, w: int) -> float:
    """Image-diagonal ratio versus the reference. 1.0 at 256x256."""
    return math.hypot(float(h), float(w)) / max(
        _FRONT_END_REF_DIAGONAL_PX, 1e-6
    )


# =============================================================================
# V1: Gabor responses and divisive normalization
# =============================================================================

def _convolve_same(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Reflect-padded same-size convolution.

    Direct for small kernels (deterministic); fftconvolve for larger
    kernels, assuming single-threaded BLAS (set by the project
    convention at perception.filters import time).
    """
    # [U25 C6] A centred "same" result exists only for ODD sides; an even
    # kernel silently shifts the response by half a pixel.
    _kh, _kw = int(kernel.shape[0]), int(kernel.shape[1])
    if _kh % 2 == 0 or _kw % 2 == 0:
        raise ValueError(
            "convolution kernel sides must be odd for a centred 'same' "
            "result, got %dx%d" % (_kh, _kw))
    # [A5#7] both sides decide the branch, and each side gets its own pad.
    if max(_kh, _kw) <= _FFT_KERNEL_THRESHOLD:
        return nd_convolve(img, kernel, mode="reflect").astype(np.float32)
    # [FIX 3] scipy.signal.fftconvolve has no padding-mode argument, so
    # mode="same" ZERO-pads -- making large-kernel and small-kernel
    # responses non-comparable at the border. Pad manually so both paths
    # reflect-pad identically.
    ph, pw = _kh // 2, _kw // 2
    padded = np.pad(img, ((ph, ph), (pw, pw)), mode="symmetric")
    return fftconvolve(padded, kernel, mode="valid").astype(np.float32)


def _border_margin_px(bank) -> int:
    """Suppress padding artifacts at extraction, not in convolution.

    [A5#5] The "path-aware" split was false: scipy's ``reflect`` IS numpy's
    ``symmetric``, so BOTH branches mirror a band of ``k // 2`` into the frame
    and both need the same margin.  With the old split, raising
    FFT_KERNEL_THRESHOLD above the kernel size would have left a 16 px margin
    under a 42 px artifact band.
    """
    k = int(bank.kernels.shape[-1])
    return max(_PEAK_BORDER_MARGIN_PX, k // 2)


def _interior_margin(border_margin_px: int, h: int, w: int) -> int:
    """[U25 F5] The margin actually applied, capped so an interior SURVIVES.

    The old cap was `_FIELD_MIN_INTERIOR_FRAC * min(h, w)`, which is the size
    of the interior it is supposed to protect -- at the cap the margin equals
    the kept region and the band discards the WHOLE frame (`m <= y < h - m` is
    empty).  The discardable amount is `(1 - _FIELD_MIN_INTERIOR_FRAC)` of the
    short side, so a margin may consume at most HALF of it:

        cap = 0.5 * (1 - _FIELD_MIN_INTERIOR_FRAC) * min(h, w)

    With the shipped 0.5 fraction that is 25% of the short side (48 px at
    192^2, where the production bank asks for 42 px, so unchanged there).  It
    is what keeps a small frame with a long-wavelength bank from emitting no
    peaks at all.
    """
    raw = max(0, int(border_margin_px))
    cap = int(0.5 * (1.0 - _FIELD_MIN_INTERIOR_FRAC) * min(int(h), int(w)))
    return max(0, min(raw, cap))


def _get_bank(scale: float, bank: Optional[GaborBank] = None):
    """[U25 C8] The Gabor bank for `scale`; caches the DEFAULT bank only."""
    if bank is not None:
        return bank
    # [BUG-03] Cache Gabor bank by (scale, parameter fingerprint)
    scale_key = (float(scale), _gabor_bank_fingerprint())
    cached = _GABOR_BANK_CACHE.get(scale_key)
    if cached is None:
        cached = GaborBank.build_default(scale=scale)
        _GABOR_BANK_CACHE[scale_key] = cached
    return cached


def v1_responses(
    lum: np.ndarray,
    scale: float = 1.0,
    bank: Optional[GaborBank] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """V1 simple, complex, and phase responses.

    simple  : (n_theta, n_sf, 2, H, W) float32 — even and odd
    complex : (n_theta, n_sf, H, W)    float32 — sqrt(even^2 + odd^2)
    phase   : (n_theta, n_sf, H, W)    float32 — atan2(odd, even)
    """
    # [U25 C8] One bank helper, and the raw-response cache is only valid for
    # the DEFAULT bank: its key is (luminance, scale, parameter fingerprint) and
    # carries no identity for a caller-supplied bank, so caching a custom bank's
    # triple would serve the default bank's response for the same luminance.
    default_bank = bank is None
    bank = _get_bank(scale, bank)
    # [O5] Reuse the raw triple the observation front end already computed.
    if default_bank and lum.dtype == np.float32:
        hit = _V1_RESPONSE_CACHE.get(_v1_cache_key(lum, scale))
        if hit is not None:
            return hit
    n_theta, n_sf, n_phase = bank.kernels.shape[:3]
    H, W = lum.shape
    simple = np.zeros((n_theta, n_sf, n_phase, H, W), dtype=np.float32)
    tasks = [
        (it, isf, ip)
        for it in range(n_theta)
        for isf in range(n_sf)
        for ip in range(n_phase)
    ]
    workers = min(_FIELD_THREADS, len(tasks)) if tasks else 1
    if workers > 1:
        def _convolve_one(task: Tuple[int, int, int]) -> None:
            simple[task] = _convolve_same(lum, bank.kernels[task])
        with ThreadPoolExecutor(max_workers=workers) as ex:
            list(ex.map(_convolve_one, tasks))
    else:
        for task in tasks:
            simple[task] = _convolve_same(lum, bank.kernels[task])
    complex_ = np.sqrt(
        simple[:, :, 0].astype(np.float32) ** 2
        + simple[:, :, 1].astype(np.float32) ** 2
    )
    phase = np.arctan2(simple[:, :, 1], simple[:, :, 0])
    out = (simple, complex_.astype(np.float32), phase.astype(np.float32))
    if default_bank and lum.dtype == np.float32:
        _V1_RESPONSE_CACHE.clear()
        _V1_RESPONSE_CACHE[_v1_cache_key(lum, scale)] = out
    return out


def _gaussian_kernel_1d(radius: int, sigma: float) -> np.ndarray:
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    k = np.exp(-(x ** 2) / (2.0 * np.float32(sigma) ** 2))
    k /= (k.sum() + np.float32(1e-12))
    return k.astype(np.float32)


def _blur_sep(img: np.ndarray, radius: int, sigma: float) -> np.ndarray:
    """Separable Gaussian blur with reflect padding."""
    k = _gaussian_kernel_1d(radius, sigma)
    H, W = img.shape
    pad = np.pad(img, ((0, 0), (radius, radius)), mode="reflect")
    out = np.zeros_like(img)
    for i in range(k.shape[0]):
        w = k[i]
        out += w * pad[:, i:i + W]
    pad = np.pad(out, ((radius, radius), (0, 0)), mode="reflect")
    out2 = np.zeros_like(out)
    for i in range(k.shape[0]):
        w = k[i]
        out2 += w * pad[i:i + H, :]
    return out2.astype(np.float32)


def divisive_normalize(
    complex_: np.ndarray,
    sigma: Optional[float] = None,
    pool_radius: Optional[float] = None,
    pool_sigma: Optional[float] = None,
    scale: float = 1.0,
) -> np.ndarray:
    """Divisive normalization: r / (sigma + pooled_summed_activity).

    Pool is a Gaussian blur of the sum over (theta, sf) of the complex
    magnitude, then broadcast back over the channel dimension. This
    follows the V1 convention that the divisor is the local total
    activity across channels, not the same channel alone. A channel-
    wise anisotropic pool is a future refinement.
    """
    sig = _DIVISIVE_SIGMA if sigma is None else float(sigma)
    pr = _DIVISIVE_POOL_RADIUS if pool_radius is None else float(pool_radius)
    ps = (_DIVISIVE_POOL_SIGMA if pool_sigma is None else float(pool_sigma)) * max(float(scale), 1e-6)
    # [FIX 5] the pool radius is a spatial scale, so it must follow the
    # image diagonal rather than being an absolute pixel count.
    # [A5#8] ceil, not round: `round(3 * ps)` truncates the pool below its own
    # 3-sigma reach (scale 0.7 -> ps 1.4 -> r 4 instead of 5) and at a very
    # small scale rounded to 0, i.e. no blur at all.
    if pool_radius is None:
        r = max(1, int(math.ceil(3.0 * ps)))
    else:
        r = max(1, int(math.ceil(pr * max(float(scale), 1e-6))))
    summed = complex_.sum(axis=(0, 1)).astype(np.float32)
    pooled = _blur_sep(summed, r, ps)
    denom = np.float32(sig) + pooled[None, None, :, :]
    return (complex_ / denom).astype(np.float32)


# =============================================================================
# V2: integration field and end-stop field
# =============================================================================

def _accumulate_shift(
    target: np.ndarray,
    source: np.ndarray,
    dx: int,
    dy: int,
    w: np.float32,
) -> None:
    """target[y+dy, x+dx] += w * source[y, x], in-place, zero-fill."""
    H, W = source.shape
    ty0 = max(0, dy); ty1 = min(H, H + dy)
    tx0 = max(0, dx); tx1 = min(W, W + dx)
    if ty0 >= ty1 or tx0 >= tx1:
        return
    sy0 = ty0 - dy; sy1 = ty1 - dy
    sx0 = tx0 - dx; sx1 = tx1 - dx
    target[ty0:ty1, tx0:tx1] += w * source[sy0:sy1, sx0:sx1]


def _accumulate_shift_subpixel(
    target: np.ndarray,
    source: np.ndarray,
    fx: float,
    fy: float,
    w: float,
) -> None:
    """target += w * source shifted by the FRACTIONAL (fx, fy), bilinear.

    [ROUND 10 / B] The beam steps advance one pixel of ARC LENGTH along the
    path, i.e. by (k*cos, k*sin). Rounding those to integer pixel shifts
    samples diagonal paths differently from cardinal ones (up to 0.5 px per
    step, and the rounded arc length is not constant), which is a source of
    the diagonal/cardinal gain anisotropy. Sampling on the exact sub-pixel
    position with bilinear interpolation makes every orientation share the
    same unit-arc-length sampling.
    """
    x0 = math.floor(fx)
    y0 = math.floor(fy)
    ax = fx - x0
    ay = fy - y0
    ix0 = int(x0)
    iy0 = int(y0)
    if ax == 0.0 and ay == 0.0:
        _accumulate_shift(target, source, ix0, iy0, np.float32(w))
        return
    # [A5#15] cardinal steps have ax == 0 or ay == 0, so half of the taps add
    # exactly zero. Skipping them is value-identical and halves the slice work.
    for _dx, _dy, _wt in (
        (0, 0, (1.0 - ax) * (1.0 - ay)),
        (1, 0, ax * (1.0 - ay)),
        (0, 1, (1.0 - ax) * ay),
        (1, 1, ax * ay),
    ):
        if _wt != 0.0:
            _accumulate_shift(target, source, ix0 + _dx, iy0 + _dy,
                              np.float32(w * _wt))


def _channel_theta(it, n_theta):
    """The angle of channel ``it`` -- the BANK's own convention.

    ``filters.py`` builds channel ``it`` at ``it * pi / n_theta`` and publishes
    ``bank.theta = [k * pi / n_theta]``.  Every index->angle conversion must
    use the same map; the half-bin form ``(it + 0.5) * pi / n_theta`` rotates
    every orientation the front end reports by half a bin (7.5 deg at
    n_theta = 12) and every integration beam marches in the wrong direction.
    Accepts a scalar or an array and returns the same shape.
    """
    scale = math.pi / float(n_theta)
    if isinstance(it, np.ndarray):
        return np.asarray(it, dtype=np.float64) * scale
    return float(it) * scale


def _af_fan_half_angle() -> float:
    """Fan half-angle from the lateral tolerance: over the beam reach
    (AF_LONGITUDINAL_MULT wavelengths) the path may deviate laterally by
    AF_PERP_FRAC wavelengths, so sin(half_angle) = AF_PERP_FRAC /
    AF_LONGITUDINAL_MULT."""
    ratio = _AF_PERP_FRAC / max(_AF_LONGITUDINAL_MULT, 1e-9)
    return math.asin(min(0.999, max(0.0, ratio)))


def _collinear_beam(
    target: np.ndarray,
    channel: np.ndarray,
    theta_ref: float,
    wavelength: float,
    direction: int,
) -> float:
    """Accumulate the COLLINEAR (driving) beam into `target`.

    The k = 0 self term (raw channel) plus Gaussian-weighted integer shifts of
    the same channel along the reference axis, forward (direction=+1) or
    backward (-1). This is exactly the pre-association-field central ray and
    defines the mass scale: returns the sum of the central weights for the
    caller's mass normalisation. The off-tangent cocircular fan is separate
    (see _bent_beam) so it can be applied modulatory, not additively.
    """
    K = int(round(_AF_LONGITUDINAL_MULT * wavelength))
    sigma_along = _AF_LONGITUDINAL_MULT * wavelength / 2.0

    target += channel  # k = 0 self term (raw channel)
    weight_total = 1.0
    if K < 1:
        return float(weight_total)

    two_sigma_sq = 2.0 * sigma_along * sigma_along
    base = theta_ref if direction > 0 else (theta_ref + math.pi)
    ct = math.cos(base)
    st = math.sin(base)
    for k in range(1, K + 1):
        w = math.exp(-(k * k) / two_sigma_sq)
        weight_total += w
        _accumulate_shift_subpixel(target, channel, k * ct, k * st, w)
    return float(weight_total)


def _bent_beam(
    target: np.ndarray,
    scale_channels: np.ndarray,
    theta_ref: float,
    wavelength: float,
    direction: int,
    n_theta: int,
) -> None:
    """Accumulate the off-tangent COCIRCULAR (facilitation) fan into `target`.

    For each arc step k along a small fan of path directions psi around the
    reference axis theta_ref, the circle through the origin tangent to
    theta_ref predicts the neighbour's orientation by the alternate-segment
    theorem: theta_pred = 2*psi - theta_ref (the angle between the tangent at
    the origin and the chord equals the angle between the chord and the tangent
    at the neighbour). The neighbour's energy is read from the RAW channel
    nearest theta_pred, so a straight edge lying along theta_ref contributes
    nothing off-tangent (no cross-orientation bleed) while a bent contour, whose
    energy genuinely rotates into the neighbouring channel, is facilitated.
    Each bend direction is weighted by its relatability,
    exp(-0.5 * (2*joff / AF_SIGMA_THETA)^2). This term carries NO mass scale of
    its own; the caller normalises it by the collinear weight total and applies
    it modulatory. direction = +1 forward, -1 backward (psi += pi).
    """
    K = int(round(_AF_LONGITUDINAL_MULT * wavelength))
    sigma_along = _AF_LONGITUDINAL_MULT * wavelength / 2.0
    half = _af_fan_half_angle()
    # One bend each side, only if a single step stays within
    # AF_MAX_DELTA_THETA. [B] The comparison was INVERTED: it refused
    # the fan when its full width was SMALLER than the limit, so a NARROW
    # (safe) fan was refused and a WIDE (over-counting) one was allowed.
    # With the shipped constants the derived fan (2*half = 0.2003 rad) is
    # inside the 0.35 rad limit, so the cocircular facilitation now
    # actually contributes -- which is what the psychophysics arc-gain
    # gate (ARC_GAIN_MIN) measures.
    if ((2.0 * half) > (_AF_MAX_DELTA_THETA + 1e-9)):
        return
    if K < 1:
        return

    two_sigma_sq = 2.0 * sigma_along * sigma_along
    base = theta_ref if direction > 0 else (theta_ref + math.pi)
    for joff in (half, -half):
        psi = base + joff
        theta_pred = 2.0 * psi - theta_ref
        # [U6.3 / CF-1] Symmetric interpolation between the two nearest channel
        # centres (primary fix). round() gave a one-sided index (fan offsets
        # +1, 0); interpolation blends the two neighbours so both bends get a
        # share of the cocircular and collinear channels. u is the continuous
        # channel coordinate on the pi-periodic axis; i0 = floor(u); f = u-i0.
        u = ((theta_pred % math.pi) / math.pi) * n_theta
        i0 = int(math.floor(u))
        f = u - i0
        i0m = i0 % n_theta
        i1m = (i0 + 1) % n_theta
        if abs(f) < 1e-9:
            src = scale_channels[i0m]
        elif abs(f - 1.0) < 1e-9:
            src = scale_channels[i1m]
        else:
            src = (1.0 - f) * scale_channels[i0m] + f * scale_channels[i1m]
        if np.asarray(src).max() <= 0.0:
            continue
        relat = math.exp(
            -0.5 * ((2.0 * joff) / max(_AF_SIGMA_THETA, 1e-9)) ** 2)
# [U4.5 / CF-3] Curve the beam along the cocircular arc instead of
        # a straight chord: the tangent rotates by joff per step, so the
        # spatial position is the cumulative sum of unit steps at rotating
        # angles. Reduces to the straight ray when joff == 0.
        # [U25 F1] DISTRIBUTE the total turn. The old `mid = psi + (k-0.5)*joff`
        # turned by joff on EVERY step, so a K-step beam ended K*joff off the
        # reference instead of on the predicted neighbour orientation: the
        # circle through the fan came out as a spiral. The beam must end at
        # theta_pred = 2*psi - theta_ref, i.e. rotate by the fan's full width
        # 2*joff over the whole reach.
        dtheta = 2.0 * joff / float(K)
        px, py = 0.0, 0.0
        for k in range(1, K + 1):
            mid = base + (k - 0.5) * dtheta
            px += math.cos(mid)
            py += math.sin(mid)
            w = math.exp(-(k * k) / two_sigma_sq)
            _accumulate_shift_subpixel(target, src, px, py, w * relat)


def integration_field(
    complex_: np.ndarray,
    sfs: np.ndarray,
    cocircular_wide: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """V2 integration, end-stop and driving fields.

    Returns
    -------
    integration : (n_theta, n_sf, H, W) float32
        MODULATORY association-field output: driving * (1 + AF_GAIN *
        facil/(driving+facil)). Facilitation amplifies but never creates local
        evidence, so a channel with zero driving energy stays zero.
    endstop     : (n_theta, n_sf, H, W) float32
        |forward - backward| normalized by forward + backward (driving only).
    driving     : (n_theta, n_sf, H, W) float32
        Pre-facilitation collinear support (forward + backward central ray),
        MODULATED by the optional wide-context gain (driving * (1 +
        COCIRCULAR_WEIGHT * wide/(driving+wide))). The link-support rule reads
        THIS field so its statistic is independent of the association field.
    """
    n_theta, n_sf, H, W = complex_.shape
    fwd = np.zeros_like(complex_)
    bwd = np.zeros_like(complex_)
    facil = np.zeros_like(complex_)

    def _integrate_one(it: int) -> None:
        theta = _channel_theta(it, n_theta)
        for isf in range(n_sf):
            ch = complex_[it, isf]
            # [BUG-03] Short-circuit zero-energy channels
            if ch.max() <= 0.0:
                continue
            sf = float(sfs[isf])
            wavelength = 1.0 / max(sf, 1e-6)
            scale_ch = complex_[:, isf]  # raw channels, for the cocircular read
            wt = _collinear_beam(fwd[it, isf], ch, theta, wavelength, +1)
            _collinear_beam(bwd[it, isf], ch, theta, wavelength, -1)
            _bent_beam(facil[it, isf], scale_ch, theta, wavelength, +1, n_theta)
            _bent_beam(facil[it, isf], scale_ch, theta, wavelength, -1, n_theta)
            # [FIX 4] mass-normalise each channel so coarse and fine
            # channels share one scale. Facilitation uses the SAME scale so it
            # is comparable to driving.
            wtn = np.float32(max(wt, 1e-9))
            fwd[it, isf] /= wtn
            bwd[it, isf] /= wtn
            facil[it, isf] /= wtn

    workers = min(_FIELD_THREADS, n_theta) if n_theta > 0 else 1
    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            list(ex.map(_integrate_one, range(n_theta)))
    else:
        for it in range(n_theta):
            _integrate_one(it)

    driving = (fwd + bwd).astype(np.float32)
    if cocircular_wide is not None:
        # [ROUND 10 / B] The wide-context term is MODULATORY, not additive.
        # The old form (driving + W*cocircular_wide) was orientation-agnostic
        # and created energy in empty channels, which then fed link support.
        # Here the wide context is a relative gain in [0, 1): it amplifies
        # existing driving but a channel with zero driving stays exactly zero.
        wide = np.asarray(cocircular_wide, dtype=np.float32)
        wide_norm = (
            wide / (driving + wide + np.float32(1e-6))
        ).astype(np.float32)
        driving = (
            driving * (np.float32(1.0) + np.float32(_COCIRCULAR_WEIGHT) * wide_norm)
        ).astype(np.float32)

    # [AF RE-DERIVATION, Round 6] Modulatory association field. facil/(driving+
    # facil) is the relative facilitation in [0, 1); driving = 0 forces 0.
    denom = (driving + facil + np.float32(1e-6)).astype(np.float32)
    facil_norm = (facil / denom).astype(np.float32)
    integration = (
        driving * (np.float32(1.0) + np.float32(_AF_GAIN) * facil_norm)
    ).astype(np.float32)

    denom_es = (fwd + bwd + np.float32(1e-6)).astype(np.float32)
    endstop = (np.abs(fwd - bwd) / denom_es).astype(np.float32)
    return integration, endstop, driving


def cocircular_context(
    complex_: np.ndarray, wavelength_mean: float,
) -> np.ndarray:
    """Broad-context support: wide Gaussian blur of the summed complex.

    Documented approximation of the plan's cocircular support. The
    full pair computation (circle through two orientations with
    radius in [5, 200] and tangent error < 0.15) is O(N^2) over
    orientation pairs and is not implemented.

    Returns shape (1, 1, H, W); callers broadcast.
    """
    summed = complex_.sum(axis=(0, 1)).astype(np.float32)
    sigma = _COCIRCULAR_WIDE_SIGMA_FRAC * wavelength_mean
    r = int(math.ceil(3.0 * sigma))
    wide = _blur_sep(summed, r, sigma)
    return wide[None, None, :, :].astype(np.float32)


# =============================================================================
# V4: continuous curvature
# =============================================================================

def _continuous_orientation(integration: np.ndarray) -> np.ndarray:
    """Population-code continuous orientation from the integration field.

    Returns (H, W) float32 in [0, pi). Orientation is pi-periodic, so the
    circular mean is computed in doubled-angle space.
    """
    n_theta = integration.shape[0]
    summed = integration.sum(axis=1).astype(np.float32)
    thetas = _channel_theta(np.arange(n_theta, dtype=np.float64),
                            n_theta).astype(np.float32)
    c = np.cos(2.0 * thetas).astype(np.float32)
    sn = np.sin(2.0 * thetas).astype(np.float32)
    cos_sum = (summed * c[:, None, None]).sum(axis=0)
    sin_sum = (summed * sn[:, None, None]).sum(axis=0)
    two_theta = np.arctan2(sin_sum, cos_sum)
    theta = 0.5 * two_theta
    theta = np.where(theta < 0.0, theta + math.pi, theta)
    return theta.astype(np.float32)


def _directed_tangent(
    tan_x: np.ndarray, tan_y: np.ndarray,
    lum: np.ndarray, wavelength_mean: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Give the undirected tangent a sign from the luminance gradient.

    The orientation field is pi-periodic, so `(cos theta, sin theta)` always
    points into the upper half-plane and the traversal direction it defines
    reverses twice around any closed contour. At an edge the smoothed
    luminance gradient is perpendicular to the contour and points
    consistently inward or outward over the whole closed contour, so choosing
    the tangent sign that puts the gradient on its LEFT gives one consistent
    traversal direction there.

    `sign = +1` where `cross(tangent, grad) > 0`, `-1` where it is negative,
    and the undirected tangent where the gradient magnitude is below
    `_CURV_GRAD_FLOOR_FRAC` of the frame's 99th-percentile gradient (no edge
    contrast to reference).
    """
    H, W = lum.shape
    sigma = max(float(wavelength_mean), 1e-6)
    radius = max(1, int(math.ceil(3.0 * sigma)))
    sm = _blur_sep(np.asarray(lum, dtype=np.float32), radius, sigma)
    # [A5#18] np.gradient: identical in the interior, ONE-SIDED at the border.
    # The old explicit slice left the first/last row and column at exactly
    # zero gradient, so the sign flipped to +1 there and kappa's sign could
    # reverse along the frame edge.
    gy, gx = np.gradient(sm)
    mag = np.hypot(gx, gy)
    floor = float(np.percentile(mag, 99.0)) * _CURV_GRAD_FLOOR_FRAC
    dot = tan_x * gy - tan_y * gx          # cross(tangent, gradient)_z
    sign = np.where(dot < 0.0, np.float32(-1.0), np.float32(1.0))
    sign = np.where(mag > np.float32(floor), sign, np.float32(1.0))
    return (tan_x * sign).astype(np.float32), (tan_y * sign).astype(np.float32)


def curvature_field(
    integration: np.ndarray,
    wavelength_mean: float,
    lum: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Signed curvature and bend angle.

    [FIX 1] kappa = d(theta)/ds where s is arc length ALONG THE TANGENT --
    the derivative of the continuous orientation field in its own tangent
    direction. The previous version took the derivative PERPENDICULAR to
    the tangent, which is ~0 for a circle and exactly 0 for a line.

    [U4 / CF-2] The tangent is DIRECTED by the luminance gradient when the
    luminance is supplied (see `_directed_tangent`): `theta` is pi-periodic,
    so an undirected tangent flips the sign of d(theta)/ds twice around a
    closed contour, and two ends of the same occluded silhouette read
    opposite curvature signs. With the gradient reference the sign is that
    of the evolute side, constant around the contour. When `lum` is None the
    undirected convention is kept (magnitude only).

    Continuous (population-code) orientation replaces the argmax bin, so
    the finite difference is not quantised to 15-degree steps. [FIX 4/5]
    bend_angle uses wavelength_mean supplied by the caller from the actual
    sfs array rather than an unscaled constant.
    """
    # [A5#3] eps == 0 when wavelength_mean <= 0 would divide by zero, and a
    # different-shaped `lum` would silently mis-register the gradient.
    if not (float(wavelength_mean) > 0.0
            and math.isfinite(float(wavelength_mean))):
        raise ValueError("wavelength_mean must be positive and finite, got %r"
                         % (wavelength_mean,))
    H, W = integration.shape[2], integration.shape[3]
    if lum is not None:
        lum = np.asarray(lum, dtype=np.float32)
        if lum.shape != (H, W):
            raise ValueError("lum shape %r != integration plane %r"
                             % (lum.shape, (H, W)))
    theta_cont = _continuous_orientation(integration)

    tan_x = np.cos(theta_cont)
    tan_y = np.sin(theta_cont)
    if lum is not None:
        tan_x, tan_y = _directed_tangent(
            tan_x, tan_y, np.asarray(lum, dtype=np.float32), wavelength_mean)
    eps = np.float32(_CURV_EPS_FRAC * wavelength_mean)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    x_fwd = np.clip(xx + eps * tan_x, 0.0, np.float32(W - 1))
    y_fwd = np.clip(yy + eps * tan_y, 0.0, np.float32(H - 1))
    x_bwd = np.clip(xx - eps * tan_x, 0.0, np.float32(W - 1))
    y_bwd = np.clip(yy - eps * tan_y, 0.0, np.float32(H - 1))
    # [A5#3] the nested copy of _sample_bilinear is gone (same arithmetic).
    # [U25 F2] Interpolate the DOUBLED-angle UNIT VECTOR, never the raw angle.
    # theta is pi-periodic but stored in [0, pi), so a bilinear window that
    # straddles the seam averages 3.13 rad with 0.01 rad into ~1.57 rad -- a
    # direction orthogonal to BOTH samples -- and the derivative at the seam
    # read 0.191 px^-1 on a 0.020 px^-1 ramp.  (cos 2t, sin 2t) is continuous
    # across the seam and the difference is then exact:
    #     2*d_theta = atan2(s2f*c2b - c2f*s2b, c2f*c2b + s2f*s2b)
    c2 = np.cos(2.0 * theta_cont).astype(np.float32)
    s2 = np.sin(2.0 * theta_cont).astype(np.float32)
    c2f = _sample_bilinear(c2, x_fwd, y_fwd)
    s2f = _sample_bilinear(s2, x_fwd, y_fwd)
    c2b = _sample_bilinear(c2, x_bwd, y_bwd)
    s2b = _sample_bilinear(s2, x_bwd, y_bwd)
    d_theta = np.arctan2(s2f * c2b - c2f * s2b,
                          c2f * c2b + s2f * s2b) / 2.0
    # [A5#3] divide by the ACTUAL separation of the two samples, not 2*eps:
    # np.clip shortens it at the frame edge and the old denominator under-read
    # curvature there.
    sep = np.hypot(x_fwd - x_bwd, y_fwd - y_bwd).astype(np.float32)
    ok = sep > np.float32(1e-3)
    kappa = np.where(ok, d_theta / np.maximum(sep, np.float32(1e-3)),
                     np.float32(0.0)).astype(np.float32)
    bend = (np.abs(kappa) * np.float32(2.0 * wavelength_mean)).astype(
        np.float32
    )
    return kappa, bend

def _sample_bilinear(
    field: np.ndarray, x: np.ndarray, y: np.ndarray,
) -> np.ndarray:
    H, W = field.shape
    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, W - 1)
    y1 = np.clip(y0 + 1, 0, H - 1)
    x0c = np.clip(x0, 0, W - 1)
    y0c = np.clip(y0, 0, H - 1)
    wx = (x - x0).astype(np.float32)
    wy = (y - y0).astype(np.float32)
    v00 = field[y0c, x0c]
    v10 = field[y0c, x1]
    v01 = field[y1, x0c]
    v11 = field[y1, x1]
    return (
        v00 * (1.0 - wx) * (1.0 - wy)
        + v10 * wx * (1.0 - wy)
        + v01 * (1.0 - wx) * wy
        + v11 * wx * wy
    ).astype(np.float32)


def _median_wavelength(sfs) -> float:
    """[A5#14] The reference wavelength of a spatial-frequency ladder.

    One reader for every caller: float64 (the float32 median differed in the
    last bits from the float64 one), list-tolerant, and floored so a zero
    frequency cannot divide by zero.
    """
    return float(np.median(
        1.0 / np.maximum(np.asarray(sfs, dtype=np.float64), 1e-6)))


def ridge_nms(
    integration: np.ndarray, min_integration: float,
    wavelength_mean: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Binary mask of ridge pixels and the summed intensity.

    Ridge pixel iff I(x, y) strictly exceeds both perpendicular
    samples at each of eps = 1 and 2 pixels, and exceeds the
    integration floor. Ties are rejected.
    """
    n_theta, n_sf, H, W = integration.shape
    I = integration.sum(axis=(0, 1)).astype(np.float32)
    flat = integration.reshape(n_theta * n_sf, H, W)
    flat_idx = flat.argmax(axis=0)
    theta_idx = (flat_idx // n_sf).astype(np.float32)
    theta_rad = _channel_theta(
        theta_idx, n_theta).astype(np.float32)
    perp_x = -np.sin(theta_rad)
    perp_y = np.cos(theta_rad)

    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    mask = I > np.float32(min_integration)
    if wavelength_mean is None or wavelength_mean <= 0.0:
        distances = (1.0, 2.0)
    else:
        distances = tuple(f * wavelength_mean
                          for f in _RIDGE_NMS_EPS_FRACS)
    for distance in distances:
        eps = np.float32(distance)
        I_p = _sample_bilinear(I, xx + eps * perp_x, yy + eps * perp_y)
        I_m = _sample_bilinear(I, xx - eps * perp_x, yy - eps * perp_y)
        mask &= (I > I_p) & (I > I_m)
    return mask, I


def _local_maxima_2d(
    field: np.ndarray, radius: int,
) -> List[Tuple[int, int, float]]:
    """Local maxima of `field` with the given NMS radius.

    Deterministic: maximum_filter is single-threaded; ties broken by
    sorted (y, x) order.
    """
    size = 2 * max(1, radius) + 1
    maxf = maximum_filter(field, size=size, mode="reflect")
    # [A5#11] on a FLAT field every pixel ties for the maximum, so a blank
    # frame built (and sorted) the whole frame as candidates. Zero is not a
    # peak of anything.
    ys, xs = np.where((field >= maxf) & (field > 0.0))
    pairs = sorted(
        ((int(y), int(x), float(field[y, x]))
         for y, x in zip(ys.tolist(), xs.tolist())),
        key=lambda t: (-t[2], t[0], t[1]),
    )
    return pairs


def _directional_nms(
    samples: List[Tuple[float, int, int, float]],
    spacing_along: float,
    spacing_perp: float,
) -> List[Tuple[float, int, int, float]]:
    """[FIX 6] Suppress a sample only if an accepted one lies within
    spacing_along ALONG its tangent AND within spacing_perp perpendicular
    to it. Strength-ordered isotropic NMS lets the first of two parallel
    contours kill the second; this spares parallel structure.

    samples: (value, y, x, theta). Caller sorts by (-value, y, x, theta).

    [SPEED] The scan over the ACCEPTED list is pruned by a uniform grid, so
    the routine is O(n * k) instead of O(n^2) on a dense field.

    VALUE IDENTITY: the pruning is exact, not approximate. The suppression
    test requires |along| <= spacing_along AND |perp| <= spacing_perp, and
    (along, perp) is a rotation of (dx, dy), so it also requires
    |dx|, |dy| <= hypot(spacing_along, spacing_perp) = CELL. Two points
    whose coordinates differ by at most CELL therefore always lie in
    ADJACENT grid cells, so the 3x3 cell neighbourhood of a candidate
    contains every point that could possibly suppress it. Points outside it
    cannot suppress and are not examined. The set of accepted samples, and
    their order, are therefore exactly what the unpruned scan produced.
    """
    accepted: List[Tuple[float, int, int, float]] = []
    sa2 = spacing_along * spacing_along
    sp2 = spacing_perp * spacing_perp
    cell = math.hypot(spacing_along, spacing_perp)
    if not (cell > 0.0):
        cell = 1e-9
    grid: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
    for v, y, x, th in samples:
        tx = math.cos(th)
        ty = math.sin(th)
        ok = True
        gy = int(math.floor(y / cell))
        gx = int(math.floor(x / cell))
        for cy in (gy - 1, gy, gy + 1):
            for cx in (gx - 1, gx, gx + 1):
                bucket = grid.get((cy, cx))
                if not bucket:
                    continue
                for ay, ax in bucket:
                    dx = x - ax
                    dy = y - ay
                    along = dx * tx + dy * ty
                    perp = -dx * ty + dy * tx
                    if along * along <= sa2 and perp * perp <= sp2:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break
        if ok:
            accepted.append((v, y, x, th))
            grid.setdefault((gy, gx), []).append((y, x))
    return accepted



def _dominant_theta_rad(integration: np.ndarray, y: int, x: int) -> float:
    """Dominant orientation in radians at one pixel (channel argmax).

    integration is indexed [theta, sf, y, x], so the pixel slice is
    (n_theta, n_sf) -- argmax over theta first, then sf.
    """
    cell = integration[:, :, y, x]        # (n_theta, n_sf)
    it = int(np.argmax(cell.max(axis=1)))
    return _channel_theta(it, integration.shape[0])


def extract_peaks(
    integration: np.ndarray,
    endstop: np.ndarray,
    curvature_signed: np.ndarray,
    bend_angle: np.ndarray,
    complex_: np.ndarray,
    phase: np.ndarray,
    sfs: np.ndarray,
    border_margin_px: int = 0,
    lum: Optional[np.ndarray] = None,   # accepted for API compatibility; unused
) -> List[Dict]:
    """Peak extraction per A1.

    Union of ridge samples (greedy NMS at 1.0 * median_wavelength),
    end-stop maxima, and |curvature| maxima.

    Each peak is a dict with position, dominant theta and sf,
    signed curvature, bend angle, phase, end-stop flag, integration
    strength, coherence.

    [U25 F5] The border pass is a HARD DISCARD again (the U25 leak gate is
    removed, see the constants block), and the margin is capped by
    ``_interior_margin`` so the discard can never consume the frame.
    """
    n_theta, n_sf, H, W = integration.shape

    # [U25 F5] tolerance for synthetic (n-D) curvature planes: reduce to 2-D.
    # [A5#13] `max` destroyed the SIGN (-0.5 and +0.1 -> +0.1); the reduction
    # keeps the largest-MAGNITUDE value, sign included.
    _cs = np.asarray(curvature_signed)
    if _cs.ndim > 2:
        _flat = _cs.reshape((-1,) + _cs.shape[-2:])
        _pick = np.abs(_flat).argmax(axis=0)
        curvature_signed = np.take_along_axis(_flat, _pick[None], axis=0)[0]
    curvature_signed = np.asarray(curvature_signed)
    _ba = np.asarray(bend_angle)
    if _ba.ndim > 2:
        bend_angle = _ba.max(axis=tuple(range(_ba.ndim - 2)))
    bend_angle = np.asarray(bend_angle)

    # [A5#1] The border leak gate is GONE, not merely re-commented: it let a
    # band candidate survive on |grad L| alone, and the recovered border peaks
    # made t_junction_order fabricate an interrupted edge on the faded scenes
    # (three figground pins failed). The border pass is a hard discard.
    # [A5#12] ... and the margin is computed ONCE, so `m == 0` makes the three
    # filters below a no-op instead of repeating the same block behind three
    # redundant guards.
    m = _interior_margin(border_margin_px, H, W)

    def _inside_interior(t) -> bool:
        """True when `t` is outside the reflect-padding artifact band.

        `>=` the margin on both axes: a pixel exactly on the boundary stays.
        """
        return m <= t[1] < H - m and m <= t[2] < W - m

    # [A5#14] ONE wavelength reader (float64, list-safe) for every caller.
    median_wavelength = wavelength_mean = _median_wavelength(sfs)
    # [P1-7/P1-8] The bank's own spatial-frequency ladder is GEOMETRIC
    # (perception/filters.py: geomspace(GABOR_SF_MIN/s, GABOR_SF_MAX/s,
    # GABOR_N_SF)), so the continuous position of a frequency on that ladder
    # is its position in LOG2 frequency. The ladder's ends are the axis's
    # first and last bin CENTRES, so a frequency that IS a band lands
    # exactly on its own bin and a frequency between two bands lands
    # between them (the graded quantity the index throws away).
    sf_log = np.log2(np.maximum(np.asarray(sfs, dtype=np.float64), 1e-9))
    sf_log_lo = float(sf_log[0]) if n_sf > 0 else 0.0
    sf_log_hi = float(sf_log[-1]) if n_sf > 0 else 0.0
    sf_log_span = (sf_log_hi - sf_log_lo) or 1.0
    peak_spacing = _PEAK_RIDGE_SPACING_FRAC * median_wavelength
    endstop_radius = int(
        round(_ENDSTOP_MAX_RADIUS_FRAC * median_wavelength)
    )
    curv_radius = int(
        round(_CURVATURE_MAX_RADIUS_FRAC * median_wavelength)
    )

    I = integration.sum(axis=(0, 1)).astype(np.float32)
    I_max = float(I.max()) if I.size else 0.0
    # [BLANK-FRAME FIX] relative AND absolute. The relative term alone
    # normalises noise against itself; the absolute term keeps a
    # structure-free frame out (0 peaks).
    min_integration = max(
        _PEAK_MIN_INTEGRATION_FRAC * I_max,
        _PEAK_MIN_INTEGRATION_ABS,
    )
    mask, _ = ridge_nms(integration, min_integration,
                        wavelength_mean=wavelength_mean)

    ys, xs = np.where(mask)
    ridge_pixels: List[Tuple[float, int, int]] = [
        (float(I[y, x]), int(y), int(x))
        for y, x in zip(ys.tolist(), xs.tolist())
    ]
    ridge_pixels.sort(key=lambda t: (-t[0], t[1], t[2]))
    # [BORDER-MARGIN FIX] drop the reflect-padding artifact band. Uses
    # >= so a pixel exactly on the margin boundary is kept.
    ridge_pixels = [t for t in ridge_pixels if _inside_interior(t)]
    # [FIX 6] direction-aware NMS: parallel contours within one
    # wavelength no longer suppress each other.
    _ridge_dn = [
        (v, int(y), int(x), _dominant_theta_rad(integration, int(y), int(x)))
        for (v, y, x) in ridge_pixels
    ]
    _ridge_dn.sort(key=lambda t: (-t[0], t[1], t[2], t[3]))
    ridge_samples = [
        (v, y, x)
        for (v, y, x, _th) in _directional_nms(
            _ridge_dn, peak_spacing,
            _SPACING_PERP_FRAC * median_wavelength)
    ]

    # [U25 F3] ENERGY-WEIGHTED MEAN, not a sum over the (theta, sf) channels.
    # The sum let a channel with end-stop 1.0 and almost no integration outvote
    # a strong one (measured: the candidate map peaked at 39.0 on a 96 px test
    # field, so `v > 0.5` could not discriminate). The weighted mean is bounded
    # by 1 and measures how terminated the energy at this pixel actually is.
    _es_num = (endstop * integration).sum(axis=(0, 1)).astype(np.float32)
    _es_den = (integration.sum(axis=(0, 1))
               + np.float32(1e-6)).astype(np.float32)
    endstop_sum = (_es_num / _es_den).astype(np.float32)
    # [BLANK-FRAME FIX] an end-stop candidate must sit on real integration
    # energy. Without this, float noise fed through the beam sum produces an
    # end-stop signature on a structure-free frame and `v > 0.5` passes.
    endstop_candidates = [
        (v, y, x)
        for (y, x, v) in _local_maxima_2d(endstop_sum, endstop_radius)
        if v > 0.5 and I[y, x] >= min_integration
    ]
    # [BORDER-MARGIN FIX] same discard for the end-stop candidate set
    endstop_candidates = [t for t in endstop_candidates
                          if _inside_interior(t)]
    _es_dn = [
        (v, int(y), int(x), _dominant_theta_rad(integration, int(y), int(x)))
        for (v, y, x) in endstop_candidates
    ]
    _es_dn.sort(key=lambda t: (-t[0], t[1], t[2], t[3]))
    endstop_peaks = [
        (v, y, x)
        for (v, y, x, _th) in _directional_nms(
            _es_dn, peak_spacing,
            _SPACING_PERP_FRAC * median_wavelength)
    ]

    curv_abs = np.abs(curvature_signed).astype(np.float32)
    # [BLANK-FRAME FIX] same energy requirement for curvature candidates.
    # [A5#4] The floor is a statistic over the pixels that can BECOME peaks.
    # Over the whole frame it was dominated by background noise (kappa is
    # defined everywhere, and empty background has a random orientation), so it
    # moved with how much of the image was empty.
    _on = I >= np.float32(min_integration)
    _pos = curv_abs[_on & (curv_abs > 0.0)]
    _curv_floor = float(np.median(_pos)) if _pos.size else 0.0
    curv_candidates = [
        (v, y, x)
        for (y, x, v) in _local_maxima_2d(curv_abs, curv_radius)
        if v > _curv_floor and I[y, x] >= min_integration
    ]
    # [BORDER-MARGIN FIX] same discard for the curvature candidate set
    curv_candidates = [t for t in curv_candidates if _inside_interior(t)]
    _cv_dn = [
        (v, int(y), int(x), _dominant_theta_rad(integration, int(y), int(x)))
        for (v, y, x) in curv_candidates
    ]
    _cv_dn.sort(key=lambda t: (-t[0], t[1], t[2], t[3]))
    curv_peaks = [
        (v, y, x)
        for (v, y, x, _th) in _directional_nms(
            _cv_dn, peak_spacing,
            _SPACING_PERP_FRAC * median_wavelength)
    ]

    # [FIX 6] the final union is directional too: same-location, different
    # source peaks must merge, parallel contours must not.
    # [U25 F4] UNIFORM UNITS. The three sources ranked by incomparable
    # quantities (ridge energy, end-stop mean, |curvature| px^-1), so the greedy
    # union compared 30.28 with 2.55 at the SAME pixel. Every candidate is
    # re-ranked by the integration energy it sits on -- the evidence all three
    # of them are about.
    _uni_dn = [
        (float(I[int(y), int(x)]), int(y), int(x),
         _dominant_theta_rad(integration, int(y), int(x)))
        for (_v, y, x) in (ridge_samples + endstop_peaks + curv_peaks)
    ]
    _uni_dn.sort(key=lambda t: (-t[0], t[1], t[2], t[3]))
    all_peaks = [
        (v, y, x)
        for (v, y, x, _th) in _directional_nms(
            _uni_dn, peak_spacing,
            _SPACING_PERP_FRAC * median_wavelength)
    ]

    peaks: List[Dict] = []
    for v, y, x in all_peaks:
        # [A5#14] ONE argmax over the (theta, sf) cell, so an exact tie breaks
        # the way every other reader in this file breaks it (lowest theta
        # first). The old per-sf loop preferred the smallest sf instead.
        _cell = integration[:, :, y, x]
        best_theta, best_sf_here = (
            int(i) for i in np.unravel_index(int(np.argmax(_cell)), _cell.shape)
        )
        # [U25 C9] the channel axis of THIS field, not the module default.
        theta_rad = _channel_theta(best_theta, n_theta)

        phase_val = float(phase[best_theta, best_sf_here, y, x])
        phase_bin = int(((phase_val + math.pi) / (2 * math.pi)) * 4) % 4

        endstop_val = float(endstop[best_theta, best_sf_here, y, x])
        endstop_flag = 1 if endstop_val > 0.5 else 0

        theta_row = integration[:, best_sf_here, y, x]
        total = float(theta_row.sum()) + 1e-6
        top = float(theta_row.max())
        coherence = top / total

        # [P1-8] The graded spatial frequency (cycles per pixel) and its
        # position on the bank's own log-frequency ladder.
        sf_here = float(sfs[best_sf_here])
        sf_axis_pos = ((math.log2(max(sf_here, 1e-9)) - sf_log_lo)
                       / sf_log_span) * float(max(n_sf - 1, 1))

        peaks.append({
            "x": int(x),
            "y": int(y),
            "theta": float(theta_rad),
            "theta_idx": int(best_theta),
            "sf_idx": int(best_sf_here),
            "sf": float(sfs[best_sf_here]),
            # [P1-8] the graded frequency and its ladder position (the
            # index is only the lookup key).
            "sf_cycles_per_px": sf_here,
            "sf_axis_pos": float(sf_axis_pos),
            "signed_curvature": float(curvature_signed[y, x]),
            "bend_angle": float(bend_angle[y, x]),
            "phase": float(phase_val),
            "phase_bin": int(phase_bin),
            "endstop_flag": int(endstop_flag),
            # [P1-7] the GRADED end-stop magnitude (the flag is its index).
            "endstop_magnitude": float(endstop_val),
            "integration_strength": float(v),
            "coherence": float(coherence),
        })
    return peaks


# =============================================================================
# Convenience top-level
# =============================================================================

def _clone_result(result: Dict) -> Dict:
    """[U25 C7] A result whose mutable containers belong to the caller.

    The numeric arrays are shared by contract (every consumer only reads
    them); `peaks` and `sfs` are cheap and are the two things a caller
    reasonably edits, so they are copied.
    """
    out = dict(result)
    out["sfs"] = [float(v) for v in result["sfs"]]
    out["peaks"] = [dict(p) for p in result["peaks"]]
    return out


def _field_cache_key(lum: np.ndarray, scale: float) -> Tuple:
    """Cache key for a computed field: the exact luminance, the scale, and
    every constant that determines the field's output.

    The luminance's own bytes are hashed (not its identity), so a DIFFERENT
    array with the same content is the same field, and an array with
    different content never is. The parameter fingerprint covers every
    module-level constant, so a constant change can never return a field
    computed under the old constants.
    """
    digest = hashlib.blake2b(
        np.ascontiguousarray(lum).tobytes(), digest_size=16
    ).digest()
    return (lum.shape, digest, float(scale), field_fingerprint())


# [SPEED] One-entry cache of the computed field.
#
# The field is a PURE, DETERMINISTIC function of (luminance, scale) and the
# module constants: `determinism_check` in this file asserts that running
# it twice on the same input gives byte-identical output. The three lenses
# that need the field in one frame (PeakLens, the V2 shape family, the V4
# family) all ask for the SAME luminance -- `planes_from_rgb` derives the
# adapted luminance from the same raw image the observation adapted -- and
# each was recomputing the whole field (the association-field integration
# alone is ~1 s per call). Caching the field makes the second and third
# requests return the SAME object, so the values are identical by
# construction rather than by recomputation.
#
# One entry bounds the memory. The cached field is the SAME dict object the
# observation already holds (`observation._contour_field_cache`), so in the
# live path this costs no extra array memory at all.
_FIELD_CACHE: Dict[Tuple, Dict] = {}


def field_from_luminance(
    lum: np.ndarray, scale: Optional[float] = None,
) -> Dict:
    """Run the full field pipeline on a luminance image.

    Returns a dict with: simple, complex, phase, integration,
    integration_driving, endstop, curvature_signed, bend_angle, peaks, scale,
    sfs.  The arrays are shared with the one-entry cache (read-only by
    contract); `peaks` and `sfs` are cloned per call.
    """
    lum = np.asarray(lum)
    # [U25 C11] An empty/3-D/non-finite plane must fail HERE, not inside the
    # cache key: a NaN luminance poisons the key (NaN != NaN, so every call
    # misses) and a 3-D plane silently reads as a stack of rows.
    # [A5#9] An EMPTY plane passes the isfinite test, gives scale = 0 and
    # reaches GaborBank.build_default(scale=0).
    if lum.ndim != 2 or lum.size == 0:
        raise ValueError(
            "field_from_luminance expects a non-empty 2-D luminance plane, "
            "got shape %r" % (lum.shape,))
    if not np.all(np.isfinite(lum)):
        raise ValueError(
            "field_from_luminance expects finite luminance (NaN/inf found)")
    if lum.dtype != np.float32:
        lum = lum.astype(np.float32)
    H, W = lum.shape
    if scale is None:
        scale = front_end_scale(H, W)
    scale = float(scale)
    if not (math.isfinite(scale) and scale > 0.0):
        raise ValueError("scale must be positive and finite, got %r" % (scale,))

    # [SPEED] Same (luminance, scale, constants) -> same field, by
    # construction (the field is a pure function of those three).
    cache_key = _field_cache_key(lum, scale)
    hit = _FIELD_CACHE.get(cache_key)
    if hit is not None:
        # [U25 C7] The arrays are shared (the consumers only read them); the
        # peak dicts AND the sfs list are cloned, so a consumer that mutated
        # either cannot corrupt the cached field.
        return _clone_result(hit)

    # [A5#2] Fetch the bank WITHOUT passing it on: `v1_responses` uses the O5
    # raw-response cache only for the DEFAULT bank, and passing `bank=` made
    # every call miss it, recomputing 120 convolutions per frame.
    bank = _get_bank(scale)
    _check_field_bank(bank)
    simple, complex_raw, phase = v1_responses(lum, scale=scale)
    complex_ = divisive_normalize(complex_raw, scale=scale)

    sfs = np.geomspace(
        _GABOR_SF_MIN / max(scale, 1e-6),
        _GABOR_SF_MAX / max(scale, 1e-6),
        _GABOR_N_SF,
    ).astype(np.float32)
    wavelength_mean = _median_wavelength(sfs)
    cocirc = cocircular_context(complex_, wavelength_mean)

    integration, endstop, integration_driving = integration_field(
        complex_, sfs, cocircular_wide=cocirc,
    )
    curvature_signed, bend_angle = curvature_field(
        integration, wavelength_mean, lum=lum)
    peaks = extract_peaks(
        integration, endstop, curvature_signed, bend_angle,
        complex_, phase, sfs,
        border_margin_px=_border_margin_px(bank),
    )
    result = {
        "simple": simple,
        "complex": complex_,
        "phase": phase,
        "integration": integration,
        "integration_driving": integration_driving,
        "endstop": endstop,
        "curvature_signed": curvature_signed,
        "bend_angle": bend_angle,
        "peaks": peaks,
        "scale": float(scale),
        "sfs": sfs.tolist(),
    }
    _FIELD_CACHE.clear()
    _FIELD_CACHE[cache_key] = result
    return _clone_result(result)


# =============================================================================
# Diagnostics
# =============================================================================

def probe_padding_schemes(
    size: int = _PADDING_TEST_IMAGE_SIZE,
) -> Dict:
    """Compare padding modes at the border.

    For each mode, convolve a random luminance image with one mid-
    scale Gabor kernel and report the ratio of mean |response| in the
    3-pixel border band to the mean |response| in the 3-pixel inset
    interior. A ratio near 1.0 means the padding does not manufacture
    border artifacts. This formalizes the 2026-10-10 front-end fix's
    finding that reflect padding did manufacture artifacts on the
    previous pipeline.

    fftconvolve is used only for large kernels and does not accept a
    padding-mode argument; the padding modes reported are those that
    ndimage.convolve accepts. If the probe kernel exceeds the FFT
    threshold only the reflect path exists (there is one production path),
    which the caller can see from `kernel_size` / `use_direct`.

    [A5#10] Named `probe_*`, not `test_*`: pytest collects `test_*` and
    errors on a function that returns a value.
    """
    rng = np.random.default_rng(0)
    lum = rng.random((size, size), dtype=np.float32)
    bank = GaborBank.build_default(scale=front_end_scale(size, size))
    nt, ns = (int(v) for v in bank.kernels.shape[:2])
    k = bank.kernels[min(3, nt - 1), min(2, ns - 1), 0]
    kernel_size = int(k.shape[0])
    use_direct = kernel_size <= _FFT_KERNEL_THRESHOLD

    b = 3
    results: Dict[str, Dict[str, float]] = {}
    modes = ("reflect", "constant", "nearest") if use_direct else ("reflect",)
    for mode in modes:
        if use_direct:
            resp = nd_convolve(lum, k, mode=mode).astype(np.float32)
        else:
            # Production large-kernel path (see _convolve_same / filters._conv2d):
            # symmetric pad + valid FFT, NOT mode="same" (which zero-pads).
            _ph = _pw = kernel_size // 2
            _pad = np.pad(lum, ((_ph, _ph), (_pw, _pw)), mode="symmetric")
            resp = fftconvolve(_pad, k, mode="valid").astype(np.float32)
        border = np.concatenate([
            np.abs(resp[:b, :]).ravel(),
            np.abs(resp[-b:, :]).ravel(),
            np.abs(resp[:, :b]).ravel(),
            np.abs(resp[:, -b:]).ravel(),
        ])
        interior = np.abs(resp[b:-b, b:-b]).ravel()
        bm = float(border.mean()) if border.size else 0.0
        im = float(interior.mean()) if interior.size else 0.0
        results[mode] = {
            "border_mean": bm,
            "interior_mean": im,
            "ratio": bm / (im + 1e-9),
        }
    results["kernel_size"] = float(kernel_size)
    results["use_direct"] = float(1.0 if use_direct else 0.0)
    return results


def probe_conv_branches(size: int = 96, k: int = 29, seed: int = 0) -> Dict:
    """[A5#10] Direct (nd_convolve reflect) vs symmetric-pad + FFT, same kernel.

    The two branches of `_convolve_same` had never been compared, because the
    old `min(..., 7)` threshold routed every production kernel onto the FFT
    path. The tolerance is PROVISIONAL (float32 FFT round-off).
    """
    rng = np.random.default_rng(seed)
    img = rng.random((size, size), dtype=np.float32)
    kern = rng.standard_normal((k, k)).astype(np.float32)
    direct = nd_convolve(img, kern, mode="reflect").astype(np.float32)
    ph = k // 2
    fft = fftconvolve(np.pad(img, ((ph, ph), (ph, ph)), mode="symmetric"),
                      kern, mode="valid").astype(np.float32)
    err = float(np.abs(direct - fft).max()) / (
        float(np.abs(direct).max()) + 1e-12)
    return {"max_rel_err": err, "passed": bool(err < 1e-4)}


def blank_frame_check(size: int = _PADDING_TEST_IMAGE_SIZE) -> Dict:
    """Blank-frame gate: uniform image yields 0 peaks."""
    # [BLANK-FRAME FIX] A blank scene is a uniform NONZERO grey plate, not
    # a zero frame: with all-zero luminance the V1 response is identically
    # zero and the gate tests nothing. The nonzero case is the one that
    # exposed the relative-threshold failure (300 peaks).
    # [U25 C17] Test white as well as grey.
    out = {}
    passed = True
    for name, val in (("grey", 0.5), ("white", 1.0)):
        lum = np.full((size, size), val, dtype=np.float32)
        result = field_from_luminance(lum)
        count = len(result["peaks"])
        out[name] = {
            "peak_count": count,
            "passed": bool(count == 0),
            "integration_max": float(result["integration"].max()),
        }
        passed = passed and (count == 0)
    out["passed"] = bool(passed)
    out["peak_count"] = out["grey"]["peak_count"]
    return out


def determinism_check(size: int = 128, seed: int = 0) -> Dict:
    """Run field twice on the same input; report byte-identity.

    [A5#17] Both caches are cleared before EACH run (a stale entry could
    otherwise serve run A), and every returned key is compared -- the old
    check skipped `integration_driving` and `sfs`.
    """
    rng = np.random.default_rng(seed)
    lum = rng.random((size, size), dtype=np.float32)
    _FIELD_CACHE.clear()
    _V1_RESPONSE_CACHE.clear()
    a = field_from_luminance(lum)
    # [SPEED] The field cache would make the second call a cache HIT, which
    # would make this check vacuous. Clear it so the second run really
    # recomputes the field.
    # [U25 C17] ... and clear the V1 raw-response cache too: it is keyed by the
    # luminance bytes, so a hit there also shortcuts the recomputation the check
    # is supposed to measure.
    _FIELD_CACHE.clear()
    _V1_RESPONSE_CACHE.clear()
    b = field_from_luminance(lum)
    arrays = ("simple", "complex", "phase", "integration",
              "integration_driving", "endstop", "curvature_signed",
              "bend_angle")
    out = {k + "_identical": bool(np.array_equal(a[k], b[k]))
           for k in arrays}
    out["peaks_identical"] = a["peaks"] == b["peaks"]
    out["sfs_identical"] = a["sfs"] == b["sfs"]
    out["passed"] = bool(all(out.values()))
    return out


# =============================================================================
# __main__: diagnostic block
# =============================================================================

if __name__ == "__main__":
    import sys
    import time

    size = 128
    rng = np.random.default_rng(0)
    lum = rng.random((size, size), dtype=np.float32)

    det = determinism_check(size=size, seed=0)
    print("=== determinism ===")
    print(det)

    print("=== blank frame ===")
    bf = blank_frame_check(size=size)
    print(bf)

    print("=== padding schemes ===")
    for mode, stat in probe_padding_schemes(size=size).items():
        print(f"  {mode}: {stat}")

    print("=== conv branches ===")
    cb = probe_conv_branches()
    print(cb)

    print("=== runtime (mean of 5, cold) ===")
    times = []
    for _ in range(5):
        # [A5#17] a warm cache would time _clone_result, not the field.
        _FIELD_CACHE.clear()
        _V1_RESPONSE_CACHE.clear()
        t0 = time.perf_counter()
        _ = field_from_luminance(lum)
        times.append(time.perf_counter() - t0)
    times.sort()
    print(f"  mean={sum(times)/len(times):.3f}s  max={times[-1]:.3f}s")

    print("=== peak count ===")
    r = field_from_luminance(lum)
    print(f"  n_peaks={len(r['peaks'])}")

    ok = (bool(det.get("passed", False))
          and bool(bf.get("passed", False))
          and bool(cb.get("passed", False)))
    sys.exit(0 if ok else 1)
