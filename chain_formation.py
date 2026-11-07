"""chain_formation.py -- Step A5 chain formation, standalone.

One public function, ``form_chains``. Pure, deterministic, numpy + stdlib.

Adjacency (verbatim from the plan; A and B are chain-adjacent iff all hold):
   1. |pos_A - pos_B| < chain_max_gap_factor * median_wavelength   (strict)
   2. axial distance between theta_A and theta_B < chain_angle_tol
   3. end-stop along segment A->B < chain_endstop_max at every one of 10
      sample points (strict)

Chain = connected component of the adjacency graph with >= chain_min_peaks.

[M2] Rules 2-3 were too weak to stop parallel over-merge and too strong for
real-image recall respectively, so they were re-expressed geometrically and
recalibrated on synthetic truth rendered through the same front end. The
adjacency predicates in force are now:
   1. distance  < chain_max_gap_factor * median_wavelength            (spec)
   2. co-circular bend: 2 sin(dtheta/2) / d <= pi/4 per MEASURED peak spacing
   3. alignment: each endpoint points at the other within ALIGNMENT_TOL
      (35 deg, the geometric midpoint between the parallel-line deviation
      of 45 deg at one wavelength and the tightest representable arc's
      28.65 deg)
   4. wiggle: sagitta (d/2) tan(dtheta/2) <= 0.25 * max_gap
   5. corner links: a sharp corner over a long chord, added on top
   6. end-stop: < CHAIN_ENDSTOP_MAX at all 10 interior samples
   7. link support: mean integration along the segment, in the segment's
      own orientation channel and distance-matched scale, relative to the
      endpoints' own peak integration, >= LINK_SUPPORT_MIN
Constants 3, 6 and 7 were calibrated from synthetic ground truth through
this front end (diagnostics/_q1_es_calib.py, _q1_support.py; rules R5 and
R7 were written down before the numbers were computed).

Design decisions the spec left open (each is argued in the report):
  D1. Angle test is the *circular axial* distance
      min(d, pi - d), d = |dtheta| mod pi. The literal "|dtheta| mod pi"
      would reject theta=0.05 vs theta=pi-0.05, which are 0.10 rad apart
      as undirected orientations.
  D2. The 10 samples are t = (k + 0.5) / 10, k = 0..9: evenly spaced and
      symmetric, so t = 0 and t = 1 are not sampled. After nearest-pixel
      rounding, a sample that lands on either endpoint's own pixel is
      ignored (short segments round their first/last sample onto the
      endpoint pixel). If every sample is ignored the test passes
      vacuously. Reason: end-stop peaks sit on end-stop maxima, so reading
      their own pixel would stop them joining any chain. Known limit: a
      smooth end-stop bump wider than one pixel around such a peak still
      blocks its neighbours.
  D3. End-stop at a sample = max over the spatial-frequency axis of the
      channel whose orientation is the axial mean of theta_A and theta_B.
      [A3#1, corrected] The bank's channel i is centred on ``i * pi /
      n_theta`` (perception/filters.py builds ``bank.theta = k * pi /
      n_theta`` and ``_channel_theta`` in contour_field.py asserts the same),
      so the NEAREST channel to an angle is ``round(theta * n_theta / pi)``,
      not its floor: floor() biased every channel choice half a bin low.
      Nearest-pixel sampling (floor(v + 0.5)), clamped to the image.
  D4. Output order within a chain: the spine of the component's minimum
      spanning tree (Euclidean edge lengths) is its longest path; the
      chain is emitted by depth-first pre-order from the spine endpoint
      with the smaller (x, y, index), side branches before continuing
      along the spine. For a path-like chain this is endpoint -> endpoint.
  D5. "Closed (ring)" is undefined in the spec. Here: >= 3 peaks, the two
      spine endpoints are themselves chain-adjacent, and the spine length
      is >= 2 * max_gap (a straight or gently curved run whose ends are
      adjacent is shorter than max_gap, so it cannot qualify). Rings are
      emitted as sorted(indices), per the spec's literal wording. The flag
      is returned alongside the chain, not inferred from its ordering.
  D6. Chains are returned ordered by the smallest (x, y, index) among
      their members.
  D7. Ties: Kruskal edges are sorted by (squared distance [exact int],
      i, j); union-find roots are the smaller index; farthest-node
      selection takes the lowest index on exact ties; all neighbour lists
      are sorted before traversal. No set/dict iteration is order-relied.
"""

from __future__ import annotations

import math
import warnings
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

# order_ring_positions lives in the geometry layer above this engine and
# imports nothing from here, so a local import keeps the engine standalone.
# [A3#8] ImportError only: `except Exception` swallowed real defects in
# chain_polylines, and the stub installed in globals() made rings silently never
# close with no way to retry.
try:
    from perception.chain_polylines import order_ring_positions as _order_ring_positions
except ImportError:  # pragma: no cover - standalone use
    _order_ring_positions = None
_RING_WARNED = False

_N_SAMPLES = 10
# [SPEED] Max n_rows * n_entries per pairwise block. This is a MEMORY knob
# only: the arithmetic of every candidate pair is computed independently of
# the block boundaries, and the pairs are re-sorted by (i, j) afterwards, so
# the block size cannot change a single value. It was lowered from 4,000,000
# to 1,000,000 so the peak footprint of the pairwise stages stays bounded
# (the pairwise stages hold ~10 float64 temporaries of this size) as the
# peak count grows.
_PAIR_BLOCK = 1_000_000  # max n_rows * n entries per pairwise block
# [SPEED] Same idea for the corner-link pass, which needs three N x N
# temporaries. Blocked, so the peak is O(block) rather than O(N^2).
_CORNER_BLOCK = 1_000_000  # max n_rows * n entries per corner block
_ENDSTOP_BLOCK = 20_000  # max candidate pairs per end-stop block
# [Q3] _RING_MIN_SPINE_GAPS retired: see _order_component's D5 note. The
# length floor was structurally wrong (it rejected exactly the loops it
# should find), so closure is decided by total turning instead.
_RING_TURN_TOLERANCE = 0.5 * 3.141592653589793  # pi/2 radians

# Junction-cut radius (px): peaks closer than this that belong to different
# contours (one stem, one through-arm) are NOT linked, so a chain does not
# [M2 multi-scale] Association-field continuity law: the per-element turn
# limit. Two contour elements continue smoothly into one contour if the
# orientation change across them is within this bound PER SAMPLE. Expressed
# as a curvature bound against the MEASURED peak spacing, it is scale-free
# and resolution-free. This value is the classic continuity limit, not a
# metric-derived threshold.
ARC_TURN_PER_SAMPLE = 0.5 * 3.141592653589793 / 2.0  # pi/4

# [M2] Alignment tolerance: each endpoint's direction must point toward the
# other within this angle.
#
# The value is bounded by geometry, not chosen for a score. Measured:
#   parallel lines, separation-axis deviation   1.0 lam = 45.0 deg
#                                                1.5 lam = 56.3 deg
#                                                3.0 lam = 71.6 deg
#   arc, chord-vs-tangent deviation             r=12px  = 28.65 deg
#                                               r=120px =  2.86 deg
# The populations are disjoint for any separation at or above one sample
# spacing (>= 45 deg) against the tightest representable arc (28.65 deg), so
# the bound is set at their midpoint, 0.61 rad (35 deg).
#
# A 0.5-lambda separation gives 26.6 deg, inside the bound. That case is
# genuinely ambiguous rather than a defect: the cross-line neighbour is then
# closer than the on-line neighbour, so the dots cannot distinguish them
# without additional evidence the representation does not carry.
ALIGNMENT_TOL = 0.196 * 3.141592653589793  # ~0.616 rad (35 deg)

# [M2] The perpendicular deviation an arc may bow away from its straight
# chord, as a fraction of the maximum link length. This is what makes the
# allowed WIGGLE shrink as the chord grows, instead of the allowed angle
# growing and the wiggle growing as its square.
SAGITTA_FRAC_OF_GAP = 0.25

# [M2][Q1][ITEM 1a] End-stop threshold along a candidate segment. The plan
# specifies CHAIN_ENDSTOP_MAX = 0.3; that value was never calibrated against
# the field it is read from. Calibrated here on synthetic contours rendered
# through the same front end (diagnostics/_q1_es_calib.py, rule R5 stated
# before computing): over 211 true same-contour adjacent pairs the worst
# end-stop along the segment has p50 = 0.268, p90 = 0.577, p99 = 0.617, so
# 0.3 rejects ~46% of clean same-contour segments. The value below is the
# smallest on a 0.05 grid that admits >= 99% of that population under the
# ALL-sf reading.
#
# [ITEM 1a] The all-sf reading is not the honest one: it takes the max over
# all five spatial-frequency channels, so an off-scale channel can decide the
# verdict. Measured (diagnostics/_item1_matched_sf.py, same corpus), the
# end-stop read at the pair's own distance-matched scale gives true
# same-contour adjacent p50 = 0.163 / p90 = 0.423 / max = 0.654 against the
# all-sf max's 0.268 / 0.577 / 0.822 -- the all-sf reading roughly doubles the
# value. With the matched-sf reading installed, rule R16 (smallest 0.05-grid
# value keeping >= 99% of true adjacent pairs) selects 0.60, and that is the
# value below. It keeps 0.995 of true pairs on the corpus.
#
# What this rule is NOT: a cross-link discriminator. On the matched-sf reading
# parallel cross pairs are LOW (p50 0.085), i.e. the signal does not point the
# way the rule wants, and no variant of the sampling tested in
# _item1_endstop_variants.py separates the two populations (best margin
# +0.014). Cross-link rejection is carried by LINK_SUPPORT_MIN below, which
# rejects 100% of the corpus's labelled cross pairs.
#
# [ROUND 10 / B] Re-checked after the wide-context term became modulatory and
# the beam steps became sub-pixel. The end-stop field is |fwd - bwd| / (fwd +
# bwd), computed from the collinear beams only, so it is mathematically
# UNCHANGED by B. `_item1_endstop_variants.py` on the regenerated corpus now
# finds NO labelled cross pairs (the geometric filter + the new peak set leave
# none within the link range), so the R10 cross-vs-same separation is undefined;
# true same-adjacent `i_max_over_sf` p90 = 0.463 (n=242). With no negative
# population to separate, R16 is unchanged -> CHAIN_ENDSTOP_MAX stays 0.60.
CHAIN_ENDSTOP_MAX = 0.60

# [M2][Q1] Link support: the mean integration along the candidate segment,
# in the segment's OWN orientation channel and the scale whose wavelength
# matches the pair distance, relative to the endpoints' own peak
# integration. A segment that runs along a contour is supported along its
# whole length; a segment that cuts between two structures is not.
#
# Calibrated on the same synthetic corpus (diagnostics/_q1_support.py, rule
# R7 stated before computing): true same-contour adjacent pairs (n = 235)
# reach 0.475 minimum with p50 = 0.90, while parallel cross pairs (n = 102)
# top out at 0.622. The smallest value on a 0.05 grid that admits ZERO
# cross pairs is 0.65, which retains ~90% of true adjacent pairs and ~84%
# of real-image pairs that pass the geometric rules (measured on 8 images:
# median support 0.73-0.84, 83-96% above the threshold).
#
# [AF RE-DERIVATION, Round 6] Re-derived after the association field became
# MODULATORY. Link support now reads the DRIVING (pre-facilitation) field
# (`integration_driving`), so it is independent of the AF.
#
# [ROUND 10 / B] Re-derived AGAIN after the wide-context (cocircular) term
# inside DRIVING became MODULATORY rather than additive (contour_field v2) and
# the beam steps became sub-pixel/equal-arc-length. The additive term used to
# manufacture support across gaps; with it gone the separation is much cleaner.
# `diagnostics/_q1_support.py` on the regenerated corpus, R7 inputs printed
# side by side: TRUE-LINK minimum 0.009 (same-adjacent n=416, p10 0.201) next
# to the RESOLVABLE (>= 1.0 wavelength) CROSS-PAIR maximum 0.180 (n=72). Only
# the sub-resolution `parallel_0.5lam` (7 px apart read at wavelength ~= 21 px,
# below that scale's resolution) reaches the low same-adj minimum; the original
# rule excludes it from the negatives. R7 applied to resolvable negatives ->
# smallest 0.05-grid value above 0.180 = 0.20, which sits just below the
# resolvable true-link p10 (0.201).
LINK_SUPPORT_MIN = 0.20


def _median_nn_spacing(xs: np.ndarray, ys: np.ndarray) -> float:
    """Median nearest-neighbour distance among the peaks, in pixels.

    A contour cannot turn more than the continuity limit between adjacent
    samples without the adjacency itself becoming ambiguous, so the per-
    sample turn bound is denominated in THIS spacing. It is measured, so a
    dense field and a sparse field get the same turn-per-sample behaviour
    rather than the same absolute angle.

    Returns +inf when the spacing cannot be measured (fewer than 2 peaks),
    which the caller already guards against.
    """
    n = xs.shape[0]
    if n < 2:
        return float("inf")
    best = np.full(n, np.inf, dtype=np.float64)
    rows_per = max(1, _PAIR_BLOCK // max(1, n))
    for r0 in range(0, n, rows_per):
        r1 = min(n, r0 + rows_per)
        rows = np.arange(r0, r1, dtype=np.int64)
        dx = xs[None, :].astype(np.float64) - xs[rows, None].astype(np.float64)
        dy = ys[None, :].astype(np.float64) - ys[rows, None].astype(np.float64)
        d = np.sqrt(dx * dx + dy * dy)
        for k, r in enumerate(rows.tolist()):
            d[k, r] = np.inf
        np.minimum.at(best, rows, d.min(axis=1))
    # [A3#17] a ZERO nearest-neighbour distance (two peaks on one pixel) made
    # the median 0 -> spacing inf -> curvature_max 0, which rejected every
    # smooth link with a non-zero turn and made the corner spacing test
    # vacuous. Unmeasurable is handled by the `size == 0` branch below.
    finite = best[np.isfinite(best) & (best > 0.0)]
    if finite.size == 0:
        return float("inf")
    spacing = float(np.median(finite))
    if not (spacing > 0.0):
        return float("inf")
    return spacing


# [M2 corner rule] Written down before testing:
#   A pair (A, B) may link as a CORNER (rather than a smooth continuation)
#   only when ALL of these hold:
#     1. the separation d is at most 1.0 * sample_spacing -- a corner is a
#        local event, so it cannot be an integration over distance;
#     2. the orientation change |dtheta| exceeds alignment_tol -- otherwise
#        the pair is a smooth link, already admitted above, and admitting it
#        twice would be redundant;
#     3. |dtheta| <= pi/2 -- the direction changes by at most a right angle;
#     4. the implied curvature kappa <= CORNER_MAX_CURVATURE -- the turn is a
#        single-sample event: at most a right angle between two dots exactly
#        one spacing apart, not a gradual turn spread over several samples;
#     5. the two dots are connected to each other's contour: either a shared
#        neighbour (a sharp bend inside a smooth run), OR one of them is a
#        DEGREE-1 ENDPOINT of a smooth run and the other is the endpoint of
#        a different smooth run (two fragments meeting at a corner). At a
#        true corner the two sides meet AT the corner dot, so requiring a
#        shared third dot would reject the very case the rule exists for.
#        Degree-1 endpoints cannot be joined to each other by accident
#        without passing conditions 1-4, which bound both the distance and
#        the turn.
# Condition 5 is why corners and accidental proximity are distinguishable at
# all: a corner is where two contour FRAGMENTS meet, not merely two dots that
# are near each other.
CORNER_MAX_TURN = 0.5 * 3.141592653589793  # pi/2
# A corner is by definition sharper than a smooth turn, so it gets a looser
# curvature limit than the smooth-continuation rule: pi/2 over one spacing
# instead of pi/4. Using the smooth limit would reject every real corner and
# make the rule a no-op.
CORNER_CURVATURE_FACTOR = 2.0


def _candidate_pairs(
    xs: np.ndarray,
    ys: np.ndarray,
    th: np.ndarray,
    max_gap2: float,
    curvature_max: float,
    alignment_tol: float,
    half_sagitta_max: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pairs (i < j) that are plausibly on the same contour.

    Three rules, written down before any measurement:

    1. ALIGNMENT (the location<->direction constraint). The direction at A
       must point TOWARD B, and the direction at B must point TOWARD A, each
       within `alignment_tol` radians. Without this, two dots lying side by
       side on parallel lines are linked: their directions agree with each
       other (dtheta ~ 0), and rule 2 then admits them. Direction agreement
       is not enough -- direction must also agree with the separation vector.

    2. BEND (co-circularity). The implied curvature over the link is within
       the per-sample continuity limit:
           kappa = 2*sin(dtheta/2)/d <= curvature_max
       A long gentle arc and a short sharp corner are then treated
       differently, which one absolute angular cap cannot do.

    3. WIGGLE (the deviation bound). The perpendicular distance the arc bows
       away from the straight chord may not exceed `half_sagitta_max`:
           sagitta = (d/2) * tan(dtheta/2) <= half_sagitta_max
       With rule 2 alone the permitted angle grows with the chord, so the
       permitted WIGGLE grows as d^2 and at long range almost anything links.
       Bounding the sagitta makes the allowed turn shrink as 1/d instead,
       which is the "allowed wiggle shrinks as they get farther apart"
       behaviour the mandate asks for.

    `alignment_tol` and the spacing behind `curvature_max` /
    `half_sagitta_max` are association-field constants, not metric-derived
    numbers.
    """
    n = xs.shape[0]
    if n < 2:
        return (
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.float64),
        )
    rows_per = max(1, _PAIR_BLOCK // n)
    # [SPEED] Column tiling: the temporaries below are (rows, cols), so the
    # peak footprint is rows_per * cols_per rather than rows_per * n. The
    # per-pair arithmetic is element-wise and therefore independent of the
    # tiling; the pairs are re-sorted by (i, j) at the end, so the output
    # order is the (i, j) lexicographic order the untiled version produced.
    cols_per = max(1, _PAIR_BLOCK // rows_per)
    parts_i: List[np.ndarray] = []
    parts_j: List[np.ndarray] = []
    parts_d: List[np.ndarray] = []
    for r0 in range(0, n, rows_per):
        r1 = min(n, r0 + rows_per)
        rows = np.arange(r0, r1, dtype=np.int64)
        for c0 in range(0, n, cols_per):
            c1 = min(n, c0 + cols_per)
            cols = np.arange(c0, c1, dtype=np.int64)
            # separation vector A(row) -> B(col)
            dx = (xs[None, :] - xs[rows, None]).astype(np.float64)
            dy = (ys[None, :] - ys[rows, None]).astype(np.float64)
            d = np.sqrt(dx * dx + dy * dy)
            dth = np.mod(np.abs(th[rows, None] - th[None, :]), math.pi)
            dth = np.minimum(dth, math.pi - dth)

            # Rule 1: each direction must point toward the other. theta is an
            # axis in [0, pi); the tangent along the separation vector is
            # atan2(dy, dx) modulo pi, so wrapping both onto [0, pi) and
            # taking the wrapped difference measures agreement in either
            # traversal.
            sep_axis = np.mod(np.arctan2(dy, dx), math.pi)
            dev_row = _axis_diff(sep_axis, np.mod(th[rows, None], math.pi))
            dev_col = _axis_diff(sep_axis, np.mod(th[None, :], math.pi))
            aligned = (dev_row <= alignment_tol) & (dev_col <= alignment_tol)

            # Rule 2: bend
            kappa = 2.0 * np.sin(dth / 2.0) / np.maximum(d, 1e-9)

            # Rule 3: wiggle
            sagitta = (d / 2.0) * np.tan(dth / 2.0)

            ok = (
                (cols[None, :] > rows[:, None])
                & ((d * d).astype(np.float64) < max_gap2)
                & (kappa <= curvature_max)
                & (sagitta <= half_sagitta_max)
                & aligned
            )
            ri, cj = np.nonzero(ok)
            parts_i.append(rows[ri])
            parts_j.append(cols[cj])
            # [A3#18] exact integer squared distance, not sqrt(...)**2.
            parts_d.append((dx[ri, cj] ** 2 + dy[ri, cj] ** 2).astype(np.float64))
    if not parts_i:
        return (
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.float64),
        )
    I = np.concatenate(parts_i)
    J = np.concatenate(parts_j)
    D2 = np.concatenate(parts_d)
    # [SPEED] Restore the (i, j) lexicographic order the untiled version
    # produced (the tiling makes the concatenation chunk-major). The sort
    # is stable, so pairs that are exactly equal keep their relative order.
    _ord = np.lexsort((J, I))
    return (I[_ord], J[_ord], D2[_ord])


def _axis_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Wrapped difference of two angles in [0, pi), in [0, pi/2]."""
    dd = np.abs(a - b)
    return np.minimum(dd, math.pi - dd)


def _corner_links(
    I: np.ndarray,
    J: np.ndarray,
    D2: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    th: np.ndarray,
    spacing: float,
    curvature_max: float,
    alignment_tol: float,
    max_gap: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Corner links: pairs meeting the documented five conditions, that were
    NOT admitted as smooth links.

    A pure ADDITION over the smooth adjacency: condition 2 excludes anything
    already linked smoothly, so this can only ADD pairs, never remove or
    rewrite an existing one.

    Returns index arrays in the same shape as `_candidate_pairs` output.
    """
    n = xs.shape[0]
    if n < 2:
        return (
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.float64),
        )
    # [SPEED] The old implementation materialised three N x N arrays with
    # np.repeat / np.tile (~600 MB at 5000 peaks). The same candidate set is
    # computed here in row blocks, so the peak footprint is O(block).
    #
    # VALUE IDENTITY: the per-pair arithmetic is identical (the same int64
    # differences cast to float64, the same sqrt, the same squared distance
    # taken from the ROUNDED distance, not from dx^2 + dy^2), and the
    # candidate pairs are produced in the same (i, j) lexicographic order
    # the flat N x N scan produced -- which the greedy corner acceptance
    # loop below is sensitive to.
    gap = float(max_gap)
    keep_i: List[np.ndarray] = []
    keep_j: List[np.ndarray] = []
    keep_d2: List[np.ndarray] = []
    rows_per = max(1, _CORNER_BLOCK // n)
    for r0 in range(0, n, rows_per):
        r1 = min(n, r0 + rows_per)
        rows = np.arange(r0, r1, dtype=np.int64)
        cols = np.arange(n, dtype=np.int64)
        dx = (xs[None, :] - xs[rows, None]).astype(np.float64)
        dy = (ys[None, :] - ys[rows, None]).astype(np.float64)
        dd = np.sqrt(dx * dx + dy * dy)
        # only i < j and within the link distance
        ok = (cols[None, :] > rows[:, None]) & (dd < gap)
        ri, cj = np.nonzero(ok)
        if ri.size:
            keep_i.append(rows[ri])
            keep_j.append(cols[cj])
            # [A3#18] exact integer squared distance, not sqrt(...)**2.
            keep_d2.append((dx[ri, cj] ** 2 + dy[ri, cj] ** 2).astype(np.float64))
    if not keep_i:
        return (
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.float64),
        )
    cI = np.concatenate(keep_i)
    cJ = np.concatenate(keep_j)
    cD2 = np.concatenate(keep_d2)

    dth = np.mod(np.abs(th[cI] - th[cJ]), math.pi)
    dth = np.minimum(dth, math.pi - dth)
    cand = (
        (np.sqrt(cD2) <= spacing)
        & (dth > alignment_tol)
        & (dth <= CORNER_MAX_TURN)
        & (
            2.0 * np.sin(dth / 2.0) / np.maximum(np.sqrt(cD2), 1e-9)
            <= curvature_max * CORNER_CURVATURE_FACTOR
        )
    )
    if not bool(cand.any()):
        return (
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.float64),
        )
    # [A3#3] condition 2 only rejects dth <= alignment_tol, and a SMOOTH pair can
    # have dth up to 2*tol while passing the kappa and sagitta tests. Such a
    # pair would be emitted twice (once smooth, once corner), inflating adj and
    # the shared-neighbour test. Exclude the smooth pairs explicitly.
    if I.shape[0]:
        smooth_keys = I.astype(np.int64) * n + J.astype(np.int64)
        cand &= ~np.isin(cI * n + cJ, smooth_keys)
        if not bool(cand.any()):
            return (
                np.zeros(0, dtype=np.int64),
                np.zeros(0, dtype=np.int64),
                np.zeros(0, dtype=np.float64),
            )

    # condition 5: the pair must connect two contour fragments that actually
    # meet. `adj` is the SMOOTH adjacency. A pair qualifies when the dots
    # share a neighbour (a sharp bend inside one run), or when one dot is a
    # degree-1 endpoint and the other is too (two runs meeting at a corner).
    # Accepting into `adj` as we go lets a multi-corner outline propagate,
    # so a rectangle's four corners chain through each other.
    adj = defaultdict(set)
    for a, b in zip(I.tolist(), J.tolist()):
        adj[a].add(b)
        adj[b].add(a)

    keep_a: List[int] = []
    keep_b: List[int] = []
    keep_d2: List[float] = []
    seen_pairs = set()
    cand_idx = np.nonzero(cand)[0].tolist()
    changed = True
    while changed:
        changed = False
        for k in cand_idx:
            a, b = int(cI[k]), int(cJ[k])
            if (a, b) in seen_pairs:
                continue
            na, nb = adj.get(a), adj.get(b)
            if not na or not nb:
                continue
            shared = na & nb
            # [A3#2] condition 5 asks for a DEGREE-1 endpoint. The old
            # `degree = {a: 1 ...}` dict made that condition true for every dot
            # that had ANY smooth link, so interior mid-chain dots were
            # corner-linked. The live neighbour count is the degree.
            if shared or (len(na) == 1 and len(nb) == 1):
                keep_a.append(a)
                keep_b.append(b)
                keep_d2.append(float(cD2[k]))
                seen_pairs.add((a, b))
                adj[a].add(b)
                adj[b].add(a)
                changed = True
    if not keep_a:
        return (
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.float64),
        )
    return (
        np.asarray(keep_a, dtype=np.int64),
        np.asarray(keep_b, dtype=np.int64),
        np.asarray(keep_d2, dtype=np.float64),
    )


def _endstop_worst(
    I: np.ndarray,
    J: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    th: np.ndarray,
    endstop: np.ndarray,
    sfs: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Worst (max over the interior samples) end-stop value per candidate pair.

    Same sampling as `_endstop_clear` (D2, D3), returning the value itself
    rather than the thresholded verdict, so a diagnostic can report where
    the pairs actually fall relative to the caller's `endstop_max`.

    [ITEM 1a] When `sfs` is given, the end-stop is read at the single
    spatial-frequency channel whose wavelength is closest to the pair's own
    distance -- the same scale the link-support rule judges the link at --
    instead of the maximum over all five sf channels.

    Why this matters: the end-stop ratio |fwd - bwd| / (fwd + bwd) is
    dominated by the forward/backward BEAM masses, and those masses are
    computed per sf channel. Taking the maximum over sf lets an off-scale
    channel decide the verdict. Measured on synthetic contours rendered
    through this front end (diagnostics/_item1_matched_sf.py), the two
    readings of the same field differ by more than a factor of two on true
    same-contour adjacent pairs:

        population                all-sf max      matched sf
        same-adjacent p50         0.268           0.163
        same-adjacent p90         0.577           0.423
        same-adjacent max         0.822           0.654
        parallel-cross p50        0.376           0.085

    The matched-sf reading is the honest one: it is close to the ~0.000 the
    field measures along a straight contour in its own channel
    (_item1_endstop_profile.py), and the residual 0.16 tracks local curvature
    rather than the channel mix. It no longer points the right way for
    cross-link rejection (cross pairs are LOW, not high), which is the
    measured reason the end-stop rule cannot be a cross-link discriminator at
    any threshold: that job belongs to the link-support rule.
    """
    n_theta, _n_sf, H, W = endstop.shape
    use_matched = sfs is not None
    if use_matched:
        wl = 1.0 / np.maximum(np.asarray(sfs, dtype=np.float64), 1e-9)
    out = np.zeros(I.shape[0], dtype=np.float64)
    frac = (np.arange(_N_SAMPLES, dtype=np.float64) + 0.5) / _N_SAMPLES
    for b0 in range(0, I.shape[0], _ENDSTOP_BLOCK):
        i = I[b0 : b0 + _ENDSTOP_BLOCK]
        j = J[b0 : b0 + _ENDSTOP_BLOCK]
        ta, tb = th[i], th[j]
        mean = 0.5 * np.arctan2(
            np.sin(2.0 * ta) + np.sin(2.0 * tb),
            np.cos(2.0 * ta) + np.cos(2.0 * tb),
        )
        mean = np.mod(mean, math.pi)
        # [A3#1] nearest channel centre (i * pi / n_theta), not floor().
        ch = np.floor(mean * (n_theta / math.pi) + 0.5).astype(np.int64) % n_theta
        xa = xs[i].astype(np.float64)
        ya = ys[i].astype(np.float64)
        xb = xs[j].astype(np.float64)
        yb = ys[j].astype(np.float64)
        dx = xb - xa
        dy = yb - ya
        d = np.sqrt(dx * dx + dy * dy)
        px = xa[:, None] + frac[None, :] * dx[:, None]
        py = ya[:, None] + frac[None, :] * dy[:, None]
        sx = np.clip(np.floor(px + 0.5).astype(np.int64), 0, W - 1)
        sy = np.clip(np.floor(py + 0.5).astype(np.int64), 0, H - 1)
        if use_matched:
            isf = np.argmin(np.abs(wl[None, :] - d[:, None]), axis=1)
            vals = endstop[ch[:, None], isf[:, None], sy, sx]  # (P, 10)
            v = vals.astype(np.float64)
        else:
            vals = endstop[ch[:, None], :, sy, sx]  # (P, 10, n_sf)
            v = vals.max(axis=2).astype(np.float64)  # (P, 10)
        ax = np.clip(xa.astype(np.int64), 0, W - 1)[:, None]
        ay = np.clip(ya.astype(np.int64), 0, H - 1)[:, None]
        bx = np.clip(xb.astype(np.int64), 0, W - 1)[:, None]
        by = np.clip(yb.astype(np.int64), 0, H - 1)[:, None]
        # [U5.3 / CH-5] Grid-aware interior detection: a sample is "on an
        # endpoint" only when its CONTINUOUS position lies within half a pixel
        # of that endpoint. Rounding alone collapses a 4 px segment's interior
        # samples onto the two endpoint pixels (the vacuous case that broke
        # test_out_of_image_peaks_do_not_crash); the continuous test keeps them
        # as real interior measurements. Genuinely degenerate pairs (d <=
        # sqrt(2)) still have every sample within half a pixel of an endpoint,
        # so their vacuous admit is preserved (pinned small-ring contract).
        dx_a = px - xa[:, None]
        dy_a = py - ya[:, None]
        dx_b = px - xb[:, None]
        dy_b = py - yb[:, None]
        on_end = (np.hypot(dx_a, dy_a) <= 0.5) | (np.hypot(dx_b, dy_b) <= 0.5)
        v[on_end] = -np.inf
        out[b0 : b0 + _ENDSTOP_BLOCK] = v.max(axis=1)
    return out


def _endstop_clear(
    I: np.ndarray,
    J: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    th: np.ndarray,
    endstop: np.ndarray,
    endstop_max: float,
) -> np.ndarray:
    """Bool per candidate pair: end-stop < endstop_max at all samples (D2, D3)."""
    return _endstop_worst(I, J, xs, ys, th, endstop) < float(endstop_max)


def _link_support(
    I: np.ndarray,
    J: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    integration: np.ndarray,
    sfs: np.ndarray,
) -> np.ndarray:
    """Segment support ratio per candidate pair (see LINK_SUPPORT_MIN).

    support = mean over the 10 interior samples of
              integration[ch(sep), sf(d), y, x] / endpoint reference

    where ch(sep) is the orientation channel of the segment's own direction
    (indexed exactly as D3), sf(d) is the scale whose wavelength is closest
    to the pair distance d, and the endpoint reference is the larger of the
    two endpoints' own peak-cell maxima (max over theta and sf at the
    endpoint pixel). Samples use the same t grid and nearest-pixel rounding
    as the end-stop test. Pairs whose endpoints sit on zero integration
    return 0.0 (no support at all), which the caller's threshold rejects.
    """
    n_theta, _n_sf, H, W = integration.shape
    wl = 1.0 / np.maximum(np.asarray(sfs, dtype=np.float64), 1e-9)
    out = np.zeros(I.shape[0], dtype=np.float64)
    frac = (np.arange(_N_SAMPLES, dtype=np.float64) + 0.5) / _N_SAMPLES
    for b0 in range(0, I.shape[0], _ENDSTOP_BLOCK):
        i = I[b0 : b0 + _ENDSTOP_BLOCK]
        j = J[b0 : b0 + _ENDSTOP_BLOCK]
        xa = xs[i].astype(np.float64)
        ya = ys[i].astype(np.float64)
        xb = xs[j].astype(np.float64)
        yb = ys[j].astype(np.float64)
        dx = xb - xa
        dy = yb - ya
        d = np.sqrt(dx * dx + dy * dy)
        sep = np.arctan2(dy, dx)
        # [A3#1] nearest channel centre (i * pi / n_theta), not floor().
        ch = np.floor(sep * (n_theta / math.pi) + 0.5).astype(np.int64) % n_theta
        isf = np.argmin(np.abs(wl[None, :] - d[:, None]), axis=1)
        px = xa[:, None] + frac[None, :] * dx[:, None]
        py = ya[:, None] + frac[None, :] * dy[:, None]
        sx = np.clip(np.floor(px + 0.5).astype(np.int64), 0, W - 1)
        sy = np.clip(np.floor(py + 0.5).astype(np.int64), 0, H - 1)
        vals = integration[ch[:, None], isf[:, None], sy, sx].astype(np.float64)
        ax = np.clip(xa.astype(np.int64), 0, W - 1)
        ay = np.clip(ya.astype(np.int64), 0, H - 1)
        bx = np.clip(xb.astype(np.int64), 0, W - 1)
        by = np.clip(yb.astype(np.int64), 0, H - 1)
        ref = np.maximum(
            integration[:, :, ay, ax].max(axis=(0, 1)).astype(np.float64),
            integration[:, :, by, bx].max(axis=(0, 1)).astype(np.float64),
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(ref > 0.0, vals.mean(axis=1) / ref, 0.0)
        out[b0 : b0 + _ENDSTOP_BLOCK] = ratio
    return out


def _total_turning_closes(
    nodes: List[int],
    adj_keys: Set[int],
    n: int,
    xl: List[int],
    yl: List[int],
) -> bool:
    """Does walking this component as one cycle close with ~2*pi of turning
    and enclose positive area?

    Size-free closure test. Endpoint-adjacency alone cannot distinguish a
    small true loop from a short straight run whose ends happen to be
    adjacent; the length floor tried to and rejected the loops instead. The
    distinguishing property of a simple closed curve is its total turning
    (Hopf Umlaufsatz): exactly +/-2*pi for a loop, ~0 for an out-and-back
    run.

    BOTH tests are required, and the second is not redundant. A DEGENERATE
    retrace -- three collinear points walked as a triangle -- satisfies the
    turning test exactly, because reversing direction at each end contributes
    pi and there are two ends: pi + pi = 2*pi. That is the accident the area
    test excludes: a simple closed curve encloses strictly positive area, and
    any collinear walk, however ordered, encloses none.

    Tolerance pi/2 is geometric, not empirical: a deviation of pi/2 would
    mean some vertex is turned the wrong way by at least a right angle, i.e.
    the walk self-crosses or retraces, and is therefore not simple.

    Returns False when the component does not admit a single simple cycle
    (e.g. a "figure eight" or a dumbbell), which is the conservative choice:
    an ambiguous component is ordinary rather than falsely closed.
    """
    if len(nodes) < 3:
        return False

    pts = [(xl[v], yl[v]) for v in nodes]

    # The cycle is chosen geometrically (two nearest neighbours in position
    # space), NOT from the chaining adjacency. This layering is required:
    # the co-circular rule deliberately admits edges beyond the immediate
    # neighbours (a gentle arc's skip-one pair), so a component's chaining
    # degrees routinely exceed 2. Requiring degree-2 adjacency here would
    # then reject exactly the rings the permissive rule was meant to close.
    #
    # What must hold instead is that every edge OF THE CYCLE is chainable --
    # a ring is a simple closed curve whose consecutive pairs all pass the
    # co-circular test.
    # [A3#8] no global stub: warn ONCE and answer "does not close", so a broken
    # import stays visible and a later successful import still works.
    global _RING_WARNED
    if _order_ring_positions is None:
        if not _RING_WARNED:
            _RING_WARNED = True
            warnings.warn(
                "chain_polylines.order_ring_positions unavailable: rings will "
                "not be detected", RuntimeWarning)
        return False
    order = _order_ring_positions(pts, list(nodes))
    if order is None or sorted(order) != list(range(len(nodes))):
        return False

    m = len(order)
    for k in range(m):
        a = nodes[order[k]]
        b = nodes[order[(k + 1) % m]]
        key = a * n + b if a < b else b * n + a
        if key not in adj_keys:
            return False  # a cycle edge is not chainable

    walk = [pts[i] for i in order]
    m = len(walk)

    # enclosed area (shoelace). The SIGN is the walk's orientation and is
    # deliberately discarded: a clockwise loop encloses just as much area as
    # a counter-clockwise one. What matters is that it is strictly nonzero --
    # any collinear walk, however ordered, encloses exactly zero.
    area2 = 0.0
    for i in range(m):
        x0, y0 = walk[i]
        x1, y1 = walk[(i + 1) % m]
        area2 += x0 * y1 - x1 * y0
    if abs(area2) <= 0.0:
        return False

    total = 0.0
    sharpest = 0.0
    for i in range(m):
        a = walk[(i - 1) % m]
        b = walk[i]
        c = walk[(i + 1) % m]
        v1x, v1y = b[0] - a[0], b[1] - a[1]
        v2x, v2y = c[0] - b[0], c[1] - b[1]
        turn = math.atan2(v1x * v2y - v1y * v2x, v1x * v2x + v1y * v2y)
        total += turn
        sharpest = max(sharpest, abs(turn))

    # Every member of a measured contour must be a continuation, not a
    # corner sharper than a right angle.  This is the same continuity bound
    # the corner rule uses, and it is what separates a CLOSED contour from an
    # OPEN arc closed by a skip-one chord: on a smooth arc the chord's two
    # end vertices each turn by ~pi - step (measured 165 deg on a 15-degree
    # semicircle sample), while a genuine ring -- and a box's right-angle
    # corner -- stay at or below pi/2.
    if sharpest > 2.0 * math.pi / 3.0 + 1e-9:
        return False

    return abs(abs(total) - 2.0 * math.pi) <= _RING_TURN_TOLERANCE


def _farthest(dist: Dict[int, float]) -> int:
    best, best_d = -1, -1.0
    for node in sorted(dist):
        if dist[node] > best_d:
            best, best_d = node, dist[node]
    return best


def continuous_orientations(
    complex_: np.ndarray, peaks: List[Dict], sfs: Optional[np.ndarray] = None
) -> np.ndarray:
    """[Q1 Route A] Continuous orientation per peak, from the field's complex
    energy.

    The discrete `theta` each peak carries is the ARGMAX over cached
    orientation bins, which quantises the local direction onto a fixed grid.
    Measured on real ETH-80 peaks, that quantisation is large: the median
    nearest-neighbour alignment deviation between two peaks on the same
    contour is 47 degrees, against a 35-degree bound set from geometry. So
    the discrete theta is too coarse for the alignment rule to recover the
    links it should.

    A first attempt sampled the field's `simple` array at the peak's own bin
    instead. That is the wrong quantity: `simple`'s third axis holds the
    Gabor even/odd PAIR, so `atan2(odd, even)` is the carrier phase, which
    is arbitrary. The energy envelope over the theta bins is the orientation
    information, which is what this uses.

    `complex_` is the field's energy with shape (n_theta, n_sf, H, W) on a
    PI-PERIODIC axis: channel i is centred on ``i * pi / n_theta`` (see
    perception/filters.py), so the doubled-angle mean below is a DIRECTION on
    the same pi-periodic axis, which is what the caller folds mod pi anyway.

    At a peak's coordinate the energy as a function of theta peaks around the
    local direction; an ENERGY-WEIGHTED CIRCULAR MEAN of that curve gives a
    sub-bin direction. The mean is circular because the theta axis wraps at
    2*pi, and energy-weighted so a narrow confident response dominates while
    a flat ambiguous spread contributes its own mean.

    Peaks whose bin pair or coordinates fall outside the field, or whose
    weighted curve is degenerate, fall back to the discrete theta so the
    array is always the same length as ``peaks`` and index-aligned.

    [B38] WHICH BIN, AND AT WHICH SCALE.  Two corrections, both to the input of
    the estimator rather than to the estimator:

    1. The centre bin is the ARGMAX OF THE CURVE BEING READ, not the peak's
       stored ``theta_idx``.  ``theta_idx`` is the argmax of `integration` (the
       accumulated association field) while this function reads `complex_` (the
       divisive-normalised energy envelope): two different arrays, so the
       parabola was being fitted around a bin the read curve does not peak at.
       `integration` accumulates along a contour and is inflated by any other
       structure inside the filter's support, so its argmax can be perpendicular
       to the contour the peak sits on; the normalised energy is not.
    2. The scale is the field's MEDIAN-WAVELENGTH channel, passed in as ``sfs``,
       not the peak's own ``sf_idx``.  ``sf_idx`` is chosen by `extract_peaks`
       as the argmax over scales of the RAW integration maximum, and raw
       integration grows with scale, so it selects the coarsest channel for
       every peak -- measured: `sf_idx == 0` for 95/95, 94/94 and 84/84
       on-contour peaks in box_over_box, cylinder_over_box and f_haze_a, i.e.
       the index carries no information at all (B39, reported not fixed: it also
       makes the peak's `sf`, `sf_cycles_per_px` and `sf_axis_pos` slots
       constant).  The coarsest channel is also the most contaminated: at
       (43,92) on f_haze_a's far box top edge -- a HORIZONTAL contour -- the
       near box's vertical edge 14 px away is inside its support, and the
       integration argmax there is 112.5 deg against the energy argmax's 7.5
       deg.  The median wavelength is the field's own characteristic scale and
       the one every distance bound in this module is denominated in
       (`max_gap = chain_max_gap_factor * median_wavelength`), so the
       orientation the alignment gate compares is read at the scale the gate's
       other bounds are measured in.  With ``sfs`` omitted the caller's previous
       behaviour (the peak's own ``sf_idx``) is kept.

    Scored against the RENDERER's own contour direction (the principal axis of
    the object's boundary pixels within 3 px of the peak; `_u13_orient.py`,
    straight-run peaks only, 246 peaks over 3 scenes, axial error in degrees):

        estimator                                   p50    p90    max   >35.3 deg
        theta_idx (integration argmax, shipped)     7.50   7.50  89.97    0.0244
        complex argmax, peak's own sf               7.51   8.02  89.65    0.0081
        complex argmax, median-wavelength scale     7.50   7.62  35.99    0.0041
        complex argmax, most coherent scale         7.50   7.53  51.34    0.0041
        complex argmax, all scales summed           7.50   7.70  86.69    0.0081

    (7.5 deg is the floor: the bank's 12 bin centres are 7.5 + 15k deg and the
    corpus's contours are axis-aligned.)  The median-scale reading is used
    because it is derived from a quantity the module already has, not because it
    wins the table -- it ties the coherence-selected reading and beats the rest.

    Downstream effect, measured on 1099 truth-adjacent same-object peak pairs
    over 14 development scenes (`_u13_links.py`): the ALIGNMENT gate is the only
    geometric gate that refuses a real contour link (bend 0, wiggle 0, end-stop
    0, tree 0), and it refuses 158 of 1099 = 14.4% of them.  Those refusals are
    what breaks the occluded contour into the 2-member chains and the isolated
    peaks the T-junction rule cannot use.
    """
    n_theta, n_sf = complex_.shape[0], complex_.shape[1]
    _h, _w = complex_.shape[2], complex_.shape[3]
    # [B38] The field's characteristic scale: the channel whose wavelength is
    # closest to the median wavelength of the bank's own frequency ladder.
    read_sf: Optional[int] = None
    if sfs is not None:
        arr = np.asarray(sfs, dtype=np.float64).ravel()
        if arr.size == n_sf and np.all(arr > 0.0):
            wl = 1.0 / arr
            med = float(np.median(wl))
            if math.isfinite(med) and med > 0.0:
                read_sf = int(np.argmin(np.abs(wl - med)))
    # [C] The bank's orientation axis is PI-PERIODIC ([0, pi), n_theta
    # bins), so the bin centres go on that axis and the weighted mean is
    # taken on the DOUBLED angle (the standard way to average an axial
    # quantity) and halved at the end. (The previous code used the full
    # 2*pi axis, so the value was not comparable with the peak's own
    # `theta` -- it was 2x off.)
    bin_theta = [(math.pi * t / n_theta) for t in range(n_theta)]
    out = np.zeros(len(peaks), dtype=np.float64)
    window = 2
    offsets = range(-window, window + 1)
    for i, p in enumerate(peaks):
        fallback = float(p.get("theta", 0.0))
        si = read_sf if read_sf is not None else int(p.get("sf_idx", 0))
        x = int(p.get("x", 0))
        y = int(p.get("y", 0))
        if not (0 <= si < n_sf) or not (0 <= y < _h and 0 <= x < _w):
            out[i] = fallback
            continue
        energies = complex_[:, si, y, x].astype(np.float64)
        # [B38] the centre bin is this curve's own argmax: the parabola below
        # refines the peak OF THE ARRAY BEING READ, and `theta_idx` is the
        # argmax of a different array (`integration`).
        ti = int(np.argmax(energies))
        if not (0 <= ti < n_theta):
            out[i] = fallback
            continue
        idxs = [(ti + d) % n_theta for d in offsets]
        ws = np.array([max(0.0, float(energies[k])) for k in idxs])
        if ws.sum() <= 1e-12:
            out[i] = fallback
            continue
        # [C] The sub-bin estimate: a PARABOLIC interpolation of the
        # peak. The energy-weighted mean over a five-bin window is
        # sensitive to the profile's SHAPE, and the shape differs
        # between resolutions (the Gabor's orientation bandwidth in
        # pixels is fixed), so the chain's orientation statistics were
        # not scale-stable. The quadratic through the peak bin and its
        # two neighbours uses the peak and its curvature, and degrades
        # to the bin centre when the peak is flat.
        y0 = float(energies[(ti - 1) % n_theta])
        y1 = float(energies[ti])
        y2 = float(energies[(ti + 1) % n_theta])
        denom = y0 - 2.0 * y1 + y2
        offset = 0.0
        if abs(denom) > 1e-12:
            offset = 0.5 * (y0 - y2) / denom
        offset = max(-0.5, min(0.5, offset))
        out[i] = (bin_theta[ti] + offset * (math.pi / n_theta)) % math.pi
    return out


def _graph_components(
    peaks: List[Dict],
    endstop: np.ndarray,
    median_wavelength: float,
    chain_max_gap_factor: float,
    chain_endstop_max: float,
    orientations: Optional[np.ndarray],
    integration: Optional[np.ndarray],
    sfs: Optional[np.ndarray],
    link_support_min: float,
) -> Dict:
    """Adjacency + spanning forest. The whole of ``form_chains``'s graph
    stage, kept separate so the occlusion pass can read the same forest
    instead of building a second, drifting copy of it."""
    if not (median_wavelength > 0.0):
        raise ValueError("median_wavelength must be > 0")
    n = len(peaks)
    if n == 0:
        return dict(
            n=0,
            xl=[],
            yl=[],
            th=[],
            tree=[],
            adj_keys=set(),
            comps={},
            max_gap=0.0,
            spacing=float("inf"),
            median_wavelength=float(median_wavelength),
        )
    es = np.asarray(endstop)
    if es.ndim != 4:
        raise ValueError("endstop must have shape (n_theta, n_sf, H, W)")

    xs = np.array([p["x"] for p in peaks], dtype=np.int64)
    ys = np.array([p["y"] for p in peaks], dtype=np.int64)
    if orientations is not None:
        # [Q1 Route A] sub-bin orientation from the field's continuous energy
        # [A3#7] A wrong-length array used to be SILENTLY ignored (different
        # geometry, no error) and a 2-D array passed the shape[0] test.
        o = np.asarray(orientations, dtype=np.float64)
        if o.shape != (n,):
            raise ValueError(
                "orientations must have shape (%d,), got %r" % (n, o.shape))
        th = o
    else:
        th = np.array([p["theta"] for p in peaks], dtype=np.float64)
    if not np.all(np.isfinite(th)):
        raise ValueError("peak orientations must be finite")
    n_sf = es.shape[1]
    if sfs is not None:
        sfs_a = np.asarray(sfs, dtype=np.float64).ravel()
        if sfs_a.shape[0] != n_sf:
            raise ValueError("sfs length must equal endstop.shape[1]")
    if integration is not None:
        integ = np.asarray(integration)
        # [A3#7] a non-4-D array used to silently DISABLE the support ranking.
        if (integ.ndim != 4 or integ.shape[:2] != es.shape[:2]
                or integ.shape[2:] != es.shape[2:]):
            raise ValueError(
                "integration must match endstop's (n_theta, n_sf, H, W)")
    xl, yl = xs.tolist(), ys.tolist()

    max_gap = float(chain_max_gap_factor) * float(median_wavelength)
    # [M2] Multi-scale co-circularity, with the alignment and wiggle bounds
    # that make the adjacency rule shape-aware rather than bend-aware only.
    # All three constants are denominated in the MEASURED peak spacing and
    # the caller's own max link length, so resolution and peak density are
    # tracked rather than assumed.
    spacing = _median_nn_spacing(xs, ys)
    curvature_max = ARC_TURN_PER_SAMPLE / spacing
    half_sagitta_max = SAGITTA_FRAC_OF_GAP * max_gap
    I, J, D2 = _candidate_pairs(
        xs, ys, th, max_gap * max_gap, curvature_max, ALIGNMENT_TOL, half_sagitta_max
    )
    # [M2] Corner links, added on top of the smooth links: a pure addition
    # (condition 2 excludes anything already linked), used so that a
    # sharp-cornered outline closes as one contour instead of breaking into
    # arcs at every corner.
    ci, cj, cd2 = _corner_links(
        I, J, D2, xs, ys, th, spacing, curvature_max, ALIGNMENT_TOL, max_gap
    )
    if ci.size:
        I = np.concatenate([I, ci])
        J = np.concatenate([J, cj])
        D2 = np.concatenate([D2, cd2])
    if I.shape[0]:
        # [ITEM 1a] The end-stop verdict is read at the pair's own scale when
        # the field's spatial frequencies are available (see
        # `_endstop_worst`): the all-sf max lets an off-scale channel decide,
        # which is what made the old reading unusable. `sfs` is None only in
        # unit tests that build an endstop array without a scale axis.
        if sfs is not None:
            keep = _endstop_worst(
                I, J, xs, ys, th, es, np.asarray(sfs, dtype=np.float64)
            ) < float(chain_endstop_max)
        else:
            keep = _endstop_clear(I, J, xs, ys, th, es, chain_endstop_max)
        # [Q1 / R7] Link support is a measured CONFIDENCE on a link, not a
        # veto on continuity.  Its own calibration says no threshold can
        # separate true links from cross links (true-link minimum 0.009,
        # cross-pair maximum 0.180), so hard-vetoing it must cut some real
        # contour links (measured: 28 of 226 consecutive truth-peak pairs).
        # It is therefore used to RANK the spanning tree (supported edges
        # first, then distance); the component structure is the geometric
        # adjacency, exactly as the [D6] note below requires for closure.
        supported = np.ones(I.shape[0], dtype=bool)
        if integration is not None and sfs is not None:
            integ = np.asarray(integration)
            if integ.ndim == 4:
                sup = _link_support(
                    I, J, xs, ys, integ, np.asarray(sfs, dtype=np.float64)
                )
                supported = sup >= float(link_support_min)
        # [D6] The GEOMETRIC adjacency: every link the geometric gates admit,
        # kept even when support is low. Closure reads this set (see
        # `_order_component`); component/tree structure reads it too, with
        # support only ordering the Kruskal preference.
        geo_I, geo_J, geo_D2 = I[keep], J[keep], D2[keep]
        geo_supported = supported[keep]
        I, J, D2 = geo_I, geo_J, geo_D2
    else:
        geo_I = geo_J = np.zeros(0, dtype=np.int64)
        geo_D2 = np.zeros(0, dtype=np.float64)
        geo_supported = np.zeros(0, dtype=bool)

    # Kruskal over adjacency edges: yields components and the MST forest.
    uf = list(range(n))

    def find(a: int) -> int:
        root = a
        while uf[root] != root:
            root = uf[root]
        while uf[a] != root:
            uf[a], a = root, uf[a]
        return root

    tree: List[List[int]] = [[] for _ in range(n)]
    adj_keys: Set[int] = set()
    # Prefer a support-backed link over a support-vetoed one of the same or
    # shorter length; then shortest first, then index order.  Support ranks,
    # it does not remove: an articulation link stays in the tree whatever its
    # support, because geometry established the continuation.
    order = np.lexsort((J, I, D2, (~geo_supported).astype(np.int64)))
    Il, Jl = I.tolist(), J.tolist()
    for k in order.tolist():
        i, j = Il[k], Jl[k]
        adj_keys.add(i * n + j)
        ri, rj = find(i), find(j)
        if ri == rj:
            continue
        lo, hi = (ri, rj) if ri < rj else (rj, ri)
        uf[hi] = lo
        tree[i].append(j)
        tree[j].append(i)
    for lst in tree:
        lst.sort()

    comps: Dict[int, List[int]] = {}
    for i in range(n):  # ascending => sorted
        comps.setdefault(find(i), []).append(i)

    # [D6] Geometric adjacency, both orientations, for the closure test.
    geo_keys: Set[int] = set()
    for a, b in zip(geo_I.tolist(), geo_J.tolist()):
        geo_keys.add(a * n + b)
        geo_keys.add(b * n + a)

    return dict(
        n=n,
        xl=xl,
        yl=yl,
        th=th,
        xs=xs,
        ys=ys,
        tree=tree,
        adj_keys=adj_keys,
        geo_keys=geo_keys,
        comps=comps,
        max_gap=max_gap,
        spacing=spacing,
        median_wavelength=float(median_wavelength),
    )


def _sweep_local(root: int, adj: Dict[int, List[int]], xl: List[int], yl: List[int]):
    """Geodesic distances and parents inside one adjacency (ring-free)."""
    dist: Dict[int, float] = {root: 0.0}
    parent: Dict[int, int] = {root: -1}
    stack = [root]
    while stack:
        u = stack.pop()
        for v in adj.get(u, ()):
            if v in dist:
                continue
            dx = xl[u] - xl[v]
            dy = yl[u] - yl[v]
            dist[v] = dist[u] + math.sqrt(float(dx * dx + dy * dy))
            parent[v] = u
            stack.append(v)
    return dist, parent


def _diameter_path(
    nodes: List[int], adj: Dict[int, List[int]], xl: List[int], yl: List[int]
) -> List[int]:
    """Longest tree path of this component, from its min-(x,y,index) end."""
    if len(nodes) <= 1:
        return list(nodes)
    d1, _ = _sweep_local(nodes[0], adj, xl, yl)
    a = _farthest(d1)
    d2, _ = _sweep_local(a, adj, xl, yl)
    b = _farthest(d2)
    s = min((a, b), key=lambda k: (xl[k], yl[k], k))
    d3, par = _sweep_local(s, adj, xl, yl)
    e = _farthest(d3)
    path: List[int] = []
    cur = e
    while cur != -1:
        path.append(cur)
        cur = par[cur]
    return path[::-1]


def _subtree_adj(
    root: int, parent: int, adj: Dict[int, List[int]], blocked=frozenset()
) -> Dict[int, List[int]]:
    """Adjacency of the subtree below ``root`` when ``parent`` is removed.

    [U4.4 / CH-3] The post-bridge forest may contain a cycle (a bridge was
    added across an occluded gap, closing a loop). The walk therefore treats
    the subtree as the graph reachable without revisiting a vertex.

    [A3#4] ... and it may not enter ``blocked`` (the caller's main path): the
    old version only refused the EDGE back to `parent`, so a cycle through
    another route re-entered main, `sub` became the whole component and
    `_decompose_paths` recursed on essentially the same graph.
    """
    block = set(blocked)
    block.add(parent)
    out: Dict[int, List[int]] = {root: []}
    stack = [root]
    while stack:
        v = stack.pop()
        for w in adj.get(v, ()):
            if w in block:
                continue
            if w not in out:
                out[w] = []
                stack.append(w)
            if w not in out[v]:
                out[v].append(w)
            if v not in out[w]:
                out[w].append(v)
    return out


def _branch_paths(
    main: List[int], adj: Dict[int, List[int]], xl: List[int], yl: List[int]
) -> List[List[int]]:
    """Decompose every multi-peak subtree hanging off ``main``."""
    on_main = set(main)
    out: List[List[int]] = []
    for v in main:
        for u in adj.get(v, ()):
            if u in on_main:
                continue
            sub = _subtree_adj(u, v, adj, on_main)
            if len(sub) == 1:
                continue
            out.extend(_decompose_paths(set(sub), sub, xl, yl))
    return out


def _decompose_paths(
    nodes: set, adj: Dict[int, List[int]], xl: List[int], yl: List[int]
) -> List[List[int]]:
    """Decompose a tree into maximal geodesic paths, main path first.

    The component's DIAMETER is its principal contour.  Every subtree hanging
    off that principal path is a measured branch (a T-junction stem, or a
    peak attached sideways); it is emitted as its own chain rather than
    threaded through the principal path.  This is the structural difference
    from the DFS pre-order, which visited every side branch before continuing
    along the spine and therefore put a contour's own members out of order
    (measured: 156 of 226 consecutive truth-peak pairs were geometrically
    admissible but not adjacent in the emitted walk).
    """
    if not nodes:
        return []
    main = _diameter_path(sorted(nodes), adj, xl, yl)
    on_main = set(main)

    def _dist(a: int, b: int) -> float:
        return math.hypot(xl[a] - xl[b], yl[a] - yl[b])

    # A one-peak branch is not a contour of its own, and dropping it would
    # discard a measured peak that the walk used to carry.  It is SPLICED into
    # the principal path at the position where it adds the least length, not
    # immediately after its tree attachment: the MST edge is only one of the
    # geometrically admissible links, and a peak lying between two spine
    # members hangs off the wrong one.  Inserting at the attachment there made
    # the walk turn ~170 deg and E3's spur prune then discarded a real contour
    # point (measured: 13 of 37 truth peaks on sphere_over_box).  Ties go to
    # the later index, which keeps coincident peaks in index order.
    spliced: List[int] = list(main)
    singles = []
    for v in main:
        for u in adj.get(v, ()):
            if u in on_main:
                continue
            if len(_subtree_adj(u, v, adj, on_main)) == 1:
                singles.append(u)
    # [A3#4] a peak reachable from two spine vertices was appended twice, and
    # then spliced into the chain twice.
    singles = sorted(set(singles))
    for u in singles:
        # Start from the tail insertion, then take any strictly cheaper
        # interior position, then the head.  Ties therefore stay at the end,
        # which keeps coincident peaks in index order.
        best_i = len(spliced)
        best_cost = _dist(u, spliced[-1]) if spliced else 0.0
        for i in range(len(spliced) - 1):
            a, b = spliced[i], spliced[i + 1]
            cost = _dist(a, u) + _dist(u, b) - _dist(a, b)
            if cost < best_cost:
                best_cost, best_i = cost, i + 1
        if spliced:
            cost_head = _dist(u, spliced[0])
            if cost_head < best_cost:
                best_i, best_cost = 0, cost_head
        spliced.insert(best_i, u)
    return [spliced] + _branch_paths(main, adj, xl, yl)


def _tree_path(a: int, b: int, adj: Dict[int, List[int]]) -> Optional[List[int]]:
    """Unique path between two vertices of a tree adjacency."""
    parent: Dict[int, int] = {a: -1}
    stack = [a]
    while stack:
        u = stack.pop()
        if u == b:
            break
        for v in adj.get(u, ()):
            if v not in parent:
                parent[v] = u
                stack.append(v)
    if b not in parent:
        return None
    path: List[int] = []
    cur = b
    while cur != -1:
        path.append(cur)
        cur = parent[cur]
    return path[::-1]


def _principal_cycle(
    nodes: List[int],
    adj: Dict[int, List[int]],
    ck: Set[int],
    n: int,
    xl: List[int],
    yl: List[int],
) -> Optional[List[int]]:
    """The component's longest chainable simple cycle, or None.

    The spanning tree contains no cycle, so every cycle of the component is
    the tree path between the ends of one non-tree geometric link plus that
    link.  Enumerating those FUNDAMENTAL cycles is exact and needs no
    search: the front silhouette's closing link is one of them, while a
    skip-one co-circular chord yields only the small sub-loop it spans.

    The principal contour is the LONGEST of them (the same rule as the open
    component's diameter path: the principal measured contour is the
    longest measured one).  A three-member cycle is a real small ring and is
    admitted when nothing longer closes; it cannot outrank a full
    silhouette.  Every candidate must pass the unchanged D5 shape test
    (``_total_turning_closes``: edge membership, simple cycle, area,
    +/-2*pi turning).
    """
    member = set(nodes)
    tree_edges = set()
    for v in nodes:
        for w in adj.get(v, ()):
            if v < w:
                tree_edges.add(v * n + w)
    best: Optional[List[int]] = None
    best_len = 0
    for key in sorted(ck):
        a, b = divmod(key, n)
        # [A3#15] `ck` holds BOTH orientations, so every non-tree link was
        # processed twice (two tree-path DFS calls), and a tree edge seen as
        # a > b was not recognised here (tree_edges stores v < w).
        if a >= b or a not in member or b not in member or key in tree_edges:
            continue
        path = _tree_path(a, b, adj)
        if path is None or len(path) < 3 or len(path) <= best_len:
            continue
        if not _total_turning_closes(path, ck, n, xl, yl):
            continue
        best, best_len = path, len(path)
    return best


def _component_paths(
    nodes: List[int],
    tree: List[List[int]],
    adj_keys: Set[int],
    n: int,
    xl: List[int],
    yl: List[int],
    max_gap: float,
    closure_keys: Optional[Set[int]] = None,
) -> List[Tuple[List[int], bool]]:
    """All chains a component contributes: its closed contour, plus branches.

    A closed contour attached to other measured peaks is the common case:
    the front object's silhouette is a cycle, and the back object's contour
    is joined to it at the T-junction where they meet.  The whole-component
    D5 test cannot close such a component (the branches are not part of its
    cycle), so the cycle is located first (``_principal_cycle``) and the
    rest of the measured tree is emitted as its own branches.
    """
    if len(nodes) == 1:
        return [(list(nodes), False)]
    member_set = set(nodes)
    adj = {v: [w for w in tree[v] if w in member_set] for v in nodes}
    ck = adj_keys if closure_keys is None else closure_keys

    ring = _principal_cycle(nodes, adj, ck, n, xl, yl)
    if ring is not None:
        # The ring's reported membership stays sorted(members): D5's contract
        # is closure, not visit order, and the traversal is derived one layer
        # up by chain_polylines.order_ring_positions.
        out: List[Tuple[List[int], bool]] = [(ring, True)]
        rest = member_set - set(ring)
        rest_adj = {v: [w for w in adj.get(v, ()) if w in rest] for v in rest}
        seen: Set[int] = set()
        for v in sorted(rest):
            if v in seen:
                continue
            comp: Set[int] = set()
            stack = [v]
            while stack:
                u = stack.pop()
                if u in comp:
                    continue
                comp.add(u)
                for w in rest_adj.get(u, ()):
                    stack.append(w)
            seen |= comp
            sub = {u: rest_adj.get(u, []) for u in comp}
            out.extend((path, False) for path in _decompose_paths(comp, sub, xl, yl))
        return out

    return [(path, False) for path in _decompose_paths(member_set, adj, xl, yl) if path]


def _chains_from(
    g: Dict,
    comps: Dict[int, List[int]],
    tree: List[List[int]],
    adj_keys: Set[int],
    chain_min_peaks: int,
    closure_keys: Optional[Set[int]] = None,
) -> List[Tuple[List[int], bool]]:
    """Order each component into chains and sort them by smallest member.

    A component can contribute more than one chain: an open component is its
    principal measured path plus its branches (``_component_paths``).  The
    minimum-member rule is applied per chain, so a one-peak branch is not a
    contour.  (Previously one component always produced exactly one chain,
    with branches threaded through the principal path.)

    ``closure_keys`` is passed through to the D5 ring test only; it defaults
    to ``adj_keys`` (the pre-D6 behaviour)."""
    n, xl, yl = g["n"], g["xl"], g["yl"]
    max_gap = g["max_gap"]
    chains: List[Tuple[List[int], bool]] = []
    for root in sorted(comps):
        members = comps[root]
        if len(members) < chain_min_peaks:
            continue
        for chain, is_ring in _component_paths(
            members, tree, adj_keys, n, xl, yl, max_gap, closure_keys=closure_keys
        ):
            if len(chain) >= chain_min_peaks:
                chains.append((chain, is_ring))
    chains.sort(
        key=lambda c: min(
            ((xl[i], yl[i], i) for i in c[0]), default=(float("inf"), float("inf"), n)
        )
    )
    return chains


def _components_of(
    tree: List[List[int]], n: int, keep_isolated: bool = False
) -> Dict[int, List[int]]:
    """Connected components of a forest, each member list ascending.

    ``keep_isolated`` keeps a peak with no surviving link as its own
    one-member component. It is off for the occlusion path (a singleton is
    not a contour) and on for the ordinary path, whose ``chain_min_peaks=1``
    contract emits it.
    """
    seen = [False] * n
    comps: Dict[int, List[int]] = {}
    for start in range(n):
        if seen[start] or not tree[start]:
            # An isolated peak (no link at all) is not a chain member in the
            # occlusion path; it never entered a component there.
            seen[start] = True
            if keep_isolated and not tree[start]:
                comps[start] = [start]
            continue
        stack = [start]
        seen[start] = True
        members: List[int] = []
        while stack:
            u = stack.pop()
            members.append(u)
            for v in tree[u]:
                if not seen[v]:
                    seen[v] = True
                    stack.append(v)
        members.sort()
        comps[members[0]] = members
    return comps


# ---------------------------------------------------------------------------
# [OCCLUSION] T-junction splitting and occlusion bridges
# ---------------------------------------------------------------------------
#
# MEASURED DEFECT (diagnostics/_sessionC_break_probe.py,
# _sessionC_arc_neighbourhood.py, 8 rendered occlusion scenes, 9 truth arcs
# whose hidden part is known by rendering the back object alone):
#
#   * at every one of the 9 arcs, a chain passes within 1.7 wavelengths
#     (0.0-7.1 px) of BOTH junctions -- the contour is found right up to the
#     occluder;
#   * at 6 of 9 arcs it is an INTERIOR member that sits there, i.e. the chain
#     did not stop at the junction: it turned onto the occluder and carried
#     on along it;
#   * 0 of 9 arcs was recovered as one contour across the occlusion.
#
# The cause is structural, not a threshold. `form_chains` decides adjacency
# from two peaks alone, so at a T-junction the stem's last peak and the bar's
# next peak are mutually admissible and the two contours become ONE component.
# A contour is a 1-D curve, so that component's spanning tree is no longer a
# path: the junction is a vertex of degree 3 with the stem hanging off it.
# (The V2 ContourLens already makes the same assumption explicitly -- "connected
# components of degree <= 2" -- so the two tiers disagree about what a contour
# is.)
#
# The fix is in two halves, both structural:
#
#   1. SPLIT. Any spanning-tree vertex of degree >= 3 is a junction: exactly
#      one pair of its incident edges is the contour passing THROUGH (the pair
#      whose directions are most anti-parallel); every other incident edge
#      terminates there. Cutting the terminating edges yields path components,
#      which is what a contour is, and leaves the junction pixel with the
#      through-contour (the occluder, the nearer surface).
#
#   2. BRIDGE. Two path ends on opposite sides of an occluder belong to one
#      contour. They are joined only on evidence, never by proximity:
#        (a) relatability -- a circle tangent to both ends must exist
#            (Parent & Zucker 1989); evaluated on the AXIAL orientations, so
#            both sign choices are tried and the best taken;
#        (b) an occlusion event -- the SAME other path must pass transversally
#            close to BOTH ends. That is the bar, seen twice; it cannot be
#            satisfied by two unrelated fragments.
#      Strength is the product of the graded components and is carried as a
#      non-identity side channel (peak indices + measurements); the chain
#      population schema is untouched.
#
# A candidate that fails any gate is SKIPPED and recorded as absent with the
# gate that refused it. No contour is ever invented.
#
# ---------------------------------------------------------------------------
# MEASURED OUTCOME AND WHY THE BRIDGE DOES NOT FIRE
# ---------------------------------------------------------------------------
# On the rendered corpus (8 scenes, 9 truth hidden arcs, diagnostics/
# _sessionC_bridge_calib.py, _sessionC_bridge_probe.json, _tmp_cross_check.py,
# _tmp_common_chain.py) this bridge accepts 0 of 9 true arcs.  The refusal
# histogram for sphere_over_box is 199 `not_relatable`, 85
# `no_transversal_bar`, 8 `bar_differs`, 1 `weak`.  That is a NULL RESULT and
# the constants below are therefore PROVISIONAL: they were NOT derived from
# accepted bridges, because there were none.  Three distinct structural fixes
# were tried and each was REFUTED BY MEASUREMENT, not by tuning:
#
#   FIX 1 -- require a TRANSVERSAL crossing of the bar at both ends.
#     Refuted: at 6 of 9 true junctions the occluder's silhouette is GRAZING
#     (|dtheta| < 30 deg) at one or both ends.  A silhouette tangency is not a
#     perpendicular T; the two curves touch, so "the bar crosses the stem" is
#     false by construction (_sessionC_arc_neighbourhood.json, dtheta_truth).
#
#   FIX 2 -- require the bridge chord to CROSS other pieces (a chord that spans
#     an occluder must cross its silhouette).
#     Refuted: the chord joins two points of the occluder's OWN silhouette, so
#     it lies INSIDE that silhouette and crosses it 0 times (box_over_box: 0;
#     sphere_over_box: 1 crossing at t = 0.988, i.e. at the endpoint).  The
#     representation does not hold the silhouette as a closed ring -- the
#     longest chain in a scene is 30 members where a closed r = 45 px ring at
#     the measured 7 px spacing needs ~40 -- so there is no polygon to test
#     enclosure against either.
#
#   FIX 3 -- identify the occluder as "the SAME other piece nearest to BOTH
#     ends".
#     Refuted: the occluder's own silhouette is itself fragmented, so the two
#     ends share a common other chain at only 2 of 9 arcs
#     (_tmp_common_chain.py).  Piece identity cannot name the occluder.
#
# Underlying these, and the deeper blocker: at the occlusion boundary the
# measured ORIENTATION is contaminated by the occluder's edge.  The two ends of
# a true hidden arc score relatability 0.352 and 1.759 against the 0.25
# tolerance, while a circle fitted to the true visible arc itself gives
# rms/R <= 0.17.  Parent & Zucker relatability is exact for two ends of one
# circle, so the failure is in the TANGENTS, not in the geometry -- and
# `_assoc_pair_geometry` / `_run_cocircular` read exactly those tangents.
# Relatability cannot adjudicate at an occlusion boundary.
#
# The ends themselves are also often absent: a chain reaches within 1.7
# wavelengths of BOTH junctions at all 9 arcs, but at 6 of 9 the near member is
# INTERIOR (the chain turned onto the occluder) and at 6 of 9 at least one side
# has no chain TERMINAL within 12 px at all (the last peak before the occluder
# is often isolated and is discarded as a singleton).
#
# UNBLOCK CONDITION (recorded, not worked around): the occluder's silhouette
# must be held as ONE closed contour so that (a) "the same occlusion event"
# is readable as one piece at both ends, and (b) the tangency can be peeled by
# curvature against that piece.  Until then no bridge is expected to fire, and
# 0 accepted bridges is the CORRECT, non-fabricating behaviour.
#
# Consequence for the downstream cues: `t_junction_order` reads "a chain
# ENDPOINT lands on another chain's INTERIOR".  Because the merge makes the
# stem and the bar ONE chain, that predicate cannot fire either -- measured
# 0/16 true T-junction order edges before AND after this pass
# (_sessionC_order_probe.py).  The bridge is a prerequisite for the cue, and
# the bridge is blocked by the same cause.

# How far apart two stem ends may be, in median wavelengths.  PROVISIONAL: the
# corpus's true hidden gaps are 8.16-12.01 wavelengths, so 15.0 admits every
# true gap with 25% headroom.  It has never been exercised by an ACCEPTED
# bridge (see above), so its upper end is untested.
BRIDGE_MAX_GAP_FACTOR = 15.0

# How close (in wavelengths) the bar must pass to a stem end.  PROVISIONAL: the
# stem's last found peak sits 0.0-1.4 wavelengths from the junction, so 2.0
# admits that with headroom.  Untested by acceptance.
BRIDGE_T_CROSS_WL = 2.0

# The transversal-crossing angle.  PROVISIONAL and, per FIX 1 above, the wrong
# discriminator at a silhouette: it is retained because it is the geometric
# meaning of "the bar crosses the stem", not because it separates the corpus.
BRIDGE_TJ_MIN_ANGLE = math.pi / 6.0

# Relatability tolerance, as the relative radius mismatch of the two tangency
# circles.  PROVISIONAL: the same-contour adjacent-member population gives
# p95 = 30 deg of tangent disagreement, and a true hidden arc's own ends score
# 0.352-1.759 (see above), so this tolerance rejects every true pair on this
# corpus.  It is not fitted to that failure.
BRIDGE_RELAT_TOL = 0.25

# Minimum graded strength to accept.  The strength is the product of the
# relatability, the two crossing grades and the curvature grade, each in
# [0, 1].  PROVISIONAL: no accepted bridge exists to calibrate against.
BRIDGE_MIN_STRENGTH = 0.30

# Diagnostics-only: the candidate list (accepted AND refused, with the gate
# that refused each) from the last `form_chains_detailed` call. It is never
# read by chain formation itself and never enters a Detection or an identity.
LAST_BRIDGE_CANDIDATES: List[Dict] = []


def _apply_junction_split(g: Dict):
    """[U3] Structural junction split on the spanning tree.

    Returns ``(tree, adj_keys, cuts, cut_keys)``. The tree is the post-split
    forest (terminating branches of every degree >= 3 vertex cut off); the
    keys are the adjacency keys that SURVIVE the split; the cuts record what
    was released.

    This is the single mechanism that separates a contour from the branch that
    terminates at a T-junction: it runs on the spanning tree BEFORE any path
    decomposition, so the through-contour (the occluding boundary) keeps its
    own path and the stem is released as its own contour instead of being
    stitched into it by the geometric adjacency.
    """
    tree, keys, cuts = _split_branches(g)
    cut_keys: Set[int] = set()
    n = g["n"]
    for c in cuts:
        cut_keys.add(int(c["junction"]) * n + int(c["branch"]))
        cut_keys.add(int(c["branch"]) * n + int(c["junction"]))
    return tree, keys, cuts, cut_keys


def _axial(x: float) -> float:
    return (x + math.pi / 2.0) % math.pi - math.pi / 2.0


def _circle_through(pa, ta, pb, tb):
    """Circle tangent to the line through `pa` at `pa` and through `pb`.

    Returns (relative_mismatch, R) where R is the mean tangency radius (inf
    for the collinear case). The construction is Parent & Zucker's: the
    centre lies on the normal at each end, so two normals fix a candidate
    centre; a completion exists only where the two radii agree. When the
    normals are parallel the edges are parallel: the completion is a
    straight line and the mismatch is the lateral offset the second end sits
    off the first end's line, relative to the chord.
    """
    d = math.hypot(pb[0] - pa[0], pb[1] - pa[1])
    if d < 1e-9:
        return float("inf"), float("inf")
    na = (-math.sin(ta), math.cos(ta))
    nb = (-math.sin(tb), math.cos(tb))
    det = na[0] * (-nb[1]) - na[1] * (-nb[0])
    if abs(det) < 1e-6:
        # parallel tangents: collinear completions are the only relatable ones
        mx = 0.5 * (pa[0] + pb[0]) - pa[0]
        my = 0.5 * (pa[1] + pb[1]) - pa[1]
        off = abs(mx * na[0] + my * na[1])
        return off / d, float("inf")
    rx, ry = pb[0] - pa[0], pb[1] - pa[1]
    u = (rx * (-nb[1]) - ry * (-nb[0])) / det
    cx, cy = pa[0] + u * na[0], pa[1] + u * na[1]
    ra = math.hypot(pa[0] - cx, pa[1] - cy)
    rb = math.hypot(pb[0] - cx, pb[1] - cy)
    mean_r = 0.5 * (ra + rb)
    return abs(ra - rb) / max(mean_r, 1e-12), mean_r


def _relatability(pa, tha, pb, thb):
    """Best over the four AXIAL sign choices of the tangency construction."""
    best = (float("inf"), float("inf"), 0.0, 0.0)
    for sa in (0.0, math.pi):
        for sb in (0.0, math.pi):
            err, r = _circle_through(pa, tha + sa, pb, thb + sb)
            if err < best[0]:
                best = (err, r, sa, sb)
    return dict(err=best[0], R=best[1], sign_a=best[2], sign_b=best[3])


def _path_ends(tree: List[List[int]], members: List[int]):
    """The two ends of a path component, ascending tie-break."""
    ends = [v for v in members if len(tree[v]) <= 1]
    return sorted(ends)


def _split_branches(g: Dict):
    """Cut every tree vertex of degree >= 3 down to its through-pair.

    Returns (tree, adj_keys, cuts). The cut edges are recorded with the
    junction vertex, the branch side, the two through-neighbours and the
    branch's own length, so a diagnostic can say which contour was released.

    [U4 / CH-1 + CH-3] One structural mechanism, ring-safe by construction:

    DEGREE >= 3: a junction has one through-pair (the most anti-parallel pair
    of incident edges) and every other incident edge terminates there. A cut
    is made only when the branch is a GENUINE TERMINATION: after removing the
    junction->branch edge, the branch can no longer reach the junction through
    any other geometric link. When it can, the component contains a real cycle
    through this vertex (a figure-eight crossing, a small ring's k-regular
    adjacency) and cutting it would destroy a closed contour -- so no cut is
    made. That is the same cycle check the bridge step needs, applied here so
    junction separation is ring-safe by construction.

    [U4.3 measured, refuted] An end-to-end-fold pass (turn > 2*pi/3 gated on
    a transversal bar, then additionally on candidate degree, then with an
    orientation-chosen cut side) was built and measured: E3 stayed 0/8 in all
    three variants, with no overlap gain. Reverted; see the U4 ledger.
    """
    tree = [list(v) for v in g["tree"]]
    keys = set(g["adj_keys"])
    xl, yl = g["xl"], g["yl"]
    n = g["n"]

    def _reaches(
        keys_now: Set[int], src: int, dst: int, skip_a: int, skip_b: int
    ) -> bool:
        """Can `src` reach `dst` over the GEOMETRIC links in `keys_now`
        without the skipped edge? (Non-tree links count: a cycle can close
        through one.)"""
        adj: Dict[int, List[int]] = {}
        for k in keys_now:
            a, b = divmod(k, n)
            adj.setdefault(a, []).append(b)
            adj.setdefault(b, []).append(a)
        stack = [src]
        seen = {src}
        while stack:
            v = stack.pop()
            for w in adj.get(v, ()):
                if (v == skip_a and w == skip_b) or (v == skip_b and w == skip_a):
                    continue
                if w in seen:
                    continue
                if w == dst:
                    return True
                seen.add(w)
                stack.append(w)
        return False

    cuts: List[Dict] = []
    while True:
        did = False
        for v in range(n):
            if len(tree[v]) < 3:
                continue
            nbrs = sorted(tree[v])
            best = None
            for a in range(len(nbrs)):
                for b in range(a + 1, len(nbrs)):
                    va = (xl[nbrs[a]] - xl[v], yl[nbrs[a]] - yl[v])
                    vb = (xl[nbrs[b]] - xl[v], yl[nbrs[b]] - yl[v])
                    na = math.hypot(*va)
                    nb = math.hypot(*vb)
                    if na < 1e-9 or nb < 1e-9:
                        continue
                    cos = (va[0] * vb[0] + va[1] * vb[1]) / (na * nb)
                    if best is None or cos < best[0]:
                        best = (cos, nbrs[a], nbrs[b])
            if best is None:
                continue
            keep = {best[1], best[2]}
            for u in nbrs:
                if u in keep:
                    continue
                # The release must genuinely separate the branch from the
                # junction; otherwise this vertex closes a cycle.
                if _reaches(keys, u, v, v, u):
                    continue
                tree[v].remove(u)
                tree[u].remove(v)
                keys.discard(v * n + u)
                keys.discard(u * n + v)
                cuts.append(
                    dict(
                        junction=int(v),
                        branch=int(u),
                        through=sorted(keep),
                        cos=float(best[0]),
                    )
                )
                did = True
        if not did:
            break
    return tree, keys, cuts


def _nearest_transversal(
    g: Dict,
    tree: List[List[int]],
    piece_of: Dict[int, int],
    end: int,
    own_piece: int,
    theta: float,
    radius: float,
    min_angle: float,
):
    """Nearest peak belonging to a DIFFERENT path component whose own
    orientation crosses this end transversally. That is the bar at a
    T-junction, read without a mask."""
    xl, yl = g["xl"], g["yl"]
    th = g["th"]
    best = None
    for v in range(g["n"]):
        if not tree[v]:
            continue
        pv = piece_of.get(v)
        if pv is None or pv == own_piece:
            continue
        d = math.hypot(xl[v] - xl[end], yl[v] - yl[end])
        if d > radius or d < 1e-9:
            continue
        ang = abs(_axial(th[v] - theta))
        if ang < min_angle:
            continue
        cand = (d, -ang, v, pv)
        if best is None or cand < best:
            best = cand
    return best


def _find_bridges(
    g: Dict,
    tree: List[List[int]],
    keys: Set[int],
    max_gap_factor: float,
    t_cross_wl: float,
    tj_min_angle: float,
    relat_tol: float,
    min_strength: float,
):
    """Accepted occlusion bridges plus the candidates that were refused.

    Only PATH ENDS are considered: a bridge joins the end of one contour to
    the end of another. Both ends must be transversally cut by the SAME other
    component, and the two ends must be relatable.
    """
    n = g["n"]
    xl, yl, th = g["xl"], g["yl"], g["th"]
    wl = g["median_wavelength"]
    pieces = _components_of(tree, n)
    piece_of: Dict[int, int] = {}
    for pid, members in pieces.items():
        for m in members:
            piece_of[m] = pid
    ends = []
    for pid, members in sorted(pieces.items()):
        for e in _path_ends(tree, members):
            ends.append((e, pid))
    band = max_gap_factor * wl
    radius = t_cross_wl * wl
    cands: List[Dict] = []
    for i in range(len(ends)):
        ea, pa_id = ends[i]
        for j in range(i + 1, len(ends)):
            eb, pb_id = ends[j]
            if pa_id == pb_id:
                continue
            d = math.hypot(xl[eb] - xl[ea], yl[eb] - yl[ea])
            if d < wl or d > band:
                continue
            rel = _relatability((xl[ea], yl[ea]), th[ea], (xl[eb], yl[eb]), th[eb])
            ta = _nearest_transversal(
                g, tree, piece_of, ea, pa_id, th[ea], radius, tj_min_angle
            )
            tb = _nearest_transversal(
                g, tree, piece_of, eb, pb_id, th[eb], radius, tj_min_angle
            )
            same_bar = bool(ta is not None and tb is not None and ta[3] == tb[3])
            ca = (
                0.0
                if ta is None
                else max(0.0, 1.0 - ta[0] / radius)
                * ((-ta[1] - tj_min_angle) / (math.pi / 2.0 - tj_min_angle))
            )
            cb = (
                0.0
                if tb is None
                else max(0.0, 1.0 - tb[0] / radius)
                * ((-tb[1] - tj_min_angle) / (math.pi / 2.0 - tj_min_angle))
            )
            ca = min(1.0, max(0.0, ca))
            cb = min(1.0, max(0.0, cb))
            grade_rel = (
                max(0.0, 1.0 - rel["err"] / relat_tol)
                if math.isfinite(rel["err"])
                else 0.0
            )
            # [A3#13] only form_chains_detailed sets g["curv"]; a direct
            # _find_bridges caller raised KeyError.
            curv = g.get("curv", {})
            ka, kb = curv.get(ea, 0.0), curv.get(eb, 0.0)
            if abs(ka) + abs(kb) > 1e-12:
                grade_curv = 1.0 - abs(ka - kb) / (abs(ka) + abs(kb))
            else:
                grade_curv = 1.0
            strength = grade_rel * ca * cb * grade_curv
            reason = None
            if not math.isfinite(rel["err"]) or rel["err"] > relat_tol:
                reason = "not_relatable"
            elif ta is None or tb is None:
                reason = "no_transversal_bar"
            elif not same_bar:
                reason = "bar_differs"
            elif strength < min_strength:
                reason = "weak"
            cands.append(
                dict(
                    a=int(ea),
                    b=int(eb),
                    piece_a=int(pa_id),
                    piece_b=int(pb_id),
                    d_px=float(d),
                    d_wl=float(d / wl),
                    relat_err=None
                    if not math.isfinite(rel["err"])
                    else float(rel["err"]),
                    R_px=None if not math.isfinite(rel["R"]) else float(rel["R"]),
                    bar_piece=None if not same_bar else int(ta[3]),
                    bar_a=None
                    if ta is None
                    else [int(ta[2]), float(ta[0] / wl), math.degrees(-ta[1])],
                    bar_b=None
                    if tb is None
                    else [int(tb[2]), float(tb[0] / wl), math.degrees(-tb[1])],
                    grade_rel=float(grade_rel),
                    grade_a=float(ca),
                    grade_b=float(cb),
                    grade_curv=float(grade_curv),
                    strength=float(strength),
                    accepted=reason is None,
                    reason=reason,
                )
            )
    # Deterministic greedy acceptance: strongest first, then (a, b). One
    # bridge per end, so no vertex ever gains a second link.
    cands.sort(key=lambda c: (-c["strength"], c["a"], c["b"]))
    used: Set[int] = set()
    accepted: List[Dict] = []
    # [A3#4] "one bridge per end" does NOT prevent a cycle: two bridges can join
    # the same pair of pieces, or three pieces can be joined in a loop. The
    # resulting cyclic "forest" made `_subtree_adj` walk back into main and
    # `_decompose_paths` recurse on essentially the same graph (RecursionError).
    # A union-find over the PIECES refuses a bridge that would close a cycle.
    puf: Dict[int, int] = {pid: pid for pid in pieces}

    def pfind(x: int) -> int:
        while puf[x] != x:
            puf[x] = puf[puf[x]]
            x = puf[x]
        return x

    for c in cands:
        if not c["accepted"]:
            continue
        if c["a"] in used or c["b"] in used:
            c["accepted"] = False
            c["reason"] = "end_already_bridged"
            continue
        ra, rb = pfind(c["piece_a"]), pfind(c["piece_b"])
        if ra == rb:
            c["accepted"] = False
            c["reason"] = "would_close_cycle"
            continue
        puf[max(ra, rb)] = min(ra, rb)
        used.add(c["a"])
        used.add(c["b"])
        accepted.append(c)
    accepted.sort(key=lambda c: (c["a"], c["b"]))
    return accepted, cands


def form_chains(
    peaks: List[Dict],
    endstop: np.ndarray,
    median_wavelength: float,
    chain_max_gap_factor: float = 2.0,
    chain_angle_tol: float = 0.3,  # [M2] superseded by co-circularity
    chain_endstop_max: float = CHAIN_ENDSTOP_MAX,
    chain_min_peaks: int = 2,
    orientations: Optional[np.ndarray] = None,
    integration: Optional[np.ndarray] = None,
    sfs: Optional[np.ndarray] = None,
    link_support_min: float = LINK_SUPPORT_MIN,
) -> List[Tuple[List[int], bool]]:
    """Group peaks into chains. Returns (chain, is_ring) tuples, each chain
    being a list of indices into ``peaks`` ordered along the chain. Chains
    smaller than ``chain_min_peaks`` are dropped. Uses only peak["x"],
    peak["y"], peak["theta"] (or ``orientations`` when supplied, which is
    the same length and index-aligned with ``peaks``).

    ``integration``/``sfs`` are the field's integration array and spatial
    frequencies. When supplied, the link-support rule (LINK_SUPPORT_MIN) is
    applied; when omitted, only the geometric, corner and end-stop rules
    gate adjacency, which is the behaviour callers that have no field (unit
    tests, synthetic peak lists) rely on.

    This is the ORDINARY path: no occlusion bridging. Junction separation
    is the SAME structural split `form_chains_detailed` uses (U4/CH-1):
    `_apply_junction_split` runs on the spanning tree before path
    decomposition, so a stem terminating on a through-contour is released as
    its own component rather than stitched into it. Nothing here bridges an
    occluded gap."""
    g = _graph_components(
        peaks,
        endstop,
        median_wavelength,
        chain_max_gap_factor,
        chain_endstop_max,
        orientations,
        integration,
        sfs,
        link_support_min,
    )
    if g["n"] == 0:
        return []
    # [U4 / CH-1] One structural mechanism owns junction separation for BOTH
    # entry points. Before this, the ordinary path applied a second, local
    # >120-degree turn cutter inside `_chains_from`, which could not fire on a
    # 90-degree T-junction and left the two entry points disagreeing.
    tree, keys, _cuts, cut_keys = _apply_junction_split(g)
    closure_keys = set(g.get("geo_keys", ())) - cut_keys
    comps = _components_of(tree, g["n"], keep_isolated=True)
    return _chains_from(
        g, comps, tree, keys, chain_min_peaks, closure_keys=closure_keys
    )


def form_chains_detailed(
    peaks: List[Dict],
    endstop: np.ndarray,
    median_wavelength: float,
    chain_max_gap_factor: float = 2.0,
    chain_angle_tol: float = 0.3,
    chain_endstop_max: float = CHAIN_ENDSTOP_MAX,
    chain_min_peaks: int = 2,
    orientations: Optional[np.ndarray] = None,
    integration: Optional[np.ndarray] = None,
    sfs: Optional[np.ndarray] = None,
    link_support_min: float = LINK_SUPPORT_MIN,
    bridge: bool = True,
    max_gap_factor: float = BRIDGE_MAX_GAP_FACTOR,
    t_cross_wl: float = BRIDGE_T_CROSS_WL,
    tj_min_angle: float = BRIDGE_TJ_MIN_ANGLE,
    relat_tol: float = BRIDGE_RELAT_TOL,
    min_strength: float = BRIDGE_MIN_STRENGTH,
) -> Tuple[List[Tuple[List[int], bool]], List[Dict]]:
    """Occlusion-aware chain formation: (chains, bridges).

    The chains are the same peaks grouped into 1-D contours, with junction
    branches released as their own contour and the two sides of an occluded
    gap joined when, and only when, the evidence supports it. ``bridges`` is
    the non-identity record of every accepted join (and, in ``diagnostics``
    form, of every refused candidate, through `bridge_candidates`).

    Identity is untouched: a bridge is a pair of PEAK INDICES plus measured
    strengths, never part of a Detection's serialized content.
    """
    # [A3#13] reset FIRST: an empty input returned without clearing the
    # previous call's candidates, so diagnostics showed stale bridges.
    global LAST_BRIDGE_CANDIDATES
    LAST_BRIDGE_CANDIDATES = []
    g = _graph_components(
        peaks,
        endstop,
        median_wavelength,
        chain_max_gap_factor,
        chain_endstop_max,
        orientations,
        integration,
        sfs,
        link_support_min,
    )
    if g["n"] == 0:
        return [], []
    g["curv"] = {
        i: float(peaks[i].get("signed_curvature", 0.0) or 0.0) for i in range(g["n"])
    }
    # [U3][U4 / CH-1] One structural mechanism owns junction separation for
    # BOTH entry points: `form_chains` calls this same `_apply_junction_split`
    # before its path decomposition. The local >120-degree turn cutter that
    # used to live inside `_chains_from` is gone; it could not fire on a
    # 90-degree T-junction and made the two entry points disagree.
    tree, keys, cuts, cut_keys = _apply_junction_split(g)
    accepted: List[Dict] = []
    cands: List[Dict] = []
    if bridge:
        accepted, cands = _find_bridges(
            g,
            tree,
            keys,
            max_gap_factor,
            t_cross_wl,
            tj_min_angle,
            relat_tol,
            min_strength,
        )
        g["bridge_candidates"] = cands
        for c in accepted:
            a, b = c["a"], c["b"]
            tree[a].append(b)
            tree[b].append(a)
            keys.add(a * g["n"] + b)
            keys.add(b * g["n"] + a)
    LAST_BRIDGE_CANDIDATES = cands
    g["branch_cuts"] = cuts
    # [A3#12] `form_chains` honours the chain_min_peaks=1 contract; this path
    # never emitted singletons.
    comps = _components_of(tree, g["n"], keep_isolated=(chain_min_peaks <= 1))
    # [D6] Closure reads the geometric adjacency (pre-support) minus the
    # junction cuts: a released stem must not be closed back onto the bar, and
    # a bridge is a completion across an occluder, not a ring.
    closure_keys = set(g.get("geo_keys", ())) - cut_keys
    chains = _chains_from(
        g, comps, tree, keys, chain_min_peaks, closure_keys=closure_keys
    )
    return chains, accepted
