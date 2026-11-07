"""perception/ownership.py -- the ONE border-ownership reader.

WHY THIS MODULE EXISTS. `BorderOwnershipLens` (perception/lenses/v2.py) emits
one detection per sampled boundary vertex with signature (arity 4):

    (arc_pos_bin, signed_dir_bin, curvature_bin, strength_bin)

`signed_dir_bin` is `_q_signed(signed_dir, max_bins=8)`, i.e. a SIGNED SCALAR
in [-1, 1] mapped to 0..7 -- NOT an angle, and NOT a cyclic direction. Three
readers in the tree disagreed about that slot:

  * perception/lenses/v2.py ContourLens: treated it as an 8-bin direction over
    2*pi and compared it with an edge bearing (a category error);
  * orchestration/binder.py `_run_border_ownership_edge`: treated it as an
    AXIAL angle over pi: `(bin + 0.5) * pi / 8` (a category error);
  * orchestration/binder.py `_bo_depth_constraints`: `(bin + 0.5)/8*2 - 1`, a
    signed scalar -- the ONLY reading that matches the producer.

This module is the single source of truth for that decode, so the readers
cannot drift again. The signed scalar is the edge-level FIGURE-GROUND signal:
`signed_dir > 0` means the detection's entity (its `alpha`) is the figure side
of the boundary, `< 0` means the named neighbour in `rho` ("neighbor") is.

SIGNED EDGE VOCABULARY. A border belongs to exactly one of its two flank
owners. We expose `figure_ground_pair` so the binder can emit the relation as
a SIGNED edge: positive coupling binds the border to the figure it owns and
negative coupling (OCCLUSION, sign=-1) marks the ground / T-occlusion side.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Optional, Tuple

__all__ = [
    "BO_BINS",
    "ownership_signed",
    "ownership_strength",
    "figure_ground_pair",
    "combine_ownership_cues",
    "combine_ownership_logodds",
    "ownership_logodds",
    "ownership_probability",
]

#: Bin count of the signed-direction slot (the producer's `_q_signed(..., 8)`).
BO_BINS = 8


def _quantized(det: Any) -> Tuple[float, ...]:
    q = getattr(getattr(det, "sigma", None), "quantized", None)
    return tuple(q) if q else ()


def ownership_signed(det: Any) -> float:
    """The signed figure-ground value of a BorderOwnership detection, in
    [-1, 1]: +1 = the entity (`alpha`) is the figure, -1 = the neighbour is.
    Decoded as the bin CENTRE of the `_q_signed(..., 8)` slot, matching the
    producer and `_bo_depth_constraints`."""
    q = _quantized(det)
    if len(q) < 2:
        return 0.0
    return (float(q[1]) + 0.5) / BO_BINS * 2.0 - 1.0


def _neighbor(det: Any) -> Optional[Any]:
    for role in getattr(det, "rho", ()) or ():
        if len(role) >= 2 and role[0] == "neighbor":
            return role[1]
    return None


def ownership_logodds(
    cues: Mapping[str, Optional[float]],
    weights: Mapping[str, float],
) -> float:
    """The UNSQUASHED log-odds of "the owner is on the LEFT of travel".

    This is the quantity ``combine_ownership_logodds`` feeds to ``tanh``.  The
    tanh is kept for the legacy graded value, but it is NOT a confidence: it
    compresses any real evidence into a narrow band near zero whenever the
    cues disagree (Session F measured p10 0.20, p50 0.214, max 0.347 over 552
    detections -- a 0.15-wide range that gates nothing).

    The raw sum is the actual evidence.  It is unbounded, so a single strong
    vote is not squashed into the same band as three weak ones, and it is the
    input the calibrated probability below is fitted on.
    """
    total = 0.0
    for name, w in weights.items():
        v = cues.get(name)
        if v is None:
            continue
        v = max(-0.999, min(0.999, float(v)))
        total += float(w) * math.log((1.0 + v) / (1.0 - v))
    return float(total)


def ownership_probability(evidence: float, *,
                          slope: float = 1.0,
                          intercept: float = 0.0) -> float:
    """P(owner side is correct) from a graded ownership statistic.

    ``evidence`` is the statistic the calibration selected (Session G: the
    multi-scale ENCLOSURE magnitude, because the raw log-odds magnitude was
    measured to carry no information about correctness -- AUC 0.43-0.53 --
    while |enclosure| carries AUC 0.65 on held-out scenes and separates a
    silhouette from an occlusion boundary at AUC 0.9997).  The shape is
    ``sigmoid(slope * |evidence| + intercept)``: the SIGN of the ownership
    claim is the claim itself, so confidence can only depend on magnitude.

    ``slope`` and ``intercept`` are fitted on the synthetic occlusion corpus
    against the renderer's object ids and recorded in
    ``parameters/_ownership_calibration.json``; the defaults are monotone so
    an uncalibrated caller gets a graded, not a false, reading.
    """
    x = abs(float(evidence))
    z = float(slope) * x + float(intercept)
    if z >= 60.0:
        return 1.0
    if z <= -60.0:
        return 0.0
    return 1.0 / (1.0 + math.exp(-z))


def combine_ownership_logodds(
    cues: Mapping[str, Optional[float]],
    weights: Mapping[str, float],
) -> float:
    """Log-odds (logit) combination of the per-fragment ownership cues.

    Each cue is a signed evidence in [-1, 1] for "the owner is on the LEFT of
    the travel direction" (+1) vs "on the right" (-1); 0 is a non-vote. The cue
    is read as a probability-like quantity and mapped to log-odds via the logit
    `log((1+v)/(1-v))`, weighted, and the summed log-odds is squashed back with
    `tanh` -- so independent agreements reinforce sub-linearly and a single
    strong dissenter cannot flip a well-supported side. A `None` cue (no
    evidence) is omitted, never treated as a 0.0 vote, so its weight does not
    dilute the survivors.

    The weights are the `BO_LOGODDS_WEIGHTS` family, fitted by logistic
    regression on a synthetic ground-truth corpus (known convex, concave and
    occluding layouts). This is the C3 combiner for the fragment producer;
    `combine_ownership_cues` remains the equal-authority mean used elsewhere.
    """
    total = 0.0
    for name, w in weights.items():
        v = cues.get(name)
        if v is None:
            continue
        v = max(-0.999, min(0.999, float(v)))
        lo = math.log((1.0 + v) / (1.0 - v))
        total += float(w) * lo
    return max(-1.0, min(1.0, math.tanh(total)))


# ---------------------------------------------------------------------------
# [F1] The three names `__all__` advertised and the module never defined.
#
# `from perception.ownership import *` raised AttributeError, and any binder
# call to `figure_ground_pair(det)` -- the SIGNED figure-ground edge this module
# exists to be the single reader of -- raised AttributeError too.  That these
# were lost rather than never written is visible in the body: `_neighbor(det)`,
# the reader `figure_ground_pair` needs in order to name the occluded side, was
# defined and had no caller at all.
# ---------------------------------------------------------------------------

#: One bin centre step of the signed slot.  `_q_signed` maps a true 0.0 to bin 4,
#: which decodes to +1/BO_BINS, so "no claim" is not readable as exactly zero:
#: anything within one bin step of zero is inside the quantiser's own resolution
#: and must be an abstention, not a weak vote for a side.
SIGNED_DEADZONE = 1.0 / BO_BINS


def ownership_strength(det: Any) -> float:
    """Magnitude of the ownership claim, in [0, 1].

    ``abs(ownership_signed(det))`` -- the SIGN of the decode is the claim (which
    flank owns the border) and the magnitude is how far from the dead centre of
    the slot it sits, i.e. the only confidence the quantised producer preserved.
    It is NOT a probability: ``ownership_probability`` is the calibrated reading
    and is fitted on the enclosure statistic, not on this magnitude (Session G
    measured the raw magnitude to carry no information about correctness, AUC
    0.43-0.53).
    """
    return abs(ownership_signed(det))


def combine_ownership_cues(
    cues: Mapping[str, Optional[float]],
    weights: Mapping[str, float],
) -> float:
    """Equal-authority mean of the signed ownership cues, in [-1, 1].

    The other combiner in this module (``combine_ownership_logodds``) weights
    the cues by the fitted ``BO_LOGODDS_WEIGHTS`` and squashes the sum; this one
    is the equal-authority mean its docstring refers to, for callers that must
    not inherit a fitted weighting.  ``weights`` is accepted and deliberately
    IGNORED so the two combiners are drop-in for one another -- a caller that
    passes a weighting gets the mean anyway, which is the point of naming it
    "equal authority".

    A ``None`` cue is omitted, never read as a 0.0 vote, so a cue that has no
    evidence does not dilute the ones that do.  With no voting cue at all the
    result is 0.0 (a non-vote).
    """
    votes = [max(-1.0, min(1.0, float(v)))
             for v in cues.values() if v is not None]
    if not votes:
        return 0.0
    return float(sum(votes) / len(votes))


def figure_ground_pair(det: Any) -> Optional[Tuple[Any, int]]:
    """The border as a SIGNED edge ``(other, sign)``, or None to abstain.

    The module contract: positive coupling binds the border to the figure it
    owns, negative coupling marks the ground / T-occlusion side.

    * ``signed > 0`` -- this detection's entity is the figure, so the named
      neighbour is the occluded side: ``(neighbor, -1)``.  This is the OCCLUSION
      edge, and it is the only place in the tree where the sign -1 is minted.
    * ``signed < 0`` -- the neighbour is the figure, so this detection's own
      entity is the ground side and the binding edge is ``(entity, +1)``.
    * ``|signed| <= SIGNED_DEADZONE`` -- inside one bin step of the slot's
      centre, i.e. the producer did not commit to a side: abstain (None).  A
      true 0.0 decodes to +1/BO_BINS, so testing for exact zero here would read
      every abstention as a weak vote for the entity's own side.
    * ``signed > 0`` with no neighbour named -- there is no other flank to bind
      the border to, so there is no edge to emit: abstain.

    No truth, no object class: the decode, the detection's own alpha and the
    ``neighbor`` role in its rho.
    """
    s = ownership_signed(det)
    if abs(s) <= SIGNED_DEADZONE:
        return None
    if s > 0.0:
        other = _neighbor(det)
        if other is None:
            return None
        return (other, -1)
    return (getattr(det, "alpha", None), +1)
