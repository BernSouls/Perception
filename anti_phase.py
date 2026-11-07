"""perception/anti_phase.py -- negative (anti-phase) coupling sources.

The coupled-phase oscillator (dynamics/oscillator.py) separates two cliques
only when the coupling matrix can say NO: a negative edge weight drives the
phase difference toward pi instead of toward 0. Measured before this module
existed (diagnostics/_a6_maskfree_result.json, M4/A6 result): the observed
graph for two-panel composites carried 146 edges, 146 with sign=+1 and ZERO
anti-phase, 118 of them EDGE_CONTACT proximity noise -- so two-panel
separation could not be positive for a structural reason, not a tuning one.

Two sources, both geometric, both computable from the live field's own
detections:

  1. Border ownership -- opposite flank owners across a shared boundary.
     A BorderOwnership detection names two entities: the entity whose
     boundary it is (rho "entity") and the entity on the far side of that
     boundary (rho "neighbor"). Those two are the owners of the opposite
     flanks of one shared boundary, which is the figure-ground / occlusion
     relation, and it is a reason to separate rather than to bind.

  2. Containment -- closed-chain interior vs exterior. A PEAK_CHAIN whose
     closure slot is set and whose member ring encloses another
     detection's anchor separates that detection (interior) from the ring
     (exterior): the ring is one surface's silhouette, the enclosed
     structure sits behind it on another depth layer.

Both sources are pure functions of detections plus anchor positions, so
they are testable without an image, and deterministic: every iteration is
over sorted keys and every tie-break is explicit.

WHY THE DIRECTION SLOT IS READ THROUGH THE UNIFIED DECODER
---------------------------------------------------------
The BorderOwnership direction slot (quantized[1]) is a SIGNED figure-ground
scalar (`_q_signed(signed_dir, 8)`: bins 0-3 = the entity is the ground, bins
4-7 = the entity is the figure), NOT an angle. Three readers once disagreed:

  * perception/lenses/v2.py ContourLens: treated it as an 8-bin direction over
    2*pi, compared with an edge bearing (a category error, since fixed);
  * orchestration/binder.py `_run_border_ownership_edge`: treated it as an
    AXIAL angle over pi, `(bin + 0.5) * pi / 8` (a category error, since fixed);
  * orchestration/binder.py `_bo_depth_constraints`: `(bin + 0.5)/8*2 - 1`, a
    signed scalar -- the reading that matches the producer.

`perception/ownership.py` is now the SINGLE decoder for that slot; every reader
above, and this module's `border_ownership_claims_directed`, go through it, so
the conventions cannot drift again. The slot's polarity orders the flank pair
figure -> ground here; the earlier note that it "carries no side information"
described a corpus where no per-vertex cue voted on figure (all-positive z), not
the slot's contract.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

from perception.chain_polylines import (
    chain_is_closed,
    member_positions,
    order_ring_positions,
)

__all__ = [
    "border_ownership_claims",
    "border_ownership_claims_directed",
    "ring_polygon",
    "point_in_polygon",
    "containment_pairs",
]

#: Minimum border-ownership strength (|signed_dir|, already in [0, 1]) for a
#: flank claim to count. Calibrated by a stated rule (diagnostics/
#: _q3_strength.py, measured before this line was written): the p10 of |z|
#: over 552 BorderOwnership detections on 3 composites + 2 real images is
#: 0.200 (p50 0.214, max 0.347), and the rule is "p10 rounded down to two
#: decimals", so the threshold is 0.19 and about 90% of claims pass.
MIN_CLAIM_STRENGTH = 0.19
_MIN_RING_MEMBERS = 3

# [O4] THE CLAIM GATE, RE-DERIVED.
#
# The audit (W9) found that MIN_CLAIM_STRENGTH = 0.19 gates NOTHING:
# the ownership confidence spans 0.20 to 0.35 (p10 0.20, p50 0.214,
# max 0.347), so ~90% of claims pass. The cause is mechanism-level, not
# a bad threshold: the three-vote log-odds (convexity 2.18, enclosure
# 2.08, T-junction 1.51) saturate tanh near zero whenever the cues
# disagree, compressing the output to a 0.15-wide band.
#
# Three distinct fixes were tried:
#   (a) RESCALE THE CUE WEIGHTS so tanh does not saturate. Rejected: the
#       weights are FITTED (parameters/_ownership_weights.json, logistic
#       regression on the synthetic corpus) and re-fitting them is a
#       corpus-level change outside this session's scope. The fitted
#       weights are the system's evidence; overriding them by hand would
#       be fabrication.
#   (b) NORMALISE THE CONFIDENCE to its empirical range (z-score against
#       the image's own distribution). Rejected: it makes the gate
#       image-dependent in a way that a single-fragment image (one claim)
#       cannot satisfy, and it hides the degeneracy instead of exposing
#       it.
#   (c) A RELATIVE GATE: the threshold is the MEDIAN of the image's own
#       |z| distribution, so ~50% of claims pass and the gate actually
#       gates. This is derived from the data, not hand-picked, and it
#       exposes the degeneracy (the median IS the p50 = 0.214) rather
#       than hiding it. Adopted.
#
# The floor MIN_CLAIM_STRENGTH is kept as a FALLBACK for the degenerate
# case (a single claim, or all claims identical): when the median cannot
# be computed, the floor applies. The unblock condition for (a) is a
# re-fit of the ownership weights on a corpus with DISAGREEING cues, so
# the log-odds span widens; until then (c) is the honest gate.
def derive_claim_threshold(bo_dets, floor: float = MIN_CLAIM_STRENGTH):
    """The median of the image's own |z| distribution, or the floor."""
    zs = sorted(abs(float(getattr(d, "z", 0.0) or 0.0)) for d in bo_dets)
    if len(zs) < 2:
        return float(floor)
    mid = len(zs) // 2
    if len(zs) % 2 == 0:
        med = 0.5 * (zs[mid - 1] + zs[mid])
    else:
        med = zs[mid]
    return float(max(floor, med))

def _claim_endpoints(det: Any) -> Optional[Tuple[Any, Any]]:
    """(entity_anchor, neighbour_anchor) of a BorderOwnership detection."""
    ent = getattr(det, "alpha", None)
    nb = None
    for role in getattr(det, "rho", ()) or ():
        if len(role) >= 2 and role[0] == "neighbor":
            nb = role[1]
            break
    if ent is None or nb is None or ent == nb:
        return None
    return ent, nb


def _claim_endpoints(det: Any) -> Optional[Tuple[Any, Any]]:
    """(entity_anchor, neighbour_anchor) of a BorderOwnership detection."""
    ent = getattr(det, "alpha", None)
    nb = None
    for role in getattr(det, "rho", ()) or ():
        if len(role) >= 2 and role[0] == "neighbor":
            nb = role[1]
            break
    if ent is None or nb is None or ent == nb:
        return None
    return ent, nb


def border_ownership_claims(
    bo_dets: Sequence[Any],
    min_strength: Optional[float] = None,
) -> List[Tuple[Any, Any, float]]:
    """Anti-phase pairs from opposite flank owners across a shared boundary.

    A BorderOwnership detection names the entity whose boundary it is (its
    anchor) and the entity on the far side of that boundary (rho
    ``neighbor``). Those two are the owners of the opposite flanks of one
    shared boundary, which separates them: the figure side stands in front
    of the ground side. The detection's ``z`` is the magnitude of the
    evidence sum, so it is a confidence, and it is the ONLY quantity this
    rule uses -- the direction slot is a signed evidence scalar, not an
    angle (module docstring), so it is deliberately not consulted.

    Multiple vertices of the same boundary are combined per ordered pair by
    maximum strength, so an over-sampled boundary cannot inflate a pair.
    Pairs are returned in deterministic (repr, repr) order.
    """
    if min_strength is None:
        min_strength = derive_claim_threshold(bo_dets)
    directed: Dict[Tuple[Any, Any], float] = {}
    for det in bo_dets:
        pair = _claim_endpoints(det)
        if pair is None:
            continue
        strength = abs(float(getattr(det, "z", 0.0) or 0.0))
        if strength < float(min_strength):
            continue
        a, b = pair
        prev = directed.get((a, b))
        if prev is None or strength > prev:
            directed[(a, b)] = strength
    out = []
    for (a, b), s in directed.items():
        lo, hi = (a, b) if repr(a) < repr(b) else (b, a)
        out.append((lo, hi, s))
    out.sort(key=lambda t: (repr(t[0]), repr(t[1])))
    return out


def border_ownership_claims_directed(
    bo_dets: Sequence[Any],
    min_strength: Optional[float] = None,
) -> List[Tuple[Any, Any, float]]:
    """Directed (figure_anchor, ground_anchor, strength) flank claims.

    Same evidence and strength gating as `border_ownership_claims`, but the
    pair is ORDERED by the figure-ground polarity read through the unified
    decoder (perception/ownership.py `ownership_signed`): the entity is the
    figure when the signed value is >= 0, otherwise the named neighbour is.
    Exactly one direction per unordered pair survives -- the one with the
    greater total evidence, ties broken by repr order -- so the emitted
    OCCLUSION edge is figure -> ground (near -> far) with no contradictory
    back-edge.
    """
    from perception.ownership import ownership_signed
    if min_strength is None:
        min_strength = derive_claim_threshold(bo_dets)
    if min_strength is None:
        min_strength = derive_claim_threshold(bo_dets)
    votes: Dict[Tuple[Any, Any], float] = {}
    for det in bo_dets:
        pair = _claim_endpoints(det)
        if pair is None:
            continue
        strength = abs(float(getattr(det, "z", 0.0) or 0.0))
        if strength < float(min_strength):
            continue
        ent, nb = pair
        fg = (ent, nb) if ownership_signed(det) >= 0.0 else (nb, ent)
        votes[fg] = votes.get(fg, 0.0) + strength
    best: Dict[Tuple[Any, Any], Tuple[Any, Any, float]] = {}
    for (a, b), s in votes.items():
        key = (a, b) if repr(a) < repr(b) else (b, a)
        cur = best.get(key)
        if (cur is None or s > cur[2]
                or (s == cur[2]
                    and repr((a, b)) < repr((cur[0], cur[1])))):
            best[key] = (a, b, s)
    out = list(best.values())
    out.sort(key=lambda t: (repr(t[0]), repr(t[1])))
    return out

def ring_polygon(
    det: Any,
    anchor_positions: Optional[Dict[Any, Any]],
    min_area: float = 0.0,
) -> Optional[List[Tuple[float, float]]]:
    """Ordered polygon of a closed PEAK_CHAIN, or None when it is not a ring.

    The chain's rho order is a valid traversal for closed chains emitted by
    form_chains' spine walker, but the ring flag (closure slot) is the
    authority: a chain without it is never a container, however its members
    happen to be ordered.
    """
    q = getattr(getattr(det, "sigma", None), "quantized", None)
    if not q or len(q) < 4 or not anchor_positions:
        return None
    # [F1] ONE reader of the closure slot.
    if not chain_is_closed(det):
        return None
    pts = member_positions(det, anchor_positions)
    if len(pts) < _MIN_RING_MEMBERS:
        return None
    order = order_ring_positions(list(pts))
    if order:
        poly = [pts[i] for i in order]
    else:
        poly = list(pts)
    area = 0.0
    n = len(poly)
    for i in range(n):
        x0, y0 = poly[i]
        x1, y1 = poly[(i + 1) % n]
        area += x0 * y1 - x1 * y0
    if 0.5 * abs(area) < float(min_area):
        return None
    return poly


def point_in_polygon(pt: Tuple[float, float],
                     poly: Sequence[Tuple[float, float]]) -> bool:
    """Even-odd ray cast, boundary points excluded."""
    x, y = float(pt[0]), float(pt[1])
    inside = False
    n = len(poly)
    for i in range(n):
        x0, y0 = poly[i]
        x1, y1 = poly[(i + 1) % n]
        if (y0 > y) != (y1 > y):
            xin = x0 + (y - y0) * (x1 - x0) / (y1 - y0)
            if x < xin:
                inside = not inside
    return inside


def _ring_scale(poly: Sequence[Tuple[float, float]]) -> float:
    cx = sum(p[0] for p in poly) / len(poly)
    cy = sum(p[1] for p in poly) / len(poly)
    rms = math.sqrt(sum((p[0] - cx) ** 2 + (p[1] - cy) ** 2
                        for p in poly) / len(poly))
    return rms


def containment_pairs(
    chain_dets: Sequence[Any],
    other_dets: Sequence[Any],
    anchor_positions: Optional[Dict[Any, Any]],
    min_area: float = 0.0,
) -> List[Tuple[Any, Any, float]]:
    """(ring_anchor, contained_anchor, depth) for closed-chain containment.

    ``depth`` is exp(-(distance from the ring centre) / (ring RMS radius)),
    so a structure at the centre of a ring is strongly contained and one
    hugging the boundary is weakly contained. Anchors that are members of
    the ring itself are excluded (a ring does not contain its own contour).
    """
    if not anchor_positions:
        return []
    rings = []
    for det in chain_dets:
        poly = ring_polygon(det, anchor_positions, min_area=min_area)
        if poly is None:
            continue
        members = {role[1] for role in (getattr(det, "rho", ()) or ())
                   if len(role) >= 2 and role[0] == "member"}
        rings.append((getattr(det, "alpha", None), poly, members))
    if not rings:
        return []
    out: List[Tuple[Any, Any, float]] = []
    for ring_anchor, poly, members in rings:
        scale = _ring_scale(poly)
        if scale <= 1e-9:
            continue
        cx = sum(p[0] for p in poly) / len(poly)
        cy = sum(p[1] for p in poly) / len(poly)
        for det in other_dets:
            anchor = getattr(det, "alpha", None)
            if anchor is None or anchor == ring_anchor or anchor in members:
                continue
            pt = anchor_positions.get(anchor)
            if pt is None or len(pt) < 2:
                continue
            if not point_in_polygon((float(pt[0]), float(pt[1])), poly):
                continue
            d = math.hypot(float(pt[0]) - cx, float(pt[1]) - cy)
            depth = math.exp(-d / scale)
            out.append((ring_anchor, anchor, depth))
    out.sort(key=lambda t: (repr(t[0]), repr(t[1])))
    return out
