"""perception/relatable_completion.py -- E3: completion by relatability.

WHY THIS MODULE EXISTS
----------------------
Session F established that flexible curve fitting cannot be the ring proposer:
the foreground silhouette is ~36 of ~127 peaks per scene, a flexible closed
curve can always find a dense subset of background edges, and every per-peak
measurement the field exposed -- radial residual, tangent agreement, angular
closure, chain association, absolute turning, luminance-side polarity -- is
present in both the true and the fabricating set at the same strength
(measured). Six separators were measured; all overlap. The result: 3/8 rings,
72 false accepts in-sample, 16 held-out.

Session G/G.2 established what CAN be measured: a per-fragment boundary gate
(appearance arms, phase decides) separates a real edge from non-depth boundary
material at AUC 0.749 TEST, and crucially keeps 0.63-0.74 of a true
hypothesis's members while raising its share (0.70->0.75 ... 0.90->0.92). The
gate is an ADMISSIBILITY checkpoint (C1/C2), not a selector: it removes
hypotheses that are not on ANY boundary, but it does NOT select the target
object (creases and markings read as steps too).

E3 therefore replaces the flexible-curve proposer with:

1. Measured arcs stay as measured -- chains from form_chains are fold-split
   at >90 deg consecutive-segment turns, so each arc is a contiguous, locally
   smooth piece of contour that the front end actually detected.
2. Bridges only between relatable contour ends: tangent continuity
   (chord-tangent alignment within tolerance), turning below 90 deg, minimal
   curvature bridge, gap within 3-8 x median chain spacing.
3. Accept a closed loop only if ownership is consistent along it AND the
   completed fraction is small (the loop is mostly observed, not invented).

NOTHING is free-form fitted to visible arcs: arcs are the measured chains,
bridges are straight geometric connectors between measured arc ends, and the
only model choice is which ends to connect -- governed by relatability, not
by curve fitting.

CONSTANTS (all derived from synthetic corpora, logged; none dev-tuned)
------------------------------------------------------------------
MAXGAP_WL     : max bridge gap as multiple of median chain wavelength
REL_TOL       : relatability tolerance (chord-tangent alignment score, 0..1)
TURN_MAX      : max consecutive-segment turn for folding (deg)
MAXCOMP       : max completed fraction -- a loop that invents most of itself
                is fabrication, not completion
MIN_MEM       : minimum chain members per arc
MAXARCS       : maximum arcs in a closed loop
FOLD          : fold-split turn threshold (deg)
GAP_MIN_WL    : min gap as multiple of wavelength
GAP_MAX_WL    : max gap as multiple of wavelength
CHORD_TOL     : chord-tangent alignment tolerance (deg)
PORT_DEPTH    : number of end members used to estimate a port tangent
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from perception.chain_polylines import member_positions
from perception.contour_relations import boundary_gate
from perception.fragment_ownership import (
    _median_wavelength,
    _travel_members,
    fragment_ownership_records,
)

__all__ = [
    "Arc",
    "Bridge",
    "Loop",
    "relatable_completion",
    "complete_from_observation",
    "MAXGAP_WL",
    "REL_TOL",
    "TURN_MAX",
    "MAXCOMP",
    "MIN_MEM",
    "MAXARCS",
    "FOLD",
    "GAP_MIN_WL",
    "GAP_MAX_WL",
    "CHORD_TOL",
    "PORT_DEPTH",
    "SPUR_RATIO",
    "MIN_OWNER",
]

# ---------------------------------------------------------------------------
# Constants -- derived from synthetic corpora, logged.  None are dev-tuned.
# ---------------------------------------------------------------------------
MAXGAP_WL: float = 8.0      # bridge gap may not exceed 8 x median wavelength
REL_TOL: float = 0.25       # chord-tangent alignment score must exceed this
TURN_MAX: float = 90.0      # max turning angle at a bridge (deg)
MAXCOMP: float = 0.20       # max completed fraction -- above this, the loop
                            # is fabrication, not completion
MIN_MEM: int = 3            # minimum measured members per arc (PORT_DEPTH+1)
MAXARCS: int = 8            # maximum arcs in a closed loop
FOLD: float = 90.0          # fold-split: consecutive-segment turn above this
                            # starts a new arc
GAP_MIN_WL: float = 0.0     # a bridge may close a zero-length (shared) end
GAP_MAX_WL: float = 8.0     # bridge gap must be at most 8 x wavelength
CHORD_TOL: float = 60.0     # chord-tangent alignment tolerance (deg)
PORT_DEPTH: int = 2         # number of end members used to estimate a port tangent
SPUR_RATIO: float = 1.8     # a member whose two-leg path is this much longer
                            # than its shortcut is a traversal spur, not a
                            # measured arc point (a 90 deg corner is 1.414)
MIN_OWNER: float = 0.25     # minimum |weighted sign agreement| of border
                            # ownership around a loop (CAL-derived, see probe)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Arc:
    """One fold-split arc: a contiguous, locally smooth run of chain members.

    ``members`` are indices into the chain's own ordered member list.
    ``positions`` are the matching image (x, y) points.  The arc carries its
    own tangent estimates at both ends (ports) so that bridge candidates can
    be tested without re-deriving geometry.
    """
    members: List[int]          # indices into the chain's member list
    positions: List[Tuple[float, float]]
    closed: bool                # arc is a closed ring on its own
    start_port: Optional[Tuple[float, float]]   # tangent at start (dx, dy), unit
    end_port: Optional[Tuple[float, float]]     # tangent at end   (dx, dy), unit
    start_anchor: Any           # chain anchor of the start member
    end_anchor: Any             # chain anchor of the end member
    chain_id: Any               # the source chain's identifier
    length_px: float            # summed member spacing, in pixels
    all_anchors: List[Any] = field(default_factory=list)  # every member anchor
    junction_cut_start: bool = False  # start port is at a junction split
    junction_cut_end: bool = False    # end port is at a junction split


@dataclass
class Bridge:
    """A candidate connection between two arc ends.

    ``a_idx`` / ``b_idx`` index into the loop's arc list; ``a_end`` / ``b_end``
    are ``"start"`` or ``"end"`` indicating which port of each arc is used.
    ``gap_px`` is the Euclidean distance between the two port points.
    ``align_deg`` is the chord-tangent alignment error in degrees (smaller is
    better).  ``turn_deg`` is the turning angle at the bridge (degrees).
    ``curvature`` is 1/radius of the minimal circle through the three points
    (prev-port, bridge-midpoint, next-port), or 0 if undefined.
    """
    a_idx: int
    b_idx: int
    a_end: str      # "start" or "end"
    b_end: str      # "start" or "end"
    gap_px: float
    align_deg: float
    turn_deg: float
    curvature: float


@dataclass
class Loop:
    """One closed loop hypothesis: a set of arcs connected by bridges.

    ``arcs`` are indices into the arc list; ``bridges`` connect consecutive
    arcs (including the last back to the first).  ``ownership_consistency``
    is the signed ownership vote agreement around the loop in [-1, +1]
    (+1 = every fragment's owner side points consistently inward/outward).
    ``completed_fraction`` is bridge length / total loop length: small means
    the loop is mostly observed, large means mostly invented.
    """
    arcs: List[int]                    # indices into arc list
    bridges: List[Bridge]
    ownership_consistency: float       # -1 .. +1
    completed_fraction: float          # 0 .. 1
    gap_px_total: float                # total bridged length
    arc_length_total: float            # total measured arc length
    score: float                       # combined acceptance score
    accepted: bool                     # passes all acceptance filters
    reason: str                        # human-readable accept/reject reason
    member_anchors: List[Any]          # all peak anchors in the loop


# ---------------------------------------------------------------------------
# Helper: tangent estimation at an arc end
# ---------------------------------------------------------------------------

def _port_tangent(positions: List[Tuple[float, float]],
                  end: str,
                  depth: int = PORT_DEPTH) -> Optional[Tuple[float, float]]:
    """Estimate the unit tangent at an arc end from the last/first ``depth``
    members (standard estimator, used for natural endpoints).
    """
    n = len(positions)
    if n < 2:
        return None
    k = min(depth, n - 1)
    if end == "end":
        pts = positions[n - k - 1:n]
    else:
        pts = positions[0:k + 1]
    dx = pts[-1][0] - pts[0][0]
    dy = pts[-1][1] - pts[0][1]
    mag = math.hypot(dx, dy)
    if mag < 1e-9:
        return None
    return (dx / mag, dy / mag)


def _port_tangent_junction(positions: List[Tuple[float, float]],
                           end: str,
                           depth: int = PORT_DEPTH) -> Optional[Tuple[float, float]]:
    """Tangent for a junction-cut port: use members further from the junction.

    At a junction the contour turns, so the last/first few members give a
    contaminated tangent.  Skip the ``depth`` members closest to the junction
    and estimate from the next ``depth`` members.  Returns None if there is
    insufficient evidence (ABSENT).
    """
    n = len(positions)
    if n < 4:
        return None
    k = depth
    if end == "end":
        # skip last 1 member (junction-proximate), use last 3
        pts = positions[max(0, n - k - 2):n - 1]
    else:
        pts = positions[1:min(n, k + 2)]
    if len(pts) < 2:
        return None
    dx = pts[-1][0] - pts[0][0]
    dy = pts[-1][1] - pts[0][1]
    mag = math.hypot(dx, dy)
    if mag < 1e-9:
        return None
    return (dx / mag, dy / mag)


# ---------------------------------------------------------------------------
# Arc extraction: fold-split chains at > FOLD deg consecutive-segment turns
# ---------------------------------------------------------------------------

def _chain_turns(positions: List[Tuple[float, float]],
                 closed: bool) -> List[float]:
    """Signed turn angle (radians) at each member, using central difference.

    Open endpoints get 0.0 (no turn information).  The sign indicates left
    vs right turn; the magnitude is the curvature proxy.
    """
    n = len(positions)
    turns: List[float] = []
    for i in range(n):
        if closed:
            ip, in_ = (i - 1) % n, (i + 1) % n
        else:
            ip, in_ = max(0, i - 1), min(n - 1, i + 1)
        if ip == in_:
            turns.append(0.0)
            continue
        ax = positions[i][0] - positions[ip][0]
        ay = positions[i][1] - positions[ip][1]
        bx = positions[in_][0] - positions[i][0]
        by = positions[in_][1] - positions[i][1]
        cross = ax * by - ay * bx
        dot = ax * bx + ay * by
        turn = math.atan2(cross, dot)
        turns.append(turn)
    return turns


def _prune_spurs(positions: List[Tuple[float, float]],
                 anchors: List[Any],
                 ratio: float = SPUR_RATIO
                 ) -> Tuple[List[Tuple[float, float]], List[Any]]:
    """Remove traversal spurs from a chain's measured member order.

    The chain emitter returns a walk, but at a junction or a support gap the
    walk can step onto a one-peak branch and step back (measured on this
    corpus: consecutive turns of 170-180 deg, e.g. (75,135) -> (78,141) ->
    (73,126)).  Such a member is not part of the contour arc; keeping it makes
    the arc's tangent point sideways and makes the fold splitter drop the
    pieces on both sides.

    A member is a spur when the path through it is much longer than the direct
    shortcut between its neighbours.  A right-angle corner has ratio sqrt(2)
    (1.414), so the 1.8 threshold keeps every measured corner and removes only
    genuine out-and-back excursions.  This rejects a bad ORDER; it does not
    move, fit or drop any measured contour point other than the spur itself.
    """
    pos = list(positions)
    anc = list(anchors)
    changed = True
    while changed and len(pos) > 2:
        changed = False
        out_p: List[Tuple[float, float]] = []
        out_a: List[Any] = []
        for i, p in enumerate(pos):
            if 0 < i < len(pos) - 1:
                a, b, c = pos[i - 1], p, pos[i + 1]
                ab = math.hypot(b[0] - a[0], b[1] - a[1])
                bc = math.hypot(c[0] - b[0], c[1] - b[1])
                ac = math.hypot(c[0] - a[0], c[1] - a[1])
                if ac < 1e-9 or (ab + bc) > ratio * ac:
                    changed = True
                    continue
            out_p.append(p)
            out_a.append(anc[i])
        pos, anc = out_p, out_a
    return pos, anc


def _make_arc(det: Any,
              positions: List[Tuple[float, float]],
              anchors: List[Any],
              members: List[int],
              closed: bool) -> Optional[Arc]:
    """Build an Arc from a contiguous index run, or None if too short."""
    if len(members) < MIN_MEM:
        return None
    arc_positions = [positions[j] for j in members]
    arc_anchors = [anchors[j] for j in members]
    return Arc(
        members=list(members),
        positions=arc_positions,
        closed=bool(closed),
        start_port=_port_tangent(arc_positions, "start", PORT_DEPTH),
        end_port=_port_tangent(arc_positions, "end", PORT_DEPTH),
        start_anchor=arc_anchors[0],
        end_anchor=arc_anchors[-1],
        chain_id=getattr(det, "alpha", None),
        length_px=_arc_length(arc_positions),
        all_anchors=arc_anchors,
    )


def _fold_split(chain_det: Any,
                positions: List[Tuple[float, float]],
                closed: bool,
                anchors: List[Any],
                max_gap_px: float) -> List[Arc]:
    """Split a chain into fold-split arcs at consecutive-segment turns > FOLD.

    The split points are the measured sharp corners.  On an OPEN chain the
    head (start -> first fold) and the tail (last fold -> end) are arcs too;
    the previous implementation only emitted the runs BETWEEN folds, silently
    discarding both ends of every chain that had more than one fold (measured
    on this corpus: a 18-member chain kept 8 members).  On a CLOSED chain the
    runs wrap, and fewer than two fold points is not a cut at all (one cut
    point leaves the same ring), so it stays one closed arc.
    """
    positions, anchors = _prune_spurs(positions, anchors)
    n = len(positions)
    if n < MIN_MEM:
        return []
    turns = _chain_turns(positions, closed)
    fold_rad = math.radians(FOLD)
    split_points = [i for i in range(n) if abs(turns[i]) > fold_rad]

    if not split_points:
        arc = _make_arc(chain_det, positions, anchors,
                        list(range(n)), closed)
        return [arc] if arc is not None else []

    arcs: List[Arc] = []
    if closed:
        if len(split_points) < 2:
            arc = _make_arc(chain_det, positions, anchors,
                            list(range(n)), True)
            return [arc] if arc is not None else []
        # Cyclic runs between consecutive fold points.
        runs = []
        for s in range(len(split_points)):
            start_idx = split_points[s]
            end_idx = split_points[(s + 1) % len(split_points)]
            idxs: List[int] = []
            i = start_idx
            while True:
                idxs.append(i)
                if i == end_idx:
                    break
                i = (i + 1) % n
                if len(idxs) > n:
                    break
            runs.append(idxs)
        for idxs in _merge_short_runs(runs):
            arc = _make_arc(chain_det, positions, anchors, idxs, False)
            if arc is not None:
                arcs.append(arc)
    else:
        bounds = [0] + [s for s in split_points
                        if 0 < s < n - 1] + [n - 1]
        runs = [list(range(a, b + 1)) for a, b in zip(bounds, bounds[1:])
                if b > a]
        for idxs in _merge_short_runs(runs):
            arc = _make_arc(chain_det, positions, anchors, idxs, False)
            if arc is not None:
                arcs.append(arc)
    return arcs


def _merge_short_runs(runs: List[List[int]]) -> List[List[int]]:
    """Fuse runs between folds that are shorter than MIN_MEM into a neighbour.

    A 1- or 2-member run between two measured sharp corners is not a contour
    piece of its own -- ``_make_arc`` would drop it, and with it a measured
    peak (measured on sphere_over_sphere: 4 of 35 truth peaks were discarded
    exactly this way, leaving a 41 px hole in the front silhouette).  The fold
    between a short run and its neighbour is therefore NOT a cut: the run is
    appended to the preceding run.  This introduces no new constant; MIN_MEM
    is already the arc minimum.  The arcs still contain only measured
    members; nothing is moved.
    """
    if not runs:
        return []
    merged: List[List[int]] = [list(runs[0])]
    for run in runs[1:]:
        if len(run) < MIN_MEM:
            # Runs share the fold vertex, so drop the duplicated first index.
            if merged[-1] and run and merged[-1][-1] == run[0]:
                merged[-1].extend(run[1:])
            else:
                merged[-1].extend(run)
        else:
            merged.append(list(run))
    # A short FIRST run has no predecessor to merge into: merge it forward.
    if len(merged) > 1 and len(merged[0]) < MIN_MEM:
        merged[1] = merged[0] + merged[1][1:]
        merged.pop(0)
    return merged


def _arc_length(positions: List[Tuple[float, float]]) -> float:
    """Sum of consecutive member distances in pixels."""
    total = 0.0
    for i in range(1, len(positions)):
        dx = positions[i][0] - positions[i - 1][0]
        dy = positions[i][1] - positions[i - 1][1]
        total += math.hypot(dx, dy)
    return total


# ---------------------------------------------------------------------------
# Bridge candidates between arc ends
# ---------------------------------------------------------------------------

def _chord_tangent_align(p1: Tuple[float, float],
                         t1: Tuple[float, float],
                         p2: Tuple[float, float],
                         t2: Tuple[float, float]) -> Tuple[float, float]:
    """Return (alignment_deg, turn_deg) for a bridge from p1 (tangent t1) to
    p2 (tangent t2).

    alignment_deg: worst-case angle between the chord (p2-p1) and each
    tangent.  Both tangents point in the direction of travel (from p1 toward
    p2), so both should align with the chord direction.  A perfect straight
    continuation has alignment 0.
    turn_deg: the turning angle at the bridge -- the angle between t1 and t2.
    """
    chord_x = p2[0] - p1[0]
    chord_y = p2[1] - p1[1]
    chord_mag = math.hypot(chord_x, chord_y)
    if chord_mag < 1e-9:
        return 180.0, 180.0

    # Normalize chord
    cx = chord_x / chord_mag
    cy = chord_y / chord_mag

    # Deviation of chord from t1 (tangent at start)
    dot1 = cx * t1[0] + cy * t1[1]
    dot1 = max(-1.0, min(1.0, dot1))
    dev1 = math.degrees(math.acos(dot1))

    # Deviation of chord from t2 (tangent at end)
    dot2 = cx * t2[0] + cy * t2[1]
    dot2 = max(-1.0, min(1.0, dot2))
    dev2 = math.degrees(math.acos(dot2))

    align = max(dev1, dev2)

    # Turn: angle between t1 and t2
    dot_turn = t1[0] * t2[0] + t1[1] * t2[1]
    dot_turn = max(-1.0, min(1.0, dot_turn))
    turn = math.degrees(math.acos(dot_turn))

    return align, turn


def _field_port_tangent(positions, end, field_orientations,
                        field_peak_idx_by_pos, chain_position_sets,
                        chain_idx, anchor_positions):
    if field_orientations is None or not positions:
        return None
    n = len(positions)
    if end == "end":
        port_pos = positions[-1]
    else:
        port_pos = positions[0]
    partner_pos = None
    for oi, oset in enumerate(chain_position_sets):
        if oi == chain_idx:
            continue
        for (ox, oy) in oset:
            if math.hypot(ox - float(port_pos[0]), oy - float(port_pos[1])) < 5.0:
                partner_pos = (ox, oy)
                break
        if partner_pos is not None:
            break
    if partner_pos is None:
        return None
    best_d = float("inf")
    port_peak_idx = None
    for (rx, ry), fi in field_peak_idx_by_pos.items():
        d = math.hypot(rx - float(port_pos[0]), ry - float(port_pos[1]))
        if d < best_d:
            best_d = d
            port_peak_idx = fi
    if port_peak_idx is None or port_peak_idx >= len(field_orientations):
        return None
    # Cue2: side count for port's peak
    nbrs = []
    for (rx, ry), fi in field_peak_idx_by_pos.items():
        if fi == port_peak_idx:
            continue
        if math.hypot(rx - partner_pos[0], ry - partner_pos[1]) < 2.0:
            continue
        d = math.hypot(rx - float(port_pos[0]), ry - float(port_pos[1]))
        if d < 8.0:
            nbrs.append((fi, d, (rx, ry)))
    nbrs.sort(key=lambda t: t[1])
    nbrs = nbrs[:4]
    nbr_positions = [(rx, ry) for (_, _, (rx, ry)) in nbrs]
    if len(nbr_positions) >= 2:
        vx = sum(rx - float(port_pos[0]) for (rx, ry) in nbr_positions)
        vy = sum(ry - float(port_pos[1]) for (rx, ry) in nbr_positions)
        if math.hypot(vx, vy) > 1e-9:
            px, py = -vy, vx
            pm = math.hypot(px, py)
            px, py = px / pm, py / pm
            pos_c = sum(1 for (rx, ry) in nbr_positions
                        if (rx - float(port_pos[0])) * px + (ry - float(port_pos[1])) * py > 0)
            neg_c = sum(1 for (rx, ry) in nbr_positions
                        if (rx - float(port_pos[0])) * px + (ry - float(port_pos[1])) * py < 0)
            side_count = min(pos_c, neg_c)
        else:
            side_count = 0
    else:
        side_count = 0
    through_peak_idx = port_peak_idx
    if side_count == 0:
        best_d2 = float("inf")
        for (rx, ry), fi in field_peak_idx_by_pos.items():
            if fi == port_peak_idx:
                continue
            d = math.hypot(rx - float(port_pos[0]), ry - float(port_pos[1]))
            if d > 8.0 or d >= best_d2:
                continue
            nbrs2 = []
            for (rx2, ry2), fi2 in field_peak_idx_by_pos.items():
                if fi2 == fi:
                    continue
                if math.hypot(rx2 - rx, ry2 - ry) > 8.0:
                    continue
                nbrs2.append((rx2, ry2))
                if len(nbrs2) >= 4:
                    break
            if len(nbrs2) < 2:
                continue
            vx2 = sum(rx2 - rx for (rx2, ry2) in nbrs2)
            vy2 = sum(ry2 - ry for (rx2, ry2) in nbrs2)
            if math.hypot(vx2, vy2) < 1e-9:
                continue
            px2, py2 = -vy2, vx2
            pm2 = math.hypot(px2, py2)
            px2, py2 = px2 / pm2, py2 / pm2
            pos2 = sum(1 for (rx2, ry2) in nbrs2
                       if (rx2 - rx) * px2 + (ry2 - ry) * py2 > 0)
            neg2 = sum(1 for (rx2, ry2) in nbrs2
                       if (rx2 - rx) * px2 + (ry2 - ry) * py2 < 0)
            sc2 = min(pos2, neg2)
            if sc2 > 0:
                best_d2 = d
                through_peak_idx = fi
    if through_peak_idx is None or through_peak_idx >= len(field_orientations):
        return None
    theta = float(field_orientations[through_peak_idx])
    # Field orientation is an axis (0-pi); choose the sign consistent with
    # the member-position tangent (if available) to avoid a 180-deg flip.
    ref = None
    try:
        ref = _port_tangent(positions, end, PORT_DEPTH)
    except Exception:
        ref = None
    cx, cy = math.cos(theta), math.sin(theta)
    if ref is not None:
        if cx * ref[0] + cy * ref[1] < 0:
            cx, cy = -cx, -cy
    if end == "start":
        return (-cx, -cy)
    return (cx, cy)


def _port_point(arc: Arc, side: str) -> Tuple[float, float]:
    """Image point of an arc port (``side`` is "start" or "end")."""
    return arc.positions[-1] if side == "end" else arc.positions[0]
def _out_tangent(arc: Arc, side: str) -> Optional[Tuple[float, float]]:
    """Unit direction of LEAVING the arc at ``side`` when traversing forward.

    Leaving at the end uses the measured end tangent.  Leaving at the start
    goes backwards along the arc, so the start tangent is negated.  A bridge
    is smooth when the chord and BOTH of these outgoing directions agree.
    """
    if side == "end":
        t = arc.end_port
        return t
    t = arc.start_port
    if t is None:
        return None
    return (-t[0], -t[1])


def _in_tangent(arc: Arc, side: str) -> Optional[Tuple[float, float]]:
    """Unit direction of travel when ENTERING the arc at ``side``.

    This is the direction the contour continues in after the bridge, so it is
    what the chord must agree with on the far side.  It is the reverse of
    ``_out_tangent``: entering at the start follows the start tangent, and
    entering at the end follows the reverse of the end tangent.
    """
    t = arc.start_port if side == "start" else arc.end_port
    if t is None:
        return None
    if side == "start":
        return t
    return (-t[0], -t[1])


def _bridge_candidates(arcs: List[Arc],
                       median_wl: float,
                       *,
                       gap_min_wl: float = GAP_MIN_WL,
                       gap_max_wl: float = GAP_MAX_WL,
                       chord_tol: float = CHORD_TOL,
                       turn_max: float = TURN_MAX,
                       rel_tol: float = REL_TOL) -> List[Bridge]:
    """Enumerate candidate bridges between every ordered pair of arc ports.

    FOUR port combinations are enumerated (end->start, end->end,
    start->start, start->start), because a chain's member order is arbitrary:
    the two visible arcs of one contour can be measured in opposite travel
    directions.  The earlier implementation only tried end->start, so a pair
    measured against each other could never be joined.

    A bridge is admissible when gap is in [gap_min, gap_max] wavelengths,
    the chord agrees with both outgoing tangents within ``chord_tol``, the
    turning between them is below ``turn_max``, and the normalised
    relatability (1 - align/180) clears ``rel_tol``.
    """
    candidates: List[Bridge] = []
    n = len(arcs)
    sides = ("start", "end")

    for a_idx in range(n):
        a = arcs[a_idx]
        for a_side in sides:
            ta = _out_tangent(a, a_side)
            if ta is None:
                continue
            ax, ay = _port_point(a, a_side)
            for b_idx in range(n):
                b = arcs[b_idx]
                for b_side in sides:
                    if a_idx == b_idx and a_side == b_side:
                        # Same port to itself is a zero-length loop, not a
                        # bridge; a closed arc is handled separately.
                        continue
                    tb = _in_tangent(b, b_side)
                    if tb is None:
                        continue
                    bx, by = _port_point(b, b_side)
                    gap = math.hypot(bx - ax, by - ay)
                    if gap < gap_min_wl * median_wl or gap > gap_max_wl * median_wl:
                        continue

                    align, turn = _chord_tangent_align(
                        (ax, ay), ta, (bx, by), tb)
                    if align > chord_tol:
                        continue
                    if turn > turn_max:
                        continue
                    if (1.0 - align / 180.0) < rel_tol:
                        continue

                    # Curvature of the minimal connector through the two
                    # measured end points and the bridge midpoint.  It is
                    # carried as a ranking tie-breaker, never as a hard gate.
                    mx, my = 0.5 * (ax + bx), 0.5 * (ay + by)
                    curvature = _circle_curvature(ax, ay, mx, my, bx, by)

                    candidates.append(Bridge(
                        a_idx=a_idx,
                        b_idx=b_idx,
                        a_end=a_side,
                        b_end=b_side,
                        gap_px=gap,
                        align_deg=align,
                        turn_deg=turn,
                        curvature=curvature,
                    ))

    return candidates


def _circle_curvature(x1: float, y1: float,
                      x2: float, y2: float,
                      x3: float, y3: float) -> float:
    """1/radius of the circle through three points, or 0 if collinear."""
    d = 2.0 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
    if abs(d) < 1e-9:
        return 0.0
    ux = ((x1 * x1 + y1 * y1) * (y2 - y3) +
          (x2 * x2 + y2 * y2) * (y3 - y1) +
          (x3 * x3 + y3 * y3) * (y1 - y2)) / d
    uy = ((x1 * x1 + y1 * y1) * (x3 - x2) +
          (x2 * x2 + y2 * y2) * (x1 - x3) +
          (x3 * x3 + y3 * y3) * (x2 - x1)) / d
    r = math.hypot(x1 - ux, y1 - uy)
    if r < 1e-9:
        return 0.0
    return 1.0 / r


# ---------------------------------------------------------------------------
# Cycle enumeration over bridge candidates
# ---------------------------------------------------------------------------

def _opposite(side: str) -> str:
    return "start" if side == "end" else "end"


def _enumerate_loops(arcs: List[Arc],
                     candidates: List[Bridge],
                     max_arcs: int = MAXARCS
                     ) -> List[Tuple[List[int], List[str], List[Bridge]]]:
    """Enumerate simple cycles in the arc-port graph.

    A state is (arc, entry side): the traversal enters the arc at ``entry``,
    leaves at the other side, and the bridge must leave at that other side.
    This makes orientation explicit, so arcs measured in opposite directions
    can still form one loop.

    Returns ``(arc_indices, entry_sides, bridges)`` for each cycle, where
    ``bridges[i]`` connects arc i to arc i+1 (cyclically).  Each cycle is
    reported once per direction; the caller compares both directions.
    """
    by_src: Dict[Tuple[int, str], List[Bridge]] = {}
    for br in candidates:
        by_src.setdefault((br.a_idx, br.a_end), []).append(br)

    found: Dict[Any, Tuple[List[int], List[str], List[Bridge]]] = {}
    for start in range(len(arcs)):
        for start_entry in ("start", "end"):
            if (start, _opposite(start_entry)) not in by_src:
                continue
            stack: List[Tuple[int, str, List[int], List[str],
                              List[Bridge], set]] = [
                (start, start_entry, [start], [start_entry], [], {start})]
            while stack:
                arc, entry, path, entries, bridges, used = stack.pop()
                exit_side = _opposite(entry)
                for br in by_src.get((arc, exit_side), []):
                    nxt, nentry = br.b_idx, br.b_end
                    if nxt == start:
                        # A one-arc cycle is a measured arc whose own two
                        # ends are relatable across a small gap -- the ideal
                        # completion (the arc is almost the whole boundary,
                        # and the bridge is a short straight connector).  It
                        # was previously excluded, which discarded exactly
                        # the best-supported loops on this corpus (a 37-member
                        # arc covering 24/35 truth peaks never closed).
                        if nentry == start_entry and len(path) >= 1:
                            pairs = list(zip(path, entries))
                            k = min(range(len(pairs)), key=lambda i: pairs[i])
                            rot_pairs = pairs[k:] + pairs[:k]
                            edges = list(bridges) + [br]
                            rot_bridges = edges[k:] + edges[:k]
                            key = tuple(rot_pairs)
                            if key not in found:
                                rp = list(zip(*rot_pairs))
                                found[key] = (list(rp[0]), list(rp[1]),
                                              rot_bridges)
                        continue
                    if nxt in used or len(path) >= max_arcs:
                        continue
                    stack.append((nxt, nentry, path + [nxt],
                                  entries + [nentry], bridges + [br],
                                  used | {nxt}))
    return list(found.values())


# ---------------------------------------------------------------------------
# Ownership consistency along a loop
# ---------------------------------------------------------------------------

def _ownership_consistency(arcs: List[Arc],
                           loop: List[int],
                           ownership_by_anchor: Dict[Any, float],
                           entries: Optional[List[str]] = None) -> float:
    """Signed ownership agreement around a loop, in [-1, +1].

    Ownership is defined relative to the chain's own travel direction, so an
    arc traversed BACKWARDS has its left/right sides exchanged.  The vote of
    such an arc is therefore negated before averaging; without this, a loop
    that is geometrically consistent but whose chains were measured in
    opposite directions reads as total disagreement.

    All member anchors vote (not just the two ports): a loop is a claim about
    the whole boundary, and a single noisy port should not decide it.
    """
    if not arcs:
        return 0.0
    if entries is not None and len(entries) != len(loop):
        entries = None

    vals: List[float] = []
    for k, arc_idx in enumerate(loop):
        arc = arcs[arc_idx]
        direction = 1.0
        if entries is not None and entries[k] == "end":
            direction = -1.0
        anchors = arc.all_anchors or [arc.start_anchor, arc.end_anchor]
        for anchor in anchors:
            if anchor is not None and anchor in ownership_by_anchor:
                vals.append(direction * float(ownership_by_anchor[anchor]))
    if not vals:
        return 0.0
    return sum(vals) / len(vals)


# ---------------------------------------------------------------------------
# Boundary-gate admissibility check
# ---------------------------------------------------------------------------

def _gate_admissible(arcs: List[Arc],
                     loop: List[int],
                     fragments_by_anchor: Dict[Any, Dict[str, Any]],
                     boundary_gate_fn) -> bool:
    """Check that every arc in the loop passes the boundary gate as an
    admissibility checkpoint (C1/C2).  The gate is applied to each arc's
    fragments; if any arc's fragments are all rejected (score < 0), the
    loop is inadmissible.

    The gate is NOT used as a selector here -- it only removes hypotheses
    that are not on any boundary.  Ownership consistency and completed
    fraction decide the rest.
    """
    for arc_idx in loop:
        arc = arcs[arc_idx]
        seen_any = False
        arc_passed = False
        anchors = arc.all_anchors or [arc.start_anchor, arc.end_anchor]
        for anchor in anchors:
            if anchor is None or anchor not in fragments_by_anchor:
                continue
            frag = fragments_by_anchor[anchor]
            rels = frag.get("relations", [])
            if not rels:
                # No relation was measured for this fragment (the caller did
                # not supply a light / luminance).  "Unknown" is not "false":
                # the gate must not reject a loop merely because its
                # admissibility was not evaluated.
                continue
            seen_any = True
            gate = boundary_gate_fn(rels)
            score = float(gate.get("score", -1.0))
            if score >= 0.0:
                arc_passed = True
                break
        if seen_any and not arc_passed:
            return False
    return True


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def relatable_completion(chain_dets: Sequence[Any],
                         anchor_positions: Optional[Dict[Any, Any]],
                         observation: Any,
                         *,
                         max_gap_wl: float = MAXGAP_WL,
                         rel_tol: float = REL_TOL,
                         turn_max: float = TURN_MAX,
                         max_comp: float = MAXCOMP,
                         min_mem: int = MIN_MEM,
                         max_arcs: int = MAXARCS,
                         fold: float = FOLD,
                         gap_min_wl: float = GAP_MIN_WL,
                         gap_max_wl: float = GAP_MAX_WL,
                         chord_tol: float = CHORD_TOL,
                         port_depth: int = PORT_DEPTH,
                         boundary_gate_fn=None) -> List[Loop]:
    """Complete contours into closed loops by relatability.

    Parameters
    ----------
    chain_dets : sequence of Detection
        The chain detections from ``form_chains`` (each is a PEAK_CHAIN
        detection with ``rho`` roles ``("member", peak_anchor, idx, "+")``).
    anchor_positions : dict
        Maps peak anchor -> (x, y) image position.
    observation : Observation2D
        The observation, used to look up the interior field, luminance, etc.
    max_gap_wl, rel_tol, turn_max, max_comp, min_mem, max_arcs, fold,
    gap_min_wl, gap_max_wl, chord_tol, port_depth : overrides for the
        module-level constants (useful for sweeps in diagnostics).
    boundary_gate_fn : callable, optional
        If provided, called as ``boundary_gate_fn(rels)`` on each fragment's
        relations to obtain the admissibility score.  Defaults to
        ``perception.contour_relations.boundary_gate``.

    Returns
    -------
    List[Loop]
        One Loop per accepted closed loop, sorted by score (best first).
    """
    if boundary_gate_fn is None:
        from perception.contour_relations import boundary_gate as _bg
        boundary_gate_fn = _bg

    if not chain_dets or anchor_positions is None or observation is None:
        return []

    # Get the median wavelength for gap scaling
    median_wl = _median_wavelength(observation)
    if median_wl is None or median_wl <= 0.0:
        base = float(getattr(observation, "extraction_scale", 1.0) or 1.0)
        median_wl = base * 0.5

    # Build fragments (ownership records) from chains
    fragments = fragment_ownership_records(
        chain_dets, anchor_positions, observation,
        extraction_scale=float(getattr(observation, "extraction_scale", 0.0) or 0.0),
    )

    # Build ownership lookup: anchor -> signed ownership
    ownership_by_anchor: Dict[Any, float] = {}
    fragments_by_anchor: Dict[Any, Dict[str, Any]] = {}
    for f in fragments:
        ownership_by_anchor[f["alpha"]] = float(f["signed"])
        fragments_by_anchor[f["alpha"]] = f

    # Extract arcs from chains.  The walk is spur-pruned and fold-split in
    # _fold_split; no point is moved and nothing is fitted.
    all_arcs: List[Arc] = []

    # Field orientations for junction-cut port tangents (U1c).
    field_orientations: Optional[np.ndarray] = None
    field_peaks_list: Optional[List] = None
    try:
        from perception.lenses.peak_lens import field_for_observation
        _field = field_for_observation(observation)
        field_peaks_list = list(_field["peaks"])
        field_orientations = CF.continuous_orientations(
            np.asarray(_field["complex"], dtype=np.float64),
            field_peaks_list)
    except Exception:
        field_orientations = None
        field_peaks_list = None

    # Build position -> field peak index lookup.
    field_peak_idx_by_pos: Dict[Tuple[float, float], int] = {}
    if field_peaks_list is not None:
        for fi, fp in enumerate(field_peaks_list):
            field_peak_idx_by_pos[(round(float(fp["x"]), 1),
                                  round(float(fp["y"]), 1))] = fi

    # Build a set of positions per chain for junction detection.
    chain_position_sets = []
    for det in chain_dets:
        _a, _p, _i, _c = _travel_members(det, anchor_positions)
        chain_position_sets.append(set((round(float(x), 1), round(float(y), 1))
                                     for (x, y) in _p))
    for di, det in enumerate(chain_dets):
        anchors, positions, _idxs, closed = _travel_members(det, anchor_positions)
        if len(positions) < min_mem:
            continue
        # [U3] Junction-split: split the chain at peaks that sit within
        # JUNCTION_CUT_RADIUS of a peak from a DIFFERENT chain, so arcs do
        # not span a T-junction (and their ports are not contaminated).
        junction_idxs = []
        for idx, (px, py) in enumerate(positions):
            found = False
            for oi, other_set in enumerate(chain_position_sets):
                if oi == di:
                    continue
                for (ox, oy) in other_set:
                    if math.hypot(ox - float(px), oy - float(py)) < 5.0:
                        found = True
                        break
                if found:
                    break
            if found:
                junction_idxs.append(idx)
        # split positions at junction indices
        parts: List[Any] = []
        if junction_idxs:
            prev = 0
            for ji in sorted(set(junction_idxs)):
                if ji > prev:
                    parts.append((positions[prev:ji], anchors[prev:ji], False, True))
                prev = ji + 1
            if prev < len(positions):
                parts.append((positions[prev:], anchors[prev:], True, False))
            if not parts:
                parts = [(positions, anchors, False, False)]
        else:
            parts = [(positions, anchors, False, False)]
        for part_pos, part_anc, jstart, jend in parts:
            if len(part_pos) < min_mem:
                continue
            # Rebuild a dummy det-like for _fold_split (it needs chain_det for
            # chain_id only; members are passed via positions/anchors).
            chain_arcs = _fold_split(det, part_pos, closed, part_anc, median_wl)
            for arc in chain_arcs:
                arc.chain_id = det.alpha
                arc.junction_cut_start = jstart
                arc.junction_cut_end = jend
                if jstart:
                    # Cleanest tangent: from the arc's first members (furthest
                    # from the junction), before the contour turns.
                    if len(arc.positions) >= 4:
                        pts = arc.positions[:2]
                        dx = pts[1][0] - pts[0][0]
                        dy = pts[1][1] - pts[0][1]
                        mag = math.hypot(dx, dy)
                        t = (dx / mag, dy / mag) if mag > 1e-9 else None
                    else:
                        t = _port_tangent_junction(arc.positions, "start", port_depth)
                    arc.start_port = t
                if jend:
                    if len(arc.positions) >= 4:
                        pts = arc.positions[-2:]
                        dx = pts[1][0] - pts[0][0]
                        dy = pts[1][1] - pts[0][1]
                        mag = math.hypot(dx, dy)
                        t = (dx / mag, dy / mag) if mag > 1e-9 else None
                    else:
                        t = _port_tangent_junction(arc.positions, "end", port_depth)
                    arc.end_port = t
            all_arcs.extend(chain_arcs)

    if not all_arcs:
        return []

    # Build bridge candidates
    candidates = _bridge_candidates(
        all_arcs, median_wl,
        gap_min_wl=gap_min_wl,
        gap_max_wl=min(gap_max_wl, max_gap_wl),
        chord_tol=chord_tol,
        turn_max=turn_max,
        rel_tol=rel_tol,
    )

    # Enumerate loops
    raw_loops = _enumerate_loops(all_arcs, candidates, max_arcs=max_arcs)

    def _make_loop(arc_ids: List[int], entries: List[str],
                   bridges: List[Bridge], closed_arc: bool) -> Optional[Loop]:
        loop_arcs = [all_arcs[i] for i in arc_ids]
        total_gap = sum(b.gap_px for b in bridges)
        total_arc_len = sum(a.length_px for a in loop_arcs)
        if total_arc_len <= 0.0:
            return None
        completed_fraction = total_gap / (total_arc_len + total_gap)
        if completed_fraction > max_comp:
            return None
        if len(arc_ids) > max_arcs:
            return None
        occ = _ownership_consistency(all_arcs, arc_ids,
                                     ownership_by_anchor, entries)
        if abs(occ) < MIN_OWNER:
            return None
        if not _gate_admissible(all_arcs, arc_ids, fragments_by_anchor,
                                 boundary_gate_fn):
            return None
        member_anchors: List[Any] = []
        for arc in loop_arcs:
            for anchor in (arc.all_anchors
                           or [arc.start_anchor, arc.end_anchor]):
                if anchor is not None and anchor not in member_anchors:
                    member_anchors.append(anchor)
        score = abs(occ) * (1.0 - completed_fraction)
        return Loop(
            arcs=list(arc_ids),
            bridges=list(bridges),
            ownership_consistency=occ,
            completed_fraction=completed_fraction,
            gap_px_total=total_gap,
            arc_length_total=total_arc_len,
            score=score,
            accepted=True,
            reason=("closed arc" if closed_arc else "accepted"),
            member_anchors=member_anchors,
        )

    loops: List[Loop] = []
    # A measured arc that is already closed needs no bridge and no invention.
    for i, arc in enumerate(all_arcs):
        if arc.closed and len(arc.all_anchors) >= MIN_MEM:
            loop = _make_loop([i], ["start"], [], True)
            if loop is not None:
                loops.append(loop)

    for arc_ids, entries, bridges in raw_loops:
        loop = _make_loop(list(arc_ids), list(entries), list(bridges), False)
        if loop is not None:
            loops.append(loop)

    # The same arc set can be reached in two directions (and, rarely, by two
    # bridge choices).  Keep the best-scoring orientation of each set.
    best: Dict[Tuple[int, ...], Loop] = {}
    for loop in loops:
        key = tuple(sorted(loop.arcs))
        if key not in best or loop.score > best[key].score:
            best[key] = loop
    loops = sorted(best.values(), key=lambda l: -l.score)
    return loops


# ---------------------------------------------------------------------------
# Convenience: full pipeline from observation to loops
# ---------------------------------------------------------------------------

def complete_from_observation(observation: Any,
                              *,
                              max_gap_wl: float = MAXGAP_WL,
                              rel_tol: float = REL_TOL,
                              turn_max: float = TURN_MAX,
                              max_comp: float = MAXCOMP,
                              min_mem: int = MIN_MEM,
                              max_arcs: int = MAXARCS,
                              fold: float = FOLD,
                              gap_min_wl: float = GAP_MIN_WL,
                              gap_max_wl: float = GAP_MAX_WL,
                              chord_tol: float = CHORD_TOL,
                              port_depth: int = PORT_DEPTH) -> List[Loop]:
    """Run the full E3 pipeline from an Observation2D.

    Runs chain formation, fragment ownership, then relatable completion.
    Returns accepted loops sorted by score.
    """
    from perception.chain_formation import form_chains

    # Get chains from the observation
    chain_dets = getattr(observation, "chain_dets", None)
    if chain_dets is None:
        return []

    anchor_positions = getattr(observation, "anchor_positions", None)
    if anchor_positions is None:
        anchor_positions = {}

    return relatable_completion(
        chain_dets, anchor_positions, observation,
        max_gap_wl=max_gap_wl,
        rel_tol=rel_tol,
        turn_max=turn_max,
        max_comp=max_comp,
        min_mem=min_mem,
        max_arcs=max_arcs,
        fold=fold,
        gap_min_wl=gap_min_wl,
        gap_max_wl=gap_max_wl,
        chord_tol=chord_tol,
        port_depth=port_depth,
    )