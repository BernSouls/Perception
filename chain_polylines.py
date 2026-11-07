"""perception/chain_polylines.py -- [W1] Ordered polylines from PEAK_CHAINs.

The old eye's Shape family (RegionShapeLens, HolisticShapeLens,
TextureBoundaryLens, BorderOwnershipLens, SurfaceLens) read closed
polylines produced by the segmenter (`observation.get_boundary_polyline`).
That made shape identity depend on a boundary tracer that the new
field-based front end does not run.

This module recovers the same object -- an ORDERED polyline plus a
curvature profile -- directly from the chain structures the field
already produces, so the Shape family can be rewired without the
segmenter.

Why chains and not raw peaks
----------------------------
An unordered point set has no intrinsic visit order, so any polyline drawn
through it is an artefact of the drawing routine. `form_chains` has already
solved this: it orders peaks along the ridge and reports closure. Reusing
that order is what makes the polyline -- and therefore the shape
descriptor computed on it -- deterministic and physically meaningful
rather than a cosmetic re-sort.

Neural analogue
---------------
A contour's trace is represented in V1/V4 as an ordered sequence of
orientation-tagged edge fragments, not as an unordered scatter. Keeping the
order and deriving curvature from consecutive turn angles is the standard
curvature-tuned description of that trace.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import parameters as P

# a chain polyline with fewer points than this cannot carry a shape: 3 is
# the minimum for a non-degenerate turn angle at the middle vertex.
CHAIN_MIN_POLYLINE = 3

# curvature of a 3-point arc is 2*sin(turn/2)/|chord|; a straight run has
# turn == 0. Bins span [0, CURV_CLIP] as a graded, unsigned quantity.
CURV_CLIP = 0.35
N_CURV_BINS = 8

# [SPEED] Memo of the ring traversal, keyed by the exact (points, indices)
# pair. The traversal is a pure function of that pair, and the four
# callers ask for the same ring several times per closed chain. The cache
# is bounded; the oldest entry is evicted first.
_RING_ORDER_CACHE: Dict[Any, Optional[List[int]]] = {}
_RING_ORDER_CACHE_MAX = 256
_RING_ORDER_MISS = object()


def member_positions(det: Any,
                     anchor_positions: Optional[Dict[Any, Any]]
                     ) -> List[Tuple[float, float]]:
    """Peak (x, y) for each member of `det`, in the chain's own order.

    Chain membership is recorded in `det.rho` as ("member", peak_anchor,
    index_in_chain, "+"). The rho order IS the chain order; the third slot
    repeats it so a consumer can sort or verify without re-parsing.
    Position is read from `anchor_positions`, which the lenses already
    maintain, so no separate index table is needed.
    """
    if det is None or anchor_positions is None:
        return []
    out: List[Tuple[float, float]] = []
    for role in getattr(det, "rho", ()) or ():
        if len(role) < 2 or role[0] != "member":
            continue
        pos = anchor_positions.get(role[1])
        if pos is None:
            continue
        try:
            x, y = float(pos[0]), float(pos[1])
        except (TypeError, IndexError, ValueError):
            continue
        out.append((x, y))
    return out


def chain_polyline(det: Any,
                   anchor_positions: Optional[Dict[Any, Any]],
                   min_points: int = CHAIN_MIN_POLYLINE) -> Optional[
                       List[Tuple[float, float]]]:
    """Ordered polyline for one PEAK_CHAIN detection, or None if unusable.

    A chain with fewer than `min_points` ordered members is rejected: the
    turn-angle sequence would be empty or single-valued, so the descriptor
    would be constant and would add a slot that carries no information.
    Closure is NOT used to close the ring here -- the caller decides
    whether to append the first vertex (see `close_polyline`).
    """
    pts = member_positions(det, anchor_positions)
    if len(pts) < min_points:
        return None
    if any(not (math.isfinite(p[0]) and math.isfinite(p[1])) for p in pts):
        return None
    return pts


def close_polyline(pts: List[Tuple[float, float]],
                   closed: bool = True) -> List[Tuple[float, float]]:
    """Append the first vertex when `closed`, making a ring walkable.

    A ring-shaped contour is analysed as a cycle, so its polyline must
    return to the start. Guarded against degenerate zero-length rings and
    against duplicating a vertex that is already present.
    """
    if not closed or len(pts) < 3:
        return pts
    first, last = pts[0], pts[-1]
    if abs(first[0] - last[0]) < 1e-9 and abs(first[1] - last[1]) < 1e-9:
        return pts
    return list(pts) + [first]


def turn_angles(pts: List[Tuple[float, float]]) -> List[float]:
    """Signed turn angle at every interior vertex, in (-pi, pi].

    The turn at vertex i is the angle between the incoming and outgoing
    segments, signed by the 2-D cross product so left and right turns are
    distinguishable. This is the standard discrete curvature of a polyline
    and is what a curvature-tuned cortical unit responds to.
    """
    out: List[float] = []
    for i in range(1, len(pts) - 1):
        ax, ay = pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]
        bx, by = pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1]
        cross = ax * by - ay * bx
        dot = ax * bx + ay * by
        out.append(math.atan2(cross, dot))
    return out


def signed_curvature_sequence(pts: List[Tuple[float, float]]) -> List[float]:
    """Per-vertex curvature 2*sin(turn/2)/|chord|, signed.

    Normalising the raw turn angle by the local chord length keeps the
    measure comparable across chain lengths, so a 3-peak tight corner and
    a 9-peak tight corner land in the same bin.
    """
    out: List[float] = []
    for i in range(1, len(pts) - 1):
        ax = pts[i][0] - pts[i - 1][0]
        ay = pts[i][1] - pts[i - 1][1]
        bx = pts[i + 1][0] - pts[i][0]
        by = pts[i + 1][1] - pts[i][1]
        chord = math.hypot(ax + bx, ay + by)
        if chord <= 1e-9:
            continue
        turn = math.atan2(ax * by - ay * bx, ax * bx + ay * by)
        out.append(2.0 * math.sin(turn / 2.0) / chord)
    return out


def curvature_histogram(curvs: List[float], n_bins: int = N_CURV_BINS,
                        clip: float = CURV_CLIP) -> Tuple[float, ...]:
    """Graded bin histogram of per-vertex curvatures, normalised to sum 1.

    Uses signed curvature split about zero so convex and concave arcs are
    not merged into one "curvy" bucket -- the sign is what distinguishes
    them, and merging is exactly the information loss this replaced.
    """
    if not curvs:
        return (0.0,) * (2 * n_bins)
    step = clip / float(n_bins)
    hist = [0.0] * (2 * n_bins)

    def idx(v: float) -> int:
        q = int(abs(v) / step)
        return min(n_bins - 1, max(0, q))

    for c in curvs:
        if not math.isfinite(c):
            continue
        v = min(clip, max(-clip, c))
        hist[idx(v) if v >= 0.0 else n_bins + idx(v)] += 1.0
    total = sum(hist)
    if total <= 0.0:
        return (0.0,) * (2 * n_bins)
    return tuple(h / total for h in hist)



def centroid_and_area(pts: List[Tuple[float, float]],
                      closed: bool = False) -> Tuple[float, float, float]:
    """(cx, cy, area) of the polyline; area is 0 unless closed.

    Shoelace formula for a closed ring; open chains have no interior, so
    reporting a nonzero area for one would be fabricated extent.
    """
    if len(pts) < 2:
        return (0.0, 0.0, 0.0)
    cx = sum(p[0] for p in pts) / len(pts)
    cy = sum(p[1] for p in pts) / len(pts)
    if not closed or len(pts) < 3:
        return (cx, cy, 0.0)
    acc = 0.0
    for i in range(len(pts) - 1):
        acc += pts[i][0] * pts[i + 1][1] - pts[i + 1][0] * pts[i][1]
    return (cx, cy, abs(acc) / 2.0)


def member_entries(det: Any,
                   anchor_positions: Optional[Dict[Any, Any]]
                   ) -> List[Tuple[int, Tuple[float, float]]]:
    """(peak_index, (x, y)) per member, in the chain's own rho order.

    `peak_index` is the peak's index in the FIELD's peak list, recovered
    from the anchor as `anchor - PEAK_ANCHOR_BASE`. That distinction
    matters: the chain's rho order can arrive permuted, so positional index
    in this list is arbitrary and must not be used for any tie-break. The
    field peak index is the stable identity.
    """
    if det is None or anchor_positions is None:
        return []
    out: List[Tuple[int, Tuple[float, float]]] = []
    base = int(P.PEAK_ANCHOR_BASE)
    for role in getattr(det, "rho", ()) or ():
        if len(role) < 2 or role[0] != "member":
            continue
        pos = anchor_positions.get(role[1])
        if pos is None:
            continue
        try:
            x, y = float(pos[0]), float(pos[1])
        except (TypeError, IndexError, ValueError):
            continue
        try:
            idx = int(role[1]) - base
        except (TypeError, ValueError):
            idx = len(out)
        out.append((idx, (x, y)))
    return out


def member_positions(det: Any,
                     anchor_positions: Optional[Dict[Any, Any]]
                     ) -> List[Tuple[float, float]]:
    """Peak (x, y) for each member of `det`, in the chain's own order."""
    return [p for _idx, p in member_entries(det, anchor_positions)]


def order_ring_positions(pts: List[Tuple[float, float]],
                         indices: Optional[List[int]] = None
                         ) -> Optional[List[int]]:
    """Traversal order for a closed ring of points, or None if not a ring.

    Why this exists
    ---------------
    `form_chains` reports rings as `sorted(members)` -- index order -- because
    a ring is "closed" and its contract does not need a visit order. But a
    polyline built from index order is NOT the contour: for the existing
    24-site test ring, index order gives steps averaging 25.6 px with a
    39.7 px maximum, against a true arc chord of ~5.3 px. That is a
    zig-zag through the ring, and every shape slot computed on it (area
    through the shoelace, turn angles, curvature histogram) is then
    describing an artefact of member numbering rather than the contour.

    So the ordering is done here, at the geometry layer, where traversal is
    actually meaningful, and `form_chains`' documented D5 contract is left
    alone.

    Method (no thresholds)
    ----------------------
    Each point's two NEAREST neighbours are taken as its ring adjacency.
    That is parameter-free: there is no distance cutoff to tune, and it
    reduces exactly to ring topology for any ring sampled finely enough to
    be a ring at all. Traversal starts at the lowest point under the same
    (x, y, ...) key form `form_chains` uses for D7 tie-breaking, and the
    first step goes toward the lower-index of that point's two neighbours.

    `indices` supplies the FIELD peak index of each entry, used for that
    tie-break. It defaults to list position, which is only meaningful when
    the caller is passing a list already in a stable order -- callers that
    can permute the list must pass the real indices so the break is a
    function of the data rather than of arrival order. Later steps continue
    the cycle; where two unvisited continuations tie, the lower index wins,
    keeping the order deterministic.

    [SPEED] The traversal is a PURE function of (points, indices), and the
    four callers that need it (chain_formation, chain_flanks,
    fragment_ownership, anti_phase) ask for the SAME ring of the SAME chain
    up to four times per closed chain. The result is therefore memoised on
    the exact (points, indices) pair. The key is the point coordinates
    THEMSELVES, not their identity, so a re-derived chain with the same
    geometry hits the cache; the returned list is a copy, so a consumer that
    mutated it could not corrupt the cache. The cache is bounded and
    FIFO-evicted.
    """
    n = len(pts)
    if n < 3:
        return None
    if indices is None:
        indices = list(range(n))
    if len(indices) != n:
        return None

    key = (tuple((float(p[0]), float(p[1])) for p in pts),
           tuple(int(i) for i in indices))
    cached = _RING_ORDER_CACHE.get(key, _RING_ORDER_MISS)
    if cached is not _RING_ORDER_MISS:
        return list(cached) if cached is not None else None

    order = _order_ring_positions_uncached(pts, indices)
    if len(_RING_ORDER_CACHE) >= _RING_ORDER_CACHE_MAX:
        # FIFO eviction: drop the oldest entry.
        for k in list(_RING_ORDER_CACHE.keys())[:1]:
            _RING_ORDER_CACHE.pop(k, None)
    _RING_ORDER_CACHE[key] = order
    return list(order) if order is not None else None


def _order_ring_positions_uncached(
    pts: List[Tuple[float, float]], indices: List[int]
) -> Optional[List[int]]:
    """The traversal itself (see `order_ring_positions` for the method)."""
    n = len(pts)

    neigh: List[List[int]] = []
    for i in range(n):
        d = sorted(
            ((math.hypot(pts[i][0] - pts[j][0], pts[i][1] - pts[j][1]), j)
             for j in range(n) if j != i)
        )
        if len(d) < 2:
            return None
        neigh.append([j for _dist, j in d[:2]])

    # degree must be exactly 2 everywhere, else it is not a simple ring
    for i in range(n):
        if len(neigh[i]) != 2:
            return None
    # and the adjacency must be symmetric: a mutual nearest-neighbour pair
    for i in range(n):
        for j in neigh[i]:
            if i not in neigh[j]:
                return None

    start = min(range(n), key=lambda k: (pts[k][0], pts[k][1], indices[k]))
    a, b = neigh[start]
    first = min((a, b), key=lambda m: indices[m])
    order = [start, first]
    seen = {start, first}
    while len(order) < n:
        cur = order[-1]
        prev = order[-2]
        cands = [j for j in neigh[cur] if j != prev and j not in seen]
        if not cands:
            return _angular_ring_order(pts, indices)
        nxt = min(cands, key=lambda m: indices[m])
        order.append(nxt)
        seen.add(nxt)
    if len(order) != n:
        return _angular_ring_order(pts, indices)
    return order


def _polygon_is_simple(pts: List[Tuple[float, float]],
                       order: List[int]) -> bool:
    """True when the closed polygon visits distinct points and no two
    non-adjacent edges cross. Parameter-free; O(n^2), n is a chain."""
    m = len(order)
    if m < 3:
        return False
    for i in range(m):
        if pts[order[i]] == pts[order[(i + 1) % m]]:
            return False
    for i in range(m):
        a0 = pts[order[i]]
        a1 = pts[order[(i + 1) % m]]
        for j in range(i + 1, m):
            if j == i or (j + 1) % m == i or (i + 1) % m == j:
                continue            # same or adjacent edge
            b0 = pts[order[j]]
            b1 = pts[order[(j + 1) % m]]
            if _segments_cross(a0, a1, b0, b1):
                return False
    return True


def _segments_cross(p: Tuple[float, float], q: Tuple[float, float],
                    r: Tuple[float, float], s: Tuple[float, float]) -> bool:
    def orient(a, b, c):
        v = ((b[0] - a[0]) * (c[1] - a[1])
             - (b[1] - a[1]) * (c[0] - a[0]))
        if v > 1e-9:
            return 1
        if v < -1e-9:
            return -1
        return 0
    o1 = orient(p, q, r)
    o2 = orient(p, q, s)
    o3 = orient(r, s, p)
    o4 = orient(r, s, q)
    return (o1 * o2 < 0) and (o3 * o4 < 0)


def _angular_ring_order(pts: List[Tuple[float, float]],
                        indices: List[int]) -> Optional[List[int]]:
    """Fallback traversal for a star-shaped ring whose sampling is uneven.

    The two-nearest-neighbour test above is exact only for evenly sampled
    rings: one member whose own two nearest neighbours are pulled off the
    contour makes the mutual-neighbour graph asymmetric and the whole ring
    unorderable (measured: a 37-member front silhouette, 5 asymmetric
    vertices).  A simple closed curve around a compact region is star-shaped
    about its centroid, so ordering by angle about the centroid IS the
    contour order there.

    The order is accepted only when it is demonstrably a ring and not an
    invented one: distinct points, a simple (non-self-intersecting) polygon,
    strictly positive enclosed area, and total turning +/-2*pi (Hopf
    Umlaufsatz).  A straight run (zero area), a figure-eight (coincident
    point / crossing edges) and a degenerate triangle all fail those checks,
    which are the same shape facts ``_total_turning_closes`` uses; no
    distance threshold is introduced.
    """
    n = len(pts)
    if n < 3:
        return None
    if len(set(pts)) != n:
        return None                     # coincident points: not a simple ring
    cx = sum(p[0] for p in pts) / n
    cy = sum(p[1] for p in pts) / n
    order = sorted(range(n), key=lambda i: (
        math.atan2(pts[i][1] - cy, pts[i][0] - cx), indices[i]))
    if not _polygon_is_simple(pts, order):
        return None
    area2 = 0.0
    for i in range(n):
        x0, y0 = pts[order[i]]
        x1, y1 = pts[order[(i + 1) % n]]
        area2 += x0 * y1 - x1 * y0
    if abs(area2) <= 1e-9:
        return None
    total = 0.0
    for i in range(n):
        a = pts[order[(i - 1) % n]]
        b = pts[order[i]]
        c = pts[order[(i + 1) % n]]
        v1x, v1y = b[0] - a[0], b[1] - a[1]
        v2x, v2y = c[0] - b[0], c[1] - b[1]
        total += math.atan2(v1x * v2y - v1y * v2x, v1x * v2x + v1y * v2y)
    if abs(abs(total) - 2.0 * math.pi) > 0.5 * math.pi:
        return None
    return order


def chain_polyline_for_detection(
    det: Any,
    anchor_positions: Optional[Dict[Any, Any]],
    min_points: int = CHAIN_MIN_POLYLINE,
) -> Optional[Dict[str, Any]]:
    """Everything the Shape lenses need, computed once per chain.

    Returns a dict (or None when the chain is unusable) with:
      points      ordered polyline
      closed      ring flag (closure_bin on the PEAK_CHAIN signature)
      curvatures  signed curvature sequence
      curvature_hist  2*n_bins graded histogram
      turns       raw turn angles
      area        shoelace area (0 when open)
      centroid    (cx, cy)
      length      total Euclidean path length

    Computing this once keeps the five Shape lenses from recomputing the
    same geometry, and makes them agree slot-for-slot by construction.
    """
    entries = member_entries(det, anchor_positions)
    if len(entries) < min_points:
        return None
    idxs = [e[0] for e in entries]
    pts = [e[1] for e in entries]
    if any(not (math.isfinite(p[0]) and math.isfinite(p[1])) for p in pts):
        return None
    # [F1] ONE reader of the closure slot.
    closed = chain_is_closed(det)
    if closed:
        # A ring's members arrive in index order, which is not a traversal.
        # Reorder them along the contour so the geometry below describes the
        # contour instead of the member numbering. The break uses the FIELD
        # peak indices, not list position, so the traversal is a function of
        # the field and not of how the chain happened to arrive. If the
        # points do not form a simple cycle the engine's order is kept
        # unchanged -- better to measure the engine's output than to invent.
        order = order_ring_positions(pts, idxs)
        if order is not None:
            pts = [pts[i] for i in order]
    walk = close_polyline(pts, closed)
    curv = signed_curvature_sequence(walk)
    cx, cy, area = centroid_and_area(walk, closed)
    length = sum(math.hypot(walk[i + 1][0] - walk[i][0],
                            walk[i + 1][1] - walk[i][1])
                 for i in range(len(walk) - 1))
    return dict(
        points=pts, closed=closed, walk=walk,
        curvatures=curv,
        curvature_hist=curvature_histogram(curv),
        turns=turn_angles(walk),
        area=area, centroid=(cx, cy), length=length,
    )



if __name__ == "__main__":
    # smoke: a square ring must give 4 roughly equal right turns.
    sq = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
    g = chain_polyline_for_detection(
        type("D", (), {"alpha": 1,
                       "sigma": type("S", (), {"quantized": (4, 0, 0, 1, 0, 0)})(),
                       "rho": (("entity", 1, 0, "+"),
                               ("member", "m0", 0, "+"),
                               ("member", "m1", 1, "+"),
                               ("member", "m2", 2, "+"),
                               ("member", "m3", 3, "+"))})(),
        {"m0": sq[0], "m1": sq[1], "m2": sq[2], "m3": sq[3]})
    print("closed:", g["closed"], "area:", round(g["area"], 2),
          "turns:", [round(x, 3) for x in g["turns"]])


def chain_is_closed(det: Any) -> bool:
    """Whether a PEAK_CHAIN detection reports itself as a closed ring.

    The flag is the chain's OWN closure slot (slot 3 of the six PEAK_CHAIN
    slots), measured by the chain lens from the total turning of the chain
    (the Hopf Umlaufsatz test). It is not inferred from a region.
    """
    try:
        q = det.sigma.quantized
        if len(q) < 4:
            return False
        return float(q[3]) >= 0.5
    except Exception:
        return False


def chain_polylines(
    dets: Optional[List[Any]],
    anchor_positions: Optional[Dict[Any, Any]],
    with_closure: bool = False,
) -> Dict[Any, Any]:
    """anchor -> ordered polyline, one per PEAK_CHAIN (a contour FRAGMENT).

    The chain's member order IS the polyline's order, so the polyline is
    determined by the field rather than by a drawing routine, and the chain's
    own closure slot says whether it is a ring. There is no region, no chord
    envelope and no entity here: a fragment is what it is.

    `with_closure=True` returns `anchor -> (polyline, closed)`.
    """
    out: Dict[Any, Any] = {}
    for det in (dets or ()):
        pts = chain_polyline(det, anchor_positions)
        if not pts:
            continue
        closed = chain_is_closed(det)
        if with_closure:
            out[det.alpha] = (close_polyline(pts, closed), closed)
        else:
            out[det.alpha] = pts
    return out
