"""perception/polar.py -- extracted from sensory_floor.py (Move 3a).

MOVE ONLY: whole top-level statements, verbatim. The re-export
shim in sensory_floor.py keeps `from sensory_floor import X`
working until the importer repoint retires it.
"""
from __future__ import annotations

import math

def _axial_wrap_sf(x: float) -> float:
    return (x + math.pi / 2.0) % math.pi - math.pi / 2.0

def signed_menger(p1, p2, p3, c) -> float:
    """Menger curvature 4A/(abc); + when the arc bends TOWARD reference point c.
    Traversal-direction independent. NOTE: c = object centroid is a shortcut for
    'object interior'; pass a figure-side reference (see inside_points) for
    non-convex objects."""
    a = math.hypot(p2[0]-p1[0], p2[1]-p1[1]); b = math.hypot(p3[0]-p2[0], p3[1]-p2[1])
    d = math.hypot(p3[0]-p1[0], p3[1]-p1[1])
    if a < 1e-9 or b < 1e-9 or d < 1e-9:
        return 0.0
    turn = (p2[0]-p1[0])*(p3[1]-p2[1]) - (p2[1]-p1[1])*(p3[0]-p2[0])
    side = (p3[0]-p1[0])*(c[1]-p1[1]) - (p3[1]-p1[1])*(c[0]-p1[0])
    s = ((turn > 0) - (turn < 0)) * ((side > 0) - (side < 0))
    return 2.0 * abs(turn) / (a * b * d) * s

def _arc_split(poly, target):
    frags, cur, acc = [], [poly[0]], 0.0
    for i in range(1, len(poly)):
        acc += math.hypot(poly[i][0]-poly[i-1][0], poly[i][1]-poly[i-1][1])
        cur.append(poly[i])
        if acc >= target:
            if len(cur) >= 3:
                frags.append(cur)
            cur, acc = [poly[i]], 0.0
    if len(cur) >= 3 and acc >= 0.5 * target:
        frags.append(cur)
    return frags

def _arc_mid(fr):
    cum = [0.0]
    for i in range(1, len(fr)):
        cum.append(cum[-1] + math.hypot(fr[i][0]-fr[i-1][0], fr[i][1]-fr[i-1][1]))
    half = 0.5 * cum[-1]
    return fr[min(range(len(fr)), key=lambda i: abs(cum[i] - half))], cum[-1]

def polar_boundary_fragments(contours, frag_frac=0.35, kappa_cap=8.0, min_points=12,
                             center=None, inside_points=None):
    """Pasupathy & Connor object-centred fragments from association-field contours.
    center: optional (x,y) overriding the vertex centroid C_obj.
    inside_points: optional {contour_idx: (x,y)} figure-side reference used ONLY
    for the curvature sign. Bins: kappa 0..8 (zero-centred, 4 = straight), theta 0..23
    (circular), r 0..7, tangent-vs-radial 0..7."""
    polys = []
    for ci, c in enumerate(contours):
        pl = c.get("polyline")
        if pl and len(pl) >= min_points:
            polys.append((ci, [(float(x), float(y)) for x, y in pl]))
    pts = [p for _, pl in polys for p in pl]
    if len(pts) < 2 * min_points:
        return [], {}
    cx, cy = center if center is not None else (sum(p[0] for p in pts)/len(pts),
                                                sum(p[1] for p in pts)/len(pts))
    dist = sorted(math.hypot(p[0]-cx, p[1]-cy) for p in pts)
    R = dist[int(0.9 * (len(dist) - 1))]
    if R < 1e-6:
        return [], {}
    target, S, out = max(4.0, frag_frac * R), math.asinh(kappa_cap), []
    for ci, pl in polys:
        ref = (inside_points or {}).get(ci, (cx, cy))
        for k, fr in enumerate(_arc_split(pl, target)):
            (mid, arc), p0, p1 = _arc_mid(fr), fr[0], fr[-1]
            kappa_n = signed_menger(p0, mid, p1, ref) * R
            kc = max(-kappa_cap, min(kappa_cap, kappa_n))
            kb = float(max(0, min(8, int(round(math.asinh(kc) / S * 4.0)) + 4)))
            th = math.atan2(mid[1]-cy, mid[0]-cx) % (2.0 * math.pi)
            tb = float(int(round(th / (2.0 * math.pi) * 24)) % 24)
            rr = math.hypot(mid[0]-cx, mid[1]-cy) / R
            rb = float(min(7, max(0, int(min(rr, 1.0) * 8.0))))
            tvr = abs(_axial_wrap_sf(math.atan2(p1[1]-p0[1], p1[0]-p0[0]) - th))
            vb = float(min(7, int(tvr / (math.pi / 2.0) * 8.0)))
            out.append({"contour": ci, "frag": k, "mid": mid, "arc": arc, "kappa_n": kappa_n,
                        "theta": th, "r": rr, "bins": (kb, tb, rb, vb),
                        # [GRADED POPULATION] The CONTINUOUS values of the
                        # four graded slots, so the emitter can build the
                        # population from the pre-quantisation measurement
                        # instead of echoing the bin: the asinh-compressed
                        # signed curvature (the same compression the level
                        # uses, so the bump and the bin agree), the polar
                        # angle, the normalised radius and the
                        # tangent-vs-radial angle.
                        "kappa_c": (math.asinh(kc) / S) if S > 0 else 0.0,
                        "theta_c": th,
                        "r_c": rr,
                        "tvr_c": tvr})
    return out, {"center": (cx, cy), "R": R, "target_len": target}



# [ITEM 4] Moved verbatim from perception/contours.py (thin-shim split; nothing deleted).
def _angle_diff(a: float, b: float) -> float:
    """Signed shortest angular difference a − b, in (−π, π]."""
    d = (a - b) % (2.0 * math.pi)
    if d > math.pi:
        d -= 2.0 * math.pi
    return d

def _axial_angle_diff(a: float, b: float) -> float:
    """Absolute shortest distance between two axial angles (mod pi)."""
    d = abs((a % math.pi) - (b % math.pi))
    return min(d, math.pi - d)
