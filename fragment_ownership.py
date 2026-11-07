"""perception/fragment_ownership.py -- [ROUND 10 / C] Per-flank fragment ownership.

WHAT THIS REPLACES
------------------
The old BorderOwnershipLens attached figure-ground to SEGMENTER region anchors
(observation.entity_indices), which is empty on this tree: it emitted nothing by
default, and when it did fire it named a region, not a contour fragment. This
module makes ownership a property of the ORIENTED EDGE FRAGMENT itself:

    one fragment = one chain member (a CONTOUR_PEAK)
    orientation  = the chain's TRAVEL direction (tangent -> left normal)
    side bit     = the SIGN of the signed ownership: + means the owner (figure)
                   lies on the LEFT of the travel direction

A closed chain's members arrive in index order, not traversal order, so the ring
is reordered with `order_ring_positions` before any tangent is taken. Ownership
is then blended with a chain-wide consensus along the contour; a lone opposite
member is absorbed into its neighbours unless it sits at a corner or a junction.

EDGES ARE DERIVED, NOT VOTED
----------------------------
Ownership is intrinsic to each fragment. Edges come afterwards: step along the
OWNER-side normal (radius relative to the fragment's own flank offset) and look
for the fragment standing there. If that fragment's own ownership points back,
the two belong to the SAME figure (a positive partner); if it diverges across the
boundary, they are an occlusion pair (a negative neighbour). Close same-chain
neighbours are never partners, but a circle's far side (many members away) may
be. No candidate -> no edge, rather than a fabricated partner.

CUES (C3)
---------
    convexity   the side of the centre of curvature (turn sign)
    enclosure   interior-field difference ACROSS the edge, at several scales
                (this is the multi-scale term that replaces a separate
                surroundedness weight)
    t_junction  proximity/side of a T-junction contact (occlusion evidence)

Combined in LOG-ODDS (perception/ownership.py `combine_ownership_logodds`).
CONTRAST POLARITY IS DELIBERATELY NOT A VOTER: V2 ownership is largely
polarity-independent, so a fragment's luminance/opponent contrast stays a
linking/descriptor property (perception/chain_flanks.py), not a side vote.

All geometry is read from positions the lenses already maintain, so this module
is testable without an image and deterministic: chains are visited in
detection-sort order and every tie-break is explicit.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import parameters as P

from core.hashing import serialize_value, _detection_sort_key
from perception.chain_flanks import ChannelSampler, PERPENDICULAR_SCALE
from perception.chain_polylines import chain_is_closed, order_ring_positions
from perception.ownership import (combine_ownership_logodds,
                                  ownership_logodds,
                                  ownership_probability)

__all__ = [
    "fragment_ownership_records",
    "DEFAULT_WEIGHTS",
]

#: Fallback log-odds weights, used when `P.BO_LOGODDS_WEIGHTS` is absent.
#: These are the seed values; the fitted values are produced by
#: diagnostics/_round10_C_fit.py on the synthetic ground-truth corpus.
DEFAULT_WEIGHTS: Dict[str, float] = {
    "convexity": 1.0,
    "enclosure": 1.2,
    "t_junction": 0.8,
}

#: |ownership| below this is treated as ambiguous: no side, and no edge step.
_AMBIGUOUS = 0.02


_WEIGHTS_CACHE: Optional[Dict[str, float]] = None
_CALIBRATION_CACHE: Optional[Dict[str, float]] = None


def _weights() -> Dict[str, float]:
    """The log-odds cue weights, in precedence order.

    1. the FITTED file (`parameters/_ownership_weights.json`), written by
       diagnostics/_round10_C_fit.py -- logistic fit on the synthetic corpus;
    2. the `P.BO_LOGODDS_WEIGHTS` override;
    3. `DEFAULT_WEIGHTS` (the seeds, identical to the shipped parameter).
    """
    global _WEIGHTS_CACHE
    if _WEIGHTS_CACHE is not None:
        return dict(_WEIGHTS_CACHE)
    fitted = None
    try:
        import json
        import os
        path = os.path.join(os.path.dirname(getattr(P, "__file__", "")),
                            "_ownership_weights.json")
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            w = data.get("weights") if isinstance(data, dict) else None
            if isinstance(w, dict) and w:
                fitted = {k: float(v) for k, v in w.items()}
    except Exception:
        fitted = None
    if fitted:
        base = dict(DEFAULT_WEIGHTS)
        base.update(fitted)
        _WEIGHTS_CACHE = base
        return dict(base)
    w = getattr(P, "BO_LOGODDS_WEIGHTS", None)
    _WEIGHTS_CACHE = dict(w) if isinstance(w, dict) and w else dict(DEFAULT_WEIGHTS)
    return dict(_WEIGHTS_CACHE)


def _calibration() -> Dict[str, float]:
    """The ownership probability calibration (Session G, E1).

    ``sigmoid(slope * |logodds| + intercept)`` fitted on the synthetic
    occlusion corpus against the renderer's object ids.  Same precedence
    pattern as ``_weights``: the fitted file, then the parameter override,
    then a monotone identity.
    """
    global _CALIBRATION_CACHE
    if _CALIBRATION_CACHE is not None:
        return dict(_CALIBRATION_CACHE)
    fitted: Optional[Dict[str, float]] = None
    try:
        import json
        import os
        path = os.path.join(os.path.dirname(getattr(P, "__file__", "")),
                            "_ownership_calibration.json")
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            got = data.get("logistic") if isinstance(data, dict) else None
            if isinstance(got, dict):
                fitted = {"slope": float(got.get("slope", 1.0)),
                          "intercept": float(got.get("intercept", 0.0))}
    except Exception:
        fitted = None
    if fitted is None:
        fitted = {"slope": float(getattr(P, "BO_CALIBRATION_SLOPE", 1.0)),
                  "intercept": float(getattr(P, "BO_CALIBRATION_INTERCEPT",
                                              0.0))}
    _CALIBRATION_CACHE = dict(fitted)
    return dict(_CALIBRATION_CACHE)


# -- geometry ---------------------------------------------------------------

def _travel_members(det: Any, anchor_positions: Dict[Any, Any]):
    """(anchors, positions, peak_indices, closed) in TRAVEL order.

    Members are taken from the chain's rho roles in the order the emitter wrote
    them. For a closed chain that order is index order, which is not a walk, so
    the ring is reordered along the contour when a simple cycle is found; if it
    is not a simple cycle the engine's order is kept (measure the output rather
    than invent).
    """
    anchors: List[Any] = []
    positions: List[Tuple[float, float]] = []
    idxs: List[int] = []
    base = int(P.PEAK_ANCHOR_BASE)
    for role in getattr(det, "rho", ()) or ():
        if len(role) < 2 or role[0] != "member":
            continue
        pos = anchor_positions.get(role[1]) if anchor_positions else None
        if pos is None or len(pos) < 2:
            continue
        try:
            x, y = float(pos[0]), float(pos[1])
        except (TypeError, IndexError, ValueError):
            continue
        anchors.append(role[1])
        positions.append((x, y))
        try:
            idxs.append(int(role[1]) - base)
        except (TypeError, ValueError):
            idxs.append(len(idxs))
    closed = False
    q = getattr(getattr(det, "sigma", None), "quantized", None)
    if q and len(q) >= 4:
        # [F1] ONE reader of the closure slot.
        closed = chain_is_closed(det)
    if closed and len(positions) >= 3:
        order = order_ring_positions(list(positions), list(idxs))
        if order is not None and len(order) == len(positions):
            anchors = [anchors[i] for i in order]
            positions = [positions[i] for i in order]
            idxs = [idxs[i] for i in order]
    return anchors, positions, idxs, closed


def _frame(positions: Sequence[Tuple[float, float]],
           closed: bool
           ) -> List[Optional[Tuple[float, float, float]]]:
    """Per-member (normal_x, normal_y, signed_turn) or None when degenerate.

    The tangent is the central difference of the neighbouring members in TRAVEL
    order; the normal is the tangent rotated +90 deg, i.e. the LEFT side of the
    travel direction. The signed turn is the discrete curvature at the member,
    normalised by pi/2 into [-1, 1]; open endpoints have no turn (0.0).
    """
    n = len(positions)
    frames: List[Optional[Tuple[float, float, float]]] = []
    for i in range(n):
        if closed:
            ip, in_ = (i - 1) % n, (i + 1) % n
        else:
            ip, in_ = max(0, i - 1), min(n - 1, i + 1)
        if ip == in_:
            frames.append(None)
            continue
        tx = positions[in_][0] - positions[ip][0]
        ty = positions[in_][1] - positions[ip][1]
        mag = math.hypot(tx, ty)
        if mag < 1e-9:
            frames.append(None)
            continue
        nx, ny = -ty / mag, tx / mag
        turn = 0.0
        if ip != i and in_ != i:
            ax = positions[i][0] - positions[ip][0]
            ay = positions[i][1] - positions[ip][1]
            bx = positions[in_][0] - positions[i][0]
            by = positions[in_][1] - positions[i][1]
            turn = math.atan2(ax * by - ay * bx, ax * bx + ay * by)
            turn = max(-1.0, min(1.0, turn / (math.pi / 2.0)))
        frames.append((nx, ny, turn))
    return frames


# -- observation-backed fields ---------------------------------------------

def _luminance(observation: Any) -> Optional[Any]:
    if observation is None:
        return None
    g = getattr(observation, "get_luminance", None)
    if callable(g):
        try:
            return g()
        except Exception:
            return None
    return None



def _median_wavelength(observation: Any) -> Optional[float]:
    field = getattr(observation, "_contour_field_cache", None) if observation is not None else None
    if not isinstance(field, dict):
        return None
    sfs = field.get("sfs")
    if sfs is None:
        return None
    arr = np.asarray(sfs, dtype=np.float64).ravel()
    if arr.size == 0:
        return None
    wl = float(np.median(1.0 / np.maximum(arr, 1e-6)))
    if not math.isfinite(wl) or wl <= 0.0:
        return None
    return wl


def _enclosure_cue(field: Optional[Dict[str, np.ndarray]],
                   point: Tuple[float, float],
                   normal: Tuple[float, float]) -> Optional[float]:
    """Multi-scale interior-field difference across the edge.

    The interior field is sampled at several probes (multiples of
    INTERIOR_PROBE_PX). The nearer probes carry more weight, because the local
    interior/exterior split at the edge is the more specific evidence; the
    wider probes supply the global enclosedness. Returns None when there is no
    field, so the cue is a non-vote rather than a fabricated 0.
    """
    if field is None:
        return None
    from perception.interior_field import ownership_from_enclosure
    probes = tuple(getattr(P, "BO_SURROUND_PROBES", (1.0, 2.0, 4.0)))
    if not probes:
        return None
    num = 0.0
    den = 0.0
    for k, scale in enumerate(probes):
        probe = float(P.INTERIOR_PROBE_PX) * float(scale)
        v = ownership_from_enclosure(field, point, normal, probe=probe)
        w = 1.0 / (k + 1.0)
        num += w * float(v)
        den += w
    if den <= 1e-12:
        return None
    return max(-1.0, min(1.0, num / den))


def _t_junctions(observation: Any,
                 explicit: Optional[Sequence[Any]]) -> List[Tuple[float, float]]:
    pts: List[Tuple[float, float]] = []
    if explicit is not None:
        for rec in explicit:
            # [B34] The junction record is ``(near, far, contact, ...)`` and the
            # CONTACT POINT is slot 2 -- the same slot the observation path below
            # reads.  The condition here was inverted (`not isinstance(...) or
            # len(rec) == 0`), so for every well-formed tuple record it took
            # `p = rec` and then read `rec[0]`/`rec[1]` -- the two chain
            # identities -- AS IF THEY WERE COORDINATES.  The two wirings of the
            # same cue therefore returned different junction locations, and a
            # malformed 2-tuple silently became a wrong coordinate instead of
            # being skipped.
            try:
                if isinstance(rec, (tuple, list)):
                    if len(rec) >= 3:
                        p = rec[2]        # (near, far, contact, ...)
                    elif (len(rec) == 2
                          and all(isinstance(v, (int, float))
                                  and not isinstance(v, bool) for v in rec)):
                        p = rec           # a bare (x, y) point
                    else:
                        continue          # malformed: nothing to read
                else:
                    p = rec               # already a bare point
                pts.append((float(p[0]), float(p[1])))
            except (TypeError, IndexError, ValueError):
                continue
        return pts
    if observation is None:
        return pts
    g = getattr(observation, "get_t_junctions", None)
    if callable(g):
        try:
            for rec in g():
                cp = rec[2]
                pts.append((float(cp[0]), float(cp[1])))
        except Exception:
            return []
    return pts


def _near_junction(point: Tuple[float, float],
                   tjs: Sequence[Tuple[float, float]],
                   radius: float) -> bool:
    for jx, jy in tjs:
        if math.hypot(jx - point[0], jy - point[1]) <= radius:
            return True
    return False


def _t_junction_cue(point: Tuple[float, float],
                    normal: Tuple[float, float],
                    tjs: Sequence[Tuple[float, float]],
                    radius: float) -> Optional[float]:
    """Signed occlusion evidence from nearby T-junction contacts.

    A T-junction marks where one surface occludes another; the stem side is the
    occluding (figure) surface. With no stem orientation available here the cue
    is conservative: a contact on the +normal side within `radius` votes for the
    owner being on the LEFT (+), a contact on the -normal side votes for the
    RIGHT (-), scaled by proximity. No contact -> None (non-vote).
    """
    if not tjs or radius <= 0.0:
        return None
    best: Optional[float] = None
    nx, ny = normal
    for jx, jy in tjs:
        d = math.hypot(jx - point[0], jy - point[1])
        if d > radius:
            continue
        side = (jx - point[0]) * nx + (jy - point[1]) * ny
        val = max(-1.0, min(1.0, side / (d + 1e-9))) * math.exp(-d / radius)
        if best is None or abs(val) > abs(best):
            best = val
    return best


# -- consensus --------------------------------------------------------------

def _consensus(s_list: List[float],
               frames: List[Optional[Tuple[float, float, float]]],
               closed: bool,
               positions: Sequence[Tuple[float, float]],
               tjs: Sequence[Tuple[float, float]],
               radius: float) -> List[float]:
    """Blend the local side vote with a chain-wide consensus.

    A moving average along the contour is taken first; a member keeps its LOCAL
    value only where it is a corner (|turn| >= BO_CORNER_FLIP_MIN_TURN) or sits
    at a junction -- the only places ownership is allowed to switch side.
    Elsewhere an isolated one-member island of the opposite sign is absorbed
    into its neighbours, so a single noisy flank cannot flip the chain.
    """
    n = len(s_list)
    if n == 0:
        return []
    w = max(1, int(round(float(P.BO_CONSENSUS_WINDOW_FRAC) * n)))
    smoothed: List[float] = []
    for i in range(n):
        acc = 0.0
        cnt = 0
        for k in range(-w, w + 1):
            j = i + k
            if closed:
                j %= n
            elif j < 0 or j >= n:
                continue
            acc += s_list[j]
            cnt += 1
        smoothed.append(acc / cnt if cnt else 0.0)

    corner = [bool(fr is not None and abs(fr[2]) >= float(P.BO_CORNER_FLIP_MIN_TURN))
              for fr in frames]
    junction = [_near_junction(positions[i], tjs, radius) for i in range(n)]
    out = [s_list[i] if (corner[i] or junction[i]) else smoothed[i]
           for i in range(n)]

    if n >= 3:
        consolidated = list(out)
        for i in range(n):
            if closed:
                ip, in_ = (i - 1) % n, (i + 1) % n
            else:
                ip, in_ = i - 1, i + 1
                if ip < 0 or in_ >= n:
                    continue
            a, b, c = out[ip], out[i], out[in_]
            if corner[i] or junction[i]:
                continue
            if (a > 0.0 > b and c > 0.0) or (a < 0.0 < b and c < 0.0):
                consolidated[i] = 0.5 * (a + c)
        out = consolidated
    return out


# -- edge derivation --------------------------------------------------------

def _pair_edges(fragments: List[Dict[str, Any]]) -> None:
    """Fill each fragment's `partner` (same figure) / `neighbor` (occlusion).

    For a fragment whose owner is on side d, the owner-side sample point is
    `p + off*d`. The nearest OTHER fragment within `radius` of that point is its
    candidate. If the candidate's own ownership points back toward `p` the pair
    is the same figure (`partner`, positive); otherwise it is an occlusion pair
    (`neighbor`, negative). Close same-chain neighbours are excluded (a circle's
    far side may still pair). No candidate -> neither field is set.

    [F2] Three eligibility rules the shipped version did not have, each of which
    was measured fabricating `neighbor` pairs that `occlusion_order_edges` then
    turned into directed depth edges:

    * a candidate must HAVE an owner side (|signed| >= _AMBIGUOUS).  The source
      loop already skipped ambiguous fragments; the candidate search did not, so
      a fragment with no side at all was named as another's occluder (measured on
      box_over_box: a neighbour pair with signed +0.10 / 0.00).
    * same-chain members are excluded by DISTANCE, not only by the
      |vertex_a - vertex_b| <= exclude index window.  The window has no
      wraparound for a chain whose members are incomplete, and it measures
      nothing about the contour: single_disk emitted an occlusion pair between
      two members of ONE closed silhouette 2.00 px apart (3 vertices apart, so
      the window let them through, and the chain's own step there is ~6 px).
      Two members closer than one sampling step of their own chain are the same
      local piece of contour.
    * an occlusion pair requires the two owner directions to DIVERGE.  With
      identical owner directions "A is on B's non-owner side" and "B is on A's
      non-owner side" are mutually exclusive, so nothing can be occluding
      anything: the two fragments are the same side of the same boundary.  The
      bound is the chain layer's own ALIGNMENT_TOL (35 deg, derived there as the
      geometric midpoint between the parallel-line deviation of 45 deg at one
      wavelength and the tightest representable arc's 28.65 deg) -- no new
      constant.  Measured on the corpus before the fix: of box_over_box's 6
      neighbour pairs, 3 had n_a . n_b in {+1.00, 0.00} and 0 of 6 were mutual.
    """
    from perception.chain_formation import ALIGNMENT_TOL
    same_side_cos = math.cos(ALIGNMENT_TOL)
    for f in fragments:
        f["partner"] = None
        f["neighbor"] = None
    radius_scale = float(P.BO_PARTNER_RADIUS_SCALE)
    exclude = int(P.BO_PARTNER_EXCLUDE_NEIGHBOURS)

    # [F2c] each chain's own sampling step, from its consecutive members
    rows_of: Dict[Any, List[Dict[str, Any]]] = {}
    for f in fragments:
        rows_of.setdefault(f["chain"], []).append(f)
    step_of: Dict[Any, float] = {}
    for chain, rows in rows_of.items():
        rows = sorted(rows, key=lambda z: int(z["index"]))
        gaps = []
        n = len(rows)
        for k in range(n - 1):
            gaps.append(math.hypot(rows[k + 1]["pos"][0] - rows[k]["pos"][0],
                                   rows[k + 1]["pos"][1] - rows[k]["pos"][1]))
        if n > 2 and rows[0].get("closed"):
            gaps.append(math.hypot(rows[0]["pos"][0] - rows[-1]["pos"][0],
                                   rows[0]["pos"][1] - rows[-1]["pos"][1]))
        if gaps:
            step_of[chain] = float(np.median(np.asarray(gaps, dtype=np.float64)))

    for i, f in enumerate(fragments):
        s = f["signed"]
        if abs(s) < _AMBIGUOUS:
            continue
        nx, ny = f["normal"]
        dx, dy = (nx, ny) if s >= 0.0 else (-nx, -ny)
        off = f["offset"]
        tx = f["pos"][0] + dx * off
        ty = f["pos"][1] + dy * off
        radius = max(radius_scale * off, 1e-6)
        best_d = None
        best_g = None
        best_key = None
        for j, g in enumerate(fragments):
            if j == i:
                continue
            if abs(g["signed"]) < _AMBIGUOUS:
                continue        # [F2d] no owner side -> cannot be a flank
            d_self = math.hypot(g["pos"][0] - f["pos"][0],
                                g["pos"][1] - f["pos"][1])
            if g["chain"] == f["chain"]:
                di = abs(g["index"] - f["index"])
                if f["closed"]:
                    di = min(di, f["n_chain"] - di)
                if di <= exclude:
                    continue
                st = step_of.get(f["chain"])
                if st and d_self < st:
                    # [F2c] closer together than the chain's own sampling step:
                    # the same local piece of one contour, not two flanks.  The
                    # bound is the measured step, so it needs no constant, and
                    # it is strict so that a chain of two members (whose median
                    # step IS their own separation) is not emptied out.
                    continue
            d = math.hypot(g["pos"][0] - tx, g["pos"][1] - ty)
            if d > radius:
                continue
            key = serialize_value(g["alpha"])
            if (best_d is None or d < best_d
                    or (d == best_d and (best_key is None or key < best_key))):
                best_d, best_g, best_key = d, g, key
        if best_g is None:
            continue
        gs = best_g["signed"]
        gnx, gny = best_g["normal"]
        gdx, gdy = (gnx, gny) if gs >= 0.0 else (-gnx, -gny)
        back = (gdx * (f["pos"][0] - best_g["pos"][0])
                + gdy * (f["pos"][1] - best_g["pos"][1]))
        # [F2a] an occlusion needs the two owner sides to diverge; two owner
        # directions inside the alignment tolerance are the same side of the
        # same boundary and carry no relation at all.
        if dx * gdx + dy * gdy > same_side_cos:
            continue
        if gs != 0.0 and back > 0.0:
            f["partner"] = best_g["alpha"]
        else:
            f["neighbor"] = best_g["alpha"]


# -- producer ---------------------------------------------------------------

def fragment_ownership_records(
    chain_dets: Sequence[Any],
    anchor_positions: Optional[Dict[Any, Any]],
    observation: Any = None,
    extraction_scale: float = 0.0,
    t_junction_positions: Optional[Sequence[Any]] = None,
) -> List[Dict[str, Any]]:
    """One ownership record per chain member (an oriented edge fragment).

    Record keys: alpha (the fragment's CONTOUR_PEAK anchor), partner (same-figure
    fragment anchor or None), neighbor (occlusion fragment anchor or None),
    signed ([-1, 1]; + = owner on the LEFT of travel), curvature ([-1, 1]),
    arc (member position as a fraction of the chain), strength (|signed|),
    vertex (member index).

    Geometry is carried as non-identity data so a consumer does not have to
    re-derive a second source of truth from the anchors:

    * ``pos`` -- image position (x, y) of the member peak.
    * ``normal`` -- unit LEFT normal of the travel direction, i.e. the side
      the owner is on when ``signed > 0``.
    * ``turn`` -- signed discrete turn in TRAVEL order, normalised by pi/2
      into [-1, 1].  Owner-relative curvature is ``turn * sign(signed)``.
    * ``chain``, ``n_chain``, ``closed`` -- chain membership and closure.

    These keys are derived from the same travel walk as ``signed``; adding
    them changes no existing value.
    """
    if not chain_dets or not anchor_positions:
        return []

    lum = _luminance(observation)
    interior = None
    try:
        from perception.interior_field import interior_field_for
        interior = interior_field_for(observation, luminance=lum)
    except Exception:
        interior = None
    median_wl = _median_wavelength(observation)
    if median_wl is None or median_wl <= 0.0:
        base = float(extraction_scale) if extraction_scale > 0.0 else 1.0
        median_wl = base * 0.5
    offset = float(PERPENDICULAR_SCALE) * float(median_wl)
    tjs = _t_junctions(observation, t_junction_positions)
    junction_radius = float(P.BO_PARTNER_RADIUS_SCALE) * offset
    weights = _weights()

    fragments: List[Dict[str, Any]] = []
    for det in sorted(chain_dets, key=_detection_sort_key):
        anchors, positions, _idxs, closed = _travel_members(det, anchor_positions)
        n = len(positions)
        if n < 2:
            continue
        frames = _frame(positions, closed)
        s_local: List[float] = []
        lo_local: List[float] = []
        enc_local: List[float] = []
        for i in range(n):
            fr = frames[i]
            if fr is None:
                s_local.append(0.0)
                lo_local.append(0.0)
                enc_local.append(0.0)
                continue
            nx, ny, curv = fr
            enc_i = _enclosure_cue(interior, positions[i], (nx, ny))
            cues = {
                "convexity": curv,
                "enclosure": enc_i,
                "t_junction": _t_junction_cue(positions[i], (nx, ny), tjs,
                                              junction_radius),
            }
            s_local.append(combine_ownership_logodds(cues, weights))
            lo_local.append(ownership_logodds(cues, weights))
            enc_local.append(float(enc_i) if enc_i is not None else 0.0)
        s_final = _consensus(s_local, frames, closed, positions, tjs,
                             junction_radius)
        # The RAW log-odds get the same consensus smoothing.  They are carried
        # alongside `signed` (which stays the tanh value, so nothing that reads
        # it changes) and are what the calibrated probability is fitted on.
        lo_final = _consensus(lo_local, frames, closed, positions, tjs,
                              junction_radius)
        cal = _calibration()
        chain_key = serialize_value(det.alpha)
        for i in range(n):
            fr = frames[i]
            if fr is None:
                continue
            fragments.append(dict(
                alpha=anchors[i],
                pos=positions[i],
                normal=(fr[0], fr[1]),
                curv=fr[2],
                signed=s_final[i],
                logodds=float(lo_final[i]),
                logodds_local=float(lo_local[i]),
                enclosure=float(enc_local[i]),
                # Calibrated on ENCLOSURE, not on the log-odds: Session G
                # measured the raw log-odds magnitude to carry no information
                # about correctness (AUC 0.43-0.53), while |enclosure| carries
                # 0.65 on held-out scenes and separates a silhouette from an
                # occlusion boundary at AUC 0.9997.
                owner_prob=ownership_probability(float(enc_local[i]),
                                                 slope=cal["slope"],
                                                 intercept=cal["intercept"]),
                chain=chain_key,
                index=i,
                n_chain=n,
                closed=closed,
                offset=offset,
            ))

    _pair_edges(fragments)

    records: List[Dict[str, Any]] = []
    for f in fragments:
        n = max(1, f["n_chain"])
        signed = max(-1.0, min(1.0, float(f["signed"])))
        records.append(dict(
            alpha=f["alpha"],
            partner=f["partner"],
            neighbor=f["neighbor"],
            signed=signed,
            curvature=max(-1.0, min(1.0, float(f["curv"]))),
            arc=(f["index"] + 0.5) / n,
            strength=abs(signed),
            vertex=int(f["index"]),
            pos=(float(f["pos"][0]), float(f["pos"][1])),
            normal=(float(f["normal"][0]), float(f["normal"][1])),
            turn=max(-1.0, min(1.0, float(f["curv"]))),
            chain=f["chain"],
            n_chain=n,
            closed=bool(f["closed"]),
            logodds=float(f["logodds"]),
            logodds_local=float(f["logodds_local"]),
            enclosure=float(f["enclosure"]),
            owner_prob=float(f["owner_prob"]),
        ))
    return records
