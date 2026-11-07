"""perception/lenses/corner_lens.py -- [Phase 1] CornerLens.

Emits one CORNER typed node per LOCAL TURN-ANGLE EXTREMUM along a boundary
chain. This is the corners arm of the Phase 1 singular-point representation:
junctions (`JunctionLens`) and end-stops (`EndStoppedLens`) already exist as
typed nodes; corners existed only as `CONTOUR` slot 5 (`corner_count_bin`), a
summary, never a node.

DERIVED FROM THE CONTINUOUS FIELD + CHAIN RELATIONS, NOT THE SHOCK GRAPH. The
lens reads the PEAK_CHAIN detections that `ChainLens` formed from the
continuous contour field's peaks (perception/chain_formation.form_chains) and
walks each chain's member order; a corner is a member whose tangent turn to its
chain neighbours is at least as sharp as both of theirs (a turn-ANGLE extremum)
and exceeds `CORNER_MIN_TURN`. Nothing here touches the dead shock structure.

Signature (arity 4, registered in core/registries.py):
    0 turn_mag_bin   graded 0..7   |signed turn| over [0, pi]
    1 orient_bin     circular 12   corner-bisector orientation over [0, pi)
    2 prom_bin       graded 0..7   local prominence over [0, pi/2]
    3 scale_bin      graded 0..7   corner arm length / (8 * extraction_scale)

The anchor scheme mirrors the peak/chain one: the corner anchor is derived from
its emission ordinal, so any consumer can compute it, and it lives in the same
far-above-the-pipeline band (`PEAK_ANCHOR_BASE + CORNER_ANCHOR_OFFSET + k`).
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import parameters as P
from core.hashing import _detection_sort_key, clamp
from core.registries import population_from_values
from core.types import (  # noqa: F401
    Detection,
    TypedSignature,
)
from perception.lenses.derived_base import HigherLens
from perception.chain_polylines import (
    chain_is_closed,
    order_ring_positions,
)

K_CORNER = "Corner"
CLASS_CORNER = "CORNER"
RELATION_CORNER = "CORNER_PROFILE"

#: slot 1 is a circular orientation axis with a 12-bin cycle over [0, pi).
N_ORIENT_BINS = 12
#: graded slot bin count (slots 0, 2, 3).
N_GRADED_BINS = 8
#: Arm length that fills the top scale bin, in wavelengths.
CORNER_FULL_ARM_WAVELENGTHS = 8.0


def _clip_bin(value: float, n_bins: int, lo: float, hi: float) -> int:
    """Clip `value` into [lo, hi] and floor-bin it into n_bins slots."""
    if not math.isfinite(value) or hi <= lo:
        return 0
    v = min(hi, max(lo, value))
    q = int(math.floor((v - lo) / ((hi - lo) / float(n_bins))))
    return min(n_bins - 1, max(0, q))


def _corner_anchor(ordinal: int) -> int:
    return int(P.PEAK_ANCHOR_BASE) + int(P.CORNER_ANCHOR_OFFSET) + int(ordinal)


def _turn_and_bisector(a: Tuple[float, float],
                       b: Tuple[float, float],
                       c: Tuple[float, float]) -> Tuple[float, float]:
    """(signed turn, bisector orientation) at vertex b.

    `signed_turn` is the angle from the incoming step (a->b) to the outgoing
    step (b->c) in (-pi, pi]; `bisector` is the orientation of the external
    angle bisector (the direction a corner "opens" toward) folded onto
    [0, pi). A straight vertex has an ill-defined bisector, so it falls back
    to the tangent direction rotated +90 deg; its turn is ~0 and it is not a
    corner anyway.
    """
    v1x, v1y = b[0] - a[0], b[1] - a[1]
    v2x, v2y = c[0] - b[0], c[1] - b[1]
    n1 = math.hypot(v1x, v1y)
    n2 = math.hypot(v2x, v2y)
    if n1 < 1e-9 or n2 < 1e-9:
        return 0.0, 0.0
    turn = math.atan2(v1x * v2y - v1y * v2x, v1x * v2x + v1y * v2y)
    # External-angle bisector: from b toward a, plus from b toward c.
    e1x, e1y = -v1x / n1, -v1y / n1
    e2x, e2y = v2x / n2, v2y / n2
    bx, by = e1x + e2x, e1y + e2y
    bn = math.hypot(bx, by)
    if bn < 1e-9:
        bx, by = -v1y / n1, v1x / n1
    return turn, math.atan2(by, bx) % math.pi


class CornerLens(HigherLens):
    """[Phase 1] Boundary corners as typed nodes (chain turn extrema)."""

    def __init__(self) -> None:
        super().__init__()

    def get_diagnostics(self) -> Dict[str, int]:
        diag = super().get_diagnostics()
        diag["emitted_total"] = self._emitted_total
        diag["candidates_evaluated"] = self._candidates_evaluated
        return diag

    def step(
        self,
        t: int,
        concept_graph: Any = None,
        sig_provider: Any = None,
        anchor_positions: Optional[Dict[Any, Tuple[float, ...]]] = None,
        anchor_velocities: Optional[Dict[Any, Tuple[float, ...]]] = None,
        current_detections: Optional[Dict[str, List[Detection]]] = None,
        extraction_scale: float = 0.0,
        envelope_mask: Optional[Tuple[Tuple[float, float, float, float], ...]] = None,
        observation: Any = None,
        **kwargs: Any,
    ) -> List[Detection]:
        if not anchor_positions or not current_detections:
            return []
        chains = sorted(current_detections.get("PEAK_CHAIN", ()),
                        key=_detection_sort_key)
        if not chains:
            return []

        min_turn = float(P.CORNER_MIN_TURN)
        emitted: List[Detection] = []
        ordinal = 0
        for det in chains:
            members = [r[1] for r in (getattr(det, "rho", ()) or ())
                       if len(r) >= 2 and r[0] == "member"]
            pts: List[Tuple[float, float]] = []
            ok = True
            for m in members:
                p = anchor_positions.get(m)
                if p is None or len(p) < 2:
                    ok = False
                    break
                pts.append((float(p[0]), float(p[1])))
            if not ok or len(pts) < 3:
                continue
            self._candidates_evaluated += 1

            q = getattr(getattr(det, "sigma", None), "quantized", None) or ()
            closed = chain_is_closed(det)
            n = len(pts)
            if closed:
                # [F5] A closed chain's members arrive in INDEX order -- a
                # zig-zag through the ring -- so every turn computed on
                # them described the member numbering, not the contour.
                # The traversal order is recovered from the geometry.
                order = order_ring_positions(pts, list(range(n)))
                if order:
                    pts = [pts[i] for i in order]
                    members = [members[i] for i in order]
            if closed:
                idx_range = range(n)          # every member, wrapping
            else:
                idx_range = range(1, n - 1)   # endpoints have no both-sides turn

            turns: Dict[int, Tuple[float, float]] = {}
            for k in idx_range:
                a = pts[(k - 1) % n]
                b = pts[k % n]
                c = pts[(k + 1) % n]
                turns[k] = _turn_and_bisector(a, b, c)

            for k, (turn, bisector) in turns.items():
                mag = abs(turn)
                if mag < min_turn:
                    continue
                # Turn-ANGLE extremum: at least as sharp as BOTH chain
                # neighbours. Non-strict, so a regular polygon (all turns
                # equal) keeps every vertex as a corner while a vertex tucked
                # beside a sharper one is suppressed.
                prev = turns.get((k - 1) % n)[0] if closed or k - 1 in turns else None
                nxt = turns.get((k + 1) % n)[0] if closed or k + 1 in turns else None
                nb = [abs(v) for v in (prev, nxt) if v is not None]
                if nb and mag < max(nb):
                    continue
                prom = mag - (max(nb) if nb else 0.0)

                turn_bin = float(_clip_bin(mag, N_GRADED_BINS, 0.0, math.pi))
                orient_bin = float(
                    int(math.floor(bisector * N_ORIENT_BINS / math.pi))
                    % N_ORIENT_BINS)
                prom_bin = float(_clip_bin(prom, N_GRADED_BINS, 0.0, math.pi / 2.0))
                if extraction_scale > 0.0:
                    a = pts[(k - 1) % n]
                    b = pts[k % n]
                    c = pts[(k + 1) % n]
                    arm = min(math.hypot(b[0] - a[0], b[1] - a[1]),
                              math.hypot(c[0] - b[0], c[1] - b[1]))
                    arm_frac = arm / (CORNER_FULL_ARM_WAVELENGTHS * extraction_scale)
                    scale_bin = float(_clip_bin(arm_frac, N_GRADED_BINS, 0.0, 1.0))
                else:
                    arm_frac = 0.0
                    scale_bin = 0.0

                slots = (turn_bin, orient_bin, prom_bin, scale_bin)
                # [GRADED POPULATION] The population is built from the
                # CONTINUOUS measurements (the turn magnitude, the bisector
                # orientation, the prominence and the arm fraction), not from
                # the bins: the bins are the lookup key, the activity vector
                # carries the sub-bin position the bin throws away.
                pop = population_from_values("CORNER", (mag, bisector, prom,
                                                         arm_frac))
                anchor = _corner_anchor(ordinal)
                ordinal += 1
                b = pts[k % n]
                if isinstance(anchor_positions, dict):
                    anchor_positions[anchor] = (b[0], b[1])
                emitted.append(Detection(
                    k=K_CORNER,
                    source_layer=1,
                    scale=0,
                    sigma=TypedSignature(
                        kind="CANONICAL",
                        class_=CLASS_CORNER,
                        relation=RELATION_CORNER,
                        quantized=slots,
                        population=pop,
                    ),
                    alpha=anchor,
                    z=float(clamp(mag / math.pi, 0.0, 1.0)),
                    t=t,
                    rho=(("entity", anchor, 0, "+"),
                         ("chain", getattr(det, "alpha", None), 0, "+"),
                         ("vertex", members[k % n], int(k), "+")),
                ))

        self._emitted_total += len(emitted)
        return emitted

