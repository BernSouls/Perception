"""perception/lenses/v2.py -- extracted from derived.py
(Move 3.5).

MOVE ONLY: whole top-level statements, verbatim.
"""
from __future__ import annotations

import numpy as np
import math
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple
import parameters as P
from perception.ownership import ownership_signed
from core.hashing import (  # noqa: F401
    _detection_sort_key,
    _node_signature_hash,
    clamp,
    serialize_value,
)
from core.slots import (  # noqa: F401
    orientation_theta,
    orientation_theta_radians,
)
from core.quantize import (  # noqa: F401
    _q_signed,
    _q_unsigned,
    q_linear,
)
from core.quantize import _aang, carrier_to_tangent  # noqa: F401
from core.registries import (  # noqa: F401
    POPULATION_SLOTS,
    population_from_values,
    population_values_in_span,
    quantise,
    slot_spec,
)
from core.types import (  # noqa: F401
    Detection,
    EdgeType,
    TypedSignature,
)
from core.spatial import _pos_dist  # noqa: F401
from parameters._text import (  # noqa: F401
    BO_COLOR_WEIGHT,
    BO_CUE_SCALE,
    BO_CURVATURE_WEIGHT,
    BO_GRAVITY_SCALE,
    BO_GRAVITY_WEIGHT,
    BO_RICHNESS_FALLBACK,
    BO_TEXTURE_BIN_MAX,
    BO_TEXTURE_WEIGHT,
)
from perception.lenses.derived_common import (  # noqa: F401
    JUNCTION_FAMILY_KINDS,
    JUNCTION_RESIDUAL_KIND,
    _JUNCTION_CANONICAL_BINS,
    _arm_tangent_stats,
    _collect_continuity_neighbours,
    _contour_measurements,
    _contour_slots,
    _contour_slots_from_measurements,
    qi,
)
from perception.lenses.derived_base import (  # noqa: F401
    HigherLens,
)

# [UNIT TABLE] The unit table of the classes this module emits, looked up BY
# NAME (never by position), so the bins path and the graded population read
# the SAME declaration of every quantity: its unit, its span, its level count
# and its quantiser. core.registries.POPULATION_SLOTS is that table.
_JUNCTION_SPEC = {s.name: s for s in POPULATION_SLOTS["JUNCTION"]}
_TEXTURE_BOUNDARY_SPEC = {
    s.name: s for s in POPULATION_SLOTS["TEXTURE_BOUNDARY"]}

def _contour_tangent(det: Any,
                      anchor_positions: Optional[Dict[Any, Any]],
                      chains: Optional[List[Any]] = None,
                      ) -> Optional[Tuple[float, float]]:
    """The travel direction of the CONTOUR that owns ``det``'s anchor.

    [U25 V1] The direction is read from the PEAK_CHAIN that owns the anchor,
    in the chain's ordinal member order.  The figure-ground gate tests the
    neighbour's LATERAL displacement against this tangent; a neighbour that
    continues ALONG the tangent carries no side information, so the gate runs
    on lateral displacement only.

    ``chains`` is the step's PEAK_CHAIN detections (``None`` -> the single
    ``det`` is treated as its own chain, the legacy call shape).
    """
    src = det.alpha
    chains = list(chains) if chains is not None else [det]
    for chain in chains:
        pairs: List[Tuple[int, Any, float, float]] = []
        for role in (getattr(chain, "rho", None) or ()):
            if len(role) < 3 or role[0] != "member":
                continue
            o = int(role[2])
            a = role[1]
            p = (anchor_positions or {}).get(a)
            if p is None or len(p) < 2:
                continue
            pairs.append((o, a, float(p[0]), float(p[1])))
        if not pairs or not any(a == src for _o, a, _x, _y in pairs):
            continue
        pairs.sort(key=lambda t: t[0])
        pts = [(x, y) for _o, _a, x, y in pairs]
        if len(pts) < 2:
            continue
        tx = pts[-1][0] - pts[0][0]
        ty = pts[-1][1] - pts[0][1]
        mag = math.hypot(tx, ty)
        if mag < 1e-9:
            return None
        return (tx / mag, ty / mag)
    return None


class BorderOwnershipLens(HigherLens):
    """B.6 [FIX v2] — V2 figure-ground as a PER-BOUNDARY-POINT FIELD.

    Emits ONE BORDER_OWNERSHIP detection per sampled boundary vertex.
    Signature (arity 4, linear):
        (arc_pos_bin, signed_dir_bin, curvature_bin, strength_bin)
    `alpha` is the live region anchor `a`; rho carries entity/neighbour/vertex.

    Field length <= BO_FIELD_SAMPLE_COUNT per boundary. Open AND closed arcs
    are sampled (the prior region-level body only handled closed polygons,
    silently dropping the many open chains). Cues are region-level (constant
    along this vertex set) except curvature, which is per-vertex.
    """

    def __init__(self) -> None:
        super().__init__()
        self._skipped_no_boundary = 0
        self._skipped_nonfinite = 0
        self._candidates_evaluated = 0

    def get_diagnostics(self) -> Dict[str, int]:
        # [A4#11] the two retired counters (never incremented) are gone.
        return {
            "emitted_total": self._emitted_total,
            "skipped_no_boundary": self._skipped_no_boundary,
            "skipped_nonfinite": self._skipped_nonfinite,
            "candidates_evaluated": self._candidates_evaluated,
        }

    def _emit_fragment_ownership(
        self,
        t: int,
        current_detections: Dict[str, List[Detection]],
        anchor_positions: Dict[Any, Tuple[float, ...]],
        extraction_scale: float,
        observation: Optional[Any],
    ) -> List[Detection]:
        """[ROUND 10 / C] One BORDER_OWNERSHIP per chain member (edge fragment).

        The signed slot is a SIGNED LINEAR scalar: + means the owner (figure) is
        on the LEFT of the chain's travel direction, magnitude is the confidence,
        near zero is ambiguous. The `neighbor` rho role names the occlusion
        (divergent) fragment; `partner` names the same-figure fragment. The
        binder turns those into a negative OCCLUSION and a positive
        EDGE_BORDER_OWNERSHIP respectively.
        """
        chain_dets = (current_detections or {}).get("PEAK_CHAIN", [])
        if not chain_dets:
            self._skipped_no_boundary += 1
            return []
        from perception.fragment_ownership import fragment_ownership_records
        records = fragment_ownership_records(
            chain_dets, anchor_positions,
            observation=observation, extraction_scale=extraction_scale)
        emitted = []
        for r in records:
            # [A4#14] a NaN arc/signed/curvature/strength reached the
            # quantisers and the population through the single reader.
            vals = (r["arc"], r["signed"], r["curvature"], r["strength"])
            if not all(math.isfinite(float(v)) for v in vals):
                self._skipped_nonfinite += 1
                continue
            signed = float(r["signed"])
            slots = (
                float(_q_unsigned(float(r["arc"]), 1.0 / 8.0, max_bins=7)),
                float(_q_signed(signed, max_bins=8)),
                float(_q_signed(float(r["curvature"]), max_bins=8)),
                float(_q_unsigned(float(r["strength"]), 1.0 / 8.0,
                                  max_bins=7)),
            )
            # [GRADED POPULATION] The CONTINUOUS ownership measurements: the
            # arc position, the SIGNED ownership direction, the signed
            # curvature and the confidence. The bins stay the lookup key; the
            # activity vector carries the sub-bin position the bin drops.
            _bo_pop = population_from_values(
                "BORDER_OWNERSHIP",
                (float(r["arc"]), signed, float(r["curvature"]),
                 float(r["strength"])))
            rho: Tuple[Tuple[str, Any, int, str], ...] = (
                ("entity", r["alpha"], 0, "+"),
                ("vertex", int(r["vertex"]), 0, "+"),
            )
            if r.get("neighbor") is not None:
                rho += (("neighbor", r["neighbor"], 0, "+"),)
            if r.get("partner") is not None:
                rho += (("partner", r["partner"], 0, "+"),)
            emitted.append(Detection(
                k="BorderOwnership", source_layer=1, scale=0,
                sigma=TypedSignature(
                    kind="CANONICAL", class_="BORDER_OWNERSHIP",
                    relation="FIGURE_SIDE_FIELD",
                    quantized=slots,
                    population=_bo_pop,
                ),
                alpha=r["alpha"], z=min(1.0, max(0.0, float(r["strength"]))), t=t, rho=rho,
            ))
        self._candidates_evaluated += len(records)
        return emitted

    def step(
        self,
        t: int,
        concept_graph: Any,
        sig_provider: Any,
        anchor_positions: Dict[Any, Tuple[float, ...]],
        anchor_velocities: Optional[Dict[Any, Tuple[float, ...]]],
        current_detections: Dict[str, List[Detection]],
        extraction_scale: float = 0.0,
        envelope_mask: Optional[
            Tuple[Tuple[float, float, float, float], ...]
        ] = None,
        observation: Optional[Any] = None,
        update_homeostasis: bool = True,
        object_envelope: Optional[
            Tuple[float, float, float, float]
        ] = None,
        scale_level: int = 0,
    ) -> List[Detection]:
        """[ROUND 10 / C] Fragment ownership is the DEFAULT: ownership is a
        property of an ORIENTED EDGE FRAGMENT (a chain member), never of a
        segmenter or field-region entity.

        [REPRESENTATION MISSION] The region-level ENTITY path is RETIRED: it
        was behind `BO_ENTITY_PATH_ENABLED` (False), so it never ran, and the
        mission's rule is "every guarded path runs or is gone". The fragment
        path is the one that measures the signed figure-ground direction.
        Logged in diagnostics/_retirement_log.jsonl.
        """
        if extraction_scale <= 0.0 or not anchor_positions:
            return []
        if bool(getattr(P, "BO_FRAGMENT_OWNERSHIP_ENABLED", True)):
            emitted = self._emit_fragment_ownership(
                t, current_detections, anchor_positions, extraction_scale,
                observation)
            self._emitted_total += len(emitted)
            return emitted
        return []


# --- [U20 LEAD 2] ONE PHYSICAL CONTOUR = ONE IDENTITY ----------------------
#
# Two spines sample the same physical contour: the FIELD spine (PEAK_CHAIN
# over the continuous contour field) and the ORIENTATION spine (Contour over
# Orientation segments).  Where both see the contour, the field spine owns the
# identity (U19 authority decision); where only the orientation spine sees it,
# it keeps its own anchor.  The mechanism is CANONICAL-ANCHOR SUBSTITUTION:
# the mapped Contour is emitted with the chain's canonical anchor, instead of
# a shared-identity link node (a link is bookkeeping, not a feature with
# content, presence and adjacency -- see the session's framework statement).
#
# The MAPPING is geometric and derived from the data: a contour is the same
# physical contour as a chain when each polyline's members are within one half
# of the other's OWN sampling step (the median inter-member gap), in BOTH
# directions (mutual coverage >= 0.5).  No absolute tolerance, no fitted
# constant (B27).

IDENTITY_MUTUAL_COVERAGE = 0.5


def _segment_distance(p: Tuple[float, float],
                      a: Tuple[float, float],
                      b: Tuple[float, float]) -> float:
    """Distance from `p` to the segment a-b (0-length segments allowed)."""
    ax, ay = a
    bx, by = b
    px, py = p
    dx, dy = bx - ax, by - ay
    den = dx * dx + dy * dy
    if den <= 1e-12:
        return math.hypot(px - ax, py - ay)
    t = ((px - ax) * dx + (py - ay) * dy) / den
    t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def _polyline_distance(p: Tuple[float, float],
                       pts: List[Tuple[float, float]],
                       closed: bool = False) -> float:
    """Distance from `p` to an open (or closed) polyline; inf when it has
    no segments at all (a single point is not a polyline)."""
    if not pts:
        return math.inf
    if len(pts) == 1:
        return math.hypot(p[0] - pts[0][0], p[1] - pts[0][1])
    best = math.inf
    pairs = list(zip(pts, pts[1:]))
    if closed and len(pts) > 2:
        pairs.append((pts[-1], pts[0]))
    for a, b in pairs:
        d = _segment_distance(p, a, b)
        if d < best:
            best = d
    return best


def _sampling_step(polylines: List[List[Tuple[float, float]]]) -> Optional[float]:
    """Median inter-member gap over the given polylines (the data's own step).

    This is the scale the identity tolerance is expressed in: half a step is
    the largest distance a point can sit from the other spine's polyline while
    still being 'the same contour sampled differently'.
    """
    gaps: List[float] = []
    for pts in polylines:
        for a, b in zip(pts, pts[1:]):
            d = math.hypot(b[0] - a[0], b[1] - a[1])
            if math.isfinite(d) and d > 0.0:
                gaps.append(d)
    if not gaps:
        return None
    gaps.sort()
    mid = len(gaps) // 2
    med = gaps[mid] if len(gaps) % 2 else 0.5 * (gaps[mid - 1] + gaps[mid])
    return float(med) if med > 0.0 else None


def chain_member_positions(det: Detection,
                           anchor_positions: Optional[Dict[Any, Tuple[float, ...]]]
                           ) -> List[Tuple[float, float]]:
    """A PEAK_CHAIN's member positions, in rho ordinal order.

    The members are the chain's own peaks; their anchors are owned by PeakLens.
    A member whose anchor has no live position is skipped (never defaulted to
    the origin, which would fabricate geometry).
    """
    if not anchor_positions:
        return []
    pairs: List[Tuple[int, Tuple[float, float]]] = []
    for role in det.rho:
        # [A4#13] length BEFORE indexing (an empty role raised IndexError), and
        # int() instead of isinstance(int) (numpy integers were dropped).
        if len(role) < 3 or role[0] != "member":
            continue
        try:
            ordinal = int(role[2])
        except (TypeError, ValueError):
            continue
        pos = anchor_positions.get(role[1])
        if pos is None or len(pos) < 2:
            continue
        pairs.append((ordinal, (float(pos[0]), float(pos[1]))))
    pairs.sort(key=lambda p: p[0])
    return [p for _, p in pairs]


def _coverage(a_pts: List[Tuple[float, float]],
              b_pts: List[Tuple[float, float]],
              tol: float, closed_b: bool = False) -> float:
    """Fraction of `a_pts` within `tol` of the `b` polyline."""
    if not a_pts:
        return 0.0
    hit = sum(1 for p in a_pts if _polyline_distance(p, b_pts, closed_b) <= tol)
    return hit / float(len(a_pts))


def canonical_anchor_map(
    contours: List[Tuple[List[Tuple[float, float]], bool]],
    chains: List[Tuple[Any, List[Tuple[float, float]]]],
    *,
    mutual_coverage: float = IDENTITY_MUTUAL_COVERAGE,
) -> Dict[int, Any]:
    """Map contour index -> canonical (chain) anchor for doubly-seen contours.

    Args:
        contours: [(member positions, closed flag)] in emission order.
        chains:   [(canonical anchor, member positions)] of the field spine.

    Returns a 1:1 assignment {contour_index: chain_anchor}.  Ambiguous pairs are
    resolved deterministically by (mutual coverage desc, contour index asc,
    chain index asc), and each chain is claimed by at most one contour, so the
    substitution cannot merge two physical contours onto one identity.
    """
    if not contours or not chains:
        return {}
    # [A4#7] Per-population tolerance: a contour point must be within half
    # the CHAIN's own step and vice versa, so neither population is scored at
    # whichever sampling happens to be coarser.
    step_c = _sampling_step([pts for pts, _ in contours])
    step_k = _sampling_step([pts for _, pts in chains])
    if step_c is None or step_k is None:
        return {}
    half_c, half_k = _canonical_tolerances(step_c, step_k)
    pad = max(half_c, half_k)

    def _bbox(pts):
        xs_ = [p[0] for p in pts]
        ys_ = [p[1] for p in pts]
        return (min(xs_), min(ys_), max(xs_), max(ys_))

    c_boxes = [_bbox(p) if p else None for p, _ in contours]
    k_boxes = [_bbox(p) if p else None for _, p in chains]
    pairs: List[Tuple[float, int, int]] = []
    for ci, (c_pts, c_closed) in enumerate(contours):
        if not c_pts or c_boxes[ci] is None:
            continue
        cb = c_boxes[ci]
        for ki, (anchor, k_pts) in enumerate(chains):
            if not k_pts or k_boxes[ki] is None:
                continue
            kb = k_boxes[ki]
            # bounding-box reject: two polylines further apart than the pad
            # cannot cover each other
            if (cb[2] + pad < kb[0] or kb[2] + pad < cb[0]
                    or cb[3] + pad < kb[1] or kb[3] + pad < cb[1]):
                continue
            c_cov = _coverage(c_pts, k_pts, half_k, closed_b=False)
            if c_cov < mutual_coverage:
                continue
            k_cov = _coverage(k_pts, c_pts, half_c, closed_b=c_closed)
            if k_cov < mutual_coverage:
                continue
            pairs.append((-min(c_cov, k_cov), ci, ki))
    pairs.sort()
    out: Dict[int, Any] = {}
    used_chains: Set[int] = set()
    for _score, ci, ki in pairs:
        if ci in out or ki in used_chains:
            continue
        out[ci] = chains[ki][0]
        used_chains.add(ki)
    return out


def _retarget_contour(det: Detection, canonical_anchor: Any) -> Detection:
    """The same Contour, with the canonical identity.

    Only the IDENTITY moves: `alpha` and the `entity` role become the chain's
    canonical anchor.  Every `member` role (the Orientation anchors) is kept
    verbatim, so the members -- and therefore the 26% only the orientation
    spine sees -- are untouched.
    """
    rho = tuple(
        (("entity", canonical_anchor, 0, "+") if role[0] == "entity" else role)
        for role in det.rho
    )
    # [A4#12] Detection is IMMUTABLE and not a dataclass: dataclasses.replace
    # raises TypeError and copy.copy raises AttributeError (core/types.py:580
    # refuses __setattr__), so the copy is explicit and lists every field.
    return Detection(
        k=det.k, source_layer=det.source_layer, scale=det.scale,
        sigma=det.sigma, alpha=canonical_anchor, z=det.z, t=det.t, rho=rho,
        substrate_hash=det.substrate_hash,
    )


#: The detection kinds that are ALL one physical oriented segment. The V1
#: emitter produces both families for the same anchor (the polarised one adds
#: the polarity relation slot); the binder's continuity edges pick one
#: representative per anchor, preferring ORIENTATION_POL (see the binder's
#: _run_orientation_continuity).  A lens that reads only one of the two
#: buckets mints a node
#: identity for the segment that no edge in the graph refers to -- measured:
#: 0 of 48 endpoint ids shared, diagnostics/_u21_id_mismatch.json.
_ORIENTATION_FAMILY_KINDS = ("Orientation", "OrientationPol")
_ORIENTATION_FAMILY_CLASSES = (
    "ORIENTATION", "ORIENTATION_POL", "ORIENTATION_GABOR",
)


def orientation_representatives(
    current_detections: Dict[str, List[Detection]],
) -> Dict[Any, Detection]:
    """ONE representative detection per ANCHOR, across the orientation families.

    The same rule the binder applies when it lays the continuity edges (prefer
    ORIENTATION_POL, deterministic tie-break by the canonical detection sort),
    so the node ids this returns are the ids those edges connect.  One anchor
    is one physical segment, so this is also the one-node-one-identity rule,
    not a merge of distinct structure.
    """
    rep: Dict[Any, Detection] = {}
    dets = []
    for kind in _ORIENTATION_FAMILY_KINDS:
        dets.extend(current_detections.get(kind, []) or [])
    for det in sorted(dets, key=_detection_sort_key):
        if getattr(det.sigma, "class_", None) not in _ORIENTATION_FAMILY_CLASSES:
            continue
        if len(det.sigma.quantized) < 3:
            continue
        cur = rep.get(det.alpha)
        if cur is None or (
            det.sigma.class_ == "ORIENTATION_POL"
            and cur.sigma.class_ != "ORIENTATION_POL"
        ):
            rep[det.alpha] = det
    return rep


def edge_association_strength(edge: Any, t: int) -> float:
    """The association strength valid DURING step `t`.

    `Edge.weight` is the channel's homeostatic EMA and is written by
    `ConceptGraph.step_28_update_edges`, which the pipeline runs in Phase 5 --
    AFTER the higher-lens tier.  Reading `weight` from a lens therefore reads
    the PREVIOUS step's value, and 0.0 for an edge's first step, so a freshly
    laid edge can never clear a threshold: measured, all 56 continuity edges
    of a fresh step read `weight 0.0` while carrying `pending_signal` 0.17-0.70
    (diagnostics/_u21_id_mismatch.json).  The raw signal recorded by the binder
    at upsert time is the field that describes the edge while the tier runs.
    """
    if getattr(edge, "observed_at", -1) == int(t):
        return float(getattr(edge, "pending_signal", 0.0))
    return float(getattr(edge, "weight", 0.0))


class ContourLens(HigherLens):
    """B.6 — V2 contour chains over Orientation segments.

    Builds a neighbour graph over Orientation pattern nodes from
    ORIENTATION_CONTINUITY and CURVATURE edges, prunes edges running
    against the figure-ground direction implied by BorderOwnership, then
    takes connected components of degree <= 2 as contour chains.
    """

    #: how many contours adopted a field-spine anchor in THIS step (U20 lead 2)
    #: [A4#11] reset at the top of step(); it kept the previous step's value
    #: whenever `pending` was empty, and the read counters had no defaults.
    _canonical_mapped = 0
    _candidates_evaluated = 0
    _skipped_no_position = 0

    def step(
        self,
        t: int,
        concept_graph: Any,
        sig_provider: Any,
        anchor_positions: Dict[Any, Tuple[float, ...]],
        anchor_velocities: Optional[Dict[Any, Tuple[float, ...]]],
        current_detections: Dict[str, List[Detection]],
        extraction_scale: float = 0.0,
        envelope_mask: Optional[
            Tuple[Tuple[float, float, float, float], ...]
        ] = None,
        observation: Optional[Any] = None,
        update_homeostasis: bool = True,
        object_envelope: Optional[
            Tuple[float, float, float, float]
        ] = None,
        scale_level: int = 0,
    ) -> List[Detection]:
        self._canonical_mapped = 0
        if extraction_scale <= 0.0:
            self._skipped_no_position += 1
            return []

        # Node ids of the Orientation segments, which are the contour's
        # members. Arity must be 3 (theta, elongation, anchor trunk).
        #
        # [U21 LEAD 1] ONE NODE PER PHYSICAL SEGMENT.  The V1 emitter produces
        # two detection families for the same anchor (ORIENTATION and
        # ORIENTATION_POL), and the binder's continuity edges pick one
        # representative per anchor, preferring the polarised family. Reading
        # only the "Orientation" bucket minted a node id per segment that NO
        # stored edge referenced: 445 detections resolved to 289 ORIENTATION
        # nodes, every continuity edge joined ORIENTATION_POL nodes, and the
        # intersection was empty (diagnostics/_u21_id_mismatch.json). The
        # member set is therefore built from the anchor representatives, which
        # is also what makes it one identity per segment rather than two.
        # [U25 LEAD 4 / V2] ONE INSTANCE PER PHYSICAL SEGMENT, carrying the
        # pattern id that AUTHORISES adjacency.  G_c is keyed by INSTANCE
        # index, so two same-signature segments at distant positions can no
        # longer collapse into one pattern node (the teleportation bug); the
        # pattern id only AUTHORISES a link and the spatial distance within
        # the link radius FORMS it.
        rep_dets: List[Detection] = []
        # [A4#11] same default as JunctionLens: one parameter, one reading.
        if getattr(P, "ORIENTATION_SPINE_WIRED", True):
            rep_dets = sorted(
                orientation_representatives(current_detections).values(),
                key=_detection_sort_key,
            )
        else:
            rep_dets = sorted(
                (
                    det
                    for det in (current_detections.get("Orientation") or [])
                    if len(det.sigma.quantized) >= 3
                ),
                key=_detection_sort_key,
            )
        inst_nid: Dict[int, int] = {}
        nid_to_inst: Dict[int, List[int]] = defaultdict(list)
        for i, det in enumerate(rep_dets):
            nid = concept_graph.get_or_create_pattern_node(
                det.k, _node_signature_hash(det.sigma), signature=det.sigma
            )
            inst_nid[i] = nid
            nid_to_inst[nid].append(i)
        inst_det: List[Detection] = rep_dets

        bo_by_anchor: Dict[Any, Detection] = {}
        for det in sorted(
            current_detections.get("BorderOwnership", []),
            key=_detection_sort_key,
        ):
            bo_by_anchor.setdefault(det.alpha, det)

        # 1. Build G_c by explicit incident-edge iteration. Per-node
        # get_outgoing/get_incoming avoids inventing a graph-wide iterator;
        # sets make the two passes idempotent.
        G_c: Dict[int, Set[int]] = defaultdict(set)
        authorised: Set[Tuple[int, int]] = set()
        for nid in sorted({inst_nid[i] for i in inst_nid}):
            for edge in concept_graph.get_outgoing_edges(nid):
                if edge.edge_type not in (
                    EdgeType.ORIENTATION_CONTINUITY, EdgeType.CURVATURE
                ):
                    continue
                if edge_association_strength(edge, t) < P.THETA_CONTOUR_EDGE:
                    continue
                if edge.child in nid_to_inst:
                    authorised.add((nid, edge.child))
            for edge in concept_graph.get_incoming_edges(nid):
                if edge.edge_type not in (
                    EdgeType.ORIENTATION_CONTINUITY, EdgeType.CURVATURE
                ):
                    continue
                if edge_association_strength(edge, t) < P.THETA_CONTOUR_EDGE:
                    continue
                if edge.parent in nid_to_inst:
                    authorised.add((edge.parent, nid))
        link_radius = max(float(P.KAPPA_CONTOUR_CONTINUITY) * extraction_scale,
                          1e-6)
        for pa, pb in sorted(authorised):
            for i in nid_to_inst.get(pa, ()):
                for j in nid_to_inst.get(pb, ()):
                    if i == j:
                        continue
                    ai = inst_det[i].alpha
                    aj = inst_det[j].alpha
                    if ai not in anchor_positions or aj not in anchor_positions:
                        continue
                    if _pos_dist(anchor_positions[ai], anchor_positions[aj]) <= link_radius:
                        G_c[i].add(j)
                        G_c[j].add(i)
        # [U25 V2] same-signature instances at the same position are duplicate
        # observations of one segment: link them so they form a chain instead
        # of being silently dropped (no teleport across distance).
        for nid, insts in nid_to_inst.items():
            for a in range(len(insts)):
                for b in range(a + 1, len(insts)):
                    i, j = insts[a], insts[b]
                    ai = inst_det[i].alpha
                    aj = inst_det[j].alpha
                    if ai not in anchor_positions or aj not in anchor_positions:
                        continue
                    if _pos_dist(anchor_positions[ai], anchor_positions[aj]) <= link_radius:
                        G_c[i].add(j)
                        G_c[j].add(i)

        # 2. Figure-ground gating. With no BorderOwnership reading for the
        # source anchor there is no evidence either way, so the edge stays.
        edges_to_remove: Set[Tuple[int, int]] = set()
        for u in sorted(G_c.keys()):
            for v in sorted(G_c[u]):
                if u >= v:
                    continue
                a_u = inst_det[u].alpha
                a_v = inst_det[v].alpha
                if a_u not in anchor_positions or a_v not in anchor_positions:
                    continue
                pos_u = anchor_positions[a_u]
                pos_v = anchor_positions[a_v]
                dx = pos_v[0] - pos_u[0]
                dy = pos_v[1] - pos_u[1]
                edge_dir_bin = int(math.floor(
                    ((math.atan2(dy, dx) % (2 * math.pi)) * 8)
                    / (2 * math.pi)
                ))
                bo_det = bo_by_anchor.get(a_u)
                if bo_det is None or len(bo_det.sigma.quantized) < 1:
                    continue
                if bo_det.sigma.relation == "FIGURE_SIDE_FIELD":
                    # [F4] The signed ownership is a SIGNED SCALAR in
                    # [-1,1] (read with the single reader,
                    # perception.ownership.ownership_signed), NOT an
                    # angle: comparing it with the edge's bearing over
                    # 2*pi was the category error ownership.py
                    # documents. The well-typed comparison is the SIDE
                    # of the neighbour relative to the SOURCE CONTOUR's
                    # travel direction, against the side the ownership
                    # says the figure is on.
                    signed = ownership_signed(bo_det)
                    if abs(signed) < 0.5:
                        continue
                    tang = _contour_tangent(inst_det[u],
                                           anchor_positions,
                                           list(current_detections.get(
                                               "PEAK_CHAIN", []) or []))
                    if tang is None:
                        continue
                    cross = tang[0] * dy - tang[1] * dx
                    if abs(cross) < 1e-9:
                        continue
                    side = 1.0 if cross > 0.0 else -1.0
                    # signed > 0: the figure is on the LEFT of the
                    # fragment's travel direction. The edge is dropped
                    # when the neighbour lies on the OTHER side.
                    if side * signed < 0.0:
                        edges_to_remove.add((u, v))
        for (u, v) in edges_to_remove:
            G_c[u].discard(v)
            G_c[v].discard(u)

        # 3. Connected components.
        visited: Set[int] = set()
        components: List[Set[int]] = []
        for start in sorted(G_c.keys()):
            if start in visited:
                continue
            comp: Set[int] = set()
            stack = [start]
            visited.add(start)
            while stack:
                x = stack.pop()
                comp.add(x)
                for y in sorted(G_c.get(x, ())):
                    if y not in visited:
                        visited.add(y)
                        stack.append(y)
            components.append(comp)

        emitted: List[Detection] = []
        # [U20 LEAD 2] Contours are held until every component has been walked:
        # the canonical-anchor mapping is a property of the whole step (one
        # contour per chain), so emission happens after the mapping exists.
        pending: List[Tuple[Detection, Optional[List[Tuple[float, float]]],
                             bool]] = []

        def _emit_chain(chain: List[int], closed: bool) -> None:
            """Emit one Contour detection for `chain` ([B5] curvature sequence)."""
            if len(chain) < 2:
                return
            self._candidates_evaluated += 1
            n_theta = P.ORIENTATION_N_THETA
            # [O2] The chain members' THETA, by name (the member
            # may be an ORIENTATION, an ORIENTATION_POL or an
            # ORIENTATION_SEGMENT).
            # [U21] RADIANS from the shared family reader, then to this
            # lens's bin: a GABOR member's theta is a bank index over a
            # 12-channel bank, so reading it as a 24-bin geometry slot would
            # halve it (base.py records that exact back-read bug).
            # [A4#5] floor, not round: a bin centre lands on exactly i + 0.5
            # after the scale, and Python's round() goes to EVEN, so the bins
            # alternated by +-1 (2.5 -> 2, 3.5 -> 4). floor(+eps) is the same
            # bin for every centre.
            bins = [
                int(math.floor(orientation_theta_radians(
                    inst_det[m].sigma)
                    * n_theta / math.pi + 1e-9)) % n_theta
                for m in chain
            ]
            member_anchors = [inst_det[m].alpha for m in chain]

            # Real geometry when every member anchor is live; otherwise
            # uniform spacing and no geometric turn resolution.
            pts = None
            if all(a in anchor_positions for a in member_anchors):
                pts = [
                    (float(anchor_positions[a][0]), float(anchor_positions[a][1]))
                    for a in member_anchors
                ]

            # [GRADED POPULATION] The CONTINUOUS contour measurements, in
            # the order the CONTOUR graded layout declares
            # (core.registries.POPULATION_SLOTS["CONTOUR"]): the arc length
            # (in members), the mean AXIAL orientation, the circular
            # variance, the closure flag, the log-compressed total turn,
            # the corner count, the net signed turn (chirality) and the 20
            # signed, log-compressed curvatures of the canonical sequence.
            # The bins stay the lookup key; the activity vector keeps the
            # sub-bin position each bin rounds off.
            #
            # [P1-7] THE CLOSURE IS TOPOLOGICAL. Open and closed are two
            # states of a property, not two points on a scale: there is no
            # quantity between them, so the slot is a FLAG and the
            # population carries it as a 0/1 activity on the unit table's
            # two-level "closure" axis -- never as a fake gradient that
            # would read "nearly closed".
            _cm = _contour_measurements(bins, closed, pts)
            quantized = _contour_slots_from_measurements(_cm)
            _contour_pop = population_from_values(
                "CONTOUR",
                population_values_in_span(
                    "CONTOUR",
                    (
                        # the arc length in MEMBERS (the unit table's span is
                        # [0, 32] members over 8 levels, the same quarter
                        # count the bin names)
                        float(_cm["n"]),
                        float(_cm["mean_theta"]),
                        float(_cm["circ_var"]),
                        1.0 if _cm["closed"] else 0.0,
                        float(_cm["total_turn"]),
                        float(_cm["corners"]),
                        float(_cm["chirality"]),
                    ) + tuple(float(c) for c in _cm["curv"]),
                ),
            )

            # alpha must be a live observation anchor: the FIRST MEMBER's
            # anchor by lowest node id (direction-invariant).
            first_member_node_id = min(chain)
            first_member_anchor = inst_det[first_member_node_id].alpha

            pending.append((Detection(
                k="Contour", source_layer=1, scale=0,
                sigma=TypedSignature(
                    kind="CANONICAL", class_="CONTOUR",
                    relation="ARC_FRAGMENT",
                    quantized=quantized,
                    population=_contour_pop,
                ),
                alpha=first_member_anchor,
                z=min(1.0, len(chain) / 8.0), t=t,
                rho=(("entity", first_member_anchor, 0, "+"),)
                    + tuple(("member", member_anchors[i], i, "+") for i in range(len(chain)))
                    + tuple(("member_node", (inst_nid[chain[i]],), i, "+") for i in range(len(chain))),
            ), pts, closed))

        for comp in components:
            branch_nodes = sorted(n for n in comp if len(G_c.get(n, ())) > 2)

            if branch_nodes:
                # [Fix 2] Split at branch points into arms.
                #
                # Pre-fix rule deleted the ENTIRE component when any node
                # branched. Corpus measurement (config C, ETH-80): 648
                # components / 10,809 nodes discarded this way — a third
                # of all contour structure, and the sole cause of the
                # single true zero-CC image (car14-map: closed silhouette
                # ring, 97 nodes, lattice-staircase degree-3 artefacts).
                #
                # Arm = maximal simple path between two branch nodes or
                # between a branch node and a dead end. Every edge belongs
                # to exactly one arm; `used` tracks edge membership with
                # undirected key `(min, max)`.
                #
                # Closure detection (self-audit A.1 fix): if the walk
                # returns to its OWN starting branch node (`curr == bn`),
                # it has traced a cycle — strip the duplicate terminal
                # node and emit `closed=True`. If it terminates at a
                # DIFFERENT branch node (or a dead end), emit
                # `closed=False`. The pre-audit draft hardcoded
                # `closed=False` for all branch arms, silently losing
                # closure information on lollipop components.
                branch_set = set(branch_nodes)
                used: Set[Tuple[int, int]] = set()
                for bn in branch_nodes:
                    for nb in sorted(G_c.get(bn, ())):
                        e_key = (min(bn, nb), max(bn, nb))
                        if e_key in used:
                            continue
                        used.add(e_key)
                        arm = [bn, nb]
                        prev, curr = bn, nb
                        while curr not in branch_set:
                            nxts = [
                                x for x in G_c.get(curr, ()) if x != prev
                            ]
                            if not nxts:
                                break
                            nxt = min(nxts)
                            e2 = (min(curr, nxt), max(curr, nxt))
                            if e2 in used:
                                break
                            used.add(e2)
                            arm.append(nxt)
                            prev, curr = curr, nxt
                        # Closure: only when the walk returned to its OWN
                        # starting branch node. A walk ending at a
                        # different branch node is an inter-branch arm
                        # and is correctly open. A walk ending on a dead
                        # end or a used edge is also correctly open.
                        is_closed_arm = (curr == bn)
                        if is_closed_arm:
                            # Strip the duplicate terminal node so rho
                            # carries each node exactly once. The stripped
                            # arm has >= 3 nodes: a cycle through a
                            # degree->=3 node traverses >= 2 intermediate
                            # nodes, so `_emit_chain`'s `len < 2` guard
                            # is not reached from this path.
                            arm = arm[:-1]
                        _emit_chain(arm, closed=is_closed_arm)
                continue

            # ---- Unbranched component: pre-fix behaviour preserved ----
            if len(comp) < 2:
                continue

            degree_one = [n for n in comp if len(G_c.get(n, ())) == 1]
            if len(degree_one) == 2:
                # Open chain: walk from the lowest-id endpoint.
                start = min(degree_one)
                chain = [start]
                prev = None
                current = start
                for _ in range(len(comp) + 1):
                    neighbors = [
                        n for n in G_c.get(current, ()) if n != prev
                    ]
                    if not neighbors:
                        break
                    nxt = min(neighbors)
                    if nxt == start:
                        break
                    chain.append(nxt)
                    prev, current = current, nxt
                closed = False
            elif len(degree_one) == 0 and all(
                len(G_c.get(n, ())) == 2 for n in comp
            ):
                # Cycle: enter it at its lowest-id node, deterministically.
                component_sorted = sorted(comp)
                start = component_sorted[0]
                neighbors = sorted(G_c[start])
                current = neighbors[0]
                chain = [start, current]
                prev = start
                for _ in range(len(comp) + 1):
                    neighbors = [
                        n for n in G_c.get(current, ()) if n != prev
                    ]
                    if not neighbors:
                        break
                    nxt = min(neighbors)
                    if nxt == start:
                        break
                    chain.append(nxt)
                    prev, current = current, nxt
                closed = True
            else:
                # Structurally impossible under connectedness; retained
                # so a corrupted G_c cannot be silently mislabelled.
                continue

            _emit_chain(chain, closed=closed)

        # [U20 LEAD 2] One physical contour, one identity.  Where this contour
        # and a field-spine PEAK_CHAIN are the same contour, the Contour adopts
        # the chain's canonical anchor; where no chain samples it, the contour
        # keeps its orientation anchor (the 26% is preserved verbatim).
        if pending:
            mapping: Dict[int, Any] = {}
            if bool(getattr(P, "SPINE_IDENTITY_UNIFIED", True)):
                chains_geo = []
                for det in sorted(current_detections.get("PEAK_CHAIN", []),
                                  key=_detection_sort_key):
                    pts_k = chain_member_positions(det, anchor_positions)
                    if len(pts_k) >= 2:
                        chains_geo.append((det.alpha, pts_k))
                mapping = canonical_anchor_map(
                    [(pts_c if pts_c is not None else [], closed_c)
                     for _det, pts_c, closed_c in pending],
                    chains_geo,
                )
            self._canonical_mapped = len(mapping)
            for i, (det, _pts, _closed) in enumerate(pending):
                anchor = mapping.get(i)
                emitted.append(det if anchor is None
                               else _retarget_contour(det, anchor))
        self._emitted_total += len(emitted)
        return emitted

def _texture_confidence(material: Any, lum_dist: Any) -> float:
    """[U25 V5] z for a TextureBoundary: material discontinuity attenuated
    by the luminance distance.  Pure texture (d_contrast/d_ori high, lum_dist
    ~0) survives; a luminance-explained boundary (lum_dist >= 0.5) scores 0."""
    _material = min(1.0, max(0.0, float(material)))
    return _material * (1.0 - min(1.0, float(lum_dist) / 0.5))


#: [A4#8] One 8-bit level. The floor keeps math.log finite near black: with
#: the old +1e-9 the ratio of a 0.5 flank to a black one reached log(5e8) = 20
#: and the value dominated every other cue.
_LUM_FLOOR = 1.0 / 255.0


def _dlogsf(brighter: Any, darker: Any) -> float:
    """[U25 V6] Log-ratio of the BRIGHTER to the DARKER flank.

    [A4#8] This is >= 0 by construction -- it is direction-independent BY
    DESIGN, so it cannot also be signed. The caller therefore has to pick the
    flanks by luminance (see `_step_field`), not by `lum_diff`'s sign.
    """
    return math.log(max(float(brighter), _LUM_FLOOR)
                    / max(float(darker), _LUM_FLOOR))


def _canonical_tolerances(step_c: Any, step_k: Any) -> Tuple[float, float]:
    """[U25 V8] Per-population canonical-anchor tolerance: half each step."""
    return (0.5 * float(step_c), 0.5 * float(step_k))


def _arm_bearing_clusters(pairs: Any, tol: float) -> List[Any]:
    """[U25 V9] Angular NMS on the arm candidate list: keep the nearest
    neighbour per bearing cluster (separation <= tol).  Eliminates the
    zero wrap-around gap case (wrap-around is merged)."""
    if not pairs:
        return []
    ordered = sorted(pairs, key=lambda p: p[0])
    clusters: List[List[Any]] = []
    for p in ordered:
        if clusters and (p[0] - clusters[-1][-1][0]) <= tol:
            clusters[-1].append(p)
        else:
            clusters.append([p])
    # wrap-around merge
    if len(clusters) > 1:
        last = clusters[-1]
        first = clusters[0]
        if (2.0 * math.pi - last[-1][0] + first[0][0]) <= tol:
            merged = last + first
            clusters = [merged] + clusters[1:-1]
    # [A4#6] the key is the DISTANCE (p[3]); p[2] is the instance index.
    return [min(cl, key=lambda p: p[3]) for cl in clusters]


def _hub_order(G: Any, comp: Any) -> List[int]:
    """[A4#2] Junction-hub CANDIDATES of one component, highest degree first.

    A component may contain several junctions; the caller used to take
    ``max(comp, key=degree)`` and collapse all but one.  Spatial suppression
    happens in the lens, AFTER pass-throughs are ruled out (a pass-through must
    not reserve the position of a real junction).
    """
    return sorted(
        (n for n in comp if len(G.get(n, ())) >= 2),
        key=lambda n: (-len(G[n]), n),
    )


def _is_pass_through(topo: int, n_ang: int) -> bool:
    """[U25 V3] A two-arm straight continuation is a pass-through, not a
    junction (topo 0, n_ang 2)."""
    return int(topo) == 0 and int(n_ang) == 2


def junction_family(topo: int, n_arms: int) -> str:
    """[U20 LEAD 3] The FAMILY of one JunctionLens candidate.

    The topo classes are JunctionLens' own hand-written classifier (see the
    `topo` assignment below); this function only NAMES the family, so the
    decision is one lookup instead of a second classifier.

    Families (each a node kind with its own presence condition):

    ======  ==========================================================
    Corner  exactly two boundary arms meet at a non-collinear angle
            (the boundary turns; nothing continues through)
    T       exactly three arms meet and one PAIR is collinear: one
            boundary continues through, a third terminates against it
    Y       exactly three arms meet, no pair collinear, every gap
            <= pi: three boundary segments meet at a common point
    ======  ==========================================================

    Residual "Junction" (kept, aliased) covers the geometries the three
    families do not describe: the two-arm continuation (topo 0, a straight
    boundary with no junction at all), the four-arm crossing (topo 4), the
    five-plus star (topo 5) and the reflex three-arm figure (topo 6, all
    three arms inside one half-plane -- a crow's foot, not a Y).
    """
    t = int(topo)
    if n_arms == 2 and t == 1:
        return "Corner"
    if n_arms == 3 and t == 2:
        return "T"
    if n_arms == 3 and t == 3:
        return "Y"
    return JUNCTION_RESIDUAL_KIND


class JunctionLens(HigherLens):
    """B.7 + B21 — V2 junctions over Orientation segments.

    Builds an instance-level spatial graph of Orientation detections,
    gated by pattern-level ORIENTATION_BEND/JUNCTION edges. Computes
    a rotation-invariant shape descriptor (topology + canonical gaps)
    anchored by the absolute reference angle of the canonical arm,
    with arm support, arm sharpness, and pairwise angle distribution (arity 19).
    """

    def step(
        self,
        t: int,
        concept_graph: Any,
        sig_provider: Any,
        anchor_positions: Dict[Any, Tuple[float, ...]],
        anchor_velocities: Optional[Dict[Any, Tuple[float, ...]]],
        current_detections: Dict[str, List[Detection]],
        extraction_scale: float = 0.0,
        envelope_mask: Optional[
            Tuple[Tuple[float, float, float, float], ...]
        ] = None,
        observation: Optional[Any] = None,
        update_homeostasis: bool = True,
        object_envelope: Optional[
            Tuple[float, float, float, float]
        ] = None,
        scale_level: int = 0,
    ) -> List[Detection]:
        if extraction_scale <= 0.0:
            self._skipped_no_position += 1
            return []

        # [U21] The same member set the continuity edges join: every family
        # that IS an oriented segment, so the pattern edges laid over the
        # polarised family are visible here as well as the GABOR geometry
        # (the two families occupy DISJOINT anchor sets -- measured 0 shared
        # anchors -- so this is a union, not a merge of two views of one
        # segment).
        if getattr(P, "ORIENTATION_SPINE_WIRED", True):
            orient_dets = sorted(
                [d for _kind in _ORIENTATION_FAMILY_KINDS
                 for d in (current_detections.get(_kind, []) or [])],
                key=_detection_sort_key,
            )
        else:
            orient_dets = sorted(
                current_detections.get("Orientation", []),
                key=_detection_sort_key,
            )
        if not orient_dets:
            return []

        det_to_nid: Dict[int, int] = {}
        for i, det in enumerate(orient_dets):
            if len(det.sigma.quantized) < 2:
                continue
            nid = concept_graph.get_or_create_pattern_node(
                det.k, _node_signature_hash(det.sigma), signature=det.sigma
            )
            det_to_nid[i] = nid

        pattern_edges: Set[Tuple[int, int]] = set()
        for nid in set(det_to_nid.values()):
            for edge in concept_graph.get_outgoing_edges(nid):
                if edge.edge_type in (EdgeType.ORIENTATION_BEND, EdgeType.ORIENTATION_JUNCTION):
                    pattern_edges.add((nid, edge.child))
                    pattern_edges.add((edge.child, nid))
            for edge in concept_graph.get_incoming_edges(nid):
                if edge.edge_type in (EdgeType.ORIENTATION_BEND, EdgeType.ORIENTATION_JUNCTION):
                    pattern_edges.add((edge.parent, nid))
                    pattern_edges.add((nid, edge.parent))

        G_j: Dict[int, Set[int]] = defaultdict(set)
        junction_radius = getattr(P, "KAPPA_JUNCTION_RADIUS", 2.0) * extraction_scale
        cell_size = max(junction_radius, 1e-6)
        metric_epsilon = 1e-3 * max(junction_radius, 1e-6)

        grid: Dict[Tuple[int, int], List[int]] = defaultdict(list)
        for i, det in enumerate(orient_dets):
            if i not in det_to_nid:
                continue
            pos = anchor_positions.get(det.alpha)
            if pos and len(pos) >= 2:
                cell = (int(pos[0] // cell_size), int(pos[1] // cell_size))
                grid[cell].append(i)

        for i, det in enumerate(orient_dets):
            if i not in det_to_nid:
                continue
            pos_i = anchor_positions.get(det.alpha)
            if not pos_i or len(pos_i) < 2:
                continue
            cell = (int(pos_i[0] // cell_size), int(pos_i[1] // cell_size))
            nid_i = det_to_nid[i]

            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for j in grid.get((cell[0] + dx, cell[1] + dy), []):
                        if i >= j:
                            continue
                        nid_j = det_to_nid[j]
                        if (nid_i, nid_j) not in pattern_edges:
                            continue
                        pos_j = anchor_positions.get(orient_dets[j].alpha)
                        if not pos_j or len(pos_j) < 2:
                            continue
                        if _pos_dist(pos_i, pos_j) <= junction_radius:
                            G_j[i].add(j)
                            G_j[j].add(i)

        CONTINUITY_NBR = _collect_continuity_neighbours(
            concept_graph, orient_dets, det_to_nid, anchor_positions,
            junction_radius, grid
        )

        visited: Set[int] = set()
        components: List[Set[int]] = []
        for start in sorted(G_j.keys()):
            if start in visited:
                continue
            comp: Set[int] = set()
            stack = [start]
            visited.add(start)
            while stack:
                x = stack.pop()
                comp.add(x)
                for y in sorted(G_j.get(x, ())):
                    if y not in visited:
                        visited.add(y)
                        stack.append(y)
            components.append(comp)

        emitted: List[Detection] = []
        for comp in components:
            if len(comp) < 3:
                continue

            # [A4#2] EVERY hub candidate, not one per component.
            accepted_hubs: List[Tuple[float, float]] = []
            for hub_idx in _hub_order(G_j, comp):
                hub_neighbors = sorted(G_j[hub_idx])
                hub_pos = anchor_positions.get(orient_dets[hub_idx].alpha)
                if not hub_pos or len(hub_pos) < 2:
                    continue

                pairs: List[Tuple[float, Any, int, float]] = []
                seen_anchors: Set[Any] = set()
                for j in hub_neighbors:
                    a = orient_dets[j].alpha
                    if a in seen_anchors:
                        continue
                    pos_j = anchor_positions.get(a)
                    if not pos_j or len(pos_j) < 2:
                        continue
                    dx = pos_j[0] - hub_pos[0]
                    dy = pos_j[1] - hub_pos[1]
                    if math.hypot(dx, dy) <= metric_epsilon:
                        continue
                    seen_anchors.add(a)
                    # [A4#10] The arm tangent was computed here and NEVER read (it
                    # gated nothing), so the call did work and could raise on a
                    # malformed signature. The arm is accepted; the bearing
                    # clustering below governs the count (V4 stays relaxed).
                    # [A4#6] carry the DISTANCE: the cluster key is the nearest
                    # neighbour per bearing, not the lowest instance index.
                    pairs.append((math.atan2(dy, dx) % (2 * math.pi), a, j,
                                  math.hypot(dx, dy)))

                # [A4#6] This helper existed and was never called, so two arms on
                # nearly the same bearing still counted as two arms (a 2-arm
                # corner scored as a 3-arm T). Tolerance is PROVISIONAL.
                pairs = _arm_bearing_clusters(
                    pairs, float(getattr(P, "JUNCTION_ARM_CLUSTER_TOL",
                                         math.pi / 36.0)))
                n_ang = len(pairs)
                if n_ang < 2:
                    continue

                self._candidates_evaluated += 1

                pairs.sort(key=lambda p: (p[0], serialize_value(p[1])))
                angles = [p[0] for p in pairs]
                member_anchors = [p[1] for p in pairs]
                arm_idx = [p[2] for p in pairs]

                gaps: List[float] = []
                for i in range(n_ang):
                    g = (angles[(i + 1) % n_ang] - angles[i]) % (2 * math.pi)
                    gaps.append(g)

                gap_bins_for_sort = [
                    math.floor(g * _JUNCTION_CANONICAL_BINS / (2 * math.pi))
                    for g in gaps
                ]
                rotations = [
                    tuple(gap_bins_for_sort[i:] + gap_bins_for_sort[:i])
                    for i in range(n_ang)
                ]
                min_seq = min(rotations)
                indices = [i for i, r in enumerate(rotations) if r == min_seq]
                min_shift = indices[0]
                ref_is_unambiguous = len(indices) == 1

                angles = angles[min_shift:] + angles[:min_shift]
                gaps = gaps[min_shift:] + gaps[:min_shift]
                member_anchors = member_anchors[min_shift:] + member_anchors[:min_shift]
                arm_idx = arm_idx[min_shift:] + arm_idx[:min_shift]

                ref_theta = angles[0]

                topo = 0
                if n_ang == 2:
                    gap = min(gaps[0], 2 * math.pi - gaps[0])
                    if abs(gap - math.pi) < P.JUNCTION_COLLINEAR_TOL:
                        topo = 0
                    else:
                        topo = 1
                elif n_ang == 3:
                    tol = P.JUNCTION_COLLINEAR_TOL
                    if any(abs(g - math.pi) < tol for g in gaps):
                        topo = 2
                    elif max(gaps) > math.pi + tol:
                        topo = 6
                    else:
                        topo = 3
                elif n_ang == 4:
                    diff_0_2 = (angles[2] - angles[0]) % (2 * math.pi)
                    diff_1_3 = (angles[3] - angles[1]) % (2 * math.pi)
                    pair_0_opposite = abs(diff_0_2 - math.pi) < P.JUNCTION_COLLINEAR_TOL
                    pair_1_opposite = abs(diff_1_3 - math.pi) < P.JUNCTION_COLLINEAR_TOL
                    if pair_0_opposite and pair_1_opposite:
                        topo = 4
                    else:
                        topo = 0
                else:
                    expected_gap = (2 * math.pi) / n_ang
                    tol = max(min(0.25 * expected_gap, P.JUNCTION_STAR_TOL), math.pi / 36.0)
                    if all(abs(g - expected_gap) < tol for g in gaps):
                        topo = 5
                    else:
                        topo = 0

                if _is_pass_through(topo, n_ang):
                    continue
                # [A4#2] a pass-through must not suppress a real junction, so
                # the radius test runs only AFTER it is ruled out.
                if any(_pos_dist(hub_pos, q) <= junction_radius
                       for q in accepted_hubs):
                    continue
                accepted_hubs.append((float(hub_pos[0]), float(hub_pos[1])))

                is_symmetric = (topo in (4, 5)) or (topo == 0 and n_ang == 2)
                if is_symmetric or not ref_is_unambiguous:
                    ref_theta_bin = 0.0
                    emit_ref_arm = False
                else:
                    ref_theta_bin = float(quantise(
                        _JUNCTION_SPEC["ref_theta"], float(ref_theta)))
                    emit_ref_arm = True

                expected_gap = (2 * math.pi) / n_ang
                spread = sum((g - expected_gap) ** 2 for g in gaps) / n_ang
                var_max = (2 * math.pi) ** 2 * (1.0 - 1.0 / n_ang) / n_ang
                spread_norm = spread / var_max if var_max > 1e-9 else 0.0
                spread_bin = float(quantise(
                    _JUNCTION_SPEC["spread"], float(spread_norm)))
                n_branches_bin = float(quantise(
                    _JUNCTION_SPEC["degree"], float(n_ang)))

                gap_bins = [
                    float(quantise(_JUNCTION_SPEC[f"gap{i}"],
                                   float(gaps[i] if i < n_ang else 0.0)))
                    for i in range(4)
                ]

                stats = [
                    _arm_tangent_stats(j, hub_idx, G_j, orient_dets, CONTINUITY_NBR)
                    for j in arm_idx
                ]
                support = sum(n_ for n_, _ in stats) / max(1, len(stats))
                measured = [r for n_, r in stats if n_ > 1]
                sharp_mean = (sum(measured) / len(measured)) if measured else 0.0
                sharp_min = min(measured) if measured else 0.0

                pair_hist = [0] * 8
                for ia in range(n_ang):
                    for ib in range(ia + 1, n_ang):
                        d = abs(angles[ia] - angles[ib]) % (2 * math.pi)
                        d = min(d, 2 * math.pi - d)
                        pair_hist[min(7, int(d / math.pi * 8))] += 1
                pair_hist_f = [
                    float(quantise(_JUNCTION_SPEC[f"arm_angle{i}"], float(c)))
                    for i, c in enumerate(pair_hist)
                ]

                extra = (
                    float(quantise(_JUNCTION_SPEC["arm_support"], float(support))),
                    float(quantise(_JUNCTION_SPEC["arm_sharp_mean"],
                                   float(sharp_mean))),
                    float(quantise(_JUNCTION_SPEC["arm_sharp_min"],
                                   float(sharp_min))),
                ) + tuple(pair_hist_f)

                quantized = (
                    n_branches_bin,
                    ref_theta_bin,
                    float(quantise(_JUNCTION_SPEC["topo"], float(topo))),
                    spread_bin,
                    gap_bins[0],
                    gap_bins[1],
                    gap_bins[2],
                    gap_bins[3],
                ) + extra
                # [GRADED POPULATION] The CONTINUOUS junction measurements, in
                # the order the JUNCTION unit table declares
                # (core.registries.POPULATION_SLOTS["JUNCTION"]): the arm count,
                # the reference orientation (radians), the topology class, the
                # NORMALISED gap spread, the four arm gaps (RADIANS), the arm
                # support (the mean number of oriented segments per arm), the two
                # arm-sharpness statistics (a doubled-angle resultant in [0, 1])
                # and the eight pairwise arm-angle counts.
                # Every value is in the unit the table declares for its slot, so
                # the graded bump and the bin above name the same measurement.
                _jn_pop = population_from_values(
                    "JUNCTION",
                    (float(n_ang),
                     float(ref_theta) if emit_ref_arm else 0.0,
                     float(topo), float(spread_norm),
                     float(gaps[0] if n_ang > 0 else 0.0),
                     float(gaps[1] if n_ang > 1 else 0.0),
                     float(gaps[2] if n_ang > 2 else 0.0),
                     float(gaps[3] if n_ang > 3 else 0.0),
                     float(support),
                     float(sharp_mean), float(sharp_min),
                     ) + tuple(float(c) for c in pair_hist))

                # [U20 LEAD 3] The FAMILY is the node kind.  The signature is
                # untouched (class JUNCTION, same quantized tuple and population),
                # so identity -- the signature hash -- cannot move: the family is
                # carried by `k`, not by the signature.
                node_kind = (junction_family(topo, n_ang)
                             if getattr(P, "JUNCTION_FAMILY_SPLIT", True)
                             else JUNCTION_RESIDUAL_KIND)

                hub_anchor = orient_dets[hub_idx].alpha
                rho = [("entity", hub_anchor, 0, "+")]
                if emit_ref_arm:
                    rho.append(("ref_arm", member_anchors[0], 0, "+"))
                for i, a in enumerate(member_anchors):
                    rho.append(("member", a, i, "+"))
                for i, j in enumerate(arm_idx):
                    if j in det_to_nid:
                        rho.append(("member_node", (det_to_nid[j],), i, "+"))

                emitted.append(Detection(
                    k=node_kind,
                    source_layer=1,
                    scale=0,
                    sigma=TypedSignature(
                        kind="CANONICAL",
                        class_="JUNCTION",
                        relation="ANGLE_RELATION",
                        quantized=quantized,
                        # [GRADED POPULATION] The CONTINUOUS junction
                        # measurements the population was built from (the arm
                        # count, the reference orientation, the topology class,
                        # the normalised gap spread, the four arm gaps, the arm
                        # support, the two arm-sharpness statistics and the
                        # eight pairwise arm-angle counts).
                        population=_jn_pop,
                    ),
                    alpha=hub_anchor,
                    z=min(1.0, max(0.0, n_ang / max(float(P.JUNCTION_SATURATION_ARMS), 1e-9))),
                    t=t,
                rho=tuple(rho),
                ))

        self._emitted_total += len(emitted)
        return emitted


class TextureBoundaryLens(HigherLens):
    """B20 — texture-defined boundary as a NODE field (arity 8).
    One detection per sampled boundary vertex whose two flanks differ in
    (theta, sf) energy distribution.
    """

    # [A4#11] class-level defaults: the step counters are read by
    # get_diagnostics and must exist before the first step runs.
    _emitted_total = 0
    _candidates_evaluated = 0
    _skipped_bad_input = 0
    _field_errors = 0
    _skipped_nonfinite = 0
    _skipped_luminance_explained = 0

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._emitted_total = 0
        self._candidates_evaluated = 0
        self._skipped_bad_input = 0
        self._field_errors = 0
        self._skipped_nonfinite = 0
        self._skipped_luminance_explained = 0

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
        observation: Optional[Any] = None,
        **kwargs: Any,
    ) -> List[Detection]:
        # [W1] Two guard paths, because the two arms need different things.
        #
        # Legacy arm: needs the gabor bank plus a per-anchor texture-profile
        # sampler. `sample_texture_profile_at` does NOT exist on
        # Observation2D at all, so this arm is dead by construction -- not by
        # a flag, and not by the entity path. That is a measured fact about
        # the class, not a tunable.
        #
        # Field arm: measures material discontinuity ACROSS a chain from the
        # front end's own luminance/opponent planes (perception.chain_flanks),
        # so it needs only the rgb plane and the chains.
        # [REPRESENTATION MISSION] The legacy arm is RETIRED, and with it
        # its body.
        #
        # It needed `observation.sample_texture_profile_at` -- an API no
        # observation implements -- so it was dead by construction, as this
        # class's own comment recorded ("dead by construction -- not by a
        # flag, and not by the entity path"). The field arm measures the same
        # material discontinuity on the front end's own planes and is the arm
        # that fires; with the field entity source on (no segmenter) it is the
        # only arm. Logged in diagnostics/_retirement_log.jsonl.
        return self._step_field(
            t, anchor_positions, current_detections, extraction_scale,
            observation, luminance=kwargs.get("luminance"))

    def _step_field(
        self,
        t: int,
        anchor_positions: Optional[Dict[Any, Tuple[float, ...]]],
        current_detections: Optional[Dict[str, List[Detection]]],
        extraction_scale: float,
        observation: Any,
        luminance: Any = None,
    ) -> List[Detection]:
        """[W1] Field-arm texture boundary: material discontinuity ACROSS a
        chain, measured on the front end's own planes.

        One detection per chain vertex where the two flanks differ, with the
        same (arc_pos, dtheta, dlogsf, lum_dist, conc_a, conc_b) slot layout
        as the legacy arm so the two are comparable slot-for-slot.

        The legacy arm's (theta, sf) profile comparison is unavailable here
        because the observation built on the region path carries no gabor
        bank and no `sample_texture_profile_at`. Substituting luminance and
        opponent contrast for the (theta, sf) energy distribution is the
        honest available measure -- it is the material difference across the
        contour, which is what "texture-defined boundary" means -- and it is
        labelled as such rather than pretending to be the gabor profile.
        """
        if anchor_positions is None:
            return []
        if observation is None:
            return []
        rgb = luminance if luminance is not None else getattr(
            observation, "_raw_image", None)
        if rgb is None:
            return []
        rgb = np.asarray(rgb)
        # [A4#9] say WHY, and require an image: a caller passing a 2-D plane
        # named `luminance` silently emitted nothing, and a 2-band plane read
        # shape[2] as 2.
        if rgb.ndim != 3 or rgb.shape[2] not in (3, 4):
            self._skipped_bad_input += 1
            return []
        from perception.contour_field import field_from_luminance
        from perception.chain_flanks import planes_from_rgb
        channels = planes_from_rgb(rgb)
        if channels is None:
            return []
        if "lum" not in channels:
            return []
        lum_plane = np.asarray(channels["lum"], dtype=np.float32)
        if lum_plane.ndim != 2 or lum_plane.size == 0:
            return []
        if not np.all(np.isfinite(lum_plane)):
            return []
        # [A4#9] counted, not swallowed: a broken front end used to look like
        # "no texture boundaries", and an empty sfs gave a NaN median silently.
        try:
            f = field_from_luminance(lum_plane)
        except (ValueError, TypeError):
            self._field_errors += 1
            return []
        sfs = np.asarray(f["sfs"], dtype=np.float64)
        if sfs.size == 0 or not np.all(np.isfinite(sfs)):
            self._field_errors += 1
            return []
        med_wl = float(np.median(1.0 / np.maximum(sfs, 1e-6)))
        if not (math.isfinite(med_wl) and med_wl > 0.0):
            self._field_errors += 1
            return []
        from perception.chain_flanks import chain_flank_fields
        chain_dets = list((current_detections or {}).get("PEAK_CHAIN", []))
        if not chain_dets:
            return []

        emitted: List[Detection] = []
        for det in chain_dets:
            anchor = det.alpha
            fl = chain_flank_fields(det, anchor_positions, med_wl, channels,
                                    image_shape=(int(rgb.shape[0]),
                                                 int(rgb.shape[1])))
            if fl is None:
                continue
            n_mem = fl['n_members']
            for m in fl['members']:
                lum_l, o0_l, o1_l = m['left']
                lum_r, o0_r, o1_r = m['right']
                # [A4#9] non-finite flanks would reach atan2/quantise.
                if not all(math.isfinite(float(v)) for v in
                           (lum_l, o0_l, o1_l, lum_r, o0_r, o1_r)):
                    self._skipped_nonfinite += 1
                    continue
                vidx = int(m['index'])
                # [A4#8] pick the flanks by LUMINANCE, not by `lum_diff`'s
                # assumed sign: _dlogsf is a brighter/darker ratio and must not
                # depend on which side the member list calls left.
                if lum_l >= lum_r:
                    (la, oa0, oa1), (lb, ob0, ob1) = m['left'], m['right']
                else:
                    (la, oa0, oa1), (lb, ob0, ob1) = m['right'], m['left']
                lum_dist = abs(la - lb)
                conc_a = min(1.0, math.hypot(oa0, oa1))
                conc_b = min(1.0, math.hypot(ob0, ob1))
                self._candidates_evaluated += 1
                dtheta = math.atan2(o1_l - o1_r, o0_l - o0_r) % math.pi
                dlogsf = _dlogsf(la, lb)
                # The two slots the SIX-slot version was missing: the
                # opponent-contrast DROP across the boundary and the
                # opponent-VECTOR difference. The class registers EIGHT slots
                # (CONSTITUTIONAL_RANGES["TEXTURE_BOUNDARY"]), so the arm
                # that emits six violates the class contract.
                d_contrast = abs(math.hypot(o0_l, o1_l)
                                 - math.hypot(o0_r, o1_r))
                d_ori = math.sqrt(0.5 * ((o0_l - o0_r) ** 2
                                         + (o1_l - o1_r) ** 2))
                frac = (vidx + 0.5) / max(1, n_mem)
                # [UNIT TABLE] The bins are written through the class' unit
                # table (core.registries.POPULATION_SLOTS["TEXTURE_BOUNDARY"]),
                # so the lookup index and the graded population are
                # quantisations of the SAME measurement in the SAME unit.
                _tb_spec = _TEXTURE_BOUNDARY_SPEC
                # [GRADED POPULATION] The CONTINUOUS material-discontinuity
                # measurements behind the eight slots, in the unit the table
                # declares for each of them.
                _tb_pop = population_from_values(
                    "TEXTURE_BOUNDARY",
                    (frac, dtheta, dlogsf, d_contrast, d_ori, lum_dist,
                     conc_a, conc_b))
                # [CONFIDENCE] z is a CONFIDENCE in [0, 1]. The old reading
                # multiplied the LUMINANCE-CONTRAST BIN INDEX (0..7) by the
                # luminance-distance attenuation, so z reached 7.0 -- not a
                # confidence at all. It is now the CONTINUOUS mean flank
                # luminance contrast (the same measurement the bin names,
                # before it was quantised) attenuated by the same factor:
                # a texture boundary is a material discontinuity that the
                # luminance alone does not explain.
                material = min(1.0, max(d_contrast, d_ori))
                z_conf = _texture_confidence(material, lum_dist)
                # [A4#15, REVERTED after measurement] The cull dropped the
                # WHOLE tier: every cell logged "TextureBoundaryLens emitted 0
                # detections despite candidates_evaluated=N", because
                # `_texture_confidence` returns 0 once the absolute flank
                # luminance difference reaches 0.5 (material * (1 - lum/0.5)),
                # which every lit-surface boundary does. A z=0 boundary is a
                # LOW-CONFIDENCE material edge, not a proven luminance edge, so
                # it emits (as before) and carries its 0 in `z`, where the
                # consumer can weight it. The counter stays for diagnostics and
                # records the boundaries that would have been culled.
                if z_conf <= 0.0:
                    self._skipped_luminance_explained += 1
                emitted.append(Detection(
                    k="TextureBoundary",
                    source_layer=1,
                    scale=0,
                    sigma=TypedSignature(
                        kind="CANONICAL", class_="TEXTURE_BOUNDARY",
                        relation="MATERIAL_DISCONTINUITY",
                        quantized=(
                            float(quantise(_tb_spec["arc_pos"], frac)),
                            float(quantise(_tb_spec["dtheta"], dtheta)),
                            float(quantise(_tb_spec["dlogsf"], dlogsf)),
                            float(quantise(_tb_spec["d_contrast"], d_contrast)),
                            float(quantise(_tb_spec["d_ori"], d_ori)),
                            float(quantise(_tb_spec["lum_dist"], lum_dist)),
                            float(quantise(_tb_spec["conc_a"], conc_a)),
                            float(quantise(_tb_spec["conc_b"], conc_b)),
                        ),
                        population=_tb_pop,
                    ),
                    alpha=anchor,
                    z=float(z_conf),
                    t=t,
                    rho=(("entity", anchor, 0, "+"),
                         ("vertex", vidx, 0, "+")),
                ))
        self._emitted_total += len(emitted)
        return emitted

    def get_diagnostics(self) -> Dict[str, int]:
        return {
            "candidates_evaluated": self._candidates_evaluated,
            "emitted_total": self._emitted_total,
            "skipped_bad_input": self._skipped_bad_input,
            "field_errors": self._field_errors,
            "skipped_nonfinite": self._skipped_nonfinite,
            "skipped_luminance_explained": self._skipped_luminance_explained,
        }

