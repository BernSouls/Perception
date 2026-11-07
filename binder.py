"""binder.py — Canonical binder module for Blueprint v31.

Implements §2.3: ten binders producing relations between two or more
distinct entities. Every binder compares lens-produced pattern nodes and emits
an order-1 edge. There are NO order-2 edge types: structural similarity is a
measurement of the graph the binders already produced, not an observed
relation, so it lives in structural_similarity.py (§1.5/§4.12) and emits
nothing.

Order-1:
    Co-activation   A and B fire in the same step          (undirected)
    Transition      A onsets at t, B onsets at t+1          (directed)
    Co-motion       A and B move together                   (undirected)
    Contrast        B appears iff A is present              (directed)
    Periodicity     A recurs with interval T relative to B  (undirected)
    Persistence     A stable while B changes                (directed)
    Topology        B is one of A's K nearest neighbors     (directed)
    Degree          B is within radius R of A               (undirected)
    Contact         A contacts B                            (undirected)
    Sequence        A precedes B with age decay             (directed)

Determinism
-----------
All iteration, sorting, and tie-breaking is by canonical bytes.
Per-channel homeostatic state mirrors §2.7 (theta, r_bar, C, F, M_prev).
Same input produces byte-identical output, run to run.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from enum import Enum
from heapq import nlargest
from typing import TYPE_CHECKING, Any, Dict, FrozenSet, List, Optional, Set, Tuple

from core.hashing import (  # noqa: F401
    clamp,
    hash_128,
    hash_64,
    serialize_bytes,
    serialize_int,
    serialize_tuple,
    serialize_value,
)
from core.schema import EdgeParams  # noqa: F401
from core.schema import edge_params as _edge_params  # noqa: F401
from core.slots import (  # noqa: F401
    orientation_theta_radians,
    slot_value,
    slot_values,
)
from core.spatial import SpatialHash  # noqa: F401
import parameters as P

from sensory_floor import Detection, TypedSignature
from lenses import _LENS_NAMES
from perception.depth import (
    _extract_cc_node_ids,
    _extract_content_node_ids,
    _extract_member_anchors,
    _extract_member_node_ids,
    _mean_position,
    _pos_dist,
    resolve_depth_constraints,
    depth_layout,
)

# [U20 LEAD 3] The junction-tier predicate: a Junction, a Corner, a T and a Y
# are all junctions. Readers use this instead of `d.k == "Junction"` so the
# family split cannot silently drop junctions from an edge channel.
from perception.lenses.derived_common import is_junction_kind  # noqa: E402

if TYPE_CHECKING:
    from concept_graph import ConceptFamily  # annotation-only (runtime import is inside _map_family to avoid the module cycle)

# §2.2 canonical lens vocabulary, sourced from lenses.py so this module cannot
# drift from the registry it classifies against.
SPATIAL_LENS_TYPES: FrozenSet[str] = frozenset(
    n for n in _LENS_NAMES if n != "EffectPresent"
)

# v32: informative lens types that carry per-patch content. Used by
# CO_LOCATED to restrict same-anchor cross-lens edges to meaningful
# pairs.
#
# [T1.2R] Two corrections:
#   * "Elongation" REMOVED -- phantom. No lens emits it (_LENS_NAMES
#     has no such entry; no _emit("Elongation", ...) call exists) and
#     content.py already documents it as vestigial.
#   * "Boundary" ADDED -- reverses the old exclusion note. Boundary is
#     the enclosed-region detector; its co-occurrence with other lenses
#     IS content (figure-ground), which is what this edge is for.
#     Position stays excluded: pure spatial scaffold.
# The bare addition inflated the candidate list ~2.5x (6952 -> 17468),
# which made _process_channel's 2000-per-step cap bind and reorder
# selection by signal. The density counterweight in _run_co_located
# (signal / max(1, k - 1) for k informative detections at the anchor)
# is what keeps that from being a pure inflation.
INFORMATIVE_LENS_TYPES: FrozenSet[str] = frozenset({
    # "OrientationPol" [option (b)]: the polarized companion of
    # Orientation. Listed so CO_LOCATED pairs it with the content
    # lenses wherever such detections exist (never on the region path).
    "Color", "Texture", "Size", "Orientation", "Boundary",
    "OrientationPol", "RegionShape",
})


# =============================================================================
# BACKWARD-COMPATIBILITY RE-EXPORTS (§1.1)
# =============================================================================


def __getattr__(name: str) -> Any:
    """Lazy re-export of `CrossSection` for legacy import paths.

    `from binder import CrossSection` must work, but `concept_graph` imports
    this module at import time (concept_graph.py line 70). A module-level
    `from concept_graph import CrossSection` here would therefore close a
    cycle: loading concept_graph executes its binder import before
    CrossSection is defined at concept_graph line 199, and the re-export
    would raise ImportError. Resolving lazily (PEP 562) breaks the cycle:
    the attribute is only looked up after both modules are fully loaded.

    Note: CrossSection is DEFINED in concept_graph; this is a re-export
    only, so there is exactly one class object and identity comparisons
    across modules hold.
    """
    if name == "CrossSection":
        from concept_graph import CrossSection

        return CrossSection
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


class ConceptGraphRef:
    """[V30 BACKWARD-COMPATIBILITY SHIM] Allows legacy test suites
    to import ConceptGraphRef from binder without collection errors."""

    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


EPSILON = P.EPSILON

# [SPEED] Upper bound on the number of Orientation anchors the
# ORIENTATION_BEND / ORIENTATION_JUNCTION relation scan will consider.
# The scan is a double loop over them, so the cost is quadratic; without a
# bound a frame that emits many Orientation anchors would pay an unbounded
# price for a relation that is a LOCAL (bend / junction) one.
#
# This is a LOCAL constant of this module with a documented default (it is
# NOT a parameter, and no parameter was changed). 4096 anchors is ~8.4M pair
# tests, which is the most the loop may ever cost.
#
# THE CAP IS NOT REACHED ON THE CORPUS. The measured maximum number of
# Orientation anchors on the ETH-80 corpus at canon 256 is reported by
# diagnostics/_speed_anchor_cap.py and is far below 4096, so every pair is
# still examined and the emitted edges are IDENTICAL to the uncapped
# version: the cap bounds a cost that the corpus never reaches, so the
# values are identical IN PRACTICE.
_ORIENTATION_ANGLE_RELATION_MAX_ANCHORS = 4096


# =============================================================================
# EDGE TYPE (§2.8)
# =============================================================================


# [reorg Move 1] EdgeType and the edge-type registries now live in
# core/types.py (single definition, imported here). The original block is
# preserved in binder.py.pre_core_move.
from core.types import (  # noqa: F401
    EdgeType,
    HEBBIAN_BOOST_TYPES,
    REPRODUCIBLE_EDGE_TYPES,
    UNDIRECTED_EDGE_TYPES,
)
from core.registries import (  # noqa: F401
    SELF_BINDER_TYPES,
)
from core.hashing import (  # noqa: F401
    _detection_sort_key,
    _node_signature_hash,
)


# Undirected edge types get canonicalized as (min_id, max_id). Directed
# types preserve their source→target ordering.
# [reorg Move 1] EdgeType and the edge-type registries now live in
# core/types.py (single definition, imported here). The original block is
# preserved in binder.py.pre_core_move.
from core.types import (  # noqa: F401
    EdgeType,
    HEBBIAN_BOOST_TYPES,
    REPRODUCIBLE_EDGE_TYPES,
    UNDIRECTED_EDGE_TYPES,
)
# [reorg Move 1] EdgeType and the edge-type registries now live in
# core/types.py (single definition, imported here). The original block is
# preserved in binder.py.pre_core_move.
from core.types import (  # noqa: F401
    EdgeType,
    HEBBIAN_BOOST_TYPES,
    REPRODUCIBLE_EDGE_TYPES,
    UNDIRECTED_EDGE_TYPES,
)
# [reorg Move 1] EdgeType and the edge-type registries now live in
# core/types.py (single definition, imported here). The original block is
# preserved in binder.py.pre_core_move.
from core.types import (  # noqa: F401
    EdgeType,
    HEBBIAN_BOOST_TYPES,
    REPRODUCIBLE_EDGE_TYPES,
    UNDIRECTED_EDGE_TYPES,
)


# =============================================================================
# FAMILY RESOLUTION (§4.14, adapted from v30 §2.8.2)
# =============================================================================


# Lens types excluded from Rule 4's substrate comparison. Two
# simultaneously-present effects are two facts about the world, not
# a cross-substrate pattern.
NON_SUBSTRATE_LENS_TYPES: FrozenSet[str] = frozenset({"EffectPresent"})


def _map_family(
    original_detections: FrozenSet[Detection],
    edges: FrozenSet[Tuple[int, int, EdgeType]] = frozenset(),
    diag: Optional[Dict[str, Any]] = None,
) -> Optional["ConceptFamily"]:
    """Map a candidate's accumulated detections and edges to a family.

    v31 rule set (v30 §2.8.2 with §11.1/§11.2 removals, plus the flat
    taxonomy's edge-based causal rule):

      Rule 1 — EVENT_CAUSAL / TEMPORAL_CHAIN:
        A SEQUENCE edge (temporal priority: A precedes B) indicates a causal
        family. sourceLayer decides which:
          max(source_layer) == 0 → EVENT_CAUSAL
          max(source_layer) >= 1 → TEMPORAL_CHAIN

      Rule 2 — SPATIAL_OBJECT:
        Two or more spatial lens types in the detection set.

      Rule 3 — RECURSIVE_COMPOSITION:
        A higher-layer detection (max(source_layer) >= 1) combined with a
        heterogeneous edge set (>= 2 distinct relation channels). A
        composition of compositions is recognized by its *mixed* relations
        rather than by an order-2 edge: the STRUCTURAL_SIMILARITY edge type
        was removed (§1.5/§4.12 — comparison is a measurement, not a
        relation), so the rule now reads the relation structure directly.
        Subgraphs already claimed by Rule 1 are excluded, so a causal chain
        is never re-classified as a recursive composition.

      Rule 4 — CROSS_DOMAIN_PATTERN:
        Same lens type, two or more distinct substrates.

    [v31 FLAT-TAXONOMY NOTE] Under v31 the six concept-derived lenses
    (ConceptSequence, ConceptStructure, Convergence, Divergence,
    ConceptDegree, ConceptRhythm) have been removed. Causality is now
    encoded as edge structure and read here. The v30 detection-only rule
    fired on raw "Sequence" + "Structure" detections; those no longer
    exist (they became the SEQUENCE binder), so the rule now fires on the corresponding edges.

    Returns None if no rule matches; the candidate is a scaffold and MUST
    NOT be emitted as a CrossSection.
    """
    from concept_graph import ConceptFamily

    if not original_detections and not edges:
        if diag is not None:
            diag["map_family_empty_input"] = (
                diag.get("map_family_empty_input", 0) + 1
            )
        return None

    lens_types: Set[str] = {d.k for d in original_detections}
    source_layers: Set[int] = {d.source_layer for d in original_detections}
    edge_types: Set[EdgeType] = {e[2] for e in edges}

    # --- Rule 1: Causal families (edge-based in v31, re-based on SEQUENCE) ---
    has_sequence_edge = EdgeType.SEQUENCE in edge_types
    # [defect fix] This name was read at Rule 3 but never assigned, so the
    # guard raised NameError the moment it was reached. Rule 1 returns on any
    # SEQUENCE edge, so at Rule 3 the pair can never both hold and the guard
    # is vacuously true for any predicate -- TOPOLOGY is the closest current
    # analogue to the historical "structure" signal, but the intended
    # predicate is UNVERIFIED. Behaviour is identical either way today.
    has_struct_delta_edge = EdgeType.TOPOLOGY in edge_types
    if has_sequence_edge:
        if diag is not None:
            diag["map_family_rule1_fired"] = (
                diag.get("map_family_rule1_fired", 0) + 1
            )
        m = max(source_layers) if source_layers else 0
        if m == 0:
            return ConceptFamily.EVENT_CAUSAL
        return ConceptFamily.TEMPORAL_CHAIN

    # --- Rule 2: SPATIAL_OBJECT ------------------------------------------
    # §2.8.2: "at least two spatial lenses". The lens vocabulary is owned by
    # lenses.py (_LENS_NAMES, the §2.2 canonical seven); reading it from there
    # rather than restating the names means adding or renaming a lens cannot
    # silently stale this rule. Under v31's flat taxonomy every lens is a
    # single-entity property extractor (§2.1), so lens-type co-occurrence is
    # the spatial-object signal; relational structure is carried by edges and
    # resolved by Rules 1 and 3 above.
    if len(lens_types & SPATIAL_LENS_TYPES) >= 2:
        if diag is not None:
            diag["map_family_rule2_fired"] = (
                diag.get("map_family_rule2_fired", 0) + 1
            )
        return ConceptFamily.SPATIAL_OBJECT

    # --- Rule 3: RECURSIVE_COMPOSITION -----------------------------------
    # A composition of compositions: a higher-layer source (the subgraph
    # contains a concept-derived detection, i.e. evidence that it is built
    # out of already-composed structure) plus heterogeneous relation
    # channels. Rule 1's causal pair is excluded so the two families stay
    # disjoint and deterministic.
    if (
        max(source_layers, default=0) >= 1
        and len(edge_types) >= 2
        and not (has_sequence_edge and has_struct_delta_edge)
    ):
        if diag is not None:
            diag["map_family_rule3_fired"] = (
                diag.get("map_family_rule3_fired", 0) + 1
            )
        return ConceptFamily.RECURSIVE_COMPOSITION

    # --- Rule 4: CROSS_DOMAIN_PATTERN ------------------------------------
    by_lens: Dict[str, List[Detection]] = defaultdict(list)
    for d in original_detections:
        by_lens[d.k].append(d)
    for _k, dets in by_lens.items():
        if _k in NON_SUBSTRATE_LENS_TYPES:
            continue
        if len(dets) < 2:
            continue
        substrates: Set[Tuple[bytes, int]] = set()
        for d in dets:
            # serialize_value is total over the closed anchor taxonomy
            # (int | 2-tuple | bytes, per ConceptGraph._anchor_kind); the
            # per-step _detection_sort_key already relies on this. The old
            # `except: repr(d.alpha).encode()` fallback was both unreachable
            # for valid anchors and a §9 determinism hazard (repr can embed a
            # memory address), so it is removed rather than masked.
            alpha_key = serialize_value(d.alpha)
            substrates.add((alpha_key, d.substrate_hash))
        if len(substrates) >= 2:
            if diag is not None:
                diag["map_family_rule4_fired"] = (
                    diag.get("map_family_rule4_fired", 0) + 1
                )
            return ConceptFamily.CROSS_DOMAIN_PATTERN

    if diag is not None:
        diag["map_family_returned_none"] = (
            diag.get("map_family_returned_none", 0) + 1
        )
        key = "_".join(str(int(x)) for x in (
            has_sequence_edge,
            len(lens_types & SPATIAL_LENS_TYPES) >= 2,
            max(source_layers, default=0) >= 1,
            len(edge_types) >= 2,
        ))
        bucket = f"map_family_none_bucket_{key}"
        diag[bucket] = diag.get(bucket, 0) + 1
    return None


# =============================================================================
# CHANNEL STATE
# =============================================================================


def _init_channel_state() -> Dict[str, Any]:
    """Per-channel homeostatic state, structurally identical to the lens
    state machine (§2.7). `last_seen` supports deterministic LRU eviction
    of `M_prev` at capacity."""
    return {
        "theta": P.THETA_CHANNEL_BASE_INIT,
        "r_bar": 0.0,
        "C": 0.0,
        "F": 0.0,
        "M_prev": {},
        "last_seen": {},
    }


# =============================================================================
# BINDER
# =============================================================================



def _axial_wrap(x: float) -> float:
    """Wrap a difference of AXIAL (mod-pi) angles into [-pi/2, pi/2)."""
    return (x + math.pi / 2.0) % math.pi - math.pi / 2.0


def _bin8(x: float, lo: float, hi: float) -> float:
    """Equal-width floor bin of x over [lo, hi) -> 0..7."""
    if not (hi > lo) or not math.isfinite(x):
        return 0.0
    return float(min(7, max(0, int((x - lo) / (hi - lo) * 8.0))))


def _axial_mean(a: float, b: float) -> float:
    """Mean of two AXIAL (mod-pi) angles in [0, pi)."""
    return (a + 0.5 * _axial_wrap(b - a)) % math.pi


def _edge_population(class_name: str,
                     positions: Tuple[float, ...]
                     ) -> Tuple[Tuple[float, ...], ...]:
    """The graded population of an edge's relational parameters.

    `positions` are the CONTINUOUS positions of each slot, in the writer's
    own bin units -- the same scale the coarse bins live on, so the bin is
    the floor of the position and the activity vector keeps the sub-bin
    part the bin rounds off. The axes are resolved BY NAME through the
    edge population table (so a reordered table cannot move a measurement
    onto another quantity).

    Returns () when the class has no axes, the arity disagrees, or a
    position is not finite: the edge then falls back to the bins path
    (never a silent, mis-shaped population).
    """
    from core.schema import EDGE_POPULATION_AXES, EDGE_UNIT_TABLE
    from core.population import soft_bump_bins
    specs = EDGE_UNIT_TABLE.get(class_name)
    axes = EDGE_POPULATION_AXES.get(class_name)
    if not specs or axes is None or len(specs) != len(positions) or len(axes) != len(positions):
        return ()
    # Resolve each axis BY NAME (not by position).
    axes_by_name = {ax.name: ax for ax in axes}
    out = []
    for pos, sp in zip(positions, specs):
        ax = axes_by_name.get(sp.name)
        if ax is None:
            return ()
        p = float(pos)
        if not math.isfinite(p):
            return ()
        out.append(soft_bump_bins(p, ax))
    return tuple(out)


def _contour_point_geometry(det, anchor_positions):
    """Real geometry from member anchors -> (rms_radius, eccentricity, angular_extent) or None."""
    pts = []
    for a in _extract_member_anchors(det):
        p = anchor_positions.get(a) if anchor_positions else None
        if p is not None and len(p) >= 2 and math.isfinite(p[0]) and math.isfinite(p[1]):
            pts.append((float(p[0]), float(p[1])))
    n = len(pts)
    if n < 3:
        return None
    cx = sum(p[0] for p in pts) / n
    cy = sum(p[1] for p in pts) / n
    sxx = syy = sxy = 0.0
    angs = []
    for x, y in pts:
        dx, dy = x - cx, y - cy
        sxx += dx * dx; syy += dy * dy; sxy += dx * dy
        angs.append(math.atan2(dy, dx))
    sxx /= n; syy /= n; sxy /= n
    tr = sxx + syy
    disc = math.sqrt(max(tr * tr / 4.0 - (sxx * syy - sxy * sxy), 0.0))
    l1, l2 = tr / 2.0 + disc, max(tr / 2.0 - disc, 0.0)
    if l1 <= 1e-12:
        return None
    angs.sort()
    gaps = [angs[i + 1] - angs[i] for i in range(n - 1)]
    gaps.append(angs[0] + 2.0 * math.pi - angs[-1])
    return math.sqrt(tr), math.sqrt(1.0 - l2 / l1), 1.0 - max(gaps) / (2.0 * math.pi)


def _assoc_pair_geometry(theta_a, pos_a, theta_b, pos_b, sigma_theta,
                         sigma_p, tau):
    """Order-symmetric association-field geometry for two oriented elements.

    Returns None if coincident, else
    (signal, chord, mean_perp, port_a, port_b).
    port_x in {0, 1}: which END of element x faces the other element.

    Signal is the product of three Gaussian gates: angular agreement,
    perpendicular offset, and along-axis gap. Averaging the two
    perpendicular offsets makes the signal independent of which element is
    named 'a'.
    """
    dx = pos_b[0] - pos_a[0]
    dy = pos_b[1] - pos_a[1]
    chord = math.hypot(dx, dy)
    if chord < 1e-9:
        return None
    phi = math.atan2(dy, dx)
    al_a = _axial_wrap(theta_a - phi)
    al_b = _axial_wrap(theta_b - phi)
    d_theta = abs(_axial_wrap(theta_a - theta_b))
    perp_a = chord * abs(math.sin(al_a))
    perp_b = chord * abs(math.sin(al_b))
    along_a = chord * abs(math.cos(al_a))
    along_b = chord * abs(math.cos(al_b))
    signal = (
        math.exp(-((d_theta / sigma_theta) ** 2))
        * math.exp(-0.5 * (perp_a ** 2 + perp_b ** 2) / (sigma_p ** 2 + EPSILON))
        * math.exp(-0.5 * (along_a + along_b) / (tau + EPSILON))
    )
    port_a = 1 if (dx * math.cos(theta_a) + dy * math.sin(theta_a)) >= 0.0 else 0
    port_b = 1 if (-dx * math.cos(theta_b) - dy * math.sin(theta_b)) >= 0.0 else 0
    return signal, chord, 0.5 * (perp_a + perp_b), port_a, port_b


#: The signature classes an ORIENTATION-SPINE edge may connect.  The spine is
#: "oriented segments and the continuity between them", so every family that
#: IS an oriented segment belongs here.  ORIENTATION_GABOR was missing: on the
#: rich 2D path every plain orientation detection is GABOR (measured: 445 of
#: 445, diagnostics/_u21_id_mismatch.json), so the continuity channel was
#: built over the 55 polarised detections only -- 11% of the spine -- and
#: never over the geometry the front end actually extracted.
ORIENTATION_SPINE_CLASSES: Tuple[str, ...] = (
    "ORIENTATION",
    "ORIENTATION_POL",
    "ORIENTATION_GABOR",
)


def _orient_spine_classes() -> Tuple[str, ...]:
    """The classes the spine admits, honouring the [U21] rollback flag."""
    if getattr(P, "ORIENTATION_SPINE_WIRED", True):
        return ORIENTATION_SPINE_CLASSES
    return ("ORIENTATION", "ORIENTATION_POL")


def _orient_class(det) -> str:
    """[O2] The signature class of an orientation detection, for a named
    slot read.  The Orientation lens emits ORIENTATION, ORIENTATION_POL
    and ORIENTATION_SEGMENT; they all carry a THETA slot, but they are
    different classes."""
    cls = getattr(getattr(det, "sigma", None), "class_", "") or ""
    if cls in ("ORIENTATION", "ORIENTATION_POL", "ORIENTATION_SEGMENT"):
        return cls
    return "ORIENTATION"


def _pol_of(d) -> Optional[int]:
    """0=OFF, 1=balanced, 2=ON for ORIENTATION_POL detections.

    Returns None for plain ORIENTATION, which carries no polarity slot.
    """
    if d.sigma.class_ != "ORIENTATION_POL" or len(d.sigma.quantized) < 4:
        return None
    # [O2] The polarity slot, by name.
    p = int(round(slot_value(d.sigma, "ORIENTATION_POL", "polarity",
                             default=0.0)))
    return p if p in (0, 1, 2) else None


def _polarity_relation(pol_t, pol_h) -> float:
    """0 = flipped, 1 = no evidence, 2 = same.

    Polarity is a property of the REGION (ON-dominant vs OFF-dominant),
    not relative to the axis direction. A contour continuing across a
    background change routinely reverses contrast, so this is recorded as
    information and never used as a veto.
    """
    if pol_t is None or pol_h is None:
        return 1.0
    if pol_t == 1 or pol_h == 1:
        return 1.0
    return 2.0 if pol_t == pol_h else 0.0


def _detection_scale(d) -> float:
    """Scale field of a Detection; 0.0 when absent or invalid."""
    s = getattr(d, "scale", None)
    try:
        s = float(s)
    except (TypeError, ValueError):
        return 0.0
    return s if (s > 0.0 and math.isfinite(s)) else 0.0


# ==========================================================================
# [Pass 23] Angular and geometric helpers for V4 non-Cartesian +
# chromatic/motion/3D relational edges (values 42-51).
# NOTE: three helpers below look up CONTOUR / SURFACE_PATCH signature slot
# positions. These match the CURRENT lens layouts only (see inline comments);
# a lens change to slot positions must update these together.
# ==========================================================================


def _angular_diff(a: float, b: float) -> float:
    """Minimum angular difference in radians, result in [0, pi]."""
    d = abs(a - b) % (2.0 * math.pi)
    return min(d, 2.0 * math.pi - d)


def _angular_diff_pi(a: float, b: float) -> float:
    """Minimum angular difference for undirected orientations (mod pi),
    result in [0, pi/2]."""
    d = abs(a - b) % math.pi
    return min(d, math.pi - d)


def _safe_atan2(dy: float, dx: float) -> float:
    """atan2 that handles zero vectors gracefully."""
    if abs(dy) < 1e-12 and abs(dx) < 1e-12:
        return 0.0
    return math.atan2(dy, dx)


def _get_orientation_theta(det: Detection, n_theta: int = 24) -> Optional[float]:
    """Orientation angle in [0, pi) from a detection's signature.

    Slot 1 of the CONTOUR summary projection is the mean-orientation bin
    (n_theta bins over [0, pi)). Returns None when unavailable.
    Orientation-class detections are handled by their own emitters; only
    Contour carries a mean-orientation slot here. CurveComplex is a
    SHAPE_ARRANGEMENT bundle whose slot 1 is a contour count, not an angle,
    so it is excluded.
    """
    q = det.sigma.quantized
    if det.k == "Contour" and len(q) > 1:
        return (float(q[1]) + 0.5) * math.pi / n_theta
    return None


def _surface_normal(det: Detection) -> Optional[Tuple[float, float, float]]:
    """Surface normal from a Surface detection's boundary-gradient cue.

    SURFACE_PATCH slot 5 is the gradient-magnitude bin (0 = flat, normal
    +z); slot 6 is the gradient side (0/1, mapped to tilt 0/pi). This is a
    coarse two-tilt proxy, NOT a full slant/tilt field: treat the returned
    vector as a direction cue, not a measured normal.
    """
    q = det.sigma.quantized
    if det.k != "Surface" or len(q) < 7:
        return None
    slant = (clamp(float(q[5]), 0.0, 7.0) / 7.0) * (math.pi / 2.0)
    tilt = 0.0 if float(q[6]) < 0.5 else math.pi
    nx = math.sin(slant) * math.cos(tilt)
    ny = math.sin(slant) * math.sin(tilt)
    nz = math.cos(slant)
    return (nx, ny, nz)


def _velocity_direction(v: Tuple[float, ...]) -> float:
    """Motion direction in [0, 2pi) from a velocity vector."""
    if len(v) < 2:
        return 0.0
    return _safe_atan2(v[1], v[0]) % (2.0 * math.pi)


def _velocity_speed(v: Tuple[float, ...]) -> float:
    """Speed (magnitude) from a velocity vector."""
    n = min(len(v), 3)
    return math.sqrt(sum(v[i] ** 2 for i in range(n)))


def _role_anchor(det: Detection, role_name: str) -> Optional[Any]:
    """First anchor of `det`'s rho role `role_name`, or None.

    Deterministic: rho is read in its emitted order, so the first matching role
    wins (a fragment carries at most one partner/neighbor role).
    """
    for role in getattr(det, "rho", ()) or ():
        if len(role) >= 2 and role[0] == role_name:
            return role[1]
    return None


class Binder:
    """Observation engine: converts detection sets into graph edges.

    One instance per pipeline. Called once per step in step order. Order-2
    binders run after all order-1 binders within the same step (§3.1).
    """

    MAX_CANDIDATES_PER_CHANNEL = 2000
    MAX_M_PREV_SIZE = 50_000

    ORDER_1_TYPES: Tuple[EdgeType, ...] = (
        EdgeType.CO_ACTIVATION,
        EdgeType.TRANSITION,
        EdgeType.CO_MOTION,
        EdgeType.CONTRAST,
        EdgeType.PERIODICITY,
        EdgeType.PERSISTENCE,
        EdgeType.TOPOLOGY,
        EdgeType.DEGREE,
        EdgeType.CONTACT,
        EdgeType.SEQUENCE,
        EdgeType.SYNCHRONY,
        EdgeType.ORIENTATION_CONTINUITY,
        EdgeType.SURFACE_GROUPING,
        EdgeType.CURVATURE,
        EdgeType.OCCLUSION,
        EdgeType.CO_LOCATED,
        EdgeType.ORIENTATION_BEND,
        EdgeType.ORIENTATION_JUNCTION,
        # [PHASE B] B.11: all six higher-tier types run as standard
        # order-1 channels so emit_pending_edges can drive them.
        EdgeType.CONTOUR_CONTINUITY,
        EdgeType.JUNCTION_CONTOUR,
        EdgeType.CURVE_CONTAINMENT,
        EdgeType.PART_CONTAINMENT,
        EdgeType.REGION_CONTENT,
        EdgeType.CONTOUR_ORIENTATION,
        EdgeType.JUNCTION_ORIENTATION,
        # [Pass 12] Relational edge types E7–E11
        EdgeType.AMODAL_LINK,
        EdgeType.COCIRCULAR,
        EdgeType.PARALLEL,
        EdgeType.SYMMETRY,
        EdgeType.BORDER_OWNERSHIP_EDGE,
        EdgeType.CURVATURE_SIGN_CONTINUITY,
        EdgeType.SHARED_BOUNDARY,
        # [Pass 23] V4 non-Cartesian + chromatic/motion/3D relational edges
        EdgeType.CONCENTRICITY,
        EdgeType.TEXTURE_BOUNDARY_EDGE,
        EdgeType.SURFACE_FOLD,
        EdgeType.RADIALITY,
        EdgeType.SPIRALITY,
        EdgeType.COLOR_OPPONENT_BORDER,
        EdgeType.MOTION_OPPOSITION,
        EdgeType.CROSS_SCALE_IDENTITY,
        EdgeType.BINOCULAR_DISPARITY,
        EdgeType.LINE_TERMINATION,
    )

    def __init__(
        self,
        concept_graph: Any,
        bootstrap_relax_orthogonality: bool = False,
    ) -> None:
        self.cg = concept_graph
        self.bootstrap_relax_orthogonality = bootstrap_relax_orthogonality
        self.t: int = -1
        self._first_step: bool = True

        self.channels: Dict[EdgeType, Dict[str, Any]] = {
            et: _init_channel_state() for et in EdgeType
        }

        # Per-pattern presence (global, across anchors).
        # Pattern = (lens_type, signature_hash).
        self._pattern_present_prev: Set[Tuple[str, bytes]] = set()
        self._pattern_onset_now: Set[Tuple[str, bytes]] = set()
        self._pattern_onset_prev: Set[Tuple[str, bytes]] = set()
        self._pattern_last_onset: Dict[Tuple[str, bytes], int] = {}
        self._pattern_history: Dict[Tuple[str, bytes], List[int]] = defaultdict(list)
        self._pattern_last_seen: Dict[Tuple[str, bytes], int] = {}
        self._pattern_anchors: Dict[Tuple[str, bytes], Set[Any]] = defaultdict(set)

        # Per-anchor pattern-set change EMA (for Persistence).
        self._anchor_pattern_set_prev: Dict[Any, FrozenSet[Tuple[str, bytes]]] = {}
        self._anchor_change_ema: Dict[Any, float] = defaultdict(float)

        # [v8] Scene median nearest-neighbour distance, recomputed
        # once at the top of each Binder.step and read by spatial
        # binders. 0.0 when fewer than two positioned anchors exist.
        self._step_l_med: float = 0.0

        # [PHASE B] Step 4. Median V1 region radius in the observation's
        # own units (pixels / meters), supplied by the Pipeline. The
        # higher-tier emitters scale their spatial gates by this
        # (KAPPA_* * _step_extraction_scale) instead of using absolute
        # radii, which were never commensurate with anchor_positions.
        self._step_extraction_scale: float = 0.0

        # [Stage 3] Ordinal depth DAG conflict accounting. `_depth_conflicts_step`
        # is reset per occlusion step; `_depth_conflicts_total` accumulates across
        # the run. `_depth_removed_by_source` counts cycle-invalidated edges by
        # constraint source ("tjunction" / "border_ownership").
        self._depth_conflicts_step: int = 0
        self._depth_conflicts_total: int = 0
        self._depth_removed_by_source: Dict[str, int] = {}

        # [Stage 1.2] Two-stream registries. Populated per step from
        # Observation2D via Pipeline/probe; empty until the first step().
        self._structural_anchors: FrozenSet[Any] = frozenset()
        self._surface_anchors: FrozenSet[Any] = frozenset()
        self._contour_surface_contacts: List[Tuple[int, int, float]] = []

        # -------------------------------------------------------------------------
        # [PHASE B] per-step lifecycle accumulators (futurework3.md B.3)
        #
        # Since Step 2 the previous-step pattern sets are committed by
        # finalize_step(), NOT at the tail of step(). The pipeline brackets a
        # step with note_step_begin() / finalize_step() so that higher-lens
        # nodes registered mid-step (bind_intra_step) join the same commit.
        # Direct Binder.step() callers (temporal_bridge, tests) do not bracket,
        # so step() self-manages the lifecycle for them: see _lifecycle_owned.
        # -------------------------------------------------------------------------
        self._step_begin_called: bool = False
        self._lifecycle_owned: bool = False
        self._v1_present_this_step: Set[Tuple[str, bytes]] = set()
        self._v1_onset_this_step: Set[Tuple[str, bytes]] = set()
        self._higher_dets_this_step: List[Detection] = []
        self._higher_node_ids: Dict[Detection, int] = {}
        # [V9 Track 2 - B.26] The node -> anchor registry lives on the
        # ConceptGraph (`cg._pattern_node_anchor` / `_pattern_node_
        # position`), NOT here. Two reasons, both measured:
        #   * reachability -- only this module holds a Binder; the
        #     Subgraph construction sites live in Consolidator, Mapper,
        #     Builder, PipelineSupport, StructuralSimilarity and
        #     Prediction, all of which already hold a ConceptGraph. A
        #     Binder-owned map would have to be threaded through 21 call
        #     sites or read after the fact.
        #   * lifetime -- a node's entry is pruned with the node
        #     (_prune_pattern_nodes) and dies with the graph, so a reused
        #     node id cannot inherit a stale anchor.
        # It is CUMULATIVE by construction: cleared only with the graph.
        # A per-step reset was tried and measured 0 entries at end of run
        # -- i.e. empty exactly when the consumer reads it.
        # The two methods below are the Binder-side accessors.
        self._higher_present_this_step: Set[Tuple[str, bytes]] = set()
        self._higher_onset_this_step: Set[Tuple[str, bytes]] = set()
        self._step_anchor_positions: Optional[Dict[Any, Tuple[float, ...]]] = None

        # Diagnostics.
        #
        # [SDRSEQ PROBE] `empty_or_zero_signal_by_type` closes the
        # observability gap that made "SD/SEQ are zero" ambiguous. The
        # three existing counters only partition candidates that actually
        # REACHED `_process_channel`:
        #     candidates_by_type        (entered)
        #   - dropped_below_threshold   (failed the threshold test)
        #   = emitted_by_type           (written to the graph)
        # A channel whose upstream producer returned early, was never
        # invoked, or was blocked by a precondition therefore reports
        # 0 / 0 / 0 — indistinguishable from "emitted nothing because
        # every candidate was rejected". This counter increments on steps
        # where the channel had no candidate with positive signal at all,
        # so `candidates_by_type == 0 AND empty_or_zero_signal_by_type > 0`
        # positively identifies "nothing was ever produced" as opposed to
        # "produced and then dropped".
        self.diagnostics: Dict[str, Any] = {
            "candidates_by_type": defaultdict(int),
            "emitted_by_type": defaultdict(int),
            "candidate_cap_hits": defaultdict(int),
            "dropped_below_threshold": defaultdict(int),
            "empty_or_zero_signal_by_type": defaultdict(int),
            # [Q3] Anti-phase observability. The anti-phase channels once
            # dropped every pair silently when either endpoint had no
            # registered pattern node, which is indistinguishable from "no
            # anti-phase structure exists". These three counters make the
            # difference visible: offered (pairs the source produced),
            # unresolved (an endpoint had no node to bind), candidates (what
            # actually reached _process_channel).
            "anti_phase_pairs_offered": defaultdict(int),
            "anti_phase_unresolved_endpoints": defaultdict(int),
            "anti_phase_candidates": defaultdict(int),
            "co_activation_steady_pairs_skipped": 0,
        }

    # -------------------------------------------------------------------------
    # Public interface
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    # [PHASE B] Step lifecycle (futurework3.md B.3)
    # -------------------------------------------------------------------------

    def note_step_begin(self, t: int) -> None:
        """Open a step: clear the per-step accumulators.

        Called by the Pipeline BEFORE step(). Idempotence guard: calling this
        twice inside one step is a wiring bug (it would silently discard
        patterns already registered by a previous bind_intra_step), so it
        raises rather than resetting.
        """
        if self._step_begin_called:
            raise RuntimeError(
                "note_step_begin called twice in one step "
                "(previous step was never finalized)"
            )
        self._step_begin_called = True
        self._lifecycle_owned = False
        self._v1_present_this_step = set()
        self._v1_onset_this_step = set()
        self._higher_dets_this_step = []
        self._higher_node_ids = {}
        self._higher_present_this_step = set()
        self._higher_onset_this_step = set()
        self._step_anchor_positions = None

    def finalize_step(self) -> None:
        """Commit the per-step accumulators into the previous-step sets.

        Runs after the whole step (Pipeline `finally`), so V1 patterns and any
        higher-lens patterns registered by bind_intra_step land in ONE commit.
        Keeping this out of step()'s tail is what lets mid-step registration
        participate; that is the entire reason the lifecycle is bracketed.
        """
        assert self._step_begin_called, (
            "finalize_step called without a matching note_step_begin"
        )
        self._pattern_present_prev = (
            self._v1_present_this_step | self._higher_present_this_step
        )
        self._pattern_onset_prev = (
            self._v1_onset_this_step | self._higher_onset_this_step
        )
        self._first_step = False
        self._v1_present_this_step = set()
        self._v1_onset_this_step = set()
        self._higher_dets_this_step = []
        self._higher_node_ids = {}
        self._higher_present_this_step = set()
        self._higher_onset_this_step = set()
        self._step_anchor_positions = None
        self._step_begin_called = False

    def _remember_node_anchor(self, nid: int, anchor: Any) -> None:
        """[V9 Track 2 - B.26] Record which anchor a pattern node came from.

        FIRST write wins. A pattern node is keyed by (lens_type,
        signature_hash), so the same node can be reached from several
        anchors; which one is recorded must not depend on iteration order.
        Callers iterate deterministic structures, so the first write is
        deterministic. Overwriting would make the map depend on the last
        emitter to run.
        """
        self.cg.note_pattern_node_anchor(nid, anchor)

    def node_anchor_map(self) -> Dict[int, Any]:
        """[V9 Track 2 - B.26] node id -> anchor, for this pipeline's nodes.

        Returns a copy so callers cannot mutate the registry.
        """
        return self.cg.pattern_node_anchor_map()

    def _remember_node_position(self, nid: int, anchor: Any) -> None:
        """[V9 Track 2 - B.26] Record the (x, y) of `nid`'s anchor for the
        CURRENT step.

        Safe by construction, not by caller discipline:
        `self._step_anchor_positions` holds the current step's map only
        between the assignment in step()'s tail and finalize_step(), and
        is None outside that window. A misordered caller therefore sees
        either the current map or nothing -- never the previous step's.
        First write wins downstream (ConceptGraph), so a node keeps the
        position of the step in which it was first bound.
        """
        pos = self._step_anchor_positions
        if pos is None:
            return
        p = pos.get(anchor)
        if p is not None:
            self.cg.note_pattern_node_position(nid, p)

    # [V9 Track 2 - B.26] Positions are captured in a DELAYED PASS, not at
    # node-minting time. Both minting sites (:734 in bind_intra_step, :855
    # in step's V1 block) run before `self._step_anchor_positions` is
    # assigned (:878), so a capture at minting time would silently record
    # the PREVIOUS step's positions -- the trap that removed the first
    # draft's node_positions(). A method returning {} post-run would have
    # been worse than no method.
    #
    # Chosen plan (option (b) of the two reviewed): the step's V1 nodes are
    # captured in a pass at the END of step(), immediately after
    # `anchor_positions` is bound and while `node_ids` is still in scope.
    # Nodes minted AFTER step() -- bind_intra_step and note_executed_action
    # -- capture inline, because for them `self._step_anchor_positions`
    # already holds the CURRENT step's map (finalize_step clears it later).


    def bind_intra_step(self, t: int, dets: FrozenSet[Detection]) -> None:
        """Register higher-lens nodes and onsets. Extends pattern history.

        Does NOT emit edges. Does NOT advance any channel EMA.
        """
        assert self._step_begin_called, (
            "bind_intra_step before note_step_begin"
        )
        assert t == self.t, "bind_intra_step out of step"
        dets_tuple = tuple(sorted(set(dets), key=_detection_sort_key))
        for d in dets_tuple:
            sig_hash = _node_signature_hash(d.sigma)
            pat = (d.k, sig_hash)
            if pat in self._higher_present_this_step:
                continue
            if pat in self._v1_present_this_step:
                # V1 already handled this pattern this step; skip to avoid
                # double history append (futurework3.md II.5).
                continue
            node_id = self.cg.get_or_create_pattern_node(
                d.k, sig_hash, signature=d.sigma
            )
            self._higher_node_ids[d] = node_id
            self._remember_node_anchor(node_id, d.alpha)
            # [V9 Track 2 - B.26] Nodes minted after step() returns: the
            # step's position map is still live here (finalize_step clears
            # it), so capture inline rather than in a later pass.
            self._remember_node_position(node_id, d.alpha)
            self._higher_dets_this_step.append(d)
            self._higher_present_this_step.add(pat)
            self._pattern_last_seen[pat] = t
            self._pattern_anchors[pat].add(d.alpha)

            prev_and_v1 = self._pattern_present_prev | self._v1_present_this_step
            if pat not in prev_and_v1:
                self._higher_onset_this_step.add(pat)
                self._pattern_last_onset[pat] = t
                self._pattern_onset_now.add(pat)  # extend for same-step readers

            hist = self._pattern_history[pat]
            hist.append(1)
            if len(hist) > P.W_HISTORY:
                hist.pop(0)

    def emit_pending_edges(self, level: str) -> None:
        """Emit edges for one higher-lens level. Each channel is emitted
        exactly once per step."""
        assert self._step_begin_called, (
            "emit_pending_edges before note_step_begin"
        )
        if not self._higher_dets_this_step:
            return
        # Sorted here as well as at registration: the level partition means
        # the same detection set is visited by more than one level, and each
        # channel must see one canonical order.
        dets_tuple = tuple(
            sorted(self._higher_dets_this_step, key=_detection_sort_key)
        )
        node_ids = dict(self._higher_node_ids)
        pos = self._step_anchor_positions
        if level == "v2":
            # Each emitter reads self._step_extraction_scale itself, so no
            # scale is threaded through this call.
            self._run_contour_continuity(dets_tuple, node_ids, pos)
            self._run_junction_contour(dets_tuple, node_ids, pos)
            # [T1.1 — V9 Track 1] The contour/junction orientation
            # emitters were called TWICE here (four lines, two pairs).
            # Beyond the wasted pass, `_process_channel` overwrites
            # M_prev in place, so the duplicate doubled the homeostatic
            # EMA rate for CONTOUR_ORIENTATION / JUNCTION_ORIENTATION --
            # the second-order effect was the real defect. One call
            # each. Measured: candidate counts halve.
            self._run_contour_orientation(dets_tuple, node_ids)
            self._run_junction_orientation(dets_tuple, node_ids)
            self._run_parallel(dets_tuple, node_ids, pos)
            self._run_border_ownership_edge(dets_tuple, node_ids, pos)
            # [Q3] Anti-phase coupling: opposite flank owners across a shared
            # boundary, and closed-chain interior vs exterior. Without these
            # the coupling matrix is all-positive and the oscillator can
            # synchronise but never separate (diagnostics/_a6_maskfree_result).
            self._run_anti_phase_border_ownership(dets_tuple, node_ids, pos)
            self._run_anti_phase_containment(dets_tuple, node_ids, pos)
            self._run_curvature_sign_continuity(dets_tuple, node_ids, pos)
            self._run_cocircular(dets_tuple, node_ids, pos)
            # [Pass 19] Emission ENABLED. The blocker was the inverted
            # evidence_score argument order in the consolidator (now fixed at
            # _evidence_margin / _scores / _classify_prefiltered): the score
            # was "what fraction of the training IMAGE does this core
            # explain?", so extra structural edges raised the negative leg and
            # pushed cow's unfiltered core past THETA_EVIDENCE_NEG.
            self._run_amodal_link(dets_tuple, node_ids, pos)
            self._run_shared_boundary(dets_tuple, node_ids, pos)
            # [Pass 23] Mid-level (V1/V2) vocabulary: colour-opponent
            # borders, end-stopped line terminations, and cross-scale
            # correspondence.
            self._run_color_opponent_border(dets_tuple, node_ids, pos)
            self._run_line_termination(dets_tuple, node_ids, pos)
            self._run_cross_scale_identity(dets_tuple, node_ids, pos)
        elif level == "v4":
            self._run_curve_containment(dets_tuple, node_ids, pos)
            self._run_symmetry(dets_tuple, node_ids, pos)
            # [Pass 23] Higher-level (V4/MT) vocabulary: non-Cartesian
            # shape structure and surface folding.
            self._run_concentricity(dets_tuple, node_ids, pos)
            self._run_radiality(dets_tuple, node_ids, pos)
            self._run_spirality(dets_tuple, node_ids, pos)
            self._run_surface_fold(dets_tuple, node_ids, pos)
            self._run_texture_boundary_edge(dets_tuple, node_ids, pos)
        elif level == "lo_pit":
            self._run_part_containment(dets_tuple, node_ids, pos)
            self._run_part_content(dets_tuple, node_ids)
            # PART_SEQUENCE, PART_AT_ANGLE, OBJECT_PART are REMOVED.
            # These relations emerge from the Consolidator's intersection
            # of containment edges, not from hardcoded binders.
        else:
            raise ValueError(f"emit_pending_edges: unknown level {level!r}")

    def clear_higher_accumulator(self) -> None:
        """[V9 Track 2 - B.8] Clear the per-step higher-lens accumulators.

        Deliberately UNCALLED by the live pipeline (single-pass per
        observation): `note_step_begin` and `finalize_step` already bracket
        these four fields, so the accumulator cannot go stale between steps
        as written. This exists for the settle-loop integration seam, and
        any caller must be gated on its own measurement -- calling it
        MID-step, after the level loop but before `finalize_step`, drops
        the same-step higher registrations from `_pattern_present_prev` /
        `_pattern_onset_prev` and therefore changes cross-step
        PERSISTENCE / TRANSITION behaviour.

        Must NOT be called from inside `emit_pending_edges` (levels 2-4
        would see an empty accumulator and never emit).
        """
        self._higher_dets_this_step = []
        self._higher_node_ids = {}
        self._higher_present_this_step = set()
        self._higher_onset_this_step = set()

    def step(
        self,
        t: int,
        D_bind: FrozenSet[Detection],
        anchor_positions: Optional[Dict[Any, Tuple[float, ...]]] = None,
        anchor_velocities: Optional[Dict[Any, Tuple[float, ...]]] = None,
        static_frame: bool = False,
        t_junctions: Optional[
            List[Tuple[int, int, Tuple[float, float], float]]
        ] = None,
        extraction_scale: float = 0.0,
        structural_anchors: Optional[FrozenSet[int]] = None,
        surface_anchors: Optional[FrozenSet[int]] = None,
        contour_surface_contacts: Optional[List[Tuple[int, int, float]]] = None,
    ) -> None:
        # [STATIC-FRAME GUARD] In single-frame (static) observations every
        # region is co-present at t=0, so all-to-all CO_ACTIVATION forms a
        # vacuous K_N clique that drowns the real spatial geometry (CONTACT/
        # TOPOLOGY/DEGREE) in edge similarity. Callers processing static
        # stills set static_frame=True; temporal pipeline steps keep the
        # channel (default False).
        self._static_frame = bool(static_frame)
        # [Stage 1.2] Two-stream registries supplied by Observation2D via
        # Pipeline/probe. Empty => predicates fall back to "structural"
        # (legacy behavior for callers without stream info).
        self._structural_anchors: FrozenSet[Any] = structural_anchors or frozenset()
        self._surface_anchors: FrozenSet[Any] = surface_anchors or frozenset()
        self._contour_surface_contacts: List[Tuple[int, int, float]] = (
            list(contour_surface_contacts) if contour_surface_contacts else []
        )
        """Run all binders for step t. Writes edges into the concept graph.

        anchor_positions and anchor_velocities are keyed by anchor. Spatial
        binders (Topology, Degree, Contact, Co-motion) skip if the required
        context is missing. See D6.
        """
        # [PHASE B] B.3: the Pipeline brackets a step with note_step_begin()
        # and finalize_step(). Callers that drive Binder.step() directly
        # (temporal_bridge.run_bridge, the real-image tests) do not bracket,
        # so self-manage here — otherwise _pattern_present_prev /
        # _pattern_onset_prev would never advance for them and every step
        # would re-report every pattern as an onset. Behaviour for those
        # callers is unchanged: they get a begin+commit pair per step, which
        # is exactly the pre-Step-2 semantics of the tail of step().
        if not self._step_begin_called:
            self.note_step_begin(t)
            self._lifecycle_owned = True
        # [PHASE B] Step 4: staged before any emitter runs, so the
        # higher-tier channels can read it later in this step through
        # emit_pending_edges.
        self._step_extraction_scale = float(extraction_scale)
        self.t = t
        dets = tuple(sorted(D_bind, key=_detection_sort_key))

        # [v8] Scene scale, computed once per step over the
        # Position-having anchors only. Read by DEGREE, CONTACT,
        # ORIENTATION_CONTINUITY. 0.0 signals "no scale available".
        self._step_l_med = 0.0
        if anchor_positions and len(anchor_positions) >= 2:
            position_anchors = {
                d.alpha for d in dets
                if d.k == "Position" and d.alpha in anchor_positions
            }
            if len(position_anchors) >= 2:
                self._step_l_med = self._median_nn_distance(
                    {a: anchor_positions[a] for a in position_anchors}
                )

        node_ids: Dict[Detection, int] = {
            d: self.cg.get_or_create_pattern_node(
                d.k, _node_signature_hash(d.sigma), signature=d.sigma
            )
            for d in dets
        }
        for _d, _nid in node_ids.items():
            self._remember_node_anchor(_nid, _d.alpha)

        self._update_pattern_history(dets)

        # Order-1.
        self._run_co_activation(dets, node_ids)
        self._run_transition(dets, node_ids)
        self._run_co_motion(dets, node_ids, anchor_velocities)
        # [Pass 23] MT motion opposition requires velocity data, so it is
        # emitted from step() rather than the higher-lens level pass.
        self._run_motion_opposition(
            dets, node_ids, anchor_positions, anchor_velocities
        )
        # [Pass 23] Depth separation is gated on genuinely 3D anchors; with
        # flat (2D) positions it can never fire, so the gate is explicit.
        # [ROUND 11 / O4] The z proxy is NOT a disparity measurement, so the
        # channel is additionally BLOCKED behind an explicit opt-in flag; it
        # cannot be produced from scene z unless an operator turns it on.
        if self._binocular_disparity_enabled(anchor_positions):
            self._run_binocular_disparity(
                dets, node_ids, anchor_positions
            )
        self._run_contrast(dets, node_ids)
        self._run_periodicity(dets, node_ids)
        self._run_persistence(dets, node_ids)
        self._run_topology(dets, node_ids, anchor_positions)
        self._run_degree(dets, node_ids, anchor_positions)
        self._run_contact(dets, node_ids, anchor_positions)
        # [Stage 1.2] Bind surface patches to contour boundary entities
        self._run_stream_binding(dets, node_ids)

        self._run_sequence(dets, node_ids)
        self._run_synchrony(dets, node_ids)
        self._run_orientation_continuity(dets, node_ids, anchor_positions)
        self._run_surface_grouping(dets, node_ids, anchor_positions)
        self._run_curvature(dets, node_ids, anchor_positions)
        self._run_occlusion(dets, node_ids, anchor_positions, t_junctions)
        self._run_co_located(dets, node_ids)
        self._run_orientation_bend(dets, node_ids, anchor_positions)
        self._run_orientation_junction(dets, node_ids, anchor_positions)

        # [PHASE B] B.3: V1 presence/onsets are staged, not committed. The
        # commit is finalize_step()'s job so that higher-lens patterns
        # registered later in the same step (bind_intra_step) join it.
        self._v1_present_this_step = {
            (d.k, _node_signature_hash(d.sigma)) for d in dets
        }
        self._v1_onset_this_step = set(self._pattern_onset_now)
        self._step_anchor_positions = anchor_positions

        # [V9 Track 2 - B.26] Delayed position capture. Both minting sites
        # ran earlier in this step, before `anchor_positions` was bound, so
        # this pass is where the step's V1 nodes get their geometry. The
        # pass is O(|dets|); first write wins, so re-visiting a node that
        # was bound in an earlier step changes nothing.
        for _d, _nid in node_ids.items():
            self._remember_node_position(_nid, _d.alpha)

        if self._lifecycle_owned:
            # Self-managed step (no Pipeline bracket) — commit here so direct
            # callers keep the pre-Step-2 contract.
            self.finalize_step()

    # -------------------------------------------------------------------------
    # [U23 LEAD 2] ORIENTATION SPINE MESH -- offered vs kept, read-only.
    # -------------------------------------------------------------------------

    def orientation_mesh_counts(self) -> Dict[str, Dict[str, int]]:
        """Offered vs kept orientation-spine edges, per channel.

        Read-only view over the counters the channels already maintain:

        * ``candidates_by_type``  -- candidates OFFERED to the winner-take-all
          rule;
        * ``emitted_by_type``       -- candidates KEPT and emitted;
        * ``dropped_below_threshold`` -- offered-but-dropped by the
          homeostasis hysteresis / threshold.

        The pruning itself (the WTA ``strict >`` update, the bipole roles) is
        NOT touched by this accessor; it is exactly the measurement the U23
        spatial-mesh lead asked for.
        """
        out: Dict[str, Dict[str, int]] = {}
        for et in (
            EdgeType.ORIENTATION_CONTINUITY,
            EdgeType.ORIENTATION_BEND,
            EdgeType.ORIENTATION_JUNCTION,
        ):
            name = et.name
            out[name] = {
                "offered": int(self.diagnostics["candidates_by_type"].get(name, 0)),
                "kept": int(self.diagnostics["emitted_by_type"].get(name, 0)),
                "dropped": int(
                    self.diagnostics["dropped_below_threshold"].get(name, 0)
                ),
            }
        return out

    def get_diagnostics(self) -> Dict[str, Any]:
        channel_snapshot = {
            et.name: {
                "theta": self.channels[et]["theta"],
                "r_bar": self.channels[et]["r_bar"],
                "C": self.channels[et]["C"],
                "F": self.channels[et]["F"],
                "M_prev_size": len(self.channels[et]["M_prev"]),
            }
            for et in EdgeType
        }
        return {
            "channels": channel_snapshot,
            "candidates_by_type": dict(self.diagnostics["candidates_by_type"]),
            "emitted_by_type": dict(self.diagnostics["emitted_by_type"]),
            "candidate_cap_hits": dict(self.diagnostics["candidate_cap_hits"]),
            "dropped_below_threshold": dict(
                self.diagnostics["dropped_below_threshold"]
            ),
            "empty_or_zero_signal_by_type": dict(
                self.diagnostics["empty_or_zero_signal_by_type"]
            ),
            "anti_phase_pairs_offered": dict(
                self.diagnostics["anti_phase_pairs_offered"]
            ),
            "anti_phase_unresolved_endpoints": dict(
                self.diagnostics["anti_phase_unresolved_endpoints"]
            ),
            "anti_phase_candidates": dict(
                self.diagnostics["anti_phase_candidates"]
            ),
        }

    def reset(self) -> None:
        self.__init__(self.cg)

    # -------------------------------------------------------------------------
    # Action-pattern onset pre-registration (B2 execution-time attribution)
    # -------------------------------------------------------------------------

    def note_executed_action(self, t: int, det: Detection) -> int:
        """Register the ONSET of an Action-lens pattern at the step the
        action was EXECUTED (called by the Pipeline in Phase 9, after
        env.execute_action).

        Why: SEQUENCE edges require strictly positive lag (parent onset
        before child onset). The effects of action a_t appear in the
        observation at step t+1 — the same step the Action lens would
        naturally emit a_t (as `prev_action`). Emission-time onset would
        make action and effect co-onset (lag 0 ⇒ no edge) and instead
        misattribute the NEXT step's effects to a_t. Pre-registering the
        onset at execution time yields lag = 1 action→effect edges —
        correct causal attribution — and consecutive actions still chain
        (A(a_t) onset t → A(a_{t+1}) onset t+1) for §5.3.4 replay order.

        Mechanics: creates/refreshes the pattern node, pins
        _pattern_last_onset to t, and marks the pattern present-prev so
        the lens emission at t+1 extends presence WITHOUT re-onsetting.
        A repeated action re-executed at t+1 refreshes its onset (each
        execution is a fresh causal event, keeping lag = 1 to its own
        effects). Deterministic; O(1).
        """
        sig_hash = _node_signature_hash(det.sigma)
        pat = (det.k, sig_hash)
        nid = self.cg.get_or_create_pattern_node(
            det.k, sig_hash, signature=det.sigma
        )
        # [V9 Track 2 - B.26] Same delayed-window rule as bind_intra_step;
        # no-op when the step's positions are already gone (None).
        self._remember_node_position(nid, det.alpha)
        self._pattern_last_onset[pat] = t
        self._pattern_last_seen[pat] = t
        # [PHASE B] Stage into the accumulator, NOT into _pattern_present_prev.
        # Since Step 2 the previous-step sets are committed by finalize_step()
        # at the end of the Pipeline step, which OVERWRITES
        # _pattern_present_prev wholesale. Writing the pre-registration there
        # directly would silently discard it, and the Action pattern would
        # re-onset at t+1 (lag 0 => no SEQUENCE edge, effects misattributed to
        # a_{t−1}). Staging it as V1 presence makes the commit carry it.
        # The Action lens is a V1 lens, so _v1_present_this_step is the right
        # bucket; it also makes bind_intra_step's duplicate guard correct.
        self._v1_present_this_step.add(pat)
        return nid

    # -------------------------------------------------------------------------
    # Workspace length scale (D2/Seam-4: metric-agnostic spatial radii)
    # -------------------------------------------------------------------------

    def _median_nn_distance(
        self, positions: Dict[Any, Tuple[float, ...]]
    ) -> float:
        """[v8] Median nearest-neighbour distance among the given
        positioned anchors. Returns 0.0 for fewer than two anchors."""
        coords = [p for p in positions.values() if p is not None]
        if len(coords) < 2:
            return 0.0
        nn_dists: List[float] = []
        for i in range(len(coords)):
            best = math.inf
            pi = coords[i]
            for j in range(len(coords)):
                if i == j:
                    continue
                pj = coords[j]
                k = min(len(pi), len(pj), 3)
                d_sq = 0.0
                for t in range(k):
                    diff = pi[t] - pj[t]
                    d_sq += diff * diff
                d = math.sqrt(d_sq)
                if d < best:
                    best = d
            if math.isfinite(best):
                nn_dists.append(best)
        if not nn_dists:
            return 0.0
        nn_dists.sort()
        mid = len(nn_dists) // 2
        if len(nn_dists) % 2 == 1:
            return nn_dists[mid]
        return 0.5 * (nn_dists[mid - 1] + nn_dists[mid])

    def _effective_degree_radius(
        self,
        positions: Optional[Dict[Any, Tuple[float, ...]]] = None,
    ) -> float:
        """[v8] Scale-adaptive radius:
        R_eff = max(κ · L_med_scene, DEGREE_RADIUS_FLOOR).

        `L_med_scene` is computed once per step in `Binder.step` and
        read from `self._step_l_med`. The `positions` argument is
        vestigial — retained for backward compatibility.

        When `_step_l_med == 0`, the result is DEGREE_RADIUS_FLOOR.
        Callers that need a physical default in this case (CONTACT)
        must branch explicitly."""
        return max(
            P.DEGREE_KAPPA * self._step_l_med,
            P.DEGREE_RADIUS_FLOOR,
        )

    # -------------------------------------------------------------------------
    # [PHASE B] V2+ higher-tier emitters (B.12)
    #
    # These read the V1 edge graph the ordinary binders laid down. Note the
    # staleness: a Contour built from ORIENTATION_CONTINUITY/CURVATURE edges
    # reflects the graph as of the PREVIOUS step, because the current
    # step's V1 edges are written before this block runs but the lenses
    # that read them were themselves built from them — see the SA-1 note in
    # the Step 5 report.
    #
    # Spatial gates are KAPPA_* * _step_extraction_scale, never absolute
    # lengths. See parameters.py for why the draft's absolute radii were
    # permanently unsatisfiable.
    # -------------------------------------------------------------------------

    def _run_contour_continuity(
        self,
        dets: Tuple[Detection, ...],
        node_ids: Dict[Detection, int],
        anchor_positions: Optional[Dict[Any, Tuple[float, ...]]],
    ) -> None:
        """Contour-to-Contour proximity + orientation agreement.

        Undirected, so candidate endpoints are canonicalized to (min, max)
        before keying.
        """
        extraction_scale = self._step_extraction_scale
        if extraction_scale <= 0.0:
            return
        R = P.KAPPA_CONTOUR_CONTINUITY * extraction_scale
        contour_dets = sorted(
            [d for d in dets if d.k == "Contour"], key=_detection_sort_key
        )
        det_pos: Dict[Detection, Optional[Tuple[float, ...]]] = {}
        det_theta: Dict[Detection, float] = {}
        for d in contour_dets:
            if len(d.sigma.quantized) < 2:
                det_pos[d] = None
                continue
            det_pos[d] = _mean_position(
                _extract_member_anchors(d), anchor_positions
            )
            # [O2] The contour MEAN THETA, by name.
            det_theta[d] = (
                (slot_value(d.sigma, "CONTOUR", "mean_theta",
                             default=0.0) + 0.5) * math.pi
                / P.ORIENTATION_N_THETA
            )

        candidates: List[Tuple[int, int, float, Tuple[int, int], EdgeParams]] = []
        pairs_seen = 0
        stop = False
        for i in range(len(contour_dets)):
            if stop:
                break
            a = contour_dets[i]
            if det_pos[a] is None:
                continue
            u = node_ids[a]
            for j in range(i + 1, len(contour_dets)):
                b = contour_dets[j]
                if det_pos[b] is None:
                    continue
                pairs_seen += 1
                if pairs_seen > P.MAX_CONTOUR_PAIRS_PER_STEP:
                    stop = True
                    break
                v = node_ids[b]
                if u == v:
                    continue
                dist = _pos_dist(det_pos[a], det_pos[b])
                delta = abs(det_theta[a] - det_theta[b])
                # theta is a half-circle quantity: pi - delta folds the
                # wrap-around without a modulo.
                delta = min(delta, math.pi - delta)
                signal = math.exp(-dist / R) * (1.0 - delta / math.pi)
                src, dst = (u, v) if u < v else (v, u)
                # [P2] THE NAMED WRITER. The parameters are the WRITER'S
                # CONTINUOUS MEASUREMENTS in the slot's own unit (the
                # separation in WAVELENGTHS, the signed turn in RADIANS, the
                # curvature agreement as a FRACTION); the unit table
                # (core.schema.EDGE_UNIT_TABLE) quantises them into the
                # lookup bins and builds the graded population from the SAME
                # numbers.  The old code addressed the slots by POSITION and
                # binned by hand, which is how the third slot ended up
                # carrying a chirality agreement under the name "curvature".
                dist_w = dist / max(extraction_scale, 1e-6)
                # [S5 contract] Signed turn binned in the lower->higher node ID
                # frame: the edge key (src, dst) is ID-ordered while a, b come
                # from detection sort order, so orient explicitly.
                raw_delta = (
                    det_theta[a] - det_theta[b]
                    if u < v
                    else det_theta[b] - det_theta[a]
                )
                turn = (raw_delta + math.pi / 2.0) % math.pi - math.pi / 2.0
                # Curvature agreement from the Contour summary chirality slot.
                # [F3] The chirality is CONTOUR slot 6; slot 4 is the
                # total turn (the binder was reading the wrong slot).
                # [O2] The contour CHIRALITY, by name (slot 6 of
                # CONTOUR; slot 4 is the total turn).
                chir_a = slot_value(a.sigma, "CONTOUR", "chirality",
                                   default=4.0)
                chir_b = slot_value(b.sigma, "CONTOUR", "chirality",
                                   default=4.0)
                curv_agree = max(0.0, min(1.0, 1.0 - abs(chir_a - chir_b) / 8.0))
                ep = _edge_params(
                    "EDGE_CONTOUR_CONTINUITY",
                    {"dist": dist_w, "turn": turn, "curv_agree": curv_agree},
                    sign=1,
                    confidence=float(clamp(signal, 0.0, 1.0)),
                )
                candidates.append((src, dst, signal, (src, dst), ep))

        self._process_channel(EdgeType.CONTOUR_CONTINUITY, candidates)

    def _run_junction_contour(
        self,
        dets: Tuple[Detection, ...],
        node_ids: Dict[Detection, int],
        anchor_positions: Optional[Dict[Any, Tuple[float, ...]]],
    ) -> None:
        """Junction-to-Contour proximity. Directed (hub -> contour)."""
        extraction_scale = self._step_extraction_scale
        if extraction_scale <= 0.0:
            return
        R = P.KAPPA_JUNCTION_CONTOUR * extraction_scale
        junction_dets = sorted(
            [d for d in dets if is_junction_kind(d.k)],
            key=_detection_sort_key
        )
        contour_dets = sorted(
            [d for d in dets if d.k == "Contour"], key=_detection_sort_key
        )

        det_pos: Dict[Detection, Optional[Tuple[float, ...]]] = {}
        for d in junction_dets + contour_dets:
            det_pos[d] = _mean_position(
                _extract_member_anchors(d), anchor_positions
            )

        candidates: List[Tuple[int, int, float, Tuple[int, int], EdgeParams]] = []
        pairs_seen = 0
        stop = False
        for j in junction_dets:
            if stop:
                break
            if det_pos[j] is None:
                continue
            u = node_ids[j]
            for c in contour_dets:
                if det_pos[c] is None:
                    continue
                pairs_seen += 1
                if pairs_seen > P.MAX_JUNCTION_CONTOUR_PAIRS:
                    stop = True
                    break
                v = node_ids[c]
                if u == v:
                    continue
                dist = _pos_dist(det_pos[j], det_pos[c])
                signal = 1.0 / (1.0 + dist / R)
                # [Pass 22] Incidence angle: the chord from the junction to
                # the contour versus the contour's own dominant tangent. High
                # incidence means a T-stem (occlusion evidence); low incidence
                # means a continuation arm. det_theta does not exist in this
                # method, so the tangent is read from the contour signature
                # slot 1 exactly as _run_contour_continuity does.
                chord_theta = math.atan2(
                    det_pos[c][1] - det_pos[j][1],
                    det_pos[c][0] - det_pos[j][0]
                ) % math.pi
                # [O2] The contour MEAN THETA, by name.
                contour_theta = (
                    (slot_value(c.sigma, "CONTOUR", "mean_theta",
                                 default=0.0) + 0.5) * math.pi
                    / P.ORIENTATION_N_THETA
                    if len(c.sigma.quantized) > 1 else 0.0
                )
                incidence = abs(chord_theta - contour_theta)
                incidence = min(incidence, math.pi - incidence)
                # Topology class straight from the Junction signature.
                # [O2] The junction TOPOLOGY CLASS, by name.
                topo_val = slot_value(j.sigma, "JUNCTION", "topo",
                                        default=3.0)
                # [P2] THE NAMED WRITER: the separation in WAVELENGTHS, the
                # INCIDENCE ANGLE in RADIANS (axial) and the junction's
                # TOPOLOGY CLASS as a categorical lookup key.
                dist_w = dist / max(extraction_scale, 1e-6)
                ep = _edge_params(
                    "EDGE_JUNCTION_CONTOUR",
                    {"dist": dist_w, "incidence": incidence,
                     "topology_class": float(min(6, max(0, int(round(topo_val)))))},
                    sign=1,
                    confidence=float(clamp(signal, 0.0, 1.0)),
                )
                candidates.append((u, v, signal, (u, v), ep))

        self._process_channel(EdgeType.JUNCTION_CONTOUR, candidates)

    def _run_contour_orientation(
        self,
        dets: Tuple[Detection, ...],
        node_ids: Dict[Detection, int],
    ) -> None:
        """Containment: Contour -> each Orientation segment it summarizes."""
        candidates: List[Tuple[int, int, float, Tuple[int, int], EdgeParams]] = []
        for d in dets:
            if d.k != "Contour":
                continue
            u = node_ids[d]
            for member_nid in _extract_member_node_ids(d):
                if u == member_nid:
                    continue
                # [P2] THE NAMED WRITER: the membership is a 0/1 FLAG (a
                # categorical lookup key -- there is no graded quantity
                # behind "this contour summarises that segment", so none is
                # invented).
                ep = _edge_params(
                    "EDGE_CONTOUR_ORIENTATION",
                    {"containment": 1.0},
                    sign=1,
                    confidence=1.0,
                )
                candidates.append((u, member_nid, 1.0, (u, member_nid), ep))
        candidates.extend(self._run_peak_chain_orientation(dets, node_ids))
        self._process_channel(EdgeType.CONTOUR_ORIENTATION, candidates)

    def _run_peak_chain_orientation(
        self,
        dets: Tuple[Detection, ...],
        node_ids: Dict[Detection, int],
    ) -> List[Tuple[int, int, float, Tuple[int, int], EdgeParams]]:
        """[A5] Containment: PEAK_CHAIN -> each member CONTOUR_PEAK.

        Source is the container (the chain), destination the member peak.
        Directed CONTOUR_ORIENTATION matches the convention already used by
        _run_contour_orientation (Contour -> Orientation segment).

        Member identity comes from the chain's `member` rho roles, which
        carry anchors under the field's PEAK_ANCHOR_BASE convention. Those
        anchors equal the CONTOUR_PEAK detections' alphas only when the
        CONTOUR_PEAK lenses were run with the same base, so a chain whose
        members were never emitted (e.g. the peak lens is disabled) simply
        produces no edges -- the emitter never invents members.
        """
        chains = [d for d in dets if d.k == "PEAK_CHAIN"]
        peaks_by_anchor: Dict[Any, Detection] = {}
        for d in dets:
            if d.k == "CONTOUR_PEAK":
                peaks_by_anchor.setdefault(d.alpha, d)
        if not chains:
            return []
        candidates: List[
            Tuple[int, int, float, Tuple[int, int], EdgeParams]
        ] = []
        for chain in chains:
            src = node_ids[chain]
            for role in chain.rho:
                if len(role) < 2 or role[0] != "member":
                    continue
                member_anchor = role[1]
                member_det = peaks_by_anchor.get(member_anchor)
                if member_det is None:
                    continue
                dst = node_ids[member_det]
                if src == dst:
                    continue
                # [P2] THE NAMED WRITER: the membership is a 0/1 FLAG (a
                # categorical lookup key -- there is no graded quantity
                # behind "this contour summarises that segment", so none is
                # invented).
                ep = _edge_params(
                    "EDGE_CONTOUR_ORIENTATION",
                    {"containment": 1.0},
                    sign=1,
                    confidence=1.0,
                )
                candidates.append((src, dst, 1.0, (src, dst), ep))
        return candidates

    def _run_junction_orientation(
        self,
        dets: Tuple[Detection, ...],
        node_ids: Dict[Detection, int],
    ) -> None:
        """Containment: Junction -> each Orientation segment it summarizes."""
        candidates: List[Tuple[int, int, float, Tuple[int, int], EdgeParams]] = []
        for d in dets:
            if not is_junction_kind(d.k):
                continue
            u = node_ids[d]
            for member_nid in _extract_member_node_ids(d):
                if u == member_nid:
                    continue
                # [P2] THE NAMED WRITER: the membership is a 0/1 FLAG.
                ep = _edge_params(
                    "EDGE_JUNCTION_ORIENTATION",
                    {"containment": 1.0},
                    sign=1,
                    confidence=1.0,
                )
                candidates.append((u, member_nid, 1.0, (u, member_nid), ep))
        self._process_channel(EdgeType.JUNCTION_ORIENTATION, candidates)

    @staticmethod
    def _containment_geometry(
        parent_det: Detection,
        child_nid: int,
        det_by_nid: Dict[int, Detection],
        anchor_positions: Optional[Dict[Any, Tuple[float, ...]]],
        r_scale: float,
    ) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        """(rel_dist_bin, polar_angle_bin, scale_ratio_bin) of a child node
        as seen from its parent detection, AND the CONTINUOUS positions
        those bins name.

        This is the posterior-IT compositional code: relative distance r/R
        to the parent, the egocentric polar angle phi, and log2 of the size
        ratio S_child/S_parent (Pasupathy & Connor 2001; Tanaka 1996). The
        angle slot is 24 bins on a circle, so it is declared with range 12.0
        and cycle 24 under the tree's N_HALF convention.

        The second triple holds the same three quantities in the writers'
        own bin units, BEFORE the rounding that names the bin: the graded
        population is built from these, so the activity vector keeps the
        sub-bin position the coarse index rounds off.

        Returns the neutral triple (0.0, 0.0, 4.0) when either end has no
        position or the child detection is unknown, so the edge is still
        emitted. [Pass 18] replaces a stub; it must never shrink the edge
        set relative to the pre-Pass-18 behaviour, or a benchmark regression
        would be attributable to the dropped edge rather than to the new
        geometry.
        """
        pa = anchor_positions.get(parent_det.alpha) if anchor_positions else None
        child = det_by_nid.get(child_nid)
        pb = (
            anchor_positions.get(child.alpha)
            if (child is not None and anchor_positions)
            else None
        )
        if pa is None or pb is None:
            return (0.0, 0.0, 4.0), (0.0, 0.0, 4.0)
        dx = pb[0] - pa[0]
        dy = pb[1] - pa[1]
        dist = math.hypot(dx, dy)
        R = max(r_scale, 1.0)
        # the CONTINUOUS relative distance, in the writer's bin units
        rel_dist_pos = (dist / (2.0 * R)) * 7.0
        rel_dist_bin = float(min(7, max(0, int(round(rel_dist_pos)))))
        angle = math.atan2(dy, dx) % (2.0 * math.pi)
        # the CONTINUOUS bearing, in the writer's bin units (24 per turn)
        angle_pos = (angle / (2.0 * math.pi)) * 24.0
        angle_bin = float(int(round(angle_pos)) % 24)
        p_s = float(getattr(parent_det, "scale", 0.0) or 0.0)
        c_s = float(getattr(child, "scale", 0.0) or 0.0)
        ratio = math.log2(c_s / p_s) if (p_s > 0.0 and c_s > 0.0) else 0.0
        # ratio 0.0 (equal scale) must land on the neutral centre bin 4.
        # the CONTINUOUS log2 scale ratio, in the writer's bin units
        scale_pos = ratio * 2.0 + 4.0
        scale_bin = float(min(7, max(0, int(round(ratio * 2.0)) + 4)))
        return (rel_dist_bin, angle_bin, scale_bin), (
            rel_dist_pos, angle_pos, scale_pos)

    @staticmethod
    def _containment_values(
        parent_det: Detection,
        child_nid: int,
        det_by_nid: Dict[int, Detection],
        anchor_positions: Optional[Dict[Any, Tuple[float, ...]]],
        r_scale: float,
    ) -> Dict[str, Optional[float]]:
        """The containment relation's CONTINUOUS measurements, in the
        unit table's units.

        [P2] THE NAMED WRITER: the separation in WAVELENGTHS (the
        relation's reach is 2 * R), the egocentric bearing in RADIANS and
        the LOG2 SIZE RATIO.  A value the writer cannot measure (no
        geometry) is None -- "no measurement", never a fabricated 0.
        """
        child = det_by_nid.get(child_nid)
        pa = anchor_positions.get(parent_det.alpha) if anchor_positions else None
        pb = (anchor_positions.get(child.alpha)
              if (child is not None and anchor_positions) else None)
        if pa is None or pb is None:
            return {"rel_dist": None, "polar_angle": None, "scale_ratio": None}
        dx = pb[0] - pa[0]
        dy = pb[1] - pa[1]
        dist = math.hypot(dx, dy)
        R = max(r_scale, 1.0)
        angle = math.atan2(dy, dx) % (2.0 * math.pi)
        p_s = float(getattr(parent_det, "scale", 0.0) or 0.0)
        c_s = float(getattr(child, "scale", 0.0) or 0.0)
        ratio = math.log2(c_s / p_s) if (p_s > 0.0 and c_s > 0.0) else None
        return {
            "rel_dist": dist / max(r_scale, 1e-6),
            "polar_angle": angle,
            "scale_ratio": ratio,
        }

    @staticmethod
    def _containment_bins(
        parent_det: Detection,
        child_nid: int,
        det_by_nid: Dict[int, Detection],
        anchor_positions: Optional[Dict[Any, Tuple[float, ...]]],
        r_scale: float,
    ) -> Tuple[float, float, float]:
        """The coarse containment bins (see `_containment_geometry`)."""
        bins, _positions = Binder._containment_geometry(
            parent_det, child_nid, det_by_nid, anchor_positions, r_scale)
        return bins

    def _run_curve_containment(
        self,
        dets: Tuple[Detection, ...],
        node_ids: Dict[Detection, int],
        anchor_positions: Optional[Dict[Any, Tuple[float, ...]]] = None,
    ) -> None:
        """Containment: CurveComplex -> each V2 node it bundles.

        [Pass 18 / C4] Payload is real polar geometry, not the (0.0,) stub.
        """
        candidates: List[Tuple[int, int, float, Tuple[int, int], EdgeParams]] = []
        det_by_nid = {nid: det for det, nid in node_ids.items()}
        r_scale = max(self._step_extraction_scale, 1.0)
        for d in dets:
            if d.k != "CurveComplex":
                continue
            if d not in node_ids:
                continue
            u = node_ids[d]
            for member_nid in _extract_member_node_ids(d):
                if u == member_nid:
                    continue
                _bins, _positions = self._containment_geometry(
                    d, member_nid, det_by_nid, anchor_positions, r_scale
                )
                values = self._containment_values(
                    d, member_nid, det_by_nid, anchor_positions, r_scale
                )
                ep = _edge_params(
                    "EDGE_CURVE_CONTAINMENT",
                    values,
                    sign=1,
                    confidence=1.0,
                )
                candidates.append((u, member_nid, 1.0, (u, member_nid), ep))
        self._process_channel(EdgeType.CURVE_CONTAINMENT, candidates)

    def _run_part_containment(
        self,
        dets: Tuple[Detection, ...],
        node_ids: Dict[Detection, int],
        anchor_positions: Optional[Dict[Any, Tuple[float, ...]]] = None,
    ) -> None:
        """Containment: RegionShape -> each CurveComplex it geometrically encloses.
        Directed: container -> contained.

        A CurveComplex is "contained" by a RegionShape when the CurveComplex's
        centroid falls within the RegionShape's spatial extent. This is a
        geometric enclosure test, not a lens extraction. The "part" emerges
        when the Consolidator intersects repeated containment patterns.
        """
        if not anchor_positions:
            return
        r_scale = max(self._step_extraction_scale, 1.0)
        if r_scale <= 0.0:
            return

        # Collect RegionShape detections (the containers)
        containers = []
        for d in dets:
            if d.k != "RegionShape":
                continue
            pos = anchor_positions.get(d.alpha)
            if pos is None or d not in node_ids:
                continue
            # RegionShape sigma: hist_bins + (area_bin, elong_bin) + (sym, axis, fold)
            # area_bin is at index -5 (after the histogram bins)
            q = d.sigma.quantized
            if len(q) < 5:
                continue
            containers.append((d, pos, node_ids[d]))

        # Collect CurveComplex detections (the potential parts)
        parts = []
        for d in dets:
            if d.k != "CurveComplex":
                continue
            pos = _mean_position(_extract_member_anchors(d), anchor_positions)
            if pos is None or d not in node_ids:
                continue
            parts.append((d, pos, node_ids[d]))

        if not containers or not parts:
            return

        candidates: List[Tuple[int, int, float, Tuple[int, int], EdgeParams]] = []
        for c_det, c_pos, c_nid in containers:
            for p_det, p_pos, p_nid in parts:
                if c_nid == p_nid:
                    continue
                dist = _pos_dist(c_pos, p_pos)
                # Containment radius: the part must be within 2x the
                # extraction scale of the container's centroid.
                if dist > 2.0 * r_scale:
                    continue
                # Signal decays with distance from container centroid.
                signal = math.exp(-dist / (2.0 * r_scale))
                if signal < 0.05:
                    continue
                # [P2] THE NAMED WRITER: the separation in WAVELENGTHS, the
                # egocentric bearing in RADIANS; the size ratio is NOT
                # measurable from this geometry, so its slot carries NO
                # MEASUREMENT (None -> UNKNOWN), never a fabricated
                # neutral value.
                angle = math.atan2(p_pos[1] - c_pos[1], p_pos[0] - c_pos[0]) % (2.0 * math.pi)
                ep = _edge_params(
                    "EDGE_PART_CONTAINMENT",
                    {"rel_dist": dist / max(r_scale, 1e-6),
                     "polar_angle": angle,
                     "scale_ratio": None},
                    sign=1,
                    confidence=float(clamp(signal, 0.0, 1.0)),
                )
                candidates.append((c_nid, p_nid, signal, (c_nid, p_nid), ep))

        self._process_channel(EdgeType.PART_CONTAINMENT, candidates)

    def _run_part_content(
        self,
        dets: Tuple[Detection, ...],
        node_ids: Dict[Detection, int],
    ) -> None:
        """Membership: RegionShape <-> content nodes inside it. Undirected.

        A content node (Color, Surface, Texture) is "inside" a RegionShape
        when they share the same anchor (entity). This is a geometric
        enclosure test based on shared entity identity, not a lens extraction.
        """
        # Collect RegionShape detections
        region_shapes = {}
        for d in dets:
            if d.k == "RegionShape" and d in node_ids:
                region_shapes[d.alpha] = node_ids[d]

        if not region_shapes:
            return

        # Collect content detections that share an anchor with a RegionShape
        candidates: List[Tuple[int, int, float, Tuple[int, int], EdgeParams]] = []
        for d in dets:
            if d.k not in ("Color", "Surface", "Texture"):
                continue
            if d.alpha not in region_shapes:
                continue
            if d not in node_ids:
                continue
            u = region_shapes[d.alpha]
            v = node_ids[d]
            if u == v:
                continue
            # Undirected: canonicalize to (min, max)
            src, dst = (u, v) if u < v else (v, u)
            # [P2] THE NAMED WRITER: the membership is a 0/1 FLAG.
            ep = _edge_params(
                "EDGE_REGION_CONTENT",
                {"containment": 1.0},
                sign=1,
                confidence=1.0,
            )
            candidates.append((src, dst, 1.0, (src, dst), ep))

        self._process_channel(EdgeType.REGION_CONTENT, candidates)

    def _run_parallel(
        self,
        dets: Tuple[Detection, ...],
        node_ids: Dict[Detection, int],
        anchor_positions: Optional[Dict[Any, Tuple[float, ...]]],
    ) -> None:
        """Contour-to-Contour parallelism relation. Undirected."""
        extraction_scale = self._step_extraction_scale
        if extraction_scale <= 0.0 or not anchor_positions:
            return
        R = P.KAPPA_CONTOUR_CONTINUITY * extraction_scale
        contour_dets = sorted(
            [d for d in dets if d.k == "Contour"], key=_detection_sort_key
        )
        det_pos: Dict[Detection, Optional[Tuple[float, ...]]] = {}
        det_theta: Dict[Detection, float] = {}
        for d in contour_dets:
            if len(d.sigma.quantized) < 2:
                det_pos[d] = None
                continue
            det_pos[d] = _mean_position(
                _extract_member_anchors(d), anchor_positions
            )
            # [O2] The contour MEAN THETA, by name.
            det_theta[d] = (
                (slot_value(d.sigma, "CONTOUR", "mean_theta",
                             default=0.0) + 0.5) * math.pi
                / P.ORIENTATION_N_THETA
            )

        candidates: List[Tuple[int, int, float, Tuple[int, int], EdgeParams]] = []
        pairs_seen = 0
        stop = False
        tol = getattr(P, "CURVE_COMPLEX_PARALLEL_TOL_RAD", math.pi / 12.0)
        for i in range(len(contour_dets)):
            if stop:
                break
            a = contour_dets[i]
            pa = det_pos[a]
            if pa is None:
                continue
            u = node_ids.get(a)
            if u is None:
                continue
            for j in range(i + 1, len(contour_dets)):
                b = contour_dets[j]
                pb = det_pos[b]
                if pb is None:
                    continue
                pairs_seen += 1
                if pairs_seen > P.MAX_CONTOUR_PAIRS_PER_STEP:
                    stop = True
                    break
                v = node_ids.get(b)
                if v is None or u == v:
                    continue
                d_theta = abs(det_theta[a] - det_theta[b])
                d_theta = min(d_theta, math.pi - d_theta)
                if d_theta > tol:
                    continue
                dist = _pos_dist(pa, pb)
                if dist > 3.0 * R:
                    continue
                dx = pb[0] - pa[0]
                dy = pb[1] - pa[1]
                theta_m = _axial_mean(det_theta[a], det_theta[b])
                ux, uy = math.cos(theta_m), math.sin(theta_m)
                perp_s = -uy * dx + ux * dy
                stag_s = ux * dx + uy * dy
                if perp_s < -1e-9 or (abs(perp_s) <= 1e-9 and stag_s < 0.0):
                    p, q, perp_s, stag_s = a, b, -perp_s, -stag_s
                else:
                    p, q = a, b
                conv = _axial_wrap(det_theta[p] - det_theta[q])

                perp_bin = _bin8(abs(perp_s), 0.0, 3.0 * R)
                stag_bin = _bin8(stag_s, -3.0 * R, 3.0 * R)
                conv_bin = _bin8(conv, -tol, tol)

                # [P2] THE NAMED WRITER: the PERPENDICULAR SEPARATION and
                # the ALONG-AXIS STAGGER in WAVELENGTHS (the relation's reach
                # is 3 * R = 6 wavelengths) and the SIGNED CONVERGENCE ANGLE
                # in RADIANS.
                signal = math.exp(-dist / R) * (1.0 - d_theta / max(tol, 1e-6))
                src, dst = (u, v) if u < v else (v, u)
                scale = max(extraction_scale, 1e-6)
                perp_w = abs(perp_s) / scale
                stag_w = stag_s / scale
                ep = _edge_params(
                    "EDGE_PARALLEL",
                    {"perp_dist": perp_w, "stagger_dist": stag_w,
                     "convergence": conv},
                    sign=1,
                    confidence=float(clamp(signal, 0.0, 1.0)),
                )
                candidates.append((src, dst, signal, (src, dst), ep))
        self._process_channel(EdgeType.PARALLEL, candidates)

    def _run_border_ownership_edge(
        self,
        dets: Tuple[Detection, ...],
        node_ids: Dict[Detection, int],
        anchor_positions: Optional[Dict[Any, Tuple[float, ...]]],
    ) -> None:
        """Border ownership binding: BorderOwnership -> RegionShape / SurfacePatch."""
        bo_dets = [d for d in dets if d.k == "BorderOwnership"]
        if not bo_dets:
            return
        # [ROUND 10 / C] Anchor -> node id, so a fragment ownership's same-figure
        # `partner` (rho) can be bound even when there is no RegionShape/Surface
        # target in the graph.
        anchor_nid: Dict[Any, int] = {}
        for d, nid in sorted(node_ids.items(),
                             key=lambda kv: _detection_sort_key(kv[0])):
            anchor_nid.setdefault(d.alpha, nid)
        target_dets = [d for d in dets if d.k in ("RegionShape", "Surface", "SurfacePatch")]
        candidates: List[Tuple[int, int, float, Tuple[int, int], EdgeParams]] = []
        for bo in bo_dets:
            u = node_ids.get(bo)
            if u is None:
                continue
            # Same-figure partner: the fragment found across the boundary whose
            # ownership points back at this one. A positive (sign=+1) coupling.
            partner = _role_anchor(bo, "partner")
            if partner is not None:
                v = anchor_nid.get(partner)
                if v is not None and v != u:
                    strength = abs(float(getattr(bo, "z", 0.0) or 0.0))
                    # [P2] THE NAMED WRITER. The SIGNED figure-ground scalar
                    # (the single reader: perception.ownership.ownership_signed)
                    # goes into slot 0 -- it is a SIGNED SCALAR in [-1, 1],
                    # NOT an angle, and the schema used to declare it as one
                    # ("ang_diff").  Slot 1 is the SEPARATION in wavelengths
                    # (the schema used to declare it as a 24-cycle "bearing",
                    # which the reader compared as an angle over 2*pi).
                    from perception.ownership import ownership_signed
                    ownership = ownership_signed(bo)
                    # [O2] The border ownership CURVATURE, by name.
                    curv = slot_value(bo.sigma, "BORDER_OWNERSHIP",
                                      "curvature", default=0.0)
                    curv_signed = (curv + 0.5) / 8.0 * 2.0 - 1.0
                    pa = anchor_positions.get(bo.alpha) if anchor_positions else None
                    pb = anchor_positions.get(partner) if anchor_positions else None
                    dist_w = None
                    if pa is not None and pb is not None:
                        dist_w = _pos_dist(pa, pb) / max(
                            self._step_extraction_scale, 1e-6)
                    ep = _edge_params(
                        "EDGE_BORDER_OWNERSHIP",
                        {"owner_signed": ownership, "dist": dist_w,
                         "curvature": curv_signed},
                        sign=1,
                        confidence=float(clamp(strength, 0.0, 1.0)),
                    )
                    candidates.append((u, v, strength, (u, v), ep))
            for tgt in target_dets:
                v = node_ids.get(tgt)
                if v is None or u == v:
                    continue
                if bo.alpha == tgt.alpha:
                    signal = 1.0
                    from perception.ownership import ownership_signed
                    ownership = ownership_signed(bo)
                    # [O2] The border ownership CURVATURE, by name.
                    curv = slot_value(bo.sigma, "BORDER_OWNERSHIP",
                                      "curvature", default=0.0)
                    curv_signed = (curv + 0.5) / 8.0 * 2.0 - 1.0
                    ep = _edge_params(
                        "EDGE_BORDER_OWNERSHIP",
                        {"owner_signed": ownership, "dist": 0.0,
                         "curvature": curv_signed},
                        sign=1,
                        confidence=1.0,
                    )
                    candidates.append((u, v, signal, (u, v), ep))
                    continue
                pa = anchor_positions.get(bo.alpha) if anchor_positions else None
                pb = anchor_positions.get(tgt.alpha) if anchor_positions else None
                if pa is None or pb is None:
                    continue
                r = max(self._step_extraction_scale, 1.0)
                dist = _pos_dist(pa, pb)
                if dist > 2.0 * r:
                    continue
                signal = math.exp(-dist / r)
                # The direction slot is a SIGNED figure-ground scalar, NOT an
                # angle (perception/ownership.py). Its sign says whether the
                # detection's entity alpha is the figure (>= 0) or the ground
                # (< 0); the magnitude is the evidence. Emitting the sign makes
                # the vocabulary signed: +1 couples the border to the figure it
                # owns, -1 marks the ground / T-occlusion side.
                from perception.ownership import ownership_signed
                ownership = ownership_signed(bo)
                curv = slot_value(bo.sigma, "BORDER_OWNERSHIP",
                                  "curvature", default=0.0)
                curv_signed = (curv + 0.5) / 8.0 * 2.0 - 1.0
                dist_w = dist / max(self._step_extraction_scale, 1e-6)
                sign = (1 if ownership >= 0.0 else -1) \
                    if bool(getattr(P, "BO_SIGNED_EDGE_ENABLED", True)) else 1
                ep = _edge_params(
                    "EDGE_BORDER_OWNERSHIP",
                    {"owner_signed": ownership, "dist": dist_w,
                     "curvature": curv_signed},
                    sign=sign,
                    confidence=float(clamp(signal, 0.0, 1.0)),
                )
                candidates.append((u, v, signal, (u, v), ep))
        self._process_channel(EdgeType.BORDER_OWNERSHIP_EDGE, candidates)

    def _run_anti_phase_border_ownership(
        self,
        dets: Tuple[Detection, ...],
        node_ids: Dict[Detection, int],
        anchor_positions: Optional[Dict[Any, Tuple[float, ...]]],
    ) -> None:
        """[Q3] Anti-phase edges from opposite flank owners across a boundary.

        A BorderOwnership detection names the entity whose boundary it is (its
        `alpha`) and the entity on the far side (rho "neighbor") -- the two
        flank owners of one shared boundary. The detection's SIGNED direction
        slot (perception/ownership.py) says which of the two is the figure, so
        the pair is emitted figure -> ground: the figure side owns the border
        and the ground side sits behind it, which separates them rather than
        binding them. Positive EDGE_BORDER_OWNERSHIP edges (entity -> the
        region it faces) are emitted by `_run_border_ownership_edge`; this
        channel is the complement and carries sign=-1 on OCCLUSION. OCCLUSION
        is a DIRECTED type, so the figure->ground order is preserved and the
        edge is near (figure) -> far (ground).
        """
        if not bool(P.ANTIPHASE_ENABLED):
            return
        bo_dets = [d for d in dets if d.k == "BorderOwnership"]
        if not bo_dets:
            return
        from perception.anti_phase import border_ownership_claims_directed
        # Anchor -> node id, resolved over the REGISTERED detections only.
        # A BorderOwnership entity anchor is a region anchor that is also
        # carried by V1 content detections (Color/Texture/Position), which
        # are not kept in `_higher_node_ids`; when no registered detection
        # sits at that anchor there is no node to bind, and the pair is
        # counted as unresolved instead of being dropped silently.
        anchor_nid: Dict[Any, int] = {}
        for d, nid in sorted(node_ids.items(),
                             key=lambda kv: _detection_sort_key(kv[0])):
            anchor_nid.setdefault(d.alpha, nid)
        pairs = border_ownership_claims_directed(
            bo_dets, min_strength=float(P.ANTIPHASE_MIN_CONFIDENCE))
        candidates: List[Tuple[int, int, float, Tuple[int, int], EdgeParams]] = []
        for a, b, strength in pairs:      # a = figure, b = ground
            u, v = anchor_nid.get(a), anchor_nid.get(b)
            if u is None or v is None:
                self.diagnostics["anti_phase_unresolved_endpoints"][
                    EdgeType.BORDER_OWNERSHIP_EDGE.name] += 1
                continue
            if u == v:
                continue
            a_id, b_id = u, v            # directed: figure -> ground
            # [Q3] Emitted on OCCLUSION, not on BORDER_OWNERSHIP_EDGE: the
            # positive border-ownership channel already writes that type for
            # the same node pairs, and one concept-graph edge per
            # (src, dst, type) key means the two emitters would overwrite
            # each other (observed: 11 anti-phase candidates, 3 edges left in
            # the store, 2 of them clobbered). The occlusion payload is the
            # schema's (depth, stack_depth, source).
            # [P2] MIS-READ FIX: the depth slot is a LAYER COUNT (the gap
            # between the occluder and the occluded).  The anti-phase claim
            # measures a claim STRENGTH, not a rank gap, so writing the
            # strength there was a category error (a fraction in [0, 1] read
            # as a layer count).  The claim says the two are ADJACENT in
            # depth (depth = 0) and its strength travels in the edge's
            # CONFIDENCE, where it belongs.
            ep = _edge_params(
                "EDGE_OCCLUSION",
                {"depth": 0.0, "stack_depth": 8.0, "source": 0.0,
                 "depth_units": None},
                sign=-1,
                confidence=float(clamp(strength, 0.0, 1.0)),
            )
            candidates.append((a_id, b_id, strength, (a_id, b_id), ep))
        self.diagnostics["anti_phase_pairs_offered"][
            EdgeType.OCCLUSION.name] += len(pairs)
        self.diagnostics["anti_phase_candidates"][
            EdgeType.OCCLUSION.name] += len(candidates)
        self._process_channel(EdgeType.OCCLUSION, candidates)

    @staticmethod
    def _det_for_anchor(dets: Tuple[Detection, ...], anchor: Any) -> Optional[Detection]:
        """First detection (detection-sorted order) carried by `anchor`.

        Deterministic: `dets` is already sorted by `_detection_sort_key`.
        """
        for d in dets:
            if d.alpha == anchor:
                return d
        return None

    def _run_anti_phase_containment(
        self,
        dets: Tuple[Detection, ...],
        node_ids: Dict[Detection, int],
        anchor_positions: Optional[Dict[Any, Tuple[float, ...]]],
    ) -> None:
        """[Q3] Anti-phase edges from closed-chain containment.

        A PEAK_CHAIN whose closure slot is set is one surface's silhouette.
        Every other detection whose anchor falls strictly inside its member
        ring sits behind it on another depth layer, so the ring (exterior)
        and the interior are separated, not bound. Emitted as sign=-1 with
        confidence = containment depth (centre = 1, boundary -> 0).
        """
        if not bool(P.ANTIPHASE_ENABLED):
            return
        chain_dets = [d for d in dets if d.k == "PEAK_CHAIN"]
        if not chain_dets:
            return
        others = [d for d in dets if d.k != "PEAK_CHAIN"]
        if not others:
            return
        from perception.anti_phase import containment_pairs
        r_scale = max(self._step_extraction_scale, 1.0)
        # Same anchor -> registered-node resolution as the border-ownership
        # channel: a PEAK_CHAIN anchor is unique, but the enclosed anchor is
        # usually a V1 content detection, which is not in `_higher_node_ids`.
        anchor_nid: Dict[Any, int] = {}
        for d, nid in sorted(node_ids.items(),
                             key=lambda kv: _detection_sort_key(kv[0])):
            anchor_nid.setdefault(d.alpha, nid)
        # Area floor: a closed contour must enclose at least a quarter of an
        # extraction-scale area, or it is peak noise rather than a silhouette.
        pairs = containment_pairs(chain_dets, others, anchor_positions,
                                  min_area=0.25 * r_scale * r_scale)
        candidates: List[Tuple[int, int, float, Tuple[int, int], EdgeParams]] = []
        for ring_anchor, inner_anchor, depth in pairs:
            u, v = anchor_nid.get(ring_anchor), anchor_nid.get(inner_anchor)
            if u is None or v is None:
                self.diagnostics["anti_phase_unresolved_endpoints"][
                    EdgeType.OCCLUSION.name] += 1
                continue
            if u == v:
                continue
            if depth < float(P.ANTIPHASE_MIN_CONFIDENCE):
                continue
            a_id, b_id = (u, v) if u < v else (v, u)
            # [Q3] OCCLUSION again (see the border-ownership channel): a ring
            # and its interior are an occlusion pair, the positive
            # PART_CONTAINMENT emitter already owns that type, and one edge
            # per key means the two would clobber each other. source = 1
            # (depth / containment) distinguishes this source from the
            # boundary one.
            # [P2] MIS-READ FIX: the depth slot is a LAYER COUNT; the
            # containment claim measures a DEPTH (a containment fraction),
            # so the claim's strength travels in the CONFIDENCE and the depth
            # reads as the adjacent layer (0).
            ep = _edge_params(
                "EDGE_OCCLUSION",
                {"depth": 0.0, "stack_depth": 8.0, "source": 1.0,
                 "depth_units": None},
                sign=-1,
                confidence=float(clamp(depth, 0.0, 1.0)),
            )
            candidates.append((a_id, b_id, depth, (a_id, b_id), ep))
        self.diagnostics["anti_phase_pairs_offered"][
            EdgeType.OCCLUSION.name] += len(pairs)
        self.diagnostics["anti_phase_candidates"][
            EdgeType.OCCLUSION.name] += len(candidates)
        self._process_channel(EdgeType.OCCLUSION, candidates)

    def _run_curvature_sign_continuity(
        self,
        dets: Tuple[Detection, ...],
        node_ids: Dict[Detection, int],
        anchor_positions: Optional[Dict[Any, Tuple[float, ...]]],
    ) -> None:
        """Pairwise continuity of curvature sign between adjacent Contour fragments."""
        contour_dets = sorted(
            [d for d in dets if d.k == "Contour"], key=_detection_sort_key
        )
        if len(contour_dets) < 2 or not anchor_positions:
            return
        extraction_scale = self._step_extraction_scale
        if extraction_scale <= 0.0:
            return
        R = P.KAPPA_CONTOUR_CONTINUITY * extraction_scale
        candidates: List[Tuple[int, int, float, Tuple[int, int], EdgeParams]] = []
        pairs_seen = 0
        for i in range(len(contour_dets)):
            a = contour_dets[i]
            pa = anchor_positions.get(a.alpha)
            if pa is None or a not in node_ids:
                continue
            u = node_ids[a]
            q_a = a.sigma.quantized
            chir_a = (q_a[6] - 4.0) if len(q_a) > 6 else 0.0
            for j in range(i + 1, len(contour_dets)):
                b = contour_dets[j]
                pb = anchor_positions.get(b.alpha)
                if pb is None or b not in node_ids:
                    continue
                v = node_ids[b]
                if u == v:
                    continue
                pairs_seen += 1
                if pairs_seen > P.MAX_CONTOUR_PAIRS_PER_STEP:
                    break
                dist = _pos_dist(pa, pb)
                if dist > 2.0 * R:
                    continue
                q_b = b.sigma.quantized
                chir_b = (q_b[6] - 4.0) if len(q_b) > 6 else 0.0
                same_sign = (chir_a * chir_b > 0.0) or (abs(chir_a) < 0.5 and abs(chir_b) < 0.5)
                if not same_sign:
                    continue
                signal = math.exp(-dist / R) * (1.0 - abs(chir_a - chir_b) / 8.0)
                if signal <= 0.0:
                    continue
                # [P2] THE NAMED WRITER: the difference of the two contours'
                # NET SIGNED TURNS in RADIANS (the chirality is the net
                # signed turn over 2*pi) and the separation in WAVELENGTHS.
                curv_diff = abs(chir_a - chir_b) * 2.0 * math.pi
                dist_w = dist / max(extraction_scale, 1e-6)
                src, dst = (u, v) if u < v else (v, u)
                ep = _edge_params(
                    "EDGE_CURVATURE_SIGN_CONTINUITY",
                    {"curv_diff": curv_diff, "dist": dist_w},
                    sign=1,
                    confidence=float(clamp(signal, 0.0, 1.0)),
                )
                candidates.append((src, dst, signal, (src, dst), ep))
        self._process_channel(EdgeType.CURVATURE_SIGN_CONTINUITY, candidates)

    def _run_cocircular(
        self,
        dets: Tuple[Detection, ...],
        node_ids: Dict[Detection, int],
        anchor_positions: Optional[Dict[Any, Tuple[float, ...]]],
    ) -> None:
        """Cocircular alignment between pairs of oriented elements (Orientation / Contour)."""
        orient_dets = sorted(
            [d for d in dets if d.k in ("Orientation", "Contour")], key=_detection_sort_key
        )
        if len(orient_dets) < 2 or not anchor_positions:
            return
        extraction_scale = self._step_extraction_scale
        if extraction_scale <= 0.0:
            return
        R = P.KAPPA_CONTOUR_CONTINUITY * extraction_scale
        candidates: List[Tuple[int, int, float, Tuple[int, int], EdgeParams]] = []
        pairs_seen = 0
        max_pairs = P.MAX_CONTOUR_PAIRS_PER_STEP
        n_theta = P.ORIENTATION_N_THETA
        tol = getattr(P, "COCIRCULAR_MAX_PATH_ANGLE", math.pi / 3.0)
        for i in range(len(orient_dets)):
            a = orient_dets[i]
            pa = anchor_positions.get(a.alpha)
            if pa is None or a not in node_ids:
                continue
            u = node_ids[a]
            # [O2] The theta slot, by name: a Contour's MEAN THETA,
            # an Orientation's THETA.
            theta_name = "mean_theta" if a.k == "Contour" else "theta"
            theta_cls = "CONTOUR" if a.k == "Contour" else "ORIENTATION"
            theta_a = (slot_value(a.sigma, theta_cls, theta_name,
                                  default=0.0) + 0.5) * math.pi / n_theta
            stop = False
            for j in range(i + 1, len(orient_dets)):
                b = orient_dets[j]
                pb = anchor_positions.get(b.alpha)
                if pb is None or b not in node_ids:
                    continue
                v = node_ids[b]
                if u == v:
                    continue
                pairs_seen += 1
                if pairs_seen > max_pairs:
                    stop = True
                    break
                dist = _pos_dist(pa, pb)
                if dist <= 1e-6 or dist > 2.5 * R:
                    continue
                theta_b_name = "mean_theta" if b.k == "Contour" else "theta"
                theta_b_cls = "CONTOUR" if b.k == "Contour" else "ORIENTATION"
                theta_b = (slot_value(b.sigma, theta_b_cls, theta_b_name,
                                       default=0.0) + 0.5) * math.pi / n_theta

                dx = pb[0] - pa[0]
                dy = pb[1] - pa[1]
                phi = math.atan2(dy, dx) % math.pi

                al_t = _axial_wrap(theta_a - phi)
                al_h = _axial_wrap(theta_b - phi)
                err = abs(_axial_wrap(al_t + al_h))
                if err > tol:
                    continue
                signal = math.exp(-dist / R) * (1.0 - err / tol)
                # [P2] THE NAMED WRITER: the co-circular DEVIATION in
                # RADIANS and the separation in WAVELENGTHS, quantised by
                # the unit table.  (The old code binned by hand with the
                # same conventions, but addressed the slots by position.)
                err_rad = err
                dist_w = dist / max(extraction_scale, 1e-6)
                src, dst = (u, v) if u < v else (v, u)
                ep = _edge_params(
                    "EDGE_COCIRCULAR",
                    {"cocirc_asym": err_rad, "dist": dist_w},
                    sign=1,
                    confidence=float(clamp(signal, 0.0, 1.0)),
                )
                candidates.append((src, dst, signal, (src, dst), ep))
            if stop:
                break
        self._process_channel(EdgeType.COCIRCULAR, candidates)

    def _run_symmetry(
        self,
        dets: Tuple[Detection, ...],
        node_ids: Dict[Detection, int],
        anchor_positions: Optional[Dict[Any, Tuple[float, ...]]],
    ) -> None:
        """Pairwise symmetry between mid-level structures (CurveComplex / Part). Undirected."""
        cc_dets = sorted(
            [d for d in dets if d.k in ("CurveComplex", "Part")], key=_detection_sort_key
        )
        if len(cc_dets) < 2 or not anchor_positions:
            return
        candidates: List[Tuple[int, int, float, Tuple[int, int], EdgeParams]] = []
        r_scale = max(self._step_extraction_scale, 1.0)
        for i in range(len(cc_dets)):
            a = cc_dets[i]
            pa = anchor_positions.get(a.alpha)
            if pa is None:
                continue
            u = node_ids.get(a)
            if u is None:
                continue
            for j in range(i + 1, len(cc_dets)):
                b = cc_dets[j]
                pb = anchor_positions.get(b.alpha)
                if pb is None:
                    continue
                v = node_ids.get(b)
                if v is None or u == v:
                    continue
                dist = _pos_dist(pa, pb)
                if dist > 4.0 * r_scale:
                    continue
                dx = pb[0] - pa[0]
                dy = pb[1] - pa[1]
                mid_theta = (math.atan2(dy, dx) + math.pi / 2.0) % math.pi
                sig_sim = a.sigma.similarity(b.sigma)
                if sig_sim < 0.5:
                    continue
                signal = sig_sim * math.exp(-dist / (2.0 * r_scale))
                # [P2] THE NAMED WRITER: the SYMMETRY SCORE as a fraction,
                # the symmetry AXIS as an AXIAL ANGLE in RADIANS and the
                # separation in WAVELENGTHS.
                dist_w = dist / max(r_scale, 1e-6)
                src, dst = (u, v) if u < v else (v, u)
                ep = _edge_params(
                    "EDGE_SYMMETRY",
                    {"sym_score": sig_sim, "axis_angle": mid_theta,
                     "dist": dist_w},
                    sign=1,
                    confidence=float(clamp(signal, 0.0, 1.0)),
                )
                candidates.append((src, dst, signal, (src, dst), ep))
        self._process_channel(EdgeType.SYMMETRY, candidates)

    def _run_amodal_link(
        self,
        dets: Tuple[Detection, ...],
        node_ids: Dict[Detection, int],
        anchor_positions: Optional[Dict[Any, Tuple[float, ...]]],
    ) -> None:
        """V2 amodal completion: link collinear Contour segments separated
        across an occluding gap (Kellman & Shipley 1991 relatability).
        Undirected.

        Two gates, both at pi/8: the two segments must agree with each
        other, AND each must agree with the CHORD joining them. The second
        gate excludes a parallel but laterally-offset pair, which is
        collinear by delta-theta yet not relatable across the gap.
        """
        contour_dets = sorted(
            [d for d in dets if d.k == "Contour"], key=_detection_sort_key
        )
        if len(contour_dets) < 2 or not anchor_positions:
            return
        r_scale = max(self._step_extraction_scale, 1.0)
        # Draft geometry from the Pass 18 spec, kept verbatim: the gap band is
        # KAPPA*2.5*r_scale and the collinearity tolerance pi/8. An earlier
        # pass narrowed these to 1.5*KAPPA / pi/12 (72 links instead of 188),
        # but measurement showed the ETH-80 regression it was meant to fix is
        # not volumetric -- 15 links and 188 links collapse the cow
        # representative identically -- so the tightening bought nothing and
        # the spec values are restored rather than silently benchmark-fitted.
        max_gap = P.KAPPA_CONTOUR_CONTINUITY * 2.5 * r_scale
        if max_gap <= 0.0:
            return
        tol = math.pi / 8.0
        candidates: List[
            Tuple[int, int, float, Tuple[int, int], EdgeParams]
        ] = []
        for i in range(len(contour_dets)):
            a = contour_dets[i]
            qa = a.sigma.quantized
            if len(qa) < 2:
                continue
            pa = anchor_positions.get(a.alpha)
            if pa is None or a not in node_ids:
                continue
            u = node_ids[a]
            th_a = (qa[1] + 0.5) * math.pi / P.ORIENTATION_N_THETA
            for j in range(i + 1, len(contour_dets)):
                b = contour_dets[j]
                qb = b.sigma.quantized
                if len(qb) < 2:
                    continue
                pb = anchor_positions.get(b.alpha)
                if pb is None or b not in node_ids:
                    continue
                v = node_ids[b]
                if u == v:
                    continue
                dist = _pos_dist(pa, pb)
                if dist < r_scale or dist > max_gap:
                    continue
                th_b = (qb[1] + 0.5) * math.pi / P.ORIENTATION_N_THETA
                d_theta = abs(th_a - th_b)
                d_theta = min(d_theta, math.pi - d_theta)
                if d_theta > tol:
                    continue
                chord = math.atan2(pb[1] - pa[1], pb[0] - pa[0]) % math.pi
                err_a = min(abs(th_a - chord), math.pi - abs(th_a - chord))
                err_b = min(abs(th_b - chord), math.pi - abs(th_b - chord))
                if err_a > tol or err_b > tol:
                    continue
                signal = math.exp(-dist / max_gap) * (
                    1.0 - (err_a + err_b) / (2.0 * tol)
                )
                # [P2] THE NAMED WRITER: the amodal GAP in WAVELENGTHS and
                # the mean collinearity ERROR in RADIANS.
                gap_w = dist / max(r_scale, 1e-6)
                err_rad = (err_a + err_b) / 2.0
                src, dst = (u, v) if u < v else (v, u)
                ep = _edge_params(
                    "EDGE_AMODAL_LINK",
                    {"gap": gap_w, "error": err_rad},
                    sign=1,
                    confidence=float(clamp(signal, 0.0, 1.0)),
                )
                candidates.append((src, dst, signal, (src, dst), ep))
        self._process_channel(EdgeType.AMODAL_LINK, candidates)

    def _run_shared_boundary(
        self,
        dets: Tuple[Detection, ...],
        node_ids: Dict[Detection, int],
        anchor_positions: Optional[Dict[Any, Tuple[float, ...]]],
    ) -> None:
        """V2 surface adjacency: bind Surface patches close enough to share
        an interface (Cortical surface representations bind adjacent regions
        across a shared boundary). Undirected.

        The congruence slot reports whether the two patches disagree on the
        field at index 6 (-> bin 7.0) or agree/are silent (-> bin 0.0),
        so opposite-handed neighbours are still linked but distinguishable.
        A short signature reads as neutral rather than raising.
        """
        surf_dets = sorted(
            [d for d in dets if d.k == "Surface"], key=_detection_sort_key
        )
        if len(surf_dets) < 2 or not anchor_positions:
            return
        r_scale = max(self._step_extraction_scale, 1.0)
        candidates: List[
            Tuple[int, int, float, Tuple[int, int], EdgeParams]
        ] = []
        for i in range(len(surf_dets)):
            a = surf_dets[i]
            pa = anchor_positions.get(a.alpha)
            if pa is None or a not in node_ids:
                continue
            u = node_ids[a]
            for j in range(i + 1, len(surf_dets)):
                b = surf_dets[j]
                pb = anchor_positions.get(b.alpha)
                if pb is None or b not in node_ids:
                    continue
                v = node_ids[b]
                if u == v:
                    continue
                dist = _pos_dist(pa, pb)
                if dist > 2.5 * r_scale:
                    continue
                signal = math.exp(-dist / (2.0 * r_scale))
                # [P2] THE NAMED WRITER: the separation in WAVELENGTHS and
                # the CONGRUENCE as a categorical lookup key (the patches
                # either agree on the field or they do not; there is no
                # graded quantity behind that statement, so none is
                # invented).
                dist_w = dist / max(r_scale, 1e-6)
                qa, qb = a.sigma.quantized, b.sigma.quantized
                congruence = (
                    7.0
                    if len(qa) > 6 and len(qb) > 6
                    and abs(qa[6] - qb[6]) > 0.5
                    else 0.0
                )
                src, dst = (u, v) if u < v else (v, u)
                ep = _edge_params(
                    "EDGE_SHARED_BOUNDARY",
                    {"dist": dist_w, "congruence": congruence},
                    sign=1,
                    confidence=float(clamp(signal, 0.0, 1.0)),
                )
                candidates.append((src, dst, signal, (src, dst), ep))
        self._process_channel(EdgeType.SHARED_BOUNDARY, candidates)

    # ======================================================================
    # [Pass 23 round 2a] V4 non-Cartesian + surface relational edges
    # (values 42-44). Biology basis:
    #   CONCENTRICITY (Gallant et al. 1996; Hedge & Van Essen 2007):
    #     V4 cells selective for concentric / polar-coordinate nesting.
    #     Directed inner -> outer (part-whole containment direction).
    #   TEXTURE_BOUNDARY_EDGE (Landy & Graham 2003):
    #     V2 second-order texture segregation across adjacent regions.
    #     Undirected: the boundary belongs to neither region alone.
    # NOTE: CurveComplex is a SHAPE_ARRANGEMENT bundle (arity 24); its slots
    # are (n_contours, n_junctions, extent, closure, 8 template bins,
    # 12 template-population bins), NOT the CONTOUR summary geometry. Only
    # Contour carries length/circ-var/mean-orientation/closure, so the
    # contour-summary slot readers (_get_orientation_theta) apply to Contour
    # alone. CurveComplex enters concentricity via its member anchors
    # (_contour_point_geometry), which is slot-independent.
    # ======================================================================

    def _run_concentricity(
        self,
        dets: Tuple[Detection, ...],
        node_ids: Dict[Detection, int],
        anchor_positions: Optional[Dict[Any, Tuple[float, ...]]],
    ) -> None:
        """V4 concentric: nested contours sharing a polar origin.

        Directed: inner -> outer.
        """
        if not anchor_positions:
            return
        r_scale = max(self._step_extraction_scale, 1.0)

        contour_info = []
        for d in dets:
            if d.k not in ("Contour", "CurveComplex"):
                continue
            pos = anchor_positions.get(d.alpha)
            if pos is None:
                continue
            nid = node_ids.get(d)
            if nid is None:
                continue
            geom = _contour_point_geometry(d, anchor_positions)
            if geom is None:
                continue
            r, ang_ext, ecc = geom
            contour_info.append((d, pos, r, ang_ext, ecc, nid))

        if len(contour_info) < 2:
            return

        candidates: List[Tuple[int, int, float, Tuple[int, int], EdgeParams]] = []
        for i in range(len(contour_info)):
            d_a, pos_a, r_a, ext_a, ecc_a, nid_a = contour_info[i]
            for j in range(i + 1, len(contour_info)):
                d_b, pos_b, r_b, ext_b, ecc_b, nid_b = contour_info[j]
                if nid_a == nid_b:
                    continue

                dist = _pos_dist(pos_a, pos_b)
                r_outer = max(r_a, r_b)
                r_inner = min(r_a, r_b)
                if dist > 1.5 * r_outer:
                    continue

                ratio = r_outer / max(r_inner, 1e-6)
                if ratio < 1.15 or ratio > 10.0:
                    continue

                log_ratio = math.log2(max(ratio, 1.0))
                radial_bin = float(clamp(int(round(log_ratio * 2.0)), 0, 7))

                norm_offset = dist / max(r_outer, 1e-6)
                offset_bin = float(clamp(int(round(norm_offset * 7.0)), 0, 7))

                min_extent = min(ext_a, ext_b)
                extent_bin = float(clamp(int(round(min_extent * 7.0)), 0, 7))

                ecc_diff = abs(ecc_a - ecc_b)
                ecc_agree_bin = float(clamp(int(round(ecc_diff * 7.0)), 0, 7))

                offset_penalty = math.exp(-3.0 * norm_offset)
                signal = offset_penalty * min_extent * (1.0 - ecc_diff)
                if signal <= 0.01:
                    continue

                # Directed: inner -> outer
                if r_a < r_b:
                    src, dst = nid_a, nid_b
                else:
                    src, dst = nid_b, nid_a

                # [P2] THE NAMED WRITER: the RADIAL RATIO as a log2
                # ratio, the NORMALISED OFFSET and the MINIMUM EXTENT as
                # fractions, and the ECCENTRICITY DIFFERENCE as a fraction.
                ep = _edge_params(
                    "EDGE_CONCENTRICITY",
                    {"radial_ratio": log_ratio, "offset": norm_offset,
                     "extent": min_extent, "ecc_difference": ecc_diff},
                    sign=1,
                    confidence=float(clamp(signal, 0.0, 1.0)),
                )
                candidates.append((src, dst, signal, (src, dst), ep))

        self._process_channel(EdgeType.CONCENTRICITY, candidates)

    # __P23_TEXTURE_BOUNDARY_ANCHOR__

    def _run_texture_boundary_edge(
        self,
        dets: Tuple[Detection, ...],
        node_ids: Dict[Detection, int],
        anchor_positions: Optional[Dict[Any, Tuple[float, ...]]],
    ) -> None:
        """V2 texture boundary: discriminant discontinuity across adjacent regions.

        Sources: SURFACE_PATCH (texture_richness q[4], lum q[1]),
        REGION_SHAPE (boundary-conformation tail: area/elong/sym/axis/fold),
        TEXTURE (roughness/isotropy), TextureBoundary lens (material-discontinuity 8-vector).
        Discriminant features: (texture_richness, luminance, hue, chroma).
        """
        if not anchor_positions:
            return
        r_scale = max(self._step_extraction_scale, 1.0)
        n_hue = 16

        region_info = []
        for d in dets:
            if d.k not in ("Surface", "RegionShape", "Texture", "TextureBoundary"):
                continue
            pos = anchor_positions.get(d.alpha)
            if pos is None:
                continue
            nid = node_ids.get(d)
            if nid is None:
                continue
            q = d.sigma.quantized
            feat: Optional[Tuple[float, float, float, float]] = None
            if d.k == "Surface" and len(q) >= 5:
                # (shape, lum, hue_circ, sat, tex_rich, grad_mag, grad_side)
                feat = (float(q[4]), float(q[1]), float(q[2] or 0.0), float(q[3] or 0.0))
            elif d.k == "TextureBoundary" and len(q) >= 6:
                # (frac, dth, dlogsf, log2energy, tex_dist, lum_dist, conc_a, conc_b)
                feat = (
                    float(q[4]) * 7.0, float(q[5]) * 14.0,
                    float(q[1]) * 16.0 / max(float(P.ORIENTATION_N_THETA), 1.0), 0.0,
                )
            elif d.k == "Texture" and len(q) >= 2:
                # (roughness, isotropy)
                feat = (float(q[0]) * 7.0 / 8.0, 0.0, 0.0, float(q[1]) * 7.0 / 8.0)
            elif d.k == "RegionShape" and len(q) >= 5:
                # boundary-conformation tail: (..., area, elong, sym, axis, fold)
                # [F3] q[-3]/q[-2] are the SYMMETRY score and the symmetry
                # AXIS (the binder was reading symmetry as texture).
                feat = (float(q[-3]) * 7.0, 0.0, float(q[-2]) * 14.0, 0.0)
            if feat is None:
                continue
            region_info.append((pos, nid, feat))

        if len(region_info) < 2:
            return

        candidates: List[Tuple[int, int, float, Tuple[int, int], EdgeParams]] = []
        for i in range(len(region_info)):
            pos_a, nid_a, fa = region_info[i]
            for j in range(i + 1, len(region_info)):
                pos_b, nid_b, fb = region_info[j]
                if nid_a == nid_b:
                    continue
                dist = _pos_dist(pos_a, pos_b)
                if dist > 2.5 * r_scale:
                    continue
                # Hue is circular on 16: fold to [0, 8] shortest arc
                hue_diff = abs(fa[2] - fb[2]) % n_hue
                hue_diff = min(hue_diff, n_hue - hue_diff)
                d_tex = abs(fa[0] - fb[0])
                d_lum = abs(fa[1] - fb[1])
                d_chroma = abs(fa[3] - fb[3])
                # Pairwise means for normalisation (guard zero-mass pairs)
                m_tex = 0.5 * (abs(fa[0]) + abs(fb[0]))
                m_lum = 0.5 * (abs(fa[1]) + abs(fb[1])) + 1.0
                m_chroma = 0.5 * (abs(fa[3]) + abs(fb[3])) + 1.0
                tex_bin = float(min(7, max(0, int(round(d_tex / (m_tex + 1e-6) * 7.0)))))
                lum_bin = float(min(7, max(0, int(round(d_lum / m_lum * 7.0)))))
                hue_bin = float(min(7, max(0, int(round(hue_diff / 8.0 * 7.0)))))
                chroma_bin = float(min(7, max(0, int(round(d_chroma / m_chroma * 7.0)))))
                raw_salience = (tex_bin + lum_bin + hue_bin + chroma_bin) / 28.0
                salience = raw_salience * math.exp(-dist / (1.5 * r_scale))
                if salience <= 0.01:
                    continue
                src, dst = (nid_a, nid_b) if nid_a < nid_b else (nid_b, nid_a)
                # [P2] THE NAMED WRITER: the texture, luminance and
                # chroma differences as FRACTIONS and the hue difference as
                # an ANGLE in RADIANS (the shortest arc on the 16-cycle).
                ep = _edge_params(
                    "EDGE_TEXTURE_BOUNDARY",
                    {"tex_diff": d_tex / (m_tex + 1e-6),
                     "lum_diff": d_lum / m_lum,
                     "hue_diff": hue_diff * (math.pi / 8.0),
                     "chroma_diff": d_chroma / m_chroma},
                    sign=1,
                    confidence=float(clamp(salience, 0.0, 1.0)),
                )
                candidates.append((src, dst, salience, (src, dst), ep))

        self._process_channel(EdgeType.TEXTURE_BOUNDARY_EDGE, candidates)

    # __P23_SURFACE_FOLD_ANCHOR__

    def _run_surface_fold(
        self,
        dets: Tuple[Detection, ...],
        node_ids: Dict[Detection, int],
        anchor_positions: Optional[Dict[Any, Tuple[float, ...]]],
    ) -> None:
        """V4/CIP surface fold: dihedral angle between oriented surfaces.

        Directed: convex side -> concave side. Normal comes from the
        coarse two-tilt SURFACE_PATCH cue: dihedral is a proxy, not a
        measured fold angle; magnitude gate + sharpness carry the evidence.
        """
        if not anchor_positions:
            return
        r_scale = max(self._step_extraction_scale, 1.0)

        surf_info = []
        for d in dets:
            if d.k != "Surface":
                continue
            pos = anchor_positions.get(d.alpha)
            if pos is None:
                continue
            nid = node_ids.get(d)
            if nid is None:
                continue
            normal = _surface_normal(d)
            if normal is None:
                continue
            surf_info.append((pos, nid, normal))

        if len(surf_info) < 2:
            return

        candidates: List[Tuple[int, int, float, Tuple[int, int], EdgeParams]] = []
        for i in range(len(surf_info)):
            pos_a, nid_a, n_a = surf_info[i]
            for j in range(i + 1, len(surf_info)):
                pos_b, nid_b, n_b = surf_info[j]
                if nid_a == nid_b:
                    continue

                dist = _pos_dist(pos_a, pos_b)
                if dist > 2.5 * r_scale:
                    continue

                dot = sum(n_a[k] * n_b[k] for k in range(3))
                dot = clamp(dot, -1.0, 1.0)
                dihedral = math.acos(dot)

                if dihedral < math.pi / 12.0:
                    continue

                dihedral_bin = float(clamp(int(round(dihedral / math.pi * 11.0)), 0, 11))

                fold_theta = _safe_atan2(pos_b[1] - pos_a[1], pos_b[0] - pos_a[0]) % math.pi
                fold_axis_bin = float(int(round(fold_theta / math.pi * 12.0)) % 12)

                conn = tuple(pos_b[k] - pos_a[k] for k in range(min(3, len(pos_a), len(pos_b))))
                if len(conn) < 3:
                    conn = conn + (0.0,) * (3 - len(conn))
                conn_len = math.sqrt(sum(c * c for c in conn))
                if conn_len > 1e-9:
                    conn_unit = tuple(c / conn_len for c in conn)
                    dn = tuple(n_a[k] - n_b[k] for k in range(3))
                    convex_score = sum(dn[k] * conn_unit[k] for k in range(3))
                    if convex_score > 0.1:
                        convexity_bin = 2.0
                    elif convex_score < -0.1:
                        convexity_bin = 0.0
                    else:
                        convexity_bin = 1.0
                else:
                    convexity_bin = 1.0

                sharpness = dihedral / max(dist, 1e-6)
                norm_sharpness = sharpness * r_scale / (math.pi / 2.0)
                sharpness_bin = float(clamp(int(round(norm_sharpness * 7.0)), 0, 7))

                signal = (dihedral / math.pi) * math.exp(-dist / (1.5 * r_scale))
                if signal <= 0.01:
                    continue

                if convexity_bin >= 1.5:
                    src, dst = nid_a, nid_b
                elif convexity_bin <= 0.5:
                    src, dst = nid_b, nid_a
                else:
                    src, dst = (nid_a, nid_b) if nid_a < nid_b else (nid_b, nid_a)

                # [P2] THE NAMED WRITER: the DIHEDRAL ANGLE in RADIANS,
                # the FOLD AXIS as an axial angle in RADIANS, the convexity
                # as a categorical lookup key and the sharpness as a
                # fraction.
                ep = _edge_params(
                    "EDGE_SURFACE_FOLD",
                    {"dihedral": dihedral, "fold_axis": fold_theta,
                     "convexity": convexity_bin,
                     "sharpness": max(0.0, min(1.0, norm_sharpness))},
                    sign=1,
                    confidence=float(clamp(signal, 0.0, 1.0)),
                )
                candidates.append((src, dst, signal, (src, dst), ep))

        self._process_channel(EdgeType.SURFACE_FOLD, candidates)

    # __P23_SPIRAL_ANCHOR__

    def _run_spirality(
        self,
        dets: Tuple[Detection, ...],
        node_ids: Dict[Detection, int],
        anchor_positions: Optional[Dict[Any, Tuple[float, ...]]],
    ) -> None:
        """V4 spiral: logarithmic-spiral contour pattern (Gallant et al. 1996).

        V4 responses to spiral patterns are documented at the single-cell level
        (passband tuning to polar/spiral form); the LOGARITHMIC-spiral claim is
        an approximation -- the emitter detects locally constant pitch
        (orientation-vs-radius angle), which a log-spiral satisfies but which
        also admits near-spiral configurations. Undirected.

        Pitch gate: excludes pure radial (~0) and pure circular (~pi/2) pairs;
        only genuinely oblique orientation-vs-radius configurations emit.

        Biology note (audit): the draft cited V4 non-Cartesian work without a
        specific spiral paper. Gallant et al. 1993/1996 (Science) reports V4
        selectivity for polar, hyperbolic, and Cartesian gratings including
        spiral-like tuning. That is cited here; no stronger claim is made.
        """
        if not anchor_positions:
            return
        r_scale = max(self._step_extraction_scale, 1.0)
        n_theta = P.ORIENTATION_N_THETA

        spiral_info = []
        for d in dets:
            if d.k not in ("Contour", "CurveComplex", "Orientation", "OrientationPol"):
                continue
            pos = anchor_positions.get(d.alpha)
            if pos is None:
                continue
            nid = node_ids.get(d)
            if nid is None:
                continue
            theta = _get_orientation_theta(d, n_theta)
            if theta is None:
                continue
            spiral_info.append((pos, nid, theta))

        if len(spiral_info) < 2:
            return

        # Pitch passband: exclude radial (~0) and circular (~pi/2) extremes.
        min_pitch = math.pi / 10.0
        max_pitch = math.pi / 2.0 - math.pi / 10.0

        candidates: List[Tuple[int, int, float, Tuple[int, int], EdgeParams]] = []
        for i in range(len(spiral_info)):
            pos_a, nid_a, theta_a = spiral_info[i]
            for j in range(i + 1, len(spiral_info)):
                pos_b, nid_b, theta_b = spiral_info[j]
                if nid_a == nid_b:
                    continue

                dist = _pos_dist(pos_a, pos_b)
                if dist > 4.0 * r_scale or dist < 0.3 * r_scale:
                    continue

                # Midpoint as the provisional spiral centre (pairwise approx).
                hub_x = (pos_a[0] + pos_b[0]) / 2.0
                hub_y = (pos_a[1] + pos_b[1]) / 2.0

                rad_a = _safe_atan2(pos_a[1] - hub_y, pos_a[0] - hub_x)
                rad_b = _safe_atan2(pos_b[1] - hub_y, pos_b[0] - hub_x)

                # Orientation is AXIAL (mod pi): fold the radius direction too.
                pitch_a = _angular_diff_pi(theta_a, rad_a % math.pi)
                pitch_b = _angular_diff_pi(theta_b, rad_b % math.pi)

                if not (min_pitch < pitch_a < max_pitch and min_pitch < pitch_b < max_pitch):
                    continue

                pitch_diff = abs(pitch_a - pitch_b)
                if pitch_diff > math.pi / 6.0:
                    continue

                mean_pitch = (pitch_a + pitch_b) / 2.0
                # Axial pitch in [0, pi/2] binned LINEARLY 0..11 (range 12.0).
                pitch_bin = float(clamp(int(round(mean_pitch / (math.pi / 2.0) * 11.0)), 0, 11))

                consistency = 1.0 - pitch_diff / (math.pi / 6.0)
                consistency_bin = float(clamp(int(round(consistency * 7.0)), 0, 7))

                # Chirality from axial cross products (orientation mod pi):
                # sign of sin(theta - rad) per element, averaged. Axial-safe
                # because both theta and rad are folded to the same space.
                cross_a = math.cos(rad_a) * math.sin(theta_a) - math.sin(rad_a) * math.cos(theta_a)
                cross_b = math.cos(rad_b) * math.sin(theta_b) - math.sin(rad_b) * math.cos(theta_b)
                mean_cross = (cross_a + cross_b) / 2.0
                if mean_cross > 0.15:
                    chirality_bin = 2.0
                elif mean_cross < -0.15:
                    chirality_bin = 0.0
                else:
                    chirality_bin = 1.0

                # Log-spiral expansion: d ln r / d phi over the pair's arc.
                r_a = math.sqrt((pos_a[0] - hub_x) ** 2 + (pos_a[1] - hub_y) ** 2)
                r_b = math.sqrt((pos_b[0] - hub_x) ** 2 + (pos_b[1] - hub_y) ** 2)
                angular_sep = _angular_diff(rad_a, rad_b)
                if angular_sep > 0.1:
                    expansion = abs(math.log(max(r_a, 0.01) / max(r_b, 0.01))) / angular_sep
                else:
                    expansion = 0.0
                expansion_bin = float(clamp(int(round(expansion * 3.0)), 0, 7))

                signal = consistency * math.exp(-dist / (3.0 * r_scale))
                if signal <= 0.01:
                    continue

                src, dst = (nid_a, nid_b) if nid_a < nid_b else (nid_b, nid_a)

                # [P2] THE NAMED WRITER: the mean PITCH in RADIANS (axial,
                # it folds), the consistency and the expansion as fractions
                # and the chirality as a categorical lookup key.
                ep = _edge_params(
                    "EDGE_SPIRALITY",
                    {"pitch": mean_pitch, "consistency": consistency,
                     "chirality": chirality_bin, "expansion": expansion},
                    sign=1,
                    confidence=float(clamp(signal, 0.0, 1.0)),
                )
                candidates.append((src, dst, signal, (src, dst), ep))

        self._process_channel(EdgeType.SPIRALITY, candidates)

    def _run_radiality(
        self,
        dets: Tuple[Detection, ...],
        node_ids: Dict[Detection, int],
        anchor_positions: Optional[Dict[Any, Tuple[float, ...]]],
    ) -> None:
        """V4 radial: orientations converging on a common hub (Gallant et al. 1996). Undirected."""
        if not anchor_positions:
            return
        r_scale = max(self._step_extraction_scale, 1.0)
        n_theta = P.ORIENTATION_N_THETA

        orient_info = []
        for d in dets:
            theta = _get_orientation_theta(d, n_theta)
            if theta is None:
                continue
            pos = anchor_positions.get(d.alpha)
            if pos is None:
                continue
            nid = node_ids.get(d)
            if nid is None:
                continue
            orient_info.append((pos, nid, theta))

        if len(orient_info) < 2:
            return

        candidates: List[Tuple[int, int, float, Tuple[int, int], EdgeParams]] = []
        tol = math.pi / 5.0

        for i in range(len(orient_info)):
            pos_a, nid_a, theta_a = orient_info[i]
            for j in range(i + 1, len(orient_info)):
                pos_b, nid_b, theta_b = orient_info[j]
                if nid_a == nid_b:
                    continue

                dist = _pos_dist(pos_a, pos_b)
                if dist > 4.0 * r_scale or dist < 0.3 * r_scale:
                    continue

                ca, sa = math.cos(theta_a), math.sin(theta_a)
                cb, sb = math.cos(theta_b), math.sin(theta_b)

                det_cross = ca * sb - sa * cb
                if abs(det_cross) < 1e-6:
                    continue

                dx = pos_b[0] - pos_a[0]
                dy = pos_b[1] - pos_a[1]
                t = (dx * sb - dy * cb) / det_cross

                hub_x = pos_a[0] + t * ca
                hub_y = pos_a[1] + t * sa

                hub_dist_a = math.sqrt((hub_x - pos_a[0]) ** 2 + (hub_y - pos_a[1]) ** 2)
                hub_dist_b = math.sqrt((hub_x - pos_b[0]) ** 2 + (hub_y - pos_b[1]) ** 2)
                max_hub_dist = 6.0 * r_scale
                if hub_dist_a > max_hub_dist or hub_dist_b > max_hub_dist:
                    continue
                if hub_dist_a < 0.1 * r_scale and hub_dist_b < 0.1 * r_scale:
                    continue

                ang_a_to_hub = _safe_atan2(hub_y - pos_a[1], hub_x - pos_a[0]) % math.pi
                ang_b_to_hub = _safe_atan2(hub_y - pos_b[1], hub_x - pos_b[0]) % math.pi
                dev_a = _angular_diff_pi(theta_a, ang_a_to_hub)
                dev_b = _angular_diff_pi(theta_b, ang_b_to_hub)

                if dev_a > tol or dev_b > tol:
                    continue

                mid_x = (pos_a[0] + pos_b[0]) / 2.0
                mid_y = (pos_a[1] + pos_b[1]) / 2.0
                hub_bearing = _safe_atan2(hub_y - mid_y, hub_x - mid_x) % (2.0 * math.pi)
                hub_bearing_bin = float(int(round(hub_bearing / (2.0 * math.pi) * 24.0)) % 24)

                mean_dev = (dev_a + dev_b) / 2.0
                align_bin = float(clamp(int(round(mean_dev / (math.pi / 2.0) * 7.0)), 0, 7))

                mean_hub_dist = (hub_dist_a + hub_dist_b) / 2.0
                hub_ratio = mean_hub_dist / max(dist, 1e-6)
                hub_ratio_bin = float(clamp(int(round(hub_ratio)), 0, 7))

                spoke_angle = abs(
                    _safe_atan2(pos_a[1] - hub_y, pos_a[0] - hub_x) -
                    _safe_atan2(pos_b[1] - hub_y, pos_b[0] - hub_x)
                )
                spoke_angle = min(spoke_angle, 2.0 * math.pi - spoke_angle)
                # Axial spoke separation on [0, pi]: 12 LINEAR bins (range 12.0).
                # NOT circular on 24: axial quantities fold, they do not wrap.
                spoke_bin = float(clamp(int(round(spoke_angle / math.pi * 11.0)), 0, 11))

                alignment_quality = 1.0 - mean_dev / tol
                distance_factor = math.exp(-dist / (3.0 * r_scale))
                signal = alignment_quality * distance_factor
                if signal <= 0.01:
                    continue

                src, dst = (nid_a, nid_b) if nid_a < nid_b else (nid_b, nid_a)

                # [P2] THE NAMED WRITER: the hub BEARING in RADIANS, the
                # mean AXIAL DEVIATION in RADIANS, the hub ratio and the
                # spoke separation (axial, it folds) in RADIANS.
                ep = _edge_params(
                    "EDGE_RADIALITY",
                    {"hub_bearing": hub_bearing, "alignment": mean_dev,
                     "hub_ratio": hub_ratio, "spoke_separation": spoke_angle},
                    sign=1,
                    confidence=float(clamp(signal, 0.0, 1.0)),
                )
                candidates.append((src, dst, signal, (src, dst), ep))

        self._process_channel(EdgeType.RADIALITY, candidates)

    # __P23_RADIAL_SPIRAL_ANCHOR__

    def _run_color_opponent_border(
        self,
        dets: Tuple[Detection, ...],
        node_ids: Dict[Detection, int],
        anchor_positions: Optional[Dict[Any, Tuple[float, ...]]],
    ) -> None:
        """[Pass 23] V1/V2 opponent-channel chromatic border. Undirected.

        Biology: V1 colour-opponent neurons (red-green, blue-yellow) respond
        to chromatic borders; V2 border-ownership cells use those borders to
        assign an edge to one side. The Color signature is ALREADY opponent
        encoded -- slot 0 = luminance, slot 1 = RG, slot 2 = BY. No RGB
        reconstruction is done here.

        Audit fix vs. the blueprint draft: the draft binned the opponent axis
        into 12 bins against a declared 24-cycle. Under a 24-cycle the wrap
        distance is min(d, 24-d), so bins 0 and 11 -- adjacent on the circle
        -- would score as maximally different. Binned to 24 to match.
        """
        if not anchor_positions:
            return
        r_scale = max(self._step_extraction_scale, 1.0)

        color_info = []
        for d in dets:
            if d.k != "Color":
                continue
            q = d.sigma.quantized
            if len(q) < 3:
                continue
            pos = anchor_positions.get(d.alpha)
            if pos is None:
                continue
            nid = node_ids.get(d)
            if nid is None:
                continue
            color_info.append((pos, nid, float(q[0]), float(q[1]), float(q[2])))

        if len(color_info) < 2:
            return

        # COLOR ranges: slot 1 (RG) spans 16 bins, slot 2 (BY) spans 12, so
        # sqrt(16^2 + 12^2) is the maximum opponent-axis difference.
        chrom_max = math.sqrt(16.0 ** 2 + 12.0 ** 2)

        candidates: List[Tuple[int, int, float, Tuple[int, int], EdgeParams]] = []
        for i in range(len(color_info)):
            pos_a, nid_a, lum_a, rg_a, by_a = color_info[i]
            for j in range(i + 1, len(color_info)):
                pos_b, nid_b, lum_b, rg_b, by_b = color_info[j]
                if nid_a == nid_b:
                    continue

                dist = _pos_dist(pos_a, pos_b)
                if dist > 2.5 * r_scale:
                    continue

                d_rg = rg_a - rg_b
                d_by = by_a - by_b
                d_lum = lum_a - lum_b

                chrom_mag = math.sqrt(d_rg * d_rg + d_by * d_by)
                lum_mag = abs(d_lum)
                if chrom_mag < 0.5:
                    continue

                opp_angle = _safe_atan2(d_by, d_rg) % (2.0 * math.pi)
                opp_angle_bin = float(
                    int(round(opp_angle / (2.0 * math.pi) * 24.0)) % 24
                )

                # Normalized: the draft used the raw magnitude, which
                # saturates the 8-bin slot at any real colour difference.
                contrast_bin = float(
                    clamp(int(round(chrom_mag / chrom_max * 7.0)), 0, 7)
                )

                total = math.sqrt(chrom_mag * chrom_mag + lum_mag * lum_mag)
                lum_indep = (chrom_mag / total) if total > 1e-6 else 0.0
                lum_indep_bin = float(clamp(int(round(lum_indep * 7.0)), 0, 7))

                extent = dist / (2.5 * r_scale)
                extent_bin = float(clamp(int(round(extent * 7.0)), 0, 7))

                proximity = math.exp(-dist / r_scale)
                signal = (
                    (chrom_mag / chrom_max)
                    * proximity
                    * (0.3 + 0.7 * lum_indep)
                )
                if signal <= 0.01:
                    continue

                src, dst = (nid_a, nid_b) if nid_a < nid_b else (nid_b, nid_a)
                # [P2] THE NAMED WRITER: the opponent ANGLE in RADIANS,
                # the normalised contrast, the luminance independence and
                # the extent as fractions.
                ep = _edge_params(
                    "EDGE_COLOR_OPPONENT_BORDER",
                    {
                        "opp_angle": opp_angle,
                        "contrast": chrom_mag / max(chrom_max, 1e-6),
                        "lum_indep": lum_indep,
                        "extent": extent,
                    },
                    sign=1,
                    confidence=float(clamp(signal, 0.0, 1.0)),
                )
                candidates.append((src, dst, signal, (src, dst), ep))

        self._process_channel(EdgeType.COLOR_OPPONENT_BORDER, candidates)

    def _run_motion_opposition(
        self,
        dets: Tuple[Detection, ...],
        node_ids: Dict[Detection, int],
        anchor_positions: Optional[Dict[Any, Tuple[float, ...]]],
        anchor_velocities: Optional[Dict[Any, Tuple[float, ...]]],
    ) -> None:
        """[Pass 23] MT common-fate contrast: velocity shear. sign = -1.

        Biology: MT direction-selective neurons respond to the common fate
        of contour elements, so two patches moving in different directions
        across the same interface are evidence of separate surfaces meeting
        (an occluding border).

        Audit fix vs. the blueprint draft: the draft binned the direction
        contrast and the boundary bearing into 12 bins against declared
        24-cycles -- the same wrap defect as the opponent axis. Both now 24.
        """
        if not anchor_velocities or not anchor_positions:
            return
        r_scale = max(self._step_extraction_scale, 1.0)

        vel_info = []
        for d in dets:
            vel = anchor_velocities.get(d.alpha)
            if vel is None or len(vel) < 2:
                continue
            pos = anchor_positions.get(d.alpha)
            if pos is None:
                continue
            nid = node_ids.get(d)
            if nid is None:
                continue
            speed = _velocity_speed(vel)
            if speed < 0.01:
                continue
            vel_info.append((pos, nid, speed, _velocity_direction(vel)))

        if len(vel_info) < 2:
            return

        candidates: List[Tuple[int, int, float, Tuple[int, int], EdgeParams]] = []
        for i in range(len(vel_info)):
            pos_a, nid_a, spd_a, dir_a = vel_info[i]
            for j in range(i + 1, len(vel_info)):
                pos_b, nid_b, spd_b, dir_b = vel_info[j]
                if nid_a == nid_b:
                    continue

                dist = _pos_dist(pos_a, pos_b)
                if dist > 3.0 * r_scale:
                    continue

                dir_diff = _angular_diff(dir_a, dir_b)
                if dir_diff < math.pi / 3.0:
                    continue

                dir_contrast_bin = float(
                    int(round(dir_diff / (2.0 * math.pi) * 24.0)) % 24
                )

                speed_rel = abs(spd_a - spd_b) / max(spd_a, spd_b, 1e-6)
                speed_contrast_bin = float(clamp(int(round(speed_rel * 7.0)), 0, 7))

                boundary = _safe_atan2(
                    pos_b[1] - pos_a[1], pos_b[0] - pos_a[0]
                ) % (2.0 * math.pi)
                boundary_bin = float(
                    int(round(boundary / (2.0 * math.pi) * 24.0)) % 24
                )

                mean_speed = (spd_a + spd_b) / 2.0
                speed_bin = float(
                    clamp(
                        int(round(mean_speed / (P.MOTION_SPEED_MAX + 1e-6) * 7.0)),
                        0,
                        7,
                    )
                )

                dir_factor = (dir_diff - math.pi / 3.0) / (math.pi - math.pi / 3.0)
                speed_factor = min(1.0, mean_speed / (P.MOTION_SPEED_MAX + 1e-6))
                proximity = math.exp(-dist / (2.0 * r_scale))
                signal = dir_factor * speed_factor * proximity
                if signal <= 0.01:
                    continue

                # Directed: fast -> slow, so the source is the surface whose
                # motion is not interrupted.
                if spd_a >= spd_b:
                    src, dst = nid_a, nid_b
                else:
                    src, dst = nid_b, nid_a

                # [P2] THE NAMED WRITER: the direction contrast and the
                # boundary angle in RADIANS, the speed contrast and the
                # mean speed as fractions.
                ep = _edge_params(
                    "EDGE_MOTION_OPPOSITION",
                    {
                        "dir_contrast": dir_diff,
                        "speed_contrast": speed_rel,
                        "boundary_angle": boundary,
                        "speed": mean_speed / max(P.MOTION_SPEED_MAX, 1e-6),
                    },
                    sign=-1,
                    confidence=float(clamp(signal, 0.0, 1.0)),
                )
                candidates.append((src, dst, signal, (src, dst), ep))

        self._process_channel(EdgeType.MOTION_OPPOSITION, candidates)

    def _run_cross_scale_identity(
        self,
        dets: Tuple[Detection, ...],
        node_ids: Dict[Detection, int],
        anchor_positions: Optional[Dict[Any, Tuple[float, ...]]],
    ) -> None:
        """[Pass 23] Cross-scale correspondence. Directed: coarser -> finer.

        Biology: multi-scale correspondence is a general property of visual
        cortex (V2/V4 pool across scales); ventral-stream scale invariance is
        a long-standing target of IT modelling. No specific cortical cell
        type is claimed beyond that: this is scale-invariance scaffolding.

        Audit fix vs. the blueprint draft: the draft derived `phase` from a
        cosine projection of the drift vector, which is not a phase at all.
        The slot now carries the fine feature's offset within the coarse
        cell -- the quantity that governs whether the two scales are in
        register -- binned to 24 against the declared 24-cycle.
        """
        if not anchor_positions:
            return
        r_scale = max(self._step_extraction_scale, 1.0)

        by_class: Dict[Tuple[str, str], List] = defaultdict(list)
        for d in dets:
            scale = _detection_scale(d)
            if scale <= 0.0:
                continue
            pos = anchor_positions.get(d.alpha)
            if pos is None:
                continue
            nid = node_ids.get(d)
            if nid is None:
                continue
            by_class[(d.k, getattr(d.sigma, "class_", d.k))].append(
                (pos, nid, scale, d)
            )

        candidates: List[Tuple[int, int, float, Tuple[int, int], EdgeParams]] = []
        for _key, group in by_class.items():
            if len(group) < 2:
                continue
            # Ascending scale: group[i] is the finer partner of group[j].
            group.sort(key=lambda g: (g[2], g[1]))

            for i in range(len(group)):
                pos_f, nid_f, scale_f, det_f = group[i]
                for j in range(i + 1, len(group)):
                    pos_c, nid_c, scale_c, det_c = group[j]
                    if nid_c == nid_f or scale_c == scale_f:
                        continue

                    scale_ratio = scale_c / max(scale_f, 1e-6)
                    if scale_ratio < 1.3 or scale_ratio > 16.0:
                        continue

                    drift = _pos_dist(pos_c, pos_f)
                    max_drift = 2.0 * scale_c * r_scale
                    if drift > max_drift:
                        continue

                    sig_sim = 0.5
                    try:
                        sig_sim = float(det_c.sigma.similarity(det_f.sigma))
                    except (TypeError, AttributeError):
                        pass
                    if sig_sim < 0.2:
                        continue

                    octaves = math.log2(max(scale_ratio, 1.0))
                    octave_bin = float(clamp(int(round(octaves)), 0, 7))

                    # Register phase: the fine feature's offset inside the
                    # coarse cell, in coarse-cell units.
                    phase = ((drift / max(scale_c, 1e-6)) % 1.0) * 2.0 * math.pi
                    phase_bin = float(
                        int(round(phase / (2.0 * math.pi) * 24.0)) % 24
                    )

                    norm_drift = drift / max(max_drift, 1e-6)
                    drift_bin = float(clamp(int(round(norm_drift * 7.0)), 0, 7))
                    sim_bin = float(clamp(int(round(sig_sim * 7.0)), 0, 7))

                    signal = (
                        sig_sim * math.exp(-norm_drift * 2.0)
                        * (1.0 / max(octaves, 0.5))
                    )
                    if signal <= 0.01:
                        continue

                    # [P2] THE NAMED WRITER: the OCTAVE RATIO as a log2
                    # ratio, the register PHASE in RADIANS, the normalised
                    # drift and the signature similarity as fractions.
                    ep = _edge_params(
                        "EDGE_CROSS_SCALE_IDENTITY",
                        {"octave_ratio": octaves, "phase_alignment": phase,
                         "drift": norm_drift, "similarity": sig_sim},
                        sign=1,
                        confidence=float(clamp(signal, 0.0, 1.0)),
                    )
                    candidates.append((nid_c, nid_f, signal, (nid_c, nid_f), ep))

        self._process_channel(EdgeType.CROSS_SCALE_IDENTITY, candidates)

    def _binocular_disparity_enabled(
        self, anchor_positions: Optional[Dict[Any, Tuple[float, ...]]]
    ) -> bool:
        """[ROUND 11 / O4] Gate for EDGE_BINOCULAR_DISPARITY.

        The channel reads the z component of an anchor position. That z is a
        scene/depth proxy, not a binocular disparity measurement, so the gate
        requires (a) an explicit opt-in flag ``BINOCULAR_DISPARITY_FROM_Z`` and
        (b) at least one genuinely 3D anchor. With the default flag the edge
        can NEVER be produced from scene z.
        """
        if not anchor_positions:
            return False
        if not bool(getattr(P, "BINOCULAR_DISPARITY_FROM_Z", False)):
            return False
        return any(
            len(_p) >= 3 and math.isfinite(float(_p[2]))
            for _p in anchor_positions.values()
        )

    def _run_binocular_disparity(
        self,
        dets: Tuple[Detection, ...],
        node_ids: Dict[Detection, int],
        anchor_positions: Optional[Dict[Any, Tuple[float, ...]]],
    ) -> None:
        """[Pass 23] 3D depth separation. Directed: nearer -> farther.

        Biology caveat: this reads stereo/structure-from-motion z already
        present on the anchor positions. It is NOT a disparity correlator and
        makes no claim about V1/V2 disparity neurons beyond the general fact
        that depth separation signals object boundaries. The channel name is
        kept (enum value 50 is reserved, never reused) but no specific
        disparity-cell mechanism is asserted.

        Audit fix vs. the blueprint draft: the draft's cell-type test
        (z_a - mean_z) * (z_b - mean_z) > 0 is NEVER true for two distinct z
        values around their own midpoint, so bins 2.0 and 3.0 were dead code.
        Replaced with a near/far-straddle classification: a disparity pair
        straddling the scene-depth midpoint is a figure-ground cue.
        """
        if not anchor_positions:
            return
        r_scale = max(self._step_extraction_scale, 1.0)

        depth_info = []
        for d in dets:
            pos = anchor_positions.get(d.alpha)
            if pos is None or len(pos) < 3:
                continue
            z = float(pos[2])
            if not math.isfinite(z):
                continue
            nid = node_ids.get(d)
            if nid is None:
                continue
            depth_info.append((pos, nid, z))

        if len(depth_info) < 2:
            return

        candidates: List[Tuple[int, int, float, Tuple[int, int], EdgeParams]] = []
        for i in range(len(depth_info)):
            pos_a, nid_a, z_a = depth_info[i]
            for j in range(i + 1, len(depth_info)):
                pos_b, nid_b, z_b = depth_info[j]
                if nid_a == nid_b:
                    continue

                d_xy = math.hypot(pos_a[0] - pos_b[0], pos_a[1] - pos_b[1])
                if d_xy > 3.0 * r_scale:
                    continue

                dz = abs(z_a - z_b)
                if dz < 0.005:
                    continue

                disp_gradient = (dz / d_xy) if d_xy > 1e-6 else 0.0
                if disp_gradient > 1.5:
                    continue

                disp_bin = float(clamp(int(round(dz / (0.3 * r_scale) * 7.0)), 0, 7))
                grad_bin = float(clamp(int(round(disp_gradient * 5.0)), 0, 7))

                # Near/far organization about the 0.5 scene-depth midpoint.
                near_a = z_a < 0.5
                near_b = z_b < 0.5
                if dz < 0.05 * r_scale:
                    cell_type_bin = 0.0     # negligible separation
                elif near_a != near_b:
                    cell_type_bin = 3.0     # straddles the depth midpoint
                elif near_a:
                    cell_type_bin = 1.0     # both in the near field
                else:
                    cell_type_bin = 2.0     # both in the far field

                if disp_gradient < 0.3:
                    conf = 1.0
                elif disp_gradient < 0.8:
                    conf = 1.0 - (disp_gradient - 0.3) / 0.5
                else:
                    conf = 0.1
                conf_bin = float(clamp(int(round(conf * 7.0)), 0, 7))

                signal = (
                    min(1.0, dz / (0.15 * r_scale))
                    * math.exp(-d_xy / (2.0 * r_scale))
                    * conf
                )
                if signal <= 0.01:
                    continue

                src, dst = (nid_a, nid_b) if z_a < z_b else (nid_b, nid_a)
                # [P2] THE NAMED WRITER: the disparity and the gradient in
                # the 3-D depth units, the near/far straddle as a
                # categorical lookup key and the confidence as a fraction.
                ep = _edge_params(
                    "EDGE_BINOCULAR_DISPARITY",
                    {"disparity": dz / (0.3 * r_scale),
                     "gradient": disp_gradient,
                     "depth_straddle": cell_type_bin,
                     "confidence": conf},
                    sign=1,
                    confidence=float(clamp(signal, 0.0, 1.0)),
                )
                candidates.append((src, dst, signal, (src, dst), ep))

        self._process_channel(EdgeType.BINOCULAR_DISPARITY, candidates)

    def _run_line_termination(
        self,
        dets: Tuple[Detection, ...],
        node_ids: Dict[Detection, int],
        anchor_positions: Optional[Dict[Any, Tuple[float, ...]]],
    ) -> None:
        """[Pass 23] End-stopped terminator -> parent segment. Directed.

        Biology: end-stopped hypercomplex cells respond to line ends and
        corners (Hubel & Wiesel; V2 end-stopping). The ENDSTOPPED signature
        slots are (theta_bin, sf_bin, ends, strength_bin) -- see
        EndStoppedLens in derived_lenses.py.

        Audit fix vs. the blueprint draft: the draft read slot 0 as
        "raw_sharpness" and slot 1 as "curvature". Those are the terminator's
        orientation and its spatial frequency; strength is slot 3. Sharpness
        now reads slot 3 (slot 2 as a 3-slot fallback), and curvature is
        derived from the adjacent parent orientations, which is what it
        physically means.

        Audit fix vs. the blueprint draft: the parent-orientation slot is
        AXIAL (mod pi), so it is binned to 12 with a declared 12-cycle and
        range 6.0. The draft declared a 24-cycle for a mod-pi quantity,
        which made adjacent axial values score as maximally different.
        """
        if not anchor_positions:
            return
        r_scale = max(self._step_extraction_scale, 1.0)

        term_info = []
        for d in dets:
            if d.k != "EndStopped" and not is_junction_kind(d.k):
                continue
            nid = node_ids.get(d)
            if nid is None:
                continue
            pos = anchor_positions.get(d.alpha)
            if pos is None:
                continue
            term_info.append((pos, nid, d.sigma.quantized))

        orient_info = []
        for d in dets:
            if d.k not in ("Orientation", "OrientationPol", "Contour"):
                continue
            pos = anchor_positions.get(d.alpha)
            if pos is None:
                continue
            nid = node_ids.get(d)
            if nid is None:
                continue
            theta = _get_orientation_theta(d, P.ORIENTATION_N_THETA)
            if theta is None and d.sigma.class_ in ("ORIENTATION", "ORIENTATION_POL"):
                q = d.sigma.quantized
                if len(q) >= 1:
                    theta = (float(q[0]) + 0.5) * math.pi / P.ORIENTATION_N_THETA
            if theta is None:
                continue
            orient_info.append((pos, nid, theta))

        if not term_info or not orient_info:
            return

        candidates: List[Tuple[int, int, float, Tuple[int, int], EdgeParams]] = []
        for t_pos, t_nid, t_q in term_info:
            nearby = []
            for o_pos, o_nid, o_theta in orient_info:
                if o_nid == t_nid:
                    continue
                dd = _pos_dist(t_pos, o_pos)
                if dd < 2.5 * r_scale:
                    nearby.append((dd, o_nid, o_theta))
            if not nearby:
                continue

            nearby.sort(key=lambda x: (x[0], x[1]))
            n_seg = len(nearby)

            if n_seg == 1:
                junction_type = 0.0
            elif n_seg == 2:
                # Collinearity from the two parent orientations: two arms of
                # one line give an axial difference near zero.
                dtheta = _angular_diff_pi(nearby[0][2], nearby[1][2])
                junction_type = 2.0 if dtheta < math.pi / 12.0 else 1.0
            elif n_seg == 3:
                junction_type = 4.0
            else:
                junction_type = 3.0

            if len(t_q) > 3:
                sharpness = clamp(float(t_q[3]) / 8.0, 0.0, 1.0)
            elif len(t_q) > 2:
                # 3-slot variant: slot 2 is `ends` in {0, 1, 2}.
                sharpness = clamp(float(t_q[2]) / 2.0, 0.0, 1.0)
            else:
                sharpness = 0.5

            if n_seg >= 2:
                corner = _angular_diff_pi(nearby[0][2], nearby[1][2])
                curvature = clamp(corner / (math.pi / 2.0), 0.0, 1.0)
            else:
                curvature = 0.0

            nearest_dist, parent_nid, parent_theta = nearby[0]

            junction_bin = float(clamp(int(junction_type), 0, 4))
            sharpness_bin = float(clamp(int(round(sharpness * 7.0)), 0, 7))
            curvature_bin = float(clamp(int(round(curvature * 7.0)), 0, 7))
            parent_bin = float(int(round(parent_theta / math.pi * 12.0)) % 12)

            proximity = math.exp(-nearest_dist / r_scale)
            signal = sharpness * proximity
            if junction_type == 2.0:
                signal *= 1.3
            signal = clamp(signal, 0.0, 1.0)
            if signal <= 0.01:
                continue

            # [P2] THE NAMED WRITER: the junction TYPE is a categorical
            # lookup key; the sharpness and the curvature are FRACTIONS; the
            # parent's orientation is an AXIAL ANGLE in RADIANS.
            ep = _edge_params(
                "EDGE_LINE_TERMINATION",
                {"junction_type": junction_bin, "sharpness": sharpness,
                 "curvature": curvature, "parent_theta": parent_theta},
                sign=1,
                confidence=float(signal),
            )
            candidates.append((t_nid, parent_nid, signal, (t_nid, parent_nid), ep))

            # A genuine corner makes the terminator evidence for BOTH arms.
            # The non-primary arm is emitted as an occluding (gating)
            # relation.
            if junction_type in (1.0, 4.0) and n_seg >= 2:
                second_nid, second_theta = nearby[1][1], nearby[1][2]
                if second_nid != t_nid:
                    ep_occl = _edge_params(
                        "EDGE_LINE_TERMINATION",
                        {"junction_type": junction_bin, "sharpness": sharpness,
                         "curvature": curvature, "parent_theta": second_theta},
                        sign=-1,
                        confidence=float(signal * 0.8),
                    )
                    candidates.append(
                        (t_nid, second_nid, signal * 0.8,
                         (t_nid, second_nid), ep_occl)
                    )

        self._process_channel(EdgeType.LINE_TERMINATION, candidates)

    # -------------------------------------------------------------------------
    # Shared emission mechanism (§2.7)
    # -------------------------------------------------------------------------

    def _process_channel(
        self,
        edge_type: EdgeType,
        candidates: List[Tuple[int, int, float, Tuple[Any, ...]]],
    ) -> None:
        """Apply the channel's threshold + hysteresis and lay down edges.

        candidates: list of (src_node_id, dst_node_id, raw_signal, key_tuple).
        key_tuple: per-channel-unique canonical tuple for M_prev tracking.
        """
        state = self.channels[edge_type]
        self.diagnostics["candidates_by_type"][edge_type.name] += len(candidates)

        # [SPEC DIVERGENCE] Homeostasis runs on every step, including
        # empty-candidate and zero-signal steps. Previously the method
        # returned early in both cases, which froze theta at its initial
        # value whenever a channel was silent on the tabletop,
        # which produces candidates on <5% of steps). The spec §2.7 requires r_channel_bar to be an
        # EMA of emission rate — an EMA that never observes silence
        # cannot adapt to it. Silent steps now contribute r = 0, driving
        # theta toward the floor so weak-but-real candidates can fire.
        emitted = 0
        if candidates:
            # D1: cap by signal using heapq.nlargest (O(N log K)).
            if len(candidates) > self.MAX_CANDIDATES_PER_CHANNEL:
                self.diagnostics["candidate_cap_hits"][edge_type.name] += 1
                candidates = nlargest(
                    self.MAX_CANDIDATES_PER_CHANNEL,
                    candidates,
                    key=lambda c: (c[2], _neg_bytes(c[3])),
                )

            max_sig = max(c[2] for c in candidates)
        else:
            max_sig = 0.0

        # [SDRSEQ PROBE] Record the "nothing to emit" state explicitly so a
        # silent channel is distinguishable from an all-rejected one. See
        # the counter's rationale in __init__.
        if max_sig <= 0.0:
            self.diagnostics["empty_or_zero_signal_by_type"][
                edge_type.name
            ] += 1

        passing: List[Tuple[int, int, float, bytes]] = []
        if max_sig > 0.0:

            theta_eff = clamp(
                state["theta"]
                - P.ETA_CTX_CHANNEL * state["C"]
                + P.ETA_FAT_CHANNEL * state["F"],
                P.THETA_ACT_FLOOR,
                P.THETA_ACT_MAX,
            )

            M_prev: Dict[bytes, int] = state["M_prev"]
            for cand in candidates:
                src, dst, signal, key = cand[0], cand[1], cand[2], cand[3]
                ep = cand[4] if len(cand) > 4 else None
                key_b = _candidate_key_bytes(key)
                norm_sig = signal / max_sig
                prev = M_prev.get(key_b, 0)
                if self._first_step:
                    threshold = theta_eff
                else:
                    threshold = (
                        theta_eff + P.DELTA_CHANNEL_HYST * (1 - 2 * prev)
                    )
                if norm_sig >= threshold:
                    passing.append((src, dst, signal, key_b, ep))

            # Emit passing edges. Preserve direction for directed types;
            # canonicalize (min, max) for undirected types.
            for src, dst, signal, _key_b, ep in passing:
                if src == dst and edge_type not in SELF_BINDER_TYPES:
                    continue
                if edge_type in UNDIRECTED_EDGE_TYPES:
                    a, b = (src, dst) if src < dst else (dst, src)
                else:
                    a, b = src, dst
                self.cg.upsert_edge(a, b, edge_type, signal, self.t, params=ep)

            emitted = len(passing)
            self.diagnostics["emitted_by_type"][edge_type.name] += emitted
            self.diagnostics["dropped_below_threshold"][edge_type.name] += (
                len(candidates) - emitted
            )

            # D10 + [v8] Retain M_prev across steps, updated in
            # place. Eviction is LRU on `last_seen` (mirrors
            passing_keys = {p[3] for p in passing}
            last_seen: Dict[bytes, int] = state["last_seen"]
            new_M_prev = dict(M_prev)
            for cand in candidates:
                key = cand[3]
                key_b = _candidate_key_bytes(key)
                new_M_prev[key_b] = 1 if key_b in passing_keys else 0
                last_seen[key_b] = self.t
            if len(new_M_prev) > self.MAX_M_PREV_SIZE:
                to_drop = len(new_M_prev) - (self.MAX_M_PREV_SIZE // 2)
                ordered = sorted(
                    new_M_prev.keys(),
                    key=lambda k: (last_seen.get(k, -1), k),
                )
                for k in ordered[:to_drop]:
                    new_M_prev.pop(k, None)
                    last_seen.pop(k, None)
            state["M_prev"] = new_M_prev

        # Homeostatic update runs unconditionally (see divergence note above).
        N_active = len(candidates)
        r = emitted / (N_active + EPSILON) if N_active > 0 else 0.0
        state["r_bar"] = (
            (1.0 - P.KAPPA_R_CHANNEL) * state["r_bar"]
            + P.KAPPA_R_CHANNEL * r
        )
        state["theta"] = clamp(
            state["theta"]
            + P.KAPPA_CHANNEL_THETA * (state["r_bar"] - P.RHO_CHANNEL_TARGET),
            P.THETA_CHANNEL_MIN,
            P.THETA_CHANNEL_MAX,
        )
        had_emission = 1.0 if emitted > 0 else 0.0
        state["C"] = (1.0 - P.KAPPA_C_CHANNEL) * state[
            "C"
        ] + P.KAPPA_C_CHANNEL * had_emission
        state["F"] = (
            P.BETA_F_CHANNEL * state["F"] + (1.0 - P.BETA_F_CHANNEL) * had_emission
        )

    # -------------------------------------------------------------------------
    # Pattern history (D9)
    # -------------------------------------------------------------------------

    def _update_pattern_history(self, dets: Tuple[Detection, ...]) -> None:
        """Update per-pattern presence, onset, and per-anchor change state."""
        present_now: Set[Tuple[str, bytes]] = set()
        anchor_patterns_now: Dict[Any, Set[Tuple[str, bytes]]] = defaultdict(set)
        for d in dets:
            pat = (d.k, _node_signature_hash(d.sigma))
            present_now.add(pat)
            anchor_patterns_now[d.alpha].add(pat)
            self._pattern_last_seen[pat] = self.t
            self._pattern_anchors[pat].add(d.alpha)

        onset_now = present_now - self._pattern_present_prev
        self._pattern_onset_now = onset_now
        for pat in sorted(onset_now):
            self._pattern_last_onset[pat] = self.t

        # Extend per-pattern sliding-window history.
        all_patterns = set(self._pattern_history.keys()) | present_now
        W = P.W_HISTORY
        for pat in sorted(all_patterns):
            hist = self._pattern_history[pat]
            hist.append(1 if pat in present_now else 0)
            if len(hist) > W:
                hist.pop(0)

        # Per-anchor pattern-set change EMA (for Persistence).
        all_anchors = set(self._anchor_pattern_set_prev.keys()) | set(
            anchor_patterns_now.keys()
        )
        beta = P.PRESENCE_EMA_BETA
        for anchor in all_anchors:
            prev_set = self._anchor_pattern_set_prev.get(anchor)
            curr_set = frozenset(anchor_patterns_now.get(anchor, set()))
            if prev_set is not None:
                change = float(len(prev_set ^ curr_set))
                self._anchor_change_ema[anchor] = (
                    beta * self._anchor_change_ema[anchor] + (1.0 - beta) * change
                )
            else:
                self._anchor_change_ema[anchor] = 0.0
            self._anchor_pattern_set_prev[anchor] = curr_set

        # Prune pattern state older than 2 * W_HISTORY.
        cutoff = self.t - 2 * W
        stale = [
            p
            for p, last in self._pattern_last_seen.items()
            if last < cutoff and p not in present_now
        ]
        for pat in sorted(stale):
            self._pattern_history.pop(pat, None)
            self._pattern_last_onset.pop(pat, None)
            self._pattern_last_seen.pop(pat, None)
            self._pattern_anchors.pop(pat, None)

        # Prune anchor change state for anchors absent a long time.
        stale_anchors = [
            a
            for a, s in self._anchor_pattern_set_prev.items()
            if not s and self._anchor_change_ema.get(a, 0.0) < 1e-6
        ]
        for a in stale_anchors:
            self._anchor_pattern_set_prev.pop(a, None)
            self._anchor_change_ema.pop(a, None)

    # -------------------------------------------------------------------------
    # Co-activation (undirected)
    # -------------------------------------------------------------------------

    def _run_co_activation(
        self, dets: Tuple[Detection, ...], node_ids: Dict[Detection, int]
    ) -> None:
        # [P2-5] Semantic change: emit CO_ACTIVATION only for pairs where at
        # least one endpoint is a fresh onset this step (0→1 transition).
        # Steady-state co-presence is uninformative: two patterns present
        # for 100 steps co-activate 100 times without new information.
        # The informative signal is "started happening together."
        #
        # P is bounded by distinct detections per step (<50 in practice).
        # Precompute signature hashes once per detection — avoids O(P²)
        # hash_128(serialize(sig)) calls inside the pair loop.
        if getattr(self, "_static_frame", False):
            return
        onset_pats = self._pattern_onset_now

        det_data = []
        for d in dets:
            pat = (d.k, _node_signature_hash(d.sigma))
            det_data.append((d, pat, node_ids[d]))

        candidates = []
        steady_pairs = 0
        for i in range(len(det_data)):
            d1, p1, src = det_data[i]
            for j in range(i + 1, len(det_data)):
                d2, p2, dst = det_data[j]
                if d1.k == d2.k and d1.sigma == d2.sigma:
                    continue
                if p1 not in onset_pats and p2 not in onset_pats:
                    steady_pairs += 1
                    continue
                if src == dst:
                    continue
                a, b = (src, dst) if src < dst else (dst, src)
                candidates.append((a, b, 1.0, (a, b)))
        self.diagnostics["co_activation_steady_pairs_skipped"] += steady_pairs
        self._process_channel(EdgeType.CO_ACTIVATION, candidates)

    # -------------------------------------------------------------------------
    # Transition (directed) — §1.12
    # -------------------------------------------------------------------------

    def _run_transition(
        self, dets: Tuple[Detection, ...], node_ids: Dict[Detection, int]
    ) -> None:
        """Signal = 1 iff parent onset at t-1 and child onset at t."""
        if not self._pattern_onset_prev or not self._pattern_onset_now:
            return

        # Map current-onset patterns to node ids.
        current_node: Dict[Tuple[str, bytes], int] = {}
        for d in dets:
            pat = (d.k, _node_signature_hash(d.sigma))
            if pat in self._pattern_onset_now and pat not in current_node:
                current_node[pat] = node_ids[d]

        candidates: List[Tuple[int, int, float, Tuple[Any, ...]]] = []
        for p_pat in sorted(self._pattern_onset_prev):
            src = self.cg.get_or_create_pattern_node(p_pat[0], p_pat[1])
            for c_pat, dst in current_node.items():
                if p_pat == c_pat:
                    continue
                if src == dst:
                    continue
                # Directed: src=parent, dst=child.
                candidates.append((src, dst, 1.0, (src, dst)))
        self._process_channel(EdgeType.TRANSITION, candidates)

    # -------------------------------------------------------------------------
    # Co-motion (undirected) — D6
    # -------------------------------------------------------------------------

    def _run_co_motion(
        self,
        dets: Tuple[Detection, ...],
        node_ids: Dict[Detection, int],
        anchor_velocities: Optional[Dict[Any, Tuple[float, ...]]],
    ) -> None:
        if not anchor_velocities or len(anchor_velocities) < 2:
            return

        rep_by_anchor: Dict[Any, Detection] = {}
        for d in dets:
            if d.alpha in anchor_velocities and d.alpha not in rep_by_anchor:
                rep_by_anchor[d.alpha] = d

        anchors = sorted(
            (a for a in rep_by_anchor.keys()),
            key=serialize_value,
        )
        candidates: List[Tuple[int, int, float, Tuple[Any, ...]]] = []
        for i in range(len(anchors)):
            for j in range(i + 1, len(anchors)):
                a, b = anchors[i], anchors[j]
                va = anchor_velocities.get(a)
                vb = anchor_velocities.get(b)
                if va is None or vb is None or len(va) < 3 or len(vb) < 3:
                    continue
                if not all(math.isfinite(x) for x in va[:3]):
                    continue
                if not all(math.isfinite(x) for x in vb[:3]):
                    continue
                na = math.sqrt(sum(x * x for x in va[:3]))
                nb = math.sqrt(sum(x * x for x in vb[:3]))
                if na < EPSILON or nb < EPSILON:
                    continue
                cos_sim = sum(x * y for x, y in zip(va[:3], vb[:3])) / (na * nb)
                signal = 0.5 * (1.0 + clamp(cos_sim, -1.0, 1.0))
                da = rep_by_anchor[a]
                db = rep_by_anchor[b]
                src = node_ids[da]
                dst = node_ids[db]
                if src == dst:
                    continue
                a_id, b_id = (src, dst) if src < dst else (dst, src)
                # [Pass 22] First real EdgeParams for CO_MOTION. Common fate
                # in MT is direction- AND speed-tuned, so a single normalised
                # dot product (which collapses toward cos_sim alone)
                # under-describes the relation. na / nb are the per-pair
                # speeds, already computed and finite-checked above.
                V = max(P.MOTION_SPEED_MAX, 1e-6)
                speed_bin = _bin8(math.log2(max(0.5 * (na + nb), V * 2.0 ** -6) / V), -6.0, 0.0)
                lead = va if na >= nb else vb
                heading = _safe_atan2(lead[1], lead[0]) % (2.0 * math.pi)
                heading_bin = float(int(round(heading / (2.0 * math.pi) * 24)) % 24)
                ratio_bin = _bin8(abs(math.log2(na / nb)), 0.0, 3.0)
                ang_bin = _bin8(math.acos(clamp(cos_sim, -1.0, 1.0)), 0.0, math.pi)
                # [P2] THE NAMED WRITER: the mean speed as a LOG2 speed
                # ratio, the heading in RADIANS, the speed ratio as an
                # absolute log2 ratio and the angle between the velocities
                # in RADIANS.
                ep = _edge_params(
                    "EDGE_CO_MOTION",
                    {"mean_speed": math.log2(max(0.5 * (na + nb),
                                                 V * 2.0 ** -6) / V),
                     "heading": heading,
                     "speed_ratio": abs(math.log2(na / nb)),
                     "angle_diff": math.acos(clamp(cos_sim, -1.0, 1.0))},
                    sign=1,
                    confidence=float(clamp(signal, 0.0, 1.0)),
                )
                candidates.append((a_id, b_id, signal, (a_id, b_id), ep))
        self._process_channel(EdgeType.CO_MOTION, candidates)

    # -------------------------------------------------------------------------
    # Contrast (directed) — D3
    # -------------------------------------------------------------------------

    def _run_contrast(
        self, dets: Tuple[Detection, ...], node_ids: Dict[Detection, int]
    ) -> None:
        # Distinct patterns present this step.
        patterns: Dict[Tuple[str, bytes], Detection] = {}
        for d in dets:
            pat = (d.k, _node_signature_hash(d.sigma))
            if pat not in patterns:
                patterns[pat] = d
        if len(patterns) < 2:
            return

        pattern_keys = sorted(patterns.keys(), key=lambda k: (k[0], k[1]))
        candidates: List[Tuple[int, int, float, Tuple[Any, ...]]] = []
        min_obs = P.CONTRAST_MIN_OBS

        for i in range(len(pattern_keys)):
            for j in range(len(pattern_keys)):
                if i == j:
                    continue
                ka = pattern_keys[i]  # cause
                kb = pattern_keys[j]  # effect
                hist_a = self._pattern_history.get(ka)
                hist_b = self._pattern_history.get(kb)
                if not hist_a or not hist_b:
                    continue

                p_b_given_a, n_a = _conditional_from_history(
                    hist_a, hist_b, condition_on_a_present=True
                )
                p_b_given_not_a, n_not_a = _conditional_from_history(
                    hist_a, hist_b, condition_on_a_present=False
                )
                if n_a + n_not_a < min_obs:
                    continue

                delta = p_b_given_a - p_b_given_not_a
                signal = 0.5 * (1.0 + math.tanh(2.0 * delta))
                if signal <= 0.0:
                    continue
                da = patterns[ka]
                db = patterns[kb]
                src = node_ids[da]
                dst = node_ids[db]
                if src == dst:
                    continue
                # Directed: cause -> effect.
                candidates.append((src, dst, signal, (src, dst)))
        self._process_channel(EdgeType.CONTRAST, candidates)

    # -------------------------------------------------------------------------
    # Periodicity (undirected) — D3, D7
    # -------------------------------------------------------------------------

    def _run_periodicity(
        self, dets: Tuple[Detection, ...], node_ids: Dict[Detection, int]
    ) -> None:
        patterns: Dict[Tuple[str, bytes], Detection] = {}
        for d in dets:
            pat = (d.k, _node_signature_hash(d.sigma))
            if pat not in patterns:
                patterns[pat] = d
        if len(patterns) < 2:
            return

        # Compute interval statistics per pattern from onset transitions.
        sigs: Dict[Tuple[str, bytes], Tuple[float, float]] = {}
        for pat, hist in sorted(self._pattern_history.items()):
            stats = _pattern_intervals(hist)
            if stats is not None:
                sigs[pat] = stats

        if len(sigs) < 2:
            return

        pattern_keys = sorted(sigs.keys(), key=lambda k: (k[0], k[1]))
        candidates: List[Tuple[int, int, float, Tuple[Any, ...]]] = []
        for i in range(len(pattern_keys)):
            for j in range(i + 1, len(pattern_keys)):
                ka = pattern_keys[i]
                kb = pattern_keys[j]
                mean_a, std_a = sigs[ka]
                mean_b, std_b = sigs[kb]
                if mean_a <= 0.0 or mean_b <= 0.0:
                    continue
                reg_a = 1.0 - clamp(std_a / (mean_a + EPSILON), 0.0, 1.0)
                reg_b = 1.0 - clamp(std_b / (mean_b + EPSILON), 0.0, 1.0)
                ratio = mean_a / mean_b if mean_a >= mean_b else mean_b / mean_a
                commensurate = _commensurability(ratio)
                signal = 0.5 * (reg_a + reg_b) * commensurate
                if signal <= 0.0:
                    continue
                da = patterns.get(ka)
                db = patterns.get(kb)
                if da is None or db is None:
                    continue
                src = node_ids[da]
                dst = node_ids[db]
                if src == dst:
                    continue
                a_id, b_id = (src, dst) if src < dst else (dst, src)
                candidates.append((a_id, b_id, signal, (a_id, b_id)))
        self._process_channel(EdgeType.PERIODICITY, candidates)

    # -------------------------------------------------------------------------
    # Persistence (directed) — D3, D8
    # -------------------------------------------------------------------------

    def _run_persistence(
        self, dets: Tuple[Detection, ...], node_ids: Dict[Detection, int]
    ) -> None:
        """A stable while B changes. Stability = presence fraction of A.
        Change = mean anchor pattern-set change EMA over B's anchors."""
        patterns: Dict[Tuple[str, bytes], Detection] = {}
        for d in dets:
            pat = (d.k, _node_signature_hash(d.sigma))
            if pat not in patterns:
                patterns[pat] = d
        if len(patterns) < 2:
            return

        pattern_keys = sorted(patterns.keys(), key=lambda k: (k[0], k[1]))
        candidates: List[Tuple[int, int, float, Tuple[Any, ...]]] = []
        threshold = P.PERSISTENCE_CHANGE_THRESHOLD

        for i in range(len(pattern_keys)):
            for j in range(len(pattern_keys)):
                if i == j:
                    continue
                ka = pattern_keys[i]  # stable side
                kb = pattern_keys[j]  # changing side

                hist_a = self._pattern_history.get(ka)
                if not hist_a:
                    continue
                stab_a = _presence_fraction(hist_a)
                if stab_a < 0.5:
                    continue

                b_anchors = self._pattern_anchors.get(kb, set())
                if not b_anchors:
                    continue
                changes = [self._anchor_change_ema.get(a, 0.0) for a in b_anchors]
                mean_change = sum(changes) / len(changes)
                if mean_change < threshold:
                    continue

                signal = stab_a * clamp(mean_change, 0.0, 1.0)
                if signal <= 0.0:
                    continue
                da = patterns[ka]
                db = patterns[kb]
                src = node_ids[da]
                dst = node_ids[db]
                if src == dst:
                    continue
                # Directed: stable -> changing.
                candidates.append((src, dst, signal, (src, dst)))
        self._process_channel(EdgeType.PERSISTENCE, candidates)

    # -------------------------------------------------------------------------
    # Topology (directed) — D6
    # -------------------------------------------------------------------------

    def _is_surface_anchor(self, anchor: Any) -> bool:
        """[Stage 1.2] True if anchor belongs strictly to the surface
        (hypercolumn) stream. Tuple segment anchors (u, seg_idx) unwrap to u."""
        a = anchor[0] if isinstance(anchor, tuple) else anchor
        if self._surface_anchors and a in self._surface_anchors:
            return True
        return False

    def _is_structural_anchor(self, anchor: Any) -> bool:
        """[Stage 1.2] True if anchor is a boundary contour, junction, or
        other structural entity. With no stream info registered (legacy
        callers without stream info) every anchor counts as structural —
        behavior unchanged for temporal/synthetic paths. Tuple segment
        anchors (u, seg_idx) unwrap to u; synthetic anchors outside both
        registries default to structural."""
        a = anchor[0] if isinstance(anchor, tuple) else anchor
        if self._surface_anchors and a in self._surface_anchors:
            return False
        if self._structural_anchors and a in self._structural_anchors:
            return True
        return True

    def _run_topology(
        self,
        dets: Tuple[Detection, ...],
        node_ids: Dict[Detection, int],
        anchor_positions: Optional[Dict[Any, Tuple[float, ...]]],
    ) -> None:
        if not anchor_positions or len(anchor_positions) < 2:
            return

        # [SPATIAL BINDERS] Rep selection must yield anchor-distinguishing
        # pattern nodes. Boundary and Onset signatures do not depend on
        # position, so all Boundary detections for different anchors
        # collapse to the same pattern node under §1.11 identity
        # (lens_type, signature_hash). Spatial edges between the same node
        # are self-loops, forbidden by §3.4. Position nodes ARE
        # anchor-distinguishing (their signature quantizes egocentric
        # coordinates, which differ per entity), so use them as the rep.
        # If no Position detection exists for an anchor, that anchor is
        # skipped for this binder; anchors without a Position observation
        # have no spatial representation to link.
        rep_by_anchor: Dict[Any, Detection] = {}
        for d in dets:
            if d.alpha not in anchor_positions:
                continue
            if d.k != "Position":
                continue
            if d.alpha not in rep_by_anchor:
                rep_by_anchor[d.alpha] = d
        anchors = sorted(rep_by_anchor.keys(), key=serialize_value)
        if len(anchors) < 2:
            return

        candidates: List[Tuple[int, int, float, Tuple[Any, ...]]] = []
        for a in anchors:
            # [Stage 1.2] Skip unless both anchors are structural (boundary/contour)
            if not self._is_structural_anchor(a):
                continue
            pa = anchor_positions[a]
            if len(pa) < 3 or not all(math.isfinite(x) for x in pa[:3]):
                continue
            dists: List[Tuple[float, Any]] = []
            for b in anchors:
                if a == b:
                    continue
                if not self._is_structural_anchor(b):
                    continue
                pb = anchor_positions[b]
                if len(pb) < 3 or not all(math.isfinite(x) for x in pb[:3]):
                    continue
                d_ab = math.sqrt(sum((x - y) ** 2 for x, y in zip(pa[:3], pb[:3])))
                if math.isfinite(d_ab):
                    dists.append((d_ab, b))
            dists.sort(key=lambda t: (t[0], serialize_value(t[1])))
            top = dists[: P.TOPOLOGY_K_MAX]
            src = node_ids[rep_by_anchor[a]]
            for rank, (_d_ab, b) in enumerate(top):
                signal = 1.0 / (1.0 + float(rank))
                dst = node_ids[rep_by_anchor[b]]
                if src == dst:
                    continue
                # Directed: A -> B means B is one of A's K-nearest.
                candidates.append((src, dst, signal, (src, dst)))
        self._process_channel(EdgeType.TOPOLOGY, candidates)

    # -------------------------------------------------------------------------
    # Degree (undirected) — D6
    # -------------------------------------------------------------------------

    def _run_degree(
        self,
        dets: Tuple[Detection, ...],
        node_ids: Dict[Detection, int],
        anchor_positions: Optional[Dict[Any, Tuple[float, ...]]],
    ) -> None:
        if not anchor_positions or len(anchor_positions) < 2:
            return
        # [SPATIAL BINDERS] Same rep-selection rationale as _run_topology:
        # only Position nodes distinguish anchors. See _run_topology for
        # the full explanation.
        rep_by_anchor: Dict[Any, Detection] = {}
        for d in dets:
            if d.alpha not in anchor_positions:
                continue
            if d.k != "Position":
                continue
            if d.alpha not in rep_by_anchor:
                rep_by_anchor[d.alpha] = d
        anchors = sorted(rep_by_anchor.keys(), key=serialize_value)
        # [D2/Seam-4] Scale-adaptive radius from the median nearest-
        # neighbour distance (see _effective_degree_radius). The former
        # getattr(P, "DEGREE_RADIUS", 0.5) read a parameter that does not
        # exist, silently pinning the radius at 0.5 — dark on every
        # symbolic grid (typical distances ≥ 5 units).
        radius = self._effective_degree_radius()
        candidates: List[Tuple[int, int, float, Tuple[Any, ...]]] = []
        for i in range(len(anchors)):
            for j in range(i + 1, len(anchors)):
                a, b = anchors[i], anchors[j]
                # [Stage 1.2] Skip unless both anchors are structural
                if not self._is_structural_anchor(a) or not self._is_structural_anchor(b):
                    continue
                pa = anchor_positions[a]
                pb = anchor_positions[b]
                if len(pa) < 3 or len(pb) < 3:
                    continue
                if not all(math.isfinite(x) for x in pa[:3]):
                    continue
                if not all(math.isfinite(x) for x in pb[:3]):
                    continue
                d_ab = math.sqrt(sum((x - y) ** 2 for x, y in zip(pa[:3], pb[:3])))
                if not math.isfinite(d_ab) or d_ab > radius:
                    continue
                src = node_ids[rep_by_anchor[a]]
                dst = node_ids[rep_by_anchor[b]]
                if src == dst:
                    continue
                a_id, b_id = (src, dst) if src < dst else (dst, src)
                # Signal: proximity strength in [0, 1].
                signal = 1.0 - clamp(d_ab / (radius + EPSILON), 0.0, 1.0)
                if signal <= 0.0:
                    signal = 1e-6
                candidates.append((a_id, b_id, signal, (a_id, b_id)))
        self._process_channel(EdgeType.DEGREE, candidates)

    # -------------------------------------------------------------------------
    # Contact (undirected) — D6
    # -------------------------------------------------------------------------

    def _run_contact(
        self,
        dets: Tuple[Detection, ...],
        node_ids: Dict[Detection, int],
        anchor_positions: Optional[Dict[Any, Tuple[float, ...]]],
    ) -> None:
        if not anchor_positions or len(anchor_positions) < 2:
            return
        # [SPATIAL BINDERS] Same rep-selection rationale as _run_topology.
        rep_by_anchor: Dict[Any, Detection] = {}
        for d in dets:
            if d.alpha not in anchor_positions:
                continue
            if d.k != "Position":
                continue
            if d.alpha not in rep_by_anchor:
                rep_by_anchor[d.alpha] = d
        anchors = sorted(rep_by_anchor.keys(), key=serialize_value)

        # [v8] Two-branch radius: scale-adaptive when positions
        # available, physical fallback otherwise.
        r_med = (
            self._effective_degree_radius()
            if self._step_l_med > 0.0
            else float(getattr(P, "CONTACT_R_MED_BOOTSTRAP", 1.0))
        )
        candidates: List[Tuple[int, int, float, Tuple[Any, ...]]] = []
        for i in range(len(anchors)):
            for j in range(i + 1, len(anchors)):
                a, b = anchors[i], anchors[j]
                # [Stage 1.2] Skip unless both anchors are structural
                if not self._is_structural_anchor(a) or not self._is_structural_anchor(b):
                    continue
                pa = anchor_positions[a]
                pb = anchor_positions[b]
                if len(pa) < 2 or len(pb) < 2:
                    continue
                if not all(math.isfinite(x) for x in pa[:2]):
                    continue
                if not all(math.isfinite(x) for x in pb[:2]):
                    continue
                dist = _pos_dist(pa[:2], pb[:2])
                if not math.isfinite(dist):
                    continue
                signal = math.exp(-dist / (r_med + EPSILON))
                src = node_ids[rep_by_anchor[a]]
                dst = node_ids[rep_by_anchor[b]]
                if src == dst:
                    continue
                a_id, b_id = (src, dst) if src < dst else (dst, src)
                # [S5 contract] Orient the normal from lower to higher node ID
                # so EDGE_CONTACT's slot-0 reversal (reflect across center,
                # +12 on cycle 24) is valid. Keys are ID-ordered but the pair
                # loop iterates serialize order, so this must be explicit.
                if src > dst:
                    pa, pb = pb, pa
                theta_n = math.atan2(pb[1] - pa[1], pb[0] - pa[0])
                has_z = (len(pa) >= 3 and len(pb) >= 3 and math.isfinite(pa[2]) and math.isfinite(pb[2]))
                # [P2] THE NAMED WRITER: the contact NORMAL is an AXIAL
                # ANGLE in RADIANS, the SEPARATION is in MEDIAN SPACINGS
                # (the contact radius r_med is the V1 field's own scale) and
                # the depth step is in the 3-D depth units.
                normal_rad = theta_n % math.pi
                dist_sp = dist / max(r_med, 1e-6)
                depth_step = (abs(pa[2] - pb[2]) / max(
                    P.OCCLUSION_3D_SIGMA_DEPTH, 1e-6)) if has_z else None
                ep = _edge_params(
                    "EDGE_CONTACT",
                    {"normal_angle": normal_rad, "dist": dist_sp,
                     "depth_step": depth_step},
                    sign=1,
                    confidence=float(clamp(signal, 0.0, 1.0)),
                )
                candidates.append((a_id, b_id, signal, (a_id, b_id), ep))
        # [Q3] Count-cap contact edges: keep only the CONTACT_MAX_PER_NODE
        # strongest incident contact edges per node. EDGE_CONTACT is
        # proximity-only, so without a cap a dense silence of neighbours
        # dominates the coupling matrix (measured: 118 of 146 edges in a
        # two-panel composite were contact edges, all positive). The cap is
        # per node, not per pair, and the distance decay above stays: the
        # surviving edges are still the strongest (nearest) ones.
        cap = int(P.CONTACT_MAX_PER_NODE)
        if bool(P.CONTACT_CAP_ENABLED) and cap > 0 and len(candidates) > cap:
            best: Dict[int, List[Tuple[float, Tuple[Any, ...], int]]] = defaultdict(list)
            for idx, (a_id, b_id, signal, key, _ep) in enumerate(candidates):
                best[a_id].append((signal, key, idx))
                best[b_id].append((signal, key, idx))
            keep_idx: Set[int] = set()
            for node in sorted(best, key=lambda n: (repr(n),)):
                rows = sorted(best[node], key=lambda r: (-r[0], repr(r[1])))
                for signal, _key, idx in rows[:cap]:
                    keep_idx.add(idx)
            candidates = [c for i, c in enumerate(candidates) if i in keep_idx]
        self._process_channel(EdgeType.CONTACT, candidates)

    # -------------------------------------------------------------------------
    # Stream Binding (PART_CONTENT) — Stage 1.2
    # -------------------------------------------------------------------------

    def _run_stream_binding(
        self,
        dets: Tuple[Detection, ...],
        node_ids: Dict[Detection, int],
    ) -> None:
        """Emits PART_CONTENT edges binding surface hypercolumn content (Color/Texture)
        to contours whose polylines pass through the hypercolumn RF."""
        if not self._contour_surface_contacts:
            return

        rep_by_anchor: Dict[Any, Detection] = {}
        for d in dets:
            if d.alpha not in rep_by_anchor:
                rep_by_anchor[d.alpha] = d

        candidates: List[Tuple[int, int, float, Tuple[Any, ...], EdgeParams]] = []
        for c_anchor, s_anchor, signal in self._contour_surface_contacts:
            d_contour = rep_by_anchor.get(c_anchor)
            d_surface = rep_by_anchor.get(s_anchor)
            if d_contour is None or d_surface is None:
                continue

            # Node ID 0 is valid — do not use `if not u`
            u = node_ids.get(d_contour)
            v = node_ids.get(d_surface)
            if u is None or v is None or u == v:
                continue

            src, dst = (u, v) if u < v else (v, u)
            # [P2] THE NAMED WRITER: the membership is a 0/1 FLAG; the
            # strength travels in the confidence.
            ep = _edge_params(
                "EDGE_REGION_CONTENT",
                {"containment": 1.0},
                sign=1,
                confidence=float(clamp(signal, 0.0, 1.0)),
            )
            candidates.append((src, dst, signal, (src, dst), ep))

        self._process_channel(EdgeType.REGION_CONTENT, candidates)

    # -------------------------------------------------------------------------
    # Sequence (directed) — D9
    # -------------------------------------------------------------------------

    def _run_sequence(
        self, dets: Tuple[Detection, ...], node_ids: Dict[Detection, int]
    ) -> None:
        """Signal = exp(-eta_seq_age * lag) for parent onset within W_SEQ
        steps before a child onset at the current step."""
        if not self._pattern_onset_now:
            return

        # Current-onset patterns and their node IDs.
        current_node: Dict[Tuple[str, bytes], int] = {}
        for d in dets:
            pat = (d.k, _node_signature_hash(d.sigma))
            if pat in self._pattern_onset_now and pat not in current_node:
                current_node[pat] = node_ids[d]

        # Parent candidates: any pattern whose onset was within W_SEQ steps
        # before now, excluding the current step (lag must be > 0).
        lag_max = P.W_SEQ
        candidate_parents: List[Tuple[Tuple[str, bytes], int]] = []
        for pat, onset_t in sorted(self._pattern_last_onset.items()):
            lag = self.t - onset_t
            if 0 < lag <= lag_max and pat not in current_node:
                candidate_parents.append((pat, lag))

        candidates: List[Tuple[int, int, float, Tuple[Any, ...]]] = []
        for p_pat, lag in candidate_parents:
            src = self.cg.get_or_create_pattern_node(p_pat[0], p_pat[1])
            signal = math.exp(-P.ETA_SEQ_AGE * float(lag))
            for c_pat, dst in current_node.items():
                if p_pat == c_pat:
                    continue
                if src == dst:
                    continue
                candidates.append((src, dst, signal, (src, dst)))
        self._process_channel(EdgeType.SEQUENCE, candidates)

    # -------------------------------------------------------------------------
    # Synchrony (undirected) — P2-2
    # -------------------------------------------------------------------------

    def _synchrony_signal(
        self, ka: Tuple[str, bytes], kb: Tuple[str, bytes]
    ) -> float:
        """[P2-2] Phase-lock signal between two patterns' onset histories.

        Over the shared trailing window W = min(len(a), len(b), W_SYNC),
        locate 0->1 onset transitions in each presence history. For every
        A onset, lag = distance to the nearest B onset. Consistency =
        (mode_count - 1) / (n_lags - 1): 1.0 when all lags equal the modal
        lag (perfect phase lock), lower as lags scatter. Requires W >= 4
        and >= 2 onsets per pattern; otherwise 0.0.
        """
        hist_a = self._pattern_history.get(ka)
        hist_b = self._pattern_history.get(kb)
        if not hist_a or not hist_b:
            return 0.0
        W = min(len(hist_a), len(hist_b), P.W_SYNC)
        if W < 4:
            return 0.0
        a_w = hist_a[-W:]
        b_w = hist_b[-W:]
        onsets_a = [i for i in range(1, W) if a_w[i] == 1 and a_w[i - 1] == 0]
        onsets_b = [i for i in range(1, W) if b_w[i] == 1 and b_w[i - 1] == 0]
        if len(onsets_a) < 2 or len(onsets_b) < 2:
            return 0.0
        lags = [min(abs(ob - oa) for ob in onsets_b) for oa in onsets_a]

        counts = Counter(lags)
        mode_count = max(counts.values())
        phase_consistency = (mode_count - 1) / (len(lags) - 1)
        return clamp(phase_consistency, 0.0, 1.0)

    def _run_synchrony(
        self, dets: Tuple[Detection, ...], node_ids: Dict[Detection, int]
    ) -> None:
        """[P2-2] Emit SYNCHRONY edges between current-frame patterns whose
        onset histories are phase-locked over the trailing W_SYNC window."""
        if getattr(self, "_static_frame", False):
            return
        if not self._pattern_onset_now:
            return

        patterns: Dict[Tuple[str, bytes], Detection] = {}
        for d in dets:
            pat = (d.k, _node_signature_hash(d.sigma))
            if pat not in patterns:
                patterns[pat] = d

        pattern_keys = sorted(patterns.keys(), key=lambda k: (k[0], k[1]))
        candidates: List[Tuple[int, int, float, Tuple[Any, ...]]] = []
        for i in range(len(pattern_keys)):
            for j in range(i + 1, len(pattern_keys)):
                ka = pattern_keys[i]
                kb = pattern_keys[j]
                signal = self._synchrony_signal(ka, kb)
                if signal <= 0.0:
                    continue
                src = node_ids[patterns[ka]]
                dst = node_ids[patterns[kb]]
                if src == dst:
                    continue
                a_id, b_id = (src, dst) if src < dst else (dst, src)
                candidates.append((a_id, b_id, signal, (a_id, b_id)))
        self._process_channel(EdgeType.SYNCHRONY, candidates)

    def _run_orientation_continuity(
        self,
        dets: Tuple[Detection, ...],
        node_ids: Dict[Detection, int],
        anchor_positions: Optional[Dict[Any, Tuple[float, ...]]],
    ) -> None:
        """[Pass 21] V1 association field with per-port competition and
        bipole roles.

        Stage 1 (V1 horizontal connections, Field et al. 1993): a symmetric
        pair field from path geometry; each segment END (port) keeps its
        single best partner -- winner-take-all per port.
        Stage 2 (bipole / end-stopping, Grossberg & Mingolla 1985): node role
        is terminal (one supported port) / weak interior / strong interior
        (sqrt(s0*s1) >= cut).
        Polarity is a RELATION SLOT (same / none / flipped), never a veto:
        contours routinely reverse contrast across background changes.
        Edge count <= 2N (one winner per port); cost O(N*k) for k neighbours
        inside the search radius, replacing the previous O(N^2) sweep.
        """
        if not anchor_positions or len(anchor_positions) < 2:
            return
        L = self._step_l_med
        if L <= 0.0:
            return

        rep: Dict[Any, Detection] = {}
        _spine_cls = _orient_spine_classes()
        for d in dets:
            if d.k not in ("Orientation", "OrientationPol"): continue
            if d.sigma.class_ not in _spine_cls: continue
            if d.alpha not in anchor_positions or len(d.sigma.quantized) < 1: continue

            cur = rep.get(d.alpha)
            if cur is None or (d.sigma.class_ == "ORIENTATION_POL" and cur.sigma.class_ != "ORIENTATION_POL"):
                rep[d.alpha] = d
        anchors = sorted(rep.keys(), key=serialize_value)
        if len(anchors) < 2:
            return

        n_theta = P.ORIENTATION_N_THETA
        # [O2] The representative ORIENTATION theta, by name.
        # [U21] Through the SHARED family reader: the GABOR family's theta is a
        # bank index over a 12-channel bank, not a 24-bin geometry slot, so
        # reading it with this channel's n_theta would halve every angle (the
        # bug base.py records).  The Contour lens reads the same function, so
        # the spine's two sides agree on what a member's angle IS.
        theta = {a: orientation_theta_radians(rep[a].sigma) for a in anchors}
        idx = {a: i for i, a in enumerate(anchors)}
        sigma_p = P.ORIENTATION_CONTINUITY_SIGMA_P_FRAC * L
        tau = P.ORIENTATION_CONTINUITY_TAU_GAP_FRAC * L
        sigma_theta = P.ORIENTATION_CONTINUITY_SIGMA_THETA
        radius = P.ORIENTATION_CONTINUITY_RANGE_TAU_MULT * tau
        min_sig = P.ORIENTATION_CONTINUITY_PORT_MIN_SIGNAL
        sh = SpatialHash({a: anchor_positions[a] for a in anchors},
                         cell_size=max(L, 1.0))

        # ---- Stage 1: per-port winner-take-all ------------------------------
        winner: Dict[Tuple[int, int], Tuple[float, Tuple[int, int]]] = {}
        pair_geom: Dict[Tuple[int, int], Tuple[float, ...]] = {}
        for a in anchors:
            i, pa = idx[a], anchor_positions[a]
            for b, _pb in sh.neighbors(pa, radius):
                j = idx.get(b)
                if j is None or j <= i:
                    continue
                g = _assoc_pair_geometry(
                    theta[a], pa, theta[b], anchor_positions[b],
                    sigma_theta, sigma_p, tau
                )
                if g is None or g[0] < min_sig:
                    continue
                pair_geom[(i, j)] = g
                for me, other in (((i, g[3]), (j, g[4])),
                                  ((j, g[4]), (i, g[3]))):
                    cur = winner.get(me)
                    if cur is None or g[0] > cur[0]:  # strict > : deterministic
                        winner[me] = (g[0], other)

        # ---- Stage 2: node roles (bipole support) ---------------------------
        port_sig: Dict[int, Dict[int, float]] = defaultdict(dict)
        for (i, side), (s, _o) in winner.items():
            port_sig[i][side] = s

        def _role(i: int) -> float:
            s = port_sig.get(i, {})
            if len(s) < 2:
                return 0.0  # terminal
            bipole = math.sqrt(s.get(0, 0.0) * s.get(1, 0.0))
            return 2.0 if bipole >= P.ORIENTATION_CONTINUITY_BIPOLE_STRONG else 1.0

        # ---- Emit -----------------------------------------------------------
        candidates: List[
            Tuple[int, int, float, Tuple[int, int], EdgeParams]
        ] = []
        emitted: Set[Tuple[int, int]] = set()
        for port in sorted(winner):
            _sig, other = winner[port]
            key = (min(port[0], other[0]), max(port[0], other[0]))
            if key in emitted:
                continue
            emitted.add(key)
            g = pair_geom.get(key)
            if g is None:
                continue
            sig, chord, perp, _pa, _pb = g
            mutual = winner.get(other, (0.0, None))[1] == port
            ti, hi = key
            pt, ph = anchor_positions[anchors[ti]], anchor_positions[anchors[hi]]
            t, h = anchors[ti], anchors[hi]
            dt, dh = rep[t], rep[h]
            src, dst = node_ids[dt], node_ids[dh]
            if src == dst:
                continue
            # Canonical traversal: first = lower node ID (deterministic).
            # [Pass 28 / draft File 5] Parameter values must be binned in the
            # SAME node-ID frame that structural_similarity._pair_edge_sim's
            # flip logic ((a1 < a2) != (b1 < b2)) assumes. Binning by spatial
            # order instead makes reversal corrections fire on ~half of the
            # edges (measured frame disagreement was 50.5%).
            a_id, b_id = (src, dst) if src < dst else (dst, src)
            if src < dst:
                fi, si, pf, ps = ti, hi, pt, ph
            else:
                fi, si, pf, ps = hi, ti, ph, pt

            f_, s_ = anchors[fi], anchors[si]
            phi = math.atan2(ps[1] - pf[1], ps[0] - pf[0])
            al_f = _axial_wrap(theta[f_] - phi)
            al_s = _axial_wrap(theta[s_] - phi)
            turn = _axial_wrap(al_s - al_f)        # signed relative to lower->higher node ID
            asym = abs(_axial_wrap(al_f + al_s))   # 0 = co-circular

            # [P2] THE NAMED WRITER: the along-axis GAP and the PERPENDICULAR
            # OFFSET in MEDIAN SPACINGS (the V1 association field's own
            # length unit L), the co-circularity asymmetry and the SIGNED
            # AXIAL TURN in RADIANS; the bipole roles and the polarity
            # relation are CATEGORICAL lookup keys.
            gap_sp = chord / max(L, 1e-6)
            perp_sp = perp / max(L, 1e-6)
            rel = _polarity_relation(_pol_of(rep[f_]), _pol_of(rep[s_]))
            signal = sig * (1.0 if mutual
                            else P.ORIENTATION_CONTINUITY_MUTUAL_FLOOR)
            if rel == 0.0:
                signal *= P.ORIENTATION_CONTINUITY_POLARITY_FLIP_GAIN

            ep = _edge_params(
                "EDGE_ORIENTATION_CONTINUITY",
                {"dist": gap_sp, "perp": perp_sp, "cocirc_asym": asym,
                 "turn": turn, "tail_role": _role(fi), "head_role": _role(si),
                 "polarity_rel": rel},
                sign=1,
                confidence=float(clamp(signal, 0.0, 1.0)),
            )
            candidates.append((a_id, b_id, signal, (a_id, b_id), ep))
        self._process_channel(EdgeType.ORIENTATION_CONTINUITY, candidates)

    def _run_surface_grouping(
        self,
        dets: Tuple[Detection, ...],
        node_ids: Dict[Detection, int],
        anchor_positions: Optional[Dict[Any, Tuple[float, ...]]],
    ) -> None:
        """[v8] Similar surface properties link. Node rep is Position
        (not Color/Texture) so the resulting edges integrate with the
        spatial subgraph used by Rule 2 family resolution.

        Capability gate: fires only when at least two anchors share a
        common surface modality. Spread gate: skips all-equal signals
        so a uniform scene cannot lay a constant-signal clique."""
        if not anchor_positions:
            return

        colors: Dict[Any, Tuple[float, float, float]] = {}
        textures: Dict[Any, Tuple[float, float]] = {}
        for d in dets:
            if d.k == "Color" and len(d.sigma.quantized) >= 3:
                # [O2] The colour triple, by name: the opponent
                # luminance / red-green / blue-yellow, or the raw
                # red / green / blue on the legacy path.
                if getattr(d.sigma, "class_", "") == "COLOR_LEGACY":
                    colors[d.alpha] = slot_values(
                        d.sigma, "COLOR_LEGACY", ("r", "g", "b"))
                else:
                    colors[d.alpha] = slot_values(
                        d.sigma, "COLOR", ("lum", "rg", "by"))
            elif d.k == "Texture" and len(d.sigma.quantized) >= 2:
                # [O2] The texture pair, by name.
                textures[d.alpha] = slot_values(
                    d.sigma, "TEXTURE", ("roughness", "isotropy"))

        if len(colors) < 2 and len(textures) < 2:
            return

        rep: Dict[Any, Detection] = {}
        for d in dets:
            if d.k != "Position":
                continue
            if d.alpha not in anchor_positions:
                continue
            if d.alpha not in rep:
                rep[d.alpha] = d
        anchors = sorted(rep.keys(), key=serialize_value)
        if len(anchors) < 2:
            return

        sigma_c = P.SURFACE_GROUPING_SIGMA_C
        sigma_t = P.SURFACE_GROUPING_SIGMA_T

        raw: List[Tuple[int, int, float, Tuple[Any, ...]]] = []
        for i in range(len(anchors)):
            for j in range(i + 1, len(anchors)):
                a, b = anchors[i], anchors[j]
                sig = 1.0
                have_any = False
                if a in colors and b in colors:
                    ca, cb = colors[a], colors[b]
                    d_c = math.sqrt(sum((x - y) ** 2 for x, y in zip(ca, cb)))
                    sig *= math.exp(-d_c / sigma_c)
                    have_any = True
                if a in textures and b in textures:
                    ta, tb = textures[a], textures[b]
                    d_t = math.sqrt(sum((x - y) ** 2 for x, y in zip(ta, tb)))
                    sig *= math.exp(-d_t / sigma_t)
                    have_any = True
                if not have_any:
                    continue
                if sig < P.SURFACE_GROUPING_MIN_SIGNAL:
                    continue
                src = node_ids[rep[a]]
                dst = node_ids[rep[b]]
                if src == dst:
                    continue
                a_id, b_id = (src, dst) if src < dst else (dst, src)
                # [Pass 22] First real EdgeParams for this channel. Every slot
                # is computed; the neutral 4.0 centres represent "this
                # modality is absent", which is a real absence, not a
                # fabricated constant.
                color_dist = None
                if a in colors and b in colors:
                    ca, cb = colors[a], colors[b]
                    d_c = math.sqrt(sum((x - y) ** 2 for x, y in zip(ca, cb)))
                    color_dist = d_c / max(sigma_c, 1e-6)
                tex_dist = None
                if a in textures and b in textures:
                    ta, tb = textures[a], textures[b]
                    d_t = math.sqrt(sum((x - y) ** 2 for x, y in zip(ta, tb)))
                    tex_dist = d_t / max(sigma_t, 1e-6)
                contrast_step = None
                if a in colors and b in colors:
                    lum_a = float(colors[a][0])
                    lum_b = float(colors[b][0])
                    contrast_step = abs(lum_a - lum_b) / (lum_a + lum_b + 1.0)
                L = self._step_l_med
                spatial_dist = None
                if L > 0.0:
                    pa_p = anchor_positions.get(a)
                    pb_p = anchor_positions.get(b)
                    if pa_p is not None and pb_p is not None and len(pa_p) >= 2 and len(pb_p) >= 2:
                        spatial_dist = math.log2(1.0 + math.hypot(
                            pb_p[0] - pa_p[0], pb_p[1] - pa_p[1]) / L)
                # [P2] THE NAMED WRITER: the colour and texture distances in
                # their own sigma units, the Weber contrast as a fraction and
                # the separation as a LOG2 SPACING.
                ep = _edge_params(
                    "EDGE_SURFACE_GROUPING",
                    {"color_dist": color_dist, "tex_dist": tex_dist,
                     "contrast_step": contrast_step,
                     "spatial_dist": spatial_dist},
                    sign=1,
                    confidence=float(clamp(sig, 0.0, 1.0)),
                )
                raw.append((a_id, b_id, sig, (a_id, b_id), ep))

        if not raw:
            return
        if len(raw) >= 2:
            sigs = [c[2] for c in raw]
            if max(sigs) - min(sigs) < P.SURFACE_GROUPING_SPREAD_MIN:
                return
        self._process_channel(EdgeType.SURFACE_GROUPING, raw)

    # -------------------------------------------------------------------------
    def _run_curvature(
        self,
        dets: Tuple[Detection, ...],
        node_ids: Dict[Detection, int],
        anchor_positions: Optional[Dict[Any, Tuple[float, ...]]] = None,
    ) -> None:
        """[v9] Adjacent boundary segments bend. Undirected.

        Groups ORIENTATION_SEGMENT detections by (chain_id_bin,
        entity), sorts by segment index (from rho), fires edges
        between consecutive segments. Closed regions wrap the last
        segment to the first via the `closed` rho flag.

        Signal = tanh(Δθ / CURVATURE_SIGMA_THETA). Straight
        continuations below CURVATURE_MIN_DELTA_THETA produce no edge.
        """
        buckets: Dict[
            Tuple[int, Any], List[Tuple[int, int, bool, Detection]]
        ] = defaultdict(list)
        for d in dets:
            if d.k != "Orientation":
                continue
            if d.sigma.class_ != "ORIENTATION_SEGMENT":
                continue
            q = d.sigma.quantized
            if len(q) < 4:
                continue
            chain_bin = int(q[2])
            entity: Any = None
            seg_idx = -1
            num_seg = 1
            closed = False
            for role in d.rho:
                if len(role) < 2:
                    continue
                if role[0] == "entity":
                    entity = role[1]
                elif role[0] == "segment":
                    seg_idx = int(role[1])
                elif role[0] == "num_segments":
                    num_seg = int(role[1])
                elif role[0] == "closed":
                    closed = bool(role[1])
            if entity is None or seg_idx < 0:
                continue
            buckets[(chain_bin, entity)].append(
                (seg_idx, num_seg, closed, d)
            )

        n_theta = P.ORIENTATION_N_THETA
        candidates: List[Tuple[int, int, float, Tuple[Any, ...]]] = []
        for _key, items in buckets.items():
            if len(items) < 2:
                continue
            items.sort(key=lambda x: (x[0], _detection_sort_key(x[3])))
            num_seg = items[0][1]
            closed = items[0][2]
            idx_to_item: Dict[int, Tuple[int, Detection]] = {
                si: (ns, d) for (si, ns, _c, d) in items
            }
            pairs: List[Tuple[int, int]] = []
            for si in sorted(idx_to_item.keys()):
                nxt = si + 1
                if nxt < num_seg:
                    pairs.append((si, nxt))
            if closed and num_seg >= 2:
                pairs.append((num_seg - 1, 0))

            for i, j in pairs:
                if i not in idx_to_item or j not in idx_to_item:
                    continue
                d_i = idx_to_item[i][1]
                d_j = idx_to_item[j][1]
                # [O2] The segment THETA, by name.
                theta_i = (
                    slot_value(d_i.sigma, "ORIENTATION_SEGMENT", "theta",
                              default=0.0) + 0.5
                ) * math.pi / n_theta
                theta_j = (
                    slot_value(d_j.sigma, "ORIENTATION_SEGMENT", "theta",
                              default=0.0) + 0.5
                ) * math.pi / n_theta
                d_theta = abs(theta_i - theta_j)
                d_theta = min(d_theta, math.pi - d_theta)
                if d_theta < P.CURVATURE_MIN_DELTA_THETA:
                    continue
                signal = math.tanh(d_theta / P.CURVATURE_SIGMA_THETA)
                if signal <= 0.0:
                    continue
                src = node_ids[d_i]
                dst = node_ids[d_j]
                if src == dst:
                    continue
                a_id, b_id = (src, dst) if src < dst else (dst, src)
                # [S5 contract] Signed turn binned in the lower->higher node ID
                # frame (key is ID-ordered; i, j are chain-order segment indices).
                turn_src, turn_dst = (
                    (theta_i, theta_j) if src < dst else (theta_j, theta_i)
                )
                turn = (turn_dst - turn_src + math.pi) % (2.0 * math.pi) - math.pi
                # [P2] THE NAMED WRITER: the SIGNED TURN in RADIANS (a full
                # 2*pi period), the CHORD LENGTH in WAVELENGTHS and the
                # CURVATURE RATE in radians per wavelength.
                #
                # [P2 / MIS-READ FIX] The chord length was a CONSTANT 4.0 --
                # a saturated slot that carried no information about the
                # pair -- and the curvature rate was the raw angle
                # difference over a fixed constant, i.e. not a rate at all.
                # Both are measured now: the chord between the two segments'
                # own positions, and the turn per unit chord.
                chord_px = 0.0
                if anchor_positions:
                    pi_a = _mean_position(
                        _extract_member_anchors(d_i), anchor_positions)
                    pi_b = _mean_position(
                        _extract_member_anchors(d_j), anchor_positions)
                    if pi_a is not None and pi_b is not None:
                        chord_px = _pos_dist(pi_a, pi_b)
                scale = max(self._step_extraction_scale, 1e-6)
                chord_w = chord_px / scale
                curv_rate = (abs(turn) / chord_w) if chord_w > 1e-6 else 0.0
                ep = _edge_params(
                    "EDGE_CURVATURE",
                    {"signed_turn": turn, "chord_len": chord_w,
                     "curv_rate": curv_rate},
                    sign=1,
                    confidence=float(clamp(signal, 0.0, 1.0)),
                )
                candidates.append((a_id, b_id, signal, (a_id, b_id), ep))
        self._process_channel(EdgeType.CURVATURE, candidates)

    def _bo_depth_constraints(self, dets):
        """[Stage 3] Border-ownership ordinal depth constraints.

        Decodes each BorderOwnership detection's signed direction bin
        (`_q_signed(v, 8)`: bins 4-7 mean the entity is the figure / nearer,
        bins 0-3 mean it is the ground / farther). A positive mean signed
        vote for (alpha, neighbor) means alpha is in front of neighbor.
        NOTE: the bin->figure mapping is inferred from the lens's signed_dir
        composition (richness difference and gravity); if the sign is
        inverted, swap near/far below (futurework6.md instruction 5).
        """
        acc = defaultdict(lambda: [0.0, 0])
        from perception.ownership import ownership_signed
        for d in dets:
            if d.k != "BorderOwnership" or len(d.sigma.quantized) < 2:
                continue
            nb = next((r[1] for r in d.rho if r[0] == "neighbor"), None)
            if nb is None or nb == d.alpha:
                continue
            v = ownership_signed(d)     # signed figure-ground bin centre
            acc[(d.alpha, nb)][0] += v; acc[(d.alpha, nb)][1] += 1
        out = []
        for (a, b), (s, n) in sorted(acc.items(), key=lambda kv: (repr(kv[0][0]), repr(kv[0][1]))):
            m = s / n
            if abs(m) >= P.DEPTH_BO_MIN_STRENGTH:
                near, far = (a, b) if m > 0 else (b, a)
                out.append((near, far, min(1.0, abs(m)), "border_ownership"))
        return out

    def _run_occlusion(
        self,
        dets: Tuple[Detection, ...],
        node_ids: Dict[Detection, int],
        anchor_positions: Optional[Dict[Any, Tuple[float, ...]]],
        t_junctions: Optional[
            List[Tuple[int, int, Tuple[float, float], float]]
        ] = None,
    ) -> None:
        """[v9/v10] Occluding region occludes occluded region. Directed.

        Source (a) 2D T-junctions: fires when t_junctions is a non-empty
        list. Each T-junction contributes a directed edge from top-bar
        region (occluding) to stem region (occluded).

        Source (b) 3D depth difference: fires ONLY when t_junctions is
        None (explicit modality gate). Two position-adjacent anchors
        with sufficient depth difference produce a directed edge from
        nearer (occluding) to farther (occluded).
        """
        rep_by_anchor: Dict[Any, Detection] = {}
        for d in dets:
            if d.k != "Position":
                continue
            if d.alpha not in rep_by_anchor:
                rep_by_anchor[d.alpha] = d
        candidates: List[Tuple[int, int, float, Tuple[Any, ...], EdgeParams]] = []

        # Source (a): 2D T-junctions (+ border ownership) via the Stage-3
        # ordinal depth DAG. All near/far constraints are merged, cycle-
        # invalidated (weakest edge removed), and laid out into longest-path
        # ranks before any OCCLUSION edge is emitted.
        if t_junctions:
            cons = [(top, stem, float(conf), "tjunction")
                    for stem, top, _pt, conf in t_junctions
                    if conf >= P.OCCLUSION_MIN_CONFIDENCE
                    and stem in rep_by_anchor and top in rep_by_anchor]
            cons += self._bo_depth_constraints(dets)
            kept, removed = resolve_depth_constraints(cons, P.DEPTH_SOURCE_TRUST)
            rank, layers = depth_layout(kept)
            self._depth_conflicts_step = len(removed)
            self._depth_conflicts_total += len(removed)
            for _a, _b, _w, src in removed:
                self._depth_removed_by_source[src] = self._depth_removed_by_source.get(src, 0) + 1
            for top, stem, conf, src in kept:
                if src != "tjunction" or top not in rep_by_anchor or stem not in rep_by_anchor:
                    continue
                s_, d_ = node_ids[rep_by_anchor[top]], node_ids[rep_by_anchor[stem]]
                if s_ == d_:
                    continue
                gap_bin = float(min(7, max(0, rank[stem] - rank[top] - 1)))     # 0 = adjacent layers
                stack_bin = float(min(7, layers[top] - 1))                      # comparable across components
                # [P2] THE NAMED WRITER: the depth slot is a LAYER COUNT and
                # the stack depth is a LAYER COUNT; source says which cue
                # fired (a categorical lookup key).
                ep = _edge_params(
                    "EDGE_OCCLUSION",
                    {"depth": gap_bin, "stack_depth": stack_bin, "source": 0.0,
                     "depth_units": None},
                    sign=-1,
                    confidence=float(clamp(conf, 0.0, 1.0)),
                )
                candidates.append((s_, d_, conf, (s_, d_), ep))

        # Source (b): 3D depth difference. Gated on t_junctions is None.
        if t_junctions is None and anchor_positions and len(anchor_positions) >= 2:
            anchors = sorted(rep_by_anchor.keys(), key=serialize_value)
            for i in range(len(anchors)):
                for j in range(i + 1, len(anchors)):
                    a, b = anchors[i], anchors[j]
                    pa = anchor_positions.get(a)
                    pb = anchor_positions.get(b)
                    if pa is None or pb is None or len(pa) < 3 or len(pb) < 3:
                        continue
                    if not (math.isfinite(pa[2]) and math.isfinite(pb[2])):
                        continue
                    d_xy = math.sqrt(
                        (pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2
                    )
                    if self._step_l_med > 0.0 and d_xy > 2.0 * self._step_l_med:
                        continue
                    dz = abs(pa[2] - pb[2])
                    if dz < P.OCCLUSION_3D_MIN_DEPTH_DIFF:
                        continue
                    near, far = (a, b) if pa[2] < pb[2] else (b, a)
                    signal = 1.0 - math.exp(
                        -dz / P.OCCLUSION_3D_SIGMA_DEPTH
                    )
                    if signal < P.OCCLUSION_MIN_CONFIDENCE:
                        continue
                    src = node_ids[rep_by_anchor[near]]
                    dst = node_ids[rep_by_anchor[far]]
                    if src == dst:
                        continue
                    dz_bin = float(min(7, max(0, int(round((dz / max(P.OCCLUSION_3D_SIGMA_DEPTH * 2.0, 1.0)) * 7.0)))))
                    # [Stage 3] Arity-3 payload: (dz_bin, stack_depth=8.0 = none,
                    # source=1.0 = depth). The 3D path has no DAG stack info, so
                    # slot 1 is explicitly 8.0 (absent) and slot 2 marks the
                    # depth source. The old conf_bin slot is dropped.
                    # [P2] THE NAMED WRITER: the 3-D path has no layer
                    # count, so the depth slot reads as the adjacent layer
                    # (0) and the stack as absent (8); the graded 3-D depth
                    # difference rides in its OWN slot, in depth units.
                    ep = _edge_params(
                        "EDGE_OCCLUSION",
                        {"depth": 0.0, "stack_depth": 8.0, "source": 1.0,
                         "depth_units": dz / max(
                             P.OCCLUSION_3D_SIGMA_DEPTH * 2.0, 1.0)},
                        sign=-1,
                        confidence=float(clamp(signal, 0.0, 1.0)),
                    )
                    candidates.append((src, dst, signal, (src, dst), ep))

        self._process_channel(EdgeType.OCCLUSION, candidates)

    # -------------------------------------------------------------------------
    # v32: Cross-lens same-anchor binder
    # -------------------------------------------------------------------------

    def _run_co_located(
        self, dets: Tuple[Detection, ...], node_ids: Dict[Detection, int]
    ) -> None:
        """CO_LOCATED: emit undirected edges between nodes of distinct
        informative lens types that fire at the same anchor.

        The edge encodes joint content at one spatial location: "this
        patch has this Color AND this Orientation." Two patches with the
        same Color but different Orientations produce different
        CO_LOCATED edges. The edge is content-bearing, not adjacency-
        bearing.

        Whitelist: informative lenses only (Color, Texture, Size,
        Orientation, Boundary). Position stays excluded (pure spatial
        scaffold); "Elongation" was a phantom and is dropped; Boundary
        is the enclosed-region detector and carries figure-ground
        contour content, so its co-occurrence is content, not a
        property of the extractor.

        Signal = geometric mean of the two endpoint z-scores, divided
        by the number of competing partners at the anchor ([T1.2R]
        density counterweight: max(1, k - 1) for k informative
        detections). An isolated pair (k = 2) is unchanged; a dense
        anchor's mass grows like k/2 instead of k(k-1)/2, so a
        Boundary-heavy patch cannot inflate the edge list past the
        per-channel cap without bound. Both endpoints must have fired
        meaningfully for the joint relation to be asserted. If either
        detection is marginal (z == 0), the edge collapses.
        """
        by_anchor: Dict[Any, List[Detection]] = defaultdict(list)
        for d in dets:
            if d.k in INFORMATIVE_LENS_TYPES:
                by_anchor[d.alpha].append(d)

        candidates: List[Tuple[int, int, float, Tuple[Any, ...]]] = []
        for anchor in sorted(by_anchor.keys(), key=serialize_value):
            group = sorted(by_anchor[anchor], key=_detection_sort_key)
            # [T1.2R] Density counterweight. k informative detections at
            # this anchor give k-1 competing partners per pair; the
            # divisor is 1 for an isolated pair, so the canonical case is
            # untouched and only dense anchors are damped.
            n_partners = float(max(1, len(group) - 1))
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    d1, d2 = group[i], group[j]
                    if d1.k == d2.k:
                        continue
                    src = node_ids[d1]
                    dst = node_ids[d2]
                    if src == dst:
                        continue
                    signal = (
                        math.sqrt(max(d1.z, 0.0) * max(d2.z, 0.0))
                        / n_partners
                    )
                    if signal <= 0.0:
                        continue
                    a_id, b_id = (src, dst) if src < dst else (dst, src)
                    sa, sb = _detection_scale(d1), _detection_scale(d2)
                    # UNSIGNED: the edge is undirected, so a signed ratio
                    # would depend on the d1/d2 pair order. abs(log2) is
                    # order-invariant.
                    scale_ratio = (
                        None if (sa <= 0.0 or sb <= 0.0)
                        else abs(math.log2(sa / sb))
                    )
                    salience = (min(d1.z, d2.z)
                                / max(max(d1.z, d2.z), 1e-6))
                    # [P2] THE NAMED WRITER: the size ratio as an ABS LOG2
                    # RATIO and the salience ratio as a FRACTION.
                    ep = _edge_params(
                        "EDGE_CO_LOCATED",
                        {"scale_ratio": scale_ratio, "salience_ratio": salience},
                        sign=1,
                        confidence=float(clamp(signal, 0.0, 1.0)),
                    )
                    candidates.append((a_id, b_id, signal, (a_id, b_id), ep))
        self._process_channel(EdgeType.CO_LOCATED, candidates)

    # -------------------------------------------------------------------------
    # v32: Orientation angle relations
    # -------------------------------------------------------------------------

    def _run_orientation_bend(
        self,
        dets: Tuple[Detection, ...],
        node_ids: Dict[Detection, int],
        anchor_positions: Optional[Dict[Any, Tuple[float, ...]]],
    ) -> None:
        """ORIENTATION_BEND: two ORIENTATION nodes at different anchors
        whose angle difference lies in [30°, 60°)."""
        self._run_orientation_angle_relation(
            dets, node_ids, anchor_positions,
            edge_type=EdgeType.ORIENTATION_BEND,
            angle_min=math.pi / 6.0,
            angle_max=math.pi / 3.0,
            angle_center=math.pi / 4.0,
            upper_inclusive=False,
        )

    def _run_orientation_junction(
        self,
        dets: Tuple[Detection, ...],
        node_ids: Dict[Detection, int],
        anchor_positions: Optional[Dict[Any, Tuple[float, ...]]],
    ) -> None:
        """ORIENTATION_JUNCTION: two ORIENTATION nodes at different anchors
        whose angle difference lies in [60°, 90°]. Upper bound is inclusive
        because d_theta saturates at π/2 after wrapping."""
        self._run_orientation_angle_relation(
            dets, node_ids, anchor_positions,
            edge_type=EdgeType.ORIENTATION_JUNCTION,
            angle_min=math.pi / 3.0,
            angle_max=math.pi / 2.0,
            angle_center=5.0 * math.pi / 12.0,
            upper_inclusive=True,
        )

    def _run_orientation_angle_relation(
        self,
        dets: Tuple[Detection, ...],
        node_ids: Dict[Detection, int],
        anchor_positions: Optional[Dict[Any, Tuple[float, ...]]],
        edge_type: EdgeType,
        angle_min: float,
        angle_max: float,
        angle_center: float,
        upper_inclusive: bool,
    ) -> None:
        """Shared implementation for ORIENTATION_BEND and ORIENTATION_JUNCTION.

        Emits undirected edges between pairs of ORIENTATION-class nodes at
        different anchors whose angle difference falls in
        [angle_min, angle_max] (or [angle_min, angle_max) if not inclusive).
        Signal = Gaussian on angle difference centered at angle_center,
        multiplied by a linear spatial decay matching ORIENTATION_CONTINUITY's
        tau_gap profile.
        """
        if not anchor_positions or len(anchor_positions) < 2:
            return
        # Spatial gate now reads the extraction scale, not _step_l_med: a
        # bend/junction is a local relation, and on a uniform patch grid
        # _step_l_med is the grid pitch, so TAU_GAP_FRAC*l_med (~64px) was a
        # 3-cell weld horizon with no hard cutoff (the blob bug).
        if self._step_extraction_scale <= 0.0:
            return

        rep: Dict[Any, Detection] = {}
        for d in dets:
            if d.k not in ("Orientation", "OrientationPol"):
                continue
            # [Option (b)] ORIENTATION_POL is orientation content here
            # too, as its own class (never merged into ORIENTATION).
            if d.sigma.class_ not in _orient_spine_classes():
                continue
            if d.alpha not in anchor_positions:
                continue
            if d.alpha not in rep:
                rep[d.alpha] = d
        anchors = sorted(rep.keys(), key=serialize_value)
        if len(anchors) < 2:
            return

        # [SPEED] The relation scan below is a DOUBLE LOOP over every
        # Orientation anchor, and it ran uncapped: the cost is quadratic in
        # the number of Orientation anchors, which the Gabor Orientation
        # path (one anchor per (region, orientation bin, scale) candidate)
        # can push into the thousands. The cap below bounds that cost.
        #
        # THE CAP IS NOT REACHED ON THE CORPUS: the measured maximum number
        # of Orientation anchors on the ETH-80 corpus at canon 256 is
        # reported by diagnostics/_speed_anchor_cap.py, and it is far below
        # the cap, so every pair is still examined and the emitted edges are
        # IDENTICAL to the uncapped version. The cap is a bound, not a
        # behaviour change: it only bites on an input that does not occur.
        #
        # The cap is a LOCAL constant of this module (not a parameter), with
        # a documented default. 4096 anchors is ~8.4M pair tests, which is
        # the most the double loop may ever cost.
        if len(anchors) > _ORIENTATION_ANGLE_RELATION_MAX_ANCHORS:
            anchors = anchors[:_ORIENTATION_ANGLE_RELATION_MAX_ANCHORS]

        n_theta = P.ORIENTATION_N_THETA
        sigma_theta = P.ORIENTATION_ANGLE_RELATION_SIGMA_THETA
        tau_gap = P.ORIENTATION_ANGLE_RELATION_KAPPA * self._step_extraction_scale

        candidates: List[Tuple[int, int, float, Tuple[Any, ...]]] = []
        for i in range(len(anchors)):
            for j in range(i + 1, len(anchors)):
                a, b = anchors[i], anchors[j]
                if a == b:
                    continue
                da = rep[a]
                db = rep[b]
                if len(da.sigma.quantized) < 1 or len(db.sigma.quantized) < 1:
                    continue
                # [O2] The orientation THETA, by name.
                # [U21] Shared family reader (GABOR theta is a bank index).
                theta_a = orientation_theta_radians(da.sigma)
                theta_b = orientation_theta_radians(db.sigma)
                d_theta = abs(theta_a - theta_b)
                d_theta = min(d_theta, math.pi - d_theta)
                if upper_inclusive:
                    if not (angle_min <= d_theta <= angle_max):
                        continue
                else:
                    if not (angle_min <= d_theta < angle_max):
                        continue
                pa = anchor_positions[a]
                pb = anchor_positions[b]
                dx = pb[0] - pa[0]
                dy = pb[1] - pa[1]
                dist = math.sqrt(dx * dx + dy * dy)
                sig_angle = math.exp(
                    -((d_theta - angle_center) / sigma_theta) ** 2
                )
                # Linear spatial decay over the LOCAL horizon
                # (ORIENTATION_ANGLE_RELATION_KAPPA * extraction_scale);
                # Gaussian decay was considered and rejected by the audit.
                sig_spatial = math.exp(-dist / (tau_gap + EPSILON))
                signal = sig_angle * sig_spatial
                if signal <= 0.0:
                    continue
                src = node_ids[da]
                dst = node_ids[db]
                if src == dst:
                    continue
                a_id, b_id = (src, dst) if src < dst else (dst, src)
                cls_name = (
                    "EDGE_ORIENTATION_BEND"
                    if edge_type == EdgeType.ORIENTATION_BEND
                    else "EDGE_ORIENTATION_JUNCTION"
                )
                # [P2] THE NAMED WRITER: the AXIAL ANGLE DIFFERENCE in
                # RADIANS, the BISECTOR ORIENTATION in RADIANS (axial) and
                # the separation in WAVELENGTHS.
                bisector_theta = ((theta_a + theta_b) / 2.0) % math.pi
                dist_w = dist / max(self._step_extraction_scale, 1e-6)
                ep = _edge_params(
                    cls_name,
                    {"angle_delta": d_theta, "bisector_theta": bisector_theta,
                     "spatial_dist": dist_w},
                    sign=1,
                    confidence=float(clamp(signal, 0.0, 1.0)),
                )
                candidates.append((a_id, b_id, signal, (a_id, b_id), ep))
        self._process_channel(edge_type, candidates)

    # Structural similarity is NOT a binder (§1.5/§4.12): it measures the graph
    # the binders produced instead of observing a new relation, so it emits no
    # edge and lives in structural_similarity.py as a pure operator. Removing
    # it from the edge taxonomy eliminates the order-2 contamination of
    # pattern-node neighborhoods (blockers.md I6).
    # -------------------------------------------------------------------------


# =============================================================================
# MODULE-LEVEL HELPERS
# =============================================================================






def _node_signature_hash_for(d: Detection) -> bytes:
    """[V9 B.19] Scale-aware pattern identity hash.

    Prepend the scale only for scale > 0. Scale-0 hashes are byte-identical
    to `_node_signature_hash(d.sigma)`, so every existing pattern node stays
    valid and no graph rebuild is required when call sites migrate.

    Declared adaptation vs the draft: `hash_64` returns an int, while this
    function's contract (and every sibling return) is bytes -- the scale
    branch is wrapped in `serialize_int` so one type crosses all branches.

    Dormant: 0 callers. The 30-site migration is pre-registered as the
    deliberate step that lands with the first producer of scale != 0
    detections; three of the sites hash a bare sigma and are ineligible.
    """
    base = _node_signature_hash(d.sigma)
    if d.scale == 0:
        return base
    return serialize_int(hash_64(serialize_tuple([
        serialize_bytes(base),
        serialize_int(d.scale),
    ])))


def _candidate_key_bytes(key: Tuple[Any, ...]) -> bytes:
    return serialize_tuple([serialize_value(v) for v in key])


def _neg_bytes(key: Tuple[Any, ...]) -> bytes:
    """Deterministic inverse key for heapq.nlargest tie-breaking by
    canonical bytes (lower bytes should win, so nlargest sees higher)."""
    return bytes(255 - b for b in _candidate_key_bytes(key))


def _conditional_from_history(
    hist_a: List[int],
    hist_b: List[int],
    condition_on_a_present: bool,
) -> Tuple[float, int]:
    """Return (P(B=1 | A=X), observations), where X is 1 or 0 depending on
    condition_on_a_present. Both histories are sliding windows of the same
    length."""
    T = min(len(hist_a), len(hist_b))
    if T == 0:
        return 0.0, 0
    n_hits = 0
    n_obs = 0
    for i in range(T):
        a_val = hist_a[i]
        b_val = hist_b[i]
        if condition_on_a_present:
            if a_val == 0:
                continue
        else:
            if a_val == 1:
                continue
        n_obs += 1
        n_hits += b_val
    if n_obs == 0:
        return 0.0, 0
    return n_hits / n_obs, n_obs


def _presence_fraction(hist: List[int]) -> float:
    if not hist:
        return 0.0
    return sum(hist) / len(hist)


def _pattern_intervals(hist: List[int]) -> Optional[Tuple[float, float]]:
    """Extract (mean, std) of inter-onset intervals from a presence history."""
    if len(hist) < 2:
        return None
    onsets: List[int] = []
    for i in range(1, len(hist)):
        if hist[i] == 1 and hist[i - 1] == 0:
            onsets.append(i)
    if len(onsets) < 2:
        return None
    intervals = [onsets[i + 1] - onsets[i] for i in range(len(onsets) - 1)]
    if not intervals:
        return None
    mean_i = sum(intervals) / len(intervals)
    var_i = sum((x - mean_i) ** 2 for x in intervals) / len(intervals)
    return mean_i, math.sqrt(max(0.0, var_i))


def _commensurability(ratio: float) -> float:
    """Score in [0, 1]; 1 = ratio near an integer."""
    if ratio <= 0.0:
        return 0.0
    nearest_int = max(1, round(ratio))
    deviation = abs(ratio - nearest_int) / max(1.0, nearest_int)
    return clamp(1.0 - deviation, 0.0, 1.0)
