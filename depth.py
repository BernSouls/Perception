"""perception/depth.py -- depth DAG helpers (Tarjan, constraint
resolution, spatial clustering).

MOVE ONLY from higher_lens_helpers.py (whole file, one concern).
"""
from __future__ import annotations

import math
from collections import defaultdict
import heapq
from typing import Any, Dict, List, Optional, Set, Tuple
from core.hashing import (  # noqa: F401
    serialize_value,
)
from core.types import (  # noqa: F401
    Detection,
)


# Original module docstring (kept as a comment so the module has
# exactly one docstring):

# Shared helpers for higher-lens construction and V2+ edge emitters.
# 
# Both binder.py (emitters) and derived_lenses.py (lenses) import from here,
# so the two modules cannot drift on how a member list is read out of
# Detection.rho or how a distance is measured.
# 
# Imports only from primitives and sensory_floor — no binder, no
# concept_graph — so it sits below both consumers in the dependency graph and
# cannot introduce an import cycle.
# 
# Every extractor here is TOTAL and ORDER-INDEPENDENT: a malformed role is
# skipped rather than raising, and results are ordered by the explicit ordinal
# carried in rho[2] (ties broken on canonical bytes) rather than by rho order.
# §9 determinism depends on that — a higher-tier edge's endpoints must not
# depend on dict or set iteration order.
def _extract_member_anchors(det: Detection) -> List[Any]:
    """Observation anchors of a higher-tier detection's members, in rho
    ordinal order."""
    pairs: List[Tuple[int, bytes, Any]] = []
    for role in det.rho:
        if role[0] != "member":
            continue
        if len(role) < 3 or not isinstance(role[2], int):
            continue
        pairs.append((role[2], serialize_value(role[1]), role[1]))
    pairs.sort(key=lambda p: (p[0], p[1]))
    return [a for _, _, a in pairs]

def _extract_node_ids_for_role(
    det: Detection, role_name: str
) -> List[int]:
    """Shared body of the four node-id extractors.

    Each carries a 1-tuple of node ids under `role_name` in rho[1] and its
    ordinal in rho[2]; the output is ordered by that ordinal. Factored out
    because the four differ ONLY in the role name — four copies of this
    loop would be four places to drift.
    """
    pairs: List[Tuple[int, int]] = []
    for role in det.rho:
        if role[0] != role_name:
            continue
        v = role[1]
        if not (isinstance(v, tuple) and len(v) == 1 and isinstance(v[0], int)):
            continue
        if len(role) < 3 or not isinstance(role[2], int):
            continue
        pairs.append((role[2], v[0]))
    pairs.sort(key=lambda p: p[0])
    return [n for _, n in pairs]

def _extract_member_node_ids(det: Detection) -> List[int]:
    """Pattern-node ids of members (B.6/B.7/B.8 role "member_node")."""
    return _extract_node_ids_for_role(det, "member_node")

def _extract_cc_node_ids(det: Detection) -> List[int]:
    """CurveComplex node ids of a Part (role "curveComplex")."""
    return _extract_node_ids_for_role(det, "curveComplex")

def _extract_content_node_ids(det: Detection) -> List[int]:
    """Content node ids of a Part (role "content_node")."""
    return _extract_node_ids_for_role(det, "content_node")

def _extract_part_node_ids(det: Detection) -> List[int]:
    """Part node ids of an ObjectShape (role "part_node")."""
    return _extract_node_ids_for_role(det, "part_node")

def _mean_position(
    member_anchors: List[Any],
    anchor_positions: Optional[Dict[Any, Tuple[float, ...]]],
) -> Optional[Tuple[float, ...]]:
    """Centroid of the members that actually have a position.

    Returns None when no member is positioned, or when the positioned
    members disagree on dimensionality — a ragged position table would make
    the centroid meaningless, and silently averaging it would put a
    higher-tier node at a position no observation supports.
    """
    anchor_positions = anchor_positions or {}
    positions = [
        anchor_positions[a] for a in member_anchors if a in anchor_positions
    ]
    if not positions:
        return None
    dims = len(positions[0])
    if any(len(p) != dims for p in positions):
        return None
    return tuple(
        sum(p[d] for p in positions) / len(positions) for d in range(dims)
    )

def _pos_dist(a: Tuple[float, ...], b: Tuple[float, ...]) -> float:
    """Euclidean distance over at most 3 dims (workspace is 2D/3D)."""
    dims = min(len(a), len(b), 3)
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(dims)))

def _anchor_pos(
    anchor: Any,
    anchor_positions: Optional[Dict[Any, Tuple[float, ...]]],
) -> Optional[Tuple[float, ...]]:
    if not anchor_positions:
        return None
    return anchor_positions.get(anchor)

def spatial_clustering(
    dets: List[Detection],
    anchor_positions: Optional[Dict[Any, Tuple[float, ...]]],
    radius: float,
) -> List[List[Detection]]:
    """Connected components by spatial proximity."""
    if not anchor_positions or not dets:
        return [dets] if dets else []
    components: List[List[Detection]] = []
    assigned: Set[Detection] = set()
    for d in dets:
        if d in assigned:
            continue
        pos = anchor_positions.get(d.alpha)
        if not pos:
            continue
        comp = [d]
        assigned.add(d)
        stack = [d]
        while stack:
            curr = stack.pop()
            p_curr = anchor_positions.get(curr.alpha)
            if not p_curr:
                continue
            for other in dets:
                if other in assigned:
                    continue
                p_other = anchor_positions.get(other.alpha)
                if not p_other:
                    continue
                if math.hypot(p_curr[0] - p_other[0], p_curr[1] - p_other[1]) <= radius:
                    comp.append(other)
                    assigned.add(other)
                    stack.append(other)
        components.append(comp)
    return components

def _k(x):
    return repr(x)

def tarjan_scc(nodes, adj):
    index, low, on, stack, out, counter = {}, {}, set(), [], [], 0
    for root in nodes:
        if root in index:
            continue
        index[root] = low[root] = counter; counter += 1
        stack.append(root); on.add(root)
        work = [(root, iter(adj.get(root, ())))]
        while work:
            v, it = work[-1]
            advanced = False
            for w in it:
                if w not in index:
                    index[w] = low[w] = counter; counter += 1
                    stack.append(w); on.add(w)
                    work.append((w, iter(adj.get(w, ()))))
                    advanced = True
                    break
                if w in on:
                    low[v] = min(low[v], index[w])
            if advanced:
                continue
            work.pop()
            if work:
                low[work[-1][0]] = min(low[work[-1][0]], low[v])
            if low[v] == index[v]:
                comp = []
                while True:
                    w = stack.pop(); on.discard(w); comp.append(w)
                    if w == v:
                        break
                out.append(comp)
    return out

def resolve_depth_constraints(constraints, trust=None):
    """constraints: (near, far, weight, source); near is in front of far.
    Effective weight = weight * trust[source]. Returns (kept, removed) with RAW
    weights; deterministic. Exact ties remove the alphabetically-first source."""
    trust = trust or {}
    merged = {}
    for near, far, w, src in constraints:
        if near == far:
            continue
        eff = float(w) * float(trust.get(src, 1.0))
        cur = merged.get((near, far))
        if cur is None or (eff, src) > (cur[0], cur[1]):
            merged[(near, far)] = (eff, src, float(w))
    removed = []
    while True:
        nodes = sorted({n for e in merged for n in e}, key=_k)
        adj = defaultdict(list)
        for a, b in sorted(merged, key=lambda e: (_k(e[0]), _k(e[1]))):
            adj[a].append(b)
        cyc = [c for c in tarjan_scc(nodes, adj) if len(c) > 1]
        if not cyc:
            break
        for comp in sorted(cyc, key=lambda c: min(_k(x) for x in c)):
            cs = set(comp)
            inside = [e for e in merged if e[0] in cs and e[1] in cs]
            if inside:
                v = min(inside, key=lambda e: (merged[e][0], merged[e][1], _k(e[0]), _k(e[1])))
                removed.append((v[0], v[1], merged[v][2], merged[v][1]))
                del merged[v]
    kept = [(a, b, raw, s) for (a, b), (_e, s, raw) in
            sorted(merged.items(), key=lambda kv: (_k(kv[0][0]), _k(kv[0][1])))]
    return kept, removed

def depth_layout(kept):
    """-> (rank, layers). rank: longest-path layer, 0 = nearest, comparable ONLY
    within a weakly connected component. layers[n]: number of layers in n's
    component (stack depth), which is comparable across components."""
    preds, succs, und, nodes = defaultdict(list), defaultdict(list), defaultdict(set), set()
    for a, b, _, _ in kept:
        succs[a].append(b); preds[b].append(a); und[a].add(b); und[b].add(a); nodes |= {a, b}
    indeg = {n: len(preds[n]) for n in nodes}
    rank = {n: 0 for n in nodes}
    ready = [(_k(n), n) for n in nodes if indeg[n] == 0]
    heapq.heapify(ready)
    while ready:
        _, n = heapq.heappop(ready)
        for m in succs[n]:
            rank[m] = max(rank[m], rank[n] + 1)
            indeg[m] -= 1
            if indeg[m] == 0:
                heapq.heappush(ready, (_k(m), m))
    layers, seen = {}, set()
    for n in sorted(nodes, key=_k):
        if n in seen:
            continue
        members, st = [], [n]
        seen.add(n)
        while st:
            x = st.pop(); members.append(x)
            for y in sorted(und[x], key=_k):
                if y not in seen:
                    seen.add(y); st.append(y)
        L = max(rank[m] for m in members) + 1
        for m in members:
            layers[m] = L
    return rank, layers

