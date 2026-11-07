"""perception/observation_2d.py -- extracted from sensory_floor.py (Move 3a).

MOVE ONLY: whole top-level statements, verbatim. The re-export
shim in sensory_floor.py keeps `from sensory_floor import X`
working until the importer repoint retires it.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.ndimage import zoom
import math
import logging
from typing import Any, Dict, List, Optional, Tuple, FrozenSet
import parameters as P

logger = logging.getLogger(__name__)
from core.hashing import (  # noqa: F401
    clamp,
)
from core.registries import (  # noqa: F401
    b1_class_name,
    b4_class_name,
)
from core.spatial import (  # noqa: F401
    SpatialHash,
)
from core.types import (  # noqa: F401
    TypedSignature,
)
from perception.polar import (  # noqa: F401
    _angle_diff,
    _axial_angle_diff,
)
from perception.rich import (  # noqa: F401
    _compute_coarseness,
    _compute_regularity,
    compute_texture_rich,
    texture_rich_population,
)
from perception.contour_structure import (  # noqa: F401
    V1RetinotopicLattice,
    _detect_t_junctions,
    _extract_association_field_contours,
    _front_end_scale,
    _local_edge_at,
)
from perception.filters import (  # noqa: F401
    GaborBank,
    _conv2d,
    color_constancy,
    color_contrast,
    compute_opponent_retinex,
    compute_responses,
    luminance_adapt,
    opponent_channels,
    split_on_off,
)
from perception.contour_field import cache_v1_responses
from perception.observation import (  # noqa: F401
    Observation,
)
from perception.rich import (  # noqa: F401
    _B1_SLOT_WEIGHTS_BY_CLASS,
    b1_region_signature,
    b1_register,
    b4_register,
    compute_responses_b1,
)

def _window_mean(arr: Any, y0: int, y1: int, x0: int, x1: int) -> Optional[float]:
    """Mean of `arr` over a window, or None when the window is empty.

    [O3] A per-region material quantity is a statistic of the region's own
    window, not the value of the single pixel nearest its centroid: one pixel
    is a sample of noise (a texture, a highlight, an anti-aliasing step), and
    two regions that look alike can land on opposite sides of it. None means
    "no measurement", so a caller keeps its own fallback instead of reading a
    fabricated zero.
    """
    a = np.asarray(arr)
    win = a[y0:y1, x0:x1]
    if win.size == 0:
        return None
    v = float(np.mean(win))
    return v if math.isfinite(v) else None


def _window_circular_mean(arr: Any, y0: int, y1: int, x0: int, x1: int,
                          period: float = 2.0 * math.pi) -> Optional[float]:
    """Circular mean of an angular `arr` over a window, or None.

    Averaging angles linearly is wrong (pi and -pi are neighbours, 0 and 2*pi
    are the same angle). The mean is taken on the unit circle and returned in
    [0, period). [O3] This is the window statistic the retinex hue and the
    ON/OFF phase planes use instead of a single pixel's phase.
    """
    a = np.asarray(arr, dtype=np.float64)
    win = a[y0:y1, x0:x1]
    if win.size == 0:
        return None
    z = np.mean(np.exp(1j * (win * (2.0 * math.pi / period))))
    if not np.isfinite(z.real) or not np.isfinite(z.imag) or abs(z) <= 1e-12:
        return None
    ang = math.atan2(z.imag, z.real) * (period / (2.0 * math.pi))
    return float(ang % period)


def _window_circular_mean_channels(arr: Any, y0: int, y1: int, x0: int, x1: int,
                                   period: float = 2.0 * math.pi) -> Any:
    """Per-channel circular mean over a window for an (..., H, W) angle map.

    Returns an array of the leading shape; channels whose resultant vanishes
    are left at 0.0 (their phase is undefined, never a fabricated angle).
    """
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim < 2:
        return np.zeros((), dtype=np.float32)
    win = a[..., y0:y1, x0:x1]
    if win.size == 0:
        return np.zeros(a.shape[:-2], dtype=np.float32)
    z = np.mean(np.exp(1j * (win * (2.0 * math.pi / period))),
                axis=(-1, -2))
    out = np.angle(z) * (period / (2.0 * math.pi))
    out = np.where(np.abs(z) > 1e-12, out % period, 0.0)
    return out.astype(np.float32)


def _memo_conv2d(bank, simple, src_img):
    """A `conv2d` for ``compute_responses_b1`` that REUSES an already
    computed raw Gabor triple.

    [SPEED / TRAP 5] ``perception.rich.compute_responses_b1`` convolves the
    bank on a plane; ``perception.filters.compute_responses`` convolves the
    SAME bank on the SAME plane, with the SAME convolution routine
    (``perception.filters._conv2d``) and the SAME kernels, so the two
    ``simple`` arrays are BIT-IDENTICAL.  Running the bank twice therefore
    buys nothing: this closure returns the already-computed plane for each
    kernel and removes one full bank pass (n_theta * n_sf * n_phase
    convolutions) per image.

    The memo is keyed by the kernel's own bytes, so a plane can only be
    returned for the exact kernel it was computed from, and it is scoped to
    one source image: if a different image is ever passed, the closure falls
    back to the real convolution instead of returning the wrong plane.

    This is value-preserving by construction: the returned plane is the very
    array the bank produced, not a recomputation of it.
    """
    memo: Dict[bytes, Any] = {}
    nt, nsf, nph = bank.kernels.shape[:3]
    for it in range(nt):
        for isf in range(nsf):
            for ip in range(nph):
                memo[bank.kernels[it, isf, ip].tobytes()] = simple[it, isf, ip]

    def _conv(img, kernel):
        if img is not src_img:
            return _conv2d(img, kernel)
        return memo[kernel.tobytes()]

    return _conv


def _polyline_shape_invariants(pts, is_closed: bool) -> Dict[str, float]:
    """The shape invariants of one region, MEASURED from its own polyline.

    [SHAPE HONESTY] These five used to be literal constants in the region
    dict ("circularity 1.0 if closed else 0.2", "roughness 0.05",
    "isotropy 0.2", "elongation 0.2 / 0.8", "area = length * 2") -- a
    fabricated shape signal in a shape-perception system, and the boundary,
    texture and region-shape lenses read them. They are computed here from
    the polyline itself:

      area        the shoelace area, in px^2
      circularity the isoperimetric ratio 4*pi*A / L^2 (1 = a circle)
      complexity  the total absolute turn in full turns (|T| / 2*pi)
      roughness   the mean |turn| per vertex, over pi (0 = smooth)
      isotropy    the directional isotropy of the segment directions
                  (|mean of the doubled-angle unit vectors|; 1 = isotropic)
      elongation  sqrt(lambda2 / lambda1) of the second-moment matrix
                  (1 = round, matching the symbolic observation's
                  convention, so the default 1.0 means "no anisotropy")
    """
    n = len(pts)
    if n < 3:
        return {"area": 0.0, "circularity": 0.0, "complexity": 0.0,
                "roughness": 0.0, "isotropy": 0.0, "elongation": 1.0}
    xs = [float(p[0]) for p in pts]
    ys = [float(p[1]) for p in pts]
    # The shoelace area and the perimeter.
    area2 = 0.0
    perim = 0.0
    seg_dirs: List[float] = []
    m = n if is_closed else n - 1
    for i in range(m):
        j = (i + 1) % n
        area2 += xs[i] * ys[j] - xs[j] * ys[i]
        dx = xs[j] - xs[i]
        dy = ys[j] - ys[i]
        d = math.hypot(dx, dy)
        perim += d
        if d > 1e-9:
            seg_dirs.append(math.atan2(dy, dx))
    area = abs(area2) * 0.5
    # The turning at each interior vertex (and the closing vertex, when the
    # contour is closed).
    turns: List[float] = []
    k = n if is_closed else n - 1
    for i in range(1, n - 1):
        ax = xs[i] - xs[i - 1]
        ay = ys[i] - ys[i - 1]
        bx = xs[i + 1] - xs[i]
        by = ys[i + 1] - ys[i]
        la = math.hypot(ax, ay)
        lb = math.hypot(bx, by)
        if la < 1e-9 or lb < 1e-9:
            continue
        cross = ax * by - ay * bx
        dot = ax * bx + ay * by
        turns.append(abs(math.atan2(cross, dot)))
    # The second-moment matrix of the boundary points (the inertia ellipse).
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = syy = sxy = 0.0
    for x, y in zip(xs, ys):
        dx = x - mx
        dy = y - my
        sxx += dx * dx
        syy += dy * dy
        sxy += dx * dy
    sxx /= n
    syy /= n
    sxy /= n
    tr = sxx + syy
    det = sxx * syy - sxy * sxy
    disc = max(0.0, tr * tr * 0.25 - det)
    l1 = tr * 0.5 + math.sqrt(disc)
    l2 = tr * 0.5 - math.sqrt(disc)
    elong = math.sqrt(max(0.0, l2) / l1) if l1 > 1e-12 else 1.0
    # The directional isotropy of the segment directions. The directions are
    # AXIAL (a segment and its reverse are the same direction), so the
    # doubled-angle resultant measures their AGREEMENT: 1 = every segment
    # shares one direction, 0 = they are spread. Isotropy is its complement
    # (1 = the directions are spread, i.e. the boundary has no preferred
    # direction), which is the convention the symbolic observation uses
    # (1 = isotropic).
    iso = 0.0
    if seg_dirs:
        sx = sum(math.sin(2.0 * a) for a in seg_dirs)
        sy = sum(math.cos(2.0 * a) for a in seg_dirs)
        iso = 1.0 - math.hypot(sx, sy) / len(seg_dirs)
    total_turn = sum(turns)
    return {
        "area": float(area),
        "circularity": (float(4.0 * math.pi * area / (perim * perim))
                         if perim > 1e-9 else 0.0),
        "complexity": float(total_turn / (2.0 * math.pi)),
        "roughness": float((total_turn / len(turns)) / math.pi
                           if turns else 0.0),
        "isotropy": float(iso),
        "elongation": float(elong),
    }


class Observation2D(Observation):
    """2D Image observation container supporting two-stream biological vision:
    dense retinotopic hypercolumns + continuous association-field contours.

    Two construction paths:
      - pixel-primary (``from_image``): runs the two-stream front-end
        (V1 lattice + contour tracer), merges into ``all_regions``,
        and passes the result to the constructor.
    """

    # [Stage 5] Optional dataset view metadata (e.g. ETH-80 azimuth /
    # elevation / instance / label). Set by the data driver, e.g.
    # ``obs.view_meta = parse_eth80_view(path)``. When set (and
    # P.ASPECT_ENABLED), Pipeline.step feeds it to the AspectGraph.
    view_meta: Optional[Dict[str, Any]] = None  # Class-level default

    def __init__(
        self,
        regions: List[Dict[str, Any]],
        image_shape: Tuple[int, int],
        flag_indices: Optional[List[int]] = None,
        effect_indices: Optional[List[int]] = None,
        contours: Optional[List[Dict[str, Any]]] = None,
    ):
        self.view_meta = None  # [Stage 5] instance-level view metadata
        self._regions = list(regions)
        self._N = len(self._regions)
        self._h, self._w = image_shape
        # [O1] THE PIXEL CONSTANTS BELOW GO THROUGH THE FRONT-END SCALE.
        # The region adjacency graph used bare pixel constants (2 px of
        # hypercolumn overlap, 10 px of contour margin, 4 px of contour
        # proximity) while the rest of the front end scales every pixel
        # threshold by _front_end_scale(h, w).  At 512^2 those constants
        # are half-size relative to the structure; at 128^2 they are
        # double.  The 256-vs-512 contract therefore held for the contour
        # extraction and NOT for the region graph.
        self._fe_scale = _front_end_scale(self._h, self._w)
        self._flag = sorted(flag_indices) if flag_indices else []
        self._effect = sorted(effect_indices) if effect_indices else []
        self._contours = list(contours) if contours is not None else []

        # [Stage 1.2] Explicit two-stream categorization:
        # structural = boundary contours/junctions, surface = hypercolumns.
        self._structural_anchors: FrozenSet[int] = frozenset(
            i for i, r in enumerate(self._regions) if r.get("is_boundary", False)
        )
        self._surface_anchors: FrozenSet[int] = frozenset(
            i for i, r in enumerate(self._regions) if not r.get("is_boundary", False)
        )

        # --- Pairwise distance & adjacency ---
        self._dist: List[List[float]] = [[float("inf")] * self._N for _ in range(self._N)]
        self._adj: List[List[int]] = [[0] * self._N for _ in range(self._N)]
        # [Stage 1.2] (contour_anchor, surface_anchor, signal) contacts used
        # by Binder._run_stream_binding to emit cross-stream PART_CONTENT.
        self._contour_surface_contacts: List[Tuple[int, int, float]] = []

        if self._N > 0:
            cents = np.array([r["centroid"] for r in self._regions], dtype=np.float64)
            rads = np.array([r["radius_px"] for r in self._regions], dtype=np.float64)
            is_bnd = np.array([bool(r.get("is_boundary", False)) for r in self._regions], dtype=bool)

            dx = cents[:, 0, None] - cents[:, 0]
            dy = cents[:, 1, None] - cents[:, 1]
            dist_mat = np.hypot(dx, dy)
            self._dist = dist_mat.tolist()
            for i in range(self._N):
                self._dist[i][i] = 0.0

            adj_mat = np.zeros((self._N, self._N), dtype=np.int32)

            hyper_idx = np.where(~is_bnd)[0]
            bnd_idx = np.where(is_bnd)[0]

            # A. Hypercolumn-to-hypercolumn contact (radial RF overlap + 2px
            # at the reference resolution; the 2 px goes through the
            # front-end scale so the contact rule is resolution-free)
            if len(hyper_idx) > 0:
                h_cents = cents[hyper_idx]
                h_rads = rads[hyper_idx]
                h_dx = h_cents[:, 0, None] - h_cents[:, 0]
                h_dy = h_cents[:, 1, None] - h_cents[:, 1]
                h_dist = np.hypot(h_dx, h_dy)
                h_contact_tol = h_rads[:, None] + h_rads[None, :] + 2.0 * self._fe_scale
                h_adj = (h_dist <= h_contact_tol) & (h_dist > 1e-6)
                for ii, orig_i in enumerate(hyper_idx):
                    for jj, orig_j in enumerate(hyper_idx):
                        if h_adj[ii, jj]:
                            adj_mat[orig_i, orig_j] = 1

            # B. Contour contact using polyline geometry (not centroids)
            if len(bnd_idx) > 0:
                # [Stage 1.2] Resolve polylines safely: primary path reads
                # self._contours via the region's _contour_idx; falls back to
                # the region's own boundary_polyline (region-built obs).
                # NOTE: the draft gated this block on `and self._contours`,
                # which would silently disable all contour adjacency for
                # from_regions() observations lacking a contours list.
                bnd_polys: List[Optional[np.ndarray]] = []
                for orig_b in bnd_idx:
                    r_b = self._regions[orig_b]
                    ci = r_b.get("_contour_idx")
                    poly_raw = None
                    if ci is not None and 0 <= ci < len(self._contours):
                        poly_raw = self._contours[ci].get("polyline")
                    if not poly_raw:
                        poly_raw = r_b.get("boundary_polyline")
                    bnd_polys.append(
                        np.array(poly_raw, dtype=np.float64)
                        if poly_raw and len(poly_raw) >= 2 else None
                    )

                # Contour-to-hypercolumn contact + stream-binding contacts
                for b_local, orig_b in enumerate(bnd_idx):
                    poly = bnd_polys[b_local]
                    if poly is None or len(poly) < 2:
                        continue
                    p_min = poly.min(axis=0) - 10.0 * self._fe_scale
                    p_max = poly.max(axis=0) + 10.0 * self._fe_scale
                    for orig_h in hyper_idx:
                        hc = cents[orig_h]
                        if not (p_min[0] <= hc[0] <= p_max[0] and p_min[1] <= hc[1] <= p_max[1]):
                            continue
                        d_vert = np.hypot(poly[:, 0] - hc[0], poly[:, 1] - hc[1])
                        min_d = float(d_vert.min())
                        hc_rad = float(rads[orig_h])
                        if min_d <= (hc_rad + 2.0 * self._fe_scale):
                            adj_mat[orig_b, orig_h] = 1
                            adj_mat[orig_h, orig_b] = 1
                            # [Stage 1.2] populate cross-stream contacts
                            sig = 1.0 - clamp(min_d / (hc_rad + 1e-6), 0.0, 1.0)
                            self._contour_surface_contacts.append(
                                (orig_b, orig_h, sig)
                            )

                # Contour-to-contour: any vertex within 4.0px at the
                # reference resolution (the constant goes through the
                # front-end scale, so the rule is resolution-free)
                for bi in range(len(bnd_idx)):
                    poly_i = bnd_polys[bi]
                    if poly_i is None or len(poly_i) < 2:
                        continue
                    orig_i = bnd_idx[bi]
                    pi_sub = poly_i[::2]
                    for bj in range(bi + 1, len(bnd_idx)):
                        poly_j = bnd_polys[bj]
                        if poly_j is None or len(poly_j) < 2:
                            continue
                        orig_j = bnd_idx[bj]
                        pj_sub = poly_j[::2]
                        d_pp = np.hypot(
                            pi_sub[:, 0, None] - pj_sub[None, :, 0],
                            pi_sub[:, 1, None] - pj_sub[None, :, 1],
                        )
                        if np.any(d_pp <= 4.0 * self._fe_scale):
                            adj_mat[orig_i, orig_j] = 1
                            adj_mat[orig_j, orig_i] = 1

            self._adj = adj_mat.tolist()

        # [v9] T-junction detection on the final region list.
        self._t_junctions = _detect_t_junctions(self._regions)

    def modality(self) -> str:
        return "2d"

    def num_atoms(self) -> int:
        return self._N

    def is_present(self, idx: int) -> bool:
        return 0 <= idx < self._N

    def get_entity_indices(self) -> List[int]:
        return list(range(self._N))

    def get_distance(self, i: int, j: int) -> float:
        if 0 <= i < self._N and 0 <= j < self._N:
            return self._dist[i][j]
        return float("inf") if i != j else 0.0

    def get_adjacency(self, i: int, j: int) -> int:
        if 0 <= i < self._N and 0 <= j < self._N:
            return self._adj[i][j]
        return 0

    def get_flag_indices(self) -> List[int]:
        return self._flag

    def get_effect_indices(self) -> List[int]:
        return self._effect

    # --- Contour accessors ---
    def get_luminance(self) -> Optional[Any]:
        """The adapted luminance plane consumed by the front end, or None.

        Prefers the plane the Gabor bank actually ran on (`_lum_adapted`,
        i.e. after luminance_adapt); falls back to the raw image's first
        channel for region-built observations that never adapted one.
        """
        lum = getattr(self, "_lum_adapted", None)
        if lum is not None:
            return lum
        raw = getattr(self, "_raw_image", None)
        if raw is not None and getattr(raw, "ndim", 0) >= 2:
            return raw[..., 0]
        return None

    def get_contours(self) -> List[Dict[str, Any]]:
        """Return the list of association-field contour dicts attached at
        construction time. Each dict has keys: polyline, theta_profile,
        strength_profile, length, is_closed."""
        return list(self._contours)

    def get_contour_polyline(self, contour_idx: int) -> Optional[List[Tuple[int, int]]]:
        """Return the pixel polyline for contour ``contour_idx``, or None."""
        if 0 <= contour_idx < len(self._contours):
            return self._contours[contour_idx].get("polyline")
        return None

    def get_contour_count(self) -> int:
        """Number of contours attached to this observation."""
        return len(self._contours)

    # --- [Stage 1.2] STREAM ACCESSORS ---
    def get_structural_anchors(self) -> FrozenSet[int]:
        """Boundary contour / junction anchors (structural stream)."""
        return self._structural_anchors

    def get_surface_anchors(self) -> FrozenSet[int]:
        """Hypercolumn anchors (surface stream)."""
        return self._surface_anchors

    def get_contour_surface_contacts(self) -> List[Tuple[int, int, float]]:
        """(contour_anchor, surface_anchor, signal) cross-stream contacts."""
        return list(self._contour_surface_contacts)

    # --- Micro-Lens Accessors queried by lenses.py ---
    def get_circularity(self, u: int) -> float:
        return self._regions[u]["circularity"] if 0 <= u < self._N else 0.5

    def get_complexity(self, u: int) -> float:
        return self._regions[u]["complexity"] if 0 <= u < self._N else 0.5

    def get_roughness(self, u: int) -> float:
        return self._regions[u]["roughness"] if 0 <= u < self._N else 0.0

    def get_isotropy(self, u: int) -> float:
        return self._regions[u]["isotropy"] if 0 <= u < self._N else 0.0

    def get_elongation(self, u: int) -> float:
        return self._regions[u]["elongation"] if 0 <= u < self._N else 1.0

    def get_radius(self, u: int) -> float:
        return self._regions[u]["radius_norm"] if 0 <= u < self._N else 0.1

    def get_extraction_scale(self) -> float:
        """[PHASE B] Step 4. Median region radius in PIXELS — the same units
        get_positions() returns (centroid pixel coordinates).

        Deliberately reads `_regions[i]["radius_px"]` rather than going
        through `get_radius()`, which returns `radius_norm` (a 0-1
        normalized value). Using the normalized form here would put the
        scale three orders of magnitude below the positions it has to be
        commensurate with.
        """
        radii = []
        for r in self._regions:
            rp = r.get("radius_px")
            if rp is None:
                continue
            try:
                v = float(rp)
            except (TypeError, ValueError):
                continue
            if math.isfinite(v) and v > 0.0:
                radii.append(v)
        if not radii:
            return 0.0
        radii.sort()
        mid = len(radii) // 2
        if len(radii) % 2 == 1:
            return radii[mid]
        return 0.5 * (radii[mid - 1] + radii[mid])

    def get_anchor_rgb(self, u: int) -> Optional[Tuple[float, float, float]]:
        return self._regions[u]["mean_rgb"] if 0 <= u < self._N else None

    def get_position(self, u: int) -> List[float]:
        # Macro-envelope-normalized address (rel_x/rel_y in [0, 1] mapped
        # back to [-1, 1]): view-stable spatial address per Patch 2.
        if 0 <= u < self._N:
            rx = self._regions[u].get("rel_x", 0.5)
            ry = self._regions[u].get("rel_y", 0.5)
            return [2.0 * rx - 1.0, 2.0 * ry - 1.0, 0.0]
        return [0.0, 0.0, 0.0]

    def get_positions(self) -> Dict[int, Tuple[float, float, float]]:
        return {i: (self._regions[i]["centroid"][0], self._regions[i]["centroid"][1], 0.0) for i in range(self._N)}

    def get_boundary_polyline(
        self, u: int
    ) -> Optional[List[Tuple[int, int]]]:
        """[v9] Ordered boundary polyline for region u, or None if the
        trace failed OR the polyline is below
        ORIENTATION_SEGMENT_MIN_POLYLINE."""
        if 0 <= u < self._N:
            poly = self._regions[u].get("boundary_polyline")
            if poly and len(poly) >= P.ORIENTATION_SEGMENT_MIN_POLYLINE:
                return poly
        return None

    def get_chain_id(self, u: int) -> Optional[int]:
        """[v9] Chain identity for region u. Always None under v10
        (chaining removed)."""
        if 0 <= u < self._N:
            return self._regions[u].get("chain_id")
        return None

    def get_chain_bounds(self, u: int) -> Optional[Tuple[float, float]]:
        """[v9] (chain_pos_start, chain_pos_end) for region u. Always
        (0.0, 1.0) under v10 (singleton chains)."""
        if 0 <= u < self._N:
            r = self._regions[u]
            return (
                r.get("chain_pos_start", 0.0),
                r.get("chain_pos_end", 1.0),
            )
        return None

    def get_t_junctions(
        self,
    ) -> List[Tuple[int, int, Tuple[float, float], float]]:
        """[v9] Precomputed T-junctions. Entries are
        (stem_region_idx, top_region_idx, contact_point, confidence)."""
        return list(getattr(self, "_t_junctions", []))

    def get_orientation(self, u: int) -> Optional[Tuple[float, float]]:
        """[v8] (angle, elongation) of the region's major axis.
        angle ∈ [0, π), elongation ∈ [0, 1]. None if unavailable."""
        if 0 <= u < self._N:
            r = self._regions[u]
            if "major_axis_angle" in r and "elongation" in r:
                return (float(r["major_axis_angle"]), float(r["elongation"]))
        return None

    def get_b4_sig(self, u: int) -> Optional[TypedSignature]:
        """Return the V2 second-order texture signature for region u.
        Bounds-checked: returns None for invalid index or missing signature."""
        if not isinstance(u, int) or u < 0 or u >= self._N:
            return None
        r = self._regions[u]
        sig = r.get("b4_sig")
        if sig is None:
            return None
        if isinstance(sig, TypedSignature):
            return sig
        nt = getattr(P, "GABOR_N_THETA", 12)
        nsf = getattr(P, "GABOR_N_SF", 5)
        cls_name = r.get("b4_class", f"TEXTURE_RICH_{nt}_{nsf}")
        return TypedSignature(
            kind="CANONICAL",
            class_=cls_name,
            relation="SECOND_ORDER_STATS",
            quantized=tuple(float(x) for x in sig),
        )

    def get_b1_sig(self, u: int) -> Optional[TypedSignature]:
        """Return the V1 hypercolumn orientation signature for region u.
        Bounds-checked: returns None for invalid index or missing signature."""
        if not isinstance(u, int) or u < 0 or u >= self._N:
            return None
        r = self._regions[u]
        sig = r.get("b1_sig")
        pop = r.get("b1_population")
        if sig is None:
            return None
        if isinstance(sig, TypedSignature):
            return sig
        nt = getattr(P, "GABOR_N_THETA", 12)
        nsf = getattr(P, "GABOR_N_SF", 5)
        cls_name = r.get("b1_class", f"ORIENTATION_RICH_{nt}_{nsf}")
        return TypedSignature(
            kind="CANONICAL",
            class_=cls_name,
            relation="HYPERCOLUMN_POPULATION",
            quantized=tuple(float(x) for x in sig),
            population=tuple(tuple(float(v) for v in axis) for axis in pop) if pop else (),
        )

    # ------------------------------------------------------------------
    # [V9 Phase 4b / draft B.6] Pixel-to-graph entry point + front-end
    # accessors. The region-primary constructor above is the live entry
    # (unchanged, byte-for-byte); these add the raw-image path alongside
    # it, per ruling (a). See the schema-delta report in the register:
    # the draft's assumed region keys already exist here.
    # ------------------------------------------------------------------

    @classmethod
    @classmethod
    def from_image(cls, raw_image, flag_indices=None, effect_indices=None,
                   contours=None, rich: bool = True):
        """Pixel-to-graph bridge: run the two-stream front-end.

        Stream 1 (V1 lattice): tessellates the visual field into
        overlapping hypercolumns via V1RetinotopicLattice.tile_with_features().
        Stream 2 (Association field): extracts curvilinear contours via
        _extract_association_field_contours().

        The two region sets are merged into ``all_regions`` and passed to
        the constructor as ``regions`` + ``contours``.

        When ``contours`` is passed explicitly (e.g. from a pre-computed
        cache), the contour region dicts are appended as boundary entities.

        ``rich=False`` skips the per-region rich-signature machinery
        (B1/B4 signatures, pooled Gabor, ON/OFF populations). Contours,
        the shock skeleton, and the aspect descriptor are computed from the
        SAME Gabor complex-energy map, so with rich=False the contours and
        descriptors are bit-identical to rich=True -- only the per-region
        response dicts are absent (get_gabor_responses returns None). The
        gate/cache paths only read contours, so they use rich=False for a
        ~3x speedup; ~73% of from_image's runtime was this machinery.

        Region-built instances (via from_regions) carry no stored
        responses: ``get_gabor_responses`` returns None for them.
        """
        h, w = raw_image.shape[:2]

        # --- Front-end: opponent colour + retinex ---
        opponent = opponent_channels(raw_image)
        corrected_rgb = color_constancy(raw_image, P.ILLUMINANT_TOP_FRAC)
        corrected_opp = opponent_channels(corrected_rgb)
        rg_contrast = color_contrast(
            corrected_opp[..., 1], P.COLOR_CONTRAST_RADIUS
        )
        by_contrast = color_contrast(
            corrected_opp[..., 2], P.COLOR_CONTRAST_RADIUS
        )
        retinex_res = compute_opponent_retinex(raw_image, extraction_scale=5.0)

        lum_raw = opponent[..., 0]
        lum = luminance_adapt(lum_raw)
        # [Scale-relative front end] Pixel constants in the contour tracer and
        # the Gabor cycles-per-pixel band are scaled to this frame's diagonal;
        # 1.0 at the 256x256 reference, so reference-resolution behaviour is
        # unchanged.
        _fe_scale = _front_end_scale(h, w)
        bank = GaborBank.build_default(_fe_scale)
        _lum_resp = compute_responses(bank, lum)
        simple = _lum_resp["simple"]
        complex_ = _lum_resp["complex"]
        phase = _lum_resp["phase"]
        # [O5] Hand the raw V1 triple to the field stage so PeakLens does not
        # recompute the same 120 convolutions. compute_responses is byte-
        # identical to contour_field.v1_responses for this (lum, scale).
        cache_v1_responses(lum, _fe_scale, simple, complex_, phase)

        # The remaining response populations feed ONLY the per-region rich
        # signatures, never the contour tracer (which reads `complex_`), so
        # they are skipped in contour-only mode.
        if rich:
            complex_pooled = gaussian_filter(
                complex_,
                sigma=(0, 0, P.DIVISIVE_POOL_SIGMA, P.DIVISIVE_POOL_SIGMA),
                mode="reflect",
            )

            # [B.46 adoption] Retinal ON/OFF substrate
            _on_img, _off_img = split_on_off(lum)
            _on_resp = compute_responses(bank, _on_img)
            _off_resp = compute_responses(bank, _off_img)

            # Attach Gabor/colour/retinex responses to hypercolumn regions
            _B1_AVAILABLE = all(name in globals() for name in (
                'compute_responses_b1', 'b1_register', 'b1_region_signature',
                'b1_class_name', '_B1_SLOT_WEIGHTS_BY_CLASS'))

            _lum_resp_b1 = None
            _on_resp_b1 = None
            _off_resp_b1 = None

            if _B1_AVAILABLE:
                b1_register(bank)
                # [SPEED / TRAP 5] The B1 populations convolve the SAME bank
                # on the SAME planes that compute_responses just convolved,
                # with the SAME convolution routine, so their raw `simple`
                # arrays are bit-identical. Passing a memoising `conv2d`
                # reuses the planes instead of running the bank a second
                # time: three bank passes per image instead of six.
                _lum_resp_b1 = compute_responses_b1(
                    bank, lum, conv2d=_memo_conv2d(bank, _lum_resp["simple"],
                                                   lum))
                _on_resp_b1 = compute_responses_b1(
                    bank, _on_img, conv2d=_memo_conv2d(bank, _on_resp["simple"],
                                                       _on_img))
                _off_resp_b1 = compute_responses_b1(
                    bank, _off_img, conv2d=_memo_conv2d(bank,
                                                         _off_resp["simple"],
                                                         _off_img))

                nt_b1, nsf_b1 = len(bank.theta), len(bank.sf)
                assert _lum_resp_b1["complex_local"].shape[:2] == (nt_b1, nsf_b1), \
                    "Bank shape mismatch"

        # --- Stream 1: V1 Retinotopic Lattice ---
        lattice = V1RetinotopicLattice()
        hypercolumn_regions = lattice.tile_with_features(
            h, w, complex_, bank.theta, bank.sf, lum, corrected_rgb
        )

        if rich:
            for r in hypercolumn_regions:
                cx = max(0, min(w - 1, int(round(r["centroid"][0]))))
                cy = max(0, min(h - 1, int(round(r["centroid"][1]))))
                win = max(P.TEXTURE_MIN_WINDOW, int(r["radius_px"]))
                y0, y1 = max(0, cy - win), min(h, cy + win + 1)
                x0, x1 = max(0, cx - win), min(w, cx + win + 1)

                window_slice = complex_[:, :, y0:y1, x0:x1]
                r["interior_variance"] = float(np.var(window_slice)) if window_slice.size > 0 else 0.0
                r["coarseness"] = _compute_coarseness(window_slice)
                r["regularity"] = _compute_regularity(window_slice)
                r["gabor_complex"] = window_slice.mean(axis=(2, 3)).astype(np.float32)

                r["gabor_simple"] = simple[:, :, :, cy, cx].copy()
                r["gabor_phase"] = phase[:, :, cy, cx].copy()
                r["gabor_complex_pooled"] = complex_pooled[:, :, y0:y1, x0:x1].mean(axis=(2, 3)).astype(np.float32)
                r["gabor_complex_on"] = _on_resp["complex"][:, :, y0:y1, x0:x1].mean(axis=(2, 3)).astype(np.float32)
                r["gabor_complex_off"] = _off_resp["complex"][:, :, y0:y1, x0:x1].mean(axis=(2, 3)).astype(np.float32)
                # [O3] WINDOW STATISTICS, NOT ONE PIXEL PER REGION.
                #
                # These planes used to be read at the single pixel nearest the
                # region's centroid. A pixel is a sample of noise -- a texture
                # grain, a highlight, an anti-aliasing step -- so two regions
                # with the SAME material could be described by opposite
                # samples, and a region straddling a boundary by whichever
                # side its centroid happened to land on. Every material plane
                # is now the statistic of the region's OWN window. Phase and
                # hue are circular, so they take a circular mean (a linear
                # mean of angles is meaningless); the rest take a mean.
                r["gabor_phase_on"] = _window_circular_mean_channels(
                    _on_resp["phase"], y0, y1, x0, x1)
                r["gabor_phase_off"] = _window_circular_mean_channels(
                    _off_resp["phase"], y0, y1, x0, x1)
                _opp_mean = [
                    _window_mean(opponent[..., _k], y0, y1, x0, x1)
                    for _k in range(3)]
                r["opponent"] = np.asarray(
                    [float(opponent[cy, cx, _k]) if _opp_mean[_k] is None
                     else _opp_mean[_k] for _k in range(3)], dtype=np.float32)
                _ret = [
                    _window_mean(retinex_res["lum_comp"], y0, y1, x0, x1),
                    _window_mean(retinex_res["rg_comp"], y0, y1, x0, x1),
                    _window_mean(retinex_res["by_comp"], y0, y1, x0, x1),
                    _window_mean(retinex_res["sat_norm"], y0, y1, x0, x1),
                    _window_circular_mean(retinex_res["hue"], y0, y1, x0, x1),
                ]
                _ret_point = (
                    float(retinex_res["lum_comp"][cy, cx]),
                    float(retinex_res["rg_comp"][cy, cx]),
                    float(retinex_res["by_comp"][cy, cx]),
                    float(retinex_res["sat_norm"][cy, cx]),
                    float(retinex_res["hue"][cy, cx]),
                )
                r["opponent_retinex"] = tuple(
                    _ret_point[_k] if _ret[_k] is None else float(_ret[_k])
                    for _k in range(5))
                _corr = [
                    _window_mean(corrected_opp[..., _k], y0, y1, x0, x1)
                    for _k in range(3)]
                r["corrected_opponent"] = np.asarray(
                    [float(corrected_opp[cy, cx, _k]) if _corr[_k] is None
                     else _corr[_k] for _k in range(3)], dtype=np.float32)
                _rgb_mean = [
                    _window_mean(corrected_rgb[..., _k], y0, y1, x0, x1)
                    for _k in range(3)]
                r["corrected_rgb"] = np.asarray(
                    [float(corrected_rgb[cy, cx, _k]) if _rgb_mean[_k] is None
                     else _rgb_mean[_k] for _k in range(3)], dtype=np.float32)
                _rgc = _window_mean(rg_contrast, y0, y1, x0, x1)
                _byc = _window_mean(by_contrast, y0, y1, x0, x1)
                r["rg_contrast"] = (float(rg_contrast[cy, cx])
                                    if _rgc is None else float(_rgc))
                r["by_contrast"] = (float(by_contrast[cy, cx])
                                    if _byc is None else float(_byc))
                r["local_edge"] = _local_edge_at(lum, cy, cx)

                # [SPEED / TRAP 4] The B1/B4 rich block is NOT computed
                # here. It used to be, and the all-regions loop below
                # computed it AGAIN for every hypercolumn, so every rich
                # region paid for the 433-slot B1 signature and the
                # second-order B4 texture signature TWICE per frame.
                #
                # VALUE IDENTITY: the all-regions loop recomputes
                # r["b1_sig"], r["b1_class"], r["b1_stats"], r["b4_sig"] and
                # r["b4_class"] for every region for which
                # `not r.get("is_boundary", False)` -- which is exactly the
                # hypercolumn set, because the contour regions are the only
                # boundary regions and they never received a B1/B4 block
                # here in the first place. The all-regions loop therefore
                # OVERWRITES every value this block would have written, so
                # dropping it leaves the region dicts byte-identical. (The
                # two blocks even use different windows -- this one is
                # uncapped in radius, the all-regions loop caps the radius
                # at 32 px -- which is precisely why the SECOND write is the
                # one that survives today, and why the first is dead work.)
                #
                # `b4_register` is still called by the all-regions loop, so
                # the TEXTURE_RICH class stays registered exactly as before.

        # --- Stream 2: Association Field Contours ---
        if contours is None:
            contours = _extract_association_field_contours(
                complex_, bank.theta, bank.sf, h, w
            )

        lattice_rad = hypercolumn_regions[0]["radius_px"] if hypercolumn_regions else 8.0
        diag_ref = math.sqrt(h * h + w * w)

        contour_regions: List[Dict[str, Any]] = []
        for c_idx, c_info in enumerate(contours):
            pts = c_info["polyline"]
            if len(pts) < 3:
                continue
            pts_arr = np.array(pts, dtype=np.float64)
            cy = float(np.mean(pts_arr[:, 1]))
            cx = float(np.mean(pts_arr[:, 0]))

            mid_idx = len(pts) // 2
            sample_x, sample_y = pts[mid_idx]
            c_int_x = max(0, min(w - 1, int(round(sample_x))))
            c_int_y = max(0, min(h - 1, int(round(sample_y))))

            dists = np.hypot(pts_arr[:, 0] - cx, pts_arr[:, 1] - cy)
            rad_px = max(2.0, float(np.max(dists)))
            _shape = _polyline_shape_invariants(pts, bool(c_info["is_closed"]))

            contour_regions.append({
                "anchor_id": len(hypercolumn_regions) + c_idx,
                "centroid": (float(cx), float(cy)),
                "radius_px": float(rad_px),
                "radius_norm": float(rad_px / (0.35 * diag_ref + 1e-6)),
                "rel_position": (float((cx / w) * 2.0 - 1.0), float((cy / h) * 2.0 - 1.0)),
                "rel_x": float(cx / w),
                "rel_y": float(cy / h),
                # [SHAPE HONESTY] These five were LITERAL CONSTANTS
                # ("circularity 1.0 if closed else 0.2", "roughness 0.05",
                # "isotropy 0.2", "elongation 0.2/0.8", "area = length * 2"):
                # fabricated shape in a shape-perception system, read by the
                # boundary, texture and region-shape lenses. They are now
                # MEASURED from the region's own polyline (see
                # `_polyline_shape_invariants`).
                "area": _shape["area"],
                "circularity": _shape["circularity"],
                "complexity": _shape["complexity"],
                "roughness": _shape["roughness"],
                "isotropy": _shape["isotropy"],
                "elongation": _shape["elongation"],
                "major_axis_angle": float(c_info["theta_profile"][0]) if c_info["theta_profile"] else 0.0,
                "mean_rgb": tuple(float(x) for x in corrected_rgb[c_int_y, c_int_x][:3]),
                "boundary_polyline": c_info["polyline"],
                # [Stage 1.2] index into the contours list for robust
                # polyline resolution in Observation2D.__init__
                "_contour_idx": c_idx,
                "is_boundary": True,
                "sample_xy": (c_int_x, c_int_y),
            })

        all_regions = hypercolumn_regions + contour_regions

        # 4. Gabor & Retinex sampling across all anchors
        # Per-region response dictionaries (rich mode only). Contours, the
        # shock skeleton and the aspect descriptor do not read these.
        if rich:
            for r in all_regions:
                if r.get("is_boundary", False):
                    cx_s, cy_s = r["sample_xy"]
                    cx, cy = cx_s, cy_s
                    win = min(int(r["radius_px"]), int(lattice_rad * 1.5))
                else:
                    cx = max(0, min(w - 1, int(round(r["centroid"][0]))))
                    cy = max(0, min(h - 1, int(round(r["centroid"][1]))))
                    win = min(max(P.TEXTURE_MIN_WINDOW, int(r["radius_px"])), 32)

                y0, y1 = max(0, cy - win), min(h, cy + win + 1)
                x0, x1 = max(0, cx - win), min(w, cx + win + 1)

                window_slice = complex_[:, :, y0:y1, x0:x1]
                r["interior_variance"] = float(np.var(window_slice)) if window_slice.size > 0 else 0.0
                r["coarseness"] = _compute_coarseness(window_slice)
                r["regularity"] = _compute_regularity(window_slice)
                r["gabor_complex"] = window_slice.mean(axis=(2, 3)).astype(np.float32) if window_slice.size > 0 else np.zeros((len(bank.theta), len(bank.sf)), dtype=np.float32)

                r["gabor_simple"] = simple[:, :, :, cy, cx].copy()
                r["gabor_phase"] = phase[:, :, cy, cx].copy()
                r["gabor_complex_pooled"] = complex_pooled[:, :, y0:y1, x0:x1].mean(axis=(2, 3)).astype(np.float32) if window_slice.size > 0 else np.zeros_like(r["gabor_complex"])
                r["gabor_complex_on"] = _on_resp["complex"][:, :, y0:y1, x0:x1].mean(axis=(2, 3)).astype(np.float32) if window_slice.size > 0 else np.zeros_like(r["gabor_complex"])
                r["gabor_complex_off"] = _off_resp["complex"][:, :, y0:y1, x0:x1].mean(axis=(2, 3)).astype(np.float32) if window_slice.size > 0 else np.zeros_like(r["gabor_complex"])
                # [O3] WINDOW STATISTICS, NOT ONE PIXEL PER REGION.
                #
                # These planes used to be read at the single pixel nearest the
                # region's centroid. A pixel is a sample of noise -- a texture
                # grain, a highlight, an anti-aliasing step -- so two regions
                # with the SAME material could be described by opposite
                # samples, and a region straddling a boundary by whichever
                # side its centroid happened to land on. Every material plane
                # is now the statistic of the region's OWN window. Phase and
                # hue are circular, so they take a circular mean (a linear
                # mean of angles is meaningless); the rest take a mean.
                r["gabor_phase_on"] = _window_circular_mean_channels(
                    _on_resp["phase"], y0, y1, x0, x1)
                r["gabor_phase_off"] = _window_circular_mean_channels(
                    _off_resp["phase"], y0, y1, x0, x1)
                _opp_mean = [
                    _window_mean(opponent[..., _k], y0, y1, x0, x1)
                    for _k in range(3)]
                r["opponent"] = np.asarray(
                    [float(opponent[cy, cx, _k]) if _opp_mean[_k] is None
                     else _opp_mean[_k] for _k in range(3)], dtype=np.float32)
                _ret = [
                    _window_mean(retinex_res["lum_comp"], y0, y1, x0, x1),
                    _window_mean(retinex_res["rg_comp"], y0, y1, x0, x1),
                    _window_mean(retinex_res["by_comp"], y0, y1, x0, x1),
                    _window_mean(retinex_res["sat_norm"], y0, y1, x0, x1),
                    _window_circular_mean(retinex_res["hue"], y0, y1, x0, x1),
                ]
                _ret_point = (
                    float(retinex_res["lum_comp"][cy, cx]),
                    float(retinex_res["rg_comp"][cy, cx]),
                    float(retinex_res["by_comp"][cy, cx]),
                    float(retinex_res["sat_norm"][cy, cx]),
                    float(retinex_res["hue"][cy, cx]),
                )
                r["opponent_retinex"] = tuple(
                    _ret_point[_k] if _ret[_k] is None else float(_ret[_k])
                    for _k in range(5))
                _corr = [
                    _window_mean(corrected_opp[..., _k], y0, y1, x0, x1)
                    for _k in range(3)]
                r["corrected_opponent"] = np.asarray(
                    [float(corrected_opp[cy, cx, _k]) if _corr[_k] is None
                     else _corr[_k] for _k in range(3)], dtype=np.float32)
                _rgb_mean = [
                    _window_mean(corrected_rgb[..., _k], y0, y1, x0, x1)
                    for _k in range(3)]
                r["corrected_rgb"] = np.asarray(
                    [float(corrected_rgb[cy, cx, _k]) if _rgb_mean[_k] is None
                     else _rgb_mean[_k] for _k in range(3)], dtype=np.float32)
                _rgc = _window_mean(rg_contrast, y0, y1, x0, x1)
                _byc = _window_mean(by_contrast, y0, y1, x0, x1)
                r["rg_contrast"] = (float(rg_contrast[cy, cx])
                                    if _rgc is None else float(_rgc))
                r["by_contrast"] = (float(by_contrast[cy, cx])
                                    if _byc is None else float(_byc))
                r["local_edge"] = _local_edge_at(lum, cy, cx)

                # Skip B1/B4 for thin contours
                if _lum_resp_b1 is not None and not r.get("is_boundary", False):
                    _sig_b1, _pop_b1, _st_b1 = b1_region_signature(
                        bank, _lum_resp_b1, _on_resp_b1, _off_resp_b1,
                        y0, y1, x0, x1, lum, _on_img, _off_img,
                        cy, cx, r["radius_px"]
                    )
                    r["b1_sig"] = _sig_b1
                    r["b1_population"] = _pop_b1
                    r["b1_class"] = b1_class_name(len(bank.theta), len(bank.sf))
                    r["b1_stats"] = _st_b1

                    b4_register(bank)
                    _win_b4 = lum[y0:y1, x0:x1]
                    _lum_mean_b4 = float(_win_b4.mean()) if _win_b4.size else 0.0
                    _sig_b4 = compute_texture_rich(
                        _lum_resp_b1["complex_local"],
                        bank.theta, bank.sf,
                        _lum_mean_b4,
                        y0, y1, x0, x1,
                    )
                    _cls_b4 = b4_class_name(len(bank.theta), len(bank.sf))
                    # [O3] The graded population is the window's own
                    # DISTRIBUTION over (theta, sf); without it the class
                    # compared only per-cell scalars measured on the window
                    # mean, so two different materials with the same mean
                    # were indistinguishable.  Empty when the window has no
                    # energy (no fabricated uniform distribution).
                    _pop_b4 = texture_rich_population(
                        _lum_resp_b1["complex_local"], y0, y1, x0, x1)
                    r["b4_sig"] = TypedSignature(
                        kind="CANONICAL",
                        class_=_cls_b4,
                        relation="SECOND_ORDER_STATS",
                        quantized=_sig_b4,
                        population=_pop_b4,
                    )
                    r["b4_class"] = _cls_b4

        # Macro envelope normalization
        if all_regions:
            all_cx = [r["centroid"][0] for r in all_regions]
            all_cy = [r["centroid"][1] for r in all_regions]
            env_cx = (min(all_cx) + max(all_cx)) / 2.0
            env_cy = (min(all_cy) + max(all_cy)) / 2.0
            env_ext = max(max(all_cx) - min(all_cx), max(all_cy) - min(all_cy), 1.0)
            for r in all_regions:
                r["rel_x"] = clamp((r["centroid"][0] - env_cx) / env_ext + 0.5, 0.0, 1.0)
                r["rel_y"] = clamp((r["centroid"][1] - env_cy) / env_ext + 0.5, 0.0, 1.0)

        for r in all_regions:
            r["chain_id"] = None
            r["chain_pos_start"] = 0.0
            r["chain_pos_end"] = 1.0

        obj = cls(all_regions, (h, w), flag_indices, effect_indices, contours=contours)
        obj._raw_image = raw_image
        obj._lum_adapted = lum
        obj._gabor_complex = complex_
        obj._gabor_thetas = bank.theta
        obj._gabor_sfs = bank.sf
        return obj

    def get_gabor_responses(self, u):
        """Per-region Gabor summary, or None on the region path.

        The data guard (not `hasattr`) is what tells the two paths apart:
        every Observation2D now HAS this method, so the B.7 branch must
        test the return value.
        """
        if 0 <= u < self._N:
            r = self._regions[u]
            if "gabor_simple" not in r:
                return None
            out = {"simple": r["gabor_simple"],
                   "complex": r["gabor_complex"],
                   "phase": r["gabor_phase"],
                   "complex_pooled": r["gabor_complex_pooled"]}
            for _pub, _src in (("complex_on", "gabor_complex_on"),
                               ("complex_off", "gabor_complex_off"),
                               ("phase_on", "gabor_phase_on"),
                               ("phase_off", "gabor_phase_off")):
                if _src in r:
                    out[_pub] = r[_src]
            return out
        return None

    def get_region_centroid_by_anchor(self, u):
        if 0 <= u < self._N:
            return (float(self._regions[u]["centroid"][0]),
                    float(self._regions[u]["centroid"][1]))
        return None

    def get_anchors_within(self, pos, radius):
        """Anchors within `radius` of `pos` (lazily built spatial hash)."""
        sh = getattr(self, "_spatial_hash", None)
        if sh is None:
            sh = SpatialHash(
                {i: (self._regions[i]["centroid"][0],
                     self._regions[i]["centroid"][1])
                 for i in range(self._N)},
                cell_size=max(P.SPATIAL_CELL_MIN, 1.0),
            )
            self._spatial_hash = sh
        return [a for a, _ in sh.neighbors(pos, radius)]

    def get_local_edge(self, u):
        if 0 <= u < self._N and "local_edge" in self._regions[u]:
            return self._regions[u]["local_edge"]
        return None

    def get_coarseness(self, u):
        if 0 <= u < self._N and "coarseness" in self._regions[u]:
            return float(self._regions[u]["coarseness"])
        return 0.0

    def get_regularity(self, u):
        if 0 <= u < self._N and "regularity" in self._regions[u]:
            return float(self._regions[u]["regularity"])
        return 0.0

    def get_opponent_channels(self, u):
        if 0 <= u < self._N:
            r = self._regions[u]
            if "opponent_retinex" in r:
                return r["opponent_retinex"]
            if "corrected_opponent" in r:
                co = r["corrected_opponent"]
                return (float(co[1]), float(co[2]), float(co[0]),
                        float(max(r["rg_contrast"], r["by_contrast"])))
        return None

    def get_anchor_opponent(self, u):
        return self.get_opponent_channels(u)

    @property
    def image_shape(self):
        return (self._h, self._w)

    def at_scale(self, level):
        """Downsampled view of this observation (raw-image path only)."""
        return downsample_observation(self, level)

def downsample_observation(obs, level):
    if not isinstance(obs, Observation2D):
        return obs
    if level == 0:
        return obs
    if getattr(obs, "_raw_image", None) is None:
        return obs
    factor = 2 ** level
    h, w = obs.image_shape
    nh = max(P.MIN_PYRAMID_DIM, h // factor)
    nw = max(P.MIN_PYRAMID_DIM, w // factor)
    small = _bilinear_resize(obs._raw_image, nh, nw)
    return Observation2D.from_image(
        small,
        flag_indices=list(obs._flag) if obs._flag else None,
        effect_indices=list(obs._effect) if obs._effect else None,
    )

def _bilinear_resize(img, nh, nw):
    from scipy.ndimage import zoom
    if img.ndim == 3:
        return zoom(img, (nh / img.shape[0], nw / img.shape[1], 1), order=1)
    return zoom(img, (nh / img.shape[0], nw / img.shape[1]), order=1)

