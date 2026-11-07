"""perception/contour_structure.py -- contour tracing, structure and lattice
symbols moved VERBATIM from perception/contours.py (ITEM 4 deletion prep,
MOVE ONLY: nothing deleted). perception/contours.py keeps a thin re-export
shim so every existing import path keeps working. Importers per
diagnostics/_item4_symbol_audit.md.
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Dict, List, Set, Tuple

import numpy as np
from scipy.ndimage import maximum_filter

import parameters as P
from core.hashing import (  # noqa: F401
    clamp,
)
from perception.polar import (  # noqa: F401
    _angle_diff,
    _axial_angle_diff,
    _axial_wrap_sf,
    signed_menger,
)


def _local_edge_at(lum, cy, cx):
    """(contrast, polarity, sharpness, orientation) at one pixel."""
    h, w = lum.shape

    def at(y, x):
        return lum[max(0, min(h - 1, y)), max(0, min(w - 1, x))]

    gx = (at(cy, cx + 1) - at(cy, cx - 1)) * 0.5
    gy = (at(cy + 1, cx) - at(cy - 1, cx)) * 0.5
    contrast = math.sqrt(gx * gx + gy * gy)
    gxx = at(cy, cx + 1) - 2 * at(cy, cx) + at(cy, cx - 1)
    gyy = at(cy + 1, cx) - 2 * at(cy, cx) + at(cy - 1, cx)
    lap = gxx + gyy
    polarity = 1.0 if lap < 0 else -1.0
    sharpness = min(1.0, abs(lap) / (contrast + 1e-6))
    orient = math.atan2(gy, gx) % math.pi
    return (contrast, polarity, sharpness, orient)

def _polyline_arc_length(polyline: List[Tuple[int, int]]) -> float:
    """Sum of consecutive Euclidean step lengths. Diagonals cost √2."""
    if len(polyline) < 2:
        return 0.0
    total = 0.0
    for i in range(len(polyline) - 1):
        dr = polyline[i + 1][0] - polyline[i][0]
        dc = polyline[i + 1][1] - polyline[i][1]
        total += math.sqrt(dr * dr + dc * dc)
    return total

def _tangent_at(
    polyline: List[Tuple[int, int]], at_start: bool, span: int = 3
) -> float:
    """Local tangent (atan2(dr, dc)) at one end of an ordered polyline,
    using up to `span` pixels of local support."""
    n = len(polyline)
    if n < 2:
        return 0.0
    span = max(1, min(span, n - 1))
    if at_start:
        p1, p2 = polyline[0], polyline[span]
    else:
        p1, p2 = polyline[n - 1 - span], polyline[n - 1]
    return math.atan2(p2[0] - p1[0], p2[1] - p1[1])

def _moore_neighbor_trace(
    coords: List[Tuple[int, int]],
) -> List[Tuple[int, int]]:
    """[v9/v10] Moore-Neighbor boundary tracing.

    Returns the OUTER contour as an ordered list of (r, c) pixels,
    traversed clockwise from the topmost-leftmost boundary pixel. Uses
    a simplified stopping criterion (halts upon re-entering the start
    pixel). For non-simple boundaries, the trace may halt early, but
    the closure check (trace[0] == trace[-1]) safely rejects them as [].

    CONVENTION: prev_dir_idx is the DIRECTION OF MOTION (set to
    found_idx after each move), not the backtrack direction. Scan start
    offset is +5 (backtrack +4, one clockwise step past it).

    Returns [] on any failure (sentinel — callers treat as "not traced"):
      - region area < 3;
      - fewer than 3 boundary pixels;
      - start pixel is isolated;
      - trace length < 3;
      - trace length < len(boundary) // 3;
      - trace does not close.
    """
    if len(coords) < 3:
        return []
    coord_set = set(coords)

    def _is_boundary(r: int, c: int) -> bool:
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            if (r + dr, c + dc) not in coord_set:
                return True
        return False

    boundary = [(r, c) for (r, c) in coords if _is_boundary(r, c)]
    if len(boundary) < 3:
        return []

    start = min(boundary, key=lambda p: (p[0], p[1]))
    bnd_set = set(boundary)
    assert start in bnd_set, "start pixel must be in bnd_set"

    dirs = [
        (0, -1), (-1, -1), (-1, 0), (-1, 1),
        (0, 1), (1, 1), (1, 0), (1, -1),
    ]

    prev_dir_idx = 0
    found = False
    for k in range(8):
        di = (prev_dir_idx + k) % 8
        dr, dc = dirs[di]
        if (start[0] + dr, start[1] + dc) in bnd_set:
            prev_dir_idx = di
            found = True
            break
    if not found:
        return []

    trace = [start]
    trace_set: Set[Tuple[int, int]] = {start}
    curr = start
    max_steps = 8 * len(boundary) + 8
    for _ in range(max_steps):
        search_start = (prev_dir_idx + 5) % 8
        found_idx = None
        for k in range(8):
            di = (search_start + k) % 8
            dr, dc = dirs[di]
            nxt = (curr[0] + dr, curr[1] + dc)
            if nxt in bnd_set:
                found_idx = di
                break
        if found_idx is None:
            break
        dr, dc = dirs[found_idx]
        nxt = (curr[0] + dr, curr[1] + dc)
        if nxt == start and len(trace) >= 3:
            trace.append(start)
            break
        if nxt in trace_set:
            trace.append(nxt)
            break
        trace.append(nxt)
        trace_set.add(nxt)
        curr = nxt
        prev_dir_idx = found_idx

    if len(trace) < 3:
        return []
    if len(trace) < len(boundary) // 3:
        return []
    if trace[0] != trace[-1]:
        return []
    return trace

def _detect_t_junctions(
    regions: List[Dict[str, Any]],
) -> List[Tuple[int, int, Tuple[float, float], float]]:
    """[v9] Geometric T-junction detection. Returns
    [(stem_region_idx, top_region_idx, contact_point, confidence)]."""
    tjs: List[Tuple[int, int, Tuple[float, float], float]] = []
    for i, r_stem in enumerate(regions):
        poly_stem = r_stem.get("boundary_polyline", [])
        if not poly_stem or len(poly_stem) < 2:
            continue
        for is_start in (True, False):
            pt = poly_stem[0] if is_start else poly_stem[-1]
            tan_stem_in = _tangent_at(poly_stem, at_start=is_start)
            tan_stem_out = (tan_stem_in + math.pi) % (2.0 * math.pi)
            for j, r_top in enumerate(regions):
                if i == j:
                    continue
                poly_top = r_top.get("boundary_polyline", [])
                if not poly_top or len(poly_top) < 2:
                    continue
                top_arc = _polyline_arc_length(poly_top)
                if top_arc < P.T_JUNCTION_MIN_FOREIGN_ARC:
                    continue
                best_d = float("inf")
                best_idx = -1
                for k, q in enumerate(poly_top):
                    dd = math.sqrt(
                        (pt[0] - q[0]) ** 2 + (pt[1] - q[1]) ** 2
                    )
                    if dd < best_d:
                        best_d = dd
                        best_idx = k
                if best_d > P.T_JUNCTION_MAX_GAP:
                    continue
                lo = max(0, best_idx - 3)
                hi = min(len(poly_top) - 1, best_idx + 3)
                if hi - lo < 1:
                    continue
                p_lo, p_hi = poly_top[lo], poly_top[hi]
                tan_top = math.atan2(
                    p_hi[0] - p_lo[0], p_hi[1] - p_lo[1]
                )
                ang = abs(_angle_diff(tan_stem_out, tan_top))
                ang = min(ang, math.pi - ang)
                perp_dev = abs(ang - math.pi / 2.0)
                if perp_dev > P.T_JUNCTION_MAX_PERP_DEV:
                    continue
                gap_conf = 1.0 - best_d / P.T_JUNCTION_MAX_GAP
                perp_conf = 1.0 - perp_dev / P.T_JUNCTION_MAX_PERP_DEV
                len_conf = min(
                    1.0, top_arc / (2.0 * P.T_JUNCTION_MIN_FOREIGN_ARC)
                )
                confidence = gap_conf * perp_conf * len_conf
                if confidence < P.OCCLUSION_MIN_CONFIDENCE:
                    continue
                contact_pt = (
                    (pt[0] + poly_top[best_idx][0]) / 2.0,
                    (pt[1] + poly_top[best_idx][1]) / 2.0,
                )
                tjs.append((i, j, contact_pt, confidence))
    return tjs

def _front_end_scale(h: int, w: int) -> float:
    """Scale of this frame relative to the calibrated reference resolution.

    Pixel-denominated front-end thresholds are multiplied by this factor and
    the Gabor cycles-per-pixel band is divided by it, so the same physical
    tuning holds at any resolution. 1.0 at the 256x256 reference frame, so
    behaviour there is unchanged. See the note on
    P.FRONT_END_REF_DIAGONAL_PX for the A/B/C experiment that motivated this.
    """
    ref = float(getattr(P, "FRONT_END_REF_DIAGONAL_PX", 362.0386719678744))
    return math.hypot(float(h), float(w)) / max(ref, 1e-6)

def _extract_association_field_contours(
    complex_energy: np.ndarray,
    bank_theta: np.ndarray,
    bank_sf: np.ndarray,
    h: int,
    w: int,
) -> List[Dict[str, Any]]:
    """Extract continuous curvilinear contours from oriented energy via association fields.

    Field, Hayes & Hess (1993): Collinear and co-circular integration along
    the edge tangent ridge (orthogonal to Gabor carrier orientation).

    [Scale-relative] Pixel-denominated constants (sigma_dist, min_length,
    max_seeds, closure distance) are scaled by _front_end_scale(h, w); the
    normalised/angular ones are resolution-independent and left as-is.
    """
    scale = _front_end_scale(h, w)
    min_len = getattr(P, "CONTOUR_MIN_LENGTH", 12) * scale
    s_theta = getattr(P, "CONTOUR_SIGMA_THETA", 0.45)
    s_dist = getattr(P, "CONTOUR_SIGMA_DIST", 3.5) * scale
    p_cos = getattr(P, "CONTOUR_COS_POWER", 2.0)
    s_thresh = getattr(P, "CONTOUR_STRENGTH_THRESHOLD", 0.12)
    max_steps = getattr(P, "CONTOUR_MAX_TRACING_STEPS", 350)
    max_seeds = max(1, int(round(getattr(P, "CONTOUR_MAX_SEEDS", 180) * scale)))

    nt = len(bank_theta)

    # 1. Energy pooling and dominant carrier orientation
    energy_over_sf = complex_energy.max(axis=1)  # (nt, H, W)
    strength_map = energy_over_sf.max(axis=0).astype(np.float64)  # (H, W)
    carrier_idx = np.argmax(energy_over_sf, axis=0)  # (H, W)

    # [Front-end fix 2026-10-10] Global-max normalisation let the frame-edge
    # convolution artifacts (the reflect-padded lowest-SF kernel manufactures
    # ridges at ~3*sigma_x inside the border) set the reference, so a small
    # object's edge energy was diluted below CONTOUR_STRENGTH_THRESHOLD and it
    # vanished entirely -- measured on the synthetic zoom sweep: a disc at
    # r <= 0.20 of a 256^2 frame yielded ZERO contours. Normalise against a
    # high percentile of the INTERIOR, with the border margin taken from the
    # lowest-SF kernel's support (scale-relative via _front_end_scale).
    sf_min = float(np.min(bank_sf)) if len(bank_sf) else P.GABOR_SF_MIN
    # bank_sf is already scale-adjusted by GaborBank.build_default(scale), so
    # this is the EFFECTIVE kernel sigma in pixels at this resolution.
    sigma_min = P.GABOR_SIGMA_X_FRAC / max(sf_min, 1e-6)
    margin = int(max(1, math.ceil(3.0 * sigma_min)))
    margin = min(margin, max(1, (min(h, w) - 1) // 2))
    interior = strength_map[margin:h - margin, margin:w - margin]
    if interior.size == 0:
        interior = strength_map
    s_ref = float(np.percentile(interior, P.CONTOUR_INTERIOR_PERCENTILE))
    if s_ref < 1e-9:
        s_ref = float(strength_map.max())
    # [front-end fix, gate 1] Absolute floor. A percentile reference has no
    # notion of "is there structure at all": on a uniform field it normalises
    # numerical noise into ridges. Measured on a blank 256^2 frame the interior
    # p99.5 is 6.9e-07 while the hardest real small object is 2.75e-01, so the
    # floor separates the two by ~5 orders of magnitude.
    if s_ref < P.CONTOUR_MIN_ABS_STRENGTH:
        return []
    strength_norm = strength_map / s_ref

    # 2. Physics Correction: Gabor carrier is normal; tangent is theta_carrier + pi/2
    carrier_rad = carrier_idx.astype(np.float64) * (math.pi / float(nt))
    tangent_rad = (carrier_rad + (math.pi / 2.0)) % math.pi

    # 3. Dense Non-Maximum Suppression with bounded seed budget
    local_max = maximum_filter(strength_norm, size=3)
    ridge_mask = (strength_norm == local_max) & (strength_norm >= s_thresh)

    ys, xs = np.where(ridge_mask)
    if len(xs) == 0:
        return []

    ridge_seeds = [(int(x), int(y), float(strength_norm[y, x])) for x, y in zip(xs, ys)]
    ridge_seeds.sort(key=lambda s: -s[2])
    ridge_seeds = ridge_seeds[:max_seeds]

    used = np.zeros((h, w), dtype=bool)
    contours: List[Dict[str, Any]] = []

    neighbor_offsets = [
        (1, 0), (-1, 0), (0, 1), (0, -1),
        (1, 1), (1, -1), (-1, 1), (-1, -1),
    ]

    def _trace_branch(start_x: int, start_y: int, heading_angle: float) -> Tuple[List[Tuple[int, int]], List[float], List[float]]:
        path: List[Tuple[int, int]] = []
        thetas: List[float] = []
        strengths: List[float] = []

        cx, cy = start_x, start_y
        curr_heading = heading_angle

        for _ in range(max_steps):
            best_next = None
            best_w = 0.0
            best_heading = curr_heading

            for dx, dy in neighbor_offsets:
                nx, ny = cx + dx, cy + dy
                if not (0 <= nx < w and 0 <= ny < h) or used[ny, nx]:
                    continue
                if strength_norm[ny, nx] < (s_thresh * 0.4):
                    continue

                step_dist = math.hypot(dx, dy)
                step_dir = math.atan2(dy, dx)

                # Forward cone check: must step in forward hemisphere
                d_head = abs((step_dir - curr_heading + math.pi) % (2.0 * math.pi) - math.pi)
                if d_head > (math.pi / 2.0):
                    continue

                # Tangent agreement (axial mod pi)
                n_tangent = tangent_rad[ny, nx]
                d_theta = _axial_angle_diff(n_tangent, curr_heading)

                # Association field equation
                w_assoc = (
                    math.exp(-(d_theta ** 2) / (s_theta ** 2))
                    * math.exp(-(step_dist ** 2) / (s_dist ** 2))
                    * (max(0.0, math.cos(d_head)) ** p_cos)
                    * float(strength_norm[ny, nx])
                )

                if w_assoc > best_w:
                    best_w = w_assoc
                    best_next = (nx, ny)
                    best_heading = step_dir

            if best_next is None or best_w < 0.05:
                break

            nx, ny = best_next
            path.append((nx, ny))
            thetas.append(float(tangent_rad[ny, nx]))
            strengths.append(float(strength_norm[ny, nx]))
            used[ny, nx] = True

            cx, cy = nx, ny
            curr_heading = best_heading

        return path, thetas, strengths

    for sx, sy, _ in ridge_seeds:
        if used[sy, sx]:
            continue

        used[sy, sx] = True
        tan_0 = tangent_rad[sy, sx]

        # Trace both branches starting from seed
        fwd_pts, fwd_th, fwd_st = _trace_branch(sx, sy, tan_0)
        bwd_pts, bwd_th, bwd_st = _trace_branch(sx, sy, (tan_0 + math.pi) % (2.0 * math.pi))

        full_pts = list(reversed(bwd_pts)) + [(sx, sy)] + fwd_pts
        full_th = list(reversed(bwd_th)) + [float(tan_0)] + fwd_th
        full_st = list(reversed(bwd_st)) + [float(strength_norm[sy, sx])] + fwd_st

        if len(full_pts) < min_len:
            continue

        d_close = math.hypot(full_pts[0][0] - full_pts[-1][0], full_pts[0][1] - full_pts[-1][1])
        is_closed = (d_close <= 3.0 * scale and len(full_pts) >= 16)
        if is_closed:
            full_pts.append(full_pts[0])
            full_th.append(full_th[0])
            full_st.append(full_st[0])

        arc_len = sum(
            math.hypot(full_pts[i+1][0] - full_pts[i][0], full_pts[i+1][1] - full_pts[i][1])
            for i in range(len(full_pts) - 1)
        )

        contours.append({
            "polyline": full_pts,
            "theta_profile": full_th,
            "strength_profile": full_st,
            "length": float(arc_len),
            "is_closed": bool(is_closed),
        })

    return contours

class V1RetinotopicLattice:
    """Tessellates the visual array into an overlapping retinotopic coordinate sheet
    of cortical hypercolumns (Hubel & Wiesel 1968; Mountcastle 1997).
    """

    def __init__(self, density: int = None, overlap_factor: float = None):
        self.density = density or getattr(P, "RETINOTOPIC_GRID_DENSITY", 16)
        self.overlap_factor = overlap_factor or getattr(P, "RETINOTOPIC_RF_OVERLAP", 1.6)

    def tile_visual_field(self, h: int, w: int) -> List[Dict[str, Any]]:
        stride_y = h / float(self.density)
        stride_x = w / float(self.density)
        radius_px = max(2.0, (min(stride_y, stride_x) / 2.0) * self.overlap_factor)
        diag_ref = math.sqrt(h * h + w * w)

        hypercolumns: List[Dict[str, Any]] = []
        for r_idx in range(self.density):
            cy = (r_idx + 0.5) * stride_y
            for c_idx in range(self.density):
                cx = (c_idx + 0.5) * stride_x
                rel_x = (cx / w) * 2.0 - 1.0
                rel_y = (cy / h) * 2.0 - 1.0

                hypercolumns.append({
                    "anchor_id": len(hypercolumns),
                    "centroid": (float(cx), float(cy)),
                    "grid_coord": (r_idx, c_idx),
                    "radius_px": float(radius_px),
                    "radius_norm": float(radius_px / (0.35 * diag_ref + 1e-6)),
                    "rel_position": (float(rel_x), float(rel_y)),
                    "rel_x": float((rel_x + 1.0) / 2.0),
                    "rel_y": float((rel_y + 1.0) / 2.0),
                    # [O1] NO FABRICATED "area".  The key used to be the
                    # area of the receptive-field DISC -- a property of
                    # the lattice, not of anything in the image, under a
                    # name that said "the area of this region".  Nothing
                    # read it (the contour regions' area is measured from
                    # their polyline in observation_2d.py).  The lattice's
                    # own quantity is kept under its own name.
                    "rf_area_px": float(math.pi * radius_px * radius_px),
                    "circularity": 1.0,
                    "complexity": 0.1,
                    "elongation": 1.0,
                    "major_axis_angle": 0.0,
                    "roughness": 0.0,
                    "isotropy": 1.0,
                    "boundary_polyline": None,
                    "is_boundary": False,
                })
        return hypercolumns

    def tile_with_features(
        self,
        h: int,
        w: int,
        complex_energy: np.ndarray,
        bank_theta: np.ndarray,
        bank_sf: np.ndarray,
        lum: np.ndarray,
        clean_rgb: np.ndarray,
    ) -> List[Dict[str, Any]]:
        regions = self.tile_visual_field(h, w)
        nt = len(bank_theta)

        for r in regions:
            cx = max(0, min(w - 1, int(round(r["centroid"][0]))))
            cy = max(0, min(h - 1, int(round(r["centroid"][1]))))
            rad = max(2, int(r["radius_px"]))
            y0, y1 = max(0, cy - rad), min(h, cy + rad + 1)
            x0, x1 = max(0, cx - rad), min(w, cx + rad + 1)

            win_energy = complex_energy[:, :, y0:y1, x0:x1]
            if win_energy.size > 0:
                marginal_th = win_energy.sum(axis=(1, 2, 3))
                tot_e = float(marginal_th.sum())
                if tot_e > 1e-9:
                    p_th = marginal_th / tot_e
                    c_m = float(np.sum(p_th * np.cos(2 * bank_theta)))
                    s_m = float(np.sum(p_th * np.sin(2 * bank_theta)))
                    r_len = math.hypot(c_m, s_m)
                    r["elongation"] = clamp(1.0 - r_len, 0.0, 1.0)
                    r["major_axis_angle"] = float(0.5 * math.atan2(s_m, c_m)) % math.pi
                    r["complexity"] = clamp(r_len, 0.0, 1.0)
                else:
                    r["elongation"] = 1.0
                    r["major_axis_angle"] = 0.0
                    r["complexity"] = 0.0

                win_spatial = win_energy.sum(axis=(0, 1))
                tot_sp = float(win_spatial.sum())
                if tot_sp > 1e-9:
                    yy, xx = np.mgrid[0:win_spatial.shape[0], 0:win_spatial.shape[1]]
                    w_n = win_spatial / tot_sp
                    my = float((w_n * yy).sum())
                    mx = float((w_n * xx).sum())
                    dists = np.hypot(yy - my, xx - mx)
                    # [O1] A MEASURED ROUNDNESS.  The old value was the
                    # MEAN NORMALISED RADIAL DISTANCE of the energy -- a
                    # radial-profile statistic filed under the name
                    # "circularity", which is not what that number
                    # measures.  This is the fraction of the region's
                    # energy inside the circle that holds half of it,
                    # relative to the fraction inside the circle that
                    # holds 95% of it: 1.0 would be "all the energy at one
                    # radius", and a UNIFORM DISC scores 0.25 / 0.9025 =
                    # 0.277.  It is a measurement of the image, it is
                    # scale-free, and it is not a copy of any other slot.
                    order = np.argsort(dists, axis=None)
                    cum = np.cumsum(w_n.ravel()[order])

                    def _mass_at(q: float) -> float:
                        i = int(np.searchsorted(cum, q))
                        i = min(max(i, 0), len(cum) - 1)
                        return float(cum[i])

                    m50 = _mass_at(0.5)
                    m95 = _mass_at(0.95)
                    r["circularity"] = clamp(m50 / (m95 + 1e-9), 0.0, 1.0)
                else:
                    r["circularity"] = 0.5
            else:
                r["elongation"] = 1.0
                r["major_axis_angle"] = 0.0
                r["circularity"] = 0.5
                r["complexity"] = 0.0

            lum_win = lum[y0:y1, x0:x1]
            if lum_win.size > 4:
                r["roughness"] = clamp(float(np.std(lum_win)) * 2.0, 0.0, 1.0)
            else:
                r["roughness"] = 0.0
            r["isotropy"] = clamp(r["elongation"], 0.0, 1.0)

            # True Linear RGB for Color lens compatibility
            pix_rgb = clean_rgb[y0:y1, x0:x1]
            if pix_rgb.size >= 3:
                r["mean_rgb"] = tuple(float(x) for x in np.mean(pix_rgb, axis=(0, 1))[:3])
            else:
                r["mean_rgb"] = (float(clean_rgb[cy, cx, 0]), float(clean_rgb[cy, cx, 1]), float(clean_rgb[cy, cx, 2]))

        return regions

def _resample_contours(contours, step, min_points=12):
    pts = []  # (x, y, theta_axial, contour_idx, arc)
    for ci, c in enumerate(contours):
        pl = c.get("polyline")
        if not pl or len(pl) < min_points:
            continue
        P_ = [(float(x), float(y)) for x, y in pl]
        cum = [0.0]
        for i in range(1, len(P_)):
            cum.append(cum[-1] + math.hypot(P_[i][0] - P_[i-1][0], P_[i][1] - P_[i-1][1]))
        if cum[-1] < 4 * step:
            continue
        smp, j = [], 0
        for k in range(int(cum[-1] / step) + 1):
            s = min(cum[-1], k * step)
            while j < len(cum) - 2 and cum[j+1] < s:
                j += 1
            seg = cum[j+1] - cum[j]
            f = 0.0 if seg < 1e-9 else (s - cum[j]) / seg
            smp.append((P_[j][0] + f * (P_[j+1][0] - P_[j][0]), P_[j][1] + f * (P_[j+1][1] - P_[j][1]), s))
        for k, (x, y, s) in enumerate(smp):
            a, b = smp[max(0, k-2)], smp[min(len(smp)-1, k+2)]
            pts.append((x, y, math.atan2(b[1] - a[1], b[0] - a[0]) % math.pi, ci, s))
    return pts

def _segment_profile(r, tol):
    """Split a radius profile at significant interior extrema (hysteresis tol).
    Returns (cut_indices, kinds) with kinds 'min' (neck) or 'max' (bulge/seed)."""
    cuts, kinds, ext_v, ext_i, mode = [], [], r[0], 0, 0
    for i in range(1, len(r)):
        v = r[i]
        if mode == 0:
            if v - ext_v > tol:
                mode, ext_v, ext_i = 1, v, i
            elif ext_v - v > tol:
                mode, ext_v, ext_i = -1, v, i
        elif mode == 1:
            if v > ext_v:
                ext_v, ext_i = v, i
            elif ext_v - v > tol:
                cuts.append(ext_i)
                kinds.append("max")
                mode, ext_v, ext_i = -1, v, i
        else:
            if v < ext_v:
                ext_v, ext_i = v, i
            elif v - ext_v > tol:
                cuts.append(ext_i)
                kinds.append("min")
                mode, ext_v, ext_i = 1, v, i
    return cuts, kinds
