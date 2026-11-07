"""perception/interior_field.py -- bounded-diffusion interior/exterior field.

THE GAP THIS FILLS. The representation had no per-pixel figure-ground signal:
regions carried a single hard `interior=(x,y)` point from polygon containment
(`perception/chain_regions.py`), and `BO_WEIGHTS` in `parameters/_text.py`
(reserved keys `enclosure`, `convexity`, `symmetry`, ...) was DEFINED but never
consumed by any lens. Enclosure -- "which side of this boundary is the object we
own" -- therefore had no source.

WHAT IT COMPUTES. A SOFT, signed interior(+)/exterior(-) scalar field. For every
pixel we ask how much EXTRA path cost the edges impose on getting to the image
frame, against the same trip through an edge-FREE image:

    geo_barrier = geodesic distance from the frame where crossing an edge costs
                  (1 + INTERIOR_EDGE_PENALTY * edge) per pixel entered,
    geo_free    = the same geodesic distance when every pixel costs 1,
    interior    = clip((geo_barrier - geo_free) / INTERIOR_ENCLOSURE_SCALE, 0, 1)
    signed      = 2*interior - 1                     (+ inside, - outside)

This is why it fills: any point sealed inside a closed high-edge boundary must
pay the extra edge cost to escape, so its interiorness is > 0 across the WHOLE
enclosed area. It is why a blank image reads 0 EVERYWHERE (the two trips are
identical -- enclosure must be EARNED by edges, and a finite-A artifact of a
naive bounded diffusion is impossible). And it is why the field is SOFT: the
geodesic is computed on a coarse working grid and bilinearly upsampled, so the
boundary transition is graded rather than a hard mask.

WHY A BOUNDED DIFFUSION, NOT A HARMONIC SOLVE. The enclosure signal lives in the
FINITE-TIME spread of the frame value; the steady state of `div(g grad u)=0` with
u=1 on the frame is u==1 everywhere for ANY positive conductance g, i.e. it
carries no enclosure at all. So the field is a bounded (fixed-sweep) min-plus
relaxation, the discrete analogue of a bounded diffusion.

CONTRACT (the mission's requirements for every mechanism):
  * deterministic; no randomness, no per-pixel ODE.
  * bounded compute: O((h+w) * hw) sweeps on a grid of longest side
    INTERIOR_WORK_SIZE, with early exit when the relaxation stops changing.
  * pure function of (luminance, parameters); memoised by content hash +
    front-end fingerprint (mirrors perception/lenses/peak_lens.py::_field_key).
  * scale-aware BY CONSTRUCTION: the working grid is a fixed size, so the same
    INTERIOR_ENCLOSURE_SCALE means the same thing at every image resolution.
"""
from __future__ import annotations

import hashlib
import math
from typing import Any, Dict, Optional, Tuple

import numpy as np
import parameters as P

_INF = 1.0e18


def _params() -> Tuple[float, float, int, int, float]:
    return (
        float(P.INTERIOR_EDGE_PENALTY),
        float(P.INTERIOR_ENCLOSURE_SCALE),
        int(P.INTERIOR_WORK_SIZE),
        int(P.INTERIOR_FIELD_CACHE_SIZE),
        float(P.INTERIOR_EDGE_FLOOR_PCT),
    )


def interior_fingerprint() -> str:
    """Stable digest of everything that changes the field, so a cache keyed on
    content hash can never hand back a field built under other parameters."""
    return "|".join(str(v) for v in _params())


def _edge_magnitude(lum: np.ndarray) -> np.ndarray:
    """Robust gradient magnitude in [0, 1] from a lightly smoothed luminance.

    The magnitude is GATED by the frame's own noise floor
    (`INTERIOR_EDGE_FLOOR_PCT`, the median gradient magnitude) before it is
    normalised by the 95th percentile. Without the gate the background's
    gradient noise sits at ~0.3 of the scale, so every escape route paid
    extra cost and the field saturated: the exterior could not be read as
    exterior (measured: 87% of a real frame at enclosure > 0.99) and the
    flank side of every contour had no support at all.

    With the gate, a gradient below the frame's median is 0 -- not a
    barrier -- and only the object's own boundary (and strong texture
    edges) impose a cost.
    """
    from scipy.ndimage import gaussian_filter

    sm = gaussian_filter(lum.astype(np.float64), sigma=1.0, mode="nearest")
    gy, gx = np.gradient(sm)
    mag = np.hypot(gx, gy)
    floor_pct = float(P.INTERIOR_EDGE_FLOOR_PCT)
    floor = float(np.percentile(mag, floor_pct)) if mag.size else 0.0
    scale = float(np.percentile(mag, 95.0))
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = float(mag.max()) if mag.size else 0.0
    if scale <= 1e-12:
        return np.zeros_like(mag)
    span = max(scale - floor, 1e-12)
    return np.clip((mag - floor) / span, 0.0, 1.0)


def _frame_mask(h: int, w: int) -> np.ndarray:
    m = np.zeros((h, w), dtype=bool)
    m[0, :] = m[-1, :] = m[:, 0] = m[:, -1] = True
    return m


def _chamfer(cost: np.ndarray) -> np.ndarray:
    """Geodesic distance from the frame through per-pixel traversal `cost`
    (8-neighbour chamfer / min-plus relaxation, diagonal step sqrt(2)).

    [O4] The old relaxation was 4-NEIGHBOUR ONLY, so it measured distance with
    the MANHATTAN metric: a diagonal escape of r steps cost 2r against r for a
    straight one. The whole field therefore carried a metric anisotropy -- a
    square boundary was "further" at its corners than at its sides, and the
    enclosure of any non-axis-aligned object depended on how its boundary
    happened to fall on the pixel grid. The diagonal term removes the bias while
    keeping the algorithm bounded: at most h+w+2 sweeps, exiting early once a
    sweep changes nothing.

    Both the barrier trip and the edge-free reference trip use the SAME
    relaxation, so the subtraction `geo_barrier - geo_free` is a comparison of
    like with like; the metric change is what makes that comparison isotropic.
    """
    h, w = cost.shape
    d = np.full((h, w), _INF, dtype=np.float64)
    frame = _frame_mask(h, w)
    d[frame] = 0.0
    sqrt2 = math.sqrt(2.0)
    cap = h + w + 2
    for _ in range(cap):
        up = np.full_like(d, _INF)
        up[1:, :] = d[:-1, :]
        dn = np.full_like(d, _INF)
        dn[:-1, :] = d[1:, :]
        lf = np.full_like(d, _INF)
        lf[:, 1:] = d[:, :-1]
        rt = np.full_like(d, _INF)
        rt[:, :-1] = d[:, 1:]
        ul = np.full_like(d, _INF)
        ul[1:, 1:] = d[:-1, :-1]
        ur = np.full_like(d, _INF)
        ur[1:, :-1] = d[:-1, 1:]
        dl = np.full_like(d, _INF)
        dl[:-1, 1:] = d[1:, :-1]
        dr = np.full_like(d, _INF)
        dr[:-1, :-1] = d[1:, 1:]
        axial = np.minimum(np.minimum(up, dn), np.minimum(lf, rt))
        diag = np.minimum(np.minimum(ul, ur), np.minimum(dl, dr))
        nb = np.minimum(axial + cost, diag + sqrt2 * cost)
        nd = np.minimum(d, nb)
        nd[frame] = 0.0
        if np.array_equal(nd, d):
            break
        d = nd
    return d


def _downsample(arr: np.ndarray, work: int) -> np.ndarray:
    """Block-mean to a grid whose longest side is <= work (deterministic)."""
    h, w = arr.shape
    f = max(1, int(np.ceil(max(h, w) / max(1, work))))
    if f == 1:
        return arr.astype(np.float64)
    hh, ww = (h + f - 1) // f, (w + f - 1) // f
    pad = np.zeros((hh * f, ww * f), dtype=np.float64)
    pad[:h, :w] = arr
    return pad.reshape(hh, f, ww, f).mean(axis=(1, 3))


def interior_field(lum: np.ndarray,
                   edge: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
    """Bounded-diffusion interior/exterior field.

    Returns {"interior": [0,1] HxW, "signed": [-1,1] HxW, "edge": [0,1] HxW}.
    """
    arr = np.asarray(lum, dtype=np.float64)
    if arr.ndim == 3:
        arr = arr[..., 0]
    h, w = arr.shape
    if h < 3 or w < 3:
        z = np.zeros((h, w), dtype=np.float32)
        return {"interior": z, "signed": z, "edge": z}

    penalty, enclose_scale, work, _cache, _floor_pct = _params()
    e_full = _edge_magnitude(arr) if edge is None else np.clip(
        np.asarray(edge, dtype=np.float64), 0.0, 1.0)

    e = _downsample(e_full, work)
    cost = 1.0 + penalty * e
    geo_barrier = _chamfer(cost)
    geo_free = _chamfer(np.ones_like(cost))
    enc = np.clip((geo_barrier - geo_free) / max(enclose_scale, 1e-6), 0.0, 1.0)

    if enc.shape != (h, w):
        from scipy.ndimage import zoom
        enc = zoom(enc, (h / enc.shape[0], w / enc.shape[1]), order=1)
        enc = np.clip(enc[:h, :w], 0.0, 1.0)

    interior = enc.astype(np.float32)
    signed = (2.0 * interior - 1.0).astype(np.float32)
    return {"interior": interior, "signed": signed,
            "edge": e_full.astype(np.float32)}


def _bilinear(field: np.ndarray, y: float, x: float) -> float:
    h, w = field.shape
    if not (0.0 <= y <= h - 1 and 0.0 <= x <= w - 1):
        return 0.0
    y0, x0 = int(np.floor(y)), int(np.floor(x))
    y1, x1 = min(y0 + 1, h - 1), min(x0 + 1, w - 1)
    fy, fx = y - y0, x - x0
    return float(
        field[y0, x0] * (1 - fy) * (1 - fx)
        + field[y1, x0] * fy * (1 - fx)
        + field[y0, x1] * (1 - fy) * fx
        + field[y1, x1] * fy * fx)


def ownership_from_enclosure(field: Dict[str, np.ndarray],
                             point: Tuple[float, float],
                             normal: Tuple[float, float],
                             probe: Optional[float] = None) -> float:
    """Enclosure cue at a boundary point: positive when the +normal side is the
    enclosed (interior) side, negative when the -normal side is, 0 when the
    field is flat there (no enclosure evidence). `point` is (x, y); `normal` is
    a (nx, ny) whose sign is the candidate figure direction."""
    interior = field.get("interior")
    if interior is None:
        return 0.0
    p = float(P.INTERIOR_PROBE_PX) if probe is None else float(probe)
    x, y = float(point[0]), float(point[1])
    nx, ny = float(normal[0]), float(normal[1])
    nrm = (nx * nx + ny * ny) ** 0.5
    if nrm <= 1e-9:
        return 0.0
    nx, ny = nx / nrm, ny / nrm
    plus = _bilinear(interior, y + ny * p, x + nx * p)
    minus = _bilinear(interior, y - ny * p, x - nx * p)
    return float(max(-1.0, min(1.0, (plus - minus) * 2.0)))


def interior_weighted_sample(
    plane: np.ndarray,
    interior: np.ndarray,
    x: float,
    y: float,
    radius: float,
    sigma: Optional[float] = None,
) -> Optional[float]:
    """Soft interior-weighted mean of `plane` around (x, y).

    The weight of each pixel is the product of the CONTINUOUS interior field
    (an enclosure strength in [0, 1], never thresholded) and a Gaussian falloff
    of radius `sigma` (default `radius`). No hard mask is ever formed: a pixel
    on the exterior simply contributes proportionally less, so a surface sample
    is pulled toward the enclosed material without a boundary cut. Returns None
    when the window falls entirely outside the plane or the weights vanish, so
    the caller can keep its own default rather than read a fabricated 0.
    """
    if plane is None or interior is None:
        return None
    p = np.asarray(plane, dtype=np.float64)
    enc = np.asarray(interior, dtype=np.float64)
    if p.ndim != 2 or enc.ndim != 2 or p.shape != enc.shape:
        return None
    h, w = p.shape
    r = max(1, int(math.ceil(float(radius))))
    s = float(sigma) if sigma and sigma > 0.0 else max(float(radius), 1e-6)
    x0, x1 = int(math.floor(x - r)), int(math.ceil(x + r))
    y0, y1 = int(math.floor(y - r)), int(math.ceil(y + r))
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w - 1, x1), min(h - 1, y1)
    if x1 < x0 or y1 < y0:
        return None
    num = 0.0
    den = 0.0
    for yy in range(y0, y1 + 1):
        dy = (yy - y) / s
        for xx in range(x0, x1 + 1):
            dx = (xx - x) / s
            gw = math.exp(-0.5 * (dx * dx + dy * dy))
            iw = float(enc[yy, xx])
            if not math.isfinite(iw):
                continue
            wgt = gw * iw
            if wgt <= 0.0:
                continue
            v = float(p[yy, xx])
            if not math.isfinite(v):
                continue
            num += wgt * v
            den += wgt
    if den <= 1e-12:
        return None
    return float(num / den)


_CACHE: Dict[Any, Dict[str, np.ndarray]] = {}


def _cache_key(lum: np.ndarray) -> Tuple:
    h = hashlib.blake2b(
        np.ascontiguousarray(np.asarray(lum, dtype=np.float32)).tobytes(),
        digest_size=16).digest()
    return (np.asarray(lum).shape, h, interior_fingerprint())


def interior_field_for(observation: Any = None,
                       luminance: Any = None) -> Optional[Dict[str, np.ndarray]]:
    """Share one interior field per observation, memoised by content hash +
    fingerprint (module-level, bounded). Returns None when no luminance is
    reachable."""
    cached = None
    if observation is not None:
        c = getattr(observation, "_interior_field_cache", None)
        if isinstance(c, dict) and "interior" in c:
            cached = c
    lum = luminance
    if lum is None and observation is not None:
        getter = getattr(observation, "get_luminance", None)
        if callable(getter):
            lum = getter()
    if lum is None:
        return cached
    arr = np.asarray(lum)
    if arr.ndim == 3:
        from perception.filters import luminance_adapt, opponent_channels
        arr = luminance_adapt(opponent_channels(arr.astype(np.float32))[..., 0])
    if arr.ndim != 2:
        return cached
    arr = arr.astype(np.float32, copy=False)
    key = _cache_key(arr)
    if cached is not None and getattr(
            observation, "_interior_field_cache_key", None) == key:
        return cached
    hit = _CACHE.get(key)
    if hit is None:
        hit = interior_field(arr)
        size = int(P.INTERIOR_FIELD_CACHE_SIZE)
        if size > 0:
            if len(_CACHE) >= size:
                _CACHE.pop(next(iter(_CACHE)))
            _CACHE[key] = hit
    if observation is not None:
        try:
            observation._interior_field_cache = hit
            observation._interior_field_cache_key = key
        except (AttributeError, TypeError):
            pass
    return hit
