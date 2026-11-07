"""perception/surface_fragments.py -- sparse SURFACE_FRAGMENTs and their cues.

WHAT THIS IS
------------
Session A's O1 representation.  A surface fragment is a sparse 2.5D sample
anchored on a contour fragment (a chain member) or on an inward sample of one:

    position          image (x, y) of the anchor
    tangent, owner    the contour frame: unit tangent and the unit image
                      normal pointing INTO the owner (figure) side
    curvature         signed local contour curvature in 1/px, from a circle
                      fit to the chain's own member positions
    depth layer       ordinal, from the ownership graph only (see below)
    cues              populations on ONE (slant, tilt) grid
    combined          the summed population

No segmentation, no object class, no mask.  The only inputs are the ownership
records (which already exist), the raw luminance, the enclosure field, and a
LIGHT PRIOR estimated from the corpus.  The renderer's truth (object id,
layer, normal, depth) never enters this module.

CUES
----
1. Occluding contour (exact geometry, no light).  At a smooth occluding
   contour the surface normal is perpendicular to the view, so slant = 90 deg,
   and it lies in the image plane along the contour normal.  The owner side
   gives the sign: the visible surface normal points AWAY from the figure, so
   the population is centred on ``-owner``.  The cue is emitted only when both
   preconditions the geometry requires hold:

   * the interior field's enclosure cue is non-negative on the owner side (a
     texture edge has enclosure ~0 on both sides), and
   * the owner surface is measurably TURNING just inside the boundary: the
     smallest monotone luminance step between successive inward disks is at
     least ``contour_min_step`` (``inward_luminance_step``).  A plane cannot
     have a smooth occluding contour, and a crease has its whole luminance
     change in one step, so both are excluded by a measured observable rather
     than by an object class.

2. Shading (pictorial, light prior from the corpus).  A local UMBILIC surface
   is fitted to a disk of raw luminance: the visible normal field of a sphere
   is exactly linear in the image offset,

       n_xy(d) = n_xy(0) + d / R,    n_z = +sqrt(1 - |n_xy|^2),

   so a patch determines (n_xy, R, albedo) given the light direction.  This is
   a differential-geometry prior (equal principal curvatures), not an object
   class.  It is exact on spheres; on a cylinder or a plane one light does not
   determine the normal (the classical ambiguity), and the fit is biased.  The
   cue abstains when the patch is not smooth, when the light response across
   the patch is too small to identify a normal, and when the residual exceeds
   the 8-bit storage-quantization floor by more than ``residual_sigma`` (a
   constant-albedo Lambertian model cannot explain texture or a shading
   discontinuity).

WIDTHS
------
Every population is a separable Gaussian on (slant, tilt) whose widths are
MEASURED on a calibration corpus from the cue's own repeatability:

    sigma_axis = sqrt(mean(error_axis^2))

The shading cue uses its GLOBAL measured width.  A modulation-depth-conditioned
table is computed and printed by the calibration harness, but the bins have
n = 15-20 and their RMS is outlier-dominated, so a binned width was measured to
be less stable on the test corpus.  Nothing is tuned against the evaluation
score; the evaluation corpus is disjoint from the calibration corpus.

POPULATION SUMMATION
--------------------
Cues emit normalized masses on the same grid.  Fusion is their PRODUCT
(equivalently, the sum of log likelihoods): the cues are conditionally
independent evidence about one normal, and each cue's measured width is its
precision.  Agreement sharpens; conflict is resolved in favour of the more
precise cue instead of being averaged away.  No learned weights.  The plain
normalized SUM (a belief mixture) is kept as `mixture_populations` and is
reported as a control, because it cannot beat its sharpest component.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import math

import numpy as np
from perception.interior_field import interior_weighted_sample
from scipy.optimize import least_squares

try:                                    # the documented leaf: constants only
    import parameters as P
except Exception:                        # pragma: no cover - degenerate tree
    P = None  # type: ignore

__all__ = [
    "SLANT_CENTERS_DEG",
    "TILT_CENTERS_DEG",
    "Population",
    "LightPrior",
    "SurfaceFragment",
    "estimate_light_prior",
    "gaussian_population",
    "fuse_populations",
    "mixture_populations",
    "population_moment",
    "contour_curvature",
    "contour_cue",
    "chain_geometry",
    "chain_radius",
    "rim_deficit_deg",
    "rim_deficit_samples",
    "rim_slant_marginal",
    "shade_umbilic",
"umbilic_ratio",
    "shading_cue",
    "texture_cue",
    "texture_cue_calibrated",
    "texture_geometry",
    "axis_population",
    "slant_marginal_population",
    "texture_calibration",
    "t_junction_order",
    "t_junction_arms",
    "chain_arms",
    "build_surface_fragments",
    "occlusion_order_edges",
    "assign_depth_layers",
    "estimate_surface",
]

#: Slant grid: 0 = facing the viewer, 90 = in the image plane.  One-degree
#: bins: a five-degree grid would put a 2.5 deg floor under every cue and hide
#: the very difference the calibration is meant to measure.
SLANT_CENTERS_DEG: Tuple[float, ...] = tuple(float(k) for k in range(0, 91))
#: Tilt grid: 180 bins of 2 deg, centres at 1, 3, ... 359.
TILT_CENTERS_DEG: Tuple[float, ...] = tuple(1.0 + 2.0 * k for k in range(180))

_AMBIGUOUS_OWNERSHIP = 0.02


# ---------------------------------------------------------------------------
# rim law (Session B)
# ---------------------------------------------------------------------------
# MEASURED, on the calibration corpus, with the continuous truth search.
#
# THE ANCHOR INSET IS SIGNED, AND USUALLY NEGATIVE.  The first version of this
# measurement conditioned on ``Scene.truth_at`` succeeding, which silently keeps
# only the anchors INSIDE the object; it reported ``d = 0.69 +- 0.47 px, every
# gated anchor inside``.  Measured without that conditioning, on the same 160
# gated calibration anchors:
#
#     P(anchor inside)  = 0.225    d | inside   = +0.66 +- 0.41 px
#     P(anchor outside) = 0.775    |d| outside  =  1.55 +- 0.79 px
#
# So the anchor is USUALLY OUTSIDE its own silhouette.  There the surface point
# nearest the anchor is the limb itself, whose normal is perpendicular to the
# view (slant 90) and whose deficit is zero.  The deficit is therefore not a
# function of the radius alone: it is a function of the anchor's unknown signed
# inset, and the cue marginalizes over the measured inset distribution,
#
#     delta(d, R) = 0                          for d <= 0
#                 = degrees(sqrt(2 d / R))     for d > 0,
#
# which gives the familiar ~10 deg only inside that sub-population (7.5 deg
# there, 3.2 deg over all anchors, sd 6.7).  The quantile table below IS the
# measured distribution: the prior's SHAPE is data, so the width is derived
# from the measurement rather than fitted to the corpus afterwards.
#
# THE RADIUS is the chain's own circle fit near the anchor.  A limb is straight
# (R = inf, deficit 0 for every d: an edge-on plane never turns away from the
# view) only when the chain's curvature is RESOLVED; see ``chain_radius``.
RIM_INSET_QUANTILES_PX: Tuple[float, ...] = (
    -5.68, -2.35, -2.06, -2.06, -2.03, -1.87, -1.85, -1.82, -1.73, -1.72,
    -1.69, -1.69, -1.59, -1.59, -1.53, -1.50, -1.47, -1.43, -1.38, -1.37,
    -1.35, -1.34, -1.31, -1.26, -1.22, -1.20, -1.14, -1.07, -0.99, -0.56,
    -0.22, -0.05, 0.16, 0.39, 0.51, 0.53, 0.57, 0.70, 1.04, 1.16, 1.93,
)
#: Inside-only law constants, kept for the sub-population that the limb law was
#: originally derived on (and for the diagnostic report).
RIM_INSET_MEAN_PX = 0.69
RIM_INSET_SD_PX = 0.47
#: Residual of the limb law against the continuous truth, inside sub-population.
RIM_RESIDUAL_DEG = 0.5
#: Kernel applied to each inset sample on the 1 deg slant grid.
RIM_GRID_SIGMA_DEG = 0.5
RIM_RADIUS_WINDOW = 12
#: A fit beyond this radius is straight for every purpose the law serves.
RIM_RADIUS_LIMIT_PX = 2000.0
#: The chain's curvature is RESOLVED when the straight-line residual is at
#: least this many times the circle residual.  Below it the fit's radius is
#: noise, and the limb counts as straight.
RIM_STRAIGHT_RATIO = 1.5
#: Fallback when no fit is possible: the corpus's median limb radius, so the
#: cue never claims more confidence than a 45 px limb.
RIM_RADIUS_FALLBACK_PX = 45.0


def _contiguous_run(keep: np.ndarray, centre: int) -> Tuple[int, int]:
    """The run of True in ``keep`` nearest ``centre``: (first, last)."""
    runs: List[Tuple[int, int]] = []
    i = 0
    while i < keep.size:
        if keep[i]:
            j = i
            while j + 1 < keep.size and keep[j + 1]:
                j += 1
            runs.append((i, j))
            i = j + 1
        else:
            i += 1
    if not runs:
        return 0, -1
    return min(runs, key=lambda ab: 0 if ab[0] <= centre <= ab[1]
               else min(abs(ab[0] - centre), abs(ab[1] - centre)))


def _line_rms(pts: np.ndarray) -> float:
    """RMS distance of ``pts`` from their own principal line."""
    if pts.shape[0] < 3:
        return float("inf")
    p = pts - pts.mean(axis=0)
    w, v = np.linalg.eigh(p.T @ p)
    d = v[:, int(np.argmax(w))]
    n = np.array([-d[1], d[0]], dtype=np.float64)
    return float(np.sqrt(np.mean((p @ n) ** 2)))


def chain_geometry(positions: Sequence[Tuple[float, float]], index: int,
                   window: int = RIM_RADIUS_WINDOW,
                   closed: bool = False) -> Optional[Dict[str, Any]]:
    """Robust circle fit of the chain near ``index``.

    Returns ``dict(r, rms_circle, rms_line, span, sigma_r, n_inlier,
    n_total)`` or None.  The fit is iterative: a contour chain is a polygon
    through peak positions, so a +-12 window spans chain ends and corners, and
    a plain algebraic fit reports 1.5-3.6 px residuals on a limb whose radius
    it in fact recovers.  Outliers are rejected on the residual median/MAD, and
    only the CONTIGUOUS surviving run containing the anchor is kept, so a corner
    cannot splice two different limbs into one circle.

    ``rms_line`` is the residual of the straight-line model on the same points;
    ``chain_radius`` uses the ratio to decide whether the curvature is real.
    ``sigma_r`` is the first-order radius uncertainty from the sagitta:
    ``sigma_r = 8 R^2 rms_circle sqrt(2/n) / span^2``.
    """
    n = len(positions)
    if n < 5:
        return None
    k = max(2, int(window))
    if closed:
        idx = [(index + j) % n for j in range(-k, k + 1)]
    else:
        idx = [min(n - 1, max(0, index + j)) for j in range(-k, k + 1)]
    pts_all = np.array([positions[j] for j in idx], dtype=np.float64)
    # Clamping duplicates the window's ends; a duplicate point would vote
    # twice in the fit.
    uniq, first = np.unique(pts_all, axis=0, return_index=True)
    pts = uniq[np.argsort(first)]
    if pts.shape[0] < 5:
        return None
    keep = np.ones(pts.shape[0], dtype=bool)
    for _ in range(4):
        fit = _circle_fit(pts[keep])
        if fit is None:
            return None
        cx, cy, r = fit
        res = np.hypot(pts[:, 0] - cx, pts[:, 1] - cy) - r
        med = float(np.median(res[keep]))
        mad = float(np.median(np.abs(res[keep] - med))) * 1.4826
        new = np.abs(res - med) <= max(0.75, 3.0 * mad)
        a, b = _contiguous_run(new, pts.shape[0] // 2)
        new = np.zeros_like(new)
        if b >= a:
            new[a:b + 1] = True
        if int(new.sum()) < 5:
            return None
        if np.array_equal(new, keep):
            break
        keep = new
    fit = _circle_fit(pts[keep])
    if fit is None:
        return None
    cx, cy, r = fit
    res = np.hypot(pts[keep, 0] - cx, pts[keep, 1] - cy) - r
    rms_c = float(np.sqrt(np.mean(res * res)))
    sub = pts[keep]
    span = float(np.hypot(*(sub[-1] - sub[0])))
    sigma_r = 0.0
    if math.isfinite(r) and r > 0.0 and span > 1e-9:
        sigma_r = (8.0 * r * r * rms_c * math.sqrt(2.0 / sub.shape[0])
                   / (span * span))
        sigma_r = min(sigma_r, r)
    return dict(r=float(r), rms_circle=rms_c, rms_line=_line_rms(sub),
                span=span, sigma_r=float(sigma_r),
                n_inlier=int(keep.sum()), n_total=int(pts.shape[0]))


def _radius_from_geometry(geom: Optional[Dict[str, Any]]
                          ) -> Tuple[Optional[float], Optional[float]]:
    """(radius, radius sd) from a ``chain_geometry`` fit.

    ``inf`` means straight to the fit's own resolution: an edge-on plane, whose
    deficit is zero at every inset.  ``None`` means no usable fit.
    """
    if geom is None:
        return None, None
    r = geom["r"]
    if not math.isfinite(r) or r <= 0.0:
        return None, None
    if r > RIM_RADIUS_LIMIT_PX:
        return float("inf"), None
    # Unresolved curvature: the straight-line model explains the points as well
    # as the circle does, so the fitted radius is noise.  A limb that does not
    # demonstrably turn is straight, and a straight limb has no deficit.
    if geom["rms_line"] < RIM_STRAIGHT_RATIO * max(geom["rms_circle"], 1e-6):
        return float("inf"), None
    return float(r), float(geom["sigma_r"])


def chain_radius(positions: Sequence[Tuple[float, float]], index: int,
                 window: int = RIM_RADIUS_WINDOW,
                 closed: bool = False) -> Optional[float]:
    """Radius (px) of the chain's own circle fit around ``index``.

    ``inf`` when the chain is straight to the fit's own resolution (an edge-on
    plane: no slant deficit at any inset), ``None`` when there are too few
    members or the fit is degenerate (unknown -> the measured prior).
    """
    geom = chain_geometry(positions, index, window=window, closed=closed)
    return _radius_from_geometry(geom)[0]


def rim_deficit_deg(radius_px: Optional[float],
                    inset_mean: float = RIM_INSET_MEAN_PX,
                    inset_sd: float = RIM_INSET_SD_PX
                    ) -> Tuple[float, float]:
    """(mean, sd) of the slant deficit on the INSIDE sub-population.

    ``delta = degrees(sqrt(2 d / R))`` with ``d`` the measured inside inset.
    This is the limb law proper; it is NOT the cue's prior over all anchors,
    because 77 % of anchors sit outside their own silhouette and have zero
    deficit.  Use ``rim_slant_marginal`` for the cue.
    """
    if radius_px is not None and math.isinf(radius_px):
        return 0.0, 0.0
    if radius_px is None or not math.isfinite(radius_px) or radius_px <= 0.0:
        radius_px = RIM_RADIUS_FALLBACK_PX
    if inset_mean <= 0.0:
        return 0.0, 0.0
    z = math.sqrt(2.0 * inset_mean / radius_px)
    dz = inset_sd / math.sqrt(2.0 * inset_mean * radius_px)
    return math.degrees(z), math.degrees(dz)


def rim_deficit_samples(radius_px: Optional[float]) -> np.ndarray:
    """Deficit (deg) for every measured inset sample, at this radius."""
    if radius_px is not None and math.isinf(radius_px):
        return np.zeros(len(RIM_INSET_QUANTILES_PX), dtype=np.float64)
    if radius_px is None or not math.isfinite(radius_px) or radius_px <= 0.0:
        radius_px = RIM_RADIUS_FALLBACK_PX
    d = np.asarray(RIM_INSET_QUANTILES_PX, dtype=np.float64)
    return np.degrees(np.sqrt(2.0 * np.maximum(d, 0.0) / float(radius_px)))


def rim_slant_marginal(radius_px: Optional[float],
                       radius_sd_px: Optional[float] = None
                       ) -> np.ndarray:
    """The contour cue's slant marginal: the measured deficit distribution.

    Each measured inset sample contributes ``90 - delta(d, R)`` with the
    measured law residual as its kernel; the radius's own uncertainty widens
    the kernel through ``d(delta)/dR = delta / 2R``.  The result is normalized,
    so the cue's shape is data and its width is derived.
    """
    deficits = rim_deficit_samples(radius_px)
    sl = np.asarray(SLANT_CENTERS_DEG, dtype=np.float64)
    w = np.zeros_like(sl)
    for dl in deficits:
        sigma = RIM_GRID_SIGMA_DEG
        if (radius_sd_px and dl > 0.0 and radius_px is not None
                and math.isfinite(radius_px) and radius_px > 0.0):
            sigma = math.hypot(sigma,
                               float(dl) * float(radius_sd_px)
                               / (2.0 * float(radius_px)))
        w += _slant_kernel(sl, 90.0 - float(dl), sigma)
    total = float(w.sum())
    if total <= 0.0:
        return np.full_like(sl, 1.0 / sl.size)
    return w / total

_DEG = math.pi / 180.0


# ---------------------------------------------------------------------------
# populations
# ---------------------------------------------------------------------------

@dataclass
class Population:
    """A normalized mass function on the shared (slant, tilt) grid."""

    mass: np.ndarray = field(
        default_factory=lambda: np.zeros(
            (len(SLANT_CENTERS_DEG), len(TILT_CENTERS_DEG)),
            dtype=np.float64))

    def normalized(self) -> "Population":
        total = float(self.mass.sum())
        if not math.isfinite(total) or total <= 0.0:
            return Population(np.zeros_like(self.mass))
        return Population(self.mass / total)

    def total(self) -> float:
        return float(self.mass.sum())

    def add(self, other: "Population") -> "Population":
        return Population(self.mass + other.mass)


def _wrap_deg(a: np.ndarray) -> np.ndarray:
    return (a + 180.0) % 360.0 - 180.0


def _slant_kernel(sl: np.ndarray, slant_deg: float,
                  sigma_slant_deg: float) -> np.ndarray:
    """Separable Gaussian in slant, folded at 0 and 90 deg.

    Slant is a polar angle on the VISIBLE hemisphere, so a plain Gaussian
    centred at 90 would lose half its mass and pull the estimate toward the
    image plane; the fold puts that mass back on the visible side, which makes
    the population's MODE exactly the cue's stated value.  The mean of the
    folded (half-normal) distribution is pulled inward by ~0.8 sigma, which is
    a property of a bounded representation: the calibration report measures the
    resulting bias and width rather than assuming them away.
    """
    ss = max(1e-3, float(sigma_slant_deg))
    d = np.exp(-0.5 * ((sl - slant_deg) / ss) ** 2)
    d += np.exp(-0.5 * ((-sl - slant_deg) / ss) ** 2)
    d += np.exp(-0.5 * ((180.0 - sl - slant_deg) / ss) ** 2)
    return d


def gaussian_population(slant_deg: float, tilt_deg: float,
                        sigma_slant_deg: float,
                        sigma_tilt_deg: float) -> Population:
    """A separable Gaussian on the grid, normalized.

    Slant is reflected at the hemisphere boundary; tilt is wrapped.  A von
    Mises would be the exact circular model, but the calibrated tilt widths
    are measured as RMS errors and the report prints them; the Gaussian is
    the repeatability model, not a hidden tuning knob.
    """
    ss = max(1e-3, float(sigma_slant_deg))
    st = max(1e-3, float(sigma_tilt_deg))
    sl = np.asarray(SLANT_CENTERS_DEG, dtype=np.float64)[:, None]
    ti = np.asarray(TILT_CENTERS_DEG, dtype=np.float64)[None, :]
    ds = _slant_kernel(sl, float(slant_deg), ss)
    dt = _wrap_deg(ti - float(tilt_deg)) / st
    w = ds * np.exp(-0.5 * dt * dt)
    return Population(w).normalized()


def _population_at_slant_tilt(slant_deg: float, tilt_deg: float,
                              sigma_slant_deg: float,
                              sigma_tilt_deg: float) -> np.ndarray:
    ss = max(1e-3, float(sigma_slant_deg))
    st = max(1e-3, float(sigma_tilt_deg))
    sl = np.asarray(SLANT_CENTERS_DEG, dtype=np.float64)[:, None]
    ti = np.asarray(TILT_CENTERS_DEG, dtype=np.float64)[None, :]
    ds = _slant_kernel(sl, float(slant_deg), ss)
    dt = _wrap_deg(ti - float(tilt_deg)) / st
    return ds * np.exp(-0.5 * dt * dt)


def fuse_populations(pops: Sequence[Population]) -> Population:
    """Product of likelihoods over the shared grid, renormalized.

    Cues are conditionally independent evidence about one normal, so their
    normalized masses multiply: agreement sharpens, disagreement is penalized
    in proportion to how confident each cue is.  No learned weights -- a cue's
    measured width is its precision.  The maximum is subtracted before
    exponentiating so the result is defined even when the cues conflict.
    """
    if not pops:
        return Population()
    acc = None
    for p in pops:
        if p.total() <= 0.0:
            continue
        q = np.maximum(p.normalized().mass, 1e-300)
        acc = np.log(q) if acc is None else acc + np.log(q)
    if acc is None:
        return Population()
    w = np.exp(acc - float(acc.max()))
    return Population(w).normalized()


def mixture_populations(pops: Sequence[Population]) -> Population:
    """Sum of normalized masses (a belief average).

    This is the control: a mixture cannot beat its sharpest component in
    general, and it is reported next to the product so the difference is
    visible rather than asserted.
    """
    if not pops:
        return Population()
    total = None
    for p in pops:
        if p.total() <= 0.0:
            continue
        q = p.normalized().mass
        total = q if total is None else total + q
    if total is None:
        return Population()
    return Population(total).normalized()


def population_moment(pop: Population) -> Optional[Tuple[float, float, float]]:
    """(slant_deg, tilt_deg, confidence) of a population.

    Slant is the marginal mean; tilt is the circular mean.  Confidence is the
    resultant length of the tilt marginal in [0, 1] -- a population with two
    symmetric lobes reports low confidence rather than a fake precise angle.
    """
    m = pop.mass
    s = float(m.sum())
    if not math.isfinite(s) or s <= 0.0:
        return None
    sl = np.asarray(SLANT_CENTERS_DEG, dtype=np.float64)
    ti = np.asarray(TILT_CENTERS_DEG, dtype=np.float64) * _DEG
    ps = m.sum(axis=1) / s
    pt = m.sum(axis=0) / s
    slant = float(np.dot(ps, sl))
    c = float(np.dot(pt, np.cos(ti)))
    sn = float(np.dot(pt, np.sin(ti)))
    tilt = math.degrees(math.atan2(sn, c))
    conf = math.hypot(c, sn)
    return slant, tilt, conf


# ---------------------------------------------------------------------------
# light prior
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LightPrior:
    """Directional light and ambient level, estimated from a corpus.

    ``direction`` is unit and points from the surface TOWARD the light, in the
    same (x right, y down, z toward viewer) frame as the image.
    """

    direction: Tuple[float, float, float]
    ambient: float = 0.0

    @property
    def vector(self) -> np.ndarray:
        return np.asarray(self.direction, dtype=np.float64)


# Corpus floor search (Session B).  The floor is the ambient of the calibration
# renders; the search covers the bottom 5 % of the pooled object luminance,
# which is where an ambient level can possibly live.  ``_LIGHT_MIN_SIGNAL`` is
# ~5 quantization steps: a pixel this far above the floor counts as lit.
_LIGHT_FLOOR_QUANTILE = 0.05
_LIGHT_MIN_SIGNAL = 0.02
_SHADOW_EPS = 0.02


def _light_fit(normals: np.ndarray, luminance: np.ndarray, floor: float,
               iters: int = 2) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """``(u, kept_normals)`` from one scene at a fixed ambient floor.

    ``u = rho * L``.  Pixels within ``_LIGHT_MIN_SIGNAL`` of the floor do not
    vote (they are shadow or transition), and the shadowed pixels
    (``n . u <= 0``) are dropped after the first fit because they obey
    ``I = floor`` rather than the linear model.
    """
    y = luminance - float(floor)
    lit = y > _LIGHT_MIN_SIGNAL
    if int(lit.sum()) < 50:
        return None
    Cn = normals[lit]
    yn = y[lit]
    try:
        u, *_ = np.linalg.lstsq(Cn, yn, rcond=None)
    except np.linalg.LinAlgError:
        return None
    m = np.ones(int(lit.sum()), dtype=bool)
    for _ in range(iters):
        m = (Cn @ u) > _SHADOW_EPS * max(float(np.linalg.norm(u)), 1e-9)
        if int(m.sum()) < 50:
            break
        try:
            u, *_ = np.linalg.lstsq(Cn[m], yn[m], rcond=None)
        except np.linalg.LinAlgError:
            return None
    if not np.all(np.isfinite(u)) or float(np.linalg.norm(u)) < 1e-6:
        return None
    return u, Cn[m]


def estimate_light_prior(luminance: Sequence[np.ndarray],
                         normals: Sequence[np.ndarray],
                         albedo_hint: float = 0.85
                         ) -> LightPrior:
    """Least-squares light direction from a corpus of untextured renders.

    Model: ``I = ambient + rho * max(0, n . L)``.  With ``u = rho * L`` this is
    linear in ``u`` on the lit pixels; the direction of ``u`` is the light
    direction and its length is ``rho``.  Only pixels clearly above the ambient
    floor vote, and the shadowed ones are dropped after the first fit because
    they obey ``I = ambient``, not the linear model.

    THE FLOOR IS CHOSEN FROM THE CORPUS, NOT FROM ONE SCENE.  Session B found
    that a per-scene "2nd percentile of the surface luminance" floor is not
    shape-agnostic: a fully lit cylinder whose darkest visible cap is at 0.20
    makes the floor 0.20, and subtracting it biases the fitted direction by
    12 deg.  Nothing in a single frame distinguishes "0.20 is the ambient"
    from "0.20 is a lit cap".  The corpus can: the ambient is the floor at
    which every scene's image is explained by the Lambertian model.  So the
    candidates are quantiles of the pooled object luminance from 0 to 5 %,
    each scene is fitted at each candidate, and the candidate with the
    smallest GLOBAL residual (over all object pixels, shadowed included) wins.
    That criterion cannot be gamed by raising the floor, because raising it
    breaks the prediction for every pixel below it.

    The truth normals here belong to the CALIBRATION CORPUS, not to any one
    fragment; this is how the corpus prior is estimated, exactly like a
    calibration target.  The fitted prior is then a constant of the estimator.

    Only calibration scenes whose visible normals span three dimensions
    contribute: a single flat face or a cylinder's side is a one-light
    ambiguity and says nothing about which direction a fixed light came from.
    The rank test is on the scene's own normals, which is calibration data.
    """
    scenes: List[Tuple[np.ndarray, np.ndarray]] = []
    pooled: List[np.ndarray] = []
    for lum, nrm in zip(luminance, normals):
        L = np.asarray(lum, dtype=np.float64).ravel()
        N = np.asarray(nrm, dtype=np.float64).reshape(-1, 3)
        ok = np.isfinite(N).all(axis=1)
        if not np.any(ok):
            continue
        scenes.append((N[ok], L[ok]))
        pooled.append(L[ok])
    if not scenes:
        return LightPrior((0.0, 0.0, 1.0), 0.0)

    # Candidate floors: 0 and the bottom 5 % of the pooled object luminance.
    lo = float(np.min(np.concatenate(pooled)))
    hi = float(np.quantile(np.concatenate(pooled), _LIGHT_FLOOR_QUANTILE))
    candidates = np.unique(np.concatenate(
        ([lo, 0.0] if lo <= 0.0 else [lo],
         np.linspace(lo, hi, 21))))

    best: Optional[Tuple[float, float, List[np.ndarray]]] = None
    for floor in candidates:
        fits = []
        score_num, score_den = 0.0, 0
        for N, L in scenes:
            fit = _light_fit(N, L, float(floor))
            if fit is None:
                continue
            u, kept_n = fit
            fits.append(fit)
            # Global residual of the FULL model, shadowed pixels included.
            pred = float(floor) + np.clip(N @ u, 0.0, None)
            score_num += float(np.sum((L - pred) ** 2))
            score_den += int(L.size)
        if not fits or score_den < 100:
            continue
        score = score_num / float(score_den)
        if best is None or score < best[0] - 1e-15:
            best = (score, float(floor), [f[0] for f in fits],
                    [f[1] for f in fits])
    if best is None:
        return LightPrior((0.0, 0.0, 1.0), 0.0)

    _, floor, us, kept = best
    kept_units: List[np.ndarray] = []
    weights: List[float] = []
    for u, kept_n in zip(us, kept):
        s = np.linalg.svd(kept_n, compute_uv=False)
        if s.size < 3 or s[0] <= 1e-12 or s[2] / s[0] < 0.05:
            continue
        n = float(np.linalg.norm(u))
        if not np.all(np.isfinite(u)) or n < 1e-6:
            continue
        kept_units.append(u / n)
        weights.append(float(kept_n.shape[0]))
    if not kept_units:
        return LightPrior((0.0, 0.0, 1.0), float(floor))
    u = np.average(np.stack(kept_units, axis=0), axis=0,
                   weights=np.asarray(weights, dtype=np.float64))
    n = float(np.linalg.norm(u))
    if n < 1e-9:
        return LightPrior((0.0, 0.0, 1.0), float(floor))
    return LightPrior(tuple(u / n), float(floor))


# ---------------------------------------------------------------------------
# contour geometry
# ---------------------------------------------------------------------------

def _circle_fit(points: np.ndarray) -> Optional[Tuple[float, float, float]]:
    """Algebraic circle fit; (cx, cy, r) or None when degenerate/straight."""
    if points.shape[0] < 5:
        return None
    x = points[:, 0]
    y = points[:, 1]
    A = np.stack([x, y, np.ones_like(x)], axis=1)
    rhs = -(x * x + y * y)
    try:
        sol, *_ = np.linalg.lstsq(A, rhs, rcond=None)
    except np.linalg.LinAlgError:
        return None
    cx = -0.5 * float(sol[0])
    cy = -0.5 * float(sol[1])
    r2 = cx * cx + cy * cy - float(sol[2])
    if not math.isfinite(r2) or r2 <= 0.0:
        return None
    return cx, cy, math.sqrt(r2)


def contour_curvature(positions: Sequence[Tuple[float, float]],
                      index: int, window: int = 3,
                      closed: bool = False) -> Optional[float]:
    """Signed curvature (1/px) at ``index`` from a local circle fit.

    The sign is in TRAVEL order: positive when the centre of curvature lies on
    the left of travel.  The span is 2*window+1 members, which is flat over
    +/-2..6 members on the synthetic corpus; +/-3 is the fixed default.
    """
    n = len(positions)
    if n < 5:
        return None
    k = max(2, int(window))
    idx = []
    for j in range(-k, k + 1):
        if closed:
            idx.append((index + j) % n)
        else:
            idx.append(min(n - 1, max(0, index + j)))
    pts = np.array([positions[j] for j in idx], dtype=np.float64)
    fit = _circle_fit(pts)
    if fit is None:
        return None
    cx, cy, r = fit
    if r > 2000.0:
        return 0.0
    px, py = positions[index]
    if closed:
        ia, ib = (index - 1) % n, (index + 1) % n
    else:
        ia, ib = max(0, index - 1), min(n - 1, index + 1)
    tx = positions[ib][0] - positions[ia][0]
    ty = positions[ib][1] - positions[ia][1]
    mag = math.hypot(tx, ty)
    if mag < 1e-9:
        return None
    lx, ly = -ty / mag, tx / mag
    side = (cx - px) * lx + (cy - py) * ly
    return float(math.copysign(1.0 / max(r, 1.0), side))


def _owner_normal(record: Dict[str, Any]) -> Optional[Tuple[float, float, float]]:
    """(unit inward x, unit inward y, signed ownership) or None."""
    signed = float(record.get("signed", 0.0) or 0.0)
    n = record.get("normal")
    if n is None or len(n) < 2:
        return None
    nx, ny = float(n[0]), float(n[1])
    mag = math.hypot(nx, ny)
    if mag < 1e-9:
        return None
    nx, ny = nx / mag, ny / mag
    if abs(signed) < _AMBIGUOUS_OWNERSHIP:
        return None
    s = 1.0 if signed > 0.0 else -1.0
    return s * nx, s * ny, s


def contour_cue(record: Dict[str, Any], curvature: Optional[float],
                sigma_slant_deg: float, sigma_tilt_deg: float,
                radius_px: Optional[float] = None,
                rim: bool = True,
                radius_sd_px: Optional[float] = None
                ) -> Tuple[Optional[Population], Dict[str, float]]:
    """The exact cue.  Returns (population or None, diagnostics).

    Tilt is the azimuth of the visible surface normal, which points away from
    the figure: ``-owner``.  Curvature sign is owner-relative: positive when
    the centre of curvature lies on the owner side.

    With ``rim=True`` (the Session B form) the slant is not a flat 90 deg and
    not a single limb-law value either: the anchor's signed inset is unknown
    (77 % of measured anchors are outside their own silhouette, where the
    deficit is zero), so the slant marginal is the limb law applied to the
    MEASURED inset distribution -- ``rim_slant_marginal``.  Its width is that
    distribution's width, widened by the radius fit's own uncertainty; it is
    not a fitted constant.  With ``rim=False`` the Session A behaviour is
    reproduced exactly, for the ablation in the report.
    """
    on = _owner_normal(record)
    if on is None:
        return None, {"valid": 0.0}
    ox, oy, s = on
    tilt = math.degrees(math.atan2(-oy, -ox))
    diag: Dict[str, float] = {
        "valid": 1.0,
        "owner_side": float(s),
        "sigma_tilt": float(sigma_tilt_deg),
    }
    if rim:
        marginal = rim_slant_marginal(radius_px, radius_sd_px)
        sl = np.asarray(SLANT_CENTERS_DEG, dtype=np.float64)
        ti = np.asarray(TILT_CENTERS_DEG, dtype=np.float64)[None, :]
        dt = _wrap_deg(ti - tilt) / max(1e-3, float(sigma_tilt_deg))
        w = marginal[:, None] * np.exp(-0.5 * dt * dt)
        pop = Population(w).normalized()
        deficit = float(np.dot(marginal, 90.0 - sl))
        var = float(np.dot(marginal, (90.0 - sl - deficit) ** 2))
        diag["slant"] = 90.0 - deficit
        diag["deficit"] = deficit
        diag["sigma_slant"] = math.sqrt(max(var, 0.0))
        diag["p_outside"] = float(np.mean(
            np.asarray(RIM_INSET_QUANTILES_PX) <= 0.0))
    else:
        pop = gaussian_population(90.0, tilt, sigma_slant_deg,
                                  sigma_tilt_deg)
        diag["slant"] = 90.0
        diag["deficit"] = 0.0
        diag["sigma_slant"] = float(sigma_slant_deg)
    if radius_px is not None and math.isfinite(radius_px):
        diag["radius_px"] = float(radius_px)
        diag["radius_kind"] = 1.0
    elif radius_px is not None and math.isinf(radius_px):
        # straight to the fit's own resolution: an edge-on plane, no deficit
        diag["radius_kind"] = 2.0
    else:
        # no usable fit: the measured inset prior at the fallback radius
        diag["radius_kind"] = 0.0
    if radius_sd_px is not None and math.isfinite(radius_sd_px):
        diag["radius_sd_px"] = float(radius_sd_px)
    if curvature is not None:
        diag["curvature"] = float(curvature)
    return pop, diag


# ---------------------------------------------------------------------------
# shading
# ---------------------------------------------------------------------------

def _patch(lum: np.ndarray, cx: float, cy: float, radius_px: float
           ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Disk samples: (dx_px, dy_px, intensity)."""
    H, W = lum.shape
    rr = int(math.ceil(radius_px))
    xi, yi = int(round(cx)), int(round(cy))
    x0, x1 = max(0, xi - rr), min(W, xi + rr + 1)
    y0, y1 = max(0, yi - rr), min(H, yi + rr + 1)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None
    ys, xs = np.mgrid[y0:y1, x0:x1]
    dx = xs + 0.5 - cx
    dy = ys + 0.5 - cy
    ok = (dx * dx + dy * dy) <= radius_px * radius_px
    if int(ok.sum()) < 12:
        return None
    return dx[ok], dy[ok], lum[y0:y1, x0:x1][ok].astype(np.float64)


def _patch_is_smooth(dx: np.ndarray, dy: np.ndarray, I: np.ndarray) -> bool:
    """Reject a patch that contains a luminance step (a boundary or texture).

    The criterion is relative to the patch's own gradient distribution, so it
    has no absolute constant: the largest neighbouring difference must not
    exceed 6x the median plus a small floor.
    """
    if I.size < 12:
        return False
    H = int(max(dx.max() - dx.min(), dy.max() - dy.min())) + 2
    grid = np.full((H + 1, H + 1), np.nan)
    gx = np.round(dx - dx.min()).astype(int)
    gy = np.round(dy - dy.min()).astype(int)
    grid[gy, gx] = I
    diffs = []
    for a, b in ((0, 1), (1, 0)):
        A = grid[: grid.shape[0] - a, : grid.shape[1] - b]
        B = grid[a:, b:]
        d = np.abs(A - B)
        d = d[np.isfinite(d)]
        if d.size:
            diffs.append(d)
    if not diffs:
        return False
    d = np.concatenate(diffs)
    med = float(np.median(d))
    return bool(float(d.max()) <= 6.0 * (med + 1e-3))


#: The umbilic model's validity test.  A sphere's shading has isotropic
#: second-order structure -- the Hessian of the luminance over the patch has two
#: comparable eigenvalues -- while a cylinder's is rank one.  The model's
#: residual CANNOT see the difference (a plane is a sphere of infinite radius,
#: so the residual sits at the storage floor), which is how the cue came to be
#: confidently wrong on cylinders: 57.9 deg median anchor error with a 17.5 deg
#: claimed width.  The measured eigenvalue ratio is bimodal with an empty gap
#: (cylinders <= 0.05, spheres mostly >= 0.10), so the gate sits in the gap and
#: is a statement about the model class, not about any object label.
UMBILIC_RATIO_MIN = 0.05


def umbilic_ratio(dx: np.ndarray, dy: np.ndarray, I: np.ndarray
                  ) -> Optional[float]:
    """Ratio of the smaller to the larger luminance Hessian eigenvalue."""
    if I.size < 12:
        return None
    A = np.stack([np.ones_like(dx), dx, dy,
                  0.5 * dx * dx, dx * dy, 0.5 * dy * dy], axis=1)
    coef, *_ = np.linalg.lstsq(A, I, rcond=None)
    hxx, hxy, hyy = float(coef[3]), float(coef[4]), float(coef[5])
    w = np.linalg.eigvalsh(np.array([[hxx, hxy], [hxy, hyy]]))
    big = float(max(abs(w[0]), abs(w[1])))
    if big <= 0.0:
        return None
    small = float(min(abs(w[0]), abs(w[1])))
    return small / big


def shade_umbilic(lum: np.ndarray, cx: float, cy: float,
                  light: LightPrior, *, patch_radius_px: float = 5.0,
                  max_radius_px: float = 600.0,
                  quantization_step: float = 1.0 / 255.0,
                  residual_sigma: float = 2.0
                  ) -> Optional[Dict[str, float]]:
    """Fit a local sphere cap to raw luminance; return the normal at (cx, cy).

    The model is derived in the module docstring.  ``radius_px`` is the
    sphere's radius in PIXELS (the ratio d/R is scale-free, so no camera
    calibration is needed).  Returns None when the patch is not smooth, the
    fit does not converge, or the residual is larger than the measurement
    noise can explain.

    ``quantization_step`` is the storage step of the luminance input (1/255
    for the 8-bit render).  The unit-step rounding noise is uniform on
    +/- half a step, so its rms is step/sqrt(12); a residual above
    ``residual_sigma`` times that is structured variation the smooth
    constant-albedo Lambertian model does NOT explain -- i.e. albedo texture
    or a shading discontinuity, not sensor noise.  Such a patch is outside
    the cue's stated domain and the cue abstains.
    """
    p = _patch(lum, cx, cy, patch_radius_px)
    if p is None:
        return None
    dx, dy, I = p
    if not _patch_is_smooth(dx, dy, I):
        return None
    L = light.vector
    a = float(light.ambient)

    def model(slant, tilt, R, rho):
        nx = math.sin(slant) * math.cos(tilt) + dx / R
        ny = math.sin(slant) * math.sin(tilt) + dy / R
        q = nx * nx + ny * ny
        nz = np.sqrt(np.maximum(0.0, 1.0 - q))
        lam = nx * L[0] + ny * L[1] + nz * L[2]
        return a + rho * np.maximum(0.0, lam)

    best = None
    for sl in np.deg2rad((0.0, 18.0, 36.0, 54.0, 70.0, 84.0)):
        for ti in np.deg2rad(np.arange(0.0, 360.0, 15.0)):
            for Rp in (6.0, 12.0, 25.0, 50.0, 100.0, 200.0):
                m = model(sl, ti, Rp, 1.0) - a
                den = float(np.sum(m * m))
                if den < 1e-9:
                    continue
                rho = float(np.sum((I - a) * m) / den)
                if not math.isfinite(rho) or rho <= 0.02 or rho >= 2.0:
                    continue
                sse = float(np.sum((a + rho * m - I) ** 2))
                if best is None or sse < best[0]:
                    best = (sse, sl, ti, Rp, rho)
    if best is None:
        return None
    _, sl0, ti0, R0, rho0 = best

    def resid(v):
        return model(v[0], v[1], v[2], v[3]) - I

    try:
        sol = least_squares(
            resid, x0=[sl0, ti0, R0, rho0],
            bounds=([0.0, -math.pi, 2.0, 0.02],
                    [math.radians(88.0), 3.0 * math.pi,
                     max_radius_px, 2.0]),
            method="trf", xtol=1e-10, ftol=1e-10, gtol=1e-10,
            max_nfev=200,
        )
    except Exception:
        return None
    if not sol.success or not np.all(np.isfinite(sol.x)):
        return None
    sl, ti, R, rho = (float(v) for v in sol.x)
    rms = float(np.sqrt(np.mean(resid(sol.x) ** 2)))
    if not math.isfinite(rms) or R >= max_radius_px - 1e-6:
        return None
    # Model validity: the residual must be explainable by the storage
    # quantization of the luminance input.  See the docstring.
    noise_sigma = float(quantization_step) / math.sqrt(12.0)
    if noise_sigma > 0.0 and rms > float(residual_sigma) * noise_sigma:
        return None
    # Identifiability.  A patch whose fitted light response barely changes
    # across it carries no normal information: a plane under one light is
    # exactly ambiguous, and a constant patch fits any normal with rho -> 0.
    # The floor is 0.03 intensity (about eight 8-bit quantization levels),
    # which is a sensor-noise scale, not a score-driven constant.  This is the
    # observable version of the bas-relief ambiguity.
    nx = math.sin(sl) * math.cos(ti) + dx / R
    ny = math.sin(sl) * math.sin(ti) + dy / R
    nz = np.sqrt(np.maximum(0.0, 1.0 - nx * nx - ny * ny))
    lam = nx * L[0] + ny * L[1] + nz * L[2]
    span = float(lam.max() - lam.min())
    if float(lam.mean()) <= 0.0 or rho * span < max(3.0 * rms, 0.03):
        return None
    # MODEL VALIDITY: the patch must be umbilic, not flat along one direction.
    # The residual cannot detect this (a cylinder's shading is explained by a
    # sphere of large radius at the storage floor), so the second-order
    # structure is tested directly.  Applied at the PATCH centre, before the
    # normal is moved to the anchor.
    anisotropy = umbilic_ratio(dx, dy, I)
    if anisotropy is not None and anisotropy < UMBILIC_RATIO_MIN:
        return None
    # Normalise the fitted normal AT THE PATCH CENTRE (ti in [0, 2pi)).
    n_c = np.array([math.sin(sl) * math.cos(ti),
                    math.sin(sl) * math.sin(ti),
                    math.cos(sl)])
    return dict(normal=tuple(n_c), radius_px=R, rho=rho, rms=rms,
                umbilic_ratio=float(anisotropy) if anisotropy is not None
                else None,
                slant=math.degrees(sl), tilt=math.degrees(ti % (2 * math.pi)),
                # Conditioning: how far the patch sits from the terminator.
                # Near 0 the clamped Lambertian term destroys the gradient
                # information, so this is the observable that predicts a
                # shading failure (reported, and used as an abstain margin).
                light_mean=float(lam.mean()), light_span=span,
                light_min=float(lam.min()), light_floor=float(a))


def extrapolate_normal(n_fit: np.ndarray, radius_px: float,
                       offset_px: Tuple[float, float],
                       tol_px: float = 0.0
                       ) -> Optional[Tuple[float, float, float]]:
    """Move a fitted sphere normal from the patch centre to the anchor.

    ``offset_px`` points from the patch centre to the anchor.  Since
    ``n_xy(d) = n_xy(0) + d / R``, the anchor's normal is
    ``n_fit_xy + offset_px / R``.

    AN ANCHOR CAN SIT OUTSIDE ITS OWN LIMB.  Measured on the gated contour
    population, 77 % of anchors are outside, by up to a few pixels.  The
    extrapolated radial distance is then beyond ``R``, and the nearest surface
    point is the limb itself, whose normal is perpendicular to the view.  The
    result is therefore CLAMPED to the limb instead of rejected.  Rejecting it
    was suppressing the shading cue on 36 of the 46 fragments where it could
    otherwise have fired, purely because the arithmetic landed on ``q = 1``.

    ``tol_px`` is how far outside the limb the anchor may be before the model
    stops describing the surface (the caller passes its patch radius, so the
    anchor is never further out than the patch was in).  Beyond it the cue
    abstains rather than inventing a normal.
    """
    if radius_px is None or radius_px <= 0.0:
        return None
    nx = float(n_fit[0]) + float(offset_px[0]) / float(radius_px)
    ny = float(n_fit[1]) + float(offset_px[1]) / float(radius_px)
    q = nx * nx + ny * ny
    if q <= 1.0:
        return nx, ny, math.sqrt(1.0 - q)
    # Beyond the limb: the limb normal is the nearest surface normal.
    if (math.sqrt(q) - 1.0) * float(radius_px) > float(tol_px):
        return None
    s = 1.0 / math.sqrt(q)
    return nx * s, ny * s, 0.0


def shading_cue(lum: np.ndarray, anchor: Tuple[float, float],
                owner: Tuple[float, float], light: LightPrior, *,
                offset_px: float = 7.0, patch_radius_px: float = 5.0,
                sigma_slant_deg: float = 20.0,
                sigma_tilt_deg: float = 20.0
                ) -> Tuple[Optional[Population], Dict[str, float]]:
    """Shading population at ``anchor``, evaluated on a patch INSIDE the owner.

    The patch is offset inward so it does not straddle the contour; the fitted
    sphere normal is then moved back to the anchor analytically, so the cue is
    comparable with the contour cue at the same point.
    """
    ox, oy = owner
    fx = anchor[0] + ox * offset_px
    fy = anchor[1] + oy * offset_px
    fit = shade_umbilic(lum, fx, fy, light,
                        patch_radius_px=patch_radius_px)
    if fit is None:
        return None, {"valid": 0.0}
    # offset from the patch centre back to the anchor, in pixels
    n = extrapolate_normal(np.asarray(fit["normal"]), fit["radius_px"],
                           (-ox * offset_px, -oy * offset_px),
                           tol_px=float(patch_radius_px))
    if n is None:
        return None, {"valid": 0.0}
    slant = math.degrees(math.acos(max(0.0, min(1.0, float(n[2])))))
    tilt = math.degrees(math.atan2(n[1], n[0]))
    pop = gaussian_population(slant, tilt, sigma_slant_deg, sigma_tilt_deg)
    diag = {
        "valid": 1.0,
        "slant": float(slant),
        "tilt": float(tilt),
        "radius_px": float(fit["radius_px"]),
        "rho": float(fit["rho"]),
        "rms": float(fit["rms"]),
        "light_mean": float(fit["light_mean"]),
        "light_span": float(fit["light_span"]),
        "light_min": float(fit["light_min"]),
        "modulation": float(fit["rho"] * fit["light_span"]),
        "umbilic_ratio": fit.get("umbilic_ratio"),
        "sigma_slant": float(sigma_slant_deg),
        "sigma_tilt": float(sigma_tilt_deg),
    }
    return pop, diag


# ---------------------------------------------------------------------------
# fragments
# ---------------------------------------------------------------------------

@dataclass
class SurfaceFragment:
    """One owner-side surface point with its graded cue populations.

    ``variation`` is the measured curvature precondition (the smallest
    monotone inward luminance step, see ``inward_luminance_step``); it is a
    gate statistic, never part of the estimate.  ``enclosure`` is the interior
    field's contour gate.  Identity, if any, never includes either.
    """

    index: int
    position: Tuple[float, float]
    tangent: Tuple[float, float]
    owner_normal: Tuple[float, float]
    owner_side: float
    curvature: Optional[float]
    curvature_sign: float
    depth_layer: Optional[int]
    alpha: Any
    chain: str
    vertex: int
    enclosure: Optional[float]
    # [O4] Enclosure-weighted luminance pooled within the fragment's
    # receptive field: the interior field weights the luminance plane so
    # that the pooled value is pulled toward enclosed (figure) material.
    enclosure_weighted_lum: Optional[float] = None
    variation: Optional[float] = None
    cues: Dict[str, Population] = field(default_factory=dict)
    cue_diag: Dict[str, Dict[str, float]] = field(default_factory=dict)
    combined: Optional[Population] = None
    estimate: Optional[Tuple[float, float, float]] = None
    tier: Optional[str] = None  # FACT / HYPOTHESIS (weighted cue fusion)

    # --- [U20 LEAD 4] MATERIAL PAYLOAD (schema v1 slots, diagnostics/_u19_payload_schema.md)
    #
    # PRESENCE CONDITION (stated once, used everywhere below): a material slot
    # is present when this fragment's OWNER-SIDE FLANK PATCH is fully inside
    # the image AND the measured material is APPROXIMATELY CONSTANT across
    # that patch -- the interquartile spread of the patch's luminance is at
    # most MATERIAL_CONSTANCY_TOL (8 of the 255 storage levels).  A region
    # whose material varies faster than that is a TEXTURE, not a constant
    # material, and the slots abstain (None) rather than average it away.
    #
    # ABSENT is None, never 0 and never a default (schema v1).  Each slot has
    # its own further condition, stated on the slot:
    #   material_lum  measured on the flank patch (luminance)
    #   material_hue  requires an RGB plane AND a chromatic patch
    #                 (saturation >= P.COLOR_ACHROMATIC_EPS; hue is noise
    #                 below it, the tree's own achromatic rule)
    #   material_sat  requires an RGB plane
    #   material_tex  measured on the flank patch (normalised autocorrelation
    #                 peak = the tree's own texture-regularity statistic)
    #   polarity      the ownership record's signed value; ABSENT on an
    #                 undecided fragment (no side claim was made)
    #   motion_x/y    requires two frames with a known displacement; ABSENT on
    #                 the single-frame (2D static) path
    material_lum: Optional[float] = None
    material_hue: Optional[float] = None
    material_sat: Optional[float] = None
    material_tex: Optional[float] = None
    polarity: Optional[float] = None
    motion_x: Optional[float] = None
    motion_y: Optional[float] = None

    def owner_is_figure(self) -> bool:
        return self.owner_side != 0.0

    def material_present(self) -> bool:
        """Is the material payload present (any measured material slot)?"""
        return (self.material_lum is not None
                or self.material_sat is not None
                or self.material_tex is not None)


#: The stated tolerance of "approximately constant material": the
#: interquartile spread of the owner-side flank patch, in storage levels of
#: 1/255.  Eight levels is the point where a smooth ramp still counts as one
#: material while a real texture does not; it is a PRESENCE tolerance, never a
#: weight (no cue is scaled by it).
MATERIAL_CONSTANCY_TOL = 8.0 / 255.0


def _flank_window(shape: Tuple[int, int], pos: Tuple[float, float],
                  normal: Tuple[float, float], offset_px: float,
                  radius_px: float
                  ) -> Optional[Tuple[Tuple[int, int, int, int], np.ndarray]]:
    """The owner-side flank window: ((x0, y0, x1, y1), mask) or None.

    The window is centred `offset_px` INSIDE the owner (along the owner
    normal) and is required to lie ENTIRELY inside the image: a material claim
    on a patch the frame cuts in half would be a claim about a region that was
    never measured, so it abstains.
    """
    H, W = int(shape[0]), int(shape[1])
    nx, ny = float(normal[0]), float(normal[1])
    mag = math.hypot(nx, ny)
    if mag <= 1e-9:
        return None
    nx, ny = nx / mag, ny / mag
    cx = float(pos[0]) + nx * float(offset_px)
    cy = float(pos[1]) + ny * float(offset_px)
    r = float(radius_px)
    rr = int(math.ceil(r))
    xi, yi = int(round(cx)), int(round(cy))
    x0, y0 = xi - rr, yi - rr
    x1, y1 = xi + rr + 1, yi + rr + 1
    if x0 < 0 or y0 < 0 or x1 > W or y1 > H:
        return None
    ys, xs = np.mgrid[y0:y1, x0:x1]
    dx = xs + 0.5 - cx
    dy = ys + 0.5 - cy
    mask = (dx * dx + dy * dy) <= r * r
    # A patch of fewer than eight samples is a point, not a region: there is
    # no SPREAD to test for constancy, so the material abstains.  (Eight, not
    # sixteen: the caller's radius is the contour sampling step (2 px), whose
    # disk holds ~13 pixels.)
    if int(mask.sum()) < 8:
        return None
    return (x0, y0, x1, y1), mask


def material_payload(lum: np.ndarray,
                     pos: Tuple[float, float],
                     normal: Tuple[float, float],
                     *,
                     offset_px: float = 2.0,
                     radius_px: float = 2.0,
                     sat_plane: Optional[np.ndarray] = None,
                     hue_plane: Optional[np.ndarray] = None,
                     ) -> Tuple[Optional[float], Optional[float],
                                Optional[float], Optional[float]]:
    """(lum, tex, sat, hue) of the owner-side flank, or Nones (ABSENT).

    Every number here is a MEASUREMENT of the flank/enclosure sample the
    fragment already owns -- no new detector and no standalone Colour or
    Texture node is invented on the 2D path (U19's finding: the base modality
    lenses emit nothing there).

    Constancy is the presence condition: when the patch's interquartile
    luminance spread exceeds MATERIAL_CONSTANCY_TOL the material is not
    approximately constant and all four slots abstain.
    """
    win = _flank_window(np.asarray(lum).shape, pos, normal, offset_px,
                        radius_px)
    if win is None:
        return None, None, None, None
    (x0, y0, x1, y1), mask = win
    patch = np.asarray(lum, dtype=np.float64)[y0:y1, x0:x1]
    vals = patch[mask]
    if vals.size < 8 or not np.all(np.isfinite(vals)):
        return None, None, None, None
    q25, q75 = np.percentile(vals, [25.0, 75.0])
    if float(q75 - q25) > MATERIAL_CONSTANCY_TOL:
        return None, None, None, None
    material_lum = float(np.median(vals))
    material_tex = None
    try:
        from perception.rich import _compute_regularity
        material_tex = float(_compute_regularity(
            patch.astype(np.float32)[None, None, :, :]))
    except Exception:
        material_tex = None
    material_sat = None
    material_hue = None
    if sat_plane is not None and hue_plane is not None:
        sat_patch = np.asarray(sat_plane, dtype=np.float64)[y0:y1, x0:x1][mask]
        hue_patch = np.asarray(hue_plane, dtype=np.float64)[y0:y1, x0:x1][mask]
        if sat_patch.size and np.all(np.isfinite(sat_patch)):
            material_sat = float(np.median(sat_patch))
            chromatic = sat_patch >= float(P.COLOR_ACHROMATIC_EPS)
            if bool(np.any(chromatic)):
                ang = hue_patch[chromatic]
                material_hue = float(
                    math.atan2(float(np.sin(ang).mean()),
                               float(np.cos(ang).mean())) % (2.0 * math.pi))
    return material_lum, material_tex, material_sat, material_hue


def _disk_median(lum: np.ndarray, cx: float, cy: float,
                 radius_px: float = 2.0) -> Optional[float]:
    """Median luminance in a small disk, or None outside the image.

    A median (not a mean) is used deliberately: albedo texture is zero-mean
    over a few pixels and does not move a median much, whereas it would move a
    mean.  This is why the curvature precondition below is texture-robust.
    """
    H, W = lum.shape
    rr = int(math.ceil(radius_px))
    xi, yi = int(round(cx)), int(round(cy))
    x0, x1 = max(0, xi - rr), min(W, xi + rr + 1)
    y0, y1 = max(0, yi - rr), min(H, yi + rr + 1)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    ys, xs = np.mgrid[y0:y1, x0:x1]
    dx = xs + 0.5 - cx
    dy = ys + 0.5 - cy
    ok = (dx * dx + dy * dy) <= radius_px * radius_px
    if int(ok.sum()) < 4:
        return None
    return float(np.median(lum[y0:y1, x0:x1][ok]))


def inward_luminance_step(lum: np.ndarray,
                          pos: Tuple[float, float],
                          owner: Tuple[float, float],
                          radii: Sequence[float] = (2.0, 4.0, 6.0, 8.0),
                          disk_radius_px: float = 2.0) -> Optional[float]:
    """Smallest monotone luminance change between successive inward disks.

    Moving inward from the owner's boundary on a SMOOTH occluding contour, the
    surface turns away from the line of sight, so the normal rotates and the
    Lambertian response changes *gradually*: every 2-px step changes the
    luminance in the same direction.  On a flat face the response is constant
    (the steps are zero); at a crease or a rim the whole change happens in one
    step.  So the minimum signed step is positive on a curved surface and zero
    or negative at a flat/creased one.

    This is the measured precondition for the exact contour cue, not a value:
    the cue still claims the exact 90 deg continuum normal.
    """
    ox, oy = owner
    vals: List[float] = []
    for d in radii:
        v = _disk_median(lum, pos[0] + ox * float(d), pos[1] + oy * float(d),
                         disk_radius_px)
        if v is None:
            return None
        vals.append(v)
    if len(vals) < 3:
        return None
    sign = 1.0 if vals[-1] >= vals[0] else -1.0
    return float(min(sign * (vals[i + 1] - vals[i])
                     for i in range(len(vals) - 1)))


# ---------------------------------------------------------------------------
# texture gradient
# ---------------------------------------------------------------------------

#: The texture cue's gradient floor, in intensity per pixel.  The storage
#: quantisation step is 1/255 and a central difference sums two steps, so a
#: patch whose largest gradient eigenvalue is below twice that carries no
#: measurable texture and the cue abstains.
TEXTURE_GRADIENT_FLOOR = 2.0 / 255.0


def _patch_grid(lum: np.ndarray, cx: float, cy: float, radius_px: float
                ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """A disk patch on its OWN GRID: (intensity, mask).  None when too small."""
    H, W = lum.shape
    rr = int(math.ceil(radius_px))
    xi, yi = int(round(cx)), int(round(cy))
    x0, x1 = max(0, xi - rr), min(W, xi + rr + 1)
    y0, y1 = max(0, yi - rr), min(H, yi + rr + 1)
    if x1 - x0 < 5 or y1 - y0 < 5:
        return None
    g = lum[y0:y1, x0:x1].astype(np.float64)
    ys, xs = np.mgrid[y0:y1, x0:x1]
    dx = xs + 0.5 - cx
    dy = ys + 0.5 - cy
    mask = (dx * dx + dy * dy) <= radius_px * radius_px
    if int(mask.sum()) < 16:
        return None
    return g, mask


def texture_geometry(lum: np.ndarray, cx: float, cy: float, *,
                     patch_radius_px: float = 6.0
                     ) -> Optional[Tuple[float, float, float, float]]:
    """(slant_deg, tilt_deg, ratio, energy) of the local texture, or None.

    THE MODEL.  A surface texture whose gradient statistics are isotropic in
    the surface plane projects to an image whose gradient second moment is
    anisotropic by exactly the foreshortening: with ``nz = cos(slant)``, the
    gradient energy along the steepest-descent (tilt) direction is ``1/nz^2``
    times the energy across it.  Hence

        slant = arccos(sqrt(lambda_min / lambda_max))

    and the tilt is the direction of the LARGEST gradient eigenvalue (the
    compressed direction).  The mean gradient is removed first, so a constant
    shading ramp does not masquerade as texture.

    THE ASSUMPTION is stated, not hidden: the texture must be isotropic in
    second-moment terms.  A single grating (stripes) violates it and the cue
    then reports a WRONG slant with high confidence -- which is why the
    corpus measurement, not the model, decides whether the cue may be fused.
    """
    p = _patch_grid(lum, cx, cy, patch_radius_px)
    if p is None:
        return None
    g, mask = p
    gy, gx = np.gradient(g)
    ax = gx[mask]
    ay = gy[mask]
    if ax.size < 16:
        return None
    ax = ax - float(ax.mean())
    ay = ay - float(ay.mean())
    jxx = float((ax * ax).mean())
    jyy = float((ay * ay).mean())
    jxy = float((ax * ay).mean())
    w = np.linalg.eigvalsh(np.array([[jxx, jxy], [jxy, jyy]]))
    lo = float(max(w[0], 0.0))
    hi = float(max(w[1], 0.0))
    if hi <= TEXTURE_GRADIENT_FLOOR ** 2:
        return None
    ratio = min(1.0, max(0.0, lo / hi))
    slant = math.degrees(math.acos(min(1.0, math.sqrt(ratio))))
    tilt = math.degrees(0.5 * math.atan2(2.0 * jxy, jxx - jyy)) % 360.0
    return slant, tilt, ratio, math.sqrt(hi)


def texture_cue(lum: np.ndarray, cx: float, cy: float,
                sigma_slant_deg: float, sigma_tilt_deg: float, *,
                patch_radius_px: float = 6.0
                ) -> Tuple[Optional[Population], Dict[str, Any]]:
    """The texture-gradient cue as a population on the shared grid.

    The widths are passed in and MUST be measured on the calibration corpus
    (the cue has no free constant of its own): a texture patch that carries
    no measurable gradient abstains rather than emitting a default.
    """
    geom = texture_geometry(lum, cx, cy, patch_radius_px=patch_radius_px)
    if geom is None:
        return None, {"abstain": 1.0}
    slant, tilt, ratio, energy = geom
    diag: Dict[str, Any] = {
        "slant": float(slant),
        "tilt": float(tilt),
        "ratio": float(ratio),
        "energy": float(energy),
        "sigma_slant": float(sigma_slant_deg),
        "sigma_tilt": float(sigma_tilt_deg),
    }
    pop = gaussian_population(slant, tilt, sigma_slant_deg, sigma_tilt_deg)
    return pop, diag


def axis_population(slant_deg: float, tilt_deg: float,
                    sigma_slant_deg: float, sigma_tilt_deg: float
                    ) -> Population:
    """A Gaussian on (slant, tilt) whose TILT is an AXIS, not a direction.

    A gradient second moment cannot tell a tilt from its opposite, so a cue
    built on one has a two-lobe likelihood 180 deg apart.  Collapsing that to a
    one-lobe population would silently claim a direction the measurement does
    not contain.
    """
    ss = max(1e-3, float(sigma_slant_deg))
    st = max(1e-3, float(sigma_tilt_deg))
    sl = np.asarray(SLANT_CENTERS_DEG, dtype=np.float64)[:, None]
    ti = np.asarray(TILT_CENTERS_DEG, dtype=np.float64)[None, :]
    ds = _slant_kernel(sl, float(slant_deg), ss)
    d0 = _wrap_deg(ti - float(tilt_deg)) / st
    d1 = _wrap_deg(ti - float(tilt_deg) - 180.0) / st
    w = ds * (np.exp(-0.5 * d0 * d0) + np.exp(-0.5 * d1 * d1))
    return Population(w).normalized()


def slant_marginal_population(slant_deg: float,
                              sigma_slant_deg: float) -> Population:
    """A population that knows the slant and NOTHING about the tilt."""
    ss = max(1e-3, float(sigma_slant_deg))
    sl = np.asarray(SLANT_CENTERS_DEG, dtype=np.float64)[:, None]
    ds = _slant_kernel(sl, float(slant_deg), ss)
    w = np.repeat(ds, len(TILT_CENTERS_DEG), axis=1)
    return Population(w).normalized()


_TEXTURE_CAL_CACHE: Optional[Dict[str, Any]] = None

#: The fallback calibration, used only when the fitted file is missing.  It
#: says "no monotone correction, wide everywhere": an uncalibrated texture cue
#: is still evidence, but weak evidence, and it must not pretend otherwise.
_TEXTURE_CAL_FALLBACK: Dict[str, Any] = dict(
    anisotropy_gate=0.20,
    anisotropy_max=0.60,
    slant_map=dict(measured=[0.0, 90.0], true=[0.0, 90.0]),
    slant_width=dict(measured=[45.0], sigma=[15.0]),
    tilt_width=dict(anisotropy=[0.20], sigma=[20.0]),
)


def texture_calibration() -> Dict[str, Any]:
    """The texture cue's fitted calibration (cached).

    ``parameters/_texture_calibration.json`` is written by
    diagnostics/_sessionG_e0_texture.py on CAL poses only.  Precedence is the
    file, then ``P.TEXTURE_CALIBRATION``, then the wide fallback above.
    """
    global _TEXTURE_CAL_CACHE
    if _TEXTURE_CAL_CACHE is not None:
        return _TEXTURE_CAL_CACHE
    got: Optional[Dict[str, Any]] = None
    try:
        import json
        import os
        path = os.path.join(os.path.dirname(getattr(P, "__file__", "")),
                            "_texture_calibration.json")
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, dict) and "slant_map" in data:
                got = data
    except Exception:
        got = None
    if got is None:
        got = getattr(P, "TEXTURE_CALIBRATION", None)
    _TEXTURE_CAL_CACHE = dict(got) if isinstance(got, dict) \
        else dict(_TEXTURE_CAL_FALLBACK)
    return _TEXTURE_CAL_CACHE


def texture_cue_calibrated(lum: np.ndarray, cx: float, cy: float, *,
                           patch_radius_px: float = 14.0
                           ) -> Tuple[Optional[Population], Dict[str, Any]]:
    """The texture-gradient cue, recalibrated and with its domain enforced.

    Three corrections, each a measured defect of the raw cue (Session G):

    1. SLANT is passed through the fitted monotone map measured -> true.  The
       map is FLAT below a measured ~33 deg, which is not a bug: in that
       regime the cue is uninformative and the raw value reads high by 25 deg.
    2. WIDTH is the fitted block width, so the population is WIDE where the map
       is flat and narrow where it has slope.  A single global sigma would
       make the cue confident precisely where it knows least.
    3. TILT is (a) an AXIS (two lobes 180 deg apart) and (b) ABSTAINED when the
       gradient second moment is near-isotropic (anisotropy below the gate):
       an isotropic patch has no tilt, so reporting one fabricates it.  The
       abstained cue still contributes its slant.

    ``diag`` carries every number a human needs to read the decision.
    """
    geom = texture_geometry(lum, cx, cy, patch_radius_px=patch_radius_px)
    if geom is None:
        return None, {"abstain": 1.0, "reason": "no measurable texture"}
    slant_raw, tilt_raw, ratio, energy = geom
    cal = texture_calibration()
    gate = float(cal.get("anisotropy_gate", 0.20))
    anis_max = float(cal.get("anisotropy_max") or 0.60)
    anis = (1.0 - float(ratio)) / (1.0 + float(ratio))

    sm = cal["slant_map"]
    slant = float(np.interp(float(slant_raw),
                            np.asarray(sm["measured"], dtype=np.float64),
                            np.asarray(sm["true"], dtype=np.float64),
                            left=float(sm["true"][0]),
                            right=float(sm["true"][-1])))
    sw = cal["slant_width"]
    sigma_slant = float(np.interp(float(slant_raw),
                                  np.asarray(sw["measured"], dtype=np.float64),
                                  np.asarray(sw["sigma"], dtype=np.float64)))

    diag: Dict[str, Any] = {
        "slant_raw": float(slant_raw),
        "slant": float(slant),
        "sigma_slant": float(sigma_slant),
        "ratio": float(ratio),
        "anisotropy": float(anis),
        "anisotropy_gate": float(gate),
        "anisotropy_max": float(anis_max),
        "energy": float(energy),
    }
    if anis > anis_max:
        # The calibration is measured on BROADBAND texture.  Above the largest
        # anisotropy it actually sampled, all the gradient energy sits in one
        # orientation (a grating, a stripe, a single groove) and the eigenvalue
        # ratio describes the PATTERN, not the surface.  Extrapolating here
        # would report a confident slant for a stripe -- the exact "guessing
        # from one cue" failure this module is built to avoid.
        diag["tilt_defined"] = 0.0
        diag["reason"] = ("texture spectrum is rank-1 (anisotropy %.2f > "
                          "%.2f): outside the calibrated regime, abstaining"
                          % (anis, anis_max))
        return None, diag
    if anis < gate:
        diag["tilt_defined"] = 0.0
        diag["reason"] = ("gradient isotropic (anisotropy %.2f < %.2f): "
                          "tilt undefined" % (anis, gate))
        return slant_marginal_population(slant, sigma_slant), diag

    tw = cal["tilt_width"]
    sigma_tilt = float(np.interp(float(anis),
                                 np.asarray(tw["anisotropy"],
                                            dtype=np.float64),
                                 np.asarray(tw["sigma"], dtype=np.float64)))
    diag["tilt"] = float(tilt_raw)
    diag["sigma_tilt"] = float(sigma_tilt)
    diag["tilt_defined"] = 1.0
    diag["reason"] = "texture gradient anisotropic by %.2f" % anis
    return axis_population(slant, tilt_raw, sigma_slant, sigma_tilt), diag


def build_surface_fragments(records: Sequence[Dict[str, Any]],
                            luminance: np.ndarray,
                            enclosure_lookup: Any,
                            light: LightPrior,
                            *,
                            interior: Any = None,
                            sigma_contour: Tuple[float, float],
                            shading_sigma: Any,
                            shading_offset_px: float = 7.0,
                            shading_patch_px: float = 5.0,
                            curvature_window: int = 3,
                            enclosure_gate: float = 0.0,
                            contour_min_step: float = 0.01,
                            contour_variation_px: float = 2.0,
                            rim: bool = True,
                            rim_radius_window: int = RIM_RADIUS_WINDOW,
                            fuse: bool = True,
                            rgb: Any = None) -> List[SurfaceFragment]:
    """Turn ownership records into surface fragments with cue populations.

    ``enclosure_lookup`` is a callable ``(pos, owner_normal) -> float | None``
    (the interior field's enclosure cue), the same observable ownership already
    uses.  ``shading_sigma`` is either a (sigma_slant, sigma_tilt) pair or a
    callable ``fit_diag -> (sigma_slant, sigma_tilt)`` given the shading fit's
    diagnostics (called with None for the provisional width).  That is how the
    calibration applies a measured width from an observable bin without a
    learned weight.
    """
    # [U20 LEAD 4] The colour planes the material payload samples, computed
    # ONCE per call when an RGB image is supplied.  They are the tree's own
    # Retinex opponent planes (perception.filters.compute_opponent_retinex),
    # so hue/saturation here mean exactly what the Colour lens means by them.
    sat_plane: Optional[np.ndarray] = None
    hue_plane: Optional[np.ndarray] = None
    if rgb is not None:
        try:
            from perception.filters import compute_opponent_retinex
            _opp = compute_opponent_retinex(rgb, extraction_scale=5.0)
            sat_plane = np.asarray(_opp["sat_norm"], dtype=np.float64)
            hue_plane = np.asarray(_opp["hue"], dtype=np.float64)
        except Exception:
            sat_plane = None
            hue_plane = None

    by_chain: Dict[str, List[Dict[str, Any]]] = {}
    for r in records:
        # [F3e] A record with no position is not a placed contour point and
        # must not take part in the chain's index set.
        if r.get("pos") is None:
            continue
        by_chain.setdefault(str(r.get("chain")), []).append(r)
    for rows in by_chain.values():
        rows.sort(key=lambda z: int(z.get("vertex", 0)))

    out: List[SurfaceFragment] = []
    for rec in records:
        pos = rec.get("pos")
        if pos is None:
            continue
        raw = rec.get("normal")
        if raw is None or len(raw) < 2:
            continue
        rnx, rny = float(raw[0]), float(raw[1])
        rmag = math.hypot(rnx, rny)
        if rmag < 1e-9:
            continue
        rnx, rny = rnx / rmag, rny / rmag
        on = _owner_normal(rec)
        alpha = rec.get("alpha")
        chain = str(rec.get("chain"))
        vertex = int(rec.get("vertex", 0))
        rows = by_chain.get(chain, [])
        positions = [tuple(map(float, r["pos"])) for r in rows]
        # [F3d] `vertex` indexes the FULL chain while `positions` holds only
        # the representable records.  Index the SUBSET by this record's slot.
        slot = {id(r): k for k, r in enumerate(rows)}
        idx = slot.get(id(rec), 0)
        curv = contour_curvature(positions, idx,
                                 window=curvature_window,
                                 closed=bool(rec.get("closed", False)))
        if on is None:
            # [F6] Existence is not ownership.  The point is placed and its
            # normal is measurable, so the fragment EXISTS -- but no owner-side
            # claim is made and no owner-dependent cue is attached.  The raw
            # unit normal is carried so the order layer keeps its geometry.
            # [F6] An undecided fragment still carries the material it sits on
            # (that measurement never needed an owner side), but NO polarity:
            # an abstention is ABSENT, never 0.
            _m_lum, _m_tex, _m_sat, _m_hue = material_payload(
                luminance, (float(pos[0]), float(pos[1])), (rnx, rny),
                offset_px=float(contour_variation_px),
                radius_px=float(contour_variation_px),
                sat_plane=sat_plane, hue_plane=hue_plane,
            )
            out.append(SurfaceFragment(
                index=len(out), position=(float(pos[0]), float(pos[1])),
                tangent=(rny, -rnx), owner_normal=(rnx, rny),
                owner_side=0.0, curvature=curv, curvature_sign=0.0,
                depth_layer=None, alpha=alpha, chain=chain, vertex=vertex,
                enclosure=None,
                material_lum=_m_lum, material_tex=_m_tex,
                material_sat=_m_sat, material_hue=_m_hue,
                polarity=None, motion_x=None, motion_y=None))
            continue
        ox, oy, side = on
        csign = 0.0
        if curv is not None and abs(curv) > 1e-9:
            csign = 1.0 if curv * side > 0.0 else -1.0
        # Tangent is the right normal of the owner normal.
        tx, ty = oy, -ox
        enc = None
        try:
            enc = float(enclosure_lookup((float(pos[0]), float(pos[1])),
                                         (ox, oy)))
        except Exception:
            enc = None
        enc_lum = None
        if interior is not None:
            try:
                enc_lum = interior_weighted_sample(luminance, interior, float(pos[0]), float(pos[1]), float(contour_variation_px) * 2.0)
            except Exception:
                enc_lum = None
        # Precondition for the exact contour cue: the owner surface must
        # curve.  A locally flat owner is a plane, and a plane cannot have a
        # smooth occluding contour (its boundary is a 3-D edge), so the cue's
        # "slant = 90" claim is invalid there.  The observable is the smallest
        # monotone luminance step between successive inward disks: gradual on
        # a turning surface, zero on a flat face, a single jump at a crease.
        # Measured on the calibration corpus; the ungated numbers are reported
        # alongside.
        variation = inward_luminance_step(
            luminance, (float(pos[0]), float(pos[1])), (ox, oy),
            radii=(2.0, 4.0, 6.0, 8.0),
            disk_radius_px=contour_variation_px)
        # [U20 LEAD 4] the material payload, measured on the owner-side flank
        mat_lum, mat_tex, mat_sat, mat_hue = material_payload(
            luminance, (float(pos[0]), float(pos[1])), (ox, oy),
            offset_px=float(contour_variation_px),
            radius_px=float(contour_variation_px),
            sat_plane=sat_plane, hue_plane=hue_plane,
        )
        raw_signed = rec.get("signed")
        polarity: Optional[float] = None
        if isinstance(raw_signed, (int, float)) and math.isfinite(float(raw_signed)):
            polarity = float(raw_signed)
        frag = SurfaceFragment(
            index=len(out), position=(float(pos[0]), float(pos[1])),
            tangent=(tx, ty), owner_normal=(ox, oy), owner_side=float(side),
            curvature=curv, curvature_sign=float(csign),
            depth_layer=None, alpha=alpha, chain=chain, vertex=vertex,
            enclosure=enc, enclosure_weighted_lum=enc_lum, variation=variation,
            material_lum=mat_lum, material_tex=mat_tex,
            material_sat=mat_sat, material_hue=mat_hue,
            polarity=polarity,
            # motion_x / motion_y stay ABSENT: this builder sees ONE frame and
            # a single frame has no displacement (schema v1 ABSENT = None).
            motion_x=None, motion_y=None,
        )
        # -- exact contour cue, gated by the interior field and by curvature
        curved = (variation is not None
                  and variation >= float(contour_min_step))
        if (enc is not None and enc >= float(enclosure_gate) and curved):
            geom = chain_geometry(positions, idx,
                                  window=rim_radius_window,
                                  closed=bool(rec.get("closed", False)))
            radius_px, radius_sd = _radius_from_geometry(geom)
            pop, diag = contour_cue(rec, curv, sigma_contour[0],
                                    sigma_contour[1],
                                    radius_px=radius_px, rim=rim,
                                    radius_sd_px=radius_sd)
            if pop is not None:
                frag.cues["contour"] = pop
                frag.cue_diag["contour"] = diag
        # -- shading cue
        s_sl, s_ti = (None, None)
        if callable(shading_sigma):
            # width depends on the fit, so fit first with a provisional width
            prov = shading_sigma(None)
            s_sl, s_ti = prov
        else:
            s_sl, s_ti = shading_sigma
        pop, diag = shading_cue(
            luminance, (float(pos[0]), float(pos[1])), (ox, oy), light,
            offset_px=shading_offset_px, patch_radius_px=shading_patch_px,
            sigma_slant_deg=float(s_sl), sigma_tilt_deg=float(s_ti))
        if pop is not None and callable(shading_sigma):
            s_sl, s_ti = shading_sigma(diag)
            pop = gaussian_population(diag["slant"], diag["tilt"],
                                      s_sl, s_ti)
            diag["sigma_slant"] = float(s_sl)
            diag["sigma_tilt"] = float(s_ti)
        if pop is not None:
            frag.cues["shading"] = pop
            frag.cue_diag["shading"] = diag
        # keep the record around for depth linking; not part of the fragment
        out.append(frag)
    if fuse:
        for f in out:
            pops = list(f.cues.values())
            f.combined = fuse_populations(pops)
            f.estimate = population_moment(f.combined)
    return out


# ---------------------------------------------------------------------------
# ordinal depth
# ---------------------------------------------------------------------------

def occlusion_order_edges(fragments: Sequence[SurfaceFragment],
                          records: Sequence[Dict[str, Any]]
                          ) -> List[Tuple[int, int]]:
    """Directed depth edges ``(nearer, farther)`` from the ownership graph.

    The ownership record's ``neighbor`` means: *this* fragment lies on the
    neighbour's NON-owner side.  The neighbour's owner is therefore the figure
    at that contour, and in front of the surface this fragment sits on.  So for
    ``A.neighbor == B`` the edge is ``(B, A)``.

    Only links that pass a geometry check are kept (A really is on B's
    non-owner side).  Session A has no T-junction stem-order cue, so this is
    the only cross-surface depth evidence available; the calibration report
    states its coverage instead of filling the gaps.
    """
    by_alpha: Dict[Any, SurfaceFragment] = {}
    for f in fragments:
        by_alpha[f.alpha] = f
    spacing = _chain_spacing(records)
    edges: List[Tuple[int, int]] = []
    seen = set()
    for rec in records:
        a = by_alpha.get(rec.get("alpha"))
        b = by_alpha.get(rec.get("neighbor"))
        if a is None or b is None or a is b:
            continue
        # [F6] An undecided owner has no side to read: an order claim built on
        # it would be an ownership abstention dressed up as evidence.
        if not (a.owner_is_figure() and b.owner_is_figure()):
            continue
        dx = a.position[0] - b.position[0]
        dy = a.position[1] - b.position[1]
        # [B42] Two ends closer than the contour layer's own sampling step are
        # one local piece of contour; there is no occlusion between them to
        # order.  Strict: a tie at the step abstains (absent is not neutral).
        if spacing is not None and math.hypot(dx, dy) <= float(spacing):
            continue
        if dx * b.owner_normal[0] + dy * b.owner_normal[1] >= 0.0:
            continue
        key = (b.index, a.index)
        if key in seen:
            continue
        seen.add(key)
        edges.append(key)
    return edges


def _chain_spacing(records: Sequence[Dict[str, Any]]) -> Optional[float]:
    """The median distance between consecutive members of the same chain.

    The T-junction tolerance must be measured from the field's own sampling:
    a fixed pixel radius would pass at one resolution and fail at another.
    """
    groups: Dict[Any, List[Dict[str, Any]]] = {}
    for r in records:
        groups.setdefault(r.get("chain"), []).append(r)
    gaps: List[float] = []
    for rs in groups.values():
        rs = sorted(rs, key=lambda r: int(r.get("vertex", 0)))
        for a, b in zip(rs, rs[1:]):
            pa, pb = a.get("pos"), b.get("pos")
            if pa is None or pb is None:
                continue
            ax, ay = pa[:2]
            bx, by = pb[:2]
            gaps.append(math.hypot(float(bx) - float(ax),
                                   float(by) - float(ay)))
    if not gaps:
        return None
    spacing = float(np.median(np.asarray(gaps, dtype=np.float64)))
    return spacing if spacing > 1e-9 else None


def _records_by_chain(records: Sequence[Dict[str, Any]]
                      ) -> Dict[Any, List[Dict[str, Any]]]:
    """Group records by chain, ordered by their index in the FULL chain."""
    groups: Dict[Any, List[Dict[str, Any]]] = {}
    for r in records:
        groups.setdefault(r.get("chain"), []).append(r)
    for rs in groups.values():
        rs.sort(key=lambda z: int(z.get("vertex", 0)))
    return groups


def _chain_count(rows: Sequence[Dict[str, Any]]) -> int:
    """The chain's real member count (`n_chain`), never the record count."""
    n = len(rows)
    for r in rows:
        v = r.get("n_chain")
        if v is not None:
            n = max(n, int(v))
    return n


def _chain_arcs(rows: Sequence[Dict[str, Any]]) -> List[float]:
    """Cumulative distance along the PRESENT records (one entry per record).

    A dropped member's position is unknown; the callers add the chain's own
    median sampling step for it (one step by construction), exactly as
    ``_u12_promotion`` does.  This keeps the bar-side balance measured on the
    chain's real extent (`n_chain`) instead of on the record subset (F3/R9).
    """
    pref: List[float] = [0.0]
    for a, b in zip(rows, rows[1:]):
        pa, pb = a.get("pos"), b.get("pos")
        if pa is None or pb is None:
            pref.append(pref[-1])
            continue
        pref.append(pref[-1] + math.hypot(float(pb[0]) - float(pa[0]),
                                          float(pb[1]) - float(pa[1])))
    return pref


def _chain_dir(rows: Sequence[Dict[str, Any]], k: int
               ) -> Optional[Tuple[float, float]]:
    """AXIAL unit direction of the chain at record index ``k``, or None."""
    if len(rows) < 2:
        return None
    if k <= 0:
        a, b = rows[0], rows[1]
    elif k >= len(rows) - 1:
        a, b = rows[-2], rows[-1]
    else:
        a, b = rows[k - 1], rows[k + 1]
    pa, pb = a.get("pos"), b.get("pos")
    if pa is None or pb is None:
        return None
    dx = float(pb[0]) - float(pa[0])
    dy = float(pb[1]) - float(pa[1])
    n = math.hypot(dx, dy)
    if n <= 1e-9:
        return None
    return dx / n, dy / n


def _end_outward(rows: Sequence[Dict[str, Any]], k: int
                 ) -> Optional[Tuple[float, float]]:
    """DIRECTED outward direction at a chain END (points past the tip)."""
    if len(rows) < 2:
        return None
    if k <= 0:
        tip, inward = rows[0], rows[1]
    else:
        tip, inward = rows[-1], rows[-2]
    pt, pi = tip.get("pos"), inward.get("pos")
    if pt is None or pi is None:
        return None
    dx = float(pt[0]) - float(pi[0])
    dy = float(pt[1]) - float(pi[1])
    n = math.hypot(dx, dy)
    if n <= 1e-9:
        return None
    return dx / n, dy / n


def _axial(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Acute angle between two AXIAL (undirected) directions, radians."""
    dot = abs(float(a[0]) * float(b[0]) + float(a[1]) * float(b[1]))
    return math.acos(max(-1.0, min(1.0, dot)))


def _continuation(groups: Dict[Any, List[Dict[str, Any]]],
                  chain_end: Any, chain_bar: Any,
                  end_rec: Dict[str, Any],
                  outward: Tuple[float, float],
                  spacing: float, band_scale: float,
                  min_align: float) -> Optional[Tuple[float, Any, Any]]:
    """The terminating contour reappearing collinearly beyond the bar.

    F7's direction test: a contour that goes BEHIND an occluder reappears on
    the far side, on the same line, running the same way.  A contour that
    merely stops -- an object's own corner, a limb tangency -- has nothing
    beyond it.  Bound: one sampling step laterally (the chain layer's own
    resolution), positive along-ray distance, axial alignment within
    ``min_align``.  Returns ``(along_distance, chain, record)`` or None.
    """
    ex, ey = float(end_rec["pos"][0]), float(end_rec["pos"][1])
    best: Optional[Tuple[float, Any, Any]] = None
    for chain, rows in groups.items():
        if chain == chain_end or chain == chain_bar:
            continue
        for k, rec in enumerate(rows):
            p = rec.get("pos")
            if p is None:
                continue
            dx = float(p[0]) - ex
            dy = float(p[1]) - ey
            along = dx * outward[0] + dy * outward[1]
            if along <= 0.0:
                continue
            lateral = abs(-dx * outward[1] + dy * outward[0])
            if lateral > float(band_scale) * spacing:
                continue
            d = _chain_dir(rows, k)
            if d is None or _axial(d, outward) > float(min_align):
                continue
            if best is None or along < best[0]:
                best = (float(along), chain, rec)
    return best


def t_junction_sites(records: Sequence[Dict[str, Any]], *,
                     tol_scale: float = 2.06,
                     min_bar_side_scale: float = 1.5,
                     require_interrupted: bool = True,
                     band_scale: float = 1.0,
                     min_turn_rad: Optional[float] = None
                     ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Every occlusion T-junction SITE, with the evidence that made it.

    A site is one chain's TRUE END landing on the INTERIOR of another chain,
    with the terminating contour reappearing collinearly beyond the bar
    (interruption) and the bar carrying real contour on both sides of the
    contact (pass-through).  The site carries no order by itself: it names the
    end, the bar and the continuation, and ``t_junction_order`` turns it into
    ``(bar, stem)``.

    The contact bound is derived, not fitted: the terminator's last DETECTED
    sample can be two gaps short (one for the sample behind the occluder, one
    for a faded contour losing the sample nearest the junction) plus the bar's
    half-gap of quantisation = sqrt(2^2 + 0.5^2) = 2.06 spacings.  The
    pass-through bound is 1.5 spacings (one gap of real contour plus the same
    half-gap).  Both are strict: a tie at a derived bound abstains.

    Rejections are named in ``diag`` so every chain end is accounted for.
    The nearest type-passing contact is the one tested; a rejected nearest is
    never replaced by a farther one (R11: the guard is subtractive over
    SITES).
    """
    groups = _records_by_chain(records)
    spacing = _chain_spacing(records)
    if min_turn_rad is None:
        try:
            from perception.chain_formation import ALIGNMENT_TOL
            min_turn_rad = float(ALIGNMENT_TOL)
        except Exception:                       # pragma: no cover - fallback
            min_turn_rad = math.radians(35.0)
    diag: Dict[str, Any] = dict(
        ends=0, sites=0, rejected_turn=0, rejected_no_contact=0,
        rejected_not_interrupted=0, rejected_short_bar=0,
        min_bar_side=0.0, min_bar_side_scale=float(min_bar_side_scale),
        spacing=None, tol=None)
    if spacing is None:
        return [], diag
    spacing = float(spacing)
    tol = float(tol_scale) * spacing
    bound = float(min_bar_side_scale) * spacing
    diag.update(spacing=spacing, tol=tol, min_bar_side=bound)

    arcs = {ch: _chain_arcs(rs) for ch, rs in groups.items()}
    sites: List[Dict[str, Any]] = []
    for chain_a, rows_a in groups.items():
        if not rows_a:
            continue
        if any(bool(r.get("closed", False)) for r in rows_a):
            continue
        n_a = _chain_count(rows_a)
        if n_a < 2:
            continue
        for want in (0, n_a - 1):
            k_a = next((k for k, r in enumerate(rows_a)
                        if int(r.get("vertex", 0)) == want
                        and r.get("pos") is not None), None)
            if k_a is None:
                continue                    # that end is not representable
            ra = rows_a[k_a]
            diag["ends"] += 1
            outward = _end_outward(rows_a, k_a)
            dir_a = _chain_dir(rows_a, k_a)
            if outward is None or dir_a is None:
                diag["rejected_no_contact"] += 1
                continue
            ax, ay = float(ra["pos"][0]), float(ra["pos"][1])
            best = None
            for chain_b, rows_b in groups.items():
                if chain_b == chain_a or not rows_b:
                    continue
                n_b = _chain_count(rows_b)
                for k_b, rb in enumerate(rows_b):
                    if rb.get("pos") is None:
                        continue
                    v_b = int(rb.get("vertex", 0))
                    if v_b <= 0 or v_b >= n_b - 1:
                        continue        # the bar must be INTERIOR here
                    bx, by = float(rb["pos"][0]), float(rb["pos"][1])
                    d = math.hypot(bx - ax, by - ay)
                    if d <= tol and (best is None or d < best[0]):
                        best = (d, chain_b, k_b, rb)
            if best is None:
                diag["rejected_no_contact"] += 1
                continue
            d, chain_b, k_b, rb = best
            dir_b = _chain_dir(groups[chain_b], k_b)
            if dir_b is None or _axial(dir_a, dir_b) <= float(min_turn_rad):
                diag["rejected_turn"] += 1
                continue
            cont = None
            if require_interrupted:
                cont = _continuation(groups, chain_a, chain_b, ra, outward,
                                     spacing, float(band_scale),
                                     float(min_turn_rad))
                if cont is None:
                    diag["rejected_not_interrupted"] += 1
                    continue
            rows_b = groups[chain_b]
            pref = arcs[chain_b]
            n_b = _chain_count(rows_b)
            miss_head = int(rows_b[0].get("vertex", 0))
            miss_tail = n_b - 1 - int(rows_b[-1].get("vertex", 0))
            side_in = pref[k_b] + miss_head * spacing
            side_out = (pref[-1] - pref[k_b]) + miss_tail * spacing
            bar_min = min(side_in, side_out)
            if bar_min <= bound:
                diag["rejected_short_bar"] += 1
                continue
            sites.append(dict(
                end=ra, end_pos=(ax, ay), chain_end=chain_a,
                contact=rb, chain_bar=chain_b, distance=float(d),
                continuation=cont,
                bar_side_in=float(side_in), bar_side_out=float(side_out),
                bar_min_side=float(bar_min),
                bar_min_side_scale=float(min_bar_side_scale),
                turn_rad=float(_axial(dir_a, dir_b))))
    diag["sites"] = len(sites)
    return sites, diag


def t_junction_order(fragments: Sequence[SurfaceFragment],
                     records: Sequence[Dict[str, Any]], *,
                     tol_scale: float = 2.06,
                     min_bar_side_scale: float = 1.5,
                     require_interrupted: bool = True,
                     return_diag: bool = False):
    """Directed depth edges ``(nearer, farther)`` from T-junction SITES.

    A T-junction is an ENDPOINT of one chain landing on the INTERIOR of
    another, with the terminating contour interrupted by it.  The terminating
    chain is the stem, and a stem terminates because the surface it lies on
    passes BEHIND the bar -- so the bar's surface is nearer and the edge is
    ``(bar, stem)``.

    The decision lives in ``t_junction_sites``; this function only maps the
    site's records to fragment indices.  A site whose end or contact has no
    fragment cannot make an order claim (its geometry is still reported by
    ``t_junction_sites``); the rest of the site list is unaffected.
    """
    sites, diag = t_junction_sites(
        records, tol_scale=tol_scale,
        min_bar_side_scale=min_bar_side_scale,
        require_interrupted=require_interrupted)
    by_alpha: Dict[Any, SurfaceFragment] = {}
    for f in fragments:
        by_alpha[f.alpha] = f
    edges: List[Tuple[int, int]] = []
    seen = set()
    for s in sites:
        fa = by_alpha.get(s["end"].get("alpha"))
        fb = by_alpha.get(s["contact"].get("alpha"))
        if fa is None or fb is None or fa is fb:
            continue
        key = (int(fb.index), int(fa.index))      # bar nearer, stem farther
        if key in seen:
            continue
        seen.add(key)
        edges.append(key)
    diag["edges"] = len(edges)
    if return_diag:
        return edges, diag
    return edges


def chain_arms(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """One ARM per open-chain END: tip position and outward direction.

    An arm is the local continuation of a chain at the point where the chain
    stops.  ``outward`` points away from the chain body, so two arms that
    belong to a contour PASSING a junction point in opposite senses are
    antiparallel, and an arm that merely ends there is not.  The direction is
    taken over a short baseline (``arm_baseline`` members) so a single noisy
    member cannot define it.

    Closed chains have no arms: a loop cannot terminate at a T.
    """
    groups: Dict[Any, List[Dict[str, Any]]] = {}
    for r in records:
        groups.setdefault(r.get("chain"), []).append(r)
    arms: List[Dict[str, Any]] = []
    for chain, rs in groups.items():
        if any(bool(r.get("closed", False)) for r in rs):
            continue
        rs = sorted(rs, key=lambda r: int(r.get("vertex", 0)))
        if len(rs) < 3:
            continue
        for at_start in (True, False):
            tip = rs[0] if at_start else rs[-1]
            base = rs[min(3, len(rs) - 1)] if at_start else rs[-min(3, len(rs) - 1) - 1]
            tx, ty = (float(v) for v in tip.get("pos", (0.0, 0.0))[:2])
            bx, by = (float(v) for v in base.get("pos", (0.0, 0.0))[:2])
            dx, dy = tx - bx, ty - by
            n = math.hypot(dx, dy)
            if n <= 1e-9:
                continue
            arms.append(dict(chain=chain, vertex=int(tip.get("vertex", 0)),
                             pos=(tx, ty), outward=(dx / n, dy / n),
                             alpha=tip.get("alpha"), at_start=at_start))
    return arms


def _cluster_arms(arms: Sequence[Dict[str, Any]], tol: float
                  ) -> List[List[Dict[str, Any]]]:
    """Single-linkage clusters of arms whose TIPS touch (within ``tol``)."""
    parent = list(range(len(arms)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(arms)):
        for j in range(i + 1, len(arms)):
            if arms[i]["chain"] == arms[j]["chain"]:
                continue
            ax, ay = arms[i]["pos"]
            bx, by = arms[j]["pos"]
            if math.hypot(ax - bx, ay - by) <= tol:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[rj] = ri
    clusters: Dict[int, List[Dict[str, Any]]] = {}
    for i in range(len(arms)):
        clusters.setdefault(find(i), []).append(arms[i])
    return [c for c in clusters.values() if len(c) >= 2]


def t_junction_arms(fragments: Sequence[SurfaceFragment],
                    records: Sequence[Dict[str, Any]], *,
                    tol_scale: float = 1.5,
                    max_through_dev_deg: float = 40.0,
                    same_figure_gate: bool = True
                    ) -> Tuple[List[Tuple[int, int]], Dict[str, Any]]:
    """T-junction stem order from the ARMS at the junction, not from proximity.

    ``t_junction_order`` decides sideways: it asks which chain's END is near
    another chain's INTERIOR.  That test cannot tell a bar that passes the
    junction from a bar that merely ends there, and it has no way to say which
    of two terminating chains is the occluder.

    The arm reading is the classic one and it uses a signal proximity throws
    away -- DIRECTION.  At a T three arms meet: the bar arrives and leaves in
    nearly OPPOSITE directions (a through-pair), and the stem arrives roughly
    perpendicular.  So:

    1. cluster arm tips that touch (single-linkage, tolerance = measured
       chain spacing);
    2. inside a cluster, the through-pair is the pair of arms most nearly
       antiparallel (``|180 - angle|`` least, and within
       ``max_through_dev_deg``); a cluster with no such pair is a corner, not
       a T, and emits nothing;
    3. every remaining arm is a stem, and its chain lies BEHIND the through
       pair's chain(s): the stem stops because it passes behind the bar.

    ``[L2] SAME-FIGURE THROUGH-PAIR (derived, not fitted).`` A through-pair is
    two arms of ONE contour passing the junction.  The pair search reads
    DIRECTION only, so at a place where two different contours cross it can
    pick one arm from each (most antiparallel by accident).  On `f_disk_a` it
    picked a disk arm and a box arm; the box arm was then treated as a stem and
    the cue emitted ``(box, disk)`` -- box nearer than the disk that occludes
    it, a fabricated wrong-direction edge.  The front end's own same-figure
    signal at a record is the ownership read ``signed``: along one contour it
    keeps its sign, and two DECIDED records with OPPOSITE signs are two
    different figures.  So when the most-antiparallel pair has two DECIDED
    ownership signs of opposite sign, the cluster is two contours crossing, not
    a T, and the site abstains.  The gate is strictly SUBTRACTIVE (it removes a
    site; it never picks a different pair and never emits or directs an edge),
    so every edge it keeps is an edge the ungated cue also emitted.

    Returns ``(edges, diag)`` where each edge is ``(nearer, farther)`` as
    fragment indices, and ``diag`` counts sites, through-pairs, corners
    rejected, cross-figure sites vetoed and stems emitted -- the numbers a
    reader needs to see why a junction was or was not used.
    """
    arms = chain_arms(records)
    spacing = _chain_spacing(records)
    if spacing is None or not arms:
        return [], dict(arms=len(arms), sites=0, through_pairs=0,
                        corners=0, stems=0)
    # alpha -> fragment index.  The alpha is the identity-carrying key shared
    # by the two views; the fragment index is what a depth layer edge uses.  A
    # T-junction STEM is often absent from ``fragments``: its ownership is
    # ambiguous (|signed| below the ambiguity floor) precisely because a stem
    # stops at the bar and owns nothing there.  So the ordering claim is
    # carried by ALPHA pairs, and the index pairs are reported for the subset
    # that is representable -- dropping the rest would hide the finding.
    idx = {f.alpha: int(f.index) for f in fragments}
    tol = float(tol_scale) * float(spacing)

    edges: List[Tuple[int, int]] = []
    seen = set()
    sites = through = corners = stems_n = cross_figure = 0
    signed_of = {r.get("alpha"): float(r.get("signed", 0.0) or 0.0)
                 for r in records}
    for arms_c in _cluster_arms(arms, tol):
        chains = {a["chain"] for a in arms_c}
        if len(chains) < 2:
            continue
        sites += 1
        best = None
        for i in range(len(arms_c)):
            for j in range(i + 1, len(arms_c)):
                a, b = arms_c[i], arms_c[j]
                if a["chain"] == b["chain"]:
                    continue
                dot = (a["outward"][0] * b["outward"][0]
                       + a["outward"][1] * b["outward"][1])
                dev = abs(180.0 - math.degrees(math.acos(max(-1.0, min(1.0, dot)))))
                if best is None or dev < best[0]:
                    best = (dev, a, b)
        if best is None or best[0] > max_through_dev_deg:
            corners += 1
            continue
        dev, bar_a, bar_b = best
        if same_figure_gate:
            sa = signed_of.get(bar_a.get("alpha"))
            sb = signed_of.get(bar_b.get("alpha"))
            if (sa is not None and sb is not None
                    and abs(sa) >= _AMBIGUOUS_OWNERSHIP
                    and abs(sb) >= _AMBIGUOUS_OWNERSHIP
                    and (sa > 0.0) != (sb > 0.0)):
                # [L2] cross-figure "through-pair": the two arms are two
                # different figures.  Subtract the whole site.
                cross_figure += 1
                corners += 1
                continue
        through += 1
        bar_chains = {bar_a["chain"], bar_b["chain"]}
        bar_tips = [bar_a, bar_b]
        for a in arms_c:
            if a["chain"] in bar_chains:
                continue
            stems_n += 1
            # The stem's chain goes behind EACH bar chain it touches.
            for bt in bar_tips:
                want = (bt["chain"], a["chain"], round(float(dev), 3))
                if want in seen:
                    continue
                seen.add(want)
                edges.append((bt.get("alpha"), a.get("alpha")))
    resolved = [(int(idx[na]), int(idx[fa])) for na, fa in edges
                if na in idx and fa in idx]
    diag = dict(arms=len(arms), sites=sites, through_pairs=through,
                corners=corners, cross_figure=cross_figure, stems=stems_n,
                tol=round(float(tol), 3),
                edges=len(edges), index_edges=len(resolved),
                stems_dropped=len(edges) - len(resolved))
    diag["resolved"] = resolved
    return edges, diag


def assign_depth_layers(fragments: Sequence[SurfaceFragment],
                        records: Sequence[Dict[str, Any]],
                        extra_edges: Sequence[Tuple[int, int]] = ()
                        ) -> Dict[int, int]:
    """Ordinal layers from the occlusion-order edges, acyclic components only.

    Edges are ``nearer -> farther``; layers are the longest-path depth inside
    each weakly connected component.  A cycle (e.g. two fragments that each
    call the other a neighbour) makes that component's order untrustworthy, so
    the whole component abstains rather than inventing an order.

    ``extra_edges`` carries other measured ORDER evidence on the same terms
    (``t_junction_order`` is the T-junction stem cue).  It is merged with the
    ownership edges and subject to the same acyclicity rule: a contradiction
    abstains instead of being silently out-voted.
    """
    edges = list(occlusion_order_edges(fragments, records))
    seen_edges = set(edges)
    for e in extra_edges:
        key = (int(e[0]), int(e[1]))
        if key not in seen_edges:
            seen_edges.add(key)
            edges.append(key)
    if not edges:
        for f in fragments:
            f.depth_layer = None
        return {}
    nodes = sorted({i for e in edges for i in e})
    adj: Dict[int, List[int]] = {i: [] for i in nodes}
    indeg: Dict[int, int] = {i: 0 for i in nodes}
    for a, b in edges:
        adj[a].append(b)
        indeg[b] += 1

    # Weak components (ignore direction).
    und: Dict[int, set] = {i: set() for i in nodes}
    for a, b in edges:
        und[a].add(b)
        und[b].add(a)
    comps: List[set] = []
    seen: set = set()
    for s in nodes:
        if s in seen:
            continue
        stack = [s]
        comp = set()
        while stack:
            u = stack.pop()
            if u in comp:
                continue
            comp.add(u)
            stack.extend(und[u] - comp)
        comps.append(comp)
        seen |= comp

    layers: Dict[int, int] = {}
    for comp in comps:
        local_indeg = {i: indeg[i] for i in comp}
        queue = [i for i in comp if local_indeg[i] == 0]
        layer = {i: 0 for i in comp}
        done = 0
        while queue:
            u = queue.pop()
            done += 1
            for v in adj[u]:
                if v not in comp:
                    continue
                layer[v] = max(layer[v], layer[u] + 1)
                local_indeg[v] -= 1
                if local_indeg[v] == 0:
                    queue.append(v)
        if done != len(comp):
            continue  # cycle: abstain for the whole component
        base = min(layer.values())
        for i in comp:
            layers[i] = layer[i] - base
    for f in fragments:
        f.depth_layer = layers.get(f.index, None)
    return dict(layers)


def estimate_surface(fragment: SurfaceFragment) -> Dict[str, Any]:
    """Convenience view of one fragment's estimate."""
    out: Dict[str, Any] = {
        "position": fragment.position,
        "owner_side": fragment.owner_side,
        "curvature": fragment.curvature,
        "curvature_sign": fragment.curvature_sign,
        "depth_layer": fragment.depth_layer,
        "cues": sorted(fragment.cues),
    }
    for name, pop in fragment.cues.items():
        m = population_moment(pop)
        if m is not None:
            out[name] = {"slant": m[0], "tilt": m[1], "confidence": m[2]}
    if fragment.combined is not None:
        m = population_moment(fragment.combined)
        if m is not None:
            out["combined"] = {"slant": m[0], "tilt": m[1],
                               "confidence": m[2]}
    return out
