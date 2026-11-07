"""perception/contour_relations.py -- what lies on each side of a contour.

WHY THIS MODULE EXISTS
----------------------
Session F established that the contour proposer cannot be fixed by geometry:
the foreground silhouette is ~36 of ~127 peaks per scene, and radial+tangent
geometry is a clustering detector.  The missing information is not the contour
-- it is the RELATION between a contour fragment and the two surfaces it
separates.  This module measures that relation, per fragment, as graded
populations over continuous quantities, and nothing else:

    cue            quantity                                   votes?
    depth_step     signed ordinal depth step  s in [-1, +1]   yes
    slant_diff     slant(right) - slant(left), deg            yes
    tilt_diff      axis difference of the two tilts, deg      yes
    appearance     relative luminance step |dI| / contrast    yes
    phase          quadrature phase, deg (0/180 edge, 90 line) yes
    kinetic        signed ordinal motion step, same axis       yes (opt-in)
    owner_slant    slant on the owner side, deg               no

``kinetic`` is the E4 slot: accretion/deletion is the SAME ordinal step
question measured by a different physical signal, so it lands on the
``depth_step`` axis instead of opening a second vocabulary.  No motion is
computed in this tree; with no measurement the cue abstains and is excluded
from the vote.

No category label is stored anywhere: a "crease", an "occluding edge" and a
"marking" are REGIONS of these continuous quantities, defined once below and
integrated when a consumer needs a decision.  The graph therefore carries
measurements, not verdicts, and a later reader can disagree about the region
boundaries without re-reading the image.

THE COMBINING RULE is the mixture (sum of the voters' graded evidence), not the
product: cues that DISAGREE go to zero instead of one confident cue vetoing the
rest, and disagreement is reported as its own diagnostic.  This is the same
rule the surface-fragment module uses for its populations, and it is the reason
a shading-only reading can never silently win.

WHAT IS *NOT* HERE: no dense depth map, no object or part class, no learned
category head.  The depth step is ORDINAL and graded -- the calibrated
confidence that one side is nearer -- because the system has no metric depth to
offer at a flank.

Session G.  E2.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from perception.surface_fragments import (LightPrior, gaussian_population,
                                          population_moment, shade_umbilic,
                                          texture_cue_calibrated)

__all__ = [
    "RELATION_CUES",
    "VOTING_CUES",
    "Relation",
    "relations_for_fragment",
    "combine_relations",
    "explain",
    "OCCLUSION_STEP",
    "CREASE_ANGLE_DEG",
    "APPEARANCE_STEP",
    "PHASE_EDGE_TOL_DEG",
    "FLANK_OFFSET_SCALE",
    "kinetic_relation",
    "boundary_gate",
]

RELATION_CUES: Tuple[str, ...] = ("depth_step", "slant_diff", "tilt_diff",
                                  "appearance", "phase", "owner_slant",
                                  "kinetic")
VOTING_CUES: Tuple[str, ...] = ("depth_step", "slant_diff", "tilt_diff",
                                "appearance", "phase", "kinetic")

# -- the region definitions, in the quantity's own units ---------------------
#
# These are DEFINITIONS of the words, not tuned thresholds: a step of a tenth
# of the ordinal axis is a step; two surface normals that differ by less than
# the measured cue width are the same normal; a luminance step below the
# storage quantisation is invisible.  Each is either derived from a measured
# width or is the resolution of the axis, and the probe reports the sensitivity
# of the separation to each one.

#: |s| above this is a depth discontinuity.  The ordinal axis runs -1..+1, so
#: 0.5 is "more evidence for a side than for equality" -- the natural midpoint
#: of the three-lobe reading, not a fitted value.
OCCLUSION_STEP = 0.5
#: Two tilts are the same axis when their difference is below the tilt cue's
#: own measured width (Session A/B: 10.9 deg); 10 deg is that width.
TILT_SAME_DEG = 10.0
#: Two slants are the same surface when they differ by less than the shading
#: cue's measured slant width (2.3 deg); 3 deg covers it with the quantisation.
SLANT_SAME_DEG = 3.0
#: ... but a CREASE is a large normal difference, and a marking has none.  The
#: crease region needs a normal difference the ownership of neither side can
#: explain away: 20 deg, i.e. two cue widths of tilt.
CREASE_ANGLE_DEG = 20.0
#: A luminance step is visible above the storage resolution.  The 8-bit step is
#: 1/255 and two patch means average ~100 pixels each, so their standard error
#: is step/sqrt(12*100) ~ 3e-4; 0.01 is ~30 standard errors.
APPEARANCE_STEP = 0.01
#: The quadrature phase of a step edge is 0 or 180; of a line, +-90.  The
#: tolerance is how far from a pure odd response a profile may sit and still
#: count as an edge.  The relation's POPULATION width is a third of it: the
#: region must be 3 sigma wide, or a correct reading would be clipped by its
#: own tolerance and the cue could never be confident.
PHASE_EDGE_TOL_DEG = 30.0

#: How many standard errors an appearance step must clear before the fragment
#: counts as a boundary at all.  This is the DECLARED significance level of
#: the arming test (a 3-sigma detection), not a fitted constant; the arming
#: also requires the step to exceed ``APPEARANCE_STEP``, the axis resolution,
#: because a statistically certain 0.002 step is not a visible boundary.
APPEARANCE_ARM_SIGMA = 3.0

#: Flank offset as a fraction of the fragment's own measured member spacing.
#: PHYSICAL DERIVATION, not a fitted value: the front-end Gabor envelope along
#: the filter's normal has sigma = GABOR_SIGMA_X_FRAC * wavelength (0.56 at the
#: reference scale), and the measured member spacing IS that wavelength.  A
#: flank sampled nearer than half the spacing therefore sits inside the
#: contour's own quadrature footprint and reads the edge, not the surface.  At
#: a full spacing the flank can instead land on the NEXT contour.  Half a
#: spacing is the first position outside the footprint and inside the gap;
#: measured on the synthetic sphere/cylinder by
#: ``diagnostics/_sessionH_flank_geometry.py`` (0.50-0.58 spacing, see the
#: Session H notebook entry).
FLANK_OFFSET_SCALE = 0.5

#: The default width of the appearance cue, in relative luminance.  It is
#: replaced by the fitted value when the calibration file is present.
_APPEARANCE_WIDTH = 0.02

#: The width of one lobe on the ORDINAL depth-step axis.  The axis is divided
#: into three equal cells (left nearer / equal / right nearer), so a lobe's
#: half-width is 1/3, and a Gaussian with sigma = 1/6 has 99.9% of its mass
#: inside its cell.  Derived from the axis, not fitted.
_ORDINAL_WIDTH = 1.0 / 6.0


@dataclass
class Relation:
    """One measured relation, as a graded population over a continuous axis.

    ``mass`` is normalised over ``grid``.  ``evidence`` is a signed vote in
    [-1, +1] for "this contour is a depth discontinuity", obtained by
    integrating ``mass`` over the declared region -- never by a lookup table of
    labels.  ``votes`` is False for relations that carry geometry but are not
    depth evidence (``owner_slant``).
    """

    cue: str
    quantity: str
    grid: np.ndarray
    mass: np.ndarray
    width: float
    evidence: float = 0.0
    votes: bool = True
    abstained: bool = False
    detail: Dict[str, Any] = field(default_factory=dict)

    def moment(self) -> Optional[Tuple[float, float]]:
        m = float(self.mass.sum())
        if m <= 0.0:
            return None
        x = float((self.grid * self.mass).sum() / m)
        var = float(((self.grid - x) ** 2 * self.mass).sum() / m)
        return x, math.sqrt(max(0.0, var))

    def region_mass(self, mask: np.ndarray) -> float:
        m = float(self.mass.sum())
        if m <= 0.0:
            return 0.0
        return float(self.mass[np.asarray(mask, dtype=bool)].sum() / m)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _wrap180(a: np.ndarray) -> np.ndarray:
    return (a + 180.0) % 360.0 - 180.0


def _axis_diff_deg(a: float, b: float) -> float:
    """Difference between two AXES (tilt is an axis): result in [0, 90]."""
    d = abs(_wrap180(a - b))
    return float(min(d, 180.0 - d))


def _bilinear(lum: np.ndarray, x: float, y: float) -> Optional[float]:
    H, W = lum.shape
    if x < 0.0 or y < 0.0 or x > W - 1.0 or y > H - 1.0:
        return None
    x0, y0 = int(math.floor(x)), int(math.floor(y))
    x1, y1 = min(x0 + 1, W - 1), min(y0 + 1, H - 1)
    fx, fy = x - x0, y - y0
    v = (lum[y0, x0] * (1 - fx) * (1 - fy) + lum[y0, x1] * fx * (1 - fy)
         + lum[y1, x0] * (1 - fx) * fy + lum[y1, x1] * fx * fy)
    return float(v)


def _patch_stats(lum: np.ndarray, cx: float, cy: float, radius: float
                 ) -> Optional[Dict[str, float]]:
    H, W = lum.shape
    rr = int(math.ceil(radius))
    xi, yi = int(round(cx)), int(round(cy))
    x0, x1 = max(0, xi - rr), min(W, xi + rr + 1)
    y0, y1 = max(0, yi - rr), min(H, yi + rr + 1)
    if x1 - x0 < 3 or y1 - y0 < 3:
        return None
    g = lum[y0:y1, x0:x1].astype(np.float64)
    ys, xs = np.mgrid[y0:y1, x0:x1]
    m = ((xs + 0.5 - cx) ** 2 + (ys + 0.5 - cy) ** 2) <= radius * radius
    if int(m.sum()) < 6:
        return None
    v = g[m]
    return dict(mean=float(v.mean()), sd=float(v.std()),
                n=int(v.size))


def _luminance_phase(lum: np.ndarray, cx: float, cy: float,
                     nx: float, ny: float, sigma: float
                     ) -> Optional[Dict[str, float]]:
    """Quadrature phase of the luminance profile across the contour.

    The profile is sampled along the contour NORMAL at unit-pixel steps and
    projected on an ORTHOGONAL PAIR at one scale, the way an energy model does
    it: the ODD member is the first derivative of a Gaussian and the EVEN
    member its second derivative.  This matters -- projecting on sin/cos at a
    fixed period does not separate a step from a line (measured: an ideal step
    read -31 deg with a LOWER explained fraction than pure noise).

    With this pair:

        step edge   -> strong ODD, ~zero EVEN   -> phase   0 or 180
        line/ridge  -> strong EVEN, ~zero ODD   -> phase +-90

    ``snr2`` is the squared amplitude over the profile's own per-sample noise
    power (the pair is normalised, so unit noise gives unit projection
    variance).  A profile that does not clear 3 sigma has no phase to report,
    and the relation abstains rather than inventing one.
    """
    half = int(max(3, round(2.5 * sigma)))
    ts = np.arange(-half, half + 1, dtype=np.float64)
    s = max(1e-6, float(sigma))
    g = np.exp(-0.5 * (ts / s) ** 2)
    odd_k = -(ts / (s * s)) * g
    even_k = ((ts * ts) / (s ** 4) - 1.0 / (s * s)) * g
    odd_k = odd_k / max(1e-12, float(np.linalg.norm(odd_k)))
    even_k = even_k / max(1e-12, float(np.linalg.norm(even_k)))
    vals = []
    for t in ts:
        v = _bilinear(lum, cx + nx * t, cy + ny * t)
        if v is None:
            return None
        vals.append(v)
    v = np.asarray(vals, dtype=np.float64)
    v = v - float(v.mean())
    odd = float(np.dot(odd_k, v))
    even = float(np.dot(even_k, v))
    amp2 = odd * odd + even * even
    phase = math.degrees(math.atan2(even, odd)) if amp2 > 0.0 else 0.0
    # Per-sample noise from the profile's own successive differences: for a
    # smooth signal the difference is dominated by noise, and var(d) = 2 s^2.
    d = np.diff(v)
    noise = float(np.sqrt(max(0.0, float(np.var(d))) / 2.0))
    noise = max(noise, 1.0 / 255.0 / math.sqrt(12.0))
    snr2 = float(amp2 / (noise * noise))
    return dict(phase=float(phase), odd=odd, even=even,
                modulation=float(math.sqrt(amp2)), snr2=snr2,
                noise=noise, r2=snr2)


def _gauss_on(grid: np.ndarray, mu: float, sigma: float
              ) -> np.ndarray:
    s = max(1e-6, float(sigma))
    return np.exp(-0.5 * ((grid - mu) / s) ** 2)


# ---------------------------------------------------------------------------
# the six relations
# ---------------------------------------------------------------------------

def _depth_step_relation(rec: Dict[str, Any], *, flat_floor: float
                         ) -> Relation:
    """Signed ordinal depth step from the fragment's calibrated ownership.

    The system has no metric depth at a flank, so the continuous axis is the
    ORDINAL step: -1 = the left flank is nearer, +1 = the right flank is
    nearer, 0 = the two flanks are at the same depth.  The mass is the
    calibrated owner probability, split by which side the owner is on, with the
    residual mass left at 0.  A fragment whose ownership is ambiguous therefore
    has most of its mass at 0 -- it abstains, rather than claiming a side.
    """
    grid = np.linspace(-1.0, 1.0, 41)
    signed = float(rec.get("signed", 0.0) or 0.0)
    prob = float(rec.get("owner_prob", 0.0) or 0.0)
    prob = min(max(prob, 0.0), 1.0)
    # `normal` is the LEFT normal of travel, so an owner on the left is a
    # nearer LEFT flank and s = -1.  ``owner_prob`` is P(the owner side is
    # correct), so the residual is not "no step": if the ownership inference is
    # wrong the step still exists and points the OTHER way.  Putting the
    # residual at 0 instead would let a confident-but-mirrored reading look
    # like an abstention, and the graded certainty of the step is |2p - 1|.
    w = _ORDINAL_WIDTH
    mass = np.zeros_like(grid)
    if abs(signed) <= 1e-12:
        # No side is claimed at all: no evidence for either lobe.
        mass += 0.5 * prob * _gauss_on(grid, -1.0, w)
        mass += 0.5 * prob * _gauss_on(grid, +1.0, w)
        mass += (1.0 - prob) * _gauss_on(grid, 0.0, w)
    elif signed > 0.0:
        mass += prob * _gauss_on(grid, -1.0, w)
        mass += (1.0 - prob) * _gauss_on(grid, +1.0, w)
    else:
        mass += prob * _gauss_on(grid, +1.0, w)
        mass += (1.0 - prob) * _gauss_on(grid, -1.0, w)
    mass /= max(1e-12, float(mass.sum()))
    step_mask = np.abs(grid) > OCCLUSION_STEP
    ev = 2.0 * float(mass[step_mask].sum()) - 1.0
    return Relation(cue="depth_step", quantity="signed ordinal depth step",
                    grid=grid, mass=mass, width=w, evidence=ev,
                    detail=dict(signed=signed, owner_prob=prob,
                                flat_floor=float(flat_floor)))


def _diff_population(grid: np.ndarray, mu: float, width: float
                     ) -> np.ndarray:
    """A Gaussian on a BOUNDED axis, with the out-of-range mass CENSORED.

    A difference cannot exceed its axis: a luminance step of 1.9 is not "off
    the chart", it is "at least 1.0", and a Gaussian centred beyond the edge
    would otherwise put almost no mass inside the axis and read as certainty
    that the step is SMALL.  Censoring moves that mass onto the edge bin, which
    is what the measurement actually says.
    """
    lo, hi = float(grid[0]), float(grid[-1])
    step = float(grid[1] - grid[0]) if grid.size > 1 else 1.0
    s = max(1e-6, float(width))
    m = _gauss_on(grid, mu, s)
    m = m / max(1e-12, float(m.sum()))
    cdf = lambda x: 0.5 * (1.0 + math.erf((x - mu) / (s * math.sqrt(2.0))))
    below = cdf(lo - 0.5 * step)
    above = 1.0 - cdf(hi + 0.5 * step)
    if below > 0.0:
        m[0] += below
    if above > 0.0:
        m[-1] += above
    return m / max(1e-12, float(m.sum()))


def _slant_relation(sl_left: Optional[float], sl_right: Optional[float],
                    width: float) -> Relation:
    grid = np.linspace(-90.0, 90.0, 91)
    if sl_left is None or sl_right is None:
        return Relation(cue="slant_diff", quantity="slant(right)-slant(left) deg",
                        grid=grid, mass=np.ones_like(grid) / grid.size,
                        width=width, evidence=0.0, abstained=True,
                        detail=dict(reason="no slant on at least one flank"))
    d = float(sl_right) - float(sl_left)
    mass = _diff_population(grid, d, width)
    # A crease needs a slant difference that the cue width cannot explain.
    step_mask = np.abs(grid) > max(SLANT_SAME_DEG, width)
    ev = 2.0 * float(mass[step_mask].sum()) - 1.0
    return Relation(cue="slant_diff", quantity="slant(right)-slant(left) deg",
                    grid=grid, mass=mass, width=width, evidence=ev,
                    detail=dict(left=float(sl_left), right=float(sl_right),
                                diff=d))


def _tilt_relation(ti_left: Optional[float], ti_right: Optional[float],
                   width: float) -> Relation:
    grid = np.linspace(0.0, 90.0, 91)
    if ti_left is None or ti_right is None:
        return Relation(cue="tilt_diff", quantity="tilt axis difference deg",
                        grid=grid, mass=np.ones_like(grid) / grid.size,
                        width=width, evidence=0.0, abstained=True,
                        detail=dict(reason="no tilt on at least one flank"))
    d = _axis_diff_deg(float(ti_right), float(ti_left))
    mass = _diff_population(grid, d, width)
    step_mask = grid > max(TILT_SAME_DEG, width)
    ev = 2.0 * float(mass[step_mask].sum()) - 1.0
    return Relation(cue="tilt_diff", quantity="tilt axis difference deg",
                    grid=grid, mass=mass, width=width, evidence=ev,
                    detail=dict(left=float(ti_left), right=float(ti_right),
                                diff=d))


def _appearance_relation(st_left: Optional[Dict[str, float]],
                         st_right: Optional[Dict[str, float]],
                         width: float) -> Relation:
    """Relative luminance step between the two flanks, WITH its uncertainty.

    The step is only meaningful next to the uncertainty of the two patch
    means.  On a flat surface the means are precise to ~1e-4 and any step is
    real; on a rough or textured one the same step can be noise.  So the
    population's width is the larger of the declared resolution and twice the
    measured standard error -- a step must clear its own measurement error
    before it is allowed to look like a marking.
    """
    grid = np.linspace(0.0, 1.0, 51)
    if st_left is None or st_right is None:
        return Relation(cue="appearance", quantity="relative luminance step",
                        grid=grid, mass=np.ones_like(grid) / grid.size,
                        width=width, evidence=0.0, abstained=True,
                        detail=dict(reason="no flank patch"))
    scale = max(1e-6, 0.5 * (st_left["mean"] + st_right["mean"]))
    d = abs(st_left["mean"] - st_right["mean"]) / scale
    se = math.hypot(st_left["sd"] / math.sqrt(max(1, st_left["n"])),
                    st_right["sd"] / math.sqrt(max(1, st_right["n"]))) / scale
    w = max(float(width), 2.0 * float(se))
    mass = _diff_population(grid, d, w)
    step_mask = grid > max(APPEARANCE_STEP, w)
    ev = 2.0 * float(mass[step_mask].sum()) - 1.0
    return Relation(cue="appearance", quantity="relative luminance step",
                    grid=grid, mass=mass, width=w, evidence=ev,
                    detail=dict(left=float(st_left["mean"]),
                                right=float(st_right["mean"]), step=d,
                                se=float(se),
                                z=float(d / max(1e-12, se))))


def _phase_relation(ph: Optional[Dict[str, float]], width: float) -> Relation:
    grid = np.linspace(-180.0, 180.0, 121)
    if ph is None:
        return Relation(cue="phase", quantity="quadrature phase deg",
                        grid=grid, mass=np.ones_like(grid) / grid.size,
                        width=width, evidence=0.0, abstained=True,
                        detail=dict(reason="profile off the image"))
    mu = float(ph["phase"])
    d = _wrap180(grid - mu)
    m = np.exp(-0.5 * (d / max(1e-6, width)) ** 2)
    mass = m / max(1e-12, float(m.sum()))
    edge = (np.abs(_wrap180(grid)) <= PHASE_EDGE_TOL_DEG) | \
           (np.abs(_wrap180(grid - 180.0)) <= PHASE_EDGE_TOL_DEG)
    ev = 2.0 * float(mass[edge].sum()) - 1.0
    # A profile that does not clear 3 sigma of its own noise has no phase to
    # report: the number would be an artefact of whatever residual is left.
    # 3 sigma is the standard detection rule, and ``snr2`` is a squared SNR,
    # so the gate is 9.
    snr2 = float(ph.get("snr2", 1e9))
    ok = snr2 >= 9.0
    if not ok:
        ev = 0.0
        mass = np.ones_like(grid) / grid.size
    return Relation(cue="phase", quantity="quadrature phase deg", grid=grid,
                    mass=mass, width=width, evidence=ev, abstained=not ok,
                    detail=dict(phase=mu, modulation=float(ph["modulation"]),
                                snr2=snr2, odd=float(ph["odd"]),
                                even=float(ph["even"])))


def _owner_slant_relation(slants: Dict[str, Optional[float]],
                          rec: Dict[str, Any], width: float) -> Relation:
    grid = np.linspace(0.0, 90.0, 91)
    signed = float(rec.get("signed", 0.0) or 0.0)
    key = "left" if signed > 0.0 else "right"
    val = slants.get(key)
    if val is None:
        return Relation(cue="owner_slant", quantity="owner-side slant deg",
                        grid=grid, mass=np.ones_like(grid) / grid.size,
                        width=width, evidence=0.0, votes=False, abstained=True,
                        detail=dict(reason="no slant on the owner side",
                                    side=key))
    mass = _diff_population(grid, float(val), width)
    return Relation(cue="owner_slant", quantity="owner-side slant deg",
                    grid=grid, mass=mass, width=width, evidence=0.0,
                    votes=False, detail=dict(side=key, slant=float(val)))


def _kinetic_relation(kinetic: Optional[Dict[str, float]],
                      rec: Dict[str, Any]) -> Relation:
    """Occlusion by accretion/deletion, on the SAME ordinal step axis (E4).

    A kinetic occlusion cue is not a different kind of evidence: when a
    surface disappears BEHIND another, the border temporarily translates and
    the occluded side is the one losing area.  That is again an ordinal step
    along the fragment's normal, so it belongs on the depth_step axis and must
    add to it rather than open a parallel vocabulary.

    This function is the INTERFACE ONLY.  No motion is computed anywhere in
    this tree: a caller that has measured the step supplies
    ``{"step": s, "confidence": p}`` with ``s`` in [-1, +1] on the same
    convention as ``depth_step`` (-1 = the left flank is nearer).  Without
    that, the relation ABSTAINS -- it contributes mass only at 0, and it is
    excluded from the vote so an empty cue cannot dilute the others.
    """
    grid = np.linspace(-1.0, 1.0, 41)
    w = _ORDINAL_WIDTH
    if not kinetic:
        return Relation(cue="kinetic", quantity="signed ordinal motion step",
                        grid=grid, mass=np.ones_like(grid) / grid.size,
                        width=w, evidence=0.0, votes=False, abstained=True,
                        detail=dict(reason="no kinetic measurement in tree"))
    s = float(kinetic.get("step", 0.0))
    p = min(max(float(kinetic.get("confidence", 0.0)), 0.0), 1.0)
    s = min(max(s, -1.0), 1.0)
    mass = p * _gauss_on(grid, s, w) + (1.0 - p) * _gauss_on(grid, 0.0, w)
    mass /= max(1e-12, float(mass.sum()))
    step_mask = np.abs(grid) > OCCLUSION_STEP
    ev = 2.0 * float(mass[step_mask].sum()) - 1.0
    return Relation(cue="kinetic", quantity="signed ordinal motion step",
                    grid=grid, mass=mass, width=w, evidence=ev, votes=True,
                    detail=dict(step=s, confidence=p))


# ---------------------------------------------------------------------------
# assembly
# ---------------------------------------------------------------------------

def _flank_slant(lum: np.ndarray, x: float, y: float, light: LightPrior,
                 patch: float) -> Tuple[Optional[float], Optional[float],
                                       str]:
    """(slant, tilt, source) at one flank: shading first, texture second.

    Shading is preferred because it is defined on a smooth LAMBERTIAN patch,
    which is what a flank is; the texture cue is the fallback where the
    shading fit is rejected (texture, or a strongly curved faceted surface).
    A flank that neither can read returns (None, None) and the relation
    abstains -- no default value is invented.
    """
    umb = shade_umbilic(lum, x, y, light, patch_radius_px=patch)
    if umb is not None:
        return float(umb["slant"]), float(umb["tilt"]), "shading"
    pop, diag = texture_cue_calibrated(lum, x, y, patch_radius_px=max(patch, 8.0))
    if pop is not None:
        # The calibrated texture cue ABSTAINS on tilt when its gradient is
        # isotropic.  The slant survives that abstention; the tilt does not,
        # and reporting a fabricated 0 deg would look like a measured normal
        # agreeing with the other flank.
        ti = float(diag["tilt"]) if diag.get("tilt_defined") else None
        return float(diag["slant"]), ti, "texture"
    return None, None, "none"


def relations_for_fragment(rec: Dict[str, Any], lum: np.ndarray,
                           light: LightPrior, *,
                           offset_px: Optional[float] = None,
                           slant_width: float = 3.0,
                           tilt_width: float = 10.9,
                           appearance_width: float = _APPEARANCE_WIDTH,
                           phase_width: float = PHASE_EDGE_TOL_DEG / 3.0,
                           flat_floor: float = 0.0,
                           kinetic: Optional[Dict[str, float]] = None
                           ) -> Tuple[List[Relation], Dict[str, Any]]:
    """The relations of one contour fragment plus its flank diagnostics."""
    pos = rec.get("pos")
    nrm = rec.get("normal")
    if pos is None or nrm is None:
        return [], dict(reason="no position or normal")
    cx, cy = float(pos[0]), float(pos[1])
    nx, ny = float(nrm[0]), float(nrm[1])
    mag = math.hypot(nx, ny)
    if mag <= 1e-9:
        return [], dict(reason="degenerate normal")
    nx, ny = nx / mag, ny / mag
    # The flank offset is HALF the measured member spacing: at a full spacing
    # the flank patch can straddle the NEXT contour, and nearer than half it
    # sits inside the contour's own blur.  ``spacing`` is supplied by the
    # caller from the chain's own geometry, so the offset is measured rather
    # than a pixel constant.
    if offset_px is not None:
        off = float(offset_px)
    elif rec.get("spacing"):
        off = float(rec["spacing"]) * FLANK_OFFSET_SCALE
    else:
        off = 5.0 * FLANK_OFFSET_SCALE
    off = max(1.0, off)
    lx, ly = cx + nx * off, cy + ny * off
    rx, ry = cx - nx * off, cy - ny * off
    patch = max(3.0, 0.45 * off)

    sl_l, ti_l, src_l = _flank_slant(lum, lx, ly, light, patch)
    sl_r, ti_r, src_r = _flank_slant(lum, rx, ry, light, patch)
    st_l = _patch_stats(lum, lx, ly, patch)
    st_r = _patch_stats(lum, rx, ry, patch)
    ph = _luminance_phase(lum, cx, cy, nx, ny, max(1.5, 0.5 * off))

    rels = [
        _depth_step_relation(rec, flat_floor=flat_floor),
        _slant_relation(sl_l, sl_r, slant_width),
        _tilt_relation(ti_l, ti_r, tilt_width),
        _appearance_relation(st_l, st_r, appearance_width),
        _phase_relation(ph, phase_width),
        _owner_slant_relation(dict(left=sl_l, right=sl_r), rec, slant_width),
        _kinetic_relation(kinetic, rec),
    ]
    diag = dict(
        pos=(cx, cy), offset=off, patch=patch,
        left=(lx, ly), right=(rx, ry),
        slant_left=sl_l, slant_right=sl_r, source_left=src_l,
        source_right=src_r,
        tilt_left=ti_l, tilt_right=ti_r,
        mean_left=None if st_l is None else st_l["mean"],
        mean_right=None if st_r is None else st_r["mean"],
        phase=None if ph is None else ph["phase"],
        modulation=None if ph is None else ph["modulation"],
    )
    return rels, diag


def boundary_gate(rels: Sequence[Relation]) -> Dict[str, Any]:
    """The two-cue boundary statistic for one fragment: appearance arms, phase decides.

    This is the statistic Session G.2 measured on the 24 scene-variants, in the
    module so the next proposer imports ONE function instead of re-deriving it.
    It is a decision STATISTIC assembled from the graded populations, not a
    category: every number it returns is read off the two relations.

    The two questions are different and the order matters:

    * ``appearance`` asks WHETHER there is a boundary here -- a step above the
      axis resolution AND above its own measurement error (3 sigma).  Without
      it the fragment is not a boundary at all, and the score is **-1, not 0**:
      a fragment the system cannot see is a MISS, not a neutral abstention.
      That choice is what stops the statistic from passing by declining to
      answer.
    * ``phase`` asks WHAT KIND of boundary it is -- a step (something ends: the
      profile is odd, phase 0/180) or a line (a ridge or stripe: even, phase
      +-90).  Its evidence is already graded in [-1, +1].

    Returns the two measurements, the assembled ``score``, and a `reason`
    string for the Diagnostic Shadow.  ``step_likeness`` is None when the phase
    cue abstained -- the boundary exists but its kind is unknown, which is a
    different statement from "a line".
    """
    by = {r.cue: r for r in rels}
    app = by.get("appearance")
    ph = by.get("phase")
    step = 0.0
    z = 0.0
    if app is not None and not app.abstained:
        step = float(app.detail.get("step", 0.0) or 0.0)
        z = float(app.detail.get("z", 0.0) or 0.0)
    armed = bool(step > APPEARANCE_STEP and z > APPEARANCE_ARM_SIGMA)

    if ph is None:
        likeness = None
        phase_deg = None
        snr2 = None
    else:
        phase_deg = ph.detail.get("phase")
        snr2 = ph.detail.get("snr2")
        likeness = None if ph.abstained else float(ph.evidence)

    if not armed:
        score = -1.0
        reason = ("no appearance step above %.3f and %.1f sigma "
                  "(step %.3f, z %.1f): not a boundary, counted as a miss"
                  % (APPEARANCE_STEP, APPEARANCE_ARM_SIGMA, step, z))
    elif likeness is None:
        score = 0.0
        reason = ("a boundary (step %.3f) but the profile does not clear its "
                  "own noise: step-or-line unknown" % step)
    elif likeness > 0.25:
        score = likeness
        reason = ("a boundary (step %.3f) with a STEP profile (phase %.0f deg, "
                  "snr^2 %.1f): something ends here" % (step, phase_deg or 0.0,
                                                        snr2 or 0.0))
    elif likeness < -0.25:
        score = likeness
        reason = ("a boundary (step %.3f) with a LINE profile (phase %.0f deg, "
                  "snr^2 %.1f): a ridge or stripe, not a step"
                  % (step, phase_deg or 0.0, snr2 or 0.0))
    else:
        score = likeness
        reason = ("a boundary (step %.3f) with a weak step reading "
                  "(phase %.0f deg): indeterminate" % (step, phase_deg or 0.0))
    return dict(armed=armed, step=float(step), z=float(z),
                step_likeness=likeness, score=float(score),
                phase_deg=(None if phase_deg is None else float(phase_deg)),
                snr2=(None if snr2 is None else float(snr2)), reason=reason)


def combine_relations(rels: Sequence[Relation], *,
                      weights: Optional[Dict[str, float]] = None
                      ) -> Dict[str, Any]:
    """Mixture of the VOTING relations: sum of graded evidence.

    Returns the combined evidence in [-1, +1] (the mean, so the range does not
    depend on how many cues voted), the per-cue evidence, the abstentions, and
    a disagreement statistic: the spread of the votes.  A large spread means
    the cues disagree and the combined number must be read as uncertain -- which
    is exactly what a sum does and a product cannot express.
    """
    votes: Dict[str, float] = {}
    abstained: List[str] = []
    for r in rels:
        if not r.votes:
            continue
        if r.abstained:
            abstained.append(r.cue)
            continue
        w = float(weights.get(r.cue, 1.0)) if weights else 1.0
        votes[r.cue] = w * float(r.evidence)
    if not votes:
        return dict(evidence=0.0, votes={}, abstained=abstained,
                    spread=None, n=0)
    vals = np.asarray(list(votes.values()), dtype=np.float64)
    ev = float(vals.mean())
    spread = float(vals.std())
    return dict(evidence=ev, votes=votes, abstained=abstained,
                spread=spread, n=int(vals.size),
                # "ambiguous" is not a threshold on the mean: it is the cues
                # carrying OPPOSITE signs with comparable weight.
                ambiguous=bool(vals.min() < -0.25 and vals.max() > 0.25))


def explain(rec: Dict[str, Any], rels: Sequence[Relation],
            combined: Dict[str, Any]) -> str:
    """The Diagnostic Shadow: one human-readable line per fragment.

    The engine stays mathematical; this names WHY, in the same words a person
    would use, and it is generated FROM the numbers -- never the reverse.
    """
    parts: List[str] = []
    for r in rels:
        if r.cue == "depth_step":
            p = float(r.detail.get("owner_prob", 0.0))
            s = float(r.detail.get("signed", 0.0))
            side = "left" if s > 0 else ("right" if s < 0 else "neither")
            parts.append("depth: owner-prob %.2f on the %s (step vote %+.2f)"
                         % (p, side, r.evidence))
        elif r.cue == "slant_diff":
            if r.abstained:
                parts.append("slant: unavailable on a flank")
            else:
                d = float(r.detail.get("diff", 0.0))
                parts.append(
                    "slant difference %.1f deg %s" %
                    (d, ("-> surfaces differ" if r.evidence > 0.25
                         else "-> same surface, not a crease")))
        elif r.cue == "tilt_diff":
            if r.abstained:
                parts.append("tilt: undefined on a flank")
            else:
                d = float(r.detail.get("diff", 0.0))
                parts.append("tilt difference %.1f deg %s" %
                             (d, ("-> crease" if r.evidence > 0.25
                                  else "-> same normal")))
        elif r.cue == "appearance":
            if r.abstained:
                parts.append("appearance: no flank patch")
            else:
                step = float(r.detail.get("step", 0.0))
                if r.evidence <= 0.25:
                    what = "(no step to speak of)"
                elif step > 1.0:
                    # A relative step above 1 needs one side near black: this is
                    # a boundary against the background or a very dark material,
                    # NOT a marking.  The words must follow the number.
                    what = ("(boundary to a near-black side -- background or "
                            "dark material, not a marking)")
                else:
                    what = "(a real reflectance boundary)"
                parts.append("appearance step %.3f %s" % (step, what))
        elif r.cue == "phase":
            if r.abstained:
                parts.append("phase: profile does not clear its own noise "
                             "(no step-or-line reading)")
            else:
                ph = float(r.detail.get("phase", 0.0))
                edge = r.evidence > 0.25
                parts.append("phase %.0f deg (snr^2 %.1f) %s" %
                             (ph, float(r.detail.get("snr2", 0.0)),
                              "(a STEP: something ends here)" if edge
                              else "(a LINE: a ridge or stripe, not a step)"))
        elif r.cue == "kinetic":
            if r.abstained:
                parts.append("motion: not measured in this tree (abstains)")
            else:
                parts.append("motion step %+.2f at confidence %.2f" %
                             (float(r.detail.get("step", 0.0)),
                              float(r.detail.get("confidence", 0.0))))
    tail = ("COMBINED %+.2f from %d cues" % (combined["evidence"],
                                            combined.get("n", 0)))
    if combined.get("ambiguous"):
        tail += " -- CUES DISAGREE (spread %.2f): unknown, not a guess" % (
            combined.get("spread") or 0.0)
    return "; ".join(parts + [tail])


#: Public alias: the E4 slot, so a future motion front end imports ONE name
#: that exists today and abstains until measurement arrives.
kinetic_relation = _kinetic_relation
