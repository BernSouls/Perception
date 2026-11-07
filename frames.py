"""perception/frames.py -- ONE source of geometry, and the FRAME BANK over it.

WHY THIS EXISTS
---------------
A surface fragment is content at a coordinate (audit item 3).  The content is
what the identity is keyed on; the coordinate is a NON-IDENTITY attribute that
says where the content was seen.  For two observations to be comparable, the
coordinate has to be expressed in a frame that is *derived from the
observation itself*, never from a truth value, an object class or a mask.

The frame bank is that derivation.  It measures ONE origin and ONE scale from
the observation's own energy --

    origin   the weighted energy centroid, in pixels
    scale    the RMS energy radius, in pixels

-- and then offers the candidate frames that can be built on it:

    upright     the image's own axes (u = +x, v = +y), theta from image-up.
                Always determined: the axes are GIVEN, not measured.
    principal   the axes of the energy's second moment (major / minor).
                Determined only as far as the energy is anisotropic: for a
                disc the major axis is undefined, and the frame says so
                (determinacy -> 0) instead of inventing an axis.
    radial      the per-point frame whose u axis points along the radius
                from the origin.  Determined only away from the centre: at
                the origin the radial direction is undefined.

Coordinates are in ENERGY RADII, so they are scale-free: doubling the frame
size leaves (u, v) unchanged.  Theta is measured from image-up (0 = above the
origin, +pi/2 = to the right), matching `memory.aspects._upright_polar`.

SELECTION IS DETERMINISTIC, AND BY MEASUREMENT
----------------------------------------------
`FrameBank.select` takes the candidate with the highest DETERMINACY -- a
geometric property of the frame, not a learned preference -- and breaks ties
by the DECLARED ORDER (lowest index).  Because the upright frame's axes are
given by the image its determinacy is 1 by construction, so the default
selection is upright unless the caller restricts the candidates.  That is the
same answer the W2 stability probe measured independently (the upright frame
beat the principal-axis frame in every class), so the declared order is also
the measured one.  Nothing here is tuned.

WHAT IS NOT HERE
----------------
No identity.  This module produces geometry only: no hash, no alpha, no
signature.  A caller attaches the coordinates as a non-identity attribute and
is then free to derive relations (distance, bearing, offset) from them.  The
relations are the SAME under every frame -- a frame is a rotation, a uniform
scale and a translation -- which is what makes "one source of geometry" true
rather than a slogan.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "FRAME_UPRIGHT",
    "FRAME_PRINCIPAL",
    "FRAME_RADIAL",
    "FRAME_ORDER",
    "Frame",
    "FrameBank",
]

FRAME_UPRIGHT = "upright"
FRAME_PRINCIPAL = "principal"
FRAME_RADIAL = "radial"

#: The declared candidate order.  Ties in determinacy resolve to the LOWEST
#: index, so this order IS the tie-break rule.
FRAME_ORDER: Tuple[str, ...] = (FRAME_UPRIGHT, FRAME_PRINCIPAL, FRAME_RADIAL)


@dataclass(frozen=True)
class Frame:
    """One rigid frame: an origin, a uniform scale, and a rotation.

    ``angle`` is the direction of the frame's u axis in IMAGE coordinates
    (x right, y down), in radians, canonicalised to [0, pi) because every
    frame this module builds is AXIAL (an axis has no head or tail).

    ``determinacy`` is the measured conditioning of the frame in [0, 1]:
    1 = the axes are fully determined, 0 = the axes are undefined.
    """

    name: str
    origin: Tuple[float, float]
    scale: float
    angle: float
    determinacy: float

    # -- image <-> frame ---------------------------------------------------
    def to_local(self, point: Sequence[float]) -> Tuple[float, float]:
        """Image (x, y) -> frame (u, v), in energy radii."""
        dx = float(point[0]) - self.origin[0]
        dy = float(point[1]) - self.origin[1]
        c, s = math.cos(self.angle), math.sin(self.angle)
        scale = self.scale if self.scale > 1e-12 else 1.0
        return ((c * dx + s * dy) / scale, (-s * dx + c * dy) / scale)

    def to_image(self, uv: Sequence[float]) -> Tuple[float, float]:
        """Frame (u, v), in energy radii -> image (x, y)."""
        u = float(uv[0]) * self.scale
        v = float(uv[1]) * self.scale
        c, s = math.cos(self.angle), math.sin(self.angle)
        return (self.origin[0] + c * u - s * v,
                self.origin[1] + s * u + c * v)

    # -- the polar address -------------------------------------------------
    def polar(self, point: Sequence[float]) -> Tuple[float, float]:
        """(r, theta): radius in energy radii, theta from image-up.

        theta is 0 above the origin, +pi/2 to the right, in (-pi, pi].  In a
        frame rotated by ``angle`` the address is the upright address MINUS
        that angle, so the frame is rotation-covariant and the radius is
        invariant.
        """
        u, v = self.to_local(point)
        return (math.hypot(u, v), math.atan2(u, -v))


def _clamp01(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return min(1.0, max(0.0, float(value)))


def _axial_angle(sxx: float, syy: float, sxy: float) -> Tuple[float, float]:
    """(major-axis angle in [0, pi), anisotropy in [0, 1]).

    The eigenvalues of the [[sxx, sxy], [sxy, syy]] covariance in closed
    form; the angle is the eigenvector of the larger one, taken on the AXIAL
    circle.  An isotropic cloud (sxx == syy, sxy == 0) has no major axis, so
    the angle is 0 and the anisotropy is 0 -- an honest "undefined", not a
    fabricated direction.
    """
    mean = 0.5 * (sxx + syy)
    half = 0.5 * (sxx - syy)
    disc = math.hypot(half, sxy)
    l1 = mean + disc
    l2 = mean - disc
    total = l1 + l2
    if total <= 1e-18:
        return (0.0, 0.0)
    aniso = _clamp01((l1 - l2) / total)
    # 2x2 eigenvector of the larger eigenvalue; the doubled-angle form keeps
    # it stable when the eigenvalues are close.
    angle = 0.5 * math.atan2(2.0 * sxy, sxx - syy)
    return (angle % math.pi, aniso)


@dataclass(frozen=True)
class FrameBank:
    """The measured origin/scale of one observation and its candidate frames.

    Build it from the observation's oriented energy (``from_energy``) or from
    the weighted coordinates of its surface fragments (``from_points``); both
    use the SAME definition (weighted centroid, RMS radius), so the geometry
    has one source.  ``None`` is returned when the observation carries no
    usable mass -- a quiet scene, never a fabricated frame.
    """

    origin: Tuple[float, float]
    scale: float
    principal_angle: float
    anisotropy: float
    mass: float
    n_points: int

    # -- construction ------------------------------------------------------
    @classmethod
    def from_points(cls, points: Iterable[Sequence[float]],
                    weights: Optional[Iterable[float]] = None
                    ) -> Optional["FrameBank"]:
        """The bank of a weighted point cloud (fragment positions)."""
        rows: List[Tuple[float, float, float]] = []
        got = list(points)
        if weights is None:
            w = [1.0] * len(got)
        else:
            w = [float(v) for v in weights]
        if len(w) != len(got):
            return None
        for p, weight in zip(got, w):
            if p is None or len(p) < 2:
                continue
            x, y = float(p[0]), float(p[1])
            if not (math.isfinite(x) and math.isfinite(y)
                    and math.isfinite(weight)):
                continue
            if weight <= 0.0:
                continue
            rows.append((x, y, weight))
        if not rows:
            return None
        rows.sort()
        mass = math.fsum(r[2] for r in rows)
        if mass <= 0.0:
            return None
        cx = math.fsum(r[0] * r[2] for r in rows) / mass
        cy = math.fsum(r[1] * r[2] for r in rows) / mass
        dx = [r[0] - cx for r in rows]
        dy = [r[1] - cy for r in rows]
        radius = math.sqrt(max(0.0, math.fsum(
            r[2] * (x * x + y * y) for r, x, y in zip(rows, dx, dy)) / mass))
        if radius <= 1e-9:
            radius = 1.0
        sxx = math.fsum(r[2] * x * x for r, x in zip(rows, dx)) / (mass * radius ** 2)
        syy = math.fsum(r[2] * y * y for r, y in zip(rows, dy)) / (mass * radius ** 2)
        sxy = math.fsum(r[2] * x * y for r, x, y in zip(rows, dx, dy)) / (mass * radius ** 2)
        angle, aniso = _axial_angle(sxx, syy, sxy)
        return cls(origin=(cx, cy), scale=radius, principal_angle=angle,
                   anisotropy=aniso, mass=mass, n_points=len(rows))

    @classmethod
    def from_energy(cls, energy: Any) -> Optional["FrameBank"]:
        """The bank of an oriented-energy map ``(n_theta, H, W)`` or ``(H, W)``.

        The definition is exactly ``memory.gist.build_gist``'s: the pixel
        centres are the sample positions, the energy is the weight, and the
        scale is the RMS distance from the centroid.  One definition, so a
        view's gist origin and a view's fragment frame origin agree.
        """
        if energy is None:
            return None
        e = np.asarray(energy, dtype=np.float64)
        if e.ndim == 3:
            e = e.sum(axis=0)
        if e.ndim != 2 or e.size == 0:
            return None
        total = float(e.sum())
        if not math.isfinite(total) or total <= 0.0:
            return None
        h, w = e.shape
        ys, xs = np.mgrid[0:h, 0:w]
        cx = float((e * xs).sum() / total)
        cy = float((e * ys).sum() / total)
        dx = xs - cx
        dy = ys - cy
        radius = float(math.sqrt((e * (dx * dx + dy * dy)).sum() / total))
        if not math.isfinite(radius) or radius <= 1e-9:
            radius = 1.0
        sxx = float((e * dx * dx).sum() / (total * radius ** 2))
        syy = float((e * dy * dy).sum() / (total * radius ** 2))
        sxy = float((e * dx * dy).sum() / (total * radius ** 2))
        angle, aniso = _axial_angle(sxx, syy, sxy)
        return cls(origin=(cx, cy), scale=radius, principal_angle=angle,
                   anisotropy=aniso, mass=total, n_points=int(e.size))

    # -- the candidates ----------------------------------------------------
    def determinacy(self, name: str,
                    point: Optional[Sequence[float]] = None) -> float:
        """The measured conditioning of one candidate frame, in [0, 1].

        upright is 1 by construction (the image supplies the axes).
        principal is the energy's anisotropy (0 for a disc).
        radial is ``min(1, r / scale)`` at ``point`` and undefined (0) when
        no point is given, because a radial FRAME is a per-point object.
        """
        if name == FRAME_UPRIGHT:
            return 1.0
        if name == FRAME_PRINCIPAL:
            return _clamp01(self.anisotropy)
        if name == FRAME_RADIAL:
            if point is None:
                return 0.0
            r, _ = self.frame(FRAME_UPRIGHT).polar(point)
            return _clamp01(r)
        return 0.0

    def frame(self, name: str,
              point: Optional[Sequence[float]] = None) -> Optional[Frame]:
        """One named frame, or None for an unknown name.

        ``radial`` needs the point it is the frame OF; without one it cannot
        be built (None) rather than silently borrowing the upright axes.
        """
        if name == FRAME_UPRIGHT:
            return Frame(FRAME_UPRIGHT, self.origin, self.scale, 0.0, 1.0)
        if name == FRAME_PRINCIPAL:
            return Frame(FRAME_PRINCIPAL, self.origin, self.scale,
                         self.principal_angle,
                         _clamp01(self.anisotropy))
        if name == FRAME_RADIAL:
            if point is None:
                return None
            dx = float(point[0]) - self.origin[0]
            dy = float(point[1]) - self.origin[1]
            if abs(dx) < 1e-12 and abs(dy) < 1e-12:
                return None
            angle = math.atan2(dy, dx) % math.pi
            r = math.hypot(dx, dy) / (self.scale if self.scale > 1e-12 else 1.0)
            return Frame(FRAME_RADIAL, self.origin, self.scale, angle,
                         _clamp01(r))
        return None

    def frames(self) -> Tuple[Frame, ...]:
        """The scene-level candidates, in the declared order."""
        out: List[Frame] = []
        for name in FRAME_ORDER:
            if name == FRAME_RADIAL:
                continue
            f = self.frame(name)
            if f is not None:
                out.append(f)
        return tuple(out)

    # -- selection ---------------------------------------------------------
    def select(self, point: Optional[Sequence[float]] = None,
               candidates: Sequence[str] = FRAME_ORDER) -> Optional[Frame]:
        """The candidate with the highest determinacy, ties to lowest index.

        A strict ``>`` keeps the FIRST of two equally determined candidates,
        which is the declared tie-break (upright first).  ``None`` only when
        every candidate is undefined (the radial frame with no point).
        """
        best: Optional[Frame] = None
        best_score = -1.0
        for name in candidates:
            score = self.determinacy(name, point)
            frame = self.frame(name, point)
            if frame is None:
                continue
            if score > best_score:
                best = frame
                best_score = score
        return best

    # -- coordinates and the relations derived from them -------------------
    def coordinates(self, points: Iterable[Sequence[float]],
                    frame: Optional[Frame] = None,
                    point: Optional[Sequence[float]] = None
                    ) -> List[Tuple[float, float]]:
        """(u, v) of each point, in energy radii.

        ``frame`` defaults to ``select(point)``; a caller that needs a
        specific frame names it or passes one from ``frame(name)``.
        """
        f = frame if frame is not None else self.select(point)
        if f is None:
            return []
        return [f.to_local(p) for p in points]

    def polar(self, points: Iterable[Sequence[float]],
              frame: Optional[Frame] = None,
              point: Optional[Sequence[float]] = None
              ) -> List[Tuple[float, float]]:
        """(r, theta) of each point, r in energy radii, theta from image-up."""
        f = frame if frame is not None else self.select(point)
        if f is None:
            return []
        return [f.polar(p) for p in points]

    def offset(self, a: Sequence[float], b: Sequence[float],
               frame: Optional[Frame] = None) -> Tuple[float, float]:
        """The frame-space displacement a -> b, in energy radii."""
        f = frame if frame is not None else self.select()
        if f is None:
            return (0.0, 0.0)
        ua, va = f.to_local(a)
        ub, vb = f.to_local(b)
        return (ub - ua, vb - va)

    def distance(self, a: Sequence[float], b: Sequence[float]) -> float:
        """The distance in energy radii.  FRAME-INVARIANT by construction."""
        return math.hypot(float(b[0]) - float(a[0]),
                          float(b[1]) - float(a[1])) / (
            self.scale if self.scale > 1e-12 else 1.0)

    def bearing(self, a: Sequence[float], b: Sequence[float],
                frame: Optional[Frame] = None) -> float:
        """The bearing a -> b, in radians from image-up, in (-pi, pi]."""
        f = frame if frame is not None else self.select()
        if f is None:
            return 0.0
        u, v = self.offset(a, b, f)
        return math.atan2(u, -v)
