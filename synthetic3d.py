"""perception/synthetic3d.py -- analytic 3D renderer with exact truth.

WHY THIS EXISTS
---------------
Session A needs a renderer whose ground truth is honest: normals, depth,
ordinal layer and motion must be computed analytically, not estimated from the
same image the perception stack reads.  This module is that renderer.

It is a ray caster over three analytic primitives (sphere, cylinder, box).  The
default camera is ORTHOGRAPHIC, deliberately: it removes camera intrinsics from
the calibration, makes the image-plane normal of a smooth silhouette exactly
perpendicular to the contour tangent, and makes the image motion of a rigid
point exactly the (x, y) part of its 3D velocity.  An optional pinhole
``Camera`` (``Scene.camera``) adds perspective; it is OFF by default, and every
Session A/B1 number was measured with it off.  Perspective changes the truth
formulae honestly rather than approximately:

* depth is the axial distance ``e = camera_z - z`` (equal to the ray parameter
  only under orthographic projection),
* slant is measured against the direction from the surface to the camera, so
  the occluding contour is ``n . view == 0`` rather than ``nz == 0``,
* image motion is the exact pinhole derivation
  ``f (Vx e + X Vz) / e^2``, which reduces to ``Vx / pixel_scale`` as
  ``e -> inf`` with ``f = e / pixel_scale``.

Optional sensor stages, also off by default: a fixed Gaussian blur
(``blur_sigma``), seeded additive Gaussian noise (``noise_sigma``,
``noise_seed``) and several directional lights (``extra_lights``).  None of
them touches the truth fields; they only change the luminance image.

CONVENTIONS
-----------
* Image x is right, image y is down, world z points TOWARD the viewer.
* A camera ray is ``origin = (x, y, camera_z)``, ``direction = (0, 0, -1)``;
  the nearest visible hit is the one with the smallest positive ray parameter
  ``t`` (``t`` is also the depth: smaller ``t`` is nearer).
* A surface normal has ``n . view > 0`` on a visible surface, i.e. ``nz > 0``.
  At a smooth occluding contour ``nz == 0`` and the normal lies in the image
  plane, perpendicular to the contour tangent.
* The renderer's object-id mask, layer, normal and depth are TRUTH.  They are
  never handed to the perception path; ``surface_fragments`` reads only the
  luminance image plus the front-end's own chains and ownership.

Nothing here is learned, sampled randomly, or tuned: every pixel is a closed
form intersection.  ``render`` is deterministic and supersampling is a fixed
grid, so two calls with the same scene are byte-identical.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import math

import numpy as np

__all__ = [
    "Sphere",
    "Cylinder",
    "Box",
    "Material",
    "Camera",
    "SceneObject",
    "Scene",
    "Hit",
    "FragmentTruth",
    "RenderTruth",
    "rotation_xyz",
    "single_object_scene",
    "occlusion_scene",
    "KNOWN_SIZE_RADII",
    "size_class_radius",
    "assign_size_class",
]

Vec3 = np.ndarray
Mat3 = np.ndarray

_EPS = 1e-12


# ---------------------------------------------------------------------------
# small linear algebra
# ---------------------------------------------------------------------------

def _unit(v: Sequence[float]) -> Vec3:
    a = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(a))
    if n < _EPS:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return a / n


def _as_dir(d: np.ndarray, n_pix: int) -> np.ndarray:
    """Ray directions as an ``(n_pix, 3)`` array.

    A shared ``(3,)`` direction (orthographic, or one primitive transform) is
    broadcast; a per-pixel ``(n_pix, 3)`` direction (perspective) passes
    through.  The result may be read-only.
    """
    a = np.asarray(d, dtype=np.float64)
    if a.ndim == 1:
        return np.broadcast_to(a, (int(n_pix), 3))
    return a


def rotation_xyz(rx: float, ry: float, rz: float) -> Mat3:
    """Right-handed rotation, applied as ``R @ p``.

    ``rx`` about x, then ``ry`` about y, then ``rz`` about z; angles in radians.
    The order is fixed so a scene is reproducible from its parameters.
    """
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    Rx = np.array([[1.0, 0.0, 0.0],
                   [0.0, cx, -sx],
                   [0.0, sx, cx]], dtype=np.float64)
    Ry = np.array([[cy, 0.0, sy],
                   [0.0, 1.0, 0.0],
                   [-sy, 0.0, cy]], dtype=np.float64)
    Rz = np.array([[cz, -sz, 0.0],
                   [sz, cz, 0.0],
                   [0.0, 0.0, 1.0]], dtype=np.float64)
    return Rz @ Ry @ Rx


# ---------------------------------------------------------------------------
# known size classes
# ---------------------------------------------------------------------------
#
# Relative size is only measurable against a KNOWN physical size.  These are
# the renderer's size bins: a name maps to one bounding radius in world units.
# The values are the radii the existing scene builders already use (0.18-0.60),
# rounded to a coarse geometric ladder, so a corpus can state "this is the
# medium class" without the estimator being allowed to see the object id.
# They are a TABLE, not a threshold: an object's apparent radius divided by its
# class radius is its inverse depth, to be compared against the other cues.
KNOWN_SIZE_RADII: Dict[str, float] = {
    "tiny": 0.15,
    "small": 0.22,
    "medium": 0.30,
    "large": 0.42,
    "huge": 0.60,
}


def size_class_radius(name: str) -> float:
    """World radius for a known size class (KeyError on an unknown name)."""
    try:
        return float(KNOWN_SIZE_RADII[str(name)])
    except KeyError:
        raise ValueError("unknown size class %r (known: %s)"
                         % (name, ", ".join(sorted(KNOWN_SIZE_RADII)))) from None


def _object_bound_radius(shape: Any) -> float:
    """Bounding radius of a primitive, in world units."""
    if isinstance(shape, Sphere):
        return float(shape.radius)
    if isinstance(shape, Cylinder):
        return float(math.hypot(shape.radius, shape.half_height))
    h = np.asarray(getattr(shape, "half_extents", (0.5, 0.5, 0.5)),
                   dtype=np.float64)
    return float(np.linalg.norm(h))


def assign_size_class(obj: "SceneObject") -> str:
    """Tag ``obj`` with the nearest class radius, on a LOG scale.

    The assignment is derived from the object's own geometry and the table
    above; it is not a perception estimate.  Log distance is used because the
    bins form a geometric ladder, so the nearest bin in log space is the
    nearest multiplicative size.
    """
    r = _object_bound_radius(obj.shape)
    best = min(KNOWN_SIZE_RADII,
               key=lambda n: abs(math.log(r / KNOWN_SIZE_RADII[n])))
    obj.size_class = best
    return best


# ---------------------------------------------------------------------------
# materials
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Material:
    """Lambertian material with an optional deterministic procedural texture.

    ``albedo`` is the diffuse reflectance.  ``texture`` is ``"none"``,
    ``"stripes"``, ``"checker"`` or ``"noise"``; the texture multiplies the
    albedo.  It is procedural (no sampled image), so a render is reproducible
    from the scene alone.  ``texture_contrast`` in [0, 1] is the modulation
    depth.

    ``"noise"`` is deterministic value noise on a lattice of
    ``texture_freq`` cells per unit of surface coordinate: a BROADBAND,
    statistically ISOTROPIC texture.  It exists because the texture-gradient
    cue assumes an isotropic texture spectrum, and the other two textures
    violate that by construction (a grating has one axis of gradient energy);
    measuring the cue on a texture that cannot satisfy its assumption would
    only measure the violation.

    The surface coordinate is METRIC: every primitive's ``uv`` is arc length
    in WORLD units, so one ``texture_freq`` cell covers ``1 / texture_freq``
    world units on every object and every face.  Session G fixed this: the
    old normalised coordinates stretched the texture by each primitive's own
    dimensions (a box face by its aspect ratio), and that false anisotropy
    was measured as a slant bias.
    """

    albedo: float = 0.85
    texture: str = "none"
    texture_freq: float = 8.0
    texture_contrast: float = 0.5
    #: How the texture is ADDRESSED.
    #:
    #: "face" (default) uses the primitive's per-face metric ``(u, v)``.  It is
    #: metric but it is NOT continuous across a fold: two faces parametrise
    #: their shared edge differently, so a box crease SLICES a stripe pattern
    #: by construction (measured, Session G.3).  That is a property of the
    #: parametrisation, not of the scene, and it made "does the pattern continue
    #: across a crease?" untestable.
    #:
    #: "object" addresses a CONTINUOUS field in the OBJECT's own frame --
    #: ``u = p_local . texture_dir`` and ``v`` along an orthogonal axis -- so one
    #: material continues across a fold, while two objects (different local
    #: frames, and a phase offset where the scene says so) remain independent.
    #: That is the physical statement "a material is continuous over one body,
    #: not across two", and it is what a crease/occlusion test needs.
    #:
    #: The default is "face" so every earlier measurement and fitted table
    #: stands unchanged; no probe is silently re-rendered.
    texture_space: str = "face"
    texture_dir: Tuple[float, float, float] = (0.7071067811865476, 0.0,
                                               0.7071067811865476)

    def _object_uv(self, p_local):
        """Texture coordinates of the object-local 3D point (fold-continuous)."""
        d1 = np.asarray(self.texture_dir, dtype=np.float64)
        n1 = float(np.linalg.norm(d1))
        d1 = d1 / n1 if n1 > 0 else np.array([1.0, 0.0, 0.0])
        anchor = (np.array([0.0, 1.0, 0.0]) if abs(d1[1]) < 0.9
                  else np.array([1.0, 0.0, 0.0]))
        d2 = np.cross(d1, anchor)
        d2 = d2 / float(np.linalg.norm(d2))
        p = np.asarray(p_local, dtype=np.float64)
        return p @ d1, p @ d2

    def uv_for(self, uv, p_local):
        """The (u, v) this material is addressed at, per ``texture_space``."""
        if self.texture_space == "object" and p_local is not None:
            return self._object_uv(p_local)
        return uv

    @staticmethod
    def _noise_scalar(u: float, v: float, freq: float) -> float:
        x = u * freq
        y = v * freq
        i = int(math.floor(x))
        j = int(math.floor(y))
        tx = x - i
        ty = y - j
        sx = tx * tx * (3.0 - 2.0 * tx)
        sy = ty * ty * (3.0 - 2.0 * ty)

        def h(a: int, b: int) -> float:
            n = (a * 374761393 + b * 668265263) & 0xFFFFFFFF
            n = ((n ^ (n >> 13)) * 1274126177) & 0xFFFFFFFF
            return ((n ^ (n >> 16)) & 0xFFFF) / 65535.0

        v0 = (h(i, j) * (1.0 - sx) + h(i + 1, j) * sx)
        v1 = (h(i, j + 1) * (1.0 - sx) + h(i + 1, j + 1) * sx)
        return v0 * (1.0 - sy) + v1 * sy

    def texture_value(self, u: float, v: float) -> float:
        if self.texture == "none":
            return 1.0
        if self.texture == "stripes":
            s = 0.5 + 0.5 * math.cos(2.0 * math.pi * self.texture_freq * u)
        elif self.texture == "checker":
            s = float((math.floor(self.texture_freq * u)
                       + math.floor(self.texture_freq * v)) % 2)
        elif self.texture == "noise":
            s = self._noise_scalar(u, v, self.texture_freq)
        else:
            raise ValueError("unknown texture %r" % (self.texture,))
        return (1.0 - self.texture_contrast) + self.texture_contrast * s

    def texture_value_vec(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        if self.texture == "none":
            return np.ones_like(u, dtype=np.float64)
        if self.texture == "stripes":
            s = 0.5 + 0.5 * np.cos(2.0 * math.pi * self.texture_freq * u)
        elif self.texture == "checker":
            s = ((np.floor(self.texture_freq * u)
                  + np.floor(self.texture_freq * v)) % 2.0)
        elif self.texture == "noise":
            x = u * self.texture_freq
            y = v * self.texture_freq
            fi = np.floor(x)
            fj = np.floor(y)
            tx = x - fi
            ty = y - fj
            sx = tx * tx * (3.0 - 2.0 * tx)
            sy = ty * ty * (3.0 - 2.0 * ty)
            i = fi.astype(np.int64)
            j = fj.astype(np.int64)

            def h(a: np.ndarray, b: np.ndarray) -> np.ndarray:
                n = ((a * 374761393 + b * 668265263) & 0xFFFFFFFF).astype(
                    np.int64)
                n = ((n ^ (n >> 13)) * 1274126177) & 0xFFFFFFFF
                return ((n ^ (n >> 16)) & 0xFFFF) / 65535.0

            v0 = h(i, j) * (1.0 - sx) + h(i + 1, j) * sx
            v1 = h(i, j + 1) * (1.0 - sx) + h(i + 1, j + 1) * sx
            s = v0 * (1.0 - sy) + v1 * sy
        else:
            raise ValueError("unknown texture %r" % (self.texture,))
        return (1.0 - self.texture_contrast) + self.texture_contrast * s


# ---------------------------------------------------------------------------
# camera
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Camera:
    """Pinhole camera at ``(0, 0, camera_z)`` looking down ``-z``.

    ``focal_px`` is the focal length in PIXELS, so a world point at axial depth
    ``e`` projects to ``(0.5 W + f X / e, 0.5 H + f Y / e)``.  ``Scene.camera
    = None`` (the default) is the orthographic camera, which is exactly the
    ``e -> inf`` limit with ``f = e / pixel_scale``: a point maps to
    ``(0.5 W + X / pixel_scale, ...)``.

    Only the focal length is exposed.  A principal-point offset would add a
    second degree of freedom without adding anything to the questions this
    renderer answers, and every extra parameter is another thing to get wrong.
    """

    focal_px: float = 800.0

    def __post_init__(self):
        if not (self.focal_px > 0.0):
            raise ValueError("camera focal_px must be positive")


# ---------------------------------------------------------------------------
# analytic primitives (local frame: centre at origin, z is the object's axis)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Sphere:
    radius: float = 0.5

    def intersect_local(self, o: np.ndarray, d: np.ndarray
                        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        n_pix = o.shape[0]
        dd = _as_dir(d, n_pix)
        # The quadratic is a t^2 + b t + c with a = |d|^2 = 1, which the
        # renderer guarantees for every ray it builds.
        b = 2.0 * (o[:, 0] * dd[:, 0] + o[:, 1] * dd[:, 1]
                   + o[:, 2] * dd[:, 2])
        c = np.einsum("ij,ij->i", o, o) - self.radius ** 2
        disc = b * b - 4.0 * c
        valid = disc >= 0.0
        sq = np.sqrt(np.maximum(disc, 0.0))
        t = (-b - sq) / 2.0
        valid &= t > _EPS
        t = np.where(valid, t, 0.0)
        p = o + t[:, None] * dd
        n = p / self.radius
        # METRIC iso-parametric coordinates: one uv unit is one world unit of
        # arc length on the equator / meridian, so `texture_freq` is cells per
        # world unit and the texture is isotropic in the tangent plane at the
        # equator.  (A sphere has no distortion-free projection; near the poles
        # the u direction still compresses as cos(latitude), which is stated,
        # not hidden.)
        u = np.arctan2(p[:, 1], p[:, 0]) * self.radius
        v = np.arccos(np.clip(p[:, 2] / self.radius, -1.0, 1.0)) * self.radius
        uv = np.stack([u, v], axis=1)
        return t, n, uv, valid

    def normal_at_local(self, p: np.ndarray) -> Vec3:
        return _unit(p)


@dataclass(frozen=True)
class Cylinder:
    radius: float = 0.4
    half_height: float = 0.7

    def intersect_local(self, o: np.ndarray, d: np.ndarray
                        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        n_pix = o.shape[0]
        dd = _as_dir(d, n_pix)
        dx, dy, dz = dd[:, 0], dd[:, 1], dd[:, 2]
        # Infinite lateral surface.
        a = dx * dx + dy * dy
        b = 2.0 * (o[:, 0] * dx + o[:, 1] * dy)
        c = o[:, 0] ** 2 + o[:, 1] ** 2 - self.radius ** 2
        disc = b * b - 4.0 * a * c
        t_side = np.full(n_pix, np.inf)
        ok = (np.abs(a) > _EPS) & (disc >= 0.0)
        sq = np.sqrt(np.maximum(disc, 0.0))
        safe_a = np.where(np.abs(a) > _EPS, a, 1.0)
        t0 = (-b - sq) / (2.0 * safe_a)
        z0 = o[:, 2] + t0 * dz
        ok &= (t0 > _EPS) & (np.abs(z0) <= self.half_height)
        t_side[ok] = t0[ok]

        # Lateral normal at the HIT point (not at the ray origin: those differ
        # under perspective, where rays are not parallel to the axis).
        p_side = o + t0[:, None] * dd
        n_side = p_side.copy()
        n_side[:, 2] = 0.0
        radial = np.maximum(np.hypot(p_side[:, 0], p_side[:, 1]), _EPS)
        n_side[:, 0] /= radial
        n_side[:, 1] /= radial

        # End caps.
        t_cap = np.full(n_pix, np.inf)
        n_cap = np.zeros((n_pix, 3), dtype=np.float64)
        cap_ok = np.abs(dz) > _EPS
        safe_dz = np.where(cap_ok, dz, 1.0)
        for sign in (1.0, -1.0):
            tc = (sign * self.half_height - o[:, 2]) / safe_dz
            x = o[:, 0] + tc * dx
            y = o[:, 1] + tc * dy
            good = cap_ok & (tc > _EPS) & (x * x + y * y <= self.radius ** 2)
            better = good & (tc < t_cap)
            t_cap[better] = tc[better]
            n_cap[better] = np.array([0.0, 0.0, sign], dtype=np.float64)

        use_side = t_side <= t_cap
        t = np.where(use_side, t_side, t_cap)
        n = np.where(use_side[:, None], n_side, n_cap)
        valid = np.isfinite(t) & (t > _EPS)
        t = np.where(valid, t, 0.0)
        p = o + t[:, None] * dd
        # METRIC coordinates (see Sphere): arc length around the axis and
        # height, both in world units.
        u = np.arctan2(p[:, 1], p[:, 0]) * self.radius
        v = p[:, 2]
        uv = np.stack([u, v], axis=1)
        return t, n, uv, valid


@dataclass(frozen=True)
class Box:
    half_extents: Tuple[float, float, float] = (0.6, 0.5, 0.4)

    def intersect_local(self, o: np.ndarray, d: np.ndarray
                        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        h = np.asarray(self.half_extents, dtype=np.float64)
        n_pix = o.shape[0]
        dd = _as_dir(d, n_pix)
        t_near = np.full(n_pix, -np.inf)
        t_far = np.full(n_pix, np.inf)
        axis_near = np.full(n_pix, -1, dtype=np.int64)
        axis_far = np.full(n_pix, -1, dtype=np.int64)
        for k in range(3):
            dk = dd[:, k]
            ok = o[:, k]
            parallel = np.abs(dk) < _EPS
            inside = (ok >= -h[k]) & (ok <= h[k])
            safe_dk = np.where(parallel, 1.0, dk)
            t1 = (-h[k] - ok) / safe_dk
            t2 = (h[k] - ok) / safe_dk
            lo = np.where(parallel, np.where(inside, -np.inf, np.inf),
                          np.minimum(t1, t2))
            hi = np.where(parallel, np.where(inside, np.inf, -np.inf),
                          np.maximum(t1, t2))
            take_lo = lo > t_near
            t_near = np.where(take_lo, lo, t_near)
            axis_near = np.where(take_lo, k, axis_near)
            take_hi = hi < t_far
            t_far = np.where(take_hi, hi, t_far)
            axis_far = np.where(take_hi, k, axis_far)

        hit_in = (t_near <= t_far) & (t_far > _EPS)
        use_near = t_near > _EPS
        t = np.where(use_near, t_near, t_far)
        axis = np.where(use_near, axis_near, axis_far)
        valid = hit_in & np.isfinite(t) & (t > _EPS)

        n = np.zeros((n_pix, 3), dtype=np.float64)
        for k in range(3):
            sel = valid & (axis == k)
            if not np.any(sel):
                continue
            p = o[sel] + t[sel, None] * dd[sel]
            sign = np.sign(p[:, k])
            sign[sign == 0.0] = 1.0
            n[sel, k] = sign

        t = np.where(valid, t, 0.0)
        p = o + t[:, None] * dd
        # Put the texture on the face the ray entered: use the two axes that
        # are not the entry axis, in WORLD units on BOTH axes.  The previous
        # version divided each axis by its own half-extent, which stretched
        # the texture by the box's aspect ratio and made an "isotropic" noise
        # texture anisotropic by up to arccos(min(h)/max(h)) ~ 44 deg -- a
        # fronto-parallel bias that belonged to the parametrisation, not to
        # the cue.  One uv unit is now one world unit of surface length.
        uv = np.zeros((n_pix, 2), dtype=np.float64)
        for k in range(3):
            sel = valid & (axis == k)
            if not np.any(sel):
                continue
            a, b = [q for q in range(3) if q != k]
            uv[sel, 0] = p[sel, a]
            uv[sel, 1] = p[sel, b]
        return t, n, uv, valid


def _gaussian_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    """Separable Gaussian blur, edge-replicated, deterministic.

    The kernel is sampled on integer offsets out to ``ceil(3 sigma)`` and
    normalised, so the result depends only on ``(img, sigma)``: no random
    support, no FFT.  ``sigma <= 0`` returns the input unchanged (identity).
    """
    if sigma <= 0.0:
        return img
    radius = max(1, int(math.ceil(3.0 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    k /= k.sum()
    rows, cols = img.shape
    pad = np.pad(img, radius, mode="edge")
    tmp = np.zeros_like(img)
    for i, w in enumerate(k):
        tmp += w * pad[radius:radius + rows, i:i + cols]
    pad = np.pad(tmp, ((radius, radius), (0, 0)), mode="edge")
    out = np.zeros_like(img)
    for i, w in enumerate(k):
        out += w * pad[i:i + rows, :]
    return out


def _defocus_blur(img: np.ndarray, sigma_px: np.ndarray,
                  max_sigma: float = 8.0) -> np.ndarray:
    """Depth-varying Gaussian splat (thin-lens defocus), in one pass.

    Each output pixel gathers with ITS OWN sigma, so a blurred region is not
    sharpened by a neighbouring sharp one.  Two honest limitations are stated
    rather than hidden:

    * the blur MAGNITUDE is the thin-lens CoC, but the occlusion-aware
      weighting of a real lens is not modelled, so an in-focus edge can pick up
      a little radiance from a blurred surface behind it;
    * the background (depth = infinity) is a constant, and blurring a constant
      is the identity, so the caller passes sigma 0 there.  Without that the
      wide background kernel would smear the foreground edge outward as a glow
      that no lens produces.

    ``max_sigma`` caps the kernel radius for cost only.  ``sigma_px`` is H x W
    (or broadcastable).
    """
    s = np.asarray(sigma_px, dtype=np.float64)
    if s.ndim == 0:
        s = np.full(img.shape, float(s), dtype=np.float64)
    s = np.minimum(np.maximum(s, 0.0), float(max_sigma))
    if float(s.max()) <= 1e-6:
        return img
    H, W = img.shape
    radius = max(1, int(math.ceil(3.0 * float(s.max()))))
    num = np.zeros_like(img, dtype=np.float64)
    den = np.zeros_like(img, dtype=np.float64)
    s2 = 2.0 * np.maximum(s, 1e-6) ** 2
    padded = np.pad(img, radius, mode="edge")
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            g = (dx * dx + dy * dy) / s2
            # exp(-r^2 / 2 sigma^2); the 3-sigma cut keeps the kernel finite.
            w = np.where(g < 9.0, np.exp(-g), 0.0)
            src = padded[radius + dy:radius + dy + H,
                         radius + dx:radius + dx + W]
            num += w * src
            den += w
    return num / np.maximum(den, 1e-12)


# ---------------------------------------------------------------------------
# scene objects
# ---------------------------------------------------------------------------

@dataclass
class SceneObject:
    """One rigid object in the scene.

    ``rotation`` maps local to world; ``centre`` is the world position of the
    local origin (also the centre for every primitive here).  ``omega`` is the
    world angular velocity in rad/s and ``velocity`` the world linear velocity
    of the centre.  ``layer`` is the scene's ordinal depth layer (0 = front);
    it is truth, used for depth-order evaluation only.
    """

    name: str
    shape: Any
    rotation: Mat3 = field(default_factory=lambda: np.eye(3, dtype=np.float64))
    centre: Vec3 = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    material: Material = field(default_factory=Material)
    omega: Vec3 = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    velocity: Vec3 = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    layer: int = 0
    # Known-size class name (a key of KNOWN_SIZE_RADII).  TRUTH: it states the
    # physical radius the renderer used, so relative size can be scored.  The
    # perception path never reads it; an empty name means "size unknown".
    size_class: str = ""

    @property
    def known_radius(self) -> Optional[float]:
        if not self.size_class:
            return None
        return size_class_radius(self.size_class)

    def world_to_local(self, p: np.ndarray) -> np.ndarray:
        return (p - self.centre) @ self.rotation

    def local_to_world(self, p: np.ndarray) -> np.ndarray:
        return p @ self.rotation.T + self.centre

    def velocity_at(self, p_world: np.ndarray) -> np.ndarray:
        r = p_world - self.centre
        return self.velocity + np.cross(self.omega, r)


@dataclass
class Hit:
    object_index: int
    object_name: str
    layer: int
    t: float
    point_world: Vec3
    normal_world: Vec3
    uv: Tuple[float, float]
    point_local: Vec3
    # Axial distance from the camera plane, ``camera_z - z``.  Equal to ``t``
    # under orthographic projection, smaller than it under perspective.  This
    # is the depth that ``RenderTruth.depth`` reports and that occlusion
    # ordering compares.
    depth: float = 0.0


@dataclass(frozen=True)
class FragmentTruth:
    """Truth at one image point, used only by the evaluator.

    ``slant`` and ``tilt`` are in DEGREES (tilt is the azimuth of the normal in
    the image plane, ``atan2(ny, nx)``).  ``motion`` is px/s.
    """

    object_index: int
    object_name: str
    layer: int
    depth: float
    point_world: Vec3
    normal_world: Vec3
    slant: float
    tilt: float
    motion: Tuple[float, float]
    edge_kind: str
    curvature_sign: float

@dataclass
class RenderTruth:
    image: np.ndarray          # H x W x 3 float32, luminance replicated
    luminance: np.ndarray      # H x W float32
    image_u8: np.ndarray       # H x W x 3 uint8, the frame a sensor would give
    depth: np.ndarray          # H x W float64, inf on background
    normal: np.ndarray         # H x W x 3 float64, nan on background
    object_id: np.ndarray      # H x W int32, -1 on background
    motion: np.ndarray         # H x W x 2 float64, image velocity
    layer: np.ndarray          # H x W int32, -1 on background
    mask: np.ndarray           # H x W bool
    light: Vec3
    camera_z: float
    width: int
    height: int
    frame: int = 0
    dt: float = 0.0
    object_names: Tuple[str, ...] = ()
    # Known-size truth: the class name per object and its table radius (nan
    # when the scene did not declare one).
    object_size_class: Tuple[str, ...] = ()
    object_radius: Tuple[float, ...] = ()


# ---------------------------------------------------------------------------
# scene
# ---------------------------------------------------------------------------

@dataclass
class Scene:
    objects: List[SceneObject]
    light: Vec3 = field(default_factory=lambda: _unit((-0.4, -0.5, 0.75)))
    background: float = 0.02
    ambient: float = 0.0
    camera_z: float = 10.0
    # World units per pixel.  The default maps a 256x256 frame to the world
    # square [-1, 1]^2, so a 0.35-radius sphere is ~45 px in radius.
    pixel_scale: float = 1.0 / 128.0
    width: int = 256
    height: int = 256
    # Optional pinhole camera; None is the orthographic default.
    camera: Optional[Camera] = None
    # Additional directional lights: ((direction, weight), ...).  ``light`` is
    # the primary one with weight 1.  Intensity is the sum of the weighted
    # Lambert terms; ``ambient`` is added once.
    extra_lights: Tuple[Tuple[Sequence[float], float], ...] = ()
    # Sensor stages, both off by default: a fixed isotropic Gaussian blur and
    # seeded additive Gaussian noise on the luminance image.
    blur_sigma: float = 0.0
    noise_sigma: float = 0.0
    noise_seed: int = 0

    # -- optics, all off by default ----------------------------------------
    #
    # Thin-lens defocus.  ``focus_depth`` is the axial object distance the lens
    # is focused at, ``focal_length`` the lens focal length (world units) and
    # ``f_number`` = focal_length / aperture diameter.  A point at depth z
    # images to a circle of confusion
    #
    #     c = A f |z - zf| / (z (zf - f)),      A = f / N
    #
    # on the image plane; it is converted to pixels by focal_px / f, i.e. the
    # same scale the pin-hole projection uses.  ``c`` is a DIAMETER; the
    # Gaussian splat uses sigma = c / 2.  focus_depth <= 0 switches it off and
    # the render is byte-identical to the pre-defocus renderer.
    focus_depth: float = 0.0
    focal_length: float = 0.05
    f_number: float = 1.0

    # Distance haze (atmospheric scattering), a radiance model, not a fudge:
    #     L' = L exp(-beta z) + airlight (1 - exp(-beta z))
    # with z the axial depth (infinity on the background, so the background
    # becomes the airlight).  beta <= 0 switches it off (identity).
    haze_beta: float = 0.0
    haze_airlight: float = 0.0

    def __post_init__(self) -> None:
        if self.focus_depth > 0.0:
            if not (self.focal_length > 0.0):
                raise ValueError("focal_length must be positive for defocus")
            if not (self.f_number > 0.0):
                raise ValueError("f_number must be positive for defocus")
            if not (self.focus_depth > self.focal_length):
                raise ValueError("focus_depth must exceed focal_length")
        if self.haze_beta < 0.0:
            raise ValueError("haze_beta must be non-negative")

    # -- optical geometry --------------------------------------------------

    @property
    def perspective(self) -> bool:
        return self.camera is not None

    @property
    def focal_px(self) -> float:
        """Focal length in PIXELS, for both camera models.

        Orthographic is the ``e -> inf`` limit with ``f = e / pixel_scale``, so
        the same number converts a world length at depth ``e`` to pixels under
        either model.  Defocus and size both use this one scale.
        """
        if self.camera is not None:
            return float(self.camera.focal_px)
        return float(self.camera_z / self.pixel_scale)

    def circle_of_confusion_px(self, depth: Any) -> np.ndarray:
        """Thin-lens circle of confusion (DIAMETER) in pixels, elementwise.

        ``depth`` is axial distance (``inf`` on the background).  Zero wherever
        defocus is off, so callers never special-case it.
        """
        d = np.asarray(depth, dtype=np.float64)
        if self.focus_depth <= 0.0:
            return np.zeros(d.shape, dtype=np.float64)
        f = float(self.focal_length)
        n = float(self.f_number)
        zf = float(self.focus_depth)
        safe = np.where(np.isfinite(d), np.maximum(d, _EPS), zf)
        rel = np.where(np.isfinite(d), np.abs(safe - zf) / safe, 1.0)
        c_world = (f / n) * f * rel / (zf - f)
        return c_world * (self.focal_px / f)

    def defocus_sigma_px(self, depth: Any) -> np.ndarray:
        """Gaussian splat width in pixels: half the CoC diameter."""
        return 0.5 * self.circle_of_confusion_px(depth)

    def pixel_to_world(self, x: float, y: float) -> Tuple[float, float]:
        """World point under the orthographic ray at pixel ``(x, y)``.

        Under perspective there is no single world point per pixel; use
        ``ray``.
        """
        return ((float(x) - 0.5 * self.width) * self.pixel_scale,
                (float(y) - 0.5 * self.height) * self.pixel_scale)

    def ray(self, x: float, y: float
            ) -> Tuple[np.ndarray, np.ndarray, float]:
        """(origin, unit direction, t-to-depth factor) at pixel ``(x, y)``.

        ``depth = t * factor`` because with a unit direction the ray parameter
        is Euclidean distance, while the z-buffer wants the axial distance.
        """
        if self.camera is None:
            wx, wy = self.pixel_to_world(x, y)
            return (np.array([wx, wy, self.camera_z], dtype=np.float64),
                    np.array([0.0, 0.0, -1.0], dtype=np.float64),
                    1.0)
        f = float(self.camera.focal_px)
        a = (float(x) - 0.5 * self.width) / f
        b = (float(y) - 0.5 * self.height) / f
        n = math.sqrt(1.0 + a * a + b * b)
        return (np.array([0.0, 0.0, self.camera_z], dtype=np.float64),
                np.array([a / n, b / n, -1.0 / n], dtype=np.float64),
                1.0 / n)

    def _rays(self, xs: np.ndarray, ys: np.ndarray
              ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Vectorised ``ray`` over pixel coordinates ``xs``, ``ys``."""
        x = np.asarray(xs, dtype=np.float64).ravel()
        y = np.asarray(ys, dtype=np.float64).ravel()
        n_pix = x.size
        if self.camera is None:
            wx = (x - 0.5 * self.width) * self.pixel_scale
            wy = (y - 0.5 * self.height) * self.pixel_scale
            O = np.stack([wx, wy,
                          np.full(n_pix, self.camera_z)], axis=1)
            D = np.zeros((n_pix, 3), dtype=np.float64)
            D[:, 2] = -1.0
            return O, D, np.ones(n_pix)
        f = float(self.camera.focal_px)
        a = (x - 0.5 * self.width) / f
        b = (y - 0.5 * self.height) / f
        n = np.sqrt(1.0 + a * a + b * b)
        O = np.zeros((n_pix, 3), dtype=np.float64)
        O[:, 2] = self.camera_z
        D = np.stack([a / n, b / n, -1.0 / n], axis=1)
        return O, D, 1.0 / n

    def _view_dir(self, p_world: np.ndarray) -> np.ndarray:
        """Unit direction from a surface point toward the camera."""
        if self.camera is None:
            return np.array([0.0, 0.0, 1.0], dtype=np.float64)
        v = np.array([0.0, 0.0, self.camera_z], dtype=np.float64) - p_world
        return _unit(v)

    def _lambert(self, normal: np.ndarray) -> float:
        """Sum of the weighted Lambert terms (no ambient, no albedo)."""
        total = max(0.0, float(normal @ self.light))
        for direction, weight in self.extra_lights:
            total += float(weight) * max(0.0, float(normal @ _unit(direction)))
        return total

    # -- ray casting -------------------------------------------------------

    def raycast(self, x: float, y: float) -> Optional[Hit]:
        """Nearest visible hit at image coordinates ``(x, y)`` in pixels."""
        O, D, to_depth = self.ray(x, y)
        best: Optional[Hit] = None
        for i, obj in enumerate(self.objects):
            o = (O - obj.centre) @ obj.rotation
            d = D @ obj.rotation
            t, n, uv, valid = obj.shape.intersect_local(
                o[None, :].copy(), np.asarray(d, dtype=np.float64)[None, :])
            if not bool(valid[0]):
                continue
            ti = float(t[0])
            if ti <= _EPS:
                continue
            if best is not None:
                # Deterministic tie-break: lower object index wins.
                if ti > best.t + 1e-12:
                    continue
                if abs(ti - best.t) <= 1e-12 and i >= best.object_index:
                    continue
            p_local = o + ti * d
            p_world = p_local @ obj.rotation.T + obj.centre
            n_world = _unit(n[0] @ obj.rotation.T)
            if float(n_world @ self._view_dir(p_world)) < 0.0:
                # A visible surface faces the viewer; keep the convention.
                n_world = -n_world
            best = Hit(
                object_index=i,
                object_name=obj.name,
                layer=int(obj.layer),
                t=ti,
                point_world=p_world,
                normal_world=n_world,
                uv=(float(uv[0, 0]), float(uv[0, 1])),
                point_local=p_local,
                depth=float(ti * to_depth),
            )
        return best

    def intensity_at(self, hit: Optional[Hit], x: float = 0.0,
                     y: float = 0.0) -> float:
        if hit is None:
            return float(self.background)
        obj = self.objects[hit.object_index]
        mat = obj.material
        albedo = mat.albedo * mat.texture_value(*hit.uv)
        return float(self.ambient + albedo * self._lambert(hit.normal_world))

    # -- truth at a point --------------------------------------------------

    def _edge_kind(self, x: float, y: float, hit: Hit,
                   radius: float = 2.0) -> str:
        """Classify the local edge as truth, for stratified reporting only.

        * ``occlusion_boundary``: another object is in front within ``radius``.
        * ``smooth_silhouette``: the point is on the owner's visible boundary
          and the surface normal is (nearly) in the image plane.
        * ``crease_or_depth_edge``: on the owner's visible boundary, but the
          normal has a significant viewer component (a box face edge, a cap
          rim, or any non-silhouette discontinuity).
        * ``interior``: no boundary nearby.
        """
        owner = hit.object_index
        best_other_depth = np.inf
        boundary = False
        # 8 directions, three radii: enough to see a boundary without assuming
        # a segmentation.  Truth only.
        for r in (radius * 0.5, radius, radius * 1.5):
            for k in range(8):
                a = 2.0 * math.pi * k / 8.0
                q = self.raycast(x + r * math.cos(a), y + r * math.sin(a))
                if q is None:
                    if r >= radius:
                        boundary = True
                    continue
                if q.object_index != owner:
                    if r >= radius:
                        boundary = True
                    if q.depth < hit.depth - 0.05:
                        best_other_depth = min(best_other_depth, q.depth)
                elif abs(q.depth - hit.depth) > 0.05 and r >= radius:
                    # Same object, a depth jump: a crease (box face edge,
                    # cylinder rim), not a smooth silhouette.
                    boundary = True
        if np.isfinite(best_other_depth):
            return "occlusion_boundary"
        if not boundary:
            return "interior"
        # A smooth silhouette is where the normal is perpendicular to the
        # viewing direction: nz == 0 orthographically, n.view == 0 in general.
        view = self._view_dir(hit.point_world)
        if abs(float(hit.normal_world @ view)) <= 0.20:
            return "smooth_silhouette"
        return "crease_or_depth_edge"

    def _mask_curvature_sign(self, x: float, y: float, owner: int,
                             radius: int = 12) -> float:
        """Sign of the owner's visible-boundary curvature, from truth mask.

        A circle is fitted to the owner's mask boundary in a local window.
        The centre of curvature lying INSIDE the owner is convex (+1); outside
        is concave (-1); a good straight fit is 0.  This is the same
        owner-relative convention the estimator uses, so the comparison is
        meaningful.
        """
        W = int(radius)
        xi, yi = int(round(x)), int(round(y))
        x0, x1 = max(0, xi - W), min(self.width, xi + W + 1)
        y0, y1 = max(0, yi - W), min(self.height, yi + W + 1)
        if x1 - x0 < 4 or y1 - y0 < 4:
            return 0.0
        # Boundary = mask pixel with a 4-neighbour outside the mask.  This is
        # computed on the FULL mask, not on the window: a window border is not
        # an object boundary, and using it labelled every interior point of a
        # large flat face as convex (a square window fits a circle).
        full = (self.object_id == owner)
        if not np.any(full[y0:y1, x0:x1]):
            return 0.0
        pad = np.pad(full, 1, mode="constant", constant_values=False)
        inner = pad[1:-1, 1:-1]
        nb = (pad[:-2, 1:-1] & pad[2:, 1:-1]
              & pad[1:-1, :-2] & pad[1:-1, 2:])
        boundary = (inner & ~nb)[y0:y1, x0:x1]
        ys, xs = np.nonzero(boundary)
        if xs.size < 6:
            return 0.0
        px = xs.astype(np.float64) + x0 - x
        py = ys.astype(np.float64) + y0 - y
        # Algebraic circle fit: x^2 + y^2 + D x + E y + F = 0.
        A = np.stack([px, py, np.ones_like(px)], axis=1)
        b = -(px * px + py * py)
        try:
            sol, *_ = np.linalg.lstsq(A, b, rcond=None)
        except np.linalg.LinAlgError:
            return 0.0
        D, E, F = float(sol[0]), float(sol[1]), float(sol[2])
        cx, cy = -D / 2.0, -E / 2.0
        r2 = cx * cx + cy * cy - F
        if not np.isfinite(r2) or r2 <= 0.0:
            return 0.0
        r = math.sqrt(r2)
        # A fit whose radius dwarfs the window is a straight line.
        if r > 8.0 * radius:
            return 0.0
        # Is the centre of curvature inside the owner mask?
        ix, iy = int(round(cx + x)), int(round(cy + y))
        if not (0 <= ix < self.width and 0 <= iy < self.height):
            return 0.0
        return 1.0 if int(self.object_id[iy, ix]) == owner else -1.0

    def _bisect_hit(self, x: float, y: float, dx: float, dy: float,
                    limit: float = 3.0, tol: float = 2e-4):
        """First surface hit along a ray that starts outside, by bisection."""
        hi = None
        s = 0.05
        while s <= limit + 1e-9:
            if self.raycast(x + dx * s, y + dy * s) is not None:
                hi = s
                break
            s += 0.05
        if hi is None:
            return None
        lo = max(0.0, hi - 0.05)
        while hi - lo > tol:
            mid = 0.5 * (lo + hi)
            if self.raycast(x + dx * mid, y + dy * mid) is not None:
                hi = mid
            else:
                lo = mid
        return self.raycast(x + dx * hi, y + dy * hi)

    def silhouette_inset(self, x: float, y: float,
                         direction: Sequence[float],
                         limit: float = 8.0, tol: float = 2e-3
                         ) -> Optional[float]:
        """Signed distance from ``(x, y)`` to the silhouette along ``direction``.

        Positive when the point is INSIDE the object (moving along
        ``direction`` leaves it).  ``direction`` is the image-plane part of the
        surface normal.  Along that direction a sphere's normal falls as
        ``nz = sqrt(1 - (rho / R)^2)``, so this distance IS the anchor inset
        ``d`` that the limb law ``slant = 90 - sqrt(2 d / R)`` needs.
        """
        dx, dy = float(direction[0]), float(direction[1])
        m = math.hypot(dx, dy)
        if m < 1e-12:
            return None
        dx, dy = dx / m, dy / m
        inside = self.raycast(x, y) is not None
        step = -1.0 if inside else 1.0
        lo, hi = 0.0, None
        s = 0.05
        while s <= limit:
            hit = self.raycast(x + step * dx * s, y + step * dy * s)
            if (hit is None) if inside else (hit is not None):
                hi = s
                break
            s += 0.05
        if hi is None:
            return None
        lo = hi - 0.05
        while hi - lo > tol:
            mid = 0.5 * (lo + hi)
            hit = self.raycast(x + step * dx * mid, y + step * dy * mid)
            if (hit is None) if inside else (hit is not None):
                hi = mid
            else:
                lo = mid
        d = 0.5 * (lo + hi)
        return d if inside else -d

    def truth_at(self, x: float, y: float,
                 inward: Optional[Sequence[float]] = None
                 ) -> Optional[FragmentTruth]:
        """Truth at ``(x, y)``; if the point misses, step inward along ``inward``.

        ``inward`` is the owner-side image normal.  A contour peak sits on or
        just outside the boundary, where a ray cast misses by a subpixel; the
        honest truth for such an anchor is the NEAREST surface point, which for
        a limb is the limb itself.  The search is therefore a continuous
        bisection to the boundary, not a ladder of 0.5 px steps: a ladder lands
        0.5 px inside the limb and reports a slant deficit that belongs to the
        ladder, not to the anchor (Session A's measured 6-15 deg bias was
        partly this artefact; the rest is the real anchor offset).
        """
        dx = float(inward[0]) if inward is not None else 0.0
        dy = float(inward[1]) if inward is not None else 0.0
        hit = self.raycast(x, y)
        if hit is None and (abs(dx) + abs(dy)) > 0.0:
            hit = self._bisect_hit(x, y, dx, dy, limit=3.0)
        if hit is None:
            return None
        nx, ny, nz = (float(hit.normal_world[0]),
                      float(hit.normal_world[1]),
                      float(hit.normal_world[2]))
        view = self._view_dir(hit.point_world)
        cos_view = max(0.0, min(1.0, float(hit.normal_world @ view)))
        # Angles in DEGREES, like every other angle in the representation and
        # in the calibration report.  (Radians here were a silent trap: the
        # harness recomputes its own reference, so nothing depended on it, but
        # a truth field that reports 1.57 for a limb is a contract bug.)
        #
        # The reference is the viewing direction, not the image z axis: under
        # perspective the two differ, and the occluding contour is where the
        # normal is perpendicular to the VIEW ray.
        slant = math.degrees(math.acos(cos_view))
        tilt = (math.degrees(math.atan2(ny, nx))
                if (abs(nx) + abs(ny)) > 1e-12 else 0.0)
        v = self.objects[hit.object_index].velocity_at(hit.point_world)
        if self.camera is None:
            # Orthographic: image velocity is the (x, y) part of the world
            # velocity, in px/s.
            motion = (float(v[0] / self.pixel_scale),
                      float(v[1] / self.pixel_scale))
        else:
            # Pinhole, camera at (0, 0, camera_z), axial depth e:
            #   u = 0.5 W + f X / e,  du/dt = f (Vx e + X Vz) / e^2
            # because de/dt = -Vz.
            e = float(hit.depth)
            f = float(self.camera.focal_px)
            motion = (float(f * (v[0] * e + hit.point_world[0] * v[2])
                           / (e * e)),
                      float(f * (v[1] * e + hit.point_world[1] * v[2])
                           / (e * e)))
        edge_kind = self._edge_kind(x, y, hit)
        curvature = self._mask_curvature_sign(x, y, hit.object_index)
        return FragmentTruth(
            object_index=hit.object_index,
            object_name=hit.object_name,
            layer=int(hit.layer),
            depth=float(hit.depth),
            point_world=hit.point_world,
            normal_world=hit.normal_world,
            slant=float(slant),
            tilt=float(tilt),
            motion=motion,
            edge_kind=edge_kind,
            curvature_sign=float(curvature),
        )

    # -- rendering ---------------------------------------------------------

    def _apply_haze(self, lum: np.ndarray, depth: np.ndarray) -> np.ndarray:
        """Atmospheric scattering on radiance, with the background at infinity."""
        beta = float(self.haze_beta)
        if beta <= 0.0:
            return lum
        d = np.asarray(depth, dtype=np.float64)
        d = np.where(np.isfinite(d), d, np.inf)
        t = np.exp(-beta * d)          # exp(-inf) == 0 on the background
        return lum * t + float(self.haze_airlight) * (1.0 - t)

    def _shade(self, ids: np.ndarray, Os: np.ndarray, Ds: np.ndarray,
               ps: np.ndarray) -> np.ndarray:
        """Luminance for one vector of subpixel rays with known object ids."""
        n_pix = ids.shape[0]
        inten = np.full(n_pix, float(self.background), dtype=np.float64)
        for i, obj in enumerate(self.objects):
            sel = ids == i
            if not np.any(sel):
                continue
            o = (Os[sel] - obj.centre) @ obj.rotation
            d = Ds[sel] @ obj.rotation
            t, n, uv, valid = obj.shape.intersect_local(o, d)
            nw = n @ obj.rotation.T
            p = ps[sel]
            if self.camera is None:
                view = np.zeros_like(p)
                view[:, 2] = 1.0
            else:
                view = np.array([0.0, 0.0, self.camera_z])[None, :] - p
            facing = np.einsum("ij,ij->i", nw, view) < 0.0
            nw[facing] *= -1.0
            lam = np.clip(nw @ self.light, 0.0, None)
            for direction, weight in self.extra_lights:
                lam = lam + float(weight) * np.clip(
                    nw @ _unit(direction), 0.0, None)
            tex = obj.material.texture_value_vec(uv[:, 0], uv[:, 1])
            inten[sel] = (self.ambient
                          + obj.material.albedo * tex * lam)
        return inten

    def render(self, width: int = 256, height: int = 256,
                supersample: int = 2, frame: int = 0, dt: float = 0.0,
                offsets: Optional[Sequence[Tuple[float, float, float]]] = None,
                ) -> RenderTruth:
        """Render one frame and its exact truth.

        ``offsets`` (optional, per-object in-plane/world translation) shifts each
        object's centre for this frame only; it is applied to both the truth and
        the luminance paths and reverted before return, so the scene object is
        not mutated across calls.

        ``supersample`` is the number of samples per axis for the luminance
        image only.  Truth (depth, normal, id, layer, motion) is evaluated at
        the pixel centre, so it is not antialiased: it is the truth at that
        point.  The image is the average of the subsamples, which is what a
        sensor would integrate, optionally blurred (``blur_sigma``) and
        corrupted with seeded noise (``noise_sigma``, ``noise_seed``).

        The truth fields are unaffected by blur and noise: they are the
        geometry, not the sensor.
        """
        self.width = int(width)
        self.height = int(height)
        H, W = int(height), int(width)
        S = max(1, int(supersample))

        # Per-object translation for this frame (in-place on copies; restored).
        _saved_centres = [np.asarray(o.centre, dtype=np.float64).copy()
                           for o in self.objects]
        if offsets is not None:
            for i, o in enumerate(self.objects):
                if i < len(offsets) and offsets[i] is not None:
                    off = np.asarray(offsets[i], dtype=np.float64).ravel()
                    if off.size >= 3:
                        o.centre = (np.asarray(o.centre, dtype=np.float64)
                                    + off[:3])
                    elif off.size == 2:
                        c = np.asarray(o.centre, dtype=np.float64).copy()
                        c[0] += float(off[0]); c[1] += float(off[1])
                        o.centre = c

        xs = np.arange(W, dtype=np.float64) + 0.5
        ys = np.arange(H, dtype=np.float64) + 0.5
        gx, gy = np.meshgrid(xs, ys)

        # Truth at pixel centres.
        O, D, to_depth = self._rays(gx, gy)

        t_best = np.full(H * W, np.inf, dtype=np.float64)
        n_best = np.zeros((H * W, 3), dtype=np.float64)
        id_best = np.full(H * W, -1, dtype=np.int32)
        uv_best = np.zeros((H * W, 2), dtype=np.float64)
        p_best = np.zeros((H * W, 3), dtype=np.float64)

        for i, obj in enumerate(self.objects):
            o = (O - obj.centre) @ obj.rotation
            d = D @ obj.rotation
            t, n, uv, valid = obj.shape.intersect_local(o, d)
            valid &= t > _EPS
            better = valid & (t < t_best - 1e-12)
            # Deterministic tie-break: lower object index wins.
            tie = valid & (np.abs(t - t_best) <= 1e-12) & (i < id_best)
            take = better | tie
            if not np.any(take):
                continue
            t_best[take] = t[take]
            n_best[take] = n[take]
            id_best[take] = i
            uv_best[take] = uv[take]
            p = (o[take] + t[take, None] * d[take]) @ obj.rotation.T
            p_best[take] = p + obj.centre

        mask = id_best >= 0
        # Axial depth: the ray parameter scaled by t-to-depth (1 orthographic).
        depth = np.where(mask, t_best * to_depth, np.inf)
        normal = np.full((H * W, 3), np.nan, dtype=np.float64)
        motion = np.zeros((H * W, 2), dtype=np.float64)
        layer = np.full(H * W, -1, dtype=np.int32)
        for i, obj in enumerate(self.objects):
            sel = id_best == i
            if not np.any(sel):
                continue
            nw = n_best[sel] @ obj.rotation.T
            p = p_best[sel]
            if self.camera is None:
                view = np.zeros_like(p)
                view[:, 2] = 1.0
            else:
                view = np.array([0.0, 0.0, self.camera_z])[None, :] - p
            facing = np.einsum("ij,ij->i", nw, view) < 0.0
            nw[facing] *= -1.0
            normal[sel] = nw
            v = obj.velocity[None, :] + np.cross(
                np.broadcast_to(obj.omega[None, :], p.shape), p - obj.centre)
            if self.camera is None:
                # Orthographic: the (x, y) part of the world velocity.
                motion[sel] = v[:, :2] / self.pixel_scale
            else:
                # Pinhole: du/dt = f (Vx e + X Vz) / e^2, per axis.
                e = depth[sel]
                f = float(self.camera.focal_px)
                motion[sel, 0] = f * (v[:, 0] * e + p[:, 0] * v[:, 2]) / (e * e)
                motion[sel, 1] = f * (v[:, 1] * e + p[:, 1] * v[:, 2]) / (e * e)
            layer[sel] = int(obj.layer)

        # Luminance image with fixed supersampling.  The subsample offsets are
        # in PIXELS, so the same loop serves both camera models.
        accum = np.zeros(H * W, dtype=np.float64)
        count = 0
        for sy in range(S):
            for sx in range(S):
                ox = (sx + 0.5) / S - 0.5
                oy = (sy + 0.5) / S - 0.5
                Os, Ds, _ = self._rays(gx + ox, gy + oy)
                ts = np.full(H * W, np.inf, dtype=np.float64)
                ids = np.full(H * W, -1, dtype=np.int32)
                ps = np.zeros((H * W, 3), dtype=np.float64)
                for i, obj in enumerate(self.objects):
                    o = (Os - obj.centre) @ obj.rotation
                    d = Ds @ obj.rotation
                    t, n, uv, valid = obj.shape.intersect_local(o, d)
                    valid &= t > _EPS
                    better = valid & (t < ts - 1e-12)
                    tie = valid & (np.abs(t - ts) <= 1e-12) & (i < ids)
                    take = better | tie
                    if not np.any(take):
                        continue
                    ts[take] = t[take]
                    ids[take] = i
                    p = (o[take] + t[take, None] * d[take]) @ obj.rotation.T
                    ps[take] = p + obj.centre
                accum += self._shade(ids, Os, Ds, ps)
                count += 1
        accum /= float(max(1, count))
        lum = accum.reshape(H, W)
        depth_img = depth.reshape(H, W)
        if float(self.haze_beta) > 0.0:
            lum = self._apply_haze(lum, depth_img)
        if float(self.focus_depth) > 0.0:
            sigma = self.defocus_sigma_px(depth_img)
            # The background is a constant, so its (maximal) CoC is a no-op
            # except for the glow it would smear over the foreground edge.
            sigma = np.where(mask.reshape(H, W), sigma, 0.0)
            lum = _defocus_blur(lum, sigma)
        lum = _gaussian_blur(lum, float(self.blur_sigma))
        if float(self.noise_sigma) > 0.0:
            rng = np.random.default_rng(int(self.noise_seed))
            lum = lum + rng.normal(0.0, float(self.noise_sigma), lum.shape)
        lum = lum.astype(np.float32)
        image = np.repeat(lum[..., None], 3, axis=2)
        image_u8 = np.repeat(
            (np.clip(lum, 0.0, 1.0) * 255.0).astype(np.uint8)[..., None],
            3, axis=2)

        # Restore object centres (translation was frame-local).
        for i, o in enumerate(self.objects):
            o.centre = _saved_centres[i]

        self.object_id = id_best.reshape(H, W)
        return RenderTruth(
            image=image,
            luminance=lum,
            image_u8=image_u8,
            depth=depth.reshape(H, W),
            normal=normal.reshape(H, W, 3),
            object_id=id_best.reshape(H, W),
            motion=motion.reshape(H, W, 2),
            layer=layer.reshape(H, W),
            mask=mask.reshape(H, W),
            light=np.asarray(self.light, dtype=np.float64),
            camera_z=float(self.camera_z),
            width=W,
            height=H,
            frame=int(frame),
            dt=float(dt),
            object_names=tuple(o.name for o in self.objects),
            object_size_class=tuple(o.size_class for o in self.objects),
            object_radius=tuple(
                (o.known_radius if o.known_radius is not None
                 else float("nan")) for o in self.objects),
        )


# ---------------------------------------------------------------------------
# scene builders used by the calibration/evaluation harness
# ---------------------------------------------------------------------------

def _obj(name: str, shape: Any, rx: float, ry: float, rz: float,
         centre: Sequence[float], material: Material,
         layer: int = 0,
         omega: Sequence[float] = (0.0, 0.0, 0.0),
         velocity: Sequence[float] = (0.0, 0.0, 0.0)) -> SceneObject:
    return SceneObject(
        name=name,
        shape=shape,
        rotation=rotation_xyz(rx, ry, rz),
        centre=np.asarray(centre, dtype=np.float64),
        material=material,
        omega=np.asarray(omega, dtype=np.float64),
        velocity=np.asarray(velocity, dtype=np.float64),
        layer=int(layer),
    )


def single_object_scene(kind: str, *, width: int = 256, height: int = 256,
                        pose: Tuple[float, float, float] = (0.0, 0.0, 0.0),
                        centre: Sequence[float] = (0.0, 0.0, 0.0),
                        material: Optional[Material] = None,
                        light: Sequence[float] = (-0.4, -0.5, 0.75),
                        omega: Sequence[float] = (0.0, 0.0, 0.0),
                        velocity: Sequence[float] = (0.0, 0.0, 0.0),
                        layer: int = 0,
                        camera: Optional[Camera] = None,
                        camera_z: float = 10.0,
                        extra_lights: Sequence[Tuple[Sequence[float], float]]
                        = (),
                        blur_sigma: float = 0.0,
                        noise_sigma: float = 0.0,
                        noise_seed: int = 0,
                        focus_depth: float = 0.0,
                        focal_length: float = 0.05,
                        f_number: float = 1.0,
                        haze_beta: float = 0.0,
                        haze_airlight: float = 0.0,
                        size_class: str = "") -> Scene:
    """One primitive, at a chosen pose.  Convenience for the harness."""
    mat = material if material is not None else Material()
    if kind == "sphere":
        shape = Sphere(radius=0.35)
    elif kind == "cylinder":
        shape = Cylinder(radius=0.30, half_height=0.55)
    elif kind == "box":
        shape = Box(half_extents=(0.42, 0.34, 0.30))
    else:
        raise ValueError("unknown object kind %r" % (kind,))
    # Keep the object inside the field of view for the canonical 256 frame:
    # the primitives are ~0.3-0.6 world units, and the image spans roughly
    # [-0.5, 0.5] * 2 at unit focal length.  The harness supplies a scale.
    obj = _obj(kind, shape, pose[0], pose[1], pose[2], centre, mat,
               layer=layer, omega=omega, velocity=velocity)
    if size_class:
        obj.size_class = size_class
    return Scene(objects=[obj], light=_unit(light), camera=camera,
                 camera_z=float(camera_z),
                 extra_lights=tuple(extra_lights),
                 blur_sigma=float(blur_sigma),
                 noise_sigma=float(noise_sigma),
                 noise_seed=int(noise_seed),
                 focus_depth=float(focus_depth),
                 focal_length=float(focal_length),
                 f_number=float(f_number),
                 haze_beta=float(haze_beta),
                 haze_airlight=float(haze_airlight))


def occlusion_scene(*, width: int = 256, height: int = 256,
                    front: str = "sphere", back: str = "box",
                    front_centre: Sequence[float] = (-0.10, -0.05, 0.55),
                    back_centre: Sequence[float] = (0.10, 0.05, 0.0),
                    front_pose: Tuple[float, float, float] = (0.0, 0.0, 0.0),
                    back_pose: Tuple[float, float, float] = (0.0, 0.2, 0.1),
                    front_material: Optional[Material] = None,
                    back_material: Optional[Material] = None,
                    light: Sequence[float] = (-0.4, -0.5, 0.75),
                    camera: Optional[Camera] = None,
                    camera_z: float = 10.0,
                    extra_lights: Sequence[Tuple[Sequence[float], float]] = (),
                    blur_sigma: float = 0.0,
                    noise_sigma: float = 0.0,
                    noise_seed: int = 0,
                    focus_depth: float = 0.0,
                    focal_length: float = 0.05,
                    f_number: float = 1.0,
                    haze_beta: float = 0.0,
                    haze_airlight: float = 0.0,
                    front_size_class: str = "",
                    back_size_class: str = "") -> Scene:
    """Two objects with a controlled ordinal layer (front layer 0)."""
    fm = front_material if front_material is not None else Material()
    bm = back_material if back_material is not None else Material()
    shapes = {
        "sphere": lambda: Sphere(radius=0.32),
        "cylinder": lambda: Cylinder(radius=0.28, half_height=0.52),
        "box": lambda: Box(half_extents=(0.38, 0.32, 0.28)),
    }
    if front not in shapes or back not in shapes:
        raise ValueError("unknown object kind")
    fo = _obj(front, shapes[front](), front_pose[0], front_pose[1],
              front_pose[2], front_centre, fm, layer=0)
    bo = _obj(back, shapes[back](), back_pose[0], back_pose[1],
              back_pose[2], back_centre, bm, layer=1)
    if front_size_class:
        fo.size_class = front_size_class
    if back_size_class:
        bo.size_class = back_size_class
    return Scene(objects=[fo, bo], light=_unit(light), camera=camera,
                 camera_z=float(camera_z),
                 extra_lights=tuple(extra_lights),
                 blur_sigma=float(blur_sigma),
                 noise_sigma=float(noise_sigma),
                 noise_seed=int(noise_seed),
                 focus_depth=float(focus_depth),
                 focal_length=float(focal_length),
                 f_number=float(f_number),
                 haze_beta=float(haze_beta),
                 haze_airlight=float(haze_airlight))
