"""perception/chain_flanks.py -- [Slice 2] Per-member flank fields on chains.

W1's identity-carrying lenses (BorderOwnership, Color, Texture, Boundary)
used to be keyed by SEGMENTER region anchors. Slice 0b proved that pathway
is empty on this tree (`entity_indices` is 0 on every image), and Slice 0c
showed the consequence: Surface emitted 8 detections that collapsed to 3
signatures because its material slots were driven by inputs keyed to
anchors that no detection shares.

This module re-keys those descriptors to CHAIN membership instead. For every
chain member it looks at both flanks of the local contour direction:

    offset = PERPENDICULAR_SCALE * median_wavelength      (scale-free)
    left flank  = centre + offset * ( -dy,  dx) / |d|
    right flank = centre + centre + offset * (  dy, -dx) / |d|

Each flank yields (luminance, opponent) sampled from the channels the front
end already computed. Ownership is then the signed difference of flank
statistics -- which side is "figure" is decided by the data, not by an
arbitrary label, and a chain with no flanks yields nothing rather than a
constant.

Neural analogue
---------------
Border-ownership (V2) is computed from texture/color/luminance contrast
ACROSS a boundary, not from the boundary alone: a contour owns the side
that is coherent with the figure. Sampling both flanks at an offset
proportional to the local spatial-frequency wavelength is the same
comparison at the scale the contour itself lives on.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import parameters as P

# perpendicular offset as a multiple of the median wavelength. Using the
# wavelength keeps the flank separation matched to the contour's own scale
# instead of a fixed pixel count, which would either cross into the
# neighbouring contour at fine scales or sample flat background at coarse.
PERPENDICULAR_SCALE = 0.5

# a flank sample is discarded when it lands outside the plane, rather than
# clamped to the border pixel: a clamped sample is systematically biased
# toward whatever the edge happens to be, which is exactly the artefact this
# measurement is meant to avoid.
FLANK_OUT_OF_BOUNDS = None


def perpendicular_offsets(pts: List[Tuple[float, float]],
                          scale_px: float
                          ) -> List[Optional[Tuple[float, float, float, float]]]:
    """Per-vertex (lx, ly, rx, ry) flank centres, or None per degenerate vertex.

    Vertex i's contour direction is the vector from vertex i-1 to vertex i+1
    (a central difference), so the normal is well defined for both open and
    closed walks. If that chord is zero -- a repeated point -- the vertex
    yields None and is skipped rather than assigned a fabricated normal.
    """
    n = len(pts)
    out: List[Optional[Tuple[float, float, float, float]]] = []
    for i in range(n):
        ip = (i - 1) % n
        in_ = (i + 1) % n
        dx = pts[in_][0] - pts[ip][0]
        dy = pts[in_][1] - pts[ip][1]
        mag = math.hypot(dx, dy)
        if mag < 1e-9:
            out.append(None)
            continue
        # normal to the contour direction, unit length
        nx, ny = -dy / mag, dx / mag
        cx, cy = pts[i]
        out.append((cx - nx * scale_px, cy - ny * scale_px,
                    cx + nx * scale_px, cy + ny * scale_px))
    return out


def planes_from_rgb(rgb: Any) -> Optional[Dict[str, Any]]:
    """The front-end planes, from ONE luminance path, for every flank consumer.

    Before this existed, chain geometry came from
    `luminance_adapt(opponent_channels(rgb)[..., 0])` while the flank
    samplers read `compute_opponent_retinex(rgb)['lum_comp']`. Measured
    directly, those two planes differ by up to 0.95 in absolute value
    (retinex mean -0.06 / min -0.53 / max 0.87; adapted mean 0.48 / range
    0.40-0.60). A flank therefore had its contrast measured on a plane the
    chain it belongs to was never derived from.

    Both now use the SAME adapted luminance, so a flank's two samples and
    the contour linking them live on one plane. The opponent channels stay
    raw and signed: `luminance_adapt` is a Weber-regime divisive transform
    defined for luminance, and applying it to a signed red-green axis would
    map it into [0,1) and destroy the sign that ownership is computed from.
    """
    if rgb is None:
        return None
    a = np.asarray(rgb)
    if a.ndim != 3 or a.shape[-1] < 3:
        return None
    if a.dtype == np.uint8:
        a = a.astype(np.float32) / 255.0
    from perception.filters import luminance_adapt, opponent_channels
    opp = opponent_channels(a.astype(np.float32))
    lum = luminance_adapt(opp[..., 0])
    if np.asarray(lum).ndim != 2 or np.asarray(lum).size == 0:
        return None
    return {"lum": np.asarray(lum, dtype=np.float64),
            "rg": np.asarray(opp[..., 1], dtype=np.float64),
            "by": np.asarray(opp[..., 2], dtype=np.float64)}


class ChannelSampler:
    """Nearest-pixel sampling of the channels that already exist.

    `channels` is the dict from `compute_opponent_retinex`-style extraction:
    a luminance plane plus opponent planes. Sampling is nearest-pixel with
    strict bounds: outside the plane yields FLANK_OUT_OF_BOUNDS so the caller
    can skip the flank instead of reading a border pixel.
    """

    def __init__(self, channels: Optional[Dict[str, Any]],
                 image_shape: Optional[Tuple[int, int]] = None):
        self.lum = None
        self.opp = None
        self.h = self.w = 0
        if channels:
            # planes_from_rgb layout first (lum / rg / by); then the
            # compute_opponent_retinex layout (lum_comp / rg_comp / by_comp)
            # so an older caller still resolves.
            for k in ('lum', 'luminance', 'lum_comp'):
                arr = np.asarray(channels.get(k, []), dtype=np.float64)
                if arr.ndim >= 2 and arr.size:
                    self.lum = arr
                    self.h, self.w = arr.shape[0], arr.shape[1]
                    break
            a = np.asarray(channels.get('rg',
                                        channels.get('rg_comp', [])),
                           dtype=np.float64)
            b = np.asarray(channels.get('by',
                                        channels.get('by_comp', [])),
                           dtype=np.float64)
            if (a.ndim >= 2 and a.size and b.ndim >= 2 and b.size
                    and a.shape == b.shape):
                self.opp = np.stack([a, b], axis=-1)
            else:
                o = channels.get('opponent')
                if o is None:
                    o = channels.get('opp')
                if o is not None:
                    o = np.asarray(o, dtype=np.float64)
                    if o.ndim == 3 and o.size:
                        self.opp = o
        if self.lum is None and image_shape is not None:
            self.h, self.w = int(image_shape[0]), int(image_shape[1])

    def _px(self, plane, x: float, y: float) -> Optional[float]:
        if plane is None or not self.h or not self.w:
            return None
        xi = int(round(x))
        yi = int(round(y))
        if xi < 0 or yi < 0 or xi >= self.w or yi >= self.h:
            return FLANK_OUT_OF_BOUNDS
        v = float(plane[yi, xi])
        return v if math.isfinite(v) else None

    def sample(self, x: float, y: float) -> Optional[Tuple[float, float, float]]:
        """(lum, o0, o1) at (x, y), or None when out of bounds."""
        if self.lum is None:
            return None
        l = self._px(self.lum, x, y)
        if l is None:
            return None
        if self.opp is not None:
            a = self._px(self.opp[..., 0], x, y)
            b = self._px(self.opp[..., 1], x, y)
            return (l, a if a is not None else 0.0, b if b is not None else 0.0)
        return (l, 0.0, 0.0)


def chain_flank_fields(
    det: Any,
    anchor_positions: Optional[Dict[Any, Any]],
    median_wavelength: Optional[float],
    channels: Optional[Dict[str, Any]],
    image_shape: Optional[Tuple[int, int]] = None,
) -> Optional[Dict[str, Any]]:
    """Both-flank statistics for every usable member of one chain.

    Returns None when there is nothing usable to say (no positions, no
    wavelength, no channels), so a caller emits nothing rather than a
    constant descriptor.

    Ownership per member is the signed flank difference of each statistic:
    positive means the left flank carries the larger value.
    """
    if det is None or anchor_positions is None:
        return None
    if median_wavelength is None or median_wavelength <= 0.0:
        return None
    from perception.chain_polylines import chain_polyline_for_detection
    g = chain_polyline_for_detection(det, anchor_positions, min_points=2)
    if g is None:
        return None
    pts = g['points']
    if not pts:
        return None

    sampler = ChannelSampler(channels, image_shape)
    offset = float(PERPENDICULAR_SCALE) * float(median_wavelength)
    offsets = perpendicular_offsets(pts, offset)

    members = []
    for i, fl in enumerate(offsets):
        if fl is None:
            continue
        lx, ly, rx, ry = fl
        ls = sampler.sample(lx, ly)
        rs = sampler.sample(rx, ry)
        if ls is None or rs is None:
            continue
        members.append(dict(
            index=i,
            vertex=(float(pts[i][0]), float(pts[i][1])),
            left=ls, right=rs,
        ))
    if not members:
        return None

    # per-member ownership: signed difference, left minus right
    for m in members:
        (ll, la, lb) = m['left']
        (rl, ra, rb) = m['right']
        m['lum_diff'] = ll - rl
        m['opp0_diff'] = la - ra
        m['opp1_diff'] = lb - rb
        m['side'] = 1 if m['lum_diff'] >= 0.0 else 0

    n = len(members)
    mean_lum = sum(m['left'][0] for m in members) / n
    mean_lum_r = sum(m['right'][0] for m in members) / n
    sat = [abs(math.hypot(m['left'][1], m['left'][2]) -
               math.hypot(m['right'][1], m['right'][2])) for m in members]
    var_lum = sum((m['left'][0] - mean_lum) ** 2 for m in members) / n

    return dict(
        members=members,
        n_members=n,
        mean_lum_left=mean_lum,
        mean_lum_right=mean_lum_r,
        mean_sat_contrast=(sum(sat) / n) if sat else 0.0,
        lum_variance=var_lum,
        closed=bool(g['closed']),
        area=g['area'],
        offset_px=offset,
        median_wavelength=float(median_wavelength),
    )


def flank_signature_slots(f: Dict[str, Any]) -> Dict[str, float]:
    """Binned slots summarising a chain's flank contrast.

    0 lum_contrast_bin  graded 0..7  mean |left-right| luminance across
                        the chain's members
    1 sat_contrast_bin  graded 0..7  mean |left-right| opponent magnitude
    2 lum_level_bin     graded 0..7  mean left-flank luminance level
    3 side_bin          graded 0..7  fraction of members whose left flank
                        is brighter

    All four are graded differences or levels -- none is a thresholded flag,
    so a chain with weak contrast still occupies its measured position
    rather than being silently binned to zero.

    Returned as a dict keyed by slot name so a consumer reads a name rather
    than a positional index: the earlier positional form was read with the
    wrong key names downstream and silently yielded 0.0 for every slot,
    which is exactly how a working signal gets lost.
    """
    def _g(v: float, hi: float, n: int = 8) -> float:
        if not isinstance(v, (int, float)) or not math.isfinite(v):
            return 0.0
        q = int(v / hi * n)
        return float(min(n - 1, max(0, q)))

    n = f['n_members']
    lum_c = sum(abs(m['lum_diff']) for m in f['members']) / n
    sat_c = sum(abs(math.hypot(m['opp0_diff'], m['opp1_diff']))
                for m in f['members']) / n
    side = sum(m['side'] for m in f['members']) / n
    return {
        "lum_contrast_bin": _g(lum_c, 1.0),
        "sat_contrast_bin": _g(sat_c, 1.0),
        "lum_level_bin": _g(f['mean_lum_left'], 1.0),
        "side_bin": _g(side, 1.0),
    }


if __name__ == '__main__':
    # smoke: a vertical edge (left dark, right bright) with 2 flanks
    # edge at x=10, so a chain running down x=10 has its two flanks at
    # x=0 (dark) and x=20 (bright) -- the edge lies BETWEEN them
    ch = {'lum': np.zeros((64, 64)) + 0.2}
    ch['lum'][:, 10:] = 0.8
    from perception.chain_polylines import chain_polyline_for_detection

    class _D:
        alpha = 1
        sigma = type('S', (), {'quantized': (3, 0, 0, 0, 0, 0)})()
        rho = (('entity', 1, 0, '+'),
               ('member', 'a', 0, '+'),
               ('member', 'b', 1, '+'),
               ('member', 'c', 2, '+'))
    pos = {'a': (10.0, 20.0), 'b': (10.0, 40.0), 'c': (10.0, 60.0)}
    f = chain_flank_fields(_D(), pos, 20.0, ch, (64, 64))
    print('members:', f['n_members'])
    print('mean lum left/right: %.3f / %.3f'
          % (f['mean_lum_left'], f['mean_lum_right']))
    print('slots:', flank_signature_slots(f))
