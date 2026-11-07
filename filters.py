"""perception/filters.py -- extracted from sensory_floor.py (Move 3a).

MOVE ONLY: whole top-level statements, verbatim. The re-export
shim in sensory_floor.py keeps `from sensory_floor import X`
working until the importer repoint retires it.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import fftconvolve as fftconvolve
from scipy.ndimage import convolve as nd_convolve
from scipy.fft import next_fast_len
import math
from dataclasses import dataclass
from typing import Any
import parameters as P

def center_surround(lum, center_sigma=None, surround_sigma=None):
    """Difference-of-Gaussians retinal receptive field. Zero-mean by
    construction, so a uniform field produces no response."""
    if center_sigma is None:
        center_sigma = float(P.RETINAL_CENTER_SIGMA)
    if surround_sigma is None:
        surround_sigma = float(P.RETINAL_SURROUND_MULT) * center_sigma
    c = gaussian_filter(lum.astype(np.float32),
                        sigma=center_sigma, mode="reflect")
    s = gaussian_filter(lum.astype(np.float32),
                        sigma=surround_sigma, mode="reflect")
    return (c - s).astype(np.float32)

def split_on_off(lum, center_sigma=None):
    """Split luminance into non-negative retinal ON and OFF channels."""
    dog = center_surround(lum, center_sigma=center_sigma)
    on = np.maximum(dog, 0.0).astype(np.float32)
    off = np.maximum(-dog, 0.0).astype(np.float32)
    return on, off

def luminance_adapt(lum: np.ndarray) -> np.ndarray:
    """Divisive local luminance adaptation (Weber regime).

    R = L / (L + L0 + A * Lbar_local).

    Scaling the whole scene by k leaves R unchanged once k*Lbar >> L0, so a
    shadowed and a sunlit copy of the same material produce the SAME
    signature. Output in [0, 1).

    [B19 NOTE] R is in [0,1) but its MEAN is well below 1 (typically ~0.4-0.5),
    so every absolute threshold calibrated on raw luminance is now too high by
    roughly 2x. B18's noise floor must be measured on ADAPTED energy; see the
    B18 note. Re-measure the B1 block weights at the same time.
    """
    from perception.observation import _SF_TINY
    lum = np.clip(np.asarray(lum, np.float64), 0.0, None)
    s = max(2.0, P.LUM_ADAPT_FRAC * min(lum.shape[:2]))
    lbar = gaussian_filter(lum, sigma=s, mode="reflect")
    return (lum / (lum + P.LUM_ADAPT_L0 + P.LUM_ADAPT_A * lbar + _SF_TINY)).astype(np.float32)

def opponent_channels(rgb):
    """Retinal/LGN colour transform.

    Hurvich & Jameson 1957; De Valois 1960; Zeki 1980; Buchsbaum 1980.
    """
    if rgb.dtype == np.uint8:
        rgb = rgb.astype(np.float32) / 255.0
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    return np.stack(
        [lum, r - g, b - 0.5 * (r + g)], axis=-1
    ).astype(np.float32)

def color_contrast(channel, radius):
    surround = gaussian_filter(channel, sigma=radius / 2.0, mode="reflect")
    return np.abs(channel - surround).astype(np.float32)

def illuminant_estimate(rgb, top_frac):
    if rgb.dtype == np.uint8:
        rgb = rgb.astype(np.float32) / 255.0
    lum = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    flat = lum.flatten()
    k = max(1, int(len(flat) * top_frac))
    thresh = np.partition(flat, -k)[-k]
    mask = lum >= thresh
    if mask.sum() == 0:
        return np.ones(3, dtype=np.float32)
    mean_rgb = rgb[mask].mean(axis=0)
    m = float(mean_rgb.max())
    if m > 0:
        return (mean_rgb / m).astype(np.float32)
    return np.ones(3, dtype=np.float32)

def color_constancy(rgb, top_frac):
    illum = illuminant_estimate(rgb, top_frac)
    return np.clip(rgb / (illum + 1e-6), 0.0, 1.0).astype(np.float32)

M_srgb_to_xyz = np.array([
    [0.4124, 0.3576, 0.1805],
    [0.2126, 0.7152, 0.0722],
    [0.0193, 0.1192, 0.9505],
], dtype=np.float64)

M_xyz_to_lms = np.array([
    [ 0.4002, 0.7076, -0.0808],
    [-0.2263, 1.1653,  0.0457],
    [ 0.0000, 0.0000,  0.9182],
], dtype=np.float64)

def srgb_to_linear(c: np.ndarray) -> np.ndarray:
    """Gamma-decode sRGB to linear RGB."""
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)

def compute_opponent_retinex(rgb_img: np.ndarray, extraction_scale: float = 5.0) -> dict:
    """Computes Retinex-normalized log-opponent (Weber-scaled) channels."""
    eps = 1e-4
    if rgb_img.dtype == np.uint8 or np.issubdtype(rgb_img.dtype, np.integer) or float(rgb_img.max()) > 1.0:
        rgb_float = rgb_img.astype(np.float64) / 255.0
    else:
        rgb_float = rgb_img.astype(np.float64)
    lin = srgb_to_linear(rgb_float)

    xyz = np.tensordot(lin, M_srgb_to_xyz, axes=([2], [1]))
    lms = np.tensordot(xyz, M_xyz_to_lms, axes=([2], [1]))

    log_L = np.log(lms[..., 0] + eps)
    log_M = np.log(lms[..., 1] + eps)
    log_S = np.log(lms[..., 2] + eps)

    sigma_fine = max(1.0, extraction_scale * P.COLOR_RETINEX_SIGMA_FINE_RATIO)
    sigma_coarse = max(2.0, sigma_fine * P.COLOR_RETINEX_SIGMA_COARSE_RATIO)

    # [AUDIT #14 FIX] Floor illuminant scale at a fraction of the image short side
    s_ill = max(2.0, max(sigma_fine, P.COLOR_ILLUM_FRAC * min(rgb_img.shape[:2])))
    illum_L = gaussian_filter(log_L, sigma=s_ill)
    illum_M = gaussian_filter(log_M, sigma=s_ill)
    illum_S = gaussian_filter(log_S, sigma=max(sigma_coarse, s_ill))

    r_L = log_L - illum_L
    r_M = log_M - illum_M
    r_S = log_S - illum_S

    sigma_smooth = max(1.0, 0.1 * sigma_fine)
    r_L = gaussian_filter(r_L, sigma=sigma_smooth)
    r_M = gaussian_filter(r_M, sigma=sigma_smooth)
    r_S = gaussian_filter(r_S, sigma=sigma_smooth)

    lum_raw = r_L + r_M
    rg_raw = r_L - r_M
    by_raw = r_S - (r_L + r_M) / 2.0

    alpha_LM = P.COLOR_LOG_OPPONENT_ALPHA_LM
    alpha_S = P.COLOR_LOG_OPPONENT_ALPHA_S

    lum_raw = np.clip(lum_raw, -2.5 * alpha_LM, 2.5 * alpha_LM)
    rg_raw = np.clip(rg_raw, -2.5 * alpha_LM, 2.5 * alpha_LM)

    by_weighted_raw = by_raw * P.COLOR_BY_WEIGHT
    by_weighted_raw = np.clip(by_weighted_raw, -2.5 * alpha_S, 2.5 * alpha_S)

    hue = np.arctan2(by_weighted_raw, rg_raw) % (2 * math.pi)
    sat_raw = np.sqrt(rg_raw**2 + by_weighted_raw**2)

    # [AUDIT #14 FIX] Below achromatic threshold, hue is noise -> clamp to 0.0
    hue = np.where(sat_raw < P.COLOR_ACHROMATIC_EPS, 0.0, hue)

    sat_norm = np.clip(sat_raw / P.COLOR_SAT_SCALE, 0.0, 1.0)

    lum_comp = np.tanh(lum_raw / alpha_LM)
    rg_comp = np.tanh(rg_raw / alpha_LM)
    by_comp = np.tanh(by_weighted_raw / alpha_S)

    return {
        "lum_comp": lum_comp,
        "rg_comp": rg_comp,
        "by_comp": by_comp,
        "sat_norm": sat_norm,
        "hue": hue,
    }

def _gabor_kernel(theta, sf, phase, sigma_x, sigma_y, size):
    """One oriented Gabor: Gaussian envelope times a sine carrier.

    ``a0`` varies along array axis 0 and ``a1`` along array axis 1
    (``indexing="ij"``). The signs below are not conventional-looking but
    they are the ones that make the channel index mean what it says: with
    them, channel ``it`` responds to a bar at exactly ``it * pi / n_theta``
    in image coordinates (x = column, y = row). Verified empirically over
    all 12 channel angles in ``diagnostics/_q1_binding_exp.py`` (option C).

    The transpose of this matrix (``a0*ct + a1*st``) was in force before and
    is a MIRROR, not a rotation: every oblique orientation was reported
    negated (a 30-degree bar answered as 150 degrees, a 45-degree bar as
    135). Axis-aligned bars are self-mirrored, so the defect was invisible
    on horizontal/vertical fixtures while oblique contours silently lost
    their orientation. See ``diagnostics/_q1_kernel_check.py``.
    """
    half = size / 2.0
    g = np.arange(-half + 0.5, half + 0.5, 1.0)
    a0, a1 = np.meshgrid(g, g, indexing="ij")
    ct, st = math.cos(theta), math.sin(theta)
    x_t = a0 * ct - a1 * st
    y_t = a0 * st + a1 * ct
    env = np.exp(-0.5 * ((x_t / sigma_x) ** 2 + (y_t / sigma_y) ** 2))
    carrier = np.cos(2.0 * math.pi * sf * x_t + phase)
    k = env * carrier
    k -= k.mean()
    n = math.sqrt(float((k ** 2).sum()))
    if n > 0:
        k /= n
    return k.astype(np.float32)

@dataclass(frozen=True)
class GaborBank:
    """V1 Gabor filter bank.

    Hubel & Wiesel 1962; De Valois & De Valois 1988; Jones & Palmer 1987.
    """

    kernels: Any
    theta: Any
    sf: Any

    @classmethod
    def build_default(cls, scale: float = 1.0):
        """Build the bank. `scale` is _front_end_scale(h, w) of the frame:
        the cycles-per-pixel band is DIVIDED by it so the bank stays tuned to
        the same physical structures at any resolution (1.0 = the 256x256
        reference, i.e. the historical band).

        Note: the top of the band can exceed Nyquist at low resolution; those
        filters simply match nothing. The bank arity (n_sf) is deliberately
        NOT reduced, because signature class names embed it.
        """
        n_theta = P.GABOR_N_THETA
        s = max(float(scale), 1e-6)
        sfs = np.geomspace(P.GABOR_SF_MIN / s, P.GABOR_SF_MAX / s, P.GABOR_N_SF)
        phases = (0.0, math.pi / 2.0)
        specs = []
        max_size = 1
        for isf, sf in enumerate(sfs):
            sx = P.GABOR_SIGMA_X_FRAC / float(sf)
            sy = sx / P.GABOR_ASPECT
            # [FIX 1] With GABOR_KERNEL_SIZE_FROM_MAX_SIGMA on, size uses
            # max(sigma_x, sigma_y) = sigma_y (since GABOR_ASPECT=0.5 makes
            # sy = 2*sx). Otherwise the old behavior uses sigma_x, clipping
            # the bar-length envelope at 1.5 * sigma_y instead of 3 * sigma_y.
            sigma_for_size = max(sx, sy) if P.GABOR_KERNEL_SIZE_FROM_MAX_SIGMA else sx
            size = int(2 * math.ceil(3.0 * sigma_for_size)) + 1
            # [ROUND 11 / O3] GABOR_KERNEL_MAX is the support cap at the
            # 256x256 reference (scale 1.0). At a higher front-end scale the
            # SAME physical support occupies more pixels, so the cap scales
            # with it; a fixed pixel cap truncates the coarsest band at high
            # resolution and the field stops being degree-invariant (same
            # scene, different detections). The cap is kept odd so kernels
            # stay centred. At scale 1.0 this is exactly the old cap.
            cap = int(round(float(P.GABOR_KERNEL_MAX) * max(1.0, s)))
            if cap % 2 == 0:
                cap += 1
            size = min(size, cap)
            if size % 2 == 0:
                size += 1
            max_size = max(max_size, size)
            specs.append((isf, float(sf), sx, sy, size))
        padded = np.zeros(
            (n_theta, len(sfs), 2, max_size, max_size), dtype=np.float32
        )
        for isf, sf, sx, sy, size in specs:
            off = (max_size - size) // 2
            for it in range(n_theta):
                theta = it * math.pi / n_theta
                for ip, phase in enumerate(phases):
                    padded[
                        it, isf, ip, off:off + size, off:off + size
                    ] = _gabor_kernel(theta, sf, phase, sx, sy, size)
        return cls(
            kernels=padded,
            theta=np.array(
                [k * math.pi / n_theta for k in range(n_theta)],
                dtype=np.float32,
            ),
            sf=sfs.astype(np.float32),
        )

def _conv2d(img, kernel):
    """Same-size convolution under ONE padding convention (reflect/symmetric).

    Direct for small kernels; for large kernels the FFT path is used, but
    scipy.signal.fftconvolve has no padding-mode argument and mode="same"
    would ZERO-pad -- making large-kernel and small-kernel responses
    non-comparable at the border. Pad manually first so both paths
    reflect-pad identically. Mirrors contour_field._convolve_same (FIX 3).
    """
    if kernel.shape[0] <= P.FFT_KERNEL_THRESHOLD:
        return nd_convolve(img, kernel, mode="reflect").astype(np.float32)
    ph = pw = int(kernel.shape[0]) // 2
    padded = np.pad(img, ((ph, ph), (pw, pw)), mode="symmetric")
    return fftconvolve(padded, kernel, mode="valid").astype(np.float32)

def compute_responses(bank, img):
    """Run the whole bank on ONE channel.

    [B.46 adoption] Returns a dict rather than a positional tuple, so
    the ON and OFF results cannot be positionally confused:
    {"simple": (n_theta, n_sf, n_phase, H, W),
     "complex": (n_theta, n_sf, H, W),
     "phase":   (n_theta, n_sf, H, W)}.
    The arrays are byte-identical to the former positional return. The
    key is "complex" (the tree's public name), not the draft
    snippet's "complex_" -- one spelling per array.
    """
    nt, nsf, nph = bank.kernels.shape[:3]
    h, w = img.shape
    simple = np.zeros((nt, nsf, nph, h, w), dtype=np.float32)
    for it in range(nt):
        for isf in range(nsf):
            for ip in range(nph):
                simple[it, isf, ip] = _conv2d(img, bank.kernels[it, isf, ip])
    complex_ = np.sqrt(simple[:, :, 0] ** 2 + simple[:, :, 1] ** 2)
    phase = np.arctan2(simple[:, :, 1], simple[:, :, 0])
    return {"simple": simple, "complex": complex_, "phase": phase}


_TINY = 1e-12

