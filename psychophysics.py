"""perception/psychophysics.py -- front-end validation metrics + gate rule.

NOT an accuracy module. Every quantity here is a mechanism-level
psychophysical measurement of the SAME synthetic stimuli the field pipeline
consumes. No ETH-80, no network, no trained score. The purpose is to convert
"the front end seems fine" into "the front end is contractually fine":

  * tuning curves (OSI, orientation half-width at half-max) -- the V1
    population's own resolution, measured rather than asserted;
  * contour detectability d' -- can the integration field find a co-circular
    contour buried in ENERGY-MATCHED clutter (the Field-Hayes-Hess test),
    does d' RISE with path length, and does it COLLAPSE under orientation
    jitter? Arc gain and link support only show facilitation on clean input;
    d' shows discrimination against matched noise.

The three-gate rule (pre-registered; see diagnostics/_FRONT_END_GATE.md):
a front-end (association-field or Gabor-kernel) edit ships only if ALL hold:
  (1) arc gain     -- cocircular facilitation on a curved path, with a
                      negligible far-blank response;
  (2) link support -- true collinear links admitted above LINK_SUPPORT_MIN
                      while resolvable false links stay below it (the rule-B
                      R7 corpus contract: >= 90% of true adjacent pairs kept,
                      zero cross pairs kept);
  (3) contour d'   -- d' >= DPRIME_MIN on the aligned contour, d' RISES with
                      path length, and d' COLLAPSES under orientation jitter.

Pure and deterministic; imports numpy and parameters only. The one measurement
that touches the filter bank (`measured_orientation_sigma_bins`) imports
perception.filters lazily so importing the metrics does not build a bank.
"""
from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

import numpy as np

#: The d' floor an aligned co-circular contour must clear to be "detectable".
#: Pre-registered (see _FRONT_END_GATE.md): 1.0 = the classic 76%-correct
#: two-alternative threshold, the weakest claim Field-Hayes-Hess support.
DPRIME_MIN = 1.0
#: Arc-gain gate: facilitation raises the curved path by this factor ...
ARC_GAIN_MIN = 1.05
#: ... while a far blank region stays at most this fraction of the path.
ARC_BLANK_RATIO_MAX = 0.10
#: d' must grow by at least this factor from the shortest usable path (5
#: elements -- a 3-element chain has a single interior element and is
#: terminal-effect dominated) to the longest (9 elements). The coherent-signal
#: prediction is sqrt(9/5) = 1.34; 1.25 is the pre-registered floor.
DPRIME_LENGTH_RATIO_MIN = 1.25
#: Jitter must knock d' down to at most this fraction of the aligned value.
DPRIME_JITTER_COLLAPSE_FRAC = 0.50
#: Link-support gate (rule-B R7 contract): at least this fraction of TRUE
#: adjacent collinear pairs must clear LINK_SUPPORT_MIN ...
LINK_TRUE_KEEP_MIN = 0.90
#: ... while ZERO resolvable false (cross) pairs may clear it.
LINK_FALSE_KEEP_MAX = 0

#: [O8] The sensor-noise ladder: Gaussian pixel noise (SD in luminance units,
#: on a 0..1 frame) at which the V1 population's own selectivity is probed.
#: The values are a MEASUREMENT ladder, not a gate: the probe reports where
#: the front end's orientation evidence falls to noise, so a calibration
#: cannot be tuned to one noise-free synthetic frame.
SENSOR_NOISE_SIGMAS: Tuple[float, ...] = (0.0, 0.02, 0.05, 0.10)
#: [O8] The orientation-jitter ladder for the contour d' probe (degrees).
JITTER_LADDER_DEG: Tuple[float, ...] = (0.0, 10.0, 20.0, 40.0)



# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

def dprime(signal: Sequence[float], noise: Sequence[float]) -> float:
    """Sensitivity index d' between a signal and a noise sample.

    Pooled-variance form: ``(mu_s - mu_n) / sqrt(0.5 (var_s + var_n))``. Pooled
    (not noise-only) because the clutter and the contour are both finite
    samples of the same field statistic; the pooled estimate is the symmetric,
    order-independent reading. Returns 0.0 when either sample is degenerate
    (fewer than two values, or zero pooled spread) -- an unmeasurable contrast
    is reported as no sensitivity, never as infinity.
    """
    s = np.asarray([float(v) for v in signal], dtype=np.float64)
    n = np.asarray([float(v) for v in noise], dtype=np.float64)
    if s.size < 2 or n.size < 2:
        return 0.0
    vs = float(np.var(s, ddof=1))
    vn = float(np.var(n, ddof=1))
    spread = math.sqrt(0.5 * (vs + vn))
    if spread <= 1e-12:
        return 0.0
    return float((float(np.mean(s)) - float(np.mean(n))) / spread)


def path_dprime(signal: Sequence[float], noise: Sequence[float]) -> float:
    """Path-level detectability: the classic coherent-signal d'.

    A contour of ``n`` co-circular elements is a single coherent object: its
    evidence ACCUMULATES while the clutter's does not. The observer decision
    variable is the signal mean against the noise's own spread, and the
    coherent sum raises the effective sensitivity by ``sqrt(n)``:

        d' = sqrt(n) * (mean(signal) - mean(noise)) / sd(noise)

    (noise spread only -- the reference distribution is the energy-matched
    clutter). This is the quantity that RISES with path length; the pooled
    per-element :func:`dprime` is length-flat because each interior element is
    already fully supported. Returns 0.0 for degenerate input.
    """
    s = np.asarray([float(v) for v in signal], dtype=np.float64)
    n = np.asarray([float(v) for v in noise], dtype=np.float64)
    if s.size < 1 or n.size < 2:
        return 0.0
    sd = float(np.std(n, ddof=1))
    if sd <= 1e-12:
        return 0.0
    return float(math.sqrt(s.size) * (float(np.mean(s)) - float(np.mean(n))) / sd)


def _circular_lerp(curve: Sequence[float], x: float) -> float:
    """Interpolate a tuning curve given in DEGREES step = 180 / len(curve)."""
    n = len(curve)
    step = 180.0 / n
    f = (x % 180.0) / step
    i0 = int(math.floor(f)) % n
    i1 = (i0 + 1) % n
    frac = f - math.floor(f)
    return float(curve[i0] * (1.0 - frac) + curve[i1] * frac)


def orientation_hwhm_deg(curve: Sequence[float]) -> float:
    """Half-width at half-maximum of an orientation curve, in degrees.

    Angles are the channel centres ``i * 180 / len(curve)``; the curve is
    interpolated so the answer is not quantised to channel spacing. Returns 90
    when the curve never falls to half peak on both sides (orientation
    unselective).
    """
    resp = [float(v) for v in curve]
    if not resp:
        return 90.0
    peak = max(resp)
    if peak <= 0.0:
        return 90.0
    pi = resp.index(peak)
    centre = pi * (180.0 / len(resp))
    half = peak / 2.0
    for d in range(1, 91):
        lo = _circular_lerp(resp, centre - d)
        hi = _circular_lerp(resp, centre + d)
        if lo < half and hi < half:
            return float(d)
    return 90.0


def orientation_osi(curve: Sequence[float]) -> float:
    """Orientation selectivity index = |sum v e^{i2theta}| / sum v, in [0, 1]."""
    resp = [float(v) for v in curve]
    n = len(resp)
    if n == 0:
        return 0.0
    sx = 0.0
    sy = 0.0
    tot = 0.0
    for i, v in enumerate(resp):
        a = 2.0 * (i * 180.0 / n) * math.pi / 180.0
        sx += v * math.cos(a)
        sy += v * math.sin(a)
        tot += v
    if tot <= 0.0:
        return 0.0
    return float(math.hypot(sx, sy) / tot)


def sigma_bins_from_hwhm_deg(hwhm_deg: float, step_deg: float) -> float:
    """Gaussian sigma (in BINS) whose half-width at half-max equals `hwhm_deg`.

    A Gaussian has HWHM = sigma * sqrt(2 ln 2); here the width is expressed in
    bins of ``step_deg`` each (one Gabor channel = one bin)."""
    if step_deg <= 0.0:
        raise ValueError("step_deg must be > 0")
    return float((hwhm_deg / step_deg) / math.sqrt(2.0 * math.log(2.0)))


# ---------------------------------------------------------------------------
# the measured V1 population width
# ---------------------------------------------------------------------------

def _grating(n: int, wavelength: float, ori_deg: float,
             contrast: float = 0.5) -> np.ndarray:
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
    w = 2.0 * math.pi / max(wavelength, 1e-6)
    t = ori_deg * math.pi / 180.0
    ct, st = math.cos(t), math.sin(t)
    c = (n - 1) / 2.0
    return (0.5 + 0.5 * contrast * np.sin(
        w * ((xx - c) * ct + (yy - c) * st))).astype(np.float32)


def measured_orientation_sigma_bins(size: int = 256) -> Dict[str, object]:
    """Measure the V1 bank's orientation tuning and return its Gaussian width.

    For every spatial-frequency band, a grating at the band's own wavelength is
    pushed through the whole bank; the per-band orientation curve at the frame
    centre gives a half-width at half-maximum (interpolated, in degrees) which
    is converted to a Gaussian sigma in BINS (one channel = one bin). The
    reported scalar is the mean sigma across bands.

    Deterministic; one full bank pass. This is the number the CONTOUR_PEAK
    ``theta`` population axis declares, and the width the assembly MDL charges
    against -- the harness re-measures it and the contract test asserts the
    declared parameters equal it, so a kernel edit cannot silently change the
    representation's own resolution.
    """
    from perception.contour_field import front_end_scale
    from perception.filters import GaborBank, compute_responses

    scale = front_end_scale(size, size)
    bank = GaborBank.build_default(scale=scale)
    n_theta = bank.kernels.shape[0]
    n_sf = bank.kernels.shape[1]
    n = size
    c = n // 2
    step_deg = 180.0 / n_theta

    per_band: List[float] = []
    osis: List[float] = []
    for isf in range(n_sf):
        wavelength = 1.0 / max(float(bank.sf[isf]), 1e-6)
        lum = _grating(n, wavelength, 0.0)
        resp = compute_responses(bank, lum)["complex"]  # (nt, nsf, H, W)
        curve = [float(resp[it, isf, c, c]) for it in range(n_theta)]
        per_band.append(orientation_hwhm_deg(curve))
        osis.append(orientation_osi(curve))

    mean_hwhm = float(np.mean(per_band)) if per_band else 90.0
    return {
        "size": int(size),
        "n_theta": int(n_theta),
        "n_sf": int(n_sf),
        "step_deg": float(step_deg),
        "hwhm_deg_per_band": [float(v) for v in per_band],
        "osi_per_band": [float(v) for v in osis],
        "mean_hwhm_deg": mean_hwhm,
        "mean_osi": float(np.mean(osis)) if osis else 0.0,
        "sigma_bins": sigma_bins_from_hwhm_deg(mean_hwhm, step_deg),
    }


# ---------------------------------------------------------------------------
# [O8] calibration probes: sensor noise and orientation jitter
# ---------------------------------------------------------------------------

def sensor_noise_probe(
    size: int = 128,
    sigmas: Sequence[float] = SENSOR_NOISE_SIGMAS,
    trials: int = 5,
    seed: int = 7,
) -> Dict[str, object]:
    """How far the V1 population's own evidence survives SENSOR NOISE.

    The calibration question O8 asks is not "does the front end work on a
    clean synthetic frame" but "at what noise level does its evidence stop
    being evidence".  For each noise SD this probe builds the same oriented
    grating and a blank frame, adds independent Gaussian noise to BOTH, runs
    the real bank, and reports:

      * ``osi``      -- orientation selectivity of the grating's tuning curve
                        at the band's own spatial frequency (a clean-frame OSI
                        should fall as noise rises);
      * ``hwhm_deg`` -- its orientation half-width (widens with noise);
      * ``dprime``   -- the peak centre response of the grating frames against
                        the blank frames' own peak (the front end's own
                        signal-vs-noise separation).

    Deterministic for a given seed.  Returns ``{str(sigma): {...}}`` plus the
    two endpoints.  This is a MEASUREMENT, not a gate: no value here is
    compared to a tuned threshold.
    """
    from perception.filters import GaborBank, compute_responses

    bank = GaborBank.build_default(scale=1.0)
    nt = len(bank.theta)
    nsf = len(bank.sf)
    c = size // 2
    isf_mid = nsf // 2
    wavelength = 1.0 / max(float(bank.sf[isf_mid]), 1e-6)
    rng = np.random.default_rng(int(seed))

    out: Dict[str, object] = {}
    for sigma in sigmas:
        signal_peak: List[float] = []
        noise_peak: List[float] = []
        osis: List[float] = []
        hwhms: List[float] = []
        for _ in range(int(trials)):
            clean = _grating(size, wavelength, 0.0)
            blank = np.full((size, size), 0.5, dtype=np.float32)
            if sigma > 0.0:
                clean = np.clip(
                    clean + rng.normal(0.0, sigma, clean.shape), 0.0, 1.0)
                blank = np.clip(
                    blank + rng.normal(0.0, sigma, blank.shape), 0.0, 1.0)
            rs = compute_responses(bank, clean.astype(np.float32))["complex"]
            rn = compute_responses(bank, blank.astype(np.float32))["complex"]
            curve = [float(rs[it, isf_mid, c, c]) for it in range(nt)]
            osis.append(orientation_osi(curve))
            hwhms.append(orientation_hwhm_deg(curve))
            signal_peak.append(max(curve) if curve else 0.0)
            noise_peak.append(float(np.max(rn[:, isf_mid, c, c])))
        noise_sd = float(np.std(noise_peak, ddof=1)) if len(noise_peak) > 1 else 0.0
        measurable = noise_sd > 1e-12
        out[str(float(sigma))] = {
            "sigma": float(sigma),
            "osi_mean": float(np.mean(osis)),
            "hwhm_deg_mean": float(np.mean(hwhms)),
            # A constant blank frame has zero response spread, so d' is not
            # defined there: report None, never a fabricated infinity/zero.
            "dprime": dprime(signal_peak, noise_peak) if measurable else None,
            "dprime_measurable": bool(measurable),
            "signal_peak_mean": float(np.mean(signal_peak)),
            "noise_peak_mean": float(np.mean(noise_peak)),
            "noise_peak_sd": noise_sd,
        }
    keys = [str(float(s)) for s in sigmas]
    out["noiseless_osi"] = out[keys[0]]["osi_mean"]
    out["noisiest_osi"] = out[keys[-1]]["osi_mean"]
    out["osi_retention"] = (
        out[keys[-1]]["osi_mean"] / max(out[keys[0]]["osi_mean"], 1e-9))
    return out


def _element(lum: np.ndarray, x: float, y: float, tangent: float,
             wavelength: float, amp: float = 0.55) -> None:
    """Add one Gabor element whose BAR runs along ``tangent`` (local helper)."""
    H, W = lum.shape
    sigma = 0.9 * wavelength
    R = int(3.0 * sigma) + 1
    x0, x1 = max(0, int(x) - R), min(W, int(x) + R + 1)
    y0, y1 = max(0, int(y) - R), min(H, int(y) + R + 1)
    if x0 >= x1 or y0 >= y1:
        return
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    ct, st = math.cos(tangent - math.pi / 2.0), math.sin(tangent - math.pi / 2.0)
    xr = (xx - x) * ct + (yy - y) * st
    yr = -(xx - x) * st + (yy - y) * ct
    env = np.exp(-(xr * xr + yr * yr) / (2.0 * sigma * sigma))
    car = np.cos(2.0 * math.pi / wavelength * xr)
    lum[y0:y1, x0:x1] += (amp * env * car).astype(np.float32)


def _contour_scene(size: int, wavelength: float, path_len: int,
                   jitter_deg: float):
    """A straight anchored contour over a fixed 8-element clutter field."""
    lum = np.full((size, size), 0.5, dtype=np.float32)
    cx = (size - 1) / 2.0
    cy = size / 2.0
    x0 = cx - (path_len - 1) / 2.0 * wavelength
    pts = []
    for i in range(path_len):
        x = x0 + i * wavelength
        j = math.radians(jitter_deg) * (1.0 if i % 2 == 0 else -1.0)
        _element(lum, x, cy, j, wavelength)
        pts.append((x, cy))
    # Fixed clutter, well clear of the path band.
    clutter = [((0.15 + 0.10 * k) * size, (0.18 if k % 2 else 0.82) * size,
                (k * 1.7) % math.pi) for k in range(8)]
    for x, y, th in clutter:
        _element(lum, x, y, th, wavelength)
    return lum, pts, [(x, y) for x, y, _ in clutter]


def _disc_sample(per_pix: np.ndarray, pts, r: int = 2) -> List[float]:
    H, W = per_pix.shape
    out = []
    for (x, y) in pts:
        xi, yi = int(round(x)), int(round(y))
        vals = []
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                xx, yy = xi + dx, yi + dy
                if 0 <= xx < W and 0 <= yy < H:
                    vals.append(float(per_pix[yy, xx]))
        out.append(float(np.mean(vals)) if vals else 0.0)
    return out


def jitter_probe(
    size: int = 128,
    n_elements: int = 9,
    jitters_deg: Sequence[float] = JITTER_LADDER_DEG,
) -> Dict[str, object]:
    """Contour d' as orientation JITTER grows, measured on the real field.

    The existing ``gate_dprime`` collapses jitter at one fixed angle.  This
    probe measures the whole ladder, so a calibration can see whether the
    collapse is a cliff or a gradual loss -- and cannot claim a jitter
    tolerance that only holds at the one point it was tuned on.

    Returns ``{str(deg): {"dprime":..., "on_mean":..., "off_mean":...}}`` plus
    ``collapse_frac`` (last / first).  Pure; imports the field lazily.
    """
    from perception.contour_field import field_from_luminance
    from perception.filters import GaborBank

    bank = GaborBank.build_default(scale=1.0)
    wavelength = 1.0 / max(float(bank.sf[len(bank.sf) // 2]), 1e-6)
    out: Dict[str, object] = {}
    for jitter in jitters_deg:
        lum, pts, clutter = _contour_scene(
            size, wavelength, int(n_elements), float(jitter))
        field = field_from_luminance(lum)
        per_pix = (np.asarray(field["integration"], np.float32)
                   - np.asarray(field["integration_driving"], np.float32)).max(axis=(0, 1))
        on = _disc_sample(per_pix, pts)
        off = _disc_sample(per_pix, clutter)
        out[str(float(jitter))] = {
            "jitter_deg": float(jitter),
            "dprime": path_dprime(on, off),
            "on_mean": float(np.mean(on)) if on else 0.0,
            "off_mean": float(np.mean(off)) if off else 0.0,
        }
    keys = [str(float(j)) for j in jitters_deg]
    out["collapse_frac"] = (
        out[keys[-1]]["dprime"] / max(out[keys[0]]["dprime"], 1e-9))
    return out


# ---------------------------------------------------------------------------
# the three-gate rule
# ---------------------------------------------------------------------------

def gate_arc(gain_arc: float, blank_ratio: float) -> Tuple[bool, str]:
    ok = (gain_arc >= ARC_GAIN_MIN) and (blank_ratio <= ARC_BLANK_RATIO_MAX)
    return ok, (f"arc gain {gain_arc:.3f} >= {ARC_GAIN_MIN} and "
                f"blank ratio {blank_ratio:.4f} <= {ARC_BLANK_RATIO_MAX}")


def gate_link_support(true_kept_frac: float, false_kept: int,
                      threshold: float) -> Tuple[bool, str]:
    """R7 contract: >= 90% of true adjacent links kept, zero cross links kept."""
    ok = (true_kept_frac >= LINK_TRUE_KEEP_MIN) and (false_kept <= LINK_FALSE_KEEP_MAX)
    return ok, (f"true links kept @ {threshold:.2f} = {true_kept_frac:.3f} >= "
                f"{LINK_TRUE_KEEP_MIN} and false links kept = {false_kept} "
                f"<= {LINK_FALSE_KEEP_MAX}")


def gate_dprime(d_by_length: Dict[int, float], jittered: float) -> Tuple[bool, str]:
    """d' must (a) clear DPRIME_MIN at the longest path, (b) rise with length,
    and (c) collapse under jitter."""
    if not d_by_length:
        return False, "no d' measurements"
    lengths = sorted(d_by_length)
    longest = float(d_by_length[lengths[-1]])
    shortest = float(d_by_length[lengths[0]])
    detectable = longest >= DPRIME_MIN
    rises = longest >= DPRIME_LENGTH_RATIO_MIN * max(shortest, 1e-9)
    collapses = jittered <= (1.0 - DPRIME_JITTER_COLLAPSE_FRAC) * longest
    ok = detectable and rises and collapses
    return ok, (f"d'(len {lengths[-1]}) {longest:.3f} >= {DPRIME_MIN}; "
                f"length ratio {longest / max(shortest, 1e-9):.2f} >= "
                f"{DPRIME_LENGTH_RATIO_MIN}; "
                f"jittered {jittered:.3f} <= "
                f"{(1.0 - DPRIME_JITTER_COLLAPSE_FRAC):.2f} of aligned")


def front_end_gate(arc: Tuple[float, float],
                   link_support: Tuple[float, float, float],
                   dprime_curves: Tuple[Dict[int, float], float]) -> Dict[str, object]:
    """Evaluate all three gates; `admissible` is their conjunction.

    Arguments are the raw measurements, not verdicts:
      arc            = (gain_arc, blank_ratio)
      link_support   = (true_kept_frac, false_kept, threshold)
      dprime_curves  = ({path_length: d'}, jittered_d')
    """
    g1, m1 = gate_arc(*arc)
    g2, m2 = gate_link_support(*link_support)
    g3, m3 = gate_dprime(*dprime_curves)
    return {
        "arc": {"pass": g1, "detail": m1},
        "link_support": {"pass": g2, "detail": m2},
        "contour_dprime": {"pass": g3, "detail": m3},
        "admissible": bool(g1 and g2 and g3),
    }
