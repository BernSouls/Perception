"""Global additive-evidence ring closure (Session E).

Replaces  P(ring) = prod_i P(link_i)  with

    LLR(ring) = sum_i  log p(res_i | this curve) - log p(res_i | background)
                + Cov + Res + Turn - c*N

on ONE curve hypothesis fitted to observed peaks.  No per-link gate is
consulted.

The per-peak term is a genuine LOG-LIKELIHOOD RATIO, not a penalty: it is
positive for a peak the curve explains better than the scene's own background.
An earlier version used -0.5 z^2, which is <= 0 for every peak, so sum(ell) FELL
with every correct member and the best hypothesis was always the smallest one
(measured: the true 40-peak ring ranked last of 28).  The scales are FIXED
(SCALE_R = 2.5 px, the corpus truth tolerance) and must not be estimated from
the hypothesis's own residual spread, which would let a contaminated set inflate
its own scale and score better.

The background is estimated leave-inliers-out over the peaks that are not
members (bandwidth = the same fixed scale), so a hypothesis cannot explain
itself.  Nothing is tuned and no truth enters the estimator.

Representation: a star-shaped radial model r(theta), fitted by ridge-regularised
least squares, with two closed bases -- periodic Fourier harmonics (`fourier`)
and a periodic cubic spline (`spline`).  The spline was chosen by measurement
(lower rms on 8/8 silhouettes, 0.95 vs 0.82 inlier on the hardest box scene).

Why radial: every silhouette in the corpus is convex, so r(theta) is single
valued; a radial basis fits an ARC without guessing the parameter range of the
unobserved part, which is where a parametric closed spline through ordered
points is ill-posed.

Decision statistic: `llr_ev_rad` (radial LLR + priors - inclusion cost).  The
field `llr` (the old penalty) is kept for ablation only.  Adding the tangent LLR
is measured to be worse (one false accept, lower d') and is not used.

Nothing is completed: the hypothesis is accepted only if it explains ENOUGH
OBSERVED peaks.  The unobserved angular sector is carried by the ridge prior,
never by extrapolated inliers.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# --- rule-first constants (all justified before use) -----------------------
# eps_r: one peak-grid quantum (peaks sit on integer pixels) plus the
# supersampled renderer's half-pixel; the same 2.5 px tolerance the corpus
# probes use to bind a chain member to rendered truth.
EPS_R = 2.5
# Angular grid for the radial model: 64 cells is 5.6 deg, well below the
# measured minimum angular peak spacing on the corpus (~9 deg).
N_GRID = 64
# Ridge: penalise the second difference of r on the grid (smoothness), with
# weight chosen so a single observed cell contributes an order of magnitude
# more than the prior.
RIDGE = 0.10
# Minimum observed peaks and angular coverage for a ring hypothesis.  A
# closed contour must be observed over most of its circumference; 0.55 is
# deliberately permissive so an occluded or sparsely detected ring is not
# vetoed by coverage (see O4).
MIN_MEMBERS = 6
MIN_COVERAGE = 0.55
# Fixed evidence scales.  CRITICAL: these must NOT be estimated from the
# hypothesis's own residual spread.  Self-standardising makes a WORSE fit score
# BETTER, because a contaminated set inflates its own median residual and so
# shrinks every normalised residual towards zero; LLR then stops being
# comparable across hypotheses (measured: the one true ring had the LOWEST LLR
# of 28).  EPS_R is the corpus truth tolerance (one peak-grid quantum) and
# SCALE_T is the corpus association tolerance used by every probe, so both are
# fixed by the task, not tuned.
SCALE_R = EPS_R
SCALE_T = math.radians(30.0)


def _grid():
    return np.arange(N_GRID + 1) * (2.0 * math.pi / N_GRID)


def _fourier_basis(theta: np.ndarray, k: int) -> np.ndarray:
    cols = [np.ones_like(theta)]
    for h in range(1, k + 1):
        cols.append(np.cos(h * theta))
        cols.append(np.sin(h * theta))
    return np.stack(cols, axis=1)


def _spline_basis(theta: np.ndarray, n_knot: int) -> np.ndarray:
    """Periodic cubic B-spline basis on n_knot uniform knots (n_knot columns)."""
    u = (theta % (2.0 * math.pi)) / (2.0 * math.pi) * n_knot
    B = np.zeros((len(theta), n_knot))
    for i, v in enumerate(u):
        j = int(math.floor(v)) % n_knot
        t = v - math.floor(v)
        # cubic B-spline weights for the 4 spans touching cell j
        w0 = (1 - t) ** 3 / 6.0
        w1 = (3 * t ** 3 - 6 * t ** 2 + 4) / 6.0
        w2 = (-3 * t ** 3 + 3 * t ** 2 + 3 * t + 1) / 6.0
        w3 = t ** 3 / 6.0
        B[i, (j - 1) % n_knot] += w0
        B[i, j] += w1
        B[i, (j + 1) % n_knot] += w2
        B[i, (j + 2) % n_knot] += w3
    return B


def _second_diff(n: int) -> np.ndarray:
    D = np.zeros((n, n))
    for i in range(n):
        D[i, i] = 2.0
        D[i, (i - 1) % n] -= 1.0
        D[i, (i + 1) % n] -= 1.0
    return D


def fit_radial(theta: np.ndarray, r: np.ndarray, kind: str = "fourier",
               order: int = 4) -> Dict[str, Any]:
    """Ridge least squares for r(theta) on a closed basis. order = harmonics
    (fourier) or number of knots (spline)."""
    theta = np.asarray(theta, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)
    if kind == "fourier":
        B = _fourier_basis(theta, int(order))
        reg = np.eye(B.shape[1]) * RIDGE
        reg[0, 0] = 0.0
    else:
        B = _spline_basis(theta, int(order))
        D = _second_diff(int(order))
        reg = RIDGE * (D.T @ D)
    A = B.T @ B + reg
    try:
        coef = np.linalg.solve(A, B.T @ r)
    except np.linalg.LinAlgError:
        coef = np.linalg.lstsq(A, B.T @ r, rcond=None)[0]
    pred = B @ coef
    return dict(kind=kind, order=int(order), coef=coef,
                rms=float(np.sqrt(np.mean((pred - r) ** 2))))


def predict(coef: np.ndarray, kind: str, order: int,
            theta: np.ndarray) -> np.ndarray:
    if kind == "fourier":
        return _fourier_basis(theta, int(order)) @ coef
    return _spline_basis(theta, int(order)) @ coef


def fit_circle(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    """Algebraic circle fit; supplies the radial model's centre without
    needing the peak order. Well-posed on an arc."""
    A = np.stack([x, y, np.ones_like(x)], axis=1)
    b = x ** 2 + y ** 2
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy = sol[0] / 2.0, sol[1] / 2.0
    rad = math.sqrt(max(sol[2] + cx ** 2 + cy ** 2, 1e-9))
    return float(cx), float(cy), float(rad)


def _tangent_of_curve(coef, kind, order, theta, h=1e-3):
    th = np.asarray(theta, dtype=np.float64)
    r0 = predict(coef, kind, order, th)
    r1 = predict(coef, kind, order, th + h)
    dr = (r1 - r0) / h
    # tangent of r(theta) in polar: (dr cos - r sin, dr sin + r cos)
    tx = dr * np.cos(th) - r0 * np.sin(th)
    ty = dr * np.sin(th) + r0 * np.cos(th)
    return r0, np.arctan2(ty, tx)


def _grow(fit, cx, cy, kind, order, xs, ys, th, max_tan=math.pi):
    """Peaks the fitted curve explains: radial residual AND tangent agreement.

    Assignment by radial residual alone lets any peak at the right radius join,
    including interior texture and the occluded object's contour, which is why
    every generator plateaued at sil_share 0.37-0.74 with coverage still ~0.9.
    The peak's own normal is the only per-peak measurement that says the peak
    lies ALONG this curve rather than merely at its radius.
    """
    rall = np.hypot(xs - cx, ys - cy)
    tall = np.arctan2(ys - cy, xs - cx)
    pred, cth = _tangent_of_curve(fit["coef"], kind, order, tall)
    res = np.abs(rall - pred)
    pk = th + math.pi / 2.0
    dth = np.abs(((pk - cth + math.pi / 2.0) % math.pi) - math.pi / 2.0)
    m = (res <= EPS_R) & (dth <= max_tan)
    return sorted(int(i) for i in np.nonzero(m)[0])


def hypothesis(peaks: Sequence[Dict], xs: np.ndarray, ys: np.ndarray,
               th: np.ndarray, seed: Sequence[int], kind: str = "fourier",
               order: int = 4, iters: int = 4,
               fixed_scale: bool = True) -> Optional[Dict[str, Any]]:
    """Fit one ring hypothesis, grown from `seed` by additive inlier evidence."""
    seed = sorted(set(int(v) for v in seed))
    if len(seed) < 3:
        return None
    cx, cy, rad = fit_circle(xs[seed].astype(float), ys[seed].astype(float))
    if not (rad > 1.0):
        return None
    members = list(seed)
    fit = None
    for _ in range(iters):
        dx = xs[members] - cx
        dy = ys[members] - cy
        rr = np.hypot(dx, dy)
        tt = np.arctan2(dy, dx)
        if len(members) < 5 or np.any(rr < 1e-6):
            break
        fit = fit_radial(tt, rr, kind=kind, order=order)
        # re-centre on the current inliers: an arc-fit centre is biased inward
        # and a biased centre inflates the radial residual of the far side
        cx, cy, rad = fit_circle(xs[members].astype(float),
                                 ys[members].astype(float))
        if not (rad > 1.0):
            break
        # assign every peak the curve explains
        new = _grow(fit, cx, cy, kind, order, xs, ys, th)
        if len(new) < 3:
            break
        if new == members:
            members = new
            break
        members = new
    if fit is None or len(members) < MIN_MEMBERS:
        return None
    return _finalise(xs, ys, th, members, cx, cy, kind, order, fixed_scale)


def _finalise(xs, ys, th, members, cx, cy, kind, order, fixed_scale):
    dx = xs[members] - cx
    dy = ys[members] - cy
    rr = np.hypot(dx, dy)
    tt = np.arctan2(dy, dx)
    rad = float(np.median(rr)) if rr.size else 0.0
    fit = fit_radial(tt, rr, kind=kind, order=order)
    pred, curve_th = _tangent_of_curve(fit["coef"], kind, order, tt)
    res = np.abs(rr - pred)

    # per-peak log-likelihood: negative squared radial residual and negative
    # squared tangent disagreement, in FIXED units (see SCALE_R / SCALE_T).
    s_r = SCALE_R if fixed_scale else max(
        float(np.median(res)) * 1.4826, 0.5)
    peak_th = th[members] + math.pi / 2.0          # peaks carry edge NORMAL
    dth = np.abs(((peak_th - curve_th + math.pi / 2.0) % math.pi)
                 - math.pi / 2.0)
    s_t = SCALE_T if fixed_scale else max(
        float(np.median(dth)) * 1.4826, math.radians(5.0))
    ell = -0.5 * (res / s_r) ** 2 - 0.5 * (dth / s_t) ** 2

    # ------------------------------------------------------------------
    # ADDITIVE EVIDENCE, CORRECTLY SIGNED.
    #
    # The expression above is a PENALTY, not a log-likelihood ratio: it is
    # <= 0 for every peak, so sum(ell) falls with every correct member and the
    # best hypothesis is always the SMALLEST one that clears MIN_MEMBERS
    # (measured: a true 40-peak ring scored -117.9 while a 6-peak accidental
    # ring scored -13.7; the true ring was LAST of 28).
    #
    # The LLR a peak contributes is
    #     log p(residual | this curve)  -  log p(residual | any curve),
    # i.e. how much more likely this peak's residual is under the hypothesis
    # than under the scene's own radial-residual background.  The background is
    # estimated from the peaks that are NOT members (leave-inliers-out, so the
    # hypothesis cannot explain itself) with a fixed bandwidth SCALE_R.  Nothing
    # is tuned: the same fixed scales as above, and the background is a
    # measurement of the scene.
    # ------------------------------------------------------------------
    rall = np.hypot(xs - cx, ys - cy)
    tall = np.arctan2(ys - cy, xs - cx)
    pred_all, curve_th_all = _tangent_of_curve(fit["coef"], kind, order, tall)
    res_all = np.abs(rall - pred_all)
    pk_all = th + math.pi / 2.0                    # peaks carry edge NORMAL
    dth_all = np.abs(((pk_all - curve_th_all + math.pi / 2.0) % math.pi)
                     - math.pi / 2.0)

    keep_out = np.ones(len(xs), dtype=bool)
    keep_out[members] = False
    bg_r = res_all[keep_out]
    bg_t = dth_all[keep_out]

    def _kde(vals, sample, h):
        if sample.size == 0:
            return np.full(len(vals), 1.0 / (2.0 * math.pi * h), dtype=float)
        z = (np.asarray(vals)[:, None] - np.asarray(sample)[None, :]) / h
        k = np.exp(-0.5 * z * z).sum(1)
        return k / (sample.size * h * math.sqrt(2.0 * math.pi))

    norm_r = SCALE_R * math.sqrt(2.0 * math.pi)
    norm_t = SCALE_T * math.sqrt(2.0 * math.pi)
    p_ring_r = np.exp(-0.5 * (res / SCALE_R) ** 2) / norm_r
    p_ring_t = np.exp(-0.5 * (dth / SCALE_T) ** 2) / norm_t
    p_bg_r = np.maximum(_kde(res, bg_r, SCALE_R), 1e-4)
    p_bg_t = np.maximum(_kde(dth, bg_t, SCALE_T), 1e-4)
    ell_rad = np.log(p_ring_r / p_bg_r)
    ell_tan = np.log(p_ring_t / p_bg_t)
    ell_ev = ell_rad + ell_tan

    # shape priors, each in its own natural unit
    ang = np.sort((tt % (2.0 * math.pi)))
    gaps = np.diff(np.r_[ang, ang[0] + 2.0 * math.pi])
    cov = float(gaps.max())                  # largest unobserved sector
    coverage = float(1.0 - cov / (2.0 * math.pi))
    nonuniform = float(gaps.std() / max(gaps.mean(), 1e-9))
    # turning of the fitted curve (should be 2*pi for a simple closed curve)
    ths = np.linspace(0.0, 2.0 * math.pi, 257)
    r_s = predict(fit["coef"], kind, order, ths)
    px, py = r_s * np.cos(ths), r_s * np.sin(ths)
    turn = 0.0
    for i in range(len(ths) - 1):
        a = (px[i], py[i])
        b = (px[i + 1], py[i + 1])
        c = (px[(i + 2) % len(ths)], py[(i + 2) % len(ths)])
        v1 = (b[0] - a[0], b[1] - a[1])
        v2 = (c[0] - b[0], c[1] - b[1])
        turn += math.atan2(v1[0] * v2[1] - v1[1] * v2[0],
                           v1[0] * v2[0] + v1[1] * v2[1])
    prior_res = -float(np.mean((res / EPS_R) ** 2))
    prior_cov = -nonuniform
    prior_turn = -((abs(turn) - 2.0 * math.pi) / (math.pi / 2.0)) ** 2
    # inclusion cost: each additional peak must earn its place.  Rule-first:
    # prior odds of one image peak belonging to one ring, pi = 0.3, measured as
    # (ring circumference in peaks)/(peaks in scene) on CAL; cost =
    # -log(pi/(1-pi)) = 0.847 nats per member.
    incl_cost = -math.log(0.3 / 0.7)
    llr = (float(ell.sum()) + prior_res + prior_cov + prior_turn
           - incl_cost * len(members))
    # Correctly signed additive evidence, with and without the tangent term.
    llr_ev = (float(ell_ev.sum()) + prior_res + prior_cov + prior_turn
              - incl_cost * len(members))
    llr_ev_rad = (float(ell_rad.sum()) + prior_res + prior_cov + prior_turn
                  - incl_cost * len(members))
    return dict(kind=kind, order=int(order), center=[cx, cy], radius=rad,
                members=[int(v) for v in members], n=len(members),
                coverage=coverage, rms=float(np.sqrt(np.mean(res ** 2))),
                s_r=s_r, s_t=s_t, ell=[float(v) for v in ell],
                sum_ell=float(ell.sum()), prior_res=prior_res,
                prior_cov=prior_cov, prior_turn=prior_turn,
                turn_deg=math.degrees(abs(turn)), llr=llr,
                llr_per_peak=llr / len(members),
                ell_rad=[float(v) for v in ell_rad],
                ell_tan=[float(v) for v in ell_tan],
                sum_ell_rad=float(ell_rad.sum()),
                sum_ell_tan=float(ell_tan.sum()),
                llr_ev=llr_ev, llr_ev_rad=llr_ev_rad,
                res=[float(v) for v in res],
                radial=[float(v) for v in tt],
                curve_tangent=[float(v) for v in curve_th],
                fixed_scale=bool(fixed_scale))


def global_hypotheses(xs: np.ndarray, ys: np.ndarray, th: np.ndarray,
                      kind: str = "spline", order: int = 16,
                      n_trials: int = 4000, min_sep: float = 12.0,
                      rng_seed: int = 0, keep: int = 40,
                      dedup: float = 0.7) -> List[Dict[str, Any]]:
    """Generate ring hypotheses GLOBALLY, not from chains.

    Measured motivation: with chain seeds, 7/8 scenes never propose a
    silhouette-spanning hypothesis at all (best share 0.07-0.67), because an arc
    seed on a straight edge or a corner gives a badly conditioned centre.  Ring
    generation must therefore be able to propose a whole closed curve from
    widely separated peaks, independently of how the chains fragmented.

    Deterministic RANSAC over circumcircles of observed peak triples (fixed rng
    seed, no random state leaking into results), refined by least-squares circle
    then the same radial-spline growth used by `hypothesis`.  Every hypothesis
    is fitted to OBSERVED peaks only; nothing is extrapolated.

    The candidate pool is ranked by INLIER COUNT (a geometric quantity) before
    the evidence terms are applied, so the LLR never selects its own seeds.
    """
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    n = len(xs)
    if n < MIN_MEMBERS:
        return []
    rng = np.random.default_rng(rng_seed)

    def circumcircle(a, b, c):
        x1, y1 = xs[a], ys[a]
        x2, y2 = xs[b], ys[b]
        x3, y3 = xs[c], ys[c]
        d = 2.0 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
        if abs(d) < 1e-6:
            return None
        s1, s2, s3 = x1 * x1 + y1 * y1, x2 * x2 + y2 * y2, x3 * x3 + y3 * y3
        ux = (s1 * (y2 - y3) + s2 * (y3 - y1) + s3 * (y1 - y2)) / d
        uy = (s1 * (x3 - x2) + s2 * (x1 - x3) + s3 * (x2 - x1)) / d
        return ux, uy, math.hypot(x1 - ux, y1 - uy)

    pool: List[Tuple[int, float, float, float, Tuple[int, ...]]] = []
    for _ in range(int(n_trials)):
        i, j, k = (int(v) for v in rng.choice(n, 3, replace=False))
        dij = math.hypot(xs[i] - xs[j], ys[i] - ys[j])
        dik = math.hypot(xs[i] - xs[k], ys[i] - ys[k])
        djk = math.hypot(xs[j] - xs[k], ys[j] - ys[k])
        if min(dij, dik, djk) < min_sep:
            continue
        c = circumcircle(i, j, k)
        if c is None:
            continue
        cx, cy, rad = c
        if not (4.0 <= rad <= 400.0):
            continue
        for _ref in range(2):
            res = np.abs(np.hypot(xs - cx, ys - cy) - rad)
            inl = np.nonzero(res <= EPS_R)[0]
            if inl.size < MIN_MEMBERS:
                break
            cx, cy, rad = fit_circle(xs[inl], ys[inl])
        res = np.abs(np.hypot(xs - cx, ys - cy) - rad)
        inl = np.nonzero(res <= EPS_R)[0]
        if inl.size < MIN_MEMBERS:
            continue
        pool.append((int(inl.size), float(cx), float(cy), float(rad),
                     tuple(sorted(int(v) for v in inl))))
    pool.sort(key=lambda t: -t[0])

    out: List[Dict[str, Any]] = []
    for _cnt, cx, cy, _rad, mem in pool:
        members = list(mem)
        if members and any(
                len(set(members) & set(o["members"]))
                / max(1, min(len(members), len(o["members"]))) > dedup
                for o in out):
            continue
        for _it in range(3):
            dx = xs[members] - cx
            dy = ys[members] - cy
            rr = np.hypot(dx, dy)
            tt = np.arctan2(dy, dx)
            if len(members) < 5 or np.any(rr < 1e-6):
                break
            fit = fit_radial(tt, rr, kind=kind, order=order)
            cx, cy, _rad2 = fit_circle(xs[members], ys[members])
            new = _grow(fit, cx, cy, kind, order, xs, ys, th)
            if len(new) < MIN_MEMBERS or new == members:
                members = new if new else members
                break
            members = new
        if len(members) < MIN_MEMBERS:
            continue
        h = _finalise(xs, ys, th, members, cx, cy, kind, order, True)
        if h is None:
            continue
        if any(
                len(set(h["members"]) & set(o["members"]))
                / max(1, min(len(h["members"]), len(o["members"]))) > dedup
                for o in out):
            continue
        out.append(h)
        if len(out) >= keep:
            break
    out.sort(key=lambda h: -h["llr_ev_rad"])
    return out


def envelope_hypotheses(xs: np.ndarray, ys: np.ndarray, th: np.ndarray,
                        centres: Sequence[Tuple[float, float]],
                        kind: str = "spline", order: int = 16,
                        bins: int = 48, dedup: float = 0.7,
                        min_normal: float = 0.30) -> List[Dict[str, Any]]:
    """Ring hypotheses from the OUTER RADIAL ENVELOPE around a centre.

    Measured motivation: circumcircle seeding only finds rings whose front
    silhouette is a circle (3/8 here: every sphere-front scene).  A box or
    cylinder silhouette is not a circle, so NO circumcircle collects its peaks
    and the true ring is never proposed (max share 0.37-0.74 with coverage
    still 0.74-0.94 -- the curve spans the object but only half its members are
    on it).

    This generator makes no circularity assumption.  Around each candidate
    centre it takes the OUTERMOST peak in each angular bin whose edge normal is
    radial-ish (|cos(theta - phi)| > min_normal; peaks already carry the
    normal), fits the radial spline to those observed peaks, then grows the set
    by radial residual exactly as the other generators do.  The outermost
    visible peak per direction is the structural prior: the front object bounds
    the visible region, so its silhouette is the outer envelope.

    Nothing is completed: bins with no observed peak contribute no points and
    the ridge prior carries them; a hypothesis with too few observed members is
    rejected.
    """
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    n = len(xs)
    out: List[Dict[str, Any]] = []
    for cx, cy in centres:
        phi = np.arctan2(ys - cy, xs - cx)
        rho = np.hypot(xs - cx, ys - cy)
        radial = np.abs(np.cos(th - phi))
        bidx = np.floor((phi + math.pi) / (2.0 * math.pi) * bins).astype(int) % bins
        pick = []
        for b in range(bins):
            m = np.nonzero(bidx == b)[0]
            if m.size == 0:
                continue
            good = m[radial[m] >= min_normal]
            cand = good if good.size else m
            pick.append(int(cand[np.argmax(rho[cand])]))
        pick = sorted(set(pick))
        if len(pick) < MIN_MEMBERS:
            continue
        members = list(pick)
        for _it in range(4):
            rr = rho[members]
            tt = phi[members]
            if len(members) < 5 or np.any(rr < 1e-6):
                break
            fit = fit_radial(tt, rr, kind=kind, order=order)
            new = _grow(fit, cx, cy, kind, order, xs, ys, th)
            if len(new) < MIN_MEMBERS:
                break
            if new == members:
                break
            members = new
        if len(members) < MIN_MEMBERS:
            continue
        h = _finalise(xs, ys, th, members, cx, cy, kind, order, True)
        if h is None:
            continue
        if any(
                len(set(h["members"]) & set(o["members"]))
                / max(1, min(len(h["members"]), len(o["members"]))) > dedup
                for o in out):
            continue
        out.append(h)
    out.sort(key=lambda h: -h["llr_ev_rad"])
    return out


def trace_hypotheses(xs: np.ndarray, ys: np.ndarray, th: np.ndarray,
                     centres: Sequence[Tuple[float, float]],
                     kind: str = "spline", order: int = 16, bins: int = 64,
                     dedup: float = 0.7) -> List[Dict[str, Any]]:
    """Ring hypotheses from a GLOBAL closed-path trace in polar coordinates.

    Why this generator: circumcircle seeding finds only circular silhouettes
    (3/8); the outer-envelope generator is worse because the outermost peak per
    direction is often on the OCCLUDED object.  What distinguishes the front
    silhouette from every other contour is that it is one CONTINUOUS CLOSED
    curve: radius changes slowly with angle and the edge normal is radial.

    So the hypothesis is the minimum-cost closed path through the observed
    peaks, one per angular bin:

        cost = sum_bins [ W * (1 - cos^2(theta_b - phi_b))      (normal is radial)
                          + |r_b - r_{b-1}| ]                   (curve is smooth)

    solved exactly by dynamic programming around the circle (the start bin's
    candidate is enumerated).  This is ONE global hypothesis over a SET of
    peaks -- not a per-link gate: there is no threshold on any single edge, and
    the back object's visible arcs are rejected because they must JUMP where
    the front object occludes them, which the path cost forbids.

    W = EPS_R is the only constant: "the peak's normal is not radial at all"
    costs the same as one residual tolerance of radial jump.

    The selected path is then refitted with the radial spline and grown by
    radial residual, exactly like the other generators.
    """
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    n = len(xs)
    out: List[Dict[str, Any]] = []
    W_NORMAL = EPS_R
    for cx, cy in centres:
        phi = np.arctan2(ys - cy, xs - cx)
        rho = np.hypot(xs - cx, ys - cy)
        radial = np.abs(np.cos(th - phi))
        bidx = np.floor((phi + math.pi) / (2.0 * math.pi) * bins).astype(int) % bins
        cand: List[List[int]] = []
        bin_of: List[int] = []
        for b in range(bins):
            m = [int(v) for v in np.nonzero(bidx == b)[0]]
            if m:
                cand.append(m)
                bin_of.append(b)
        if len(cand) < MIN_MEMBERS:
            continue

        emit = [W_NORMAL * (1.0 - radial[c] ** 2) for c in cand]

        def run(start_choice: int):
            # dp[j] = cost of best path ending at candidate j of the current bin
            prev_idx = cand[0]
            dp = emit[0].copy()
            back: List[List[int]] = []
            for b in range(1, len(cand)):
                cur_idx = cand[b]
                trans = np.abs(rho[cur_idx][None, :] - rho[prev_idx][:, None])
                # transition between bins that are adjacent in ANGLE gets the
                # radial-jump cost; a skipped (empty) bin costs nothing extra
                # because the ridge prior, not an observed peak, carries it.
                tot = dp[:, None] + trans
                arg = np.argmin(tot, axis=0)
                dp = tot[arg, np.arange(len(cur_idx))] + emit[b]
                back.append([int(v) for v in arg])
                prev_idx = cur_idx
            # close the loop back to the start choice
            close = np.abs(rho[cand[0][start_choice]] - rho[cand[-1]])
            tot_close = dp + close
            end = int(np.argmin(tot_close))
            path = [end]
            for b in range(len(cand) - 1, 0, -1):
                end = back[b - 1][path[-1]]
                path.append(end)
            path.reverse()
            return float(tot_close[int(path[-1])]), path

        best = None
        for s in range(len(cand[0])):
            total, path = run(s)
            if best is None or total < best[0]:
                best = (total, path)
        if best is None:
            continue
        members = sorted(int(cand[b][j]) for b, j in enumerate(best[1]))

        for _it in range(4):
            rr = rho[members]
            tt = phi[members]
            if len(members) < 5 or np.any(rr < 1e-6):
                break
            fit = fit_radial(tt, rr, kind=kind, order=order)
            new = _grow(fit, cx, cy, kind, order, xs, ys, th)
            if len(new) < MIN_MEMBERS:
                break
            if new == members:
                break
            members = new
        if len(members) < MIN_MEMBERS:
            continue
        h = _finalise(xs, ys, th, members, cx, cy, kind, order, True)
        if h is None:
            continue
        if any(
                len(set(h["members"]) & set(o["members"]))
                / max(1, min(len(h["members"]), len(o["members"]))) > dedup
                for o in out):
            continue
        out.append(h)
    out.sort(key=lambda h: -h["llr_ev_rad"])
    return out


def refit_set(xs: np.ndarray, ys: np.ndarray, th: np.ndarray,
               members: Sequence[int], kind: str = "spline", order: int = 16,
               iters: int = 4) -> Optional[Dict[str, Any]]:
    """Refit one global curve to a candidate SET of peaks, then grow it.

    Used to assemble a ring from several arc hypotheses: a box or cylinder
    silhouette is covered by 3-4 chain fragments, each of which alone gives an
    ill-conditioned centre, so growth from any one fragment cannot reach the
    far side.  The union of two fragments spans enough of the curve to be
    well-posed, after which the ordinary radial growth completes the set from
    OBSERVED peaks only.
    """
    members = sorted(set(int(v) for v in members))
    if len(members) < MIN_MEMBERS:
        return None
    cx, cy, rad = fit_circle(xs[members].astype(float), ys[members].astype(float))
    if not (rad > 1.0):
        return None
    fit = None
    for _ in range(iters):
        rr = np.hypot(xs[members] - cx, ys[members] - cy)
        tt = np.arctan2(ys[members] - cy, xs[members] - cx)
        if len(members) < 5 or np.any(rr < 1e-6):
            break
        fit = fit_radial(tt, rr, kind=kind, order=order)
        new = _grow(fit, cx, cy, kind, order, xs, ys, th)
        if len(new) < MIN_MEMBERS:
            break
        if new == members:
            break
        members = new
    if fit is None or len(members) < MIN_MEMBERS:
        return None
    return _finalise(xs, ys, th, members, cx, cy, kind, order, True)


def merge_hypotheses(xs: np.ndarray, ys: np.ndarray, th: np.ndarray,
                     hs: Sequence[Dict[str, Any]], kind: str = "spline",
                     order: int = 16, dedup: float = 0.7,
                     max_union: int = 3) -> List[Dict[str, Any]]:
    """Assemble rings by merging 2..max_union candidate member sets and
    refitting the global curve.  Deterministic (pairs in list order)."""
    sets = [tuple(sorted(h["members"])) for h in hs]
    seen_sets = set(sets)
    out: List[Dict[str, Any]] = []
    combos = []
    n = len(sets)
    for i in range(n):
        combos.append((i,))
        for j in range(i + 1, n):
            combos.append((i, j))
            for k in range(j + 1, n):
                if max_union >= 3:
                    combos.append((i, j, k))
    for combo in combos:
        mem = sorted({v for idx in combo for v in sets[idx]})
        key = tuple(mem)
        if key in seen_sets:
            continue
        seen_sets.add(key)
        h = refit_set(xs, ys, th, mem, kind=kind, order=order)
        if h is None:
            continue
        if any(
                len(set(h["members"]) & set(o["members"]))
                / max(1, min(len(h["members"]), len(o["members"]))) > dedup
                for o in out):
            continue
        out.append(h)
    out.sort(key=lambda h: -h["llr_ev_rad"])
    return out


def ring_hypotheses(peaks, xs, ys, th, seeds, kind="fourier", order=4,
                    dedup=0.7, fixed_scale: bool = True) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for seed in seeds:
        h = hypothesis(peaks, xs, ys, th, seed, kind=kind, order=order,
                       fixed_scale=fixed_scale)
        if h is None:
            continue
        ms = set(h["members"])
        dup = False
        for o in out:
            inter = len(ms & set(o["members"]))
            if inter / max(1, min(len(ms), len(o["members"]))) > dedup:
                dup = True
                break
        if not dup:
            out.append(h)
    out.sort(key=lambda h: -h["llr"])
    return out
