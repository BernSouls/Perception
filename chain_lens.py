"""perception/lenses/chain_lens.py -- [A5] ChainLens.

Groups the continuous contour field's peaks into chains via
`perception.chain_formation.form_chains` and emits one PEAK_CHAIN detection
per chain, with rho roles linking it to its member CONTOUR_PEAKs.

Why this exists
---------------
`form_chains` returns (indices, is_ring) tuples. A bare tuple is not visible
to the graph: this lens emits a canonical chain detection with explicit
membership roles. PeakLens owns member peak detections; the Binder resolves
those roles into containment edges (container -> member).

Anchor scheme
-------------
Peak and chain anchors are computed from their index in the field output, so
any lens can reference one without a lookup table:

    peak  i            -> PEAK_ANCHOR_BASE + i
    chain ordinal k    -> PEAK_ANCHOR_BASE + CHAIN_ANCHOR_OFFSET + k

PEAK_ANCHOR_BASE is far above any pipeline-internal anchor counter, so the
field's anchors cannot collide with region anchors or binder synthesised
anchors.

Signature (arity 6, registered in core/registries.py)
----------------------------------------------------
    0 n_peaks_bin          graded  0..15   min(15, len(members))
    1 mean_theta_bin       circular 12     circular mean of member thetas
    2 theta_var_bin        graded  0..7    1 - |R_bar| bounded circular spread
    3 closure_bin          binary          from form_chains' ring flag
    4 curvature_summary_bin graded 0..7    mean |signed_curvature|
    5 length_bin           graded  0..7    total length / (8 * median_wavelength)

Slot 1 is circular: CONTOUR_PEAK's theta axis is pi-periodic, so the mean is
computed on doubled angles and folded back, and the bin is taken modulo 12 to
satisfy the registry's N/2 range convention (range 6.0 over cycle 12).
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import parameters as P
from core.types import (  # noqa: F401
    Detection,
    TypedSignature,
)
from perception.chain_formation import form_chains
from perception.lenses.peak_lens import field_for_observation
from perception.lenses.derived_base import (  # noqa: F401
    HigherLens,
)

K_CHAIN = "PEAK_CHAIN"
CLASS_CHAIN = "PEAK_CHAIN"
RELATION_CHAIN = "CHAIN_PROFILE"

# arity-6 slot bin counts
N_PEAKS_BINS = 16        # slot 0 max, clipped to 15
N_THETA_BINS = 12        # slot 1: circular bins over [0, pi)
N_VAR_BINS = 8           # slot 2 max, clipped to 7
N_CURV_BINS = 8          # slot 4 max, clipped to 7
N_LEN_BINS = 8           # slot 5 max, clipped to 7

# length normalisation: a full chain of 8 wavelengths fills the top bin.
LEN_FULL_WAVELENGTHS = 8.0


def _circular_mean_theta(thetas: List[float]) -> Optional[float]:
    """Circular mean of orientations on the pi-periodic axis, in [0, pi).

    Doubling maps each orientation onto the 2*pi circle, where an ordinary
    mean is well defined, then halving folds back to the orientation axis.
    Returns None when no member has a usable theta.
    """
    if not thetas:
        return None
    s = sum(math.sin(2.0 * t) for t in thetas)
    c = sum(math.cos(2.0 * t) for t in thetas)
    if abs(s) < 1e-12 and abs(c) < 1e-12:
        return None
    mean = 0.5 * math.atan2(s, c)
    return mean % math.pi


def _circular_spread_bin(thetas: List[float]) -> int:
    """Circular spread of orientations mapped to 0..7.

    Uses the resultant length R_bar of the doubled angles: R_bar == 1 means
    all members agree, 0 means uniformly spread. `1 - R_bar` is therefore a
    0..1 concentration-to-uniform measure, which is what the bin encodes.
    Falls back to 0 (tightly grouped) for a degenerate empty-or-opposing set.
    """
    if not thetas:
        return 0
    s = sum(math.sin(2.0 * t) for t in thetas) / len(thetas)
    c = sum(math.cos(2.0 * t) for t in thetas) / len(thetas)
    r_bar = math.hypot(s, c)
    spread = 1.0 - r_bar
    if not math.isfinite(spread):
        return 0
    spread = min(1.0, max(0.0, spread))
    q = int(spread * N_VAR_BINS)
    return min(N_VAR_BINS - 1, max(0, q))



def _clip_bin(value: Any, n_bins: int, lo: float = 0.0,
              hi: float = 1.0) -> int:
    """Clip a float into [lo, hi] and bin it into n_bins slots."""
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        return 0
    v = min(hi, max(lo, float(value)))
    q = int(math.floor((v - lo) / ((hi - lo) / float(n_bins))))
    return min(n_bins - 1, max(0, q))


def _peak_anchor(index: int) -> int:
    return int(P.PEAK_ANCHOR_BASE) + int(index)


def _chain_anchor(ordinal: int) -> int:
    return int(P.PEAK_ANCHOR_BASE) + int(P.CHAIN_ANCHOR_OFFSET) + int(ordinal)


class ChainLens(HigherLens):
    """[A5] Field peak chains. Emits PEAK_CHAIN detections.

    Reads the continuous contour field (peaks + end-stop), groups its peaks
    into chains with perception.chain_formation.form_chains, and emits one
    PEAK_CHAIN per chain with member roles pointing at the member peaks'
    computed anchors. Members are consumed by the Binder's
    `_run_peak_chain_orientation`, which turns them into CONTOUR_ORIENTATION
    containment edges.
    """

    def __init__(self) -> None:
        super().__init__()
        self._candidates_evaluated = 0
        self._field_rebuilt = 0
        self._no_luminance = 0
        self._field_missing = 0

    def get_diagnostics(self) -> Dict[str, int]:
        return {
            "emitted_total": self._emitted_total,
            "candidates_evaluated": self._candidates_evaluated,
            "field_rebuilt": self._field_rebuilt,
            "field_missing": self._field_missing,
        }

    # -- field access -----------------------------------------------------
    def _field_output(self, observation: Any,
                      luminance: Optional[Any] = None) -> Optional[Dict[str, Any]]:
        """Consume V1's cached field, or compute one on direct invocation."""
        cached = (getattr(observation, "_contour_field_cache", None)
                  if observation is not None else None)
        if not isinstance(cached, dict):
            self._field_rebuilt += 1
        field = field_for_observation(observation, luminance)
        if field is None:
            self._field_missing += 1
        return field

    @staticmethod
    def _median_wavelength(field: Dict[str, Any]) -> Optional[float]:
        """Median wavelength of the field's spatial-frequency band."""
        sfs = field.get("sfs")
        if sfs is None:
            return None
        arr = np.asarray(sfs, dtype=np.float64).ravel()
        if arr.size == 0:
            return None
        wavelengths = 1.0 / np.maximum(arr, 1e-6)
        wl = float(np.median(wavelengths))
        if not math.isfinite(wl) or wl <= 0.0:
            return None
        return wl

    # -- HigherLens API ---------------------------------------------------
    def step(
        self,
        t: int,
        concept_graph: Any,
        sig_provider: Any,
        anchor_positions: Dict[Any, Tuple[float, ...]],
        anchor_velocities: Optional[Dict[Any, Tuple[float, ...]]],
        current_detections: Dict[str, List[Detection]],
        extraction_scale: float = 0.0,
        envelope_mask: Optional[
            Tuple[Tuple[float, float, float, float], ...]
        ] = None,
        observation: Optional[Any] = None,
        luminance: Optional[Any] = None,
        **kwargs: Any,
    ) -> List[Detection]:
        if observation is None and luminance is None:
            # Nothing to extract from: neither an observation to inspect
            # nor an image plane. Explicit luminance is sufficient on its
            # own so callers on the region-construction path (which discard
            # the image) can still drive the lens.
            return []

        field = self._field_output(observation, luminance=luminance)
        if field is None:
            return []

        peaks: List[Dict[str, Any]] = list(field.get("peaks") or [])
        if not peaks:
            return []

        median_wl = self._median_wavelength(field)
        if median_wl is None:
            return []

        endstop = np.asarray(field["endstop"])
        if endstop.ndim != 4:
            return []

        try:
            # [Q1] The geometric, corner, end-stop and link-support rules all
            # come from the field the peaks were extracted from, so the
            # integration array and the spatial frequencies travel with the
            # call. Without them the support rule is skipped and recall is
            # whatever the geometric rules alone give.
            # [AF RE-DERIVATION, Round 6] Link support reads the DRIVING
            # (pre-facilitation) field: the association field is modulatory and
            # must not inflate the support statistic, which is a ratio of
            # evidence along the segment to endpoint evidence.
            support_field = field.get("integration_driving")
            if support_field is None:
                support_field = field["integration"]
            # [C] The CONTINUOUS orientation per peak, read from the
            # field's own energy envelope. It goes to the chain's
            # ALIGNMENT rule (so the 35-degree tolerance is applied to a
            # graded angle, not a 15-degree argmax) and to the chain's
            # circular mean and spread.
            orientations = None
            try:
                from perception.chain_formation import (
                    continuous_orientations)
                _cplx = field.get("complex")
                if _cplx is not None:
                    # [B38] `sfs` selects the field's median-wavelength channel:
                    # the peak's own `sf_idx` is the argmax of the RAW
                    # integration over scales and is the coarsest channel for
                    # every peak, where a neighbouring structure inside the
                    # filter's support decides the orientation.
                    orientations = continuous_orientations(
                        np.asarray(_cplx, dtype=np.float64), peaks,
                        sfs=np.asarray(field["sfs"], dtype=np.float64))
            except Exception:
                orientations = None
            chains = form_chains(peaks, endstop, median_wl,
                                 integration=np.asarray(support_field),
                                 sfs=np.asarray(field["sfs"],
                                                dtype=np.float64),
                                 orientations=orientations)
        except ValueError:
            # form_chains raises on a non-positive median wavelength or a
            # malformed end-stop shape; both mean "no usable field".
            return []

        emitted: List[Detection] = []

        # Peak anchor positions are owned by PeakLens, which emits the
        # CONTOUR_PEAK detections and populates anchor_positions itself.
        # This lens READS them (through member_positions in
        # perception.chain_polylines) rather than writing them: a second
        # writer would silently mask an ordering or anchor bug in the owner.

        # CONTOUR_PEAK detections are owned by the earlier PeakLens.
        for ordinal, (members, is_ring) in enumerate(chains):
            self._candidates_evaluated += 1
            if not members:
                continue

            if orientations is not None and len(orientations) == len(peaks):
                member_thetas = [float(orientations[i]) for i in members]
            else:
                member_thetas = [float(peaks[i].get("theta", 0.0))
                                 for i in members]
            slot0 = float(min(15, len(members)))

            # Slot 1: circular mean folded onto the 12-bin orientation axis.
            mean_theta = _circular_mean_theta(member_thetas)
            slot1 = float(int(math.floor(mean_theta * N_THETA_BINS / math.pi))
                          % N_THETA_BINS) if mean_theta is not None else 0.0

            # Slot 2: circular spread (0 = all members agree).
            slot2 = float(_circular_spread_bin(member_thetas))

            # Slot 3: closure flag from form_chains.
            # [P1-7] THE CLOSURE IS TOPOLOGICAL, NOT A GRADIENT. "Open" and
            # "closed" are two states of a property, not two points on a
            # scale: there is no quantity between them, so the slot stays a
            # FLAG (0 open, 1 closed) and is never dressed up as a
            # gradient. The graded companion of a chain is the ENSEMBLE's
            # evidence (the member orientation distribution, the member
            # curvature mixture, the normalised length), which the
            # population carries; the closure is a separate, binary fact
            # that the population layout (core.registries.POPULATION_AXES,
            # owned by core) does not declare an axis for.
            slot3 = 1.0 if is_ring else 0.0

            # Slot 4: mean |signed curvature| over members.
            curv_vals = []
            for i in members:
                v = peaks[i].get("signed_curvature", 0.0)
                if isinstance(v, (int, float)) and math.isfinite(float(v)):
                    curv_vals.append(abs(float(v)))
            curv_mean = (sum(curv_vals) / len(curv_vals)) if curv_vals else 0.0
            slot4 = float(_clip_bin(curv_mean, N_CURV_BINS, 0.0, 1.0))

            # Slot 5: total chain length, in units of 8 wavelengths.
            length = 0.0
            for a, b in zip(members, members[1:]):
                pa, pb = peaks[a], peaks[b]
                length += math.hypot(float(pa.get("x", 0)) - float(pb.get("x", 0)),
                                     float(pa.get("y", 0)) - float(pb.get("y", 0)))
            len_frac = length / (LEN_FULL_WAVELENGTHS * median_wl)
            slot5 = float(_clip_bin(len_frac, N_LEN_BINS, 0.0, 1.0))

            # [POPULATION CODING] The ensemble is a break-stable DISTRIBUTION of
            # evidence, not one collapsed number: the orientation axis holds the
            # mixture of member orientations (so a spread-out chain is faithful),
            # the curvature axis the mixture of member |curvature|, and the length
            # axis a bump on the normalised length. Bins above stay the index.
            from core.population import soft_bump_bins, mixture_bins
            from core.registries import population_axis
            # [U22] distributions stay distributions (a mixture per axis is
            # the right construct for an ensemble); what changes is that each
            # axis is fetched BY NAME, not by its index in the layout.
            _ax_theta = population_axis(CLASS_CHAIN, "theta_dist")
            _ax_curv = population_axis(CLASS_CHAIN, "curvature")
            _ax_len = population_axis(CLASS_CHAIN, "length")
            if _ax_theta is None or _ax_curv is None or _ax_len is None:
                population = ()
            else:
                theta_centers = [((t % math.pi) / math.pi) * N_THETA_BINS
                                 for t in member_thetas]
                curv_centers = [min(1.0, v) * N_CURV_BINS for v in curv_vals]
                population = (
                    mixture_bins(theta_centers, _ax_theta),
                    mixture_bins(curv_centers, _ax_curv),
                    soft_bump_bins(min(1.0, max(0.0, len_frac)) * N_LEN_BINS,
                                   _ax_len),
                )

            sigma = TypedSignature(
                kind="CANONICAL",
                class_=CLASS_CHAIN,
                relation=RELATION_CHAIN,
                quantized=(slot0, slot1, slot2, slot3, slot4, slot5),
                population=population,
            )

            anchor = _chain_anchor(ordinal)

            # Member roles: the anchor scheme is index-derived, so the role
            # target is computable by any consumer without a lookup table.
            rho: Tuple[Tuple[str, Any, int, str], ...] = (
                ("entity", anchor, 0, "+"),
            )
            rho += tuple(
                ("member", _peak_anchor(i), k, "+")
                for k, i in enumerate(members)
            )

            if isinstance(anchor_positions, dict):
                xs = [float(peaks[i].get("x", 0.0)) for i in members]
                ys = [float(peaks[i].get("y", 0.0)) for i in members]
                anchor_positions[anchor] = (
                    sum(xs) / len(xs), sum(ys) / len(ys),
                )

            emitted.append(
                Detection(
                    k=K_CHAIN,
                    source_layer=1,
                    scale=0,
                    sigma=sigma,
                    alpha=anchor,
                    z=1.0,
                    t=t,
                    rho=rho,
                )
            )

        self._emitted_total += len(emitted)
        return emitted


if __name__ == "__main__":
    import sys

    def _selftest() -> int:
        import perception.observation_2d as obs_mod

        img = np.zeros((64, 64, 3), dtype=np.float32)
        for i in range(8, 56):
            img[32, i, 0] = 1.0
        obs = obs_mod.Observation2D.from_image(
            img, contours=None, rich=False
        )
        lens = ChainLens()
        dets = lens.step(0, None, None, {}, None, {}, observation=obs)
        print("emitted=%d diagnostics=%s"
              % (len(dets), lens.get_diagnostics()))
        for d in dets[:3]:
            print(" ", d.sigma.quantized, "rho=%d" % len(d.rho))
        return 0

    sys.exit(_selftest())
