"""perception/lenses/derived_base.py -- extracted from derived.py
(Move 3.5).

MOVE ONLY: whole top-level statements, verbatim.
"""
from __future__ import annotations

import math

import numpy as np
from typing import Any, Dict, List, Optional, Tuple
import parameters as P
from core.hashing import (  # noqa: F401
    clamp,
    hash_64,
    serialize_int,
    serialize_tuple,
    serialize_value,
)
from core.quantize import (  # noqa: F401
    q_linear,
)
from perception.lenses.derived_common import (  # noqa: F401
    compute_endstopped_map,
)
from core.types import (  # noqa: F401
    Detection,
    TypedSignature,
)
from core.registries import population_from_values  # noqa: F401

class ContourGainState:
    """B.14 recurrent contour grouping.

    Per-anchor gain that accumulates across steps from contour membership
    and decays otherwise. The Pipeline reads `.gain` and threads it into
    the Orientation lens as `gain_by_anchor`, so a repeatedly-contoured
    anchor gets a wider receptive field. Constructed in Step 4, wired in
    Step 9.

    Bounded on both axes: values saturate at MAX_CONTOUR_GAIN_VALUE, and
    the table is trimmed to MAX_CONTOUR_GAIN_ENTRIES by evicting the
    SMALLEST gains (ties broken on canonical bytes, so eviction is
    deterministic rather than insertion-order dependent).
    """

    def __init__(self) -> None:
        self.gain: Dict[Any, float] = {}

    def decay(self) -> None:
        for a in list(self.gain.keys()):
            self.gain[a] *= P.CONTOUR_GAIN_DECAY
            if self.gain[a] < P.CONTOUR_GAIN_FLOOR:
                del self.gain[a]

    def bounded_increment(self, anchor: Any) -> None:
        self.gain[anchor] = min(
            P.MAX_CONTOUR_GAIN_VALUE,
            self.gain.get(anchor, 0.0) + P.CONTOUR_GAIN_INCREMENT,
        )
        if len(self.gain) > P.MAX_CONTOUR_GAIN_ENTRIES:
            ordered = sorted(
                self.gain.items(),
                key=lambda kv: (kv[1], serialize_value(kv[0])),
            )
            for k, _ in ordered[: len(self.gain) - P.MAX_CONTOUR_GAIN_ENTRIES]:
                del self.gain[k]

class HigherLens:
    """Base class for all higher lenses. Each subclass initializes its
    per-instance counters in __init__ and implements step()."""

    def __init__(self) -> None:
        self._emitted_total: int = 0
        self._skipped_no_position: int = 0
        self._skipped_no_content: int = 0
        self._skipped_no_color: int = 0
        self._skipped_no_size: int = 0
        self._skipped_no_component: int = 0
        # [PHASE F] Content detections excluded from Part pooling by the
        # BorderOwnership figure-ground gate (ground side of a boundary).
        self._skipped_ground_side: int = 0
        # [PHASE B] Step 5. Candidates that passed every structural pre-check
        # and so reached the signature/emission stage. This is the
        # diagnostic that distinguishes "the lens found nothing to work
        # with" (candidates == 0) from "the lens found candidates and threw
        # all of them away" (candidates > 0, emitted == 0) — the second is a
        # bug, the first is usually just a quiet scene.
        self._candidates_evaluated: int = 0

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
        update_homeostasis: bool = True,
        object_envelope: Optional[
            Tuple[float, float, float, float]
        ] = None,
        scale_level: int = 0,
    ) -> List[Detection]:
        raise NotImplementedError

    def get_diagnostics(self) -> Dict[str, int]:
        return {
            "emitted_total": self._emitted_total,
            "skipped_no_position": self._skipped_no_position,
            "skipped_no_content": self._skipped_no_content,
            "skipped_no_color": self._skipped_no_color,
            "skipped_no_size": self._skipped_no_size,
            "skipped_no_component": self._skipped_no_component,
            "skipped_ground_side": self._skipped_ground_side,
        }

class EndStoppedLens(HigherLens):
    """[D1] Hypercomplex (End-Stopped) Lens.
    Emits ENDSTOPPED detections at termination peaks."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._emitted_total = 0
        self._candidates_evaluated = 0

    def step(
        self,
        t: int,
        concept_graph: Any = None,
        sig_provider: Any = None,
        anchor_positions: Optional[Dict[Any, Tuple[float, ...]]] = None,
        anchor_velocities: Optional[Dict[Any, Tuple[float, ...]]] = None,
        current_detections: Optional[Dict[str, List[Detection]]] = None,
        extraction_scale: float = 0.0,
        envelope_mask: Optional[Tuple[Tuple[float, float, float, float], ...]] = None,
        observation: Any = None,
        **kwargs: Any,
    ) -> List[Detection]:
        if observation is None:
            return []

        Eg = getattr(observation, "_gabor_complex", None)
        if Eg is None:
            return []

        thetas = getattr(observation, "_gabor_thetas", None)
        sfs = getattr(observation, "_gabor_sfs", None)
        if thetas is None or sfs is None:
            return []

        from scipy.ndimage import maximum_filter

        gmax = float(Eg.max()) if Eg.size > 0 else 0.0
        if gmax < 1e-6:
            return []

        rel_s, rel_d, side = compute_endstopped_map(
            Eg, thetas, sfs, kappa=P.ENDSTOP_FLANK_KAPPA
        )

        energy_ok = Eg >= P.ENDSTOP_ENERGY_FRAC * gmax
        rel_best = np.where(rel_d >= P.ENDSTOP_DOUBLE_RATIO * rel_s, rel_d, rel_s)
        win = np.max(np.where(energy_ok, rel_best, -np.inf), axis=(0, 1))

        L_px = max(1.0, P.ENDSTOP_FLANK_KAPPA * 3.0)
        size = max(3, int(round(P.ENDSTOP_NMS_MULT * L_px)))
        peaks = (win >= maximum_filter(win, size=size)) & energy_ok.any(axis=(0, 1))

        rim = int(round(P.ENDSTOP_RIM_FRAC * L_px))
        if rim > 0:
            peaks[:rim, :] = False
            peaks[-rim:, :] = False
            peaks[:, :rim] = False
            peaks[:, -rim:] = False

        emitted: List[Detection] = []
        n_theta = P.ORIENTATION_N_THETA
        nsf = len(sfs)
        ys, xs = np.nonzero(peaks)
        self._candidates_evaluated += len(ys)
        for y, x in zip(ys.tolist(), xs.tolist()):
            channel = int(np.argmax(rel_best[:, :, y, x]))
            it, isf = divmod(channel, rel_best.shape[1])
            rel_s_v = float(rel_s[it, isf, y, x])
            rel_d_v = float(rel_d[it, isf, y, x])
            strength = max(rel_s_v, rel_d_v)
            if rel_d_v >= P.ENDSTOP_DOUBLE_RATIO * rel_s_v:
                ends = 0
            else:
                ends = 2 if int(side[it, isf, y, x]) > 0 else 1

            theta_bin = int(round((it / max(len(thetas), 1)) * n_theta)) % n_theta
            sf_bin = q_linear(float(isf), 8, 0.0, float(max(nsf - 1, 1)))
            strength_bin = q_linear(clamp(strength, 0.0, 1.0), 8, 0.0, 1.0)

            syn = hash_64(
                serialize_tuple([
                    serialize_int(y),
                    serialize_int(x),
                    serialize_int(it),
                    serialize_int(isf),
                ])
            )
            if anchor_positions is not None and isinstance(anchor_positions, dict):
                anchor_positions[syn] = (float(x), float(y))

            _es_slots = (
                float(theta_bin),
                float(sf_bin),
                float(ends),
                float(strength_bin),
            )
            # [GRADED POPULATION] Built from the CONTINUOUS measurements: the
            # terminator's orientation (in radians), the scale index
            # (normalised), the SIGNED open side (-1 open toward -a, +1
            # toward +a, 0 no end) and the end-stop strength. The bins stay
            # the lookup key.
            _es_pop = population_from_values(
                "ENDSTOPPED",
                ((it / max(len(thetas), 1)) * math.pi,
                 (isf / max(nsf - 1, 1)),
                 (0.0 if ends == 0 else (1.0 if ends == 2 else -1.0)),
                 float(clamp(strength, 0.0, 1.0))))
            emitted.append(
                Detection(
                    k="EndStopped",
                    source_layer=0,
                    scale=0,
                    sigma=TypedSignature(
                        kind="CANONICAL",
                        class_="ENDSTOPPED",
                        relation="LINE_TERMINATION",
                        quantized=_es_slots,
                        population=_es_pop,
                    ),
                    alpha=syn,
                    z=strength,
                    t=t,
                    rho=(("entity", syn, 0, "+"),),
                )
            )
        self._emitted_total += len(emitted)
        return emitted

    def get_diagnostics(self) -> Dict[str, int]:
        return {
            "candidates_evaluated": self._candidates_evaluated,
            "emitted_total": self._emitted_total,
        }

