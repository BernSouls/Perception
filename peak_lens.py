"""V1 peak lens. Owns the contour-field cache and emits CONTOUR_PEAK only.

Peak i in the field's original order has anchor PEAK_ANCHOR_BASE + i. The
field's order is never re-sorted when allocating anchors: chain membership
indexes that same list. Position is stored outside the arity-8 signature.
"""
from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import parameters as P
from core.types import Detection
from perception.contour_field import field_from_luminance, field_fingerprint
from perception.contour_peak import peaks_to_contour_peak_detections
from perception.lenses.derived_base import HigherLens


def _field_key(lum: np.ndarray) -> Tuple:
    """(shape, content hash, parameter fingerprint) for the field cache.

    [BUG-03 / A6] The observation-attached cache must not hand a grayscale run
    the field built from a colour run (or vice versa), so the key folds in the
    exact input luminance, not just observation identity.
    """
    h = hashlib.blake2b(np.ascontiguousarray(lum).tobytes(), digest_size=16).digest()
    return (lum.shape, h, field_fingerprint())


def field_for_observation(observation: Any, luminance: Any = None) -> Optional[Dict]:
    """Share one field per observation between V1 PeakLens and V2 ChainLens."""
    cached = None
    if observation is not None:
        c = getattr(observation, "_contour_field_cache", None)
        if isinstance(c, dict) and "peaks" in c and "endstop" in c:
            cached = c
    if luminance is None and observation is not None:
        getter = getattr(observation, "get_luminance", None)
        if callable(getter):
            luminance = getter()
    lum = None
    if luminance is not None:
        arr = np.asarray(luminance)
        if arr.ndim == 3:
            from perception.filters import luminance_adapt, opponent_channels
            arr = luminance_adapt(opponent_channels(arr.astype(np.float32))[..., 0])
        if arr.ndim == 2:
            lum = arr.astype(np.float32, copy=False)
    key = _field_key(lum) if lum is not None else None
    if cached is not None:
        cached_key = getattr(observation, "_contour_field_cache_key", None)
        # No key available (no luminance): trust the observation-scoped cache.
        if key is None or cached_key == key:
            return cached
    if lum is None:
        return None
    field = field_from_luminance(lum)
    if observation is not None:
        try:
            observation._contour_field_cache = field
            if key is not None:
                observation._contour_field_cache_key = key
        except (AttributeError, TypeError):
            pass
    return field


class PeakLens(HigherLens):
    """First-stage peak detections; chain formation belongs exclusively to V2."""

    def step(
        self, t: int, concept_graph: Any, sig_provider: Any,
        anchor_positions: Dict[Any, Tuple[float, ...]],
        anchor_velocities: Optional[Dict[Any, Tuple[float, ...]]],
        current_detections: Dict[str, List[Detection]],
        extraction_scale: float = 0.0, envelope_mask: Any = None,
        observation: Any = None, luminance: Any = None, **kwargs: Any,
    ) -> List[Detection]:
        field = field_for_observation(observation, luminance)
        if field is None:
            return []
        peaks = field.get("peaks") or []
        if len(peaks) >= int(P.CHAIN_ANCHOR_OFFSET):
            raise ValueError("peak count collides with chain anchor range")
        # The adapter uses the field's exact list order when index-addressing.
        next_index = iter(range(len(peaks)))
        def allocate(y: int, x: int) -> int:
            return int(P.PEAK_ANCHOR_BASE) + next(next_index)
        out = peaks_to_contour_peak_detections(
            peaks, allocate, anchor_positions, extraction_scale,
            preserve_input_order=True, t=t,
        )
        self._emitted_total += len(out)
        return out
