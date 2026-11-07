"""perception/lenses/surface_fragment_lens.py -- the SURFACE_FRAGMENT emitter.

WHAT THIS IS
------------
Session B / B3.  ``perception.surface_fragments`` measures a SPARSE field of
surface fragments: one 2.5D orientation estimate per anchored contour point,
fused from the occluding-contour and shading cues on a shared (slant, tilt)
grid.  This module turns those fragments into detections of the registered
node class ``SURFACE_FRAGMENT`` (core/registries.py: four slots -- slant,
tilt, curvature sign, confidence).

THE IDENTITY CONTRACT
---------------------
The identity of a SURFACE_FRAGMENT is CONTENT-KEYED.  ``alpha`` is the hash of
the class, the relation and the four QUANTISED slot values -- nothing else:

    * the image coordinates of the fragment are NOT in the identity, the
      detection hash, or the golden contract;
    * the graded population (the sub-bin tuning vectors) is NOT in the
      identity either: ``TypedSignature`` already excludes it from its hash
      and equality, and ``alpha`` is built from the bins alone.

Two fragments with identical content therefore produce the SAME identity, and
their coordinates ride beside the detection in the lens' non-identity
``positions`` map (one list per content identity).  This is the
"presence + tolerance" reading of an assembly member: the content names the
member, the coordinate says where the member was seen.

WHY THE EMITTER IS STANDALONE (no pipeline wiring in this pass)
--------------------------------------------------------------
Adding a lens to ``LENS_DECLARATION_BY_NAME`` / the declared lens order is a
contract another session's pipeline depends on, so this pass ships the emitter
with the interface documented but UNWIRED.  A future wiring session provides
the fragments -- ``build_surface_fragments`` needs the ownership records, the
raw luminance and the corpus light prior -- and either passes them explicitly,
registers a ``fragment_source`` callable, or attaches them to the observation
as ``observation.surface_fragments``.

Nothing in this module reads a truth value, an object class, a mask or a
fabricated constant.  Every slot is a measurement the cue populations already
carry; a fragment with no usable cue ABSTAINS rather than emitting a default.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional, Tuple

from core.hashing import hash_64, serialize_float, serialize_tuple
from core.population import validate_population
from core.registries import (
    get_population_axes,
    population_for_bins,
    population_from_values,
    quantise,
    slot_specs,
)
from core.types import Detection, TypedSignature
from perception.lenses.derived_base import HigherLens
from perception.frames import FrameBank
from perception.surface_fragments import (
    SurfaceFragment,
    population_moment,
)

__all__ = [
    "SURFACE_FRAGMENT_CLASS",
    "SURFACE_FRAGMENT_RELATION",
    "SURFACE_FRAGMENT_K",
    "MAX_FRAGMENTS_PER_FRAME",
    "surface_fragment_values",
    "surface_fragment_signature",
    "surface_fragment_alpha",
    "surface_fragment_detection",
    "SurfaceFragmentLens",
]

#: The registered node class (core/registries.py: unit table + ranges).
SURFACE_FRAGMENT_CLASS = "SURFACE_FRAGMENT"

#: The relation of the emitted signature.  One relation, so the class' axes
#: are never silently compared under two different meanings.
SURFACE_FRAGMENT_RELATION = "SURFACE_FRAGMENT_ORIENTATION"

#: The detection key (``d.k``) the pipeline stores it under.
SURFACE_FRAGMENT_K = "SurfaceFragment"

#: Per-frame budget.  A fragment field is dense along every contour; the cap
#: keeps the emitter O(budget) on real images.  The highest-confidence
#: fragments are kept (the ones whose fused population is most concentrated),
#: so the budget never silently keeps noise over signal.  The value is a
#: storage budget, not a threshold: the cap is reported in the diagnostics.
MAX_FRAGMENTS_PER_FRAME = 512


def surface_fragment_values(
    fragment: SurfaceFragment,
) -> Optional[Tuple[float, float, float, float]]:
    """The four CONTINUOUS slot values of one fragment, in slot order.

    ``(slant_rad, tilt_rad, curvature_sign, confidence)``, or None when the
    fragment carries no fused cue (ABSTAIN -- never a fabricated default).

    Slant and tilt are the fused population's marginal mean and circular mean;
    confidence is its resultant length.  The curvature SIGN is the fragment's
    owner-relative sign, which is what the cue resolves.  Non-finite or
    out-of-span measurements abstain: a serialization must never be handed a
    NaN.
    """
    if fragment.combined is None:
        return None
    moment = population_moment(fragment.combined)
    if moment is None:
        return None
    slant_deg, tilt_deg, confidence = (float(v) for v in moment)
    if not (math.isfinite(slant_deg) and math.isfinite(tilt_deg)
            and math.isfinite(confidence)):
        return None
    # The grid the cues live on is 0 .. 90 deg slant and any tilt; a mean that
    # lands a hair outside (floating point) is clamped, not rejected.
    slant_deg = min(max(slant_deg, 0.0), 90.0)
    slant = math.radians(slant_deg)
    # TILT WRAP: population_moment returns atan2 in [-180, 180]; the slot is
    # [0, 2pi).  A value that rounds up to exactly 2pi is 0 (the same angle).
    tilt = math.radians(tilt_deg % 360.0)
    if tilt >= 2.0 * math.pi:
        tilt = 0.0
    # The sign is already discretised by the cue (|x| <= 1); keep the measured
    # value in the slot's span so the graded axis and the bins agree.
    sign = float(fragment.curvature_sign)
    if not math.isfinite(sign):
        sign = 0.0
    sign = min(max(sign, -1.0), 1.0)
    confidence = min(max(confidence, 0.0), 1.0)
    return slant, tilt, sign, confidence


def _quantised(values: Tuple[float, float, float, float]
               ) -> Tuple[float, ...]:
    specs = slot_specs(SURFACE_FRAGMENT_CLASS)
    if len(specs) != len(values):
        return ()
    return tuple(float(quantise(spec, value))
                 for spec, value in zip(specs, values))


def _population(values: Tuple[float, float, float, float],
                quantized: Tuple[float, ...]) -> Tuple[Tuple[float, ...], ...]:
    population = population_from_values(SURFACE_FRAGMENT_CLASS, values)
    if not population:
        population = population_for_bins(SURFACE_FRAGMENT_CLASS, quantized)
    axes = get_population_axes(SURFACE_FRAGMENT_CLASS)
    if population and axes is not None:
        validate_population(population, axes)
    return population


def surface_fragment_signature(
    fragment: SurfaceFragment,
) -> Optional[TypedSignature]:
    """The CANONICAL signature of one fragment, or None when it abstains.

    The bins are ``quantise`` applied to the unit table's own slots, and the
    population is ``population_from_values`` on the SAME continuous values, so
    the lookup key and the graded bump can never name different bins.  If the
    graded layout is unavailable the coarse-bins population is used instead of
    dropping the graded reading (never a mis-shaped population).
    """
    values = surface_fragment_values(fragment)
    if values is None:
        return None
    quantized = _quantised(values)
    if not quantized:
        return None
    return TypedSignature(
        kind="CANONICAL",
        class_=SURFACE_FRAGMENT_CLASS,
        relation=SURFACE_FRAGMENT_RELATION,
        quantized=quantized,
        population=_population(values, quantized),
    )


def surface_fragment_alpha(signature: TypedSignature) -> int:
    """The CONTENT identity of a fragment signature.

    A hash of (class, relation, quantised slots) only.  Deliberately ignores
    the population and every coordinate, so a re-render at another resolution
    that lands in the same bins is the SAME member, and a coordinate never
    leaks into identity.
    """
    payload = serialize_tuple([
        SURFACE_FRAGMENT_CLASS.encode("utf-8"),
        SURFACE_FRAGMENT_RELATION.encode("utf-8"),
        serialize_tuple([serialize_float(float(v))
                         for v in signature.quantized]),
    ])
    return int(hash_64(payload))


def _detection_from_signature(
    signature: TypedSignature,
    confidence: float,
    t: int,
    *,
    scale: int = 0,
    substrate_hash: int = 0,
) -> Detection:
    alpha = surface_fragment_alpha(signature)
    return Detection(
        k=SURFACE_FRAGMENT_K,
        source_layer=1,
        scale=int(scale),
        sigma=signature,
        alpha=alpha,
        z=float(confidence),
        t=int(t),
        rho=(),
        substrate_hash=int(substrate_hash),
    )


def surface_fragment_detection(
    fragment: SurfaceFragment,
    t: int,
    *,
    scale: int = 0,
    substrate_hash: int = 0,
) -> Optional[Detection]:
    """One Detection for one fragment, or None when the fragment abstains.

    ``alpha`` is content-keyed (see ``surface_fragment_alpha``); the position
    is NOT stored on the detection.  ``z`` is the fused confidence -- a
    salience the pipeline may read, never part of the signature identity --
    and ``rho`` is empty because binding a fragment to an entity/part is out
    of scope for this pass (the aspect-graph migration note describes it).
    """
    signature = surface_fragment_signature(fragment)
    if signature is None:
        return None
    values = surface_fragment_values(fragment)
    if values is None:  # pragma: no cover - signature implies values
        return None
    return _detection_from_signature(
        signature, values[3], t, scale=scale,
        substrate_hash=substrate_hash)


class SurfaceFragmentLens(HigherLens):
    """Emit SURFACE_FRAGMENT detections from a sparse fragment field.

    Input resolution, in order:
      1. the ``fragments`` argument (a sequence of ``SurfaceFragment``);
      2. ``fragment_source()`` if it is a callable;
      3. ``observation.surface_fragments`` (the documented interface: the
         caller sets the attribute; no observation class is modified here).

    No fragments -> no detections (a quiet scene, not an error).  The order is
    deterministic and independent of the input order: content key first,
    position only as a tie-break.  Content-identical fragments collapse to ONE
    detection (their coordinates are collected in ``positions`` under that
    content identity), which is exactly the "presence + tolerance" reading.
    """

    def __init__(self, max_fragments: int = MAX_FRAGMENTS_PER_FRAME) -> None:
        super().__init__()
        self.max_fragments = int(max_fragments)
        # NON-IDENTITY side channel: content alpha -> the image coordinates
        # that produced it this step.  Rebuilt every step; never hashed.
        self.positions: Dict[int, List[Tuple[float, float]]] = {}
        # [B4] The same coordinates in the selected frame, in ENERGY RADII,
        # plus their polar address (r, theta).  Also NON-IDENTITY: the frame
        # bank and these coordinates are geometry, not signature content.
        self.coordinates: Dict[int, List[Tuple[float, float]]] = {}
        self.polar: Dict[int, List[Tuple[float, float]]] = {}
        self.frames: Optional[FrameBank] = None
        self.frame_name: str = ""
        self._skipped_no_cue = 0
        self._skipped_non_finite = 0
        self._capped = 0
        self._collapsed = 0

    # -- input ---------------------------------------------------------
    @staticmethod
    def fragments_from(
        observation: Any = None,
        fragments: Optional[Iterable[SurfaceFragment]] = None,
        fragment_source: Optional[Any] = None,
    ) -> List[SurfaceFragment]:
        """Resolve the fragment field from the three accepted sources."""
        if fragments is not None:
            return list(fragments)
        if callable(fragment_source):
            got = fragment_source()
            return list(got) if got is not None else []
        if observation is not None:
            got = getattr(observation, "surface_fragments", None)
            if got is None:
                getter = getattr(observation, "get_surface_fragments", None)
                if callable(getter):
                    got = getter()
            return list(got) if got is not None else []
        return []

    # -- step ----------------------------------------------------------
    def step(
        self,
        t: int,
        concept_graph: Any = None,
        sig_provider: Any = None,
        anchor_positions: Optional[Dict[Any, Tuple[float, ...]]] = None,
        anchor_velocities: Optional[Dict[Any, Tuple[float, ...]]] = None,
        current_detections: Optional[Dict[str, List[Detection]]] = None,
        extraction_scale: float = 0.0,
        observation: Optional[Any] = None,
        luminance: Optional[Any] = None,
        fragments: Optional[Iterable[SurfaceFragment]] = None,
        fragment_source: Optional[Any] = None,
        frames: Optional[FrameBank] = None,
        scale: int = 0,
        **kwargs: Any,
    ) -> List[Detection]:
        self.positions = {}
        self.coordinates = {}
        self.polar = {}
        self.frame_name = ""
        field = self.fragments_from(observation, fragments, fragment_source)
        if not field:
            return []

        # [B4] THE FRAME BANK.  One origin and one scale measured from the
        # observation's own field, then coordinates in ENERGY RADII.  The
        # bank is supplied by the caller when the observation has a better
        # energy measure (the oriented-energy map); otherwise the fragment
        # positions are the measure.  Either way it is NON-IDENTITY data:
        # it never enters a signature, an alpha or a detection hash.
        bank = frames
        if bank is None:
            bank = FrameBank.from_points([f.position for f in field])
        self.frames = bank
        frame = bank.select() if bank is not None else None
        self.frame_name = frame.name if frame is not None else ""

        rows: List[Tuple[Tuple[float, ...], Tuple[float, float], float,
                          TypedSignature]] = []
        for fragment in field:
            if fragment.combined is None:
                self._skipped_no_cue += 1
                continue
            values = surface_fragment_values(fragment)
            signature = surface_fragment_signature(fragment)
            if values is None or signature is None:
                # A fused population exists but no slot could be measured
                # (empty moment or a non-finite component): abstain.
                self._skipped_non_finite += 1
                continue
            pos = (float(fragment.position[0]), float(fragment.position[1]))
            rows.append((signature.quantized, pos, values[3], signature))

        # Budget: keep the most concentrated populations, then restore the
        # content order so the OUTPUT does not depend on the confidence order.
        if self.max_fragments > 0 and len(rows) > self.max_fragments:
            self._capped += len(rows) - self.max_fragments
            rows.sort(key=lambda r: (-r[2], r[0], r[1]))
            rows = rows[: self.max_fragments]
        rows.sort(key=lambda r: (r[0], r[1]))

        seen: Dict[int, TypedSignature] = {}
        out: List[Detection] = []
        for _bins, pos, confidence, signature in rows:
            alpha = surface_fragment_alpha(signature)
            self.positions.setdefault(alpha, []).append(pos)
            if frame is not None:
                self.coordinates.setdefault(alpha, []).append(
                    frame.to_local(pos))
                self.polar.setdefault(alpha, []).append(frame.polar(pos))
            if alpha in seen:
                self._collapsed += 1
                continue
            seen[alpha] = signature
            out.append(_detection_from_signature(
                signature, confidence, t, scale=scale))
        self._emitted_total += len(out)
        self._candidates_evaluated += len(field)
        return out

    # -- diagnostics ---------------------------------------------------
    def get_diagnostics(self) -> Dict[str, int]:
        out = dict(super().get_diagnostics())
        out.update({
            "skipped_no_cue": self._skipped_no_cue,
            "skipped_non_finite": self._skipped_non_finite,
            "capped": self._capped,
            "collapsed": self._collapsed,
            "distinct_identities": len(self.positions),
        })
        return out
