"""perception/lenses/v4.py -- extracted from derived.py
(Move 3.5).

MOVE ONLY: whole top-level statements, verbatim.
"""
from __future__ import annotations

import numpy as np
import math
import itertools
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple
import parameters as P
from core.hashing import (  # noqa: F401
    _detection_sort_key,
    _node_signature_hash,
    clamp,
    hash_64,
    serialize_int,
    serialize_tuple,
    serialize_value,
)
from core.quantize import _aang, _opens_toward, carrier_to_tangent  # noqa: F401
from core.types import (  # noqa: F401
    Detection,
    EdgeType,
    TypedSignature,
)
from core.registries import (  # noqa: F401
    POPULATION_SLOTS,
    get_population_axes,
    population_axis,
    population_for_bins,
    population_for_slot_value,
    population_from_named_vectors,
    population_from_values,
    population_slot_vector,
    quantise,
    slot_axes,
)
from core.population import soft_bump_from_value  # noqa: F401
from core.slots import slot_slice, slot_value  # noqa: F401
from perception.contour_field import field_from_luminance  # noqa: F401
from perception.chain_flanks import planes_from_rgb  # noqa: F401
from perception.interior_field import (  # noqa: F401
    interior_field_for,
    interior_weighted_sample,
)
from perception.rich import (  # noqa: F401
    texture_richness,
)
from perception.depth import (  # noqa: F401
    _extract_member_anchors,
    spatial_clustering,
)
from perception.lenses.derived_common import (  # noqa: F401
    CURVATURE_TEMPLATES,
    _canonical_curvature_sequence,
    _chamfer,
    _resample_closed,
    polyline_shape_descriptor,
    contour_self_symmetry,
    match_templates,
    qi,
    qi_circ,
    sample_curvature_profile,
)
from perception.lenses.derived_base import (  # noqa: F401
    HigherLens,
)

# [UNIT TABLE] The unit tables of the classes this module emits, looked up BY
# NAME, so the bins path and the graded population read the SAME declaration
# of every quantity (unit, span, level count, quantiser).
_SURFACE_SPEC = {s.name: s for s in POPULATION_SLOTS["SURFACE_PATCH"]}
# [O2] The CONTOUR curvature SEQUENCE, by name.  The 20 signed-curvature
# levels of a contour chain, in slot order.
_CONTOUR_CURV_SEQUENCE = tuple(f"curv{k}" for k in range(20))
_CURVE_COMPLEX_SPEC = {s.name: s for s in POPULATION_SLOTS["CURVE_COMPLEX"]}
_REGION_SHAPE_SPEC = {s.name: s for s in POPULATION_SLOTS["REGION_SHAPE"]}
_ILLUSORY_SPEC = {s.name: s for s in POPULATION_SLOTS["ILLUSORY_CONTOUR"]}


def _level_bin(spec, level: float) -> float:
    """A SOURCE detection's LEVEL, written back through the unit table.

    The material slots of a surface carry the level of the SOURCE detection
    (the colour detection's own bins, the ownership field's own bins): their
    bin IS that level's index and their graded axis is that level's axis.
    Expressing the level as the fraction of the axis keeps the two readings
    on the same level.
    """
    return float(quantise(spec, float(level) / float(spec.n_bins)))


def _flank_lum_contrast(flank: Optional[Dict[str, Any]]) -> float:
    """[P1-10] The flank fields' mean |left - right| LUMINANCE contrast.

    A MEASUREMENT (a difference of the two flanks' sampled luminances,
    averaged over the chain's members), not a bin: the flank fields carry
    it per member and the surface slot names it.
    """
    if not flank:
        return 0.0
    members = flank.get("members") or ()
    if not members:
        return 0.0
    total = 0.0
    for m in members:
        try:
            total += abs(float(m.get("lum_diff", 0.0)))
        except (TypeError, ValueError):
            continue
    return total / len(members)


def _surface_population(
    *,
    shape_c: float,
    color_pop: Tuple[Tuple[float, ...], ...],
    flank: Optional[Dict[str, Any]],
    has_color: bool,
    tex: float,
    grad_side_bin: float,
    best_bo: Any,
    pooled: Optional[Dict[str, Optional[float]]] = None,
) -> Tuple[Tuple[float, ...], ...]:
    """The SURFACE_PATCH graded population, axis by axis.

    The layout (core.registries.POPULATION_SLOTS["SURFACE_PATCH"]) is
    (shape, lum, hue, sat, tex, grad_mag, grad_side). Every axis is built
    from a MEASUREMENT:

      shape     the CONTINUOUS spread of the boundary curvature;
      lum/hue/sat  the colour detection's own graded vectors (its
                CONTINUOUS opponent levels), else the flank fields'
                continuous statistics -- the mean flank luminance, the
                mean opponent contrast and the mean flank luminance
                difference (the same quantity the hue slot carries);
      tex       the CONTINUOUS texture richness;
      grad_mag  the ownership field's CONTINUOUS strength (its z);
      grad_side the ownership field's SIGNED direction, taken from its own
                graded vector (the signed slot of BORDER_OWNERSHIP).

    A vector projected from a source detection is carried over APART FROM
    ITS FRAME: the ownership's signed direction and the colour's luminance
    are both read on axes with the same bin count but not the same span
    (colour lum -1..1, patch lum 0..1), so the patch's own lookup key and
    the copied vector can name different bins at mid-grey.  That frame
    mismatch is measured and tracked as debt row 4 in
    `diagnostics/_u22_debt_ledger.md`; re-basing it re-mints every surface
    hash and was deliberately not folded into this hygiene pass.

    Returns () when no continuous source is available for an axis, so the
    signature falls back to its bins (never a mis-shaped population).
    """
    pairs = slot_axes("SURFACE_PATCH")
    if not pairs:
        return ()

    def _vec(slot_name: str, value: float) -> Tuple[float, ...]:
        """ONE slot's activity vector, addressed BY NAME (U22).

        The slot's own unit-table entry supplies the span, the circular
        wrap and the bin count, so this reads the registry the same way the
        bins path quantises it.  Addressed by NAME, never by position: a
        reordered POPULATION_SLOTS cannot move a measurement onto another
        quantity (tests/perception/test_population_scramble_u22.py).
        """
        return population_for_slot_value("SURFACE_PATCH", slot_name, value)

    side_axis = population_axis("SURFACE_PATCH", "grad_side")
    if side_axis is None:
        return ()

    # --- the material: the colour detection's graded vectors, else the
    # --- flank fields' continuous statistics
    c_vecs = tuple(color_pop) if len(color_pop) == 5 else ()
    lum_vec: Optional[Tuple[float, ...]] = None
    hue_vec: Optional[Tuple[float, ...]] = None
    sat_vec: Optional[Tuple[float, ...]] = None
    if has_color:
        if not c_vecs:
            # the colour detection carries no graded vector: its continuous
            # levels are unavailable, so NO population is claimed for the
            # patch (the bins path reads it instead).
            return ()
        # [U22] The source vectors are fetched from the COLOUR population by
        # their axis NAMES, not by their positions in that tuple.
        # [U23 Lead 6] COLOR lum is -1..1; patch lum is 0..1. Apply the
        # affine re-base instead of copying the vector verbatim (debt row 4).
        color_lum_vec = population_slot_vector(c_vecs, "COLOR", "lum") or None
        if color_lum_vec:
            from core.population import population_centre_bins
            _color_axis = population_axis("COLOR", "lum")
            if _color_axis is not None:
                _centre = population_centre_bins(color_lum_vec, _color_axis)
                _value = (_centre / float(_color_axis.n_bins)) * 2.0 - 1.0
                _transformed = (_value + 1.0) / 2.0
                lum_vec = _vec("lum", _transformed)
            else:
                lum_vec = color_lum_vec
        else:
            lum_vec = None
        sat_vec = population_slot_vector(c_vecs, "COLOR", "sat") or None
        hue_vec = population_slot_vector(c_vecs, "COLOR", "hue") or None
    elif pooled is not None and any(
            pooled.get(k) is not None for k in ("lum", "sat", "hue")):
        # [O4] The enclosure-weighted material, when a field was available.
        # These are the SAME spans the flank path uses (lum 0..1, sat 0..1,
        # hue 0..2*pi), so the axis and the measurement still agree.
        lum_vec = (_vec("lum", float(pooled["lum"]))
                   if pooled.get("lum") is not None else _vec("lum", 0.0))
        sat_vec = (_vec("sat", float(pooled["sat"]))
                   if pooled.get("sat") is not None else _vec("sat", 0.0))
        hue_vec = (_vec("hue", float(pooled["hue"]))
                   if pooled.get("hue") is not None else _vec("hue", 0.0))
    elif flank is not None:
        members = flank.get("members") or ()
        n_mem = len(members)
        lum_diff = (sum(abs(float(m.get("lum_diff", 0.0))) for m in members) / n_mem
                    if n_mem else 0.0)
        lum_vec = _vec("lum", float(flank.get("mean_lum_left", 0.0)))
        sat_vec = _vec("sat", float(flank.get("mean_sat_contrast", 0.0)))
        # [P1-10] The hue slot of the flank path carries the luminance
        # CONTRAST level, and the graded value is now the SAME quantity the
        # lookup key names: the contrast as an ANGLE on the slot's own
        # 16-cycle (a contrast of 1.0 is the full turn). It used to be
        # half that (contrast * pi), so the graded bump sat on a different
        # bin than the lookup key -- two quantisations of one measurement.
        hue_vec = _vec("hue", lum_diff * 2.0 * math.pi)
    else:
        # no measurement at all: the emitter's own defaults (zero level)
        lum_vec = _vec("lum", 0.0)
        sat_vec = _vec("sat", 0.0)
        hue_vec = _vec("hue", 0.0)
    if not lum_vec or not sat_vec or not hue_vec:
        return ()

    # --- the texture: the CONTINUOUS richness (a fraction in [0, 1]) ---
    tex_vec = _vec("tex", float(tex))
    if not tex_vec:
        return ()

    # --- the gradient: the ownership field's strength and signed side ---
    grad_strength = 0.0
    grad_side_vec: Optional[Tuple[float, ...]] = None
    if best_bo is not None:
        # the ownership detection's z IS its continuous strength
        grad_strength = float(getattr(best_bo, "z", 0.0) or 0.0)
        bo_pop = tuple(getattr(best_bo.sigma, "population", ()) or ())
        # [U22] the ownership's sign is its NAMED axis, not its position.
        grad_side_vec = (population_slot_vector(bo_pop, "BORDER_OWNERSHIP",
                                                "signed_dir") or None)
    grad_mag_vec = _vec("grad_mag", grad_strength)
    if not grad_mag_vec:
        return ()
    if grad_side_vec is None or len(grad_side_vec) != side_axis.n_bins:
        # no graded ownership vector: fall back to the signed direction the
        # bin names (the ONLY reading available in that case)
        signed = (float(grad_side_bin) + 0.5) / 4.0 - 1.0
        grad_side_vec = _vec("grad_side", (signed + 1.0) / 2.0)
    if not grad_side_vec or len(grad_side_vec) != side_axis.n_bins:
        return ()

    # Assembled in the registry's CURRENT slot order, from vectors addressed
    # by NAME -- so the tuple a reader zips with `slot_axes` is always the
    # tuple the registry describes.
    return population_from_named_vectors("SURFACE_PATCH", {
        "shape": _vec("shape", shape_c),
        "lum": lum_vec,
        "hue": hue_vec,
        "sat": sat_vec,
        "tex": tex_vec,
        "grad_mag": grad_mag_vec,
        "grad_side": grad_side_vec,
    })


def _enclosure_pooled_material(
    interior: Any,
    planes: Optional[Dict[str, Any]],
    x: float,
    y: float,
    radius: float,
) -> Optional[Dict[str, Optional[float]]]:
    """[O4] Surface material pooled by ENCLOSURE WEIGHT around a fragment.

    A surface statistic used to be a fixed window (the flanks' offset samples,
    or a region's own level): the window straddles the boundary, so the same
    fragment was described by a blend of figure and GROUND material. Here every
    pixel contributes `interior * gaussian`, so the enclosed side dominates
    without a hard mask. The planes are the front end's own:

      lum      the adapted luminance (0..1)
      sat      |opponent| = hypot(rg, by) (0..1-ish; the slot clamps)
      hue      circular mean of atan2(by, rg), in [0, 2*pi)
      tex      local |lum - local mean| normalised by the plane's p95,
               a second-order texture richness in [0, 1]

    Returns None when there is no field/planes; a per-key None means that
    plane's weighted mean had no support (all weights vanished) and the caller
    keeps its own fallback -- never a fabricated zero.
    """
    if interior is None or not planes:
        return None
    enc = interior.get("interior") if isinstance(interior, dict) else interior
    if enc is None:
        return None

    def _s(key: str) -> Optional[float]:
        p = planes.get(key)
        if p is None:
            return None
        return interior_weighted_sample(p, enc, x, y, radius)

    lum = _s("lum")
    sat = _s("sat")
    tc = _s("hue_cos")
    ts = _s("hue_sin")
    tex = _s("tex")
    hue: Optional[float] = None
    if tc is not None and ts is not None and (abs(tc) + abs(ts)) > 1e-12:
        hue = float(math.atan2(ts, tc) % (2.0 * math.pi))
    return {"lum": lum, "sat": sat, "hue": hue, "tex": tex}


def _symmetry_measurements(pts):
    """The symmetry SCORE and the symmetry AXIS ANGLE of a closed outline.

    The same computation contour_self_symmetry performs (perception/lenses/
    derived_common.py), with the SAME resampling and chamfer helpers, but
    returning the CONTINUOUS score and the axis angle instead of their
    bins: the graded code needs the measurement the bins round off.

    The axis angle is None when the outline is isotropic (the score spread
    over the axes is below P.SYM_ISOTROPIC_SPREAD): there is no preferred
    axis, and the emitter's lookup key keeps the undefined sentinel.
    """
    R = _resample_closed(pts, P.SYM_RESAMPLE_N)
    if R is None:
        return 0.0, None
    R = R - R.mean(axis=0)
    rms = math.sqrt(float((R ** 2).sum(axis=1).mean())) + 1e-12
    tol = P.SYM_TAU * rms
    scores = []
    for k in range(P.SYM_N_AXES):
        phi = k * math.pi / P.SYM_N_AXES
        c2, s2 = math.cos(2 * phi), math.sin(2 * phi)
        M = np.array([[c2, s2], [s2, -c2]])
        scores.append(max(0.0, 1.0 - _chamfer(R @ M.T, R) / tol))
    best_k = int(np.argmax(scores))
    spread = max(scores) - min(scores)
    axis = (None if spread < P.SYM_ISOTROPIC_SPREAD
            else best_k * math.pi / P.SYM_N_AXES)
    return float(max(scores)), axis


def _region_shape_measurements(poly, observation, closed):
    """The CONTINUOUS measurements behind the REGION_SHAPE slots.

    polyline_shape_descriptor (perception/lenses/derived_common.py) returns
    only the BINS; the graded code needs the pre-quantisation values, so
    they are recomputed here from the SAME polyline with the SAME formulas
    (the curvature histogram, the shoelace area and the elongation
    eigenvalue ratio), so the graded vector and the lookup index describe
    ONE measurement.

    Returns (hist_values, area_norm, elong_norm, sym_score, axis_angle).
    """
    n_pts = len(poly)
    periodic = bool(closed) and n_pts >= 3
    raw_area_sum = 0.0
    if periodic:
        for i in range(n_pts):
            p1 = poly[i]
            p2 = poly[(i + 1) % n_pts]
            raw_area_sum += p1[0] * p2[1] - p2[0] * p1[1]
    area = abs(raw_area_sum) / 2.0

    hist = [0.0] * P.REGION_SHAPE_CURV_BINS
    turns = [0.0] * n_pts
    _arc = 0.0
    for i in range(n_pts):
        p_prev = (poly[(i - 1) % n_pts] if periodic
                  else poly[max(0, i - 1)])
        p_curr = poly[i]
        p_next = (poly[(i + 1) % n_pts] if periodic
                  else poly[min(n_pts - 1, i + 1)])
        v1x = p_curr[0] - p_prev[0]
        v1y = p_curr[1] - p_prev[1]
        v2x = p_next[0] - p_curr[0]
        v2y = p_next[1] - p_curr[1]
        cross = v1x * v2y - v1y * v2x
        dot = v1x * v2x + v1y * v2y
        theta = math.atan2(cross, dot)
        if raw_area_sum < 0:
            theta = -theta
        turns[i] = theta
        bin_idx = int(
            math.floor(
                (theta + math.pi) / (2 * math.pi) * P.REGION_SHAPE_CURV_BINS
            )
        )
        bin_idx = min(P.REGION_SHAPE_CURV_BINS - 1, max(0, bin_idx))
        L_local = (math.hypot(v1x, v1y) + math.hypot(v2x, v2y)) / 2.0
        hist[bin_idx] += L_local
        _arc += L_local

    total_L = sum(hist)
    if total_L > 1e-6:
        hist = [h / total_L for h in hist]
    hist_values = [math.sqrt(h) for h in hist]

    img_area = float(getattr(observation, "image_area", 0.0)) or 1.0
    area_norm = math.log10(max(area / img_area, 1e-6))

    _cx = sum(p[0] for p in poly) / n_pts
    _cy = sum(p[1] for p in poly) / n_pts
    _sxx = sum((p[0] - _cx) ** 2 for p in poly) / n_pts
    _syy = sum((p[1] - _cy) ** 2 for p in poly) / n_pts
    _sxy = sum((p[0] - _cx) * (p[1] - _cy) for p in poly) / n_pts
    _tr = _sxx + _syy
    _det = _sxx * _syy - _sxy * _sxy
    _disc = math.sqrt(max(0.0, _tr * _tr / 4.0 - _det))
    _l1 = _tr / 2.0 + _disc
    _l2 = _tr / 2.0 - _disc
    elong = math.sqrt(_l1 / (_l2 + 1e-9)) if _l2 > 1e-9 else 1.0
    elong_norm = min(1.0, (elong - 1.0) / 4.0)

    sym_score, axis_angle = _symmetry_measurements(poly)
    return hist_values, area_norm, elong_norm, sym_score, axis_angle


class SurfaceLens(HigherLens):
    """B12 [FIX v2] — V2/V4 Surface as Bounded Material Field (arity 7)."""

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
        **kwargs,
    ) -> List[Detection]:
        if not current_detections:
            return []
        color_by = {d.alpha: d for d in current_detections.get("Color", [])}
        tex_by = {d.alpha: d for d in current_detections.get("Texture", [])}
        bo_by: Dict[Any, List[Any]] = defaultdict(list)
        for d in current_detections.get("BorderOwnership", []):
            bo_by[d.alpha].append(d)

        # The FRAGMENT source: one surface per PEAK_CHAIN (a contour
        # fragment). There is no region and no entity any more -- the
        # material of a fragment is measured on its two flanks, with the
        # continuous interior field deciding which side is figure and which
        # is ground.
        chain_dets = list((current_detections or {}).get("PEAK_CHAIN", []))
        emitted = []
        if chain_dets:
            # Field entities: iterate anchors in the source's own order. Each
            # member of the chain contributes its flank contrast, so a chain
            # emits ONE detection per chain (not per member) whose slots are
            # the aggregate over members -- one figure, one surface.
            from perception.chain_flanks import (
                chain_flank_fields, flank_signature_slots,
            )
            # The front end's own colour planes, and the field's median
            # wavelength, are both derived from the SAME rgb plane the chain
            # lens was driven with. Probing the observation for them is not
            # usable here: the region-construction path carries no luminance
            # and no channels.
            channels = None
            med_wl = None
            shape = None
            rgb = luminance
            if rgb is None:
                rgb = getattr(observation, "_raw_image", None)
            if rgb is not None:
                rgb = np.asarray(rgb)
                if rgb.ndim >= 2:
                    shape = (int(rgb.shape[0]), int(rgb.shape[1]))
                if rgb.ndim == 3:
                    # planes_from_rgb gives the SAME adapted luminance the
                    # chains were derived from; the retinex lum_comp this
                    # replaced is a different plane (max difference 0.95).
                    channels = planes_from_rgb(rgb)
                    if channels is not None:
                        try:
                            f = field_from_luminance(
                                np.asarray(channels.get("lum", [0]),
                                           dtype=np.float32))
                            sfs = np.asarray(f["sfs"], dtype=np.float64)
                            med_wl = float(np.median(
                                1.0 / np.maximum(sfs, 1e-6)))
                        except Exception:
                            med_wl = None

            # [O4] THE ENCLOSURE FIELD AND THE MATERIAL PLANES, computed ONCE
            # per frame. The surface of every fragment is pooled from these
            # planes by the continuous interior field, so a fragment's
            # material is the FIGURE's material and not a fixed window that
            # straddles the boundary. Built only when the front end's planes
            # exist: otherwise `pool_planes`/`interior` stay None and the
            # emission path is byte-identical to the pre-O4 path.
            pool_planes = None
            interior = None
            if channels is not None:
                _lum_p = np.asarray(channels.get("lum"), dtype=np.float64)
                _rg_p = np.asarray(channels.get("rg"), dtype=np.float64)
                _by_p = np.asarray(channels.get("by"), dtype=np.float64)
                if (_lum_p.ndim == 2 and _rg_p.shape == _lum_p.shape
                        and _by_p.shape == _lum_p.shape):
                    from scipy.ndimage import gaussian_filter as _gf
                    _hue_p = np.arctan2(_by_p, _rg_p) % (2.0 * math.pi)
                    _loc = _gf(_lum_p, sigma=2.0, mode="nearest")
                    _dev = np.abs(_lum_p - _loc)
                    _scale = float(np.percentile(_dev, 95.0)) if _dev.size else 0.0
                    if not np.isfinite(_scale) or _scale <= 1e-12:
                        _scale = float(_dev.max()) if _dev.size else 1.0
                    pool_planes = {
                        "lum": _lum_p,
                        "sat": np.hypot(_rg_p, _by_p),
                        "hue_cos": np.cos(_hue_p),
                        "hue_sin": np.sin(_hue_p),
                        "tex": np.clip(_dev / max(_scale, 1e-12), 0.0, 1.0),
                    }
                    try:
                        interior = interior_field_for(None, luminance=_lum_p)
                    except Exception:
                        interior = None
                        pool_planes = None

            for det in chain_dets:
                anchor = det.alpha
                flank = None
                if det is not None and med_wl and channels:
                    try:
                        fld = chain_flank_fields(
                            det, anchor_positions, med_wl, channels,
                            image_shape=shape)
                        if fld is not None:
                            # flatten the named slot dict into the flank dict
                            # so _emit_one reads plain keys, and keep the raw
                            # slot set too for diagnostics
                            fld.update(flank_signature_slots(fld))
                            fld["slots"] = fld
                            flank = fld
                    except Exception:
                        flank = None
                bo_field = bo_by.get(anchor, [])
                c_det = color_by.get(anchor)
                t_det = tex_by.get(anchor)
                emitted.append(self._emit_one(
                    t, anchor, 1.0, bo_field, c_det, t_det,
                    flank, current_detections, observation,
                    anchor_positions, interior, pool_planes))
        self._emitted_total += len(emitted)
        return emitted

    def _emit_one(
        self,
        t: int,
        anchor: Any,
        z_val: Any,
        bo_field: List[Any],
        c_det: Any,
        t_det: Any,
        flank: Optional[Dict[str, Any]],
        current_detections: Optional[Dict[str, List[Detection]]],
        observation: Optional[Any],
        anchor_positions: Optional[Dict[Any, Tuple[float, ...]]],
        interior: Optional[Dict[str, Any]] = None,
        pool_planes: Optional[Dict[str, Any]] = None,
    ) -> Detection:
        """One Surface detection for one entity, from either entity source.

        The material slots come from the entity's own Colour/Texture
        detections when present, else from the field region's surface sample
        -- that fallback is what keeps a field region emittable without the
        segmenter's region descriptors.
        """
        curvs = [
            slot_value(d.sigma, "BORDER_OWNERSHIP", "curvature", default=0.0)
            for d in bo_field
            if len(d.sigma.quantized) >= 4
        ]
        shape_bin = 0.0
        shape_c = 0.0
        if curvs:
            mean_c = sum(curvs) / len(curvs)
            var_c = sum((c - mean_c) ** 2 for c in curvs) / len(curvs)
            # the CONTINUOUS curvature spread (the emitter normalises it by
            # the same 12.25 the shape bin uses)
            shape_c = var_c / 12.25
            shape_bin = float(quantise(_SURFACE_SPEC["shape"], shape_c))

        # [O1] THE COLOUR IS READ BY NAME, FROM THE COLOUR'S OWN UNIT
        # TABLE -- not by position across a class boundary.  The surface's
        # lum/hue/sat slots carry the colour's own LEVELS (its bin index
        # on the colour's own axis), which is exactly what the unit table
        # says they are.
        c_q = ()
        if (c_det is not None
                and getattr(c_det.sigma, "class_", "") == "COLOR"
                and len(c_det.sigma.quantized) >= 5):
            c_q = c_det.sigma.quantized
        lum = slot_value(c_det.sigma, "COLOR", "lum", default=0.0) if c_q else 0.0
        sat = slot_value(c_det.sigma, "COLOR", "sat", default=0.0) if c_q else 0.0
        hue = slot_value(c_det.sigma, "COLOR", "hue", default=0.0) if c_q else 0.0
        # [ROUND 10 / D] Texture. `texture_richness` already prefers the B4
        # rich signature (second-order statistics of the pooled Gabor complex
        # energy) when the region carries one; for a field-region surface with
        # no B4 detection, the flank fields' luminance variance is the
        # second-order stand-in computed from the same front-end energies.
        # Both are FRACTIONS in [0, 1] -- the unit table's unit for the
        # texture slot.
        tex = texture_richness(t_det) if t_det is not None else 0.0
        if tex <= 0.0 and flank is not None:
            tex = min(1.0, float(flank.get("lum_variance", 0.0)))

        # [O4] ENCLOSURE-WEIGHTED SURFACE POOLING. When the frame's interior
        # field and material planes exist, a fragment's material is the mean
        # of the planes weighted by the continuous interior field: pixels on
        # the FIGURE side dominate without a hard mask, so the surface is the
        # enclosed material and not a fixed window straddling the boundary.
        # `None` (no field / no planes) leaves every value below untouched,
        # so the pre-O4 emission path is byte-identical.
        _pooled: Optional[Dict[str, Optional[float]]] = None
        if interior is not None and pool_planes is not None:
            _pos = (anchor_positions or {}).get(anchor)
            if _pos is not None and len(_pos) >= 2:
                _radius = float(P.INTERIOR_PROBE_PX)
                if flank is not None and flank.get("offset_px"):
                    _radius = max(_radius, float(flank["offset_px"]))
                _radius = max(2.0, 2.0 * _radius)
                _pooled = _enclosure_pooled_material(
                    interior, pool_planes, float(_pos[0]), float(_pos[1]),
                    _radius)
                # The texture slot is a MATERIAL property in every path, so
                # the enclosure-pooled richness replaces the flank variance
                # whenever it is available.
                if _pooled is not None and _pooled.get("tex") is not None:
                    tex = float(_pooled["tex"])

        # [W1] A field region with no region-level Color/Texture detection
        # still carries a material description. Flank contrast is preferred
        # (Slice 2): it is what makes a field region emittable AND what makes
        # its slots vary -- Slice 0c showed the interior-only fallback
        # collapses 80 emissions to 3 signatures, which dedupe to 0 new graph
        # nodes. The interior sample is the last-resort fallback.
        if not c_q:
            if flank is not None:
                # [P1-10] The CONTINUOUS measurements the flank fields carry,
                # NOT the bins re-binned.
                #
                # The slots used to be built from `lum_level_bin`,
                # `sat_contrast_bin` and `lum_contrast_bin` -- the flank
                # fields' own 8-bin summaries, re-binned onto this class'
                # axes. That threw away the sub-bin position AND, for the hue
                # slot, half the axis: the luminance contrast (a fraction) was
                # carried through an 8-level intermediate and then placed on a
                # 16-cycle, so the lookup key could only ever name the lower
                # half of the hue axis.
                #
                # The flank fields carry the measurements themselves:
                #   mean_lum_left      the mean flank luminance level;
                #   mean_sat_contrast  the mean opponent magnitude difference;
                #   the members' lum_diff  the per-member luminance difference.
                # The levels below are those measurements expressed in the
                # slots' own level counts (the unit table's convention: the
                # level IS the bin index), so the lookup key and the graded
                # population name the SAME measurement.
                lum = (float(flank.get("mean_lum_left", 0.0))
                       * float(_SURFACE_SPEC["lum"].n_bins))
                sat = (float(flank.get("mean_sat_contrast", 0.0))
                       * float(_SURFACE_SPEC["sat"].n_bins))
                hue = _flank_lum_contrast(flank) * 16.0
            # [O4] The enclosure-pooled material, when available, is the
            # surface's OWN measurement and supersedes the flank estimate.
            # The spans are the same (lum/sat 0..1, hue 0..2*pi) mapped onto
            # the slots' own level counts, exactly as the flank path does.
            if _pooled is not None:
                if _pooled.get("lum") is not None:
                    lum = (float(_pooled["lum"])
                           * float(_SURFACE_SPEC["lum"].n_bins))
                if _pooled.get("sat") is not None:
                    sat = (float(_pooled["sat"])
                           * float(_SURFACE_SPEC["sat"].n_bins))
                if _pooled.get("hue") is not None:
                    hue = (float(_pooled["hue"]) / (2.0 * math.pi) * 16.0)

        best = max(
            bo_field,
            key=lambda d: slot_value(d.sigma, "BORDER_OWNERSHIP", "strength",
                                     default=-1.0),
        ) if bo_field else None
        grad_mag_bin, grad_side = 0.0, 0.0
        if best is not None and len(best.sigma.quantized) >= 4:
            grad_side = slot_value(best.sigma, "BORDER_OWNERSHIP", "signed_dir",
                                   default=0.0)
            grad_mag_bin = slot_value(best.sigma, "BORDER_OWNERSHIP", "strength",
                                      default=0.0)

        quantized = (
            float(shape_bin),
            float(_level_bin(_SURFACE_SPEC["lum"], lum)),
            # The hue slot carries the SOURCE detection's level: its bin IS
            # that level's index on the 16-cycle (the angular axis' index),
            # and the graded axis is that same axis. The level is written
            # directly because the level -> angle conversion is not exactly
            # invertible in floating point (11/16 of a turn is not a
            # representable multiple of pi/8).
            float(qi_circ(hue, 16.0, 16)),
            float(_level_bin(_SURFACE_SPEC["sat"], sat)),
            float(quantise(_SURFACE_SPEC["tex"], float(tex or 0.0))),
            float(_level_bin(_SURFACE_SPEC["grad_mag"], grad_mag_bin)),
            float(_level_bin(_SURFACE_SPEC["grad_side"], grad_side)),
        )

        # [GRADED POPULATION] The CONTINUOUS material measurements of the
        # patch, in the order the SURFACE_PATCH graded layout declares
        # (core.registries.POPULATION_SLOTS["SURFACE_PATCH"]): the shape
        # (the spread of the boundary curvature), the luminance, the hue,
        # the saturation, the texture richness, the gradient magnitude (the
        # ownership field's strength) and the gradient side (its signed
        # direction). The bins stay the lookup key; the activity vector
        # keeps the sub-bin position each bin rounds off.
        #
        # Two sources, both MEASUREMENTS, never a bin index:
        #   * the colour detection's own graded vectors (its CONTINUOUS
        #     luminance / saturation / hue), projected onto this axis: the
        #     colour's span (-1..1 for the luminance) and this axis' span
        #     (0..1) are the same affine map, so the projection is exact
        #     and the patch's vector IS the colour's vector;
        #   * the flank fields' continuous statistics (the mean flank
        #     luminance, the mean opponent contrast and the mean flank
        #     luminance difference), which is what the flank path measures.
        _pop = _surface_population(
            shape_c=shape_c,
            color_pop=(getattr(c_det.sigma, "population", ())
                       if c_det is not None else ()),
            flank=flank,
            has_color=bool(c_q),
            tex=clamp(float(tex or 0.0), 0.0, 1.0),
            grad_side_bin=grad_side,
            best_bo=best,
            pooled=_pooled,
        )
        if not _pop:
            # No continuous source for some axis: keep the coarse-bins
            # population (never a mis-shaped one) rather than dropping the
            # graded reading entirely.
            _pop = population_for_bins("SURFACE_PATCH", quantized)
        return Detection(
            k="Surface",
            source_layer=1,
            scale=0,
            sigma=TypedSignature(
                kind="CANONICAL",
                class_="SURFACE_PATCH",
                relation="SURFACE_MATERIAL",
                quantized=quantized,
                population=_pop,
            ),
            alpha=anchor,
            z=z_val,
            t=t,
            rho=(
                ("entity", anchor, 0, "+"),
                ("boundary", anchor, 0, "+"),
            ),
        )

class IllusoryContourLens(HigherLens):
    """B17 — V2 Modal Completion (Illusory Contours) (arity 3)."""

    def step(
        self,
        t: int,
        concept_graph: Any = None,
        sig_provider: Any = None,
        anchor_positions: Optional[Dict[Any, Tuple[float, ...]]] = None,
        anchor_velocities: Optional[Dict[Any, Tuple[float, ...]]] = None,
        current_detections: Optional[Dict[str, List[Detection]]] = None,
        extraction_scale: float = 0.0,
        **kwargs,
    ) -> List[Detection]:
        if not current_detections:
            return []
        end_dets = current_detections.get("EndStopped", [])
        if not end_dets:
            end_dets = current_detections.get("Termination", [])
        if not end_dets or extraction_scale <= 0.0:
            return []

        max_gap = P.KAPPA_ILLUSORY_MAX_GAP * extraction_scale
        angle_tol = P.ILLUSORY_ANGLE_TOL_RAD
        candidates = []
        n_theta = P.ORIENTATION_N_THETA

        grid = defaultdict(list)
        for d in sorted(end_dets, key=_detection_sort_key):
            p = (anchor_positions or {}).get(d.alpha)
            if p is None and isinstance(d.alpha, tuple) and len(d.alpha) >= 2:
                p = d.alpha
            if p is not None and len(p) >= 2:
                grid[(int(p[0] // max_gap), int(p[1] // max_gap))].append((d, p))

        seen_pairs = set()
        for (cx, cy), cell in grid.items():
            neigh = []
            for off in itertools.product((-1, 0, 1), repeat=2):
                neigh.extend(grid.get((cx + off[0], cy + off[1]), ()))
            for i, (d1, p1) in enumerate(neigh):
                theta1_bin = slot_value(d1.sigma, "ILLUSORY_CONTOUR", "theta",
                                        default=0.0)
                s1 = d1.z
                for d2, p2 in neigh[i + 1 :]:
                    pair_key = (min(id(d1), id(d2)), max(id(d1), id(d2)))
                    if pair_key in seen_pairs:
                        continue
                    seen_pairs.add(pair_key)
                    if d1.alpha == d2.alpha:
                        continue
                    theta2_bin = slot_value(d2.sigma, "ILLUSORY_CONTOUR",
                                            "theta", default=0.0)
                    s2 = d2.z

                    dx = p2[0] - p1[0]
                    dy = p2[1] - p1[1]
                    dist = math.hypot(dx, dy)
                    if dist < 1e-6 or dist > max_gap:
                        continue

                    t1 = carrier_to_tangent((theta1_bin + 0.5) * math.pi / n_theta)
                    t2 = carrier_to_tangent((theta2_bin + 0.5) * math.pi / n_theta)
                    if _aang(t1, t2) > angle_tol:
                        continue
                    phi = math.atan2(dy, dx) % math.pi
                    if (
                        _aang(phi, t1) > angle_tol
                        or _aang(phi, t2) > angle_tol
                    ):
                        continue
                    if not (
                        _opens_toward(d1, (dx, dy), n_theta)
                        and _opens_toward(d2, (-dx, -dy), n_theta)
                    ):
                        continue

                    strength = s1 * s2 * math.exp(-dist / max_gap)
                    candidates.append((strength, t1, dist, p1, p2, d1, d2))

        candidates.sort(key=lambda x: -x[0])

        used, kept = set(), []
        for cand in candidates[: P.ILLUSORY_MAX_CANDIDATES]:
            k1 = serialize_value(cand[5].alpha)
            k2 = serialize_value(cand[6].alpha)
            if k1 in used or k2 in used:
                continue
            used.update((k1, k2))
            kept.append(cand)

        emitted = []
        for strength, theta, dist, p1, p2, d1, d2 in kept:
            if strength < 0.05:
                continue
            mid_x = (p1[0] + p2[0]) / 2.0
            mid_y = (p1[1] + p2[1]) / 2.0
            synthetic_anchor = hash_64(
                serialize_tuple([
                    serialize_value(d1.alpha),
                    serialize_value(d2.alpha),
                    serialize_int(int(round(mid_x))),
                    serialize_int(int(round(mid_y))),
                ])
            )
            alpha = synthetic_anchor
            # [UNIT TABLE] The bins come from the class' unit table
            # (core.registries.POPULATION_SLOTS["ILLUSORY_CONTOUR"]), so the
            # lookup index and the graded population are quantisations of the
            # SAME measurements in the SAME units.
            theta_bin = int(quantise(_ILLUSORY_SPEC["theta"], float(theta)))
            length_norm = min(1.0, dist / max_gap)
            length_bin = int(quantise(_ILLUSORY_SPEC["length"],
                                      float(length_norm)))
            strength_norm = min(1.0, strength)
            strength_bin = int(quantise(_ILLUSORY_SPEC["strength"],
                                        float(strength)))

            # [GRADED POPULATION] The CONTINUOUS completion measurements: the
            # axial orientation (radians), the gap length as a fraction of
            # the maximum illusory gap, and the completion strength.
            _ic_pop = population_from_values(
                "ILLUSORY_CONTOUR",
                (float(theta), float(length_norm), float(strength)),
            )

            emitted.append(
                Detection(
                    k="IllusoryContour",
                    source_layer=1,
                    scale=0,
                    sigma=TypedSignature(
                        kind="CANONICAL",
                        class_="ILLUSORY_CONTOUR",
                        relation="MODAL_COMPLETION",
                        quantized=(
                            float(theta_bin),
                            float(length_bin),
                            float(strength_bin),
                        ),
                        population=_ic_pop,
                    ),
                    alpha=alpha,
                    # z is a CONFIDENCE in [0, 1]: the completion strength.
                    z=float(min(1.0, max(0.0, strength))),
                    t=t,
                    rho=(
                        ("entity", alpha, 0, "+"),
                        ("inducer_1", d1.alpha, 0, "+"),
                        ("inducer_2", d2.alpha, 0, "+"),
                    ),
                )
            )
        self._emitted_total += len(emitted)
        return emitted

class CurveComplexLens(HigherLens):
    """B.8 — V4 curve-bundles over Contour and Junction.

    Builds a neighbour graph over the V2 nodes from CONTOUR_CONTINUITY and
    JUNCTION_CONTOUR edges, then takes connected components that contain at
    least one Contour. The component's SHAPE_ARRANGEMENT signature (arity 24,
    matching CONSTITUTIONAL_RANGES["CURVE_COMPLEX"]) is:
        (n_contours, n_junctions, extent, closure,
         8 curvature-template bins, 12 template-population bins)
    """

    def __init__(self) -> None:
        super().__init__()
        self._skipped_no_position = 0
        self._skipped_no_component = 0
        self._candidates_evaluated = 0

    def get_diagnostics(self) -> Dict[str, int]:
        return {
            "emitted_total": self._emitted_total,
            "skipped_no_position": self._skipped_no_position,
            "skipped_no_component": self._skipped_no_component,
            "candidates_evaluated": self._candidates_evaluated,
        }

    def step(
        self,
        t: int,
        concept_graph: Any,
        sig_provider: Any,
        anchor_positions: Dict[Any, Tuple[float, ...]],
        anchor_velocities: Optional[Dict[Any, Tuple[float, ...]]],
        current_detections: Dict[str, List[Detection]],
        extraction_scale: float = 0.0,
    ) -> List[Detection]:
        if extraction_scale <= 0.0:
            self._skipped_no_position += 1
            return []

        contour_dets = current_detections.get("Contour", [])
        junction_dets = current_detections.get("Junction", [])
        if not contour_dets and not junction_dets:
            self._skipped_no_component += 1
            return []

        node_id_to_detection: Dict[int, Detection] = {}
        node_id_to_kind: Dict[int, str] = {}
        for det in sorted(contour_dets, key=_detection_sort_key):
            if len(det.sigma.quantized) < 4:
                continue
            nid = concept_graph.get_or_create_pattern_node(
                det.k, _node_signature_hash(det.sigma), signature=det.sigma
            )
            node_id_to_detection.setdefault(nid, det)
            node_id_to_kind.setdefault(nid, "Contour")
        for det in sorted(junction_dets, key=_detection_sort_key):
            if len(det.sigma.quantized) < 3:
                continue
            nid = concept_graph.get_or_create_pattern_node(
                det.k, _node_signature_hash(det.sigma), signature=det.sigma
            )
            node_id_to_detection.setdefault(nid, det)
            node_id_to_kind.setdefault(nid, "Junction")

        # Weight filter: THETA_CONTOUR_EDGE for BOTH edge types. V4 is
        # structural topology over pairs that have already proven their
        # continuity through the V2 EMA, so one gate is right; a second,
        # stricter gate would drop bundles whose junction links are
        # legitimately weaker than their contour links.
        # Every admitted V2 node enters the component graph, including
        # degree-0 nodes: an isolated contour is a singleton component and
        # may emit as topology class 0 (single closed loop) or 1 (single
        # open contour). Seeding only edge-endpoints made those classes
        # unreachable even after the size guard was lifted.
        G_v4: Dict[int, Set[int]] = defaultdict(set)
        for nid in node_id_to_detection:
            G_v4[nid]  # touch: register degree-0 nodes as components
        for nid in node_id_to_detection:
            for edge in concept_graph.get_outgoing_edges(nid):
                if edge.edge_type not in (
                    EdgeType.CONTOUR_CONTINUITY, EdgeType.JUNCTION_CONTOUR
                ):
                    continue
                if edge.weight < P.THETA_CONTOUR_EDGE:
                    continue
                if edge.child not in node_id_to_detection:
                    continue
                G_v4[nid].add(edge.child)
                G_v4[edge.child].add(nid)
            for edge in concept_graph.get_incoming_edges(nid):
                if edge.edge_type not in (
                    EdgeType.CONTOUR_CONTINUITY, EdgeType.JUNCTION_CONTOUR
                ):
                    continue
                if edge.weight < P.THETA_CONTOUR_EDGE:
                    continue
                if edge.parent not in node_id_to_detection:
                    continue
                G_v4[nid].add(edge.parent)
                G_v4[edge.parent].add(nid)

        visited: Set[int] = set()
        components: List[Set[int]] = []
        for start in sorted(G_v4.keys()):
            if start in visited:
                continue
            comp: Set[int] = set()
            stack = [start]
            visited.add(start)
            while stack:
                x = stack.pop()
                comp.add(x)
                for y in sorted(G_v4.get(x, ())):
                    if y not in visited:
                        visited.add(y)
                        stack.append(y)
            components.append(comp)

        emitted: List[Detection] = []
        for comp in components:
            # [Fix: unreachable topology classes] The old `len(comp) < 2`
            # guard skipped single-contour components, which made topology
            # classes 0 (single closed loop) and 1 (single open contour) --
            # two of the four classes this lens was designed to emit --
            # unreachable, and starved the whole Part/ObjectShape tier
            # whenever an object's structure condenses into one outline.
            # The contour-content check just below already rejects pure-
            # junction comps; size is not evidence.
            contour_ids = sorted(
                [n for n in comp if node_id_to_kind[n] == "Contour"]
            )
            junction_ids = sorted(
                [n for n in comp if node_id_to_kind[n] == "Junction"]
            )
            # A bundle of pure junctions carries no curve structure.
            if len(contour_ids) < 1:
                continue

            self._candidates_evaluated += 1

            # [UNIT TABLE] The bins come from the class' unit table
            # (core.registries.POPULATION_SLOTS["CURVE_COMPLEX"]), so the
            # lookup index and the graded population are quantisations of
            # the SAME measurements in the SAME units.
            n_c_bin = quantise(_CURVE_COMPLEX_SPEC["n_contours"],
                               float(len(contour_ids)))
            n_j_bin = quantise(_CURVE_COMPLEX_SPEC["n_junctions"],
                               float(len(junction_ids)))

            sorted_member_ids = sorted(comp)
            # alpha is a live observation anchor, so it is the FIRST
            # MEMBER's anchor, not a synthesised centroid (which would not be
            # a key in anchor_positions and could not be looked up later).
            first_member_node_id = sorted_member_ids[0]
            first_member_anchor = node_id_to_detection[
                first_member_node_id
            ].alpha

            # [Object-center pooling] Carry the complex's full V1 member
            # footprint ("member" anchor roles) so downstream lenses can
            # locate the structure CENTROID. Rim-only identity (first
            # member) sits on the object boundary; the footprint centroid
            # sits inside it. Deterministic: anchors deduplicated and
            # canonically ordered; ordinals re-indexed after ordering.
            footprint: List[Any] = []
            seen_anchor: Set[Any] = set()
            for mid in sorted_member_ids:
                for a in _extract_member_anchors(node_id_to_detection[mid]):
                    if a in seen_anchor:
                        continue
                    seen_anchor.add(a)
                    footprint.append(a)
            if not footprint:
                footprint = [first_member_anchor]
            footprint = sorted(footprint, key=serialize_value)

            # Spatial extent: normalised span of the component's anchors.
            _pts: List[Tuple[float, float]] = []
            for a in footprint:
                p = anchor_positions.get(a) if anchor_positions else None
                if p is not None and len(p) >= 2:
                    _pts.append((float(p[0]), float(p[1])))
            if len(_pts) >= 2:
                _span = max(
                    math.hypot(pa[0] - pb[0], pa[1] - pb[1])
                    for pa in _pts for pb in _pts
                )
            else:
                _span = 0.0
            _span_norm = min(1.0, _span / max(1.0, 6.0 * extraction_scale))
            extent_bin = quantise(_CURVE_COMPLEX_SPEC["extent"], _span_norm)

            # Closure: mean closure flag over the component's contours.
            _closure_sum = 0.0
            _valid_c = 0
            for cid in contour_ids:
                cq = node_id_to_detection[cid].sigma.quantized
                if len(cq) >= 4:
                    _closure_sum += float(cq[3])
                    _valid_c += 1
            _closure = _closure_sum / max(1, _valid_c)
            closure_bin = quantise(_CURVE_COMPLEX_SPEC["closure"],
                                   clamp(_closure, 0.0, 1.0))

            # Curvature-template bins over the contour members' sequences.
            template_hits = [0.0] * P.CURVE_COMPLEX_TEMPLATE_BINS
            for cid in contour_ids:
                cq = node_id_to_detection[cid].sigma.quantized
                if len(cq) >= 27:
                    seq = cq[7:27]
                    for k in range(20):
                        t_bin = int((k / 20.0) * P.CURVE_COMPLEX_TEMPLATE_BINS)
                        t_bin = min(P.CURVE_COMPLEX_TEMPLATE_BINS - 1, t_bin)
                        template_hits[t_bin] += (float(seq[k]) - 4.0) / 4.0
                elif len(cq) >= 7:
                    for idx, k in enumerate((4, 5, 6)):
                        t_bin = int((idx / 3.0) * P.CURVE_COMPLEX_TEMPLATE_BINS)
                        t_bin = min(P.CURVE_COMPLEX_TEMPLATE_BINS - 1, t_bin)
                        template_hits[t_bin] += (float(cq[k]) - 3.5) / 3.5
            _max_abs = max(1e-6, max(abs(h) for h in template_hits))
            # The CONTINUOUS template energies: each bin's accumulated
            # curvature energy, normalised by the bundle's strongest one.
            template_energies = [h / _max_abs for h in template_hits]
            template_bins = [
                float(quantise(_CURVE_COMPLEX_SPEC[f"template{i}"], e))
                for i, e in enumerate(template_energies)
            ]

            # Template-population bins over sampled curvature profiles.
            shape_pop = [0.0] * len(CURVATURE_TEMPLATES)
            _n_pops = 0
            for cid in contour_ids:
                prof = sample_curvature_profile(
                    node_id_to_detection[cid], anchor_positions, n=5
                )
                if prof is None or float(np.abs(prof).sum()) < 1e-6:
                    continue
                _best, _score, pop = match_templates(prof)
                for i in range(len(shape_pop)):
                    shape_pop[i] += float(pop[i])
                _n_pops += 1
            if _n_pops > 0:
                shape_pop = [v / _n_pops for v in shape_pop]
            shape_bins = [
                float(quantise(_CURVE_COMPLEX_SPEC[f"shape{i}"], v))
                for i, v in enumerate(shape_pop)
            ]

            quantized = (
                (float(n_c_bin), float(n_j_bin),
                 float(extent_bin), float(closure_bin))
                + tuple(float(b) for b in template_bins)
                + tuple(float(b) for b in shape_bins)
            )

            # [GRADED POPULATION] The CONTINUOUS bundle measurements, in the
            # order the CURVE_COMPLEX unit table declares: the contour count,
            # the junction count, the spatial extent (the normalised span),
            # the closure (the mean closure flag), the eight curvature-
            # template energies (each normalised to [-1, 1]) and the twelve
            # curvature-template populations (the softmax weight of each
            # template).
            _cc_pop = population_from_values(
                "CURVE_COMPLEX",
                (float(len(contour_ids)),
                 float(len(junction_ids)),
                 float(_span_norm),
                 float(clamp(_closure, 0.0, 1.0)))
                + tuple(float(e) for e in template_energies)
                + tuple(float(v) for v in shape_pop),
            )

            emitted.append(Detection(
                k="CurveComplex", source_layer=2, scale=0,
                sigma=TypedSignature(
                    kind="CANONICAL", class_="CURVE_COMPLEX",
                    relation="SHAPE_ARRANGEMENT",
                    quantized=quantized,
                    population=_cc_pop,
                ),
                alpha=first_member_anchor,
                z=min(1.0, (len(contour_ids) + len(junction_ids)) / 6.0), t=t,
                rho=(("entity", first_member_anchor, 0, "+"),)
                    + tuple(
                        ("member", footprint[i], i, "+")
                        for i in range(len(footprint))
                    )
                    + tuple(
                        ("member_node", (sorted_member_ids[i],), i, "+")
                        for i in range(len(sorted_member_ids))
                    ),
            ))

        self._emitted_total += len(emitted)
        return emitted

class CurveComplexLens_B13(HigherLens):
    """B13 — V4 Curve Arrangements & Shape Fragments (arity 24).

    STATE (ROUND 11 / O4): RETIRED — never constructed by the pipeline. The
    live CurveComplex path is ``CurveComplexLens`` (arity 3). This class is
    retained for enum/arity history and backward import compatibility only;
    see ``diagnostics/_O4_DEAD_MECHANISM_LEDGER.md``.
    """

    def _spatial_clustering(self, dets, anchor_positions, radius):
        return spatial_clustering(dets, anchor_positions, radius)

    def step(
        self,
        t: int,
        concept_graph: Any = None,
        sig_provider: Any = None,
        anchor_positions: Optional[Dict[Any, Tuple[float, ...]]] = None,
        anchor_velocities: Optional[Dict[Any, Tuple[float, ...]]] = None,
        current_detections: Optional[Dict[str, List[Detection]]] = None,
        extraction_scale: float = 0.0,
        **kwargs,
    ) -> List[Detection]:
        if not current_detections:
            return []
        contour_dets = current_detections.get("Contour", [])
        junction_dets = current_detections.get("Junction", [])
        if not contour_dets:
            return []

        radius = (
            P.KAPPA_CURVE_COMPLEX_RADIUS * extraction_scale
            if extraction_scale > 0
            else 50.0
        )
        clusters = (
            self._spatial_clustering(contour_dets, anchor_positions, radius)
            if anchor_positions
            else []
        )
        if not clusters:
            clusters = [contour_dets]

        emitted = []
        for cluster in clusters:
            emitted.extend(
                self._emit_cluster(
                    t, cluster, junction_dets, anchor_positions or {}, extraction_scale, concept_graph=concept_graph
                )
            )
        self._emitted_total += len(emitted)
        return emitted

    def _emit_cluster(
        self,
        t: int,
        contour_dets: List[Detection],
        junction_dets: List[Detection],
        anchor_positions: Dict[Any, Tuple[float, ...]],
        extraction_scale: float,
        concept_graph: Any = None,
    ) -> List[Detection]:
        n_contours = len(contour_dets)
        if n_contours == 0:
            return []
        n_junctions = len(junction_dets)

        _pos = [anchor_positions.get(c.alpha) for c in contour_dets]
        _pos = [p for p in _pos if p is not None and len(p) >= 2]
        if _pos:
            span = max(
                math.hypot(a[0] - b[0], a[1] - b[1])
                for a in _pos
                for b in _pos
            )
        else:
            span = 0.0
        span_norm = min(1.0, span / max(1.0, 6.0 * extraction_scale))
        extent_bin = qi(span_norm, 8, 0.0, 1.0)

        closure_sum = 0.0
        valid_c = 0
        for c in contour_dets:
            if len(c.sigma.quantized) >= 4:
                closure_sum += slot_value(c.sigma, "CONTOUR", "closure",
                                          default=0.0)
                valid_c += 1
        closure_score = closure_sum / max(1, valid_c)

        template_hits = [0.0] * P.CURVE_COMPLEX_TEMPLATE_BINS
        for c in contour_dets:
            if len(c.sigma.quantized) >= 27:
                seq = slot_slice(c.sigma, "CONTOUR", _CONTOUR_CURV_SEQUENCE)
                for k in range(20):
                    t_bin = int((k / 20.0) * P.CURVE_COMPLEX_TEMPLATE_BINS)
                    t_bin = min(P.CURVE_COMPLEX_TEMPLATE_BINS - 1, t_bin)
                    energy = (seq[k] - 4.0) / 4.0
                    template_hits[t_bin] += energy
            elif len(c.sigma.quantized) >= 7:
                for idx, nm in enumerate(("total_turn", "corner_count",
                                          "chirality")):
                    t_bin = int((idx / 3.0) * P.CURVE_COMPLEX_TEMPLATE_BINS)
                    energy = (slot_value(c.sigma, "CONTOUR", nm, default=3.5)
                              - 3.5) / 3.5
                    template_hits[t_bin] += energy

        max_abs = max(1e-6, max(abs(h) for h in template_hits))
        template_bins = [
            min(7.0, max(0.0, (h / max_abs) * 3.5 + 3.5))
            for h in template_hits
        ]

        n_c_bin = qi(n_contours, 8, 0.0, 7.0)
        n_j_bin = qi(n_junctions, 8, 0.0, 7.0)
        c_bin = qi(clamp(closure_score, 0.0, 1.0), 8, 0.0, 1.0)

        shape_pop = [0.0] * len(CURVATURE_TEMPLATES)
        n_pops = 0
        for c in contour_dets:
            prof = sample_curvature_profile(c, anchor_positions, n=5)
            if prof is None or float(np.abs(prof).sum()) < 1e-6:
                continue
            _best, _score, pop = match_templates(prof)
            for i in range(len(shape_pop)):
                shape_pop[i] += float(pop[i])
            n_pops += 1
        if n_pops > 0:
            shape_pop = [v / n_pops for v in shape_pop]
        shape_bins = [qi(v, 8, 0.0, 1.0) for v in shape_pop]

        quantized = (
            (float(n_c_bin), float(n_j_bin), float(extent_bin), float(c_bin))
            + tuple(float(b) for b in template_bins)
            + tuple(float(b) for b in shape_bins)
        )

        contour_dets_sorted = sorted(contour_dets, key=_detection_sort_key)
        first_anchor = contour_dets_sorted[0].alpha

        c_nids = []
        if concept_graph is not None:
            for d in contour_dets_sorted:
                nid = concept_graph.get_or_create_pattern_node(
                    d.k, _node_signature_hash(d.sigma), signature=d.sigma
                )
                c_nids.append(nid)

        rho = (
            (("entity", first_anchor, 0, "+"),)
            + tuple(
                ("member", d.alpha, i, "+")
                for i, d in enumerate(contour_dets_sorted)
            )
            + tuple(
                ("member_node", (c_nids[i],), i, "+")
                for i in range(len(c_nids))
            )
        )

        return [
            Detection(
                k="CurveComplex",
                source_layer=2,
                scale=0,
                sigma=TypedSignature(
                    kind="CANONICAL",
                    class_="CURVE_COMPLEX",
                    relation="SHAPE_ARRANGEMENT",
                    quantized=quantized,
                ),
                alpha=first_anchor,
                z=max(extent_bin / 7.0, closure_score),
                t=t,
                rho=rho,
            )
        ]

class RegionShapeLens(HigherLens):
    """B14 + B22 — V4/IT Boundary Conformation & Self-Symmetry (arity 21)."""

    def step(
        self,
        t: int,
        concept_graph: Any = None,
        sig_provider: Any = None,
        anchor_positions: Optional[Dict[Any, Tuple[float, ...]]] = None,
        anchor_velocities: Optional[Dict[Any, Tuple[float, ...]]] = None,
        current_detections: Optional[Dict[str, List[Detection]]] = None,
        extraction_scale: float = 0.0,
        envelope_mask: Optional[
            Tuple[Tuple[float, float, float, float], ...]
        ] = None,
        observation: Optional[Any] = None,
        **kwargs,
    ) -> List[Detection]:
        if observation is None:
            return []
        # The shape of a CONTOUR FRAGMENT: the chain's own member order,
        # closed only when the chain says it is a ring.
        from perception.chain_polylines import chain_polylines
        polys = chain_polylines(
            (current_detections or {}).get("PEAK_CHAIN", []),
            anchor_positions, with_closure=True)
        emitted = []
        for u, (poly, closed) in polys.items():
            if not poly or len(poly) < 3:
                continue
            (hist_bins, area_bin, elong_bin,
             sym_bin, axis_bin, fold_bin, _seq) = polyline_shape_descriptor(
                poly, observation, closed=closed
            )

            # [GRADED POPULATION] The CONTINUOUS shape measurements, in the
            # order the REGION_SHAPE unit table declares
            # (core.registries.POPULATION_SLOTS["REGION_SHAPE"]): the 16
            # curvature-histogram masses (the square root of the normalised
            # arc mass in each turn bin), the area (the log10 of the region's
            # area as a fraction of the image), the elongation (the
            # eigenvalue ratio, normalised), the symmetry score, the symmetry
            # AXIS (the axis angle in radians over the 12 registered axes)
            # and the fold count.
            # Every value is a measurement the emitter made; the axis is
            # None (an uninformative, uniform activity) when the outline is
            # isotropic and there is no preferred axis.
            (hist_values, area_norm, elong_norm,
             sym_score, axis_angle) = _region_shape_measurements(
                poly, observation, closed)
            if axis_angle is None:
                # The outline is isotropic: there is no preferred axis, so the
                # graded axis carries no measurement (a uniform activity) and
                # the lookup key keeps the undefined sentinel.
                axis_angle = None
                axis_bin_out = float(axis_bin)
            else:
                axis_bin_out = float(quantise(_REGION_SHAPE_SPEC["axis"],
                                              float(axis_angle)))
            _rs_pop = population_from_values(
                "REGION_SHAPE",
                tuple(float(v) for v in hist_values)
                + (float(area_norm),
                   float(elong_norm),
                   float(sym_score),
                   (float(axis_angle) if axis_angle is not None else None),
                   float(fold_bin)),
            )

            quantized = hist_bins + (float(area_bin),
                                     float(max(0.0, elong_bin)))
            quantized = quantized + (sym_bin, axis_bin_out, fold_bin)

            _det = Detection(
                k="RegionShape",
                source_layer=0,
                scale=0,
                sigma=TypedSignature(
                    kind="CANONICAL",
                    class_="REGION_SHAPE",
                    relation="BOUNDARY_CONFORMATION",
                    quantized=quantized,
                    population=_rs_pop,
                ),
                alpha=u,
                z=1.0,
                t=t,
                rho=(("entity", u, 0, "+"),),
            )
            object.__setattr__(_det, "_curv_profile", _seq)
            emitted.append(_det)
        self._emitted_total += len(emitted)
        return emitted

