"""perception/lenses/derived.py -- the derived-lens aggregator (Move 3.5).

    Tiers live in derived_base / v2 / v4 / s4; shared constants in
    derived_common. Their public names are re-exported here, so the
    legacy `from derived_lenses import X` surface is unchanged.

    [U22.5] `DerivedLenses` is NOT a perception lens: it owns Recognition,
    Operators and Chunks, reads the concept graph and writes memory. That is
    Tier 4, so the class now lives in `memory/derived_processes.py`; this
    module keeps a re-export at the same place in its namespace, which
    preserves every `from derived_lenses import DerivedLenses` call site.
    """
from __future__ import annotations

import math
from itertools import combinations
from typing import Any, Dict, List, Optional, Set, Tuple
import parameters as P
from concept_graph import (  # noqa: F401
    ARITY_MAX,
    Concept,
    ConceptStatus,
    EPSILON,
    Edge,
    KAPPA_BIRTH,
    TAU_EDGE_INACTIVE,
    TAU_OP_COOLDOWN,
    TAU_REVERT,
    THETA_EDGE_ACTIVE,
    THETA_HYST_OP,
)
from core.hashing import (  # noqa: F401
    _detection_sort_key,
    clamp,
    hash_64,
    serialize_bytes,
    serialize_float,
    serialize_int,
    serialize_set,
    serialize_tuple,
    serialize_value,
)
from core.types import (  # noqa: F401
    Detection,
    EdgeType,
    TypedSignature,
)
from learning.fertility import (  # noqa: F401
    KAPPA_EMA,
    RF_MAX,
)
from memory.cores import (  # noqa: F401
    _sig_similarity,
)
from parameters._text import (  # noqa: F401
    B_MAX,
    DELTA_CAT,
    DELTA_THETA_CAT_MAX,
    KAPPA_CAT,
    K_CATEGORIZE,
    RHO_CAT_TARGET,
    RHO_PROGRESS_MIN,
    TAU_PROGRESS,
    THETA_CATEGORIZE,
    THETA_CATEGORIZE_MAX,
    THETA_CATEGORIZE_MIN,
    THETA_SIM,
    W_CHUNK,
    W_PLAN_CHUNK,
)

# [U22.5] Tier 4 processes moved out of the perception tier.  This
# re-export keeps the legacy `derived` / `derived_lenses` surface intact.
from memory.derived_processes import DerivedLenses  # noqa: F401



# --- tier re-exports (legacy surface) ---
from perception.lenses.derived_base import (  # noqa: F401
    ContourGainState,
    EndStoppedLens,
    HigherLens,
)

from perception.lenses.v2 import (  # noqa: F401
    BorderOwnershipLens,
    ContourLens,
    JunctionLens,
    TextureBoundaryLens,
)

from perception.lenses.v4 import (  # noqa: F401
    CurveComplexLens,
    CurveComplexLens_B13,
    IllusoryContourLens,
    RegionShapeLens,
    SurfaceLens,
)


from perception.lenses.chain_lens import (  # noqa: F401  [A5]
    ChainLens,
)

from perception.lenses.derived_common import (  # noqa: F401
    ARITY_MAX,
    BO_COLOR_WEIGHT,
    BO_CUE_SCALE,
    BO_CURVATURE_WEIGHT,
    BO_GRAVITY_SCALE,
    BO_GRAVITY_WEIGHT,
    BO_RICHNESS_FALLBACK,
    BO_TEXTURE_BIN_MAX,
    BO_TEXTURE_WEIGHT,
    B_MAX,
    CURVATURE_TEMPLATES,
    DELTA_CAT,
    DELTA_CHUNK,
    DELTA_OP,
    DELTA_THETA_CAT_MAX,
    D_CHUNK_MAX,
    EPSILON,
    ETA_CTX_OP,
    ETA_FAT_DECAY_OP,
    ETA_FAT_OP,
    HIGHER_LENS_NAMES,
    KAPPA_BIRTH,
    KAPPA_CAT,
    KAPPA_CHUNK_HOMEO,
    KAPPA_EMA,
    KAPPA_OP,
    K_CATEGORIZE,
    K_CHUNK_CAND,
    K_CHUNK_PLAN,
    K_CHUNK_STAR,
    K_OP,
    RF_MAX,
    RHO_BIRTH,
    RHO_CAT_TARGET,
    RHO_CHUNK_TARGET,
    RHO_OP_TARGET,
    RHO_PROGRESS_MIN,
    RHO_PROMO_RF,
    SIGMA_MAX_OP,
    TAU_EDGE_INACTIVE,
    TAU_OP_COOLDOWN,
    TAU_PROGRESS,
    TAU_PROMO_MIN,
    TAU_REVERT,
    THETA_CATEGORIZE,
    THETA_CATEGORIZE_MAX,
    THETA_CATEGORIZE_MIN,
    THETA_CHUNK_EMIT,
    THETA_CHUNK_EMIT_MAX,
    THETA_CHUNK_EMIT_MIN,
    THETA_EDGE_ACTIVE,
    THETA_FLOOR_OP,
    THETA_HYST_OP,
    THETA_PROMO_OP,
    THETA_SIM,
    W_CHUNK,
    W_PLAN_CHUNK,
    Y_FLOOR_OP,
    _CONTOUR_RANGES,
    _JUNCTION_CANONICAL_BINS,
    _SIM_MEMO,
    _SIM_MEMO_CAP,
    _THETA_FRAME_SIGN,
    _TIE_MIN_COS,
    _TINY,
    _arm_tangent_stats,
    _canonical_curvature_sequence,
    _chamfer,
    _collect_continuity_neighbours,
    _compat_key,
    _contour_slots,
    _contour_turns,
    _envelope_cov,
    _oriented_tangents,
    _point_in_poly,
    _resample_closed,
    _sig_compatible,
    _sig_similarity,
    _signed_log_level,
    _sim_memo,
    _splat,
    _turn_positions,
    circ_mean_bin,
    compute_endstopped_map,
    contour_self_symmetry,
    diagnostic_derived_lenses,
    match_templates,
    qi,
    qi_circ,
    reflection_symmetry,
    sample_curvature_profile,
    texture_richness,
)
