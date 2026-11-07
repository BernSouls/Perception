"""perception/observation.py -- extracted from sensory_floor.py (Move 3a).

MOVE ONLY: whole top-level statements, verbatim. The re-export
shim in sensory_floor.py keeps `from sensory_floor import X`
working until the importer repoint retires it.
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from core.hashing import (  # noqa: F401
    clamp,
    hash_64,
    serialize_value,
)
from core.types import (  # noqa: F401
    TypedSignature,
)

_SF_TINY = 1e-12

EPSILON = 1e-9

HASH_BITS = 64

W_T = 10  # Presence / coherence window

W_R = 20  # Rhythm window (kept for external compat)

W_CM = 10  # Co-motion window (kept for external compat)

W_SEQ = 20  # Sequence window (kept for external compat)

DELTA_COHERENCE = 1  # Lag for temporal coherence autocorrelation

TAU_LIFE = 100.0  # Anchor age normalization (Ch 10)

TAU_NORM = 10.0  # State recency normalization (Ch 11)

R_LOCAL_3D = 0.30  # Ch 1 local spatial contrast radius (m)

R_CLUSTER = 0.20  # Common-fate spatial clustering (m) — adapter

R_CLUTTER = 0.30  # Ch 9 relational clutter radius (m)

R_STR_3D = 0.30  # 3D Structure lens radius (m) — kept for compat

V_MAX = 1.0  # Maximum linear speed for Ch 13 (m/s)

V_THRESH = 0.02  # Dynamic entity velocity threshold (m/s)

EPS_VEL = 0.03  # Common-fate velocity coherence threshold (m/s)

H_REF = 1.0  # Reference height for Ch 14 (m)

THETA_DEPTH_HIGH = 0.25  # Depth discontinuity upper trigger (m)

THETA_DEPTH_LOW = 0.12  # Depth discontinuity lower recovery (m)

R_MIN_SEP = 0.25  # 3D NMS separation (m)

EPS_GATE = 0.15  # Predictive centroid gating (m)

TAU_PERSIST = 10  # Occlusion persistence window (frames)

EPS_CONTACT = 0.03  # Surface contact tolerance for A matrix (m)

ALPHA_P = 0.20  # Low-pass position filter coefficient

M_KEYPOINTS = 100  # Max tracked sparse corner points

R_MIN = 0.04  # Minimum entity bounding radius (m)

R_MAX_OBJ = 0.50  # Reference maximum radius (m)

N_ANCHOR_MAX = 16  # Working-memory bound (slot count)

U_AGENT = 0  # Agent effector anchor slot

Q_S = 0.1  # Legacy: used by _select_representative (2D only)

DEG_MAX = 50  # 2D contact degree normalization

DEG_MAX_3D = 8  # 3D contact degree normalization (§4.3)

BETA_T = 0.90  # Ch 3 temporal presence EMA decay

ACTION_LENS_NAME = "Action"

ACTION_SIG_TAG = "ACTION"

ACTION_ROLE_NAME = "action"

def action_signature(action: Any) -> TypedSignature:
    """Canonical signature of an executed action under the Action lens.

    OPAQUE by design (root-cause decision against the draft's CANONICAL
    bucket-mod-N signature): actions are CATEGORICAL — their identity must
    be exact. Mod-bucket quantization collides distinct actions with
    birthday probability (~60% for a 12-action repertoire at 64 bins), and
    a collision merges two actions into ONE pattern node under the §1.11
    identity (lens_type, signature_hash) — silently corrupting vocabulary
    recovery, because the per-node action registry can then hold only one
    of the two actions. The full 64-bit canonical hash makes node identity
    exact (collision probability ≤ 2⁻⁶⁴), and OPAQUE×OPAQUE similarity
    (§2.21.5) is exact equality — the correct semantics for categorical
    identity: no spurious ordinal distance between unrelated actions, and
    no CONSTITUTIONAL_RANGES registration required.
    """
    return TypedSignature.opaque(ACTION_SIG_TAG, hash_64(serialize_value(action)))

def resolve_anchor_position(obs: Any, anchor: Any) -> Optional[Tuple[float, float]]:
    """[Seam 2] Environment-agnostic position probe for one anchor.

    Handles both observation conventions: symbolic envs expose
    `obs.positions` as a dict keyed by atom index; 3D observations expose
    `get_positions()` as a sequence. The symbolic (-1, -1) flag
    placeholder — and any non-numeric, non-finite, or short entry —
    resolves to None (NO position). Single source of truth: the
    UtilityEngine's goal-anchor fallback and the Pipeline's trigger-anchor
    grounding must agree on what "has a position" means, or dual
    grounding desynchronizes.
    """
    if anchor is None or obs is None:
        return None
    positions = getattr(obs, "positions", None)
    if positions is None:
        getter = getattr(obs, "get_positions", None)
        if callable(getter):
            positions = getter()
    if positions is None:
        return None
    pos: Any = None
    if isinstance(positions, dict):
        pos = positions.get(anchor)
    elif isinstance(anchor, int) and not isinstance(anchor, bool) and 0 <= anchor < len(positions):
        pos = positions[anchor]
    if pos is None or not isinstance(pos, (list, tuple)) or len(pos) < 2:
        return None
    try:
        x = float(pos[0])
        y = float(pos[1])
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(x) and math.isfinite(y)):
        return None
    if x == -1.0 and y == -1.0:
        return None  # flag placeholder — not a position
    return (x, y)

#: [STATE LAYOUT] The 16 channels of the state pattern, by index. This is
#: the legend the tree never had: every consumer of `StatePattern.X` reads a
#: channel by index, and the same index used to mean different things in the
#: symbolic, 2-D and 3-D paths. The tuple below is the single statement of
#: what each channel is, and which path fills it.
#:
#:  0 presence / flag value    all paths   (2-D: presence strength)
#:  1 spatial contrast (Y_C_space)   2-D, 3-D, symbolic (measured, image-derived)
#:  2 temporal contrast (Y_C_time)   2-D, 3-D   (interface: the motion session)
#:  3 temporal coherence (Y_T)   2-D, 3-D   (interface: the motion session)
#:  4 -- RESERVED (the entity flag was removed: it was 1.0 for every atom)
#:  5 flag marker               symbolic
#:  6 effect marker             symbolic
#:  7 flag value                symbolic
#:  8 degree (normalised)       2-D, 3-D
#:  9 clutter count / relative degree  symbolic, 3-D
#: 10 age                       symbolic, 3-D (interface: motion session)
#: 11 recency                   symbolic, 3-D (interface: motion session)
#: 12 radius                    3-D
#: 13 speed                     3-D          (interface: the motion session)
#: 14 elevation                 3-D
#: 15 planarity                 3-D
#:
#: (The legend follows the CODE, which is what the three paths write:
#:  symbolic  9 = relative degree, 10 = age, 11 = recency;
#:  3-D       9 = clutter count, 10 = age, 11 = recency, 12 = radius,
#:            13 = speed, 14 = elevation, 15 = planarity.)
STATE_CHANNEL_NAMES: Tuple[str, ...] = (
    "presence",
    "spatial_contrast",
    "temporal_contrast",
    "temporal_coherence",
    "reserved",
    "flag_marker",
    "effect_marker",
    "flag_value",
    "degree",
    "age",
    "recency",
    "radius",
    "speed",
    "elevation",
    "planarity",
    "clutter_count",
)

STATE_CHANNEL_COUNT: int = len(STATE_CHANNEL_NAMES)


@dataclass
class StatePattern:
    """The per-atom state. `X` is a fixed 16-channel row per atom.

    The channel semantics are documented once, in `STATE_CHANNEL_NAMES`
    (this module). Channels 1..3 are the transduction outputs (spatial
    contrast, temporal contrast, temporal coherence); channels 9, 10 and 12
    are temporal and are left as INTERFACE for the motion session: the
    image path does not fill them, and does not pretend to.
    """

    t: int
    X: List[List[float]]
    R: List[List[float]]
    A: List[List[int]]
    entity_indices: List[int] = field(default_factory=list)
    raw_indices: List[int] = field(default_factory=list)
    modality: str = "symbolic"
    Y_C_time: List[float] = field(default_factory=list)
    state_hash: int = 0

    def num_atoms(self) -> int:
        return len(self.X)

@dataclass
class Transition:
    prev_state: StatePattern
    action: Any
    next_state: StatePattern
    t: int

class Observation:
    def modality(self) -> str:
        raise NotImplementedError

    def num_atoms(self) -> int:
        raise NotImplementedError

    def get_feature(self, idx: int, col: int) -> float:
        raise NotImplementedError

    def get_distance(self, i: int, j: int) -> float:
        raise NotImplementedError

    def get_adjacency(self, i: int, j: int) -> int:
        raise NotImplementedError

    def is_present(self, idx: int) -> bool:
        raise NotImplementedError

    def get_entity_indices(self) -> List[int]:
        raise NotImplementedError

    def get_flag_indices(self) -> List[int]:
        raise NotImplementedError

    def get_effect_indices(self) -> List[int]:
        raise NotImplementedError

    def get_extraction_scale(self) -> float:
        """Median V1 region radius in the observation's coordinate units
        (same units as get_positions()). Returns 0.0 for symbolic modality
        or observations with no per-region radii.

        [PHASE B] Step 4. The higher tier's spatial gates are multiples of
        this value (see the KAPPA_* constants in parameters.py) rather than
        absolute radii, so they track the extractor's actual granularity.
        """
        return 0.0

class SymbolicObservation(Observation):
    def __init__(
        self,
        entity_indices,
        flag_indices,
        effect_indices,
        distances,
        adjacency_threshold,
        presence=None,
        flag_values=None,
    ):
        self._entity = sorted(entity_indices)
        self._flag = sorted(flag_indices)
        self._effect = sorted(effect_indices)
        self._dist = distances
        self._adj_thresh = adjacency_threshold
        self._presence = presence if presence is not None else [True] * len(distances)
        self._flag_values = (
            flag_values if flag_values is not None else [1.0] * len(self._flag)
        )
        self._N = len(distances)
        self._flag_value_map = dict(zip(self._flag, self._flag_values))
        self.entity_indices = self._entity
        self.presence = self._presence
        self.distances = self._dist

    def modality(self):
        return "symbolic"

    def num_atoms(self):
        return self._N

    def get_feature(self, idx, col):
        raise NotImplementedError

    def get_distance(self, i, j):
        return self._dist[i][j]

    def get_adjacency(self, i, j):
        return 1 if self._dist[i][j] <= self._adj_thresh and i != j else 0

    def get_entity_indices(self):
        return self._entity

    def get_flag_indices(self):
        return self._flag

    def get_effect_indices(self):
        return self._effect

    def is_present(self, idx):
        return self._presence[idx]

    def get_flag_value(self, idx):
        return self._flag_value_map.get(idx, 0.0)

class ContinuousObservation(Observation):
    """DEPRECATED: Retained for legacy 2D continuous environments only.
    New environments must use Observation3D (3D) or SymbolicObservation
    (discrete). No current v31 environment uses this path."""

    def __init__(
        self,
        raw_features,
        distances,
        adjacency_threshold,
        flag_indices=None,
        effect_indices=None,
        flag_values=None,
    ):
        self._feat = raw_features
        self._dist = distances
        self._adj_thresh = adjacency_threshold
        self._N = len(raw_features)
        self._flag = sorted(flag_indices) if flag_indices else []
        self._effect = sorted(effect_indices) if effect_indices else []
        self._flag_values = flag_values if flag_values else [1.0] * len(self._flag)
        self._flag_value_map = dict(zip(self._flag, self._flag_values))

    def modality(self):
        return "continuous"

    def num_atoms(self):
        return self._N

    def get_feature(self, idx, col):
        if 0 <= col < 4:
            return self._feat[idx][col]
        raise ValueError("Only columns 0-3 provided for continuous.")

    def get_distance(self, i, j):
        return self._dist[i][j]

    def get_adjacency(self, i, j):
        return 1 if self._dist[i][j] <= self._adj_thresh and i != j else 0

    def get_flag_indices(self):
        return self._flag

    def get_effect_indices(self):
        return self._effect

    def get_flag_value(self, idx):
        return self._flag_value_map.get(idx, 0.0)

    def is_present(self, idx: int) -> bool:
        return True

class Observation3D(Observation):
    """3D observation container produced by the RGBDCommonFateTracker adapter.
    Bounds-safe: accessors never raise IndexError on arbitrary atom queries.
    """

    def __init__(
        self,
        positions: List[List[float]],
        radii: List[float],
        velocities: List[List[float]],
        presence: List[bool],
        birth_times: List[int],
        planarity: List[float],
        agent_index: int = 0,
        elongation: Optional[List[float]] = None,
        flag_indices: Optional[List[int]] = None,
        effect_indices: Optional[List[int]] = None,
        flag_values: Optional[List[float]] = None,
        anchor_rgb: Optional[List[Optional[Tuple[float, float, float]]]] = None,
    ):
        self._positions = positions
        self._radii = radii
        self._velocities = velocities
        self._presence = presence
        self._birth_times = birth_times
        self._planarity = planarity
        # SIZE channel 1 (option (a)): inertia-axis ratio per anchor.
        # Isotropic default 1.0 when the adapter provides no shape tensor.
        self._elongation = (
            list(elongation) if elongation is not None
            else [1.0] * len(positions)
        )
        self._agent_index = agent_index
        self._N = len(positions)
        self._flag = sorted(flag_indices) if flag_indices else []
        self._effect = sorted(effect_indices) if effect_indices else []
        self._flag_values = flag_values if flag_values else [1.0] * len(self._flag)
        self._flag_value_map = dict(zip(self._flag, self._flag_values))

        if anchor_rgb is None:
            self._anchor_rgb: List[Optional[Tuple[float, float, float]]] = [
                None
            ] * self._N
        else:
            if len(anchor_rgb) != self._N:
                raise ValueError(
                    f"anchor_rgb length {len(anchor_rgb)} != atom count {self._N}"
                )
            self._anchor_rgb = list(anchor_rgb)

        self._dist: List[List[float]] = [
            [float("inf")] * self._N for _ in range(self._N)
        ]
        for i in range(self._N):
            self._dist[i][i] = 0.0
            if not self._presence[i]:
                continue
            for j in range(i + 1, self._N):
                if not self._presence[j]:
                    continue
                dx = positions[i][0] - positions[j][0]
                dy = positions[i][1] - positions[j][1]
                dz = positions[i][2] - positions[j][2]
                d = math.sqrt(dx * dx + dy * dy + dz * dz)
                self._dist[i][j] = d
                self._dist[j][i] = d

        self._adj: List[List[int]] = [[0] * self._N for _ in range(self._N)]
        for i in range(self._N):
            for j in range(i + 1, self._N):
                if not self._presence[i] or not self._presence[j]:
                    continue
                contact_dist = radii[i] + radii[j] + EPS_CONTACT
                if self._dist[i][j] <= contact_dist:
                    self._adj[i][j] = 1
                    self._adj[j][i] = 1

    @property
    def presence(self) -> List[bool]:
        return self._presence

    def modality(self) -> str:
        return "3d"

    def num_atoms(self) -> int:
        return self._N

    def get_feature(self, idx: int, col: int) -> float:
        raise NotImplementedError(
            "Observation3D features are computed in _construct_3d_state"
        )

    def get_distance(self, i: int, j: int) -> float:
        if i < 0 or i >= self._N or j < 0 or j >= self._N:
            return float("inf") if i != j else 0.0
        return self._dist[i][j]

    def get_adjacency(self, i: int, j: int) -> int:
        if i < 0 or i >= self._N or j < 0 or j >= self._N:
            return 0
        return self._adj[i][j]

    def is_present(self, idx: int) -> bool:
        if idx < 0 or idx >= self._N:
            return False
        return self._presence[idx]

    def get_entity_indices(self) -> List[int]:
        return [
            i
            for i in range(self._N)
            if self._presence[i] and i not in self._flag and i not in self._effect
        ]

    def get_flag_indices(self) -> List[int]:
        return self._flag

    def get_effect_indices(self) -> List[int]:
        return self._effect

    def get_flag_value(self, idx: int) -> float:
        return self._flag_value_map.get(idx, 0.0)

    def get_position(self, u: int) -> List[float]:
        if u < 0 or u >= self._N:
            return [0.0, 0.0, 0.0]
        return self._positions[u]

    def get_radius(self, u: int) -> float:
        if u < 0 or u >= self._N:
            return R_MIN
        return self._radii[u]

    def get_extraction_scale(self) -> float:
        """[PHASE B] Step 4. Median entity radius in METERS, the same units
        get_positions() returns. Reads _radii directly rather than
        get_radius() so the value is never silently swapped for a
        normalized quantity.
        """
        radii = []
        for u in self.get_entity_indices():
            try:
                r = float(self._radii[u])
            except (IndexError, TypeError, ValueError):
                continue
            if math.isfinite(r) and r > 0.0:
                radii.append(r)
        if not radii:
            return 0.0
        radii.sort()
        mid = len(radii) // 2
        if len(radii) % 2 == 1:
            return radii[mid]
        return 0.5 * (radii[mid - 1] + radii[mid])

    def get_velocity(self, u: int) -> List[float]:
        if u < 0 or u >= self._N:
            return [0.0, 0.0, 0.0]
        return self._velocities[u]

    def get_speed(self, u: int) -> float:
        if u < 0 or u >= self._N:
            return 0.0
        v = self._velocities[u]
        return math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)

    def get_birth_time(self, u: int) -> int:
        if u < 0 or u >= self._N:
            return 0
        return self._birth_times[u]

    def get_elongation(self, u: int) -> float:
        """Inertia-axis elongation sqrt(lambda2/lambda1) in [0, 1].
        Bounds-safe; isotropic 1.0 for absent/out-of-range anchors."""
        if u < 0 or u >= self._N:
            return 1.0
        return self._elongation[u]

    # --- 2D-channel proxies for the 3D path (documented): the micro-lens
    # channels introduced by the visual enrichment patches have honest 3D
    # analogs in the anchor point-cluster shape invariants already carried
    # by this container (planarity, elongation).
    def get_circularity(self, u: int) -> float:
        """Roundness proxy: cluster elongation (1.0 = isotropic)."""
        return self.get_elongation(u)

    def get_complexity(self, u: int) -> float:
        """Surface complexity proxy: planarity index."""
        return self.get_planarity(u)

    def get_roughness(self, u: int) -> float:
        """Surface regularity carry-over: planarity index."""
        return self.get_planarity(u)

    def get_isotropy(self, u: int) -> float:
        """Directional isotropy proxy: cluster elongation."""
        return self.get_elongation(u)

    def get_planarity(self, u: int) -> float:
        if u < 0 or u >= self._N:
            return 0.0
        return self._planarity[u]

    def get_anchor_rgb(self, u: int) -> Optional[Tuple[float, float, float]]:
        if u < 0 or u >= self._N:
            return None
        return self._anchor_rgb[u]

    def get_elevation(self, u: int) -> float:
        if u < 0 or u >= self._N or not self._presence[u]:
            return 0.5
        if (
            self._agent_index < 0
            or self._agent_index >= self._N
            or not self._presence[self._agent_index]
        ):
            return 0.5
        y_u = self._positions[u][1]
        y_agent = self._positions[self._agent_index][1]
        return clamp((y_u - y_agent) / H_REF + 0.5, 0.0, 1.0)

    def get_agent_index(self) -> int:
        return self._agent_index

    def get_positions(self) -> List[List[float]]:
        return self._positions

@dataclass
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float

@dataclass
class AdapterConfig:
    M: int = M_KEYPOINTS
    R_cluster: float = R_CLUSTER
    eps_vel: float = EPS_VEL
    v_thresh: float = V_THRESH
    theta_depth_high: float = THETA_DEPTH_HIGH
    theta_depth_low: float = THETA_DEPTH_LOW
    R_min_sep: float = R_MIN_SEP
    eps_gate: float = EPS_GATE
    tau_persist: int = TAU_PERSIST
    N_anchor_max: int = N_ANCHOR_MAX
    r_min: float = R_MIN
    R_max_obj: float = R_MAX_OBJ
    eps_contact: float = EPS_CONTACT
    alpha_p: float = ALPHA_P
    depth_grid_w: int = 64
    depth_grid_h: int = 48

@dataclass
class AnchorState:
    id: int
    centroid: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    filtered_centroid: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    velocity: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    radius: float = R_MIN
    birth_step: int = 0
    last_seen_step: int = 0
    presence_ema: float = 0.0
    salience_ema: float = 0.0
    is_agent: bool = False
    active: bool = True

class AnchorSet:
    """Bounded container for AnchorState records.
    Enforces |U_t| <= N_ANCHOR_MAX with deterministic tie-break on salience_ema.
    Slot 0 is permanently reserved for the agent effector."""

    def __init__(self, capacity: int = N_ANCHOR_MAX):
        self._capacity = capacity
        self._anchors: Dict[int, AnchorState] = {}

    def add(self, anchor: AnchorState) -> bool:
        self._anchors[anchor.id] = anchor
        if len(self._anchors) > self._capacity:
            self.truncate_to_cap()
        return True

    def get(self, anchor_id: int) -> Optional[AnchorState]:
        return self._anchors.get(anchor_id)

    def remove(self, anchor_id: int):
        self._anchors.pop(anchor_id, None)

    def active_anchors(self) -> List[AnchorState]:
        return [a for a in self._anchors.values() if a.active]

    def truncate_to_cap(self):
        if U_AGENT not in self._anchors:
            warnings.warn(
                f"AnchorSet.truncate_to_cap: U_AGENT ({U_AGENT}) not in "
                f"_anchors. Adapter contract violated.",
                stacklevel=2,
            )
        non_agent = [a for a in self._anchors.values() if not a.is_agent]
        non_agent.sort(key=lambda a: (-a.salience_ema, a.id))
        keep_ids = {U_AGENT}
        for a in non_agent[: self._capacity - 1]:
            keep_ids.add(a.id)
        to_remove = [aid for aid in self._anchors if aid not in keep_ids]
        for aid in to_remove:
            del self._anchors[aid]

def median(values):
    if not values:
        return 1.0
    sv = sorted(values)
    n = len(sv)
    if n % 2 == 0:
        return (sv[n // 2 - 1] + sv[n // 2]) / 2.0
    return sv[n // 2]

def normalize(vals):
    mv = max(vals) if vals else 0.0
    if mv <= 0:
        return [0.0] * len(vals)
    return [v / (mv + EPSILON) for v in vals]

