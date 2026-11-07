"""perception/floor.py -- extracted from sensory_floor.py (Move 3a).

MOVE ONLY: whole top-level statements, verbatim. The re-export
shim in sensory_floor.py keeps `from sensory_floor import X`
working until the importer repoint retires it.
"""
from __future__ import annotations

import copy
import math
import warnings
from collections import deque
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from core.hashing import (  # noqa: F401
    clamp,
    hash_64,
    serialize_float,
    serialize_int,
    serialize_set,
    serialize_tuple,
)
from core.types import (  # noqa: F401
    Detection,
)
import numpy as np

from perception.observation import (  # noqa: F401
    BETA_T,
    STATE_CHANNEL_COUNT,
    ContinuousObservation,
    DEG_MAX,
    DEG_MAX_3D,
    DELTA_COHERENCE,
    EPSILON,
    Observation,
    Observation3D,
    R_CLUTTER,
    R_LOCAL_3D,
    R_MAX_OBJ,
    StatePattern,
    SymbolicObservation,
    TAU_LIFE,
    TAU_NORM,
    Transition,
    V_MAX,
    W_T,
    median,
)
from perception.observation_2d import (  # noqa: F401
    Observation2D,
)

class SensoryFloor:
    """State construction engine. Owns history tracking, StatePattern
    construction, and state hashing. Delegates raw lens execution to a
    LensExecutor instance (created lazily or injected).
    """

    def __init__(
        self,
        env_median_distance: Optional[float] = None,
        env_max_radius: Optional[float] = None,
        lens_executor: Optional[Any] = None,
    ):
        self.t = -1
        self.prev_state: Optional[StatePattern] = None
        self.prev_action: Optional[Any] = None
        self.R_med = env_median_distance
        self.R_env_max = env_max_radius

        # History for temporal operators
        self.presence_history: Dict[int, deque] = {}
        self.cont_history: deque = deque(maxlen=W_T + DELTA_COHERENCE)
        self.prev_X_C: Optional[List[List[float]]] = None
        # Within-step memo for _distance_aware_smoothing.
        self._smoothing_cache: Dict[Tuple[int, int, int, float], List[List[float]]] = {}

        # Onset and presence tracking (used by state construction and hash)
        self.onset_times: Dict[int, int] = {}
        self.last_presence: Dict[int, bool] = {}
        self.atom_birth: Dict[int, int] = {}
        self.last_change_time: Dict[int, int] = {}
        self.prev_features: Dict[int, List[float]] = {}
        self.prev_flag_values: Dict[int, float] = {}

        # Relational distance profile tracking (entity atoms only)
        self.prev_distances: Dict[int, List[float]] = {}

        # Last-present feature cache
        self.last_present_features: Dict[int, List[float]] = {}

        # 3D EMA presence tracker (Ch 3)
        self.presence_ema_3d: Dict[int, float] = {}

        # Anchor spawning flag (public after step)
        self.new_anchor_spawned = False
        self._prev_entity_indices: Set[int] = set()

        # Radius defaults (updated on first state construction)
        self.R_BOUNDARY: Optional[float] = self.R_med
        self.R_TOPOLOGY: Optional[float] = (2 * self.R_med) if self.R_med else None
        self.R_DEGREE: Optional[float] = self.R_med
        self.R_STR: Optional[float] = self.R_med
        if self.R_env_max is None and self.R_med is not None:
            self.R_env_max = 10 * self.R_med

        # Lens executor: optional injection. Created lazily on first step()
        # to avoid a module-scope import cycle with lenses.py.
        self._lens_executor = lens_executor

    # -------------------------------------------------------------------------
    # Public interface
    # -------------------------------------------------------------------------

    def step(
        self,
        O_t: Observation,
        a_prev: Optional[Any] = None,
        dir_tuples: Optional[List[Tuple[int, int, int]]] = None,
        concept_effects: Optional[Set[int]] = None,
        gain_by_anchor: Optional[Dict[Any, float]] = None,
        envelope_mask: Optional[
            Tuple[Tuple[float, float, float, float], ...]
        ] = None,
        update_homeostasis: bool = True,
        object_envelope: Optional[
            Tuple[float, float, float, float]
        ] = None,
        scale_level: int = 0,
        anchor_positions: Optional[Dict[Any, Tuple]] = None,
    ) -> Tuple[Set[Detection], StatePattern, int, Optional[Transition]]:
        """Process one step: update history, construct state, compute hash,
        delegate lens execution, return (D_raw, S_t, s_t, T_prev).

        gain_by_anchor — [Phase B B.14] recurrent-grouping gain, threaded
        through to LensExecutor.run_all. None ⇒ no gain.
        """
        self._smoothing_cache.clear()
        self.t += 1
        t = self.t

        self._update_history(O_t)
        S_t = self._construct_state(O_t)
        self._validate_state(S_t)
        s_t = self._compute_state_hash(S_t, dir_tuples=dir_tuples)
        S_t.state_hash = s_t

        T_prev = None
        if t > 0 and self.prev_state is not None:
            T_prev = Transition(
                prev_state=self.prev_state, action=a_prev, next_state=S_t, t=t - 1
            )

        D_raw = self._get_lens_executor().run_all(
            S_t, O_t, t, dir_tuples, prev_action=a_prev,
            concept_effects=concept_effects,
            gain_by_anchor=gain_by_anchor,
            envelope_mask=envelope_mask,
            update_homeostasis=update_homeostasis,
            object_envelope=object_envelope,
            scale_level=scale_level,
            anchor_positions=anchor_positions,
        )

        self.prev_state = S_t
        self.prev_action = a_prev

        return D_raw, S_t, s_t, T_prev

    # -------------------------------------------------------------------------
    # Lens executor access (lazy import to avoid circular dependency)
    # -------------------------------------------------------------------------

    def _get_lens_executor(self) -> Any:
        if self._lens_executor is None:
            from perception.lenses.base import LensExecutor

            self._lens_executor = LensExecutor()
        return self._lens_executor

    def make_action_detection(self, action: Any, t: int) -> Any:
        """Public delegation to the Action lens (B2): construct the
        Detection for an action executed at step `t`. The Pipeline calls
        this at execution time to pre-register the action pattern's onset
        in the Binder (correct SEQUENCE attribution lag); the lens itself
        re-emits the identical pattern at t+1 for binding presence."""
        return self._get_lens_executor().make_action_detection(action, t)

    # -------------------------------------------------------------------------
    # State construction
    # -------------------------------------------------------------------------

    def _validate_state(self, S_t: StatePattern):
        for row in S_t.X:
            for val in row:
                assert 0.0 <= val <= 1.0, "Feature out of [0,1]"
        for i in range(S_t.num_atoms()):
            assert S_t.A[i][i] == 0, "Adjacency diagonal must be zero"

    def _construct_state(
        self,
        O_t: Union[SymbolicObservation, ContinuousObservation, Observation3D],
    ) -> StatePattern:
        if self.R_med is None:
            all_dists = []
            N = O_t.num_atoms()
            for i in range(N):
                for j in range(i + 1, N):
                    d = O_t.get_distance(i, j)
                    if d < float("inf"):
                        all_dists.append(d)
            self.R_med = max(median(all_dists), EPSILON) if all_dists else 1.0
            self.R_BOUNDARY = self.R_med
            self.R_TOPOLOGY = 2 * self.R_med
            self.R_DEGREE = self.R_med
            self.R_STR = self.R_med
            if self.R_env_max is None:
                self.R_env_max = 10 * self.R_med

        modality = O_t.modality()
        if modality == "symbolic":
            return self._construct_symbolic_state(O_t)
        elif modality == "3d":
            return self._construct_3d_state(O_t)
        elif modality == "2d":
            return self._construct_2d_state(O_t)
        elif modality == "continuous":
            return self._construct_continuous_state_legacy(O_t)
        raise ValueError(f"Unknown modality: {modality}")

    def _features_2d(self, O_t: Observation2D, N: int) -> List[List[float]]:
        """MEASURED per-atom feature vectors for the image path.

        Four numbers per atom, all read from the observation:
          0  the atom's pooled Gabor energy (how much structure it
             carries), normalised by the strongest atom in the frame so
             the measure is relative and scale-free;
          1  its roughness (a second-order texture measure);
          2  its elongation;
          3  presence.

        These seed the transduction, so the spatial-contrast channel is
        a MEASUREMENT. Before this, the 2-D feature vector was
        [1, 0, 0, 0] for every atom, so the contrast was identically
        zero and the transduction was unreachable for images.
        """
        raw: List[float] = []
        getter = getattr(O_t, "get_gabor_responses", None)
        for i in range(N):
            energy = 0.0
            g = None
            if callable(getter):
                try:
                    g = getter(i)
                except Exception:
                    g = None
            if g:
                c = g.get("complex_pooled")
                if c is not None:
                    try:
                        arr = np.asarray(c, dtype=np.float64)
                        energy = float(arr.mean()) if arr.size else 0.0
                    except Exception:
                        energy = 0.0
            raw.append(max(0.0, energy))
        peak = max(raw) if raw else 0.0
        scale = peak if peak > 1e-9 else 1.0

        out: List[List[float]] = []
        for i in range(N):
            try:
                rough = float(O_t.get_roughness(i))
            except Exception:
                rough = 0.0
            try:
                elong = float(O_t.get_elongation(i))
            except Exception:
                elong = 1.0
            out.append([
                min(1.0, raw[i] / scale),
                min(1.0, max(0.0, rough)),
                min(1.0, max(0.0, elong)),
                1.0,
            ])
        return out

    def _construct_2d_state(self, O_t: Observation2D) -> StatePattern:
        N = O_t.num_atoms()
        X = [[0.0] * STATE_CHANNEL_COUNT for _ in range(N)]
        R = [[O_t.get_distance(i, j) for j in range(N)] for i in range(N)]
        A = [[O_t.get_adjacency(i, j) for j in range(N)] for i in range(N)]

        entity_indices = list(range(N))
        for i in entity_indices:
            X[i][0] = 1.0  # Presence
            deg = sum(A[i])
            X[i][8] = min(deg / 8.0, 1.0)

        # [STATE] The spatial / temporal channels are the transduction
        # outputs. The image path seeds them with MEASURED features, so
        # channel 1 is a measurement (it used to be 0.0 for every atom,
        # which made the state -- and its hash -- blind to the image).
        _feats = self._features_2d(O_t, N)
        _Y_I, _Y_C_space, _Y_C_time, _Y_T = (
            self._compute_transduction_responses(
                X, R, A, modality="2d", O_t=O_t, features=_feats))
        for i in range(N):
            X[i][1] = _Y_C_space[i]
            X[i][2] = _Y_C_time[i]
            X[i][3] = _Y_T[i]

        return StatePattern(
            t=self.t,
            X=X,
            R=R,
            A=A,
            entity_indices=entity_indices,
            raw_indices=list(range(N)),
            modality="2d",
        )

    # -------------------------------------------------------------------------
    # Symbolic state construction (unchanged from v30)
    # -------------------------------------------------------------------------

    def _construct_symbolic_state(self, O_t: SymbolicObservation) -> StatePattern:
        N = O_t.num_atoms()
        entity_indices = O_t.get_entity_indices()
        flag_indices = O_t.get_flag_indices()
        effect_indices = O_t.get_effect_indices()
        presence = [O_t.is_present(i) for i in range(N)]

        X = [[0.0] * 16 for _ in range(N)]
        R = [[float("inf")] * N for _ in range(N)]
        A = [[0] * N for _ in range(N)]

        for i in range(N):
            for j in range(N):
                d = O_t.get_distance(i, j)
                if d < float("inf"):
                    R[i][j] = d
                    if i != j and O_t.get_adjacency(i, j):
                        A[i][j] = 1

        for i in range(N):
            if i in entity_indices:
                deg = sum(A[i])
                X[i][8] = min(deg / DEG_MAX, 1.0)
                rel_deg = sum(1 for j in range(N) if R[i][j] < float("inf"))
                X[i][9] = min(rel_deg / DEG_MAX, 1.0)
                if i not in self.atom_birth:
                    self.atom_birth[i] = self.t
                X[i][10] = min((self.t - self.atom_birth[i]) / TAU_LIFE, 1.0)
                if i not in self.last_change_time:
                    self.last_change_time[i] = self.t
                X[i][11] = clamp(
                    1.0 / (1.0 + (self.t - self.last_change_time[i]) / TAU_NORM),
                    0.0,
                    1.0,
                )
                X[i][0] = 1.0 if presence[i] else 0.0
                if i not in self.presence_history:
                    self.presence_history[i] = deque(maxlen=W_T)
                    for _ in range(min(W_T, self.t)):
                        self.presence_history[i].append(0.0)
                X[i][3] = sum(self.presence_history[i]) / W_T

            if i in flag_indices:
                X[i][5] = 1.0
                X[i][0] = O_t.get_flag_value(i)

            if i in effect_indices:
                X[i][6] = 1.0
                X[i][0] = 1.0 if presence[i] else 0.0


        # Relational onsets for entity atoms
        for i in entity_indices:
            if presence[i]:
                curr_dists = [O_t.get_distance(i, j) for j in range(N)]
                prev_d = self.prev_distances.get(i)
                if prev_d is not None:
                    if any(abs(c - p) > 1e-5 for c, p in zip(curr_dists, prev_d)):
                        self.onset_times[i] = self.t
                        self.last_change_time[i] = self.t
                self.prev_distances[i] = curr_dists

        # Store last-present features
        for i in range(N):
            if presence[i]:
                self.last_present_features[i] = X[i][:]

        for i in range(N):
            if i in self.prev_features:
                if any(abs(a - b) > 1e-9 for a, b in zip(X[i], self.prev_features[i])):
                    self.last_change_time[i] = self.t
            else:
                self.last_change_time[i] = self.t
            self.prev_features[i] = X[i][:]

        return StatePattern(
            t=self.t,
            X=X,
            R=R,
            A=A,
            entity_indices=sorted(entity_indices),
            raw_indices=list(range(N)),
            modality="symbolic",
        )

    # -------------------------------------------------------------------------
    # 3D state construction
    # -------------------------------------------------------------------------

    def _construct_3d_state(self, O_t: Observation3D) -> StatePattern:
        """Constructs StatePattern from an Observation3D. Anchor IDs are
        persistent and 1:1 with entities; the adapter resolved identities."""
        N = O_t.num_atoms()
        agent_idx = O_t.get_agent_index()

        # Adapter contract check: anchor array size must be stable.
        if self.prev_state is not None and self.prev_state.modality == "3d":
            prev_N = self.prev_state.num_atoms()
            if N != prev_N:
                warnings.warn(
                    f"Observation3D anchor count changed: {prev_N} → {N}. "
                    f"Index-keyed history may be inconsistent. "
                    f"Adapter must maintain fixed-size anchor arrays.",
                    stacklevel=2,
                )

        total_atoms = max(
            N,
            (max(O_t.get_flag_indices()) + 1) if O_t.get_flag_indices() else 0,
            (max(O_t.get_effect_indices()) + 1) if O_t.get_effect_indices() else 0,
        )

        R = [[float("inf")] * total_atoms for _ in range(total_atoms)]
        A = [[0] * total_atoms for _ in range(total_atoms)]

        for i in range(total_atoms):
            R[i][i] = 0.0

        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                d = O_t.get_distance(i, j)
                if d < float("inf"):
                    R[i][j] = d
                A[i][j] = O_t.get_adjacency(i, j)

        X = [[0.0] * 16 for _ in range(total_atoms)]

        for i in range(N):
            X[i][0] = 1.0 if O_t.is_present(i) else 0.0

        Y_I, Y_C_space, Y_C_time, Y_T = self._compute_transduction_responses(
            X, R, A, modality="3d", O_t=O_t
        )
        for i in range(N):
            X[i][0] = Y_I[i]
            X[i][1] = Y_C_space[i]
            X[i][2] = Y_C_time[i]
            X[i][3] = Y_T[i]

        entity_indices = O_t.get_entity_indices()

        flag_indices = O_t.get_flag_indices()
        for i in flag_indices:
            X[i][5] = 1.0
            X[i][0] = O_t.get_flag_value(i)

        effect_indices = O_t.get_effect_indices()
        for i in effect_indices:
            X[i][6] = 1.0
            X[i][0] = 1.0 if O_t.is_present(i) else 0.0

        if 0 <= agent_idx < total_atoms:
            X[agent_idx][7] = 1.0
            if O_t.is_present(agent_idx):
                X[agent_idx][0] = 1.0

        for i in range(N):
            if not O_t.is_present(i):
                continue
            deg = sum(A[i][j] for j in range(N) if j != i)
            X[i][8] = min(1.0, deg / (DEG_MAX_3D + EPSILON))

        for i in range(N):
            if not O_t.is_present(i):
                continue
            clutter_count = sum(
                1
                for j in range(N)
                if j != i and R[i][j] < R_CLUTTER and R[i][j] < float("inf")
            )
            X[i][9] = min(1.0, clutter_count / (DEG_MAX_3D + EPSILON))

        for i in range(N):
            if not O_t.is_present(i):
                continue
            if i not in self.atom_birth:
                adapter_birth = O_t.get_birth_time(i)
                self.atom_birth[i] = (
                    adapter_birth if adapter_birth <= self.t else self.t
                )
            age = max(0, self.t - self.atom_birth[i])
            X[i][10] = min(1.0, age / TAU_LIFE)

        for i in range(N):
            if not O_t.is_present(i):
                continue
            if i not in self.last_change_time:
                self.last_change_time[i] = self.t
            X[i][11] = clamp(
                1.0 / (1.0 + (self.t - self.last_change_time[i]) / TAU_NORM),
                0.0,
                1.0,
            )

        self._write_3d_specific_channels(X, O_t)


        for i in entity_indices:
            if O_t.is_present(i):
                curr_dists = [R[i][j] for j in range(total_atoms)]
                prev_d = self.prev_distances.get(i)
                if prev_d is not None:
                    if any(abs(c - p) > 1e-5 for c, p in zip(curr_dists, prev_d)):
                        self.onset_times[i] = self.t
                        self.last_change_time[i] = self.t
                self.prev_distances[i] = curr_dists

        for i in range(total_atoms):
            if i < N and O_t.is_present(i):
                self.last_present_features[i] = X[i][:]

        for i in range(total_atoms):
            if i in self.prev_features:
                if any(abs(a - b) > 1e-9 for a, b in zip(X[i], self.prev_features[i])):
                    self.last_change_time[i] = self.t
            else:
                self.last_change_time[i] = self.t
            self.prev_features[i] = X[i][:]

        current_entities = set(entity_indices)
        self.new_anchor_spawned = bool(current_entities - self._prev_entity_indices)
        self._prev_entity_indices = current_entities

        return StatePattern(
            t=self.t,
            X=X,
            R=R,
            A=A,
            entity_indices=sorted(entity_indices),
            raw_indices=list(range(total_atoms)),
            modality="3d",
            Y_C_time=Y_C_time,
        )

    def _write_3d_specific_channels(
        self, X: List[List[float]], O_t: Observation3D
    ) -> None:
        for u in O_t.get_entity_indices():
            if not O_t.is_present(u):
                continue
            X[u][12] = min(1.0, O_t.get_radius(u) / (R_MAX_OBJ + EPSILON))
            X[u][13] = min(1.0, O_t.get_speed(u) / (V_MAX + EPSILON))
            X[u][14] = O_t.get_elevation(u)
            X[u][15] = clamp(O_t.get_planarity(u), 0.0, 1.0)

    # -------------------------------------------------------------------------
    # Legacy 2D continuous state construction (deprecated)
    # -------------------------------------------------------------------------

    def _construct_continuous_state_legacy(
        self, O_t: ContinuousObservation
    ) -> StatePattern:
        """DEPRECATED: Legacy 2D continuous path. No v31 environment uses
        this. Retained for backward compatibility during migration."""
        N = O_t.num_atoms()
        X = [[0.0] * 16 for _ in range(N)]
        for i in range(N):
            for col in range(4):
                X[i][col] = O_t.get_feature(i, col)

        R = [[float("inf")] * N for _ in range(N)]
        A = [[0] * N for _ in range(N)]
        for i in range(N):
            for j in range(N):
                d = O_t.get_distance(i, j)
                if d < float("inf"):
                    R[i][j] = d
                    if i != j and O_t.get_adjacency(i, j):
                        A[i][j] = 1

        flag_indices = O_t.get_flag_indices()
        effect_indices = O_t.get_effect_indices()
        for i in flag_indices:
            X[i][5] = 1.0
            X[i][0] = O_t.get_flag_value(i)
        for i in effect_indices:
            X[i][6] = 1.0
            X[i][0] = 1.0

        Y_I, Y_C_space, Y_C_time, Y_T = self._compute_transduction_responses(
            X, R, A, modality="continuous"
        )
        for i in range(N):
            X[i][0] = Y_I[i]
            X[i][1] = Y_C_space[i]
            X[i][2] = Y_C_time[i]
            X[i][3] = Y_T[i]

        # Legacy entity-synthesis controller was removed in v31: the adapter
        # owns entity identity. The legacy path degrades to "each atom is an
        # entity" without continuity of identity across frames.
        entity_indices = list(range(N))

        for i in range(N):
            if i not in self.atom_birth:
                self.atom_birth[i] = self.t
            X[i][10] = min((self.t - self.atom_birth[i]) / TAU_LIFE, 1.0)
            if i not in self.last_change_time:
                self.last_change_time[i] = self.t
            X[i][11] = clamp(
                1.0 / (1.0 + (self.t - self.last_change_time[i]) / TAU_NORM),
                0.0,
                1.0,
            )
            self.last_present_features[i] = X[i][:]

        for i in range(N):
            if i in self.prev_features:
                if any(abs(a - b) > 1e-9 for a, b in zip(X[i], self.prev_features[i])):
                    self.last_change_time[i] = self.t
            else:
                self.last_change_time[i] = self.t
            self.prev_features[i] = X[i][:]

        return StatePattern(
            t=self.t,
            X=X,
            R=R,
            A=A,
            entity_indices=entity_indices,
            raw_indices=list(range(N)),
            modality="continuous",
            Y_C_time=Y_C_time,
        )

    # -------------------------------------------------------------------------
    # Transduction
    # -------------------------------------------------------------------------

    def _compute_transduction_responses(
        self,
        X: List[List[float]],
        R: List[List[float]],
        A: List[List[int]],
        modality: str = "symbolic",
        O_t: Optional[Observation] = None,
        features: Optional[List[List[float]]] = None,
    ) -> Tuple[List[float], List[float], List[float], List[float]]:
        N = len(X)
        X_I = self._distance_aware_smoothing(X, A, R, ell=float("inf"))
        Y_I = [X_I[i][0] for i in range(N)]

        if modality == "3d":
            ell_C = R_LOCAL_3D
            R_C = R_LOCAL_3D
        else:
            ell_C = self.R_med if self.R_med else 1.0
            R_C = self.R_med if self.R_med else 1.0

        X_C = self._distance_aware_smoothing(X, A, R, ell=ell_C)
        if features is not None:
            # The caller's MEASURED feature vectors REPLACE the state's
            # own first four channels, which are degenerate in the image
            # path (presence only) -- but they are SMOOTHED first, so the
            # distance-aware smoothing still applies to the quantities the
            # contrast reads. (Replacing the smoothed rows outright made
            # the smoothing a no-op on this path.)
            X_F = self._distance_aware_smoothing(
                [list(f) + [0.0] * 12 for f in features],
                A, R, ell=ell_C)
            X_C = [
                [float(X_F[i][0]), float(X_F[i][1]),
                 float(X_F[i][2]), float(X_F[i][3])]
                + list(X_C[i][4:])
                for i in range(N)
            ]

        Y_C_space = []
        for i in range(N):
            num, den = 0.0, 0.0
            xi = X_C[i][:4]
            for j in range(N):
                if i != j and A[i][j] == 1 and R[i][j] <= R_C:
                    w = math.exp(-R[i][j] / (ell_C + EPSILON))
                    xj = X_C[j][:4]
                    diff = math.sqrt(sum((a - b) ** 2 for a, b in zip(xi, xj)))
                    num += w * diff
                    den += w
            raw_c_space = num / den if den > 0 else 0.0
            Y_C_space.append(clamp(raw_c_space, 0.0, 1.0))

        Y_C_time = [0.0] * N
        if self.prev_X_C is not None:
            for i in range(N):
                if i < len(self.prev_X_C):
                    prev_xi = self.prev_X_C[i][:4]
                    curr_xi = X_C[i][:4]
                    diff = math.sqrt(
                        sum((a - b) ** 2 for a, b in zip(curr_xi, prev_xi))
                    )
                    Y_C_time[i] = clamp(diff / (1.0 + EPSILON), 0.0, 1.0)
        self.prev_X_C = copy.deepcopy(X_C)

        if modality == "3d":
            Y_T = [0.0] * N
            for i in range(N):
                present = X[i][0] > 0.5
                if O_t is not None:
                    present = O_t.is_present(i)

                if not self.last_presence.get(i, False):
                    self.presence_ema_3d[i] = 0.0

                prev_ema = self.presence_ema_3d.get(i, 0.0)
                self.presence_ema_3d[i] = BETA_T * prev_ema + (1.0 - BETA_T) * (
                    1.0 if present else 0.0
                )
                Y_T[i] = clamp(self.presence_ema_3d[i], 0.0, 1.0)
            self.cont_history.append(copy.deepcopy(X_I))
        else:
            self.cont_history.append(copy.deepcopy(X_I))
            Y_T = [0.0] * N
            if (
                self.t >= DELTA_COHERENCE + W_T - 1
                and len(self.cont_history) >= DELTA_COHERENCE + W_T
            ):
                for i in range(N):
                    num, den = 0.0, 0.0
                    for tau in range(W_T):
                        idx_curr = -(tau + 1)
                        idx_prev = -(tau + 1 + DELTA_COHERENCE)
                        curr_vec = self.cont_history[idx_curr][i][:4]
                        prev_vec = self.cont_history[idx_prev][i][:4]
                        norm_curr = math.sqrt(sum(x * x for x in curr_vec)) + EPSILON
                        norm_prev = math.sqrt(sum(x * x for x in prev_vec)) + EPSILON
                        dot = sum(a * b for a, b in zip(curr_vec, prev_vec))
                        cos_sim = dot / (norm_curr * norm_prev)
                        weight = BETA_T**tau
                        num += weight * cos_sim
                        den += weight
                    Y_T[i] = clamp(num / (den + EPSILON), 0.0, 1.0)

        return Y_I, Y_C_space, Y_C_time, Y_T

    def _distance_aware_smoothing(self, X, A, R, ell):
        cache_key = (id(X), id(A), id(R), ell)
        cached = self._smoothing_cache.get(cache_key)
        if cached is not None:
            return cached
        N = len(X)
        d = 16
        W = [[0.0] * N for _ in range(N)]
        for i in range(N):
            for j in range(N):
                if A[i][j] == 1:
                    W[i][j] = 1.0 if ell == float("inf") else math.exp(-R[i][j] / ell)
        D_diag = [sum(W[i]) for i in range(N)]
        X_smooth = [[0.0] * d for _ in range(N)]
        for i in range(N):
            for col in range(d):
                val = X[i][col]
                for j in range(N):
                    if i != j:
                        val += W[i][j] * X[j][col]
                X_smooth[i][col] = clamp(val / (D_diag[i] + 1.0), 0.0, 1.0)
        self._smoothing_cache[cache_key] = X_smooth
        return X_smooth

    # -------------------------------------------------------------------------
    # Pure state hash (no history mutation)
    # -------------------------------------------------------------------------

    def _compute_state_hash_pure(
        self,
        O_t: Observation,
        dir_tuples: Optional[List[Tuple[int, int, int]]] = None,
    ) -> int:
        """Compute the state hash for an observation WITHOUT mutating history."""
        self._smoothing_cache.clear()
        if self.R_med is None:
            raise RuntimeError(
                "SensoryFloor.step() must be called before _compute_state_hash_pure()"
            )

        saved_atom_birth = dict(self.atom_birth)
        saved_last_change_time = dict(self.last_change_time)
        saved_prev_distances = {k: v[:] for k, v in self.prev_distances.items()}
        saved_prev_features = {k: v[:] for k, v in self.prev_features.items()}
        saved_last_present_features = {
            k: v[:] for k, v in self.last_present_features.items()
        }
        saved_onset_times = dict(self.onset_times)
        saved_prev_flag_values = dict(self.prev_flag_values)
        saved_presence_history = {
            k: deque(v, maxlen=v.maxlen) for k, v in self.presence_history.items()
        }
        saved_last_presence = dict(self.last_presence)
        saved_presence_ema_3d = dict(self.presence_ema_3d)
        saved_prev_entity_indices = set(self._prev_entity_indices)

        saved_prev_X_C = copy.deepcopy(self.prev_X_C)
        saved_cont_history = copy.deepcopy(self.cont_history)
        saved_new_anchor = self.new_anchor_spawned

        S_t = self._construct_state(O_t)
        h = self._compute_state_hash(S_t, dir_tuples=dir_tuples)

        self.atom_birth = saved_atom_birth
        self.last_change_time = saved_last_change_time
        self.prev_distances = saved_prev_distances
        self.prev_features = saved_prev_features
        self.last_present_features = saved_last_present_features
        self.onset_times = saved_onset_times
        self.prev_flag_values = saved_prev_flag_values
        self.presence_history = saved_presence_history
        self.last_presence = saved_last_presence
        self.presence_ema_3d = saved_presence_ema_3d
        self._prev_entity_indices = saved_prev_entity_indices

        self.prev_X_C = saved_prev_X_C
        self.cont_history = saved_cont_history
        self.new_anchor_spawned = saved_new_anchor

        return h

    # -------------------------------------------------------------------------
    # State hash
    # -------------------------------------------------------------------------

    def _compute_state_hash(
        self, S_t: StatePattern, dir_tuples: Optional[List[Tuple[int, int, int]]] = None
    ) -> int:
        parts = []

        # [STATE] The measured spatial-contrast channel, per atom. Without
        # it the hash was a function of the adjacency and the flags
        # alone, so two DIFFERENT images hashed IDENTICALLY.
        for i in range(S_t.num_atoms()):
            v = float(S_t.X[i][1])
            if v > 0.0:
                parts.append(
                    serialize_tuple(
                        [
                            serialize_int(4),
                            serialize_int(i),
                            serialize_float(v),
                        ]
                    )
                )

        # Flags (tag 0)
        for i in range(S_t.num_atoms()):
            if S_t.X[i][5] == 1.0:
                parts.append(
                    serialize_tuple(
                        [
                            serialize_int(0),
                            serialize_int(i),
                            serialize_float(S_t.X[i][0]),
                        ]
                    )
                )

        # Effects (tag 2)
        for i in range(S_t.num_atoms()):
            if S_t.X[i][6] == 1.0:
                parts.append(
                    serialize_tuple(
                        [
                            serialize_int(2),
                            serialize_int(i),
                            serialize_float(S_t.X[i][0]),
                        ]
                    )
                )

        # Entity-level adjacency (tag 1)
        for u in S_t.entity_indices:
            adjacent = False
            for v in S_t.entity_indices:
                if v == u:
                    continue
                if S_t.A[u][v] == 1:
                    adjacent = True
                    break
            if adjacent:
                parts.append(serialize_tuple([serialize_int(1), serialize_int(u)]))

        # Directional relational invariants (tag 3)
        if dir_tuples:
            for r, d, s in dir_tuples:
                parts.append(
                    serialize_tuple(
                        [
                            serialize_int(3),
                            serialize_int(r),
                            serialize_int(d),
                            serialize_int(s),
                        ]
                    )
                )

        return hash_64(serialize_set(parts))

    # -------------------------------------------------------------------------
    # History updates
    # -------------------------------------------------------------------------

    def _update_history(self, O_t: Observation):
        modality = O_t.modality()
        if modality == "3d":
            self._update_history_3d(O_t)
        elif modality == "symbolic":
            self._update_history_symbolic(O_t)
        elif modality == "2d":
            self._update_history_2d(O_t)
        elif modality == "continuous":
            self._update_history_symbolic(O_t)
        else:
            raise ValueError(f"Unknown modality: {modality}")

    def _atom_strength_2d(self, O_t: Observation, i: int) -> float:
        """The atom's normalised structural energy, in [0, 1].

        Read from the observation's own Gabor summary when it has one
        (the image path), and 1.0 otherwise (an observation that
        carries no Gabor data has no structure to measure).
        """
        getter = getattr(O_t, "get_gabor_responses", None)
        if not callable(getter):
            return 1.0
        try:
            g = getter(i)
        except Exception:
            return 1.0
        if not g:
            return 1.0
        c = g.get("complex_pooled")
        if c is None:
            return 1.0
        try:
            arr = np.asarray(c, dtype=np.float64)
            v = float(arr.mean()) if arr.size else 0.0
        except Exception:
            return 1.0
        return min(1.0, max(0.0, v))

    def _update_history_2d(self, O_t: Observation) -> None:
        """[Batch 2D] History update for 2D still-image observations.
        Regions are always present within a still image (no presence
        transitions, no flags/effects), so presence_history is populated
        at 1.0 for every atom from the first step. atom_birth and
        last_change_time are initialized so downstream state construction
        reads defined values."""
        N = O_t.num_atoms()
        for i in range(N):
            if i not in self.presence_history:
                self.presence_history[i] = deque(maxlen=W_T)
                for _ in range(min(W_T, max(0, self.t))):
                    self.presence_history[i].append(0.0)
            # The history records an IMAGE-DERIVED presence strength
            # (the atom's normalised structural energy), not a constant:
            # a still image's atoms are always PRESENT, but how much
            # structure each one carries is a measurement.
            self.presence_history[i].append(self._atom_strength_2d(O_t, i))
            self.last_presence[i] = True
            if i not in self.atom_birth:
                self.atom_birth[i] = self.t
            if i not in self.last_change_time:
                self.last_change_time[i] = self.t

    def _update_history_symbolic(self, O_t: Observation):
        N = O_t.num_atoms()
        new_entity_appeared = False

        for i in range(N):
            present = O_t.is_present(i) if O_t.modality() == "symbolic" else True

            if O_t.modality() == "symbolic":
                if i not in self.presence_history:
                    self.presence_history[i] = deque(maxlen=W_T)
                    for _ in range(min(W_T, self.t)):
                        self.presence_history[i].append(0.0)
                self.presence_history[i].append(1.0 if present else 0.0)

            prev_present = self.last_presence.get(i, False)
            if present and not prev_present:
                self.onset_times[i] = self.t
                if O_t.modality() == "symbolic" and i in O_t.get_entity_indices():
                    new_entity_appeared = True
            self.last_presence[i] = present

            if O_t.modality() == "symbolic" and i in O_t.get_flag_indices():
                current_val = O_t.get_flag_value(i)
                prev_val = self.prev_flag_values.get(i, None)
                if prev_val is not None and current_val != prev_val:
                    self.onset_times[i] = self.t
                    self.last_change_time[i] = self.t
                self.prev_flag_values[i] = current_val

            if i not in self.atom_birth:
                self.atom_birth[i] = self.t
            if i not in self.last_change_time:
                self.last_change_time[i] = self.t

        if O_t.modality() == "symbolic":
            self.new_anchor_spawned = new_entity_appeared

    def _update_history_3d(self, O_t: Observation3D):
        """3D history update. All tracking keyed by anchor ID (= atom index)."""
        N = O_t.num_atoms()

        for i in range(N):
            present = O_t.is_present(i)

            if i not in self.presence_history:
                self.presence_history[i] = deque(maxlen=W_T)
                for _ in range(min(W_T, max(0, self.t))):
                    self.presence_history[i].append(0.0)
            self.presence_history[i].append(1.0 if present else 0.0)

            prev_present = self.last_presence.get(i, False)
            if present and not prev_present:
                self.onset_times[i] = self.t
            self.last_presence[i] = present

            if i in O_t.get_flag_indices():
                current_val = O_t.get_flag_value(i)
                prev_val = self.prev_flag_values.get(i, None)
                if prev_val is not None and current_val != prev_val:
                    self.onset_times[i] = self.t
                    self.last_change_time[i] = self.t
                self.prev_flag_values[i] = current_val

            # Birth tracking: 3D uses the adapter's birth_time (see
            # _construct_3d_state). Only initialize last_change_time here.
            if i not in self.last_change_time:
                self.last_change_time[i] = self.t

def diagnostic_report(sf: SensoryFloor, D_raw: Set[Detection]) -> Dict[str, Any]:
    """Per-step detection summary. Lens-specific homeostatic state lives in
    the LensExecutor; use LensExecutor.get_diagnostics() for that."""
    counts: Dict[str, int] = {}
    for det in D_raw:
        counts[det.k] = counts.get(det.k, 0) + 1

    entity_count = len(sf.prev_state.entity_indices) if sf.prev_state else 0

    lens_diag: Dict[str, Any] = {}
    if sf._lens_executor is not None and hasattr(sf._lens_executor, "get_diagnostics"):
        try:
            lens_diag = sf._lens_executor.get_diagnostics()
        except Exception:
            lens_diag = {}

    return {
        "total": len(D_raw),
        "per_lens": counts,
        "entity_count": entity_count,
        "state_modality": sf.prev_state.modality if sf.prev_state else None,
        "lens_diagnostics": lens_diag,
    }

