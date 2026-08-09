"""Shared RC/CPG backend for the multi-link mass robot."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class MassCondition:
    link_masses: tuple[float, ...]
    payload_mass: float

    @property
    def vector(self) -> np.ndarray:
        return np.asarray((*self.link_masses, self.payload_mass), dtype=np.float32)


class CPGController:
    def __init__(self, n_actions: int, omega=2.0, amplitudes=None, phases=None, offsets=None):
        self.n = int(n_actions)
        self.omega = float(omega)
        self.amplitudes = np.ones(self.n, dtype=np.float64) if amplitudes is None else np.asarray(amplitudes, dtype=np.float64).reshape(self.n)
        self.phases = np.zeros(self.n, dtype=np.float64) if phases is None else np.asarray(phases, dtype=np.float64).reshape(self.n)
        self.offsets = np.zeros(self.n, dtype=np.float64) if offsets is None else np.asarray(offsets, dtype=np.float64).reshape(self.n)

    @classmethod
    def from_vector(cls, vec: Sequence[float], n_actions: int):
        vec = np.asarray(vec, dtype=np.float64)
        return cls(
            n_actions,
            omega=float(vec[0]),
            amplitudes=vec[1:1 + n_actions],
            phases=vec[1 + n_actions:1 + 2 * n_actions],
            offsets=vec[1 + 2 * n_actions:1 + 3 * n_actions],
        )

    def vector(self) -> np.ndarray:
        return np.concatenate(([self.omega], self.amplitudes, self.phases, self.offsets)).astype(np.float64)

    def step(self, t: float) -> np.ndarray:
        theta = self.omega * float(t) + self.phases
        return self.offsets + self.amplitudes * np.sin(theta)


class EchoStateNetwork:
    def __init__(self, n_inputs, n_reservoir, spectral_radius=0.95, input_scaling=1.0, leak_rate=0.3, sparsity=0.1, seed=42):
        rng = np.random.RandomState(seed)
        self.n_reservoir = int(n_reservoir)
        self.leak = float(leak_rate)
        self.Win = rng.uniform(-input_scaling, input_scaling, (self.n_reservoir, int(n_inputs))).astype(np.float64)

        W = rng.uniform(-0.5, 0.5, (self.n_reservoir, self.n_reservoir))
        mask = rng.random((self.n_reservoir, self.n_reservoir)) < sparsity
        W *= mask
        eigvals = np.linalg.eigvals(W)
        max_abs = np.max(np.abs(eigvals)) if eigvals.size else 1.0
        if max_abs > 1e-12:
            W *= spectral_radius / max_abs
        self.W = W.astype(np.float64)
        self.state = np.zeros(self.n_reservoir, dtype=np.float64)
        self.W_out = None

    def reset(self):
        self.state = np.zeros(self.n_reservoir, dtype=np.float64)

    def update(self, x):
        x = np.asarray(x, dtype=np.float64)
        pre = self.W @ self.state + self.Win @ x
        self.state = (1.0 - self.leak) * self.state + self.leak * np.tanh(pre)
        return self.state.copy()

    def collect_states(self, episodes_X, washout=50):
        all_states = []
        all_inputs = []
        all_indices = []
        for ep_idx, ep_X in enumerate(episodes_X):
            self.reset()
            for t, x in enumerate(ep_X):
                s = self.update(x)
                if t >= washout:
                    all_states.append(s)
                    all_inputs.append(np.asarray(x, dtype=np.float64))
                    all_indices.append((ep_idx, t))
        return np.asarray(all_states, dtype=np.float64), np.asarray(all_inputs, dtype=np.float64), all_indices

    def _augment(self, H, X_in):
        bias = np.ones((len(H), 1), dtype=np.float64)
        return np.concatenate([bias, X_in, H], axis=1)

    def fit(self, episodes_X, episodes_Y, ridge=1e-4, washout=50):
        H, X_in, indices = self.collect_states(episodes_X, washout=washout)
        H_aug = self._augment(H, X_in)
        Y_mat = np.asarray([episodes_Y[ep_idx][t] for ep_idx, t in indices], dtype=np.float64)
        A = H_aug.T @ H_aug + ridge * np.eye(H_aug.shape[1])
        b = H_aug.T @ Y_mat
        self.W_out = np.linalg.solve(A, b)
        return self

    def predict(self, x):
        x = np.asarray(x, dtype=np.float64)
        s = self.update(x)
        v = np.concatenate([[1.0], x, s])
        return v @ self.W_out
