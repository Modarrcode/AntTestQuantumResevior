"""Behavior cloning for the multi-link mass robot.

This trainer fits a ridge-regression policy from the generated rollout dataset
and saves a compact model that can be used to drive the robot environment.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


@dataclass
class LinearPolicy:
    weights: np.ndarray
    bias: np.ndarray

    def predict(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32).reshape(-1)
        return obs @ self.weights + self.bias


def _flatten_episodes(episodes):
    arrays = [np.asarray(ep, dtype=np.float32) for ep in episodes if len(ep) > 0]
    if not arrays:
        return np.empty((0, 0), dtype=np.float32)
    return np.concatenate(arrays, axis=0)


def load_dataset(dataset_path: Path):
    data = np.load(dataset_path, allow_pickle=True)
    observations = _flatten_episodes(data["observations"])
    actions = _flatten_episodes(data["actions"])
    if observations.shape[0] != actions.shape[0]:
        raise ValueError("Observation and action sample counts do not match")
    return observations, actions, data


def fit_linear_policy(observations: np.ndarray, actions: np.ndarray, ridge: float = 1e-3) -> LinearPolicy:
    observations = np.asarray(observations, dtype=np.float64)
    actions = np.asarray(actions, dtype=np.float64)
    if observations.ndim != 2 or actions.ndim != 2:
        raise ValueError("Expected 2D observation and action arrays")

    ones = np.ones((observations.shape[0], 1), dtype=np.float64)
    design = np.concatenate([observations, ones], axis=1)
    gram = design.T @ design + ridge * np.eye(design.shape[1], dtype=np.float64)
    rhs = design.T @ actions
    solution = np.linalg.solve(gram, rhs)
    weights = solution[:-1]
    bias = solution[-1]
    return LinearPolicy(weights=weights.astype(np.float32), bias=bias.astype(np.float32))


def evaluate_policy(policy: LinearPolicy, observations: np.ndarray, actions: np.ndarray) -> float:
    predictions = np.asarray([policy.predict(obs) for obs in observations], dtype=np.float32)
    mse = float(np.mean((predictions - actions) ** 2))
    return mse


def main():
    parser = argparse.ArgumentParser(description="Train a behavior-cloning policy for the multi-link robot")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path, default=Path("multilink_bc_policy.pkl"))
    parser.add_argument("--ridge", type=float, default=1e-3)
    args = parser.parse_args()

    observations, actions, data = load_dataset(args.dataset)
    policy = fit_linear_policy(observations, actions, ridge=float(args.ridge))
    mse = evaluate_policy(policy, observations, actions)

    payload = {
        "policy_weights": policy.weights,
        "policy_bias": policy.bias,
        "n_links": int(data["n_links"]),
        "link_masses": np.asarray(data["mass_profiles"][0][: int(data["n_links"])], dtype=np.float32),
        "payload_mass": float(data["payload_mass"]),
        "observation_dim": int(observations.shape[1]),
        "action_dim": int(actions.shape[1]),
        "mse": mse,
    }

    with args.output.open("wb") as handle:
        np.savez_compressed(handle, **payload)

    log.info("Saved policy to %s", args.output)
    log.info("Training MSE: %.6f", mse)


if __name__ == "__main__":
    main()