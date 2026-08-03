"""Train a single-payload RC hold model for the multi-link robot.

Usage:
    python train_single_payload.py --payload 0.20 --output model_payload0.20.npz
"""
from __future__ import annotations

import argparse
from pathlib import Path
import logging

import numpy as np

import train_multilink_rc_cpg as trainer
from multilink_mass_robot_env import MassProfile, MultiLinkMassRobotEnv
from multilink_rc_cpg_backend import EchoStateNetwork, MassCondition

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def train_single(payload_mass: float, links: int = 4, masses: str = "0.20,0.25,0.30,0.35", episodes: int = 20, steps: int = 300, reservoir: int = 600, output: Path | str = "multilink_rc_hold_model_payload.npz"):
    base_link_masses = tuple(float(x) for x in masses.split(","))
    if len(base_link_masses) != links:
        raise ValueError(f"Expected {links} link masses in --masses")

    profile = MassProfile(link_masses=base_link_masses, payload_mass=float(payload_mass))
    env = MultiLinkMassRobotEnv(n_links=links, mass_profile=profile, render_mode=None)
    log.info("Collecting episodes for payload=%.3f", float(payload_mass))
    episodes_X, episodes_Y = trainer.collect_episodes(env, MassCondition(profile.link_masses, profile.payload_mass), n_episodes=episodes, episode_length=steps)
    env.close()

    if not episodes_X:
        raise RuntimeError("No episodes collected")

    feature_dim = episodes_X[0].shape[1]
    action_dim = episodes_Y[0].shape[1]
    esn = EchoStateNetwork(n_inputs=feature_dim, n_reservoir=reservoir, seed=42)
    esn.fit(episodes_X, episodes_Y, ridge=1e-4, washout=trainer.WASHOUT)

    model_path = Path(output)
    with model_path.open("wb") as handle:
        np.savez_compressed(
            handle,
            n_links=np.int32(links),
            link_masses=np.asarray(base_link_masses, dtype=np.float32),
            payload_mass=np.float32(profile.payload_mass),
            reservoir_Win=esn.Win,
            reservoir_W=esn.W,
            reservoir_W_out=esn.W_out,
            reservoir_leak=np.float32(esn.leak),
            reservoir_n=np.int32(esn.n_reservoir),
            feature_dim=np.int32(feature_dim),
            action_dim=np.int32(action_dim),
            profile_records=np.asarray([{"payload_mass": profile.payload_mass}], dtype=object),
        )

    log.info("Saved single-payload model to %s", str(model_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a single-payload RC hold model")
    parser.add_argument("--payload", type=float, required=True)
    parser.add_argument("--links", type=int, default=4)
    parser.add_argument("--masses", type=str, default="0.20,0.25,0.30,0.35")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--reservoir", type=int, default=600)
    parser.add_argument("--output", type=Path, default=Path("multilink_rc_hold_model_payload.npz"))
    args = parser.parse_args()

    train_single(args.payload, links=args.links, masses=args.masses, episodes=args.episodes, steps=args.steps, reservoir=args.reservoir, output=args.output)
