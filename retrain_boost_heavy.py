"""Retrain a single heavy-payload model with a stronger hold teacher.
Usage:
  python retrain_boost_heavy.py --payload 0.40 --output <out.npz>
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

import train_multilink_rc_cpg as trainer
from multilink_mass_robot_env import MassProfile, MultiLinkMassRobotEnv
from multilink_rc_cpg_backend import EchoStateNetwork, MassCondition

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Retrain heavy payload with boosted teacher")
    parser.add_argument("--payload", type=float, default=0.40)
    parser.add_argument("--links", type=int, default=4)
    parser.add_argument("--masses", type=str, default="0.20,0.25,0.30,0.35")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--reservoir", type=int, default=600)
    parser.add_argument("--output", type=Path, default=Path("multilink_rc_hold_model_payload0.40_boost.npz"))
    parser.add_argument("--extra-lift", type=float, default=0.30, help="Extra constant torque per joint added to teacher for the target payload")
    args = parser.parse_args()

    payload_mass = float(args.payload)
    base_link_masses = tuple(float(x) for x in args.masses.split(","))
    profile = MassProfile(link_masses=base_link_masses, payload_mass=payload_mass)

    # Wrap trainer.hold_teacher to add extra lift for the target payload
    orig_hold = trainer.hold_teacher

    def boosted_hold(obs: np.ndarray, n_links: int, mass_condition: MassCondition, segment_length: float = 0.18) -> np.ndarray:
        out = orig_hold(obs, n_links, mass_condition, segment_length)
        if abs(float(mass_condition.payload_mass) - payload_mass) < 1e-6:
            extra = np.full(n_links, float(args.extra_lift), dtype=np.float32)
            return (out + extra).astype(np.float32)
        return out

    trainer.hold_teacher = boosted_hold

    env = MultiLinkMassRobotEnv(n_links=args.links, mass_profile=profile, render_mode=None)
    log.info("Collecting episodes with boosted teacher for payload=%.3f", payload_mass)
    episodes_X, episodes_Y = trainer.collect_episodes(env, MassCondition(profile.link_masses, profile.payload_mass), n_episodes=args.episodes, episode_length=args.steps)
    env.close()

    feature_dim = episodes_X[0].shape[1]
    action_dim = episodes_Y[0].shape[1]
    esn = EchoStateNetwork(n_inputs=feature_dim, n_reservoir=args.reservoir, seed=42)
    esn.fit(episodes_X, episodes_Y, ridge=1e-4, washout=trainer.WASHOUT)

    model_path = args.output
    with model_path.open("wb") as handle:
        np.savez_compressed(
            handle,
            n_links=np.int32(args.links),
            link_masses=np.asarray(base_link_masses, dtype=np.float32),
            payload_mass=np.float32(profile.payload_mass),
            reservoir_Win=esn.Win,
            reservoir_W=esn.W,
            reservoir_W_out=esn.W_out,
            reservoir_leak=np.float32(esn.leak),
            reservoir_n=np.int32(esn.n_reservoir),
            feature_dim=np.int32(feature_dim),
            action_dim=np.int32(action_dim),
            profile_records=np.asarray([{"payload_mass": profile.payload_mass, "boosted": True}], dtype=object),
        )

    log.info("Saved boosted heavy-payload model to %s", model_path)


if __name__ == "__main__":
    main()
