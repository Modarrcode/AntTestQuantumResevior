"""Ant-style RC hold training for the upright multi-link mass robot."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

from multilink_mass_robot_env import MassProfile, MultiLinkMassRobotEnv
from multilink_rc_cpg_backend import EchoStateNetwork, MassCondition


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

WASHOUT = 40
OUTPUT_DIR = Path("multilink_rc_hold_model")
TRAIN_PAYLOADS = (0.20, 0.30, 0.40)


def hold_teacher(
    obs: np.ndarray,
    n_links: int,
    mass_condition: MassCondition,
    segment_length: float = 0.18,
) -> np.ndarray:
    obs = np.asarray(obs, dtype=np.float32)
    joint_positions = obs[:n_links]
    joint_velocities = obs[n_links:2 * n_links]
    tip_pos = obs[2 * n_links:2 * n_links + 3]
    tip_vel = obs[2 * n_links + 3:2 * n_links + 6]

    target_height = segment_length * float(n_links)
    height_error = target_height - float(tip_pos[2])
    total_mass = float(np.sum(mass_condition.vector))
    payload_bias = 0.10 * float(mass_condition.payload_mass)
    joint_stiffness = -(2.20 + 0.03 * total_mass) * joint_positions
    joint_damping = -(0.90 + 0.02 * total_mass) * joint_velocities
    lift = (0.50 + 0.02 * total_mass) * height_error - (0.12 + 0.01 * total_mass) * float(tip_vel[2])
    gravity_comp = np.full(n_links, 0.04 * total_mass / max(n_links, 1), dtype=np.float32)
    shape = np.linspace(0.18 + payload_bias, 0.08 + payload_bias * 0.5, n_links, dtype=np.float32)
    return (joint_stiffness + joint_damping + lift + gravity_comp + shape).astype(np.float32)


def build_features(obs: np.ndarray, prev_obs: np.ndarray, mass_condition: MassCondition) -> np.ndarray:
    obs = np.asarray(obs, dtype=np.float32)
    prev_obs = np.asarray(prev_obs, dtype=np.float32)
    delta = obs - prev_obs
    return np.concatenate([obs, delta, mass_condition.vector]).astype(np.float32)


def collect_episodes(env: MultiLinkMassRobotEnv, mass_condition: MassCondition, n_episodes: int = 20, episode_length: int = 300):
    episodes_X = []
    episodes_Y = []
    for episode_index in range(n_episodes):
        obs, _ = env.reset(seed=episode_index)
        prev_obs = obs.copy()
        ep_X = []
        ep_Y = []
        for step in range(episode_length):
            teacher_action = hold_teacher(obs, env.n_links, mass_condition, segment_length=env.segment_length)
            teacher_action = np.clip(teacher_action, env.action_space.low, env.action_space.high)
            ep_X.append(build_features(obs, prev_obs, mass_condition))
            ep_Y.append(teacher_action.copy())
            prev_obs = obs.copy()
            obs, _, terminated, truncated, _ = env.step(teacher_action)
            if terminated or truncated:
                break
        episodes_X.append(np.asarray(ep_X, dtype=np.float32))
        episodes_Y.append(np.asarray(ep_Y, dtype=np.float32))
    return episodes_X, episodes_Y


def main():
    parser = argparse.ArgumentParser(description="Train RC hold policy on the upright multi-link mass robot")
    parser.add_argument("--links", type=int, default=4)
    parser.add_argument("--masses", type=str, default="0.20,0.25,0.30,0.35")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--reservoir", type=int, default=600)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR.with_suffix(".npz"))
    args = parser.parse_args()

    base_link_masses = tuple(float(x) for x in args.masses.split(","))
    if len(base_link_masses) != args.links:
        raise ValueError(f"Expected {args.links} link masses in --masses")

    payloads = TRAIN_PAYLOADS
    all_episodes_X = []
    all_episodes_Y = []
    profile_records = []
    best_profile = None
    best_score = -float("inf")

    for payload_mass in payloads:
        profile = MassProfile(link_masses=base_link_masses, payload_mass=payload_mass)
        env = MultiLinkMassRobotEnv(n_links=args.links, mass_profile=profile, render_mode=None)
        log.info("Training mass profile %s", profile)
        episodes_X, episodes_Y = collect_episodes(env, MassCondition(profile.link_masses, profile.payload_mass), n_episodes=args.episodes, episode_length=args.steps)
        score = float(np.mean([np.mean(ep[:, 2 * args.links + 2]) if len(ep) else 0.0 for ep in episodes_X]))
        all_episodes_X.extend(episodes_X)
        all_episodes_Y.extend(episodes_Y)
        profile_records.append({"payload_mass": payload_mass, "score": score})
        if score > best_score:
            best_score = score
            best_profile = profile
        env.close()

    feature_dim = all_episodes_X[0].shape[1]
    action_dim = all_episodes_Y[0].shape[1]
    esn = EchoStateNetwork(n_inputs=feature_dim, n_reservoir=args.reservoir, seed=42)
    esn.fit(all_episodes_X, all_episodes_Y, ridge=1e-4, washout=WASHOUT)

    model_path = args.output
    with model_path.open("wb") as handle:
        np.savez_compressed(
            handle,
            n_links=np.int32(args.links),
            link_masses=np.asarray(base_link_masses, dtype=np.float32),
            payload_mass=np.float32(best_profile.payload_mass),
            reservoir_Win=esn.Win,
            reservoir_W=esn.W,
            reservoir_W_out=esn.W_out,
            reservoir_leak=np.float32(esn.leak),
            reservoir_n=np.int32(esn.n_reservoir),
            feature_dim=np.int32(feature_dim),
            action_dim=np.int32(action_dim),
            profile_records=np.asarray(profile_records, dtype=object),
        )

    summary_path = model_path.with_suffix(".txt")
    with summary_path.open("w", encoding="utf-8") as handle:
        handle.write("Multi-link RC/CPG training summary\n")
        handle.write(f"Best payload mass: {best_profile.payload_mass:.3f}\n")
        handle.write(f"Best score: {best_score:.6f}\n")
        for record in profile_records:
            handle.write(f"payload={record['payload_mass']:.3f} score={record['score']:.6f}\n")

    log.info("Saved RC/CPG model to %s", model_path)
    log.info("Best payload mass=%.3f score=%.3f", best_profile.payload_mass, best_score)


if __name__ == "__main__":
    main()