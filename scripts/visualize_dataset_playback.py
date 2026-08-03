#!/usr/bin/env python3
"""Replay dataset actions in Gym environments and record the motion.

This script loads a dataset saved by `AntWithRes.generate_dataset` or similar
(`.npz` with `X`, `Y`, `ep_lens`, optional `meta`) and replays each episode's
actions in one or more Gym environments.

It supports:
- replaying the final dataset in a visible or recordable mode
- running the same dataset through multiple Gym env IDs
- saving video recordings per environment/episode
- plotting the robot base trajectory from replay
"""

import argparse
import logging
import os
from pathlib import Path

try:
    import gymnasium as gym
    GYM_USE_GYMNASIUM = True
except ImportError:
    try:
        import gym
        GYM_USE_GYMNASIUM = False
    except ImportError as exc:
        raise ImportError(
            "Neither gymnasium nor gym is installed. Install one of them to run this script."
        ) from exc

import numpy as np

try:
    import imageio
except ImportError:
    imageio = None

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def set_floor_friction(env, mu: float):
    """Set floor geom friction to `mu` for MuJoCo-style envs."""
    try:
        model = env.unwrapped.model
    except Exception:
        return

    ngeom = getattr(model, "ngeom", None)
    if ngeom is None:
        try:
            ngeom = len(model.geom)
        except Exception:
            return

    for i in range(ngeom):
        name = None
        if hasattr(model, "geom_names"):
            try:
                geom_name = model.geom_names[i]
                name = geom_name.decode("utf-8") if isinstance(geom_name, bytes) else str(geom_name)
            except Exception:
                pass
        if name is None:
            try:
                g = model.geom[i]
                name = getattr(g, "name", None)
                if name is None:
                    name = str(g)
                name = name.decode("utf-8") if isinstance(name, bytes) else str(name)
            except Exception:
                pass
        if name is None:
            continue
        if "floor" in name or "ground" in name or "geom_floor" in name:
            try:
                model.geom_friction[i] = np.array([mu, 0.0, 0.0])
            except Exception:
                try:
                    model.geom_friction[i] = [mu, 0.0, 0.0]
                except Exception:
                    pass


def get_base_xy(env):
    """Return approximate base XY position for trajectory plotting."""
    try:
        dat = env.unwrapped.data
        qpos = getattr(dat, "qpos", None)
        if qpos is None:
            return None
        if len(qpos) >= 2:
            return float(qpos[0]), float(qpos[1])
        if len(qpos) >= 1:
            return float(qpos[0]), 0.0
    except Exception:
        pass
    return None


def load_dataset(path):
    log.info("Loading dataset from %s", path)
    data = np.load(path, allow_pickle=True)
    X = data["X"]
    Y = data["Y"]
    ep_lens = None
    if "ep_lens" in data:
        ep_lens = np.asarray(data["ep_lens"]).astype(int).tolist()
    elif "meta" in data:
        meta = data["meta"]
        if isinstance(meta, np.ndarray) and meta.size > 0:
            lengths = []
            for item in meta:
                if isinstance(item, dict) and "episode_length" in item:
                    lengths.append(int(item["episode_length"]))
            if lengths:
                ep_lens = lengths
    if ep_lens is None:
        ep_lens = [len(Y)]

    episodes = []
    idx = 0
    for length in ep_lens:
        episodes.append({"X": X[idx: idx + length], "Y": Y[idx: idx + length]})
        idx += length
    if idx != len(Y):
        log.warning("Dataset episode lengths do not sum to total Y rows: %d vs %d", idx, len(Y))
    return episodes, data


def get_dataset_env_id(data):
    if "env_id" not in data:
        return None
    env_id = data["env_id"]
    if isinstance(env_id, np.ndarray):
        if env_id.shape == ():
            env_id = env_id.item()
        elif len(env_id) == 1:
            env_id = env_id[0]
    if isinstance(env_id, bytes):
        env_id = env_id.decode("utf-8")
    return str(env_id)


def pad_or_truncate_action(action, target_shape):
    action = np.asarray(action, dtype=np.float32)
    if action.shape == target_shape:
        return action
    if action.size < np.prod(target_shape):
        padded = np.zeros(target_shape, dtype=np.float32)
        flat = padded.reshape(-1)
        flat[: action.size] = action.reshape(-1)
        return padded
    flat = action.reshape(-1)[: np.prod(target_shape)]
    return flat.reshape(target_shape)


def make_video_writer(path, fps):
    if imageio is None:
        raise RuntimeError("imageio is required to write video recordings. Install it with pip install imageio.")
    parent = Path(path).parent
    parent.mkdir(parents=True, exist_ok=True)
    return imageio.get_writer(path, fps=fps)


def reset_env(env):
    """Reset an environment and return obs, info for gym/gymnasium compatibility."""
    try:
        result = env.reset()
    except TypeError:
        result = env.reset(return_info=True)

    if isinstance(result, tuple) and len(result) == 2:
        return result
    return result, {}


def step_env(env, action):
    """Step an environment and normalize outputs for gym/gymnasium compatibility."""
    result = env.step(action)
    if len(result) == 5:
        obs, reward, terminated, truncated, info = result
        return obs, reward, terminated, truncated, info
    obs, reward, done, info = result
    return obs, reward, done, False, info


def replay_episode(env, actions, render_mode, record_writer=None, record_every=1, save_trajectory=False):
    obs, info = reset_env(env)
    positions = []
    start_xy = get_base_xy(env)

    if render_mode == "rgb_array":
        frame = env.render()
        if record_writer is not None and (0 % record_every == 0):
            record_writer.append_data(frame)

    terminated = False
    truncated = False
    for step, action in enumerate(actions):
        action = pad_or_truncate_action(action, env.action_space.shape)
        obs, reward, terminated, truncated, info = step_env(env, action)
        if render_mode == "rgb_array":
            frame = env.render()
            if record_writer is not None and ((step + 1) % record_every == 0):
                record_writer.append_data(frame)
        if save_trajectory:
            xy = get_base_xy(env)
            if xy is not None:
                positions.append(xy)
        if terminated or truncated:
            break

    end_xy = get_base_xy(env)
    forward = None
    if start_xy is not None and end_xy is not None:
        try:
            forward = float(end_xy[0] - start_xy[0])
        except Exception:
            forward = None

    return positions, step + 1, forward


def save_trajectory_plot(trajectories, out_path):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 6))
    for label, pts in trajectories.items():
        pts = np.asarray(pts)
        if pts.size == 0:
            continue
        ax.plot(pts[:, 0], pts[:, 1], marker="o", markersize=2, label=label)
    ax.set_title("Base XY Trajectory")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.legend(loc="best")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    log.info("Saved trajectory plot to %s", out_path)


def main():
    parser = argparse.ArgumentParser(description="Replay a dataset through Gym environments and record the motion.")
    parser.add_argument("dataset", type=str, help="Path to the dataset .npz file")
    parser.add_argument("--envs", type=str, default="Ant-v5,Hopper-v5,HalfCheetah-v5,Walker2d-v5",
                        help="Comma-separated Gym env IDs to replay the dataset in")
    parser.add_argument("--episodes", type=int, default=1,
                        help="Number of dataset episodes to replay" )
    parser.add_argument("--render-mode", type=str, default="rgb_array", choices=["rgb_array", "human"],
                        help="Gym render mode for playback")
    parser.add_argument("--record-dir", type=str, default=None,
                        help="Directory to save video recordings")
    parser.add_argument("--record-fps", type=int, default=20,
                        help="FPS for saved videos")
    parser.add_argument("--show-trajectory", action="store_true",
                        help="Save a trajectory plot of the robot base positions")
    parser.add_argument("--trajectory-path", type=str, default="trajectory.png",
                        help="Output path for trajectory plot")
    parser.add_argument("--dataset-episodes", type=int, default=None,
                        help="Optional number of episodes from the dataset to replay. Default uses all available episodes")
    parser.add_argument("--record-every", type=int, default=1,
                        help="Record every Nth frame for faster video generation")
    args = parser.parse_args()

    episodes, data = load_dataset(args.dataset)
    if args.dataset_episodes is not None:
        episodes = episodes[: args.dataset_episodes]
    if args.episodes > len(episodes):
        log.warning("Requested %d episodes but dataset contains only %d; using %d episodes.",
                    args.episodes, len(episodes), len(episodes))
        args.episodes = len(episodes)
    episodes = episodes[: args.episodes]

    dataset_env = get_dataset_env_id(data)
    env_ids = [env_id.strip() for env_id in args.envs.split(",") if env_id.strip()]
    if dataset_env is not None and dataset_env not in env_ids:
        log.info("Dataset contains env_id=%s; adding it to replay list.", dataset_env)
        env_ids.insert(0, dataset_env)
    if len(env_ids) == 0:
        raise ValueError("At least one Gym env ID is required.")

    trajectory_data = {}
    for env_index, env_id in enumerate(env_ids):
        log.info("\n=== Replay on env %d/%d: %s ===", env_index + 1, len(env_ids), env_id)
        try:
            env = gym.make(env_id, render_mode=args.render_mode)
        except Exception as exc:
            log.error("Could not create env %s: %s", env_id, exc)
            continue

        if args.record_dir and args.render_mode == "rgb_array" and imageio is None:
            log.warning("imageio not installed; skipping recording for %s", env_id)

        video_writer = None
        if args.record_dir and args.render_mode == "rgb_array" and imageio is not None:
            env_safe = env_id.replace("/", "_").replace(":", "_")
            record_dir = Path(args.record_dir)
            record_dir.mkdir(parents=True, exist_ok=True)
            video_path = record_dir / f"dataset_playback_{env_safe}.mp4"
            video_writer = make_video_writer(str(video_path), fps=args.record_fps)

        env_trajectory = []
        for episode_idx, ep in enumerate(episodes):
            log.info(" Replaying episode %d/%d (length=%d)", episode_idx + 1, len(episodes), len(ep["Y"]))
            positions, executed_steps, forward = replay_episode(
                env,
                ep["Y"],
                render_mode=args.render_mode,
                record_writer=video_writer,
                record_every=max(1, args.record_every),
                save_trajectory=args.show_trajectory,
            )
            if args.show_trajectory:
                env_trajectory.extend(positions)
            if forward is not None:
                log.info("  Episode %d completed: %d executed steps, forward=%.3f", episode_idx + 1, executed_steps, forward)
            else:
                log.info("  Episode %d completed: %d executed steps", episode_idx + 1, executed_steps)

        if video_writer is not None:
            video_writer.close()
            log.info("Saved video for %s to %s", env_id, video_path)

        if args.show_trajectory:
            label = f"{env_id}"
            trajectory_data[label] = env_trajectory

        env.close()

    if args.show_trajectory and trajectory_data:
        save_trajectory_plot(trajectory_data, args.trajectory_path)


if __name__ == "__main__":
    main()
