#!/usr/bin/env python3
"""Visualize the trained RC models for multiple MuJoCo environments."""

import argparse
import logging
import os
import pickle
from pathlib import Path

try:
    import gymnasium as gym
    GYM_USE_GYMNASIUM = True
except ImportError:
    try:
        import gym
        GYM_USE_GYMNASIUM = False
    except ImportError as exc:
        raise ImportError("Neither gymnasium nor gym is installed. Install one of them to run this script.") from exc

import numpy as np

try:
    import imageio
except ImportError:
    imageio = None

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def sanitize_env_id(env_id: str) -> str:
    return env_id.replace("/", "_").replace(":", "_")


def reset_env(env):
    try:
        result = env.reset()
    except TypeError:
        result = env.reset(return_info=True)
    if isinstance(result, tuple) and len(result) == 2:
        return result
    return result, {}


def step_env(env, action):
    result = env.step(action)
    if len(result) == 5:
        obs, reward, terminated, truncated, info = result
        return obs, reward, terminated, truncated, info
    obs, reward, done, info = result
    return obs, reward, done, False, info


def make_video_writer(path, fps):
    if imageio is None:
        raise RuntimeError("imageio is required to write video recordings. Install it with pip install imageio.")
    parent = Path(path).parent
    parent.mkdir(parents=True, exist_ok=True)
    return imageio.get_writer(path, fps=fps)


def load_model(model_path):
    log.info("Loading model %s", model_path)
    with open(model_path, "rb") as f:
        return pickle.load(f)


def run_model(env, model_data, episode_length, num_episodes, render_mode, record_writer=None, record_every=1):
    esn = model_data["rc_model"]
    input_mean = model_data["input_mean"]
    input_std = model_data["input_std"]

    distances = []
    rewards = []

    for ep in range(num_episodes):
        log.info("  Episode %d/%d", ep + 1, num_episodes)
        obs, info = reset_env(env)
        start_x = get_base_x(env)
        total_reward = 0.0

        if render_mode == "rgb_array":
            frame = env.render()
            if record_writer is not None and (0 % record_every == 0):
                record_writer.append_data(frame)

        for step in range(episode_length):
            obs_norm = (obs - input_mean) / input_std
            action = esn.run(obs_norm.reshape(1, -1)).flatten()
            try:
                action = np.clip(action, env.action_space.low, env.action_space.high)
            except Exception:
                action = np.clip(action, -1.0, 1.0)

            obs, reward, terminated, truncated, info = step_env(env, action)
            total_reward += float(reward)

            if render_mode == "rgb_array":
                frame = env.render()
                if record_writer is not None and ((step + 1) % record_every == 0):
                    record_writer.append_data(frame)

            if terminated or truncated:
                break

        end_x = get_base_x(env)
        distances.append(end_x - start_x)
        rewards.append(total_reward)
        log.info("    Done: distance=%.3f reward=%.3f steps=%d", distances[-1], rewards[-1], step + 1)

    return distances, rewards


def get_base_x(env):
    try:
        dat = env.unwrapped.data
        qpos = getattr(dat, "qpos", None)
        if qpos is not None and len(qpos) >= 1:
            return float(qpos[0])
    except Exception:
        pass
    return 0.0


def save_results(results, out_path):
    with open(out_path, "w") as f:
        for item in results:
            f.write("Env: %s\n" % item["env_id"])
            f.write("Model: %s\n" % item["model_path"])
            f.write("  episodes=%d\n" % item["num_episodes"])
            f.write("  avg_distance=%.3f std_distance=%.3f\n" % (item["avg_distance"], item["std_distance"]))
            f.write("  avg_reward=%.3f std_reward=%.3f\n" % (item["avg_reward"], item["std_reward"]))
            f.write("\n")
    log.info("Saved summary results to %s", out_path)


def find_model_for_env(output_dir, env_id):
    env_safe = sanitize_env_id(env_id)
    candidate = Path(output_dir) / f"rc_model_{env_safe}.pkl"
    if candidate.exists():
        return str(candidate)
    # fallback: search for env_id in filenames
    for p in Path(output_dir).glob("*.pkl"):
        if env_id.replace("-", "").lower() in p.name.replace("-", "").lower():
            return str(p)
    return None


def main():
    parser = argparse.ArgumentParser(description="Visualize RC models trained for multiple MuJoCo environments.")
    parser.add_argument("--envs", type=str, default="Ant-v5,Hopper-v5,HalfCheetah-v5,Walker2d-v5",
                        help="Comma-separated Gym env IDs to visualize")
    parser.add_argument("--model-dir", type=str, default="multienv_rc_models",
                        help="Directory containing saved RC model pickle files")
    parser.add_argument("--num-episodes", type=int, default=1,
                        help="Number of episodes to run per model (use 1 for faster playback)")
    parser.add_argument("--episode-length", type=int, default=500,
                        help="Max steps per episode")
    parser.add_argument("--render-mode", type=str, default="none", choices=["none", "human", "rgb_array"],
                        help="Render mode for playback; use none for fastest execution")
    parser.add_argument("--record-dir", type=str, default=None,
                        help="Directory to save video recordings when using rgb_array")
    parser.add_argument("--record-fps", type=int, default=20,
                        help="FPS for saved videos")
    parser.add_argument("--record-every", type=int, default=1,
                        help="Record every Nth frame")
    parser.add_argument("--summary-path", type=str, default="visualization_summary.txt",
                        help="Path to save per-model summary results")
    args = parser.parse_args()

    env_ids = [env_id.strip() for env_id in args.envs.split(",") if env_id.strip()]
    if len(env_ids) == 0:
        raise ValueError("At least one Gym env ID is required.")

    results = []
    for env_id in env_ids:
        model_path = find_model_for_env(args.model_dir, env_id)
        if model_path is None:
            log.error("No RC model found for env %s in %s", env_id, args.model_dir)
            continue

        model_data = load_model(model_path)
        render_mode = None if args.render_mode == "none" else args.render_mode
        env = gym.make(env_id, render_mode=render_mode)

        record_writer = None
        video_path = None
        if args.render_mode == "rgb_array" and args.record_dir is not None:
            if imageio is None:
                log.error("imageio is required to record videos. Install it with pip install imageio.")
                env.close()
                continue
            video_dir = Path(args.record_dir)
            video_dir.mkdir(parents=True, exist_ok=True)
            video_path = video_dir / f"visualize_{sanitize_env_id(env_id)}.mp4"
            record_writer = make_video_writer(str(video_path), fps=args.record_fps)

        log.info("\n=== Visualizing env %s with model %s ===", env_id, model_path)
        distances, rewards = run_model(env, model_data,
                                      episode_length=args.episode_length,
                                      num_episodes=args.num_episodes,
                                      render_mode=args.render_mode,
                                      record_writer=record_writer,
                                      record_every=max(1, args.record_every))

        if record_writer is not None:
            record_writer.close()
            log.info("Saved video to %s", video_path)

        env.close()

        avg_distance = float(np.mean(distances)) if distances else 0.0
        std_distance = float(np.std(distances)) if distances else 0.0
        avg_reward = float(np.mean(rewards)) if rewards else 0.0
        std_reward = float(np.std(rewards)) if rewards else 0.0

        results.append({
            "env_id": env_id,
            "model_path": model_path,
            "num_episodes": len(distances),
            "avg_distance": avg_distance,
            "std_distance": std_distance,
            "avg_reward": avg_reward,
            "std_reward": std_reward,
        })

    if results:
        save_results(results, args.summary_path)


if __name__ == "__main__":
    main()
