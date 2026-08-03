#!/usr/bin/env python3
"""Generate separate multi-CPG datasets and train one RC per MuJoCo environment."""

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
        raise ImportError(
            "Neither gymnasium nor gym is installed. Install one of them to run this script."
        ) from exc

import numpy as np
from reservoirpy.nodes import Reservoir, Ridge
from reservoirpy.model import Model

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

TRAIN_FRICTIONS = [0.5, 1.0, 1.5]
DEFAULT_ENVS = ["Ant-v5", "Hopper-v5", "HalfCheetah-v5", "Walker2d-v5"]


def sanitize_env_id(env_id: str) -> str:
    return env_id.replace("/", "_").replace(":", "_")


def set_floor_friction(env, mu: float):
    """Set floor geom friction to `mu` for MuJoCo envs."""
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


def get_base_x(env) -> float:
    """Get base X position for forward-distance evaluation."""
    try:
        dat = env.unwrapped.data
        qpos = getattr(dat, "qpos", None)
        if qpos is not None and len(qpos) >= 1:
            return float(qpos[0])
    except Exception:
        pass
    return 0.0


class CPGController:
    """Simple sinusoidal CPG controller."""

    def __init__(self, n_actions: int, omega: float = 2.0, amplitudes=None, phases=None, offsets=None):
        self.n = n_actions
        self.omega = float(omega)
        self.amplitudes = np.ones(self.n) if amplitudes is None else np.asarray(amplitudes).reshape(self.n)
        self.phases = np.zeros(self.n) if phases is None else np.asarray(phases).reshape(self.n)
        self.offsets = np.zeros(self.n) if offsets is None else np.asarray(offsets).reshape(self.n)

    @classmethod
    def from_vector(cls, vec, n_actions):
        omega = float(vec[0])
        amps = vec[1:1 + n_actions]
        phases = vec[1 + n_actions:1 + 2 * n_actions]
        offsets = vec[1 + 2 * n_actions:1 + 3 * n_actions]
        return cls(n_actions, omega, amps, phases, offsets)

    def step(self, t: float):
        theta = self.omega * t + self.phases
        return self.offsets + self.amplitudes * np.sin(theta)


def random_search_tune(env, n_actions, n_iters=100, episode_length=500):
    """Tune a CPG using random search."""
    best_vec = None
    best_score = -float("inf")
    for i in range(n_iters):
        omega = np.random.uniform(0.5, 4.0)
        amps = np.random.uniform(0.0, 1.0, size=n_actions)
        phases = np.random.uniform(-np.pi, np.pi, size=n_actions)
        offsets = np.random.uniform(-0.5, 0.5, size=n_actions)
        vec = np.concatenate(([omega], amps, phases, offsets))
        cpg = CPGController.from_vector(vec, n_actions)
        forward, _, _, _, _ = evaluate_controller(env, cpg, episode_length=episode_length)
        if forward > best_score:
            best_score = forward
            best_vec = vec.copy()
            if (i + 1) % 25 == 0 or i == 0:
                log.info("  Tune iter %d: best forward=%.3f", i + 1, best_score)
    return best_vec, best_score


def evaluate_controller(env, controller, episode_length=500):
    obs, info = env.reset()
    start_x = get_base_x(env)
    total_reward = 0.0
    max_speed = 0.0
    slip_count = 0
    prev_x = start_x
    dt = 0.02
    for step in range(episode_length):
        t = step * dt
        action = np.asarray(controller.step(t), dtype=np.float32)
        try:
            action = np.clip(action, env.action_space.low, env.action_space.high)
        except Exception:
            action = np.clip(action, -1.0, 1.0)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        curr_x = get_base_x(env)
        speed = (curr_x - prev_x) / dt
        max_speed = max(max_speed, abs(speed))
        prev_x = curr_x
        if reward < -1.0 or abs(speed) < 0.01:
            slip_count += 1
        if terminated or truncated:
            break
    return get_base_x(env) - start_x, total_reward, step + 1, max_speed, slip_count


def collect_cpg_dataset(env_id, env, cpg, episodes_per_friction=15, min_forward=0.5, episode_length=500):
    """Collect CPG demonstration data for one environment and controller."""
    X = []
    Y = []
    ep_lens = []
    attempts = 0
    good_eps = 0
    max_attempts = episodes_per_friction * 6
    best_forward = -float("inf")
    best_episode = None

    while good_eps < episodes_per_friction and attempts < max_attempts:
        attempts += 1
        obs, info = env.reset()
        start_x = get_base_x(env)
        ep_obs = []
        ep_actions = []

        for step in range(episode_length):
            t = step * 0.02
            action = np.asarray(cpg.step(t), dtype=np.float32)
            try:
                action = np.clip(action, env.action_space.low, env.action_space.high)
            except Exception:
                action = np.clip(action, -1.0, 1.0)
            ep_obs.append(obs.copy())
            ep_actions.append(action.copy())
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break

        forward = get_base_x(env) - start_x
        if len(ep_obs) >= 2 and forward > best_forward:
            best_forward = forward
            best_episode = (ep_obs.copy(), ep_actions.copy())

        if forward < min_forward or len(ep_obs) < 2:
            continue

        X.extend(ep_obs)
        Y.extend(ep_actions)
        ep_lens.append(len(ep_obs))
        good_eps += 1

    if good_eps == 0 and best_episode is not None:
        log.warning(
            "No episodes met min_forward=%.3f for env=%s. Using best fallback episode with forward=%.3f.",
            min_forward,
            env_id,
            best_forward,
        )
        ep_obs, ep_actions = best_episode
        X.extend(ep_obs)
        Y.extend(ep_actions)
        ep_lens.append(len(ep_obs))
        good_eps = 1

    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32), ep_lens


def collect_dataset_for_env(env_id, frictions, dataset_dir, cpg_dir, episodes_per_friction=15, cpg_tune_iters=120):
    dataset_dir = Path(dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    cpg_dir = Path(cpg_dir)
    cpg_dir.mkdir(parents=True, exist_ok=True)

    env_safe = sanitize_env_id(env_id)
    dataset_path = dataset_dir / f"dataset_{env_safe}.npz"
    model_path = dataset_dir / f"dataset_{env_safe}_cpgs.pkl"

    cpg_stats = []
    all_X = []
    all_Y = []
    all_ep_lens = []
    all_frictions = []
    cpg_vecs = []

    for idx, mu in enumerate(frictions):
        log.info("\n=== Collecting env=%s friction=%.1f ===", env_id, mu)
        env = gym.make(env_id, render_mode="rgb_array")
        set_floor_friction(env, mu)
        n_actions = env.action_space.shape[0]

        vec_path = cpg_dir / f"{env_safe}_cpg_{idx}.npy"
        if vec_path.exists():
            best_vec = np.load(vec_path)
            log.info("Loaded cached CPG vector from %s", vec_path)
        else:
            best_vec, best_score = random_search_tune(env, n_actions, n_iters=cpg_tune_iters, episode_length=500)
            np.save(vec_path, best_vec)
            log.info("Saved CPG vector to %s (best forward=%.3f)", vec_path, best_score)

        cpg = CPGController.from_vector(best_vec, n_actions)
        forward, reward, steps, _, _ = evaluate_controller(env, cpg, episode_length=500)
        log.info("CPG eval: forward=%.3f reward=%.3f steps=%d", forward, reward, steps)

        min_forward_val = 0.5 if "Ant" in env_id else 0.1
        X, Y, ep_lens = collect_cpg_dataset(
            env_id,
            env,
            cpg,
            episodes_per_friction=episodes_per_friction,
            min_forward=min_forward_val,
            episode_length=500,
        )
        env.close()

        if len(ep_lens) == 0:
            raise RuntimeError("No valid episodes collected for env=%s friction=%.1f" % (env_id, mu))

        all_X.append(X)
        all_Y.append(Y)
        all_ep_lens.extend(ep_lens)
        all_frictions.extend([mu] * X.shape[0])
        cpg_stats.append({"friction": float(mu), "forward": float(forward), "reward": float(reward), "cpg_path": str(vec_path)})
        cpg_vecs.append(best_vec)

    X_all = np.vstack(all_X)
    Y_all = np.vstack(all_Y)
    frictions_array = np.asarray(all_frictions, dtype=np.float32).reshape(-1, 1)

    np.savez_compressed(
        dataset_path,
        env_id=np.array(env_id, dtype=object),
        X=X_all,
        Y=Y_all,
        ep_lens=np.asarray(all_ep_lens, dtype=np.int32),
        frictions=frictions_array,
        cpg_stats=np.array(cpg_stats, dtype=object),
    )

    with open(model_path, "wb") as f:
        pickle.dump({"env_id": env_id, "cpg_vecs": cpg_vecs, "cpg_stats": cpg_stats}, f)

    log.info("Saved dataset to %s", dataset_path)
    log.info("Saved CPG metadata to %s", model_path)
    return dataset_path, model_path


def train_rc_from_dataset(env_id, dataset_path, output_dir, n_reservoir=1000, ridge=1e-6):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    env_safe = sanitize_env_id(env_id)

    data = np.load(dataset_path, allow_pickle=True)
    X = data["X"]
    Y = data["Y"]

    input_mean = X.mean(axis=0)
    input_std = X.std(axis=0)
    input_std[input_std < 1e-8] = 1.0
    X_norm = (X - input_mean) / input_std

    env = gym.make(env_id, render_mode="rgb_array")
    n_inputs = env.observation_space.shape[0]
    n_outputs = env.action_space.shape[0]
    env.close()

    if X_norm.shape[1] != n_inputs:
        raise ValueError(f"Dataset input dim {X_norm.shape[1]} does not match env obs dim {n_inputs}")

    np.random.seed(42)
    Win = np.random.uniform(-2.0, 2.0, size=(n_reservoir, n_inputs))
    density = 0.1
    mask = (np.random.rand(n_reservoir, n_reservoir) < density)
    W = np.random.uniform(-0.5, 0.5, size=(n_reservoir, n_reservoir)) * mask.astype(float)
    eigvals = np.linalg.eigvals(W)
    max_abs = np.max(np.abs(eigvals))
    if max_abs > 1e-12:
        W *= 0.95 / max_abs
    else:
        W = np.random.uniform(-0.1, 0.1, size=(n_reservoir, n_reservoir))

    reservoir = Reservoir(units=n_reservoir, input_dim=n_inputs, Win=Win, W=W)
    readout = Ridge(input_dim=n_reservoir, output_dim=n_outputs, ridge=ridge)
    esn = Model([reservoir, readout], edges=[(reservoir, 0, readout)])

    log.info("Training RC for env %s on %d samples", env_id, X_norm.shape[0])
    esn.fit(X_norm, Y)

    model_path = output_dir / f"rc_model_{env_safe}.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({
            "env_id": env_id,
            "rc_model": esn,
            "input_mean": input_mean,
            "input_std": input_std,
            "dataset_path": str(dataset_path),
        }, f)

    summary_path = output_dir / f"training_summary_{env_safe}.txt"
    with open(summary_path, "w") as f:
        f.write(f"RC model trained on env {env_id}\n")
        f.write(f"Dataset: {dataset_path}\n")
        f.write(f"Samples: {X.shape[0]}\n")
        f.write(f"Input dim: {n_inputs}\n")
        f.write(f"Output dim: {n_outputs}\n")
    log.info("Saved RC model to %s", model_path)
    log.info("Saved training summary to %s", summary_path)
    return model_path


def main():
    parser = argparse.ArgumentParser(description="Generate separate datasets and RC models for multiple MuJoCo environments.")
    parser.add_argument("--envs", type=str, default=','.join(DEFAULT_ENVS),
                        help="Comma-separated Gym environment IDs")
    parser.add_argument("--dataset-dir", type=str, default="datasets",
                        help="Directory to save generated datasets")
    parser.add_argument("--cpg-dir", type=str, default="cpg_vectors",
                        help="Directory to cache CPG vectors")
    parser.add_argument("--output-dir", type=str, default="multienv_rc_models",
                        help="Directory to save trained RC models")
    parser.add_argument("--episodes-per-friction", type=int, default=15,
                        help="Episodes per friction for dataset collection")
    parser.add_argument("--cpg-tune-iters", type=int, default=120,
                        help="Random search iterations for CPG tuning")
    parser.add_argument("--no-train-rc", action="store_true",
                        help="Only generate datasets and skip RC training")
    parser.add_argument("--render-mode", type=str, default="rgb_array", choices=["rgb_array", "human"],
                        help="Render mode for environment during tuning and collection")
    parser.add_argument("--frictions", type=str, default=','.join(str(mu) for mu in TRAIN_FRICTIONS),
                        help="Comma-separated friction values for 3 CPGs")
    args = parser.parse_args()

    env_ids = [env_id.strip() for env_id in args.envs.split(',') if env_id.strip()]
    frictions = [float(mu) for mu in args.frictions.split(',') if mu.strip()]

    for env_id in env_ids:
        log.info("\n=== Processing env %s ===", env_id)
        dataset_path, metadata_path = collect_dataset_for_env(
            env_id,
            frictions,
            args.dataset_dir,
            args.cpg_dir,
            episodes_per_friction=args.episodes_per_friction,
            cpg_tune_iters=args.cpg_tune_iters,
        )
        if not args.no_train_rc:
            train_rc_from_dataset(env_id, dataset_path, args.output_dir)


if __name__ == "__main__":
    main()
