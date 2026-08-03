"""
Export multifunctional RC+AE evaluation into Excel-ready CSV files.

Outputs:
- multifunctional_rc_ae_model/excel_episode_results.csv
- multifunctional_rc_ae_model/excel_summary_results.csv
"""

import argparse
import csv
import os
import pickle

import gymnasium as gym
import numpy as np


DEFAULT_MODEL = os.path.join("multifunctional_rc_ae_model", "multifunctional_rc_autoencoder.pkl")


def set_floor_friction(env, mu: float):
    """Set floor friction on MuJoCo floor/ground geoms."""
    try:
        model = env.unwrapped.model
    except Exception:
        return

    def geom_name_at(i):
        if hasattr(model, "geom_names"):
            try:
                n = model.geom_names[i]
                return n.decode("utf-8") if isinstance(n, bytes) else str(n)
            except Exception:
                pass
        if hasattr(model, "geom"):
            try:
                g = model.geom[i]
                name = getattr(g, "name", None)
                if name is None:
                    try:
                        name = g[0]
                    except Exception:
                        name = str(g)
                return name.decode("utf-8") if isinstance(name, bytes) else str(name)
            except Exception:
                pass
        return f"geom_{i}"

    ngeom = getattr(model, "ngeom", None)
    if ngeom is None:
        try:
            ngeom = len(model.geom)
        except Exception:
            ngeom = 0

    for i in range(ngeom):
        name = geom_name_at(i)
        if "floor" in name or "ground" in name or "geom_floor" in name:
            try:
                model.geom_friction[i] = np.array([mu, 0.0, 0.0])
            except Exception:
                try:
                    model.geom_friction[i] = [mu, 0.0, 0.0]
                except Exception:
                    pass


def get_base_x(env) -> float:
    try:
        return float(env.unwrapped.data.qpos[0])
    except Exception:
        return 0.0


class CPGController:
    """Simple sinusoidal controller from saved vectors."""

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


class NumpyAutoencoderInference:
    """Inference-only wrapper for saved AE weights."""

    def __init__(self, state):
        self.W1 = state["W1"]
        self.b1 = state["b1"]
        self.W2 = state["W2"]
        self.b2 = state["b2"]

    def encode(self, x):
        h1 = np.tanh(x @ self.W1 + self.b1)
        return np.tanh(h1 @ self.W2 + self.b2)


class MultifunctionalPolicy:
    """Policy wrapper for RC + AE model."""

    def __init__(self, model_data):
        self.rc_model = model_data["rc_model"]
        self.ae = NumpyAutoencoderInference(model_data["autoencoder"])
        self.obs_mean = model_data["obs_mean"]
        self.obs_std = model_data["obs_std"]
        self.action_mean = model_data["action_mean"]
        self.action_std = model_data["action_std"]

    def reset(self):
        self.rc_model.reset()

    def predict(self, obs, friction):
        obs_norm = (obs - self.obs_mean) / self.obs_std
        z = self.ae.encode(obs_norm.reshape(1, -1)).reshape(-1)
        rc_in = np.concatenate([z, [friction]]).astype(np.float32)
        rc_out = self.rc_model.run(rc_in.reshape(1, -1)).reshape(-1)

        n_actions = self.action_mean.shape[0]
        action_norm = rc_out[:n_actions]
        return action_norm * self.action_std + self.action_mean


def run_episode(env, policy, friction, episode_length=500):
    policy.reset()
    set_floor_friction(env, friction)
    obs, _ = env.reset()
    start_x = get_base_x(env)
    total_reward = 0.0
    steps = 0

    for _ in range(episode_length):
        action = policy.predict(obs, friction)
        action = np.clip(action, env.action_space.low, env.action_space.high)
        obs, reward, term, trunc, _ = env.step(action)
        total_reward += float(reward)
        steps += 1
        if term or trunc:
            break

    distance = get_base_x(env) - start_x
    return float(total_reward), float(distance), int(steps)


def run_episode_cpg(env, cpg, episode_length=500):
    obs, _ = env.reset()
    start_x = get_base_x(env)
    total_reward = 0.0
    steps = 0

    for step in range(episode_length):
        action = np.asarray(cpg.step(step * 0.02), dtype=np.float32)
        action = np.clip(action, env.action_space.low, env.action_space.high)
        obs, reward, term, trunc, _ = env.step(action)
        total_reward += float(reward)
        steps += 1
        if term or trunc:
            break

    distance = get_base_x(env) - start_x
    return float(total_reward), float(distance), int(steps)


def summary_stats(values):
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "median": float(np.median(arr)),
    }


def export_results(model_path, episodes=20, episode_length=500):
    with open(model_path, "rb") as f:
        model_data = pickle.load(f)

    frictions = model_data.get("frictions", [0.5, 1.0, 1.5])
    policy = MultifunctionalPolicy(model_data)

    out_dir = os.path.dirname(model_path)
    episode_csv = os.path.join(out_dir, "excel_episode_results.csv")
    summary_csv = os.path.join(out_dir, "excel_summary_results.csv")

    episode_rows = []
    summary_rows = []

    for idx, friction in enumerate(frictions):
        env = gym.make("Ant-v5", render_mode="rgb_array")
        set_floor_friction(env, float(friction))

        cpg_path = os.path.join("datasets", f"cpg_idx{idx}.npy")
        cpg = None
        if os.path.exists(cpg_path):
            cpg_vec = np.load(cpg_path)
            cpg = CPGController.from_vector(cpg_vec, env.action_space.shape[0])

        rc_rewards = []
        rc_distances = []
        cpg_rewards = []
        cpg_distances = []

        for ep in range(1, episodes + 1):
            rc_reward, rc_dist, rc_steps = run_episode(env, policy, float(friction), episode_length=episode_length)
            rc_rewards.append(rc_reward)
            rc_distances.append(rc_dist)

            row = {
                "friction": float(friction),
                "episode": ep,
                "policy": "multifunctional_rc_ae",
                "reward": rc_reward,
                "forward_distance": rc_dist,
                "steps": rc_steps,
            }

            if cpg is not None:
                set_floor_friction(env, float(friction))
                cpg_reward, cpg_dist, cpg_steps = run_episode_cpg(env, cpg, episode_length=episode_length)
                cpg_rewards.append(cpg_reward)
                cpg_distances.append(cpg_dist)
                episode_rows.append(row)
                episode_rows.append({
                    "friction": float(friction),
                    "episode": ep,
                    "policy": "cpg",
                    "reward": cpg_reward,
                    "forward_distance": cpg_dist,
                    "steps": cpg_steps,
                })
            else:
                episode_rows.append(row)

        rc_r = summary_stats(rc_rewards)
        rc_d = summary_stats(rc_distances)

        summary_rows.append({
            "friction": float(friction),
            "policy": "multifunctional_rc_ae",
            "episodes": episodes,
            "reward_mean": rc_r["mean"],
            "reward_std": rc_r["std"],
            "reward_median": rc_r["median"],
            "reward_min": rc_r["min"],
            "reward_max": rc_r["max"],
            "distance_mean": rc_d["mean"],
            "distance_std": rc_d["std"],
            "distance_median": rc_d["median"],
            "distance_min": rc_d["min"],
            "distance_max": rc_d["max"],
        })

        if len(cpg_rewards) > 0:
            cpg_r = summary_stats(cpg_rewards)
            cpg_d = summary_stats(cpg_distances)
            summary_rows.append({
                "friction": float(friction),
                "policy": "cpg",
                "episodes": episodes,
                "reward_mean": cpg_r["mean"],
                "reward_std": cpg_r["std"],
                "reward_median": cpg_r["median"],
                "reward_min": cpg_r["min"],
                "reward_max": cpg_r["max"],
                "distance_mean": cpg_d["mean"],
                "distance_std": cpg_d["std"],
                "distance_median": cpg_d["median"],
                "distance_min": cpg_d["min"],
                "distance_max": cpg_d["max"],
            })

        env.close()

    with open(episode_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["friction", "episode", "policy", "reward", "forward_distance", "steps"],
        )
        writer.writeheader()
        writer.writerows(episode_rows)

    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "friction",
                "policy",
                "episodes",
                "reward_mean",
                "reward_std",
                "reward_median",
                "reward_min",
                "reward_max",
                "distance_mean",
                "distance_std",
                "distance_median",
                "distance_min",
                "distance_max",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    return episode_csv, summary_csv, summary_rows


def parse_args():
    parser = argparse.ArgumentParser(description="Export multifunctional RC results to Excel-ready CSV")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--length", type=int, default=500)
    return parser.parse_args()


def main():
    args = parse_args()
    episode_csv, summary_csv, summary_rows = export_results(
        model_path=args.model,
        episodes=args.episodes,
        episode_length=args.length,
    )

    print("Export complete")
    print(f"Episode CSV: {episode_csv}")
    print(f"Summary CSV: {summary_csv}")
    for row in summary_rows:
        print(row)


if __name__ == "__main__":
    main()
