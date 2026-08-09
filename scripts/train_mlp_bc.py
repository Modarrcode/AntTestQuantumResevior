"""
Train Dense Neural Network for CPG imitation using scikit-learn (no PyTorch/TensorFlow).
Purpose: Test if standard neural networks can achieve better RC alignment.
"""

import argparse
import logging
import os
import pickle
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

TRAIN_FRICTIONS = [0.5, 1.0, 1.5]
OUTPUT_DIR = "mlp_bc_model"


def set_floor_friction(env, mu: float):
    """Set floor friction."""
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
    """Get ant base x position."""
    try:
        return float(env.unwrapped.data.qpos[0])
    except Exception:
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

    def step(self, t):
        phase = self.omega * t + self.phases
        return self.offsets + self.amplitudes * np.sin(phase)


@dataclass
class DatasetBundle:
    observations: np.ndarray
    actions: np.ndarray
    frictions: np.ndarray
    cpg_stats: dict


def collect_multifriction_dataset(frictions, episodes_per_friction, min_forward, cpg_tune_iters):
    """Collect demonstration data from CPG across multiple frictions."""
    all_obs = []
    all_actions = []
    all_frictions = []
    cpg_stats = {}

    for mu in frictions:
        env = gym.make("Ant-v5", render_mode="rgb_array")
        set_floor_friction(env, mu)

        # Tune CPG parameters
        best_dist = -float("inf")
        best_cpg = None
        for _ in range(cpg_tune_iters):
            vec = np.random.randn(1 + 8 + 8 + 8)
            cpg = CPGController.from_vector(vec, 8)
            set_floor_friction(env, mu)
            obs, _ = env.reset()
            start_x = get_base_x(env)

            for step in range(500):
                action = np.asarray(cpg.step(step * 0.02), dtype=np.float32)
                action = np.clip(action, env.action_space.low, env.action_space.high)
                obs, _, term, trunc, _ = env.step(action)
                if term or trunc:
                    break

            dist = get_base_x(env) - start_x
            if dist > best_dist:
                best_dist = dist
                best_cpg = cpg

        cpg_stats[mu] = {"forward": float(best_dist), "cpg": best_cpg}
        log.info("Friction %.1f CPG: forward=%.3f", mu, best_dist)

        # Collect demonstrations
        good_eps = 0
        attempts = 0
        max_attempts = episodes_per_friction * 3

        while good_eps < episodes_per_friction and attempts < max_attempts:
            attempts += 1
            ep_obs = []
            ep_actions = []
            obs, _ = env.reset()
            start_x = get_base_x(env)

            for step in range(500):
                action = np.asarray(best_cpg.step(step * 0.02), dtype=np.float32)
                action = np.clip(action, env.action_space.low, env.action_space.high)
                ep_obs.append(obs.copy())
                ep_actions.append(action.copy())

                obs, _, term, trunc, _ = env.step(action)
                if term or trunc:
                    break

            forward = get_base_x(env) - start_x
            if forward < min_forward or len(ep_obs) < 2:
                continue

            ep_obs = np.asarray(ep_obs, dtype=np.float32)
            ep_actions = np.asarray(ep_actions, dtype=np.float32)

            all_obs.append(ep_obs)
            all_actions.append(ep_actions)
            all_frictions.append(np.full((len(ep_obs), 1), mu, dtype=np.float32))
            good_eps += 1

            if good_eps % 5 == 0:
                log.info("  Friction %.1f collected %d/%d good episodes", mu, good_eps, episodes_per_friction)

        env.close()
        log.info("Friction %.1f: %d good episodes from %d attempts", mu, good_eps, attempts)

    obs = np.vstack(all_obs).astype(np.float32)
    actions = np.vstack(all_actions).astype(np.float32)
    fric = np.vstack(all_frictions).astype(np.float32).reshape(-1)

    return DatasetBundle(obs, actions, fric, cpg_stats)


class MLPBCPolicy:
    """MLP-based behavioral cloning policy."""

    def __init__(self, model, scaler_in, scaler_out):
        self.model = model
        self.scaler_in = scaler_in
        self.scaler_out = scaler_out

    def predict(self, obs, friction):
        x = np.concatenate([obs, [friction]])
        x_norm = self.scaler_in.transform(x.reshape(1, -1))
        action_norm = self.model.predict(x_norm).flatten()
        action = self.scaler_out.inverse_transform(action_norm.reshape(1, -1)).flatten()
        return action


def train_mlp_bc(
    frictions,
    episodes_per_friction=15,
    min_forward=0.5,
    cpg_tune_iters=200,
    hidden_layers=(512, 256),
    alpha=1e-4,  # L2 regularization
    learning_rate_init=1e-3,
    max_iter=1000,
):
    """Train MLP behavioral cloning model."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    log.info("=" * 70)
    log.info("MLP BEHAVIORAL CLONING TRAINING")
    log.info("=" * 70)

    data = collect_multifriction_dataset(
        frictions=frictions,
        episodes_per_friction=episodes_per_friction,
        min_forward=min_forward,
        cpg_tune_iters=cpg_tune_iters,
    )

    log.info("Collected dataset: obs=%s actions=%s", data.observations.shape, data.actions.shape)

    # Prepare input data (obs + friction)
    X = np.concatenate([data.observations, data.frictions.reshape(-1, 1)], axis=1)
    y = data.actions

    # Normalize
    scaler_in = StandardScaler()
    X_norm = scaler_in.fit_transform(X)

    scaler_out = StandardScaler()
    y_norm = scaler_out.fit_transform(y)

    log.info("Training data: X=%s y=%s", X_norm.shape, y_norm.shape)

    # Train MLP
    log.info("Training MLP BC (hidden_layers=%s, alpha=%.1e)...", hidden_layers, alpha)
    model = MLPRegressor(
        hidden_layer_sizes=hidden_layers,
        activation="relu",
        solver="adam",
        alpha=alpha,
        batch_size=64,
        learning_rate_init=learning_rate_init,
        max_iter=max_iter,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        verbose=True,
    )
    model.fit(X_norm, y_norm)

    log.info("Training complete")

    # Create policy
    policy = MLPBCPolicy(model, scaler_in, scaler_out)

    # Evaluate
    eval_results = []
    for mu in frictions:
        rewards = []
        distances = []
        
        for ep in range(5):
            env = gym.make("Ant-v5", render_mode="rgb_array")
            set_floor_friction(env, mu)
            obs, _ = env.reset()
            start_x = get_base_x(env)
            ep_reward = 0

            for step in range(500):
                action = policy.predict(obs, mu)
                action = np.clip(action, env.action_space.low, env.action_space.high)
                obs, reward, term, trunc, _ = env.step(action)
                ep_reward += reward
                if term or trunc:
                    break

            dist = get_base_x(env) - start_x
            distances.append(dist)
            rewards.append(ep_reward)
            env.close()

        avg_dist = float(np.mean(distances))
        avg_reward = float(np.mean(rewards))
        eval_results.append({"friction": float(mu), "bc_reward": avg_reward, "bc_fwd": avg_dist})
        log.info("Eval friction %.1f -> forward=%.3f reward=%.1f", mu, avg_dist, avg_reward)

    # Save
    model_data = {
        "model": model,
        "scaler_in": scaler_in,
        "scaler_out": scaler_out,
        "eval_results": eval_results,
        "cpg_stats": data.cpg_stats,
    }

    pkl_path = os.path.join(OUTPUT_DIR, "mlp_bc.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(model_data, f)
    log.info("Saved model to: %s", pkl_path)

    # Summary
    summary_path = os.path.join(OUTPUT_DIR, "summary.txt")
    with open(summary_path, "w") as f:
        f.write("MLP Behavioral Cloning Summary\n")
        f.write("=" * 70 + "\n")
        f.write(f"Model path: {pkl_path}\n")
        f.write(f"Hidden layers: {hidden_layers}\n")
        f.write(f"Alpha (L2 reg): {alpha}\n\n")
        f.write("CPG Baseline Stats:\n")
        for mu in frictions:
            cpg_fwd = data.cpg_stats[mu]["forward"]
            f.write(f"  friction {mu}: cpg_fwd={cpg_fwd:.3f}\n")
        f.write("\nMLP BC Evaluation Stats:\n")
        for res in eval_results:
            f.write(f"  friction {res['friction']}: bc_fwd={res['bc_fwd']:.3f}, bc_reward={res['bc_reward']:.1f}\n")

    log.info("Saved summary to: %s", summary_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes-per-friction", type=int, default=15)
    parser.add_argument("--cpg-tune-iters", type=int, default=200)
    parser.add_argument("--hidden-layers", type=str, default="512,256")
    parser.add_argument("--alpha", type=float, default=1e-4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--max-iter", type=int, default=1000)

    args = parser.parse_args()
    
    hidden_layers = tuple(map(int, args.hidden_layers.split(",")))

    train_mlp_bc(
        frictions=TRAIN_FRICTIONS,
        episodes_per_friction=args.episodes_per_friction,
        cpg_tune_iters=args.cpg_tune_iters,
        hidden_layers=hidden_layers,
        alpha=args.alpha,
        learning_rate_init=args.learning_rate,
        max_iter=args.max_iter,
    )
