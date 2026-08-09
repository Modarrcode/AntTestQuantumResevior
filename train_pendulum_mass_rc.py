"""
Standalone reservoir-computing trainer for Pendulum-v1 with varying payload mass.

This is a dedicated pendulum-only version of the RC + autoencoder workflow.
It uses a mass-aware energy-shaping swing-up controller as the teacher and
trains a reservoir model to imitate that controller across multiple masses.
"""

import argparse
import logging
import os
import pickle

import gymnasium as gym
import numpy as np


logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


MASS_VALUES = (1.0, 1.5)
OUTPUT_DIR = "pendulum_physical_throw_band_rc_model"
WASHOUT = 20


def set_pendulum_mass(env, mass: float):
    """Set Pendulum-v1 payload mass when the environment exposes it."""
    try:
        env.unwrapped.m = float(mass)
    except Exception:
        pass


def build_pendulum_features(env, obs, mass: float):
    """Compact state vector for pendulum imitation learning."""
    cos_theta, sin_theta, theta_dot = np.asarray(obs, dtype=np.float64).reshape(3)
    theta = np.arctan2(sin_theta, cos_theta)
    unwrapped = env.unwrapped
    length = float(getattr(unwrapped, "l", 1.0))
    gravity = float(getattr(unwrapped, "g", 10.0))
    payload_mass = float(getattr(unwrapped, "m", mass))
    current_energy = 0.5 * payload_mass * (length * theta_dot) ** 2 + payload_mass * gravity * length * (1.0 - cos_theta)
    target_energy = payload_mass * gravity * length
    energy_error = target_energy - current_energy
    return np.array([cos_theta, sin_theta, theta, theta_dot, energy_error, mass], dtype=np.float32)


def pendulum_reward_from_obs(obs):
    """Gymnasium Pendulum reward uses the current state after the step."""
    cos_theta, sin_theta, theta_dot = np.asarray(obs, dtype=np.float64).reshape(3)
    theta = np.arctan2(sin_theta, cos_theta)
    return -(theta * theta + 0.1 * theta_dot * theta_dot)


class PendulumSwingUpTeacher:
    """Energy-shaping swing-up controller with upright stabilization."""

    def __init__(self, energy_gain=2.5, stabilize_kp=12.0, stabilize_kd=2.0,
                 stabilize_angle=0.25, stabilize_speed=1.0):
        self.energy_gain = float(energy_gain)
        self.stabilize_kp = float(stabilize_kp)
        self.stabilize_kd = float(stabilize_kd)
        self.stabilize_angle = float(stabilize_angle)
        self.stabilize_speed = float(stabilize_speed)

    def step(self, env, obs, mass=1.0):
        cos_theta, sin_theta, theta_dot = np.asarray(obs, dtype=np.float64).reshape(3)
        theta = np.arctan2(sin_theta, cos_theta)

        if abs(theta) < self.stabilize_angle and abs(theta_dot) < self.stabilize_speed:
            torque = -(self.stabilize_kp * theta + self.stabilize_kd * theta_dot)
        else:
            energy = 0.5 * theta_dot * theta_dot + cos_theta
            energy_target = 1.0
            sign_term = np.sign(theta_dot * cos_theta) if abs(theta_dot) > 1e-6 else np.sign(sin_theta)
            if sign_term == 0.0:
                sign_term = 1.0
            torque = -self.energy_gain * sign_term * (energy - energy_target)

        torque = np.clip(torque, -2.0, 2.0)
        return np.array([torque], dtype=np.float32)


def collect_pendulum_episodes(env, teacher, mass_value, n_episodes=20, episode_length=200):
    episodes_X, episodes_Y = [], []
    for ep in range(n_episodes):
        obs, _ = env.reset()
        ep_X, ep_Y = [], []
        for _ in range(episode_length):
            action = teacher.step(env, obs, mass=mass_value)
            ep_X.append(build_pendulum_features(env, obs, mass_value))
            ep_Y.append(action.copy())
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break
        episodes_X.append(np.asarray(ep_X, dtype=np.float32))
        episodes_Y.append(np.asarray(ep_Y, dtype=np.float32))
        if (ep + 1) % 10 == 0:
            log.info("      Collected %d/%d episodes", ep + 1, n_episodes)
    return episodes_X, episodes_Y


class EchoStateNetwork:
    def __init__(self, n_inputs, n_reservoir, spectral_radius=0.95, input_scaling=1.0,
                 leak_rate=0.3, sparsity=0.1, seed=42):
        rng = np.random.RandomState(seed)
        self.n_reservoir = int(n_reservoir)
        self.leak = float(leak_rate)
        self.Win = rng.uniform(-input_scaling, input_scaling, (self.n_reservoir, n_inputs)).astype(np.float64)

        W = rng.uniform(-0.5, 0.5, (self.n_reservoir, self.n_reservoir))
        mask = rng.random((self.n_reservoir, self.n_reservoir)) < sparsity
        W *= mask
        eigvals = np.linalg.eigvals(W)
        max_abs = np.max(np.abs(eigvals))
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

    def collect_states(self, episodes_X, washout=WASHOUT):
        states, inputs, indices = [], [], []
        for ep_idx, ep_X in enumerate(episodes_X):
            self.reset()
            for t, x in enumerate(ep_X):
                s = self.update(x)
                if t >= washout:
                    states.append(s)
                    inputs.append(np.asarray(x, dtype=np.float64))
                    indices.append((ep_idx, t))
        return np.asarray(states, dtype=np.float64), np.asarray(inputs, dtype=np.float64), indices

    def _augment(self, H, X_in):
        bias = np.ones((len(H), 1), dtype=np.float64)
        return np.concatenate([bias, X_in, H], axis=1)

    def fit(self, episodes_X, episodes_Y, ridge=1e-4, washout=WASHOUT):
        H, X_in, indices = self.collect_states(episodes_X, washout=washout)
        H_aug = self._augment(H, X_in)
        Y_mat = np.asarray([episodes_Y[ep_idx][t] for ep_idx, t in indices], dtype=np.float64)
        A = H_aug.T @ H_aug + ridge * np.eye(H_aug.shape[1])
        b = H_aug.T @ Y_mat
        self.W_out = np.linalg.solve(A, b)
        self._ridge = ridge
        return self

    def predict(self, x):
        x = np.asarray(x, dtype=np.float64)
        s = self.update(x)
        v = np.concatenate([[1.0], x, s])
        return v @ self.W_out


class NumpyAutoencoder:
    def __init__(self, input_dim, latent_dim, hidden_dim=64, seed=42):
        rng = np.random.default_rng(seed)
        self.input_dim = int(input_dim)
        self.latent_dim = int(latent_dim)
        self.hidden_dim = int(hidden_dim)

        s1 = np.sqrt(2.0 / (self.input_dim + self.hidden_dim))
        s2 = np.sqrt(2.0 / (self.hidden_dim + self.latent_dim))

        self.W1 = rng.normal(0.0, s1, size=(self.input_dim, self.hidden_dim)).astype(np.float32)
        self.b1 = np.zeros((1, self.hidden_dim), dtype=np.float32)
        self.W2 = rng.normal(0.0, s2, size=(self.hidden_dim, self.latent_dim)).astype(np.float32)
        self.b2 = np.zeros((1, self.latent_dim), dtype=np.float32)
        self.W3 = rng.normal(0.0, s2, size=(self.latent_dim, self.hidden_dim)).astype(np.float32)
        self.b3 = np.zeros((1, self.hidden_dim), dtype=np.float32)
        self.W4 = rng.normal(0.0, s1, size=(self.hidden_dim, self.input_dim)).astype(np.float32)
        self.b4 = np.zeros((1, self.input_dim), dtype=np.float32)

    @staticmethod
    def _tanh(x):
        return np.tanh(x)

    @staticmethod
    def _dtanh(y):
        return 1.0 - y * y

    def _forward(self, x):
        h1 = self._tanh(x @ self.W1 + self.b1)
        z = self._tanh(h1 @ self.W2 + self.b2)
        h2 = self._tanh(z @ self.W3 + self.b3)
        recon = h2 @ self.W4 + self.b4
        return h1, z, h2, recon

    def fit(self, x, epochs=20, batch_size=512, lr=1e-3, weight_decay=1e-5, log_every=5):
        n = x.shape[0]
        for epoch in range(1, epochs + 1):
            perm = np.random.permutation(n)
            x_shuffled = x[perm]
            epoch_loss = 0.0
            batches = 0

            for start in range(0, n, batch_size):
                xb = x_shuffled[start:start + batch_size]
                if xb.shape[0] == 0:
                    continue

                h1, z, h2, recon = self._forward(xb)
                err = recon - xb
                epoch_loss += float(np.mean(err * err))
                batches += 1

                d_recon = (2.0 / xb.shape[0]) * err
                dW4 = h2.T @ d_recon + weight_decay * self.W4
                db4 = np.sum(d_recon, axis=0, keepdims=True)

                d_h2 = (d_recon @ self.W4.T) * self._dtanh(h2)
                dW3 = z.T @ d_h2 + weight_decay * self.W3
                db3 = np.sum(d_h2, axis=0, keepdims=True)

                d_z = (d_h2 @ self.W3.T) * self._dtanh(z)
                dW2 = h1.T @ d_z + weight_decay * self.W2
                db2 = np.sum(d_z, axis=0, keepdims=True)

                d_h1 = (d_z @ self.W2.T) * self._dtanh(h1)
                dW1 = xb.T @ d_h1 + weight_decay * self.W1
                db1 = np.sum(d_h1, axis=0, keepdims=True)

                self.W4 -= lr * dW4
                self.b4 -= lr * db4
                self.W3 -= lr * dW3
                self.b3 -= lr * db3
                self.W2 -= lr * dW2
                self.b2 -= lr * db2
                self.W1 -= lr * dW1
                self.b1 -= lr * db1

            mean_loss = epoch_loss / max(1, batches)
            if epoch % log_every == 0 or epoch == 1 or epoch == epochs:
                log.info("    AE epoch %d/%d mse=%.6f", epoch, epochs, mean_loss)

        return self

    def encode(self, x):
        h1 = self._tanh(x @ self.W1 + self.b1)
        return self._tanh(h1 @ self.W2 + self.b2)


def evaluate_config_mse(esn, val_episodes_X, val_episodes_Y, washout=WASHOUT):
    if esn.W_out is None:
        return float("inf")
    total_err, total_n = 0.0, 0
    for ep_X, ep_Y in zip(val_episodes_X, val_episodes_Y):
        esn.reset()
        for t, (x, y) in enumerate(zip(ep_X, ep_Y)):
            pred = esn.predict(x)
            if t >= washout:
                total_err += np.mean((pred - y.astype(np.float64)) ** 2)
                total_n += 1
    return total_err / max(total_n, 1)


def evaluate_teacher(env, teacher, mass_value, episode_length=200):
    obs, _ = env.reset()
    total_reward = 0.0
    upright_steps = 0

    for _ in range(episode_length):
        action = teacher.step(env, obs, mass=mass_value)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += float(reward)
        if abs(np.arctan2(obs[1], obs[0])) < 0.35:
            upright_steps += 1
        if terminated or truncated:
            break

    return total_reward, upright_steps / max(1, episode_length)


def evaluate_policy(env, esn, autoencoder, X_mean, X_std, Y_mean, Y_std, mass_value, episode_length=200):
    obs, _ = env.reset()
    esn.reset()
    total_reward = 0.0
    upright_steps = 0

    for _ in range(episode_length):
        obs_aug = build_pendulum_features(env, obs, mass_value).astype(np.float64)
        obs_norm = (obs_aug - X_mean) / X_std
        obs_input = autoencoder.encode(obs_norm.reshape(1, -1)).reshape(-1)
        action_norm = esn.predict(obs_input)
        action = np.clip((action_norm * Y_std + Y_mean).astype(np.float32), env.action_space.low, env.action_space.high)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += float(reward)
        if abs(np.arctan2(obs[1], obs[0])) < 0.35:
            upright_steps += 1
        if terminated or truncated:
            break

    return total_reward, upright_steps / max(1, episode_length)


def train_pendulum_rc(
    masses=MASS_VALUES,
    n_episodes=20,
    n_reservoir=800,
    ae_hidden_dim=64,
    ae_latent_dim=16,
    ae_epochs=20,
):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    log.info("=" * 70)
    log.info("DEDICATED RC FOR PENDULUM MASS VARIATIONS")
    log.info("=" * 70)
    log.info("Masses=%s | n_episodes=%d | n_reservoir=%d", masses, n_episodes, n_reservoir)

    teacher = PendulumSwingUpTeacher()
    all_episodes_X, all_episodes_Y, mass_per_ep = [], [], []
    teacher_stats = {}

    for mass in masses:
        log.info("\n[Mass %.2f]", mass)
        env = gym.make("Pendulum-v1", render_mode="rgb_array")
        set_pendulum_mass(env, mass)

        teacher_reward, teacher_upright = evaluate_teacher(env, teacher, mass)
        teacher_stats[float(mass)] = {
            "teacher_reward": float(teacher_reward),
            "teacher_upright": float(teacher_upright),
        }
        log.info("  Teacher reward=%.3f upright=%.1f%%", teacher_reward, teacher_upright * 100.0)

        eps_X, eps_Y = collect_pendulum_episodes(env, teacher, mass, n_episodes=n_episodes)
        for ep_X, ep_Y in zip(eps_X, eps_Y):
            all_episodes_X.append(ep_X)
            all_episodes_Y.append(ep_Y)
            mass_per_ep.append(float(mass))
        env.close()

    X_flat = np.vstack(all_episodes_X)
    Y_flat = np.vstack(all_episodes_Y)
    input_dim = X_flat.shape[1]
    output_dim = Y_flat.shape[1]

    X_mean = X_flat.mean(axis=0)
    X_std = X_flat.std(axis=0)
    X_std[X_std < 1e-8] = 1.0
    Y_mean = Y_flat.mean(axis=0)
    Y_std = Y_flat.std(axis=0)
    Y_std[Y_std < 1e-8] = 1.0

    norm_episodes_X = [(ep - X_mean) / X_std for ep in all_episodes_X]
    norm_episodes_Y = [(ep - Y_mean) / Y_std for ep in all_episodes_Y]
    X_norm_flat = (X_flat - X_mean) / X_std

    log.info("\nTotal episodes: %d | Input dim: %d | Output dim: %d", len(all_episodes_X), input_dim, output_dim)
    log.info("Training autoencoder (input=%d latent=%d, epochs=%d)...", input_dim, ae_latent_dim, ae_epochs)
    autoencoder = NumpyAutoencoder(input_dim=input_dim, latent_dim=ae_latent_dim, hidden_dim=ae_hidden_dim)
    autoencoder.fit(X_norm_flat, epochs=ae_epochs, batch_size=512, lr=1e-3, weight_decay=1e-5, log_every=max(1, ae_epochs // 5))
    encoded_episodes_X = [autoencoder.encode(ep) for ep in norm_episodes_X]

    mass_episode_indices = {mass: [i for i, m in enumerate(mass_per_ep) if m == mass] for mass in masses}
    for mass, idxs in mass_episode_indices.items():
        log.info("Mass %.2f episodes available: %d", mass, len(idxs))

    configs = [
        {"spectral_radius": 0.90, "input_scaling": 1.0, "leak_rate": 0.3, "ridge": 1e-4, "name": "SR90-L03-R1e4"},
        {"spectral_radius": 0.95, "input_scaling": 1.0, "leak_rate": 0.3, "ridge": 1e-3, "name": "SR95-L03-R1e3"},
        {"spectral_radius": 0.99, "input_scaling": 1.5, "leak_rate": 0.2, "ridge": 1e-3, "name": "SR99-L02-R1e3"},
        {"spectral_radius": 0.99, "input_scaling": 2.0, "leak_rate": 0.1, "ridge": 1e-2, "name": "SR99-L01-R1e2"},
        {"spectral_radius": 0.95, "input_scaling": 2.0, "leak_rate": 0.3, "ridge": 1e-2, "name": "SR95-L03-R1e2"},
        {"spectral_radius": 0.99, "input_scaling": 1.0, "leak_rate": 0.5, "ridge": 1e-3, "name": "SR99-L05-R1e3"},
    ]

    log.info("\nSearching hyperparameters (%d configs) per mass...", len(configs))
    mass_models = {}
    eval_results = []

    for mass in masses:
        idxs = mass_episode_indices[mass]
        split = max(1, int(len(idxs) * 0.8))
        train_idxs = idxs[:split]
        val_idxs = idxs[split:]
        train_X = [encoded_episodes_X[i] for i in train_idxs]
        train_Y = [norm_episodes_Y[i] for i in train_idxs]
        val_X = [encoded_episodes_X[i] for i in val_idxs] if val_idxs else list(train_X)
        val_Y = [norm_episodes_Y[i] for i in val_idxs] if val_idxs else list(train_Y)

        log.info("  Mass %.2f train episodes=%d val episodes=%d", mass, len(train_X), len(val_X))

        best_esn, best_cfg, best_mse = None, None, float("inf")
        for cfg in configs:
            log.info("    [%s]", cfg["name"])
            esn = EchoStateNetwork(
                n_inputs=ae_latent_dim,
                n_reservoir=n_reservoir,
                spectral_radius=cfg["spectral_radius"],
                input_scaling=cfg["input_scaling"],
                leak_rate=cfg["leak_rate"],
                sparsity=0.1,
                seed=42,
            )
            esn.fit(train_X, train_Y, ridge=cfg["ridge"], washout=WASHOUT)
            mse = evaluate_config_mse(esn, val_X, val_Y, washout=WASHOUT)
            log.info("      Val MSE: %.6f", mse)
            if mse < best_mse:
                best_mse = mse
                best_cfg = cfg
                best_esn = esn

        mass_models[float(mass)] = {"esn": best_esn, "config": best_cfg, "val_mse": best_mse}
        log.info("  Best config for mass %.2f: %s (val MSE=%.6f)", mass, best_cfg["name"], best_mse)

        env = gym.make("Pendulum-v1", render_mode="rgb_array")
        set_pendulum_mass(env, mass)
        rewards, upright = [], []
        for _ in range(5):
            reward, upright_ratio = evaluate_policy(
                env,
                best_esn,
                autoencoder,
                X_mean,
                X_std,
                Y_mean,
                Y_std,
                mass_value=float(mass),
            )
            rewards.append(reward)
            upright.append(upright_ratio)
        env.close()

        avg_reward = float(np.mean(rewards))
        avg_upright = float(np.mean(upright))
        eval_results.append({"mass": float(mass), "reward": avg_reward, "upright_ratio": avg_upright, "config": best_cfg})
        log.info("  Mass %.2f: RC reward=%.3f upright=%.1f%%", mass, avg_reward, avg_upright * 100.0)

    model_data = {
        "task": "pendulum_mass",
        "masses": masses,
        "mass_models": mass_models,
        "autoencoder": autoencoder,
        "X_mean": X_mean,
        "X_std": X_std,
        "Y_mean": Y_mean,
        "Y_std": Y_std,
        "config": {float(m): model_data["config"] for m, model_data in mass_models.items()},
        "teacher_stats": teacher_stats,
        "eval_results": eval_results,
    }

    model_path = os.path.join(OUTPUT_DIR, "pendulum_rc.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model_data, f)
    log.info("Saved model to: %s", model_path)

    summary_path = os.path.join(OUTPUT_DIR, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Dedicated Pendulum RC Training\n")
        f.write("=" * 70 + "\n")
        f.write(f"Mass values: {masses}\n")
        f.write(f"Model: {model_path}\n")
        f.write("Best configs by mass:\n")
        for mass, model_info in mass_models.items():
            f.write(f"  mass {mass}: {model_info['config']} (val MSE={model_info['val_mse']:.6f})\n")
        f.write("\n")
        f.write("Teacher stats:\n")
        for mass, stats in teacher_stats.items():
            f.write(f"  mass {mass}: reward={stats['teacher_reward']:.3f}, upright={stats['teacher_upright'] * 100.0:.1f}%\n")
        f.write("\nRC results:\n")
        for res in eval_results:
            f.write(f"  mass {res['mass']}: reward={res['reward']:.3f}, upright={res['upright_ratio'] * 100.0:.1f}%\n")

    log.info("Saved summary to: %s", summary_path)
    log.info("DONE")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-episodes", type=int, default=20)
    parser.add_argument("--n-reservoir", type=int, default=800)
    parser.add_argument("--ae-hidden-dim", type=int, default=64)
    parser.add_argument("--ae-latent-dim", type=int, default=16)
    parser.add_argument("--ae-epochs", type=int, default=20)
    args = parser.parse_args()

    train_pendulum_rc(
        n_episodes=args.n_episodes,
        n_reservoir=args.n_reservoir,
        ae_hidden_dim=args.ae_hidden_dim,
        ae_latent_dim=args.ae_latent_dim,
        ae_epochs=args.ae_epochs,
    )