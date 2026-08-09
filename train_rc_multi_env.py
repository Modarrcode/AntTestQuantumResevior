"""
Generalized reservoir-computing training for Gymnasium locomotion and pendulum tasks.

Supported tasks:
- swimmer: Swimmer-v5 with friction-conditioned CPG imitation
- cheetah: HalfCheetah-v5 with friction-conditioned CPG imitation
- pendulum: Pendulum-v1 with mass-conditioned swing-up imitation

The locomotion tasks reuse the same reservoir + autoencoder framework as the
optimized Ant training, but with environment-specific action dimensions and
condition labels. The pendulum task treats the payload weight as a domain
parameter and trains on multiple masses.
"""

import argparse
import logging
import os
import pickle
from dataclasses import dataclass

import gymnasium as gym
import numpy as np


logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


WASHOUT = 50


@dataclass(frozen=True)
class TaskSpec:
    name: str
    env_id: str
    output_dir: str
    condition_name: str
    condition_values: tuple
    episode_length: int
    mode: str


TASK_SPECS = {
    "swimmer": TaskSpec(
        name="swimmer",
        env_id="Swimmer-v5",
        output_dir="swimmer_rc_model",
        condition_name="friction",
        condition_values=(0.5, 1.0, 1.5),
        episode_length=400,
        mode="locomotion",
    ),
    "cheetah": TaskSpec(
        name="cheetah",
        env_id="HalfCheetah-v5",
        output_dir="halfcheetah_rc_model",
        condition_name="friction",
        condition_values=(0.5, 1.0, 1.5),
        episode_length=500,
        mode="locomotion",
    ),
    "pendulum": TaskSpec(
        name="pendulum",
        env_id="Pendulum-v1",
        output_dir="pendulum_mass_rc_model",
        condition_name="mass",
        condition_values=(0.5, 1.0, 1.5, 2.0),
        episode_length=200,
        mode="pendulum",
    ),
}


def set_floor_friction(env, mu: float):
    """Set floor/ground friction on MuJoCo geoms when available."""
    try:
        model = env.unwrapped.model
    except Exception:
        return

    def geom_name_at(i):
        if hasattr(model, "geom_names"):
            try:
                name = model.geom_names[i]
                return name.decode("utf-8") if isinstance(name, bytes) else str(name)
            except Exception:
                pass
        if hasattr(model, "geom"):
            try:
                geom = model.geom[i]
                name = getattr(geom, "name", None)
                if name is None:
                    try:
                        name = geom[0]
                    except Exception:
                        name = str(geom)
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


def set_pendulum_mass(env, mass: float):
    """Set the payload mass of Pendulum-v1 when supported by the environment."""
    try:
        unwrapped = env.unwrapped
    except Exception:
        return

    for attr in ("m", "mass"):
        if hasattr(unwrapped, attr):
            try:
                setattr(unwrapped, attr, float(mass))
            except Exception:
                pass


def build_pendulum_features(env, obs, mass: float):
    """Build a compact feature vector for Pendulum-v1."""
    cos_theta, sin_theta, theta_dot = np.asarray(obs, dtype=np.float64).reshape(3)
    theta = np.arctan2(sin_theta, cos_theta)
    unwrapped = env.unwrapped
    length = float(getattr(unwrapped, "l", 1.0))
    gravity = float(getattr(unwrapped, "g", 10.0))
    effective_mass = float(getattr(unwrapped, "m", mass))
    current_energy = 0.5 * effective_mass * (length * theta_dot) ** 2 + effective_mass * gravity * length * (1.0 - cos_theta)
    target_energy = effective_mass * gravity * length
    energy_error = target_energy - current_energy
    return np.array([cos_theta, sin_theta, theta, theta_dot, energy_error, mass], dtype=np.float32)


def get_forward_position(env) -> float:
    try:
        return float(env.unwrapped.data.qpos[0])
    except Exception:
        return 0.0


class CPGController:
    def __init__(self, n_actions, omega=2.0, amplitudes=None, phases=None, offsets=None):
        self.n = int(n_actions)
        self.omega = float(omega)
        self.amplitudes = np.ones(self.n) if amplitudes is None else np.asarray(amplitudes).reshape(self.n)
        self.phases = np.zeros(self.n) if phases is None else np.asarray(phases).reshape(self.n)
        self.offsets = np.zeros(self.n) if offsets is None else np.asarray(offsets).reshape(self.n)

    @classmethod
    def from_vector(cls, vec, n_actions):
        vec = np.asarray(vec, dtype=np.float64)
        return cls(
            n_actions,
            float(vec[0]),
            vec[1:1 + n_actions],
            vec[1 + n_actions:1 + 2 * n_actions],
            vec[1 + 2 * n_actions:1 + 3 * n_actions],
        )

    def step(self, t):
        return self.offsets + self.amplitudes * np.sin(self.omega * t + self.phases)


class PendulumSwingUpTeacher:
    """Simple mass-aware swing-up and stabilization teacher for Pendulum-v1."""

    def __init__(self, max_torque=2.0, kp=9.0, kd=1.6, swing_gain=0.06, stabilize_angle=0.35, stabilize_speed=1.5):
        self.max_torque = float(max_torque)
        self.kp = float(kp)
        self.kd = float(kd)
        self.swing_gain = float(swing_gain)
        self.stabilize_angle = float(stabilize_angle)
        self.stabilize_speed = float(stabilize_speed)

    def step(self, env, obs, mass=1.0):
        cos_theta, sin_theta, theta_dot = np.asarray(obs, dtype=np.float64).reshape(3)
        theta = np.arctan2(sin_theta, cos_theta)
        unwrapped = env.unwrapped
        length = float(getattr(unwrapped, "l", 1.0))
        gravity = float(getattr(unwrapped, "g", 10.0))
        effective_mass = float(getattr(unwrapped, "m", mass))

        current_energy = 0.5 * effective_mass * (length * theta_dot) ** 2 + effective_mass * gravity * length * (1.0 - cos_theta)
        target_energy = effective_mass * gravity * length
        energy_error = target_energy - current_energy

        if abs(theta) < self.stabilize_angle and abs(theta_dot) < self.stabilize_speed:
            torque = -(self.kp * theta + self.kd * theta_dot)
        else:
            direction = np.sign(theta_dot * cos_theta)
            if direction == 0.0:
                direction = np.sign(np.sin(theta))
            torque = self.swing_gain * energy_error * direction - 0.25 * theta_dot

        return np.array([np.clip(torque, -self.max_torque, self.max_torque)], dtype=np.float32)


def tune_cpg(env, n_actions, cpg_iters=300, episode_length=500):
    best_dist, best_vec = -float("inf"), None

    for i in range(cpg_iters):
        omega = np.random.uniform(0.5, 4.0)
        amps = np.random.uniform(0.0, 1.0, size=n_actions)
        phases = np.random.uniform(-np.pi, np.pi, size=n_actions)
        offsets = np.random.uniform(-0.5, 0.5, size=n_actions)
        vec = np.concatenate(([omega], amps, phases, offsets))
        cpg = CPGController.from_vector(vec, n_actions)
        dist, _, _ = evaluate_locomotion_controller(env, cpg, episode_length=episode_length)
        if dist > best_dist:
            best_dist, best_vec = dist, vec.copy()
        if (i + 1) % 100 == 0 or i == 0:
            log.info("    CPG tune iter %d: best_forward=%.3f", i + 1, best_dist)

    return CPGController.from_vector(best_vec, n_actions), best_dist


def evaluate_locomotion_controller(env, controller, episode_length=500):
    obs, _ = env.reset()
    start_x = get_forward_position(env)
    total_reward = 0.0

    for step in range(episode_length):
        action = np.asarray(controller.step(step * 0.02), dtype=np.float32)
        action = np.clip(action, env.action_space.low, env.action_space.high)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += float(reward)
        if terminated or truncated:
            break

    return get_forward_position(env) - start_x, total_reward, step + 1


def collect_locomotion_episodes(env, cpg, n_episodes=30, min_forward=0.5, episode_length=500, condition_value=1.0):
    episodes_X = []
    episodes_Y = []
    good, attempts = 0, 0
    max_attempts = n_episodes * 5

    while good < n_episodes and attempts < max_attempts:
        attempts += 1
        obs, _ = env.reset()
        prev_obs = obs.copy()
        start_x = get_forward_position(env)
        ep_X, ep_Y = [], []

        for step in range(episode_length):
            t_sec = step * 0.02
            action = np.asarray(cpg.step(t_sec), dtype=np.float32)
            action = np.clip(action, env.action_space.low, env.action_space.high)
            vel = obs - prev_obs
            cpg_phase = (cpg.omega * t_sec + cpg.phases).astype(np.float32)
            cond = np.array([condition_value], dtype=np.float32)
            ep_X.append(np.concatenate([obs, vel, cpg_phase, action, cond]).astype(np.float32))
            ep_Y.append(action.copy())
            prev_obs = obs.copy()
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break

        forward = get_forward_position(env) - start_x
        if forward >= min_forward:
            episodes_X.append(np.array(ep_X, dtype=np.float32))
            episodes_Y.append(np.array(ep_Y, dtype=np.float32))
            good += 1
            if good % 10 == 0:
                log.info("      Collected %d/%d good episodes", good, n_episodes)

    log.info("      Final: %d good episodes from %d attempts", good, attempts)
    return episodes_X, episodes_Y


def collect_pendulum_episodes(env, teacher, n_episodes=30, episode_length=200, mass_value=1.0):
    episodes_X = []
    episodes_Y = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        prev_obs = obs.copy()
        ep_X, ep_Y = [], []

        for step in range(episode_length):
            action = teacher.step(env, obs, mass=mass_value)
            ep_X.append(build_pendulum_features(env, obs, mass_value))
            ep_Y.append(action.copy())
            prev_obs = obs.copy()
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break

        episodes_X.append(np.array(ep_X, dtype=np.float32))
        episodes_Y.append(np.array(ep_Y, dtype=np.float32))

        if (ep + 1) % 10 == 0:
            log.info("      Collected %d/%d episodes", ep + 1, n_episodes)

    return episodes_X, episodes_Y


class EchoStateNetwork:
    def __init__(self, n_inputs, n_reservoir, spectral_radius=0.95, input_scaling=1.0, leak_rate=0.3, sparsity=0.1, seed=42):
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
        self.state = (1 - self.leak) * self.state + self.leak * np.tanh(pre)
        return self.state.copy()

    def collect_states(self, episodes_X, washout=WASHOUT):
        all_states = []
        all_inputs = []
        all_indices = []
        for ep_idx, ep_X in enumerate(episodes_X):
            self.reset()
            for t, x in enumerate(ep_X):
                s = self.update(x)
                if t >= washout:
                    all_states.append(s)
                    all_inputs.append(np.asarray(x, dtype=np.float64))
                    all_indices.append((ep_idx, t))
        return np.array(all_states, dtype=np.float64), np.array(all_inputs, dtype=np.float64), all_indices

    def _augment(self, H, X_in):
        bias = np.ones((len(H), 1), dtype=np.float64)
        return np.concatenate([bias, X_in, H], axis=1)

    def fit(self, episodes_X, episodes_Y, ridge=1e-4, washout=WASHOUT):
        H, X_in, indices = self.collect_states(episodes_X, washout=washout)
        H_aug = self._augment(H, X_in)
        Y_mat = np.array([episodes_Y[ep_idx][t] for ep_idx, t in indices], dtype=np.float64)
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
    def __init__(self, input_dim, latent_dim, hidden_dim=128, seed=42):
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

    def fit(self, x, epochs=60, batch_size=512, lr=1e-3, weight_decay=1e-5, log_every=10):
        n = x.shape[0]
        losses = []

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
                loss = np.mean(err * err)
                epoch_loss += float(loss)
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
            losses.append(mean_loss)
            if epoch % log_every == 0 or epoch == 1 or epoch == epochs:
                log.info("    AE epoch %d/%d mse=%.6f", epoch, epochs, mean_loss)

        return losses

    def encode(self, x):
        h1 = self._tanh(x @ self.W1 + self.b1)
        z = self._tanh(h1 @ self.W2 + self.b2)
        return z


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


def split_train_val(episodes_X, episodes_Y, conditions):
    train_X, train_Y, val_X, val_Y = [], [], [], []
    for cond in sorted(set(conditions)):
        idxs = [i for i, c in enumerate(conditions) if c == cond]
        split = max(1, int(len(idxs) * 0.8))
        for i in idxs[:split]:
            train_X.append(episodes_X[i])
            train_Y.append(episodes_Y[i])
        for i in idxs[split:]:
            val_X.append(episodes_X[i])
            val_Y.append(episodes_Y[i])
    return train_X, train_Y, val_X, val_Y


def evaluate_locomotion_policy(env, esn, cpg, autoencoder, X_mean, X_std, Y_mean, Y_std, condition_value, episode_length=500):
    obs, _ = env.reset()
    esn.reset()
    start_x = get_forward_position(env)
    total_reward = 0.0
    prev_obs = obs.copy()

    for step in range(episode_length):
        vel = obs - prev_obs
        t_sec = step * 0.02
        cpg_phase = (cpg.omega * t_sec + cpg.phases).astype(np.float64)
        cpg_out = np.clip(cpg.step(t_sec).astype(np.float64), env.action_space.low, env.action_space.high)
        obs_aug = np.concatenate([obs, vel, cpg_phase, cpg_out, [condition_value]]).astype(np.float64)
        obs_norm = (obs_aug - X_mean) / X_std
        obs_input = autoencoder.encode(obs_norm.reshape(1, -1)).reshape(-1)
        action_norm = esn.predict(obs_input)
        action = np.clip((action_norm * Y_std + Y_mean).astype(np.float32), env.action_space.low, env.action_space.high)

        prev_obs = obs.copy()
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += float(reward)
        if terminated or truncated:
            break

    return get_forward_position(env) - start_x, total_reward


def evaluate_pendulum_policy(env, esn, autoencoder, X_mean, X_std, Y_mean, Y_std, mass_value, episode_length=200):
    obs, _ = env.reset()
    esn.reset()
    prev_obs = obs.copy()
    total_reward = 0.0
    upright_steps = 0

    for _ in range(episode_length):
        obs_aug = build_pendulum_features(env, obs, mass_value).astype(np.float64)
        obs_norm = (obs_aug - X_mean) / X_std
        obs_input = autoencoder.encode(obs_norm.reshape(1, -1)).reshape(-1)
        action_norm = esn.predict(obs_input)
        action = np.clip((action_norm * Y_std + Y_mean).astype(np.float32), env.action_space.low, env.action_space.high)

        prev_obs = obs.copy()
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += float(reward)
        if abs(np.arctan2(obs[1], obs[0])) < 0.35:
            upright_steps += 1
        if terminated or truncated:
            break

    upright_ratio = upright_steps / max(1, episode_length)
    return total_reward, upright_ratio


def train_task(task_name, n_episodes=30, n_reservoir=1200, cpg_iters=300, ae_hidden_dim=128, ae_latent_dim=64, ae_epochs=60, ridge=1e-3):
    if task_name not in TASK_SPECS:
        raise ValueError(f"Unknown task: {task_name}")

    spec = TASK_SPECS[task_name]
    os.makedirs(spec.output_dir, exist_ok=True)

    log.info("=" * 70)
    log.info("GENERALIZED RC TRAINING: %s", spec.name.upper())
    log.info("=" * 70)
    log.info("Env=%s | condition=%s | values=%s", spec.env_id, spec.condition_name, spec.condition_values)

    all_episodes_X = []
    all_episodes_Y = []
    condition_per_ep = []
    teacher_per_condition = {}
    baseline_stats = {}

    for condition_value in spec.condition_values:
        log.info("\n[%s %.2f]", spec.condition_name.capitalize(), condition_value)
        env = gym.make(spec.env_id, render_mode="rgb_array")

        if spec.mode == "locomotion":
            set_floor_friction(env, float(condition_value))
            n_actions = env.action_space.shape[0]
            log.info("  Tuning CPG (%d iterations)...", cpg_iters)
            cpg, cpg_forward = tune_cpg(env, n_actions, cpg_iters=cpg_iters, episode_length=spec.episode_length)
            teacher_per_condition[condition_value] = cpg
            baseline_stats[condition_value] = {"teacher_forward": float(cpg_forward)}
            log.info("  Teacher forward=%.3f", cpg_forward)

            log.info("  Collecting %d episodes...", n_episodes)
            eps_X, eps_Y = collect_locomotion_episodes(
                env,
                cpg,
                n_episodes=n_episodes,
                min_forward=0.0,
                episode_length=spec.episode_length,
                condition_value=float(condition_value),
            )
        else:
            set_pendulum_mass(env, float(condition_value))
            teacher = PendulumSwingUpTeacher()
            teacher_per_condition[condition_value] = teacher
            baseline_stats[condition_value] = {}
            log.info("  Collecting %d episodes...", n_episodes)
            eps_X, eps_Y = collect_pendulum_episodes(
                env,
                teacher,
                n_episodes=n_episodes,
                episode_length=spec.episode_length,
                mass_value=float(condition_value),
            )

        for ep_X, ep_Y in zip(eps_X, eps_Y):
            all_episodes_X.append(ep_X)
            all_episodes_Y.append(ep_Y)
            condition_per_ep.append(float(condition_value))

        log.info("  Collected %d episodes (%d total steps)", len(eps_X), sum(len(e) for e in eps_X))
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

    log.info("\nTotal episodes: %d | Input dim: %d | Output dim: %d", len(all_episodes_X), input_dim, output_dim)

    log.info("Training autoencoder (input=%d latent=%d, epochs=%d)...", input_dim, ae_latent_dim, ae_epochs)
    autoencoder = NumpyAutoencoder(input_dim=input_dim, latent_dim=ae_latent_dim, hidden_dim=ae_hidden_dim)
    X_norm_flat = (X_flat - X_mean) / X_std
    autoencoder.fit(X_norm_flat, epochs=ae_epochs, batch_size=512, lr=1e-3, weight_decay=1e-5, log_every=max(1, ae_epochs // 10))
    encoded_episodes_X = [autoencoder.encode(ep) for ep in norm_episodes_X]

    train_X, train_Y, val_X, val_Y = split_train_val(encoded_episodes_X, norm_episodes_Y, condition_per_ep)
    log.info("Train episodes: %d | Val episodes: %d", len(train_X), len(val_X))

    configs = [
        {"spectral_radius": 0.90, "input_scaling": 1.0, "leak_rate": 0.3, "ridge": 1e-4, "name": "SR90-L03-R1e4"},
        {"spectral_radius": 0.95, "input_scaling": 1.0, "leak_rate": 0.3, "ridge": 1e-3, "name": "SR95-L03-R1e3"},
        {"spectral_radius": 0.99, "input_scaling": 1.5, "leak_rate": 0.2, "ridge": 1e-3, "name": "SR99-L02-R1e3"},
        {"spectral_radius": 0.99, "input_scaling": 2.0, "leak_rate": 0.1, "ridge": 1e-2, "name": "SR99-L01-R1e2"},
        {"spectral_radius": 0.95, "input_scaling": 2.0, "leak_rate": 0.3, "ridge": 1e-2, "name": "SR95-L03-R1e2"},
        {"spectral_radius": 0.99, "input_scaling": 1.0, "leak_rate": 0.5, "ridge": 1e-3, "name": "SR99-L05-R1e3"},
    ]

    log.info("\nSearching hyperparameters (%d configs)...", len(configs))
    best_esn, best_cfg, best_mse = None, None, float("inf")

    for cfg in configs:
        log.info("  [%s]", cfg["name"])
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
        log.info("    Val MSE: %.6f", mse)

        if mse < best_mse:
            best_mse = mse
            best_cfg = cfg
            best_esn = esn

    log.info("\nBest config: %s (val MSE=%.6f)", best_cfg["name"], best_mse)

    eval_results = []
    if spec.mode == "locomotion":
        log.info("\nEvaluating locomotion policy across conditions...")
        for condition_value in spec.condition_values:
            env = gym.make(spec.env_id, render_mode="rgb_array")
            set_floor_friction(env, float(condition_value))
            cpg = teacher_per_condition[condition_value]

            forward_scores = []
            rewards = []
            for _ in range(5):
                forward, reward = evaluate_locomotion_policy(
                    env,
                    best_esn,
                    cpg,
                    autoencoder,
                    X_mean,
                    X_std,
                    Y_mean,
                    Y_std,
                    float(condition_value),
                    episode_length=spec.episode_length,
                )
                forward_scores.append(forward)
                rewards.append(reward)

            avg_forward = float(np.mean(forward_scores))
            avg_reward = float(np.mean(rewards))
            teacher_forward = baseline_stats[condition_value]["teacher_forward"]
            alignment = (avg_forward / teacher_forward * 100.0) if abs(teacher_forward) > 1e-8 else 0.0

            eval_results.append({
                spec.condition_name: float(condition_value),
                "rc_forward": avg_forward,
                "rc_reward": avg_reward,
                "teacher_forward": teacher_forward,
                "alignment_pct": alignment,
            })
            log.info("  %s %.2f: RC=%.3f, Teacher=%.3f, Alignment=%.1f%%", spec.condition_name.capitalize(), condition_value, avg_forward, teacher_forward, alignment)
            env.close()
    else:
        log.info("\nEvaluating pendulum policy across masses...")
        for condition_value in spec.condition_values:
            env = gym.make(spec.env_id, render_mode="rgb_array")
            set_pendulum_mass(env, float(condition_value))

            rewards = []
            upright = []
            for _ in range(5):
                reward, upright_ratio = evaluate_pendulum_policy(
                    env,
                    best_esn,
                    autoencoder,
                    X_mean,
                    X_std,
                    Y_mean,
                    Y_std,
                    float(condition_value),
                    episode_length=spec.episode_length,
                )
                rewards.append(reward)
                upright.append(upright_ratio)

            avg_reward = float(np.mean(rewards))
            avg_upright = float(np.mean(upright))
            eval_results.append({
                spec.condition_name: float(condition_value),
                "rc_reward": avg_reward,
                "upright_ratio": avg_upright,
            })
            log.info("  Mass %.2f: RC reward=%.3f, upright=%.1f%%", condition_value, avg_reward, avg_upright * 100.0)
            env.close()

    model_data = {
        "task": spec.name,
        "env_id": spec.env_id,
        "condition_name": spec.condition_name,
        "condition_values": spec.condition_values,
        "esn": best_esn,
        "autoencoder": autoencoder,
        "X_mean": X_mean,
        "X_std": X_std,
        "Y_mean": Y_mean,
        "Y_std": Y_std,
        "config": best_cfg,
        "eval_results": eval_results,
        "teacher_per_condition": teacher_per_condition,
    }

    pkl_path = os.path.join(spec.output_dir, f"{spec.name}_rc.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(model_data, f)
    log.info("Saved model to: %s", pkl_path)

    summary_path = os.path.join(spec.output_dir, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"Generalized RC training for {spec.name}\n")
        f.write("=" * 70 + "\n")
        f.write(f"Env: {spec.env_id}\n")
        f.write(f"Condition: {spec.condition_name}\n")
        f.write(f"Condition values: {spec.condition_values}\n")
        f.write(f"Model: {pkl_path}\n")
        f.write(f"Best config: {best_cfg}\n\n")
        f.write("Results:\n")
        for res in eval_results:
            if spec.mode == "locomotion":
                f.write(
                    f"  {spec.condition_name} {res[spec.condition_name]}: RC={res['rc_forward']:.3f}, "
                    f"Teacher={res['teacher_forward']:.3f}, Alignment={res['alignment_pct']:.1f}%\n"
                )
            else:
                f.write(
                    f"  mass {res[spec.condition_name]}: reward={res['rc_reward']:.3f}, "
                    f"upright={res['upright_ratio'] * 100.0:.1f}%\n"
                )

    log.info("Saved summary to: %s", summary_path)
    log.info("DONE")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=sorted(TASK_SPECS.keys()), required=True)
    parser.add_argument("--n-episodes", type=int, default=30)
    parser.add_argument("--n-reservoir", type=int, default=1200)
    parser.add_argument("--cpg-iters", type=int, default=300)
    parser.add_argument("--ae-hidden-dim", type=int, default=128)
    parser.add_argument("--ae-latent-dim", type=int, default=64)
    parser.add_argument("--ae-epochs", type=int, default=60)
    parser.add_argument("--ridge", type=float, default=1e-3)
    args = parser.parse_args()

    train_task(
        task_name=args.task,
        n_episodes=args.n_episodes,
        n_reservoir=args.n_reservoir,
        cpg_iters=args.cpg_iters,
        ae_hidden_dim=args.ae_hidden_dim,
        ae_latent_dim=args.ae_latent_dim,
        ae_epochs=args.ae_epochs,
        ridge=args.ridge,
    )