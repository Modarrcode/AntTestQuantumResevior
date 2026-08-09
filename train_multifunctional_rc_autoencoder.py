"""
Train a multifunctional Reservoir Computing (RC) controller with an autoencoder bottleneck.

Multifunctional objective:
- Primary task: imitate CPG actions.
- Auxiliary task: predict next latent state (planning signal).

This follows the paper direction of using RC memory for control + state prediction,
and extends the existing multi-friction pipeline in this repository.
"""

import argparse
import logging
import os
import pickle
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from reservoirpy.model import Model
from reservoirpy.nodes import Reservoir, Ridge


logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


TRAIN_FRICTIONS = [0.5, 1.0, 1.5]
OUTPUT_DIR = "multifunctional_rc_ae_model"


def set_floor_friction(env, mu: float):
    """Set floor friction for MuJoCo geoms that match floor/ground names."""
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
    """Get ant base x position for forward-distance metrics."""
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

    def step(self, t: float):
        theta = self.omega * t + self.phases
        return self.offsets + self.amplitudes * np.sin(theta)


def random_search_tune(env, n_actions, n_iters=100, episode_length=500):
    """Tune CPG parameters by random search."""
    best_vec = None
    best_score = -float("inf")

    for i in range(n_iters):
        omega = np.random.uniform(0.5, 4.0)
        amps = np.random.uniform(0.0, 1.0, size=n_actions)
        phases = np.random.uniform(-np.pi, np.pi, size=n_actions)
        offsets = np.random.uniform(-0.5, 0.5, size=n_actions)

        vec = np.concatenate(([omega], amps, phases, offsets))
        cpg = CPGController.from_vector(vec, n_actions)

        forward, _, _ = evaluate_cpg(env, cpg, episode_length=episode_length)
        if forward > best_score:
            best_score = forward
            best_vec = vec.copy()
            if (i + 1) % 25 == 0 or i == 0:
                log.info("  CPG iter %d: best forward=%.3f", i + 1, best_score)

    return best_vec, best_score


def evaluate_cpg(env, cpg, episode_length=500):
    """Evaluate CPG controller."""
    obs, _ = env.reset()
    start_x = get_base_x(env)
    total_reward = 0.0

    for step in range(episode_length):
        action = np.asarray(cpg.step(step * 0.02), dtype=np.float32)
        action = np.clip(action, env.action_space.low, env.action_space.high)
        obs, reward, term, trunc, _ = env.step(action)
        total_reward += float(reward)
        if term or trunc:
            break

    return get_base_x(env) - start_x, total_reward, step + 1


@dataclass
class DatasetBundle:
    observations: np.ndarray
    next_observations: np.ndarray
    actions: np.ndarray
    frictions: np.ndarray
    cpg_stats: list


def load_or_tune_cpg_vector(env, friction: float, friction_idx: int, n_iters: int, cpg_dir: str):
    """Load cached CPG params if present; otherwise tune and cache."""
    os.makedirs(cpg_dir, exist_ok=True)
    cpg_path = os.path.join(cpg_dir, f"cpg_idx{friction_idx}.npy")

    if os.path.exists(cpg_path):
        vec = np.load(cpg_path)
        log.info("Loaded CPG params from %s", cpg_path)
        return vec

    log.info("No cached CPG for friction %.1f. Tuning...", friction)
    best_vec, best_score = random_search_tune(
        env,
        env.action_space.shape[0],
        n_iters=n_iters,
        episode_length=500,
    )
    np.save(cpg_path, best_vec)
    log.info("Saved CPG params to %s (forward %.3f)", cpg_path, best_score)
    return best_vec


def collect_multifriction_dataset(frictions, episodes_per_friction=15, min_forward=0.5, cpg_tune_iters=120):
    """Collect sequential demonstrations for all frictions."""
    all_obs = []
    all_next_obs = []
    all_actions = []
    all_frictions = []
    cpg_stats = []

    for idx, mu in enumerate(frictions):
        env = gym.make("Ant-v5", render_mode="rgb_array")
        set_floor_friction(env, mu)
        n_actions = env.action_space.shape[0]

        cpg_vec = load_or_tune_cpg_vector(env, mu, idx, cpg_tune_iters, cpg_dir="datasets")
        cpg = CPGController.from_vector(cpg_vec, n_actions)

        cpg_forward, cpg_reward, _ = evaluate_cpg(env, cpg, episode_length=500)
        cpg_stats.append({
            "friction": float(mu),
            "cpg_fwd": float(cpg_forward),
            "cpg_reward": float(cpg_reward),
        })
        log.info("Friction %.1f CPG: forward=%.3f reward=%.1f", mu, cpg_forward, cpg_reward)

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
                action = np.asarray(cpg.step(step * 0.02), dtype=np.float32)
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
            ep_next_obs = np.vstack([ep_obs[1:], ep_obs[-1:]])

            all_obs.append(ep_obs)
            all_next_obs.append(ep_next_obs)
            all_actions.append(ep_actions)
            all_frictions.append(np.full((len(ep_obs), 1), mu, dtype=np.float32))
            good_eps += 1

            if good_eps % 5 == 0:
                log.info("  Friction %.1f collected %d/%d good episodes", mu, good_eps, episodes_per_friction)

        env.close()
        log.info("Friction %.1f: %d good episodes from %d attempts", mu, good_eps, attempts)

    obs = np.vstack(all_obs).astype(np.float32)
    next_obs = np.vstack(all_next_obs).astype(np.float32)
    actions = np.vstack(all_actions).astype(np.float32)
    fric = np.vstack(all_frictions).astype(np.float32).reshape(-1)

    return DatasetBundle(obs, next_obs, actions, fric, cpg_stats)


class NumpyAutoencoder:
    """Small tanh autoencoder implemented in NumPy."""

    def __init__(self, input_dim: int, latent_dim: int, hidden_dim=64, seed=42):
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

    def fit(self, x, epochs=80, batch_size=512, lr=1e-3, weight_decay=1e-5, log_every=10):
        n = x.shape[0]
        losses = []

        for epoch in range(1, epochs + 1):
            perm = np.random.permutation(n)
            x_shuffled = x[perm]
            epoch_loss = 0.0
            batches = 0

            for start in range(0, n, batch_size):
                xb = x_shuffled[start:start + batch_size]
                bsz = xb.shape[0]
                if bsz == 0:
                    continue

                h1, z, h2, recon = self._forward(xb)
                err = recon - xb
                loss = np.mean(err * err)
                epoch_loss += float(loss)
                batches += 1

                d_recon = (2.0 / bsz) * err

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
                log.info("AE epoch %d/%d mse=%.6f", epoch, epochs, mean_loss)

        return losses

    def encode(self, x):
        h1 = self._tanh(x @ self.W1 + self.b1)
        z = self._tanh(h1 @ self.W2 + self.b2)
        return z

    def reconstruct(self, x):
        _, _, _, recon = self._forward(x)
        return recon

    def state_dict(self):
        return {
            "input_dim": self.input_dim,
            "latent_dim": self.latent_dim,
            "hidden_dim": self.hidden_dim,
            "W1": self.W1,
            "b1": self.b1,
            "W2": self.W2,
            "b2": self.b2,
            "W3": self.W3,
            "b3": self.b3,
            "W4": self.W4,
            "b4": self.b4,
        }


class MultifunctionalRCPolicy:
    """Inference wrapper for RC+AE model."""

    def __init__(
        self,
        rc_model,
        autoencoder: NumpyAutoencoder,
        obs_mean,
        obs_std,
        action_mean,
        action_std,
    ):
        self.rc_model = rc_model
        self.autoencoder = autoencoder
        self.obs_mean = obs_mean
        self.obs_std = obs_std
        self.action_mean = action_mean
        self.action_std = action_std

    def reset(self):
        self.rc_model.reset()

    def predict(self, obs, friction):
        obs_norm = (obs - self.obs_mean) / self.obs_std
        z = self.autoencoder.encode(obs_norm.reshape(1, -1)).reshape(-1)
        rc_in = np.concatenate([z, [friction]]).astype(np.float32)
        rc_out = self.rc_model.run(rc_in.reshape(1, -1)).reshape(-1)

        n_actions = self.action_mean.shape[0]
        act_norm = rc_out[:n_actions]
        action = act_norm * self.action_std + self.action_mean
        return action


def build_reservoir_weights(n_reservoir, n_inputs, spectral_radius=0.95, density=0.1, input_scaling=2.0, seed=42):
    """Create sparse reservoir matrices with desired spectral radius."""
    rng = np.random.default_rng(seed)

    win = rng.uniform(-input_scaling, input_scaling, size=(n_reservoir, n_inputs)).astype(np.float32)

    mask = rng.random((n_reservoir, n_reservoir)) < density
    w = rng.uniform(-0.5, 0.5, size=(n_reservoir, n_reservoir)).astype(np.float32)
    w *= mask.astype(np.float32)

    eigvals = np.linalg.eigvals(w)
    max_abs = float(np.max(np.abs(eigvals)))
    if max_abs > 1e-12:
        w *= spectral_radius / max_abs

    return win, w


def evaluate_policy(env, policy: MultifunctionalRCPolicy, friction: float, n_episodes=3, episode_length=500):
    """Evaluate policy distance and reward at fixed friction."""
    rewards = []
    distances = []

    for _ in range(n_episodes):
        policy.reset()
        set_floor_friction(env, friction)
        obs, _ = env.reset()
        start_x = get_base_x(env)
        ep_reward = 0.0

        for _ in range(episode_length):
            action = policy.predict(obs, friction)
            action = np.clip(action, env.action_space.low, env.action_space.high)
            obs, reward, term, trunc, _ = env.step(action)
            ep_reward += float(reward)
            if term or trunc:
                break

        rewards.append(ep_reward)
        distances.append(get_base_x(env) - start_x)

    return float(np.mean(rewards)), float(np.mean(distances))


def train_multifunctional_rc_autoencoder(
    frictions,
    episodes_per_friction=20,
    min_forward=0.5,
    cpg_tune_iters=200,
    latent_dim=48,
    ae_hidden_dim=128,
    ae_epochs=120,
    n_reservoir=1500,
):
    """End-to-end training for multifunctional RC + autoencoder."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    log.info("=" * 70)
    log.info("MULTIFUNCTIONAL RC + AUTOENCODER TRAINING")
    log.info("=" * 70)

    data = collect_multifriction_dataset(
        frictions=frictions,
        episodes_per_friction=episodes_per_friction,
        min_forward=min_forward,
        cpg_tune_iters=cpg_tune_iters,
    )

    log.info(
        "Collected dataset: obs=%s next_obs=%s actions=%s",
        data.observations.shape,
        data.next_observations.shape,
        data.actions.shape,
    )

    obs_mean = data.observations.mean(axis=0)
    obs_std = data.observations.std(axis=0)
    obs_std[obs_std < 1e-8] = 1.0

    obs_norm = (data.observations - obs_mean) / obs_std
    next_obs_norm = (data.next_observations - obs_mean) / obs_std

    log.info("Training autoencoder (input=%d latent=%d)", obs_norm.shape[1], latent_dim)
    ae = NumpyAutoencoder(
        input_dim=obs_norm.shape[1],
        latent_dim=latent_dim,
        hidden_dim=ae_hidden_dim,
        seed=42,
    )
    ae_losses = ae.fit(obs_norm, epochs=ae_epochs, batch_size=512, lr=1e-3, weight_decay=1e-5)

    z_t = ae.encode(obs_norm)
    z_next = ae.encode(next_obs_norm)

    action_mean = data.actions.mean(axis=0)
    action_std = data.actions.std(axis=0)
    action_std[action_std < 1e-8] = 1.0
    action_norm = (data.actions - action_mean) / action_std

    rc_inputs = np.concatenate([z_t, data.frictions.reshape(-1, 1)], axis=1)
    rc_targets = np.concatenate([action_norm, z_next], axis=1)

    log.info("RC inputs shape: %s", rc_inputs.shape)
    log.info("RC targets shape (actions + next-latent): %s", rc_targets.shape)

    win, w = build_reservoir_weights(
        n_reservoir=n_reservoir,
        n_inputs=rc_inputs.shape[1],
        spectral_radius=0.92,
        density=0.12,
        input_scaling=2.5,
        seed=42,
    )

    reservoir = Reservoir(units=n_reservoir, input_dim=rc_inputs.shape[1], Win=win, W=w)
    readout = Ridge(input_dim=n_reservoir, output_dim=rc_targets.shape[1], ridge=1e-6)
    rc_model = Model([reservoir, readout], edges=[(reservoir, 0, readout)])

    log.info("Training multifunctional RC...")
    rc_model.fit(rc_inputs, rc_targets)
    log.info("RC training complete")

    policy = MultifunctionalRCPolicy(
        rc_model=rc_model,
        autoencoder=ae,
        obs_mean=obs_mean,
        obs_std=obs_std,
        action_mean=action_mean,
        action_std=action_std,
    )

    eval_results = []
    for mu in frictions:
        env = gym.make("Ant-v5", render_mode="rgb_array")
        set_floor_friction(env, mu)
        rew, dist = evaluate_policy(env, policy, friction=mu, n_episodes=3, episode_length=500)
        env.close()
        eval_results.append({"friction": float(mu), "rc_reward": rew, "rc_fwd": dist})
        log.info("Eval friction %.1f -> forward=%.3f reward=%.1f", mu, dist, rew)

    model_data = {
        "type": "multifunctional_rc_autoencoder",
        "frictions": list(frictions),
        "cpg_stats": data.cpg_stats,
        "eval_results": eval_results,
        "rc_model": rc_model,
        "autoencoder": ae.state_dict(),
        "obs_mean": obs_mean,
        "obs_std": obs_std,
        "action_mean": action_mean,
        "action_std": action_std,
        "latent_dim": latent_dim,
        "ae_hidden_dim": ae_hidden_dim,
        "ae_losses": ae_losses,
        "config": {
            "episodes_per_friction": episodes_per_friction,
            "min_forward": min_forward,
            "cpg_tune_iters": cpg_tune_iters,
            "ae_epochs": ae_epochs,
            "n_reservoir": n_reservoir,
        },
    }

    model_path = os.path.join(OUTPUT_DIR, "multifunctional_rc_autoencoder.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model_data, f)

    summary_path = os.path.join(OUTPUT_DIR, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Multifunctional RC + Autoencoder Summary\n")
        f.write("=" * 70 + "\n")
        f.write(f"Model path: {model_path}\n")
        f.write(f"Latent dim: {latent_dim}\n")
        f.write(f"Reservoir units: {n_reservoir}\n")
        f.write(f"AE final loss: {ae_losses[-1]:.6f}\n\n")

        f.write("CPG Baseline Stats:\n")
        for stat in data.cpg_stats:
            f.write(
                f"  friction {stat['friction']:.1f}: "
                f"cpg_fwd={stat['cpg_fwd']:.3f}, cpg_reward={stat['cpg_reward']:.1f}\n"
            )

        f.write("\nRC Evaluation Stats:\n")
        for res in eval_results:
            f.write(
                f"  friction {res['friction']:.1f}: "
                f"rc_fwd={res['rc_fwd']:.3f}, rc_reward={res['rc_reward']:.1f}\n"
            )

    log.info("Saved model to: %s", model_path)
    log.info("Saved summary to: %s", summary_path)

    return model_path, summary_path


def parse_args():
    parser = argparse.ArgumentParser(description="Train multifunctional RC with an autoencoder bottleneck")
    parser.add_argument("--episodes-per-friction", type=int, default=20)
    parser.add_argument("--min-forward", type=float, default=0.5)
    parser.add_argument("--cpg-tune-iters", type=int, default=200)
    parser.add_argument("--latent-dim", type=int, default=48)
    parser.add_argument("--ae-hidden-dim", type=int, default=128)
    parser.add_argument("--ae-epochs", type=int, default=120)
    parser.add_argument("--n-reservoir", type=int, default=1500)
    return parser.parse_args()


def main():
    args = parse_args()
    train_multifunctional_rc_autoencoder(
        frictions=TRAIN_FRICTIONS,
        episodes_per_friction=args.episodes_per_friction,
        min_forward=args.min_forward,
        cpg_tune_iters=args.cpg_tune_iters,
        latent_dim=args.latent_dim,
        ae_hidden_dim=args.ae_hidden_dim,
        ae_epochs=args.ae_epochs,
        n_reservoir=args.n_reservoir,
    )


if __name__ == "__main__":
    main()
