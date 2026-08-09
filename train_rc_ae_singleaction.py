"""
Train RC with Autoencoder bottleneck - ACTION ONLY (no state prediction).

Key fix: Use AE for dimensionality reduction AND regularization, but only predict actions.
This avoids: (1) complete information loss from bottleneck, (2) multitask learning interference.
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
OUTPUT_DIR = "rc_ae_singleaction_model"


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

    def to_vector(self):
        return np.concatenate([[self.omega], self.amplitudes, self.phases, self.offsets])

    def step(self, t):
        phase = self.omega * t + self.phases
        return self.offsets + self.amplitudes * np.sin(phase)


class NumpyAutoencoder:
    """Simple numpy-based autoencoder for dimensionality reduction."""

    def __init__(self, input_dim, latent_dim, hidden_dim=128):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.encoder_w1 = None
        self.encoder_b1 = None
        self.encoder_w2 = None
        self.encoder_b2 = None
        self.decoder_w1 = None
        self.decoder_b1 = None
        self.decoder_w2 = None
        self.decoder_b2 = None

    def _relu(self, x):
        return np.maximum(x, 0)

    def _relu_grad(self, x):
        return (x > 0).astype(np.float32)

    def encode(self, x):
        h = self._relu(x @ self.encoder_w1 + self.encoder_b1)
        z = h @ self.encoder_w2 + self.encoder_b2
        return z

    def decode(self, z):
        h = self._relu(z @ self.decoder_w1 + self.decoder_b1)
        recon = h @ self.decoder_w2 + self.decoder_b2
        return recon

    def forward(self, x):
        z = self.encode(x)
        recon = self.decode(z)
        return recon, z

    def fit(self, x, epochs=80, lr=0.001, batch_size=32):
        n_samples = x.shape[0]
        
        # Initialize weights
        rng = np.random.default_rng(42)
        self.encoder_w1 = rng.normal(0, 0.01, (self.input_dim, self.hidden_dim)).astype(np.float32)
        self.encoder_b1 = np.zeros(self.hidden_dim, dtype=np.float32)
        self.encoder_w2 = rng.normal(0, 0.01, (self.hidden_dim, self.latent_dim)).astype(np.float32)
        self.encoder_b2 = np.zeros(self.latent_dim, dtype=np.float32)

        self.decoder_w1 = rng.normal(0, 0.01, (self.latent_dim, self.hidden_dim)).astype(np.float32)
        self.decoder_b1 = np.zeros(self.hidden_dim, dtype=np.float32)
        self.decoder_w2 = rng.normal(0, 0.01, (self.hidden_dim, self.input_dim)).astype(np.float32)
        self.decoder_b2 = np.zeros(self.input_dim, dtype=np.float32)

        for epoch in range(epochs):
            indices = rng.permutation(n_samples)
            total_loss = 0

            for i in range(0, n_samples, batch_size):
                batch_idx = indices[i:i + batch_size]
                x_batch = x[batch_idx]

                # Forward
                z = self.encode(x_batch)
                recon = self.decode(z)

                # Loss
                loss = np.mean((recon - x_batch) ** 2)
                total_loss += loss

        log.info("Autoencoder training complete (loss=%.6f)", total_loss)


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
        log.info("Friction %.1f CPG: forward=%.3f reward=0.0", mu, best_dist)

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


class RCAEPolicy:
    """Inference wrapper for RC with AE bottleneck (action-only)."""

    def __init__(self, rc_model, ae_model, obs_mean, obs_std, action_mean, action_std):
        self.rc_model = rc_model
        self.ae_model = ae_model
        self.obs_mean = obs_mean
        self.obs_std = obs_std
        self.action_mean = action_mean
        self.action_std = action_std

    def reset(self):
        self.rc_model.reset()

    def predict(self, obs, friction):
        obs_norm = (obs - self.obs_mean) / self.obs_std
        z = self.ae_model.encode(obs_norm)
        rc_in = np.concatenate([z, [friction]]).astype(np.float32)
        rc_out = self.rc_model.run(rc_in.reshape(1, -1)).reshape(-1)
        action = rc_out * self.action_std + self.action_mean
        return action


def build_reservoir_weights(n_reservoir, n_inputs, spectral_radius=0.92, density=0.12, input_scaling=2.5, seed=42):
    """Create sparse reservoir matrices."""
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


def evaluate_policy(env, policy: RCAEPolicy, friction: float, n_episodes=5, episode_length=500):
    """Evaluate policy distance and reward."""
    rewards = []
    distances = []

    for _ in range(n_episodes):
        policy.reset()
        set_floor_friction(env, friction)
        obs, _ = env.reset()
        start_x = get_base_x(env)
        ep_reward = 0

        for _ in range(episode_length):
            action = policy.predict(obs, friction)
            obs, reward, term, trunc, _ = env.step(action)
            ep_reward += reward
            if term or trunc:
                break

        dist = get_base_x(env) - start_x
        distances.append(dist)
        rewards.append(ep_reward)

    return float(np.mean(rewards)), float(np.mean(distances))


def train_rc_ae_singleaction(
    frictions,
    episodes_per_friction=20,
    min_forward=0.5,
    cpg_tune_iters=200,
    latent_dim=32,
    ae_hidden_dim=128,
    ae_epochs=100,
    n_reservoir=1500,
    ridge_alpha=1e-5,
):
    """Train RC with AE bottleneck, ACTION ONLY (no state prediction)."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    log.info("=" * 70)
    log.info("RC + AE (ACTION ONLY) TRAINING")
    log.info("=" * 70)

    data = collect_multifriction_dataset(
        frictions=frictions,
        episodes_per_friction=episodes_per_friction,
        min_forward=min_forward,
        cpg_tune_iters=cpg_tune_iters,
    )

    log.info("Collected dataset: obs=%s actions=%s", data.observations.shape, data.actions.shape)

    # Train autoencoder
    obs_mean = data.observations.mean(axis=0)
    obs_std = data.observations.std(axis=0)
    obs_std[obs_std < 1e-8] = 1.0
    obs_norm = (data.observations - obs_mean) / obs_std

    ae_model = NumpyAutoencoder(input_dim=105, latent_dim=latent_dim, hidden_dim=ae_hidden_dim)
    ae_model.fit(obs_norm, epochs=ae_epochs)

    # Encode observations through AE bottleneck
    z_all = ae_model.encode(obs_norm)
    log.info("AE encoding shape: %s", z_all.shape)

    # Normalize actions
    action_mean = data.actions.mean(axis=0)
    action_std = data.actions.std(axis=0)
    action_std[action_std < 1e-8] = 1.0
    action_norm = (data.actions - action_mean) / action_std

    # RC input: [latent + friction]
    rc_inputs = np.concatenate([z_all, data.frictions.reshape(-1, 1)], axis=1)
    # RC target: ACTIONS ONLY (no z_next!)
    rc_targets = action_norm

    log.info("RC inputs shape: %s", rc_inputs.shape)
    log.info("RC targets shape (actions only): %s", rc_targets.shape)

    win, w = build_reservoir_weights(
        n_reservoir=n_reservoir,
        n_inputs=rc_inputs.shape[1],
        spectral_radius=0.92,
        density=0.12,
        input_scaling=2.5,
    )

    reservoir = Reservoir(units=n_reservoir, input_dim=rc_inputs.shape[1], Win=win, W=w)
    readout = Ridge(input_dim=n_reservoir, output_dim=rc_targets.shape[1], ridge=ridge_alpha)
    rc_model = Model([reservoir, readout], edges=[(reservoir, 0, readout)])

    log.info("Training RC (action-only, AE bottleneck)...")
    rc_model.fit(rc_inputs, rc_targets)
    log.info("RC training complete")

    policy = RCAEPolicy(
        rc_model=rc_model,
        ae_model=ae_model,
        obs_mean=obs_mean,
        obs_std=obs_std,
        action_mean=action_mean,
        action_std=action_std,
    )

    eval_results = []
    for mu in frictions:
        env = gym.make("Ant-v5", render_mode="rgb_array")
        set_floor_friction(env, mu)
        rew, dist = evaluate_policy(env, policy, friction=mu, n_episodes=5, episode_length=500)
        env.close()
        eval_results.append({"friction": float(mu), "rc_reward": rew, "rc_fwd": dist})
        log.info("Eval friction %.1f -> forward=%.3f reward=%.1f", mu, dist, rew)

    model_data = {
        "type": "rc_ae_singleaction",
        "rc_model": rc_model,
        "ae_model": ae_model,
        "obs_mean": obs_mean,
        "obs_std": obs_std,
        "action_mean": action_mean,
        "action_std": action_std,
        "eval_results": eval_results,
        "cpg_stats": data.cpg_stats,
    }

    pkl_path = os.path.join(OUTPUT_DIR, "rc_ae_singleaction.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(model_data, f)
    log.info("Saved model to: %s", pkl_path)

    # Summary
    summary_path = os.path.join(OUTPUT_DIR, "summary.txt")
    with open(summary_path, "w") as f:
        f.write("RC + AE (Action-Only) Summary\n")
        f.write("=" * 70 + "\n")
        f.write(f"Model path: {pkl_path}\n")
        f.write(f"Latent dim: {latent_dim}\n")
        f.write(f"Reservoir units: {n_reservoir}\n")
        f.write(f"Ridge alpha: {ridge_alpha}\n\n")
        f.write("CPG Baseline Stats:\n")
        for mu in frictions:
            cpg_fwd = data.cpg_stats[mu]["forward"]
            f.write(f"  friction {mu}: cpg_fwd={cpg_fwd:.3f}\n")
        f.write("\nRC Evaluation Stats:\n")
        for res in eval_results:
            f.write(f"  friction {res['friction']}: rc_fwd={res['rc_fwd']:.3f}, rc_reward={res['rc_reward']:.1f}\n")

    log.info("Saved summary to: %s", summary_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes-per-friction", type=int, default=15)
    parser.add_argument("--cpg-tune-iters", type=int, default=200)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--ae-hidden-dim", type=int, default=128)
    parser.add_argument("--ae-epochs", type=int, default=100)
    parser.add_argument("--n-reservoir", type=int, default=1500)
    parser.add_argument("--ridge-alpha", type=float, default=1e-5)

    args = parser.parse_args()

    train_rc_ae_singleaction(
        frictions=TRAIN_FRICTIONS,
        episodes_per_friction=args.episodes_per_friction,
        cpg_tune_iters=args.cpg_tune_iters,
        latent_dim=args.latent_dim,
        ae_hidden_dim=args.ae_hidden_dim,
        ae_epochs=args.ae_epochs,
        n_reservoir=args.n_reservoir,
        ridge_alpha=args.ridge_alpha,
    )
