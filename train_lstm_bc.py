"""
Train LSTM neural network for CPG imitation (standard behavioral cloning).
Purpose: Test if the problem is with Reservoir Computing or with the approach itself.
"""

import argparse
import logging
import os
import pickle
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset


logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

TRAIN_FRICTIONS = [0.5, 1.0, 1.5]
OUTPUT_DIR = "lstm_bc_model"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


class LSTMBCPolicy(nn.Module):
    """LSTM network for behavioral cloning."""

    def __init__(self, obs_dim=105, action_dim=8, hidden_dim=256, num_layers=2):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        # Input: [obs (105) + friction (1)]
        self.lstm = nn.LSTM(
            input_size=obs_dim + 1,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, action_dim)
        )
        
        self.hidden_state = None

    def forward(self, x):
        """x shape: (batch_size, seq_len, obs_dim+1)"""
        lstm_out, self.hidden_state = self.lstm(x)
        # Take last timestep
        last_out = lstm_out[:, -1, :]
        action = self.fc(last_out)
        return action

    def reset_hidden(self, batch_size=1):
        """Reset hidden state for new episode."""
        self.hidden_state = None


class LSTMBCInference:
    """Inference wrapper for LSTM BC policy."""

    def __init__(self, model, obs_mean, obs_std, action_mean, action_std):
        self.model = model
        self.obs_mean = obs_mean
        self.obs_std = obs_std
        self.action_mean = action_mean
        self.action_std = action_std
        self.seq_buffer = []  # Store recent observations

    def reset(self):
        """Reset for new episode."""
        self.model.reset_hidden(batch_size=1)
        self.seq_buffer = []

    def predict(self, obs, friction, seq_len=10):
        """Predict action using recent observation history."""
        obs_norm = (obs - self.obs_mean) / self.obs_std
        
        # Add observation to buffer
        self.seq_buffer.append(np.concatenate([obs_norm, [friction]]))
        
        # Keep only last seq_len observations
        if len(self.seq_buffer) > seq_len:
            self.seq_buffer = self.seq_buffer[-seq_len:]
        
        # Pad if needed
        if len(self.seq_buffer) < seq_len:
            pad_obs = np.concatenate([np.zeros(105), [friction]])
            seq_data = [pad_obs] * (seq_len - len(self.seq_buffer)) + self.seq_buffer
        else:
            seq_data = self.seq_buffer
        
        # Convert to tensor
        x = torch.from_numpy(np.asarray(seq_data, dtype=np.float32)).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            action_norm = self.model(x).cpu().numpy().flatten()
        
        # Denormalize
        action = action_norm * self.action_std + self.action_mean
        return action


def train_lstm_bc(
    frictions,
    episodes_per_friction=15,
    min_forward=0.5,
    cpg_tune_iters=200,
    hidden_dim=256,
    num_layers=2,
    batch_size=64,
    epochs=50,
    learning_rate=1e-3,
    seq_len=10,
):
    """Train LSTM behavioral cloning model."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    log.info("=" * 70)
    log.info("LSTM BEHAVIORAL CLONING TRAINING")
    log.info("=" * 70)

    data = collect_multifriction_dataset(
        frictions=frictions,
        episodes_per_friction=episodes_per_friction,
        min_forward=min_forward,
        cpg_tune_iters=cpg_tune_iters,
    )

    log.info("Collected dataset: obs=%s actions=%s", data.observations.shape, data.actions.shape)

    # Normalize
    obs_mean = data.observations.mean(axis=0)
    obs_std = data.observations.std(axis=0)
    obs_std[obs_std < 1e-8] = 1.0
    obs_norm = (data.observations - obs_mean) / obs_std

    action_mean = data.actions.mean(axis=0)
    action_std = data.actions.std(axis=0)
    action_std[action_std < 1e-8] = 1.0
    action_norm = (data.actions - action_mean) / action_std

    # Create sequences
    X = []
    y = []
    for i in range(len(obs_norm) - seq_len):
        x_seq = np.concatenate([obs_norm[i:i+seq_len], data.frictions[i:i+seq_len].reshape(-1, 1)], axis=1)
        X.append(x_seq)
        y.append(action_norm[i + seq_len])
    
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)

    log.info("Training data shape: X=%s y=%s", X.shape, y.shape)

    # Create model
    model = LSTMBCPolicy(obs_dim=105, action_dim=8, hidden_dim=hidden_dim, num_layers=num_layers).to(DEVICE)
    optimizer = Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    # Training
    dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    log.info("Training LSTM BC for %d epochs...", epochs)
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            
            optimizer.zero_grad()
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            log.info("Epoch %d: loss=%.6f", epoch + 1, total_loss / len(dataloader))

    log.info("Training complete")

    # Evaluate
    model.eval()
    policy = LSTMBCInference(model, obs_mean, obs_std, action_mean, action_std)

    eval_results = []
    for mu in frictions:
        rewards = []
        distances = []
        
        for _ in range(5):
            policy.reset()
            env = gym.make("Ant-v5", render_mode="rgb_array")
            set_floor_friction(env, mu)
            obs, _ = env.reset()
            start_x = get_base_x(env)
            ep_reward = 0

            for _ in range(500):
                action = policy.predict(obs, mu, seq_len=seq_len)
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
        eval_results.append({"friction": float(mu), "rc_reward": avg_reward, "rc_fwd": avg_dist})
        log.info("Eval friction %.1f -> forward=%.3f reward=%.1f", mu, avg_dist, avg_reward)

    # Save
    model_data = {
        "model": model.state_dict(),
        "obs_mean": obs_mean,
        "obs_std": obs_std,
        "action_mean": action_mean,
        "action_std": action_std,
        "eval_results": eval_results,
        "cpg_stats": data.cpg_stats,
    }

    pkl_path = os.path.join(OUTPUT_DIR, "lstm_bc.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(model_data, f)
    log.info("Saved model to: %s", pkl_path)

    # Summary
    summary_path = os.path.join(OUTPUT_DIR, "summary.txt")
    with open(summary_path, "w") as f:
        f.write("LSTM Behavioral Cloning Summary\n")
        f.write("=" * 70 + "\n")
        f.write(f"Model path: {pkl_path}\n")
        f.write(f"Hidden dim: {hidden_dim}, Num layers: {num_layers}\n\n")
        f.write("CPG Baseline Stats:\n")
        for mu in frictions:
            cpg_fwd = data.cpg_stats[mu]["forward"]
            f.write(f"  friction {mu}: cpg_fwd={cpg_fwd:.3f}\n")
        f.write("\nLSTM BC Evaluation Stats:\n")
        for res in eval_results:
            f.write(f"  friction {res['friction']}: bc_fwd={res['rc_fwd']:.3f}, bc_reward={res['rc_reward']:.1f}\n")

    log.info("Saved summary to: %s", summary_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes-per-friction", type=int, default=15)
    parser.add_argument("--cpg-tune-iters", type=int, default=200)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seq-len", type=int, default=10)

    args = parser.parse_args()

    train_lstm_bc(
        frictions=TRAIN_FRICTIONS,
        episodes_per_friction=args.episodes_per_friction,
        cpg_tune_iters=args.cpg_tune_iters,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        seq_len=args.seq_len,
    )
