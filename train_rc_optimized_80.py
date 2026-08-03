"""
Optimized RC with target: Reach 80%+ alignment with CPG.
Key fix: proper sequential reservoir state collection per episode,
manual ridge regression on harvested states, correct config selection.
"""

import argparse
import logging
import os
import pickle

import gymnasium as gym
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import time

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

TRAIN_FRICTIONS = [0.5, 1.0, 1.5]
OUTPUT_DIR = "optimized_rc_80_model"
WASHOUT = 50  # steps to discard at start of each episode


def _save_checkpoint(tag, **kwargs):
    """Save a lightweight checkpoint of the current training state.
    The function is defensive about missing values so it can be called
    at several points during the pipeline.
    """
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        path = os.path.join(OUTPUT_DIR, f"rc_80_checkpoint_{int(time.time())}_{tag}.pkl")
        data = {}
        # Only include known items if present to avoid huge dumps
        for k in ("esn", "autoencoder", "X_mean", "X_std", "Y_mean", "Y_std", "config", "cpg_per_friction", "frictions", "eval_results"):
            if k in kwargs and kwargs[k] is not None:
                data[k] = kwargs[k]
        # Always include lightweight metadata
        data["tag"] = tag
        data["ts"] = time.time()
        with open(path, "wb") as f:
            pickle.dump(data, f)
        log.info("Saved checkpoint: %s", path)
    except Exception as e:
        log.warning("Failed to save checkpoint '%s': %s", tag, e)


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------

def _get_floor_geom_indices(model):
    indices = []
    ngeom = model.ngeom
    geom_names = getattr(model, "geom_names", None)
    for i in range(ngeom):
        name = None
        if geom_names is not None:
            try:
                n = geom_names[i]
                name = n.decode("utf-8") if isinstance(n, bytes) else str(n)
            except Exception:
                pass
        is_plane = int(model.geom_type[i]) == 0
        is_named_floor = name is not None and ("floor" in name or "ground" in name)
        if is_plane or is_named_floor:
            indices.append(i)
    return indices


def set_floor_friction(env, mu: float):
    """Set floor friction on all plane-type geoms and named floor geoms."""
    try:
        model = env.unwrapped.model
    except Exception:
        return
    for i in _get_floor_geom_indices(model):
        try:
            model.geom_friction[i] = np.array([mu, 0.005, 0.0001])
        except Exception:
            pass


def set_floor_color(env, color, zone_idx: int = 0):
    """Tint the floor geometry so each friction zone is visually distinct in 3D."""
    try:
        model = env.unwrapped.model
    except Exception:
        return
    color_arr = np.array(color, dtype=np.float32)
    for i in _get_floor_geom_indices(model):
        try:
            model.geom_rgba[i] = color_arr
        except Exception:
            pass


def get_base_x(env) -> float:
    try:
        return float(env.unwrapped.data.qpos[0])
    except Exception:
        return 0.0


def get_base_y(env) -> float:
    try:
        return float(env.unwrapped.data.qpos[1])
    except Exception:
        return 0.0


def get_forward_progress(env, start_x: float, start_y: float, previous_progress: float = 0.0) -> float:
    """Use the ant's planar displacement from the start pose as a robust progress signal."""
    try:
        qpos = np.asarray(env.unwrapped.data.qpos[:2], dtype=float)
    except Exception:
        return previous_progress
    try:
        com = np.asarray(env.unwrapped.data.subtree_com[0][:2], dtype=float)
    except Exception:
        com = qpos
    start_pos = np.array([start_x, start_y], dtype=float)
    displacement = max(np.linalg.norm(qpos - start_pos), np.linalg.norm(com - start_pos))
    return max(previous_progress, float(displacement))


def _parse_switch_sequence(raw_sequence):
    if raw_sequence is None:
        return []
    values = []
    for part in str(raw_sequence).split(","):
        part = part.strip()
        if not part:
            continue
        try:
            values.append(float(part))
        except ValueError:
            continue
    return values


def _annotate_frame(frame, step, current_mu, switched=False, reason="surface",
                    progress=0.0, next_boundary=0.0, switch_values=None,
                    current_idx=0, total_zones=0):
    try:
        img = Image.fromarray(frame).convert("RGBA")
    except Exception:
        return frame
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    draw.text((10, 10), f"Step {step} | Active CPG {current_mu:.1f}", fill=(255, 255, 255, 255), font=font)
    draw.text((10, 32), f"Progress {progress:.2f}m | Next zone {next_boundary:.2f}m",
              fill=(255, 255, 255, 255), font=font)

    if switch_values:
        zone_palette = [(70, 140, 255), (255, 190, 60), (90, 220, 90)]
        bar_h = max(28, int(img.height * 0.16))
        bar_y = max(8, img.height - bar_h - 10)
        width = img.width
        for idx, val in enumerate(switch_values):
            x0 = int(idx / max(1, len(switch_values)) * width)
            x1 = int((idx + 1) / max(1, len(switch_values)) * width)
            color = zone_palette[idx % len(zone_palette)]
            alpha = 190 if idx == current_idx else 120
            draw.rectangle([x0, bar_y, x1, img.height - 8], fill=color + (alpha,))
            draw.line([x0, bar_y, x0, img.height - 8], fill=(255, 255, 255, 180), width=2)
            label = f"CPG {val:.1f}" if idx == len(switch_values) - 1 else f"{val:.1f}"
            draw.text((x0 + 8, bar_y + 6), label, fill=(255, 255, 255, 255), font=font)

        if current_idx is not None:
            current_color = zone_palette[current_idx % len(zone_palette)]
            draw.rounded_rectangle(
                [10, img.height - bar_h - 46, 180, img.height - bar_h - 16],
                radius=8,
                fill=current_color + (220,),
                outline=(255, 255, 255, 255),
            )
            draw.text((22, img.height - bar_h - 40), f"ACTIVE SURFACE {current_idx + 1}/{max(1, total_zones)}", fill=(255, 255, 255, 255), font=font)

    if switched:
        draw.rectangle((6, 6, img.width - 7, img.height - 7), outline=(255, 0, 0, 255), width=3)
        label = "ENVIRONMENT FORCED SWITCH" if reason == "surface" else "RC SWITCHED CPG"
        draw.text((10, 52), label, fill=(255, 0, 0, 255), font=font)
    return np.asarray(img.convert("RGB"))


def visualize_saved_model(model_path, episode_length=500, output_gif=None,
                          switch_every=80, switch_sequence=None,
                          render_mode="rgb_array", switch_distance=0.3, action_scale=1.2,
                          cpg_speed_scale=1.0, cpg_amp_scale=1.0, action_smooth=0.0, cpg_mix=0.0):
    """Load a saved RC model and record a single rollout with CPG-switch annotations."""
    if output_gif is None:
        output_gif = os.path.join(OUTPUT_DIR, "rc_switch_demo.gif")

    os.makedirs(os.path.dirname(output_gif) or ".", exist_ok=True)

    with open(model_path, "rb") as f:
        model_data = pickle.load(f)

    esn = model_data["esn"]
    autoencoder = model_data.get("autoencoder")
    cpg_per_friction = model_data.get("cpg_per_friction", {})
    X_mean = model_data["X_mean"]
    X_std = model_data["X_std"]
    Y_mean = model_data["Y_mean"]
    Y_std = model_data["Y_std"]

    available_friction = []
    if hasattr(esn, "W_out_per_friction"):
        available_friction = [float(mu) for mu in esn.W_out_per_friction.keys()]
    if not available_friction:
        available_friction = _parse_switch_sequence(switch_sequence)
    if not available_friction:
        available_friction = [0.5, 1.0, 1.5]

    requested_sequence = _parse_switch_sequence(switch_sequence)
    if requested_sequence:
        switch_values = requested_sequence
    else:
        switch_values = available_friction

    if not switch_values:
        switch_values = [0.5, 1.0, 1.5]

    log.info("Visualizing saved model from %s", model_path)
    log.info("Recording GIF to %s", output_gif)
    log.info("Using multi-surface switching over %.2fm zones: %s", switch_distance, switch_values)

    env = gym.make("Ant-v5", render_mode=render_mode)
    current_mu = float(switch_values[0])
    set_floor_friction(env, current_mu)
    zone_palette = [(0.22, 0.38, 0.95, 1.0), (0.95, 0.45, 0.20, 1.0), (0.18, 0.80, 0.32, 1.0)]
    set_floor_color(env, zone_palette[0], zone_idx=0)
    if hasattr(esn, "W_out_per_friction") and current_mu in esn.W_out_per_friction:
        esn.W_out = esn.W_out_per_friction[current_mu]
    esn.reset()

    obs, _ = env.reset()
    prev_obs = obs.copy()
    prev_action_unclipped = None
    start_x = get_base_x(env)
    start_y = get_base_y(env)
    frames = []
    current_idx = 0
    current_cpg = None
    progress = 0.0
    zone_boundaries = [0.0] + [switch_distance * (i + 1) for i in range(len(switch_values) - 1)]

    for step in range(episode_length):
        switched = False
        progress = get_forward_progress(env, start_x, start_y, previous_progress=progress)
        while current_idx + 1 < len(switch_values) and progress >= zone_boundaries[current_idx + 1]:
            next_idx = current_idx + 1
            next_mu = float(switch_values[next_idx])
            if next_mu != current_mu:
                current_idx = next_idx
                current_mu = next_mu
                set_floor_friction(env, current_mu)
                zone_palette = [(0.22, 0.38, 0.95, 1.0), (0.95, 0.45, 0.20, 1.0), (0.18, 0.80, 0.32, 1.0)]
                set_floor_color(env, zone_palette[current_idx % len(zone_palette)], zone_idx=current_idx)
                if hasattr(esn, "W_out_per_friction") and current_mu in esn.W_out_per_friction:
                    esn.W_out = esn.W_out_per_friction[current_mu]
                esn.reset()
                switched = True
                log.info("  Environment changed surface -> CPG %.1f at %.2fm", current_mu, progress)
            else:
                break

        vel = obs - prev_obs
        t_sec = step * 0.02
        cpg = cpg_per_friction.get(current_mu)
        if cpg is None:
            cpg = CPGController(n_actions=8)
        else:
            # allow temporary scaling of CPG speed/amplitude for visualization trials
            try:
                cpg.omega = float(cpg.omega) * float(cpg_speed_scale)
                cpg.amplitudes = np.asarray(cpg.amplitudes) * float(cpg_amp_scale)
            except Exception:
                pass
        current_cpg = cpg
        cpg_phase = (cpg.omega * t_sec + cpg.phases).astype(np.float64)
        cpg_out = np.clip(cpg.step(t_sec), env.action_space.low, env.action_space.high).astype(np.float64)

        obs_aug = np.concatenate([obs, vel, cpg_phase, cpg_out, [current_mu]]).astype(np.float64)
        obs_norm = (obs_aug - X_mean) / X_std
        if autoencoder is not None:
            obs_input = autoencoder.encode(obs_norm.reshape(1, -1)).reshape(-1)
        else:
            obs_input = obs_norm

        action_norm = esn.predict(obs_input)
        action_scaled = action_norm * action_scale
        action_unclipped = (action_scaled * Y_std + Y_mean)
        # Optionally mix a fraction of the open-loop CPG output to guarantee movement
        # and avoid short pauses where the learned controller outputs near-zero actions.
        if 'cpg_mix' in locals() and float(cpg_mix) > 0.0:
            try:
                cpg_frac = float(cpg_mix)
                # cpg_out is already clipped to env.action_space earlier
                action_unclipped = (1.0 - cpg_frac) * action_unclipped + cpg_frac * cpg_out
            except Exception:
                pass
        # optional smoothing to avoid abrupt large jumps (visualization only)
        if action_smooth is not None and float(action_smooth) > 0.0:
            if prev_action_unclipped is None:
                prev_action_unclipped = action_unclipped.copy()
            action_unclipped = prev_action_unclipped * float(action_smooth) + action_unclipped * (1.0 - float(action_smooth))
            prev_action_unclipped = action_unclipped.copy()
        action = np.clip(action_unclipped, env.action_space.low, env.action_space.high)

        prev_obs = obs.copy()
        obs, reward, term, trunc, _ = env.step(action)

        frame = env.render()
        if frame is not None:
            frame_arr = np.asarray(frame)
            if frame_arr.ndim == 2:
                frame_arr = np.repeat(frame_arr[..., None], 3, axis=2)
            next_boundary = zone_boundaries[current_idx + 1] if current_idx + 1 < len(zone_boundaries) else zone_boundaries[-1]
            frame_arr = _annotate_frame(
                frame_arr,
                step,
                current_mu,
                switched=switched,
                reason="surface",
                progress=progress,
                next_boundary=next_boundary,
                switch_values=switch_values,
                current_idx=current_idx,
                total_zones=len(switch_values),
            )
            frames.append(frame_arr)

        if term or trunc:
            break

    end_x = get_base_x(env)
    env.close()

    if frames:
        gif_images = [Image.fromarray(frame) for frame in frames]
        gif_images[0].save(output_gif, save_all=True, append_images=gif_images[1:],
                          loop=0, duration=16)  # 16ms per frame ≈ 60fps for smooth motion
        log.info("Saved rollout GIF with %d frames to %s (playback: %.1f sec)", len(frames), output_gif, len(frames) * 0.016)
    else:
        log.warning("No frames captured; GIF was not written.")

    log.info("Episode distance: %.3f", end_x - start_x)


# ---------------------------------------------------------------------------
# CPG
# ---------------------------------------------------------------------------

class CPGController:
    def __init__(self, n_actions, omega=2.0, amplitudes=None, phases=None, offsets=None):
        self.n = n_actions
        self.omega = float(omega)
        self.amplitudes = np.ones(self.n) if amplitudes is None else np.asarray(amplitudes).reshape(self.n)
        self.phases    = np.zeros(self.n) if phases    is None else np.asarray(phases).reshape(self.n)
        self.offsets   = np.zeros(self.n) if offsets   is None else np.asarray(offsets).reshape(self.n)

    @classmethod
    def from_vector(cls, vec, n_actions):
        return cls(n_actions, float(vec[0]),
                   vec[1:1+n_actions],
                   vec[1+n_actions:1+2*n_actions],
                   vec[1+2*n_actions:1+3*n_actions])

    def step(self, t):
        return self.offsets + self.amplitudes * np.sin(self.omega * t + self.phases)


def _eval_cpg_vec(env, vec):
    """Evaluate a CPG vector, return forward distance."""
    cpg = CPGController.from_vector(vec, 8)
    obs, _ = env.reset()
    start_x = get_base_x(env)
    for step in range(500):
        action = np.clip(cpg.step(step * 0.02).astype(np.float32),
                         env.action_space.low, env.action_space.high)
        obs, _, term, trunc, _ = env.step(action)
        if term or trunc:
            break
    return get_base_x(env) - start_x


def tune_cpg(env, cpg_iters=1000):
    """Random search + hill-climbing refinement for best CPG."""
    best_dist, best_vec = -float("inf"), None

    # Phase 1: random search
    for i in range(cpg_iters):
        vec = np.random.randn(25)
        dist = _eval_cpg_vec(env, vec)
        if dist > best_dist:
            best_dist, best_vec = dist, vec.copy()
        if (i + 1) % 200 == 0:
            log.info("    CPG tune iter %d: best_dist=%.3f", i + 1, best_dist)

    # Phase 2: hill-climbing refinement (500 steps, shrinking noise)
    log.info("    Hill-climbing from best_dist=%.3f...", best_dist)
    current_vec = best_vec.copy()
    sigma = 0.3
    for i in range(500):
        candidate = current_vec + np.random.randn(25) * sigma
        dist = _eval_cpg_vec(env, candidate)
        if dist > best_dist:
            best_dist = dist
            current_vec = candidate.copy()
            best_vec = candidate.copy()
        if (i + 1) % 100 == 0:
            sigma *= 0.7
    log.info("    After hill-climbing: best_dist=%.3f", best_dist)

    return CPGController.from_vector(best_vec, 8), best_dist


# ---------------------------------------------------------------------------
# Reservoir (pure numpy — no reservoirpy sequential bug)
# ---------------------------------------------------------------------------

class EchoStateNetwork:
    """Minimal leaky ESN with numpy. Proper sequential state updates."""

    def __init__(self, n_inputs, n_reservoir, spectral_radius=0.95,
                 input_scaling=1.0, leak_rate=0.3, sparsity=0.1, seed=42):
        rng = np.random.RandomState(seed)
        self.n_reservoir = n_reservoir
        self.leak = leak_rate

        # Input weights
        self.Win = rng.uniform(-input_scaling, input_scaling,
                               (n_reservoir, n_inputs)).astype(np.float64)

        # Recurrent weights (sparse)
        W = rng.uniform(-0.5, 0.5, (n_reservoir, n_reservoir))
        mask = rng.random((n_reservoir, n_reservoir)) < sparsity
        W *= mask
        eigvals = np.linalg.eigvals(W)
        max_abs = np.max(np.abs(eigvals))
        if max_abs > 1e-12:
            W *= spectral_radius / max_abs
        self.W = W.astype(np.float64)

        self.state = np.zeros(n_reservoir, dtype=np.float64)
        self.W_out = None  # set after fit

    def reset(self):
        self.state = np.zeros(self.n_reservoir, dtype=np.float64)

    def update(self, x):
        """Single-step update, returns new state."""
        x = x.astype(np.float64)
        pre = self.W @ self.state + self.Win @ x
        new_state = (1 - self.leak) * self.state + self.leak * np.tanh(pre)
        self.state = new_state
        return self.state.copy()

    def collect_states(self, episodes_X, washout=WASHOUT):
        """
        episodes_X: list of 2-D arrays (T_ep x n_inputs), one per episode.
        Returns (H, X_in, indices) where:
          H     — reservoir states (n_samples, n_reservoir)
          X_in  — corresponding input vectors (n_samples, n_inputs)
          indices — list of (ep_idx, t) for aligning Y
        """
        all_states = []
        all_inputs = []
        all_ep_indices = []
        for ep_idx, ep_X in enumerate(episodes_X):
            self.reset()
            for t, x in enumerate(ep_X):
                s = self.update(x)
                if t >= washout:
                    all_states.append(s)
                    all_inputs.append(x.astype(np.float64))
                    all_ep_indices.append((ep_idx, t))
        return (np.array(all_states, dtype=np.float64),
                np.array(all_inputs, dtype=np.float64),
                all_ep_indices)

    def _augment(self, H, X_in):
        """Paper Eq.3: augment readout input as [1; In; s]."""
        bias = np.ones((len(H), 1), dtype=np.float64)
        return np.concatenate([bias, X_in, H], axis=1)

    def fit(self, episodes_X, episodes_Y, ridge=1e-4, washout=WASHOUT):
        """
        Harvest reservoir states episode-by-episode, then solve ridge regression.
        Uses paper Eq.3 output: W_out @ [1; In; s] so readout has direct
        access to inputs (CPG phase, friction) without reservoir encoding them.
        episodes_X / episodes_Y: lists of per-episode arrays.
        """
        log.info("    Harvesting reservoir states (%d episodes)...", len(episodes_X))
        H, X_in, indices = self.collect_states(episodes_X, washout=washout)
        H_aug = self._augment(H, X_in)   # [1; In; s]

        # Build target matrix aligned to harvested states
        Y_list = [episodes_Y[ep_idx][t] for ep_idx, t in indices]
        Y_mat = np.array(Y_list, dtype=np.float64)

        log.info("    Fitting readout: H_aug=%s Y=%s ridge=%.1e", H_aug.shape, Y_mat.shape, ridge)

        # Activation stats sanity check
        mean_act = np.mean(np.abs(H))
        log.info("    Reservoir activation: mean=%.4f std=%.4f", mean_act, np.std(H))
        if mean_act < 0.01:
            log.warning("    WARNING: reservoir nearly silent — increase input_scaling")
        elif mean_act > 0.95:
            log.warning("    WARNING: reservoir saturated — decrease spectral_radius")

        # Ridge regression on augmented features: W_out = (H_aug'H_aug + ridge*I)^{-1} H_aug' Y
        A = H_aug.T @ H_aug + ridge * np.eye(H_aug.shape[1])
        b = H_aug.T @ Y_mat
        self.W_out = np.linalg.solve(A, b)  # (1 + n_inputs + n_reservoir, n_outputs)
        log.info("    W_out norm=%.4f max=%.4f", np.linalg.norm(self.W_out), np.max(np.abs(self.W_out)))

        # Store for DAgger fine-tuning
        self._train_X = list(episodes_X)
        self._train_Y = list(episodes_Y)
        self._ridge   = ridge

        return self

    def predict(self, x):
        """Single-step predict (updates state). Uses [1; In; s] per paper Eq.3."""
        x = np.asarray(x, dtype=np.float64)
        s = self.update(x)
        v = np.concatenate([[1.0], x, s])
        return v @ self.W_out

    def predict_sequence(self, X, washout=0):
        """Predict over a sequence without resetting."""
        preds = []
        for t, x in enumerate(X):
            p = self.predict(x)
            if t >= washout:
                preds.append(p)
        return np.array(preds)


class NumpyAutoencoder:
    """Small tanh autoencoder implemented in NumPy."""

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


# ---------------------------------------------------------------------------
# Data collection (now returns per-episode lists)
# ---------------------------------------------------------------------------

def collect_episodes(env, cpg, n_episodes=50, min_forward=1.0):
    """Returns (episodes_obs, episodes_actions) as lists of arrays.
    Input includes CPG phase angles and output so reservoir knows gait phase."""
    episodes_obs = []
    episodes_actions = []
    good, attempts = 0, 0
    max_attempts = n_episodes * 5

    while good < n_episodes and attempts < max_attempts:
        attempts += 1
        ep_obs, ep_act = [], []
        obs, _ = env.reset()
        start_x = get_base_x(env)
        prev_obs = obs.copy()

        for step in range(500):
            t = step * 0.02
            action = np.clip(cpg.step(t).astype(np.float32),
                             env.action_space.low, env.action_space.high)
            vel = obs - prev_obs
            # CPG phase angles (raw, before clipping) and clipped output
            cpg_phase = (cpg.omega * t + cpg.phases).astype(np.float32)  # 8-dim
            cpg_out   = action.copy()                                      # 8-dim
            ep_obs.append(np.concatenate([obs, vel, cpg_phase, cpg_out]).astype(np.float32))
            ep_act.append(action.copy())
            prev_obs = obs.copy()
            obs, _, term, trunc, _ = env.step(action)
            if term or trunc:
                break

        forward = get_base_x(env) - start_x
        if forward >= min_forward:
            episodes_obs.append(np.array(ep_obs, dtype=np.float32))
            episodes_actions.append(np.array(ep_act, dtype=np.float32))
            good += 1
            if good % 10 == 0:
                log.info("      Collected %d/%d good episodes", good, n_episodes)

    log.info("      Final: %d good episodes from %d attempts", good, attempts)
    return episodes_obs, episodes_actions


# ---------------------------------------------------------------------------
# Config selection: actual held-out prediction error
# ---------------------------------------------------------------------------

def evaluate_config_mse(esn, val_episodes_X, val_episodes_Y, washout=WASHOUT):
    """MSE on held-out episodes — lower is better."""
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


# ---------------------------------------------------------------------------
# DAgger closed-loop fine-tuning (per-friction readouts, explicit labels)
# ---------------------------------------------------------------------------

def _dagger_rollout(esn, env, cpg, mu, X_mean, X_std, Y_mean, Y_std, n_eps, W_out_override=None, autoencoder=None):
    """Run ESN closed-loop with optional W_out override, label with CPG."""
    saved_W_out = esn.W_out
    if W_out_override is not None:
        esn.W_out = W_out_override
    eps_X, eps_Y = [], []
    for _ in range(n_eps):
        obs, _ = env.reset()
        esn.reset()
        prev_obs = obs.copy()
        ep_X, ep_Y = [], []
        for step in range(500):
            vel = obs - prev_obs
            t_sec = step * 0.02
            cpg_phase = (cpg.omega * t_sec + cpg.phases).astype(np.float64)
            cpg_out   = np.clip(cpg.step(t_sec),
                                np.full(8, -1.0), np.full(8, 1.0)).astype(np.float64)
            obs_aug = np.concatenate([obs, vel, cpg_phase, cpg_out, [mu]]).astype(np.float64)
            obs_norm = (obs_aug - X_mean) / X_std
            if autoencoder is not None:
                obs_norm = autoencoder.encode(obs_norm.reshape(1, -1)).reshape(-1)
            action_norm = esn.predict(obs_norm)
            action = np.clip(
                (action_norm * Y_std + Y_mean).astype(np.float32),
                env.action_space.low, env.action_space.high,
            )
            cpg_action = np.clip(
                cpg.step(step * 0.02).astype(np.float32),
                env.action_space.low, env.action_space.high,
            )
            ep_X.append(obs_norm)
            ep_Y.append((cpg_action - Y_mean) / Y_std)
            prev_obs = obs.copy()
            obs, _, term, trunc, _ = env.step(action)
            if term or trunc:
                break
        eps_X.append(np.array(ep_X, dtype=np.float64))
        eps_Y.append(np.array(ep_Y, dtype=np.float64))
    esn.W_out = saved_W_out
    return eps_X, eps_Y


def _fit_readout(esn, episodes_X, episodes_Y, ridge, washout, base_n_states=None):
    """Fit a single W_out on given episodes using paper Eq.3: [1; In; s].
    Scales ridge proportionally to dataset size so regularisation stays meaningful."""
    H, X_in, indices = esn.collect_states(episodes_X, washout=washout)
    H_aug = esn._augment(H, X_in)
    Y_list = [episodes_Y[ep_idx][t] for ep_idx, t in indices]
    Y_mat = np.array(Y_list, dtype=np.float64)
    # Scale ridge with dataset size relative to initial size
    n_states = len(indices)
    scale = (n_states / base_n_states) if (base_n_states is not None and base_n_states > 0) else 1.0
    effective_ridge = ridge * scale
    A = H_aug.T @ H_aug + effective_ridge * np.eye(H_aug.shape[1])
    return np.linalg.solve(A, H_aug.T @ Y_mat)


def closed_loop_finetune(esn, cpg_per_friction, frictions, n_episodes,
                         X_mean, X_std, Y_mean, Y_std,
                         friction_per_ep,          # explicit friction label per episode
                         norm_episodes_X,           # all normalised open-loop eps
                         norm_episodes_Y,
                         autoencoder=None,
                         n_rounds=5, episodes_per_round=10, washout=WASHOUT):
    """
    Per-friction DAgger with explicit episode labels (no normalisation recovery).
    Each friction keeps its own episode pool and readout.
    esn.W_out_per_friction[mu] is set after training.
    At eval time caller selects the right readout per friction.
    """
    # Seed per-friction pools from open-loop data using explicit labels
    pool_X = {mu: [] for mu in frictions}
    pool_Y = {mu: [] for mu in frictions}
    for ep_X, ep_Y, mu in zip(norm_episodes_X, norm_episodes_Y, friction_per_ep):
        pool_X[mu].append(ep_X)
        pool_Y[mu].append(ep_Y)

    log.info("  Initial pool sizes: %s", {mu: len(pool_X[mu]) for mu in frictions})

    # Fit initial per-friction readouts from open-loop data
    W_out_pf = {}
    base_n_states = {}   # track initial state count for ridge scaling
    for mu in frictions:
        H_init, X_in_init, idx_init = esn.collect_states(pool_X[mu], washout=washout)
        base_n_states[mu] = len(idx_init)
        H_aug_init = esn._augment(H_init, X_in_init)
        Y_init = np.array([pool_Y[mu][ei][t] for ei, t in idx_init], dtype=np.float64)
        A = H_aug_init.T @ H_aug_init + esn._ridge * np.eye(H_aug_init.shape[1])
        W_out_pf[mu] = np.linalg.solve(A, H_aug_init.T @ Y_init)
        log.info("  Initial W_out[%.1f] norm=%.4f (base_n_states=%d)",
                 mu, np.linalg.norm(W_out_pf[mu]), base_n_states[mu])

    for round_idx in range(n_rounds):
        log.info("  DAgger round %d/%d", round_idx + 1, n_rounds)

        for mu in frictions:
            env = gym.make("Ant-v5", render_mode="rgb_array")
            set_floor_friction(env, mu)

            new_X, new_Y = _dagger_rollout(
                esn, env, cpg_per_friction[mu], mu,
                X_mean, X_std, Y_mean, Y_std,
                episodes_per_round,
                W_out_override=W_out_pf[mu],
                autoencoder=autoencoder,
            )
            env.close()

            pool_X[mu].extend(new_X)
            pool_Y[mu].extend(new_Y)
            n_states = sum(len(e) for e in new_X)
            log.info("    mu=%.1f: +%d episodes (%d new states, pool=%d eps)",
                     mu, episodes_per_round, n_states, len(pool_X[mu]))

            W_out_pf[mu] = _fit_readout(
                esn, pool_X[mu], pool_Y[mu], esn._ridge, washout,
                base_n_states=base_n_states[mu]
            )
            log.info("    mu=%.1f W_out norm=%.4f", mu, np.linalg.norm(W_out_pf[mu]))

    esn.W_out_per_friction = W_out_pf
    # Set global W_out to average (used if friction unknown)
    esn.W_out = np.mean(list(W_out_pf.values()), axis=0)
    try:
        _save_checkpoint(f"dagger_round_{round_idx}", esn=esn, autoencoder=autoencoder, frictions=frictions, cpg_per_friction=cpg_per_friction)
    except Exception:
        pass
    return esn


# ---------------------------------------------------------------------------
# Main training
# ---------------------------------------------------------------------------

def train_optimized_rc(
    frictions=TRAIN_FRICTIONS,
    n_episodes=50,
    n_reservoir=1500,
    cpg_iters=1000,
    min_forward=1.0,
    ae_hidden_dim=128,
    ae_latent_dim=64,
    ae_epochs=80,
):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    log.info("=" * 70)
    log.info("OPTIMIZED RC FOR 80%+ ALIGNMENT")
    log.info("=" * 70)
    log.info("Config: n_episodes=%d, n_reservoir=%d, cpg_iters=%d, ae_latent_dim=%d, ae_epochs=%d",
             n_episodes, n_reservoir, cpg_iters, ae_latent_dim, ae_epochs)

    # ------------------------------------------------------------------
    # 1. Collect data per friction
    # ------------------------------------------------------------------
    all_episodes_X = []   # flat list of episode arrays
    all_episodes_Y = []
    friction_per_ep = []  # which friction each episode belongs to
    cpg_stats = {}
    cpg_per_friction = {}

    for mu in frictions:
        log.info("\n[Friction %.1f]", mu)
        env = gym.make("Ant-v5", render_mode="rgb_array")
        set_floor_friction(env, mu)

        log.info("  Tuning CPG (%d iterations)...", cpg_iters)
        cpg, cpg_dist = tune_cpg(env, cpg_iters=cpg_iters)
        cpg_stats[mu] = {"forward": float(cpg_dist)}
        cpg_per_friction[mu] = cpg
        log.info("  CPG tuned: forward=%.3f", cpg_dist)

        log.info("  Collecting %d episodes (min_forward=%.1f)...", n_episodes, min_forward)
        eps_obs, eps_act = collect_episodes(env, cpg, n_episodes=n_episodes, min_forward=min_forward)

        # Append friction feature to each episode's observations
        for ep_obs, ep_act in zip(eps_obs, eps_act):
            mu_col = np.full((len(ep_obs), 1), mu, dtype=np.float32)
            ep_obs_with_mu = np.concatenate([ep_obs, mu_col], axis=1)
            all_episodes_X.append(ep_obs_with_mu)
            all_episodes_Y.append(ep_act)
            friction_per_ep.append(mu)

        log.info("  Collected %d episodes (%d total steps)",
                 len(eps_obs), sum(len(e) for e in eps_obs))
        env.close()
        # checkpoint after collecting episodes for this friction
        try:
            _save_checkpoint(f"collected_mu_{mu}", cpg_per_friction=cpg_per_friction, frictions=frictions)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # 2. Normalize across all data
    # ------------------------------------------------------------------
    X_flat = np.vstack(all_episodes_X)
    Y_flat = np.vstack(all_episodes_Y)
    input_dim = X_flat.shape[1]
    n_outputs = Y_flat.shape[1]

    X_mean = X_flat.mean(axis=0)
    X_std  = X_flat.std(axis=0);  X_std[X_std < 1e-8] = 1.0
    Y_mean = Y_flat.mean(axis=0)
    Y_std  = Y_flat.std(axis=0);  Y_std[Y_std < 1e-8] = 1.0

    X_norm_flat = (X_flat - X_mean) / X_std
    # Normalise each episode
    norm_episodes_X = [(ep - X_mean) / X_std for ep in all_episodes_X]
    norm_episodes_Y = [(ep - Y_mean) / Y_std for ep in all_episodes_Y]

    log.info("\nTotal episodes: %d | Raw input dim: %d | Output dim: %d",
             len(all_episodes_X), input_dim, n_outputs)

    # ------------------------------------------------------------------
    # 2.5 Train autoencoder on normalized inputs
    # ------------------------------------------------------------------
    log.info("Training autoencoder (input=%d latent=%d, epochs=%d)...",
             input_dim, ae_latent_dim, ae_epochs)
    autoencoder = NumpyAutoencoder(input_dim=input_dim,
                                   latent_dim=ae_latent_dim,
                                   hidden_dim=ae_hidden_dim)
    ae_losses = autoencoder.fit(X_norm_flat,
                                epochs=ae_epochs,
                                batch_size=512,
                                lr=1e-3,
                                weight_decay=1e-5,
                                log_every=max(1, ae_epochs // 10))
    encoded_episodes_X = [autoencoder.encode(ep) for ep in norm_episodes_X]

    # checkpoint after autoencoder training
    try:
        _save_checkpoint("post_ae", autoencoder=autoencoder, frictions=frictions, cpg_per_friction=cpg_per_friction)
    except Exception:
        pass

    n_inputs = ae_latent_dim
    log.info("Encoded episode input dim: %d", n_inputs)

    # ------------------------------------------------------------------
    # 3. Split train/val (80/20 per friction)
    # ------------------------------------------------------------------
    train_X, train_Y, val_X, val_Y = [], [], [], []
    for mu in frictions:
        idxs = [i for i, f in enumerate(friction_per_ep) if f == mu]
        split = max(1, int(len(idxs) * 0.8))
        for i in idxs[:split]:
            train_X.append(encoded_episodes_X[i])
            train_Y.append(norm_episodes_Y[i])
        for i in idxs[split:]:
            val_X.append(encoded_episodes_X[i])
            val_Y.append(norm_episodes_Y[i])

    log.info("Train episodes: %d | Val episodes: %d", len(train_X), len(val_X))

    # ------------------------------------------------------------------
    # 4. Hyperparameter search with correct MSE scoring
    # ------------------------------------------------------------------
    configs = [
        {"spectral_radius": 0.90, "input_scaling": 1.0, "leak_rate": 0.3, "ridge": 1e-4,  "name": "SR90-L03-R1e4"},
        {"spectral_radius": 0.95, "input_scaling": 1.0, "leak_rate": 0.3, "ridge": 1e-3,  "name": "SR95-L03-R1e3"},
        {"spectral_radius": 0.99, "input_scaling": 1.5, "leak_rate": 0.2, "ridge": 1e-3,  "name": "SR99-L02-R1e3"},
        {"spectral_radius": 0.99, "input_scaling": 2.0, "leak_rate": 0.1, "ridge": 1e-2,  "name": "SR99-L01-R1e2"},
        {"spectral_radius": 0.95, "input_scaling": 2.0, "leak_rate": 0.3, "ridge": 1e-2,  "name": "SR95-L03-R1e2"},
        {"spectral_radius": 0.99, "input_scaling": 1.0, "leak_rate": 0.5, "ridge": 1e-3,  "name": "SR99-L05-R1e3"},
    ]

    log.info("\nSearching hyperparameters (%d configs)...", len(configs))
    best_esn, best_cfg, best_mse = None, None, float("inf")

    for cfg in configs:
        log.info("  [%s]", cfg["name"])
        esn = EchoStateNetwork(
            n_inputs=n_inputs,
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
            best_cfg  = cfg
            best_esn  = esn

    log.info("\nBest config: %s (val MSE=%.6f)", best_cfg["name"], best_mse)

    # checkpoint after hyperparameter search
    try:
        _save_checkpoint("post_search", esn=best_esn, config=best_cfg, X_mean=X_mean, X_std=X_std, Y_mean=Y_mean, Y_std=Y_std, cpg_per_friction=cpg_per_friction, frictions=frictions)
    except Exception:
        pass

    # ------------------------------------------------------------------
    # 5. DAgger closed-loop fine-tuning
    # ------------------------------------------------------------------
    log.info("\nClosed-loop fine-tuning (DAgger, 5 rounds x %d episodes/friction)...",
             max(1, n_episodes // 5))
    encoded_norm_episodes_X = [autoencoder.encode(ep) for ep in norm_episodes_X]

    best_esn = closed_loop_finetune(
        best_esn,
        cpg_per_friction=cpg_per_friction,
        frictions=frictions,
        n_episodes=n_episodes,
        X_mean=X_mean, X_std=X_std,
        Y_mean=Y_mean, Y_std=Y_std,
        friction_per_ep=friction_per_ep,
        norm_episodes_X=encoded_norm_episodes_X,
        norm_episodes_Y=norm_episodes_Y,
        n_rounds=5,
        episodes_per_round=max(1, n_episodes // 5),
        washout=WASHOUT,
        autoencoder=autoencoder,
    )

    # ------------------------------------------------------------------
    # 6. Evaluate on all frictions
    # ------------------------------------------------------------------
    log.info("\nEvaluating on all frictions (5 episodes each)...")
    eval_results = []

    for mu in frictions:
        env = gym.make("Ant-v5", render_mode="rgb_array")
        set_floor_friction(env, mu)

        # Select per-friction readout
        if hasattr(best_esn, "W_out_per_friction") and mu in best_esn.W_out_per_friction:
            best_esn.W_out = best_esn.W_out_per_friction[mu]
            log.info("  [mu=%.1f] using per-friction readout (norm=%.4f)",
                     mu, np.linalg.norm(best_esn.W_out))

        rewards, distances = [], []
        for ep in range(5):
            obs, _ = env.reset()
            best_esn.reset()          # reset reservoir between episodes
            start_x = get_base_x(env)
            ep_reward = 0.0
            prev_obs = obs.copy()

            for step in range(500):
                vel = obs - prev_obs
                t_sec = step * 0.02
                cpg_ref = cpg_per_friction[mu]
                cpg_phase = (cpg_ref.omega * t_sec + cpg_ref.phases).astype(np.float64)
                cpg_out   = np.clip(cpg_ref.step(t_sec),
                                    env.action_space.low, env.action_space.high).astype(np.float64)
                obs_aug = np.concatenate([obs, vel, cpg_phase, cpg_out, [mu]]).astype(np.float64)
                obs_norm = (obs_aug - X_mean) / X_std
                obs_input = autoencoder.encode(obs_norm.reshape(1, -1)).reshape(-1)

                action_norm = best_esn.predict(obs_input)
                action = (action_norm * Y_std + Y_mean).astype(np.float32)
                action = np.clip(action, env.action_space.low, env.action_space.high)

                prev_obs = obs.copy()
                obs, reward, term, trunc, _ = env.step(action)
                ep_reward += reward
                if term or trunc:
                    break

            distances.append(get_base_x(env) - start_x)
            rewards.append(ep_reward)

        avg_dist   = float(np.mean(distances))
        avg_reward = float(np.mean(rewards))
        cpg_fwd    = cpg_stats[mu]["forward"]
        alignment  = (avg_dist / cpg_fwd * 100) if cpg_fwd != 0 else 0.0

        eval_results.append({
            "friction": mu, "rc_fwd": avg_dist, "rc_reward": avg_reward,
            "cpg_fwd": cpg_fwd, "alignment_pct": alignment,
        })
        log.info("  Friction %.1f: RC=%.3f, CPG=%.3f, Alignment=%.1f%%",
                 mu, avg_dist, cpg_fwd, alignment)
        env.close()

    avg_alignment = float(np.mean([r["alignment_pct"] for r in eval_results]))
    log.info("\nAverage Alignment: %.1f%%", avg_alignment)

    # ------------------------------------------------------------------
    # 7. Save
    # ------------------------------------------------------------------
    model_data = {
        "esn": best_esn,
        "autoencoder": autoencoder,
        "X_mean": X_mean, "X_std": X_std,
        "Y_mean": Y_mean, "Y_std": Y_std,
        "config": best_cfg,
        "eval_results": eval_results,
        "cpg_stats": cpg_stats,
        "cpg_per_friction": cpg_per_friction,
    }
    pkl_path = os.path.join(OUTPUT_DIR, "rc_80_optimized.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(model_data, f)
    log.info("Saved model to: %s", pkl_path)

    summary_path = os.path.join(OUTPUT_DIR, "summary.txt")
    with open(summary_path, "w") as f:
        f.write("Optimized RC for 80%+ Alignment\n")
        f.write("=" * 70 + "\n")
        f.write(f"Model: {pkl_path}\n")
        f.write(f"Config: {best_cfg}\n\n")
        f.write("Results:\n")
        for res in eval_results:
            f.write(f"  Friction {res['friction']}: {res['alignment_pct']:.1f}%"
                    f" (RC={res['rc_fwd']:.3f}, CPG={res['cpg_fwd']:.3f})\n")
        f.write(f"\nAverage Alignment: {avg_alignment:.1f}%\n")
    log.info("Saved summary to: %s", summary_path)
    log.info("\nDONE!")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-episodes",  type=int,   default=50)
    parser.add_argument("--n-reservoir", type=int,   default=1500)
    parser.add_argument("--cpg-iters",   type=int,   default=1000)
    parser.add_argument("--min-forward", type=float, default=1.0)
    parser.add_argument("--ae-hidden-dim", type=int, default=128)
    parser.add_argument("--ae-latent-dim", type=int, default=64)
    parser.add_argument("--ae-epochs", type=int, default=80)
    parser.add_argument("--visualize-model", action="store_true",
                        help="Load a saved model and record a single rollout with highlighted CPG switches")
    parser.add_argument("--model-path", type=str, default=os.path.join(OUTPUT_DIR, "rc_80_optimized.pkl"),
                        help="Path to the saved RC model pickle")
    parser.add_argument("--record-gif", type=str, default=None,
                        help="Optional output GIF path for the rollout recording")
    parser.add_argument("--switch-every", type=int, default=80,
                        help="Legacy step-based switch interval (kept for compatibility)")
    parser.add_argument("--switch-distance", type=float, default=0.18,
                        help="Distance between friction-surface zones that trigger a controller switch")
    parser.add_argument("--switch-sequence", type=str, default="0.5,1.0,1.5",
                        help="Comma-separated friction/CPG values to cycle through")
    parser.add_argument("--action-scale", type=float, default=1.2,
                        help="Scale factor for action magnitude to make the ant move faster")
    parser.add_argument("--cpg-speed-scale", type=float, default=1.0,
                        help="Scale factor applied to CPG frequency for visualization only")
    parser.add_argument("--cpg-amp-scale", type=float, default=1.0,
                        help="Scale factor applied to CPG amplitudes for visualization only")
    parser.add_argument("--action-smooth", type=float, default=0.0,
                        help="Smoothing factor [0..1) for visualizer actions (0=no smoothing, closer to 1 more smoothing)")
    parser.add_argument("--cpg-mix", type=float, default=0.0,
                        help="Blend factor [0..1] to mix CPG action into ESN action during visualization (helps continuous walking)")
    parser.add_argument("--episode-length", type=int, default=500,
                        help="Number of steps to render in the single rollout")
    parser.add_argument("--render-mode", type=str, default="rgb_array",
                        help="Gym render mode to use for the rollout (rgb_array by default)")
    args = parser.parse_args()

    if args.visualize_model:
        visualize_saved_model(
            model_path=args.model_path,
            episode_length=args.episode_length,
            output_gif=args.record_gif,
            switch_every=args.switch_every,
            switch_sequence=args.switch_sequence,
            render_mode=args.render_mode,
            switch_distance=args.switch_distance,
            action_scale=args.action_scale,
            cpg_speed_scale=args.cpg_speed_scale,
            cpg_amp_scale=args.cpg_amp_scale,
            action_smooth=args.action_smooth,
            cpg_mix=args.cpg_mix,
        )
    else:
        train_optimized_rc(
            n_episodes=args.n_episodes,
            n_reservoir=args.n_reservoir,
            cpg_iters=args.cpg_iters,
            min_forward=args.min_forward,
            ae_hidden_dim=args.ae_hidden_dim,
            ae_latent_dim=args.ae_latent_dim,
            ae_epochs=args.ae_epochs,
        )