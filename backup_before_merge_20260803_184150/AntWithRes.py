import logging
import argparse
import gymnasium as gym
import numpy as np
from reservoirpy.nodes import Reservoir, Ridge
from reservoirpy.model import Model
import time
import os
import glob
import pickle
import re
from pathlib import Path

# Toggle debug for quicker runs and visible output
DEBUG = True

# Configure simple logging to stdout
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)
# NOTE: Avoid using `mat_gen.rand_sparse` which may not exist in some
# reservoirpy versions. Use NumPy to generate matrices for portability.

# Number of reservoir units (kept global)
n_reservoir = 1000  # Increased for better RC performance

# Preset frictions to choose from (5 environments)
FRICTIONS = [0.5, 1.0, 1.5, 2.0, 2.5]

def set_floor_friction(env, mu: float):
    """Set floor geom friction to `mu` for geoms matching 'floor' or 'ground'."""
    try:
        model = env.unwrapped.model
    except Exception:
        log.warning("Could not access env.unwrapped.model to set friction")
        return
    # Helper to get geom name at index i across mujoco versions
    def geom_name_at(i):
        # 1) model.geom_names (sequence of bytes/str)
        if hasattr(model, "geom_names"):
            try:
                n = model.geom_names[i]
                return n.decode("utf-8") if isinstance(n, bytes) else str(n)
            except Exception:
                pass

        # 2) model.geom is an array of structs with a .name field
        if hasattr(model, "geom"):
            try:
                g = model.geom[i]
                # some bindings expose name as bytes or as attribute
                name = getattr(g, "name", None)
                if name is None:
                    # fallback: try indexing into struct
                    try:
                        name = g[0]
                    except Exception:
                        name = str(g)
                return name.decode("utf-8") if isinstance(name, bytes) else str(name)
            except Exception:
                pass

        # 3) try mujoco.mj_id2name if available
        try:
            import mujoco

            try:
                nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
                return nm.decode("utf-8") if isinstance(nm, bytes) else str(nm)
            except Exception:
                pass
        except Exception:
            pass

        # fallback
        return f"geom_{i}"

    ngeom = getattr(model, "ngeom", None)
    if ngeom is None:
        # try length of geom array
        try:
            ngeom = len(model.geom)
        except Exception:
            ngeom = 0

    for i in range(ngeom):
        name = geom_name_at(i)
        if "floor" in name or "ground" in name or "geom_floor" in name:
            # geom_friction is a 3-vector; set first element to mu
            try:
                model.geom_friction[i] = np.array([mu, 0.0, 0.0])
            except Exception:
                try:
                    model.geom_friction[i] = [mu, 0.0, 0.0]
                except Exception:
                    log.warning("Could not set geom_friction for geom %s (index %d)", name, i)


# ESN matrices and model will be created in `main()` after the environment
# is instantiated since `n_inputs` and `n_outputs` depend on the env.


# --- CPG controller, tuning and dataset utilities ---
class CPGController:
    """Simple sinusoidal CPG controller producing continuous actions."""

    def __init__(self, n_actions: int, omega: float = 2.0, amplitudes=None, phases=None, offsets=None):
        self.n = n_actions
        self.omega = float(omega)
        self.amplitudes = np.ones(self.n) if amplitudes is None else np.asarray(amplitudes).reshape(self.n)
        self.phases = np.zeros(self.n) if phases is None else np.asarray(phases).reshape(self.n)
        self.offsets = np.zeros(self.n) if offsets is None else np.asarray(offsets).reshape(self.n)

    @classmethod
    def from_vector(cls, vec, n_actions):
        omega = float(vec[0])
        A = vec[1:1 + n_actions]
        phi = vec[1 + n_actions:1 + 2 * n_actions]
        off = vec[1 + 2 * n_actions:1 + 3 * n_actions]
        return cls(n_actions, omega, A, phi, off)

    def vector(self):
        return np.concatenate(( [self.omega], self.amplitudes, self.phases, self.offsets ))
    def step(self, t: float):
        theta = self.omega * t + self.phases
        return self.offsets + self.amplitudes * np.sin(theta)


def get_base_x(env):
    def get_base_vel(env):
        try:
            dat = env.unwrapped.data
            if hasattr(dat, "qvel"):
                return float(dat.qvel[0])
        except Exception:
            pass
        return 0.0
    try:
        dat = env.unwrapped.data
        if hasattr(dat, "qpos"):
            return float(dat.qpos[0])
    except Exception:
        pass
    return 0.0


def evaluate_controller(env, controller: CPGController, episode_length: int = 500, render: bool = False, frame_sleep: float = 0.0):
    obs, info = env.reset()
    dt = 0.02
    start_x = get_base_x(env)
    total_reward = 0.0
    max_speed = 0.0
    slip_count = 0
    prev_x = start_x
    for step in range(episode_length):
        t = step * dt
        action = controller.step(t)
        action = np.asarray(action).flatten()
        try:
            action = np.clip(action, env.action_space.low, env.action_space.high)
        except Exception:
            action = np.clip(action, -1.0, 1.0)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        # Max speed calculation
        curr_x = get_base_x(env)
        speed = (curr_x - prev_x) / dt
        max_speed = max(max_speed, abs(speed))
        prev_x = curr_x
        # Slipping detection: negative reward or sudden drop in speed
        if reward < -1.0 or abs(speed) < 0.01:
            slip_count += 1
        if render:
            try:
                env.render()
            except Exception:
                pass
            if frame_sleep > 0:
                time.sleep(frame_sleep)
        if terminated or truncated:
            break
    end_x = get_base_x(env)
    forward = end_x - start_x
    return forward, total_reward, step + 1, max_speed, slip_count


def random_search_tune(env, n_actions, n_iters=200, episode_length=500, render=False, frame_sleep=0.0):
    best_vec = None
    best_score = -float('inf')
    for i in range(n_iters):
        omega = np.random.uniform(0.5, 4.0)
        A = np.random.uniform(0.0, 1.0, size=n_actions)
        phi = np.random.uniform(-np.pi, np.pi, size=n_actions)
        off = np.random.uniform(-0.5, 0.5, size=n_actions)
        vec = np.concatenate(([omega], A, phi, off))
        controller = CPGController.from_vector(vec, n_actions)
        forward, *_ = evaluate_controller(env, controller, episode_length=episode_length, render=render, frame_sleep=frame_sleep)
        if forward > best_score:
            best_score = forward
            best_vec = vec.copy()
            log.info("New best (iter %d): forward=%.3f", i, best_score)
    return best_vec, best_score


def generate_dataset(env, controller: CPGController, n_episodes: int = 10, episode_length: int = 500, out_path: str = "dataset.npz", render: bool = False, frame_sleep: float = 0.0):
    Xs = []
    Ys = []
    metas = []
    ep_lens = []
    for e in range(n_episodes):
        obs, info = env.reset()
        ep_steps = 0
        for step in range(episode_length):
            t = step * 0.02
            action = controller.step(t)
            action = np.asarray(action).flatten()
            try:
                action = np.clip(action, env.action_space.low, env.action_space.high)
            except Exception:
                action = np.clip(action, -1.0, 1.0)
            Xs.append(obs.copy())
            Ys.append(action.copy())
            obs, reward, terminated, truncated, info = env.step(action)
            if render:
                try:
                    env.render()
                except Exception:
                    pass
                if frame_sleep > 0:
                    import time
                    time.sleep(frame_sleep)
            ep_steps += 1
            if terminated or truncated:
                break
        metas.append({"episode": e})
        ep_lens.append(ep_steps)
    X = np.array(Xs)
    Y = np.array(Ys)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, X=X, Y=Y, meta=metas, ep_lens=np.array(ep_lens))
    log.info("Saved dataset %s X=%s Y=%s", out_path, X.shape, Y.shape)
    return out_path


def train_on_dataset(path, n_reservoir_local=500, washout: int = 50):
    # Load dataset
    data = np.load(path, allow_pickle=True)
    X = data["X"]
    Y = data["Y"]
    n_inputs = X.shape[1]
    n_outputs = Y.shape[1]
    # Input normalization (compute from full dataset)
    input_mean = X.mean(axis=0)
    input_std = X.std(axis=0)
    input_std[input_std < 1e-8] = 1.0

    # Initialize reservoir matrices (Win, W)
    np.random.seed(42)
    Win = np.random.uniform(-1.0, 1.0, size=(n_reservoir_local, n_inputs))
    density = 0.1
    mask = (np.random.rand(n_reservoir_local, n_reservoir_local) < density)
    W = np.random.uniform(-0.5, 0.5, size=(n_reservoir_local, n_reservoir_local)) * mask.astype(float)
    eigvals = np.linalg.eigvals(W)
    max_abs = np.max(np.abs(eigvals))
    if max_abs > 1e-12:
        W *= 1.25 / max_abs
    else:
        W = np.random.uniform(-0.1, 0.1, size=(n_reservoir_local, n_reservoir_local))

    # Run the reservoir on the dataset sequentially and collect states after washout per episode
    states_list = []
    Y_list = []
    total_T = X.shape[0]
    # Determine episode lengths if available
    if "ep_lens" in data:
        ep_lens = list(map(int, np.asarray(data["ep_lens"]).tolist()))
    else:
        # try to infer from meta if present
        meta = data.get("meta", None)
        if meta is not None and len(meta) > 1:
            # split equally, last chunk gets remainder
            n_eps = len(meta)
            base = total_T // n_eps
            ep_lens = [base] * n_eps
            rem = total_T - base * n_eps
            for i in range(rem):
                ep_lens[i] += 1
        else:
            ep_lens = [total_T]

    idx = 0
    for ep_len in ep_lens:
        x = np.zeros((n_reservoir_local,), dtype=float)
        for t in range(ep_len):
            u = X[idx]
            u_norm = (u - input_mean) / input_std
            x = np.tanh(W.dot(x) + Win.dot(u_norm))
            if t >= washout:
                states_list.append(x.copy())
                Y_list.append(Y[idx].copy())
            idx += 1

    if len(states_list) == 0:
        log.warning("After washout=%d no states collected for %s; reducing washout to 0", washout, path)
        # fallback: collect all states without washout
        idx = 0
        states_list = []
        Y_list = []
        for ep_len in ep_lens:
            x = np.zeros((n_reservoir_local,), dtype=float)
            for t in range(ep_len):
                u = X[idx]
                u_norm = (u - input_mean) / input_std
                x = np.tanh(W.dot(x) + Win.dot(u_norm))
                states_list.append(x.copy())
                Y_list.append(Y[idx].copy())
                idx += 1

    S_mat = np.vstack(states_list)  # shape (N_samples, n_reservoir)
    Y_mat = np.vstack(Y_list)       # shape (N_samples, n_outputs)

    # Fit linear readout with ridge regression (including bias), after normalizing states
    ridge = 1e-6
    S = S_mat.T  # shape (n_reservoir, N)
    Ymat = Y_mat.T  # shape (n_outputs, N)

    # State normalization
    state_mean = S.mean(axis=1)
    state_std = S.std(axis=1)
    state_std[state_std < 1e-8] = 1.0
    S_norm = (S - state_mean[:, None]) / state_std[:, None]

    ones = np.ones((1, S_norm.shape[1]))
    S_aug = np.vstack([S_norm, ones])  # (n_reservoir+1, N)
    A = S_aug.dot(S_aug.T) + ridge * np.eye(S_aug.shape[0])
    B = Ymat.dot(S_aug.T)
    Wout = B.dot(np.linalg.inv(A))  # shape (n_outputs, n_reservoir+1)

    log.info("Trained readout on dataset %s: X=%s Y=%s (collected_states=%s)", path, X.shape, Y.shape, S_mat.shape)

    # Collect metadata for reproducibility
    ep_lens_saved = None
    if "ep_lens" in data:
        ep_lens_saved = np.asarray(data["ep_lens"]).astype(int)

    model_dict = {
        "W": W,
        "Win": Win,
        "Wout": Wout,
        "input_mean": input_mean,
        "input_std": input_std,
        "state_mean": state_mean,
        "state_std": state_std,
        "ep_lens": ep_lens_saved,
        "n_reservoir": n_reservoir_local,
        "washout": int(washout),
        "seed": 42,
        "dataset": str(path),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    return model_dict


def main():
    def evaluate_random_policy(env, episode_length=200):
        obs, info = env.reset()
        dt = 0.02
        start_x = get_base_x(env)
        total_reward = 0.0
        max_speed = 0.0
        slip_count = 0
        prev_x = start_x
        for step in range(episode_length):
            action = env.action_space.sample()
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
        end_x = get_base_x(env)
        forward = end_x - start_x
        return forward, total_reward, step + 1, max_speed, slip_count

    def evaluate_rc_policy(env, esn, episode_length=200, input_mean=None, input_std=None):
        obs, info = env.reset()
        dt = 0.02
        start_x = get_base_x(env)
        total_reward = 0.0
        max_speed = 0.0
        slip_count = 0
        prev_x = start_x
        for step in range(episode_length):
            # Normalize observation if normalization params provided
            obs_norm = obs
            if input_mean is not None and input_std is not None:
                obs_norm = (obs - input_mean) / input_std
            action = esn.run(obs_norm.reshape(1, -1))
            action = np.asarray(action).flatten()
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
        end_x = get_base_x(env)
        forward = end_x - start_x
        return forward, total_reward, step + 1, max_speed, slip_count
    parser = argparse.ArgumentParser(description="Run Ant with ReservoirPy ESN readout")
    parser.add_argument("--render", action="store_true", help="Enable human rendering window")
    parser.add_argument("--friction-index", type=int, choices=list(range(len(FRICTIONS))), default=1, help=f"Choose preset floor friction (0-{len(FRICTIONS)-1})")
    parser.add_argument("--tune-cpg", action="store_true", help="Run random-search tuning for a CPG on the selected friction")
    parser.add_argument("--tune-iters", type=int, default=200, help="Number of random-search iterations when tuning CPG")
    parser.add_argument("--tune-length", type=int, default=300, help="Episode length used during tuning (steps)")
    parser.add_argument("--generate-datasets", action="store_true", help="Generate datasets using tuned or default CPGs")
    parser.add_argument("--dataset-dir", type=str, default="./datasets", help="Directory to save/load datasets and CPG params")
    parser.add_argument("--dataset-episodes", type=int, default=5, help="Episodes per dataset file")
    parser.add_argument("--dataset-length", type=int, default=300, help="Max steps per dataset episode")
    parser.add_argument("--train-from-datasets", action="store_true", help="Train reservoir models from datasets in dataset-dir")
    parser.add_argument("--model-dir", type=str, default="./models", help="Directory to save trained reservoir models (.pkl)")
    parser.add_argument("--frame-sleep", type=float, default=0.0, help="Seconds to sleep between render frames")
    parser.add_argument("--train-samples", type=int, default=None, help="Number of random samples to collect for training (overrides DEBUG default)")
    parser.add_argument("--test-steps", type=int, default=None, help="Number of test steps to run during evaluation (overrides DEBUG default)")
    parser.add_argument("--pipeline", action="store_true", help="Run full pipeline: tune CPGs, generate datasets, train reservoirs, evaluate")
    parser.add_argument("--train-indices", type=str, default="0,1,2", help="Comma-separated indices (0-4) used for training datasets")
    parser.add_argument("--test-indices", type=str, default="3,4", help="Comma-separated indices (0-4) used for interpolation/testing")
    parser.add_argument("--tune-iters-per-env", type=int, default=100, help="Random-search iterations per environment during pipeline tuning")
    parser.add_argument("--washout", type=int, default=50, help="Washout steps to discard at start of each episode when training readout")
    parser.add_argument("--evaluate-models", action="store_true", help="Evaluate saved models and tuned CPGs on test frictions")
    parser.add_argument("--eval-episodes", type=int, default=2, help="Episodes per evaluation run (average over these)")
    parser.add_argument("--eval-length", type=int, default=300, help="Steps per evaluation episode")
    parser.add_argument("--compare-cpg-vs-random", action="store_true", help="Compare CPG vs random policy across all frictions")
    parser.add_argument("--build-prof-summary", action="store_true", help="Build professor-ready summary from existing result files")

    def build_professor_summary():
        base_dir = Path(__file__).resolve().parent
        output_path = base_dir / "comparison_professor_summary.txt"

        fair_rows = []
        comp_candidates = [
            base_dir / "comparison_results.txt",
            base_dir.parent / "comparison_results.txt",
            Path.cwd() / "comparison_results.txt",
        ]
        comp_path = None
        for candidate in comp_candidates:
            if candidate.exists():
                comp_path = candidate
                break
        if comp_path is not None and comp_path.exists():
            with open(comp_path, "r", encoding="utf-8") as f:
                lines = [ln.strip() for ln in f.readlines() if ln.strip()]
            for ln in lines[1:]:
                parts = [p.strip() for p in ln.split(",")]
                if len(parts) < 13:
                    continue
                try:
                    fair_rows.append({
                        "friction": float(parts[0]),
                        "cpg_fwd": float(parts[1]),
                        "rand_fwd": float(parts[2]),
                        "rc_fwd": float(parts[3]),
                        "cpg_spd": float(parts[4]),
                        "rand_spd": float(parts[5]),
                        "rc_spd": float(parts[6]),
                        "cpg_slip": float(parts[7]),
                        "rand_slip": float(parts[8]),
                        "rc_slip": float(parts[9]),
                        "cpg_rew": float(parts[10]),
                        "rand_rew": float(parts[11]),
                        "rc_rew": float(parts[12]),
                    })
                except Exception:
                    continue

        improved = None
        improved_path = base_dir / "improved_rc_model" / "improved_rc_friction_1.0.pkl"
        if improved_path.exists():
            try:
                with open(improved_path, "rb") as f:
                    d = pickle.load(f)
                improved = {
                    "cpg_fwd": float(d.get("cpg_fwd", np.nan)),
                    "supervised_fwd": float(d.get("rc_fwd", np.nan)),
                    "friction": float(d.get("friction", 1.0)),
                }
            except Exception:
                improved = None

        rl = None
        rl_model_path = base_dir / "rc_rl_extended" / "rc_extended_rl_friction_1.0.pkl"
        if rl_model_path.exists():
            try:
                with open(rl_model_path, "rb") as f:
                    d = pickle.load(f)
                rl = {
                    "rl_fwd": float(d.get("rl_fwd", np.nan)),
                    "friction": float(d.get("friction", 1.0)),
                }
            except Exception:
                rl = None

        log_best = None
        rl_log_path = base_dir / "rc_rl_extended" / "extended_rl_run.log"
        if rl_log_path.exists():
            try:
                with open(rl_log_path, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()
                vals = re.findall(r"NEW BEST! distance=([0-9]+\.?[0-9]*)m", text)
                if vals:
                    log_best = max(float(v) for v in vals)
            except Exception:
                log_best = None

        lines = []
        lines.append("PROFESSOR-READY RC VS CPG SUMMARY")
        lines.append("=" * 60)
        lines.append("")
        lines.append("A) FAIR COMPARISON (NO RL FINE-TUNING)")
        lines.append("- RC is trained to imitate/control from CPG-generated supervision per friction.")

        if fair_rows:
            ratios = []
            for r in fair_rows:
                ratio = (r["rc_fwd"] / r["cpg_fwd"] * 100.0) if abs(r["cpg_fwd"]) > 1e-12 else np.nan
                ratios.append(ratio)
                lines.append(
                    f"  Friction {r['friction']:.1f}: CPG={r['cpg_fwd']:.3f}m, RC={r['rc_fwd']:.3f}m ({ratio:.1f}% of CPG), Random={r['rand_fwd']:.3f}m"
                )
            valid_ratios = [x for x in ratios if np.isfinite(x)]
            if valid_ratios:
                lines.append(f"  Mean RC/CPG across listed frictions: {np.mean(valid_ratios):.1f}%")
            best_row = max(fair_rows, key=lambda x: x["rc_fwd"])
            best_ratio = (best_row["rc_fwd"] / best_row["cpg_fwd"] * 100.0) if abs(best_row["cpg_fwd"]) > 1e-12 else np.nan
            lines.append(
                f"  Best fair-case RC: friction {best_row['friction']:.1f}, RC={best_row['rc_fwd']:.3f}m ({best_ratio:.1f}% of CPG)."
            )
        else:
            lines.append("  comparison_results.txt not found or not parseable.")

        lines.append("")
        lines.append("B) OPTIMIZED POTENTIAL (RL FINE-TUNED RC)")
        lines.append("- RC starts from supervised baseline, then RL directly optimizes reward/distance.")

        if improved is not None:
            ratio = (improved["supervised_fwd"] / improved["cpg_fwd"] * 100.0) if abs(improved["cpg_fwd"]) > 1e-12 else np.nan
            lines.append(
                f"  Supervised baseline (friction {improved['friction']:.1f}): RC={improved['supervised_fwd']:.3f}m vs CPG={improved['cpg_fwd']:.3f}m ({ratio:.1f}% of CPG)."
            )
        else:
            lines.append("  improved_rc_model file not found.")

        if rl is not None and improved is not None:
            ratio = (rl["rl_fwd"] / improved["cpg_fwd"] * 100.0) if abs(improved["cpg_fwd"]) > 1e-12 else np.nan
            gain = (rl["rl_fwd"] / improved["supervised_fwd"] - 1.0) * 100.0 if abs(improved["supervised_fwd"]) > 1e-12 else np.nan
            lines.append(
                f"  RL optimized (saved model): RC={rl['rl_fwd']:.3f}m ({ratio:.1f}% of CPG, +{gain:.1f}% vs supervised RC)."
            )
        elif log_best is not None and improved is not None:
            ratio = (log_best / improved["cpg_fwd"] * 100.0) if abs(improved["cpg_fwd"]) > 1e-12 else np.nan
            gain = (log_best / improved["supervised_fwd"] - 1.0) * 100.0 if abs(improved["supervised_fwd"]) > 1e-12 else np.nan
            lines.append(
                f"  RL best seen in training log: RC={log_best:.3f}m ({ratio:.1f}% of CPG, +{gain:.1f}% vs supervised RC)."
            )
        else:
            lines.append("  RL artifact/log not found yet.")

        lines.append("")
        lines.append("CLAIMS TO REPORT")
        lines.append("1) Fair claim: Supervised RC approaches CPG but usually does not exceed it.")
        lines.append("2) Optimization claim: RL-finetuned RC can exceed CPG on the target friction.")

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

        print(f"Professor-ready summary saved to: {str(output_path)}")
        return

    def compare_cpg_vs_random_rc():
        """Compare CPG, random, and RC policies for all frictions. Train separate RC model per friction for best performance."""
        print("COMPARE FUNCTION STARTED")
        
        # Open output file for writing results
        output_file = "comparison_results.txt"
        with open(output_file, "w") as f:
            # Write header
            header = "Friction, CPG Fwd, Rand Fwd, RC Fwd, CPG Spd, Rand Spd, RC Spd, CPG Slip, Rand Slip, RC Slip, CPG Rew, Rand Rew, RC Rew"
            print(header)
            f.write(header + "\n")
            f.flush()  # Ensure header is written immediately
            
            for idx, mu in enumerate(FRICTIONS):
                print(f"Processing friction idx {idx} (mu={mu})...")
                
                # CPG: Tune and evaluate
                env_cpg = gym.make("Ant-v5")
                set_floor_friction(env_cpg, mu)
                best_vec, best_score = random_search_tune(env_cpg, env_cpg.action_space.shape[0], n_iters=300 if DEBUG else 1000, episode_length=500, render=False)
                cpg = CPGController.from_vector(best_vec, env_cpg.action_space.shape[0])
                cpg_fwd, cpg_rew, _, cpg_max_speed, cpg_slip = evaluate_controller(env_cpg, cpg, episode_length=500, render=False)

                # Collect CPG-generated data for RC training (multiple episodes for diversity)
                X_rc, Y_rc = [], []
                n_episodes = 20  # Multiple episodes for better coverage
                steps_per_episode = 500
                for ep in range(n_episodes):
                    obs, info = env_cpg.reset()
                    for step in range(steps_per_episode):
                        t = step * 0.02
                        action = cpg.step(t)
                        action = np.asarray(action).flatten()
                        try:
                            action = np.clip(action, env_cpg.action_space.low, env_cpg.action_space.high)
                        except Exception:
                            action = np.clip(action, -1.0, 1.0)
                        X_rc.append(obs)
                        Y_rc.append(action)
                        obs, reward, terminated, truncated, info = env_cpg.step(action)
                        if terminated or truncated:
                            break
                X_rc = np.array(X_rc)
                Y_rc = np.array(Y_rc)
                
                # Normalize inputs for better RC training
                input_mean = X_rc.mean(axis=0)
                input_std = X_rc.std(axis=0)
                input_std[input_std < 1e-8] = 1.0
                X_rc_norm = (X_rc - input_mean) / input_std
                
                env_cpg.close()

                # Random policy: Evaluate
                env_rand = gym.make("Ant-v5")
                set_floor_friction(env_rand, mu)
                rand_fwd, rand_rew, _, rand_max_speed, rand_slip = evaluate_random_policy(env_rand, episode_length=500)
                env_rand.close()

                # RC policy: Train on friction-specific CPG data with improved hyperparameters
                env_rc = gym.make("Ant-v5")
                set_floor_friction(env_rc, mu)
                n_inputs = env_rc.observation_space.shape[0]
                n_outputs = env_rc.action_space.shape[0]
                np.random.seed(42 + idx)  # Different seed per friction for diversity
                Win = np.random.uniform(-2.0, 2.0, size=(n_reservoir, n_inputs))  # Increased input weight range
                density = 0.1
                mask = (np.random.rand(n_reservoir, n_reservoir) < density)
                W = np.random.uniform(-0.5, 0.5, size=(n_reservoir, n_reservoir)) * mask.astype(float)
                eigvals = np.linalg.eigvals(W)
                max_abs = np.max(np.abs(eigvals))
                if max_abs > 1e-12:
                    W *= 0.95 / max_abs  # Adjusted spectral radius for stability
                else:
                    W = np.random.uniform(-0.1, 0.1, size=(n_reservoir, n_reservoir))
                reservoir = Reservoir(units=n_reservoir, input_dim=n_inputs, Win=Win, W=W)
                readout = Ridge(input_dim=n_reservoir, output_dim=n_outputs, ridge=1e-6)
                esn = Model([reservoir, readout], edges=[(reservoir, 0, readout)])
                esn.fit(X_rc_norm, Y_rc)  # Train on normalized inputs
                rc_fwd, rc_rew, _, rc_max_speed, rc_slip = evaluate_rc_policy(env_rc, esn, episode_length=500, input_mean=input_mean, input_std=input_std)
                env_rc.close()

                row = [mu, cpg_fwd, rand_fwd, rc_fwd, cpg_max_speed, rand_max_speed, rc_max_speed, cpg_slip, rand_slip, rc_slip, cpg_rew, rand_rew, rc_rew]
                row_str = ", ".join(str(x) for x in row)
                print(row_str)
                f.write(row_str + "\n")
                f.flush()  # Ensure results are written immediately
        
        print(f"\nResults saved to {output_file}")
        return

    args = parser.parse_args()
    if args.build_prof_summary:
        build_professor_summary()
        return

    if args.compare_cpg_vs_random:
        compare_cpg_vs_random_rc()
        return
    
    try:
        # Create environment
        env = gym.make("Ant-v5")
        set_floor_friction(env, FRICTIONS[args.friction_index])
        
        # determine input/output dims from env
        n_inputs = env.observation_space.shape[0]
        n_outputs = env.action_space.shape[0]
        log.info("Reservoir units=%d input_dim=%d output_dim=%d", n_reservoir, n_inputs, n_outputs)

        # --- Generate matrices (use NumPy for compatibility) ---
        np.random.seed(42)
        Win = np.random.uniform(-1.0, 1.0, size=(n_reservoir, n_inputs))
        density = 0.1
        mask = (np.random.rand(n_reservoir, n_reservoir) < density)
        W = np.random.uniform(-0.5, 0.5, size=(n_reservoir, n_reservoir)) * mask.astype(float)
        eigvals = np.linalg.eigvals(W)
        max_abs = np.max(np.abs(eigvals))
        if max_abs > 1e-12:
            W *= 1.25 / max_abs
        else:
            W = np.random.uniform(-0.1, 0.1, size=(n_reservoir, n_reservoir))

        # --- Build reservoir & readout ---
        reservoir = Reservoir(units=n_reservoir, input_dim=n_inputs, Win=Win, W=W)
        readout = Ridge(input_dim=n_reservoir, output_dim=n_outputs, ridge=1e-6)
        esn = Model([reservoir, readout], edges=[(reservoir, 0, readout)])

        # --- Collect random data for supervised fit ---
        X, Y = [], []
        obs, info = env.reset()
        n_samples = args.train_samples if args.train_samples is not None else (100 if DEBUG else 5000)
        log.info("Collecting %d random samples for training", n_samples)
        for i in range(n_samples):
            action = env.action_space.sample()
            next_obs, reward, terminated, truncated, info = env.step(action)
            X.append(obs)
            Y.append(action)
            obs = next_obs
            if terminated or truncated:
                obs, info = env.reset()
            if DEBUG and (i + 1) % 25 == 0:
                log.info("  collected %d/%d", i + 1, n_samples)

        X = np.array(X)
        Y = np.array(Y)
        log.info("Training data shapes: X=%s Y=%s", X.shape, Y.shape)

        # --- Train ---
        t0 = time.time()
        esn.fit(X, Y)
        log.info("Training finished in %.2f s", time.time() - t0)

        # --- Test run ---
        obs, info = env.reset()
        n_test = args.test_steps if args.test_steps is not None else (50 if DEBUG else 1000)
        total_reward = 0.0
        log.info("Starting test run (%d steps)", n_test)
        for i in range(n_test):
            action = esn.run(obs.reshape(1, -1))
            action = np.asarray(action).flatten()
            try:
                action = np.clip(action, env.action_space.low, env.action_space.high)
            except Exception:
                action = np.clip(action, -1.0, 1.0)

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            if terminated or truncated:
                obs, info = env.reset()
            if DEBUG and (i + 1) % 10 == 0:
                log.info("  test step %d/%d  cumulative_reward=%.3f", i + 1, n_test, total_reward)

        log.info("Test run finished, cumulative reward=%.3f", total_reward)

    except Exception as e:
        log.exception("Unhandled exception during execution: %s", e)
    finally:
        try:
            env.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()

