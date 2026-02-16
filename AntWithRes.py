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
from pathlib import Path

# Toggle debug for quicker runs and visible output
DEBUG = True

# Configure simple logging to stdout
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)
# NOTE: Avoid using `mat_gen.rand_sparse` which may not exist in some
# reservoirpy versions. Use NumPy to generate matrices for portability.

# Number of reservoir units (kept global)
n_reservoir = 500

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
        return np.concatenate(([self.omega], self.amplitudes, self.phases, self.offsets))

    def step(self, t: float):
        theta = self.omega * t + self.phases
        return self.offsets + self.amplitudes * np.sin(theta)


def get_base_x(env):
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
    return forward, total_reward, step + 1


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
        forward, _, _ = evaluate_controller(env, controller, episode_length=episode_length, render=render, frame_sleep=frame_sleep)
        if forward > best_score:
            best_score = forward
            best_vec = vec.copy()
            log.info("New best (iter %d): forward=%.3f", i, best_score)
    return best_vec, best_score


def generate_dataset(env, controller: CPGController, n_episodes: int = 10, episode_length: int = 500, out_path: str = "dataset.npz"):
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
    args = parser.parse_args()

    # Create environment with or without rendering
    log.info("Creating environment 'Ant-v5' (render=%s)", args.render)
    if args.render:
        env = gym.make("Ant-v5", render_mode="human")
    else:
        env = gym.make("Ant-v5")

    # Apply selected friction preset
    mu = FRICTIONS[args.friction_index]
    log.info("Setting floor friction to %.3f (index=%d)", mu, args.friction_index)
    set_floor_friction(env, mu)

    # If requested, generate datasets for specified training indices
    if args.generate_datasets:
        ds_dir = Path(args.dataset_dir)
        ds_dir.mkdir(parents=True, exist_ok=True)
        train_idxs = [int(x) for x in args.train_indices.split(",") if x.strip() != ""]
        for idx in train_idxs:
            if idx < 0 or idx >= len(FRICTIONS):
                log.warning("Skipping invalid friction index %s", idx)
                continue
            mu_i = FRICTIONS[idx]
            log.info("Generating dataset for friction index %d (mu=%.3f)", idx, mu_i)
            # create a fresh env for this friction
            try:
                env_i = gym.make("Ant-v5")
                set_floor_friction(env_i, mu_i)
                # tune a small CPG (keep tuning quick by default)
                tune_iters = args.tune_iters_per_env if hasattr(args, "tune_iters_per_env") else args.tune_iters
                best_vec, best_score = random_search_tune(env_i, env_i.action_space.shape[0], n_iters=tune_iters, episode_length=args.tune_length, render=args.render, frame_sleep=args.frame_sleep)
                if best_vec is None:
                    log.warning("No CPG found for idx %d; using default random CPG", idx)
                    best_vec = CPGController(env_i.action_space.shape[0]).vector()
                # save cpg params
                vec_path = ds_dir / f"cpg_idx{idx}.npy"
                np.save(vec_path, best_vec)
                log.info("Saved tuned CPG params to %s (score=%.3f)", vec_path, best_score)
                # generate dataset
                ds_path = ds_dir / f"dataset_idx{idx}.npz"
                controller = CPGController.from_vector(best_vec, env_i.action_space.shape[0])
                generate_dataset(env_i, controller, n_episodes=args.dataset_episodes, episode_length=args.dataset_length, out_path=str(ds_path))
            except Exception:
                log.exception("Failed to generate dataset for friction idx %d", idx)
            finally:
                try:
                    env_i.close()
                except Exception:
                    pass

    try:
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

        # If user requested to train models from dataset files, handle that now
        if args.train_from_datasets:
            model_dir = Path(args.model_dir)
            model_dir.mkdir(parents=True, exist_ok=True)
            ds_dir = Path(args.dataset_dir)
            npz_files = sorted(glob.glob(str(ds_dir / "*.npz")))
            if len(npz_files) == 0:
                log.warning("No dataset files (*.npz) found in %s", ds_dir)
            for ds in npz_files:
                try:
                    log.info("Training on dataset %s", ds)
                    model_dict = train_on_dataset(ds, n_reservoir_local=n_reservoir, washout=args.washout)
                    out_name = model_dir / (Path(ds).stem + ".npz")
                    np.savez(
                        out_name,
                        W=model_dict["W"],
                        Win=model_dict["Win"],
                        Wout=model_dict["Wout"],
                        input_mean=model_dict.get("input_mean"),
                        input_std=model_dict.get("input_std"),
                        state_mean=model_dict.get("state_mean"),
                        state_std=model_dict.get("state_std"),
                        ep_lens=model_dict.get("ep_lens"),
                        n_reservoir=model_dict.get("n_reservoir"),
                        washout=model_dict.get("washout"),
                        seed=model_dict.get("seed"),
                        dataset=model_dict.get("dataset"),
                        timestamp=model_dict.get("timestamp"),
                    )
                    log.info("Saved trained model arrays to %s", out_name)
                except Exception:
                    log.exception("Failed to train/save model for dataset %s", ds)

        # Evaluate saved models vs tuned CPGs on test indices
        if args.evaluate_models:
            model_dir = Path(args.model_dir)
            ds_dir = Path(args.dataset_dir)
            # load trained models that match dataset_idx{train_idx}.pkl
            # load .npz model files (W, Win, Wout)
            model_files = sorted(glob.glob(str(model_dir / "*.npz")))
            models = {}
            for mf in model_files:
                try:
                    stem = Path(mf).stem
                    if "dataset_idx" in stem:
                        tidx = int(stem.split("dataset_idx")[-1])
                    else:
                        tidx = stem
                    data = np.load(mf)
                    models[tidx] = {
                        "W": data["W"],
                        "Win": data["Win"],
                        "Wout": data["Wout"],
                        "input_mean": data.get("input_mean") if "input_mean" in data else None,
                        "input_std": data.get("input_std") if "input_std" in data else None,
                        "state_mean": data.get("state_mean") if "state_mean" in data else None,
                        "state_std": data.get("state_std") if "state_std" in data else None,
                        "ep_lens": data.get("ep_lens") if "ep_lens" in data else None,
                        "n_reservoir": int(data.get("n_reservoir")) if "n_reservoir" in data else None,
                        "washout": int(data.get("washout")) if "washout" in data else None,
                        "seed": int(data.get("seed")) if "seed" in data else None,
                        "dataset": str(data.get("dataset")) if "dataset" in data else None,
                        "timestamp": str(data.get("timestamp")) if "timestamp" in data else None,
                    }
                    log.info("Loaded model arrays %s for train_idx=%s", mf, tidx)
                except Exception:
                    log.exception("Failed to load model file %s", mf)

            # load tuned CPG params if available
            cpgs = {}
            cpg_files = sorted(glob.glob(str(ds_dir / "cpg_idx*.npy")))
            for cf in cpg_files:
                try:
                    stem = Path(cf).stem
                    tidx = int(stem.split("cpg_idx")[-1])
                    vec = np.load(cf)
                    cpgs[tidx] = CPGController.from_vector(vec, env.action_space.shape[0])
                    log.info("Loaded CPG params %s for train_idx=%d", cf, tidx)
                except Exception:
                    log.exception("Failed to load CPG params %s", cf)

            test_idxs = [int(x) for x in args.test_indices.split(",") if x.strip() != ""]
            results = []
            for test_idx in test_idxs:
                if test_idx < 0 or test_idx >= len(FRICTIONS):
                    log.warning("Skipping invalid test index %s", test_idx)
                    continue
                mu_test = FRICTIONS[test_idx]
                log.info("Evaluating on test friction index %d (mu=%.3f)", test_idx, mu_test)
                env_test = None
                try:
                    env_test = gym.make("Ant-v5")
                    set_floor_friction(env_test, mu_test)
                    for tidx, controller in cpgs.items():
                        fwd_sum = 0.0
                        rew_sum = 0.0
                        for ep in range(args.eval_episodes):
                            fwd, rew, steps = evaluate_controller(env_test, controller, episode_length=args.eval_length, render=args.render, frame_sleep=args.frame_sleep)
                            fwd_sum += fwd
                            rew_sum += rew
                        results.append({"test_idx": test_idx, "train_idx": tidx, "policy": "CPG", "forward": fwd_sum / args.eval_episodes, "reward": rew_sum / args.eval_episodes})

                    for tidx, model in models.items():
                        W = model["W"]
                        Win = model["Win"]
                        Wout = model["Wout"]
                        n_res_local = W.shape[0]
                        fwd_sum = 0.0
                        rew_sum = 0.0
                        for ep in range(args.eval_episodes):
                            obs, info = env_test.reset()
                            start_x = get_base_x(env_test)
                            total_r = 0.0
                            x = np.zeros((n_res_local,), dtype=float)
                            for step in range(args.eval_length):
                                u = np.asarray(obs).flatten()
                                # apply input normalization if available
                                if model.get("input_mean") is not None and model.get("input_std") is not None:
                                    u_norm = (u - model["input_mean"]) / model["input_std"]
                                else:
                                    u_norm = u
                                # reservoir update
                                x = np.tanh(W.dot(x) + Win.dot(u_norm))
                                # apply state normalization before readout if available
                                if model.get("state_mean") is not None and model.get("state_std") is not None:
                                    x_norm = (x - model["state_mean"]) / model["state_std"]
                                else:
                                    x_norm = x
                                x_aug = np.concatenate([x_norm, [1.0]])
                                action = Wout.dot(x_aug)
                                try:
                                    action = np.clip(action, env_test.action_space.low, env_test.action_space.high)
                                except Exception:
                                    action = np.clip(action, -1.0, 1.0)
                                obs, reward, terminated, truncated, info = env_test.step(action)
                                total_r += float(reward)
                                if terminated or truncated:
                                    break
                            end_x = get_base_x(env_test)
                            fwd_sum += (end_x - start_x)
                            rew_sum += total_r
                        results.append({"test_idx": test_idx, "train_idx": tidx, "policy": "Reservoir", "forward": fwd_sum / args.eval_episodes, "reward": rew_sum / args.eval_episodes})
                except Exception:
                    log.exception("Failed evaluation on test idx %d", test_idx)
                finally:
                    try:
                        if env_test is not None:
                            env_test.close()
                    except Exception:
                        pass

            # summarize results
            log.info("Evaluation summary:")
            for r in results:
                log.info(" test_idx=%s train_idx=%s policy=%s forward=%.3f reward=%.3f", r["test_idx"], r["train_idx"], r["policy"], r["forward"], r["reward"]) 

    except Exception as e:
        log.exception("Unhandled exception during execution: %s", e)
    finally:
        try:
            env.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()

