"""
Train 3 CPGs (frictions 0.5, 1.0, 1.5).
Collect data from all 3 CPGs.
Train a SINGLE RC model on combined data.
Evaluate on each friction and show best result.
"""
import logging
import numpy as np
import gymnasium as gym
from reservoirpy.nodes import Reservoir, Ridge
from reservoirpy.model import Model
import pickle
import os

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# Configuration
n_reservoir = 1000
TRAIN_FRICTIONS = [0.5, 1.0, 1.5]
OUTPUT_DIR = "best_models"

def set_floor_friction(env, mu: float):
    """Set floor geom friction to `mu` for geoms matching 'floor' or 'ground'."""
    try:
        model = env.unwrapped.model
    except Exception:
        log.warning("Could not access env.unwrapped.model to set friction")
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
        try:
            import mujoco
            try:
                nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
                return nm.decode("utf-8") if isinstance(nm, bytes) else str(nm)
            except Exception:
                pass
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


def get_base_x(env):
    """Get base X position."""
    try:
        dat = env.unwrapped.data
        if hasattr(dat, "qpos"):
            return float(dat.qpos[0])
    except Exception:
        pass
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
        A = vec[1:1 + n_actions]
        phi = vec[1 + n_actions:1 + 2 * n_actions]
        off = vec[1 + 2 * n_actions:1 + 3 * n_actions]
        return cls(n_actions, omega, A, phi, off)

    def step(self, t: float):
        theta = self.omega * t + self.phases
        return self.offsets + self.amplitudes * np.sin(theta)


def random_search_tune(env, n_actions, n_iters=100, episode_length=500):
    """Tune CPG parameters via random search."""
    best_vec = None
    best_score = -float('inf')
    for i in range(n_iters):
        omega = np.random.uniform(0.5, 4.0)
        A = np.random.uniform(0.0, 1.0, size=n_actions)
        phi = np.random.uniform(-np.pi, np.pi, size=n_actions)
        off = np.random.uniform(-0.5, 0.5, size=n_actions)
        vec = np.concatenate(([omega], A, phi, off))
        controller = CPGController.from_vector(vec, n_actions)
        forward, *_ = evaluate_controller(env, controller, episode_length=episode_length)
        if forward > best_score:
            best_score = forward
            best_vec = vec.copy()
            if (i + 1) % 100 == 0 or i == 0:
                log.info("  Iter %d: best forward=%.3f", i, best_score)
    return best_vec, best_score


def evaluate_controller(env, controller, episode_length=500):
    """Evaluate a CPG controller."""
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


def evaluate_rc_policy(env, esn, episode_length=500, input_mean=None, input_std=None):
    """Evaluate RC policy."""
    obs, info = env.reset()
    dt = 0.02
    start_x = get_base_x(env)
    total_reward = 0.0
    max_speed = 0.0
    slip_count = 0
    prev_x = start_x
    for step in range(episode_length):
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


def train_unified_rc():
    """Train 3 CPGs, collect combined data, train 1 RC on all data."""
    print("=== UNIFIED RC TRAINING ===", flush=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    cpg_controllers = {}
    all_X = []
    all_Y = []
    
    # Step 1: Train and collect data from all 3 CPGs
    log.info("STEP 1: Training 3 CPG controllers and collecting data...")
    for friction_idx, mu in enumerate(TRAIN_FRICTIONS):
        log.info("-" * 60)
        log.info("CPG %d: Friction %.1f", friction_idx + 1, mu)
        log.info("-" * 60)
        
        # Train CPG
        log.info("  Tuning CPG...")
        env_cpg = gym.make("Ant-v5", render_mode="rgb_array")
        set_floor_friction(env_cpg, mu)
        best_vec, best_score = random_search_tune(env_cpg, env_cpg.action_space.shape[0], n_iters=100, episode_length=500)
        cpg = CPGController.from_vector(best_vec, env_cpg.action_space.shape[0])
        cpg_fwd, cpg_rew, _, cpg_max_speed, cpg_slip = evaluate_controller(env_cpg, cpg, episode_length=500)
        log.info("  CPG performance: forward=%.3f, reward=%.3f", cpg_fwd, cpg_rew)
        
        # Collect data from this CPG
        log.info("  Collecting 20 episodes of CPG data...")
        X_cpg, Y_cpg = [], []
        for ep in range(20):
            obs, info = env_cpg.reset()
            for step in range(500):
                t = step * 0.02
                action = cpg.step(t)
                action = np.asarray(action).flatten()
                try:
                    action = np.clip(action, env_cpg.action_space.low, env_cpg.action_space.high)
                except Exception:
                    action = np.clip(action, -1.0, 1.0)
                X_cpg.append(obs)
                Y_cpg.append(action)
                obs, reward, terminated, truncated, info = env_cpg.step(action)
                if terminated or truncated:
                    break
        
        env_cpg.close()
        all_X.extend(X_cpg)
        all_Y.extend(Y_cpg)
        cpg_controllers[mu] = cpg
        log.info("  Collected %d samples from CPG (friction=%.1f)", len(X_cpg), mu)
    
    # Convert to arrays
    all_X = np.array(all_X)
    all_Y = np.array(all_Y)
    log.info("\nCombined dataset size: X=%s, Y=%s", all_X.shape, all_Y.shape)
    
    # Step 2: Train single RC on combined data
    log.info("\n" + "=" * 60)
    log.info("STEP 2: Training single RC on combined data from all 3 CPGs...")
    log.info("=" * 60)
    
    # Normalize inputs
    input_mean = all_X.mean(axis=0)
    input_std = all_X.std(axis=0)
    input_std[input_std < 1e-8] = 1.0
    X_norm = (all_X - input_mean) / input_std
    
    # Create environment to get dimensions
    env_dummy = gym.make("Ant-v5", render_mode="rgb_array")
    n_inputs = env_dummy.observation_space.shape[0]
    n_outputs = env_dummy.action_space.shape[0]
    env_dummy.close()
    
    # Train RC
    np.random.seed(42)
    Win = np.random.uniform(-2.0, 2.0, size=(n_reservoir, n_inputs))
    density = 0.1
    mask = (np.random.rand(n_reservoir, n_reservoir) < density)
    W = np.random.uniform(-0.5, 0.5, size=(n_reservoir, n_reservoir)) * mask.astype(float)
    eigvals = np.linalg.eigvals(W)
    max_abs = np.max(np.abs(eigvals))
    if max_abs > 1e-12:
        W *= 0.95 / max_abs
    else:
        W = np.random.uniform(-0.1, 0.1, size=(n_reservoir, n_reservoir))
    
    reservoir = Reservoir(units=n_reservoir, input_dim=n_inputs, Win=Win, W=W)
    readout = Ridge(input_dim=n_reservoir, output_dim=n_outputs, ridge=1e-6)
    esn = Model([reservoir, readout], edges=[(reservoir, 0, readout)])
    log.info("Training RC model on %d samples...", X_norm.shape[0])
    esn.fit(X_norm, all_Y)
    log.info("RC training complete!")
    
    # Step 3: Evaluate on each friction
    log.info("\n" + "=" * 60)
    log.info("STEP 3: Evaluating unified RC on each friction...")
    log.info("=" * 60)
    
    results = []
    for mu in TRAIN_FRICTIONS:
        log.info("\nEvaluating RC on friction %.1f...", mu)
        env = gym.make("Ant-v5", render_mode="rgb_array")
        set_floor_friction(env, mu)
        rc_fwd, rc_rew, _, rc_max_speed, rc_slip = evaluate_rc_policy(env, esn, episode_length=500, input_mean=input_mean, input_std=input_std)
        env.close()
        log.info("  RC: forward=%.3f, reward=%.3f, max_speed=%.3f", rc_fwd, rc_rew, rc_max_speed)
        results.append({
            "friction": mu,
            "rc_fwd": rc_fwd,
            "rc_rew": rc_rew,
            "rc_max_speed": rc_max_speed,
        })
    
    # Find best friction
    best_result = max(results, key=lambda x: x["rc_fwd"])
    
    log.info("\n" + "=" * 60)
    log.info("BEST PERFORMANCE: Friction %.1f (RC forward=%.3f)", best_result["friction"], best_result["rc_fwd"])
    log.info("=" * 60)
    
    # Save unified model
    model_data = {
        "type": "unified_rc",
        "frictions_trained_on": TRAIN_FRICTIONS,
        "rc_model": esn,
        "input_mean": input_mean,
        "input_std": input_std,
        "cpg_controllers": cpg_controllers,
        "best_friction": best_result["friction"],
        "results": results,
    }
    
    model_path = os.path.join(OUTPUT_DIR, "unified_rc_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model_data, f)
    log.info("Saved unified model to %s", model_path)
    
    # Save summary
    summary_path = os.path.join(OUTPUT_DIR, "unified_rc_summary.txt")
    with open(summary_path, "w") as f:
        f.write("UNIFIED RC TRAINING SUMMARY\n")
        f.write("=" * 60 + "\n")
        f.write(f"Single RC trained on combined data from 3 CPGs (frictions: {TRAIN_FRICTIONS})\n")
        f.write(f"Training data: {all_X.shape[0]} samples\n")
        f.write("\nPerformance on each friction:\n")
        for res in results:
            f.write(f"  Friction {res['friction']}: forward={res['rc_fwd']:.3f}, reward={res['rc_rew']:.3f}\n")
        f.write("\n" + "=" * 60 + "\n")
        f.write(f"BEST: Friction {best_result['friction']} with RC forward={best_result['rc_fwd']:.3f}\n")
        f.write(f"Model: {model_path}\n")
    
    log.info("Summary saved to %s", summary_path)
    
    return model_path


if __name__ == "__main__":
    best_model = train_unified_rc()
    log.info("\nTo visualize the best result, run:")
    log.info("  python visualize_unified_rc.py %s", best_model)
