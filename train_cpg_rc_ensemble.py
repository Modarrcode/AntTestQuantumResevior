"""
Train CPG and RC models on 3 selected frictions (0.5, 1.0, 1.5).
Save the best performing one for visualization.
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
                    log.warning("Could not set geom_friction for geom %s (index %d)", name, i)


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


def random_search_tune(env, n_actions, n_iters=1000, episode_length=500):
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


def train_and_evaluate(frictions):
    """Train CPG and RC on specified frictions, return best model."""
    print("DEBUG: Starting train_and_evaluate", flush=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"DEBUG: Output directory created: {OUTPUT_DIR}", flush=True)
    
    results = []
    
    for friction_idx, mu in enumerate(frictions):
        log.info("=" * 60)
        log.info("Training friction %.1f (index %d/%d)", mu, friction_idx + 1, len(frictions))
        log.info("=" * 60)
        
        # Train CPG
        log.info("Tuning CPG...")
        print(f"DEBUG: Creating Ant-v5 environment for friction {mu}", flush=True)
        env_cpg = gym.make("Ant-v5", render_mode="rgb_array")
        set_floor_friction(env_cpg, mu)
        best_vec, best_score = random_search_tune(env_cpg, env_cpg.action_space.shape[0], n_iters=100, episode_length=500)
        cpg = CPGController.from_vector(best_vec, env_cpg.action_space.shape[0])
        cpg_fwd, cpg_rew, _, cpg_max_speed, cpg_slip = evaluate_controller(env_cpg, cpg, episode_length=500)
        log.info("CPG: forward=%.3f, reward=%.3f, max_speed=%.3f", cpg_fwd, cpg_rew, cpg_max_speed)
        
        # Collect CPG data
        log.info("Collecting CPG training data (20 episodes)...")
        X_rc, Y_rc = [], []
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
                X_rc.append(obs)
                Y_rc.append(action)
                obs, reward, terminated, truncated, info = env_cpg.step(action)
                if terminated or truncated:
                    break
        X_rc = np.array(X_rc)
        Y_rc = np.array(Y_rc)
        
        # Normalize inputs
        input_mean = X_rc.mean(axis=0)
        input_std = X_rc.std(axis=0)
        input_std[input_std < 1e-8] = 1.0
        X_rc_norm = (X_rc - input_mean) / input_std
        
        env_cpg.close()
        
        # Train RC
        log.info("Training RC model...")
        env_rc = gym.make("Ant-v5")
        set_floor_friction(env_rc, mu)
        n_inputs = env_rc.observation_space.shape[0]
        n_outputs = env_rc.action_space.shape[0]
        
        np.random.seed(42 + friction_idx)
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
        esn.fit(X_rc_norm, Y_rc)
        
        # Evaluate RC
        rc_fwd, rc_rew, _, rc_max_speed, rc_slip = evaluate_rc_policy(env_rc, esn, episode_length=500, input_mean=input_mean, input_std=input_std)
        log.info("RC: forward=%.3f, reward=%.3f, max_speed=%.3f", rc_fwd, rc_rew, rc_max_speed)
        env_rc.close()
        
        # Save this model
        model_data = {
            "friction": mu,
            "cpg_params": best_vec,
            "cpg_fwd": cpg_fwd,
            "rc_model": esn,
            "input_mean": input_mean,
            "input_std": input_std,
            "rc_fwd": rc_fwd,
            "rc_rew": rc_rew,
            "rc_max_speed": rc_max_speed,
        }
        model_path = os.path.join(OUTPUT_DIR, f"model_friction_{mu}.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(model_data, f)
        log.info("Saved model to %s", model_path)
        
        results.append({
            "friction": mu,
            "cpg_fwd": cpg_fwd,
            "rc_fwd": rc_fwd,
            "rc_rew": rc_rew,
            "model_path": model_path,
        })
    
    # Find best
    best_result = max(results, key=lambda x: x["rc_fwd"])
    log.info("\n" + "=" * 60)
    log.info("BEST MODEL: Friction %.1f (RC forward=%.3f)", best_result["friction"], best_result["rc_fwd"])
    log.info("Model path: %s", best_result["model_path"])
    log.info("=" * 60)
    
    # Save summary
    summary_path = os.path.join(OUTPUT_DIR, "summary.txt")
    with open(summary_path, "w") as f:
        f.write("Training Summary\n")
        f.write("=" * 60 + "\n")
        for res in results:
            f.write(f"Friction {res['friction']}: CPG={res['cpg_fwd']:.3f}, RC={res['rc_fwd']:.3f}, Reward={res['rc_rew']:.3f}\n")
        f.write("\n" + "=" * 60 + "\n")
        f.write(f"BEST: Friction {best_result['friction']} with RC forward={best_result['rc_fwd']:.3f}\n")
        f.write(f"Model: {best_result['model_path']}\n")
    
    log.info("Summary saved to %s", summary_path)
    
    return best_result["model_path"]


if __name__ == "__main__":
    print("=" * 60, flush=True)
    print("STARTING TRAINING SCRIPT", flush=True)
    print("=" * 60, flush=True)
    best_model = train_and_evaluate(TRAIN_FRICTIONS)
    log.info("\nTo visualize the best model, run:")
    log.info("  python visualize_best_model.py %s", best_model)
