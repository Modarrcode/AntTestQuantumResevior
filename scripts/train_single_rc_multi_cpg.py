"""
Train 3 CPG controllers (frictions 0.5, 1.0, 1.5).
Collect data from all 3 CPGs.
Train ONE RC model on the combined data.
Test the RC on all 3 frictions and visualize the best result.
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
OUTPUT_DIR = "single_rc_model"

def set_floor_friction(env, mu: float):
    """Set floor geom friction to `mu`."""
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
            if (i + 1) % 50 == 0 or i == 0:
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


def train_single_rc_on_multiple_cpgs(frictions):
    """Train 3 CPGs, collect all data, train one RC."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    log.info("=" * 60)
    log.info("STEP 1: Train 3 CPG controllers")
    log.info("=" * 60)
    
    all_X = []
    all_Y = []
    cpg_results = []
    
    # Train CPGs and collect data
    for friction_idx, mu in enumerate(frictions):
        log.info("\nTraining CPG for friction %.1f (%d/%d)", mu, friction_idx + 1, len(frictions))
        
        env_cpg = gym.make("Ant-v5", render_mode="rgb_array")
        set_floor_friction(env_cpg, mu)
        
        # Tune CPG
        best_vec, best_score = random_search_tune(env_cpg, env_cpg.action_space.shape[0], n_iters=100, episode_length=500)
        cpg = CPGController.from_vector(best_vec, env_cpg.action_space.shape[0])
        cpg_fwd, cpg_rew, _, cpg_max_speed, cpg_slip = evaluate_controller(env_cpg, cpg, episode_length=500)
        
        log.info("  CPG performance: forward=%.3f, reward=%.3f", cpg_fwd, cpg_rew)
        
        cpg_results.append({
            "friction": mu,
            "cpg_params": best_vec,
            "cpg_fwd": cpg_fwd,
            "cpg_rew": cpg_rew,
        })
        
        # Collect CPG data (20 episodes per friction)
        log.info("  Collecting CPG data (20 episodes)...")
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
                all_X.append(obs)
                all_Y.append(action)
                obs, reward, terminated, truncated, info = env_cpg.step(action)
                if terminated or truncated:
                    break
        
        env_cpg.close()
        log.info("  Collected %d samples from friction %.1f", len(all_X), mu)
    
    # Combine all data
    X_combined = np.array(all_X)
    Y_combined = np.array(all_Y)
    
    log.info("\n" + "=" * 60)
    log.info("STEP 2: Train ONE RC on combined data")
    log.info("=" * 60)
    log.info("Total training samples: %d", len(X_combined))
    
    # Normalize inputs
    input_mean = X_combined.mean(axis=0)
    input_std = X_combined.std(axis=0)
    input_std[input_std < 1e-8] = 1.0
    X_norm = (X_combined - input_mean) / input_std
    
    # Create and train RC
    env_test = gym.make("Ant-v5", render_mode="rgb_array")
    n_inputs = env_test.observation_space.shape[0]
    n_outputs = env_test.action_space.shape[0]
    env_test.close()
    
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
    
    log.info("Training RC...")
    esn.fit(X_norm, Y_combined)
    log.info("RC training complete!")
    
    # Test RC on all frictions
    log.info("\n" + "=" * 60)
    log.info("STEP 3: Test RC on all frictions")
    log.info("=" * 60)
    
    test_results = []
    for mu in frictions:
        log.info("\nTesting on friction %.1f...", mu)
        env_test = gym.make("Ant-v5", render_mode="rgb_array")
        set_floor_friction(env_test, mu)
        
        rc_fwd, rc_rew, _, rc_max_speed, rc_slip = evaluate_rc_policy(env_test, esn, episode_length=500, input_mean=input_mean, input_std=input_std)
        env_test.close()
        
        log.info("  RC performance: forward=%.3f, reward=%.3f", rc_fwd, rc_rew)
        
        test_results.append({
            "friction": mu,
            "rc_fwd": rc_fwd,
            "rc_rew": rc_rew,
            "rc_max_speed": rc_max_speed,
        })
    
    # Find best friction
    best_result = max(test_results, key=lambda x: x["rc_fwd"])
    
    log.info("\n" + "=" * 60)
    log.info("BEST RC PERFORMANCE: Friction %.1f", best_result["friction"])
    log.info("  Forward distance: %.3f", best_result["rc_fwd"])
    log.info("  Reward: %.3f", best_result["rc_rew"])
    log.info("=" * 60)
    
    # Save the model
    model_data = {
        "rc_model": esn,
        "input_mean": input_mean,
        "input_std": input_std,
        "best_friction": best_result["friction"],
        "cpg_results": cpg_results,
        "test_results": test_results,
    }
    
    model_path = os.path.join(OUTPUT_DIR, "single_rc_trained_on_3cpgs.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model_data, f)
    log.info("\nModel saved to: %s", model_path)
    
    # Save summary
    summary_path = os.path.join(OUTPUT_DIR, "summary.txt")
    with open(summary_path, "w") as f:
        f.write("Training Summary: Single RC trained on 3 CPGs\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("CPG Training Results:\n")
        for res in cpg_results:
            f.write(f"  Friction {res['friction']}: CPG forward={res['cpg_fwd']:.3f}, reward={res['cpg_rew']:.3f}\n")
        
        f.write("\nRC Testing Results (single RC tested on all frictions):\n")
        for res in test_results:
            f.write(f"  Friction {res['friction']}: RC forward={res['rc_fwd']:.3f}, reward={res['rc_rew']:.3f}\n")
        
        f.write("\n" + "=" * 60 + "\n")
        f.write(f"BEST: RC performs best at friction {best_result['friction']}\n")
        f.write(f"  Forward: {best_result['rc_fwd']:.3f}\n")
        f.write(f"  Reward: {best_result['rc_rew']:.3f}\n")
        f.write(f"\nModel: {model_path}\n")
    
    log.info("Summary saved to: %s", summary_path)
    
    return model_path, best_result["friction"]


if __name__ == "__main__":
    model_path, best_friction = train_single_rc_on_multiple_cpgs(TRAIN_FRICTIONS)
    log.info("\n" + "=" * 60)
    log.info("To visualize the RC model at its best friction:")
    log.info("  python visualize_single_rc.py")
    log.info("=" * 60)
