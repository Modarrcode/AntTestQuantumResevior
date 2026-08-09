import logging
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)
import numpy as np
import gymnasium as gym
from reservoirpy.nodes import Reservoir, Ridge
from reservoirpy.model import Model
import pickle
import os


n_reservoir = 1000
TRAIN_FRICTIONS = [0.5, 1.0, 1.5]
OUTPUT_DIR = "improved_rc_model"


def set_floor_friction(env, mu: float):
    """Set floor geom friction."""
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
    """CPG controller."""
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
    """Tune CPG with MORE iterations."""
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
            if (i + 1) % 100 == 0:
                log.info("  Iter %d: best forward=%.3f", i, best_score)
    return best_vec, best_score


def evaluate_controller(env, controller, episode_length=500):
    """Evaluate CPG."""
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
        if terminated or truncated:
            break
    end_x = get_base_x(env)
    forward = end_x - start_x
    return forward, total_reward, step + 1


def collect_good_cpg_data(env, cpg, n_episodes=30, episode_length=500, min_forward=0.5):
    """
    FIX 1: Only collect data from SUCCESSFUL episodes.
    Filter out episodes where ant falls or moves backward.
    """
    X_all, Y_all = [], []
    good_episodes = 0
    attempts = 0
    max_attempts = n_episodes * 3  # Allow multiple attempts
    
    while good_episodes < n_episodes and attempts < max_attempts:
        attempts += 1
        X_ep, Y_ep = [], []
        obs, info = env.reset()
        start_x = get_base_x(env)
        
        for step in range(episode_length):
            t = step * 0.02
            action = cpg.step(t)
            action = np.asarray(action).flatten()
            try:
                action = np.clip(action, env.action_space.low, env.action_space.high)
            except Exception:
                action = np.clip(action, -1.0, 1.0)
            
            X_ep.append(obs)
            Y_ep.append(action)
            
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        
        end_x = get_base_x(env)
        forward = end_x - start_x
        
        # FIX: Only keep successful episodes
        if forward >= min_forward:
            X_all.extend(X_ep)
            Y_all.extend(Y_ep)
            good_episodes += 1
            if good_episodes % 5 == 0:
                log.info("    Collected %d/%d good episodes (forward=%.2f)", good_episodes, n_episodes, forward)
    
    log.info("    Final: %d good episodes from %d attempts", good_episodes, attempts)
    return np.array(X_all), np.array(Y_all)


def augment_temporal_features(X, window=3):
    """
    FIX 2: Add temporal context by including recent observations.
    Augments state with velocity (difference from previous state).
    """
    X_aug = []
    for i in range(len(X)):
        # Current state
        curr = X[i]
        
        # Velocity (difference from previous frame)
        if i > 0:
            vel = X[i] - X[i-1]
        else:
            vel = np.zeros_like(curr)
        
        # Concatenate [current_state, velocity]
        augmented = np.concatenate([curr, vel])
        X_aug.append(augmented)
    
    return np.array(X_aug)


def train_improved_rc(friction, cpg_params, n_reservoir_local=1000):
    """
    Train RC with ALL improvements on a single friction.
    """
    log.info("\n" + "=" * 60)
    log.info("Training IMPROVED RC for friction %.1f", friction)
    log.info("=" * 60)
    
    # Create environment
    env = gym.make("Ant-v5", render_mode="rgb_array")
    set_floor_friction(env, friction)
    n_actions = env.action_space.shape[0]
    
    # Load CPG
    cpg = CPGController.from_vector(cpg_params, n_actions)
    
    # FIX 1: Collect only good episodes
    log.info("Collecting high-quality CPG demonstrations...")
    X_raw, Y_raw = collect_good_cpg_data(env, cpg, n_episodes=30, episode_length=500, min_forward=1.0)
    log.info("Collected %d samples", len(X_raw))
    
    # FIX 2: Augment with temporal features
    log.info("Adding temporal features...")
    X = augment_temporal_features(X_raw)
    Y = Y_raw
    
    # FIX 3: Normalize both inputs AND outputs
    log.info("Normalizing inputs and outputs...")
    input_mean = X.mean(axis=0)
    input_std = X.std(axis=0)
    input_std[input_std < 1e-8] = 1.0
    X_norm = (X - input_mean) / input_std
    
    output_mean = Y.mean(axis=0)
    output_std = Y.std(axis=0)
    output_std[output_std < 1e-8] = 1.0
    Y_norm = (Y - output_mean) / output_std
    
    n_inputs = X_norm.shape[1]
    n_outputs = Y_norm.shape[1]
    
    # FIX 4: Try multiple hyperparameter configurations
    log.info("Testing hyperparameter configurations...")
    configs = [
        {"spectral_radius": 0.9, "input_scaling": 1.5, "ridge": 1e-7},
        {"spectral_radius": 0.95, "input_scaling": 2.0, "ridge": 1e-6},
        {"spectral_radius": 1.0, "input_scaling": 2.5, "ridge": 1e-5},
    ]
    
    best_config = None
    best_esn = None
    best_score = -float('inf')
    
    for cfg_idx, cfg in enumerate(configs):
        log.info("  Config %d: SR=%.2f, IS=%.2f, ridge=%.1e", 
                 cfg_idx+1, cfg["spectral_radius"], cfg["input_scaling"], cfg["ridge"])
        
        # Create reservoir with this config
        np.random.seed(42 + cfg_idx)
        Win = np.random.uniform(-cfg["input_scaling"], cfg["input_scaling"], 
                               size=(n_reservoir_local, n_inputs))
        
        density = 0.1
        mask = (np.random.rand(n_reservoir_local, n_reservoir_local) < density)
        W = np.random.uniform(-0.5, 0.5, size=(n_reservoir_local, n_reservoir_local)) * mask.astype(float)
        eigvals = np.linalg.eigvals(W)
        max_abs = np.max(np.abs(eigvals))
        if max_abs > 1e-12:
            W *= cfg["spectral_radius"] / max_abs
        
        reservoir = Reservoir(units=n_reservoir_local, input_dim=n_inputs, Win=Win, W=W)
        readout = Ridge(input_dim=n_reservoir_local, output_dim=n_outputs, ridge=cfg["ridge"])
        esn = Model([reservoir, readout], edges=[(reservoir, 0, readout)])
        
        # Train
        esn.fit(X_norm, Y_norm)
        
        # Quick validation
        env_val = gym.make("Ant-v5", render_mode="rgb_array")
        set_floor_friction(env_val, friction)
        score = evaluate_rc_policy(env_val, esn, episode_length=500, 
                                   input_mean=input_mean, input_std=input_std,
                                   output_mean=output_mean, output_std=output_std,
                                   augment=True)
        env_val.close()
        
        log.info("    Validation forward: %.3f", score[0])
        
        if score[0] > best_score:
            best_score = score[0]
            best_config = cfg
            best_esn = esn
    
    log.info("\nBest config: %s", best_config)
    log.info("Best validation forward: %.3f", best_score)
    
    env.close()
    
    return best_esn, input_mean, input_std, output_mean, output_std, best_config


def evaluate_rc_policy(env, esn, episode_length=500, 
                       input_mean=None, input_std=None,
                       output_mean=None, output_std=None, augment=True):
    """Evaluate RC with all improvements."""
    obs, info = env.reset()
    prev_obs = obs.copy()
    start_x = get_base_x(env)
    total_reward = 0.0
    
    for step in range(episode_length):
        # Augment with temporal features
        if augment:
            vel = obs - prev_obs if step > 0 else np.zeros_like(obs)
            obs_aug = np.concatenate([obs, vel])
        else:
            obs_aug = obs
        
        # Normalize
        if input_mean is not None:
            obs_norm = (obs_aug - input_mean) / input_std
        else:
            obs_norm = obs_aug
        
        # Run RC
        action_norm = esn.run(obs_norm.reshape(1, -1))
        action_norm = np.asarray(action_norm).flatten()
        
        # Denormalize output
        if output_mean is not None:
            action = action_norm * output_std + output_mean
        else:
            action = action_norm
        
        try:
            action = np.clip(action, env.action_space.low, env.action_space.high)
        except Exception:
            action = np.clip(action, -1.0, 1.0)
        
        prev_obs = obs.copy()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        
        if terminated or truncated:
            break
    
    end_x = get_base_x(env)
    forward = end_x - start_x
    return forward, total_reward, step + 1


def main():
    """Train improved RC models for all frictions."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    log.info("=" * 60)
    log.info("IMPROVED RC TRAINING")
    log.info("=" * 60)
    log.info("Improvements:")
    log.info("1. Better CPG tuning (1000 iterations)")
    log.info("2. Filter failed episodes (only learn success)")
    log.info("3. Temporal feature augmentation")
    log.info("4. Output normalization")
    log.info("5. Hyperparameter grid search")
    log.info("=" * 60)
    
    # --- Multi-friction RC training ---
    # Collect CPG data for all frictions, augment with friction value
    all_X = []
    all_Y = []
    all_friction = []
    cpg_stats = []
    for friction in TRAIN_FRICTIONS:
        log.info("\n" + "=" * 60)
        log.info("FRICTION %.1f", friction)
        log.info("=" * 60)
        # Step 1: Train better CPG
        log.info("Step 1: Training CPG (1000 iterations)...")
        env_cpg = gym.make("Ant-v5", render_mode="rgb_array")
        set_floor_friction(env_cpg, friction)
        best_vec, best_score = random_search_tune(env_cpg, env_cpg.action_space.shape[0], n_iters=1000, episode_length=500)
        cpg = CPGController.from_vector(best_vec, env_cpg.action_space.shape[0])
        cpg_fwd, cpg_rew, _ = evaluate_controller(env_cpg, cpg, episode_length=500)
        log.info("CPG final: forward=%.3f, reward=%.3f", cpg_fwd, cpg_rew)
        # Step 2: Collect CPG data (only good episodes)
        X_raw, Y_raw = collect_good_cpg_data(env_cpg, cpg, n_episodes=30, episode_length=500, min_forward=1.0)
        env_cpg.close()
        # Step 3: Augment with temporal features
        X_aug = augment_temporal_features(X_raw)
        # Step 4: Add friction as extra input
        friction_col = np.full((X_aug.shape[0], 1), friction, dtype=np.float32)
        X_aug_fric = np.concatenate([X_aug, friction_col], axis=1)
        all_X.append(X_aug_fric)
        all_Y.append(Y_raw)
        all_friction.append(np.full((X_aug.shape[0],), friction))
        cpg_stats.append({"friction": friction, "cpg_fwd": cpg_fwd, "cpg_rew": cpg_rew})

    # Combine all data
    X_all = np.vstack(all_X)
    Y_all = np.vstack(all_Y)

    # Normalize inputs and outputs
    input_mean = X_all.mean(axis=0)
    input_std = X_all.std(axis=0)
    input_std[input_std < 1e-8] = 1.0
    X_norm = (X_all - input_mean) / input_std
    output_mean = Y_all.mean(axis=0)
    output_std = Y_all.std(axis=0)
    output_std[output_std < 1e-8] = 1.0
    Y_norm = (Y_all - output_mean) / output_std

    n_inputs = X_norm.shape[1]
    n_outputs = Y_norm.shape[1]

    # Hyperparameter configs
    configs = [
        {"spectral_radius": 0.9, "input_scaling": 1.5, "ridge": 1e-7},
        {"spectral_radius": 0.95, "input_scaling": 2.0, "ridge": 1e-6},
        {"spectral_radius": 1.0, "input_scaling": 2.5, "ridge": 1e-5},
    ]
    best_config = None
    best_esn = None
    best_score = -float('inf')
    for cfg_idx, cfg in enumerate(configs):
        log.info("  Config %d: SR=%.2f, IS=%.2f, ridge=%.1e", cfg_idx+1, cfg["spectral_radius"], cfg["input_scaling"], cfg["ridge"])
        np.random.seed(42 + cfg_idx)
        Win = np.random.uniform(-cfg["input_scaling"], cfg["input_scaling"], size=(n_reservoir, n_inputs))
        density = 0.1
        mask = (np.random.rand(n_reservoir, n_reservoir) < density)
        W = np.random.uniform(-0.5, 0.5, size=(n_reservoir, n_reservoir)) * mask.astype(float)
        eigvals = np.linalg.eigvals(W)
        max_abs = np.max(np.abs(eigvals))
        if max_abs > 1e-12:
            W *= cfg["spectral_radius"] / max_abs
        reservoir = Reservoir(units=n_reservoir, input_dim=n_inputs, Win=Win, W=W)
        readout = Ridge(input_dim=n_reservoir, output_dim=n_outputs, ridge=cfg["ridge"])
        esn = Model([reservoir, readout], edges=[(reservoir, 0, readout)])
        esn.fit(X_norm, Y_norm)
        # Validation: evaluate on each friction separately
        val_scores = []
        for stat in cpg_stats:
            friction = stat["friction"]
            # Build validation set for this friction
            idxs = np.where(np.abs(X_all[:, -1] - friction) < 1e-6)[0]
            if len(idxs) == 0:
                continue
            X_val = X_all[idxs]
            Y_val = Y_all[idxs]
            # Normalize
            X_val_norm = (X_val - input_mean) / input_std
            # Evaluate
            env_val = gym.make("Ant-v5", render_mode="rgb_array")
            set_floor_friction(env_val, friction)
            # Use a custom evaluation function that augments obs with friction
            def eval_policy_with_friction(env, esn, friction, episode_length=500):
                obs, info = env.reset()
                prev_obs = obs.copy()
                start_x = get_base_x(env)
                total_reward = 0.0
                for step in range(episode_length):
                    vel = obs - prev_obs if step > 0 else np.zeros_like(obs)
                    obs_aug = np.concatenate([obs, vel, [friction]])
                    obs_norm = (obs_aug - input_mean) / input_std
                    action_norm = esn.run(obs_norm.reshape(1, -1))
                    action_norm = np.asarray(action_norm).flatten()
                    action = action_norm * output_std + output_mean
                    try:
                        action = np.clip(action, env.action_space.low, env.action_space.high)
                    except Exception:
                        action = np.clip(action, -1.0, 1.0)
                    prev_obs = obs.copy()
                    obs, reward, terminated, truncated, info = env.step(action)
                    total_reward += float(reward)
                    if terminated or truncated:
                        break
                end_x = get_base_x(env)
                forward = end_x - start_x
                return forward, total_reward, step + 1
            fwd, rew, _ = eval_policy_with_friction(env_val, esn, friction)
            val_scores.append(fwd)
            env_val.close()
        mean_score = np.mean(val_scores) if val_scores else -float('inf')
        log.info("    Validation mean forward: %.3f", mean_score)
        if mean_score > best_score:
            best_score = mean_score
            best_config = cfg
            best_esn = esn

    # Save the multi-friction RC model
    model_data = {
        "rc_model": best_esn,
        "input_mean": input_mean,
        "input_std": input_std,
        "output_mean": output_mean,
        "output_std": output_std,
        "config": best_config,
        "frictions": TRAIN_FRICTIONS,
        "cpg_stats": cpg_stats,
        "type": "multi_friction_rc"
    }
    model_path = os.path.join(OUTPUT_DIR, "improved_rc_multi_friction.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model_data, f)
    log.info("Saved multi-friction RC to: %s", model_path)

    log.info("\n" + "=" * 60)
    log.info("FINAL RESULTS")
    log.info("=" * 60)
    for stat in cpg_stats:
        log.info("Friction %.1f: CPG=%.3fm, Reward=%.3f", stat["friction"], stat["cpg_fwd"], stat["cpg_rew"])


if __name__ == "__main__":
    main()
