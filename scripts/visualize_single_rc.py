"""
Visualize the single RC model trained on 3 CPGs at its best friction.
"""
import logging
import pickle
import numpy as np
import gymnasium as gym
import os

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

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


def visualize_rc_model(model_path, num_episodes=5, episode_length=500):
    """Load and visualize the single RC model at its best friction."""
    
    if not os.path.exists(model_path):
        log.error("Model not found: %s", model_path)
        log.info("Run train_single_rc_multi_cpg.py first!")
        return
    
    log.info("Loading model from %s", model_path)
    
    with open(model_path, "rb") as f:
        model_data = pickle.load(f)
    
    esn = model_data["rc_model"]
    input_mean = model_data["input_mean"]
    input_std = model_data["input_std"]
    best_friction = model_data["best_friction"]
    
    log.info("RC trained on data from 3 CPGs (frictions: 0.5, 1.0, 1.5)")
    log.info("Best performance at friction: %.1f", best_friction)
    log.info("\n" + "=" * 60)
    log.info("VISUALIZING RC at friction %.1f", best_friction)
    log.info("=" * 60)
    
    # Create environment with best friction
    env = gym.make("Ant-v5", render_mode="human")
    set_floor_friction(env, best_friction)
    
    episode_distances = []
    episode_rewards = []
    
    for ep in range(num_episodes):
        log.info("\n--- Episode %d/%d ---", ep + 1, num_episodes)
        obs, info = env.reset()
        start_x = get_base_x(env)
        total_reward = 0.0
        
        for step in range(episode_length):
            # Normalize observation
            obs_norm = (obs - input_mean) / input_std
            
            # Get action from RC
            action = esn.run(obs_norm.reshape(1, -1))
            action = np.asarray(action).flatten()
            try:
                action = np.clip(action, env.action_space.low, env.action_space.high)
            except Exception:
                action = np.clip(action, -1.0, 1.0)
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            
            if terminated or truncated:
                break
        
        end_x = get_base_x(env)
        distance = end_x - start_x
        episode_distances.append(distance)
        episode_rewards.append(total_reward)
        
        log.info("Episode %d: Distance=%.3f, Reward=%.3f", ep + 1, distance, total_reward)
    
    env.close()
    
    # Print summary
    avg_distance = np.mean(episode_distances)
    avg_reward = np.mean(episode_rewards)
    std_distance = np.std(episode_distances)
    std_reward = np.std(episode_rewards)
    
    log.info("\n" + "=" * 60)
    log.info("Visualization Summary:")
    log.info("  RC trained on: 3 CPGs (frictions 0.5, 1.0, 1.5 combined)")
    log.info("  Tested at friction: %.1f", best_friction)
    log.info("  Avg Distance: %.3f ± %.3f", avg_distance, std_distance)
    log.info("  Avg Reward:   %.3f ± %.3f", avg_reward, std_reward)
    log.info("=" * 60)


if __name__ == "__main__":
    model_path = "single_rc_model/single_rc_trained_on_3cpgs.pkl"
    visualize_rc_model(model_path, num_episodes=5, episode_length=500)
