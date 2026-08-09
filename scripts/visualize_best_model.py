"""
Visualization script for the best CPG/RC model.
Load and display the model in action with rendering.
"""
import logging
import sys
import pickle
import numpy as np
import gymnasium as gym
import time

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

def set_floor_friction(env, mu: float):
    """Set floor geom friction to `mu`."""
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


def visualize_model(model_path, num_episodes=3, episode_length=500, render_mode="human"):
    """Load and visualize a saved model."""
    log.info("Loading model from %s", model_path)
    
    with open(model_path, "rb") as f:
        model_data = pickle.load(f)
    
    friction = model_data["friction"]
    esn = model_data["rc_model"]
    input_mean = model_data["input_mean"]
    input_std = model_data["input_std"]
    
    log.info("Model trained on friction: %.1f", friction)
    log.info("Input normalization stats: mean_shape=%s, std_shape=%s", input_mean.shape, input_std.shape)
    
    # Create environment with same friction
    env = gym.make("Ant-v5", render_mode=render_mode)
    set_floor_friction(env, friction)
    
    log.info("Running %d episodes with rendering...", num_episodes)
    
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
            
            # Get action from RC model
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
    log.info("Visualization Summary (friction=%.1f):", friction)
    log.info("  Avg Distance: %.3f ± %.3f", avg_distance, std_distance)
    log.info("  Avg Reward:   %.3f ± %.3f", avg_reward, std_reward)
    log.info("=" * 60)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Auto-detect best model
        import os
        best_models_dir = "best_models"
        if os.path.exists(best_models_dir):
            files = [f for f in os.listdir(best_models_dir) if f.startswith("model_friction_")]
            if files:
                # Find best from summary
                summary_path = os.path.join(best_models_dir, "summary.txt")
                if os.path.exists(summary_path):
                    with open(summary_path, "r") as f:
                        content = f.read()
                        # Extract best model path from last line
                        for line in reversed(content.split("\n")):
                            if line.startswith("Model:"):
                                model_path = line.split("Model:", 1)[1].strip()
                                break
                else:
                    # Just use first model
                    model_path = os.path.join(best_models_dir, files[0])
                
                log.info("No model specified, using best model: %s", model_path)
                visualize_model(model_path, num_episodes=3, episode_length=500)
            else:
                log.error("No models found in %s", best_models_dir)
        else:
            log.error("Usage: python visualize_best_model.py <model_path>")
    else:
        model_path = sys.argv[1]
        num_episodes = int(sys.argv[2]) if len(sys.argv) > 2 else 3
        visualize_model(model_path, num_episodes=num_episodes, episode_length=500)
