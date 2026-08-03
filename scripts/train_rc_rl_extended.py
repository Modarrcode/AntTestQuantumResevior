"""
Extended RL training starting from the improved RC baseline (76% of CPG).
This combines all improvements + extensive RL optimization for best results.
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

FRICTION = 1.0  # Best performing friction from improved RC
RL_GENERATIONS = 200  # Extended training
POPULATION_SIZE = 30  # Larger population for better exploration
OUTPUT_DIR = "rc_rl_extended"


def set_floor_friction(env, mu: float):
    """Set floor friction."""
    try:
        model = env.unwrapped.model
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
                    pass
    except Exception:
        pass


def get_base_x(env):
    """Get X position."""
    try:
        return float(env.unwrapped.data.qpos[0])
    except Exception:
        return 0.0


class RCPolicyFromModel:
    """RC policy that uses the full trained ReservoirPy model."""
    def __init__(self, rc_model, input_mean, input_std, output_mean=None, output_std=None, augment=True):
        self.rc_model = rc_model
        self.input_mean = input_mean
        self.input_std = input_std
        self.output_mean = output_mean
        self.output_std = output_std
        self.augment = augment
        self.prev_obs = None
        
        # Get readout weights for RL optimization
        nodes = list(rc_model.nodes)
        self.readout = nodes[1]
        self.reservoir = nodes[0]
        
    def reset(self):
        """Reset state."""
        self.rc_model.reset()
        self.prev_obs = None
    
    def predict(self, obs):
        """Get action using the model's run method."""
        # Augment with velocity if enabled
        if self.augment:
            if self.prev_obs is not None:
                vel = obs - self.prev_obs
            else:
                vel = np.zeros_like(obs)
            obs_aug = np.concatenate([obs, vel])
        else:
            obs_aug = obs
        
        # Normalize
        obs_norm = (obs_aug - self.input_mean) / self.input_std
        
        # Run through model
        action_norm = self.rc_model.run(obs_norm.reshape(1, -1)).flatten()
        
        # Denormalize if output normalization was used
        if self.output_mean is not None:
            action = action_norm * self.output_std + self.output_mean
        else:
            action = action_norm
        
        self.prev_obs = obs.copy()
        return action
    
    def set_weights(self, weights):
        """Set readout weights."""
        # Reshape and set to readout node
        # Wout should be (n_reservoir, n_outputs)
        n_reservoir = self.readout.input_dim
        n_outputs = self.readout.output_dim
        self.readout.Wout = weights.reshape(n_reservoir, n_outputs)
    
    def get_weights(self):
        """Get readout weights as flat vector."""
        return self.readout.Wout.flatten()


def evaluate_policy(env, policy, n_episodes=3, episode_length=500):
    """Evaluate policy."""
    total_rewards = []
    total_distances = []
    
    for ep in range(n_episodes):
        policy.reset()
        obs, _ = env.reset()
        start_x = get_base_x(env)
        ep_reward = 0.0
        
        for step in range(episode_length):
            action = policy.predict(obs)
            action = np.clip(action, env.action_space.low, env.action_space.high)
            obs, reward, term, trunc, _ = env.step(action)
            ep_reward += reward
            
            if term or trunc:
                break
        
        distance = get_base_x(env) - start_x
        total_rewards.append(ep_reward)
        total_distances.append(distance)
    
    return np.mean(total_rewards), np.mean(total_distances)


def evolution_strategy_optimize(env, policy, n_generations=200, population_size=30, 
                                noise_std=0.03, learning_rate=0.01):
    """
    Extended ES optimization with adaptive parameters.
    """
    log.info("\n" + "=" * 60)
    log.info("EXTENDED RL OPTIMIZATION")
    log.info("=" * 60)
    log.info("Generations: %d, Population: %d", n_generations, population_size)
    
    current_weights = policy.get_weights()
    n_params = len(current_weights)
    log.info("Optimizing %d parameters", n_params)
    
    # Evaluate baseline
    baseline_reward, baseline_dist = evaluate_policy(env, policy, n_episodes=5)
    log.info("Baseline (improved supervised): reward=%.2f, distance=%.3f", baseline_reward, baseline_dist)
    
    best_weights = current_weights.copy()
    best_reward = baseline_reward
    best_distance = baseline_dist
    
    # Adaptive noise
    current_noise_std = noise_std
    no_improvement_count = 0
    
    for gen in range(n_generations):
        # Generate population
        population = []
        noises = []
        
        for _ in range(population_size):
            noise = np.random.randn(n_params) * current_noise_std
            noises.append(noise)
            population.append(current_weights + noise)
        
        # Evaluate
        rewards = []
        distances = []
        for weights in population:
            policy.set_weights(weights)
            reward, dist = evaluate_policy(env, policy, n_episodes=2)
            rewards.append(reward)
            distances.append(dist)
        
        rewards = np.array(rewards)
        distances = np.array(distances)
        
        # Fitness shaping (rank-based)
        # Use distance as primary metric since we care about forward locomotion
        ranks = np.argsort(np.argsort(distances))
        utilities = np.maximum(0, np.log(population_size/2 + 1) - np.log(ranks + 1))
        utilities /= utilities.sum()
        
        # Gradient update
        gradient = np.zeros(n_params)
        for noise, util in zip(noises, utilities):
            gradient += util * noise
        
        current_weights += learning_rate * gradient
        
        # Track best based on distance
        best_idx = np.argmax(distances)
        if distances[best_idx] > best_distance:
            best_distance = distances[best_idx]
            best_reward = rewards[best_idx]
            best_weights = population[best_idx].copy()
            no_improvement_count = 0
            log.info("Gen %d: NEW BEST! distance=%.3fm, reward=%.2f", 
                     gen+1, best_distance, best_reward)
        else:
            no_improvement_count += 1
        
        # Adaptive noise: increase if stuck
        if no_improvement_count > 20:
            current_noise_std *= 1.1
            no_improvement_count = 0
            log.info("Gen %d: Increasing exploration (noise=%.4f)", gen+1, current_noise_std)
        
        if (gen + 1) % 20 == 0:
            log.info("Gen %d: best_dist=%.3fm (%.0f%% of baseline), avg_dist=%.3fm, noise=%.4f", 
                     gen+1, best_distance, (best_distance/baseline_dist)*100, 
                     distances.mean(), current_noise_std)
    
    # Set to best
    policy.set_weights(best_weights)
    
    # Final evaluation (10 episodes)
    final_reward, final_dist = evaluate_policy(env, policy, n_episodes=10)
    
    log.info("\n" + "=" * 60)
    log.info("RL OPTIMIZATION RESULTS")
    log.info("=" * 60)
    log.info("Before RL: distance=%.3fm, reward=%.2f", baseline_dist, baseline_reward)
    log.info("After RL:  distance=%.3fm, reward=%.2f", final_dist, final_reward)
    log.info("Improvement: %.1f%% distance, %.1f%% reward", 
             ((final_dist/baseline_dist - 1)*100) if baseline_dist > 0 else 0,
             ((final_reward/baseline_reward - 1)*100) if baseline_reward > 0 else 0)
    log.info("=" * 60)
    
    return best_weights, final_reward, final_dist


def main():
    """Load improved RC and run extended RL optimization."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    log.info("=" * 60)
    log.info("EXTENDED RC + RL TRAINING")
    log.info("=" * 60)
    log.info("Strategy: Start from improved RC (76%% baseline)")
    log.info("          + Extended RL optimization (200 generations)")
    log.info("=" * 60)
    
        # Step 1: Load multi-friction RC model
        improved_model_path = "improved_rc_model/improved_rc_multi_friction.pkl"
        if not os.path.exists(improved_model_path):
            log.error("Multi-friction RC model not found at: %s", improved_model_path)
            log.error("Please run train_improved_rc.py to generate it!")
            return
        log.info("\nStep 1: Loading multi-friction RC model...")
        with open(improved_model_path, "rb") as f:
            model_data = pickle.load(f)
        rc_model = model_data["rc_model"]
        input_mean = model_data["input_mean"]
        input_std = model_data["input_std"]
        output_mean = model_data.get("output_mean", None)
        output_std = model_data.get("output_std", None)
        config = model_data["config"]
        frictions = model_data["frictions"]
        cpg_stats = model_data["cpg_stats"]
        log.info("Loaded multi-friction RC model. Frictions: %s", frictions)
        log.info("  Config: %s", config)

        # Step 2: RL policy wrapper that takes friction as input
        class RCPolicyMultiFriction:
            def __init__(self, rc_model, input_mean, input_std, output_mean=None, output_std=None, augment=True):
                self.rc_model = rc_model
                self.input_mean = input_mean
                self.input_std = input_std
                self.output_mean = output_mean
                self.output_std = output_std
                self.augment = augment
                self.prev_obs = None
                nodes = list(rc_model.nodes)
                self.readout = nodes[1]
                self.reservoir = nodes[0]
            def reset(self):
                self.rc_model.reset()
                self.prev_obs = None
            def predict(self, obs, friction):
                # Augment with velocity if enabled
                if self.augment:
                    if self.prev_obs is not None:
                        vel = obs - self.prev_obs
                    else:
                        vel = np.zeros_like(obs)
                    obs_aug = np.concatenate([obs, vel, [friction]])
                else:
                    obs_aug = np.concatenate([obs, [friction]])
                obs_norm = (obs_aug - self.input_mean) / self.input_std
                action_norm = self.rc_model.run(obs_norm.reshape(1, -1)).flatten()
                if self.output_mean is not None:
                    action = action_norm * self.output_std + self.output_mean
                else:
                    action = action_norm
                self.prev_obs = obs.copy()
                return action
            def set_weights(self, weights):
                n_reservoir = self.readout.input_dim
                n_outputs = self.readout.output_dim
                self.readout.Wout = weights.reshape(n_reservoir, n_outputs)
            def get_weights(self):
                return self.readout.Wout.flatten()

        # Step 3: RL environment and evaluation with random friction per episode
        def evaluate_policy_multi(env, policy, frictions, n_episodes=3, episode_length=500):
            total_rewards = []
            total_distances = []
            for ep in range(n_episodes):
                policy.reset()
                # Randomly select friction for this episode
                friction = float(np.random.choice(frictions))
                set_floor_friction(env, friction)
                obs, _ = env.reset()
                start_x = get_base_x(env)
                ep_reward = 0.0
                for step in range(episode_length):
                    action = policy.predict(obs, friction)
                    action = np.clip(action, env.action_space.low, env.action_space.high)
                    obs, reward, term, trunc, _ = env.step(action)
                    ep_reward += reward
                    if term or trunc:
                        break
                distance = get_base_x(env) - start_x
                total_rewards.append(ep_reward)
                total_distances.append(distance)
            return np.mean(total_rewards), np.mean(total_distances)

        def evolution_strategy_optimize_multi(env, policy, frictions, n_generations=200, population_size=30, noise_std=0.03, learning_rate=0.01):
            log.info("\n" + "=" * 60)
            log.info("EXTENDED RL OPTIMIZATION (MULTI-FRICTION)")
            log.info("=" * 60)
            log.info("Generations: %d, Population: %d", n_generations, population_size)
            current_weights = policy.get_weights()
            n_params = len(current_weights)
            log.info("Optimizing %d parameters", n_params)
            baseline_reward, baseline_dist = evaluate_policy_multi(env, policy, frictions, n_episodes=5)
            log.info("Baseline (improved supervised): reward=%.2f, distance=%.3f", baseline_reward, baseline_dist)
            best_weights = current_weights.copy()
            best_reward = baseline_reward
            best_distance = baseline_dist
            current_noise_std = noise_std
            no_improvement_count = 0
            for gen in range(n_generations):
                population = []
                noises = []
                for _ in range(population_size):
                    noise = np.random.randn(n_params) * current_noise_std
                    noises.append(noise)
                    population.append(current_weights + noise)
                rewards = []
                distances = []
                for weights in population:
                    policy.set_weights(weights)
                    reward, dist = evaluate_policy_multi(env, policy, frictions, n_episodes=2)
                    rewards.append(reward)
                    distances.append(dist)
                rewards = np.array(rewards)
                distances = np.array(distances)
                ranks = np.argsort(np.argsort(distances))
                utilities = np.maximum(0, np.log(population_size/2 + 1) - np.log(ranks + 1))
                utilities /= utilities.sum()
                gradient = np.zeros(n_params)
                for noise, util in zip(noises, utilities):
                    gradient += util * noise
                current_weights += learning_rate * gradient
                best_idx = np.argmax(distances)
                if distances[best_idx] > best_distance:
                    best_distance = distances[best_idx]
                    best_reward = rewards[best_idx]
                    best_weights = population[best_idx].copy()
                    no_improvement_count = 0
                    log.info("Gen %d: NEW BEST! distance=%.3fm, reward=%.2f", gen+1, best_distance, best_reward)
                else:
                    no_improvement_count += 1
                if no_improvement_count > 20:
                    current_noise_std *= 1.1
                    no_improvement_count = 0
                    log.info("Gen %d: Increasing exploration (noise=%.4f)", gen+1, current_noise_std)
                if (gen + 1) % 20 == 0:
                    log.info("Gen %d: best_dist=%.3fm (%.0f%% of baseline), avg_dist=%.3fm, noise=%.4f", gen+1, best_distance, (best_distance/baseline_dist)*100, distances.mean(), current_noise_std)
            policy.set_weights(best_weights)
            final_reward, final_dist = evaluate_policy_multi(env, policy, frictions, n_episodes=10)
            log.info("\n" + "=" * 60)
            log.info("RL OPTIMIZATION RESULTS (MULTI-FRICTION)")
            log.info("=" * 60)
            log.info("Before RL: distance=%.3fm, reward=%.2f", baseline_dist, baseline_reward)
            log.info("After RL:  distance=%.3fm, reward=%.2f", final_dist, final_reward)
            log.info("Improvement: %.1f%% distance, %.1f%% reward", ((final_dist/baseline_dist - 1)*100) if baseline_dist > 0 else 0, ((final_reward/baseline_reward - 1)*100) if baseline_reward > 0 else 0)
            log.info("=" * 60)
            return best_weights, final_reward, final_dist

        # Step 4: RL training
        log.info("\nStep 2: Creating RL-trainable multi-friction policy...")
        policy = RCPolicyMultiFriction(rc_model, input_mean, input_std, output_mean, output_std, augment=True)
        log.info("Policy created. Readout weights shape: %s", policy.readout.Wout.shape)
        log.info("\nStep 3: Setting up environment...")
        env = gym.make("Ant-v5", render_mode="rgb_array")
        log.info("\nStep 4: Running extended RL optimization (multi-friction)...")
        best_weights, rl_reward, rl_dist = evolution_strategy_optimize_multi(
            env, policy, frictions,
            n_generations=RL_GENERATIONS,
            population_size=POPULATION_SIZE,
            noise_std=0.02,
            learning_rate=0.01
        )
        # Step 5: Save result
        log.info("\nStep 5: Saving RL-optimized multi-friction model...")
        final_model_data = {
            "rc_model": policy.rc_model,
            "readout_weights": policy.readout.Wout,
            "input_mean": input_mean,
            "input_std": input_std,
            "output_mean": output_mean,
            "output_std": output_std,
            "frictions": frictions,
            "cpg_stats": cpg_stats,
            "rl_fwd": rl_dist,
            "config": config,
            "type": "extended_rl_optimized_multi_friction"
        }
        model_path = os.path.join(OUTPUT_DIR, "rc_extended_rl_multi_friction.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(final_model_data, f)
        log.info("Saved RL-optimized multi-friction model to: %s", model_path)
        log.info("\n" + "=" * 60)
        log.info("FINAL RESULTS")
        log.info("=" * 60)
        for stat in cpg_stats:
            log.info("CPG Teacher (friction %.1f): %.3fm", stat["friction"], stat["cpg_fwd"])
        improvement = ((rl_dist/rc_fwd_supervised)-1)*100
        log.info("\n✅ RL improved RC by %.1f%%", improvement)
        if rl_dist > cpg_fwd * 0.9:
            log.info("⭐ RC achieved >90%% of CPG performance!")
    else:
        log.info("\n❌ RL did not improve - may need more generations")
    
    log.info("\nModel saved to: %s", model_path)
    log.info("=" * 60)
    
    env.close()


if __name__ == "__main__":
    main()
