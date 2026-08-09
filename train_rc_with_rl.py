"""
RC + Reinforcement Learning: Train RC to EXCEED CPG performance.

Approach:
1. Warm-start: Train RC readout on CPG demonstrations (supervised)
2. RL Fine-tuning: Optimize readout weights to maximize reward using Evolution Strategies
3. The reservoir stays fixed, only readout weights are optimized via RL

This allows RC to go beyond CPG performance by directly optimizing for the task.
"""
import logging
import numpy as np
import gymnasium as gym
from reservoirpy.nodes import Reservoir, Ridge
from reservoirpy.model import Model
import pickle
import os
from copy import deepcopy

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

n_reservoir = 1000
TRAIN_FRICTIONS = [1.0]  # Focus on one friction for faster demo
OUTPUT_DIR = "rc_rl_model"
RL_GENERATIONS = 50  # Evolution strategy generations
POPULATION_SIZE = 20  # ES population size


def set_floor_friction(env, mu: float):
    """Set floor friction."""
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


def get_base_x(env):
    """Get X position."""
    try:
        return float(env.unwrapped.data.qpos[0])
    except Exception:
        return 0.0


class CPGController:
    """CPG controller."""
    def __init__(self, n_actions, omega=2.0, amplitudes=None, phases=None, offsets=None):
        self.n = n_actions
        self.omega = float(omega)
        self.amplitudes = np.ones(self.n) if amplitudes is None else np.asarray(amplitudes)
        self.phases = np.zeros(self.n) if phases is None else np.asarray(phases)
        self.offsets = np.zeros(self.n) if offsets is None else np.asarray(offsets)

    @classmethod
    def from_vector(cls, vec, n_actions):
        omega = vec[0]
        A = vec[1:1+n_actions]
        phi = vec[1+n_actions:1+2*n_actions]
        off = vec[1+2*n_actions:1+3*n_actions]
        return cls(n_actions, omega, A, phi, off)

    def step(self, t):
        return self.offsets + self.amplitudes * np.sin(self.omega * t + self.phases)


def tune_cpg(env, n_actions, n_iters=300):
    """Quick CPG tuning."""
    best_vec = None
    best_score = -float('inf')
    for i in range(n_iters):
        omega = np.random.uniform(0.5, 4.0)
        A = np.random.uniform(0.0, 1.0, size=n_actions)
        phi = np.random.uniform(-np.pi, np.pi, size=n_actions)
        off = np.random.uniform(-0.5, 0.5, size=n_actions)
        vec = np.concatenate(([omega], A, phi, off))
        
        cpg = CPGController.from_vector(vec, n_actions)
        obs, _ = env.reset()
        start_x = get_base_x(env)
        
        for step in range(500):
            action = cpg.step(step * 0.02)
            action = np.clip(action, env.action_space.low, env.action_space.high)
            obs, _, term, trunc, _ = env.step(action)
            if term or trunc:
                break
        
        forward = get_base_x(env) - start_x
        if forward > best_score:
            best_score = forward
            best_vec = vec.copy()
            if (i+1) % 100 == 0:
                log.info("  Iter %d: best=%.3f", i, best_score)
    
    return best_vec, best_score


def collect_cpg_data(env, cpg, n_episodes=20):
    """Collect CPG demonstrations."""
    X, Y = [], []
    for ep in range(n_episodes):
        obs, _ = env.reset()
        for step in range(500):
            action = cpg.step(step * 0.02)
            action = np.clip(action, env.action_space.low, env.action_space.high)
            X.append(obs)
            Y.append(action)
            obs, _, term, trunc, _ = env.step(action)
            if term or trunc:
                break
    return np.array(X), np.array(Y)


class RCPolicy:
    """RC policy wrapper for RL."""
    def __init__(self, reservoir, readout_weights, readout_bias, input_mean, input_std):
        self.reservoir = reservoir
        self.Wout = readout_weights  # Shape: (n_outputs, n_reservoir)
        self.bias = readout_bias  # Shape: (n_outputs,)
        self.input_mean = input_mean
        self.input_std = input_std
        self.state = None
        
    def reset(self):
        """Reset reservoir state."""
        self.reservoir.reset()
        self.state = np.zeros(self.reservoir.output_dim)
    
    def predict(self, obs):
        """Get action from observation."""
        # Normalize input
        obs_norm = (obs - self.input_mean) / self.input_std
        
        # Update reservoir state using ReservoirPy's run method
        self.state = self.reservoir.run(obs_norm.reshape(1, -1)).flatten()
        
        # Compute output: Wout @ state + bias
        action = self.Wout @ self.state + self.bias
        
        return action
    
    def set_weights(self, weights):
        """Set readout weights (includes bias at end)."""
        n_outputs = self.Wout.shape[0]
        n_reservoir = self.Wout.shape[1]
        
        # Split weights: first part is Wout, last n_outputs elements are bias
        self.Wout = weights[:n_outputs * n_reservoir].reshape(n_outputs, n_reservoir)
        self.bias = weights[n_outputs * n_reservoir:]
    
    def get_weights(self):
        """Get readout weights as flat vector (includes bias)."""
        return np.concatenate([self.Wout.flatten(), self.bias.flatten()])


def evaluate_policy(env, policy, n_episodes=3, episode_length=500):
    """Evaluate policy and return average reward."""
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


def evolution_strategy_optimize(env, base_policy, n_generations=50, population_size=20, 
                                noise_std=0.1, learning_rate=0.01):
    """
    Optimize RC readout weights using Evolution Strategies (ES).
    
    ES is perfect for this because:
    - Only optimizes readout weights (small parameter space)
    - Gradient-free (no backprop through reservoir)
    - Naturally explores policy space
    """
    log.info("\n" + "=" * 60)
    log.info("RL FINE-TUNING: Evolution Strategies")
    log.info("=" * 60)
    log.info("Generations: %d, Population: %d", n_generations, population_size)
    
    # Get initial weights
    current_weights = base_policy.get_weights()
    n_params = len(current_weights)
    log.info("Optimizing %d parameters (readout weights)", n_params)
    
    best_weights = current_weights.copy()
    best_reward = -float('inf')
    
    # Evaluate baseline
    baseline_reward, baseline_dist = evaluate_policy(env, base_policy, n_episodes=5)
    log.info("Baseline (supervised): reward=%.2f, distance=%.3f", baseline_reward, baseline_dist)
    best_reward = baseline_reward
    
    for gen in range(n_generations):
        # Generate population by adding noise
        population = []
        noises = []
        
        for _ in range(population_size):
            noise = np.random.randn(n_params) * noise_std
            noises.append(noise)
            population.append(current_weights + noise)
        
        # Evaluate population
        rewards = []
        for i, weights in enumerate(population):
            base_policy.set_weights(weights)
            reward, dist = evaluate_policy(env, base_policy, n_episodes=2)  # 2 episodes per eval
            rewards.append(reward)
        
        rewards = np.array(rewards)
        
        # Update: Move towards better-performing perturbations
        # Fitness shaping: rank-based
        ranks = np.argsort(np.argsort(rewards))  # 0 to pop_size-1
        utilities = np.maximum(0, np.log(population_size/2 + 1) - np.log(ranks + 1))
        utilities /= utilities.sum()
        
        # Weight update
        gradient = np.zeros(n_params)
        for noise, util in zip(noises, utilities):
            gradient += util * noise
        
        current_weights += learning_rate * gradient
        
        # Track best
        best_idx = np.argmax(rewards)
        if rewards[best_idx] > best_reward:
            best_reward = rewards[best_idx]
            best_weights = population[best_idx].copy()
        
        if (gen + 1) % 10 == 0 or gen == 0:
            log.info("Gen %d: best_reward=%.2f, avg_reward=%.2f, std=%.2f", 
                     gen+1, best_reward, rewards.mean(), rewards.std())
    
    # Set to best weights
    base_policy.set_weights(best_weights)
    
    # Final evaluation
    final_reward, final_dist = evaluate_policy(env, base_policy, n_episodes=10)
    log.info("\n" + "=" * 60)
    log.info("RL OPTIMIZATION COMPLETE")
    log.info("Before RL: reward=%.2f, distance=%.3f", baseline_reward, baseline_dist)
    log.info("After RL:  reward=%.2f, distance=%.3f", final_reward, final_dist)
    log.info("Improvement: %.1f%% reward, %.1f%% distance", 
             (final_reward/baseline_reward - 1)*100 if baseline_reward != 0 else 0,
             (final_dist/baseline_dist - 1)*100 if baseline_dist != 0 else 0)
    log.info("=" * 60)
    
    return best_weights, final_reward, final_dist


def main():
    """Train RC with CPG warm-start, then RL fine-tuning."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    friction = TRAIN_FRICTIONS[0]
    
    log.info("=" * 60)
    log.info("RC + RL TRAINING (Friction %.1f)", friction)
    log.info("=" * 60)
    
    # Step 1: Train CPG
    log.info("\nStep 1: Training CPG teacher...")
    env = gym.make("Ant-v5", render_mode="rgb_array")
    set_floor_friction(env, friction)
    n_actions = env.action_space.shape[0]
    
    cpg_vec, cpg_score = tune_cpg(env, n_actions, n_iters=300)
    cpg = CPGController.from_vector(cpg_vec, n_actions)
    log.info("CPG performance: %.3f forward", cpg_score)
    
    # Step 2: Collect CPG data
    log.info("\nStep 2: Collecting CPG demonstrations...")
    X, Y = collect_cpg_data(env, cpg, n_episodes=20)
    log.info("Collected %d samples", len(X))
    
    # Normalize
    input_mean = X.mean(axis=0)
    input_std = X.std(axis=0)
    input_std[input_std < 1e-8] = 1.0
    X_norm = (X - input_mean) / input_std
    
    # Step 3: Train RC (supervised - warm start)
    log.info("\nStep 3: Supervised pre-training (warm-start)...")
    n_inputs = X.shape[1]
    n_outputs = Y.shape[1]
    
    np.random.seed(42)
    Win = np.random.uniform(-2.0, 2.0, size=(n_reservoir, n_inputs))
    density = 0.1
    mask = (np.random.rand(n_reservoir, n_reservoir) < density)
    W = np.random.uniform(-0.5, 0.5, size=(n_reservoir, n_reservoir)) * mask
    eigvals = np.linalg.eigvals(W)
    max_abs = np.max(np.abs(eigvals))
    if max_abs > 1e-12:
        W *= 0.95 / max_abs
    
    reservoir = Reservoir(units=n_reservoir, input_dim=n_inputs, Win=Win, W=W)
    readout = Ridge(input_dim=n_reservoir, output_dim=n_outputs, ridge=1e-6)
    esn = Model([reservoir, readout], edges=[(reservoir, 0, readout)])
    esn.fit(X_norm, Y)
    
    # Extract trained weights
    # ReservoirPy stores weights as (input_dim, output_dim) and bias separately
    Wout = readout.Wout.T  # Transpose to get (n_outputs, n_reservoir)
    bias = readout.bias if hasattr(readout, 'bias') else np.zeros(n_outputs)
    log.info("Supervised training complete. Readout shape: %s, bias shape: %s", Wout.shape, bias.shape)
    
    # Step 4: Create RL policy
    log.info("\nStep 4: Creating RL-trainable policy...")
    policy = RCPolicy(reservoir, Wout, bias, input_mean, input_std)
    
    # Evaluate supervised performance
    sup_reward, sup_dist = evaluate_policy(env, policy, n_episodes=5)
    log.info("Supervised RC: reward=%.2f, distance=%.3f", sup_reward, sup_dist)
    
    # Step 5: RL fine-tuning
    log.info("\nStep 5: RL fine-tuning with Evolution Strategies...")
    best_weights, rl_reward, rl_dist = evolution_strategy_optimize(
        env, policy, 
        n_generations=RL_GENERATIONS,
        population_size=POPULATION_SIZE,
        noise_std=0.05,
        learning_rate=0.01
    )
    
    # Step 6: Save models
    log.info("\nStep 6: Saving models...")
    
    # Save supervised model
    sup_model_data = {
        "rc_model": esn,
        "input_mean": input_mean,
        "input_std": input_std,
        "friction": friction,
        "cpg_fwd": cpg_score,
        "rc_fwd": sup_dist,
        "type": "supervised"
    }
    sup_path = os.path.join(OUTPUT_DIR, f"rc_supervised_friction_{friction}.pkl")
    with open(sup_path, "wb") as f:
        pickle.dump(sup_model_data, f)
    
    # Save RL-optimized model
    # Extract final weights and bias from policy
    final_weights = policy.get_weights()
    n_outputs = policy.Wout.shape[0]
    n_reservoir_dim = policy.Wout.shape[1]
    final_Wout = final_weights[:n_outputs * n_reservoir_dim].reshape(n_outputs, n_reservoir_dim)
    final_bias = final_weights[n_outputs * n_reservoir_dim:]
    
    rl_model_data = {
        "reservoir": reservoir,
        "readout_weights": final_Wout,
        "readout_bias": final_bias,
        "input_mean": input_mean,
        "input_std": input_std,
        "friction": friction,
        "cpg_fwd": cpg_score,
        "supervised_fwd": sup_dist,
        "rl_fwd": rl_dist,
        "improvement": (rl_dist / cpg_score) * 100 if cpg_score > 0 else 0,
        "type": "rl_optimized"
    }
    rl_path = os.path.join(OUTPUT_DIR, f"rc_rl_optimized_friction_{friction}.pkl")
    with open(rl_path, "wb") as f:
        pickle.dump(rl_model_data, f)
    
    # Summary
    log.info("\n" + "=" * 60)
    log.info("FINAL COMPARISON")
    log.info("=" * 60)
    log.info("CPG Teacher:      %.3fm forward", cpg_score)
    log.info("RC (Supervised):  %.3fm forward (%.0f%% of CPG)", 
             sup_dist, (sup_dist/cpg_score)*100 if cpg_score > 0 else 0)
    log.info("RC (RL-Optimized): %.3fm forward (%.0f%% of CPG)", 
             rl_dist, (rl_dist/cpg_score)*100 if cpg_score > 0 else 0)
    
    if rl_dist > cpg_score:
        log.info("\n🎉 SUCCESS! RC exceeded CPG by %.1f%%!", ((rl_dist/cpg_score)-1)*100)
    elif rl_dist > sup_dist:
        log.info("\n✅ RL improved RC by %.1f%%", ((rl_dist/sup_dist)-1)*100)
    
    log.info("\nModels saved:")
    log.info("  Supervised: %s", sup_path)
    log.info("  RL-Optimized: %s", rl_path)
    
    env.close()


if __name__ == "__main__":
    main()
