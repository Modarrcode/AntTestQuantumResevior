This file has been removed as part of cleanup.
import numpy as np
import gymnasium as gym
from visualize_rl_vs_cpg import CPGController, load_rl_policy, set_floor_friction, get_base_x

env = gym.make("Ant-v5", render_mode="rgb_array")
set_floor_friction(env, 1.0)

# Load actual trained CPG
cpg_vec = np.load("datasets/best_cpg_f2.npy")
cpg = CPGController.from_vector(cpg_vec, 8)

# Test CPG
print("=" * 60)
print("TESTING CPG (best_cpg_f2)")
print("=" * 60)
obs, _ = env.reset()
start_x = get_base_x(env)

total_reward_cpg = 0
steps_cpg = 0
for step in range(500):
    action = np.asarray(cpg.step(step * 0.005)).flatten()
    obs, reward, term, trunc, _ = env.step(action)
    total_reward_cpg += reward
    steps_cpg = step + 1
    if term or trunc:
        break

end_x = get_base_x(env)
distance_cpg = end_x - start_x
print(f"CPG - Distance: {distance_cpg:.4f}m, Total Reward: {total_reward_cpg:.2f}, Steps: {steps_cpg}")
print()

# Test RL
print("=" * 60)
print("TESTING RC+RL")
print("=" * 60)
policy, kind, path = load_rl_policy(".", 1.0)
obs, _ = env.reset()
policy.reset()
start_x = get_base_x(env)

total_reward_rl = 0
steps_rl = 0
for step in range(500):
    action = np.asarray(policy.predict(obs)).flatten()
    action = np.clip(action, env.action_space.low, env.action_space.high)
    obs, reward, term, trunc, _ = env.step(action)
    total_reward_rl += reward
    steps_rl = step + 1
    if term or trunc:
        break

end_x = get_base_x(env)
distance_rl = end_x - start_x
print(f"RC+RL - Distance: {distance_rl:.4f}m, Total Reward: {total_reward_rl:.2f}, Steps: {steps_rl}")
print()

# Comparison
print("=" * 60)
print("COMPARISON")
print("=" * 60)
ratio = distance_rl / distance_cpg if distance_cpg > 0 else 0
print(f"CPG moves:     {distance_cpg:.4f}m (reward: {total_reward_cpg:.2f})")
print(f"RC+RL moves:   {distance_rl:.4f}m (reward: {total_reward_rl:.2f})")
print()
if distance_cpg > 0:
    print(f"RC+RL is {ratio*100:.1f}% of CPG distance")
    if ratio >= 0.9:
        print("✅ RC+RL performs comparably to CPG!")
    elif ratio >= 0.5:
        print("⚠️  RC+RL is slower than CPG")
    else:
        print("❌ RC+RL significantly underperforms CPG")
else:
    print(f"CPG moved backward, RC+RL clearly superior")

env.close()
import numpy as np
import gymnasium as gym
from visualize_rl_vs_cpg import CPGController, load_rl_policy, set_floor_friction, get_base_x

env = gym.make("Ant-v5", render_mode="rgb_array")
set_floor_friction(env, 1.0)

# Load actual trained CPG
cpg_vec = np.load("datasets/best_cpg_f2.npy")
cpg = CPGController.from_vector(cpg_vec, 8)

# Test CPG
print("=" * 60)
print("TESTING CPG (best_cpg_f2)")
print("=" * 60)
obs, _ = env.reset()
start_x = get_base_x(env)

total_reward_cpg = 0
steps_cpg = 0
for step in range(500):
    action = np.asarray(cpg.step(step * 0.005)).flatten()
    obs, reward, term, trunc, _ = env.step(action)
    total_reward_cpg += reward
    steps_cpg = step + 1
    if term or trunc:
        break

end_x = get_base_x(env)
distance_cpg = end_x - start_x
print(f"CPG - Distance: {distance_cpg:.4f}m, Total Reward: {total_reward_cpg:.2f}, Steps: {steps_cpg}")
print()

# Test RL
print("=" * 60)
print("TESTING RC+RL")
print("=" * 60)
policy, kind, path = load_rl_policy(".", 1.0)
obs, _ = env.reset()
policy.reset()
start_x = get_base_x(env)

total_reward_rl = 0
steps_rl = 0
for step in range(500):
    action = np.asarray(policy.predict(obs)).flatten()
    action = np.clip(action, env.action_space.low, env.action_space.high)
    obs, reward, term, trunc, _ = env.step(action)
    total_reward_rl += reward
    steps_rl = step + 1
    if term or trunc:
        break

end_x = get_base_x(env)
distance_rl = end_x - start_x
print(f"RC+RL - Distance: {distance_rl:.4f}m, Total Reward: {total_reward_rl:.2f}, Steps: {steps_rl}")
print()

# Comparison
print("=" * 60)
print("COMPARISON")
print("=" * 60)
ratio = distance_rl / distance_cpg if distance_cpg > 0 else 0
print(f"CPG moves:     {distance_cpg:.4f}m (reward: {total_reward_cpg:.2f})")
print(f"RC+RL moves:   {distance_rl:.4f}m (reward: {total_reward_rl:.2f})")
print()
if distance_cpg > 0:
    print(f"RC+RL is {ratio*100:.1f}% of CPG distance")
    if ratio >= 0.9:
        print("✅ RC+RL performs comparably to CPG!")
    elif ratio >= 0.5:
        print("⚠️  RC+RL is slower than CPG")
    else:
        print("❌ RC+RL significantly underperforms CPG")
else:
    print(f"CPG moved backward, RC+RL clearly superior")

env.close()
