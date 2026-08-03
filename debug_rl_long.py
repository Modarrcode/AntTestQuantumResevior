This file has been removed as part of cleanup.
import numpy as np
import gymnasium as gym
from visualize_rl_vs_cpg import set_floor_friction, get_base_x, load_rl_policy

env = gym.make('Ant-v5', render_mode='rgb_array')
set_floor_friction(env, 1.0)
policy, kind, path = load_rl_policy('.', 1.0)
obs, _ = env.reset()
policy.reset()
start_x = get_base_x(env)
for i in range(200):
    a = np.asarray(policy.predict(obs)).flatten()
    a = np.clip(a, env.action_space.low, env.action_space.high)
    obs, r, term, trunc, _ = env.step(a)
    if i % 20 == 0:
        print(i, 'x', get_base_x(env)-start_x, 'reward', float(r), 'term', term, 'trunc', trunc)
    if term or trunc:
        print('ended_at', i, 'x', get_base_x(env)-start_x)
        break
env.close()
import numpy as np
import gymnasium as gym
from visualize_rl_vs_cpg import set_floor_friction, get_base_x, load_rl_policy

env = gym.make('Ant-v5', render_mode='rgb_array')
set_floor_friction(env, 1.0)
policy, kind, path = load_rl_policy('.', 1.0)
obs, _ = env.reset()
policy.reset()
start_x = get_base_x(env)
for i in range(200):
    a = np.asarray(policy.predict(obs)).flatten()
    a = np.clip(a, env.action_space.low, env.action_space.high)
    obs, r, term, trunc, _ = env.step(a)
    if i % 20 == 0:
        print(i, 'x', get_base_x(env)-start_x, 'reward', float(r), 'term', term, 'trunc', trunc)
    if term or trunc:
        print('ended_at', i, 'x', get_base_x(env)-start_x)
        break
env.close()
