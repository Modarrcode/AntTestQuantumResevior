This file has been removed as part of cleanup.
import numpy as np
import gymnasium as gym
from visualize_rl_vs_cpg import set_floor_friction, get_base_x, load_rl_policy

env = gym.make('Ant-v5', render_mode='rgb_array')
set_floor_friction(env, 1.0)
policy, kind, path = load_rl_policy('.', 1.0)
print('model_kind', kind)
print('model_path', path)
obs, _ = env.reset()
policy.reset()
start_x = get_base_x(env)
for i in range(15):
    a = np.asarray(policy.predict(obs)).flatten()
    print(i, 'act_mean', float(np.mean(np.abs(a))), 'act_max', float(np.max(np.abs(a))))
    a = np.clip(a, env.action_space.low, env.action_space.high)
    obs, r, term, trunc, _ = env.step(a)
    print('   reward', float(r), 'term', term, 'trunc', trunc, 'x', get_base_x(env) - start_x)
    if term or trunc:
        break
env.close()
import numpy as np
import gymnasium as gym
from visualize_rl_vs_cpg import set_floor_friction, get_base_x, load_rl_policy

env = gym.make('Ant-v5', render_mode='rgb_array')
set_floor_friction(env, 1.0)
policy, kind, path = load_rl_policy('.', 1.0)
print('model_kind', kind)
print('model_path', path)
obs, _ = env.reset()
policy.reset()
start_x = get_base_x(env)
for i in range(15):
    a = np.asarray(policy.predict(obs)).flatten()
    print(i, 'act_mean', float(np.mean(np.abs(a))), 'act_max', float(np.max(np.abs(a))))
    a = np.clip(a, env.action_space.low, env.action_space.high)
    obs, r, term, trunc, _ = env.step(a)
    print('   reward', float(r), 'term', term, 'trunc', trunc, 'x', get_base_x(env) - start_x)
    if term or trunc:
        break
env.close()
