import gymnasium as gym
from reservoirpy.nodes import Reservoir, Ridge
import numpy as np

env = gym.make("Ant-v5", render_mode="human")

obs_size = env.observation_space.shape[0]
act_size = env.action_space.shape[0]

# For ReservoirPy 0.4+, supported parameters are:
# - units
# - input_dim
# - sr (spectral radius)
# - input_scaling
# - noise
# - activation
reservoir = Reservoir(
    units=500,
    input_dim=obs_size,
    sr=0.95,            # spectral radius
    input_scaling=0.5,
    noise=0.001,
    activation="tanh"
)

# Linear readout (ridge regression)
readout = Ridge(ridge=1e-6, input_dim=500, output_dim=act_size)

model = reservoir >> readout

obs, _ = env.reset()

for step in range(1000):
    obs_norm = np.tanh(obs)
    action = model.run(obs_norm.reshape(1, -1))[0]
    action = np.clip(action, -1, 1)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, _ = env.reset()

env.close()
