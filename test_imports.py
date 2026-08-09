This file has been removed as part of cleanup.
#!/usr/bin/env python
"""Minimal test of the training pipeline."""
import sys
print("SCRIPT STARTED", flush=True)

import logging
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)
print("LOGGING CONFIGURED", flush=True)

import numpy as np
print("NUMPY IMPORTED", flush=True)

import gymnasium as gym
print("GYM IMPORTED", flush=True)

from reservoirpy.nodes import Reservoir, Ridge
from reservoirpy.model import Model
print("RESERVOIRPY IMPORTED", flush=True)

import pickle
import os
print("PICKLE/OS IMPORTED", flush=True)

print("ALL IMPORTS SUCCESSFUL", flush=True)

# Try minimal environment creation
print("Creating environment...", flush=True)
env = gym.make("Ant-v5", render_mode="rgb_array")
print("Environment created successfully!", flush=True)
env.close()
print("Environment closed", flush=True)

print("TEST COMPLETE", flush=True)
#!/usr/bin/env python
"""Minimal test of the training pipeline."""
import sys
print("SCRIPT STARTED", flush=True)

import logging
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)
print("LOGGING CONFIGURED", flush=True)

import numpy as np
print("NUMPY IMPORTED", flush=True)

import gymnasium as gym
print("GYM IMPORTED", flush=True)

from reservoirpy.nodes import Reservoir, Ridge
from reservoirpy.model import Model
print("RESERVOIRPY IMPORTED", flush=True)

import pickle
import os
print("PICKLE/OS IMPORTED", flush=True)

print("ALL IMPORTS SUCCESSFUL", flush=True)

# Try minimal environment creation
print("Creating environment...", flush=True)
env = gym.make("Ant-v5", render_mode="rgb_array")
print("Environment created successfully!", flush=True)
env.close()
print("Environment closed", flush=True)

print("TEST COMPLETE", flush=True)
