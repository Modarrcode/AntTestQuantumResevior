# Documentation for `train_rc_optimized_80.py`

This document explains `train_rc_optimized_80.py` in easy-to-follow sections. It is written for readers who are not deeply familiar with CPGs (Central Pattern Generators), RC (Reservoir Computing), or the specific training flow used here.

---

## 1. What this script does

This script trains a robot controller that tries to match the performance of a simple rhythmic controller called a CPG. The goal is to reach at least `80%` of the forward motion produced by the CPG, across different surface frictions.

The script uses:
- a CPG as a teacher to generate reference actions,
- a reservoir computer (Echo State Network) to learn from those actions,
- an autoencoder to compress the controller input,
- a DAgger-style closed-loop fine-tuning step to improve performance.

It then evaluates the learned controller and reports how closely it matches the CPG.

---

## 2. High-level structure

The file is organized into these main chunks:

1. Imports and global settings
2. Environment helper functions
3. CPG controller and tuning functions
4. Reservoir computing implementation (`EchoStateNetwork`)
5. Autoencoder implementation (`NumpyAutoencoder`)
6. Data collection function
7. Validation and configuration selection helpers
8. Closed-loop fine-tuning code
9. Main training flow: `train_optimized_rc()`
10. Command-line interface

---

## 3. Imports and global settings

At the top, the script imports standard libraries and configures logging.

Important constants:
- `TRAIN_FRICTIONS = [0.5, 1.0, 1.5]` — the surface friction values used during training.
- `OUTPUT_DIR = "optimized_rc_80_model"` — where results are saved.
- `WASHOUT = 50` — the first 50 time steps of each episode are ignored when training the reservoir.

The washout period is common in reservoir computing: the reservoir needs a short warmup before its state becomes useful.

---

## 4. Environment helpers

### `set_floor_friction(env, mu)`

This function changes the friction of the floor in the MuJoCo `Ant-v5` environment.

Why it exists:
- The script trains controllers for multiple friction values,
- so it must change the environment physics without creating a different environment from scratch.

### `get_base_x(env)`

This reads the robot’s current x-position from the environment state.

Used for:
- measuring how far the robot moved forward during an episode.

---

## 5. CPG controller and tuning

A CPG is a simple rhythmic signal generator. In this script, it generates action vectors for the robot at each time step.

### `CPGController`

This class represents the CPG.

- `n_actions` is the number of output joints/actions (8 for `Ant-v5`).
- `omega` is the rhythm speed.
- `amplitudes`, `phases`, and `offsets` shape the waveform.

The `step(t)` method returns one action vector for time `t`.

### `tune_cpg(env, cpg_iters=1000)`

This function finds a good CPG configuration by:
1. Random search to find a promising parameter vector.
2. Hill-climbing refinement to improve the result.

It evaluates each candidate using `_eval_cpg_vec(env, vec)`:
- run the environment for 500 steps,
- return the forward distance traveled.

The tuned CPG is used as the reference teacher for the reservoir.

---

## 6. Reservoir computing implementation

### `EchoStateNetwork`

This is the reservoir computer used to learn the CPG’s behavior.

Key ideas:
- The reservoir is a fixed random recurrent network.
- Only the output weights are learned.
- The reservoir state is updated sequentially, which helps it remember recent history.

Important parts:
- `__init__`: builds random input and recurrent weights.
- `reset()`: clears the reservoir state before each episode.
- `update(x)`: runs one time step of reservoir dynamics.
- `collect_states(episodes_X, washout)`: collects reservoir states for each episode after the washout period.
- `fit(episodes_X, episodes_Y, ridge, washout)`: trains the output layer using ridge regression.
- `predict(x)`: makes a single prediction and updates the state.
- `predict_sequence(X, washout)`: predicts over a sequence.

The reservoir uses a feature vector that includes: bias, input, and reservoir state.

### Why this is useful

Reservoir computing is effective at learning temporal patterns with limited training. It is well suited for robot controllers because it can capture time-varying dynamics without training the entire recurrent network.

---

## 7. Autoencoder implementation

### `NumpyAutoencoder`

This is a simple neural network that compresses the controller inputs into a smaller vector.

Why an autoencoder is used:
- the input vector is large,
- the reservoir works better when the input dimensionality is smaller,
- the autoencoder can learn a compact representation of the relevant input features.

Main methods:
- `_forward(x)`: computes encoder and decoder outputs.
- `fit(x, epochs, batch_size, lr, weight_decay)`: trains the autoencoder by minimizing reconstruction error.
- `encode(x)`: maps inputs into the compressed latent space.
- `reconstruct(x)`: reconstructs input from encoding.

This autoencoder is trained on normalized inputs before the reservoir is trained.

---

## 8. Data collection

### `collect_episodes(env, cpg, n_episodes, min_forward)`

This function collects training episodes using the tuned CPG.

Steps:
1. Reset the environment.
2. Run the CPG to produce actions.
3. Save observations and actions for each step.
4. Keep only episodes that move forward at least `min_forward` meters.

Each observation contains:
- the raw observation from the environment,
- the velocity change from the previous step,
- the CPG phase values,
- the CPG output itself.

This gives the reservoir enough information to learn the CPG’s rhythm.

---

## 9. Config selection and validation

### `evaluate_config_mse(esn, val_episodes_X, val_episodes_Y, washout)`

This evaluates the reservoir’s prediction error on held-out validation episodes.

It measures mean squared error (MSE) between the predicted actions and the CPG actions.

### Why validation matters

The script tries multiple reservoir hyperparameters and chooses the best one based on validation error.

---

## 10. Closed-loop fine-tuning with DAgger

### DAgger overview

DAgger stands for Dataset Aggregation. It is a way to improve a learned policy by letting the policy act in the environment, then labeling the visited states with the teacher’s actions.

That helps the policy recover from its own mistakes.

### `_dagger_rollout(...)`

This function uses the current reservoir policy to run episodes in closed loop.

- It uses the policy’s output as the action sent to the environment.
- It keeps the teacher action from the CPG as the label.
- It saves the resulting states and labels for later retraining.

### `closed_loop_finetune(...)`

This is the fine-tuning loop.

Key ideas:
- It maintains a separate pool of episodes for each friction level.
- For each friction, it runs the current learned controller and collects new data.
- It retrains a per-friction readout from the aggregated data.
- It repeats for several rounds.

This makes the reservoir more robust in closed-loop execution.

---

## 11. Main training flow: `train_optimized_rc(...)`

This is the central function that runs training from start to finish.

### Step-by-step summary

1. Create the output directory.
2. For each friction value:
   - create an `Ant-v5` environment,
   - tune a CPG for that friction,
   - collect good episodes using the tuned CPG,
   - store the episode data and friction labels.
3. Normalize all inputs and outputs.
4. Train the autoencoder on the normalized inputs.
5. Encode the training inputs using the autoencoder.
6. Split the episodes into training and validation sets.
7. Search reservoir hyperparameters and choose the best config.
8. Perform closed-loop fine-tuning with DAgger.
9. Evaluate the final controller on each friction.
10. Save the trained model and a summary report.

### Why these steps matter

- Data collection establishes the reference behavior.
- normalization keeps the network stable.
- the autoencoder reduces input size.
- hyperparameter search finds a good reservoir.
- DAgger improves actual control performance.
- evaluation measures how close the learned controller is to the CPG.

---

## 12. Command-line interface

At the bottom of the file, the script reads command-line arguments:

- `--n-episodes` — number of episodes per friction.
- `--n-reservoir` — reservoir size.
- `--cpg-iters` — how many search iterations to use when tuning the CPG.
- `--min-forward` — the minimum forward distance required for a training episode.
- `--ae-hidden-dim` — hidden layer size in the autoencoder.
- `--ae-latent-dim` — latent dimension of the autoencoder.
- `--ae-epochs` — training epochs for the autoencoder.

These let you experiment with speed and performance.

---

## 13. How to read the final results

The script writes two output files:

- `optimized_rc_80_model/rc_80_optimized.pkl` — the saved model and normalizers.
- `optimized_rc_80_model/summary.txt` — a human-readable summary.

The summary reports:
- the friction values tested,
- the learned controller’s forward distance,
- the CPG forward distance,
- the alignment percentage.

The alignment percentage is the key metric: it tells you how close the learned controller is to the CPG.

---

## 14. Short definitions for non-experts

### What is a CPG?
A Central Pattern Generator is a simple rhythmic generator. In this script, it produces repeated action patterns that make the robot walk.

### What is Reservoir Computing?
It is a machine learning approach where a fixed random network (the reservoir) transforms inputs into a rich dynamic state. Only the output layer is trained, which makes learning fast.

### What is an Autoencoder?
An autoencoder is a small neural network that learns to compress input data into a smaller representation and then reconstruct it.

### What is DAgger?
DAgger is a technique that improves a learned policy by collecting new data from the policy’s own behavior, then labeling that behavior with a teacher.

---

## 15. Practical advice

If you want to use this code for another robot or task, you should:
- change the environment and action dimensions,
- update observation features to match the new task,
- keep the basic flow: collect teacher data, normalize, encode, train reservoir, fine-tune, evaluate.

If you want a lighter test run, reduce `--n-episodes`, `--cpg-iters`, `--ae-epochs`, and `--n-reservoir`.

---

## 16. Suggested reading order for the code

1. Start with `train_optimized_rc()` to understand the overall flow.
2. Read the CPG section to see how the teacher is generated.
3. Read `EchoStateNetwork` to understand the reservoir learner.
4. Read `NumpyAutoencoder` to see how input compression happens.
5. Read `collect_episodes` and `closed_loop_finetune` to see how the data is collected and improved.

---

## 17. What this script does not cover

This script is not a general reinforcement learning agent. It is an imitation-style controller that learns from a CPG teacher. It also assumes the same observation and action format as `Ant-v5`.

If you want to adapt it to another environment, the main changes are in the observation construction and the teacher signal.
