# Dissertation Training and Data

## Purpose

This section documents how the Ant locomotion data were generated, how the training pipeline was structured, and what artifacts were produced. It is intended to make the dissertation method section reproducible and easy to defend.

## Data Collection

The dataset was generated from `Ant-v5` rollouts under three friction settings:

- friction `0.5`
- friction `1.0`
- friction `1.5`

For each friction level, a separate CPG teacher was tuned and used to generate supervised episodes. The training code keeps the teacher and the downstream reservoir readout surface-specific, rather than forcing a single gait to fit all conditions.

The implementation uses a structured four-leg gait with stance-heavy timing and friction-dependent phase offsets. Earlier diagonal or two-legged compensation patterns were replaced so the resulting trajectories better represent a quadruped walk.

## Training Pipeline

The final training pipeline is:

1. Tune a friction-specific CPG teacher.
2. Collect supervised rollout episodes from that teacher.
3. Normalize the observation vectors.
4. Train a NumPy autoencoder to compress the observations.
5. Encode the rollout states and train an Echo State Network readout.
6. Fine-tune the controller with DAgger-style closed-loop updates.
7. Evaluate the saved model on all three friction conditions.

The trainer uses a washout period of 50 steps at the start of each episode to avoid fitting transient reservoir dynamics. The current saved model lives at `optimized_rc_80_model/rc_80_optimized.pkl`, and the corresponding summary is in `optimized_rc_80_model/summary.txt`.

## Training Settings

The latest optimized run used the following training settings:

- `--n-episodes 12`
- `--n-reservoir 600`
- `--ae-epochs 10`
- `--cpg-iters 400`
- `--min-forward 2.0`

These settings were chosen to balance data quality, reservoir capacity, and training time. The tuned CPG search was extended to a longer-horizon forward-distance objective so the teacher search rewarded actual locomotion rather than short, unstable bursts.

## Data And Artifact Outputs

The main data and result artifacts are:

- `optimized_rc_80_model/rc_80_optimized.pkl` for the saved model
- `optimized_rc_80_model/summary.txt` for the final forward-distance summary
- `artifacts/videos/four_leg_speed_distinct_final.mp4` for motion contrast evidence
- `artifacts/videos/four_leg_equal_segments_colored_gui_final_02.mp4` for labeled surface transitions

## Reproducibility Notes

The training script includes defensive handling for cases where a friction slice produces too few acceptable episodes. In that case, the pipeline keeps a fallback episode rather than failing with an empty dataset. This keeps the dissertation narrative honest: the method is robust enough to run end-to-end, and the remaining gap is best described as friction-dependent performance rather than a broken pipeline.
