# Scripts: overview

This folder contains runnable entrypoints for training, visualization, sweeping, and utilities used in the Ant RC project. The file `train_rc_optimized_80.py` is the primary training pipeline used to build the `optimized_rc_80_model` artifacts and produce presentation visuals.

## Quickstart

1. Create and activate a Python virtualenv:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r ../requirements.txt   # create if needed
```
2. Run a short training / reproduce example (example used during development):
```bash
python train_rc_optimized_80.py --n-episodes 24 --n-reservoir 800 --cpg-iters 600 --ae-epochs 40
```
3. Visualize a saved model (example):
```bash
python scripts/visualize_best_model.py --model ../optimized_rc_80_model/rc_80_optimized.pkl --action-smooth 0.75 --cpg-mix 0.05
```

Note: many scripts expect to be run from the repository root; adjust relative paths as needed.

## Key CLI knobs used in practice
- `--n-episodes` — number of collected episodes per friction during training
- `--n-reservoir` — ESN reservoir size
- `--cpg-iters` — iterations to tune the open-loop CPG
- `--ae-epochs` — autoencoder training epochs
- `--action-smooth` / `--cpg-mix` — visualization-only smoothing / mixing to reduce pauses during demos

## Where outputs live (presentation-ready)
- `artifacts/videos/` — MP4 presentation clips
- `artifacts/gifs/` — generated GIFs
- `artifacts/csv/` — sweep CSVs and results

## Development history (how `train_rc_optimized_80.py` reached this state)
This is a concise chronological list of the design decisions, features and fixes applied while iterating the script and surrounding tooling:

- Added an end-to-end RC training pipeline (AE encoder + Echo State Network readouts) and saved model `optimized_rc_80_model/rc_80_optimized.pkl`.
- Implemented per-friction CPG tuning (`--cpg-iters`) and hill-climbing to find open-loop gait parameters.
- Collected multi-friction episodes with checks for successful forward progress and added checkpointing after collection milestones (`_save_checkpoint`).
- Trained a small autoencoder (`NumpyAutoencoder`) to reduce input dimensionality before fitting ESN readouts.
- Performed grid search over ESN hyperparameters and selected best config by validation MSE.
- Added closed-loop fine-tuning (DAgger) rounds to collect readouts more robust to the controller's distribution.
- Implemented a visualization suite (`visualize_saved_model`) that:
  - Records single-rollout visualizations and saves GIF/MP4s.
  - Annotates frames when the RC switches CPGs (environment-driven switching).
  - Applies `--action-smooth` and `--cpg-mix` for demo smoothing without retraining.
- Implemented environment-driven friction zones and visual tinting (`set_floor_friction`, `set_floor_color`) so switches are visible.
- Added flip detection utilities (`check_flips.py`) to automatically detect falling runs (base z and body-up checks).
- Added robust GIF→MP4 conversion utilities (ffmpeg) and created comparison visuals (side-by-side) for presentations.
- Instrumented the pipeline with checkpoints after major steps (post-collection, post-AE, post-search, post-DAgger) to mitigate long-run failures.
- Addressed unpickle issues by ensuring model classes are importable during loading (pickling compatibility fixes).
- Organized repository for presentation: added Git LFS for large media, grouped media under `artifacts/`, created `reports/` and `logs/`, and moved runnable scripts into `scripts/` for clarity.

## Files to inspect for details
- `scripts/train_rc_optimized_80.py` — main training & visualization pipeline
- `scripts/visualize_best_model.py` — visualization helper and annotation logic
- `scripts/auto_sweep_full.py` / `scripts/auto_sweep_visualizer.py` — parameter sweeps and quick visual search
- `scripts/check_flips.py` — flip detection and validation harness

## Next recommended steps
- Generate a `requirements.txt` from the environment to make reproduction easier (I can do this).
- If you will share the repo, consider adding a `scripts/README.md` (this file) and a short `USAGE.md` with example commands for each script — I can expand it to cover more scripts on request.
