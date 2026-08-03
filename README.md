# AntTestQuantumReservoir

This repository contains experiments and training code for reservoir computing (RC) with CPG controllers applied to the OpenAI Gym Ant environment.

Key files and folders:
- `train_rc_optimized_80.py` — training and visualization pipeline (AE + ESN + DAgger fine-tuning).
- `visualize_*` scripts — visualization helpers and GIF/MP4 exporters.
- `optimized_rc_80_model/` — excluded from repo (models and large artifacts).

Organized folders (presentation-ready):
- `artifacts/videos/` — presentation MP4s (tracked via Git LFS).
- `artifacts/gifs/` — generated GIFs (tracked via Git LFS).
- `artifacts/csv/` — CSV outputs and sweep results.
- `reports/` — human-readable summaries and notes (TXT).
- `logs/` — training and monitoring logs.

Notes:
- Large model files and datasets remain excluded by `.gitignore` (see `.gitignore`).
- Presentation videos and GIFs have been added via Git LFS to keep the repository lightweight.
- If you want additional artifacts tracked (e.g., `.npy`, `.pkl`), I can enable Git LFS for those too.
- To reproduce training, create a Python virtualenv and install dependencies (numpy, gymnasium, Pillow, etc.).

Example quick start:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt  # create if needed
python train_rc_optimized_80.py --help
```

## Multilink RC/ESN Hold Controller

This repository also includes an upright multi-link holding environment and ESN-based reservoir controller scripts developed alongside the Ant experiments.

Files of interest (new):

- `multilink_mass_robot_env.py` — PyBullet/Gym environment for an upright multi-link robot with configurable payload.
- `train_multilink_rc_cpg.py` — Collects teacher episodes and trains an ESN readout for mass-conditioned holding.
- `train_single_payload.py` — Train a dedicated ESN readout for a single payload value.
- `retrain_boost_heavy.py` — Retrain the heavy-payload teacher with an extra lift to improve hold performance for heavy payloads.
- `run_multilink_rc_cpg_demo.py` — Run a saved ESN model and export a demo GIF and reward plot.

Quick demo (Windows PowerShell):

```powershell
python train_single_payload.py --payload 0.20 --output multilink_rc_hold_model_payload0.20.npz
python run_multilink_rc_cpg_demo.py multilink_rc_hold_model_payload0.20.npz --output-gif multilink_rc_hold_demo_payload0.20.gif --reward-plot multilink_rc_hold_reward_payload0.20.png
```

Backups:

If files were overwritten during recent merges, a backup folder `backup_before_merge_YYYYMMDD_HHMMSS` exists in the repo root containing the original local copies.

If you want, I can add a dedicated `requirements.txt` for the multilink scripts or expand this README with per-script flag descriptions.
