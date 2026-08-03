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

Detailed CLI reference (multilink scripts)

- `train_multilink_rc_cpg.py`
	- Description: Train a shared ESN readout across multiple payloads using the scripted hold teacher.
	- Key flags:
		- `--links` : number of links in the chain (default: 4)
		- `--masses` : comma-separated base link masses (default: "0.20,0.25,0.30,0.35")
		- `--episodes` : episodes per payload (default: 20)
		- `--steps` : steps per episode (default: 300)
		- `--reservoir` : reservoir size (default: 600)
		- `--output` : output `.npz` path (default: `multilink_rc_hold_model.npz`)

- `train_single_payload.py`
	- Description: Train a dedicated ESN readout for a single payload value.
	- Key flags:
		- `--payload` : payload mass (required)
		- `--links`, `--masses`, `--episodes`, `--steps`, `--reservoir`, `--output` (same meanings as above)

- `retrain_boost_heavy.py`
	- Description: Retrain a heavy payload using an augmented teacher that adds a constant extra lift to the target payload.
	- Key flags:
		- `--payload` : target payload mass (default: 0.40)
		- `--extra-lift` : extra constant torque per joint added to the teacher for the target payload (default: 0.30)
		- other flags: `--links`, `--masses`, `--episodes`, `--steps`, `--reservoir`, `--output`

- `run_multilink_rc_cpg_demo.py`
	- Description: Run a saved ESN model and export a visual demo (GIF) and reward plot.
	- Usage example:
		```powershell
		python run_multilink_rc_cpg_demo.py multilink_rc_hold_model_payload0.20.npz --output-gif demo.gif --reward-plot reward.png
		```

Notes:
- The multilink scripts require a working PyBullet installation for simulation. For headless runs, the scripts set `render_mode=None` during training to avoid opening a GUI.
- If you need a tailored `requirements.txt` (pinned versions from your environment), I can generate one automatically.
