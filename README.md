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
