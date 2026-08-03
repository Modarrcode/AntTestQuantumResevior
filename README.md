# AntTestQuantumReservoir

This repository contains experiments and training code for reservoir computing (RC) with CPG controllers applied to the OpenAI Gym Ant environment.

Key files and folders:
- `train_rc_optimized_80.py` — training and visualization pipeline (AE + ESN + DAgger fine-tuning).
- `visualize_*` scripts — visualization helpers and GIF/MP4 exporters.
- `optimized_rc_80_model/` — excluded from repo (models and large artifacts).

Notes:
- Large model files, datasets, and media are intentionally excluded by `.gitignore`. If you want specific artifacts included, let me know and I can add Git LFS configuration.
- To reproduce training, create a Python virtualenv and install dependencies (numpy, gymnasium, Pillow, etc.).

Example quick start:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt  # create if needed
python train_rc_optimized_80.py --help
```
