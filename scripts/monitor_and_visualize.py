#!/usr/bin/env python3
"""
Monitor the optimized model file and run the visualizer when it's updated.
Usage: python monitor_and_visualize.py

This script polls the model file mtime and when it increases, it runs the
visualizer using the project's playback venv. Logs are written to
`monitor_and_visualize.log`.
"""
import os
import time
import subprocess
import sys

ROOT = os.path.dirname(__file__)
MODEL = os.path.join(ROOT, 'optimized_rc_80_model', 'rc_80_optimized.pkl')
LOG = os.path.join(ROOT, 'monitor_and_visualize.log')
PYTHON = os.path.join(ROOT, '..', '.venv_playback', 'bin', 'python')
# fallback to system python
if not os.path.exists(PYTHON):
    PYTHON = sys.executable

VIS_CMD = [
    PYTHON,
    os.path.join(ROOT, 'train_rc_optimized_80.py'),
    '--visualize-model',
    '--model-path', MODEL,
    '--record-gif', os.path.join(ROOT, 'ant_rc_switch_surface_auto.gif'),
    '--episode-length', '1500',
    '--switch-distance', '0.5',
    '--switch-sequence', '0.5,1.0,1.5',
    '--action-scale', '1.0',
]

POLL_INTERVAL = 8


def write(msg):
    ts = time.strftime('%Y-%m-%d %H:%M:%S')
    with open(LOG, 'a') as f:
        f.write(f"[{ts}] {msg}\n")
    print(msg)


def main():
    write('monitor_and_visualize started')
    if not os.path.exists(MODEL):
        write(f'Model not found at {MODEL}; waiting for creation...')
        prev_mtime = 0
    else:
        prev_mtime = os.path.getmtime(MODEL)
        write(f'Found existing model mtime={prev_mtime}')

    try:
        while True:
            if os.path.exists(MODEL):
                mtime = os.path.getmtime(MODEL)
                if mtime > prev_mtime:
                    write(f'Model updated: mtime={mtime} (prev={prev_mtime})')
                    write('Starting visualizer...')
                    with open(LOG, 'a') as outf:
                        proc = subprocess.Popen(VIS_CMD, stdout=outf, stderr=outf)
                        write(f'Visualizer started with PID {proc.pid}')
                        ret = proc.wait()
                        write(f'Visualizer finished with exit code {ret}')
                    prev_mtime = mtime
                    write('Monitoring continues for further updates...')
            time.sleep(POLL_INTERVAL)
    except KeyboardInterrupt:
        write('monitor_and_visualize stopped by KeyboardInterrupt')


if __name__ == '__main__':
    main()
