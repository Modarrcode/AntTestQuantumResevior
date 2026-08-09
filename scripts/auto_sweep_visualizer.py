import subprocess, shlex, csv, re, os

MODEL='optimized_rc_80_model/rc_80_optimized.pkl'
PY='/home/modarr/Desktop/Masters Project/.venv_playback/bin/python'
SCRIPT='train_rc_optimized_80.py'
OUT_CSV='sweep_quick_results.csv'

cpg_mixes = [0.0, 0.05, 0.08, 0.12]
action_smooths = [0.75, 0.85]
episode_length = 300

if not os.path.exists(MODEL):
    print('Model not found:', MODEL)
    raise SystemExit(1)

results = []
for mix in cpg_mixes:
    for smooth in action_smooths:
        gif_name = f'ant_quick_a{int(smooth*100)}_m{int(mix*100)}.gif'
        cmd_list = [PY, SCRIPT, '--visualize-model', '--model-path', MODEL, '--record-gif', gif_name,
                '--episode-length', str(episode_length), '--action-scale', '1.6',
                '--cpg-speed-scale', '1.0', '--cpg-amp-scale', '1.0',
                '--action-smooth', str(smooth), '--cpg-mix', str(mix)]
        print('Running', ' '.join(shlex.quote(x) for x in cmd_list))
        p = subprocess.Popen(cmd_list, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        out_lines = []
        while True:
            line = p.stdout.readline()
            if not line and p.poll() is not None:
                break
            if line:
                print(line.rstrip())
                out_lines.append(line)
        ret = p.wait()
        out = '\n'.join(out_lines)
        m = re.search(r'Episode distance:\s*([0-9]+\.[0-9]+)', out)
        dist = float(m.group(1)) if m else 0.0
        results.append({'cpg_mix': mix, 'action_smooth': smooth, 'distance': dist, 'gif': gif_name})

# save CSV
with open(OUT_CSV, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['cpg_mix','action_smooth','distance','gif'])
    w.writeheader()
    for r in results:
        w.writerow(r)

# pick best
best = max(results, key=lambda r: r['distance'])
print('Best combo:', best)

# generate full gif for best
full_gif = 'ant_rc_best_full.gif'
full_cmd = [PY, SCRIPT, '--visualize-model', '--model-path', MODEL, '--record-gif', full_gif,
            '--episode-length', '600', '--action-scale', '1.6', '--cpg-speed-scale', '1.0',
            '--cpg-amp-scale', '1.0', '--action-smooth', str(best['action_smooth']), '--cpg-mix', str(best['cpg_mix'])]
print('Generating full gif with best combo:', ' '.join(shlex.quote(x) for x in full_cmd))
subprocess.run(full_cmd)

# create comparison gif
comp = 'ant_rc_compare_best.gif'
comp_cmd = [PY, 'compare_gifs_better.py', 'ant_rc_switch_surface_regen.gif', full_gif, comp]
print('Creating comparison gif:', ' '.join(shlex.quote(x) for x in comp_cmd))
subprocess.run(comp_cmd)
print('Saved results to', OUT_CSV, 'and', comp)
