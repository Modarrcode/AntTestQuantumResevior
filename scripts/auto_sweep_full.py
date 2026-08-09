import subprocess, shlex, csv, os, itertools

PY='/home/modarr/Desktop/Masters Project/.venv_playback/bin/python'
SCRIPT='train_rc_optimized_80.py'
MODEL='optimized_rc_80_model/rc_80_optimized.pkl'
OUT_CSV='sweep_full_results.csv'

# parameter grid (moderate size)
action_scales = [1.2, 1.4, 1.6]
cpg_speeds = [1.0, 1.25]
cpg_amps = [1.0]
cpg_mixes = [0.0, 0.05, 0.1]
action_smooths = [0.6, 0.75, 0.85]

episode_short = 300
episode_full = 600
num_top = 3

if not os.path.exists(MODEL):
    print('Model missing:', MODEL)
    raise SystemExit(1)

results = []
all_combos = list(itertools.product(action_scales, cpg_speeds, cpg_amps, cpg_mixes, action_smooths))
print('Running', len(all_combos), 'short rollouts')
for (a_scale, s_speed, amp, mix, smooth) in all_combos:
    tag = f'a{int(a_scale*10)}_s{int(s_speed*100)}_amp{int(amp*10)}_m{int(mix*100)}_sm{int(smooth*100)}'
    gif = f'ant_full_quick_{tag}.gif'
    cmd = [PY, SCRIPT, '--visualize-model', '--model-path', MODEL, '--record-gif', gif,
           '--episode-length', str(episode_short), '--action-scale', str(a_scale),
           '--cpg-speed-scale', str(s_speed), '--cpg-amp-scale', str(amp),
           '--action-smooth', str(smooth), '--cpg-mix', str(mix)]
    print('Running', ' '.join(shlex.quote(x) for x in cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    out = proc.stdout + '\n' + proc.stderr
    dist = 0.0
    for line in out.splitlines():
        if 'Episode distance:' in line:
            try:
                dist = float(line.split(':')[-1].strip())
            except:
                pass
    results.append({'tag': tag, 'action_scale': a_scale, 'cpg_speed': s_speed, 'cpg_amp': amp, 'cpg_mix': mix, 'action_smooth': smooth, 'distance': dist, 'quick_gif': gif})

# save CSV
with open(OUT_CSV, 'w', newline='') as f:
    import csv
    w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
    w.writeheader()
    for r in results:
        w.writerow(r)

# pick top non-flipping combos by distance
sorted_res = sorted(results, key=lambda r: r['distance'], reverse=True)
selected = sorted_res[:num_top]
print('Top combos:', selected)

full_gifs = []
for r in selected:
    tag = r['tag']
    full_gif = f'ant_full_best_{tag}.gif'
    cmd = [PY, SCRIPT, '--visualize-model', '--model-path', MODEL, '--record-gif', full_gif,
           '--episode-length', str(episode_full), '--action-scale', str(r['action_scale']),
           '--cpg-speed-scale', str(r['cpg_speed']), '--cpg-amp-scale', str(r['cpg_amp']),
           '--action-smooth', str(r['action_smooth']), '--cpg-mix', str(r['cpg_mix'])]
    print('Generating full gif:', ' '.join(shlex.quote(x) for x in cmd))
    subprocess.run(cmd)
    full_gifs.append({'tag': tag, 'gif': full_gif, 'params': r})

# create comparison side-by-side for each full gif vs baseline
baseline = 'ant_rc_switch_surface_regen.gif'
compare_script = [PY, 'compare_gifs_better.py']
comp_outputs = []
for fg in full_gifs:
    out_comp = f'compare_{fg["tag"]}.gif'
    cmd = compare_script + [baseline, fg['gif'], out_comp]
    print('Creating comparison:', ' '.join(shlex.quote(x) for x in cmd))
    subprocess.run(cmd)
    comp_outputs.append(out_comp)

# convert full gifs to mp4 using ffmpeg if available
mp4_outputs = []
for fg in full_gifs:
    mp4 = fg['gif'].replace('.gif', '.mp4')
    gif = fg['gif']
    # ffmpeg command
    cmd = ['ffmpeg', '-y', '-i', gif, '-movflags', 'faststart', '-pix_fmt', 'yuv420p', '-vf', "scale=trunc(iw/2)*2:trunc(ih/2)*2", mp4]
    print('Converting to mp4:', ' '.join(shlex.quote(x) for x in cmd))
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        mp4_outputs.append(mp4)
    except Exception as e:
        print('ffmpeg failed:', e)

print('Done. Results CSV:', OUT_CSV)
print('Full GIFs:', [x['gif'] for x in full_gifs])
print('MP4s:', mp4_outputs)
