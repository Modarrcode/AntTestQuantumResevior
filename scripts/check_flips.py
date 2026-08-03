import pickle, os, csv
import numpy as np
import gymnasium as gym
import train_rc_optimized_80 as tr
from train_rc_optimized_80 import CPGController

MODEL='optimized_rc_80_model/rc_80_optimized.pkl'
FULL_GIFS=['ant_full_best_a12_s100_amp10_m5_sm60.gif','ant_full_best_a12_s100_amp10_m10_sm60.gif','ant_full_best_a12_s100_amp10_m10_sm75.gif']

# helper to parse params from filename tag format used earlier
# tag example: ant_full_best_a12_s100_amp10_m5_sm60.gif -> a12 => action_scale=1.2, s100=>cpg_speed=1.0, m5=>cpg_mix=0.05, sm60=>action_smooth=0.6

def parse_tag(fname):
    base = os.path.basename(fname).replace('.gif','')
    parts = base.split('_')
    # find parts with a*, s*, amp*, m*, sm*
    params = {}
    for part in parts:
        if part.startswith('a') and 'a'==part[0] and part[1:].isdigit():
            params['action_scale'] = int(part[1:]) / 10.0
        if part.startswith('s') and part[1:].isdigit():
            params['cpg_speed'] = int(part[1:]) / 100.0
        if part.startswith('amp') and part[3:].isdigit():
            params['cpg_amp'] = int(part[3:]) / 10.0
        if part.startswith('m') and part[1:].isdigit():
            params['cpg_mix'] = int(part[1:]) / 100.0
        if part.startswith('sm') and part[2:].isdigit():
            params['action_smooth'] = int(part[2:]) / 100.0
    return params


def load_model(path=MODEL):
    import sys
    # some models were pickled when train script was __main__; ensure
    # those class names resolve by temporarily mapping __main__ to train module
    orig_main = sys.modules.get('__main__')
    sys.modules['__main__'] = tr
    try:
        with open(path,'rb') as f:
            d = pickle.load(f)
    finally:
        if orig_main is not None:
            sys.modules['__main__'] = orig_main
        else:
            del sys.modules['__main__']
    return d


def simulate(params, steps=600):
    d = load_model()
    esn = d['esn']
    autoencoder = d.get('autoencoder')
    cpg_per_friction = d.get('cpg_per_friction', {})
    X_mean = d['X_mean']; X_std = d['X_std']; Y_mean = d['Y_mean']; Y_std = d['Y_std']
    env = gym.make('Ant-v5')
    # run with default switch sequence [0.5,1.0,1.5]
    switch_values = [0.5,1.0,1.5]
    switch_distance = 0.18
    current_mu = switch_values[0]
    if hasattr(esn,'W_out_per_friction') and current_mu in esn.W_out_per_friction:
        esn.W_out = esn.W_out_per_friction[current_mu]
    esn.reset()
    obs,_ = env.reset()
    prev_obs = obs.copy()
    start_x = float(env.unwrapped.data.qpos[0])
    fallen_frames = 0
    min_z = float(env.unwrapped.data.qpos[2])
    for step in range(steps):
        progress = np.linalg.norm(np.asarray(env.unwrapped.data.qpos[:2]) - np.asarray([start_x,0.0]))
        # switch surfaces based on progress
        idx = int(min(len(switch_values)-1, progress // switch_distance))
        mu = switch_values[idx]
        if mu != current_mu:
            current_mu = mu
            if hasattr(esn,'W_out_per_friction') and current_mu in esn.W_out_per_friction:
                esn.W_out = esn.W_out_per_friction[current_mu]
            esn.reset()
        # compute action
        vel = obs - prev_obs
        t_sec = step * 0.02
        cpg = cpg_per_friction.get(current_mu)
        if cpg is None:
            cpg = CPGController(n_actions=8)
        try:
            cpg_out = np.clip(cpg.step(t_sec), env.action_space.low, env.action_space.high).astype(np.float64)
        except Exception:
            cpg_out = np.zeros(env.action_space.shape)
        cpg_phase = (cpg.omega * t_sec + cpg.phases).astype(np.float64)
        obs_aug = np.concatenate([obs, vel, cpg_phase, cpg_out, [current_mu]]).astype(np.float64)
        obs_norm = (obs_aug - X_mean) / X_std
        if autoencoder is not None:
            obs_input = autoencoder.encode(obs_norm.reshape(1,-1)).reshape(-1)
        else:
            obs_input = obs_norm
        action_norm = esn.predict(obs_input)
        action_scaled = action_norm * params.get('action_scale',1.0)
        action_unclipped = (action_scaled * Y_std + Y_mean)
        # mix
        mix = params.get('cpg_mix',0.0)
        if mix>0:
            action_unclipped = (1.0-mix)*action_unclipped + mix*cpg_out
        # smooth
        # (no smoothing needed for flip detection)
        action = np.clip(action_unclipped, env.action_space.low, env.action_space.high)
        prev_obs = obs.copy()
        obs, reward, term, trunc, _ = env.step(action)
        z = float(env.unwrapped.data.qpos[2])
        min_z = min(min_z, z)
        # detect upside-down using base quaternion (qpos[3:7]) if present
        q = np.asarray(env.unwrapped.data.qpos)
        up_z = 1.0
        if len(q) >= 7:
            # assume quaternion order [x,y,z,w] or [w,x,y,z]; try both
            q3 = q[3:7]
            # try as [w,x,y,z]
            w,xq,yq,zq = float(q3[0]), float(q3[1]), float(q3[2]), float(q3[3])
            up_z_wxyz = 1 - 2*(xq*xq + yq*yq)
            # try as [x,y,z,w]
            xq2,yq2,zq2,w2 = float(q3[0]), float(q3[1]), float(q3[2]), float(q3[3])
            up_z_xyzw = 1 - 2*(xq2*xq2 + yq2*yq2)
            # choose a plausible value (the same formula yields same result if mapping differs only by order of components?)
            up_z = up_z_wxyz if abs(up_z_wxyz) <= 1.1 else up_z_xyzw
        if up_z < 0.0:
            # upside-down
            fallen_frames += 1
        if z < 0.18:
            fallen_frames += 1
        if term or trunc:
            break
    env.close()
    return {'fallen_frames': fallen_frames, 'min_z': min_z, 'terminated': (term or trunc)}

if __name__=='__main__':
    for g in FULL_GIFS:
        if not os.path.exists(g):
            print('Missing', g)
            continue
        params = parse_tag(g)
        print('\nChecking', g, 'params=', params)
        res = simulate(params, steps=600)
        print('Result:', res)
