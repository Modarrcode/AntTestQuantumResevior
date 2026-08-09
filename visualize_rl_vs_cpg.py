"""
Visualize CPG vs RC policies on Ant-v5 for the same friction.

This script runs CPG episodes first, then RC (reservoir computing) episodes, 
and prints metrics for an easy visual + numeric comparison.

RC model priority: improved_rc_model (supervised on CPG) > rc_rl_extended > rc_rl_legacy
"""
import argparse
import logging
import os
import pickle
import time
import warnings

import gymnasium as gym
import numpy as np

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

FRICTIONS = [0.5, 1.0, 1.5, 2.0, 2.5]


def safe_close_env(env, label: str = "env"):
    """Close env while ignoring known GLFW teardown warning/noise."""
    if env is None:
        return
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r".*GLFW library is not initialized.*")
        try:
            env.close()
        except Exception as exc:
            log.debug("Ignoring %s close error: %s", label, exc)


def set_floor_friction(env, mu: float):
    """Set floor geom friction to `mu`."""
    try:
        model = env.unwrapped.model
    except Exception:
        return

    def geom_name_at(i):
        if hasattr(model, "geom_names"):
            try:
                n = model.geom_names[i]
                return n.decode("utf-8") if isinstance(n, bytes) else str(n)
            except Exception:
                pass
        if hasattr(model, "geom"):
            try:
                g = model.geom[i]
                name = getattr(g, "name", None)
                if name is None:
                    try:
                        name = g[0]
                    except Exception:
                        name = str(g)
                return name.decode("utf-8") if isinstance(name, bytes) else str(name)
            except Exception:
                pass
        return f"geom_{i}"

    ngeom = getattr(model, "ngeom", None)
    if ngeom is None:
        try:
            ngeom = len(model.geom)
        except Exception:
            ngeom = 0

    for i in range(ngeom):
        name = geom_name_at(i)
        if "floor" in name or "ground" in name or "geom_floor" in name:
            try:
                model.geom_friction[i] = np.array([mu, 0.0, 0.0])
            except Exception:
                pass


def get_base_x(env):
    """Get base x position."""
    try:
        return float(env.unwrapped.data.qpos[0])
    except Exception:
        return 0.0


class CPGController:
    """Simple sinusoidal CPG."""

    def __init__(self, n_actions: int, omega: float = 2.0, amplitudes=None, phases=None, offsets=None):
        self.n = n_actions
        self.omega = float(omega)
        self.amplitudes = np.ones(self.n) if amplitudes is None else np.asarray(amplitudes).reshape(self.n)
        self.phases = np.zeros(self.n) if phases is None else np.asarray(phases).reshape(self.n)
        self.offsets = np.zeros(self.n) if offsets is None else np.asarray(offsets).reshape(self.n)

    @classmethod
    def from_vector(cls, vec, n_actions):
        omega = float(vec[0])
        A = vec[1:1 + n_actions]
        phi = vec[1 + n_actions:1 + 2 * n_actions]
        off = vec[1 + 2 * n_actions:1 + 3 * n_actions]
        return cls(n_actions, omega, A, phi, off)

    def step(self, t: float):
        return self.offsets + self.amplitudes * np.sin(self.omega * t + self.phases)


class RLExtendedPolicy:
    """Policy wrapper for extended RL model format (contains full rc_model)."""

    def __init__(self, model_data):
        self.rc_model = model_data["rc_model"]
        self.input_mean = model_data["input_mean"]
        self.input_std = model_data["input_std"]
        self.output_mean = model_data.get("output_mean", None)
        self.output_std = model_data.get("output_std", None)
        self.prev_obs = None

    def reset(self):
        self.rc_model.reset()
        self.prev_obs = None

    def predict(self, obs):
        if self.prev_obs is None:
            vel = np.zeros_like(obs)
        else:
            vel = obs - self.prev_obs
        obs_aug = np.concatenate([obs, vel])

        obs_norm = (obs_aug - self.input_mean) / self.input_std
        action_norm = self.rc_model.run(obs_norm.reshape(1, -1)).flatten()
        if self.output_mean is not None and self.output_std is not None:
            action = action_norm * self.output_std + self.output_mean
        else:
            action = action_norm

        self.prev_obs = obs.copy()
        return action


class RLLegacyPolicy:
    """Policy wrapper for legacy RL model format (reservoir + readout weights)."""

    def __init__(self, model_data):
        self.reservoir = model_data["reservoir"]
        self.readout_weights = np.asarray(model_data["readout_weights"])  # (n_outputs, n_res)
        self.readout_bias = np.asarray(model_data["readout_bias"])        # (n_outputs,)
        self.input_mean = model_data["input_mean"]
        self.input_std = model_data["input_std"]

    def reset(self):
        self.reservoir.reset()

    def predict(self, obs):
        obs_norm = (obs - self.input_mean) / self.input_std
        state = self.reservoir.run(obs_norm.reshape(1, -1)).flatten()
        return self.readout_weights @ state + self.readout_bias


def load_rl_policy(base_dir: str, friction: float):
    """Load best available RC policy artifact (prioritizes improved RC over legacy RL)."""
    # Try improved RC first (supervised training on CPG data)
    improved_path = os.path.join(base_dir, "improved_rc_model", f"improved_rc_friction_{friction}.pkl")
    if os.path.exists(improved_path):
        with open(improved_path, "rb") as f:
            model_data = pickle.load(f)
        return RLExtendedPolicy(model_data), "improved_rc", improved_path
    
    # Fall back to extended RL (if available)
    ext_path = os.path.join(base_dir, "rc_rl_extended", f"rc_extended_rl_friction_{friction}.pkl")
    if os.path.exists(ext_path):
        with open(ext_path, "rb") as f:
            model_data = pickle.load(f)
        return RLExtendedPolicy(model_data), "rc_rl_extended", ext_path

    # Fall back to legacy RL
    legacy_path = os.path.join(base_dir, "rc_rl_model", f"rc_rl_optimized_friction_{friction}.pkl")
    if os.path.exists(legacy_path):
        with open(legacy_path, "rb") as f:
            model_data = pickle.load(f)
        return RLLegacyPolicy(model_data), "rc_rl_legacy", legacy_path

    raise FileNotFoundError(
        "No RC model found. Expected one of:\n"
        f"  - {improved_path}\n"
        f"  - {ext_path}\n"
        f"  - {legacy_path}"
    )


def load_cpg(base_dir: str, friction: float, n_actions: int):
    """Load CPG vector from datasets/cpg_idx*.npy based on friction index."""
    try:
        idx = FRICTIONS.index(float(friction))
    except ValueError as e:
        raise ValueError(f"Friction {friction} is not in presets {FRICTIONS}") from e

    vec_path = os.path.join(base_dir, "datasets", f"cpg_idx{idx}.npy")
    if not os.path.exists(vec_path):
        raise FileNotFoundError(
            f"CPG vector not found at {vec_path}. "
            "Generate/tune CPG datasets first."
        )

    vec = np.load(vec_path)
    return CPGController.from_vector(vec, n_actions), vec_path


def rollout_policy(env, policy_name: str, policy_obj, episodes: int, episode_length: int, frame_sleep: float):
    """Run policy episodes and return aggregate metrics."""
    distances = []
    rewards = []

    for ep in range(episodes):
        if hasattr(policy_obj, "reset"):
            policy_obj.reset()

        obs, _ = env.reset()
        start_x = get_base_x(env)
        total_reward = 0.0

        for step in range(episode_length):
            t = step * 0.02
            if policy_name == "cpg":
                action = policy_obj.step(t)
            else:
                action = policy_obj.predict(obs)

            action = np.asarray(action).flatten()
            action = np.clip(action, env.action_space.low, env.action_space.high)

            obs, reward, term, trunc, _ = env.step(action)
            total_reward += float(reward)

            # Some Gymnasium/MuJoCo setups need explicit render calls to update GUI.
            try:
                env.render()
            except Exception:
                pass

            if frame_sleep > 0:
                time.sleep(frame_sleep)
            if term or trunc:
                break

        dist = get_base_x(env) - start_x
        distances.append(dist)
        rewards.append(total_reward)
        log.info("%s episode %d/%d: distance=%.3f, reward=%.2f", policy_name.upper(), ep + 1, episodes, dist, total_reward)

    return {
        "avg_dist": float(np.mean(distances)),
        "std_dist": float(np.std(distances)),
        "avg_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
    }


def rollout_side_by_side(env_cpg, env_rl, cpg, rl_policy, episodes: int, episode_length: int, frame_sleep: float):
    """Run CPG and RL in parallel lockstep using two environments."""
    cpg_distances, cpg_rewards = [], []
    rl_distances, rl_rewards = [], []

    for ep in range(episodes):
        if hasattr(cpg, "reset"):
            cpg.reset()
        if hasattr(rl_policy, "reset"):
            rl_policy.reset()

        obs_cpg, _ = env_cpg.reset()
        obs_rl, _ = env_rl.reset()
        start_x_cpg = get_base_x(env_cpg)
        start_x_rl = get_base_x(env_rl)
        total_reward_cpg = 0.0
        total_reward_rl = 0.0

        done_cpg = False
        done_rl = False
        cpg_last_step = 0
        rl_last_step = 0

        for step in range(episode_length):
            t = step * 0.02

            if not done_cpg:
                action_cpg = np.asarray(cpg.step(t)).flatten()
                action_cpg = np.clip(action_cpg, env_cpg.action_space.low, env_cpg.action_space.high)
                obs_cpg, reward_cpg, term_cpg, trunc_cpg, _ = env_cpg.step(action_cpg)
                total_reward_cpg += float(reward_cpg)
                cpg_last_step = step
                done_cpg = bool(term_cpg or trunc_cpg)

            if not done_rl:
                action_rl = np.asarray(rl_policy.predict(obs_rl)).flatten()
                action_rl = np.clip(action_rl, env_rl.action_space.low, env_rl.action_space.high)
                obs_rl, reward_rl, term_rl, trunc_rl, _ = env_rl.step(action_rl)
                total_reward_rl += float(reward_rl)
                rl_last_step = step
                done_rl = bool(term_rl or trunc_rl)

            # Force GUI refresh for both windows.
            try:
                env_cpg.render()
            except Exception:
                pass
            try:
                env_rl.render()
            except Exception:
                pass

            if frame_sleep > 0:
                time.sleep(frame_sleep)

            # End this paired episode once both policies are done.
            if done_cpg and done_rl:
                break

        cpg_dist = get_base_x(env_cpg) - start_x_cpg
        rl_dist = get_base_x(env_rl) - start_x_rl
        cpg_distances.append(cpg_dist)
        cpg_rewards.append(total_reward_cpg)
        rl_distances.append(rl_dist)
        rl_rewards.append(total_reward_rl)

        log.info(
            "SIDE-BY-SIDE episode %d/%d | CPG: dist=%.3f reward=%.2f steps=%d | RL: dist=%.3f reward=%.2f steps=%d",
            ep + 1,
            episodes,
            cpg_dist,
            total_reward_cpg,
            cpg_last_step + 1,
            rl_dist,
            total_reward_rl,
            rl_last_step + 1,
        )

    cpg_metrics = {
        "avg_dist": float(np.mean(cpg_distances)),
        "std_dist": float(np.std(cpg_distances)),
        "avg_reward": float(np.mean(cpg_rewards)),
        "std_reward": float(np.std(cpg_rewards)),
    }
    rl_metrics = {
        "avg_dist": float(np.mean(rl_distances)),
        "std_dist": float(np.std(rl_distances)),
        "avg_reward": float(np.mean(rl_rewards)),
        "std_reward": float(np.std(rl_rewards)),
    }
    return cpg_metrics, rl_metrics


def main():
    parser = argparse.ArgumentParser(description="Visual compare: CPG vs RC on Ant-v5")
    parser.add_argument("--policy", type=str, default="both", choices=["cpg", "rc", "rl", "both"], help="Which policy to run (rl is accepted as alias for rc)")
    parser.add_argument("--friction", type=float, default=1.0, help="Friction to evaluate")
    parser.add_argument("--episodes", type=int, default=2, help="Episodes per policy")
    parser.add_argument("--length", type=int, default=500, help="Max steps per episode")
    parser.add_argument("--render-mode", type=str, default="human", choices=["human", "rgb_array"], help="Render mode")
    parser.add_argument("--frame-sleep", type=float, default=0.0, help="Optional sleep per step for slower playback")
    parser.add_argument("--pause-between", type=float, default=1.5, help="Pause between CPG and RL runs")
    parser.add_argument("--side-by-side", action="store_true", help="Run CPG and RC+RL simultaneously in two GUI windows")
    args = parser.parse_args()

    # Backward compatibility: older commands used --policy rl.
    if args.policy == "rl":
        args.policy = "rc"

    base_dir = os.path.dirname(os.path.abspath(__file__))

    env_probe = gym.make("Ant-v5", render_mode="rgb_array")
    n_actions = env_probe.action_space.shape[0]
    safe_close_env(env_probe, "env_probe")

    cpg = None
    cpg_path = None
    rc_policy = None
    rc_kind = None
    rc_path = None

    if args.policy in ("cpg", "both"):
        cpg, cpg_path = load_cpg(base_dir, args.friction, n_actions)
        log.info("Loaded CPG: %s", cpg_path)

    if args.policy in ("rc", "both"):
        rc_policy, rc_kind, rc_path = load_rl_policy(base_dir, args.friction)
        log.info("Loaded RC (%s): %s", rc_kind, rc_path)

    log.info("Running visual comparison at friction %.1f", args.friction)

    cpg_metrics = None
    rc_metrics = None

    if args.side_by_side and args.policy == "both":
        if args.render_mode != "human":
            log.warning("--side-by-side is most useful with --render-mode human.")

        env_cpg = gym.make("Ant-v5", render_mode=args.render_mode)
        env_rc = gym.make("Ant-v5", render_mode=args.render_mode)
        set_floor_friction(env_cpg, args.friction)
        set_floor_friction(env_rc, args.friction)

        log.info("\n=== SIDE-BY-SIDE RUN (CPG + RC) ===")
        cpg_metrics, rc_metrics = rollout_side_by_side(
            env_cpg,
            env_rc,
            cpg,
            rc_policy,
            args.episodes,
            args.length,
            args.frame_sleep,
        )

        safe_close_env(env_cpg, "env_cpg")
        safe_close_env(env_rc, "env_rc")
    else:
        env = gym.make("Ant-v5", render_mode=args.render_mode)
        set_floor_friction(env, args.friction)

        if args.policy in ("cpg", "both"):
            log.info("\n=== CPG RUN ===")
            cpg_metrics = rollout_policy(env, "cpg", cpg, args.episodes, args.length, args.frame_sleep)

        if args.policy == "both" and args.pause_between > 0:
            log.info("Pausing %.1fs before RC run...", args.pause_between)
            time.sleep(args.pause_between)

        if args.policy in ("rc", "both"):
            log.info("\n=== RC RUN ===")
            rc_metrics = rollout_policy(env, "rc", rc_policy, args.episodes, args.length, args.frame_sleep)

        safe_close_env(env, "env")

    if args.policy == "both":
        log.info("\n" + "=" * 60)
        log.info("VISUAL COMPARISON SUMMARY (friction=%.1f)", args.friction)
        log.info("CPG    : dist=%.3f +/- %.3f, reward=%.2f +/- %.2f",
                 cpg_metrics["avg_dist"], cpg_metrics["std_dist"], cpg_metrics["avg_reward"], cpg_metrics["std_reward"])
        log.info("RC     : dist=%.3f +/- %.3f, reward=%.2f +/- %.2f",
                 rc_metrics["avg_dist"], rc_metrics["std_dist"], rc_metrics["avg_reward"], rc_metrics["std_reward"])

        if abs(cpg_metrics["avg_dist"]) > 1e-12:
            ratio = 100.0 * rc_metrics["avg_dist"] / cpg_metrics["avg_dist"]
            log.info("RC as %% of CPG (distance): %.1f%%", ratio)
        log.info("=" * 60)
    elif args.policy == "cpg" and cpg_metrics is not None:
        log.info("\n" + "=" * 60)
        log.info("CPG SUMMARY (friction=%.1f)", args.friction)
        log.info("dist=%.3f +/- %.3f, reward=%.2f +/- %.2f",
                 cpg_metrics["avg_dist"], cpg_metrics["std_dist"], cpg_metrics["avg_reward"], cpg_metrics["std_reward"])
        log.info("=" * 60)
    elif args.policy == "rc" and rc_metrics is not None:
        log.info("\n" + "=" * 60)
        log.info("RC SUMMARY (friction=%.1f)", args.friction)
        log.info("dist=%.3f +/- %.3f, reward=%.2f +/- %.2f",
                 rc_metrics["avg_dist"], rc_metrics["std_dist"], rc_metrics["avg_reward"], rc_metrics["std_reward"])
        log.info("=" * 60)


if __name__ == "__main__":
    main()
