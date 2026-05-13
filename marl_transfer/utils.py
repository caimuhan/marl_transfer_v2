"""
utils.py - environment factory (MPE2 + VMAS) and tensor utilities
"""
import numpy as np
import torch
import gym_vecenv
from mpe2_adapter import create_mpe2_env


# ================================================================
#  Normalization
# ================================================================
def normalize_obs(obs, mean, std):
    """Normalize observations (handles both numpy and torch tensors)."""
    if mean is not None:
        if isinstance(obs, torch.Tensor):
            return (obs - mean) / std
        else:
            return np.divide((obs - mean), std)
    return obs


# ================================================================
#  MPE2 env factory  (legacy: simple_formation, simple_line)
# ================================================================
def make_multiagent_env(env_id, num_agents, dist_threshold, arena_size,
                        identity_size, success_bonus=False):
    return create_mpe2_env(
        env_id=env_id,
        num_agents=num_agents,
        dist_threshold=dist_threshold,
        arena_size=arena_size,
        identity_size=identity_size,
        success_bonus=success_bonus,
    )


def make_env(env_id, seed, rank, num_agents, dist_threshold, arena_size,
             identity_size, success_bonus=False):
    def _thunk():
        env = make_multiagent_env(env_id, num_agents, dist_threshold,
                                  arena_size, identity_size, success_bonus)
        return env
    return _thunk


def make_parallel_envs(args):
    """Create parallel MPE2 environments (legacy path)."""
    envs = [make_env(args.env_name, args.seed, i, args.num_agents,
                     args.dist_threshold, args.arena_size, args.identity_size,
                     args.success_bonus)
            for i in range(args.num_processes)]

    if args.num_processes > 1:
        envs = gym_vecenv.SubprocVecEnv(envs)
    else:
        envs = gym_vecenv.DummyVecEnv(envs)

    envs = gym_vecenv.MultiAgentVecNormalize(envs, ob=False, ret=True)
    return envs


# ================================================================
#  VMAS env factory  (GPU-native)
# ================================================================
def make_vmas_env(args):
    """
    Create a VMAS-powered multi-environment for simple_spread.

    Returns a VMASSimpleSpread instance that behaves like a gym VecEnv:
      - .reset(seed) → tensor [num_envs, num_agents, obs_dim]
      - .step(actions) → (obs, reward, done, info) all tensors
      - .world / .action_space / .observation_space for learner compat
      - .ob_rms = None (for eval compat)
    """
    from vmas_env.simple_spread import VMASSimpleSpread

    env = VMASSimpleSpread(
        num_agents=args.num_agents,
        num_envs=args.num_envs,
        arena_size=args.arena_size,
        dist_threshold=args.dist_threshold,
        identity_size=args.identity_size,
        success_bonus=args.success_bonus,
        device=args.device,
    )
    return env


# ================================================================
#  Weight init
# ================================================================
def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module