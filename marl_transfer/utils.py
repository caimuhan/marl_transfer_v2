"""
utils.py - 使用 mpe2 环境
"""
import numpy as np
import gym_vecenv
from mpe2_adapter import create_mpe2_env


def normalize_obs(obs, mean, std):
    """归一化观察"""
    if mean is not None:
        return np.divide((obs - mean), std)
    else:
        return obs


def make_multiagent_env(env_id, num_agents, dist_threshold, arena_size, identity_size, success_bonus=False):
    """
    创建多智能体环境 - 使用 mpe2 适配器

    Args:
        env_id: 环境名称 ('simple_spread', 'simple_formation', 'simple_line')
        num_agents: 智能体数量
        dist_threshold: 距离阈值
        arena_size: 场地大小
        identity_size: 身份向量大小
        success_bonus: 是否启用成功奖励（仅 simple_spread）

    Returns:
        env: 与 marl_transfer 兼容的环境
    """
    return create_mpe2_env(
        env_id=env_id,
        num_agents=num_agents,
        dist_threshold=dist_threshold,
        arena_size=arena_size,
        identity_size=identity_size,
        success_bonus=success_bonus
    )


def make_env(env_id, seed, rank, num_agents, dist_threshold, arena_size, identity_size, success_bonus=False):
    """创建单个环境的工厂函数（用于并行环境）"""
    def _thunk():
        env = make_multiagent_env(env_id, num_agents, dist_threshold, arena_size, identity_size, success_bonus)
        return env
    return _thunk


def make_parallel_envs(args):
    """创建并行环境"""
    envs = [make_env(args.env_name, args.seed, i, args.num_agents,
                     args.dist_threshold, args.arena_size, args.identity_size, args.success_bonus)
            for i in range(args.num_processes)]

    if args.num_processes > 1:
        envs = gym_vecenv.SubprocVecEnv(envs)
    else:
        envs = gym_vecenv.DummyVecEnv(envs)

    envs = gym_vecenv.MultiAgentVecNormalize(envs, ob=False, ret=True)
    return envs


def init(module, weight_init, bias_init, gain=1):
    """初始化模块权重"""
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module