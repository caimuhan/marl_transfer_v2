import numpy as np
from mpe2 import my_custom_env_v1  # OET 的 PettingZoo 环境

class OETMarlTransferEnv:
    """包装 OET 的 parallel_env，使其类似 marl_transfer 的 MultiAgentEnv（离散）。"""
    metadata = {'render.modes': ['human']}

    def __init__(self, seed=0, max_cycles=25, render_mode=None):
        self.raw_env = my_custom_env_v1.parallel_env(
            continuous_actions=False,  # 离散
            max_cycles=max_cycles,
            render_mode=render_mode,
        )
        self.raw_env.reset(seed=seed)
        self.agents = list(self.raw_env.agents)
        self.n = len(self.agents)  # 20 个智能体
        self.action_space = [self.raw_env.action_space(a) for a in self.agents]
        self.observation_space = [self.raw_env.observation_space(a) for a in self.agents]

    @property
    def episode_limit(self):
        return self.raw_env.max_cycles

    def reset(self, seed=None):
        obs_dict = self.raw_env.reset(seed=seed)
        return [obs_dict[a] for a in self.agents]

    def step(self, action_n):
        # action_n: 长度为 n 的 int 列表，每个在 {0..4}
        actions = {agent: action_n[i] for i, agent in enumerate(self.agents)}
        obs, rew, term, trunc, info = self.raw_env.step(actions)
        obs_n = [obs[a] for a in self.agents]
        rew_n = [rew[a] for a in self.agents]
        done_n = [term[a] or trunc[a] for a in self.agents]
        info_n = {'n': [info.get(a, {}) for a in self.agents]}
        return obs_n, rew_n, done_n, info_n

    # 如果 marl_transfer 需要，可选的辅助函数：
    def get_obs(self):
        cur = self.raw_env.observe()
        return [cur[a] for a in self.agents]

    def get_state(self):
        return np.concatenate(self.get_obs())

    def get_avail_actions(self):
        n_actions = self.action_space[0].n
        return np.ones((self.n, n_actions))
