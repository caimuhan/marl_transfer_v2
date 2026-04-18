"""
mpe2_adapter.py - MPE2 (PettingZoo) 到 marl_transfer 的适配层
将 PettingZoo 风格的环境包装成 marl_transfer 期望的 gym.Env 风格接口
"""

import numpy as np
from scipy.optimize import linear_sum_assignment

# 从 mpe2 导入基础类
from mpe2._mpe_utils.core import Agent as MPE2Agent, Landmark, World as MPE2World
from mpe2._mpe_utils.scenario import BaseScenario
from mpe2._mpe_utils.simple_env import SimpleEnv, make_env as mpe2_make_env
from gymnasium.utils import EzPickle
from gymnasium import spaces
from pettingzoo.utils.conversions import parallel_wrapper_fn


# ============================================================
# 扩展类 - 添加 marl_transfer 需要的属性
# ============================================================

class Agent(MPE2Agent):
    """扩展 Agent，添加 iden 和 adversary 属性"""

    def __init__(self, iden=None):
        super().__init__()
        self.iden = iden if iden is not None else 0
        self.adversary = False


class World(MPE2World):
    """扩展 World，添加 marl_transfer 需要的属性"""

    def __init__(self):
        super().__init__()
        self.steps = 0
        self.max_steps_episode = 50
        self.dists = []
        self.min_dists = None
        self.dist_thres = 0.1
        self.collaborative = False

    @property
    def policy_agents(self):
        """返回可控制的 agents（兼容旧代码）"""
        return [agent for agent in self.agents if agent.action_callback is None]


# ============================================================
# Gym 兼容包装器
# ============================================================

class MPE2GymWrapper:
    """
    将 PettingZoo 环境包装成 marl_transfer 期望的 gym.Env 风格接口

    主要功能：
    1. 将 PettingZoo 的 dict 接口转换为 list 接口
    2. 提供 world 属性访问
    3. 兼容 marl_transfer 的评估代码
    """

    def __init__(self, pz_env, scenario, world):
        """
        Args:
            pz_env: PettingZoo parallel_env
            scenario: Scenario 对象
            world: World 对象
        """
        self.env = pz_env
        self.scenario = scenario
        self.world = world

        # 初始化环境获取 agents 列表
        self.env.reset()
        self.agent_names = list(self.env.agents)
        self.n = len(self.agent_names)

        # 设置 action_space 和 observation_space（列表形式）
        self.action_space = [self.env.action_space(a) for a in self.agent_names]
        self.observation_space = [self.env.observation_space(a) for a in self.agent_names]

        # 兼容属性
        self.discrete_action_input = True
        self.shared_reward = getattr(world, 'collaborative', False)

    @property
    def policy_agents(self):
        """返回可控制的 agents（兼容旧代码）"""
        return self.world.policy_agents

    @property
    def episode_limit(self):
        """返回最大步数"""
        return getattr(self.world, 'max_steps_episode', 50)

    def seed(self, seed=None):
        """设置随机种子 - PettingZoo 在 reset 时处理"""
        self._seed = seed

    def reset(self):
        """重置环境，返回观察列表"""
        if hasattr(self, '_seed') and self._seed is not None:
            obs_dict, _ = self.env.reset(seed=self._seed)
            self._seed += 1
        else:
            obs_dict, _ = self.env.reset()

        # 重置 world 状态
        self.world.steps = 0
        # self.scenario.is_success = False
        self.scenario.min_dists = None

        return [obs_dict[a] for a in self.agent_names]

    def step(self, action_n):
        """
        执行动作

        Args:
            action_n: 动作列表 [action_agent0, action_agent1, ...]

        Returns:
            obs_n, reward_n, done_n, info_n (与旧接口兼容)
        """
        # 转换动作格式为字典
        action_dict = {}
        for i, agent_name in enumerate(self.agent_names):
            action = action_n[i]
            # 处理不同的动作格式
            if isinstance(action, np.ndarray):
                if action.size == 1:
                    action = int(action.item())
                else:
                    action = int(action[0])
            elif hasattr(action, '__len__') and len(action) == 1:
                action = int(action[0])
            action_dict[agent_name] = action

        # 执行步骤
        obs_dict, reward_dict, term_dict, trunc_dict, info_dict = self.env.step(action_dict)

        # 更新 world.steps
        self.world.steps += 1

        success_done = False
        if self.n > 0:
            # 使用第一个 agent 来触发检查
            agent_0 = self.world.agents[0]
            # 更新 is_success 状态，并在成功时提前结束
            success_done = self.scenario.done(agent_0, self.world)

        # 转换回列表形式
        obs_n = []
        reward_n = []
        done_n = []

        for i, agent_name in enumerate(self.agent_names):
            obs_n.append(obs_dict.get(agent_name, np.zeros(self.observation_space[i].shape)))
            reward_n.append(reward_dict.get(agent_name, 0.0))
            base_done = term_dict.get(agent_name, False) or trunc_dict.get(agent_name, False)
            done_n.append(base_done or success_done)

        # 处理共享奖励
        if self.shared_reward:
            total_reward = sum(reward_n)
            reward_n = [total_reward] * self.n

        # 组装 info（兼容 marl_transfer 格式）
        # info_n = {'n': [info_dict.get(agent_name, {}) for agent_name in self.agent_names]}
        infos_list = []
        for agent in self.world.agents:
            # 这里的 agent 是 mpe2 的 Agent 对象
            # 直接调用 scenario.info 获取最新的统计数据 (is_success, dists 等)
            info_data = self.scenario.info(agent, self.world)
            infos_list.append(info_data)

        info_n = {'n': infos_list}
        return obs_n, reward_n, done_n, info_n

    def render(self, mode='human', attn=None):
        """渲染环境"""
        return self.env.render()

    def close(self):
        """关闭环境"""
        self.env.close()

    def get_env_info(self):
        """获取环境信息"""
        obs_shape = self.observation_space[0].shape[0]
        n_actions = self.action_space[0].n if hasattr(self.action_space[0], 'n') else 5

        return {
            "state_shape": obs_shape * self.n,
            "obs_shape": obs_shape,
            "n_actions": n_actions,
            "n_agents": self.n,
            "episode_limit": self.episode_limit
        }

    def get_state(self):
        """获取全局状态"""
        if hasattr(self.env, 'state'):
            return self.env.state()
        return None

    def get_avail_actions(self):
        """获取可用动作"""
        n_actions = self.action_space[0].n if hasattr(self.action_space[0], 'n') else 5
        return np.ones((self.n, n_actions))


# ============================================================
# Simple Spread Scenario (对齐 marl_transfer)
# ============================================================
from mpe2.my_custom_env.my_custom_env import Scenario as SimpleSpreadScenario
# class SimpleSpreadScenario(BaseScenario):
#     """
#     Simple Spread 场景 - 完全对齐 marl_transfer 的逻辑
#
#     特点：
#     1. 支持 identity_size 参数
#     2. 奖励结构与 marl_transfer 一致
#     """
#
#     def __init__(self, num_agents=3, dist_threshold=0.1, arena_size=1, identity_size=0):
#         self.num_agents = num_agents
#         self.rewards = np.zeros(self.num_agents)
#         self.dist_threshold = dist_threshold
#         self.arena_size = arena_size
#         self.identity_size = identity_size
#         self.is_success = False
#         self.min_dists = None
#
#     def make_world(self):
#         world = World()
#         world.dim_c = 0  # 禁用通信
#         world.collaborative = False
#         world.dist_thres = self.dist_threshold
#         world.max_steps_episode = 50
#
#         # 创建带 iden 的 agents
#         world.agents = [Agent(iden=i) for i in range(self.num_agents)]
#         for i, agent in enumerate(world.agents):
#             agent.name = f'agent_{i}'
#             agent.collide = True
#             agent.silent = True
#             agent.size = 0.05
#             agent.adversary = False
#
#         # 创建 landmarks
#         world.landmarks = [Landmark() for _ in range(self.num_agents)]
#         for i, landmark in enumerate(world.landmarks):
#             landmark.name = f'landmark_{i}'
#             landmark.collide = False
#             landmark.movable = False
#
#         return world
#
#     def reset_world(self, world, np_random):
#         """重置世界状态"""
#         # 设置颜色
#         for agent in world.agents:
#             agent.color = np.array([0.35, 0.35, 0.85])
#             agent.state.p_pos = np_random.uniform(-self.arena_size, self.arena_size, world.dim_p)
#             agent.state.p_vel = np.zeros(world.dim_p)
#             agent.state.c = np.zeros(world.dim_c)
#
#         for landmark in world.landmarks:
#             landmark.color = np.array([0.25, 0.25, 0.25])
#             landmark.state.p_pos = np_random.uniform(-self.arena_size, self.arena_size, world.dim_p)
#             landmark.state.p_vel = np.zeros(world.dim_p)
#
#         # 重置状态
#         world.steps = 0
#         world.dists = []
#         self.is_success = False
#         self.min_dists = None
#
#     def reward(self, agent, world):
#         """计算奖励 - 只在第一个 agent 时计算一次"""
#         if agent.iden == 0:
#             # 计算距离矩阵
#             world.dists = np.array([
#                 [np.linalg.norm(a.state.p_pos - l.state.p_pos) for l in world.landmarks]
#                 for a in world.agents
#             ])
#             # 匈牙利算法求最优匹配
#             ri, ci = linear_sum_assignment(world.dists)
#             self.min_dists = world.dists[ri, ci]
#             world.min_dists = self.min_dists
#
#             # 计算联合奖励
#             joint_reward = np.clip(-np.mean(self.min_dists), -15, 15)
#             self.rewards = np.full(self.num_agents, joint_reward)
#
#         return self.rewards.mean()
#
#     def observation(self, agent, world):
#         """获取观察 - 与 marl_transfer 一致"""
#         # landmark 相对位置
#         entity_pos = [l.state.p_pos - agent.state.p_pos for l in world.landmarks]
#
#         # 基础观察：[vel, pos, landmark_rel_pos]
#         obs = np.concatenate([agent.state.p_vel, agent.state.p_pos] + entity_pos)
#
#         # 如果启用了 identity，添加 one-hot 向量
#         if self.identity_size > 0:
#             identity = np.eye(self.identity_size)[agent.iden]
#             obs = np.append(identity, obs)
#
#         return obs
#
#     def done(self, agent, world):
#         """判断是否结束"""
#         if self.min_dists is not None:
#             self.is_success = np.all(self.min_dists < world.dist_thres)
#         return world.steps >= world.max_steps_episode or self.is_success
#
#     def info(self, agent, world):
#         """返回额外信息"""
#         return {
#             'is_success': self.is_success,
#             'world_steps': world.steps,
#             'reward': float(self.rewards.mean()) if self.rewards is not None else 0,
#             'dists': float(self.min_dists.mean()) if self.min_dists is not None else 0
#         }


# ============================================================
# Simple Formation Scenario
# ============================================================

def get_thetas(poses):
    """计算角度"""
    thetas = []
    for pose in poses:
        angle = np.arctan2(pose[1], pose[0])
        if angle < 0:
            angle += 2 * np.pi
        thetas.append(angle)
    return thetas


class SimpleFormationScenario(BaseScenario):
    """Simple Formation 场景"""

    def __init__(self, num_agents=4, dist_threshold=0.1, arena_size=1, identity_size=0):
        self.num_agents = num_agents
        self.target_radius = 0.5
        self.ideal_theta_separation = (2 * np.pi) / self.num_agents
        self.arena_size = arena_size
        self.dist_thres = 0.05
        self.theta_thres = 0.1
        self.identity_size = identity_size
        self.rewards = np.zeros(self.num_agents)
        self.is_success = False
        self.min_dists = None

    def make_world(self):
        world = World()
        world.dim_c = 0
        world.collaborative = False
        world.max_steps_episode = 50

        world.agents = [Agent(iden=i) for i in range(self.num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = f'agent_{i}'
            agent.collide = True
            agent.silent = True
            agent.size = 0.05
            agent.adversary = False

        # 一个中心 landmark
        world.landmarks = [Landmark()]
        world.landmarks[0].name = 'landmark_0'
        world.landmarks[0].collide = False
        world.landmarks[0].movable = False
        world.landmarks[0].size = 0.03

        return world

    def reset_world(self, world, np_random):
        for agent in world.agents:
            agent.color = np.array([0.35, 0.35, 0.85])
            agent.state.p_pos = np_random.uniform(-self.arena_size, self.arena_size, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)

        for landmark in world.landmarks:
            landmark.color = np.array([0.25, 0.25, 0.25])
            landmark.state.p_pos = np_random.uniform(-0.5 * self.arena_size, 0.5 * self.arena_size, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)

        world.steps = 0
        world.dists = []
        self.is_success = False
        self.min_dists = None

    def reward(self, agent, world):
        if agent.iden == 0:
            landmark = world.landmarks[0]
            dists = [np.linalg.norm(a.state.p_pos - landmark.state.p_pos) for a in world.agents]
            rel_pos = [a.state.p_pos - landmark.state.p_pos for a in world.agents]
            thetas = get_thetas(rel_pos)

            dist_rew = -np.mean(np.abs(np.array(dists) - self.target_radius))

            sorted_thetas = np.sort(thetas)
            theta_diffs = np.diff(sorted_thetas)
            theta_diffs = np.append(theta_diffs, 2 * np.pi - sorted_thetas[-1] + sorted_thetas[0])
            theta_rew = -np.std(theta_diffs)

            joint_reward = np.clip(dist_rew + theta_rew, -15, 15)
            self.rewards = np.full(self.num_agents, joint_reward)
            world.dists = np.array(dists)

        return self.rewards.mean()

    def observation(self, agent, world):
        landmark_pos = [l.state.p_pos - agent.state.p_pos for l in world.landmarks]
        other_pos = [other.state.p_pos - agent.state.p_pos for other in world.agents if other is not agent]

        obs = np.concatenate([agent.state.p_vel, agent.state.p_pos] + landmark_pos + other_pos)

        if self.identity_size > 0:
            identity = np.eye(self.identity_size)[agent.iden]
            obs = np.append(identity, obs)
        return obs

    def done(self, agent, world):
        return world.steps >= world.max_steps_episode

    def info(self, agent, world):
        return {
            'is_success': self.is_success,
            'world_steps': world.steps,
            'reward': float(self.rewards.mean())
        }


# ============================================================
# Simple Line Scenario
# ============================================================

class SimpleLineScenario(BaseScenario):
    """Simple Line 场景"""

    def __init__(self, num_agents=4, dist_threshold=0.1, arena_size=1, identity_size=0):
        self.num_agents = num_agents
        self.arena_size = arena_size
        self.total_sep = 1.25 * arena_size
        self.ideal_sep = self.total_sep / (num_agents - 1) if num_agents > 1 else 0
        self.dist_thres = 0.05
        self.identity_size = identity_size
        self.rewards = np.zeros(self.num_agents)
        self.is_success = False
        self.min_dists = None

    def make_world(self):
        world = World()
        world.dim_c = 0
        world.collaborative = False
        world.max_steps_episode = 50

        world.agents = [Agent(iden=i) for i in range(self.num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = f'agent_{i}'
            agent.collide = True
            agent.silent = True
            agent.size = 0.03
            agent.adversary = False

        # 两个 landmarks 定义线的端点
        world.landmarks = [Landmark(), Landmark()]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = f'landmark_{i}'
            landmark.collide = False
            landmark.movable = False
            landmark.size = 0.02

        return world

    def reset_world(self, world, np_random):
        for agent in world.agents:
            agent.color = np.array([0.35, 0.35, 0.85])
            agent.state.p_pos = np_random.uniform(-self.arena_size, self.arena_size, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)

        for landmark in world.landmarks:
            landmark.color = np.array([0.25, 0.25, 0.25])

        # 第一个 landmark
        world.landmarks[0].state.p_pos = np_random.uniform(-0.25 * self.arena_size, 0.25 * self.arena_size, world.dim_p)
        world.landmarks[0].state.p_vel = np.zeros(world.dim_p)

        # 第二个 landmark - 与第一个保持固定距离
        theta = np_random.uniform(0, 2 * np.pi)
        loc = world.landmarks[0].state.p_pos + self.total_sep * np.array([np.cos(theta), np.sin(theta)])

        while not (abs(loc[0]) < self.arena_size and abs(loc[1]) < self.arena_size):
            theta += np.radians(5)
            loc = world.landmarks[0].state.p_pos + self.total_sep * np.array([np.cos(theta), np.sin(theta)])

        world.landmarks[1].state.p_pos = loc
        world.landmarks[1].state.p_vel = np.zeros(world.dim_p)

        world.steps = 0
        world.dists = []
        self.is_success = False
        self.min_dists = None

    def reward(self, agent, world):
        if agent.iden == 0:
            l0 = world.landmarks[0].state.p_pos
            l1 = world.landmarks[1].state.p_pos
            line_vec = l1 - l0

            # 计算理想位置
            ideal_positions = []
            for i in range(self.num_agents):
                t = i / (self.num_agents - 1) if self.num_agents > 1 else 0.5
                ideal_positions.append(l0 + t * line_vec)

            # 计算距离矩阵
            dists = np.array([
                [np.linalg.norm(a.state.p_pos - ideal_pos) for ideal_pos in ideal_positions]
                for a in world.agents
            ])

            ri, ci = linear_sum_assignment(dists)
            self.min_dists = dists[ri, ci]
            world.min_dists = self.min_dists
            world.dists = dists

            joint_reward = np.clip(-np.mean(self.min_dists), -15, 15)
            self.rewards = np.full(self.num_agents, joint_reward)

        return self.rewards.mean()

    def observation(self, agent, world):
        landmark_pos = [l.state.p_pos - agent.state.p_pos for l in world.landmarks]
        other_pos = [other.state.p_pos - agent.state.p_pos for other in world.agents if other is not agent]

        obs = np.concatenate([agent.state.p_vel, agent.state.p_pos] + landmark_pos + other_pos)

        if self.identity_size > 0:
            identity = np.eye(self.identity_size)[agent.iden]
            obs = np.append(identity, obs)
        return obs

    def done(self, agent, world):
        return world.steps >= world.max_steps_episode

    def info(self, agent, world):
        return {
            'is_success': self.is_success,
            'world_steps': world.steps,
            'reward': float(self.rewards.mean())
        }


# ============================================================
# 环境创建函数
# ============================================================

def create_mpe2_env(env_id, num_agents=3, dist_threshold=0.1, arena_size=1, identity_size=0, success_bonus=False):
    """
    创建 mpe2 环境并包装成 marl_transfer 兼容接口

    Args:
        env_id: 环境名称 ('simple_spread', 'simple_formation', 'simple_line')
        num_agents: 智能体数量
        dist_threshold: 距离阈值
        arena_size: 场地大小
        identity_size: 身份向量大小（0 表示不使用）
        success_bonus: 是否启用成功奖励（仅 simple_spread）

    Returns:
        MPE2GymWrapper: 包装后的环境，与 marl_transfer 兼容
    """
    # 选择场景
    if env_id == 'simple_spread':
        scenario = SimpleSpreadScenario(num_agents, dist_threshold, arena_size, identity_size, success_bonus)
    elif env_id == 'simple_formation':
        scenario = SimpleFormationScenario(num_agents, dist_threshold, arena_size, identity_size)
    elif env_id == 'simple_line':
        scenario = SimpleLineScenario(num_agents, dist_threshold, arena_size, identity_size)
    else:
        # 默认使用 simple_spread
        scenario = SimpleSpreadScenario(num_agents, dist_threshold, arena_size, identity_size, success_bonus)

    world = scenario.make_world()

    # 创建自定义环境类
    class CustomMPE2Env(SimpleEnv, EzPickle):
        def __init__(self):
            EzPickle.__init__(self)
            SimpleEnv.__init__(
                self,
                scenario=scenario,
                world=world,
                render_mode=None,
                # render_mode="human",
                max_cycles=50,
                continuous_actions=False,
                local_ratio=None,
                dynamic_rescaling=False,
            )
            self.metadata["name"] = f"{env_id}_v1"

    # 创建环境实例
    raw = CustomMPE2Env()
    wrapped = mpe2_make_env(lambda **kw: raw)
    pz_env = parallel_wrapper_fn(wrapped)()

    # 包装成兼容接口
    return MPE2GymWrapper(pz_env, scenario, world)
