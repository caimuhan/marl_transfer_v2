

"""
My Custom Environment - 完全对齐 marl_transfer 的 simple_spread
"""

import numpy as np
from gymnasium.utils import EzPickle
from pettingzoo.utils.conversions import parallel_wrapper_fn
from scipy.optimize import linear_sum_assignment

from mpe2._mpe_utils.core import Agent, Landmark, World
from mpe2._mpe_utils.scenario import BaseScenario
from mpe2._mpe_utils.simple_env import SimpleEnv, make_env

from mpe2 import word_generator
# ============================================================
# 扩展 Agent 类以支持 iden 属性
# ============================================================
class IdentifiedAgent(Agent):
    """带有 iden 属性的 Agent，与 M 完全一致"""

    def __init__(self, iden=0):
        super().__init__()
        self.iden = iden


# ============================================================
# 扩展 World 类以支持 steps 和 max_steps_episode
# ============================================================
class ExtendedWorld(World):
    """扩展的 World，与 M 完全一致"""

    def __init__(self):
        super().__init__()
        self.steps = 0
        self.max_steps_episode = 50
        self.dists = []
        self.min_dists = None
        self.dist_thres = 0.1


# ============================================================
# 环境类
# ============================================================
class raw_env(SimpleEnv, EzPickle):
    def __init__(
            self,
            N=4,
            local_ratio=None,  # 不使用 local_ratio，与 M 一致
            max_cycles=50,  # 与 M 的 max_steps_episode 一致
            continuous_actions=False,  # 离散动作
            # render_mode=None,
            render_mode="human",
            dynamic_rescaling=False,
            # 新增参数，与 M 对齐
            dist_threshold=0.1,
            arena_size=1,
            identity_size=0,
    ):
        EzPickle.__init__(
            self,
            N=N,
            max_cycles=max_cycles,
            continuous_actions=continuous_actions,
            render_mode=render_mode,
            dist_threshold=dist_threshold,
            arena_size=arena_size,
            identity_size=identity_size,
        )
        scenario = Scenario(
            num_agents=N,
            dist_threshold=dist_threshold,
            arena_size=arena_size,
            identity_size=identity_size,
        )
        world = scenario.make_world()
        SimpleEnv.__init__(
            self,
            scenario=scenario,
            world=world,
            render_mode=render_mode,
            max_cycles=max_cycles,
            continuous_actions=continuous_actions,
            local_ratio=local_ratio,
            dynamic_rescaling=dynamic_rescaling,
        )
        self.metadata["name"] = "my_custom_env_v1"


env = make_env(raw_env)
parallel_env = parallel_wrapper_fn(env)


# ============================================================
# Scenario 类 - 完全对齐 M 的 simple_spread
# ============================================================
class Scenario(BaseScenario):
    """
    完全对齐 marl_transfer/mape/multiagent/scenarios/simple_spread.py
    """

    def __init__(self, num_agents=4, dist_threshold=0.1, arena_size=1, identity_size=0):
        self.num_agents = num_agents
        self.rewards = np.zeros(self.num_agents)
        self.temp_done = False#
        self.dist_threshold = dist_threshold
        self.arena_size = arena_size
        self.identity_size = identity_size
        self.is_success = False
        self.min_dists = None

    def make_world(self):
        world = ExtendedWorld()
        # set any world properties first
        world.dim_c = 0
        num_agents = self.num_agents
        num_landmarks = num_agents
        world.collaborative = False
        world.dist_thres = self.dist_threshold

        # add agents - 使用 IdentifiedAgent 以支持 iden
        world.agents = [IdentifiedAgent(iden=i) for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.size = 0.02
            agent.adversary = False

        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
            landmark.size=0.02
        # make initial conditions
        self.reset_world(world, np.random)
        world.dists = []
        return world

    def reset_world(self, world, np_random):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.35, 0.85])

        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([1, 1, 1])

        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np_random.uniform(-self.arena_size, self.arena_size, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)

        '''
        radius = self.arena_size * 0.8
        angles = np.linspace(0, 2 * np.pi, len(world.landmarks), endpoint=False)

        for i, landmark in enumerate(world.landmarks):
            x = radius * np.cos(angles[i])
            y = radius * np.sin(angles[i])
            landmark.state.p_pos = np.array([x, y])
            landmark.state.p_vel = np.zeros(world.dim_p)
        '''
        segments = self._get_letter_segments("S")

        points = self._sample_points_on_segments(
            segments,
            len(world.landmarks)
        )

        # 缩放
        points = points * self.arena_size * 0.8

        # for i, landmark in enumerate(world.landmarks):
        #     landmark.state.p_pos = points[i]
        #     landmark.state.p_vel = np.zeros(world.dim_p)
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np_random.uniform(-self.arena_size, self.arena_size, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)

        world.steps = 0
        world.dists = []
        self.is_success = False
        self.min_dists = None

    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    def reward(self, agent, world):
        # 只在第一个 agent 时计算一次（与 M 完全一致）
        if agent.iden == 0:
            # each column represents distance of all agents from the respective landmark
            world.dists = np.array([
                [np.linalg.norm(a.state.p_pos - l.state.p_pos) for l in world.landmarks]
                for a in world.agents
            ])
            # optimal 1:1 agent-landmark pairing (bipartite matching algorithm)
            self.min_dists = self._bipartite_min_dists(world.dists)
            # the reward is normalized by the number of agents
            joint_reward = np.clip(-np.mean(self.min_dists), -15, 15)
            self.rewards = np.full(self.num_agents, joint_reward)
            world.min_dists = self.min_dists
        return self.rewards.mean()

    def _bipartite_min_dists(self, dists):
        ri, ci = linear_sum_assignment(dists)
        min_dists = dists[ri, ci]
        return min_dists

    def observation(self, agent, world):
        """
        观察空间 - 与 M 完全一致

        结构:  [identity (可选), vel(2), pos(2), landmark_rel_pos(2*N)]

        - identity_size > 0 时:  前面添加 one-hot 身份向量
        - vel(2) + pos(2) = 4 维
        - landmark_rel_pos = 2 * num_landmarks 维
        """
        # positions of all entities in this agent's reference frame
        entity_pos = [entity.state.p_pos - agent.state.p_pos for entity in world.landmarks]
        default_obs = np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos)

        if self.identity_size != 0:
            identified_obs = np.append(np.eye(self.identity_size)[agent.iden], default_obs)
            return identified_obs
        return default_obs

    def done(self, agent, world):
        # condition1 = world.steps >= world.max_steps_episode
        # self.is_success = np.all(self.min_dists < world.dist_thres)
        # return condition1 or self.is_success
        # try:
        #     if self.min_dists is not None:
        #
        #         success_now = np.all(self.min_dists < world.dist_thres)
        #         # 只有当它是 True 时才更新，或者每一帧都更新，取出决于你是否希望"曾经成功过就算成功"
        #         # 这里我们用：只要当前帧满足条件，就是成功
        #         self.is_success = success_now
        #
        #     else:
        #         self.is_success =True
        # except Exception:
        #     self.is_success = False

        # # 2. 这里的修改很关键：仅仅当步数耗尽时才返回 done=True
        # # 不要因为 success 就结束环境，否则 eval.py 可能来不及统计到
        # return world.steps >= world.max_steps_episode

        if self.min_dists is not None:
            self.is_success = np.all(self.min_dists < world.dist_thres)
        else:
            self.is_success = False

        return world.steps >= world.max_steps_episode or self.is_success
    def info(self, agent, world):
        return {
            'is_success': self.is_success,
            'world_steps': world.steps,
            'reward': self.rewards.mean() if self.rewards is not None else 0,
            'dists': self.min_dists.mean() if self.min_dists is not None else 0
        }

    import numpy as np
    def _sample_points_on_segments(self, segments, total_points):
        """
        segments: list of [(x1,y1),(x2,y2)]
        total_points: landmark数量
        """
        # 计算每条线段长度
        lengths = []
        for (p1, p2) in segments:
            p1 = np.array(p1)
            p2 = np.array(p2)
            lengths.append(np.linalg.norm(p2 - p1))

        lengths = np.array(lengths)
        total_length = lengths.sum()

        # 按比例分配点数
        points_per_segment = np.maximum(
            1,
            np.round(total_points * lengths / total_length).astype(int)
        )

        # 修正数量误差
        diff = total_points - points_per_segment.sum()
        points_per_segment[0] += diff

        # 生成点
        points = []
        for (p1, p2), n in zip(segments, points_per_segment):
            p1 = np.array(p1)
            p2 = np.array(p2)
            for t in np.linspace(0, 1, n, endpoint=False):
                points.append(p1 + t * (p2 - p1))

        return np.array(points[:total_points])

    def _get_letter_segments(self, letter):
        """
        所有字母都归一化在 [-1,1] 空间
        """

        if letter == "S":
            return [
                ((-0.8, 0.8), (0.8, 0.8)),
                ((-0.8, 0.8), (-0.8, 0.0)),
                ((-0.8, 0.0), (0.8, 0.0)),
                ((0.8, 0.0), (0.8, -0.8)),
                ((-0.8, -0.8), (0.8, -0.8)),
            ]

        elif letter == "J":
            return [
                ((-0.8, 0.8), (0.8, 0.8)),
                ((0.0, 0.6), (0.0, -0.6)),
                ((0.0, -0.7), (-0.5, -0.8)),
            ]

        elif letter == "T":
            return [
                ((-0.8, 0.8), (0.8, 0.8)),
                ((0.0, 0.8), (0.0, -0.8)),
            ]

        elif letter == "U":
            return [
                ((-0.8, 0.8), (-0.8, -0.8)),
                ((0.8, 0.8), (0.8, -0.8)),
                ((-0.8, -0.8), (0.8, -0.8)),
            ]

        else:
            raise ValueError("Unknown letter")

