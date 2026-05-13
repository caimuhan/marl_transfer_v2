"""
vmas_env/simple_spread.py

VMAS-powered simple_spread with the same observation/action interface as
marl_transfer's MPE2 environment.  Trained MPNN checkpoints can run
inference on this environment with zero weight modification.

Observation (entity_mp=True, identity_size=0):
  [vel(2), pos(2), landmark_rel(2*N)]

Action space: Discrete(5) → [noop, left, right, down, up]
              bridged to VMAS continuous forces via ACTION_MAP.
"""
import os
import sys

# Ensure VMAS simulator is importable (cloned at ../VectorizedMultiAgentSimulator)
_vmas_sim = os.path.join(os.path.dirname(__file__), "..", "VectorizedMultiAgentSimulator")
if _vmas_sim not in sys.path:
    sys.path.insert(0, _vmas_sim)

# Gym compatibility: VMAS uses gym, but older gym versions may not support
# array-like Box bounds.  Replace with gymnasium if available.
try:
    import gymnasium
    import gym.spaces
    gym.spaces.Box = gymnasium.spaces.Box
except ImportError:
    pass

import numpy as np
import torch
from torch import Tensor
from scipy.optimize import linear_sum_assignment

from vmas.simulator.core import Agent, Landmark, Sphere, World
from vmas.simulator.dynamics.holonomic import Holonomic
from vmas.simulator.environment.environment import Environment
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import Color, X, Y

# ============================================================
#  Constants matching PyMunkSimpleSpread
# ============================================================
AGENT_SIZE = 0.05
LANDMARK_SIZE = 0.05
MAX_SPEED = 2.0
ACTION_FORCE = 5.0
DT = 0.1
MAX_STEPS = 50

# Discrete action → (fx, fy)  (unit forces, scaled by u_multiplier)
ACTION_MAP = {
    0: (0.0, 0.0),     # noop
    1: (-1.0, 0.0),    # left
    2: (1.0, 0.0),     # right
    3: (0.0, -1.0),    # down
    4: (0.0, 1.0),     # up
}


# ============================================================
#  VMAS Scenario  (entity_mp aligned)
# ============================================================
class SimpleSpreadScenario(BaseScenario):
    """
    VMAS scenario matching MPE2 simple_spread with entity_mp=True.
    Observation order: [vel, pos, landmark_rel] (vel first, matching MPE2).
    Reward: Hungarian matching (same as original).
    """

    def make_world(self, batch_dim: int, device: torch.device, **kwargs) -> World:
        n_agents = kwargs.pop("n_agents", 3)
        self.arena_size = kwargs.pop("arena_size", 1.0)
        self.dist_threshold = kwargs.pop("dist_threshold", 0.1)
        self.success_bonus = kwargs.pop("success_bonus", True)
        self.identity_size = kwargs.pop("identity_size", 0)
        self.obs_agents = kwargs.pop("obs_agents", False)

        self.n_agents = n_agents
        self.num_landmarks = n_agents

        world = World(
            batch_dim=batch_dim,
            device=device,
            dt=DT,
            x_semidim=self.arena_size,
            y_semidim=self.arena_size,
        )

        # Agents
        for i in range(n_agents):
            agent = Agent(
                name=f"agent_{i}",
                collide=True,
                shape=Sphere(radius=AGENT_SIZE),
                mass=1.0,
                max_speed=MAX_SPEED,
                u_multiplier=ACTION_FORCE,
                u_range=1.0,
                color=Color.BLUE,
            )
            world.add_agent(agent)

        # Landmarks (static)
        for i in range(n_agents):
            landmark = Landmark(
                name=f"landmark_{i}",
                collide=False,
                shape=Sphere(radius=LANDMARK_SIZE),
                color=Color.GRAY,
            )
            world.add_landmark(landmark)

        return world

    def reset_world_at(self, env_index: int = None):
        rng = torch.zeros if env_index is not None else lambda shape, **kw: \
            torch.zeros(shape, **kw).uniform_(-self.arena_size, self.arena_size)

        for agent in self.world.agents:
            shape = (1, self.world.dim_p) if env_index is not None else \
                    (self.world.batch_dim, self.world.dim_p)
            pos = torch.zeros(shape, device=self.world.device,
                              dtype=torch.float32).uniform_(-self.arena_size,
                                                            self.arena_size)
            agent.set_pos(pos, batch_index=env_index)

        for landmark in self.world.landmarks:
            shape = (1, self.world.dim_p) if env_index is not None else \
                    (self.world.batch_dim, self.world.dim_p)
            pos = torch.zeros(shape, device=self.world.device,
                              dtype=torch.float32).uniform_(-self.arena_size,
                                                            self.arena_size)
            landmark.set_pos(pos, batch_index=env_index)

    def observation(self, agent: Agent) -> Tensor:
        """[vel(2), pos(2), landmark_rel(2*N)] — vel first, matching MPE2."""
        obs = [agent.state.vel, agent.state.pos]
        for landmark in self.world.landmarks:
            obs.append(landmark.state.pos - agent.state.pos)
        if self.obs_agents:
            for other in self.world.agents:
                if other != agent:
                    obs.append(other.state.pos - agent.state.pos)
        return torch.cat(obs, dim=-1)

    def reward(self, agent: Agent) -> Tensor:
        """Hungarian matching reward — computed once per step (first agent)."""
        if agent != self.world.agents[0]:
            return self._reward

        B = self.world.batch_dim
        agent_pos = torch.stack([a.state.pos for a in self.world.agents], dim=1)
        landmark_pos = torch.stack([l.state.pos for l in self.world.landmarks], dim=1)
        dists = torch.cdist(agent_pos, landmark_pos)  # [B, N, N]

        self._reward = torch.zeros(B, device=self.world.device)
        for b in range(B):
            d = dists[b].cpu().numpy()
            ri, ci = linear_sum_assignment(d)
            avg_d = d[ri, ci].mean()
            r = float(np.clip(-avg_d, -15, 15))
            if self.success_bonus and bool(np.all(d[ri, ci] < self.dist_threshold)):
                r += 5.0
            self._reward[b] = r
        return self._reward

    def info(self, agent: Agent):
        return {}


# ============================================================
#  Precomputed action map (GPU tensor)
# ============================================================
_ACTION_MAP_NP = np.array([
    [0.0, 0.0],     # 0: noop
    [-1.0, 0.0],    # 1: left
    [1.0, 0.0],     # 2: right
    [0.0, -1.0],    # 3: down
    [0.0, 1.0],     # 4: up
], dtype=np.float32)


def _build_action_map(device):
    return torch.as_tensor(_ACTION_MAP_NP, device=device)


# ============================================================
#  Wrapper — multi-env, GPU-native tensor interface
# ============================================================
class VMASSimpleSpread:
    """
    VMAS-powered simple_spread with GPU-native tensor interface.

    Compatible with:
      - learner.setup_master()   (world.policy_agents, action_space, obs_space)
      - test_inference.py        (reset/step/seed/close/render)
    """

    def __init__(self, num_agents=3, num_envs=32, arena_size=1.0,
                 dist_threshold=0.1, identity_size=0, success_bonus=True,
                 device="cpu"):
        self.num_agents = num_agents
        self.num_landmarks = num_agents
        self.num_envs = num_envs
        self.n = num_agents  # compat: gym_vecenv VecEnv expects .n
        self.arena_size = arena_size
        self.dist_threshold = dist_threshold
        self.identity_size = identity_size
        self.success_bonus = success_bonus
        self.device = device

        self.input_size = identity_size + 4
        self.obs_dim = self.input_size + 2 * self.num_landmarks

        # ---- VMAS environment (batched) -------------------------------
        self._env = Environment(
            scenario=SimpleSpreadScenario(),
            num_envs=num_envs,
            device=device,
            continuous_actions=True,
            max_steps=MAX_STEPS,
            n_agents=num_agents,
            arena_size=arena_size,
            dist_threshold=dist_threshold,
            success_bonus=success_bonus,
            identity_size=identity_size,
            obs_agents=False,
        )

        # ---- Action map (GPU tensor) ----------------------------------
        self._action_map = _build_action_map(device)

        # ---- Dummy world for learner compatibility --------------------
        self._dummy_agent_list = [_DummyAgent(i) for i in range(num_agents)]
        self.world = _DummyWorld(self._dummy_agent_list)

        # ---- Action / observation spaces ------------------------------
        try:
            import gymnasium as gym
            self.action_space = [gym.spaces.Discrete(5) for _ in range(num_agents)]
            self.observation_space = [gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
            ) for _ in range(num_agents)]
        except ImportError:
            self.action_space = None
            self.observation_space = None

        # ---- State ----------------------------------------------------
        self.ob_rms = None  # eval compat (may be set externally)
        self._seed_val = None
        self._is_success = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.min_dists = None

        self._reset_all()

    # ================================================================
    #  Reset
    # ================================================================
    def _reset_all(self):
        obs_list = self._env.reset()  # list of num_agents tensors [num_envs, obs_dim]
        self._is_success.zero_()
        self.min_dists = None
        return torch.stack(obs_list, dim=1)  # [num_envs, num_agents, obs_dim]

    def reset(self, seed=None):
        if seed is not None:
            self._seed_val = seed
        if self._seed_val is not None:
            self._env.seed(self._seed_val)
        return self._reset_all()

    # ================================================================
    #  Step  (tensor-in, tensor-out)
    # ================================================================
    def step(self, actions):
        """
        actions: list of num_agents tensors, each [num_envs, 1] (int 0-4).

        Returns:
            obs:    tensor [num_envs, num_agents, obs_dim]
            reward: tensor [num_envs, num_agents]
            done:   tensor [num_envs, num_agents] (bool)
            info:   dict  {'n': [per_agent_dict, ...]}
        """
        # Discrete(5) → continuous forces via lookup table
        forces = []
        for act in actions:
            if not isinstance(act, torch.Tensor):
                act = torch.as_tensor(act, dtype=torch.float32, device=self.device)
            elif act.device != self.device:
                act = act.to(self.device)
            idx = act.squeeze(-1).long()  # [num_envs]
            forces.append(self._action_map[idx])  # [num_envs, 2]

        obs_list, rews_list, vmas_dones, _infos = self._env.step(forces)
        # obs_list:  list of num_agents tensors [num_envs, obs_dim]
        # rews_list: list of num_agents tensors [num_envs]
        # vmas_dones: tensor [num_envs]  (True when max_steps reached)

        self._compute_success_vectorized()

        done_env = vmas_dones | self._is_success  # [num_envs] bool

        # --- Auto-reset environments that reached terminal state -----
        done_idx = torch.where(done_env)[0]
        if len(done_idx) > 0:
            for i in done_idx.tolist():
                self._env._reset_at(i, return_observations=False)
            # Re-read observations (mix of reset + continuing envs)
            obs_list = [
                self._env.scenario.observation(a).clone()
                for a in self._env.agents
            ]

        # --- Pack outputs ---------------------------------------------
        obs = torch.stack(obs_list, dim=1)            # [num_envs, num_agents, obs_dim]
        reward = torch.stack(rews_list, dim=1)        # [num_envs, num_agents]
        done = done_env.unsqueeze(1).expand(-1, self.num_agents)  # broadcast

        # Info dict for eval compatibility (uses env 0 stats)
        info = {'n': [{
            'is_success': self._is_success[0].item(),
            'world_steps': self._env.steps[0].item(),
            'reward': reward[0, 0].item(),
        } for _ in range(self.num_agents)]}

        return obs, reward, done, info

    # ================================================================
    #  Success detection (vectorised across envs)
    # ================================================================
    def _compute_success_vectorized(self):
        agent_pos = torch.stack(
            [a.state.pos for a in self._env.world.agents], dim=1
        )  # [num_envs, num_agents, 2]
        landmark_pos = torch.stack(
            [l.state.pos for l in self._env.world.landmarks], dim=1
        )  # [num_envs, num_landmarks, 2]
        dists = torch.cdist(agent_pos, landmark_pos)  # [num_envs, N, N]

        self.min_dists = torch.zeros(self.num_envs, self.num_agents,
                                      device=self.device)
        for b in range(self.num_envs):
            d = dists[b].cpu().numpy()
            ri, ci = linear_sum_assignment(d)
            self.min_dists[b] = torch.as_tensor(d[ri, ci], device=self.device)
            self._is_success[b] = bool(np.all(d[ri, ci] < self.dist_threshold))

        # Reset success flag for just-reset envs (steps == 0)
        just_reset = self._env.steps == 0
        self._is_success[just_reset] = False

    # ================================================================
    #  Seed
    # ================================================================
    def seed(self, seed=None):
        if seed is not None:
            self._seed_val = seed
            self._env.seed(seed)

    def close(self):
        pass

    # ================================================================
    #  Render  (delegated to VMAS)
    # ================================================================
    def render(self):
        try:
            self._env.render(env_index=0)
        except Exception:
            pass

    # ================================================================
    #  Info
    # ================================================================
    def get_env_info(self):
        return {
            "state_shape": self.obs_dim * self.num_agents,
            "obs_shape": self.obs_dim,
            "n_actions": 5,
            "n_agents": self.num_agents,
            "episode_limit": MAX_STEPS,
        }


# ============================================================
#  Dummy classes for learner.py compatibility
# ============================================================
class _DummyAgent:
    def __init__(self, idx):
        self.iden = idx
        self.adversary = False


class _DummyWorld:
    def __init__(self, agents):
        self.agents = agents
        self.policy_agents = agents


# ============================================================
#  Smoke-test
# ============================================================
if __name__ == "__main__":
    print("VMAS Simple Spread — random policy test")
    print("=" * 50)

    env = VMASSimpleSpread(num_agents=5, num_envs=1, arena_size=1.0, device="cpu")

    obs = env.reset()
    print(f"Observation shape: {obs.shape}")
    print(f"Expected: ({1}, {env.num_agents}, {env.obs_dim})")

    total_reward = 0
    for step in range(MAX_STEPS):
        acts = [torch.randint(0, 5, (1, 1)) for _ in range(env.num_agents)]
        obs, reward, done, info = env.step(acts)
        total_reward += reward[0, 0].item()

        if step % 10 == 0:
            min_d = env.min_dists[0].mean().item() if env.min_dists is not None else float('nan')
            print(f"  step {step:2d}  reward={reward[0,0].item():.3f}  "
                  f"min_dist={min_d:.3f}  success={env._is_success[0].item()}")

        if done[0, 0]:
            break

    print(f"\nEpisode done, total_reward={total_reward:.2f}")
    env.close()
    print("Done.")
