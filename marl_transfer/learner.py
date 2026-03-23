"""
learner.py - 适配 mpe2 环境
"""
import numpy as np
import torch
from rlcore.algo import JointPPO
from rlagent import Neo
from mpnn import MPNN
from utils import make_multiagent_env


def setup_master(args, env=None, return_env=False):
    """
    设置训练主控

    Args:
        args: 命令行参数
        env: 环境实例（可选）
        return_env: 是否返回环境

    Returns:
        master: Learner 实例
        env: 环境实例（如果 return_env=True）
    """
    if env is None:
        env = make_multiagent_env(
            args.env_name,
            num_agents=args.num_agents,
            dist_threshold=args.dist_threshold,
            arena_size=args.arena_size,
            identity_size=args.identity_size
        )

    policy1 = None
    policy2 = None
    team1 = []
    team2 = []

    # 统计对手和友方数量
    num_adversary = 0
    num_friendly = 0

    # 获取 policy_agents（兼容 mpe2）
    policy_agents = env.world.policy_agents

    for i, agent in enumerate(policy_agents):
        if hasattr(agent, 'adversary') and agent.adversary:
            num_adversary += 1
        else:
            num_friendly += 1

    # 获取动作空间
    action_space = env.action_space[0]  # 假设所有智能体动作空间相同

    # 确定实体数量
    entity_mp = args.entity_mp
    if args.env_name == 'simple_spread':
        num_entities = args.num_agents
    elif args.env_name == 'simple_formation':
        num_entities = 1
    elif args.env_name == 'simple_line':
        num_entities = 2
    else:
        num_entities = args.num_agents  # 默认

    # 计算策略观察维度
    obs_dim = env.observation_space[0].shape[0]
    if entity_mp:
        pol_obs_dim = obs_dim - 2 * num_entities
    else:
        pol_obs_dim = obs_dim

    # 位置索引
    pos_index = args.identity_size + 2

    # 为每个智能体创建策略
    for i, agent in enumerate(policy_agents):
        agent_obs_dim = env.observation_space[i].shape[0]

        if hasattr(agent, 'adversary') and agent.adversary:
            # 对手智能体
            if policy1 is None:
                policy1 = MPNN(
                    input_size=pol_obs_dim,
                    num_agents=num_adversary,
                    num_entities=num_entities,
                    action_space=action_space,
                    pos_index=pos_index,
                    mask_dist=args.mask_dist,
                    entity_mp=entity_mp
                ).to(args.device)
            team1.append(Neo(args, policy1, (agent_obs_dim,), action_space))
        else:
            # 友方智能体
            if policy2 is None:
                policy2 = MPNN(
                    input_size=pol_obs_dim,
                    num_agents=num_friendly,
                    num_entities=num_entities,
                    action_space=action_space,
                    pos_index=pos_index,
                    mask_dist=args.mask_dist,
                    entity_mp=entity_mp
                ).to(args.device)
            team2.append(Neo(args, policy2, (agent_obs_dim,), action_space))

    master = Learner(args, [team1, team2], [policy1, policy2], env)

    if args.continue_training:
        print("Loading pretrained model")
        master.load_models(torch.load(args.load_dir, weights_only=False)['models'])

    if return_env:
        return master, env
    return master


class Learner(object):
    """
    Learner 类 - 支持多智能体集中式训练
    """

    def __init__(self, args, teams_list, policies_list, env):
        """
        初始化 Learner

        Args:
            args: 命令行参数
            teams_list: 团队列表 [[team1_agents], [team2_agents]]
            policies_list: 策略列表 [policy1, policy2]
            env: 环境实例
        """
        self.teams_list = [x for x in teams_list if len(x) != 0]
        self.all_agents = [agent for team in teams_list for agent in team]
        self.policies_list = [x for x in policies_list if x is not None]
        self.trainers_list = [
            JointPPO(
                policy,
                args.clip_param,
                args.ppo_epoch,
                args.num_mini_batch,
                args.value_loss_coef,
                args.entropy_coef,
                lr=args.lr,
                max_grad_norm=args.max_grad_norm,
                use_clipped_value_loss=args.clipped_value_loss
            )
            for policy in self.policies_list
        ]
        self.device = args.device
        self.env = env

    @property
    def all_policies(self):
        """返回所有智能体的策略状态字典"""
        return [agent.actor_critic.state_dict() for agent in self.all_agents]

    @property
    def team_attn(self):
        """返回注意力矩阵"""
        if self.policies_list and hasattr(self.policies_list[0], 'attn_mat'):
            return self.policies_list[0].attn_mat
        return None

    def initialize_obs(self, obs):
        """
        初始化观察

        Args:
            obs: 观察 (num_processes x num_agents x obs_dim)
        """
        for i, agent in enumerate(self.all_agents):
            agent.initialize_obs(torch.from_numpy(obs[:, i, :]).float().to(self.device))
            agent.rollouts.to(self.device)

    def act(self, step):
        """
        获取动作

        Args:
            step: 当前步数

        Returns:
            actions_list: 动作列表
        """
        actions_list = []
        for team, policy in zip(self.teams_list, self.policies_list):
            # 拼接所有输入
            all_obs = torch.cat([agent.rollouts.obs[step] for agent in team])
            all_hidden = torch.cat([agent.rollouts.recurrent_hidden_states[step] for agent in team])
            all_masks = torch.cat([agent.rollouts.masks[step] for agent in team])

            # 单次前向传播
            props = policy.act(all_obs, all_hidden, all_masks, deterministic=False)

            # 分割输出
            n = len(team)
            all_value, all_action, all_action_log_prob, all_states = [torch.chunk(x, n) for x in props]
            for i in range(n):
                team[i].value = all_value[i]
                team[i].action = all_action[i]
                team[i].action_log_prob = all_action_log_prob[i]
                team[i].states = all_states[i]
                actions_list.append(all_action[i].cpu().numpy())

        return actions_list

    def update(self):
        """更新策略"""
        return_vals = []
        for i, trainer in enumerate(self.trainers_list):
            rollouts_list = [agent.rollouts for agent in self.teams_list[i]]
            vals = trainer.update(rollouts_list)
            return_vals.append([np.array(vals)] * len(rollouts_list))

        return np.stack([x for v in return_vals for x in v]).reshape(-1, 3)

    def wrap_horizon(self):
        """处理 horizon 结束"""
        for team, policy in zip(self.teams_list, self.policies_list):
            last_obs = torch.cat([agent.rollouts.obs[-1] for agent in team])
            last_hidden = torch.cat([agent.rollouts.recurrent_hidden_states[-1] for agent in team])
            last_masks = torch.cat([agent.rollouts.masks[-1] for agent in team])

            with torch.no_grad():
                next_value = policy.get_value(last_obs, last_hidden, last_masks)

            all_value = torch.chunk(next_value, len(team))
            for i in range(len(team)):
                team[i].wrap_horizon(all_value[i])

    def after_update(self):
        """更新后处理"""
        for agent in self.all_agents:
            agent.after_update()

    def update_rollout(self, obs, reward, masks):
        """更新 rollout"""
        obs_t = torch.from_numpy(obs).float().to(self.device)
        for i, agent in enumerate(self.all_agents):
            agent_obs = obs_t[:, i, :]
            agent.update_rollout(agent_obs, reward[:, i].unsqueeze(1), masks[:, i].unsqueeze(1))

    def load_models(self, policies_list):
        """加载模型"""
        for agent, policy in zip(self.all_agents, policies_list):
            agent.load_model(policy)

    def eval_act(self, obs, recurrent_hidden_states, mask):
        """
        评估模式下的动作选择

        Args:
            obs: 观察列表
            recurrent_hidden_states: 循环隐藏状态
            mask: 掩码

        Returns:
            actions: 动作数组
        """
        obs1 = []
        obs2 = []
        all_obs = []

        # 获取 policy_agents
        policy_agents = self.env.world.policy_agents

        for i in range(len(obs)):
            agent = policy_agents[i]
            obs_tensor = torch.as_tensor(obs[i], dtype=torch.float, device=self.device).view(1, -1)

            if hasattr(agent, 'adversary') and agent.adversary:
                obs1.append(obs_tensor)
            else:
                obs2.append(obs_tensor)

        if len(obs1) != 0:
            all_obs.append(obs1)
        if len(obs2) != 0:
            all_obs.append(obs2)

        actions = []
        for team, policy, obs_list in zip(self.teams_list, self.policies_list, all_obs):
            if len(obs_list) != 0:
                _, action, _, _ = policy.act(
                    torch.cat(obs_list).to(self.device),
                    None,
                    None,
                    deterministic=True
                )
                actions.append(action.squeeze(1).cpu().numpy())

        return np.hstack(actions)

    def set_eval_mode(self):
        """设置评估模式"""
        for agent in self.all_agents:
            agent.actor_critic.eval()

    def set_train_mode(self):
        """设置训练模式"""
        for agent in self.all_agents:
            agent.actor_critic.train()