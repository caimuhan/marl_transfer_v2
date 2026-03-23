"""
eval.py - 评估脚本，适配 mpe2 环境
"""
import numpy as np
import torch
from arguments import get_args
from utils import normalize_obs
from learner import setup_master
import time


def evaluate(args, seed, policies_list, ob_rms=None, render=False, env=None, master=None, render_attn=True):
    """
    RL 评估函数

    Args:
        args: 命令行参数
        seed: 随机种子
        policies_list: 策略列表
        ob_rms: 观察归一化参数
        render: 是否渲染
        env: 环境实例
        master: Learner 实例
        render_attn: 是否渲染注意力

    Returns:
        all_episode_rewards: 所有回合奖励
        per_step_rewards: 每步奖励
        final_min_dists: 最终最小距离
        num_success: 成功次数
        episode_length: 平均回合长度
    """
    if env is None or master is None:
        master, env = setup_master(args, return_env=True)

    if seed is None:
        seed = np.random.randint(0, 100000)
        # seed = 42
    print("Evaluation Seed:", seed)
    env.seed(seed)

    if ob_rms is not None:
        obs_mean, obs_std = ob_rms
    else:
        obs_mean = None
        obs_std = None

    master.load_models(policies_list)
    master.set_eval_mode()

    num_eval_episodes = args.num_eval_episodes
    all_episode_rewards = np.full((num_eval_episodes, env.n), 0.0)
    per_step_rewards = np.full((num_eval_episodes, env.n), 0.0)

    recurrent_hidden_states = None
    mask = None

    final_min_dists = []
    num_success = 0
    episode_length = 0

    for t in range(num_eval_episodes):
        obs = env.reset()
        obs = normalize_obs(obs, obs_mean, obs_std)
        done = [False] * env.n
        episode_rewards = np.full(env.n, 0.0)
        episode_steps = 0

        if render:
            attn = None if not render_attn else master.team_attn
            if attn is not None and len(attn.shape) == 3:
                attn = attn.max(0)
            env.render(attn=attn)

        while not np.all(done):
            with torch.no_grad():
                actions = master.eval_act(obs, recurrent_hidden_states, mask)
            episode_steps += 1
            obs, reward, done, info = env.step(actions)
            obs = normalize_obs(obs, obs_mean, obs_std)
            episode_rewards += np.array(reward)

            if render:
                attn = None if not render_attn else master.team_attn
                if attn is not None and len(attn.shape) == 3:
                    attn = attn.max(0)
                env.render(attn=attn)
                if args.record_video:
                    time.sleep(0.08)

        per_step_rewards[t] = episode_rewards / episode_steps

        # 获取 info
        if info['n'] and len(info['n']) > 0:
            ep_info = info['n'][0]
            num_success += ep_info.get('is_success', 10)
            episode_length = (episode_length * t + ep_info.get('world_steps', episode_steps)) / (t + 1)

        # 获取最终距离
        if args.env_name == 'simple_spread':
            if hasattr(env.world, 'min_dists') and env.world.min_dists is not None:
                final_min_dists.append(env.world.min_dists)
        elif args.env_name in ['simple_formation', 'simple_line']:
            if hasattr(env.world, 'dists') and len(env.world.dists) > 0:
                final_min_dists.append(env.world.dists)

        if render:
            print("Ep {} | Success: {} \n Av per-step reward: {:.2f} | Ep Length {}".format(
                t,
                ep_info.get('is_success', False),
                per_step_rewards[t][0],
                ep_info.get('world_steps', episode_steps)
            ))

        all_episode_rewards[t, :] = episode_rewards

        if args.record_video:
            input('Press enter to continue: ')

    return all_episode_rewards, per_step_rewards, final_min_dists, num_success, episode_length


if __name__ == '__main__':
    args = get_args()
    checkpoint = torch.load(args.load_dir, map_location=lambda storage, loc: storage, weights_only=False)
    policies_list = checkpoint['models']
    ob_rms = checkpoint['ob_rms']

    all_episode_rewards, per_step_rewards, final_min_dists, num_success, episode_length = evaluate(
        args, args.seed, policies_list, ob_rms, args.render, render_attn=args.masking
    )

    print("Average Per Step Reward {}\nNum Success {}/{} | Av. Episode Length {:.2f})".format(
        per_step_rewards.mean(0), num_success, args.num_eval_episodes, episode_length
    ))

    if final_min_dists:
        print("Final Min Dists {}".format(np.stack(final_min_dists).mean(0)))