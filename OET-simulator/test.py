from mpe2 import my_custom_env_v1

# 创建环境
env = my_custom_env_v1.env(render_mode="human")  # 启用连续动作空间
env.reset(seed=42)

# 运行环境
for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()

    if termination or truncation:
        action = None

    else:
        # 随机动作
        action = env.action_space(agent).sample()
        print(f"智能体 {agent} 执行动作: {action}")

    env.step(action)

env.close()