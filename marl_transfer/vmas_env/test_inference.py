"""
test_inference.py

Load a trained MPNN checkpoint and run deterministic inference on the
VMAS simple_spread environment.  Verifies the observation/action bridge.

Usage:
    cd marl_transfer
    python vmas_env/test_inference.py
"""
import os
import sys
import numpy as np
import torch

# Make marl_transfer/ importable (parent of vmas_env/)
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from mpnn import MPNN
from vmas_env.simple_spread import VMASSimpleSpread, MAX_STEPS


def load_checkpoint(ckpt_path, device="cpu"):
    ckpt = torch.load(ckpt_path, map_location=device)
    return ckpt["models"][0]


def build_mpnn(state_dict, num_agents, num_entities, input_size, device="cpu"):
    entity_mp = "entity_encoder.0.weight" in state_dict
    h_dim = state_dict["encoder.0.weight"].shape[0]
    num_actions = state_dict["dist.linear.bias"].shape[0]

    class DummyActionSpace:
        n = num_actions

    policy = MPNN(
        action_space=DummyActionSpace(),
        num_agents=num_agents,
        num_entities=num_entities,
        input_size=input_size,
        hidden_dim=h_dim,
        pos_index=2,
        entity_mp=entity_mp,
    ).to(device)

    policy.load_state_dict(state_dict, strict=True)
    policy.eval()
    return policy


def run_eval_episode(env, policy, device="cpu", render=False):
    obs_list = env.reset(seed=42)
    total_reward = 0.0

    for step in range(MAX_STEPS):
        obs_batch = torch.as_tensor(np.stack(obs_list), dtype=torch.float, device=device)

        with torch.no_grad():
            _, action, _, _ = policy.act(obs_batch, None, None, deterministic=True)
        actions = action.squeeze(-1).cpu().numpy()

        obs_list, reward_list, done_list, info = env.step(actions)
        total_reward += reward_list[0]

        if render:
            env.render()

        if all(done_list):
            break

    success = env.is_success
    steps = env.steps
    final_dist = env.min_dists.mean() if env.min_dists is not None else float("nan")
    return total_reward, success, steps, final_dist


def main():
    ckpt_dir = os.path.join(os.path.dirname(__file__), "..", "..", "marlsave", "save_new")

    candidates = [
        ("N=3  (3_0418)",  os.path.join(ckpt_dir, "3_0418", "ep1200.pt"),  3),
        ("N=5  (5_0418)",  os.path.join(ckpt_dir, "5_0418", "ep400.pt"),   5),
        ("N=10 (10_0419)", os.path.join(ckpt_dir, "10_0419", "ep1200.pt"), 10),
        ("N=16 (16_0419)", os.path.join(ckpt_dir, "16_0419", "ep1550.pt"), 16),
    ]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_eval_episodes = 10

    print("=" * 70)
    print(f"VMAS Inference Test — device: {device}")
    print("=" * 70)

    results = []

    for label, ckpt_path, n_agents in candidates:
        if not os.path.exists(ckpt_path):
            print(f"\n{label}: checkpoint not found — skip")
            continue

        print(f"\n{label}  <-  {os.path.relpath(ckpt_path)}")

        state_dict = load_checkpoint(ckpt_path, device)
        input_size = state_dict["encoder.0.weight"].shape[1]
        entity_mp = "entity_encoder.0.weight" in state_dict
        num_entities = n_agents

        print(f"  input_size={input_size}  entity_mp={entity_mp}  "
              f"num_entities={num_entities}  num_agents={n_agents}")

        policy = build_mpnn(state_dict, n_agents, num_entities, input_size, device)

        env = VMASSimpleSpread(
            num_agents=n_agents,
            arena_size=1.0,
            dist_threshold=0.1,
            identity_size=0,
            success_bonus=True,
        )
        env.seed(42)

        successes = 0
        total_r = 0.0
        final_dists = []

        for ep in range(num_eval_episodes):
            render = (ep == 0)
            r, succ, steps, fd = run_eval_episode(env, policy, device, render=render)
            successes += int(succ)
            total_r += r
            final_dists.append(fd)

        rate = 100.0 * successes / num_eval_episodes
        avg_dist = np.mean(final_dists)
        print(f"  Success: {successes}/{num_eval_episodes} ({rate:.0f}%)  "
              f"avg_final_dist={avg_dist:.3f}  avg_reward={total_r / num_eval_episodes:.2f}")

        results.append((label, rate, avg_dist))
        env.close()

    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("-" * 70)
    for label, rate, dist in results:
        print(f"  {label:20s}  success={rate:5.0f}%  avg_dist={dist:.4f}")
    print()

    if all(r[1] >= 80 for r in results):
        print("OK  All checkpoints transfer successfully to VMAS environment.")
    elif any(r[1] >= 50 for r in results):
        print("~  Some checkpoints transfer.  Observation bridge is compatible.")
    else:
        print("FAIL  None succeeded.  Check observation/action bridge.")
    print("\nDone.")


if __name__ == "__main__":
    main()
