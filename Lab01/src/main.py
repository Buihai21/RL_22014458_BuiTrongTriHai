"""
Chương trình tổng hợp Mini-project Agent-Environment cho CartPole-v1
File: Lab01/src/main.py
"""

import os
import sys
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

# Đảm bảo in UTF-8 không bị lỗi mã hóa trên Windows console
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def create_environment(env_name="CartPole-v1", render_mode=None):
    return gym.make(env_name, render_mode=render_mode)

def policy(observation, env):
    pole_angle = observation[2]
    pole_angular_velocity = observation[3]
    signal = pole_angle + 0.15 * pole_angular_velocity
    return 1 if signal > 0 else 0

def run_episode(env, policy_func, seed=None, max_steps=1000):
    obs, info = env.reset(seed=seed)
    total_reward = 0.0
    length = 0
    terminated_flag = False
    truncated_flag = False
    
    for _ in range(max_steps):
        action = policy_func(obs, env)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        length += 1
        
        if terminated or truncated:
            terminated_flag = terminated
            truncated_flag = truncated
            break
            
    return {
        "reward": total_reward,
        "length": length,
        "terminated": terminated_flag,
        "truncated": truncated_flag
    }

def evaluate_policy(env, policy_func, n_episodes=500, seed=42):
    rewards = []
    lengths = []
    
    for ep in range(n_episodes):
        ep_seed = seed + ep if seed is not None else None
        res = run_episode(env, policy_func, seed=ep_seed)
        rewards.append(res["reward"])
        lengths.append(res["length"])
        
    rewards_arr = np.array(rewards)
    lengths_arr = np.array(lengths)
    
    best_idx = int(np.argmax(rewards_arr))
    worst_idx = int(np.argmin(rewards_arr))
    
    return {
        "rewards": rewards_arr,
        "lengths": lengths_arr,
        "mean_reward": float(np.mean(rewards_arr)),
        "std_reward": float(np.std(rewards_arr)),
        "best_episode": (best_idx + 1, float(rewards_arr[best_idx])),
        "worst_episode": (worst_idx + 1, float(rewards_arr[worst_idx]))
    }

def plot_results(rewards, window_size=10, save_dir="figures"):
    os.makedirs(save_dir, exist_ok=True)
    episodes = np.arange(1, len(rewards) + 1)
    
    ret = np.cumsum(rewards, dtype=float)
    ret[window_size:] = ret[window_size:] - ret[:-window_size]
    ma = ret[window_size - 1:] / window_size
    ma_padded = np.concatenate([[np.nan] * (window_size - 1), ma])
    
    plt.figure(figsize=(12, 5))
    plt.plot(episodes, rewards, color='skyblue', alpha=0.6, label='Raw Episode Reward')
    plt.plot(episodes, ma_padded, color='crimson', linewidth=2, label=f'Moving Average (window={window_size})')
    plt.title("Mini-project CartPole-v1 Policy Evaluation (500 Episodes)")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.grid(True)
    plt.legend()
    
    save_path = os.path.join(save_dir, "mini_project_results.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Da luu bieu do ket qua tai: {save_path}")

def main():
    print("=== CHUONG TRINH TONG HOP MAIN.PY (CARTPOLE-V1) ===")
    env = create_environment("CartPole-v1")
    
    n_episodes = 500
    seed_val = 42
    stats = evaluate_policy(env, policy, n_episodes=n_episodes, seed=seed_val)
    env.close()
    
    print("\n--- THONG KE KET QUA ---")
    print(f"Mean Reward     : {stats['mean_reward']:.2f}")
    print(f"Std Reward      : {stats['std_reward']:.2f}")
    print(f"Episode tot nhat: Episode #{stats['best_episode'][0]} (Reward: {stats['best_episode'][1]})")
    print(f"Episode te nhat : Episode #{stats['worst_episode'][0]} (Reward: {stats['worst_episode'][1]})")
    
    plot_results(stats["rewards"], window_size=10)

if __name__ == "__main__":
    main()
