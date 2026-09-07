"""
Bài 18: Tính và vẽ Moving Average của phần thưởng
Yêu cầu:
- Viết hàm moving_average(values, window_size) tính trung bình trượt.
- Không sử dụng pandas.
- Vẽ đồng thời reward ban đầu và đường moving average (window_size=10).
- Lưu biểu đồ vào figures/moving_average.png
"""

import os
import sys
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def moving_average(values, window_size=10):
    ret = np.cumsum(values, dtype=float)
    ret[window_size:] = ret[window_size:] - ret[:-window_size]
    ma = ret[window_size - 1:] / window_size
    pad = [np.nan] * (window_size - 1)
    return np.concatenate([pad, ma])

def run_episode(env):
    obs, info = env.reset()
    total_reward = 0.0
    while True:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break
    return total_reward

if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    
    num_episodes = 100
    episodes = np.arange(1, num_episodes + 1)
    rewards = np.array([run_episode(env) for _ in range(num_episodes)])
    env.close()
    
    ma_rewards = moving_average(rewards, window_size=10)
    
    os.makedirs("figures", exist_ok=True)
    
    plt.figure(figsize=(10, 5))
    plt.plot(episodes, rewards, color='lightblue', linestyle='-', label='Raw Reward', alpha=0.8)
    plt.plot(episodes, ma_rewards, color='red', linewidth=2, label='Moving Average (window=10)')
    
    plt.title("CartPole-v1 Rewards with Moving Average")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True)
    plt.legend()
    
    output_path = os.path.join("figures", "moving_average.png")
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    print(f"Da luu bieu do Moving Average vao: {output_path}")
