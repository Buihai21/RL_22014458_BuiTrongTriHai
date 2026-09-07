"""
Bài 17: Vẽ biểu đồ reward theo episode
Yêu cầu:
- Sử dụng Matplotlib vẽ biểu đồ: trục X là episode, trục Y là total reward.
- Bắt buộc phải có title, xlabel, ylabel, grid.
- Lưu biểu đồ vào file figures/reward_cartpole.png
"""

import os
import sys
import gymnasium as gym
import matplotlib.pyplot as plt

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

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
    episodes = list(range(1, num_episodes + 1))
    rewards = [run_episode(env) for _ in range(num_episodes)]
    env.close()
    
    os.makedirs("figures", exist_ok=True)
    
    plt.figure(figsize=(10, 5))
    plt.plot(episodes, rewards, marker='o', linestyle='-', color='b', alpha=0.7, label='Episode Reward')
    plt.title("CartPole-v1 Episode Rewards (Random Agent)")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.grid(True)
    plt.legend()
    
    output_path = os.path.join("figures", "reward_cartpole.png")
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    print(f"Da luu bieu do vao: {output_path}")
