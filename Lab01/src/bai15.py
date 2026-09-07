"""
Bài 15: Tính thống kê reward bằng NumPy
Yêu cầu:
- Chạy 100 episode bằng Random Agent.
- Dùng NumPy tính: mean, min, max, std.
- In kết quả định dạng 2 chữ số thập phân.
"""

import gymnasium as gym
import numpy as np

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
    
    episode_rewards = [run_episode(env) for _ in range(100)]
    rewards_arr = np.array(episode_rewards)
    
    mean_reward = np.mean(rewards_arr)
    min_reward = np.min(rewards_arr)
    max_reward = np.max(rewards_arr)
    std_reward = np.std(rewards_arr)
    
    print(f"Mean reward : {mean_reward:.2f}")
    print(f"Min reward  : {min_reward:.2f}")
    print(f"Max reward  : {max_reward:.2f}")
    print(f"Std reward  : {std_reward:.2f}")
    
    env.close()
