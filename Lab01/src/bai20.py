"""
Bài 20: So sánh hiệu năng hai seed khác nhau
Yêu cầu:
- Thử nghiệm với seed = 42 và seed = 100.
- Mỗi seed chạy 20 episode với Random Agent.
- Tính phần thưởng trung bình của từng nhóm seed.
"""

import gymnasium as gym
import numpy as np

def run_experiment_for_seed(seed_val, n_episodes=20):
    env = gym.make("CartPole-v1")
    rewards = []
    
    for ep in range(n_episodes):
        # Đặt seed cho mỗi episode (seed_val + ep để các episode khác nhau nhưng tái lập được)
        obs, info = env.reset(seed=seed_val + ep)
        total_reward = 0.0
        while True:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        rewards.append(total_reward)
        
    env.close()
    return np.mean(rewards)

if __name__ == "__main__":
    mean_seed_42 = run_experiment_for_seed(42, 20)
    mean_seed_100 = run_experiment_for_seed(100, 20)
    
    print(f"Reward trung bình với Seed 42  : {mean_seed_42:.2f}")
    print(f"Reward trung bình với Seed 100 : {mean_seed_100:.2f}")
