"""
Bài 22: Viết hàm thí nghiệm có seed
Yêu cầu:
- Viết hàm experiment(seed, n_episodes)
- Trả về dictionary chứa thông tin thống kê: seed, mean_reward, std_reward, max_reward, min_reward.
- Chạy thí nghiệm với ít nhất 5 seed khác nhau và in kết quả.
"""

import gymnasium as gym
import numpy as np

def experiment(seed, n_episodes=50):
    env = gym.make("CartPole-v1")
    rewards = []
    
    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep)
        env.action_space.seed(seed + ep)
        total_reward = 0.0
        while True:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        rewards.append(total_reward)
        
    env.close()
    
    rewards_arr = np.array(rewards)
    return {
        "seed": seed,
        "mean_reward": round(float(np.mean(rewards_arr)), 2),
        "std_reward": round(float(np.std(rewards_arr)), 2),
        "max_reward": round(float(np.max(rewards_arr)), 2),
        "min_reward": round(float(np.min(rewards_arr)), 2)
    }

if __name__ == "__main__":
    seeds = [10, 42, 100, 2024, 9999]
    print(f"{'Seed':^10} | {'Mean':^10} | {'Std':^10} | {'Max':^10} | {'Min':^10}")
    print("-" * 58)
    
    for s in seeds:
        res = experiment(s, n_episodes=50)
        print(f"{res['seed']:^10d} | {res['mean_reward']:^10.2f} | {res['std_reward']:^10.2f} | {res['max_reward']:^10.2f} | {res['min_reward']:^10.2f}")
