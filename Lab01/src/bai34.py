"""
Bài 34: Xây dựng hàm evaluate_policy()
Yêu cầu:
- Xây dựng hàm evaluate_policy(env_name, policy, n_episodes=100, seed=42)
- Trả về dictionary chứa: mean_reward, std_reward, min_reward, max_reward, mean_length.
"""

import gymnasium as gym
import numpy as np

def run_episode(env, policy, seed=None):
    obs, info = env.reset(seed=seed)
    total_reward = 0.0
    length = 0
    while True:
        action = policy(obs, env)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        length += 1
        if terminated or truncated:
            break
    return total_reward, length

def evaluate_policy(env_name, policy, n_episodes=100, seed=42):
    env = gym.make(env_name)
    rewards = []
    lengths = []
    
    for ep in range(n_episodes):
        ep_seed = seed + ep if seed is not None else None
        r, l = run_episode(env, policy, seed=ep_seed)
        rewards.append(r)
        lengths.append(l)
        
    env.close()
    
    rewards_arr = np.array(rewards)
    lengths_arr = np.array(lengths)
    
    return {
        "mean_reward": round(float(np.mean(rewards_arr)), 2),
        "std_reward": round(float(np.std(rewards_arr)), 2),
        "min_reward": round(float(np.min(rewards_arr)), 2),
        "max_reward": round(float(np.max(rewards_arr)), 2),
        "mean_length": round(float(np.mean(lengths_arr)), 2)
    }

def random_policy(observation, env):
    return env.action_space.sample()

if __name__ == "__main__":
    results = evaluate_policy("CartPole-v1", random_policy, n_episodes=100, seed=42)
    print("Kết quả đánh giá Random Policy trên CartPole-v1:")
    for k, v in results.items():
        print(f"  - {k}: {v}")
