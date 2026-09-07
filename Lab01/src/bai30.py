"""
Bài 30: Policy luôn chọn một action cố định
Yêu cầu:
- Xây dựng always_left_policy(observation) luôn trả về 0 (đẩy xe sang trái).
- Xây dựng always_right_policy(observation) luôn trả về 1 (đẩy xe sang phải).
- Chạy mỗi policy 100 episode và so sánh reward trung bình.
"""

import gymnasium as gym
import numpy as np

def always_left_policy(observation):
    return 0

def always_right_policy(observation):
    return 1

def evaluate_policy(env, policy_func, n_episodes=100):
    rewards = []
    for _ in range(n_episodes):
        obs, info = env.reset()
        ep_reward = 0.0
        while True:
            action = policy_func(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            if terminated or truncated:
                break
        rewards.append(ep_reward)
    return np.mean(rewards)

if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    
    mean_left = evaluate_policy(env, always_left_policy, 100)
    mean_right = evaluate_policy(env, always_right_policy, 100)
    
    print(f"Mean Reward (Always Left Policy)  : {mean_left:.2f}")
    print(f"Mean Reward (Always Right Policy) : {mean_right:.2f}")
    
    env.close()
