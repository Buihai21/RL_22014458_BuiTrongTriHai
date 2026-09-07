"""
Bài 32: Cải tiến Heuristic Policy dùng nhiều thành phần Observation
Yêu cầu:
- Sử dụng tối thiểu 2 thành phần trong observation (chẳng hạn góc pole và vận tốc góc).
- Quyết định luật chọn action dựa trên sự kết hợp giữa Pole Angle và Pole Angular Velocity.
- Mục tiêu: mean reward thu được vượt trội hơn hẳn so với Random Policy và Angle-only Policy.
"""

import gymnasium as gym
import numpy as np

def improved_heuristic_policy(observation):
    pole_angle = observation[2]
    pole_angular_velocity = observation[3]
    
    # Kết hợp góc và vận tốc góc để dự đoán chiều nghiêng tiếp theo
    # Luật: nếu (góc + 0.1 * vận tốc góc) > 0 thì đẩy sang phải (1), ngược lại đẩy sang trái (0)
    signal = pole_angle + 0.15 * pole_angular_velocity
    return 1 if signal > 0 else 0

def random_policy(observation, env):
    return env.action_space.sample()

def evaluate(env, policy_func, n_episodes=100):
    rewards = []
    for _ in range(n_episodes):
        obs, info = env.reset()
        ep_reward = 0.0
        while True:
            action = policy_func(obs) if policy_func != random_policy else policy_func(obs, env)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            if terminated or truncated:
                break
        rewards.append(ep_reward)
    return np.mean(rewards)

if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    
    mean_improved = evaluate(env, improved_heuristic_policy, 100)
    mean_random = evaluate(env, random_policy, 100)
    
    print(f"Mean Reward (Improved Heuristic Policy) : {mean_improved:.2f}")
    print(f"Mean Reward (Random Policy)             : {mean_random:.2f}")
    print(f"Kết quả cải tiến > Random Policy?: {mean_improved > mean_random}")
    
    env.close()
