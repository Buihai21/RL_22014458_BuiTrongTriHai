"""
Bài 31: Policy dựa trên Observation (Góc nghiêng của con lắc)
Yêu cầu:
- Xây dựng angle_based_policy(observation) dựa trên góc của pole (observation[2]).
- Nếu pole_angle > 0: đẩy xe sang phải (action 1) để đỡ con lắc.
- Ngược lại: đẩy xe sang trái (action 0).
- Chạy 100 episode và so sánh với Random Policy.
"""

import gymnasium as gym
import numpy as np

def angle_based_policy(observation):
    pole_angle = observation[2]
    if pole_angle > 0:
        return 1  # Đẩy xe sang phải
    else:
        return 0  # Đẩy xe sang trái

def random_policy(observation, env):
    return env.action_space.sample()

def evaluate(env, policy_type="angle", n_episodes=100):
    rewards = []
    for _ in range(n_episodes):
        obs, info = env.reset()
        ep_reward = 0.0
        while True:
            if policy_type == "angle":
                action = angle_based_policy(obs)
            else:
                action = random_policy(obs, env)
                
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            if terminated or truncated:
                break
        rewards.append(ep_reward)
    return np.mean(rewards)

if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    
    mean_angle = evaluate(env, "angle", 100)
    mean_random = evaluate(env, "random", 100)
    
    print(f"Mean Reward (Angle-based Policy) : {mean_angle:.2f}")
    print(f"Mean Reward (Random Policy)      : {mean_random:.2f}")
    print(f"Hiệu quả tăng gấp khoảng {mean_angle / mean_random:.1f} lần so với Random Policy!")
    
    env.close()
