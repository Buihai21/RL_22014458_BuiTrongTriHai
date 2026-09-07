"""
Bài 16: Tìm Episode tốt nhất
Yêu cầu:
- Chạy 100 episode bằng Random Agent.
- Tìm episode có reward lớn nhất, reward tương ứng, độ dài episode tương ứng.
- Không duyệt lại môi trường để tìm kết quả.
"""

import gymnasium as gym
import numpy as np

def run_episode(env):
    obs, info = env.reset()
    total_reward = 0.0
    length = 0
    while True:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        length += 1
        if terminated or truncated:
            break
    return total_reward, length

if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    
    history = []  # Lưu (episode_idx, reward, length)
    for ep in range(1, 101):
        r, l = run_episode(env)
        history.append((ep, r, l))
        
    # Tìm episode có reward lớn nhất (không chạy lại môi trường)
    best_episode = max(history, key=lambda x: x[1])
    
    print(f"Best Episode Index : {best_episode[0]}")
    print(f"Best Reward        : {best_episode[1]}")
    print(f"Best Episode Length: {best_episode[2]}")
    
    env.close()
