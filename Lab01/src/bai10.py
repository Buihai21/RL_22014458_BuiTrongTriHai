"""
Bài 10: Tính tổng reward của 1 episode
Yêu cầu:
- Khởi tạo total_reward = 0.0
- Cộng dồn reward sau mỗi bước tương tác.
- In ra Episode length và Total reward khi episode kết thúc.
"""

import gymnasium as gym

env = gym.make("CartPole-v1")
obs, info = env.reset(seed=42)

total_reward = 0.0
episode_length = 0

for t in range(500):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    episode_length += 1
    
    if terminated or truncated:
        break

print(f"Episode length: {episode_length}")
print(f"Total reward: {total_reward}")

env.close()
