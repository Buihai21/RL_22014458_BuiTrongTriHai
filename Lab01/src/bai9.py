"""
Bài 9: Chạy tối đa 20 bước tương tác
Yêu cầu:
- Chạy vòng lặp tối đa 20 timestep.
- Ở mỗi timestep in ra timestep t, action, reward.
- Dừng ngay lập tức nếu terminated hoặc truncated là True.
"""

import gymnasium as gym

env = gym.make("CartPole-v1")
obs, info = env.reset(seed=42)

for t in range(20):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Step {t:2d} | Action: {action} | Reward: {reward}")
    
    if terminated or truncated:
        print(f"Episode kết thúc tại bước t = {t} (Terminated: {terminated}, Truncated: {truncated})")
        break

env.close()
