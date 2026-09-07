"""
Bài 13: Chạy 10 episode bằng Random Agent
Yêu cầu:
- Chạy 10 episode với Random Agent.
- In ra bảng thống kê định dạng: Episode | Reward | Length
"""

import gymnasium as gym

def random_agent(env):
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
    
    print(f"{'Episode':^10} | {'Reward':^10} | {'Length':^10}")
    print("-" * 36)
    
    for ep in range(1, 11):
        reward, length = random_agent(env)
        print(f"{ep:^10d} | {reward:^10.1f} | {length:^10d}")
        
    env.close()
