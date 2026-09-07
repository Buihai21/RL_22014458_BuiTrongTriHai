"""
Bài 29: Viết Policy dưới dạng hàm
Yêu cầu:
- Đĩnh nghĩa hàm policy(observation, env) trả về action ngẫu nhiên.
- Thay thế toàn bộ lời gọi env.action_space.sample() bằng policy(observation, env).
"""

import gymnasium as gym

def policy(observation, env):
    """
    Policy trả về action ngẫu nhiên từ quan sát observation.
    """
    return env.action_space.sample()

def run_agent_with_policy(env, max_steps=500):
    obs, info = env.reset(seed=42)
    total_reward = 0.0
    steps = 0
    
    for _ in range(max_steps):
        # Chọn action thông qua hàm policy thay vì sample trực tiếp
        action = policy(obs, env)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        
        if terminated or truncated:
            break
            
    return total_reward, steps

if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    reward, length = run_agent_with_policy(env)
    print(f"Policy-based Agent -> Reward: {reward}, Steps: {length}")
    env.close()
