"""
Bài 11: Viết Random Agent hoàn chỉnh
Yêu cầu:
- Xây dựng hàm random_agent(env, max_steps=500)
- Reset môi trường, tương tác chọn ngẫu nhiên hành động cho đến khi dừng.
- Trả về (total_reward, episode_length).
"""

import gymnasium as gym

def random_agent(env, max_steps=500):
    """
    Thực thi 1 episode với agent chọn action ngẫu nhiên.
    Returns:
        total_reward (float): Tổng phần thưởng đạt được.
        episode_length (int): Độ dài episode (số bước).
    """
    obs, info = env.reset()
    total_reward = 0.0
    episode_length = 0
    
    for _ in range(max_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        episode_length += 1
        
        if terminated or truncated:
            break
            
    return total_reward, episode_length

if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    reward, length = random_agent(env)
    print(f"Random Agent -> Total reward: {reward}, Episode length: {length}")
    env.close()
