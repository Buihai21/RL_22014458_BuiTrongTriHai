"""
Bài 14: Chạy 100 episode và lưu danh sách phần thưởng
Yêu cầu:
- Chạy Random Agent trong 100 episode.
- Lưu reward của từng episode vào mảng episode_rewards = []
- Không in từng timestep ra màn hình.
"""

import gymnasium as gym

def run_episode(env):
    obs, info = env.reset()
    total_reward = 0.0
    while True:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break
    return total_reward

if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    
    episode_rewards = []
    num_episodes = 100
    
    for ep in range(num_episodes):
        r = run_episode(env)
        episode_rewards.append(r)
        
    print(f"Đã hoàn thành {len(episode_rewards)} episode.")
    print("5 phần thưởng đầu tiên:", episode_rewards[:5])
    
    env.close()
