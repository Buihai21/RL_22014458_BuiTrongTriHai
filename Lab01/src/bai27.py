"""
Bài 27: Đánh giá Reward trong FrozenLake với Random Policy
Yêu cầu:
- Chạy 100 episode bằng Random Policy trên FrozenLake.
- Đếm số episode thành công (success) và thất bại (failure).
- Tính success_rate = success / total_episodes.
"""

import gymnasium as gym

def evaluate_random_frozenlake(n_episodes=100):
    env = gym.make("FrozenLake-v1", is_slippery=False)
    
    success = 0
    failure = 0
    
    for ep in range(n_episodes):
        obs, info = env.reset()
        while True:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                if reward == 1.0:
                    success += 1
                else:
                    failure += 1
                break
                
    env.close()
    
    success_rate = success / n_episodes
    return success, failure, success_rate

if __name__ == "__main__":
    n_episodes = 100
    succ, fail, rate = evaluate_random_frozenlake(n_episodes)
    
    print(f"Tổng số episode chạy : {n_episodes}")
    print(f"Số lần Thành công (Success) : {succ}")
    print(f"Số lần Thất bại  (Failure) : {fail}")
    print(f"Tỷ lệ thành công (Success Rate): {rate * 100:.2f}%")
