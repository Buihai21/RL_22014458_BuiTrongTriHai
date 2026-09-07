"""
Bài 28: So sánh môi trường Deterministic (is_slippery=False) và Stochastic (is_slippery=True)
Yêu cầu:
- Chạy 500 episode cho cả 2 trường hợp is_slippery=False và is_slippery=True.
- So sánh success rate, average reward, average episode length.
- Viết kết luận bằng comment trong code.
"""

import gymnasium as gym
import numpy as np

def run_experiment(is_slippery, n_episodes=500):
    env = gym.make("FrozenLake-v1", is_slippery=is_slippery)
    
    rewards = []
    lengths = []
    successes = 0
    
    for ep in range(n_episodes):
        obs, info = env.reset()
        ep_reward = 0.0
        ep_length = 0
        while True:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            ep_length += 1
            if terminated or truncated:
                if reward == 1.0:
                    successes += 1
                break
        rewards.append(ep_reward)
        lengths.append(ep_length)
        
    env.close()
    
    return {
        "success_rate": successes / n_episodes,
        "avg_reward": np.mean(rewards),
        "avg_length": np.mean(lengths)
    }

if __name__ == "__main__":
    n_ep = 500
    res_det = run_experiment(is_slippery=False, n_episodes=n_ep)
    res_sto = run_experiment(is_slippery=True, n_episodes=n_ep)
    
    print(f"{'Chỉ số':<25} | {'is_slippery=False':<20} | {'is_slippery=True':<20}")
    print("-" * 70)
    print(f"{'Success Rate':<25} | {res_det['success_rate']*100:>18.2f}% | {res_sto['success_rate']*100:>18.2f}%")
    print(f"{'Average Reward':<25} | {res_det['avg_reward']:>20.4f} | {res_sto['avg_reward']:>20.4f}")
    print(f"{'Average Episode Length':<25} | {res_det['avg_length']:>20.2f} | {res_sto['avg_length']:>20.2f}")

"""
KẾT LUẬN:
1. Trong môi trường Deterministic (is_slippery=False), khi thực hiện một action thì Agent chắc chắn di chuyển
   theo hướng đó. Đối với Random Policy, khả năng tình cờ chạm mốc Goal vẫn xảy ra nhưng tỷ lệ thấp.
2. Trong môi trường Stochastic (is_slippery=True), mặt băng trượt khiến hướng di chuyển của Agent bị xác xuất 
   (chỉ 1/3 đi đúng hướng, 2/3 trượt sang 2 hướng vuông góc). Việc chọn action ngẫu nhiên khiến tỷ lệ tới đích thành công
   giảm đáng kể và độ dài episode trung bình thay đổi rõ rệt.
"""
