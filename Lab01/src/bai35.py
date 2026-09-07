"""
Bài 35: So sánh hiệu năng của ba Agent trên CartPole-v1
Yêu cầu:
- Đánh giá 3 Agent: Random Policy, Angle-based Policy, Improved Heuristic Policy.
- Chạy 500 episode cho mỗi agent.
- Lập bảng so sánh các chỉ số (Mean reward, Std, Min, Max, Mean length).
- Vẽ biểu đồ so sánh và lưu vào figures/comparison_agents.png.
- Viết 5-10 dòng nhận xét đánh giá.
"""

import os
import sys
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def random_policy(obs, env):
    return env.action_space.sample()

def angle_based_policy(obs, env):
    return 1 if obs[2] > 0 else 0

def improved_policy(obs, env):
    pole_angle = obs[2]
    pole_angular_vel = obs[3]
    return 1 if (pole_angle + 0.15 * pole_angular_vel) > 0 else 0

def evaluate_agent(env_name, policy_func, n_episodes=500, seed=42):
    env = gym.make(env_name)
    rewards = []
    lengths = []
    
    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep)
        r_sum = 0.0
        l_cnt = 0
        while True:
            action = policy_func(obs, env)
            obs, reward, terminated, truncated, info = env.step(action)
            r_sum += reward
            l_cnt += 1
            if terminated or truncated:
                break
        rewards.append(r_sum)
        lengths.append(l_cnt)
        
    env.close()
    return rewards, lengths

if __name__ == "__main__":
    n_episodes = 500
    env_name = "CartPole-v1"
    
    print("Dang danh gia 3 Agent qua 500 episode...")
    r_rand, l_rand = evaluate_agent(env_name, random_policy, n_episodes)
    r_angle, l_angle = evaluate_agent(env_name, angle_based_policy, n_episodes)
    r_impr, l_impr = evaluate_agent(env_name, improved_policy, n_episodes)
    
    agents_data = [
        ("Random", r_rand, l_rand),
        ("Angle-based", r_angle, l_angle),
        ("Improved", r_impr, l_impr)
    ]
    
    print(f"\n{'Agent':<15} | {'Mean Reward':<12} | {'Std':<10} | {'Min':<8} | {'Max':<8} | {'Mean Length':<12}")
    print("-" * 75)
    for name, r_list, l_list in agents_data:
        r_arr, l_arr = np.array(r_list), np.array(l_list)
        print(f"{name:<15} | {np.mean(r_arr):>12.2f} | {np.std(r_arr):>10.2f} | {np.min(r_arr):>8.1f} | {np.max(r_arr):>8.1f} | {np.mean(l_arr):>12.2f}")
        
    os.makedirs("figures", exist_ok=True)
    plt.figure(figsize=(12, 6))
    
    plt.plot(r_rand, label='Random Policy', alpha=0.5, color='gray')
    plt.plot(r_angle, label='Angle-based Policy', alpha=0.7, color='orange')
    plt.plot(r_impr, label='Improved Policy (Angle + Velocity)', alpha=0.9, color='green')
    
    plt.title("Comparison of 3 Agents on CartPole-v1 (500 Episodes)")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True)
    plt.legend()
    
    out_file = os.path.join("figures", "comparison_agents.png")
    plt.savefig(out_file, dpi=300)
    plt.close()
    
    print(f"\nDa luu bieu do so sanh vao: {out_file}")

"""
NHẬN XÉT ĐÁNH GIÁ (5-10 DÒNG):
1. Random Policy hoạt động rất kém trên CartPole-v1 với phần thưởng trung bình chỉ đạt khoảng 20-23 điểm.
2. Angle-based Policy (chỉ nhìn vào góc con lắc) giúp cải thiện đáng kể hiệu năng, đẩy phần thưởng trung bình lên khoảng 40-45 điểm.
3. Improved Policy (kết hợp góc con lắc và vận tốc góc) mang lại hiệu quả vượt trội, giữ con lắc cân bằng lâu hơn rất nhiều.
4. Việc kết hợp vận tốc góc giúp Agent dự đoán được đà di chuyển của con lắc trước khi nó bị nghiêng quá sâu, từ đó đưa ra hành động phản ứng kịp thời hơn.
5. Kết quả thí nghiệm khẳng định việc tận dụng thông tin trạng thái (observation) một cách hợp lý là yếu tố quyết định chất lượng của một Policy trong các bài toán điều khiển.
"""
