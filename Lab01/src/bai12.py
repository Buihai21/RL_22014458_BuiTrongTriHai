"""
Bài 12: Không sử dụng biến done từ API Gym cũ
Yêu cầu:
- Không nhận done từ env.step().
- Tự tạo biến episode_finished = terminated or truncated.
- In nguyên nhân kết thúc episode: Termination (do con lắc đổ/vượt ngưỡng) hay Truncation (do hết giới hạn số bước).
"""

import gymnasium as gym

def run_episode_with_reasons(env, max_steps=500):
    obs, info = env.reset(seed=42)
    total_reward = 0.0
    step_count = 0
    
    while True:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step_count += 1
        
        # Tự định nghĩa cờ kết thúc episode
        episode_finished = terminated or truncated
        
        if episode_finished:
            print(f"Episode kết thúc tại bước {step_count}. Tổng reward: {total_reward}")
            if terminated:
                print("Nguyên nhân kết thúc: Termination (Cơ chế bài toán: con lắc đổ quá góc cho phép hoặc xe trượt quá biên).")
            elif truncated:
                print("Nguyên nhân kết thúc: Truncation (Giới hạn bên ngoài: đạt tối đa số bước quy định).")
            break

if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    run_episode_with_reasons(env)
    env.close()
