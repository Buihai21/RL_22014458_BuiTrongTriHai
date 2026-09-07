"""
Bài 8: Viết hàm run_one_step()
Yêu cầu:
- Xây dựng hàm run_one_step(env, action) thực hiện env.step(action)
- Trả về observation, reward, terminated, truncated, info
- Thử nghiệm hàm với ít nhất 5 bước (5 actions).
"""

import gymnasium as gym

def run_one_step(env, action):
    """
    Thực hiện một bước tương tác với môi trường theo action được truyền vào.
    """
    observation, reward, terminated, truncated, info = env.step(action)
    return observation, reward, terminated, truncated, info

if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    obs, info = env.reset(seed=42)
    print("Khởi tạo môi trường thành công. Trạng thái ban đầu:", obs)
    
    # Kiểm thử với 5 bước tương tác
    for step in range(1, 6):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = run_one_step(env, action)
        print(f"Step {step} | Action: {action} | Obs: {obs} | Reward: {reward} | Terminated: {terminated} | Truncated: {truncated}")
        if terminated or truncated:
            print("Episode kết thúc sớm!")
            break
            
    env.close()
