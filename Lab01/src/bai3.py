"""
Bài 3: Khám phá action space của CartPole-v1
Yêu cầu:
- In đối tượng env.action_space
- Tự động xác định số action có thể thực hiện (dùng env.action_space.n)
- Kết quả in ra có dạng "Number of actions: ..."
"""

import gymnasium as gym

# 1. Tạo môi trường
env = gym.make("CartPole-v1")

# 2. In action space
print("Action space:", env.action_space)

# 3. Tự động xác định số action (không gán trực tiếp bằng hằng số)
num_actions = env.action_space.n
print(f"Number of actions: {num_actions}")

# 4. Đóng môi trường
env.close()
