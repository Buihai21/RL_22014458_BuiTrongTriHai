"""
Bài 23: Tạo môi trường FrozenLake (rời rạc)
Yêu cầu:
- Khởi tạo env = gym.make("FrozenLake-v1", is_slippery=False)
- In env.observation_space và env.action_space
- Xác định bằng code số lượng state và số lượng action.
"""

import gymnasium as gym

# 1. Tạo môi trường FrozenLake không trượt (is_slippery=False)
env = gym.make("FrozenLake-v1", is_slippery=False)

# 2. In không gian trạng thái và hành động
print("Observation space:", env.observation_space)
print("Action space:", env.action_space)

# 3. Tự động xác định số state và số action bằng code
num_states = env.observation_space.n
num_actions = env.action_space.n

print(f"Số lượng state (trạng thái): {num_states}")
print(f"Số lượng action (hành động): {num_actions}")

# 4. Đóng môi trường
env.close()
