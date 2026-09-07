"""
Bài 4: Khám phá observation space của CartPole-v1
Yêu cầu:
- In env.observation_space
- Xác định shape, kiểu dữ liệu (dtype), giới hạn dưới (low), giới hạn trên (high)
"""

import gymnasium as gym

# 1. Tạo môi trường
env = gym.make("CartPole-v1")

# 2. In observation space
print("Observation space:", env.observation_space)

# 3. Xác định các thông số của observation space
print("Shape of observation:", env.observation_space.shape)
print("Data type:", env.observation_space.dtype)
print("Lower bounds (low):", env.observation_space.low)
print("Upper bounds (high):", env.observation_space.high)

# 4. Đóng môi trường
env.close()
