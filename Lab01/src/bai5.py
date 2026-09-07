"""
Bài 5: Quan sát trạng thái ban đầu của CartPole-v1
Yêu cầu:
- Gọi observation, info = env.reset(seed=42)
- In Observation, Type, Shape, Info
- Comment giải thích kiểu dữ liệu và ý nghĩa của từng phần tử trong observation.
"""

import gymnasium as gym

# 1. Tạo môi trường
env = gym.make("CartPole-v1")

# 2. Reset môi trường với seed=42
observation, info = env.reset(seed=42)

# 3. In thông tin trạng thái ban đầu
print("Observation:", observation)
print("Type:", type(observation))
print("Shape:", observation.shape)
print("Info:", info)

"""
Giải thích chi tiết 4 phần tử trong observation của CartPole-v1:
- observation[0]: Cart Position (float64) - Vị trí của xe trượt trên trục X (-4.8 đến 4.8).
- observation[1]: Cart Velocity (float64) - Vận tốc của xe trượt (-inf đến +inf).
- observation[2]: Pole Angle (float64) - Góc nghiêng của con lắc tính theo radian (~ -0.418 đến 0.418 rad).
- observation[3]: Pole Angular Velocity (float64) - Vận tốc góc của con lắc (-inf đến +inf).
"""

# 4. Đóng môi trường
env.close()
