"""
Bài 7: Một bước tương tác với môi trường CartPole-v1
Yêu cầu:
- Khởi tạo môi trường và chỉ thực hiện đúng 1 action.
- In ra State before action, Action, State after action, Reward, Terminated, Truncated, Info.
"""

import gymnasium as gym

# 1. Tạo môi trường và reset
env = gym.make("CartPole-v1")
state_before, info = env.reset(seed=42)

# 2. Chọn 1 action
action = env.action_space.sample()

# 3. Thực hiện 1 bước tương tác
state_after, reward, terminated, truncated, info = env.step(action)

# 4. In thông tin bước tương tác
print("State before action:", state_before)
print("Action:", action)
print("State after action:", state_after)
print("Reward:", reward)
print("Terminated:", terminated)
print("Truncated:", truncated)
print("Info:", info)

# 5. Đóng môi trường
env.close()
