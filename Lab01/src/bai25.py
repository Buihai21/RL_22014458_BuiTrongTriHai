"""
Bài 25: Ánh xạ action sang tên hành động tương ứng
Yêu cầu:
- Xác định ý nghĩa của các action trong FrozenLake (0: LEFT, 1: DOWN, 2: RIGHT, 3: UP).
- Tạo dictionary ACTION_NAMES
- Sinh action ngẫu nhiên và in theo định dạng "Action <id> -> <NAME>".
"""

import gymnasium as gym

# Dictionary ánh xạ action ID sang tên bước đi
ACTION_NAMES = {
    0: "LEFT",
    1: "DOWN",
    2: "RIGHT",
    3: "UP"
}

env = gym.make("FrozenLake-v1", is_slippery=False)
obs, info = env.reset(seed=42)

# Sinh action ngẫu nhiên
action = env.action_space.sample()

# In kết quả ánh xạ
print(f"Action {action} -> {ACTION_NAMES[action]}")

env.close()
