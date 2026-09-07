"""
Bài 26: Điều khiển FrozenLake bằng chuỗi hành động cố định
Yêu cầu:
- Thiết lập môi trường is_slippery=False.
- Tự xây dựng chuỗi actions = [...] đưa Agent từ ô Start (S) đến ô Goal (G).
- In trạng thái (state, reward, terminated) sau mỗi bước tương tác.
"""

import gymnasium as gym

ACTION_NAMES = {0: "LEFT", 1: "DOWN", 2: "RIGHT", 3: "UP"}

env = gym.make("FrozenLake-v1", is_slippery=False, render_mode="ansi")
obs, info = env.reset(seed=42)

# Chuỗi hành động để di chuyển từ (0,0) đến (3,3):
# DOWN (1), DOWN (1), RIGHT (2), RIGHT (2), DOWN (1), RIGHT (2)
actions = [1, 1, 2, 2, 1, 2]

print("Bản đồ ban đầu:")
print(env.render())

for step, act in enumerate(actions, 1):
    obs, reward, terminated, truncated, info = env.step(act)
    print(f"Bước {step} | Thực hiện Action: {act} ({ACTION_NAMES[act]}) -> State mới: {obs} | Reward: {reward}")
    print(env.render())
    
    if terminated or truncated:
        if reward == 1.0:
            print("Chúc mừng! Agent đã tới Đích (Goal) thành công!")
        else:
            print("Rất tiếc! Agent đã rơi xuống hố (Hole).")
        break

env.close()
