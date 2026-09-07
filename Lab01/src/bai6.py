"""
Bài 6: Sinh action ngẫu nhiên
Yêu cầu:
- Sinh 20 action ngẫu nhiên bằng env.action_space.sample()
- Lưu vào một Python list và in ra toàn bộ danh sách.
- Tính tần suất xuất hiện của từng action.
"""

from collections import Counter
import gymnasium as gym

# 1. Tạo môi trường
env = gym.make("CartPole-v1")

# 2. Sinh 20 action ngẫu nhiên và lưu vào danh sách
actions = [env.action_space.sample() for _ in range(20)]

# 3. In toàn bộ danh sách action
print("Danh sách 20 action đã sinh:", actions)

# 4. Tính và in tần suất xuất hiện của từng action
counts = Counter(actions)
print("\nTần suất xuất hiện của từng action:")
for action, count in sorted(counts.items()):
    print(f"- Action {action}: {count} lần ({count / len(actions) * 100:.1f}%)")

# 5. Đóng môi trường
env.close()
