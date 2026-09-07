"""
Bài 24: Hiển thị FrozenLake dạng văn bản (text/ansi)
Yêu cầu:
- Tạo môi trường với render_mode="ansi"
- Reset môi trường và in kết quả env.render()
- Quan sát cấu trúc bản đồ: Start (S), Frozen (F), Hole (H), Goal (G).
"""

import gymnasium as gym

# 1. Khởi tạo môi trường với render_mode="ansi"
env = gym.make("FrozenLake-v1", is_slippery=False, render_mode="ansi")

# 2. Reset môi trường
obs, info = env.reset(seed=42)

# 3. Render bản đồ dạng chuỗi ANSI và in ra màn hình
map_text = env.render()
print("Bản đồ FrozenLake (ansi format):")
print(map_text)
print("Chú thích:")
print("S: Start (Vị trí xuất phát)")
print("F: Frozen (Mặt băng an toàn)")
print("H: Hole (Hố nước nguy hiểm - rớt xuống sẽ kết thúc)")
print("G: Goal (Đích đến chiến thắng)")

env.close()
