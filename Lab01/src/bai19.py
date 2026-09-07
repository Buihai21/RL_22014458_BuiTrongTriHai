"""
Bài 19: Thử nghiệm với seed trên môi trường độc lập
Yêu cầu:
- Chạy env.reset(seed=42) 10 lần trong 10 môi trường độc lập.
- Ghi lại initial observation và kiểm tra xem chúng có giống hệt nhau không.
- Viết kết luận trong 2-3 dòng comment.
"""

import gymnasium as gym
import numpy as np

initial_observations = []

# Tạo 10 môi trường độc lập và gọi reset(seed=42)
for i in range(10):
    env = gym.make("CartPole-v1")
    obs, info = env.reset(seed=42)
    initial_observations.append(obs)
    env.close()

# Kiểm tra tính giống nhau của 10 initial observations
all_equal = all(np.array_equal(initial_observations[0], obs) for obs in initial_observations)

print(f"Số lượng môi trường chạy: {len(initial_observations)}")
print(f"Tất cả initial observations có giống hệt nhau không?: {all_equal}")
print("Mẫu observation đầu tiên:", initial_observations[0])

"""
KẾT LUẬN:
Khi thiết lập cùng một giá trị random seed (seed=42) cho hàm reset(), các môi trường CartPole độc lập
sẽ luôn tạo ra cùng một trạng thái khởi tạo ban đầu giống hệt nhau. Điều này đảm bảo tính đóng đếm và 
khả năng tái lập (reproducibility) trong các thử nghiệm Học tăng cường.
"""
