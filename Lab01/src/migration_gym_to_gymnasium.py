"""
Bài bắt buộc (Mục 8): Chuyển đổi mã nguồn từ Gym cũ sang Gymnasium chuẩn
Mô tả:
File này giải quyết yêu cầu chuyển đổi bài toán ví dụ từ API Gym cũ sang API Gymnasium chuẩn hiện tại.

GIẢI THÍCH CHI TIẾT CÁC CÂU HỎI TRONG COMMENT:

# 1. terminated có ý nghĩa gì?
# `terminated = True` đại diện cho việc Episode kết thúc do BẢN CHẤT CỦA BÀI TOÁN hay quy luật môi trường.
# Ví dụ trong CartPole: con lắc bị nghiêng quá góc 12 độ hoặc xe trượt vượt quá phạm vi vị trí cho phép.

# 2. truncated có ý nghĩa gì?
# `truncated = True` đại diện cho việc Episode bị dừng do GIỚI HẠN BÊN NGOÀI (như giới hạn thời gian / số bước tối đa).
# Ví dụ trong CartPole-v1: episode tự động dừng khi đạt tới bước thứ 500 dù con lắc vẫn đang giữ đứng được.

# 3. Vì sao không nên dùng done của API cũ?
# Trong API Gym cũ, biến `done` gộp chung cả 2 trường hợp (terminated và truncated). Điều này gây ra khó khăn 
# và sai lệch lớn khi huấn luyện các thuật toán Học tăng cường (như Q-Learning, PPO, SAC), vì:
# - Khi `terminated = True`: Trạng thái tiếp theo thực sự là trạng thái kết thúc (Value function V(S_next) = 0).
# - Khi `truncated = True`: Episode chỉ bị ngắt ngang do hết thời gian, trạng thái tiếp theo V(S_next) VẪN CÓ GIÁ TRỊ 
#   và không được coi là trạng thái kết thúc thực sự. Nếu gộp thành `done`, thuật toán sẽ học sai giá trị trạng thái.
"""

import sys
import gymnasium as gym

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def main():
    # 1. Sử dụng import gymnasium as gym và phiên bản CartPole-v1 hiện tại
    env = gym.make("CartPole-v1")
    
    # 2. env.reset() trong Gymnasium trả về tuple (observation, info)
    observation, info = env.reset(seed=42)
    
    for t in range(1000):
        action = env.action_space.sample()
        
        # 3. env.step() trả về 5 giá trị: observation, reward, terminated, truncated, info
        observation, reward, terminated, truncated, info = env.step(action)
        
        # 4. Kiểm tra điều kiện kết thúc bằng terminated hoặc truncated thay vì done cũ
        if terminated or truncated:
            print(f"Episode ket thuc tai buoc t = {t}.")
            if terminated:
                print("Ly do: Terminated (Con lac bi do).")
            elif truncated:
                print("Ly do: Truncated (Het gioi han so buoc).")
            break
            
    # 5. Luôn đóng môi trường sau khi hoàn thành
    env.close()

if __name__ == "__main__":
    main()
