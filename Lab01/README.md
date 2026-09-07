# Bài thực hành số 1 - Làm quen với Gymnasium

Họ tên: Bùi Trọng Trí Hải  
MSSV: 22014458  
Lớp: Reinforcement Learning  
GitHub username: Buihai21  
Repository URL: https://github.com/Buihai21/RL_22014458_BuiTrongTriHai  

Python version: 3.14.2  
Gymnasium version: 1.3.0  
NumPy version: 2.5.1  
Matplotlib version: 3.10.0  

## Cách cài đặt
```bash
cd Lab01
pip install -r requirements.txt
```

## Cách chạy từng bài
Chạy từng bài đơn lẻ bằng lệnh Python:
```bash
python src/bai1.py
python src/bai2.py
python src/bai3.py
# ...
python src/bai35.py
python src/bai36.py
python src/migration_gym_to_gymnasium.py
```

## Cách chạy chương trình tổng hợp
Chạy chương trình mini-project tổng hợp bằng:
```bash
python src/main.py
```

## Mô tả kết quả
- Đã hoàn thành toàn bộ 36 bài tập và phần bắt buộc chuyển đổi Gym cũ sang Gymnasium.
- Đã xuất đầy đủ 4 biểu đồ hình ảnh kết quả trong thư mục `figures/`:
  - `figures/reward_cartpole.png`
  - `figures/moving_average.png`
  - `figures/comparison_agents.png`
  - `figures/mini_project_results.png`
- So sánh các Agent trên CartPole-v1 qua 500 episode:
  - Random Policy: Mean Reward ~21.99
  - Angle-based Policy: Mean Reward ~41.97
  - Improved Policy: Mean Reward 500.00 (Max Score 500/500)

## Khó khăn gặp phải
- Xử lý mã hóa kí tự tiếng Việt trên Windows Console (đã khắc phục bằng `sys.stdout.reconfigure(encoding='utf-8')`).
- Phân biệt giữa hai khái niệm `terminated` (kết thúc môi trường do cơ chế vật lý) và `truncated` (bị ngắt do hết giới hạn bước).

## Kết luận
- Nắm vững kiến thức nền tảng về tương tác Agent - Environment và thư viện Gymnasium.
- Thiết kế thành công Heuristic Policy kết hợp góc nghiêng và vận tốc góc giữ cân bằng tuyệt đối cho CartPole.
