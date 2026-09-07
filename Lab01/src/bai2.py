import gymnasium as gym

# 1. Tạo môi trường CartPole-v1
env = gym.make("CartPole-v1")

# 2. In đối tượng môi trường env
print("Environment:", env)

# 3. Đóng môi trường sau khi hoàn thành
env.close()
