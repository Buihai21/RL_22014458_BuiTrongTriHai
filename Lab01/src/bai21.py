"""
Bài 21: Seed cho action space
Yêu cầu:
- Thiết lập seed cho env.action_space (dùng env.action_space.seed(42)).
- Sinh 20 action ngẫu nhiên.
- Chạy hai lần độc lập và kiểm tra hai chuỗi action có trùng khớp hoàn toàn không.
"""

import gymnasium as gym

def generate_actions_with_seed(seed_value=42, count=20):
    env = gym.make("CartPole-v1")
    env.reset(seed=seed_value)
    # Thiết lập seed trực tiếp cho không gian hành động action_space
    env.action_space.seed(seed_value)
    
    actions = [env.action_space.sample() for _ in range(count)]
    env.close()
    return actions

if __name__ == "__main__":
    run1 = generate_actions_with_seed(42, 20)
    run2 = generate_actions_with_seed(42, 20)
    
    print("Chuỗi Action lần 1:", run1)
    print("Chuỗi Action lần 2:", run2)
    print("Hai chuỗi action có giống hệt nhau không?:", run1 == run2)
