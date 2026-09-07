"""
Bài 33: Xây dựng hàm run_episode() tổng quát
Yêu cầu:
- Xây dựng hàm tổng quát run_episode(env, policy, seed=None, max_steps=1000)
- Trả về dictionary {"reward": ..., "length": ..., "terminated": ..., "truncated": ...}
- Hàm phải dùng chung được cho mọi môi trường Gymnasium (không gắn chặt vào CartPole).
"""

import gymnasium as gym

def run_episode(env, policy, seed=None, max_steps=1000):
    """
    Hàm thực thi 1 episode tổng quát cho bất kỳ môi trường Gymnasium nào.
    """
    obs, info = env.reset(seed=seed)
    total_reward = 0.0
    length = 0
    terminated_flag = False
    truncated_flag = False
    
    for _ in range(max_steps):
        action = policy(obs, env)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        length += 1
        
        if terminated or truncated:
            terminated_flag = terminated
            truncated_flag = truncated
            break
            
    return {
        "reward": total_reward,
        "length": length,
        "terminated": terminated_flag,
        "truncated": truncated_flag
    }

def random_policy(observation, env):
    return env.action_space.sample()

if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    res = run_episode(env, random_policy, seed=42)
    print("Kết quả chạy episode tổng quát (CartPole):", res)
    env.close()
    
    # Kiểm thử tính tổng quát trên môi trường rời rạc FrozenLake
    env_fl = gym.make("FrozenLake-v1", is_slippery=False)
    res_fl = run_episode(env_fl, random_policy, seed=42)
    print("Kết quả chạy episode tổng quát (FrozenLake):", res_fl)
    env_fl.close()
