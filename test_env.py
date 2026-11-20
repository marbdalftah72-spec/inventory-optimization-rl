import gymnasium as gym
import numpy as np
from inventory_env import InventoryEnv

def random_agent_test(episodes=5, seed=42):
    env = InventoryEnv()
    np.random.seed(seed)
    total_rewards = []
    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0
        while not done:
            action = env.action_space.sample()
            obs, reward, done, _, _ = env.step(action)
            ep_reward += reward
        total_rewards.append(ep_reward)
    print("Random agent rewards per episode:", total_rewards)
    print("ENV TEST PASSED")

if __name__ == "__main__":
    random_agent_test()
