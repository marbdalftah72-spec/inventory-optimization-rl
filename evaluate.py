# Placeholder evaluate module
import time
import json
import os
import pandas as pd
import numpy as np
from config import SEED, EVAL_EPISODES
from inventory_env import InventoryEnv

def evaluate_policy(policy_fn, algorithm_name, results_dir="results"):
    """Runs `policy_fn(env)` for EVAL_EPISODES episodes.
    policy_fn should take an env and return total reward for one episode.
    Saves per‑episode rewards CSV and a summary JSON.
    Returns (mean_reward, stockout_rate, avg_inventory).
    """
    np.random.seed(SEED)
    env = InventoryEnv()
    rewards = []
    stockouts = []
    inventories = []
    start = time.time()
    for ep in range(EVAL_EPISODES):
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0
        while not done:
            action = policy_fn(env, obs)
            obs, reward, done, _, _ = env.step(action)
            ep_reward += reward
        rewards.append(ep_reward)
        # compute episode metrics from env tracking
        stockout_rate = np.mean(env.stockout_rates) if env.stockout_rates else 0.0
        avg_inv = np.mean(env.inventory_levels) if env.inventory_levels else 0.0
        stockouts.append(stockout_rate)
        inventories.append(avg_inv)
    training_time = time.time() - start
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, f"{algorithm_name}_rewards.csv")
    pd.DataFrame({"episode": np.arange(EVAL_EPISODES), "episode_reward": rewards}).to_csv(csv_path, index=False)
    summary = {
        "algorithm": algorithm_name,
        "mean_reward_last_100": float(np.mean(rewards[-100:])),
        "stockout_rate": float(np.mean(stockouts)),
        "avg_inventory": float(np.mean(inventories)),
        "training_time_sec": training_time,
    }
    json_path = os.path.join(results_dir, f"{algorithm_name}_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    return summary

# Example dummy policy (random) – real algorithms will replace this.
def random_policy(env, observation):
    return env.action_space.sample()
