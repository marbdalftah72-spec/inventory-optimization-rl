import gymnasium as gym
import numpy as np
from gymnasium import spaces
from config import (
    MAX_INVENTORY,
    INITIAL_INVENTORY,
    EPISODE_LENGTH,
    ORDER_ACTIONS,
    DEMAND_LAMBDA,
    SALE_PRICE,
    PURCHASE_COST,
    HOLDING_COST,
    STOCKOUT_PENALTY,
    SEED,
)

class InventoryEnv(gym.Env):
    """Warehouse inventory management environment.

    Observation: current inventory level (int) – Box(0, MAX_INVENTORY)
    Action: Discrete(len(ORDER_ACTIONS)) – index into ORDER_ACTIONS list.
    """

    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(low=0, high=MAX_INVENTORY, shape=(1,), dtype=np.int32)
        self.action_space = spaces.Discrete(len(ORDER_ACTIONS))
        self.reset(seed=SEED)
        self.stockout_rates = []
        self.inventory_levels = []

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.current_inventory = INITIAL_INVENTORY
        self.current_step = 0
        self.stockout_rates = []
        self.inventory_levels = []
        return np.array([self.current_inventory], dtype=np.int32), {}

    def step(self, action):
        order_qty = ORDER_ACTIONS[action]
        # Immediate replenishment, cap to max inventory
        self.current_inventory = min(self.current_inventory + order_qty, MAX_INVENTORY)
        # Sample demand (Poisson)
        demand = np.random.poisson(DEMAND_LAMBDA)
        sales = min(self.current_inventory, demand)
        unmet = demand - sales
        # Compute reward
        reward = (
            sales * SALE_PRICE
            - order_qty * PURCHASE_COST
            - self.current_inventory * HOLDING_COST
            - unmet * STOCKOUT_PENALTY
        )
        # Update inventory after sales
        self.current_inventory -= sales
        # Track metrics
        self.stockout_rates.append(unmet / demand if demand > 0 else 0.0)
        self.inventory_levels.append(self.current_inventory)
        self.current_step += 1
        done = self.current_step >= EPISODE_LENGTH
        info = {}
        return np.array([self.current_inventory], dtype=np.int32), reward, done, False, info

    def render(self):
        pass

    def close(self):
        pass
