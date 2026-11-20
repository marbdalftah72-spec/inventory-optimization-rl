# Shared configuration constants for the Warehouse Inventory Management RL project
SEED = 42
MAX_INVENTORY = 50
INITIAL_INVENTORY = 20
EPISODE_LENGTH = 30  # days (each step represents one day)
ORDER_ACTIONS = [0, 5, 10, 15, 20, 25]
DEMAND_LAMBDA = 5  # Poisson demand mean per day
SALE_PRICE = 10
PURCHASE_COST = 5
HOLDING_COST = 0.5  # per unit per day
STOCKOUT_PENALTY = 8  # per unmet unit
TRAIN_EPISODES = 1000
EVAL_EPISODES = 100
