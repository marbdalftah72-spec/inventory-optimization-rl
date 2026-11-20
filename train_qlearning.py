# Placeholder training script for Tabular Q‑Learning
"""This file will contain the Tabular Q‑Learning implementation.
It currently only imports the shared environment so the module can be run
without errors.
"""

from inventory_env import InventoryEnv

if __name__ == "__main__":
    # Simple sanity check import
    env = InventoryEnv()
    print("InventoryEnv imported successfully for Q‑Learning placeholder.")
