"""
Configuration file for all hyperparameters - FIXED VERSION
"""

import torch

class Config:
    # Environment parameters
    NUM_UES = 8                    # L: Number of UEs
    PACKETS_PER_UE = 50           # P: Packets each UE must transmit
    SLOT_DURATION = 0.01          # Δt = 10ms
    MAX_BUFFER_SIZE = 50
    
    # Episode parameters
    MAX_EPISODE_LENGTH = 3000     # ✅ Reduced from 5000 (force faster learning)
    NUM_EPISODES = 1000           # Training episodes
    
    # ============= ✅ FIXED REWARD WEIGHTS =============
    W_GOODPUT = 1.0
    W_COLLISION = 0.2              # ✅ Reduced from 0.5 (less afraid of collisions)
    W_LOSS = 5.0                   # ✅ Increased from 1.5 (punish losses harder)
    W_EFFICIENCY = 0.05            # ✅ Reduced from 0.1
    
    # Reward values
    ALPHA_GOODPUT = 20.0           # ✅ Increased from 10.0 (reward success more)
    ALPHA_COLLISION = 3.0          # ✅ Reduced from 5.0 (less penalty)
    ALPHA_LOSS = 30.0              # ✅ Increased from 15.0 (harsh penalty)
    ALPHA_EFFICIENCY = 0.1
    # ==============================================
