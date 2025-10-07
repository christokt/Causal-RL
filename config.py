"""
Configuration file for all hyperparameters - FIXED VERSION
Includes adjusted reward weights and all necessary parameters
"""

import torch

class Config:
    # ==================== Environment Parameters ====================
    NUM_UES = 8                    # L: Number of UEs
    PACKETS_PER_UE = 50           # P: Packets each UE must transmit
    SLOT_DURATION = 0.01          # Δt = 10ms
    MAX_BUFFER_SIZE = 50
    
    # ==================== Episode Parameters ====================
    MAX_EPISODE_LENGTH = 3000     # Maximum slots per episode (reduced from 5000)
    NUM_EPISODES = 1000           # Training episodes
    
    # ==================== Reward Weights (FIXED) ====================
    # These weights balance the different objectives in the reward function
    W_GOODPUT = 1.0               # Weight for goodput reward
    W_COLLISION = 0.2             # Reduced from 0.5 (less afraid of collisions)
    W_LOSS = 5.0                  # Increased from 1.5 (punish packet losses harder)
    W_EFFICIENCY = 0.05           # Reduced from 0.1
    
    # Reward scaling factors
    ALPHA_GOODPUT = 20.0          # Increased from 10.0 (reward success more)
    ALPHA_COLLISION = 3.0         # Reduced from 5.0 (less collision penalty)
    ALPHA_LOSS = 30.0             # Increased from 15.0 (harsh penalty for packet loss)
    ALPHA_EFFICIENCY = 0.1        # Small bonus for channel utilization
    
    # ==================== Q-Learning Parameters ====================
    LEARNING_RATE_Q = 0.15        # α: Increased from 0.1 (learn faster)
    DISCOUNT_FACTOR = 0.95        # γ: How much to value future rewards
    EPSILON_START = 0.5           # Starting exploration rate (increased from 0.3)
    EPSILON_END = 0.05            # Minimum exploration rate
    EPSILON_DECAY = 300           # Episodes over which to decay epsilon (reduced from 500)
    
    # ==================== PPO Parameters ====================
    LEARNING_RATE_PPO = 3e-4      # Learning rate for PPO
    PPO_EPOCHS = 10               # K: Number of PPO update epochs
    PPO_CLIP_EPSILON = 0.2        # ε: Clipping parameter for PPO
    GAE_LAMBDA = 0.95             # λ: GAE parameter
    MINI_BATCH_SIZE = 256         # M: Mini-batch size for training
    
    # ==================== Causal World Model Parameters ====================
    HIDDEN_DIM = 128              # d_h: Embedding dimension for neural networks
    N_GRAPH_UPDATE = 50           # Update causal graph every n episodes
    K_STEP_ROLLOUT = 5            # k: Number of steps for model rollout
    HISTORY_WINDOW = 5            # N: History length for state representation
    
    # ==================== Counterfactual Planning ====================
    PLANNING_HORIZON = 5          # H: Lookahead steps for BS planning
    
    # ==================== Replay Buffer ====================
    BUFFER_SIZE = 50000           # Maximum size of replay buffer
    
    # ==================== Constraints ====================
    MAX_PLR = 0.05                # Maximum acceptable packet loss rate (5%)
    
    # ==================== Device Configuration ====================
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
