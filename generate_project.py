"""
Script to generate complete project structure
"""

import os

# Define all file contents
FILES = {
    'requirements.txt': '''numpy>=1.21.0
torch>=1.10.0
matplotlib>=3.4.0
networkx>=2.6.0
scipy>=1.7.0
''',

    'config.py': '''"""
Configuration file for all hyperparameters
"""

import torch

class Config:
    # Environment parameters
    NUM_UES = 8                    # L: Number of UEs
    PACKETS_PER_UE = 50           # P: Packets each UE must transmit
    SLOT_DURATION = 0.01          # Δt = 10ms
    MAX_BUFFER_SIZE = 50
    
    # Episode parameters
    MAX_EPISODE_LENGTH = 5000     # Maximum slots per episode
    NUM_EPISODES = 2000           # Training episodes
    
    # Reward weights
    W_GOODPUT = 1.0
    W_COLLISION = 0.5
    W_LOSS = 1.5
    W_EFFICIENCY = 0.1
    
    # Reward values
    ALPHA_GOODPUT = 10.0
    ALPHA_COLLISION = 5.0
    ALPHA_LOSS = 15.0
    ALPHA_EFFICIENCY = 0.1
    
    # Q-learning parameters
    LEARNING_RATE_Q = 0.1
    DISCOUNT_FACTOR = 0.95        # γ
    EPSILON_START = 0.3
    EPSILON_END = 0.05
    EPSILON_DECAY = 500           # Episodes to decay over
    
    # PPO parameters
    LEARNING_RATE_PPO = 3e-4
    PPO_EPOCHS = 10               # K: Number of PPO update epochs
    PPO_CLIP_EPSILON = 0.2        # ε for clipping
    GAE_LAMBDA = 0.95             # λ for GAE
    MINI_BATCH_SIZE = 256         # M
    
    # Causal world model parameters
    HIDDEN_DIM = 128              # d_h: Embedding dimension
    N_GRAPH_UPDATE = 50           # Update causal graph every n episodes
    K_STEP_ROLLOUT = 5            # k-step model rollout
    HISTORY_WINDOW = 5            # N: History length
    
    # Counterfactual planning
    PLANNING_HORIZON = 5          # H: Lookahead steps for BS
    
    # Replay buffer
    BUFFER_SIZE = 50000
    
    # Target PLR constraint
    MAX_PLR = 0.05                # 5% maximum packet loss rate
    
    # Device
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
''',

    'README.md': '''# Causality Aware Wireless Scheduling via Explainable Multi-Agent Reinforcement Learning

Implementation of the paper "Causality Aware Wireless Scheduling via Explainable Multi-Agent Reinforcement Learning"

## Installation
```bash
pip install -r requirements.txt
