"""
UE and BS agents using Q-learning
"""

import numpy as np
from typing import Dict
from config import Config

class UEAgent:
    """Q-learning agent for each UE"""
    def __init__(self, ue_id: int, config: Config, state_dim: int, action_dim: int = 3):
        self.ue_id = ue_id
        self.config = config
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Q-table: Use discretized states
        # State discretization: (buffer_level, grant, collision, last_action, collision_rate_bin)
        self.n_buffer_bins = 10
        self.n_collision_rate_bins = 5
        
        # Q-table shape
        q_shape = (self.n_buffer_bins, 2, 2, action_dim, self.n_collision_rate_bins, action_dim)
        self.Q = np.zeros(q_shape)
        
        self.epsilon = config.EPSILON_START
        self.alpha = config.LEARNING_RATE_Q
        self.gamma = config.DISCOUNT_FACTOR
    
    def discretize_state(self, state: np.ndarray) -> Tuple:
        """Convert continuous state to discrete indices"""
        buffer_norm, grant, collision, last_action_norm, collision_rate = state
        
        buffer_bin = int(buffer_norm * (self.n_buffer_bins - 1))
        grant_bin = int(grant)
        collision_bin = int(collision)
        last_action_bin = int(last_action_norm * 2)  # Maps [0, 0.5, 1.0] to [0, 1, 2]
        collision_rate_bin = int(collision_rate * (self.n_collision_rate_bins - 1))
        
        return (buffer_bin, grant_bin, collision_bin, last_action_bin, collision_rate_bin)
    
    def select_action(self, state: np.ndarray, buffer: int, explore: bool = True) -> int:
        """Select action using ε-greedy policy"""
        if buffer == 0:
            return 0  # Must be IDLE if no packets
        
        if explore and np.random.random() < self.epsilon:
            return np.random.randint(0, self.action_dim)
        else:
            state_idx = self.discretize_state(state)
            q_values = self.Q[state_idx]
            return np.argmax(q_values)
    
    def update(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray):
        """Q-learning update (Eq. 4)"""
        state_idx = self.discretize_state(state)
        next_state_idx = self.discretize_state(next_state)
        
        # Q-learning update
        best_next_action = np.argmax(self.Q[next_state_idx])
        td_target = reward + self.gamma * self.Q[next_state_idx][best_next_action]
        td_error = td_target - self.Q[state_idx][action]
        
        self.Q[state_idx][action] += self.alpha * td_error
    
    def decay_epsilon(self, episode: int):
        """Decay exploration rate"""
        self.epsilon = self.config.EPSILON_END + \
                      (self.config.EPSILON_START - self.config.EPSILON_END) * \
                      np.exp(-episode / self.config.EPSILON_DECAY)

class BSAgent:
    """Base station agent for grant decisions"""
    def __init__(self, config: Config, num_ues: int):
        self.config = config
        self.L = num_ues
        
        # Simple policy: prioritize UE with highest buffer
        # Can be extended to learned policy
    
    def select_grant(self, bs_state: Dict, causal_model=None) -> int:
        """
        Select which UE to grant
        
        Args:
            bs_state: Dictionary with buffer states, etc.
            causal_model: Optional causal world model for counterfactual planning
        
        Returns:
            grant: Integer in {0, 1, ..., L} where 0 = no grant
        """
        buffers = bs_state['buffers']
        
        # Simple heuristic: grant to UE with most packets
        if np.max(buffers) == 0:
            return 0  # No grant if all buffers empty
        
        ue_with_max_buffer = np.argmax(buffers)
        return ue_with_max_buffer + 1  # Grant indices are 1-indexed
    
    def counterfactual_planning(self, bs_state: Dict, causal_model, horizon: int = 5):
        """
        Counterfactual planning using causal world model (Algorithm 6)
        
        This is placeholder for now - full implementation requires trained causal model
        """
        # TODO: Implement full counterfactual rollout
        # For now, fall back to heuristic
        return self.select_grant(bs_state)
