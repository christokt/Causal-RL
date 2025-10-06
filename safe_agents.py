"""
Safe UE agents with DELETE constraints
"""

import numpy as np
from typing import Dict, Tuple
from config import Config


class SafeUEAgent:
    """Q-learning agent for each UE with SAFE DELETE constraint"""
    
    def __init__(self, ue_id: int, config: Config, state_dim: int, action_dim: int = 3):
        self.ue_id = ue_id
        self.config = config
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Q-table: Use discretized states
        self.n_buffer_bins = 10
        self.n_collision_rate_bins = 5
        
        # Q-table shape
        q_shape = (self.n_buffer_bins, 2, 2, action_dim, self.n_collision_rate_bins, action_dim)
        self.Q = np.zeros(q_shape)
        
        self.epsilon = config.EPSILON_START
        self.alpha = config.LEARNING_RATE_Q
        self.gamma = config.DISCOUNT_FACTOR
        
        # ========= ✅ SAFETY TRACKING =========
        self.last_transmitted_step = -999  # Track when we last transmitted
        self.current_step = 0
        self.last_action_taken = 0
        # ====================================
    
    def discretize_state(self, state: np.ndarray) -> Tuple[int, int, int, int, int]:
        """Convert continuous state to discrete indices"""
        buffer_norm, grant, collision, last_action_norm, collision_rate = state
        
        buffer_bin = int(np.clip(buffer_norm * (self.n_buffer_bins - 1), 0, self.n_buffer_bins - 1))
        grant_bin = int(grant)
        collision_bin = int(collision)
        last_action_bin = int(np.clip(last_action_norm * 2, 0, 2))
        collision_rate_bin = int(np.clip(collision_rate * (self.n_collision_rate_bins - 1), 0, self.n_collision_rate_bins - 1))
        
        return (buffer_bin, grant_bin, collision_bin, last_action_bin, collision_rate_bin)
    
    def select_action(self, state: np.ndarray, buffer: int, 
                     last_action: int, collision: bool, 
                     explore: bool = True) -> int:
        """
        Select action using ε-greedy policy with SAFETY CONSTRAINTS
        
        SAFETY RULE: Only allow DELETE if:
        - Last action was TRANSMIT (action = 1)
        - No collision occurred
        - Very recent (within 2 steps)
        """
        
        self.current_step += 1
        
        # Rule 0: Must be IDLE if no packets
        if buffer == 0:
            return 0
        
        # ========= ✅ SAFETY CONSTRAINT FOR DELETE =========
        can_safely_delete = (
            last_action == 1 and                                      # Just transmitted
            not collision and                                          # No collision
            (self.current_step - self.last_transmitted_step) <= 2     # Recent (within 2 steps)
        )
        
        if not can_safely_delete:
            # DELETE is FORBIDDEN - only choose IDLE or TRANSMIT
            available_actions = [0, 1]
        else:
            # All actions available
            available_actions = [0, 1, 2]
        # =================================================
        
        # Track when we transmitted successfully
        if last_action == 1 and not collision:
            self.last_transmitted_step = self.current_step
        
        # ε-greedy selection from available actions only
        if explore and np.random.random() < self.epsilon:
            action = np.random.choice(available_actions)
        else:
            state_idx = self.discretize_state(state)
            q_values = self.Q[state_idx]
            
            # Only consider Q-values for available actions
            masked_q = np.full(self.action_dim, -np.inf)
            for a in available_actions:
                masked_q[a] = q_values[a]
            
            action = np.argmax(masked_q)
        
        self.last_action_taken = action
        return action
    
    def update(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray):
        """Q-learning update (Eq. 4 from paper)"""
        state_idx = self.discretize_state(state)
        next_state_idx = self.discretize_state(next_state)
        
        # Q(o_t, a_t) ← Q(o_t, a_t) + α[r_t + γ max_a Q(o_{t+1}, a) - Q(o_t, a_t)]
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
    
    def select_grant(self, bs_state: Dict, causal_model=None) -> int:
        """
        Select which UE to grant
        Simple heuristic: grant to UE with most packets
        """
        buffers = bs_state['buffers']
        
        # No grant if all buffers empty
        if np.max(buffers) == 0:
            return 0
        
        # Grant to UE with maximum buffer
        ue_with_max_buffer = np.argmax(buffers)
        return ue_with_max_buffer + 1  # Grant indices are 1-indexed
