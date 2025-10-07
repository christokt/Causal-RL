"""
Wireless uplink scheduling environment with collision dynamics
"""

import numpy as np
from typing import Dict, Tuple, List
from config import Config

class WirelessSchedulingEnv:
    def __init__(self, config: Config):
        self.config = config
        self.L = config.NUM_UES
        self.P = config.PACKETS_PER_UE
        
        # State variables
        self.buffers = None           # B^u_t for each UE
        self.losses = None            # L^u_t cumulative losses
        self.completed = None         # Successfully delivered packets
        self.last_actions = None      # a^u_{t-1}
        self.last_grant = None        # m_{t-1}
        self.collision_history = None # Recent collisions
        
        self.current_step = 0
        self.reset()
    
    def reset(self) -> Dict:
        """Reset environment to initial state"""
        self.buffers = np.full(self.L, self.P, dtype=np.int32)
        self.losses = np.zeros(self.L, dtype=np.int32)
        self.completed = np.zeros(self.L, dtype=np.int32)
        self.last_actions = np.zeros(self.L, dtype=np.int32)
        self.last_grant = 0
        self.collision_history = []
        self.current_step = 0
        
        return self._get_observations()
    
    def step(self, ue_actions: np.ndarray, bs_grant: int) -> Tuple[Dict, float, bool, Dict]:
        """
        Execute one time step
        
        Args:
            ue_actions: Array of shape (L,) with actions {0, 1, 2} for each UE
            bs_grant: Integer in {0, 1, ..., L} indicating which UE is granted
        
        Returns:
            observations: Dict with observations for each UE and BS
            reward: Scalar reward
            done: Boolean indicating episode termination
            info: Dict with additional information
        """
        # Count transmissions (action = 1)
        num_transmitters = np.sum(ue_actions == 1)
        
        # Determine collision
        collision = (num_transmitters >= 2)
        
        # Determine successful transmission
        success = np.zeros(self.L, dtype=bool)
        if num_transmitters == 1:
            successful_ue = np.where(ue_actions == 1)[0][0]
            success[successful_ue] = True
        
        # Determine packet losses (DELETE when previous transmission failed)
        losses = np.zeros(self.L, dtype=bool)
        for u in range(self.L):
            if ue_actions[u] == 2:  # DELETE action
                # Loss occurs if last action wasn't transmit OR if it was but collided
                if self.last_actions[u] != 1:
                    losses[u] = True
                elif len(self.collision_history) > 0 and self.collision_history[-1]:
                    # Check if last step had collision involving this UE
                    losses[u] = True
        
        # Update buffers
        for u in range(self.L):
            if ue_actions[u] == 2:  # DELETE
                self.buffers[u] = max(0, self.buffers[u] - 1)
                if losses[u]:
                    self.losses[u] += 1
                else:
                    self.completed[u] += 1
            elif success[u]:  # Successful transmission
                self.buffers[u] = max(0, self.buffers[u] - 1)
                self.completed[u] += 1
        
        # Compute reward
        reward = self._compute_reward(success, collision, losses, ue_actions)
        
        # Update history
        self.last_actions = ue_actions.copy()
        self.last_grant = bs_grant
        self.collision_history.append(collision)
        if len(self.collision_history) > self.config.HISTORY_WINDOW:
            self.collision_history.pop(0)
        
        self.current_step += 1
        
        # Check termination
        total_packets_accounted = np.sum(self.buffers) + np.sum(self.completed) + np.sum(self.losses)
        done = (total_packets_accounted == 0) or (self.current_step >= self.config.MAX_EPISODE_LENGTH)
        
        # Get observations
        observations = self._get_observations()
        
        # Info dictionary
        info = {
            'collision': collision,
            'success': success,
            'losses': losses,
            'num_transmitters': num_transmitters,
            'buffers': self.buffers.copy(),
            'completed': self.completed.copy(),
            'total_losses': self.losses.copy(),
            'step': self.current_step
        }
        
        return observations, reward, done, info
    
    def _compute_reward(self, success: np.ndarray, collision: bool, 
                       losses: np.ndarray, actions: np.ndarray) -> float:
        """Compute multi-objective reward"""
        cfg = self.config
        
        # Goodput reward
        r_goodput = cfg.ALPHA_GOODPUT * np.sum(success)
        
        # Collision penalty
        r_collision = -cfg.ALPHA_COLLISION * float(collision)
        
        # Packet loss penalty
        r_loss = -cfg.ALPHA_LOSS * np.sum(losses)
        
        # Channel utilization (penalize idling)
        num_idle = np.sum(actions == 0)
        r_efficiency = cfg.ALPHA_EFFICIENCY * (self.L - num_idle)
         # ✅ ADD: Penalty for having packets remaining (urgency)
        total_remaining = np.sum(self.buffers)
        r_urgency = -0.01 * total_remaining  # Small penalty per packet remaining
        # Total reward
        reward = (cfg.W_GOODPUT * r_goodput + 
                 cfg.W_COLLISION * r_collision + 
                 cfg.W_LOSS * r_loss + 
                 cfg.W_EFFICIENCY * r_efficiency +
                 r_urgency)
        
        return reward
    
    def _get_observations(self) -> Dict:
        """Get observations for each agent"""
        observations = {}
        
        # UE observations
        for u in range(self.L):
            grant_received = 1 if self.last_grant == (u + 1) else 0
            collision_indicator = int(self.collision_history[-1]) if len(self.collision_history) > 0 else 0
            
            obs_u = {
                'buffer': self.buffers[u],
                'grant': grant_received,
                'collision': collision_indicator,
                'last_action': self.last_actions[u],
                'collision_rate': np.mean(self.collision_history) if len(self.collision_history) > 0 else 0.0
            }
            observations[f'ue_{u}'] = obs_u
        
        # BS observation
        observations['bs'] = {
            'buffers': self.buffers.copy(),
            'losses': self.losses.copy(),
            'completed': self.completed.copy(),
            'last_actions': self.last_actions.copy(),
            'collision': int(self.collision_history[-1]) if len(self.collision_history) > 0 else 0,
            'collision_rate': np.mean(self.collision_history) if len(self.collision_history) > 0 else 0.0
        }
        
        return observations
    
    def get_ue_state(self, ue_id: int) -> np.ndarray:
        """Get state vector for UE (for RL agent)"""
        grant_received = 1 if self.last_grant == (ue_id + 1) else 0
        collision_indicator = int(self.collision_history[-1]) if len(self.collision_history) > 0 else 0
        collision_rate = np.mean(self.collision_history) if len(self.collision_history) > 0 else 0.0
        
        state = np.array([
            self.buffers[ue_id] / self.P,  # Normalized buffer
            grant_received,
            collision_indicator,
            self.last_actions[ue_id] / 2.0,  # Normalized action
            collision_rate
        ], dtype=np.float32)
        
        return state
    
    def get_bs_state(self) -> np.ndarray:
        """Get state vector for BS"""
        state = np.concatenate([
            self.buffers / self.P,  # Normalized buffers
            self.last_actions / 2.0,  # Normalized actions
            [self.collision_history[-1] if len(self.collision_history) > 0 else 0],
            [np.mean(self.collision_history) if len(self.collision_history) > 0 else 0.0]
        ]).astype(np.float32)
        
        return state
