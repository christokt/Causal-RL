"""
Proximal Policy Optimization (PPO) implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple
from config import Config

class PolicyNetwork(nn.Module):
    """Policy network π_θ(a|x)"""
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, state):
        logits = self.net(state)
        return logits
    
    def get_action(self, state, deterministic=False):
        """Sample action from policy"""
        logits = self.forward(state)
        probs = F.softmax(logits, dim=-1)
        
        if deterministic:
            action = torch.argmax(probs, dim=-1)
        else:
            action = torch.multinomial(probs, 1).squeeze(-1)
        
        return action, probs
    
    def log_prob(self, state, action):
        """Compute log probability of action"""
        logits = self.forward(state)
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Gather log probs for taken actions
        action_log_probs = log_probs.gather(1, action.unsqueeze(-1).long()).squeeze(-1)
        return action_log_probs

class ValueNetwork(nn.Module):
    """Value network V_φ(x)"""
    def __init__(self, state_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state):
        return self.net(state).squeeze(-1)

def compute_gae(rewards: List[float], values: List[float], dones: List[bool], 
                gamma: float, lam: float) -> np.ndarray:
    """
    Compute Generalized Advantage Estimation (GAE)
    
    Args:
        rewards: List of rewards
        values: List of value estimates V(x_t)
        dones: List of done flags
        gamma: Discount factor
        lam: GAE lambda parameter
    
    Returns:
        advantages: NumPy array of advantage estimates
    """
    advantages = []
    gae = 0
    
    # Compute advantages backward from end of trajectory
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = 0 if dones[t] else values[t]
        else:
            next_value = values[t + 1]
        
        # TD error: δ_t = r_t + γ*V(x_{t+1}) - V(x_t)
        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
        
        # GAE: Â_t = δ_t + (γλ)*Â_{t+1}
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        advantages.insert(0, gae)
    
    return np.array(advantages)

class PPOAgent:
    """PPO agent for UE or BS"""
    def __init__(self, state_dim: int, action_dim: int, config: Config):
        self.config = config
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Networks
        self.policy = PolicyNetwork(state_dim, action_dim)
        self.value = ValueNetwork(state_dim)
        
        # Optimizers
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), 
                                                lr=config.LEARNING_RATE_PPO)
        self.value_optimizer = torch.optim.Adam(self.value.parameters(), 
                                               lr=config.LEARNING_RATE_PPO)
        
        self.device = config.DEVICE
        self.policy.to(self.device)
        self.value.to(self.device)
    
    def select_action(self, state: np.ndarray, deterministic: bool = False):
        """Select action from policy"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action, probs = self.policy.get_action(state_tensor, deterministic)
            value = self.value(state_tensor)
        
        return action.item(), value.item(), probs.squeeze(0).cpu().numpy()
    
    def update(self, trajectories: List[Dict]):
        """
        Update policy using PPO (Algorithm 2, Eq. 13)
        
        Args:
            trajectories: List of trajectory dictionaries containing:
                - states: List of states
                - actions: List of actions
                - rewards: List of rewards
                - dones: List of done flags
                - log_probs: List of old log probabilities
                - values: List of value estimates
        """
        # Prepare data
        all_states = []
        all_actions = []
        all_advantages = []
        all_returns = []
        all_old_log_probs = []
        
        for traj in trajectories:
            states = traj['states']
            actions = traj['actions']
            rewards = traj['rewards']
            dones = traj['dones']
            old_log_probs = traj['log_probs']
            values = traj['values']
            
            # Compute advantages using GAE
            advantages = compute_gae(rewards, values, dones, 
                                   self.config.DISCOUNT_FACTOR, 
                                   self.config.GAE_LAMBDA)
            
            # Compute returns (for value function training)
            returns = advantages + values
            
            all_states.extend(states)
            all_actions.extend(actions)
            all_advantages.extend(advantages)
            all_returns.extend(returns)
            all_old_log_probs.extend(old_log_probs)
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(all_states)).to(self.device)
        actions = torch.LongTensor(all_actions).to(self.device)
        advantages = torch.FloatTensor(all_advantages).to(self.device)
        returns = torch.FloatTensor(all_returns).to(self.device)
        old_log_probs = torch.FloatTensor(all_old_log_probs).to(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update for K epochs
        for epoch in range(self.config.PPO_EPOCHS):
            # Generate random mini-batches
            indices = torch.randperm(len(states))
            
            for start in range(0, len(states), self.config.MINI_BATCH_SIZE):
                end = start + self.config.MINI_BATCH_SIZE
                batch_indices = indices[start:end]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                
                # Compute new log probabilities
                new_log_probs = self.policy.log_prob(batch_states, batch_actions)
                
                # Compute ratio ρ_t = π_θ / π_θ_old (Eq. 14)
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                
                # Compute clipped objective (Eq. 13)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 
                                   1 - self.config.PPO_CLIP_EPSILON, 
                                   1 + self.config.PPO_CLIP_EPSILON) * batch_advantages
                
                # PPO objective (to MAXIMIZE, so we MINIMIZE negative)
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Add entropy bonus for exploration
                logits = self.policy(batch_states)
                probs = F.softmax(logits, dim=-1)
                entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()
                policy_loss = policy_loss - 0.01 * entropy  # Entropy coefficient
                
                # Update policy
                self.policy_optimizer.zero_grad()
                policy_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.policy_optimizer.step()
                
                # Update value function
                values_pred = self.value(batch_states)
                value_loss = F.mse_loss(values_pred, batch_returns)
                
                self.value_optimizer.zero_grad()
                value_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.value.parameters(), 0.5)
                self.value_optimizer.step()
