"""
Causal world model with inference networks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple
from config import Config

class VariableEncoder(nn.Module):
    """Encoder for mapping variables to fixed-dimensional embeddings"""
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def forward(self, x):
        return self.net(x)

class InferenceNetwork(nn.Module):
    """
    Inference network f^j for predicting variable v_j
    Uses GRU for actions and attention for state variables
    """
    def __init__(self, config: Config, output_type: str = 'continuous'):
        super().__init__()
        self.config = config
        self.hidden_dim = config.HIDDEN_DIM
        self.output_type = output_type  # 'continuous' or 'discrete'
        
        # GRU for processing action sequence
        self.gru = nn.GRU(input_size=self.hidden_dim, 
                         hidden_size=self.hidden_dim, 
                         batch_first=True)
        
        # Linear transforms for query and action contribution
        self.W_q = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.W_a = nn.Linear(self.hidden_dim, self.hidden_dim)
        
        # Linear transform for state contributions
        self.W_s = nn.Linear(self.hidden_dim, self.hidden_dim)
        
        # Distribution decoder
        if output_type == 'continuous':
            self.decoder_mean = nn.Linear(self.hidden_dim, 1)
            self.decoder_logvar = nn.Linear(self.hidden_dim, 1)
        else:  # discrete
            self.decoder_logits = nn.Linear(self.hidden_dim, 3)  # 3 actions
    
    def forward(self, state_encodings: torch.Tensor, action_encodings: torch.Tensor, 
                keys: torch.Tensor) -> Tuple:
        """
        Args:
            state_encodings: (batch, num_states, hidden_dim)
            action_encodings: (batch, num_actions, hidden_dim)
            keys: (num_states, hidden_dim) - shared key vectors
        
        Returns:
            If continuous: (mean, logvar)
            If discrete: logits
        """
        batch_size = state_encodings.shape[0]
        
        # Process actions with GRU (Eq. 6)
        gru_out, _ = self.gru(action_encodings)
        e_j = gru_out[:, -1, :]  # Take last hidden state (batch, hidden_dim)
        
        # Compute query and action contribution (Eq. 7-8)
        q_j = self.W_q(e_j)  # (batch, hidden_dim)
        c_a = self.W_a(e_j)  # (batch, hidden_dim)
        
        # Compute state contributions (Eq. 5)
        c_states = self.W_s(state_encodings)  # (batch, num_states, hidden_dim)
        
        # Compute attention weights (Eq. 9)
        # Expand query for broadcasting
        q_j_expanded = q_j.unsqueeze(1)  # (batch, 1, hidden_dim)
        keys_expanded = keys.unsqueeze(0)  # (1, num_states, hidden_dim)
        
        # Compute attention scores
        scores = torch.sum(keys_expanded * q_j_expanded, dim=-1)  # (batch, num_states)
        
        # Softmax with action
        scores_with_action = torch.cat([scores, torch.zeros(batch_size, 1, device=scores.device)], dim=1)
        attention_weights = F.softmax(scores_with_action, dim=1)
        
        alpha_states = attention_weights[:, :-1]  # (batch, num_states)
        alpha_action = attention_weights[:, -1:]  # (batch, 1)
        
        # Compute hidden representation (Eq. 10)
        h_j = torch.sum(alpha_states.unsqueeze(-1) * c_states, dim=1) + alpha_action * c_a
        
        # Decode to distribution parameters (Eq. 11)
        if self.output_type == 'continuous':
            mean = self.decoder_mean(h_j)
            logvar = self.decoder_logvar(h_j)
            return mean, logvar, attention_weights
        else:
            logits = self.decoder_logits(h_j)
            return logits, attention_weights

class CausalWorldModel(nn.Module):
    """
    Complete causal world model with inference networks
    """
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.L = config.NUM_UES
        self.hidden_dim = config.HIDDEN_DIM
        
        # Variable encoders (shared across all inference networks)
        self.buffer_encoder = VariableEncoder(1, self.hidden_dim)
        self.action_encoder = VariableEncoder(3, self.hidden_dim)  # One-hot encoded
        self.grant_encoder = VariableEncoder(1, self.hidden_dim)
        self.collision_encoder = VariableEncoder(1, self.hidden_dim)
        
        # Shared key vectors for state variables
        self.register_parameter('keys_buffer', 
                              nn.Parameter(torch.randn(self.L, self.hidden_dim)))
        self.register_parameter('key_collision', 
                              nn.Parameter(torch.randn(1, self.hidden_dim)))
        
        # Inference networks for predicting next states
        self.buffer_predictors = nn.ModuleList([
            InferenceNetwork(config, output_type='continuous') 
            for _ in range(self.L)
        ])
        
        self.collision_predictor = InferenceNetwork(config, output_type='discrete')
        
    def encode_variables(self, buffers: torch.Tensor, actions: torch.Tensor, 
                        grants: torch.Tensor, collision: torch.Tensor) -> Dict:
        """
        Encode all variables
        
        Args:
            buffers: (batch, L)
            actions: (batch, L) - integer actions
            grants: (batch, L) - binary grants
            collision: (batch, 1)
        """
        batch_size = buffers.shape[0]
        
        # Encode buffers
        buffer_encodings = []
        for u in range(self.L):
            enc = self.buffer_encoder(buffers[:, u:u+1])  # (batch, hidden_dim)
            buffer_encodings.append(enc)
        buffer_encodings = torch.stack(buffer_encodings, dim=1)  # (batch, L, hidden_dim)
        
        # Encode actions (one-hot)
        actions_onehot = F.one_hot(actions.long(), num_classes=3).float()  # (batch, L, 3)
        action_encodings = []
        for u in range(self.L):
            enc = self.action_encoder(actions_onehot[:, u, :])
            action_encodings.append(enc)
        action_encodings = torch.stack(action_encodings, dim=1)  # (batch, L, hidden_dim)
        
        # Encode grants
        grant_encodings = self.grant_encoder(grants.unsqueeze(-1))  # (batch, L, hidden_dim)
        
        # Encode collision
        collision_encoding = self.collision_encoder(collision)  # (batch, hidden_dim)
        
        return {
            'buffers': buffer_encodings,
            'actions': action_encodings,
            'grants': grant_encodings,
            'collision': collision_encoding
        }
    
    def predict_next_buffer(self, ue_id: int, encodings: Dict, keys: torch.Tensor) -> Tuple:
        """
        Predict next buffer state for UE ue_id
        
        Returns:
            mean, logvar, attention_weights
        """
        # Gather parent variables for buffer prediction
        # Parents: current buffer, action, grant, other UE actions (for collision)
        state_enc = encodings['buffers'][:, ue_id:ue_id+1, :]  # (batch, 1, hidden_dim)
        action_enc = encodings['actions'][:, ue_id:ue_id+1, :]  # (batch, 1, hidden_dim)
        
        # Use inference network
        mean, logvar, attention = self.buffer_predictors[ue_id](
            state_enc, action_enc, keys[:1, :]
        )
        
        return mean, logvar, attention
    
    def predict_collision(self, encodings: Dict) -> Tuple:
        """
        Predict collision given all UE actions
        
        Returns:
            logits, attention_weights
        """
        # All actions are parents of collision
        action_enc = encodings['actions']  # (batch, L, hidden_dim)
        
        # Use dummy state (collision has no state parents, only actions)
        dummy_state = torch.zeros(action_enc.shape[0], 1, self.hidden_dim, 
                                 device=action_enc.device)
        dummy_keys = torch.zeros(1, self.hidden_dim, device=action_enc.device)
        
        logits, attention = self.collision_predictor(dummy_state, action_enc, dummy_keys)
        
        return logits, attention
    
    def forward(self, buffers: torch.Tensor, actions: torch.Tensor, 
               grants: torch.Tensor, collision: torch.Tensor) -> Dict:
        """
        Full forward pass predicting all next-step variables
        
        Returns dict with predictions and attention weights
        """
        # Encode all variables
        encodings = self.encode_variables(buffers, actions, grants, collision)
        
        # Predict next buffers for all UEs
        buffer_predictions = []
        buffer_attentions = []
        for u in range(self.L):
            mean, logvar, attention = self.predict_next_buffer(
                u, encodings, self.keys_buffer
            )
            buffer_predictions.append((mean, logvar))
            buffer_attentions.append(attention)
        
        # Predict collision
        collision_logits, collision_attention = self.predict_collision(encodings)
        
        return {
            'buffer_predictions': buffer_predictions,
            'buffer_attentions': buffer_attentions,
            'collision_logits': collision_logits,
            'collision_attention': collision_attention
        }
    
    def compute_loss(self, predictions: Dict, targets: Dict) -> torch.Tensor:
        """
        Compute negative log-likelihood loss (Eq. 12)
        
        Args:
            predictions: Output from forward()
            targets: Dict with 'next_buffers' and 'collision'
        """
        loss = 0.0
        
        # Buffer prediction loss (Gaussian NLL)
        for u in range(self.L):
            mean, logvar = predictions['buffer_predictions'][u]
            target = targets['next_buffers'][:, u:u+1]
            
            # Negative log-likelihood of Gaussian
            nll = 0.5 * (logvar + ((target - mean) ** 2) / torch.exp(logvar))
            loss += nll.mean()
        
        # Collision prediction loss (Cross-entropy)
        collision_logits = predictions['collision_logits']
        collision_target = targets['collision'].long().squeeze()
        
        ce_loss = F.cross_entropy(collision_logits, collision_target)
        loss += ce_loss
        
        return loss

def train_world_model(model: CausalWorldModel, replay_buffer: List, 
                     config: Config, num_iterations: int = 100):
    """
    Train the causal world model (maximize log-likelihood, Eq. 12)
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE_PPO)
    
    for iteration in range(num_iterations):
        # Sample mini-batch
        indices = np.random.choice(len(replay_buffer), 
                                  size=min(config.MINI_BATCH_SIZE, len(replay_buffer)), 
                                  replace=False)
        
        batch = [replay_buffer[i] for i in indices]
        
        # Prepare batch tensors
        buffers = torch.tensor([t['buffers'] for t in batch], dtype=torch.float32)
        actions = torch.tensor([t['actions'] for t in batch], dtype=torch.long)
        grants = torch.tensor([t['grants'] for t in batch], dtype=torch.float32)
        collision = torch.tensor([t['collision'] for t in batch], dtype=torch.float32).unsqueeze(-1)
        next_buffers = torch.tensor([t['next_buffers'] for t in batch], dtype=torch.float32)
        next_collision = torch.tensor([t['next_collision'] for t in batch], dtype=torch.long)
        
        # Forward pass
        predictions = model(buffers, actions, grants, collision)
        
        # Compute loss
        targets = {'next_buffers': next_buffers, 'collision': next_collision}
        loss = model.compute_loss(predictions, targets)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if iteration % 10 == 0:
            print(f"  World model training iteration {iteration}/{num_iterations}, Loss: {loss.item():.4f}")
