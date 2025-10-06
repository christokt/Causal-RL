"""
Main training loop with causal world model
"""

import numpy as np
import torch
from typing import List, Dict
import matplotlib.pyplot as plt

from config import Config
from environment import WirelessSchedulingEnv
from causal_model import CausalWorldModel, train_world_model
from agents import UEAgent, BSAgent
from ppo import PPOAgent

class ReplayBuffer:
    """Replay buffer for storing transitions"""
    def __init__(self, max_size: int):
        self.max_size = max_size
        self.buffer = []
    
    def add(self, transition: Dict):
        if len(self.buffer) >= self.max_size:
            self.buffer.pop(0)
        self.buffer.append(transition)
    
    def __len__(self):
        return len(self.buffer)
    
    def __getitem__(self, idx):
        return self.buffer[idx]

def model_rollout(env: WirelessSchedulingEnv, ue_agents: List, bs_agent, 
                 causal_model: CausalWorldModel, k: int, config: Config) -> List[Dict]:
    """
    Perform k-step model rollout (Algorithm 2)
    
    Args:
        env: Environment (for starting state)
        ue_agents: List of UE agents
        bs_agent: BS agent
        causal_model: Trained causal world model
        k: Number of rollout steps
    
    Returns:
        simulated_transitions: List of simulated transitions
    """
    simulated_transitions = []
    
    # Get current state from environment
    current_buffers = env.buffers.copy()
    current_actions = env.last_actions.copy()
    current_grant = env.last_grant
    
    for tau in range(k):
        # Sample actions from current policy
        ue_actions = []
        for u, agent in enumerate(ue_agents):
            state = env.get_ue_state(u)
            if isinstance(agent, PPOAgent):
                action, _, _ = agent.select_action(state)
            else:
                action = agent.select_action(state, current_buffers[u])
            ue_actions.append(action)
        
        ue_actions = np.array(ue_actions)
        
        # BS selects grant
        bs_state = env.get_bs_state()
        bs_grant = bs_agent.select_grant({'buffers': current_buffers})
        
        # Use causal model to predict next state
        with torch.no_grad():
            buffers_tensor = torch.FloatTensor(current_buffers).unsqueeze(0)
            actions_tensor = torch.LongTensor(ue_actions).unsqueeze(0)
            grants_tensor = torch.zeros(1, config.NUM_UES)
            if bs_grant > 0:
                grants_tensor[0, bs_grant - 1] = 1
            collision_tensor = torch.FloatTensor([[0]])  # Dummy
            
            predictions = causal_model(buffers_tensor, actions_tensor, 
                                      grants_tensor, collision_tensor)
            
            # Extract predicted next buffers
            next_buffers = []
            for u in range(config.NUM_UES):
                mean, logvar = predictions['buffer_predictions'][u]
                next_buffer = mean.item()
                next_buffers.append(max(0, min(config.PACKETS_PER_UE, next_buffer)))
            
            next_buffers = np.array(next_buffers)
            
            # Predict collision
            collision_logits = predictions['collision_logits']
            collision_prob = torch.softmax(collision_logits, dim=-1)[0, 1].item()
            predicted_collision = (collision_prob > 0.5)
        
        # Compute reward using known reward function
        # Determine success
        num_transmitters = np.sum(ue_actions == 1)
        collision = (num_transmitters >= 2)
        success = np.zeros(config.NUM_UES, dtype=bool)
        if num_transmitters == 1:
            successful_ue = np.where(ue_actions == 1)[0][0]
            success[successful_ue] = True
        
        # Compute losses (simplified for rollout)
        losses = np.zeros(config.NUM_UES, dtype=bool)
        
        # Compute reward
        r_goodput = config.ALPHA_GOODPUT * np.sum(success)
        r_collision = -config.ALPHA_COLLISION * float(collision)
        r_loss = -config.ALPHA_LOSS * np.sum(losses)
        num_idle = np.sum(ue_actions == 0)
        r_efficiency = config.ALPHA_EFFICIENCY * (config.NUM_UES - num_idle)
        
        reward = (config.W_GOODPUT * r_goodput + 
                 config.W_COLLISION * r_collision + 
                 config.W_LOSS * r_loss + 
                 config.W_EFFICIENCY * r_efficiency)
        
        # Store simulated transition
        transition = {
            'buffers': current_buffers.copy(),
            'actions': ue_actions.copy(),
            'grant': bs_grant,
            'reward': reward,
            'next_buffers': next_buffers.copy(),
            'collision': collision
        }
        simulated_transitions.append(transition)
        
        # Update state for next iteration
        current_buffers = next_buffers
        current_actions = ue_actions
        current_grant = bs_grant
    
    return simulated_transitions

def train_causal_marl(config: Config, use_ppo: bool = False):
    """
    Main training loop (Algorithm 1 and Algorithm 2)
    
    Args:
        config: Configuration object
        use_ppo: If True, use PPO; if False, use Q-learning
    """
    # Initialize environment
    env = WirelessSchedulingEnv(config)
    
    # Initialize agents
    if use_ppo:
        ue_agents = [PPOAgent(state_dim=5, action_dim=3, config=config) 
                    for _ in range(config.NUM_UES)]
    else:
        ue_agents = [UEAgent(u, config, state_dim=5) 
                    for u in range(config.NUM_UES)]
    
    bs_agent = BSAgent(config, config.NUM_UES)
    
    # Initialize causal world model
    causal_model = CausalWorldModel(config).to(config.DEVICE)
    
    # Initialize replay buffer
    replay_buffer = ReplayBuffer(config.BUFFER_SIZE)
    
    # Training statistics
    episode_lengths = []
    episode_rewards = []
    episode_goodputs = []
    episode_plrs = []
    episode_collisions = []
    
    print("Starting training...")
    print(f"Method: {'PPO' if use_ppo else 'Q-Learning'}")
    print(f"Episodes: {config.NUM_EPISODES}")
    print(f"UEs: {config.NUM_UES}, Packets/UE: {config.PACKETS_PER_UE}")
    print("-" * 60)
    
    # Main training loop (Algorithm 1, line 2)
    for episode in range(config.NUM_EPISODES):
        # Reset environment
        observations = env.reset()
        
        episode_reward = 0
        episode_transitions = []
        
        # Episode trajectory storage for PPO
        ue_trajectories = [{
            'states': [], 'actions': [], 'rewards': [], 
            'dones': [], 'log_probs': [], 'values': []
        } for _ in range(config.NUM_UES)]
        
        done = False
        step = 0
        
        # Collect trajectory (Algorithm 1, line 3)
        while not done:
            # Each UE selects action
            ue_actions = []
            ue_states = []
            ue_log_probs = []
            ue_values = []
            
            for u in range(config.NUM_UES):
                state = env.get_ue_state(u)
                ue_states.append(state)
                
                if use_ppo:
                    action, value, probs = ue_agents[u].select_action(state)
                    log_prob = np.log(probs[action] + 1e-8)
                    ue_log_probs.append(log_prob)
                    ue_values.append(value)
                else:
                    action = ue_agents[u].select_action(state, env.buffers[u])
                
                ue_actions.append(action)
            
            ue_actions = np.array(ue_actions)
            
            # BS selects grant
            bs_state_dict = {'buffers': env.buffers.copy()}
            bs_grant = bs_agent.select_grant(bs_state_dict, causal_model)
            
            # Execute actions in environment
            next_observations, reward, done, info = env.step(ue_actions, bs_grant)
            
            episode_reward += reward
            
            # Store transition in replay buffer for world model training
            transition = {
                'buffers': info['buffers'] - (info['success'].astype(int) + 
                          (ue_actions == 2).astype(int)),  # Previous buffers
                'actions': ue_actions.copy(),
                'grants': np.array([1 if bs_grant == (u+1) else 0 
                                  for u in range(config.NUM_UES)]),
                'collision': float(info['collision']),
                'next_buffers': info['buffers'].copy(),
                'next_collision': float(info['collision']),
                'reward': reward
            }
            replay_buffer.add(transition)
            episode_transitions.append(transition)
            
            # Store for PPO update
            if use_ppo:
                for u in range(config.NUM_UES):
                    ue_trajectories[u]['states'].append(ue_states[u])
                    ue_trajectories[u]['actions'].append(ue_actions[u])
                    ue_trajectories[u]['rewards'].append(reward / config.NUM_UES)  # Shared reward
                    ue_trajectories[u]['dones'].append(done)
                    ue_trajectories[u]['log_probs'].append(ue_log_probs[u])
                    ue_trajectories[u]['values'].append(ue_values[u])
            else:
                # Q-learning update
                for u in range(config.NUM_UES):
                    next_state = env.get_ue_state(u)
                    ue_agents[u].update(ue_states[u], ue_actions[u], 
                                       reward / config.NUM_UES, next_state)
            
            step += 1
            
            if step >= config.MAX_EPISODE_LENGTH:
                break
        
        # Episode statistics
        episode_lengths.append(step)
        episode_rewards.append(episode_reward)
        
        goodput = np.sum(env.completed) / (step * config.SLOT_DURATION)
        episode_goodputs.append(goodput)
        
        plr = np.sum(env.losses) / (config.NUM_UES * config.PACKETS_PER_UE)
        episode_plrs.append(plr)
        
        collision_rate = np.mean(env.collision_history) if len(env.collision_history) > 0 else 0
        episode_collisions.append(collision_rate)
        
        # Update causal world model periodically (Algorithm 1, line 4-7)
        if episode > 0 and episode % config.N_GRAPH_UPDATE == 0 and len(replay_buffer) > config.MINI_BATCH_SIZE:
            print(f"\n  Updating causal world model at episode {episode}...")
            train_world_model(causal_model, replay_buffer.buffer, config)
        
        # PPO policy update (Algorithm 1, line 8-11)
        if use_ppo and episode % 10 == 0 and episode > 0:
            print(f"  Updating PPO policies at episode {episode}...")
            
            # Generate simulated data via k-step rollout (Algorithm 2)
            if episode >= config.N_GRAPH_UPDATE:  # Only after world model is trained
                simulated_transitions = model_rollout(
                    env, ue_agents, bs_agent, causal_model, 
                    config.K_STEP_ROLLOUT, config
                )
                # Add simulated transitions to trajectories (simplified)
            
            # Update each UE's policy
            for u in range(config.NUM_UES):
                if len(ue_trajectories[u]['states']) > 0:
                    ue_agents[u].update([ue_trajectories[u]])
        
        # Decay epsilon for Q-learning
        if not use_ppo:
            for agent in ue_agents:
                agent.decay_epsilon(episode)
        
        # Print progress
        if episode % 10 == 0:
            avg_length = np.mean(episode_lengths[-10:])
            avg_reward = np.mean(episode_rewards[-10:])
            avg_goodput = np.mean(episode_goodputs[-10:])
            avg_plr = np.mean(episode_plrs[-10:])
            avg_collision = np.mean(episode_collisions[-10:])
            
            print(f"Episode {episode}/{config.NUM_EPISODES}")
            print(f"  Avg Length: {avg_length:.1f}, Avg Reward: {avg_reward:.2f}")
            print(f"  Goodput: {avg_goodput:.2f} pkts/sec, PLR: {avg_plr:.3f}")
            print(f"  Collision Rate: {avg_collision:.3f}")
            if not use_ppo:
                print(f"  Epsilon: {ue_agents[0].epsilon:.3f}")
    
    print("\n" + "=" * 60)
    print("Training completed!")
    print("=" * 60)
    
    # Final statistics
    print(f"\nFinal Performance (last 100 episodes):")
    print(f"  Avg Episode Length: {np.mean(episode_lengths[-100:]):.1f} slots")
    print(f"  Avg Goodput: {np.mean(episode_goodputs[-100:]):.2f} packets/sec")
    print(f"  Avg PLR: {np.mean(episode_plrs[-100:]):.4f}")
    print(f"  Avg Collision Rate: {np.mean(episode_collisions[-100:]):.3f}")
    
    # Plot results
    plot_training_results(episode_lengths, episode_goodputs, episode_plrs, 
                         episode_collisions, use_ppo)
    
    return {
        'episode_lengths': episode_lengths,
        'episode_goodputs': episode_goodputs,
        'episode_plrs': episode_plrs,
        'episode_collisions': episode_collisions,
        'ue_agents': ue_agents,
        'bs_agent': bs_agent,
        'causal_model': causal_model
    }

def plot_training_results(episode_lengths, episode_goodputs, episode_plrs, 
                         episode_collisions, use_ppo):
    """Plot training curves"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    method_name = "Causal MARL (PPO)" if use_ppo else "Causal MARL (Q-Learning)"
    
    # Episode length
    axes[0, 0].plot(episode_lengths, alpha=0.3, label='Raw')
    axes[0, 0].plot(smooth(episode_lengths, 50), linewidth=2, label='Smoothed')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Episode Length (slots)')
    axes[0, 0].set_title('Episode Length vs Training')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Goodput
    axes[0, 1].plot(episode_goodputs, alpha=0.3, label='Raw')
    axes[0, 1].plot(smooth(episode_goodputs, 50), linewidth=2, label='Smoothed')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Goodput (packets/sec)')
    axes[0, 1].set_title('Goodput vs Training')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # PLR
    axes[1, 0].plot(episode_plrs, alpha=0.3, label='Raw')
    axes[1, 0].plot(smooth(episode_plrs, 50), linewidth=2, label='Smoothed')
    axes[1, 0].axhline(y=0.05, color='r', linestyle='--', label='Max PLR (5%)')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Packet Loss Rate')
    axes[1, 0].set_title('PLR vs Training')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Collision rate
    axes[1, 1].plot(episode_collisions, alpha=0.3, label='Raw')
    axes[1, 1].plot(smooth(episode_collisions, 50), linewidth=2, label='Smoothed')
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Collision Rate')
    axes[1, 1].set_title('Collision Rate vs Training')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle(f'{method_name} Training Results', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'training_results_{"ppo" if use_ppo else "qlearn"}.png', dpi=300)
    print(f"\nPlot saved as 'training_results_{'ppo' if use_ppo else 'qlearn'}.png'")

def smooth(data, window=50):
    """Smooth data using moving average"""
    if len(data) < window:
        return data
    smoothed = []
    for i in range(len(data)):
        start = max(0, i - window // 2)
        end = min(len(data), i + window // 2)
        smoothed.append(np.mean(data[start:end]))
    return smoothed

if __name__ == "__main__":
    # Create config
    config = Config()
    
    # Train with Q-learning
    print("\n" + "=" * 60)
    print("TRAINING WITH Q-LEARNING")
    print("=" * 60)
    results_qlearn = train_causal_marl(config, use_ppo=False)
    
    # Train with PPO
    print("\n\n" + "=" * 60)
    print("TRAINING WITH PPO")
    print("=" * 60)
    results_ppo = train_causal_marl(config, use_ppo=True)
    
    # Compare results
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)
    print(f"\nQ-Learning (last 100 episodes):")
    print(f"  Avg Episode Length: {np.mean(results_qlearn['episode_lengths'][-100:]):.1f}")
    print(f"  Avg Goodput: {np.mean(results_qlearn['episode_goodputs'][-100:]):.2f} pkts/sec")
    print(f"  Avg PLR: {np.mean(results_qlearn['episode_plrs'][-100:]):.4f}")
    
    print(f"\nPPO (last 100 episodes):")
    print(f"  Avg Episode Length: {np.mean(results_ppo['episode_lengths'][-100:]):.1f}")
    print(f"  Avg Goodput: {np.mean(results_ppo['episode_goodputs'][-100:]):.2f} pkts/sec")
    print(f"  Avg PLR: {np.mean(results_ppo['episode_plrs'][-100:]):.4f}")
