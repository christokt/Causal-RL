"""
Utility functions
"""

import numpy as np
import torch
import pickle
import os

def save_checkpoint(agents, causal_model, episode, filepath):
    """Save training checkpoint"""
    checkpoint = {
        'episode': episode,
        'causal_model_state_dict': causal_model.state_dict(),
        'ue_agents': []
    }
    
    for agent in agents:
        if hasattr(agent, 'Q'):  # Q-learning agent
            checkpoint['ue_agents'].append({
                'type': 'qlearning',
                'Q': agent.Q,
                'epsilon': agent.epsilon
            })
        else:  # PPO agent
            checkpoint['ue_agents'].append({
                'type': 'ppo',
                'policy_state_dict': agent.policy.state_dict(),
                'value_state_dict': agent.value.state_dict()
            })
    
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")

def load_checkpoint(filepath, agents, causal_model):
    """Load training checkpoint"""
    checkpoint = torch.load(filepath)
    
    causal_model.load_state_dict(checkpoint['causal_model_state_dict'])
    
    for i, agent_data in enumerate(checkpoint['ue_agents']):
        if agent_data['type'] == 'qlearning':
            agents[i].Q = agent_data['Q']
            agents[i].epsilon = agent_data['epsilon']
        else:
            agents[i].policy.load_state_dict(agent_data['policy_state_dict'])
            agents[i].value.load_state_dict(agent_data['value_state_dict'])
    
    print(f"Checkpoint loaded from {filepath}")
    return checkpoint['episode']

def set_seed(seed):
    """Set random seeds for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def evaluate_policy(env, ue_agents, bs_agent, num_episodes=100, deterministic=True):
    """
    Evaluate trained policy
    """
    episode_lengths = []
    episode_goodputs = []
    episode_plrs = []
    episode_collision_rates = []
    
    for ep in range(num_episodes):
        observations = env.reset()
        done = False
        step = 0
        
        while not done:
            # UEs select actions
            ue_actions = []
            for u in range(len(ue_agents)):
                state = env.get_ue_state(u)
                
                if hasattr(ue_agents[u], 'Q'):  # Q-learning (SafeUEAgent)
                    last_collision = env.collision_history[-1] if len(env.collision_history) > 0 else False
                    # ✅ Pass extra parameters for SafeUEAgent
                    action = ue_agents[u].select_action(
                        state, 
                        env.buffers[u],
                        env.last_actions[u],
                        last_collision,
                        explore=not deterministic
                    )
                else:  # PPO
                    action, _, _ = ue_agents[u].select_action(state, deterministic)
                
                ue_actions.append(action)
            
            ue_actions = np.array(ue_actions)
            
            # BS selects grant
            bs_state_dict = {'buffers': env.buffers.copy()}
            bs_grant = bs_agent.select_grant(bs_state_dict)
            
            # Execute
            observations, reward, done, info = env.step(ue_actions, bs_grant)
            step += 1
            
            if step >= env.config.MAX_EPISODE_LENGTH:
                break
        
        # Compute metrics
        episode_lengths.append(step)
        
        goodput = np.sum(env.completed) / (step * env.config.SLOT_DURATION)
        episode_goodputs.append(goodput)
        
        plr = np.sum(env.losses) / (env.config.NUM_UES * env.config.PACKETS_PER_UE)
        episode_plrs.append(plr)
        
        collision_rate = np.mean(env.collision_history) if len(env.collision_history) > 0 else 0
        episode_collision_rates.append(collision_rate)
    
    return {
        'mean_episode_length': np.mean(episode_lengths),
        'std_episode_length': np.std(episode_lengths),
        'mean_goodput': np.mean(episode_goodputs),
        'std_goodput': np.std(episode_goodputs),
        'mean_plr': np.mean(episode_plrs),
        'std_plr': np.std(episode_plrs),
        'mean_collision_rate': np.mean(episode_collision_rates),
        'std_collision_rate': np.std(episode_collision_rates)
    }

def visualize_causal_graph(causal_model, config, save_path='causal_graph.png'):
    """
    Visualize learned causal graph with attention weights
    
    Args:
        causal_model: Trained causal world model
        config: Configuration
        save_path: Path to save visualization
    """
    try:
        import networkx as nx
        import matplotlib.pyplot as plt
    except ImportError:
        print("NetworkX or Matplotlib not available for graph visualization")
        return
    
    # Create directed graph
    G = nx.DiGraph()
    
    # Add nodes
    for u in range(config.NUM_UES):
        G.add_node(f'B_{u}', node_type='state')  # Buffer
        G.add_node(f'a_{u}', node_type='action')  # Action
        G.add_node(f'm_{u}', node_type='grant')  # Grant
    
    G.add_node('collision', node_type='outcome')
    G.add_node('reward', node_type='reward')
    
    # Add temporal edges (example structure)
    for u in range(config.NUM_UES):
        # State to next state
        G.add_edge(f'B_{u}', f'B_{u}\'', weight=0.8)
        # Action to next state
        G.add_edge(f'a_{u}', f'B_{u}\'', weight=0.6)
        # Grant to next state
        G.add_edge(f'm_{u}', f'B_{u}\'', weight=0.5)
        # Action to collision
        G.add_edge(f'a_{u}', 'collision', weight=0.7)
    
    # Collision to reward
    G.add_edge('collision', 'reward', weight=0.9)
    
    # Layout
    pos = nx.spring_layout(G, k=2, iterations=50)
    
    # Draw
    plt.figure(figsize=(14, 10))
    
    # Node colors by type
    node_colors = []
    for node in G.nodes():
        if 'B_' in node:
            node_colors.append('lightblue')
        elif 'a_' in node:
            node_colors.append('lightgreen')
        elif 'm_' in node:
            node_colors.append('yellow')
        elif node == 'collision':
            node_colors.append('red')
        elif node == 'reward':
            node_colors.append('gold')
        else:
            node_colors.append('gray')
    
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=800, alpha=0.9)
    nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold')
    
    # Draw edges with varying thickness based on weights
    edges = G.edges()
    weights = [G[u][v]['weight'] for u, v in edges]
    nx.draw_networkx_edges(G, pos, width=[w*3 for w in weights], 
                          alpha=0.6, arrows=True, arrowsize=15)
    
    plt.title('Learned Causal Graph Structure', fontsize=16, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Causal graph saved to {save_path}")

def print_attention_weights(causal_model, sample_data, config):
    """
    Print attention weights from causal model to understand causal relationships
    
    Args:
        causal_model: Trained causal world model
        sample_data: Sample data point (dict with buffers, actions, etc.)
        config: Configuration
    """
    causal_model.eval()
    
    with torch.no_grad():
        buffers = torch.FloatTensor(sample_data['buffers']).unsqueeze(0)
        actions = torch.LongTensor(sample_data['actions']).unsqueeze(0)
        grants = torch.FloatTensor(sample_data['grants']).unsqueeze(0)
        collision = torch.FloatTensor([[sample_data['collision']]])
        
        predictions = causal_model(buffers, actions, grants, collision)
        
        print("\n" + "="*60)
        print("CAUSAL ATTENTION WEIGHTS")
        print("="*60)
        
        # Buffer predictions
        for u in range(config.NUM_UES):
            attention = predictions['buffer_attentions'][u]
            print(f"\nUE {u} Next Buffer Prediction:")
            print(f"  Attention on current buffer: {attention[0, 0].item():.3f}")
            print(f"  Attention on action: {attention[0, -1].item():.3f}")
        
        # Collision prediction
        collision_attention = predictions['collision_attention']
        print(f"\nCollision Prediction:")
        print(f"  Attention distribution over UE actions:")
        for u in range(config.NUM_UES):
            if u < collision_attention.shape[1]:
                print(f"    UE {u}: {collision_attention[0, u].item():.3f}")

def compute_reward_difference_explanation(ue_agent, state, action1, action2, config):
    """
    Compute Reward Difference Explanation (RDX) for explainability
    
    Args:
        ue_agent: UE agent (must be PPO agent with decomposed Q-values)
        state: Current state
        action1: First action to compare
        action2: Second action to compare
        config: Configuration
    
    Returns:
        Dictionary with RDX components
    """
    # This is a simplified version - full implementation requires
    # decomposed Q-functions for each reward component
    
    if hasattr(ue_agent, 'policy'):  # PPO agent
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        with torch.no_grad():
            logits = ue_agent.policy(state_tensor)
            probs = torch.softmax(logits, dim=-1)
            
            prob_a1 = probs[0, action1].item()
            prob_a2 = probs[0, action2].item()
        
        explanation = {
            'action1': action1,
            'action2': action2,
            'prob_a1': prob_a1,
            'prob_a2': prob_a2,
            'preference': 'action1' if prob_a1 > prob_a2 else 'action2',
            'preference_strength': abs(prob_a1 - prob_a2)
        }
        
        return explanation
    else:
        # Q-learning agent
        state_idx = ue_agent.discretize_state(state)
        q1 = ue_agent.Q[state_idx][action1]
        q2 = ue_agent.Q[state_idx][action2]
        
        explanation = {
            'action1': action1,
            'action2': action2,
            'q_value_a1': q1,
            'q_value_a2': q2,
            'preference': 'action1' if q1 > q2 else 'action2',
            'preference_strength': abs(q1 - q2)
        }
        
        return explanation

class ExperimentLogger:
    """Logger for tracking experiments"""
    def __init__(self, log_dir='logs'):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.logs = []
    
    def log(self, episode, metrics):
        """Log metrics for an episode"""
        log_entry = {'episode': episode, **metrics}
        self.logs.append(log_entry)
    
    def save(self, filename='experiment_log.pkl'):
        """Save logs to file"""
        filepath = os.path.join(self.log_dir, filename)
        with open(filepath, 'wb') as f:
            pickle.dump(self.logs, f)
        print(f"Logs saved to {filepath}")
    
    def load(self, filename='experiment_log.pkl'):
        """Load logs from file"""
        filepath = os.path.join(self.log_dir, filename)
        with open(filepath, 'rb') as f:
            self.logs = pickle.load(f)
        print(f"Logs loaded from {filepath}")
        return self.logs

def compare_with_baselines(env, trained_agents, bs_agent, config, num_episodes=100):
    """
    Compare trained agents with baseline methods
    
    Baselines:
    1. Random action
    2. Always transmit (aggressive)
    3. Round-robin with grant
    """
    print("\n" + "="*60)
    print("BASELINE COMPARISON")
    print("="*60)
    
    # Evaluate trained agents
    print("\nEvaluating trained Causal MARL agents...")
    trained_results = evaluate_policy(env, trained_agents, bs_agent, num_episodes)
    
    # Baseline 1: Random actions
    print("\nEvaluating Random baseline...")
    random_agents = [type('obj', (object,), {
        'select_action': lambda self, s, b, explore=True: np.random.randint(0, 3) if b > 0 else 0
    })() for _ in range(config.NUM_UES)]
    random_results = evaluate_policy(env, random_agents, bs_agent, num_episodes)
    
    # Baseline 2: Always transmit (when buffer not empty)
    print("\nEvaluating Always-Transmit baseline...")
    transmit_agents = [type('obj', (object,), {
        'select_action': lambda self, s, b, explore=True: 1 if b > 0 else 0
    })() for _ in range(config.NUM_UES)]
    transmit_results = evaluate_policy(env, transmit_agents, bs_agent, num_episodes)
    
    # Print comparison
    print("\n" + "-"*60)
    print(f"{'Method':<25} {'Goodput':>12} {'PLR':>8} {'Collision':>12} {'Ep Length':>12}")
    print("-"*60)
    
    print(f"{'Causal MARL':<25} "
          f"{trained_results['mean_goodput']:>12.2f} "
          f"{trained_results['mean_plr']:>8.4f} "
          f"{trained_results['mean_collision_rate']:>12.4f} "
          f"{trained_results['mean_episode_length']:>12.1f}")
    
    print(f"{'Random':<25} "
          f"{random_results['mean_goodput']:>12.2f} "
          f"{random_results['mean_plr']:>8.4f} "
          f"{random_results['mean_collision_rate']:>12.4f} "
          f"{random_results['mean_episode_length']:>12.1f}")
    
    print(f"{'Always Transmit':<25} "
          f"{transmit_results['mean_goodput']:>12.2f} "
          f"{transmit_results['mean_plr']:>8.4f} "
          f"{transmit_results['mean_collision_rate']:>12.4f} "
          f"{transmit_results['mean_episode_length']:>12.1f}")
    
    print("-"*60)
    
    # Calculate improvements
    goodput_improvement = ((trained_results['mean_goodput'] - random_results['mean_goodput']) / 
                          random_results['mean_goodput'] * 100)
    
    print(f"\nGoodput improvement over Random: {goodput_improvement:.1f}%")
    
    return {
        'trained': trained_results,
        'random': random_results,
        'transmit': transmit_results
    }
