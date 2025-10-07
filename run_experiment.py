"""
Example script to run complete experiments
"""
from safe_agents import SafeUEAgent, BSAgent  # ✅ Changed from: from agents import UEAgent, BSAgent

import numpy as np
import torch
from config import Config
from train import train_causal_marl
from utils import (set_seed, save_checkpoint, evaluate_policy, 
                  compare_with_baselines, ExperimentLogger,
                  visualize_causal_graph, print_attention_weights)
from environment import WirelessSchedulingEnv

def main():
    """Run complete experiment"""
    
    # Set random seed for reproducibility
    set_seed(42)
    
    # Create configuration
    config = Config()
    
    # Modify config for experiment if needed
    config.NUM_EPISODES = 1000  # Shorter for testing
    config.NUM_UES = 8
    config.PACKETS_PER_UE = 50
    
    print("="*70)
    print(" "*15 + "CAUSAL MARL FOR WIRELESS SCHEDULING")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Number of UEs: {config.NUM_UES}")
    print(f"  Packets per UE: {config.PACKETS_PER_UE}")
    print(f"  Total packets to transmit: {config.NUM_UES * config.PACKETS_PER_UE}")
    print(f"  Training episodes: {config.NUM_EPISODES}")
    print(f"  Causal graph update frequency: every {config.N_GRAPH_UPDATE} episodes")
    print(f"  Device: {config.DEVICE}")
    print("-"*70)
    
    # Initialize logger
    logger = ExperimentLogger(log_dir='logs')
    
    # ========================================
    # EXPERIMENT 1: Train with Q-Learning
    # ========================================
    print("\n" + "="*70)
    print("EXPERIMENT 1: Q-LEARNING WITH CAUSAL WORLD MODEL")
    print("="*70)
    
    results_qlearn = train_causal_marl(config, use_ppo=False)
    
    # Save checkpoint
    save_checkpoint(
        results_qlearn['ue_agents'], 
        results_qlearn['causal_model'],
        config.NUM_EPISODES,
        'checkpoints/qlearning_final.pt'
    )
    
    # Evaluate
    env = WirelessSchedulingEnv(config)
    qlearn_eval = evaluate_policy(
        env, 
        results_qlearn['ue_agents'], 
        results_qlearn['bs_agent'],
        num_episodes=100,
        deterministic=True
    )
    
    print("\n" + "-"*70)
    print("Q-Learning Final Evaluation (100 episodes):")
    print(f"  Episode Length: {qlearn_eval['mean_episode_length']:.1f} ± {qlearn_eval['std_episode_length']:.1f}")
    print(f"  Goodput: {qlearn_eval['mean_goodput']:.2f} ± {qlearn_eval['std_goodput']:.2f} pkts/sec")
    print(f"  PLR: {qlearn_eval['mean_plr']:.4f} ± {qlearn_eval['std_plr']:.4f}")
    print(f"  Collision Rate: {qlearn_eval['mean_collision_rate']:.4f} ± {qlearn_eval['std_collision_rate']:.4f}")
    
    # ========================================
    # EXPERIMENT 2: Train with PPO
    # ========================================
    print("\n" + "="*70)
    print("EXPERIMENT 2: PPO WITH CAUSAL WORLD MODEL")
    print("="*70)
    
    # Reset seed for fair comparison
    set_seed(42)
    
    results_ppo = train_causal_marl(config, use_ppo=True)
    
    # Save checkpoint
    save_checkpoint(
        results_ppo['ue_agents'], 
        results_ppo['causal_model'],
        config.NUM_EPISODES,
        'checkpoints/ppo_final.pt'
    )
    
    # Evaluate
    ppo_eval = evaluate_policy(
        env, 
        results_ppo['ue_agents'], 
        results_ppo['bs_agent'],
        num_episodes=100,
        deterministic=True
    )
    
    print("\n" + "-"*70)
    print("PPO Final Evaluation (100 episodes):")
    print(f"  Episode Length: {ppo_eval['mean_episode_length']:.1f} ± {ppo_eval['std_episode_length']:.1f}")
    print(f"  Goodput: {ppo_eval['mean_goodput']:.2f} ± {ppo_eval['std_goodput']:.2f} pkts/sec")
    print(f"  PLR: {ppo_eval['mean_plr']:.4f} ± {ppo_eval['std_plr']:.4f}")
    print(f"  Collision Rate: {ppo_eval['mean_collision_rate']:.4f} ± {ppo_eval['std_collision_rate']:.4f}")
    
    # ========================================
    # EXPERIMENT 3: Compare with Baselines
    # ========================================
    print("\n" + "="*70)
    print("EXPERIMENT 3: BASELINE COMPARISON")
    print("="*70)
    
    baseline_comparison = compare_with_baselines(
        env,
        results_ppo['ue_agents'],  # Use PPO as it typically performs better
        results_ppo['bs_agent'],
        config,
        num_episodes=100
    )
    
    # ========================================
    # EXPERIMENT 4: Causal Explainability
    # ========================================
    print("\n" + "="*70)
    print("EXPERIMENT 4: CAUSAL EXPLAINABILITY ANALYSIS")
    print("="*70)
    
    # Visualize learned causal graph
    visualize_causal_graph(results_ppo['causal_model'], config, 
                          save_path='results/causal_graph.png')
    
    # Sample data for attention analysis
    env.reset()
    sample_transition = {
        'buffers': env.buffers.copy(),
        'actions': np.array([1, 0, 1, 0, 1, 0, 0, 1]),  # Mixed actions
        'grants': np.array([1, 0, 0, 0, 0, 0, 0, 0]),   # Grant to UE 0
        'collision': 1.0  # Collision occurred
    }
    
    print_attention_weights(results_ppo['causal_model'], sample_transition, config)
    
    # ========================================
    # FINAL SUMMARY
    # ========================================
    print("\n" + "="*70)
    print(" "*25 + "FINAL SUMMARY")
    print("="*70)
    
    print("\nPerformance Comparison:")
    print(f"\n{'Metric':<30} {'Q-Learning':>15} {'PPO':>15} {'Best':>15}")
    print("-"*75)
    
    metrics = [
        ('Episode Length', qlearn_eval['mean_episode_length'], ppo_eval['mean_episode_length'], 'lower'),
        ('Goodput (pkts/sec)', qlearn_eval['mean_goodput'], ppo_eval['mean_goodput'], 'higher'),
        ('PLR', qlearn_eval['mean_plr'], ppo_eval['mean_plr'], 'lower'),
        ('Collision Rate', qlearn_eval['mean_collision_rate'], ppo_eval['mean_collision_rate'], 'lower')
    ]
    
    for metric_name, qlearn_val, ppo_val, criterion in metrics:
        if criterion == 'lower':
            best = 'Q-Learning' if qlearn_val < ppo_val else 'PPO'
        else:
            best = 'Q-Learning' if qlearn_val > ppo_val else 'PPO'
        
        print(f"{metric_name:<30} {qlearn_val:>15.3f} {ppo_val:>15.3f} {best:>15}")
    
    print("\n" + "="*70)
    print("Experiments completed successfully!")
    print("Results saved in:")
    print("  - training_results_qlearn.png")
    print("  - training_results_ppo.png")
    print("  - results/causal_graph.png")
    print("  - checkpoints/qlearning_final.pt")
    print("  - checkpoints/ppo_final.pt")
    print("="*70)

if __name__ == "__main__":
    # Create necessary directories
    import os
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # Run experiments
    main()
