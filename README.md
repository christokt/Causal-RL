# Causality Aware Wireless Scheduling via Explainable Multi-Agent Reinforcement Learning

Implementation of the paper "Causality Aware Wireless Scheduling via Explainable Multi-Agent Reinforcement Learning"

## Overview

This codebase implements a causal world model-based multi-agent reinforcement learning approach for wireless uplink scheduling with collision dynamics.

### Key Features

- **Causal World Model**: Learns structural causal models using inference networks with attention mechanisms
- **Multi-Agent Learning**: Supports both Q-learning and PPO for decentralized UE agents
- **Explainability**: Provides reward difference explanations (RDX) and attention-based causal chain analysis
- **Collision Dynamics**: Models realistic wireless channel with collision detection
- **DELETE Action**: Implements optimistic packet deletion with loss tracking

## Installation
```bash
pip install -r requirements.txt
