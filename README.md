# MHENST-DQN

A deep Q-learning framework combining cutting-edge reinforcement learning approaches. Integrates VAE, GNN, Transformer, and metacognition mechanisms to create an adaptive agent capable of handling multiple environment types.

## Features

- **Variational Autoencoder (VAE)**: Learns efficient low-dimensional representations of high-dimensional inputs
- **Graph Neural Network (GNN)**: Models relationships and knowledge graphs within environments
- **Transformer-based DQN**: Analyzes temporal context for sequential decision making
- **Metacognition System**: Self-optimizes through dynamic adjustment of learning and exploration rates
- **Multimodal Support**: Handles diverse input formats including images, text, and numerical data
- **Mixed Precision Support**: Improves GPU memory efficiency and training speed

## Supported Environments

- CartPole
- MiniGrid-DoorKey
- SequenceMemory
- MultiModalPuzzle

## Experiment Visualization

Automatically analyzes and visualizes key metrics including reward curves, exploration rate changes, memory usage, VAE reconstruction error, and average Q-values, generating comprehensive PDF reports.

## Requirements

- PyTorch
- PyTorch Geometric
- Gymnasium
- MiniGrid
- matplotlib
- tqdm
- psutil

## Usage Example

```python
from mhenst_dqn import run_mhenst_dqn_on_env, EnvAdapter
import gymnasium as gym

env_raw = gym.make("CartPole-v1")
env_adapter = EnvAdapter(env_raw)

rewards, model, vae, gnn, meta, vae_loss, q_values = run_mhenst_dqn_on_env(
    env_adapter, 
    input_dim=4, 
    action_dim=2,
    num_episodes=500
)
```
