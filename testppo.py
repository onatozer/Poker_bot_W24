import sys
import torch

from arguments import get_args
from ppo import PPO
from network import FeedForwardNN
from eval_policy import eval_policy







def main():
  hyperparams = {
      'timesteps_per_batch': 2048,
      'max_timesteps_per_episode': 1000,
      'gamma': 0.999,
      'n_updates_per_iteration': 10,
      'lr': 3e-4,
      'eps_clip': 0.1,
      'hidden_dim': 64
  }
  