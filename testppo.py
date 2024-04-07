import sys
import torch

from arguments import get_args
from ppo import PPO
import Siamese as Siamese
from eval_policy import eval_policy


def train(env, hyperparameters, action_model, critic_model):
  print(f"Training")
  model = PPO(policy_class = Siamese, env = env, **hyperparameters)
  if action_model != '' and critic_model != '':
    model.actor.load_state_dict(torch.load(action_model))
    model.critic.load_state_dict(torch.load(critic_model))
  else:
      print("Training from scratch")

  model.learn(total_timesteps=69000)

def test(action_model):
  print(f"Testing Phase")
  if action_model == '':
	  print(f"Didn't specify model file. Exiting.", flush=True)
  model = Siamese()
  



def main(args):
  hyperparams = {
      'timesteps_per_batch': 2048,
      'max_timesteps_per_episode': 1000,
      'gamma': 0.999,
      'n_updates_per_iteration': 10,
      'lr': 3e-4,
      'eps_clip': 0.1,
      'hidden_dim': 64
  }
  trainortest = 'train'
  if trainortest == 'train':
     train(env, hyperparams)
  else:
     test(env)

  
