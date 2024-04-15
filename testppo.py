import sys
import torch

import torchrl
from torchrl.objectives import ClipPPOLoss, ValueEstimators
from tensordict.nn import TensorDictModule
from torchrl.objectives.value import GAE
from torch.utils.data import ReplayBuffer
from torch.distributions import LazyTensorStorage
from torch.utils.data.sampler import SamplerWithoutReplacement

from ppo import PPO
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator
import Siamese as Sm




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
  frames_per_batch = 2048

  

  #whatever the gpu is for 
  device = torch.device('cpu')

  
  gamma = 0.999

  #in_keys for both is either one or 64
  policy_module = TensorDictModule(Sm.SiamesePolicy(),in_keys = ["observation"],out_keys = ["fold","check", "call", "raise_1/2","raise_3/4","raise_1","raise_3/2","raise_2","all_in"])
  value_module = ValueOperator(module = Sm.SiameseReward, in_keys = ["observation"])
  advantage_module = GAE(gamma = gamma, lmbda = 0.95, value_network = value_module, average_GAE = True)
  
  loss_module = torchrl.objectives.ClipPPOLoss(action_network = policy_module, critic_newtork = value_module, clip_epsilon = 0.1, entropy_bonus = True, entropy_coef = 1e-4)
  optim = optim.Adam(loss_module.parameters(),3e-4)

  loss_module.make_value_estimator(
    ValueEstimators.GAE, gamma=gamma 
  )  # We build GAE
  GAE = loss_module.value_estimator

  optim = torch.optim.Adam(loss_module.parameters(), hyperparams['lr'])
  
  #   scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
  #     optim, total_frames // frames_per_batch, 0.0
  # )
  logs = dict(list)
  datastorage = torchrl.collectors.DataCollectorBase(policy_module)
  #gotta figure this one out but cardinality of sub samples
  sub_batch_size = 9
  num_epochs = 69 #nice
  replay_buffer = ReplayBuffer(
    storage=LazyTensorStorage(max_size=frames_per_batch),
    sampler=SamplerWithoutReplacement(),
)
  
  
  for i, tensordict_data in enumerate(datastorage):
    for _ in range(num_epochs):
      advantage_module(tensordict_data)
      data_view = tensordict_data.reshape(-1)
      replay_buffer.extend(data_view.cpu())
      for _ in range(frames_per_batch // sub_batch_size):
        subdata = replay_buffer.sample(sub_batch_size)
        loss_vals = loss_module(subdata.to(device))
        loss_value = (
            loss_vals["loss_objective"]
            + loss_vals["loss_critic"]
            + loss_vals["loss_entropy"]
        )
        loss_value.backward()
        torch.nn.utils.clip_grad_norm_(loss_module.parameters(), 1.0)
        optim.step()
        optim.zero_grad()

    logs["reward"].append(tensordict_data["next", "reward"].mean().item())
    logs["step_count"].append(tensordict_data["step_count"].max().item())
    logs["lr"].append(optim.param_groups[0]["lr"])
    lr_str = f"lr policy: {logs['lr'][-1]: 4.4f}"
    if i % 10 == 0:
        # we need some way to evaluate this every ten batches
    
        

        






  





  #   for idx, data in enumerate(datastorage):
  #     for _ in range(num_epochs):
  #       advantage_module(data)
        
  #       for _ in range(frames_per_batch//sub_batch_size):
        


  
