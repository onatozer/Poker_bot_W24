import torch
import torch.nn as nn
import torch.nn.functional as F


class Siamese(nn.Module):
    # TODO: Write in pytorch

    def __init__(self, output_space, hidden_dim):
        super().__init__()
        # input space should be 16x16x6

        self.output_space = output_space
        self.hidden_dim = hidden_dim

        # Card Net BS
        self.conv1 = nn.Conv2d(in_channels=6,
                               out_channels=6, kernel_size=(3, 3))

        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)

        self.conv2 = nn.Conv2d(in_channels=6,
                               out_channels=16, kernel_size=(3, 3))

        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=1)

        self.card_net = nn.Sequential(
            self.conv1,
            nn.ReLU(),
            self.pool1,
            self.conv2,
            nn.ReLU(),
            self.pool2
        )

        # action net
        self.action_conv1 = nn.Conv2d(filter_size=3, in_channels=24,
                                      out_channels=12, kernel_size=(3, 3))

        self.action_avg_pool_1 = nn.MaxPool2d(kernel_size=(2, 2))

        self.action_conv2 = nn.Conv2d(filter_size=3, in_channels=12,
                                      out_channels=32, kernel_size=(3, 3))
        self.action_avg_pool_2 = nn.MaxPool2d(kernel_size=2, stride=1)

        self.action_net = nn.Sequential(
            self.action_conv1,
            nn.Tanh(),
            self.action_avg_pool_1,
            self.action_conv2,
            nn.Tanh(),
        )

        # linear layer for policy
        self.policy_fc1 = nn.Linear(4 * 4 * 32 + 4 * 4 * 16, self.hidden_dim)
        self.policy_fc2 = nn.Linear(self.hidden_dim, output_space)

        self.policy_net = nn.Sequential(
            self.policy_fc1.
            nn.ReLU(),
            self.policy_fc2,
            nn.Softmax()
        )

        # linear layer for reward (for value loss)
        
        self.reward_fc1 = nn.Linear(4 * 4 * 32 + 4 * 4 * 16, self.hidden_dim)
        self.reward_fc2 = nn.Linear(self.hidden_dim, 1)
        self.reward_fc_net = nn.Sequential(
            self.reward_fc1,
            nn.ReLU(),
            self.reward_fc2
        )
        self.flatten_card = nn.Flatten(4 * 4 * 16, 1)
        self.flatten_action = nn.Flatten(4 * 4 * 32, 1)

    def forward(self, card_state, game_state):
        card_output = self.card_net(card_state.float())
        game_output = self.action_net(game_state.float())

        flat_card = self.flatten_card(card_output)
        flat_game = self.flatten_action(game_output)
        
        concatted_game = torch.cat(flat_card, flat_game)

        reward = self.reward_fc_net(concatted_game)
        actions = self.policy_net(concatted_game)

        return reward, actions


#   ''''
#    model.add(layers.Conv2D(activation = 'relu', input_shape = (17, 17), kernel_size = (4,4)))


#   #flatten before we use dense layers
#   model.add(layers.flatten())
#   '''
#   model.add(layers.flatten())
#   model.summary()
#   return model

# # def game_model():
