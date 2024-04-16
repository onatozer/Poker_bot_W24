import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch.nn.Module import NormalParamExtractor
import torchrl 


class SiamesePolicy(nn.Module):
    def __init__(self):
        super(SiamesePolicy, self).__init__()
        # input space should be 16x16x6

        self.output_space = 9
        self.hidden_dim = 64

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

        # Action net is the games state net
        self.action_conv1 = nn.Conv2d(in_channels=24,
                                      out_channels=12, kernel_size=(3, 3))

        self.action_avg_pool_1 = nn.MaxPool2d(kernel_size=(2, 2))

        self.action_conv2 = nn.Conv2d(in_channels=12,
                                      out_channels=32, kernel_size=(3, 3))
        self.action_avg_pool_2 = nn.MaxPool2d(kernel_size=2, stride=1)

        self.action_net = nn.Sequential(
            self.action_conv1,
            nn.Tanh(),
            self.action_avg_pool_1,
            self.action_conv2,
            nn.Tanh(),
            self.action_avg_pool_2
        )

        # linear layer for policy
        self.policy_fc1 = nn.Linear(4 * 4 * 32 + 4 * 4 * 16, self.hidden_dim)
        self.policy_fc2 = nn.Linear(self.hidden_dim, self.output_space)

        ##No idea why softmax is being used here
        self.policy_net = nn.Sequential(
            self.policy_fc1,
            nn.ReLU(),
            self.policy_fc2,
            # NormalParamExtractor(),
        )

        # linear layer for reward (for value loss)

        # self.reward_fc1 = nn.Linear(4 * 4 * 32 + 4 * 4 * 16, self.hidden_dim)
        # self.reward_fc2 = nn.Linear(self.hidden_dim, 1)

        # self.reward_fc_net = nn.Sequential(
        #     self.reward_fc1,
        #     nn.ReLU(),
        #     self.reward_fc2
        # )
        self.flatten_card = nn.Flatten(start_dim=0)
        self.flatten_action = nn.Flatten(start_dim=0)

    def forward(self, card_state, game_state):

        print((card_state.shape))
        print("Shape:", card_state.shape)
        print("Type:", type(card_state))
        print("Dtype:", card_state.dtype)
        for param in card_state:
            print(param.dtype, '-->', param.shape)

        card_state = torch.from_numpy(card_state).float()
        game_state = torch.from_numpy(game_state).float()
        card_output = self.card_net(card_state)
        flat_card = self.flatten_card(card_output)
        game_output = self.action_net(game_state)

        

        flat_card = self.flatten_card(card_output)
        flat_game = self.flatten_action(game_output)

        # print(f"Flattened:\n{flat_card.shape}, {flat_game.shape}")
        

        concatted_game = torch.cat((flat_card, flat_game),dim=0)
        # print("Game shape:")
        # print(concatted_game.shape)

        actions = self.policy_net(concatted_game)
        actions = F.softmax(actions,dim = -1)
        
        return actions



class SiameseReward(nn.Module):
    # TODO: Write in pytorch

    def __init__(self):
        super(SiameseReward, self).__init__()
        # input space should be 16x16x6

        self.output_space = 1
        self.hidden_dim = 64

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

        # action net)
        self.action_conv1 = nn.Conv2d(in_channels=24,
                                      out_channels=12, kernel_size=(3, 3))

        self.action_avg_pool_1 = nn.MaxPool2d(kernel_size=(2, 2))

        self.action_conv2 = nn.Conv2d(in_channels=12,
                                      out_channels=32, kernel_size=(3, 3))
        self.action_avg_pool_2 = nn.MaxPool2d(kernel_size=2, stride=1)

        self.action_net = nn.Sequential(
            self.action_conv1,
            nn.Tanh(),
            self.action_avg_pool_1,
            self.action_conv2,
            nn.Tanh(),
            self.action_avg_pool_2
        )

        # # linear layer for policy
        # self.policy_fc1 = nn.Linear(4 * 4 * 32 + 4 * 4 * 16, self.hidden_dim)
        # self.policy_fc2 = nn.Linear(self.hidden_dim, self.output_space)

        # ##No idea why softmax is being used here
        # self.policy_net = nn.Sequential(
        #     self.policy_fc1,
        #     nn.ReLU(),
        #     self.policy_fc2,
        #     nn.Softmax()
        # )

        # linear layer for reward (for value loss)

        self.reward_fc1 = nn.Linear(4 * 4 * 32 + 4 * 4 * 16, self.hidden_dim)
        self.reward_fc2 = nn.Linear(self.hidden_dim, 1)

        self.reward_fc_net = nn.Sequential(
            self.reward_fc1,
            nn.ReLU(),
            self.reward_fc2
        )
        self.flatten_card = nn.Flatten(start_dim=0)
        self.flatten_action = nn.Flatten(start_dim=0)

    def forward(self, card_state, game_state):

        print((card_state.shape))
        print("Shape:", card_state.shape)
        print("Type:", type(card_state))
        print("Dtype:", card_state.dtype)
        for param in card_state:
            print(param.dtype, '-->', param.shape)

        card_state = torch.from_numpy(card_state).float()
        game_state = torch.from_numpy(game_state).float()
        card_output = self.card_net(card_state)
        flat_card = self.flatten_card(card_output)
        game_output = self.action_net(game_state)


        flat_card = self.flatten_card(card_output)
        flat_game = self.flatten_action(game_output)

        # print(f"Flattened:\n{flat_card.shape}, {flat_game.shape}")

        concatted_game = torch.cat((flat_card, flat_game),dim=0)
        # print("Game shape:")
        # print(concatted_game.shape)

        reward = self.reward_fc_net(concatted_game)
       
        return reward

