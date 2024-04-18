from pypokerengine.players import BasePokerPlayer
from Siamese import SiamesePolicy, SiameseReward
import pprint as pp
import numpy as np
from pypokerengine.api.emulator import Emulator
from pypokerengine.utils.card_utils import gen_cards
from pypokerengine.utils.game_state_utils import restore_game_state, attach_hole_card, attach_hole_card_from_deck
from torch.distributions.categorical import Categorical
from cartpole_ppo_follow import PPOTrainer, calculate_gae, discount_rewards

class AlphaPlayer(BasePokerPlayer):
    def __init__(self, name = "player 1", is_training = False):
        super().__init__()
        self.card_state = np.zeros((6, 16, 16))

        self.hole_card_updated = False
        self.encodings = np.zeros((24, 4, 9))

        self.encodings_zero_pad = np.zeros((24, 16, 16))

        self.name = name
        self.uuid = ""

        self.stage = "preflop"
        self.own_chips = 0
        self.opponent_chips = 0
        
        #TODO: Create list of tensors which store all the hand information
        self.past_hands = []

        #Instantiate the model
        self.policy = SiamesePolicy()
        self.critic = SiameseReward()
        
        # self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)
        # self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=3e-4)

        self.is_training = is_training

        #[(game-state tensor, card-state tensor), (game-state tensor, card-state tensor)]


    #  we define the logic to make an action through this method. (so this method would be the core of your AI)
    def declare_action(self, valid_actions, hole_card, round_state):
        # valid_actions format => [raise_action_info, call_action_info, fold_action_info]
        '''
        print('START NEW ROUND')
        print('valid actions: ', valid_actions)
        print('hole card: ', hole_card)
        pp.pprint(round_state)
        print("Game State matrix:")
        '''
        self.encodings_zero_pad = self.encode_game(valid_actions, round_state)
        self.card_state = self.update_card_state(hole_card, round_state)

        
        # model = SiamesePolicy()

        #Run forward pass through the policy network and return the action generated 
        

        ##instantiate the emualtor with the current_state
        current_state = self._setup_game_state(round_state, hole_card)


        #perform a rollout on this action and return the data, then train the network on it
        loop_pot = round_state['pot']['main']['amount']
        loop_valid_actions = valid_actions

        if(self.is_training):
            trainer = PPOTrainer(policy_network = self.policy, critic_network = self.critic)
            train_data = [[], [], [], []] # obs, act, values, act_log_probs
        #     # obs = env.reset()

            ep_reward = 0
            done = False
            obs = (self.encodings_zero_pad, self.card_state)
            while(not done):
                policy_output = self.policy.forward(game_state=self.encodings_zero_pad,
                                    card_state=self.card_state)
                        
                critic_output = self.critic.forward(game_state=self.encodings_zero_pad,
                                    card_state=self.card_state)
                
                
                logits = policy_output
                act_distribution = Categorical(logits=logits)
                act = act_distribution.sample()

                act_log_prob = act_distribution.log_prob(act)

                val = critic_output.item()

                action, amount = self.convert_output_into_action(act.item(),loop_valid_actions,loop_pot)

                print("emulator output")

                next_turn_state, events = self.emulator.apply_action(current_state,action,amount)
                pp.pprint(next_turn_state)
                pp.pprint((events))

                #pypoker is lowkey aids
                reward = 0
                for event in events:
                    if 'players' in event:
                        for player in event['players']:
                            if(player['uuid'] == self.uuid):
                                reward = player['stack'] - self.initial_pot

                    if event['type'] == 'event_round_finish':
                        done = True

                if(not done):
                    #get most recent event
                    round_state = events[-1]['round_state']
                    valid_actions = events[-1]['valid_actions']

                    #next_obs = (card state, game state)
                next_obs = (self.update_card_state(hole_card,round_state), self.encode_game(valid_actions,round_state))
        #         next_obs, reward, done, _ = env.step(act)

                
                for i, item in enumerate((obs, act, val, act_log_prob)):
                    # print('item:')
                    # print(item)
                    train_data[i].append(item)

                obs = next_obs
                print("observation")
                print(obs)
                ep_reward += reward
                if done:
                    break

            print("Train data")
            print(train_data)

            gae_rewards = calculate_gae(ep_reward,train_data[2])

            #TODO: update model parameters:
            print("Gay:")
            print(gae_rewards)

            print("thomas")
            # print(train_data[0])

            for i, item in enumerate(train_data[0]):
                # (train_data[0][i], train_data[1][i], train_data[3[i]], gae_rewards[i]])
                # print(f"obs: {train_data[0][i]}, action {train_data[1][i]}, log probs {train_data[3][i]}")
                trainer.train_policy(train_data[0][i], train_data[1][i], train_data[3][i], gae_rewards[i])
                # trainer.train_value(train_data[0][i], train_data[1[i], train_data[3[i]], gae_rewards[i]])

      

        policy_output = self.policy.forward(game_state=self.encodings_zero_pad,
                                    card_state=self.card_state)
        pot_amount = round_state['pot']['main']['amount']


        # adv = gae
        # ratio = (log_prob - old_log_prob).exp()

        # # actor_loss
        # surr_loss = ratio * adv
        # clipped_surr_loss = (
        #     torch.clamp(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon) * adv
        # )

        # surr3 = 3*adv

        # # entropy
        # entropy = dist.entropy().mean()

        # actor_loss = (
        #     -torch.min(surr_loss, clipped_surr_loss,surr3).mean()
        #     - entropy * self.entropy_weight
        # )
        # # critic_loss
        # value = self.critic(state)
        # #clipped_value = old_value + (value - old_value).clamp(-0.5, 0.5)

        # clampreturn = torch.clamp(return_,-self.ownchips,self.opponentchips)
        # critic_loss = (clampreturn - value).pow(2).mean()

        # # train critic
        # self.critic_optimizer.zero_grad()
        # critic_loss.backward(retain_graph=True)
        # self.critic_optimizer.step()

        # # train actor
        # self.actor_optimizer.zero_grad()
        # actor_loss.backward()
        # self.actor_optimizer.step()

        # actor_losses.append(actor_loss.item())
        # critic_losses.append(critic_loss.item())

        act_distribution = Categorical(logits=policy_output)
        act = act_distribution.sample()
      
        
        return self.convert_output_into_action(act.item(), valid_actions,pot_amount)

    def receive_game_start_message(self, game_info):
        # print("round started")

        nb_player = game_info['player_num']
        max_round = game_info['rule']['max_round']
        sb_amount = game_info['rule']['small_blind_amount']
        ante_amount = game_info['rule']['ante']

        self.emulator = Emulator()
        self.emulator.set_game_rule(nb_player, max_round, sb_amount, ante_amount)
        for player_info in game_info['seats']:
            uuid = player_info['uuid']
            player_model = AlphaPlayer(name = uuid, is_training=False)
            # player_model = self.my_model if uuid == self.uuid else self.opponents_model

            ##record the alphaPlayer's uuid (important for later)
            #TODO: change so that this updates start of each round
            if(player_info['name'] == self.name):
                self.uuid = uuid
                self.initial_pot = player_info['stack']

            self.emulator.register_player(uuid, player_model)

        #stupid ass pypoker shit, emulator uuid's gon be different from game uuid
        # players_info = {
        #     "uuid-1": { "name": config.players_info[0]['name'], "stack": config.initial_stack},
        #     "uuid-2": { "name": config.players_info[1]['name'], "stack": config.initial_stack},
        # }
        # self.initial_state = self.emulator.generate_initial_game_state()


        #maybe we want this for later (calculate amount bet for each player)
        # for player in game_info['seats']:
        #     if player['name'] == self.name:
        #         self.uuid = player['uuid']
        
    def receive_round_start_message(self, round_count, hole_card, seats):
        ##reset private member variables at the start of each round
        print("seats")
        pp.pprint(seats)
        self.card_state = np.zeros((6, 16, 16))

        self.hole_card_updated = False
        self.encodings = np.zeros((24, 4, 9))

        self.encodings_zero_pad = np.zeros((24, 16, 16))

        
    def receive_street_start_message(self, street, round_state):
        pass

    #set the emulator state to the current state
    def _setup_game_state(self, round_state, my_hole_card):
        game_state = restore_game_state(round_state)
        game_state['table'].deck.shuffle()
        player_uuids = [player_info['uuid'] for player_info in round_state['seats']]
        for uuid in player_uuids:
            if uuid == self.uuid:
                game_state = attach_hole_card(game_state, uuid, gen_cards(my_hole_card))  # attach my holecard
            else:
                game_state = attach_hole_card_from_deck(game_state, uuid)  # attach opponents holecard at random
        return game_state

    def receive_game_update_message(self, action, round_state):
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
        # self.update_card_state()
        pass

    def update_card_state(self, hole_card, round_state):
        card_state = self.card_state
        # update hole cards
        if not self.hole_card_updated:
            card_state[0, ...] = self.encode_card(hole_card)
            self.hole_card_updated = True

        # update flop cards
        elif len(round_state['community_card']) == 3:
            card_state[1, ...] = self.encode_card(
                round_state['community_card'])
        # update turn card
        elif len(round_state['community_card']) == 4:
            card_state[2, ...] = self.encode_card(
                round_state['community_card'][3:])
        # update river card
        elif len(round_state['community_card']) == 5:
            card_state[3, ...] = self.encode_card(
                round_state['community_card'][4:])

        # update all public cards
        if len(round_state['community_card']) > 0:
            card_state[4, ...] = self.encode_card(
                round_state['community_card'])

        # update all hole and public cards combined
        if len(round_state['community_card']) > 0:
            card_state[5, ...] = (self.card_state[0, ...].astype(
                bool) | self.card_state[4, ...].astype(bool)).astype(int)
        else:
            card_state[5, ...] = self.card_state[0, ...]

        return card_state

    def encode_card(self, cards):
        def pad_vector(vec):
            padded = np.pad(vec, ((6, 6), (1, 2)),
                            mode='constant', constant_values=0)
            return padded
        suits = {'C': 0, 'S': 1, 'H': 2, 'D': 3}
        vals = {'A': 12, 'K': 11, 'Q': 10, 'J': 9, 'T': 8}
        for i in range(2, 10):
            vals[str(i)] = i - 2
        encoding = np.zeros((4, 13))
        for card in cards:
            suit = suits[card[0]]
            val = vals[card[1]]
            encoding[suit, val] = 1
        return pad_vector(encoding)

    def print_card_state(self):
        state = ['hole', 'flop', 'turn', 'river',
                 'all public', 'all hole and public']
        for i in range(6):
            # print(state[i] + ' cards')
            # pp.pprint(self.card_state[i, ...])
            pass

        # action returned here is sent to the poker engine

    # code to represent game state information as matrix
    def encode_game(self, valid_actions, round_state):
        currPot = round_state['pot']['main']['amount']
        # encodings=np.zeros((4,9))
        currPlayer = None
        if (round_state["next_player"] == 1):
            currPlayer = 0
        else:
            currPlayer = 1

        curr_plyer_ammt = round_state["seats"][currPlayer]["stack"]
        ratio = curr_plyer_ammt/currPot

        # For later create dictionary mapping uid into index
        # p1_uid_to_idx = {}
        # p2_uid_to_idx = {}

        print("Action histories")
        # NOTE: Not the most efficient way to implement this functionality, don't care right now
        for pftr in round_state['action_histories']:
            if (pftr == "preflop"):
                matrix_index = 0

            if (pftr == "flop"):
                matrix_index = 6

            if (pftr == "turn"):
                matrix_index = 12

            if (pftr == "river"):
                matrix_index = 18
            for action in round_state['action_histories'][pftr]:
                # print(f"action: {action}")

                # print(f"how to get action {action['action']}")
                # print(f"how to get amount {action['amount']}")

                # print(round_state['action_histories'][pftr])
                # small blind is coutned as betting the full pot in the paper
                # 9 actions
                # all in
                if action['action'] == "RAISE" and action['amount'] == curr_plyer_ammt:
                    self.encodings[matrix_index][currPlayer][8] += 1

                if (action['action'] == "BIGBLIND"):
                    self.encodings[matrix_index][currPlayer][7] += 1

                if action['action'] == "RAISE" and action['amount'] >= currPot*2:
                    self.encodings[matrix_index][currPlayer][7] += 1

                if action['action'] == "RAISE" and action['amount'] >= currPot*1.5 and action['amount'] < currPot*2:
                    self.encodings[matrix_index][currPlayer][6] += 1

                if (action['action'] == "SMALLBLIND"):
                    self.encodings[matrix_index][currPlayer][5] += 1

                if action['action'] == "RAISE" and action['amount'] >= currPot*1 and action['amount'] < currPot*1.5:
                    self.encodings[matrix_index][currPlayer][5] += 1

                if action['action'] == "RAISE" and action['amount'] >= currPot*0.75 and action['amount'] < currPot*1:
                    self.encodings[matrix_index][currPlayer][4] += 1

                if action['action'] == "RAISE" and action['amount'] >= currPot*0.5 and action['amount'] < currPot*0.75:
                    self.encodings[matrix_index][currPlayer][3] += 1

                # fold
                if (action['action'] == "FOLD"):
                    self.encodings[matrix_index][currPlayer][0] += 1

                # This is a check
                if (action['action'] == "CALL" and action['amount'] == 0):
                    self.encodings[matrix_index][currPlayer][1] += 1

                # This is a call
                if (action['action'] == "CALL"):
                    self.encodings[matrix_index][currPlayer][2] += 1

                # add the sum row
                self.encodings[matrix_index][2] = self.encodings[matrix_index][0] + \
                    self.encodings[matrix_index][1]

                # encoding for the legal actions player is allowed to take
                for action in valid_actions:
                    self.encodings[matrix_index][3][0] = 1
                    if (action['action'] == "call" and action['amount'] == 0):  # check
                        self.encodings[matrix_index][3][1] = 1
                        self.encodings[matrix_index][3][2] = 0
                    if (action['action'] == "call"):  # call
                        self.encodings[matrix_index][3][1] = 0
                        if (action['amount'] > curr_plyer_ammt):
                            self.encodings[matrix_index][3][2] = 0
                    else:
                        self.encodings[matrix_index][3][2] = 1
                    if (ratio > 1/2):
                        self.encodings[matrix_index][3][3] = 1
                    if (ratio > 3/4):
                        self.encodings[matrix_index][3][4] = 1
                    if (ratio > 1):
                        self.encodings[matrix_index][3][5] = 1
                    if (ratio > 3/2):
                        self.encodings[matrix_index][3][6] = 1
                    if (ratio > 2):
                        self.encodings[matrix_index][3][7] = 1
                    self.encodings[matrix_index][3][8] = 1

                matrix_index += 1
        # zero pad encodings to be 24x13x13 matrix for neural network in seperate padded matrix variable
        # return encodings

        # Calculate padding
        padding = [(0, 0),  # First dimension (24), no padding needed
                   (6, 6),  # Second dimension (4 to 16), pad 6 on each side
                   (3, 4)]  # Third dimension (9 to 16), pad 3 before, 4 after

        # Apply padding
        return np.pad(self.encodings, pad_width=padding, mode='constant', constant_values=0)

    def convert_output_into_action(self,output, valid_actions, pot_amount):
        #amount you're allowed to call is always in 2nd value of valid actions
        print("output")
        
        # print(act)

        call_amount = valid_actions[1]['amount']
        max_bet = valid_actions[2]['amount']['max']



        output_to_action_dict = {0: ('fold', 0),
                                1: ('call', call_amount),    #check is encoded as call with amount 0 in pypoker
                                2: ('call', call_amount),
                                3: ('raise', pot_amount*.5),
                                4: ('raise', pot_amount*.75),
                                5: ('raise', pot_amount),
                                6: ('raise', pot_amount*1.5),
                                7: ('raise', pot_amount*2),
                                8: ('raise', max_bet)
                                }


        action, amount = output_to_action_dict[output]
        print(f"Alpha player did {action} for amount {amount}")

        return action, amount

    def update_past_hands(self):
        hand_tuple = (self.encodings_zero_pad, self.card_state)
        self.past_hands.append(hand_tuple)

    def update_amount_bet(self, round_state):

        if('turn' in round_state['action_histories']):
            
            
            return
        
        if('river' in round_state['action_histories']):
            
            return
        
        if('flop' in round_state['action_histories']):
            
            return
        
        if('preflop' in round_state['action_histories']):
            
            return
    
    
    def train(self):
        mycock = "1inch"
        return mycock
        