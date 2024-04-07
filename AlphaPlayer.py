from pypokerengine.players import BasePokerPlayer
from Siamese import Siamese
import pprint as pp
import numpy as np


class AlphaPlayer(BasePokerPlayer):
    def __init__(self):
        super().__init__()
        self.card_state = np.zeros((6, 16, 16))
        self.hole_card_updated = False
        self.encodings = np.zeros((24, 4, 9))
        self.encodings_zero_pad = np.zeros((24, 16, 16))
        self.own_chips = 0
        self.opponent_chips = 0

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

        # print("Card state representation:")
        # self.update_card_state(hole_card, round_state)
        # pp.pprint(round_state)

        self.print_card_state()

        # TODO: Actual action has to be
        call_action_info = valid_actions[1]
        action, amount = call_action_info["action"], call_action_info["amount"]

        model = Siamese(output_space=9, hidden_dim=1)

        _ , model_output = model.forward(game_state=self.encodings_zero_pad,
                      card_state=self.card_state)
        
        pot_amount = round_state['pot']['main']['amount']
        
        # print(f"Pot: {pot_amount}")
        # print(f"Valid actions: {valid_actions}")
        
        return self.convert_output_into_action(model_output, valid_actions,pot_amount)

    def receive_game_start_message(self, game_info):
        pass

    def receive_round_start_message(self, round_count, hole_card, seats):
        pass

    def receive_street_start_message(self, street, round_state):
        pass

    def receive_game_update_message(self, action, round_state):
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
        pass

    def update_card_state(self, hole_card, round_state):
        # update hole cards
        if not self.hole_card_updated:
            self.card_state[0, ...] = self.encode_card(hole_card)
            self.hole_card_updated = True

        # update flop cards
        elif len(round_state['community_card']) == 3:
            self.card_state[1, ...] = self.encode_card(
                round_state['community_card'])
        # update turn card
        elif len(round_state['community_card']) == 4:
            self.card_state[2, ...] = self.encode_card(
                round_state['community_card'][3:])
        # update river card
        elif len(round_state['community_card']) == 5:
            self.card_state[3, ...] = self.encode_card(
                round_state['community_card'][4:])

        # update all public cards
        if len(round_state['community_card']) > 0:
            self.card_state[4, ...] = self.encode_card(
                round_state['community_card'])

        # update all hole and public cards combined
        if len(round_state['community_card']) > 0:
            self.card_state[5, ...] = (self.card_state[0, ...].astype(
                bool) | self.card_state[4, ...].astype(bool)).astype(int)
        else:
            self.card_state[5, ...] = self.card_state[0, ...]

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
        # print(type(output))
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

        ##TODO: What if neural net selects invalid action, and how to add randomness

        action, amount = output_to_action_dict[output.argmax().item()]
        print(f"Alpha player did {action} for amount {amount}")

        return action, amount
