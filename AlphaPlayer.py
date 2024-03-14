from pypokerengine.players import BasePokerPlayer
import pprint as pp
import numpy as np

class AlphaPlayer(BasePokerPlayer):  
    def __init__(self):
        super().__init__()
        self.card_state = np.zeros((6, 17, 17))
        self.hole_card_updated = False

    #  we define the logic to make an action through this method. (so this method would be the core of your AI)
    def declare_action(self, valid_actions, hole_card, round_state):
        # valid_actions format => [raise_action_info, call_action_info, fold_action_info]
        print('START NEW ROUND')
        print('valid actions: ', valid_actions)
        print('hole card: ', hole_card)
        pp.pprint(round_state)
        print("Game State matrix:")
        self.encode_game(valid_actions,round_state)


        print("Card state representation:")
        self.update_card_state(hole_card, round_state)
        self.print_card_state()

        call_action_info = valid_actions[1]
        action, amount = call_action_info["action"], call_action_info["amount"]
        return action, amount 

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
            self.card_state[1, ...] = self.encode_card(round_state['community_card'])
        # update turn card
        elif len(round_state['community_card']) == 4:
            self.card_state[2, ...] = self.encode_card(round_state['community_card'][3:])
        # update river card
        elif len(round_state['community_card']) == 5:
            self.card_state[3, ...] = self.encode_card(round_state['community_card'][4:])

        # update all public cards
        if len(round_state['community_card']) > 0:
            self.card_state[4, ...] = self.encode_card(round_state['community_card'])

        # update all hole and public cards combined
        if len(round_state['community_card']) > 0:
            self.card_state[5, ...] = (self.card_state[0, ...].astype(bool) | self.card_state[4, ...].astype(bool)).astype(int)
        else:
            self.card_state[5, ...] = self.card_state[0, ...]
        
    def encode_card(self, cards):
        def pad_vector(vec):
            padded = np.pad(vec, ((6, 7), (2, 2)), mode='constant')
            return padded
        suits = {'C': 0,'S':1, 'H':2, 'D':3}
        vals = {'A':12,'K':11, 'Q':10,'J': 9, 'T':8}
        for i in range(2,10):
          vals[str(i)] = i - 2
        encoding = np.zeros((4, 13))
        for card in cards:
          suit = suits[card[0]]
          val = vals[card[1]]
          encoding[suit, val] = 1
        return pad_vector(encoding)

    def print_card_state(self):
        state = ['hole', 'flop', 'turn', 'river', 'all public', 'all hole and public']
        for i in range(6):
            print(state[i] + ' cards')
            pp.pprint(self.card_state[i, ...])
        
        # action returned here is sent to the poker engine
            
    ##code to represent game state information as matrix
    def encode_game(self, valid_actions,round_state):
        currPot=round_state['pot']['main']['amount']
        encodings=np.zeros((4,9))
        currPlayer=None
        if(round_state["next_player"]==1):
            currPlayer=0
        else:
            currPlayer=1

        curr_plyer_ammt=round_state["seats"][currPlayer]["stack"]
        ratio=curr_plyer_ammt/currPot


        #For later create dictionary mapping uid into index 
        # p1_uid_to_idx = {}
        # p2_uid_to_idx = {}


        print("Action histories")
        ##NOTE: Not the most efficient way to implement this functionality, don't care right now
        for action in round_state['action_histories']:
            print(round_state['action_histories'][action])
            #small blind is coutned as betting the full pot in the paper
            if(round_state['action_histories'][action] == "SMALLBLIND"):
                encodings[currPlayer][4] = 1

            #big blind would then, by definition, be betting double the current pot :-)
            if(round_state['action_histories'][action] == "BIGBLIND"):
                encodings[currPlayer][7] = 1
            
            if(round_state['action_histories'][action] == "FOLD"):
                encodings[currPlayer][0] = 1
            
            #This is a check
            if(round_state['action_histories'][action] == "CALL" and round_state['amount'][action] == 0):
                encodings[currPlayer][1] = 1

            

            
    

        #encoding for the legal actions player is allowed to take
        for action in valid_actions:
            encodings[3][0]=1
            if(action['action']=="call" and action['amount']==0): #check
                encodings[3][1]=1
                encodings[3][2]=0
            if(action['action']=="call"): #call
                encodings[3][1]=0
                if(action['amount'] > curr_plyer_ammt):
                    encodings[3][2]=0
            else:
                encodings[3][2]=1
            if(ratio>1/2):
                encodings[3][3]=1
            if(ratio>3/4):
                encodings[3][4]=1
            if(ratio>1):
                encodings[3][5]=1
            if(ratio>3/2):
                encodings[3][6]=1
            if(ratio>2):
                encodings[3][7]=1
            encodings[3][8]=1

        
        print(encodings)
    
    



