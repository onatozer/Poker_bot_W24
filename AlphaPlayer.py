from pypokerengine.players import BasePokerPlayer
import tensorflow
import pprint as pp
import numpy as np

class AlphaPlayer(BasePokerPlayer):  # Do not forget to make parent class as "BasePokerPlayer"

    #  we define the logic to make an action through this method. (so this method would be the core of your AI)
    def declare_action(self, valid_actions, hole_card, round_state):
        # valid_actions format => [raise_action_info, call_action_info, fold_action_info]
        print('START NEW ROUND')
        print('valid actions: ', valid_actions)
        print('hole card: ', hole_card)
        pp.pprint(round_state)
        encode_game(valid_actions,round_state)

        call_action_info = valid_actions[1]
        action, amount = call_action_info["action"], call_action_info["amount"]
        return action, amount   # action returned here is sent to the poker engine

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


##code to represent card information as matrix
def encode_card(cards):
  
  def pad_vector(vec):
    padded = np.pad(vec, ((6, 7), (2, 2)), mode='constant', constant_values=(4, 4))
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



##code to represent game state information as matrix
def encode_game(valid_actions,round_state):

  currPot=round_state['pot']['main']['amount']
  encodings=np.zeros((4,9))
  currPlayer=None
  if(round_state["next_player"]==1):
    currPlayer=0
  else:
    currPlayer=1

    #
  curr_plyer_ammt=round_state["seats"][currPlayer]["stack"]
  ratio=curr_plyer_ammt/currPot
  #encoding for the legal actions player is allowed to take
  for action in valid_actions:

    encodings[3][0]=1
    if(action['action']=="call" and action['amount']==0): #check
      encodings[3][1]=1
      encodings[3][2]=0
    if(action['action']=="call"): #call
      encodings[3][1]=0
      if(action['amount']>curr_plyer_ammt):
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
