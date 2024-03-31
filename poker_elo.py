from pypokerengine.api.game import setup_config, start_poker
from FishPlayer import FishPlayer
from ConsolePlayer import ConsolePlayer
from EmulatorPlayer import EmulatorPlayer, MyModel
from AlphaPlayer import AlphaPlayer
from RandomPlayer import RandomPlayer
import pprint as pp





# Want to take in two agents and run them against each other n amount of times to determine who's the better agent
def run_simulation(agent_1 = RandomPlayer(), agent_2 = RandomPlayer(), config = setup_config(max_round=1, initial_stack=100, small_blind_amount=5)):
   
    config.register_player(name="player 1", algorithm=agent_1)
    config.register_player(name="player 2", algorithm=agent_2)

    game_result = start_poker(config,verbose=0)


    player_list = game_result['players']

    player_1_winnings = player_list[0]['stack']
    player_2_winnings = player_list[1]['stack']

    player_1_elo, player_2_elo = calculate_elo(score_a=player_1_winnings, score_b=player_2_winnings)
    print("elo:")
    print(player_1_elo, player_2_elo)





#Elo alg I made from a different codign project
def calculate_elo(elo_a: float = 1500, elo_b: float = 1500, K: int = 16, score_a: int = 115, score_b: int = 115):
  win_a = 0
  win_b = 0

  if(score_a > score_b):
    win_a = 1
  else:
    win_b = 1

  expected_score_a = 1/(1+10**((elo_b - elo_a)/400))
  expected_score_b = 1/(1+10**((elo_a - elo_b)/400))

  # print(expected_score_a, " ", expected_score_b)
  return K*(win_a - expected_score_a), K*(win_b - expected_score_b)



def main():
    config = setup_config(max_round=100, initial_stack=100, small_blind_amount=5)
    
    agent_1 = RandomPlayer()
    agent_2 = RandomPlayer()

    run_simulation(agent_1 = agent_1, agent_2 = agent_2, config= config)


if __name__ == "__main__":
    main()
