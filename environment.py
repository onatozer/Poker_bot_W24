from pypokerengine.api.emulator import Emulator
from pypokerengine.api.game import setup_config, start_poker
from FishPlayer import FishPlayer
from ConsolePlayer import ConsolePlayer
from EmulatorPlayer import EmulatorPlayer, MyModel
from AlphaPlayer import AlphaPlayer
from RandomPlayer import RandomPlayer
import pprint as pp

class environment:
    def __init__(self,config,p1):
        
        self.config = config
        self.emulator = Emulator()
        #only doing heads-up for now, so two players, otherwise, passing in config member variables into emulator
        self.emulator.set_game_rule(player_num=2,max_round=config.max_round, small_blind_amount=config.sb_amount, ante_amount= config.ante)
        self.p1 = p1
        #pypoker on some dumb shit
        players_info = {
            "uuid-1": { "name": config.players_info[0]['name'], "stack": config.initial_stack},
            "uuid-2": { "name": config.players_info[1]['name'], "stack": config.initial_stack},
        }
        self.initial_state = self.emulator.generate_initial_game_state(players_info)

        game_state, event  = self.emulator.start_new_round(self.initial_state)
        next_turn_state, events = self.emulator.apply_action(game_state, 'call', 10)

        self.game_state = game_state
        # print('next turn state')
        # print(next_turn_state)

        # print("sanity check")
        # print(self.p1.uuid)


        # print("card state:")
        # print(self.p1.card_state)
        # print("has been updated")
        # print(self.p1.hole_card_updated)


        # self.observation_space = self.p1.encodings_zero_pad()
        # self.action_space = meallthetimebecauseIhavetonsofpremaritalsex

    # def observationspace(self):
    #     game_state, events = self.emulator.start_new_round(self.initial_state)

    #     print("game state")
    #     pp.pprint(game_state)


    #     print("p1")
    #     print(self.p1.encodings_zero_pad)
        
    #     print("events")
    #     pp.pprint(events)

    def end_it(self):
        return self.emulator.run_until_round_finish(self.game_state)
    '''
    def step():
    #goal is to move onto next action
        observation = penis
        return observation, reward, done, info
    def reset():
        return observation #assuming that we just get new game

    def render():
        #not necessary but would generally show the action being played specifically in the test stage
    '''



def main():
    config = setup_config(max_round=1, initial_stack=100, small_blind_amount=5)
    p1 = AlphaPlayer("player 1")
    config.register_player(name="player 1", algorithm=FishPlayer())
    config.register_player(name="player 2", algorithm=FishPlayer())
    # pp.pprint(config.players_info)
    env = environment(config,p1)

    pp.pprint(env.end_it())
    # print("sanity check")
    # print(p1.uuid)


    # env.observation_space()

if __name__ == "__main__":
    main()
