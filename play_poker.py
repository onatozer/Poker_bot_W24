from pypokerengine.api.game import setup_config, start_poker
from FishPlayer import FishPlayer
from ConsolePlayer import ConsolePlayer
from EmulatorPlayer import EmulatorPlayer, MyModel
from AlphaPlayer import AlphaPlayer
import pprint as pp


def main():
    config = setup_config(max_round=1, initial_stack=100, small_blind_amount=5)
    config.register_player(name="fish player 1", algorithm=AlphaPlayer())
    config.register_player(name="fish player 2", algorithm=FishPlayer())
    game_result = start_poker(config, verbose=1)
    pp.pprint(game_result)


if __name__ == "__main__":
    main()
