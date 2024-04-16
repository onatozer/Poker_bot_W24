from pypokerengine.api.game import setup_config, start_poker
from FishPlayer import FishPlayer
from ConsolePlayer import ConsolePlayer
from EmulatorPlayer import EmulatorPlayer, MyModel
from AlphaPlayer import AlphaPlayer
from RandomPlayer import RandomPlayer
from pypokerengine.api.emulator import Emulator
import pprint as pp


def main():
    config = setup_config(max_round=1, initial_stack=100, small_blind_amount=5)
    config.register_player(name="player 1", algorithm=AlphaPlayer("player 1"))
    config.register_player(name="player 2", algorithm=RandomPlayer())
    game_result = start_poker(config, verbose=1)
    pp.pprint(game_result)


if __name__ == "__main__":
    main()
