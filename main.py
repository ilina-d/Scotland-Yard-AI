import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = 'hide'

from utils.game import Game
from utils.players import RandomPlayer, ExplorerPlayer, UserPlayer
from utils.helpers import Graph, Database


game = Game(
    detective_r = UserPlayer('r'),
    detective_g = ExplorerPlayer('g'),
    detective_b = ExplorerPlayer('b'),
    detective_o = ExplorerPlayer('o'),
    detective_p = ExplorerPlayer('p'),
    mr_x = RandomPlayer('x'),
    always_show_x = False,
    visuals = True,
    wait_after_move = 400
)

game.play()
