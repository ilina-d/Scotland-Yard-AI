import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = 'hide'

from utils.game import Game
from utils.players import RandomPlayer, ExplorerPlayer, UserPlayer, DetectiveNetPlayer, MrXNetPlayer
from utils.neural_nets.detective_nets import TestDetectiveNet
from utils.neural_nets.mrx_nets import TestMrXNet
from utils.helpers import Graph, Database


game = Game(
    detective_r = RandomPlayer('r'),
    detective_g = RandomPlayer('g'),
    detective_b = RandomPlayer('b'),
    detective_o = RandomPlayer('o'),
    detective_p = RandomPlayer('p'),
    mr_x = RandomPlayer('x'),
    always_show_x = True,
    visuals = True,
    wait_after_move = 500
)

game.play()
