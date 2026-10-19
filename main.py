import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # FORCES CPU USAGE!
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = 'hide'

from utils.game import Game
from utils.players import RandomPlayer, ExplorerPlayer, UserPlayer, DetectiveNetPlayer, MrXNetPlayer
from utils.neural_nets.detective_nets import TestDetectiveNet
from utils.neural_nets.mrx_nets import TestMrXNet
from utils.neural_nets.trainers import DetectiveTrainer, MrXTrainer
from utils.helpers import Graph, Database


# Things to train:
# - Detectives
#   > Try with 4 and 5 layers of same width
#   > Try with 3 layers with greater width
#   > Try with 3 layers and FastRGCNConv
# - Mr X
#   > Same layer experiments, but after we find the optimal params

game = Game(
    detective_r = RandomPlayer('r'),
    detective_g = RandomPlayer('g'),
    detective_b = RandomPlayer('b'),
    detective_o = RandomPlayer('o'),
    detective_p = RandomPlayer('p'),
    mr_x = UserPlayer('x'),
    visuals = True,
    wait_after_move = None,
    always_show_x = True
)
game.play()
