import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # FORCES CPU USAGE!
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = 'hide'

from utils.game import Game
from utils.players import RandomPlayer, UserPlayer, MrXNetPlayer, DetectiveNetPlayer
from utils.neural_nets.trainers import MrXTrainer, DetectiveTrainer
from utils.neural_nets.detective_nets import DetectiveNet3L
from utils.neural_nets.mrx_nets import MrXNet3L


# Things to train:
# - Detectives
#   > Try with 4 and 5 layers of same width
#   > Try with 3 layers with greater width
#   > Try with 3 layers and FastRGCNConv
# - Mr X
#   > Same layer experiments, but after we find the optimal params

net = DetectiveNet3L()
DetectiveTrainer.train(net, update_freq = 'per_reveal')
