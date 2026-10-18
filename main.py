import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # FORCES CPU USAGE!
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = 'hide'

from utils.game import Game
from utils.players import RandomPlayer, ExplorerPlayer, UserPlayer, DetectiveNetPlayer, MrXNetPlayer
from utils.neural_nets.detective_nets import TestDetectiveNet
from utils.neural_nets.mrx_nets import TestMrXNet
from utils.neural_nets.trainers import DetectiveTrainer, MrXTrainer
from utils.helpers import Graph, Database


detective_net = TestDetectiveNet()

DetectiveTrainer.train(
    detective_net,
    lr = 0.001,
    wd = 0.01,
    epochs = 100,
    opponent = RandomPlayer('x'),
    gamma = 0.95,
    value_coef = 0.6,
    entropy_coef = 0.9,
    update_freq = 'per_reveal'
)
