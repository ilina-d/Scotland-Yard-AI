import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # FORCES CPU USAGE!
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = 'hide'

from utils.game import Game
from utils.players import RandomPlayer, ExplorerPlayer, UserPlayer, DetectiveNetPlayer, MrXNetPlayer
from utils.neural_nets.detective_nets import TestDetectiveNet
from utils.neural_nets.mrx_nets import TestMrXNet
from utils.neural_nets.trainers import DetectiveTrainer
from utils.helpers import Graph, Database


det_net = TestDetectiveNet()
det_net.load_weights('InitialTrainingTest_PerRound')

DetectiveTrainer.train(
    detective_net = det_net,
    lr = 0.001,
    wd = 0.0001,
    epochs = 10_000,
    opponent = RandomPlayer('x'),
    gamma = 0.99,
    value_coef = 0.5,
    entropy_coef = 0.01,
    per_round = True
)

DetectiveTrainer.save(
    detective_net = det_net,
    file_name = 'InitialTrainingTest_PerRound',
    comment = '/'
)
