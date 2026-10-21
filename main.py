import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # FORCES CPU USAGE!
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = 'hide'

from multiprocessing import freeze_support
from utils.game import Game
from utils.simulator import Simulator

from utils.players import \
    RandomPlayer, UserPlayer, ExplorerPlayer, GreedyDetPlayer, \
    XNetPlayer, XNetSemiRandomPlayer, TrainingXNetPlayer, \
    DNetPlayer, TrainingDNetPlayer

from utils.neural_nets.trainers import MrXTrainer, DetectiveTrainer

from utils.neural_nets.detective_nets import DNet3L
from utils.neural_nets.mrx_nets import XNet3L


# Things to train:
# - Detectives
#   > Try with 4 and 5 layers of same width
#   > Try with 3 layers with greater width
#   > Try with 3 layers and FastRGCNConv
# - Mr X
#   > Same layer experiments, but after we find the optimal params


def main():
    simulator = Simulator(
        detective_r = GreedyDetPlayer('r'),
        detective_g = GreedyDetPlayer('g'),
        detective_b = GreedyDetPlayer('b'),
        detective_o = GreedyDetPlayer('o'),
        detective_p = GreedyDetPlayer('p'),
        mr_x = ExplorerPlayer('x'),
        num_sims = 1000,
        measure_performance = False,
        num_workers = 99
    )

    simulator.start()


if __name__ == '__main__':
    freeze_support()
    main()
    input('... preventing program from closing by waiting for input ...')
