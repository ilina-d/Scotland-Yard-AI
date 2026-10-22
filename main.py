import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # FORCES CPU USAGE!
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = 'hide'

from multiprocessing import freeze_support
from utils.game import Game
from utils.simulator import Simulator

from utils.players import \
    RandomPlayer, UserPlayer, ExplorerPlayer, GreedyPlayer, \
    XNetPlayer, XNetSemiRandomPlayer, TrainingXNetPlayer, \
    DNetPlayer, TrainingDNetPlayer

from utils.neural_nets.trainers import MrXTrainer, DetectiveTrainer

from utils.neural_nets.detective_nets import DNet3L, DNet5L, DNet5LW, DNet5LRes
from utils.neural_nets.mrx_nets import XNet3L, XNet5L, XNet5LW, XNet5LRes


# Things to train:
# - Detectives
#   > Try with 4 and 5 layers of same width
#   > Try with 3 layers with greater width
#   > Try with 3 layers and FastRGCNConv
# - Mr X
#   > Same layer experiments, but after we find the optimal params


# match int(input('Training Case: ')):
#     case 1:
#         net = XNet3L()
#         MrXTrainer.train(mrx_net = net)
#         MrXTrainer.save(net, '3L_pg_vsRandom_plt', '/')
#     case 2:
#         net = XNet5L()
#         MrXTrainer.train(mrx_net = net)
#         MrXTrainer.save(net, '5L_pg_vsRandom_plt', '/')
#     case 3:
#         net = XNet5LW()
#         MrXTrainer.train(mrx_net = net)
#         MrXTrainer.save(net, '5LW_pg_vsRandom_plt', '/')
#     case 4:
#         net = XNet5LRes()
#         MrXTrainer.train(mrx_net = net)
#         MrXTrainer.save(net, '5LRes_pg_vsRandom_plt', '/')
#     case 5:
#         net = DNet3L()
#         DetectiveTrainer.train(detective_net = net)
#         DetectiveTrainer.save(net, '3L_pg_vsRandom_plt', '/')
#     case 6:
#         net = DNet5L()
#         DetectiveTrainer.train(detective_net = net)
#         DetectiveTrainer.save(net, '5L_pg_vsRandom_plt', '/')
#     case 7:
#         net = DNet5LW()
#         DetectiveTrainer.train(detective_net = net)
#         DetectiveTrainer.save(net, '5LW_pg_vsRandom_plt', '/')
#     case 8:
#         net = DNet5LRes()
#         DetectiveTrainer.train(detective_net = net)
#         DetectiveTrainer.save(net, '5LRes_pg_vsRandom_plt', '/')
#
#
# input('... preventing program from continuing by waiting for input ...')


def main():
    simulator = Simulator(
        detective_r = GreedyPlayer('r'),
        detective_g = GreedyPlayer('g'),
        detective_b = GreedyPlayer('b'),
        detective_o = GreedyPlayer('o'),
        detective_p = GreedyPlayer('p'),
        mr_x = GreedyPlayer('x'),
        num_sims = 1000,
        measure_performance = False,
        num_workers = 20
    )

    simulator.start()


if __name__ == '__main__':
    freeze_support()
    main()
    input('... preventing program from closing by waiting for input ...')
