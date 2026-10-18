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
#   > Same param experiments
#   > Same layer experiments, but after we find the optimal params


lr = 0.001
wd = 0.0001
epochs = 10_000
gamma = 0.99
value_coef = 0.5
entropy_coef = 0.01
update_freq = 'per_game'
comment = '/'

det_net = TestDetectiveNet()
mrx_net = TestMrXNet()

expn = int(input('Training Case: '))
match expn:
    case 1:
        net = det_net
        trainer = DetectiveTrainer
        update_freq = 'per_reveal'
        comment = 'Detective training with initial params and per-reveal updates.'

    case 2:
        net = det_net
        trainer = DetectiveTrainer
        lr = 0.0005
        wd = 0.00005
        update_freq = 'per_reveal'
        comment = 'Detective training with lower lr and wd and per-reveal updates.'

    case 3:
        net = det_net
        trainer = DetectiveTrainer
        lr = 0.0005
        wd = 0.00005
        update_freq = 'per_round'
        comment = 'Detective training with lower lr and wd and per-round updates.'

    case 4:
        net = mrx_net
        trainer = MrXTrainer
        comment = 'MrX training with initial params and per-game updates.'

    case 5:
        net = mrx_net
        trainer = MrXTrainer
        update_freq = 'per_round'
        comment = 'MrX training with initial params and per-round updates.'

    case 6:
        net = mrx_net
        trainer = MrXTrainer
        update_freq = 'per_reveal'
        comment = 'MrX training with initial params and per-reveal updates.'

    case 7:
        net = mrx_net
        trainer = MrXTrainer
        update_freq = 'per_game'
        lr = 0.0005
        wd = 0.00005
        comment = 'MrX training with lower lr and wd and per-game updates.'

    case 8:
        net = det_net
        trainer = DetectiveTrainer
        lr = 0.0001
        wd = 0.00001
        epochs = 20_000
        comment = 'Detective training with lower lr and wd, and more epochs'

    # Same experiments as for detectives, but first we need to find initial MrX params
    # case 8:
    #     net = mrx_net
    #     lr = 0.01
    #     comment = 'MrX Exp1'
    #
    # case 9:
    #     net = mrx_net
    #     lr = 0.005
    #     comment = 'MrX Exp2'
    #
    # case 10:
    #     net = mrx_net
    #     wd = 0.001
    #     comment = 'MrX Exp3'
    #
    # case 11:
    #     net = mrx_net
    #     wd = 0.005
    #     comment = 'MrX Exp4'
    #
    # case 12:
    #     net = mrx_net
    #     epochs = 20_000
    #     update_freq = 'per_round'
    #     comment = 'MrX Exp5'
    #
    # case 13:
    #     net = mrx_net
    #     gamma = 0.95
    #     comment = 'MrX Exp6'
    #
    # case 14:
    #     net = mrx_net
    #     lr = 0.05
    #     wd = 0.005
    #     comment = 'MrX Exp7'


if trainer == DetectiveTrainer:
    trainer.train(
        detective_net = net,
        lr = lr,
        wd = wd,
        epochs = epochs,
        opponent = RandomPlayer('x'),
        gamma = gamma,
        value_coef = value_coef,
        entropy_coef = entropy_coef,
        update_freq = update_freq
    )

if trainer == MrXTrainer:
    trainer.train(
        mrx_net = net,
        lr = lr,
        wd = wd,
        epochs = epochs,
        opponent_r = RandomPlayer('r'),
        opponent_g = RandomPlayer('g'),
        opponent_b = RandomPlayer('b'),
        opponent_o = RandomPlayer('o'),
        opponent_p = RandomPlayer('p'),
        gamma = gamma,
        value_coef = value_coef,
        entropy_coef = entropy_coef,
        update_freq = update_freq
    )

trainer.save(
    net, f'TrainingExperiment{expn}', comment
)
