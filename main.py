import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = 'hide'

from utils.game import Game
from utils.players import RandomPlayer, ExplorerPlayer, UserPlayer, DetectiveNetPlayer, MrXNetPlayer
from utils.neural_nets.detective_nets import TestDetectiveNet
from utils.neural_nets.mrx_nets import TestMrXNet
from utils.helpers import Graph, Database


import time
dw = 0
xw = 0
avgt = []
for i in range(1, 1000 + 1):
    ts = time.time()
    game = Game(
        detective_r = RandomPlayer('r'),
        detective_g = RandomPlayer('g'),
        detective_b = RandomPlayer('b'),
        detective_o = RandomPlayer('o'),
        detective_p = RandomPlayer('p'),
        mr_x = RandomPlayer('x'),
        always_show_x = True,
        visuals = False,
        wait_after_move = None
    )

    w = game.play()
    avgt.append(time.time() - ts)

    if w == 'Mr. X':
        xw += 1
    else:
        dw += 1

    if i % 100 == 0:
        print(f'{i} / 1000  |  Avg Time: {sum(avgt)/len(avgt):.2f}')

print(f'x: {xw}\nd: {dw}')

