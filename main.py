from utils.game import Game
from utils.players import RandomPlayer, ExplorerPlayer
from utils.helpers import Graph, Database


game = Game(
    detective_r = ExplorerPlayer('r'),
    detective_g = ExplorerPlayer('g'),
    detective_b = ExplorerPlayer('b'),
    detective_o = ExplorerPlayer('o'),
    detective_p = ExplorerPlayer('p'),
    mr_x = RandomPlayer('x')
)

game.play()
