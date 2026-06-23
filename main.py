from utils.game import Game
from utils.players import RandomPlayer
from utils.helpers import Graph


game = Game(
    detective_r = RandomPlayer('r'),
    detective_g = RandomPlayer('g'),
    detective_b = RandomPlayer('b'),
    detective_o = RandomPlayer('o'),
    detective_p = RandomPlayer('p'),
    mr_x = RandomPlayer('x')
)

game.play()
