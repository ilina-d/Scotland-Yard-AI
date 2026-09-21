from .base_player import BasePlayer
from utils.helpers import State


class UserPlayer(BasePlayer):
    """ Player agent controlled by the user. """

    def make_move(self, state: State) -> tuple[str, int | None]:
        # Logic handled by Game
        return '', None


__all__ = ["UserPlayer"]