from abc import ABC, abstractmethod


class BasePlayer(ABC):
    """ Abstract representation of a player. """

    def __init__(self) -> None:
        """ Create a player instance. """
        pass


__all__ = ["BasePlayer"]