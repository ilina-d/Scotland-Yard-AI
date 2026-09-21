import pygame as _pg


# Display Config
MAP_IMAGE_PATH = 'utils/display/gameboard.png'
MAP_NODES_PATH = 'utils/display/nodes_pos.json'

MAP_WIDTH, MAP_HEIGHT = 2570, 1926

DEFAULT_WINDOW_WIDTH, DEFAULT_WINDOW_HEIGHT = 1400, 900
MIN_WINDOW_WIDTH, MIN_WINDOW_HEIGHT = 900, 640
PANEL_WIDTH = 340

MAX_ZOOM = 6.0
ZOOM_STEP = 1.15
DRAG_CLICK_THRESHOLD = 6

NODE_RADIUS = 22
NODE_MIN_SCREEN_RADIUS = 12


# Display Colors
COLOR_WHITE = (255, 255, 255)

COLOR_PANEL = (26, 28, 32)
COLOR_PANEL_SECTION = (34, 36, 41)
COLOR_PANEL_BORDER = (60, 63, 70)

COLOR_TEXT = (232, 232, 235)
COLOR_TEXT_MUTED = (150, 153, 160)
COLOR_TEXT_HEADER = (255, 214, 122)

COLOR_TURN_HIGHLIGHT_INNER = (137, 194, 188)
COLOR_TURN_HIGHLIGHT_OUTER = (122, 255, 242)

COLOR_RING_OUTER = (255, 120, 120)
COLOR_RING_INNER = (219, 59, 59)
COLOR_RING_GLOW = (255, 99, 99, 90)

COLOR_POPUP_INNER = (38, 42, 48)
COLOR_POPUP_OUTER = (90, 96, 108)
COLOR_POPUP_HOVER = (60, 68, 82)

COLOR_TICKET = {
    'taxi' : (239, 229, 82),
    'bus' : (49, 166, 149),
    'metro' : (244, 84, 72),
    'black' : (26, 26, 30),
    'double' : (242, 78, 218)
}

COLOR_PLAYER = {
    'r' : (220, 70, 70),
    'g' : (70, 200, 100),
    'b' : (70, 140, 240),
    'o' : (240, 160, 50),
    'p' : (180, 100, 220),
    'x' : (26, 26, 32),
}

COLOR_REVEAL = (255, 255, 255)


# Display Labels
LABEL_TICKET = {
    'taxi' : 'T',
    'bus' : 'B',
    'metro' : 'M',
    'black' : '?',
    'double' : '*'
}

LABEL_PLAYER = {
    'r' : 'Detective Red',
    'g' : 'Detective Green',
    'b' : 'Detective Blue',
    'o' : 'Detective Orange',
    'p' : 'Detective Purple',
    'x' : 'Mr. X'
}


# Helper Classes
class UserTurn:
    """ Helper class for handling user input during their turn. """

    is_user_turn: bool = False
    player_name: str | None = None
    chosen_move: tuple[str, int | None] | None = None

    reachable_nodes: dict[int, list[str]] | None = None
    selected_node: int | None = None

    is_popup_shown: bool | None = False
    popup_ticket_options: list[str] | None = None
    popup_anchor_pos: tuple[float, float] | None = None
    popup_hover_idx: int | None = None
    popup_box: _pg.Rect | None = None
    popup_ticket_boxes: list[_pg.Rect] | None = None


    def __init__(self) -> None:
        """ Create and initialize a UserTurn instance. """

        self.reset()


    def reset(self) -> None:
        """ Reset variables. """

        self.is_user_turn = False
        self.player_name = None
        self.chosen_move = None
        self.reachable_nodes = None
        self.selected_node = None
        self.is_popup_shown = False
        self.popup_ticket_options = None
        self.popup_anchor_pos = None
        self.popup_hover_idx = None
        self.popup_box = None
        self.popup_ticket_boxes = None
