import pygame
import json
import math
from .assets import *
from .camera import Camera
from utils.helpers import State


class Display:
    """ Game board rendering and events logic. """

    def __init__(self, state: State, wait_after_move: int | str | None, always_show_x: bool) -> None:
        """
        Create and initialize a Display instance.

        Waiting Methods:
            - int | Number of milliseconds to wait.
            - "input" | Wait until space is pressed.
            - None | No waiting.

        Arguments:
            state: The state object.
            wait_after_move: The method for waiting after each move.
            always_show_x: Whether to always show Mr. X's location.
        """

        self.state = state

        pygame.init()
        pygame.display.set_caption('Scotland Yard')

        self.FONT_SMALL = pygame.font.SysFont('arial', 12)
        self.FONT_BODY = pygame.font.SysFont('arial', 18)
        self.FONT_BOLD = pygame.font.SysFont('arial', 15, bold = True)
        self.FONT_TITLE = pygame.font.SysFont('arial', 19, bold = True)
        self.FONT_HUGE = pygame.font.SysFont('arial', 34, bold = True)

        self.wait_after_move = wait_after_move
        self.always_show_x = always_show_x

        self.window_size = (DEFAULT_WINDOW_WIDTH, DEFAULT_WINDOW_HEIGHT)
        self.screen = pygame.display.set_mode(self.window_size, pygame.RESIZABLE)
        self.clock = pygame.time.Clock()
        self.is_running = True

        self.map_image = self._load_map_image()
        self.map_nodes = self._load_map_nodes()

        self.viewport_w = self.window_size[0] - PANEL_WIDTH
        self.viewport_h = self.window_size[1]
        self.viewport_rect = pygame.Rect(0, 0, self.viewport_w, self.viewport_h)
        self.camera = Camera(MAP_WIDTH, MAP_HEIGHT, self.viewport_w, self.viewport_h)

        self.dragging = False
        self.drag_moved = False
        self.mouse_down_pos = (0, 0)
        self.last_mouse_pos = (0, 0)

        self.paused_for_input = False
        self.paused_time_left = 0
        self.user_turn = UserTurn()


    def _commit_move(self, destination: int, chosen_ticket: str) -> None:
        """ Helper function for saving the user's chosen move. """

        self.user_turn.chosen_move = (chosen_ticket, destination)
        self.hide_popup()
        self.start_wait_after_move()


    def start_wait_after_move(self) -> None:
        """ Helper function for waiting after a move. """

        if self.wait_after_move is None:
            return

        if isinstance(self.wait_after_move, int):
            self.update_paused_time()

        elif self.wait_after_move == 'input':
            self.paused_for_input = True


    def update_paused_time(self, dt: float = None) -> None:
        """
        Update the remaining time to wait after a move.

        Arguments:
            dt: The time passed since the last update.
        """

        if dt is None:
            self.paused_time_left = max(0.0, self.wait_after_move / 1000)
        else:
            self.paused_time_left = max(0.0, self.paused_time_left - dt)


    @staticmethod
    def _load_map_nodes() -> dict[int, dict[str, float]]:
        """ Helper function for loading the station positions on the map. """

        with open(MAP_NODES_PATH, 'r') as file:
            data = json.load(file)

        nodes_pos = {}
        for entry in data:
            nodes_pos[entry['id']] = {
                'x' : float(entry['x']),
                'y' : float(entry['y'])
            }

        return nodes_pos


    @staticmethod
    def _load_map_image() -> pygame.Surface:
        """ Helper function for loading the game map image. """

        return pygame.image.load(MAP_IMAGE_PATH).convert()


    def update_window_size(self, w: float, h: float) -> None:
        """
        Adjust class variables after a window resize.

        Arguments:
            w: New window width.
            h: New window height.
        """

        w = max(w, MIN_WINDOW_WIDTH)
        h = max(h, MIN_WINDOW_HEIGHT)
        self.window_size = (w, h)
        self.viewport_w = w - PANEL_WIDTH
        self.viewport_h = h
        self.viewport_rect = pygame.Rect(0, 0, self.viewport_w, self.viewport_h)
        self.camera.update_viewport(self.viewport_w, self.viewport_h)


    def process_events(self) -> None:
        """ Process game events. """

        mouse_pos = pygame.mouse.get_pos()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.is_running = False

            elif event.type == pygame.VIDEORESIZE:
                self.update_window_size(event.w, event.h)

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.is_running = False

                if event.key == pygame.K_SPACE:
                    self.paused_for_input = False

            elif event.type == pygame.MOUSEWHEEL and self.viewport_rect.collidepoint(mouse_pos):
                factor = ZOOM_STEP if event.y > 0 else 1 / ZOOM_STEP
                self.camera.zoom_at(mouse_pos, factor)

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                self._handle_left_click_start(event.pos)

            elif event.type == pygame.MOUSEMOTION:
                self._handle_mouse_movement(event.pos)

            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1 and self.dragging:
                self._handle_left_click_complete(event.pos)


    def _get_viewport_center_pos(self) -> tuple[float, float]:
        """ Helper function for getting the viewport center coordinates. """

        return self.viewport_w / 2, self.viewport_h / 2


    def _handle_left_click_start(self, pos: tuple[float, float]) -> None:
        """ Helper function for handling left clicks. """

        if self.user_turn.is_user_turn:
            if self.user_turn.is_popup_shown:
                idx = self._is_pos_in_popup(pos)
                if idx >= 0:
                    self._commit_move(
                        destination = self.user_turn.selected_node,
                        chosen_ticket = self.user_turn.popup_ticket_options[idx]
                    )
                    self.hide_popup()
                    return

            selected_node = self.get_node_near_pos(pos)
            if selected_node and selected_node in self.user_turn.reachable_nodes:
                self.user_turn.selected_node = selected_node
                self.show_popup(selected_node, self.user_turn.reachable_nodes[selected_node])
                return

        if self.viewport_rect.collidepoint(pos):
            self.dragging = True
            self.drag_moved = False
            self.mouse_down_pos = pos
            self.last_mouse_pos = pos


    def _handle_mouse_movement(self, pos: tuple[float, float]) -> None:
        """ Helper function for handling mouse movement. """

        if self.dragging:
            dx = pos[0] - self.last_mouse_pos[0]
            dy = pos[1] - self.last_mouse_pos[1]

            xdiff = abs(pos[0] - self.mouse_down_pos[0])
            ydiff = abs(pos[1] - self.mouse_down_pos[1])
            if xdiff > DRAG_CLICK_THRESHOLD or ydiff > DRAG_CLICK_THRESHOLD:
                self.drag_moved = True

            if self.drag_moved:
                self.camera.pan(dx, dy)

            self.last_mouse_pos = pos

        else:
            self.popup_hover_idx = self._is_pos_in_popup(pos)


    def _handle_left_click_complete(self, pos: tuple[float, float]) -> None:
        """ Helper function for handling completed left clicks. """

        was_click = not self.drag_moved
        self.dragging = False
        self.drag_moved = False
        if was_click and self.user_turn.is_popup_shown and not self._is_pos_in_popup(pos):
            if self.user_turn.is_user_turn:
                self.user_turn.selected_node = None
            self.hide_popup()


    def _is_pos_in_popup(self, pos: tuple[float, float]) -> int:
        """ Helper function for checking if the given position is inside a popup. """

        if self.user_turn.is_popup_shown:
            rects = self._popup_option_rects()
            for i, r in enumerate(rects):
                if r.collidepoint(pos):
                    return i

        return -1


    def get_node_near_pos(self, pos: tuple[float, float]) -> int | None:
        """
        Get the closest node to the given position.

        Arguments:
            pos: x-y coordinates.

        Returns:
            The node ID or None if there aren't any nearby nodes.
        """

        if not self.viewport_rect.collidepoint(pos):
            return None

        mx, my = self.camera.screen_to_map(pos[0], pos[1])
        r = max(NODE_RADIUS, NODE_MIN_SCREEN_RADIUS / max(self.camera.zoom, 1e-6))

        best_id, best_d2 = None, r * r
        for nid, node in self.map_nodes.items():
            dx, dy = node['x'] - mx, node['y'] - my
            d2 = dx * dx + dy * dy
            if d2 <= best_d2:
                best_id, best_d2 = nid, d2

        return best_id


    def get_node_pos(self, node: int) -> tuple[float, float] | None:
        """
        Get the xy coordinates for the specified node ID.

        Arguments:
            node: Node ID.

        Returns:
            The node's xy coordinates or None if not found.
        """

        node = self.map_nodes[node]
        if node is None:
            return None

        return self.camera.map_to_screen(node['x'], node['y'])


    def draw(self) -> None:
        """ Render the entire game visuals. """

        self._draw_map()
        self._draw_highlighted_nodes()
        self._draw_tokens()
        self._draw_panel()
        self._draw_popup()
        pygame.display.flip()


    def _draw_map(self) -> None:
        """ Helper function for rendering the game map. """

        vw, vh = self.camera.get_visible_map_dims()
        src = pygame.Rect(
            int(self.camera.x), int(self.camera.y),
            max(1, math.ceil(vw)), max(1, math.ceil(vh))
        ).clip(self.map_image.get_rect())

        if src.width <= 0 or src.height <= 0:
            return

        sub = self.map_image.subsurface(src)
        scaled = pygame.transform.scale(sub, (self.viewport_w, self.viewport_h))
        self.screen.blit(scaled, (0, 0))


    def _draw_highlighted_nodes(self) -> None:
        """ Helper function for rendering highlighted nodes. """

        if not self.user_turn.reachable_nodes:
            return

        pulse = 0.5 + 0.5 * math.sin(pygame.time.get_ticks() / 220.0)
        r = max(NODE_MIN_SCREEN_RADIUS + 4, NODE_RADIUS * self.camera.zoom + 6)
        ring_w = max(2, int(r * 0.16))

        for nid in self.user_turn.reachable_nodes:
            sp = self.get_node_pos(nid)
            if sp is None:
                continue

            sx, sy = sp
            if not (-r <= sx <= self.viewport_w + r and -r <= sy <= self.viewport_h + r):
                continue

            glow = pygame.Surface((int(r * 4), int(r * 4)), pygame.SRCALPHA)
            gc = (COLOR_RING_GLOW[0], COLOR_RING_GLOW[1], COLOR_RING_GLOW[2],
                  int(COLOR_RING_GLOW[3] * (0.5 + 0.5 * pulse)))

            pygame.draw.circle(glow, gc, (r * 2, r * 2), int(r * 1.6))
            self.screen.blit(glow, (sx - r * 2, sy - r * 2))

            pygame.draw.circle(self.screen, COLOR_RING_INNER, (sx, sy), int(r), ring_w)
            pygame.draw.circle(self.screen, COLOR_RING_OUTER, (sx, sy), int(r), max(1, ring_w // 2))


    def _draw_tokens(self) -> None:
        """ Helper function for rendering player positions. """

        for name, node_id in self._get_player_positions():
            sp = self.get_node_pos(node_id)
            if sp is None:
                continue

            sx, sy = sp
            r = max(9, int(NODE_RADIUS * self.camera.zoom * 0.75))
            color = COLOR_PLAYER.get(name, (200, 200, 200))

            pygame.draw.circle(self.screen, (0, 0, 0), (int(sx + 1), int(sy + 2)), r)
            pygame.draw.circle(self.screen, color, (int(sx), int(sy)), r)
            pygame.draw.circle(self.screen, (12, 12, 14), (int(sx), int(sy)), r, 2)

            if name == 'x':
                d = r * 0.55
                pygame.draw.line(self.screen, (235, 235, 240), (sx - d, sy - d), (sx + d, sy + d), 3)
                pygame.draw.line(self.screen, (235, 235, 240), (sx - d, sy + d), (sx + d, sy - d), 3)

            else:
                letter = name.upper()
                surf = self.FONT_BOLD.render(letter, True, (12, 12, 14))
                self.screen.blit(surf, surf.get_rect(center = (int(sx), int(sy))))


    def _get_player_positions(self) -> list[tuple[str, int]]:
        """ Helper function for getting player positions. """

        pos = []
        for player, node in self.state.positions.items():
            if player == 'x' and not self.always_show_x and self.user_turn.player_name != 'x':
                continue
            pos.append((player, node))

        return pos


    def _draw_panel(self) -> None:
        """ Helper function for rendering the side panel. """

        px0 = self.viewport_w
        panel_rect = pygame.Rect(px0, 0, PANEL_WIDTH, self.window_size[1])
        pygame.draw.rect(self.screen, COLOR_PANEL, panel_rect)
        pygame.draw.line(self.screen, COLOR_PANEL_BORDER, (px0, 0), (px0, self.window_size[1]), 2)

        x = px0 + 18
        y = 16

        title = self.FONT_TITLE.render('Scotland Yard', True, COLOR_TEXT_HEADER)
        self.screen.blit(title, (x, y))
        y += 28

        if self.state.winner:
            surf = self.FONT_BODY.render(f'Winner: {self.state.winner}', True, COLOR_TEXT_HEADER)
            self.screen.blit(surf, (x, y))

        y += 22
        y = self._draw_horizontal_line(x, y, px0)

        y = self._draw_section_header(x, y, 'Detectives')
        for key in LABEL_PLAYER:
            if key != 'x':
                y = self._draw_detectives_tickets(x, y, key)
        y += 4
        y = self._draw_horizontal_line(x, y, px0)

        y = self._draw_section_header(x, y, 'Mr. X')
        y = self._mr_x_row(x, y)
        y += 4
        y = self._draw_horizontal_line(x, y, px0)

        y = self._draw_section_header(x, y, 'Mr. X\'s Travel Logs')
        self._draw_x_log(x, y)


    def _draw_horizontal_line(self, x: float, y: float, panel_x0: float) -> float:
        """ Helper function for rendering a horizontal line on the side panel. """

        pygame.draw.line(self.screen, COLOR_PANEL_BORDER, (x, y), (panel_x0 + PANEL_WIDTH - 18, y))
        return y + 10


    def _draw_section_header(self, x: float, y: float, text: str) -> float:
        """ Helper function for rendering a section header text on the side panel. """

        surf = self.FONT_BOLD.render(text.upper(), True, COLOR_TEXT_MUTED)
        self.screen.blit(surf, (x, y))
        return y + 20


    def _draw_detectives_tickets(self, x: float, y: float, player: str) -> float:
        """ Helper function for rendering the detectives' turn indicators and tickets on the side panel. """

        row_h = 40
        rect = pygame.Rect(x - 6, y, PANEL_WIDTH - 24, row_h)
        cy = y + row_h // 2
        if player == self.state.current_player:
            pygame.draw.rect(self.screen, COLOR_TURN_HIGHLIGHT_INNER, rect, border_radius = 6)
            pygame.draw.rect(self.screen, COLOR_TURN_HIGHLIGHT_OUTER, rect, 1, border_radius = 6)
            name_color = COLOR_PANEL
        else:
            name_color = COLOR_TEXT

        pygame.draw.circle(self.screen, COLOR_PLAYER[player], (x + 10, cy), 9)
        pygame.draw.circle(self.screen, (12, 12, 14), (x + 10, cy), 9, 1)

        label = self.FONT_BODY.render(LABEL_PLAYER[player], True, name_color)
        self.screen.blit(label, (x + 26, cy - label.get_height() // 2))

        tickets = self.state.tickets[player]
        bx = x + 150
        for ticket in ('taxi', 'bus', 'metro'):
            bx = self._badge(bx, cy, ticket, tickets[ticket])

        return y + row_h + 4


    def _mr_x_row(self, x: float, y: float) -> float:
        """ Helper function for rendering Mr. X's turn indicator and tickets on the side panel. """

        row_h = 40
        rect = pygame.Rect(x - 6, y, PANEL_WIDTH - 24, row_h)
        cy = y + row_h // 2
        if self.state.current_player == 'x':
            pygame.draw.rect(self.screen, COLOR_TURN_HIGHLIGHT_INNER, rect, border_radius = 6)
            pygame.draw.rect(self.screen, COLOR_TURN_HIGHLIGHT_OUTER, rect, 1, border_radius = 6)
            name_color = COLOR_PANEL
        else:
            name_color = COLOR_TEXT

        pygame.draw.circle(self.screen, COLOR_PLAYER['x'], (x + 10, cy), 9)
        pygame.draw.circle(self.screen, (12, 12, 14), (x + 10, cy), 9, 1)

        label = self.FONT_BODY.render(LABEL_PLAYER['x'], True, name_color)
        self.screen.blit(label, (x + 26, cy - label.get_height() // 2))

        tickets = self.state.tickets['x']
        bx = x + 150
        for ticket in ('black', 'double'):
            bx = self._badge(bx, cy, ticket, tickets.get(ticket, 0))

        return y + row_h + 4


    def _badge(self, x: float, cy: float, ticket: str, count: int) -> float:
        """ Helper function for rendering remaining tickets on the side panel. """

        color = COLOR_TICKET.get(ticket, (120, 120, 120))
        text_color = (20, 20, 20) if ticket != 'black' else (235, 235, 235)
        surf = self.FONT_BODY.render(str(count), True, text_color)

        pad_x, pad_y = 7, 5
        w = 20 + pad_x * 2
        h = surf.get_height() + pad_y * 2

        rect = pygame.Rect(x, cy - h // 2, w, h)
        pygame.draw.rect(self.screen, color, rect, border_radius = 5)
        pygame.draw.rect(self.screen, (0, 0, 0), rect, 1, border_radius = 5)
        self.screen.blit(surf, surf.get_rect(center = rect.center))

        return x + w + 5


    def _draw_x_log(self, x: float, y: float) -> None:
        """ Helper function for rendering Mr. X's travel logs on the side panel. """

        cols, rows = 3, 8

        num_w = 20
        gap_num = 4
        cell = 50
        row_gap = 3
        col_gap = 26

        grid_x = x
        grid_y = y + 6

        for c in range(cols):
            col_x = grid_x + c * (num_w + gap_num + cell + col_gap)

            for r in range(rows):
                idx = c * rows + r
                num = idx + 1
                cy = grid_y + r * (cell + row_gap)

                num_surf = self.FONT_BODY.render(str(num), True, COLOR_TEXT)
                self.screen.blit(num_surf, num_surf.get_rect(midright = (col_x + num_w - 2, cy + cell // 2)))
                cell_x = col_x + num_w + gap_num
                rect = pygame.Rect(cell_x, cy, cell, cell)

                if idx < len(self.state.travel_logs):
                    ticket = self.state.travel_logs[idx]
                    if isinstance(ticket, int):
                        color = COLOR_WHITE
                        letter = str(ticket)
                    else:
                        color = COLOR_TICKET[ticket]
                        letter = LABEL_TICKET[ticket]
                    pygame.draw.rect(self.screen, color, rect, border_radius = 3)
                    pygame.draw.rect(self.screen, (10, 10, 10), rect, 1, border_radius = 3)
                    tc = (235, 235, 235) if ticket == 'black' else (20, 20, 20)
                    surf = self.FONT_BODY.render(letter, True, tc)
                    self.screen.blit(surf, surf.get_rect(center = rect.center))

                else:
                    pygame.draw.rect(self.screen, COLOR_PANEL_SECTION, rect, border_radius = 3)
                    pygame.draw.rect(self.screen, COLOR_PANEL_BORDER, rect, 1, border_radius = 3)


    def _popup_option_rects(self) -> list[pygame.Rect]:
        """ Helper function for rendering the popup ticket options. """

        if not self.user_turn.is_user_turn:
            return []

        ax, ay = self.user_turn.popup_anchor_pos
        ticket_options = self.user_turn.popup_ticket_options
        w, row_h, pad, header_h = 150, 28, 8, 22
        total_h = header_h + len(ticket_options) * (row_h + 4) + pad

        x = ax + 24 if ax + 24 + w <= self.viewport_w else ax - w - 24
        y = ay - total_h // 2
        y = max(8, min(y, self.viewport_h - total_h - 8))
        x = max(8, min(x, self.viewport_w - w - 8))

        rects = []
        for i in range(len(ticket_options)):
            rects.append(pygame.Rect(x + pad, y + header_h + i * (row_h + 4), w - pad * 2, row_h))

        self.user_turn.popup_box = pygame.Rect(x, y, w, total_h)
        self.user_turn.popup_ticket_boxes = rects

        return rects


    def _draw_popup(self) -> None:
        """ Helper function for rendering the ticket choice popup. """

        if not self.user_turn.is_popup_shown:
            return

        rects = self._popup_option_rects()
        box = self.user_turn.popup_box

        sh = pygame.Surface((box.w + 6, box.h + 6), pygame.SRCALPHA)
        pygame.draw.rect(sh, (0, 0, 0, 120), (0, 0, box.w + 6, box.h + 6), border_radius = 10)
        self.screen.blit(sh, (box.x - 3, box.y + 3))

        pygame.draw.rect(self.screen, COLOR_POPUP_INNER, box, border_radius = 8)
        pygame.draw.rect(self.screen, COLOR_POPUP_OUTER, box, 2, border_radius = 8)

        header = self.FONT_BOLD.render('Pick Ticket:', True, COLOR_TEXT)
        self.screen.blit(header, (box.x + 10, box.y + 4))

        for i, (r, ticket) in enumerate(zip(rects, self.user_turn.popup_ticket_options)):
            hovered = (i == self.popup_hover_idx)
            color = COLOR_POPUP_HOVER if hovered else COLOR_PANEL_SECTION
            pygame.draw.rect(self.screen, color, r, border_radius = 5)
            pygame.draw.rect(self.screen, COLOR_POPUP_OUTER, r, 1, border_radius = 5)

            chip = pygame.Rect(r.x + 6, r.y + 5, 22, r.h - 10)
            pygame.draw.rect(self.screen, COLOR_TICKET[ticket], chip, border_radius = 3)
            pygame.draw.rect(self.screen, (10, 10, 10), chip, 1, border_radius = 3)

            letter = LABEL_TICKET[ticket]
            tc = (235, 235, 235) if ticket == 'black' else (20, 20, 20)
            ls = self.FONT_SMALL.render(letter, True, tc)
            self.screen.blit(ls, ls.get_rect(center=chip.center))

            lbl = self.FONT_BODY.render(ticket.capitalize(), True, COLOR_TEXT)
            self.screen.blit(lbl, (r.x + 36, r.y + (r.h - lbl.get_height()) // 2))


    def show_popup(self, node_id: int, tickets: list[str]) -> None:
        """
        Show the ticket selection popup.

        Arguments:
            node_id: The node ID at which the popup shows.
            tickets: A set of available tickets.
        """

        sp = self.get_node_pos(node_id)
        if sp is None:
            return

        self.user_turn.is_popup_shown = True
        self.user_turn.selected_node = node_id
        self.user_turn.popup_anchor_pos = sp
        self.user_turn.popup_ticket_options = tickets
        self.user_turn.popup_hover_idx = -1


    def hide_popup(self):
        """ Hide the ticket selection popup. """

        self.user_turn.is_popup_shown = False
        self.user_turn.selected_node = None
        self.user_turn.popup_anchor_pos = None
        self.user_turn.popup_ticket_options = None
        self.user_turn.popup_hover_idx = -1
        self.user_turn.popup_box = None
        self.user_turn.popup_ticket_boxes = None


__all__ = ['Display']
