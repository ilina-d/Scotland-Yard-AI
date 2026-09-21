from .assets import *


class Camera:
    """ Game board movement and zoom logic. """

    def __init__(self, map_w: int, map_h: int, viewport_w: int, viewport_h: int) -> None:
        """
        Create and initialize a Camera instance.

        Arguments:
            map_w: Native map width.
            map_h: Native map height.
            viewport_w: Viewport width.
            viewport_h: Viewport height.
        """

        self.map_w, self.map_h = map_w, map_h
        self.viewport_w, self.viewport_h = viewport_w, viewport_h
        self.min_zoom = self._get_min_zoom(viewport_w, viewport_h)
        self.zoom = self.min_zoom
        self.max_zoom = MAX_ZOOM
        self.x = 0.0
        self.y = 0.0
        self._clip_if_out_of_bounds()


    def _get_min_zoom(self, vw: float, vh: float) -> float:
        """ Helper function for calculating the minimum allowed zoom factor. """

        return max(vw / self.map_w, vh / self.map_h)


    def get_visible_map_dims(self) -> tuple[float, float]:
        """ Calculate the current visible portion of the map. """

        return self.viewport_w / self.zoom, self.viewport_h / self.zoom


    def _clip_if_out_of_bounds(self) -> None:
        """ Helper function for ensuring that the camera never goes outside the map. """

        vw, vh = self.get_visible_map_dims()
        max_x = max(0.0, self.map_w - vw)
        max_y = max(0.0, self.map_h - vh)
        self.x = min(max(self.x, 0.0), max_x)
        self.y = min(max(self.y, 0.0), max_y)


    def map_to_screen(self, mx: float, my: float) -> tuple[float, float]:
        """
        Calculate where the given map point appears on screen.

        Arguments:
            mx: x coordinate on the map.
            my: y coordinate on the map.

        Returns:
            The corresponding coordinates on the screen.
        """

        return (mx - self.x) * self.zoom, (my - self.y) * self.zoom


    def screen_to_map(self, sx: float, sy: float) -> tuple[float, float]:
        """
        Calculate where the given screen point appears on the map.

        Arguments:
            sx: x coordinate on the screen.
            sy: y coordinate on the screen.

        Returns:
            The corresponding coordinates on the map.
        """

        return self.x + sx / self.zoom, self.y + sy / self.zoom


    def pan(self, dx: float, dy: float) -> None:
        """
        Pan across the game board.

        Arguments:
            dx: x coordinate delta.
            dy: y coordinate delta.
        """

        self.x -= dx / self.zoom
        self.y -= dy / self.zoom
        self._clip_if_out_of_bounds()


    def zoom_at(self, screen_pos: tuple[float, float], factor: float) -> None:
        """
        Zoom in or out of the given point.

        Arguments:
            screen_pos: x-y coordinates of the screen point.
            factor: Zoom factor.
        """

        sx, sy = screen_pos
        map_x, map_y = self.screen_to_map(sx, sy)

        new_zoom = min(max(self.zoom * factor, self.min_zoom), self.max_zoom)
        if new_zoom == self.zoom:
            return

        self.zoom = new_zoom
        self.x = map_x - sx / self.zoom
        self.y = map_y - sy / self.zoom
        self._clip_if_out_of_bounds()


    def reset(self) -> None:
        """ Reset the camera's position. """

        self.zoom = self.min_zoom
        self.x = 0.0
        self.y = 0.0
        self._clip_if_out_of_bounds()


    def update_viewport(self, vw: float, vh: float) -> None:
        """
        Adjust class variables after a viewport update.

        Arguments:
            vw: New viewport width.
            vh: New viewport height.
        """

        old_vw, old_vh = self.get_visible_map_dims()
        cx = self.x + old_vw / 2
        cy = self.y + old_vh / 2

        self.viewport_w, self.viewport_h = vw, vh
        self.min_zoom = self._get_min_zoom(vw, vh)
        self.zoom = min(max(self.zoom, self.min_zoom), self.max_zoom)

        nvw, nvh = self.get_visible_map_dims()
        self.x = cx - nvw / 2
        self.y = cy - nvh / 2
        self._clip_if_out_of_bounds()


__all__ = ['Camera']
