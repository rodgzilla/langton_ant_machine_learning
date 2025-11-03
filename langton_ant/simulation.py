"""Core Langton's Ant simulation engine"""

import numpy as np
from typing import Optional, Tuple
from .highway_detector import HighwayDetector


class LangtonAnt:
    """
    Langton's Ant cellular automaton simulator with expandable grid.

    Rules:
    - If on white cell (0): turn right, flip to black (1), move forward
    - If on black cell (1): turn left, flip to white (0), move forward

    Directions: 0=North, 1=East, 2=South, 3=West
    """

    # Direction vectors: North, East, South, West
    DIRECTION_VECTORS = {
        0: (0, -1),   # North (up, negative y)
        1: (1, 0),    # East (right, positive x)
        2: (0, 1),    # South (down, positive y)
        3: (-1, 0)    # West (left, negative x)
    }

    DIRECTION_NAMES = {0: 'N', 1: 'E', 2: 'S', 3: 'W'}

    def __init__(
        self,
        initial_size: int = 100,
        start_position: Optional[Tuple[int, int]] = None,
        start_direction: int = 0,
        initial_grid: Optional[np.ndarray] = None,
        expansion_margin: int = 10
    ):
        """
        Initialize Langton's Ant simulation.

        Args:
            initial_size: Initial grid size (will be initial_size x initial_size)
            start_position: Starting (x, y) position of ant. If None, starts at center
            start_direction: Starting direction (0=N, 1=E, 2=S, 3=W)
            initial_grid: Optional pre-existing grid pattern (2D numpy array of 0s and 1s)
            expansion_margin: When ant is within this many cells of edge, expand grid
        """
        if initial_grid is not None:
            self.grid = initial_grid.astype(np.uint8).copy()
            self.height, self.width = self.grid.shape
        else:
            self.width = initial_size
            self.height = initial_size
            self.grid = np.zeros((self.height, self.width), dtype=np.uint8)

        # Set starting position (default to center)
        if start_position is None:
            self.ant_x = self.width // 2
            self.ant_y = self.height // 2
        else:
            self.ant_x, self.ant_y = start_position

        # Validate starting direction
        if start_direction not in [0, 1, 2, 3]:
            raise ValueError(f"start_direction must be 0-3, got {start_direction}")

        self.ant_direction = start_direction
        self.expansion_margin = expansion_margin
        self.step_count = 0
        self.expansion_count = 0

        # Highway detector
        self.highway_detector = HighwayDetector()

    def step(self) -> None:
        """
        Execute one step of Langton's Ant simulation.

        1. Check current cell color
        2. Turn right (if white/0) or left (if black/1)
        3. Flip current cell color
        4. Move forward
        5. Expand grid if necessary
        """
        # Get current cell value
        current_cell = self.grid[self.ant_y, self.ant_x]

        # Turn: right if white (0), left if black (1)
        if current_cell == 0:
            # Turn right
            self.ant_direction = (self.ant_direction + 1) % 4
        else:
            # Turn left
            self.ant_direction = (self.ant_direction - 1) % 4

        # Flip current cell
        self.grid[self.ant_y, self.ant_x] = 1 - current_cell

        # Move forward in current direction
        dx, dy = self.DIRECTION_VECTORS[self.ant_direction]
        self.ant_x += dx
        self.ant_y += dy

        # Check if we need to expand grid
        self._check_and_expand_grid()

        # Update step count
        self.step_count += 1

        # Update highway detector
        self.highway_detector.add_position(self.ant_x, self.ant_y, self.ant_direction)

    def _check_and_expand_grid(self) -> None:
        """
        Check if ant is near grid boundary and expand if necessary.
        Expands by doubling the grid size in the necessary direction(s).
        """
        needs_expansion = False
        expand_left = expand_right = expand_top = expand_bottom = 0

        # Check if near boundaries
        if self.ant_x < self.expansion_margin:
            expand_left = self.width
            needs_expansion = True
        elif self.ant_x >= self.width - self.expansion_margin:
            expand_right = self.width
            needs_expansion = True

        if self.ant_y < self.expansion_margin:
            expand_top = self.height
            needs_expansion = True
        elif self.ant_y >= self.height - self.expansion_margin:
            expand_bottom = self.height
            needs_expansion = True

        if needs_expansion:
            self._expand_grid(expand_left, expand_right, expand_top, expand_bottom)
            self.expansion_count += 1

    def _expand_grid(self, left: int, right: int, top: int, bottom: int) -> None:
        """
        Expand the grid by adding rows/columns.

        Args:
            left: Number of columns to add on the left
            right: Number of columns to add on the right
            top: Number of rows to add on top
            bottom: Number of rows to add on bottom
        """
        new_height = self.height + top + bottom
        new_width = self.width + left + right
        new_grid = np.zeros((new_height, new_width), dtype=np.uint8)

        # Copy old grid into new grid at appropriate offset
        new_grid[top:top + self.height, left:left + self.width] = self.grid

        # Update ant position
        self.ant_x += left
        self.ant_y += top

        # Update grid reference and dimensions
        self.grid = new_grid
        self.width = new_width
        self.height = new_height

    def run_until_highway(self, max_steps: int = 100000, check_interval: int = 500) -> Optional[str]:
        """
        Run simulation until highway is detected or max_steps is reached.

        Args:
            max_steps: Maximum number of steps to run
            check_interval: How often to check for highway pattern

        Returns:
            Highway direction ('NE', 'NW', 'SE', 'SW') or None if not detected
        """
        while self.step_count < max_steps:
            self.step()

            # Check for highway periodically
            if self.step_count % check_interval == 0:
                if self.highway_detector.is_highway_detected():
                    return self.highway_detector.get_highway_direction()

        # Max steps reached without highway detection
        return None

    def get_state(self) -> dict:
        """
        Get current state of the simulation.

        Returns:
            Dictionary with current state information
        """
        highway_direction = None
        if self.highway_detector.is_highway_detected():
            highway_direction = self.highway_detector.get_highway_direction()

        return {
            'step_count': self.step_count,
            'ant_position': (self.ant_x, self.ant_y),
            'ant_direction': self.ant_direction,
            'ant_direction_name': self.DIRECTION_NAMES[self.ant_direction],
            'grid_size': (self.width, self.height),
            'expansion_count': self.expansion_count,
            'highway_detected': self.highway_detector.is_highway_detected(),
            'highway_direction': highway_direction
        }

    def get_grid_copy(self) -> np.ndarray:
        """Return a copy of the current grid state."""
        return self.grid.copy()
