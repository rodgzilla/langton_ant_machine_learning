"""Pygame visualization for Langton's Ant simulation"""

import pygame
import numpy as np
from typing import Optional, Tuple
from .simulation import LangtonAnt


class Visualizer:
    """
    Pygame-based visualizer for Langton's Ant simulation.

    Controls:
    - SPACE: Pause/unpause simulation
    - UP/DOWN: Increase/decrease simulation speed
    - R: Reset simulation
    - Q/ESC: Quit
    """

    # Colors
    COLOR_WHITE = (255, 255, 255)
    COLOR_BLACK = (0, 0, 0)
    COLOR_ANT = (255, 0, 0)  # Red
    COLOR_BG = (200, 200, 200)  # Light gray
    COLOR_TEXT = (0, 0, 0)

    def __init__(
        self,
        ant: LangtonAnt,
        window_size: Tuple[int, int] = (800, 800),
        cell_size: Optional[int] = None,
        steps_per_frame: int = 1
    ):
        """
        Initialize the visualizer.

        Args:
            ant: LangtonAnt simulation instance
            window_size: Window dimensions (width, height)
            cell_size: Size of each cell in pixels (auto-calculated if None)
            steps_per_frame: Number of simulation steps per frame
        """
        pygame.init()

        self.ant = ant
        self.window_size = window_size
        self.steps_per_frame = steps_per_frame
        self.paused = False
        self.running = True

        # Auto-calculate cell size if not provided
        if cell_size is None:
            # Try to fit the grid in the window
            cell_size = min(
                window_size[0] // ant.width,
                window_size[1] // ant.height
            )
            # Ensure cell size is at least 1 but not too large
            cell_size = max(1, min(cell_size, 20))

        self.cell_size = cell_size

        # Calculate viewport (which part of the grid to show)
        self.viewport_x = 0
        self.viewport_y = 0
        self.update_viewport()

        # Create window
        self.screen = pygame.display.set_mode(window_size)
        pygame.display.set_caption("Langton's Ant Simulation")

        # Font for status text
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 18)

        # Clock for controlling frame rate
        self.clock = pygame.time.Clock()
        self.fps = 60

    def update_viewport(self) -> None:
        """Update viewport to center on ant position"""
        # Calculate how many cells fit in the window
        cells_x = self.window_size[0] // self.cell_size
        cells_y = self.window_size[1] // self.cell_size

        # Center viewport on ant
        self.viewport_x = max(0, min(self.ant.ant_x - cells_x // 2, self.ant.width - cells_x))
        self.viewport_y = max(0, min(self.ant.ant_y - cells_y // 2, self.ant.height - cells_y))

        # Ensure viewport is valid
        self.viewport_x = max(0, self.viewport_x)
        self.viewport_y = max(0, self.viewport_y)

    def should_update_viewport(self) -> bool:
        """Check if viewport needs updating (ant near edge of visible area)"""
        # Calculate how many cells fit in the window
        cells_x = self.window_size[0] // self.cell_size
        cells_y = self.window_size[1] // self.cell_size

        # Define margin - only update if ant is within this distance from edge
        margin = min(cells_x // 4, cells_y // 4, 20)  # 25% of viewport or 20 cells

        # Calculate ant position relative to viewport
        ant_viewport_x = self.ant.ant_x - self.viewport_x
        ant_viewport_y = self.ant.ant_y - self.viewport_y

        # Check if ant is near edges
        near_left = ant_viewport_x < margin
        near_right = ant_viewport_x > cells_x - margin
        near_top = ant_viewport_y < margin
        near_bottom = ant_viewport_y > cells_y - margin

        return near_left or near_right or near_top or near_bottom

    def handle_events(self) -> None:
        """Handle pygame events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.paused = not self.paused

                elif event.key == pygame.K_UP:
                    # Increase speed
                    self.steps_per_frame = min(self.steps_per_frame * 2, 1000)

                elif event.key == pygame.K_DOWN:
                    # Decrease speed
                    self.steps_per_frame = max(self.steps_per_frame // 2, 1)

                elif event.key == pygame.K_q or event.key == pygame.K_ESCAPE:
                    self.running = False

                elif event.key == pygame.K_EQUALS or event.key == pygame.K_PLUS:
                    # Zoom in
                    self.cell_size = min(self.cell_size + 1, 20)

                elif event.key == pygame.K_MINUS:
                    # Zoom out
                    self.cell_size = max(self.cell_size - 1, 1)

    def draw_grid(self) -> None:
        """Draw the grid and ant"""
        # Fill background
        self.screen.fill(self.COLOR_BG)

        # Calculate visible area
        cells_x = self.window_size[0] // self.cell_size
        cells_y = self.window_size[1] // self.cell_size

        # Draw cells
        for grid_y in range(self.viewport_y, min(self.viewport_y + cells_y + 1, self.ant.height)):
            for grid_x in range(self.viewport_x, min(self.viewport_x + cells_x + 1, self.ant.width)):
                # Calculate screen position
                screen_x = (grid_x - self.viewport_x) * self.cell_size
                screen_y = (grid_y - self.viewport_y) * self.cell_size

                # Draw cell if it's black
                if self.ant.grid[grid_y, grid_x] == 1:
                    pygame.draw.rect(
                        self.screen,
                        self.COLOR_BLACK,
                        (screen_x, screen_y, self.cell_size, self.cell_size)
                    )
                else:
                    pygame.draw.rect(
                        self.screen,
                        self.COLOR_WHITE,
                        (screen_x, screen_y, self.cell_size, self.cell_size)
                    )

        # Draw ant
        ant_screen_x = (self.ant.ant_x - self.viewport_x) * self.cell_size
        ant_screen_y = (self.ant.ant_y - self.viewport_y) * self.cell_size

        # Draw ant as a colored square with a direction indicator
        pygame.draw.rect(
            self.screen,
            self.COLOR_ANT,
            (ant_screen_x, ant_screen_y, self.cell_size, self.cell_size)
        )

        # Draw direction indicator (small triangle)
        if self.cell_size >= 5:
            center_x = ant_screen_x + self.cell_size // 2
            center_y = ant_screen_y + self.cell_size // 2
            size = self.cell_size // 3

            # Direction vectors for triangle
            if self.ant.ant_direction == 0:  # North
                points = [
                    (center_x, center_y - size),
                    (center_x - size // 2, center_y + size // 2),
                    (center_x + size // 2, center_y + size // 2)
                ]
            elif self.ant.ant_direction == 1:  # East
                points = [
                    (center_x + size, center_y),
                    (center_x - size // 2, center_y - size // 2),
                    (center_x - size // 2, center_y + size // 2)
                ]
            elif self.ant.ant_direction == 2:  # South
                points = [
                    (center_x, center_y + size),
                    (center_x - size // 2, center_y - size // 2),
                    (center_x + size // 2, center_y - size // 2)
                ]
            else:  # West
                points = [
                    (center_x - size, center_y),
                    (center_x + size // 2, center_y - size // 2),
                    (center_x + size // 2, center_y + size // 2)
                ]

            pygame.draw.polygon(self.screen, self.COLOR_WHITE, points)

    def draw_status(self) -> None:
        """Draw status information"""
        state = self.ant.get_state()

        # Create status lines
        status_lines = [
            f"Step: {state['step_count']}",
            f"Position: ({state['ant_position'][0]}, {state['ant_position'][1]})",
            f"Direction: {state['ant_direction_name']}",
            f"Grid: {state['grid_size'][0]}x{state['grid_size'][1]}",
            f"Expansions: {state['expansion_count']}",
            f"Speed: {self.steps_per_frame} steps/frame",
        ]

        if state['highway_detected']:
            status_lines.append(f"Highway: {state['highway_direction']}")
        else:
            status_lines.append("Highway: Searching...")

        if self.paused:
            status_lines.append("PAUSED")

        # Draw semi-transparent background for status
        status_height = len(status_lines) * 25 + 10
        status_surface = pygame.Surface((300, status_height))
        status_surface.set_alpha(200)
        status_surface.fill((50, 50, 50))
        self.screen.blit(status_surface, (10, 10))

        # Draw status text
        y_offset = 15
        for line in status_lines:
            if line.startswith("PAUSED"):
                text_surface = self.font.render(line, True, (255, 255, 0))
            elif line.startswith("Highway:") and state['highway_detected']:
                text_surface = self.font.render(line, True, (0, 255, 0))
            else:
                text_surface = self.font.render(line, True, (255, 255, 255))

            self.screen.blit(text_surface, (15, y_offset))
            y_offset += 25

        # Draw controls help at bottom
        controls = [
            "Controls:",
            "SPACE: Pause/Unpause",
            "UP/DOWN: Speed",
            "+/-: Zoom",
            "Q/ESC: Quit"
        ]

        help_height = len(controls) * 20 + 10
        help_surface = pygame.Surface((250, help_height))
        help_surface.set_alpha(180)
        help_surface.fill((50, 50, 50))
        self.screen.blit(help_surface, (10, self.window_size[1] - help_height - 10))

        y_offset = self.window_size[1] - help_height - 5
        for line in controls:
            text_surface = self.small_font.render(line, True, (255, 255, 255))
            self.screen.blit(text_surface, (15, y_offset))
            y_offset += 20

    def run(self) -> None:
        """Run the visualization loop"""
        while self.running:
            self.handle_events()

            # Update simulation if not paused
            if not self.paused:
                for _ in range(self.steps_per_frame):
                    self.ant.step()
                    # Stop if highway detected (optional)
                    # if self.ant.highway_detector.is_highway_detected():
                    #     self.paused = True
                    #     break

                # Update viewport to follow ant (only when near edge)
                if self.should_update_viewport():
                    self.update_viewport()

            # Draw everything
            self.draw_grid()
            self.draw_status()

            # Update display
            pygame.display.flip()
            self.clock.tick(self.fps)

        pygame.quit()

    def run_until_highway(self, max_steps: int = 100000) -> Optional[str]:
        """
        Run visualization until highway is detected or max steps reached.

        Args:
            max_steps: Maximum steps to run

        Returns:
            Highway direction or None
        """
        while self.running and self.ant.step_count < max_steps:
            self.handle_events()

            # Update simulation if not paused
            if not self.paused:
                for _ in range(self.steps_per_frame):
                    self.ant.step()

                    # Check if highway detected
                    if self.ant.highway_detector.is_highway_detected():
                        self.paused = True
                        highway_dir = self.ant.highway_detector.get_highway_direction()

                        # Continue showing visualization but paused
                        while self.running:
                            self.handle_events()
                            self.draw_grid()
                            self.draw_status()
                            pygame.display.flip()
                            self.clock.tick(self.fps)

                        pygame.quit()
                        return highway_dir

                    if self.ant.step_count >= max_steps:
                        break

                # Update viewport to follow ant (only when near edge)
                if self.should_update_viewport():
                    self.update_viewport()

            # Draw everything
            self.draw_grid()
            self.draw_status()

            # Update display
            pygame.display.flip()
            self.clock.tick(self.fps)

        pygame.quit()
        return self.ant.highway_detector.get_highway_direction()
