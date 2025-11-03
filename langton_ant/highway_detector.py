"""Highway pattern detection for Langton's Ant"""

from collections import deque
from typing import Optional, Tuple


class HighwayDetector:
    """
    Detects when Langton's Ant enters the "highway" phase.

    The highway is a repeating 104-step pattern where the ant moves diagonally.
    This detector tracks the ant's position history and checks for this pattern.
    """

    HIGHWAY_PERIOD = 104  # The highway repeats every 104 steps
    CYCLES_TO_CONFIRM = 3  # Number of cycles to observe before confirming

    def __init__(self):
        """Initialize the highway detector."""
        self.positions = deque(maxlen=self.HIGHWAY_PERIOD * self.CYCLES_TO_CONFIRM * 2)
        self.directions = deque(maxlen=self.HIGHWAY_PERIOD * self.CYCLES_TO_CONFIRM * 2)
        self._highway_detected = False
        self._highway_direction = None

    def add_position(self, x: int, y: int, direction: int) -> None:
        """
        Add a new position to the history.

        Args:
            x: Current x position
            y: Current y position
            direction: Current direction (0-3)
        """
        self.positions.append((x, y))
        self.directions.append(direction)

        # Check for highway pattern periodically
        if len(self.positions) >= self.HIGHWAY_PERIOD * self.CYCLES_TO_CONFIRM:
            if not self._highway_detected:
                self._check_for_highway()

    def _check_for_highway(self) -> None:
        """
        Check if the recent position history shows a highway pattern.

        A highway is confirmed when we see the same relative displacement pattern
        repeat for CYCLES_TO_CONFIRM consecutive cycles.
        """
        if len(self.positions) < self.HIGHWAY_PERIOD * self.CYCLES_TO_CONFIRM:
            return

        # Get the last CYCLES_TO_CONFIRM complete cycles
        positions_list = list(self.positions)
        directions_list = list(self.directions)

        # Extract displacement patterns for each cycle
        cycle_displacements = []
        for cycle_idx in range(self.CYCLES_TO_CONFIRM):
            start_idx = -(self.CYCLES_TO_CONFIRM - cycle_idx) * self.HIGHWAY_PERIOD
            end_idx = start_idx + self.HIGHWAY_PERIOD
            if end_idx == 0:
                end_idx = None

            cycle_positions = positions_list[start_idx:end_idx]
            cycle_directions = directions_list[start_idx:end_idx]

            if len(cycle_positions) < self.HIGHWAY_PERIOD:
                return

            # Calculate displacement pattern for this cycle
            displacements = []
            for i in range(len(cycle_positions) - 1):
                dx = cycle_positions[i + 1][0] - cycle_positions[i][0]
                dy = cycle_positions[i + 1][1] - cycle_positions[i][1]
                displacements.append((dx, dy, cycle_directions[i]))

            cycle_displacements.append(displacements)

        # Check if all cycles have the same pattern
        if self._patterns_match(cycle_displacements):
            self._highway_detected = True
            self._highway_direction = self._calculate_direction(cycle_displacements[0])

    def _patterns_match(self, cycle_displacements: list) -> bool:
        """
        Check if all displacement patterns match.

        Args:
            cycle_displacements: List of displacement patterns, one per cycle

        Returns:
            True if all patterns match
        """
        if len(cycle_displacements) < 2:
            return False

        reference_pattern = cycle_displacements[0]

        for pattern in cycle_displacements[1:]:
            if len(pattern) != len(reference_pattern):
                return False

            # Allow small variations due to grid expansion or other effects
            mismatches = sum(
                1 for i in range(len(pattern))
                if pattern[i] != reference_pattern[i]
            )

            # If more than 5% of steps don't match, patterns are different
            if mismatches > len(pattern) * 0.05:
                return False

        return True

    def _calculate_direction(self, displacement_pattern: list) -> str:
        """
        Calculate the overall direction of the highway from displacement pattern.

        Args:
            displacement_pattern: List of (dx, dy, direction) tuples for one cycle

        Returns:
            Direction string: 'NE', 'NW', 'SE', or 'SW'
        """
        # Calculate net displacement over one cycle
        total_dx = sum(dx for dx, dy, _ in displacement_pattern)
        total_dy = sum(dy for dx, dy, _ in displacement_pattern)

        # Determine direction based on net displacement
        # Note: In screen coordinates, y increases downward
        if total_dx > 0 and total_dy < 0:
            return 'NE'  # Right and up
        elif total_dx < 0 and total_dy < 0:
            return 'NW'  # Left and up
        elif total_dx > 0 and total_dy > 0:
            return 'SE'  # Right and down
        elif total_dx < 0 and total_dy > 0:
            return 'SW'  # Left and down
        else:
            # No clear direction (shouldn't happen for valid highway)
            # Default based on predominant direction
            if abs(total_dx) > abs(total_dy):
                return 'NE' if total_dx > 0 else 'NW'
            else:
                return 'SE' if total_dy > 0 else 'SW'

    def is_highway_detected(self) -> bool:
        """Return True if highway has been detected."""
        return self._highway_detected

    def get_highway_direction(self) -> Optional[str]:
        """
        Get the direction of the detected highway.

        Returns:
            Direction string ('NE', 'NW', 'SE', 'SW') or None if not detected
        """
        return self._highway_direction

    def reset(self) -> None:
        """Reset the detector state."""
        self.positions.clear()
        self.directions.clear()
        self._highway_detected = False
        self._highway_direction = None
