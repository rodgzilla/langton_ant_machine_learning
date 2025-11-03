"""Unit tests for the core Langton's Ant simulation"""

import pytest
import numpy as np
from langton_ant.simulation import LangtonAnt


class TestBasicRules:
    """Test that basic Langton's Ant rules are correctly implemented"""

    def test_turn_right_on_white_cell(self):
        """Ant should turn right when on a white (0) cell"""
        ant = LangtonAnt(initial_size=10, start_position=(5, 5), start_direction=0)

        # Ant starts on white cell facing North
        assert ant.grid[5, 5] == 0
        assert ant.ant_direction == 0  # North

        ant.step()

        # After step: should have turned right (to East)
        assert ant.ant_direction == 1  # East

    def test_turn_left_on_black_cell(self):
        """Ant should turn left when on a black (1) cell"""
        ant = LangtonAnt(initial_size=10, start_position=(5, 5), start_direction=0)

        # Set starting cell to black
        ant.grid[5, 5] = 1
        assert ant.ant_direction == 0  # North

        ant.step()

        # After step: should have turned left (to West)
        assert ant.ant_direction == 3  # West

    def test_cell_flip_white_to_black(self):
        """White cell should flip to black after ant visits"""
        ant = LangtonAnt(initial_size=10, start_position=(5, 5), start_direction=0, expansion_margin=2)

        # Cell starts white
        assert ant.grid[5, 5] == 0

        ant.step()

        # Cell should now be black
        assert ant.grid[5, 5] == 1

    def test_cell_flip_black_to_white(self):
        """Black cell should flip to white after ant visits"""
        ant = LangtonAnt(initial_size=10, start_position=(5, 5), start_direction=0)

        # Set cell to black
        ant.grid[5, 5] = 1

        ant.step()

        # Cell should now be white
        assert ant.grid[5, 5] == 0

    def test_move_forward_north(self):
        """Ant should move up (negative y) when facing North"""
        ant = LangtonAnt(initial_size=10, start_position=(5, 5), start_direction=1, expansion_margin=2)
        ant.grid[5, 5] = 1  # Make it turn left to face North

        ant.step()

        # Should have moved to (5, 4) - up one row
        assert ant.ant_x == 5
        assert ant.ant_y == 4

    def test_move_forward_east(self):
        """Ant should move right (positive x) when facing East"""
        ant = LangtonAnt(initial_size=10, start_position=(5, 5), start_direction=0, expansion_margin=2)
        # Starts on white, will turn right to East

        ant.step()

        # Should have moved to (6, 5) - right one column
        assert ant.ant_x == 6
        assert ant.ant_y == 5

    def test_move_forward_south(self):
        """Ant should move down (positive y) when facing South"""
        ant = LangtonAnt(initial_size=10, start_position=(5, 5), start_direction=3, expansion_margin=2)
        ant.grid[5, 5] = 1  # Make it turn left to face South

        ant.step()

        # Should have moved to (5, 6) - down one row
        assert ant.ant_x == 5
        assert ant.ant_y == 6

    def test_move_forward_west(self):
        """Ant should move left (negative x) when facing West"""
        ant = LangtonAnt(initial_size=10, start_position=(5, 5), start_direction=2, expansion_margin=2)
        # Starts on white, will turn right to West

        ant.step()

        # Should have moved to (4, 5) - left one column
        assert ant.ant_x == 4
        assert ant.ant_y == 5

    def test_direction_wraparound(self):
        """Test that direction correctly wraps around (0-3)"""
        ant = LangtonAnt(initial_size=10, start_position=(5, 5), start_direction=3)

        # Facing West, turn right should wrap to North
        ant.step()
        assert ant.ant_direction == 0  # North

        # Reset, test left turn from North
        ant2 = LangtonAnt(initial_size=10, start_position=(5, 5), start_direction=0)
        ant2.grid[5, 5] = 1  # Make it turn left
        ant2.step()
        assert ant2.ant_direction == 3  # West


class TestGridExpansion:
    """Test grid expansion functionality"""

    def test_grid_expands_when_approaching_right_edge(self):
        """Grid should expand when ant approaches right edge"""
        ant = LangtonAnt(initial_size=20, start_position=(15, 10), start_direction=1, expansion_margin=5)

        initial_width = ant.width

        # Move ant closer to right edge
        for _ in range(10):
            ant.step()

        # Grid should have expanded
        assert ant.width > initial_width

    def test_grid_expands_when_approaching_bottom_edge(self):
        """Grid should expand when ant approaches bottom edge"""
        ant = LangtonAnt(initial_size=20, start_position=(10, 15), start_direction=2, expansion_margin=5)

        initial_height = ant.height

        # Move ant closer to bottom edge
        for _ in range(10):
            ant.step()

        # Grid should have expanded
        assert ant.height > initial_height

    def test_ant_position_adjusted_after_expansion(self):
        """Ant position should be correctly adjusted when grid expands"""
        ant = LangtonAnt(initial_size=20, start_position=(2, 10), start_direction=3, expansion_margin=5)

        # Record position before expansion
        pos_before = (ant.ant_x, ant.ant_y)

        # Move towards left edge to trigger expansion
        for _ in range(5):
            ant.step()

        # Ant should still be in valid position
        assert 0 <= ant.ant_x < ant.width
        assert 0 <= ant.ant_y < ant.height

    def test_cell_states_preserved_during_expansion(self):
        """Cell states should be preserved when grid expands"""
        ant = LangtonAnt(initial_size=20, start_position=(15, 10), start_direction=1, expansion_margin=5)

        # Create a known pattern
        ant.grid[5, 5] = 1
        ant.grid[5, 6] = 1
        ant.grid[6, 5] = 1

        initial_width = ant.width

        # Trigger expansion
        for _ in range(15):
            ant.step()

        if ant.width > initial_width:
            # Pattern should still exist (may be shifted due to expansion)
            # Check that total number of black cells is at least what we set
            # (plus any created by ant movement)
            assert np.sum(ant.grid) >= 3

    def test_expansion_count_increments(self):
        """Expansion count should increment when grid expands"""
        ant = LangtonAnt(initial_size=20, start_position=(2, 2), start_direction=3, expansion_margin=5)

        assert ant.expansion_count == 0

        # Trigger expansion
        for _ in range(10):
            ant.step()

        assert ant.expansion_count > 0


class TestInitialConditions:
    """Test initialization with different starting conditions"""

    def test_default_initialization(self):
        """Test default initialization parameters"""
        ant = LangtonAnt()

        assert ant.width == 100
        assert ant.height == 100
        assert ant.ant_x == 50
        assert ant.ant_y == 50
        assert ant.ant_direction == 0  # North
        assert ant.step_count == 0

    def test_custom_starting_position(self):
        """Test custom starting position"""
        ant = LangtonAnt(initial_size=50, start_position=(10, 20))

        assert ant.ant_x == 10
        assert ant.ant_y == 20

    def test_custom_starting_direction(self):
        """Test all four starting directions"""
        for direction in range(4):
            ant = LangtonAnt(initial_size=10, start_direction=direction)
            assert ant.ant_direction == direction

    def test_invalid_starting_direction_raises_error(self):
        """Invalid starting direction should raise ValueError"""
        with pytest.raises(ValueError):
            LangtonAnt(start_direction=4)

        with pytest.raises(ValueError):
            LangtonAnt(start_direction=-1)

    def test_custom_initial_grid(self):
        """Test initialization with custom grid pattern"""
        custom_grid = np.zeros((30, 30), dtype=np.uint8)
        custom_grid[10:20, 10:20] = 1  # 10x10 black square

        ant = LangtonAnt(initial_grid=custom_grid, start_position=(15, 15))

        assert ant.width == 30
        assert ant.height == 30
        assert ant.grid[15, 15] == 1  # Should be black
        assert np.sum(ant.grid) == 100  # Should have 100 black cells

    def test_initial_grid_is_copied(self):
        """Initial grid should be copied, not referenced"""
        original_grid = np.zeros((10, 10), dtype=np.uint8)
        ant = LangtonAnt(initial_grid=original_grid)

        # Modify ant's grid
        ant.grid[5, 5] = 1

        # Original should be unchanged
        assert original_grid[5, 5] == 0


class TestStateManagement:
    """Test state tracking and retrieval"""

    def test_step_count_increments(self):
        """Step count should increment with each step"""
        ant = LangtonAnt(initial_size=10, start_position=(5, 5))

        assert ant.step_count == 0

        ant.step()
        assert ant.step_count == 1

        ant.step()
        assert ant.step_count == 2

        for _ in range(98):
            ant.step()
        assert ant.step_count == 100

    def test_get_state_returns_correct_info(self):
        """get_state should return accurate simulation state"""
        ant = LangtonAnt(initial_size=50, start_position=(25, 25), start_direction=2)

        state = ant.get_state()

        assert state['step_count'] == 0
        assert state['ant_position'] == (25, 25)
        assert state['ant_direction'] == 2
        assert state['ant_direction_name'] == 'S'
        assert state['grid_size'] == (50, 50)
        assert state['expansion_count'] == 0
        assert state['highway_detected'] == False
        assert state['highway_direction'] is None

    def test_get_grid_copy_returns_copy(self):
        """get_grid_copy should return a copy, not reference"""
        ant = LangtonAnt(initial_size=10)

        grid_copy = ant.get_grid_copy()
        grid_copy[5, 5] = 1

        # Original grid should be unchanged
        assert ant.grid[5, 5] == 0

    def test_direction_names_mapping(self):
        """Direction names should be correctly mapped"""
        assert LangtonAnt.DIRECTION_NAMES[0] == 'N'
        assert LangtonAnt.DIRECTION_NAMES[1] == 'E'
        assert LangtonAnt.DIRECTION_NAMES[2] == 'S'
        assert LangtonAnt.DIRECTION_NAMES[3] == 'W'


class TestMultipleSteps:
    """Test behavior over multiple steps"""

    def test_known_sequence(self):
        """Test a known short sequence of moves"""
        # Starting on empty grid at center facing North
        ant = LangtonAnt(initial_size=20, start_position=(10, 10), start_direction=0)

        positions_visited = [(ant.ant_x, ant.ant_y)]

        # Execute a few steps and track
        for _ in range(4):
            ant.step()
            positions_visited.append((ant.ant_x, ant.ant_y))

        # After 4 steps on empty grid, ant should have made a square pattern
        # Start: (10, 10), N, white -> turn R to E, flip, move to (11, 10)
        # Step 2: (11, 10), E, white -> turn R to S, flip, move to (11, 11)
        # Step 3: (11, 11), S, white -> turn R to W, flip, move to (10, 11)
        # Step 4: (10, 11), W, white -> turn R to N, flip, move to (10, 10)

        assert positions_visited[0] == (10, 10)
        assert positions_visited[1] == (11, 10)
        assert positions_visited[2] == (11, 11)
        assert positions_visited[3] == (10, 11)
        assert positions_visited[4] == (10, 10)

    def test_deterministic_behavior(self):
        """Same initial conditions should produce same results"""
        ant1 = LangtonAnt(initial_size=50, start_position=(25, 25), start_direction=1)
        ant2 = LangtonAnt(initial_size=50, start_position=(25, 25), start_direction=1)

        for _ in range(100):
            ant1.step()
            ant2.step()

        assert ant1.ant_x == ant2.ant_x
        assert ant1.ant_y == ant2.ant_y
        assert ant1.ant_direction == ant2.ant_direction
        assert np.array_equal(ant1.grid, ant2.grid)
