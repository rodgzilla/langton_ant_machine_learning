"""Unit tests for highway detection"""

import pytest
from langton_ant.highway_detector import HighwayDetector
from langton_ant.simulation import LangtonAnt


class TestHighwayDetectorBasics:
    """Test basic highway detector functionality"""

    def test_initialization(self):
        """Test detector initializes correctly"""
        detector = HighwayDetector()

        assert not detector.is_highway_detected()
        assert detector.get_highway_direction() is None
        assert len(detector.positions) == 0

    def test_add_position(self):
        """Test adding positions to detector"""
        detector = HighwayDetector()

        detector.add_position(0, 0, 0)
        assert len(detector.positions) == 1

        detector.add_position(1, 0, 1)
        assert len(detector.positions) == 2

    def test_reset(self):
        """Test resetting detector state"""
        detector = HighwayDetector()

        detector.add_position(0, 0, 0)
        detector.add_position(1, 0, 1)

        detector.reset()

        assert len(detector.positions) == 0
        assert len(detector.directions) == 0
        assert not detector.is_highway_detected()
        assert detector.get_highway_direction() is None

    def test_no_detection_with_insufficient_data(self):
        """Detector should not detect highway with insufficient position data"""
        detector = HighwayDetector()

        # Add only a few positions
        for i in range(100):
            detector.add_position(i, i, 0)

        # Should not detect highway with so few positions
        assert not detector.is_highway_detected()


class TestHighwayDetectionIntegration:
    """Test highway detection with actual simulation"""

    def test_standard_langton_ant_eventually_forms_highway(self):
        """Standard Langton's Ant (empty grid, center start) should form highway"""
        ant = LangtonAnt(initial_size=100, start_position=(50, 50), start_direction=0)

        # Run for a reasonable number of steps
        # Standard Langton's Ant typically forms highway around 10,000-11,000 steps
        max_steps = 15000

        highway_direction = ant.run_until_highway(max_steps=max_steps, check_interval=200)

        # Should have detected a highway
        assert highway_direction is not None
        assert highway_direction in ['NE', 'NW', 'SE', 'SW']

    def test_highway_detected_flag_is_set(self):
        """When highway is detected, the flag should be set"""
        ant = LangtonAnt(initial_size=100, start_position=(50, 50), start_direction=0)

        # Run until highway detected
        ant.run_until_highway(max_steps=15000, check_interval=200)

        state = ant.get_state()
        if state['highway_detected']:
            assert state['highway_direction'] is not None
            assert state['highway_direction'] in ['NE', 'NW', 'SE', 'SW']

    def test_different_starting_positions_produce_highways(self):
        """Different starting positions should still produce highways"""
        starting_positions = [(30, 30), (50, 50), (70, 70)]

        for pos in starting_positions:
            ant = LangtonAnt(initial_size=150, start_position=pos, start_direction=0)
            highway_direction = ant.run_until_highway(max_steps=15000, check_interval=200)

            # Each should eventually form a highway
            assert highway_direction is not None

    def test_different_starting_directions_produce_highways(self):
        """Different starting directions should still produce highways"""
        for direction in range(4):
            ant = LangtonAnt(initial_size=100, start_position=(50, 50), start_direction=direction)
            highway_direction = ant.run_until_highway(max_steps=15000, check_interval=200)

            # Each should eventually form a highway
            assert highway_direction is not None


class TestHighwayDirectionClassification:
    """Test that highway directions are correctly classified"""

    def test_direction_is_consistent(self):
        """Same initial conditions should produce same highway direction"""
        ant1 = LangtonAnt(initial_size=100, start_position=(50, 50), start_direction=0)
        ant2 = LangtonAnt(initial_size=100, start_position=(50, 50), start_direction=0)

        dir1 = ant1.run_until_highway(max_steps=15000, check_interval=200)
        dir2 = ant2.run_until_highway(max_steps=15000, check_interval=200)

        assert dir1 == dir2

    def test_highway_direction_valid(self):
        """Highway direction should be one of the four valid directions"""
        ant = LangtonAnt(initial_size=100, start_position=(50, 50), start_direction=0)
        highway_direction = ant.run_until_highway(max_steps=15000, check_interval=200)

        assert highway_direction in ['NE', 'NW', 'SE', 'SW']


class TestEdgeCases:
    """Test edge cases and boundary conditions"""

    def test_no_highway_detected_within_max_steps(self):
        """If max_steps is too small, no highway should be detected"""
        ant = LangtonAnt(initial_size=50, start_position=(25, 25), start_direction=0)

        # Run for very few steps (highway takes ~10k+ steps)
        highway_direction = ant.run_until_highway(max_steps=100, check_interval=50)

        # Should not have detected highway
        assert highway_direction is None
        assert not ant.highway_detector.is_highway_detected()

    def test_detector_maxlen_respected(self):
        """Position deque should respect maxlen"""
        detector = HighwayDetector()
        maxlen = detector.positions.maxlen

        # Add more positions than maxlen
        for i in range(maxlen + 100):
            detector.add_position(i, i, 0)

        # Length should not exceed maxlen
        assert len(detector.positions) == maxlen

    def test_small_grid_with_expansion(self):
        """Highway detection should work even with grid expansion"""
        ant = LangtonAnt(initial_size=50, start_position=(25, 25), start_direction=0, expansion_margin=10)

        highway_direction = ant.run_until_highway(max_steps=15000, check_interval=200)

        # Should still detect highway despite expansions
        assert highway_direction is not None

        # Should have had some expansions
        assert ant.expansion_count > 0


class TestPerformance:
    """Test performance characteristics"""

    def test_highway_detected_in_reasonable_time(self):
        """Highway should be detected within expected step count"""
        ant = LangtonAnt(initial_size=100, start_position=(50, 50), start_direction=0)

        highway_direction = ant.run_until_highway(max_steps=15000, check_interval=200)

        # Should detect highway
        assert highway_direction is not None

        # Should detect within expected range (typically 10k-12k steps)
        # Allow some margin for variations
        assert ant.step_count < 15000
