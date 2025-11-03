"""Unit tests for dataset management"""

import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path
from langton_ant.dataset import SimulationConfig, SimulationResult, DatasetGenerator


class TestSimulationConfig:
    """Test SimulationConfig functionality"""

    def test_initialization(self):
        """Test basic initialization"""
        config = SimulationConfig(
            start_position=(50, 50),
            start_direction=1,
            grid_size=100
        )

        assert config.start_position == (50, 50)
        assert config.start_direction == 1
        assert config.grid_size == 100
        assert config.initial_grid is None

    def test_initialization_with_grid(self):
        """Test initialization with custom grid"""
        grid = np.zeros((30, 30), dtype=np.uint8)
        grid[10:20, 10:20] = 1

        config = SimulationConfig(
            start_position=(15, 15),
            start_direction=2,
            initial_grid=grid
        )

        assert config.initial_grid is not None
        assert np.array_equal(config.initial_grid, grid)

    def test_to_dict_without_grid(self):
        """Test conversion to dictionary without grid"""
        config = SimulationConfig(
            start_position=(25, 30),
            start_direction=3,
            grid_size=50
        )

        config_dict = config.to_dict()

        assert config_dict['start_position'] == [25, 30]
        assert config_dict['start_direction'] == 3
        assert config_dict['grid_size'] == 50
        assert config_dict['has_initial_grid'] is False

    def test_to_dict_with_grid(self):
        """Test conversion to dictionary with grid"""
        grid = np.ones((20, 20), dtype=np.uint8)
        config = SimulationConfig(
            start_position=(10, 10),
            start_direction=0,
            initial_grid=grid
        )

        config_dict = config.to_dict()

        assert config_dict['has_initial_grid'] is True
        assert config_dict['initial_grid_shape'] == [20, 20]

    def test_from_dict(self):
        """Test creating config from dictionary"""
        config_dict = {
            'start_position': [40, 45],
            'start_direction': 2,
            'grid_size': 80,
            'has_initial_grid': False
        }

        config = SimulationConfig.from_dict(config_dict)

        assert config.start_position == (40, 45)
        assert config.start_direction == 2
        assert config.grid_size == 80
        assert config.initial_grid is None

    def test_save_and_load_without_grid(self):
        """Test saving and loading config without grid"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = SimulationConfig(
                start_position=(30, 35),
                start_direction=1,
                grid_size=60
            )

            filepath = Path(tmpdir) / "test_config"
            config.save(filepath)

            # Load it back
            loaded_config = SimulationConfig.load(filepath)

            assert loaded_config.start_position == config.start_position
            assert loaded_config.start_direction == config.start_direction
            assert loaded_config.grid_size == config.grid_size
            assert loaded_config.initial_grid is None

    def test_save_and_load_with_grid(self):
        """Test saving and loading config with grid"""
        with tempfile.TemporaryDirectory() as tmpdir:
            grid = np.random.randint(0, 2, (25, 25), dtype=np.uint8)
            config = SimulationConfig(
                start_position=(12, 12),
                start_direction=3,
                initial_grid=grid
            )

            filepath = Path(tmpdir) / "test_config"
            config.save(filepath)

            # Load it back
            loaded_config = SimulationConfig.load(filepath)

            assert loaded_config.start_position == config.start_position
            assert loaded_config.start_direction == config.start_direction
            assert loaded_config.initial_grid is not None
            assert np.array_equal(loaded_config.initial_grid, grid)


class TestSimulationResult:
    """Test SimulationResult functionality"""

    def test_initialization(self):
        """Test basic initialization"""
        config = SimulationConfig(
            start_position=(50, 50),
            start_direction=0,
            grid_size=100
        )

        result = SimulationResult(
            config=config,
            highway_direction='NE',
            steps_to_highway=10500,
            grid_expansions=2,
            final_grid_size=(200, 200)
        )

        assert result.highway_direction == 'NE'
        assert result.steps_to_highway == 10500
        assert result.grid_expansions == 2
        assert result.final_grid_size == (200, 200)
        assert result.timestamp is not None

    def test_to_dict(self):
        """Test conversion to dictionary"""
        config = SimulationConfig(
            start_position=(50, 50),
            start_direction=1,
            grid_size=100
        )

        result = SimulationResult(
            config=config,
            highway_direction='SW',
            steps_to_highway=11000,
            grid_expansions=3,
            final_grid_size=(250, 250),
            timestamp='2024-01-01T00:00:00'
        )

        result_dict = result.to_dict()

        assert result_dict['highway_direction'] == 'SW'
        assert result_dict['steps_to_highway'] == 11000
        assert result_dict['grid_expansions'] == 3
        assert result_dict['final_grid_size'] == [250, 250]
        assert result_dict['timestamp'] == '2024-01-01T00:00:00'

    def test_save_and_load(self):
        """Test saving and loading result"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = SimulationConfig(
                start_position=(50, 50),
                start_direction=2,
                grid_size=100
            )

            result = SimulationResult(
                config=config,
                highway_direction='NW',
                steps_to_highway=10200,
                grid_expansions=1,
                final_grid_size=(150, 150)
            )

            filepath = Path(tmpdir) / "test_result.json"
            result.save(filepath)

            # Load it back
            loaded_result = SimulationResult.load(filepath)

            assert loaded_result.highway_direction == result.highway_direction
            assert loaded_result.steps_to_highway == result.steps_to_highway
            assert loaded_result.grid_expansions == result.grid_expansions
            assert loaded_result.final_grid_size == result.final_grid_size

    def test_save_and_load_with_grid(self):
        """Test saving and loading result with initial grid"""
        with tempfile.TemporaryDirectory() as tmpdir:
            grid = np.random.randint(0, 2, (30, 30), dtype=np.uint8)
            config = SimulationConfig(
                start_position=(15, 15),
                start_direction=0,
                initial_grid=grid
            )

            result = SimulationResult(
                config=config,
                highway_direction='SE',
                steps_to_highway=10800,
                grid_expansions=2,
                final_grid_size=(180, 180)
            )

            filepath = Path(tmpdir) / "test_result.json"
            result.save(filepath)

            # Load it back
            loaded_result = SimulationResult.load(filepath)

            assert loaded_result.config.initial_grid is not None
            assert np.array_equal(loaded_result.config.initial_grid, grid)

    def test_null_highway_direction(self):
        """Test result with no highway detected"""
        config = SimulationConfig(
            start_position=(50, 50),
            start_direction=0,
            grid_size=100
        )

        result = SimulationResult(
            config=config,
            highway_direction=None,
            steps_to_highway=100000,
            grid_expansions=5,
            final_grid_size=(300, 300)
        )

        assert result.highway_direction is None

        # Should still be saveable
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test_result.json"
            result.save(filepath)

            loaded_result = SimulationResult.load(filepath)
            assert loaded_result.highway_direction is None


class TestDatasetGenerator:
    """Test DatasetGenerator functionality"""

    def test_initialization(self):
        """Test generator initialization"""
        with tempfile.TemporaryDirectory() as tmpdir:
            generator = DatasetGenerator(tmpdir)
            assert generator.output_dir.exists()

    def test_generate_random_config(self):
        """Test random configuration generation"""
        with tempfile.TemporaryDirectory() as tmpdir:
            generator = DatasetGenerator(tmpdir)

            config = generator.generate_random_config(grid_size=50)

            # Check position is within bounds
            assert 0 <= config.start_position[0] < 50
            assert 0 <= config.start_position[1] < 50

            # Check direction is valid
            assert config.start_direction in [0, 1, 2, 3]

            # Check grid size
            assert config.grid_size == 50

    def test_generate_random_config_no_pattern(self):
        """Test random config generation without initial pattern"""
        with tempfile.TemporaryDirectory() as tmpdir:
            generator = DatasetGenerator(tmpdir)

            # Generate multiple configs, none should have patterns
            for _ in range(10):
                config = generator.generate_random_config(
                    grid_size=40,
                    allow_initial_pattern=False
                )
                assert config.initial_grid is None

    def test_run_simulation(self):
        """Test running a single simulation"""
        with tempfile.TemporaryDirectory() as tmpdir:
            generator = DatasetGenerator(tmpdir)

            config = SimulationConfig(
                start_position=(50, 50),
                start_direction=0,
                grid_size=100
            )

            # Run simulation (use short timeout for testing)
            result = generator.run_simulation(config, max_steps=1000, check_interval=100)

            assert result.config.start_position == config.start_position
            assert result.config.start_direction == config.start_direction
            assert result.steps_to_highway <= 1000

    def test_generate_small_dataset(self):
        """Test generating a small dataset"""
        with tempfile.TemporaryDirectory() as tmpdir:
            generator = DatasetGenerator(tmpdir)

            # Generate small dataset with short timeout
            results = generator.generate_dataset(
                num_simulations=3,
                grid_size=50,
                max_steps=1000,
                check_interval=100,
                allow_initial_pattern=False
            )

            assert len(results) == 3

            # Check files were created
            json_files = list(generator.output_dir.glob("sim_*.json"))
            assert len(json_files) == 3

    def test_load_dataset(self):
        """Test loading a saved dataset"""
        with tempfile.TemporaryDirectory() as tmpdir:
            generator = DatasetGenerator(tmpdir)

            # Generate dataset
            generator.generate_dataset(
                num_simulations=5,
                grid_size=50,
                max_steps=1000,
                check_interval=100,
                allow_initial_pattern=False
            )

            # Load it back
            loaded_results = generator.load_dataset()

            assert len(loaded_results) == 5

            # Each result should be valid
            for result in loaded_results:
                assert result.config is not None
                assert result.steps_to_highway > 0

    def test_custom_prefix(self):
        """Test using custom prefix for filenames"""
        with tempfile.TemporaryDirectory() as tmpdir:
            generator = DatasetGenerator(tmpdir)

            generator.generate_dataset(
                num_simulations=2,
                grid_size=40,
                max_steps=500,
                check_interval=100,
                prefix="test"
            )

            # Check files use custom prefix
            test_files = list(generator.output_dir.glob("test_*.json"))
            assert len(test_files) == 2

    def test_results_are_reproducible(self):
        """Test that same config produces same results"""
        with tempfile.TemporaryDirectory() as tmpdir:
            generator = DatasetGenerator(tmpdir)

            config = SimulationConfig(
                start_position=(50, 50),
                start_direction=0,
                grid_size=100
            )

            result1 = generator.run_simulation(config, max_steps=5000, check_interval=200)
            result2 = generator.run_simulation(config, max_steps=5000, check_interval=200)

            # Same config should produce same results
            assert result1.highway_direction == result2.highway_direction
            assert result1.steps_to_highway == result2.steps_to_highway


class TestIntegration:
    """Integration tests for dataset pipeline"""

    def test_end_to_end_dataset_creation(self):
        """Test complete dataset creation and loading pipeline"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create generator
            generator = DatasetGenerator(tmpdir)

            # Generate small dataset
            num_sims = 5
            results = generator.generate_dataset(
                num_simulations=num_sims,
                grid_size=50,
                max_steps=2000,
                check_interval=200,
                allow_initial_pattern=True,
                pattern_density=0.05
            )

            # Verify results
            assert len(results) == num_sims

            # Load dataset back
            loaded_results = generator.load_dataset()
            assert len(loaded_results) == num_sims

            # Compare results
            for original, loaded in zip(results, loaded_results):
                assert original.highway_direction == loaded.highway_direction
                assert original.steps_to_highway == loaded.steps_to_highway
                assert original.config.start_position == loaded.config.start_position
                assert original.config.start_direction == loaded.config.start_direction
