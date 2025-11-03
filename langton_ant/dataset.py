"""Dataset generation and management for Langton's Ant simulations"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple, List
from datetime import datetime
from .simulation import LangtonAnt


class SimulationConfig:
    """Configuration for a single simulation run"""

    def __init__(
        self,
        start_position: Tuple[int, int],
        start_direction: int,
        initial_grid: Optional[np.ndarray] = None,
        grid_size: Optional[int] = None
    ):
        """
        Initialize simulation configuration.

        Args:
            start_position: (x, y) starting position
            start_direction: Starting direction (0-3)
            initial_grid: Optional initial grid pattern
            grid_size: Grid size if initial_grid is None
        """
        self.start_position = start_position
        self.start_direction = start_direction
        self.initial_grid = initial_grid
        self.grid_size = grid_size if grid_size is not None else 100

    def to_dict(self) -> Dict:
        """Convert configuration to dictionary for JSON serialization"""
        config_dict = {
            'start_position': list(self.start_position),
            'start_direction': int(self.start_direction),
            'grid_size': int(self.grid_size)
        }

        if self.initial_grid is not None:
            # Store grid shape, we'll save the actual grid separately as .npy
            config_dict['has_initial_grid'] = True
            config_dict['initial_grid_shape'] = list(self.initial_grid.shape)
        else:
            config_dict['has_initial_grid'] = False

        return config_dict

    @classmethod
    def from_dict(cls, config_dict: Dict, initial_grid: Optional[np.ndarray] = None) -> 'SimulationConfig':
        """Create configuration from dictionary"""
        return cls(
            start_position=tuple(config_dict['start_position']),
            start_direction=config_dict['start_direction'],
            initial_grid=initial_grid,
            grid_size=config_dict.get('grid_size', 100)
        )

    def save(self, filepath: Path) -> None:
        """
        Save configuration to file.

        Args:
            filepath: Path to save configuration (without extension)
        """
        filepath = Path(filepath)

        # Save JSON metadata
        with open(filepath.with_suffix('.json'), 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

        # Save grid if it exists
        if self.initial_grid is not None:
            np.save(filepath.with_suffix('.npy'), self.initial_grid)

    @classmethod
    def load(cls, filepath: Path) -> 'SimulationConfig':
        """
        Load configuration from file.

        Args:
            filepath: Path to configuration file (without extension)

        Returns:
            SimulationConfig instance
        """
        filepath = Path(filepath)

        # Load JSON metadata
        with open(filepath.with_suffix('.json'), 'r') as f:
            config_dict = json.load(f)

        # Load grid if it exists
        initial_grid = None
        if config_dict.get('has_initial_grid', False):
            npy_path = filepath.with_suffix('.npy')
            if npy_path.exists():
                initial_grid = np.load(npy_path)

        return cls.from_dict(config_dict, initial_grid)


class SimulationResult:
    """Results from a simulation run"""

    def __init__(
        self,
        config: SimulationConfig,
        highway_direction: Optional[str],
        steps_to_highway: int,
        grid_expansions: int,
        final_grid_size: Tuple[int, int],
        timestamp: Optional[str] = None
    ):
        """
        Initialize simulation result.

        Args:
            config: Configuration used for this simulation
            highway_direction: Detected highway direction or None
            steps_to_highway: Number of steps taken
            grid_expansions: Number of times grid expanded
            final_grid_size: Final (width, height) of grid
            timestamp: ISO format timestamp
        """
        self.config = config
        self.highway_direction = highway_direction
        self.steps_to_highway = steps_to_highway
        self.grid_expansions = grid_expansions
        self.final_grid_size = final_grid_size
        self.timestamp = timestamp if timestamp else datetime.now().isoformat()

    def to_dict(self) -> Dict:
        """Convert result to dictionary for JSON serialization"""
        return {
            'configuration': self.config.to_dict(),
            'highway_direction': self.highway_direction,
            'steps_to_highway': self.steps_to_highway,
            'grid_expansions': self.grid_expansions,
            'final_grid_size': list(self.final_grid_size),
            'timestamp': self.timestamp
        }

    @classmethod
    def from_dict(cls, result_dict: Dict, initial_grid: Optional[np.ndarray] = None) -> 'SimulationResult':
        """Create result from dictionary"""
        config = SimulationConfig.from_dict(result_dict['configuration'], initial_grid)

        return cls(
            config=config,
            highway_direction=result_dict['highway_direction'],
            steps_to_highway=result_dict['steps_to_highway'],
            grid_expansions=result_dict['grid_expansions'],
            final_grid_size=tuple(result_dict['final_grid_size']),
            timestamp=result_dict.get('timestamp')
        )

    def save(self, filepath: Path) -> None:
        """
        Save result to file.

        Args:
            filepath: Path to save result
        """
        filepath = Path(filepath)

        # Ensure parent directory exists
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Save result as JSON
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

        # Save grid if config has one
        if self.config.initial_grid is not None:
            grid_path = filepath.with_suffix('.grid.npy')
            np.save(grid_path, self.config.initial_grid)

    @classmethod
    def load(cls, filepath: Path) -> 'SimulationResult':
        """
        Load result from file.

        Args:
            filepath: Path to result file

        Returns:
            SimulationResult instance
        """
        filepath = Path(filepath)

        with open(filepath, 'r') as f:
            result_dict = json.load(f)

        # Try to load grid if it exists
        initial_grid = None
        grid_path = filepath.with_suffix('.grid.npy')
        if grid_path.exists():
            initial_grid = np.load(grid_path)

        return cls.from_dict(result_dict, initial_grid)


class DatasetGenerator:
    """Generate datasets of Langton's Ant simulations"""

    def __init__(self, output_dir: Path):
        """
        Initialize dataset generator.

        Args:
            output_dir: Directory to save dataset files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_random_config(
        self,
        grid_size: int = 100,
        allow_initial_pattern: bool = True,
        pattern_density: float = 0.1
    ) -> SimulationConfig:
        """
        Generate a random initial configuration.

        Args:
            grid_size: Size of the grid
            allow_initial_pattern: Whether to include random initial patterns
            pattern_density: If using initial pattern, density of black cells (0.0-1.0)

        Returns:
            Random SimulationConfig
        """
        # Random starting position (avoid edges)
        margin = grid_size // 4
        start_x = np.random.randint(margin, grid_size - margin)
        start_y = np.random.randint(margin, grid_size - margin)

        # Random starting direction
        start_direction = np.random.randint(0, 4)

        # Create random initial pattern
        initial_grid = None
        if allow_initial_pattern:
            initial_grid = np.random.random((grid_size, grid_size)) < pattern_density
            initial_grid = initial_grid.astype(np.uint8)

        return SimulationConfig(
            start_position=(start_x, start_y),
            start_direction=start_direction,
            initial_grid=initial_grid,
            grid_size=grid_size
        )

    def run_simulation(
        self,
        config: SimulationConfig,
        max_steps: int = 100000,
        check_interval: int = 500
    ) -> SimulationResult:
        """
        Run a single simulation with given configuration.

        Args:
            config: Simulation configuration
            max_steps: Maximum steps to run
            check_interval: How often to check for highway

        Returns:
            SimulationResult
        """
        # Create simulation
        ant = LangtonAnt(
            initial_size=config.grid_size,
            start_position=config.start_position,
            start_direction=config.start_direction,
            initial_grid=config.initial_grid
        )

        # Run until highway detected or max steps
        highway_direction = ant.run_until_highway(max_steps=max_steps, check_interval=check_interval)

        # Get final state
        state = ant.get_state()

        return SimulationResult(
            config=config,
            highway_direction=highway_direction,
            steps_to_highway=state['step_count'],
            grid_expansions=state['expansion_count'],
            final_grid_size=state['grid_size']
        )

    def generate_dataset(
        self,
        num_simulations: int,
        grid_size: int = 100,
        max_steps: int = 100000,
        check_interval: int = 500,
        allow_initial_pattern: bool = True,
        pattern_density: float = 0.1,
        prefix: str = "sim"
    ) -> List[SimulationResult]:
        """
        Generate a dataset of multiple simulations.

        Args:
            num_simulations: Number of simulations to run
            grid_size: Grid size for each simulation
            max_steps: Maximum steps per simulation
            check_interval: How often to check for highway
            allow_initial_pattern: Whether to include random initial patterns
            pattern_density: Density of black cells in initial patterns
            prefix: Prefix for output filenames

        Returns:
            List of SimulationResults
        """
        results = []

        for i in range(num_simulations):
            # Generate random configuration
            config = self.generate_random_config(
                grid_size=grid_size,
                allow_initial_pattern=allow_initial_pattern,
                pattern_density=pattern_density
            )

            # Run simulation
            result = self.run_simulation(config, max_steps, check_interval)

            # Save result
            output_path = self.output_dir / f"{prefix}_{i:06d}.json"
            result.save(output_path)

            results.append(result)

            # Print progress
            if (i + 1) % 10 == 0 or i == 0:
                print(f"Completed {i + 1}/{num_simulations} simulations")

        return results

    def load_dataset(self, pattern: str = "sim_*.json") -> List[SimulationResult]:
        """
        Load all results matching pattern from output directory.

        Args:
            pattern: Glob pattern for result files

        Returns:
            List of SimulationResults
        """
        results = []
        for filepath in sorted(self.output_dir.glob(pattern)):
            if not filepath.name.endswith('.grid.npy'):  # Skip grid files
                result = SimulationResult.load(filepath)
                results.append(result)

        return results
