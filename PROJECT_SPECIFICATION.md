# Langton's Ant Simulation Project

## Project Overview

This project implements a Langton's Ant cellular automaton simulator designed to generate training data for a machine learning system. The goal is to predict the direction of the "highway" pattern that emerges from different initial grid configurations.

## Background: Langton's Ant

Langton's Ant is a two-dimensional universal Turing machine with simple rules:
- The ant starts on a grid of white and black cells
- At each step:
  - If on a white cell: turn 90° right, flip the cell to black, move forward
  - If on a black cell: turn 90° left, flip the cell to white, move forward

After an initial chaotic phase (~10,000 steps), the ant typically enters a repeating pattern called a "highway" that extends diagonally indefinitely.

## Project Goals

### Primary Objectives
1. **Simulation**: Create a fast, accurate Langton's Ant simulator with configurable initial conditions
2. **Highway Detection**: Automatically detect when the highway pattern emerges and determine its direction
3. **Dataset Generation**: Generate large datasets of simulation runs with varying initial configurations
4. **Visualization**: Provide a graphical interface to observe and debug simulations

### Machine Learning Goal (Out of Scope for Simulation)
The ultimate goal is to train an ML model to predict the highway direction (NE, NW, SE, SW) given only the initial grid configuration. The simulation component provides the training data; ML implementation is handled separately.

## Technical Specifications

### Technology Stack
- **Language**: Python 3.x
- **Core Libraries**:
  - `numpy`: Fast numerical operations for grid manipulation
  - `pygame`: Graphical visualization
  - `pytest`: Unit testing framework
- **Data Storage**: JSON/NPY format for datasets
- **Code Style**: Readable, maintainable Python (no heavy optimization like Numba)
- **Testing**: Comprehensive unit tests for all components

### Components

#### 1. Core Simulation Engine
- **Expandable Grid**: Starts with a small grid (e.g., 100x100) and expands dynamically when the ant approaches edges
- **Efficient State Management**: Uses NumPy arrays for cell states
- **Step Execution**: Processes ant movement and cell flipping according to Langton's rules

#### 2. Highway Detection System
- **Pattern Recognition**: Detects the repeating 104-step highway cycle
- **Direction Classification**: Determines highway direction (NE, NW, SE, SW)
- **Automatic Termination**: Stops simulation once highway is confirmed

#### 3. Visualization Module (Pygame)
- **Real-time Display**: Shows grid, ant position, and current direction
- **Configurable Speed**: Adjustable simulation speed for observation
- **Status Information**: Displays step count, highway status
- **Restart Capability**: Load and visualize any saved configuration

#### 4. Dataset Management
- **Configuration Storage**: Save/load initial conditions
  - Ant starting position (x, y)
  - Ant starting direction (N, E, S, W or 0, 1, 2, 3)
  - Initial grid pattern (2D binary array)
- **Results Storage**: Record simulation outcomes
  - Highway direction detected
  - Number of steps to highway formation
  - Metadata (timestamp, grid size, etc.)
- **Batch Processing**: Run multiple simulations with randomized or specified initial conditions

### Configurable Parameters

#### Initial Conditions
1. **Starting Position**: (x, y) coordinates on the grid
2. **Starting Direction**: North, East, South, or West
3. **Initial Grid Pattern**: 2D array of black (1) and white (0) cells

#### Simulation Parameters
- Initial grid size (default: small, e.g., 100x100)
- Maximum steps (timeout if highway not detected)
- Highway detection confidence threshold

### Data Format

#### Input Configuration
```json
{
  "start_position": [x, y],
  "start_direction": 0-3,
  "initial_grid": [[0, 1, 0, ...], ...] or "path/to/grid.npy"
}
```

#### Output Results
```json
{
  "configuration": { /* input config */ },
  "highway_direction": "NE"|"NW"|"SE"|"SW"|null,
  "steps_to_highway": integer,
  "grid_expansions": integer,
  "final_grid_size": [width, height]
}
```

## Implementation Architecture

### Module Structure
```
langton_ant/
├── simulation.py       # Core simulation engine
├── highway_detector.py # Highway pattern detection
├── visualizer.py       # Pygame visualization
├── dataset.py          # Dataset generation and management
├── utils.py           # Helper functions
└── tests/
    ├── test_simulation.py
    ├── test_highway_detector.py
    ├── test_dataset.py
    └── test_utils.py
```

### Key Classes

#### `LangtonAnt`
- Manages grid state and ant state
- Executes simulation steps
- Handles grid expansion
- Integrates with highway detector

#### `HighwayDetector`
- Tracks ant position history
- Detects repeating patterns
- Classifies highway direction

#### `Visualizer`
- Renders grid with pygame
- Handles user input (pause, speed control)
- Displays simulation statistics

#### `DatasetGenerator`
- Creates random initial configurations
- Runs batch simulations
- Saves results in structured format

## Testing Strategy

### Testing Framework
- **pytest**: Primary testing framework
- **pytest-cov**: Code coverage analysis
- **Target Coverage**: >90% for core simulation logic

### Unit Test Coverage

#### 1. Simulation Tests (`test_simulation.py`)
- **Basic Rules**:
  - Ant turns right on white cell and flips it to black
  - Ant turns left on black cell and flips it to white
  - Ant moves forward after turning
- **Grid Expansion**:
  - Grid expands when ant approaches boundaries
  - Ant position is correctly adjusted after expansion
  - Cell states are preserved during expansion
- **Initial Conditions**:
  - Correctly sets starting position
  - Correctly sets starting direction
  - Loads custom initial grid patterns
- **State Management**:
  - Grid state is correctly maintained
  - Step counter increments properly
  - Direction changes are accurate

#### 2. Highway Detection Tests (`test_highway_detector.py`)
- **Pattern Detection**:
  - Detects the 104-step repeating highway pattern
  - Correctly identifies when highway has NOT formed (during chaotic phase)
  - Handles edge cases (ant leaving and re-entering detection zone)
- **Direction Classification**:
  - Correctly classifies NE (northeast) highways
  - Correctly classifies NW (northwest) highways
  - Correctly classifies SE (southeast) highways
  - Correctly classifies SW (southwest) highways
- **Known Scenarios**:
  - Test against standard Langton's Ant (starts at origin, empty grid)
  - Verify highway forms after ~10,000 steps
  - Verify highway direction matches expected behavior

#### 3. Dataset Tests (`test_dataset.py`)
- **Configuration Management**:
  - Save configuration to JSON/NPY format
  - Load configuration from JSON/NPY format
  - Round-trip test (save and load produces identical config)
- **Result Storage**:
  - Correctly saves simulation results
  - Includes all required fields (highway direction, steps, etc.)
  - Handles null highway direction (timeout cases)
- **Batch Generation**:
  - Generates requested number of simulations
  - Creates unique initial configurations
  - Properly saves all results

#### 4. Utilities Tests (`test_utils.py`)
- **Helper Functions**:
  - Direction encoding/decoding (N/E/S/W ↔ 0/1/2/3)
  - Grid manipulation utilities
  - Any other utility functions

#### 5. Integration Tests
- **End-to-End Simulation**:
  - Run complete simulation from start to highway detection
  - Verify results are reproducible with same initial conditions
  - Test with various initial configurations
- **Dataset Generation Pipeline**:
  - Generate small dataset (10 simulations)
  - Verify all files are created correctly
  - Load and replay simulations from dataset

### Test Execution

Run all tests:
```bash
pytest tests/
```

Run with coverage:
```bash
pytest --cov=langton_ant --cov-report=html tests/
```

Run specific test file:
```bash
pytest tests/test_simulation.py
```

### Continuous Testing
- Tests should run quickly (entire suite < 30 seconds)
- Use smaller grids and shorter timeouts for tests
- Mock pygame components in visualizer tests to avoid display dependency

## Usage Scenarios

### 1. Single Visualization
Run a single simulation with visualization to observe behavior:
```python
python visualize.py --config my_config.json
```

### 2. Dataset Generation
Generate N simulations with random initial conditions:
```python
python generate_dataset.py --count 10000 --output dataset/
```

### 3. Replay from Dataset
Visualize a specific configuration from the dataset:
```python
python visualize.py --dataset-sample dataset/simulation_0042.json
```

## Performance Considerations

- **Grid Representation**: NumPy boolean arrays for memory efficiency
- **Expansion Strategy**: Double grid size when needed to minimize reallocations
- **Highway Detection**: Efficient pattern matching without storing entire history
- **Headless Mode**: No rendering overhead for dataset generation

## Future Extensions (Optional)

- Multi-color Langton's Ant variants
- Parallel simulation execution
- Interactive grid editor for initial patterns
- Statistical analysis tools for dataset
- 3D visualization of ant trajectory over time

## Success Criteria

1. Simulator correctly implements Langton's Ant rules
2. Highway detection works reliably (>99% accuracy on standard cases)
3. Can generate 10,000+ simulations in reasonable time (hours, not days)
4. Visualization is smooth and informative
5. Dataset format is suitable for ML training
6. Code is readable and well-documented
