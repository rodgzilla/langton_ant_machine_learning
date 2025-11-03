#!/usr/bin/env python3
"""
Visualize a Langton's Ant simulation.

Usage:
    python visualize.py                          # Standard simulation
    python visualize.py --config config.json     # From config file
    python visualize.py --dataset result.json    # From dataset sample
"""

import argparse
import sys
from pathlib import Path
from langton_ant.simulation import LangtonAnt
from langton_ant.visualizer import Visualizer
from langton_ant.dataset import SimulationConfig, SimulationResult


def main():
    parser = argparse.ArgumentParser(description='Visualize Langton\'s Ant simulation')

    parser.add_argument(
        '--config',
        type=str,
        help='Path to configuration JSON file'
    )

    parser.add_argument(
        '--dataset',
        type=str,
        help='Path to dataset result JSON file to replay'
    )

    parser.add_argument(
        '--grid-size',
        type=int,
        default=100,
        help='Grid size (default: 100)'
    )

    parser.add_argument(
        '--start-x',
        type=int,
        help='Starting X position (default: center)'
    )

    parser.add_argument(
        '--start-y',
        type=int,
        help='Starting Y position (default: center)'
    )

    parser.add_argument(
        '--direction',
        type=int,
        choices=[0, 1, 2, 3],
        default=0,
        help='Starting direction: 0=N, 1=E, 2=S, 3=W (default: 0)'
    )

    parser.add_argument(
        '--speed',
        type=int,
        default=10,
        help='Initial steps per frame (default: 10)'
    )

    parser.add_argument(
        '--window-size',
        type=int,
        nargs=2,
        default=[800, 800],
        metavar=('WIDTH', 'HEIGHT'),
        help='Window size in pixels (default: 800 800)'
    )

    parser.add_argument(
        '--cell-size',
        type=int,
        help='Cell size in pixels (auto if not specified)'
    )

    args = parser.parse_args()

    # Load configuration
    config = None

    if args.dataset:
        # Load from dataset result
        print(f"Loading configuration from dataset: {args.dataset}")
        result = SimulationResult.load(Path(args.dataset))
        config = result.config
        print(f"Original result: Highway {result.highway_direction} after {result.steps_to_highway} steps")

    elif args.config:
        # Load from config file
        print(f"Loading configuration from file: {args.config}")
        config = SimulationConfig.load(Path(args.config))

    else:
        # Create default configuration
        start_x = args.start_x if args.start_x is not None else args.grid_size // 2
        start_y = args.start_y if args.start_y is not None else args.grid_size // 2

        config = SimulationConfig(
            start_position=(start_x, start_y),
            start_direction=args.direction,
            grid_size=args.grid_size
        )

    # Create simulation
    ant = LangtonAnt(
        initial_size=config.grid_size,
        start_position=config.start_position,
        start_direction=config.start_direction,
        initial_grid=config.initial_grid
    )

    print(f"Starting simulation:")
    print(f"  Position: {config.start_position}")
    print(f"  Direction: {config.start_direction} ({ant.DIRECTION_NAMES[config.start_direction]})")
    print(f"  Grid size: {config.grid_size}x{config.grid_size}")
    print(f"  Has initial pattern: {config.initial_grid is not None}")
    print()
    print("Controls:")
    print("  SPACE: Pause/Unpause")
    print("  UP/DOWN: Increase/Decrease speed")
    print("  +/-: Zoom in/out")
    print("  Q/ESC: Quit")
    print()

    # Create and run visualizer
    visualizer = Visualizer(
        ant=ant,
        window_size=tuple(args.window_size),
        cell_size=args.cell_size,
        steps_per_frame=args.speed
    )

    visualizer.run()

    # Print final state
    state = ant.get_state()
    print(f"\nFinal state:")
    print(f"  Steps: {state['step_count']}")
    print(f"  Position: {state['ant_position']}")
    print(f"  Grid size: {state['grid_size']}")
    print(f"  Expansions: {state['expansion_count']}")
    if state['highway_detected']:
        print(f"  Highway detected: {state['highway_direction']}")
    else:
        print(f"  Highway: Not detected")


if __name__ == '__main__':
    main()
