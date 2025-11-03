#!/usr/bin/env python3
"""
Generate a dataset of Langton's Ant simulations.

Usage:
    python generate_dataset.py --count 100 --output dataset/
    python generate_dataset.py --count 1000 --grid-size 150 --max-steps 150000
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
from langton_ant.dataset import DatasetGenerator


def main():
    parser = argparse.ArgumentParser(description='Generate Langton\'s Ant simulation dataset')

    parser.add_argument(
        '--count',
        type=int,
        required=True,
        help='Number of simulations to generate'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='dataset',
        help='Output directory for dataset (default: dataset/)'
    )

    parser.add_argument(
        '--grid-size',
        type=int,
        default=100,
        help='Grid size for simulations (default: 100)'
    )

    parser.add_argument(
        '--max-steps',
        type=int,
        default=100000,
        help='Maximum steps per simulation (default: 100000)'
    )

    parser.add_argument(
        '--check-interval',
        type=int,
        default=500,
        help='Steps between highway checks (default: 500)'
    )

    parser.add_argument(
        '--no-patterns',
        action='store_true',
        help='Disable random initial patterns (all start with empty grid)'
    )

    parser.add_argument(
        '--pattern-density',
        type=float,
        default=0.1,
        help='Density of black cells in initial patterns (default: 0.1)'
    )

    parser.add_argument(
        '--prefix',
        type=str,
        default='sim',
        help='Prefix for output filenames (default: sim)'
    )

    args = parser.parse_args()

    # Validate arguments
    if args.count <= 0:
        print("Error: count must be positive")
        sys.exit(1)

    if args.grid_size < 10:
        print("Error: grid-size must be at least 10")
        sys.exit(1)

    if args.pattern_density < 0 or args.pattern_density > 1:
        print("Error: pattern-density must be between 0 and 1")
        sys.exit(1)

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Langton's Ant Dataset Generation")
    print("=" * 60)
    print(f"Number of simulations: {args.count}")
    print(f"Output directory: {output_dir.absolute()}")
    print(f"Grid size: {args.grid_size}x{args.grid_size}")
    print(f"Max steps per simulation: {args.max_steps}")
    print(f"Check interval: {args.check_interval}")
    print(f"Allow initial patterns: {not args.no_patterns}")
    if not args.no_patterns:
        print(f"Pattern density: {args.pattern_density}")
    print(f"Filename prefix: {args.prefix}")
    print("=" * 60)
    print()

    # Confirm with user for large datasets
    if args.count > 100:
        response = input(f"Generate {args.count} simulations? This may take a while. [y/N]: ")
        if response.lower() not in ['y', 'yes']:
            print("Cancelled.")
            sys.exit(0)

    # Create generator
    generator = DatasetGenerator(output_dir)

    # Record start time
    start_time = datetime.now()
    print(f"Starting generation at {start_time.strftime('%H:%M:%S')}")
    print()

    # Generate dataset
    try:
        results = generator.generate_dataset(
            num_simulations=args.count,
            grid_size=args.grid_size,
            max_steps=args.max_steps,
            check_interval=args.check_interval,
            allow_initial_pattern=not args.no_patterns,
            pattern_density=args.pattern_density,
            prefix=args.prefix
        )

        # Record end time
        end_time = datetime.now()
        duration = end_time - start_time

        print()
        print("=" * 60)
        print("Generation Complete!")
        print("=" * 60)
        print(f"Total simulations: {len(results)}")
        print(f"Duration: {duration}")
        print(f"Average time per simulation: {duration / len(results)}")
        print()

        # Analyze results
        highways_detected = sum(1 for r in results if r.highway_direction is not None)
        avg_steps = sum(r.steps_to_highway for r in results) / len(results)

        print("Dataset Statistics:")
        print(f"  Highways detected: {highways_detected}/{len(results)} ({100*highways_detected/len(results):.1f}%)")
        print(f"  Average steps: {avg_steps:.0f}")

        if highways_detected > 0:
            # Direction distribution
            directions = {}
            for r in results:
                if r.highway_direction:
                    directions[r.highway_direction] = directions.get(r.highway_direction, 0) + 1

            print(f"  Direction distribution:")
            for direction in ['NE', 'NW', 'SE', 'SW']:
                count = directions.get(direction, 0)
                print(f"    {direction}: {count} ({100*count/highways_detected:.1f}%)")

        print()
        print(f"Dataset saved to: {output_dir.absolute()}")
        print("=" * 60)

    except KeyboardInterrupt:
        print()
        print("Generation interrupted by user.")
        print(f"Partial dataset saved to: {output_dir.absolute()}")
        sys.exit(1)

    except Exception as e:
        print()
        print(f"Error during generation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
