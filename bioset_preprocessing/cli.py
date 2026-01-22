"""
Command-line interface for the volumetric pipeline.

Usage:
    bioset run --config config.yaml
    bioset run --url https://... --channels 0 1 2 --threshold otsu
    bioset info --url https://...
"""

import argparse
import logging
import sys
from pathlib import Path

from .config import Config
from .data import load_zarr_array
from .pipeline import Pipeline
from .processing import create_tiling_scheme, list_threshold_methods


def setup_logging(verbose: bool = False) -> None:
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def cmd_run(args: argparse.Namespace) -> int:
    """Run the pipeline."""
    setup_logging(args.verbose)
    
    if args.config:
        # Load from config file
        pipeline = Pipeline.from_config(args.config)
    elif args.url:
        # Build config from CLI arguments
        channels = args.channels if args.channels else None
        
        config = Config(
            zarr_url=args.url,
            zarr_component=args.component,
            channels=channels,
            threshold_method=args.threshold,
            tile_size=args.tile_size,
            output_dir=Path(args.output),
            save_masks=args.save_masks,
            resume=not args.no_resume,
        )
        pipeline = Pipeline(config)
    else:
        print("Error: Must provide either --config or --url", file=sys.stderr)
        return 1
    
    try:
        results = pipeline.run()
        print(f"\nCompleted: {results['completed']} tiles")
        print(f"Output: {results['output_dir']}")
        return 0
    except Exception as e:
        logging.error(f"Pipeline failed: {e}")
        if args.verbose:
            raise
        return 1


def cmd_info(args: argparse.Namespace) -> int:
    """Show information about a dataset."""
    setup_logging(args.verbose)
    
    if not args.url:
        print("Error: --url required", file=sys.stderr)
        return 1
    
    try:
        arr, info = load_zarr_array(args.url, args.component)
        
        print("\n" + "=" * 50)
        print("DATASET INFORMATION")
        print("=" * 50)
        print(f"URL: {args.url}")
        print(f"Component: {args.component}")
        print(f"\n{info}")
        
        scheme = create_tiling_scheme(info)
        print(f"\n{scheme}")
        
        return 0
    except Exception as e:
        logging.error(f"Failed to load dataset: {e}")
        if args.verbose:
            raise
        return 1


def cmd_methods(args: argparse.Namespace) -> int:
    """List available threshold methods."""
    print("\nAvailable threshold methods:")
    print("-" * 30)
    for method in list_threshold_methods():
        print(f"  - {method}")
    return 0


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        prog="bioset",
        description="Process large volumetric microscopy data",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output",
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Run command
    run_parser = subparsers.add_parser("run", help="Run the pipeline")
    run_parser.add_argument(
        "-c", "--config",
        type=Path,
        help="Path to config file (YAML or JSON)",
    )
    run_parser.add_argument(
        "--url",
        help="Zarr URL or path",
    )
    run_parser.add_argument(
        "--component",
        default="0",
        help="Resolution level (default: 0)",
    )
    run_parser.add_argument(
        "--channels",
        type=int,
        nargs="+",
        help="Channels to process (default: all)",
    )
    run_parser.add_argument(
        "--threshold",
        default="percentile_95",
        help="Threshold method (default: percentile_95)",
    )
    run_parser.add_argument(
        "--tile-size",
        type=int,
        help="Tile size (default: auto)",
    )
    run_parser.add_argument(
        "--output", "-o",
        default="./results",
        help="Output directory (default: ./results)",
    )
    run_parser.add_argument(
        "--save-masks",
        action="store_true",
        help="Save binary masks as TIFFs",
    )
    run_parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Don't resume from checkpoint",
    )
    run_parser.set_defaults(func=cmd_run)
    
    # Info command
    info_parser = subparsers.add_parser("info", help="Show dataset information")
    info_parser.add_argument(
        "--url",
        required=True,
        help="Zarr URL or path",
    )
    info_parser.add_argument(
        "--component",
        default="0",
        help="Resolution level (default: 0)",
    )
    info_parser.set_defaults(func=cmd_info)
    
    # Methods command
    methods_parser = subparsers.add_parser("methods", help="List threshold methods")
    methods_parser.set_defaults(func=cmd_methods)
    
    # Parse and execute
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return 0
    
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
