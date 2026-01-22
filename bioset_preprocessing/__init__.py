"""
Volumetric Pipeline - Process large volumetric microscopy data.

A pipeline for processing large multi-channel volumetric datasets,
computing thresholds, and analyzing channel co-localization.

Quick Start:
    >>> from volumetric_pipeline import Pipeline, Config
    
    >>> # From config file
    >>> pipeline = Pipeline.from_config("config.yaml")
    >>> results = pipeline.run()
    
    >>> # From URL
    >>> pipeline = Pipeline.from_url("https://example.com/data.zarr")
    >>> results = pipeline.run(channels=[0, 1, 2])
    
    >>> # Programmatic
    >>> config = Config(
    ...     zarr_url="https://example.com/data.zarr",
    ...     channels=[0, 1, 2],
    ...     threshold_method="otsu",
    ... )
    >>> pipeline = Pipeline(config)
    >>> results = pipeline.run()

See Also:
    - Config: Configuration dataclass
    - Pipeline: Main pipeline class
    - run_pipeline: Convenience function
"""

__version__ = "0.1.0"

# Main API
from .config import Config
from .pipeline import Pipeline, run_pipeline

# Data loading
from .data import ArrayInfo, load_zarr_array

# Processing utilities
from .processing import (
    # Tiling
    TileInfo,
    TilingScheme,
    calculate_optimal_tile_size,
    create_tiling_scheme,
    # Thresholding
    ThresholdResult,
    apply_threshold,
    list_threshold_methods,
    register_threshold_method,
    is_gpu_available,
    # Overlap
    OverlapResult,
    compute_overlaps,
    get_channel_combinations,
)

# I/O
from .io import (
    ResultSaver,
    CheckpointManager,
    save_mask_tiff,
    is_tiff_available,
)

__all__ = [
    # Version
    "__version__",
    # Main API
    "Config",
    "Pipeline",
    "run_pipeline",
    # Data
    "ArrayInfo",
    "load_zarr_array",
    # Tiling
    "TileInfo",
    "TilingScheme",
    "calculate_optimal_tile_size",
    "create_tiling_scheme",
    # Thresholding
    "ThresholdResult",
    "apply_threshold",
    "list_threshold_methods",
    "register_threshold_method",
    "is_gpu_available",
    # Overlap
    "OverlapResult",
    "compute_overlaps",
    "get_channel_combinations",
    # I/O
    "ResultSaver",
    "CheckpointManager",
    "save_mask_tiff",
    "is_tiff_available",
]
