"""Processing utilities for thresholding, tiling, and overlap computation."""

from .overlap import OverlapResult, compute_overlaps, compute_pairwise_matrix, get_channel_combinations
from .thresholding import (
    ThresholdResult,
    apply_threshold,
    get_threshold_method,
    is_gpu_available,
    list_threshold_methods,
    register_threshold_method,
    threshold_fixed,
    threshold_mean_std,
    threshold_otsu,
    threshold_percentile,
)
from .tiling import TileInfo, TilingScheme, calculate_optimal_tile_size, create_tiling_scheme

__all__ = [
    # Tiling
    "TileInfo",
    "TilingScheme",
    "calculate_optimal_tile_size",
    "create_tiling_scheme",
    # Thresholding
    "ThresholdResult",
    "apply_threshold",
    "get_threshold_method",
    "list_threshold_methods",
    "register_threshold_method",
    "threshold_percentile",
    "threshold_otsu",
    "threshold_mean_std",
    "threshold_fixed",
    "is_gpu_available",
    # Overlap
    "OverlapResult",
    "compute_overlaps",
    "compute_pairwise_matrix",
    "get_channel_combinations",
]
