"""
Channel overlap computation for co-localization analysis.

Computes the number of voxels that are simultaneously active
across different combinations of channels.
"""

import itertools
import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from .thresholding import _init_gpu, _GPU_AVAILABLE, _cp

logger = logging.getLogger(__name__)


@dataclass 
class OverlapResult:
    """
    Result of overlap computation for a single tile.
    
    Attributes:
        overlaps: Dict mapping channel combination tuple to voxel count
        combinations: List of channel combinations computed
        total_voxels: Total voxels in the tile
    """
    overlaps: Dict[Tuple[int, ...], int]
    combinations: List[Tuple[int, ...]]
    total_voxels: int
    
    def get_overlap(self, *channels: int) -> int:
        """Get overlap count for specific channels."""
        key = tuple(sorted(channels))
        return self.overlaps.get(key, 0)
    
    def get_overlap_fraction(self, *channels: int) -> float:
        """Get overlap as fraction of total voxels."""
        return self.get_overlap(*channels) / self.total_voxels
    
    def to_dict(self) -> Dict[str, int]:
        """Convert to dict with string keys (for CSV/JSON)."""
        return {
            "_".join(map(str, k)): v 
            for k, v in self.overlaps.items()
        }
    
    def __repr__(self) -> str:
        lines = ["OverlapResult("]
        for combo, count in self.overlaps.items():
            combo_str = " ∩ ".join(map(str, combo))
            pct = 100 * count / self.total_voxels
            lines.append(f"  {combo_str}: {count:,} ({pct:.2f}%)")
        lines.append(")")
        return "\n".join(lines)


def get_channel_combinations(
    channels: List[int],
    min_size: int = 2,
    max_size: int = None,
) -> List[Tuple[int, ...]]:
    """
    Generate all combinations of channels.
    
    Args:
        channels: List of channel indices
        min_size: Minimum combination size (default 2)
        max_size: Maximum combination size (default: all channels)
    
    Returns:
        List of channel index tuples
        
    Example:
        >>> get_channel_combinations([0, 1, 2])
        [(0, 1), (0, 2), (1, 2), (0, 1, 2)]
    """
    if max_size is None:
        max_size = len(channels)
    
    combinations = []
    for r in range(min_size, max_size + 1):
        combinations.extend(itertools.combinations(sorted(channels), r))
    
    return combinations


def compute_overlaps(
    masks: Dict[int, np.ndarray],
    combinations: List[Tuple[int, ...]] = None,
) -> OverlapResult:
    """
    Compute overlap voxel counts for channel combinations.
    
    For each combination of channels, counts the number of voxels
    where ALL channels in the combination are active (logical AND).
    
    Args:
        masks: Dict mapping channel index to binary mask (numpy array)
        combinations: Specific combinations to compute (default: all pairs and higher)
    
    Returns:
        OverlapResult with overlap counts
        
    Example:
        >>> masks = {0: mask0, 1: mask1, 2: mask2}
        >>> result = compute_overlaps(masks)
        >>> print(result.get_overlap(0, 1))  # Voxels where both 0 and 1 are active
    """
    _init_gpu()
    
    channels = sorted(masks.keys())
    
    if combinations is None:
        combinations = get_channel_combinations(channels)
    
    # Get total voxels from first mask
    first_mask = masks[channels[0]]
    total_voxels = first_mask.size
    
    overlaps = {}
    
    if _GPU_AVAILABLE:
        # Transfer all masks to GPU once
        masks_gpu = {ch: _cp.asarray(mask) for ch, mask in masks.items()}
        
        for combo in combinations:
            # Start with first channel's mask
            overlap_mask = masks_gpu[combo[0]].copy()
            
            # AND with remaining channels
            for ch in combo[1:]:
                overlap_mask = _cp.logical_and(overlap_mask, masks_gpu[ch])
            
            overlaps[combo] = int(_cp.sum(overlap_mask))
        
        # Clean up GPU memory
        del masks_gpu
        _cp.get_default_memory_pool().free_all_blocks()
    else:
        for combo in combinations:
            overlap_mask = masks[combo[0]].astype(bool).copy()
            
            for ch in combo[1:]:
                overlap_mask = np.logical_and(overlap_mask, masks[ch])
            
            overlaps[combo] = int(np.sum(overlap_mask))
    
    return OverlapResult(
        overlaps=overlaps,
        combinations=combinations,
        total_voxels=total_voxels,
    )


def compute_pairwise_matrix(
    masks: Dict[int, np.ndarray],
) -> np.ndarray:
    """
    Compute pairwise overlap matrix.
    
    Returns a symmetric matrix where entry (i, j) is the number of
    voxels where both channel i and channel j are active.
    
    Args:
        masks: Dict mapping channel index to binary mask
    
    Returns:
        2D numpy array of shape (n_channels, n_channels)
        Diagonal entries are the total active voxels per channel.
    """
    channels = sorted(masks.keys())
    n = len(channels)
    
    matrix = np.zeros((n, n), dtype=np.int64)
    
    # Diagonal: individual channel counts
    for i, ch in enumerate(channels):
        matrix[i, i] = int(np.sum(masks[ch]))
    
    # Off-diagonal: pairwise overlaps
    pairs = get_channel_combinations(channels, min_size=2, max_size=2)
    result = compute_overlaps(masks, pairs)
    
    for (ch1, ch2), count in result.overlaps.items():
        i = channels.index(ch1)
        j = channels.index(ch2)
        matrix[i, j] = count
        matrix[j, i] = count  # Symmetric
    
    return matrix
