"""
Tiling utilities for processing large volumes in manageable chunks.

Calculates optimal tile sizes based on available memory and generates
tile coordinates for iteration.
"""

import logging
from dataclasses import dataclass
from typing import Iterator, List, Tuple

from ..data.loader import ArrayInfo

logger = logging.getLogger(__name__)


@dataclass
class TileInfo:
    """
    Information about a single tile.
    
    Attributes:
        tile_y: Tile index in Y direction (0-indexed)
        tile_x: Tile index in X direction (0-indexed)
        y_start: Start index in Y (inclusive)
        y_end: End index in Y (exclusive)
        x_start: Start index in X (inclusive)
        x_end: End index in X (exclusive)
    """
    tile_y: int
    tile_x: int
    y_start: int
    y_end: int
    x_start: int
    x_end: int
    
    @property
    def y_slice(self) -> slice:
        """Return slice for Y dimension."""
        return slice(self.y_start, self.y_end)
    
    @property
    def x_slice(self) -> slice:
        """Return slice for X dimension."""
        return slice(self.x_start, self.x_end)
    
    @property
    def tile_shape_yx(self) -> Tuple[int, int]:
        """Return (height, width) of this tile."""
        return (self.y_end - self.y_start, self.x_end - self.x_start)
    
    def __repr__(self) -> str:
        return (
            f"TileInfo(idx=({self.tile_y}, {self.tile_x}), "
            f"y=[{self.y_start}:{self.y_end}], x=[{self.x_start}:{self.x_end}])"
        )


@dataclass
class TilingScheme:
    """
    Complete tiling scheme for a volume.
    
    Attributes:
        tile_size_y: Tile height
        tile_size_x: Tile width
        n_tiles_y: Number of tiles in Y direction
        n_tiles_x: Number of tiles in X direction
        total_tiles: Total number of tiles
        array_info: Information about the source array
    """
    tile_size_y: int
    tile_size_x: int
    n_tiles_y: int
    n_tiles_x: int
    array_info: ArrayInfo
    
    @property
    def total_tiles(self) -> int:
        """Total number of tiles."""
        return self.n_tiles_y * self.n_tiles_x
    
    @property
    def tile_voxels(self) -> int:
        """Number of voxels in a full tile (Z * tile_Y * tile_X)."""
        return self.array_info.n_z * self.tile_size_y * self.tile_size_x
    
    @property
    def tile_memory_mb(self) -> float:
        """Memory for raw tile data in MB."""
        return self.tile_voxels * self.array_info.bytes_per_voxel / 1e6
    
    @property
    def peak_memory_mb(self) -> float:
        """Peak memory during computation in MB (float64 conversion)."""
        # Raw (2 bytes) + float64 for computation (8 bytes) + mask (1 byte)
        return self.tile_voxels * 11 / 1e6
    
    def get_tile(self, tile_y: int, tile_x: int) -> TileInfo:
        """
        Get TileInfo for a specific tile index.
        
        Args:
            tile_y: Tile index in Y direction
            tile_x: Tile index in X direction
            
        Returns:
            TileInfo for the specified tile
            
        Raises:
            IndexError: If tile indices are out of range
        """
        if tile_y < 0 or tile_y >= self.n_tiles_y:
            raise IndexError(
                f"tile_y={tile_y} out of range [0, {self.n_tiles_y})"
            )
        if tile_x < 0 or tile_x >= self.n_tiles_x:
            raise IndexError(
                f"tile_x={tile_x} out of range [0, {self.n_tiles_x})"
            )
        
        y_start = tile_y * self.tile_size_y
        y_end = min(y_start + self.tile_size_y, self.array_info.n_y)
        
        x_start = tile_x * self.tile_size_x
        x_end = min(x_start + self.tile_size_x, self.array_info.n_x)
        
        return TileInfo(
            tile_y=tile_y,
            tile_x=tile_x,
            y_start=y_start,
            y_end=y_end,
            x_start=x_start,
            x_end=x_end,
        )
    
    def iter_tiles(self) -> Iterator[TileInfo]:
        """
        Iterate over all tiles in row-major order.
        
        Yields:
            TileInfo for each tile
        """
        for tile_y in range(self.n_tiles_y):
            for tile_x in range(self.n_tiles_x):
                yield self.get_tile(tile_y, tile_x)
    
    def __repr__(self) -> str:
        return (
            f"TilingScheme(\n"
            f"  tile_size=({self.tile_size_y}, {self.tile_size_x}),\n"
            f"  grid=({self.n_tiles_y}, {self.n_tiles_x}),\n"
            f"  total_tiles={self.total_tiles},\n"
            f"  tile_memory={self.tile_memory_mb:.1f} MB,\n"
            f"  peak_memory={self.peak_memory_mb:.1f} MB\n"
            f")"
        )


def calculate_optimal_tile_size(
    array_info: ArrayInfo,
    available_memory_gb: float = 8.0,
    safety_factor: float = 0.5,
    n_channels_simultaneous: int = 1,
    min_tile_size: int = 128,
    max_tile_size: int = 2048,
    alignment: int = 64,
) -> int:
    """
    Calculate optimal square tile size based on memory constraints.
    
    The calculation accounts for:
    - Raw data in memory (uint16 = 2 bytes)
    - Float64 conversion during percentile computation (8 bytes)
    - Binary mask output (1 byte)
    - Safety factor to leave headroom
    
    Args:
        array_info: Information about the array
        available_memory_gb: Available RAM/VRAM in gigabytes
        safety_factor: Fraction of memory to actually use (0.0-1.0)
        n_channels_simultaneous: Number of channels loaded at once
        min_tile_size: Minimum tile size
        max_tile_size: Maximum tile size
        alignment: Round tile size to this value for memory alignment
    
    Returns:
        Optimal tile size (square, applied to both Y and X)
        
    Example:
        >>> tile_size = calculate_optimal_tile_size(
        ...     array_info,
        ...     available_memory_gb=8.0,
        ...     safety_factor=0.5
        ... )
        >>> print(f"Optimal tile size: {tile_size}")
        Optimal tile size: 1344
    """
    available_bytes = available_memory_gb * 1e9 * safety_factor
    
    # Peak memory per voxel during computation:
    # raw (2) + float64 computation (8) + mask (1) = 11 bytes
    bytes_per_voxel_peak = 11 * n_channels_simultaneous
    
    # Z dimension is fixed, solve for Y * X
    # Z * Y * X * bytes_per_voxel_peak <= available_bytes
    # Y * X <= available_bytes / (Z * bytes_per_voxel_peak)
    n_z = array_info.n_z
    max_yx_voxels = available_bytes / (n_z * bytes_per_voxel_peak)
    
    # Square tiles: Y = X, so tile_size^2 <= max_yx_voxels
    tile_size = int(max_yx_voxels ** 0.5)
    
    # Round down to alignment
    tile_size = (tile_size // alignment) * alignment
    
    # Clamp to bounds
    tile_size = max(min_tile_size, min(tile_size, max_tile_size))
    
    logger.info(
        f"Calculated optimal tile size: {tile_size} "
        f"(available={available_memory_gb:.1f}GB, Z={n_z}, "
        f"safety={safety_factor:.0%})"
    )
    
    return tile_size


def create_tiling_scheme(
    array_info: ArrayInfo,
    tile_size: int = None,
    available_memory_gb: float = 8.0,
    safety_factor: float = 0.5,
) -> TilingScheme:
    """
    Create a tiling scheme for the given array.
    
    If tile_size is not provided, it will be automatically calculated
    based on available memory.
    
    Args:
        array_info: Information about the array
        tile_size: Fixed tile size (optional, auto-calculated if None)
        available_memory_gb: Available memory for auto-calculation
        safety_factor: Safety factor for auto-calculation
    
    Returns:
        TilingScheme instance
        
    Example:
        >>> scheme = create_tiling_scheme(array_info, tile_size=1024)
        >>> for tile in scheme.iter_tiles():
        ...     process(tile)
    """
    if tile_size is None:
        tile_size = calculate_optimal_tile_size(
            array_info,
            available_memory_gb=available_memory_gb,
            safety_factor=safety_factor,
        )
    
    n_y = array_info.n_y
    n_x = array_info.n_x
    
    n_tiles_y = (n_y + tile_size - 1) // tile_size
    n_tiles_x = (n_x + tile_size - 1) // tile_size
    
    scheme = TilingScheme(
        tile_size_y=tile_size,
        tile_size_x=tile_size,
        n_tiles_y=n_tiles_y,
        n_tiles_x=n_tiles_x,
        array_info=array_info,
    )
    
    logger.info(
        f"Created tiling scheme: {n_tiles_y}x{n_tiles_x} = {scheme.total_tiles} tiles"
    )
    
    return scheme
