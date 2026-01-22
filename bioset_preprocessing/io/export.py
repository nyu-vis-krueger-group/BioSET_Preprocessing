"""
Export utilities for saving results as images.

Supports TIFF export for visualization in Fiji/ImageJ, napari, etc.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Check for tifffile
try:
    import tifffile
    TIFF_AVAILABLE = True
except ImportError:
    TIFF_AVAILABLE = False
    logger.warning("tifffile not installed. TIFF export disabled.")


def save_mask_tiff(
    mask: np.ndarray,
    path: Path,
    scale_to_255: bool = True,
    compression: str = "zlib",
) -> Path:
    """
    Save a binary mask as a TIFF file.
    
    Args:
        mask: Binary mask array (uint8, values 0 or 1)
        path: Output path
        scale_to_255: Scale mask values to 0-255 for visibility
        compression: TIFF compression method
    
    Returns:
        Path to saved file
        
    Raises:
        ImportError: If tifffile is not installed
    """
    if not TIFF_AVAILABLE:
        raise ImportError(
            "tifffile required for TIFF export. "
            "Install with: pip install tifffile"
        )
    
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    if scale_to_255:
        mask = (mask * 255).astype(np.uint8)
    
    tifffile.imwrite(path, mask, compression=compression)
    logger.debug(f"Saved mask: {path}")
    
    return path


def save_raw_tiff(
    data: np.ndarray,
    path: Path,
    compression: str = "zlib",
) -> Path:
    """
    Save raw data as a TIFF file.
    
    Args:
        data: Data array (any dtype)
        path: Output path
        compression: TIFF compression method
    
    Returns:
        Path to saved file
    """
    if not TIFF_AVAILABLE:
        raise ImportError(
            "tifffile required for TIFF export. "
            "Install with: pip install tifffile"
        )
    
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    tifffile.imwrite(path, data, compression=compression)
    logger.debug(f"Saved raw data: {path}")
    
    return path


def generate_tile_filename(
    channel: int,
    tile_y: int,
    tile_x: int,
    suffix: str = "mask",
    method: Optional[str] = None,
    extension: str = ".tiff",
) -> str:
    """
    Generate a standardized filename for a tile.
    
    Args:
        channel: Channel index
        tile_y: Tile Y index
        tile_x: Tile X index
        suffix: File suffix (e.g., "mask", "raw")
        method: Threshold method name (optional)
        extension: File extension
    
    Returns:
        Filename string
        
    Example:
        >>> generate_tile_filename(0, 1, 2, "mask", "otsu")
        "ch00_tile_y01_x02_otsu_mask.tiff"
    """
    parts = [
        f"ch{channel:02d}",
        f"tile_y{tile_y:02d}",
        f"x{tile_x:02d}",
    ]
    
    if method:
        parts.append(method)
    
    parts.append(suffix)
    
    return "_".join(parts) + extension


def is_tiff_available() -> bool:
    """Check if TIFF export is available."""
    return TIFF_AVAILABLE
