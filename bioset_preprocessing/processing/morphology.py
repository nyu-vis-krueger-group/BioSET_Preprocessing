"""
Morphological operations: connected component filtering and dilation.
"""

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_GPU_AVAILABLE = False
_cp = None


def _init_gpu():
    """Initialize GPU support if available."""
    global _GPU_AVAILABLE, _cp
    
    if _cp is not None:
        return _GPU_AVAILABLE
    
    try:
        import cupy as cp
        _ = cp.array([1, 2, 3])
        cp.get_default_memory_pool().free_all_blocks()
        _cp = cp
        _GPU_AVAILABLE = True
        logger.info("GPU acceleration available for morphology (CuPy)")
    except Exception as e:
        _cp = np
        _GPU_AVAILABLE = False
        logger.debug(f"GPU not available for morphology: {e}")
    
    return _GPU_AVAILABLE


@dataclass
class CCFilterResult:
    """Result of connected component filtering."""
    mask: np.ndarray
    original_components: int
    remaining_components: int
    removed_components: int
    original_voxels: int
    remaining_voxels: int
    
    def __repr__(self) -> str:
        return (
            f"CCFilterResult(components: {self.original_components} -> {self.remaining_components}, "
            f"voxels: {self.original_voxels} -> {self.remaining_voxels})"
        )


def filter_connected_components(
    mask: np.ndarray,
    min_volume_um3: float,
    voxel_volume_um3: float,
    connectivity: int = 26,
) -> CCFilterResult:
    """
    Remove connected components smaller than minimum volume.
    
    Args:
        mask: Binary mask (uint8, values 0 or 1/255)
        min_volume_um3: Minimum component volume in cubic micrometers
        voxel_volume_um3: Volume of a single voxel in cubic micrometers
        connectivity: Connectivity for labeling (6, 18, or 26 for 3D)
        
    Returns:
        CCFilterResult with filtered mask and statistics
    """
    min_voxels = int(np.ceil(min_volume_um3 / voxel_volume_um3))
    logger.debug(f"Min volume: {min_volume_um3} μm³ = {min_voxels} voxels")
    
    binary_mask = (mask > 0).astype(np.uint8)
    original_voxels = int(np.sum(binary_mask))
    
    if original_voxels == 0:
        return CCFilterResult(
            mask=binary_mask,
            original_components=0,
            remaining_components=0,
            removed_components=0,
            original_voxels=0,
            remaining_voxels=0,
        )
    
    try:
        import cc3d
        logger.debug("Using cc3d for connected component analysis")
        labels, n_components = cc3d.connected_components(
            binary_mask, 
            connectivity=connectivity,
            return_N=True,
        )
        component_sizes = cc3d.statistics(labels)['voxel_counts']
        valid_labels = np.where(component_sizes >= min_voxels)[0]
        valid_labels = valid_labels[valid_labels > 0]
        
        if len(valid_labels) == 0:
            filtered_mask = np.zeros_like(binary_mask)
        else:
            filtered_mask = cc3d.dust(
                binary_mask,
                threshold=min_voxels,
                connectivity=connectivity,
                in_place=False,
            )
        
    except ImportError:
        logger.debug("cc3d not available, using scipy.ndimage.label")
        from scipy.ndimage import label
        
        if connectivity == 6:
            struct = np.array([
                [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
                [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
            ])
        else:
            struct = np.ones((3, 3, 3), dtype=np.uint8)
        
        labels, n_components = label(binary_mask, structure=struct)
        
        component_sizes = np.bincount(labels.ravel())
        
        valid_labels = np.where(component_sizes >= min_voxels)[0]
        valid_labels = valid_labels[valid_labels > 0]
        
        if len(valid_labels) == 0:
            filtered_mask = np.zeros_like(binary_mask)
        else:
            valid_mask = np.zeros(len(component_sizes), dtype=bool)
            valid_mask[valid_labels] = True
            filtered_mask = valid_mask[labels].astype(np.uint8)
    
    remaining_voxels = int(np.sum(filtered_mask))
    remaining_components = len(valid_labels) if 'valid_labels' in dir() else 0
    
    return CCFilterResult(
        mask=filtered_mask,
        original_components=n_components,
        remaining_components=remaining_components,
        removed_components=n_components - remaining_components,
        original_voxels=original_voxels,
        remaining_voxels=remaining_voxels,
    )


def dilate_mask(
    mask: np.ndarray,
    radius_um: float,
    spacing_um: Tuple[float, float, float],
) -> np.ndarray:
    """
    Dilate a binary mask by a physical radius.
    
    Uses distance transform for efficiency (compute once, threshold at any radius).
    
    Args:
        mask: Binary mask (Z, Y, X)
        radius_um: Dilation radius in micrometers
        spacing_um: Voxel spacing (Z, Y, X) in micrometers
        
    Returns:
        Dilated binary mask (uint8)
    """
    from scipy.ndimage import distance_transform_edt
    
    if radius_um <= 0:
        return mask.copy()
    
    binary_mask = (mask > 0).astype(np.uint8)
    
    if np.sum(binary_mask) == 0:
        return binary_mask
    
    distances = distance_transform_edt(
        binary_mask == 0,  
        sampling=spacing_um,  
    )
    
    dilated = ((distances <= radius_um) | (binary_mask > 0)).astype(np.uint8)
    
    return dilated


def compute_distance_transform(
    mask: np.ndarray,
    spacing_um: Tuple[float, float, float],
) -> np.ndarray:
    """
    Compute distance transform for a mask.
    
    Returns distance from each background voxel to nearest foreground voxel.
    Can be reused for multiple dilation radii.
    
    Uses GPU acceleration via CuPy when available.
    
    Args:
        mask: Binary mask (Z, Y, X)
        spacing_um: Voxel spacing (Z, Y, X) in micrometers
        
    Returns:
        Distance array (float32) in micrometers
    """
    _init_gpu()
    
    binary_mask = (mask > 0).astype(np.uint8)
    
    if np.sum(binary_mask) == 0:
        return np.full(mask.shape, np.inf, dtype=np.float32)
    
    if _GPU_AVAILABLE:
        try:
            from cupyx.scipy.ndimage import distance_transform_edt as gpu_edt
            
            mask_gpu = _cp.asarray(binary_mask == 0)
            
            distances_gpu = gpu_edt(
                mask_gpu,
                sampling=spacing_um,
                return_distances=True,
                return_indices=False,
                float64_distances=False,
            )
            
            distances = _cp.asnumpy(distances_gpu).astype(np.float32)
            
            del mask_gpu, distances_gpu
            _cp.get_default_memory_pool().free_all_blocks()
            
            logger.debug("Distance transform computed on GPU")
            return distances
            
        except Exception as e:
            logger.warning(f"GPU distance transform failed, falling back to CPU: {e}")
    
    # CPU fallback using scipy
    from scipy.ndimage import distance_transform_edt
    
    distances = distance_transform_edt(
        binary_mask == 0,
        sampling=spacing_um,
    ).astype(np.float32)
    
    return distances


def dilate_from_distance_transform(
    distances: np.ndarray,
    original_mask: np.ndarray,
    radius_um: float,
) -> np.ndarray:
    """
    Create dilated mask from pre-computed distance transform.
    
    Args:
        distances: Distance transform array
        original_mask: Original binary mask
        radius_um: Dilation radius in micrometers
        
    Returns:
        Dilated binary mask (uint8)
    """
    if radius_um <= 0:
        return (original_mask > 0).astype(np.uint8)
    
    dilated = ((distances <= radius_um) | (original_mask > 0)).astype(np.uint8)
    return dilated