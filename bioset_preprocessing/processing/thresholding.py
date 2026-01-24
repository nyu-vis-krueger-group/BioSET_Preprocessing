"""
Thresholding methods for binary segmentation.

Supports GPU acceleration via CuPy when available.
"""

import logging
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple
# from skimage.filters import threshold_otsu as sk_threshold_otsu

import numpy as np

logger = logging.getLogger(__name__)

# GPU availability check
_GPU_AVAILABLE = False
_cp = None

def _init_gpu():
    """Initialize GPU support if available."""
    global _GPU_AVAILABLE, _cp
    
    if _cp is not None:
        return _GPU_AVAILABLE
    
    try:
        import cupy as cp
        # Test that GPU actually works
        _ = cp.array([1, 2, 3])
        cp.get_default_memory_pool().free_all_blocks()
        _cp = cp
        _GPU_AVAILABLE = True
        logger.info("GPU acceleration available (CuPy)")
    except Exception as e:
        _cp = np  # Fallback to numpy
        _GPU_AVAILABLE = False
        logger.info(f"GPU not available, using CPU: {e}")
    
    return _GPU_AVAILABLE


def is_gpu_available() -> bool:
    """Check if GPU acceleration is available."""
    _init_gpu()
    return _GPU_AVAILABLE


@dataclass
class ThresholdResult:
    """
    Result of a thresholding operation.
    
    Attributes:
        mask: Binary mask (uint8, 0 or 1)
        threshold_value: The threshold value used
        active_voxels: Number of voxels above threshold
        active_fraction: Fraction of voxels above threshold
        method: Name of the thresholding method used
    """
    mask: np.ndarray
    threshold_value: float
    active_voxels: int
    active_fraction: float
    method: str
    
    def __repr__(self) -> str:
        return (
            f"ThresholdResult(method='{self.method}', "
            f"threshold={self.threshold_value:.2f}, "
            f"active={self.active_fraction:.2%})"
        )


# =============================================================================
# Thresholding Functions
# =============================================================================

def threshold_percentile(
    data: np.ndarray,
    percentile: float = 95.0,
) -> ThresholdResult:
    """
    Threshold using percentile.
    
    Voxels above the given percentile are considered "active".
    
    Args:
        data: Input array (any shape)
        percentile: Percentile value (0-100)
    
    Returns:
        ThresholdResult with binary mask and metadata
    """
    _init_gpu()
    
    if _GPU_AVAILABLE:
        data_gpu = _cp.asarray(data)
        thresh = float(_cp.percentile(data_gpu, percentile))
        mask = _cp.asnumpy(data_gpu > thresh).astype(np.uint8)
        del data_gpu
        _cp.get_default_memory_pool().free_all_blocks()
    else:
        thresh = float(np.percentile(data, percentile))
        mask = (data > thresh).astype(np.uint8)
    
    active = int(np.sum(mask))
    
    return ThresholdResult(
        mask=mask,
        threshold_value=thresh,
        active_voxels=active,
        active_fraction=active / mask.size,
        method=f"percentile_{int(percentile)}",
    )


def threshold_otsu(data: np.ndarray, nbins: Optional[int] = None) -> ThresholdResult:
    """
    Otsu's automatic thresholding.
    
    Finds the threshold that minimizes intra-class variance
    (equivalently, maximizes inter-class variance).
    
    Args:
        data: Input array (any shape)
        nbins: Number of bins for histogram (default: 256)
    
    Returns:
        ThresholdResult with binary mask and metadata
    """
    _init_gpu()
    
    if nbins is None:
        dtype_str = str(data.dtype)
        if data.dtype == np.uint16 or dtype_str in ('>u2', '<u2'):
            nbins = 65536  # 16 bit full precision
        elif data.dtype == np.uint8:
            nbins = 256
        else:
            nbins = 4096  # default
    
    if _GPU_AVAILABLE:
        thresh = _otsu_threshold_gpu(data, nbins)
        data_gpu = _cp.asarray(data)
        mask = _cp.asnumpy(data_gpu > thresh).astype(np.uint8)
        del data_gpu
        _cp.get_default_memory_pool().free_all_blocks()
    else:
        thresh = _otsu_threshold_cpu(data, nbins)
        mask = (data > thresh).astype(np.uint8)
    
    active = int(np.sum(mask))
    
    return ThresholdResult(
        mask=mask,
        threshold_value=thresh,
        active_voxels=active,
        active_fraction=active / mask.size,
        method="otsu",
    )

def _otsu_threshold_cpu(data: np.ndarray, nbins: int) -> float:
    """
    Compute Otsu threshold on CPU.
    
    Uses histogram-based approach for efficiency.
    """
    data_min = float(np.min(data))
    data_max = float(np.max(data))
    
    if data_min == data_max:
        return data_min
    
    hist, bin_edges = np.histogram(data.ravel(), bins=nbins, range=(data_min, data_max))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    hist = hist.astype(np.float64)
    hist_norm = hist / hist.sum()
    
    weight1 = np.cumsum(hist_norm)               
    weight2 = np.cumsum(hist_norm[::-1])[::-1]   
    
    mean1 = np.cumsum(hist_norm * bin_centers)   
    mean2 = np.cumsum((hist_norm * bin_centers)[::-1])[::-1]  
    with np.errstate(divide='ignore', invalid='ignore'):
        mean1 = np.where(weight1 > 0, mean1 / weight1, 0)
        mean2 = np.where(weight2 > 0, mean2 / weight2, 0)
    
    variance_between = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2
    
    variance_between = np.nan_to_num(variance_between, nan=0.0)

    idx = np.argmax(variance_between)
    thresh = float(bin_centers[idx])
    
    return thresh


def _otsu_threshold_gpu(data: np.ndarray, nbins: int) -> float:
    """
    Compute Otsu threshold on GPU using CuPy.
    
    Accelerates histogram computation and variance calculation.
    """
    data_gpu = _cp.asarray(data.ravel())
    
    data_min = float(_cp.min(data_gpu))
    data_max = float(_cp.max(data_gpu))
    
    if data_min == data_max:
        del data_gpu
        _cp.get_default_memory_pool().free_all_blocks()
        return data_min
    
    hist, bin_edges = _cp.histogram(data_gpu, bins=nbins, range=(data_min, data_max))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    del data_gpu
    
    hist = hist.astype(_cp.float64)
    hist_norm = hist / hist.sum()

    weight1 = _cp.cumsum(hist_norm)
    weight2 = _cp.cumsum(hist_norm[::-1])[::-1]
    
    mean1 = _cp.cumsum(hist_norm * bin_centers)
    mean2 = _cp.cumsum((hist_norm * bin_centers)[::-1])[::-1]
    mean1 = _cp.where(weight1 > 0, mean1 / weight1, 0)
    mean2 = _cp.where(weight2 > 0, mean2 / weight2, 0)

    variance_between = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2
    variance_between = _cp.nan_to_num(variance_between, nan=0.0)
    idx = int(_cp.argmax(variance_between))
    thresh = float(bin_centers[idx])
    
    # Cleanup
    del hist, bin_edges, bin_centers, hist_norm
    del weight1, weight2, mean1, mean2, variance_between
    _cp.get_default_memory_pool().free_all_blocks()
    
    return thresh


def threshold_mean_std(
    data: np.ndarray,
    n_std: float = 2.0,
) -> ThresholdResult:
    """
    Threshold at mean + n * standard_deviation.
    
    Args:
        data: Input array (any shape)
        n_std: Number of standard deviations above mean
    
    Returns:
        ThresholdResult with binary mask and metadata
    """
    mean = float(np.mean(data))
    std = float(np.std(data))
    thresh = mean + n_std * std
    
    mask = (data > thresh).astype(np.uint8)
    active = int(np.sum(mask))
    
    return ThresholdResult(
        mask=mask,
        threshold_value=thresh,
        active_voxels=active,
        active_fraction=active / mask.size,
        method=f"mean_{int(n_std)}std",
    )


def threshold_fixed(
    data: np.ndarray,
    value: float,
) -> ThresholdResult:
    """
    Fixed threshold value.
    
    Args:
        data: Input array (any shape)
        value: Threshold value
    
    Returns:
        ThresholdResult with binary mask and metadata
    """
    mask = (data > value).astype(np.uint8)
    active = int(np.sum(mask))
    
    return ThresholdResult(
        mask=mask,
        threshold_value=value,
        active_voxels=active,
        active_fraction=active / mask.size,
        method="fixed",
    )


# =============================================================================
# Method Registry
# =============================================================================

# Type alias for threshold functions
ThresholdFunction = Callable[[np.ndarray], ThresholdResult]

# Registry of available methods
_THRESHOLD_METHODS: Dict[str, ThresholdFunction] = {
    "percentile_95": lambda d: threshold_percentile(d, 95),
    "percentile_90": lambda d: threshold_percentile(d, 90),
    "percentile_99": lambda d: threshold_percentile(d, 99),
    "otsu": threshold_otsu,
    "mean_2std": lambda d: threshold_mean_std(d, 2.0),
    "mean_3std": lambda d: threshold_mean_std(d, 3.0),
}


def get_threshold_method(name: str) -> ThresholdFunction:
    """
    Get a threshold function by name.
    
    Args:
        name: Method name (e.g., "otsu", "percentile_95")
    
    Returns:
        Threshold function
        
    Raises:
        ValueError: If method name is not recognized
    """
    if name not in _THRESHOLD_METHODS:
        available = ", ".join(_THRESHOLD_METHODS.keys())
        raise ValueError(
            f"Unknown threshold method: '{name}'. "
            f"Available methods: {available}"
        )
    return _THRESHOLD_METHODS[name]


def list_threshold_methods() -> list:
    """Return list of available threshold method names."""
    return list(_THRESHOLD_METHODS.keys())


def register_threshold_method(
    name: str,
    func: ThresholdFunction,
) -> None:
    """
    Register a custom threshold method.
    
    The function should take a numpy array and return a ThresholdResult.
    
    Args:
        name: Name for the method
        func: Threshold function
        
    Example:
        >>> def my_threshold(data):
        ...     thresh = np.median(data) * 2
        ...     mask = (data > thresh).astype(np.uint8)
        ...     return ThresholdResult(
        ...         mask=mask,
        ...         threshold_value=thresh,
        ...         active_voxels=int(np.sum(mask)),
        ...         active_fraction=np.sum(mask) / mask.size,
        ...         method="my_custom"
        ...     )
        >>> register_threshold_method("my_custom", my_threshold)
    """
    _THRESHOLD_METHODS[name] = func
    logger.info(f"Registered custom threshold method: {name}")


def apply_threshold(
    data: np.ndarray,
    method: str = "percentile_95",
) -> ThresholdResult:
    """
    Apply a threshold method to data.
    
    Convenience function that looks up the method and applies it.
    
    Args:
        data: Input array
        method: Method name
    
    Returns:
        ThresholdResult
        
    Example:
        >>> result = apply_threshold(tile_data, method="otsu")
        >>> print(f"Threshold: {result.threshold_value}")
    """
    func = get_threshold_method(method)
    return func(data)
