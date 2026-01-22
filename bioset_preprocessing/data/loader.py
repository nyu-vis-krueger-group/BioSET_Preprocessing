"""
Data loading utilities for Zarr arrays stored locally or on S3.

Handles both 4D (C, Z, Y, X) and 5D (T, C, Z, Y, X) arrays.
"""

import logging
from dataclasses import dataclass
from typing import Tuple, Union

import dask.array as da
from ome_zarr.io import parse_url

logger = logging.getLogger(__name__)


@dataclass
class ArrayInfo:
    """
    Information about a loaded volumetric array.

    Attributes:
        shape: Full array shape
        dtype: Data type (e.g., '>u2' for uint16 big-endian)
        ndim: Number of dimensions (4 or 5)
        n_timepoints: Number of timepoints (1 if 4D)
        n_channels: Number of channels
        n_z: Number of z-slices
        n_y: Y dimension size
        n_x: X dimension size
        bytes_per_voxel: Size in bytes of each voxel
    """
    shape: Tuple[int, ...]
    dtype: str
    ndim: int
    n_timepoints: int
    n_channels: int
    n_z: int
    n_y: int
    n_x: int
    bytes_per_voxel: int

    @property
    def spatial_shape(self) -> Tuple[int, int, int]:
        """Return (Z, Y, X) shape."""
        return (self.n_z, self.n_y, self.n_x)

    @property
    def total_voxels(self) -> int:
        """Total number of voxels per channel."""
        return self.n_z * self.n_y * self.n_x

    @property
    def channel_size_gb(self) -> float:
        """Size of one channel in gigabytes."""
        return self.total_voxels * self.bytes_per_voxel / 1e9

    def __repr__(self) -> str:
        return (
            f"ArrayInfo(\n"
            f"  shape={self.shape},\n"
            f"  dtype='{self.dtype}',\n"
            f"  dimensions: T={self.n_timepoints}, C={self.n_channels}, "
            f"Z={self.n_z}, Y={self.n_y}, X={self.n_x},\n"
            f"  bytes_per_voxel={self.bytes_per_voxel},\n"
            f"  channel_size={self.channel_size_gb:.2f} GB\n"
            f")"
        )


def load_zarr_array(
    url: str,
    component: str = "0",
) -> Tuple[da.Array, ArrayInfo]:
    """
    Load a Zarr array from a URL or local path.

    Supports both local paths and S3/HTTP URLs. Handles OME-Zarr format.

    Args:
        url: URL or path to Zarr store
        component: Resolution level ("0" = full res, "1" = 2x downsample, etc.)

    Returns:
        Tuple of (dask_array, array_info)

    Example:
        >>> arr, info = load_zarr_array(
        ...     "https://example.com/data.zarr",
        ...     component="2"
        ... )
        >>> print(info)
        ArrayInfo(shape=(1, 70, 194, 1377, 2727), ...)
    """
    logger.info(f"Loading Zarr from: {url}")
    logger.info(f"Resolution level: {component}")

    root = parse_url(url, mode="r")
    if root is None:
        raise ValueError(f"Could not parse URL: {url}")

    store = root.store
    arr = da.from_zarr(store, component=component)
    info = _parse_array_info(arr)

    logger.info(f"Loaded array: {info.shape}, dtype={info.dtype}")
    logger.info(
        f"Dimensions: T={info.n_timepoints}, C={info.n_channels}, "
        f"Z={info.n_z}, Y={info.n_y}, X={info.n_x}"
    )

    return arr, info


def _parse_array_info(arr: da.Array) -> ArrayInfo:
    """
    Parse array information from a dask array.

    Handles both 4D (C, Z, Y, X) and 5D (T, C, Z, Y, X) formats.
    """
    shape = arr.shape
    dtype = str(arr.dtype)
    ndim = len(shape)

    bytes_per_voxel = _get_bytes_per_voxel(dtype)

    if ndim == 5:
        # (T, C, Z, Y, X)
        n_t, n_c, n_z, n_y, n_x = shape
    elif ndim == 4:
        # (C, Z, Y, X)
        n_t = 1
        n_c, n_z, n_y, n_x = shape
    else:
        raise ValueError(
            f"Expected 4D or 5D array, got {ndim}D with shape {shape}"
        )

    return ArrayInfo(
        shape=shape,
        dtype=dtype,
        ndim=ndim,
        n_timepoints=n_t,
        n_channels=n_c,
        n_z=n_z,
        n_y=n_y,
        n_x=n_x,
        bytes_per_voxel=bytes_per_voxel,
    )


def _get_bytes_per_voxel(dtype: str) -> int:
    """
    Get bytes per voxel from dtype string.

    Common dtypes:
    - '>u2', '<u2', 'uint16': 2 bytes
    - '>u1', '<u1', 'uint8': 1 byte
    - '>f4', '<f4', 'float32': 4 bytes
    - '>f8', '<f8', 'float64': 8 bytes
    """
    import numpy as np
    return np.dtype(dtype).itemsize


def get_tile_data(
    arr: da.Array,
    info: ArrayInfo,
    channel: int,
    y_slice: slice,
    x_slice: slice,
    timepoint: int = 0,
) -> "np.ndarray":
    """
    Extract a tile of data from the array.

    Args:
        arr: Dask array
        info: ArrayInfo for the array
        channel: Channel index
        y_slice: Slice for Y dimension
        x_slice: Slice for X dimension
        timepoint: Timepoint index (default 0)

    Returns:
        NumPy array with shape (Z, Y_tile, X_tile)
    """
    if info.ndim == 5:
        tile = arr[timepoint, channel, :, y_slice, x_slice]
    else:
        tile = arr[channel, :, y_slice, x_slice]

    return tile.compute()
