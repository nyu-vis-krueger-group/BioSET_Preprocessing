from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import cupy as cp
from cupyx.scipy.ndimage import label as cc_label  
from cupyx.scipy.ndimage import sum_labels         

@dataclass
class CCStats:
    n_components: int
    min_voxels: int

class ConnectedComponentsFilter:
    def __init__(self, min_obj_vol_um3: float, voxel_vol_um3: float, connectivity: int = 26):
        self.min_voxels = int(np.ceil(min_obj_vol_um3 / voxel_vol_um3))
        self.connectivity = connectivity

    def _structure(self) -> cp.ndarray:
        if self.connectivity == 6:
            s = cp.zeros((3,3,3), dtype=cp.int8)
            s[1,1,0] = s[1,1,2] = 1
            s[1,0,1] = s[1,2,1] = 1
            s[0,1,1] = s[2,1,1] = 1
            s[1,1,1] = 1
            return s
        return cp.ones((3,3,3), dtype=cp.int8)

    def __call__(self, mask_gpu: cp.ndarray) -> tuple[cp.ndarray, CCStats]:
        labels, n = cc_label(mask_gpu, structure=self._structure())
        if n == 0:
            return mask_gpu, CCStats(n_components=0, min_voxels=self.min_voxels)

        ones = cp.ones(labels.shape, dtype=cp.int32)
        idx = cp.arange(1, n + 1, dtype=cp.int32)
        sizes = sum_labels(ones, labels=labels, index=idx)

        keep = idx[sizes >= self.min_voxels]
        if keep.size == 0:
            return cp.zeros_like(mask_gpu), CCStats(n_components=int(n), min_voxels=self.min_voxels)

        filtered = cp.isin(labels, keep)
        return filtered, CCStats(n_components=int(n), min_voxels=self.min_voxels)
