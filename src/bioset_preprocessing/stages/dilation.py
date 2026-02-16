from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence, Dict, Optional
import cupy as cp
from cupyx.scipy.ndimage import distance_transform_edt  

@dataclass
class DilationResult:
    dist_um: Optional[cp.ndarray]  
    dilated: Dict[float, cp.ndarray]  

class EDTSweepDilation:
    def __init__(self, radii_um: Sequence[float], sampling_zyx_um: tuple[float, float, float], float64_distances: bool = False):
        self.radii_um = [float(r) for r in radii_um]
        self.sampling = sampling_zyx_um
        self.float64_distances = float64_distances

    def __call__(self, mask_gpu: cp.ndarray) -> DilationResult:
        dist = distance_transform_edt(~mask_gpu, sampling=self.sampling, float64_distances=self.float64_distances)
        if dist.dtype != cp.float32 and not self.float64_distances:
            dist = dist.astype(cp.float32)

        dilated = {r: (dist <= r) for r in self.radii_um}
        return DilationResult(dist_um=dist, dilated=dilated)
