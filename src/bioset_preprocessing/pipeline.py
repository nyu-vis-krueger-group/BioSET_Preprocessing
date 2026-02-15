from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Iterator, Sequence, List

import numpy as np
import cupy as cp

from .config import PipelineConfig
from .io import ZarrPyramid
from .tiling import iter_tiles_xy, tile_slices, TileIndex
from .stages.threshold import AlphaThreshold, ThresholdStats
from .stages.cc_filter import ConnectedComponentsFilter, CCStats
from .stages.dilation import EDTSweepDilation, DilationResult

@dataclass
class TileOutput:
    channel: int
    tile: TileIndex
    threshold: ThresholdStats
    cc: CCStats
    masks: Dict[float, cp.ndarray] 

def chunked(seq: Sequence[int], k: int) -> Iterator[List[int]]:
    for i in range(0, len(seq), k):
        yield list(seq[i:i+k])

class Pipeline:
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.pyr = ZarrPyramid.open(cfg.zarr_url)

        self.comp_hi, self.A_hi = self.pyr.highest_res()
        if self.pyr.is_multiplexed() and cfg.prefer_lowest_res_for_global:
            self.comp_lo, self.A_lo = self.pyr.lowest_res()
        else:
            self.comp_lo, self.A_lo = self.comp_hi, self.A_hi

        self.th = AlphaThreshold(alpha=cfg.alpha, trim_q=cfg.trim_q)
        self.cc = ConnectedComponentsFilter(
            min_obj_vol_um3=cfg.min_obj_vol_um3,
            voxel_vol_um3=cfg.voxel_size_um.voxel_volume_um3,
            connectivity=cfg.connectivity,
        )
        self.dil = EDTSweepDilation(
            radii_um=cfg.dilate_um,
            sampling_zyx_um=cfg.voxel_size_um.sampling_zyx,
            float64_distances=cfg.float64_distances,  
        )

        self._t_global: Dict[int, float] = {}

    def compute_global_thresholds(self) -> Dict[int, float]:
        for ch in self.cfg.channels:
            vol_lr = self.A_lo[0, ch, :, :, :].compute().astype(np.float32)
            self._t_global[ch] = self.th.compute_global(vol_lr)
        return dict(self._t_global)

    def iter_tile_outputs(self) -> Iterator[TileOutput]:
        if not self._t_global:
            self.compute_global_thresholds()

        _, _, z, y, x = self.A_hi.shape
        tile_y, tile_x = self.cfg.tile_xy

        for tile in iter_tiles_xy(y, x, tile_y=tile_y, tile_x=tile_x):
            ys, xs = tile_slices(tile, tile_y, tile_x)

            for ch_batch in chunked(self.cfg.channels, self.cfg.channel_batch):
                vol_cpu = self.A_hi[0, ch_batch, :, ys, xs].compute()     
                vol_gpu = cp.asarray(vol_cpu)                             

                for i, ch in enumerate(ch_batch):
                    v = vol_gpu[i]  
                    tstats = self.th.compute_tile_gpu(v, self._t_global[ch])
                    mask0 = self.th.apply_gpu(v, tstats.t_final)

                    mask1, ccstats = self.cc(mask0)

                    dilres: DilationResult = self.dil(mask1)

                    yield TileOutput(
                        channel=ch,
                        tile=tile,
                        threshold=tstats,
                        cc=ccstats,
                        masks=dilres.dilated,
                    )
