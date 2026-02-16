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
from .stages.overlaps import OverlapTileResult, OverlapMiner

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

        self.pyr_local = ZarrPyramid.open(cfg.zarr_path) if cfg.zarr_path else None
        self.pyr_remote = ZarrPyramid.open(cfg.zarr_url) if cfg.zarr_url else None

        if not self.pyr_local and not self.pyr_remote:
            raise ValueError("Provide at least one of zarr_path (local) or zarr_url (remote).")

        def has_multi(pyr: ZarrPyramid) -> bool:
            return len(pyr.arrays) > 1

        if self.pyr_local:
            self.comp_hi, self.A_hi = self.pyr_local.highest_res()
            hi_source = "local"
        else:
            self.comp_hi, self.A_hi = self.pyr_remote.highest_res()
            hi_source = "remote"

        
        if self.pyr_local and has_multi(self.pyr_local):
            self.comp_lo, self.A_lo = self.pyr_local.lowest_res()
            global_source = "local(lowest)"
        elif self.pyr_local and not has_multi(self.pyr_local):
            if self.pyr_remote and has_multi(self.pyr_remote):
                self.comp_lo, self.A_lo = self.pyr_remote.lowest_res()
                global_source = "remote(lowest)"
            elif self.pyr_remote:
                self.comp_lo, self.A_lo = self.pyr_remote.highest_res()
                global_source = "remote(highest)"
            else:
                self.comp_lo, self.A_lo = self.comp_hi, self.A_hi
                global_source = f"{hi_source}(highest)"
        else:
            if has_multi(self.pyr_remote):
                self.comp_lo, self.A_lo = self.pyr_remote.lowest_res()
                global_source = "remote(lowest)"
            else:
                self.comp_lo, self.A_lo = self.pyr_remote.highest_res()
                global_source = "remote(highest)"

        print(f"[Pipeline] High res source: {hi_source} component={self.comp_hi} shape={self.A_hi.shape}")
        print(f"[Pipeline] Global threshold source: {global_source} component={self.comp_lo} shape={self.A_lo.shape}")

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

        self.overlap_miner = OverlapMiner(
            radii_um=cfg.dilate_um,
            max_set_size=cfg.max_set_size,
            min_marker_vox=cfg.min_marker_vox,
            min_support_pair=cfg.min_support_pair,
            min_support_set=cfg.min_support_set,
            aggressive_stop_on_fail=cfg.aggressive_stop_on_fail,
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

    def iter_tile_overlap_outputs(self) -> Iterator[OverlapTileResult]:
        if not self._t_global:
            self.compute_global_thresholds()

        _, _, z, y, x = self.A_hi.shape
        tile_y, tile_x = self.cfg.tile_xy
        radii = sorted(float(r) for r in self.cfg.dilate_um)
        
        n_tiles_y = (y + tile_y - 1) // tile_y
        n_tiles_x = (x + tile_x - 1) // tile_x
        total_tiles = n_tiles_y * n_tiles_x
        tile_count = 0

        for tile in iter_tiles_xy(y, x, tile_y=tile_y, tile_x=tile_x):
            tile_count += 1
            ys, xs = tile_slices(tile, tile_y, tile_x)

            masks: Dict[float, Dict[int, cp.ndarray]] = {r: {} for r in radii}
            marker_vox: Dict[float, Dict[int, int]] = {r: {} for r in radii}

            for ch_batch in chunked(self.cfg.channels, self.cfg.channel_batch):
                vol_cpu = self.A_hi[0, ch_batch, :, ys, xs].compute()
                vol_gpu = cp.asarray(vol_cpu)

                for i, ch in enumerate(ch_batch):
                    v = vol_gpu[i]
                    tstats = self.th.compute_tile_gpu(v, self._t_global[ch])
                    mask0 = self.th.apply_gpu(v, tstats.t_final)
                    mask1, _ = self.cc(mask0)
                    
                    dilres: DilationResult = self.dil(mask1)

                    for r in radii:
                        m = dilres.dilated[r]
                        masks[r][ch] = m
                        marker_vox[r][ch] = int(cp.count_nonzero(m))
            
            del vol_gpu
            
            result = self.overlap_miner.run(
                tile_x=tile.tx,
                tile_y=tile.ty,
                masks=masks,
                marker_vox=marker_vox,
            )
            
            del masks
            cp.get_default_memory_pool().free_all_blocks()
            
            yield result