from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Iterator, Sequence, List
from pathlib import Path

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

            actual_z = z
            actual_y = ys.stop - ys.start
            actual_x = xs.stop - xs.start
            tile_shape = (actual_z, actual_y, actual_x)
            total_voxels = actual_z * actual_y * actual_x

            masks: Dict[float, Dict[int, cp.ndarray]] = {r: {} for r in radii}
            
            marker_vox_gpu: Dict[float, Dict[int, cp.ndarray]] = {r: {} for r in radii}
            sum_intensity_gpu: Dict[float, Dict[int, cp.ndarray]] = {r: {} for r in radii}

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
                        marker_vox_gpu[r][ch] = cp.count_nonzero(m)
                        if r == radii[0]:
                            if cp.any(m):
                                sum_intensity_gpu[r][ch] = cp.sum(v[m])
                            else:
                                sum_intensity_gpu[r][ch] = cp.asarray(0.0)
                        else:
                            sum_intensity_gpu[r][ch] = sum_intensity_gpu[radii[0]][ch]
                
                del vol_gpu

            marker_vox: Dict[float, Dict[int, int]] = {r: {} for r in radii}
            sum_intensity: Dict[float, Dict[int, float]] = {r: {} for r in radii}
            
            gpu_scalars = []
            scalar_keys = []  
            
            for r in radii:
                for ch in self.cfg.channels:
                    gpu_scalars.append(marker_vox_gpu[r][ch])
                    scalar_keys.append(('vox', r, ch))
                    gpu_scalars.append(sum_intensity_gpu[r][ch])
                    scalar_keys.append(('int', r, ch))
            
           
            cpu_values = [float(s.get()) for s in gpu_scalars]  
            
            for i, (typ, r, ch) in enumerate(scalar_keys):
                if typ == 'vox':
                    marker_vox[r][ch] = int(cpu_values[i])
                else:
                    sum_intensity[r][ch] = cpu_values[i]
            
            del marker_vox_gpu
            del sum_intensity_gpu
            
            result = self.overlap_miner.run(
                tile_x=tile.tx,
                tile_y=tile.ty,
                tile_shape=tile_shape,
                total_voxels=total_voxels,
                masks=masks,
                marker_vox=marker_vox,
                sum_intensity=sum_intensity,  
            )
            
            del masks
            cp.get_default_memory_pool().free_all_blocks()
            
            yield result

    def run_full_analysis(
        self,
        channel_names: List[str] | None = None,
    ) -> Path:
        from .aggregation import HierarchicalAggregator
        from .writer import BiosetWriter
        
        if not self._t_global:
            self.compute_global_thresholds()
        
        _, _, z, y, x = self.A_hi.shape
        tile_y, tile_x = self.cfg.tile_xy
        
        if channel_names is None:
            channel_names = [f"ch{i}" for i in self.cfg.channels]
        
        aggregator = HierarchicalAggregator(
            base_tile_y=tile_y,
            base_tile_x=tile_x,
            n_levels=self.cfg.hierarchy_levels,
        )
        
        # Process all tiles
        print(f"Processing tiles...")
        tile_count = 0
        for result in self.iter_tile_overlap_outputs():
            aggregator.add_tile_result(result)
            tile_count += 1
            if tile_count % 5 == 0:
                print(f"  Processed {tile_count} tiles...")
        
        print(f"Total tiles: {tile_count}")
        
        # Aggregate
        print("Aggregating hierarchies...")
        levels = aggregator.aggregate()
        
        # Build hierarchy level metadata
        hierarchy_meta = []
        for lvl in levels:
            hierarchy_meta.append({
                "level": lvl.level,
                "tile_size_x": lvl.tile_size_x,
                "tile_size_y": lvl.tile_size_y,
                "n_channels": len(lvl.channels),
                "n_pairs": len(lvl.pairs),
                "n_sets": len(lvl.sets),
            })
        
        # Write to database
        output_path = Path(self.cfg.output_dir) / f"{self.cfg.output_name}.bioset"
        print(f"Writing to {output_path}...")
        
        writer = BiosetWriter(
            output_path=output_path,
            channel_names=channel_names,
            dilation_amounts=self.cfg.dilate_um,
            volume_shape=(z, y, x),
        )
        
        writer.write_metadata(hierarchy_meta)
        
        for level in levels:
            print(f"  Writing level {level.level}...")
            writer.write_hierarchy_level(level)
        
        final_path = writer.finalize()
        print(f"Done! Output: {final_path}")
        
        return final_path