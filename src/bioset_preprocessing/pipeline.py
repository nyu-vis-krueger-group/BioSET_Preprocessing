from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Iterator, Sequence, List, Set, Tuple
from pathlib import Path
import time

import numpy as np
import cupy as cp

from .config import PipelineConfig
from .io import ZarrPyramid
from .tiling import iter_tiles_xy, tile_slices, TileIndex
from .stages.threshold import AlphaThreshold, ThresholdStats
from .stages.cc_filter import ConnectedComponentsFilter, CCStats
from .stages.dilation import EDTSweepDilation, DilationResult
from .stages.overlaps import OverlapTileResult, OverlapMiner
from .checkpoint import (
    save_tile_checkpoint, 
    get_completed_tiles, 
    load_all_checkpoints,
    get_checkpoint_stats,
)

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

    def _process_single_tile(
        self,
        tile: TileIndex,
        radii: List[float],
        z: int,
    ) -> OverlapTileResult:
        """Process a single tile and return the result."""
        tile_y, tile_x = self.cfg.tile_xy
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
        
        return result

    def run_tile_processing(
        self,
        resume: bool = True,
    ) -> int:
        """
        STAGE 1 (GPU): Process tiles and save checkpoints.
        
        This can be interrupted and resumed. Each tile is saved immediately
        after processing.
        
        Args:
            resume: If True, skip tiles that have already been checkpointed.
        
        Returns:
            Number of tiles processed in this run.
        """
        if not self._t_global:
            self.compute_global_thresholds()

        _, _, z, y, x = self.A_hi.shape
        tile_y, tile_x = self.cfg.tile_xy
        radii = sorted(float(r) for r in self.cfg.dilate_um)
        
        n_tiles_y = (y + tile_y - 1) // tile_y
        n_tiles_x = (x + tile_x - 1) // tile_x
        total_tiles = n_tiles_y * n_tiles_x
        
        checkpoint_dir = Path(self.cfg.checkpoint_dir) / self.cfg.output_name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        completed: Set[Tuple[int, int]] = set()
        if resume:
            completed = get_completed_tiles(checkpoint_dir)
            if completed:
                print(f"[Checkpoint] Found {len(completed)} completed tiles, resuming...")
        
        print(f"[Pipeline] Processing {total_tiles} tiles ({n_tiles_y} x {n_tiles_x})")
        print(f"[Pipeline] Checkpoints: {checkpoint_dir}")
        
        tiles_processed = 0
        tiles_skipped = 0
        start_time = time.time()
        
        for tile in iter_tiles_xy(y, x, tile_y=tile_y, tile_x=tile_x):
            tile_key = (tile.tx, tile.ty)
            
            if tile_key in completed:
                tiles_skipped += 1
                continue
            
            result = self._process_single_tile(tile, radii, z)
            
            save_tile_checkpoint(checkpoint_dir, result)
            
            tiles_processed += 1
            total_done = tiles_processed + tiles_skipped
            
            if tiles_processed % 5 == 0:
                elapsed = time.time() - start_time
                rate = tiles_processed / elapsed if elapsed > 0 else 0
                remaining = (total_tiles - total_done) / rate if rate > 0 else 0
                print(
                    f"  [{total_done}/{total_tiles}] "
                    f"Processed: {tiles_processed}, Skipped: {tiles_skipped}, "
                    f"Rate: {rate:.2f} tiles/sec, ETA: {remaining/60:.1f} min"
                )
        
        elapsed = time.time() - start_time
        print(f"\n[Pipeline] Tile processing complete!")
        print(f"  Processed: {tiles_processed} tiles")
        print(f"  Skipped (resumed): {tiles_skipped} tiles")
        print(f"  Time: {elapsed/60:.1f} minutes")
        print(f"  Checkpoints saved to: {checkpoint_dir}")
        
        return tiles_processed

    def run_aggregation(
        self,
        channel_names: List[str] | None = None,
    ) -> Path:
        """
        STAGE 2 (CPU): Load checkpoints, aggregate, and write database.
        
        This runs entirely on CPU and can be run separately from tile processing.
        
        Returns:
            Path to the output .bioset file.
        """
        from .aggregation import HierarchicalAggregator
        from .writer import BiosetWriter
        
        _, _, z, y, x = self.A_hi.shape
        tile_y, tile_x = self.cfg.tile_xy
        
        if channel_names is None:
            channel_names = [f"ch{i}" for i in self.cfg.channels]
        
        checkpoint_dir = Path(self.cfg.checkpoint_dir) / self.cfg.output_name
        
        stats = get_checkpoint_stats(checkpoint_dir)
        print(f"[Aggregation] Loading checkpoints from: {checkpoint_dir}")
        print(f"[Aggregation] Found {stats['n_completed']} checkpointed tiles")
        
        if stats['n_completed'] == 0:
            raise RuntimeError(
                f"No checkpoints found in {checkpoint_dir}. "
                "Run run_tile_processing() first."
            )
        
        print("[Aggregation] Loading tile results...")
        start_time = time.time()
        results = load_all_checkpoints(checkpoint_dir)
        print(f"  Loaded {len(results)} tiles in {time.time() - start_time:.1f}s")
        
        aggregator = HierarchicalAggregator(
            base_tile_y=tile_y,
            base_tile_x=tile_x,
            n_levels=self.cfg.hierarchy_levels,
        )
        
        print("[Aggregation] Adding results to aggregator...")
        for result in results:
            aggregator.add_tile_result(result)
        
        print("[Aggregation] Computing hierarchical aggregation...")
        start_time = time.time()
        levels = aggregator.aggregate()
        print(f"  Aggregation took {time.time() - start_time:.1f}s")
        
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
            print(f"  Level {lvl.level}: {len(lvl.pairs)} pairs, {len(lvl.sets)} sets")
        
        output_dir = Path(self.cfg.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{self.cfg.output_name}.bioset"
        
        print(f"[Aggregation] Writing to {output_path}...")
        start_time = time.time()
        
        writer = BiosetWriter(
            output_path=output_path,
            channel_names=channel_names,
            dilation_amounts=self.cfg.dilate_um,
            volume_shape=(z, y, x),
        )
        
        writer.write_metadata(hierarchy_meta)
        
        for level in levels:
            writer.write_hierarchy_level(level)
        
        final_path = writer.finalize()
        print(f"  Database written in {time.time() - start_time:.1f}s")
        print(f"\n[Aggregation] Complete! Output: {final_path}")
        
        return final_path

    def run_full_analysis(
        self,
        channel_names: List[str] | None = None,
        resume: bool = True,
    ) -> Path:
        self.run_tile_processing(resume=resume)
        return self.run_aggregation(channel_names=channel_names)

    def iter_tile_overlap_outputs(self) -> Iterator[OverlapTileResult]:
        """Iterate over tile results (legacy method, no checkpointing)."""
        if not self._t_global:
            self.compute_global_thresholds()

        _, _, z, y, x = self.A_hi.shape
        tile_y, tile_x = self.cfg.tile_xy
        radii = sorted(float(r) for r in self.cfg.dilate_um)

        for tile in iter_tiles_xy(y, x, tile_y=tile_y, tile_x=tile_x):
            yield self._process_single_tile(tile, radii, z)