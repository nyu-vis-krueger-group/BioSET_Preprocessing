"""
Main pipeline orchestration.

Provides the Pipeline class that ties together data loading, processing,
and result saving.
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import dask.array as da

from .config import Config
from .data import ArrayInfo, get_tile_data, load_zarr_array, load_ome_metadata, OMEMetadata
from .io import CheckpointManager, save_mask_tiff, generate_tile_filename
from .io.bioset_writer import BiosetWriter
from .io.profiler import PipelineProfiler
from .processing import (
    TilingScheme,
    apply_threshold,
    create_tiling_scheme,
    is_gpu_available,
    filter_connected_components,
    compute_distance_transform,
    dilate_from_distance_transform,
)
from .processing.overlap import (
    compute_all_overlaps,
    PairwiseResult,
    EnrichmentResult,
    HigherOrderResult,
)
from .io.hierarchy import aggregate_hierarchy

logger = logging.getLogger(__name__)


class Pipeline:
    """
    Main pipeline for volumetric processing.
    
    Orchestrates the complete workflow:
    1. Load data from Zarr
    2. Divide into tiles
    3. Threshold each channel (per tile)
    4. Compute pairwise overlaps via matrix multiply
    5. Compute enrichment metrics
    6. Prune and compute higher-order combinations
    7. Save results to .bioset database
    
    Args:
        config: Configuration object
    """
    
    def __init__(self, config: Config):
        self.config = config
        self._array: Optional[da.Array] = None
        self._array_info: Optional[ArrayInfo] = None
        self._tiling_scheme: Optional[TilingScheme] = None
        self._metadata: Optional[OMEMetadata] = None
    
    # ---- Factory methods (unchanged) ----
    
    @classmethod
    def from_config(cls, path: Union[str, Path]) -> "Pipeline":
        path = Path(path)
        if path.suffix in (".yaml", ".yml"):
            config = Config.from_yaml(path)
        elif path.suffix == ".json":
            config = Config.from_json(path)
        else:
            raise ValueError(f"Unknown config format: {path.suffix}")
        return cls(config)
    
    @classmethod
    def from_url(cls, url: str, component: str = "0", **kwargs) -> "Pipeline":
        config = Config(zarr_url=url, zarr_component=component, **kwargs)
        return cls(config)
    
    # ---- Data loading (unchanged) ----
    
    def load_data(self) -> "Pipeline":
        logger.info("Loading data...")
        self._array, self._array_info = load_zarr_array(
            url=self.config.zarr_url,
            component=self.config.zarr_component,
        )
        logger.info(f"Array info: {self._array_info}")
        return self
    
    def setup_tiling(self) -> "Pipeline":
        if self._array_info is None:
            self.load_data()
        self._tiling_scheme = create_tiling_scheme(
            self._array_info,
            tile_size=self.config.tile_size,
            available_memory_gb=self.config.available_memory_gb,
            safety_factor=self.config.memory_safety_factor,
        )
        logger.info(f"Tiling scheme: {self._tiling_scheme}")
        return self
    
    @property
    def array(self) -> da.Array:
        if self._array is None:
            self.load_data()
        return self._array
    
    @property
    def array_info(self) -> ArrayInfo:
        if self._array_info is None:
            self.load_data()
        return self._array_info
    
    @property
    def tiling_scheme(self) -> TilingScheme:
        if self._tiling_scheme is None:
            self.setup_tiling()
        return self._tiling_scheme
    
    @property
    def metadata(self) -> Optional[OMEMetadata]:
        if self._metadata is None:
            self._metadata = load_ome_metadata(
                metadata_url=self.config.metadata_url,
                zarr_url=self.config.zarr_url,
            )
        return self._metadata
    
    # ================================================================
    # run() — main entry point
    # ================================================================
    
    def run(
        self,
        channels: Optional[List[int]] = None,
        resume: Optional[bool] = None,
    ) -> Dict:
        """Run the complete pipeline."""
        
        # Resolve parameters
        channels = channels or self.config.channels
        resume = resume if resume is not None else self.config.resume
        if channels is None:
            channels = list(range(self.array_info.n_channels))
        
        # ---- Setup profiler ----
        profiler = PipelineProfiler()
        profiler.start_pipeline()
        profiler.total_channels = len(channels)
        
        # ---- Resolve channel names ----
        channel_names = {}
        for ch in channels:
            if self.metadata:
                channel_names[ch] = self.metadata.get_channel_name(ch)
            else:
                channel_names[ch] = f"ch{ch}"
        
        # ---- Physical dimensions ----
        if self.metadata and self.metadata.physical_dimensions:
            phys = self.metadata.physical_dimensions
            voxel_volume_um3 = phys.voxel_volume_um3
            spacing_um = (phys.z_um, phys.y_um, phys.x_um)
            phys_dict = {"x_um": phys.x_um, "y_um": phys.y_um, "z_um": phys.z_um}
        else:
            voxel_volume_um3 = 1.0
            spacing_um = (1.0, 1.0, 1.0)
            phys_dict = None
            if self.config.cc_filter_enabled or self.config.dilation_radii_um:
                logger.warning("No metadata available, using default 1μm voxel size")
        
        # ---- Dilation radii ----
        dilation_radii = self.config.dilation_radii_um or [0]
        if 0 not in dilation_radii:
            dilation_radii = [0] + list(dilation_radii)
        
        # ---- Setup output ----
        output_dir = Path(self.config.output_dir)
        checkpoint = CheckpointManager(output_dir)
        _ = self.tiling_scheme  # ensure data + tiling ready
        
        # ---- Open bioset writer ----
        bioset_path = output_dir / "analysis.bioset"
        writer = BiosetWriter(bioset_path)
        writer.open()
        
        # Write metadata
        hierarchy_level_info = [
            {"level": 0, "tile_size": self.tiling_scheme.tile_size_y,
             "aggregation": "none"},
        ]
        for lvl in range(1, self.config.hierarchy_levels + 1):
            hierarchy_level_info.append({
                "level": lvl,
                "tile_size": self.tiling_scheme.tile_size_y * (2 ** lvl),
                "aggregation": "2x2_pooled_counts",
            })
        
        writer.write_metadata(
            channels=[channel_names[ch] for ch in channels],
            channel_indices=channels,
            hierarchy_levels=hierarchy_level_info,
            dilation_amounts=[int(r) for r in dilation_radii],
            volume_bounds={
                "x": [0, self.array_info.n_x],
                "y": [0, self.array_info.n_y],
                "z": [0, self.array_info.n_z],
            },
            physical_dimensions=phys_dict,
            threshold_method=self.config.threshold_method,
        )
        
        # ---- Log summary ----
        n_tiles = self.tiling_scheme.total_tiles
        logger.info("=" * 60)
        logger.info("BIOSET PREPROCESSING PIPELINE")
        logger.info("=" * 60)
        logger.info(f"GPU: {'ENABLED' if is_gpu_available() else 'DISABLED'}")
        logger.info(f"Channels: {len(channels)} ({channels[:5]}{'...' if len(channels)>5 else ''})")
        logger.info(f"Output: {bioset_path}")
        logger.info(f"CC Filtering: {'ENABLED' if self.config.cc_filter_enabled else 'DISABLED'}")
        logger.info(f"Dilation radii: {dilation_radii} μm")
        logger.info(f"Max combo size: {self.config.max_num_channels_in_comb}")
        logger.info(f"Enrichment threshold: {self.config.enrichment_threshold}")
        logger.info(f"Tiles: {n_tiles}")
        
        # ---- Process tiles ----
        completed = 0
        skipped = 0
        
        for tile in self.tiling_scheme.iter_tiles():
            if resume and checkpoint.is_completed(tile.tile_y, tile.tile_x):
                profiler.record_skip()
                skipped += 1
                continue
            
            logger.info(f"Processing tile ({tile.tile_y}, {tile.tile_x})...")
            
            profiler.start_tile(tile.tile_y, tile.tile_x)
            
            self._process_tile(
                tile=tile,
                channels=channels,
                channel_names=channel_names,
                writer=writer,
                profiler=profiler,
                spacing_um=spacing_um,
                voxel_volume_um3=voxel_volume_um3,
                dilation_radii=dilation_radii,
            )
            
            profiler.end_tile()
            checkpoint.mark_completed(tile.tile_y, tile.tile_x)
            completed += 1
        
        # ---- Hierarchical aggregation ----
        if self.config.hierarchy_levels > 0:
            # Ensure all pending writes are flushed before SQL aggregation
            writer._flush_channel_stats()
            writer._flush_pairwise()
            writer._flush_tiles()
            writer._conn.commit()
            
            logger.info(
                f"Building {self.config.hierarchy_levels} hierarchy levels..."
            )
            with profiler.stage("hierarchy"):
                hierarchy_summary = aggregate_hierarchy(
                    conn=writer._conn,
                    base_tile_size_y=self.tiling_scheme.tile_size_y,
                    base_tile_size_x=self.tiling_scheme.tile_size_x,
                    num_levels=self.config.hierarchy_levels,
                )
        
        # ---- Finalize ----
        profiler.end_pipeline()
        report = profiler.report()
        
        # Store profiling in metadata
        writer.write_metadata(
            channels=[channel_names[ch] for ch in channels],
            channel_indices=channels,
            hierarchy_levels=hierarchy_level_info,
            dilation_amounts=[int(r) for r in dilation_radii],
            volume_bounds={
                "x": [0, self.array_info.n_x],
                "y": [0, self.array_info.n_y],
                "z": [0, self.array_info.n_z],
            },
            extra={"profiling": profiler.to_dict()},
        )
        
        final_path = writer.close()
        
        logger.info("=" * 60)
        logger.info("COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Processed: {completed} tiles")
        logger.info(f"Skipped: {skipped} tiles (from checkpoint)")
        logger.info(f"Output: {final_path}")
        
        return {
            "completed": completed,
            "skipped": skipped,
            "total_tiles": n_tiles,
            "output_path": str(final_path),
            "profiling": profiler.to_dict(),
        }
    
    # ================================================================
    # _process_tile() — core per-tile processing
    # ================================================================
    
    def _process_tile(
        self,
        tile,
        channels: List[int],
        channel_names: Dict[int, str],
        writer: BiosetWriter,
        profiler: PipelineProfiler,
        spacing_um: tuple,
        voxel_volume_um3: float,
        dilation_radii: List[float],
    ) -> None:
        """
        Process a single tile:
        1. Load + threshold + CC filter each channel
        2. Write per-channel stats
        3. Compute distance transforms
        4. For each dilation: overlap analysis + enrichment + write
        """
        
        masks = {}
        
        # Total voxels in this tile (computed from first channel load)
        tile_total_voxels = None
        
        # ----------------------------------------------------------
        # Stage 1: Load, threshold, CC filter each channel
        # ----------------------------------------------------------
        for channel in channels:
            
            # ---- Load ----
            with profiler.stage("load"):
                tile_data = get_tile_data(
                    self.array, self.array_info,
                    channel=channel,
                    y_slice=tile.y_slice,
                    x_slice=tile.x_slice,
                )
            
            if tile_total_voxels is None:
                tile_total_voxels = tile_data.size
            
            # ---- Intensity stats (before threshold) ----
            with profiler.stage("intensity_stats"):
                mean_intensity = float(np.mean(tile_data))
                max_intensity = float(np.max(tile_data))
            
            # ---- Threshold ----
            with profiler.stage("threshold"):
                result = apply_threshold(tile_data, self.config.threshold_method)
            
            mask = result.mask
            
            # ---- CC Filtering ----
            if self.config.cc_filter_enabled:
                with profiler.stage("cc_filter"):
                    cc_result = filter_connected_components(
                        mask=mask,
                        min_volume_um3=self.config.cc_min_volume_um3,
                        voxel_volume_um3=voxel_volume_um3,
                    )
                    mask = cc_result.mask
                    logger.debug(
                        f"  Ch {channel}: CC {cc_result.original_components}"
                        f" -> {cc_result.remaining_components} components"
                    )
            
            masks[channel] = mask
            
            # ---- Write channel stats to bioset ----
            with profiler.stage("write"):
                writer.write_channel_stat(
                    tile_x0=tile.x_start, tile_x1=tile.x_end,
                    tile_y0=tile.y_start, tile_y1=tile.y_end,
                    hierarchy_level=0,
                    channel=channel_names[channel],
                    channel_idx=channel,
                    threshold_value=result.threshold_value,
                    active_voxels=result.active_voxels,
                    active_fraction=result.active_fraction,
                    total_voxels=tile_total_voxels,
                    mean_intensity=mean_intensity,
                    max_intensity=max_intensity,
                )
            
            logger.debug(
                f"  Ch {channel} ({channel_names[channel]}): "
                f"thresh={result.threshold_value:.2f}, "
                f"active={result.active_fraction:.2%}, "
                f"mean_int={mean_intensity:.1f}"
            )
            
            # ---- Optional: save mask TIFF ----
            if self.config.save_masks:
                mask_path = self.config.output_dir / generate_tile_filename(
                    channel=channel,
                    tile_x=tile.tile_x, tile_x_start=tile.x_start, tile_x_end=tile.x_end,
                    tile_y=tile.tile_y, tile_y_start=tile.y_start, tile_y_end=tile.y_end,
                    suffix="mask", method=self.config.threshold_method,
                )
                save_mask_tiff(mask, mask_path)
            
            del tile_data
        
        # ----------------------------------------------------------
        # Stage 2: Distance transforms (once, reused for all radii)
        #          Parallelized — scipy releases the GIL
        # ----------------------------------------------------------
        distance_transforms = {}
        if any(r > 0 for r in dilation_radii):
            with profiler.stage("distance_transform"):
                # Only compute for non-empty masks
                channels_to_compute = [
                    ch for ch, mask in masks.items() if np.any(mask)
                ]
                
                if channels_to_compute:
                    n_workers = min(len(channels_to_compute), os.cpu_count() or 4)
                    
                    def _dt_worker(ch):
                        return ch, compute_distance_transform(masks[ch], spacing_um)
                    
                    from concurrent.futures import ThreadPoolExecutor
                    with ThreadPoolExecutor(max_workers=n_workers) as pool:
                        for ch, dt in pool.map(_dt_worker, channels_to_compute):
                            distance_transforms[ch] = dt
                
                    logger.debug(
                        f"  Distance transforms: {len(distance_transforms)} channels, "
                        f"{n_workers} threads"
                    )
        
        # ----------------------------------------------------------
        # Stage 3: Per-dilation overlap analysis
        # ----------------------------------------------------------
        for radius_um in dilation_radii:
            
            # ---- Build dilated masks ----
            with profiler.stage("dilation"):
                if radius_um == 0:
                    analysis_masks = masks
                else:
                    analysis_masks = {}
                    for channel, mask in masks.items():
                        if channel in distance_transforms:
                            analysis_masks[channel] = dilate_from_distance_transform(
                                distance_transforms[channel], mask, radius_um
                            )
                        else:
                            analysis_masks[channel] = mask
            
            # ---- Core overlap + enrichment analysis ----
            with profiler.stage("overlap"):
                pairwise_result, enrichment_result, higher_order_result = compute_all_overlaps(
                    masks=analysis_masks,
                    max_higher_order_size=self.config.max_num_channels_in_comb,
                    enrichment_threshold=self.config.enrichment_threshold,
                    compute_higher_order=(self.config.max_num_channels_in_comb >= 3),
                )
            
            dilation_int = int(radius_um)
            
            # ---- Write results to bioset ----
            with profiler.stage("write"):
                writer.begin_transaction()
                
                self._write_tile_results(
                    writer=writer,
                    tile=tile,
                    pairwise_result=pairwise_result,
                    enrichment_result=enrichment_result,
                    higher_order_result=higher_order_result,
                    channel_names=channel_names,
                    dilation=dilation_int,
                    hierarchy_level=0,
                )
                
                writer.commit_transaction()
            
            # ---- Log enrichment stats ----
            suffix = f" (dilation={radius_um}μm)" if radius_um > 0 else ""
            n_ch = len(pairwise_result.channels)
            n_pairs = n_ch * (n_ch - 1) // 2
            # Count enriched pairs from enrichment matrix
            emat = enrichment_result.enrichment_matrix
            n_enriched = int(np.sum(
                emat[np.triu_indices(n_ch, k=1)] >= self.config.enrichment_threshold
            ))
            logger.info(
                f"  Overlaps{suffix}: "
                f"{n_ch} active ch, "
                f"{n_enriched}/{n_pairs} enriched pairs, "
                f"{higher_order_result.n_candidates} higher-order candidates, "
                f"{len(higher_order_result.overlaps)} with overlap"
            )
            
            # Cleanup dilated masks
            if radius_um > 0:
                del analysis_masks
        
        # Cleanup
        del distance_transforms
        del masks
    
    # ================================================================
    # _write_tile_results() — write one tile+dilation to bioset
    # ================================================================
    
    def _write_tile_results(
        self,
        writer: BiosetWriter,
        tile,
        pairwise_result: PairwiseResult,
        enrichment_result: EnrichmentResult,
        higher_order_result: HigherOrderResult,
        channel_names: Dict[int, str],
        dilation: int,
        hierarchy_level: int,
    ) -> None:
        """Write all overlap results for one tile at one dilation level."""
        
        tx0, tx1 = tile.x_start, tile.x_end
        ty0, ty1 = tile.y_start, tile.y_end
        total = pairwise_result.total_voxels
        channels = pairwise_result.channels
        n_ch = len(channels)
        
        # ---- Pairwise metrics (asymmetric, per-pair) ----
        for i in range(n_ch):
            for j in range(i + 1, n_ch):
                ch_a, ch_b = channels[i], channels[j]
                count_a = pairwise_result.channel_counts.get(ch_a, 0)
                count_b = pairwise_result.channel_counts.get(ch_b, 0)
                count_ab = int(pairwise_result.overlap_matrix[i, j])
                enrichment = float(enrichment_result.enrichment_matrix[i, j])
                name_a = channel_names.get(ch_a, f"ch{ch_a}")
                name_b = channel_names.get(ch_b, f"ch{ch_b}")
                
                # Write detailed asymmetric metrics
                writer.write_pairwise_metric(
                    tile_x0=tx0, tile_x1=tx1,
                    tile_y0=ty0, tile_y1=ty1,
                    hierarchy_level=hierarchy_level,
                    dilation=dilation,
                    channel_a=name_a, channel_b=name_b,
                    channel_a_idx=ch_a, channel_b_idx=ch_b,
                    count_a=count_a, count_b=count_b,
                    count_ab=count_ab, total_voxels=total,
                )
                
                # Write pair as a combination + tile (for UpSet / heatmap)
                channels_str = f"{name_a}|{name_b}"
                indices_str = f"{ch_a}|{ch_b}"
                
                writer.write_combination_with_tile(
                    channels=channels_str,
                    channel_indices=indices_str,
                    channel_count=2,
                    dilation=dilation,
                    hierarchy_level=hierarchy_level,
                    overlap_count=count_ab,
                    total_voxels=total,
                    tile_x0=tx0, tile_x1=tx1,
                    tile_y0=ty0, tile_y1=ty1,
                    tile_count=count_ab,
                    enrichment_ratio=enrichment,
                    tile_enrichment=enrichment,
                )
        
        # ---- Higher-order combinations ----
        for combo, count in higher_order_result.overlaps.items():
            names = [channel_names.get(ch, f"ch{ch}") for ch in combo]
            channels_str = "|".join(names)
            indices_str = "|".join(str(ch) for ch in combo)
            
            e_indep = higher_order_result.enrichment_indep.get(combo)
            e_higher = higher_order_result.enrichment_higher.get(combo)
            
            writer.write_combination_with_tile(
                channels=channels_str,
                channel_indices=indices_str,
                channel_count=len(combo),
                dilation=dilation,
                hierarchy_level=hierarchy_level,
                overlap_count=count,
                total_voxels=total,
                tile_x0=tx0, tile_x1=tx1,
                tile_y0=ty0, tile_y1=ty1,
                tile_count=count,
                enrichment_ratio=e_indep,
                higher_order_enrichment=e_higher,
                tile_enrichment=e_indep,
            )


# ================================================================
# Convenience function (updated)
# ================================================================

def run_pipeline(
    config: Optional[Config] = None,
    config_path: Optional[Union[str, Path]] = None,
    **kwargs,
) -> Dict:
    """
    Convenience function to run the pipeline.
    
    Args:
        config: Config object (optional)
        config_path: Path to config file (optional)
        **kwargs: Override config parameters
    
    Returns:
        Dict with results summary
    """
    if config is not None:
        pipeline = Pipeline(config)
    elif config_path is not None:
        pipeline = Pipeline.from_config(config_path)
    elif "zarr_url" in kwargs:
        config = Config(**kwargs)
        pipeline = Pipeline(config)
    else:
        raise ValueError("Must provide either config, config_path, or zarr_url")
    
    return pipeline.run()