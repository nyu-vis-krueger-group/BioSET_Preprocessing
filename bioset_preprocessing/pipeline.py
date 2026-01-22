"""
Main pipeline orchestration.

Provides the Pipeline class that ties together data loading, processing,
and result saving.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import dask.array as da

from .config import Config
from .data import ArrayInfo, get_tile_data, load_zarr_array
from .io import CheckpointManager, ResultSaver, save_mask_tiff, generate_tile_filename
from .processing import (
    TilingScheme,
    apply_threshold,
    compute_overlaps,
    create_tiling_scheme,
    get_channel_combinations,
    is_gpu_available,
)

logger = logging.getLogger(__name__)


class Pipeline:
    """
    Main pipeline for volumetric processing.
    
    Orchestrates the complete workflow:
    1. Load data from Zarr
    2. Divide into tiles
    3. Threshold each channel
    4. Compute channel overlaps
    5. Save results
    
    Args:
        config: Configuration object
        
    Example:
        >>> # From config file
        >>> pipeline = Pipeline.from_config("config.yaml")
        >>> results = pipeline.run()
        
        >>> # From URL with defaults
        >>> pipeline = Pipeline.from_url("https://...")
        >>> results = pipeline.run(channels=[0, 1, 2])
        
        >>> # Programmatic
        >>> config = Config(zarr_url="https://...", channels=[0, 1, 2])
        >>> pipeline = Pipeline(config)
        >>> results = pipeline.run()
    """
    
    def __init__(self, config: Config):
        self.config = config
        self._array: Optional[da.Array] = None
        self._array_info: Optional[ArrayInfo] = None
        self._tiling_scheme: Optional[TilingScheme] = None
    
    @classmethod
    def from_config(cls, path: Union[str, Path]) -> "Pipeline":
        """
        Create pipeline from a YAML or JSON config file.
        
        Args:
            path: Path to config file
            
        Returns:
            Pipeline instance
        """
        path = Path(path)
        
        if path.suffix in (".yaml", ".yml"):
            config = Config.from_yaml(path)
        elif path.suffix == ".json":
            config = Config.from_json(path)
        else:
            raise ValueError(f"Unknown config format: {path.suffix}")
        
        return cls(config)
    
    @classmethod
    def from_url(
        cls,
        url: str,
        component: str = "0",
        **kwargs,
    ) -> "Pipeline":
        """
        Create pipeline from a Zarr URL with default settings.
        
        Args:
            url: Zarr URL or path
            component: Resolution level
            **kwargs: Additional Config parameters
            
        Returns:
            Pipeline instance
        """
        config = Config(
            zarr_url=url,
            zarr_component=component,
            **kwargs,
        )
        return cls(config)
    
    def load_data(self) -> "Pipeline":
        """
        Load the Zarr data.
        
        Returns:
            self (for method chaining)
        """
        logger.info("Loading data...")
        
        self._array, self._array_info = load_zarr_array(
            url=self.config.zarr_url,
            component=self.config.zarr_component,
        )
        
        logger.info(f"Array info: {self._array_info}")
        
        return self
    
    def setup_tiling(self) -> "Pipeline":
        """
        Set up the tiling scheme.
        
        Returns:
            self (for method chaining)
        """
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
        """Get the loaded array, loading if necessary."""
        if self._array is None:
            self.load_data()
        return self._array
    
    @property
    def array_info(self) -> ArrayInfo:
        """Get array info, loading if necessary."""
        if self._array_info is None:
            self.load_data()
        return self._array_info
    
    @property
    def tiling_scheme(self) -> TilingScheme:
        """Get tiling scheme, setting up if necessary."""
        if self._tiling_scheme is None:
            self.setup_tiling()
        return self._tiling_scheme
    
    def run(
        self,
        channels: Optional[List[int]] = None,
        resume: Optional[bool] = None,
    ) -> Dict:
        """
        Run the complete pipeline.
        
        Args:
            channels: Override config channels
            resume: Override config resume setting
            
        Returns:
            Dict with summary statistics
        """
        # Resolve parameters
        channels = channels or self.config.channels
        resume = resume if resume is not None else self.config.resume
        
        # If channels still None, use all channels
        if channels is None:
            channels = list(range(self.array_info.n_channels))
        
        logger.info("=" * 60)
        logger.info("VOLUMETRIC PROCESSING PIPELINE")
        logger.info("=" * 60)
        logger.info(f"GPU: {'ENABLED' if is_gpu_available() else 'DISABLED'}")
        logger.info(f"Channels: {channels}")
        logger.info(f"Output: {self.config.output_dir}")
        
        # Setup
        output_dir = Path(self.config.output_dir)
        saver = ResultSaver(output_dir, channels)
        checkpoint = CheckpointManager(output_dir)
        
        # Ensure data and tiling are ready
        _ = self.tiling_scheme
        
        n_tiles = self.tiling_scheme.total_tiles
        channel_combos = get_channel_combinations(channels)
        
        logger.info(f"Tiles: {n_tiles}")
        logger.info(f"Channel combinations: {len(channel_combos)}")
        
        # Process tiles
        completed = 0
        skipped = 0
        
        for tile in self.tiling_scheme.iter_tiles():
            # Check checkpoint
            if resume and checkpoint.is_completed(tile.tile_y, tile.tile_x):
                skipped += 1
                continue
            
            logger.info(f"Processing tile ({tile.tile_y}, {tile.tile_x})...")
            
            # Process this tile
            self._process_tile(
                tile=tile,
                channels=channels,
                saver=saver,
            )
            
            # Mark completed
            checkpoint.mark_completed(tile.tile_y, tile.tile_x)
            completed += 1
        
        # Summary
        logger.info("=" * 60)
        logger.info("COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Processed: {completed} tiles")
        logger.info(f"Skipped: {skipped} tiles (from checkpoint)")
        logger.info(f"Results: {output_dir.absolute()}")
        
        return {
            "completed": completed,
            "skipped": skipped,
            "total_tiles": n_tiles,
            "output_dir": str(output_dir.absolute()),
            "summary": saver.get_results_summary(),
        }
    
    def _process_tile(
        self,
        tile,
        channels: List[int],
        saver: ResultSaver,
    ) -> None:
        """Process a single tile: threshold all channels and compute overlaps."""
        masks = {}
        
        # Process each channel
        for channel in channels:
            # Load tile data
            tile_data = get_tile_data(
                self.array,
                self.array_info,
                channel=channel,
                y_slice=tile.y_slice,
                x_slice=tile.x_slice,
            )
            
            # Apply threshold
            result = apply_threshold(tile_data, self.config.threshold_method)
            masks[channel] = result.mask
            
            # Save threshold result
            saver.save_threshold(
                tile_y=tile.tile_y,
                tile_x=tile.tile_x,
                channel=channel,
                threshold=result.threshold_value,
                active_voxels=result.active_voxels,
                active_fraction=result.active_fraction,
                y_start=tile.y_start,
                y_end=tile.y_end,
                x_start=tile.x_start,
                x_end=tile.x_end,
            )
            
            logger.debug(
                f"  Ch {channel}: thresh={result.threshold_value:.2f}, "
                f"active={result.active_fraction:.2%}"
            )
            
            # Optionally save mask
            if self.config.save_masks:
                mask_path = self.config.output_dir / generate_tile_filename(
                    channel=channel,
                    tile_y=tile.tile_y,
                    tile_x=tile.tile_x,
                    suffix="mask",
                    method=self.config.threshold_method,
                )
                save_mask_tiff(result.mask, mask_path)
            
            del tile_data
        
        # Compute overlaps
        overlap_result = compute_overlaps(masks)
        
        # Save overlaps
        saver.save_overlaps(
            tile_y=tile.tile_y,
            tile_x=tile.tile_x,
            y_start=tile.y_start,
            y_end=tile.y_end,
            x_start=tile.x_start,
            x_end=tile.x_end,
            overlap_result=overlap_result,
        )
        
        # Log overlap summary
        for combo, count in overlap_result.overlaps.items():
            combo_str = "+".join(map(str, combo))
            logger.debug(f"  Overlap {combo_str}: {count:,}")
        
        del masks


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
        
    Example:
        >>> results = run_pipeline(config_path="config.yaml")
        >>> results = run_pipeline(
        ...     zarr_url="https://...",
        ...     channels=[0, 1, 2],
        ...     threshold_method="otsu"
        ... )
    """
    if config is not None:
        pipeline = Pipeline(config)
    elif config_path is not None:
        pipeline = Pipeline.from_config(config_path)
    elif "zarr_url" in kwargs:
        config = Config(**kwargs)
        pipeline = Pipeline(config)
    else:
        raise ValueError(
            "Must provide either config, config_path, or zarr_url"
        )
    
    return pipeline.run()
