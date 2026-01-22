"""
Result saving utilities for CSV output.

Handles incremental saving of threshold and overlap results.
"""

import csv
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

from ..processing.overlap import OverlapResult, get_channel_combinations

logger = logging.getLogger(__name__)


class ResultSaver:
    """
    Handles incremental saving of results to CSV files.
    
    Creates two CSV files:
    - thresholds.csv: Per-channel threshold values for each tile
    - overlaps.csv: Channel combination overlap counts for each tile
    
    Results are appended after each tile, so partial results are preserved
    if processing is interrupted.
    
    Args:
        output_dir: Directory for output files
        channels: List of channel indices being processed
        
    Example:
        >>> saver = ResultSaver(Path("./results"), channels=[0, 1, 2])
        >>> saver.save_threshold(tile_y=0, tile_x=0, channel=0, ...)
        >>> saver.save_overlaps(tile_y=0, tile_x=0, ...)
    """
    
    def __init__(
        self,
        output_dir: Path,
        channels: List[int],
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.channels = channels
        self.channel_combinations = get_channel_combinations(channels)
        
        self.thresholds_path = self.output_dir / "thresholds.csv"
        self.overlaps_path = self.output_dir / "overlaps.csv"
        
        self._init_csv_files()
    
    def _init_csv_files(self) -> None:
        """Initialize CSV files with headers if they don't exist."""
        # Thresholds CSV
        if not self.thresholds_path.exists():
            with open(self.thresholds_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "tile_y", "tile_x", "channel", "threshold",
                    "active_voxels", "active_fraction",
                    "y_start", "y_end", "x_start", "x_end",
                    "timestamp"
                ])
            logger.info(f"Created {self.thresholds_path}")
        
        # Overlaps CSV
        if not self.overlaps_path.exists():
            with open(self.overlaps_path, "w", newline="") as f:
                writer = csv.writer(f)
                # Create column names for each combination
                combo_cols = [
                    "_".join(map(str, combo)) 
                    for combo in self.channel_combinations
                ]
                writer.writerow([
                    "tile_y", "tile_x",
                    "y_start", "y_end", "x_start", "x_end",
                    "timestamp"
                ] + combo_cols)
            logger.info(f"Created {self.overlaps_path}")
    
    def save_threshold(
        self,
        tile_y: int,
        tile_x: int,
        channel: int,
        threshold: float,
        active_voxels: int,
        active_fraction: float,
        y_start: int,
        y_end: int,
        x_start: int,
        x_end: int,
    ) -> None:
        """
        Save a threshold result for a single channel/tile.
        
        Args:
            tile_y: Tile Y index
            tile_x: Tile X index
            channel: Channel index
            threshold: Threshold value used
            active_voxels: Number of voxels above threshold
            active_fraction: Fraction of voxels above threshold
            y_start, y_end: Y slice bounds
            x_start, x_end: X slice bounds
        """
        with open(self.thresholds_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                tile_y, tile_x, channel, threshold,
                active_voxels, f"{active_fraction:.6f}",
                y_start, y_end, x_start, x_end,
                datetime.now().isoformat()
            ])
    
    def save_overlaps(
        self,
        tile_y: int,
        tile_x: int,
        y_start: int,
        y_end: int,
        x_start: int,
        x_end: int,
        overlap_result: OverlapResult,
    ) -> None:
        """
        Save overlap results for a tile.
        
        Args:
            tile_y: Tile Y index
            tile_x: Tile X index
            y_start, y_end: Y slice bounds
            x_start, x_end: X slice bounds
            overlap_result: OverlapResult from compute_overlaps()
        """
        with open(self.overlaps_path, "a", newline="") as f:
            writer = csv.writer(f)
            # Get counts in same order as header
            counts = [
                overlap_result.overlaps.get(combo, 0)
                for combo in self.channel_combinations
            ]
            writer.writerow([
                tile_y, tile_x,
                y_start, y_end, x_start, x_end,
                datetime.now().isoformat()
            ] + counts)
    
    def get_results_summary(self) -> Dict:
        """
        Get summary of saved results.
        
        Returns:
            Dict with counts of saved thresholds and overlaps
        """
        threshold_count = 0
        overlap_count = 0
        
        if self.thresholds_path.exists():
            with open(self.thresholds_path) as f:
                threshold_count = sum(1 for _ in f) - 1  # Exclude header
        
        if self.overlaps_path.exists():
            with open(self.overlaps_path) as f:
                overlap_count = sum(1 for _ in f) - 1  # Exclude header
        
        return {
            "thresholds_saved": threshold_count,
            "overlaps_saved": overlap_count,
            "thresholds_path": str(self.thresholds_path),
            "overlaps_path": str(self.overlaps_path),
        }
