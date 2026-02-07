"""
Pipeline profiling and timing utilities.

Tracks per-stage and per-tile timing with summary statistics.
Designed to be lightweight — just wraps time.perf_counter() with
structured accumulation and reporting.

Usage in pipeline:
    profiler = PipelineProfiler()
    
    for tile in tiles:
        profiler.start_tile(tile_y, tile_x)
        
        with profiler.stage("load"):
            data = load_tile(...)
        
        with profiler.stage("threshold"):
            mask = threshold(data)
        
        profiler.end_tile()
    
    profiler.report()
    profiler.to_dict()  # for JSON serialization
"""

import logging
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TileTimings:
    """Timing record for a single tile."""
    tile_y: int
    tile_x: int
    stages: Dict[str, float] = field(default_factory=dict)
    total: float = 0.0


class PipelineProfiler:
    """
    Accumulates per-stage timing across tiles and produces summary statistics.
    
    Thread-safe for single-pipeline use (not designed for concurrent pipelines).
    """
    
    def __init__(self):
        self._tile_timings: List[TileTimings] = []
        self._current_tile: Optional[TileTimings] = None
        self._stage_start: Optional[float] = None
        self._pipeline_start: Optional[float] = None
        self._pipeline_end: Optional[float] = None
        
        # Accumulated stage totals (for fast summary without iterating tiles)
        self._stage_totals: Dict[str, float] = defaultdict(float)
        self._stage_counts: Dict[str, int] = defaultdict(int)
        
        # Global counters
        self.tiles_processed: int = 0
        self.tiles_skipped: int = 0
        self.total_channels: int = 0
        self.total_combinations: int = 0
    
    def start_pipeline(self) -> None:
        """Mark pipeline start time."""
        self._pipeline_start = time.perf_counter()
    
    def end_pipeline(self) -> None:
        """Mark pipeline end time."""
        self._pipeline_end = time.perf_counter()
    
    def start_tile(self, tile_y: int, tile_x: int) -> None:
        """Begin timing a new tile."""
        self._current_tile = TileTimings(tile_y=tile_y, tile_x=tile_x)
        self._current_tile.total = time.perf_counter()
    
    def end_tile(self) -> None:
        """Finish timing current tile, log, and accumulate."""
        if self._current_tile is None:
            return
        
        self._current_tile.total = time.perf_counter() - self._current_tile.total
        self._tile_timings.append(self._current_tile)
        self.tiles_processed += 1
        
        # Log per-tile breakdown
        parts = [f"{k}={v:.2f}s" for k, v in self._current_tile.stages.items()]
        parts.append(f"total={self._current_tile.total:.2f}s")
        logger.info(
            f"  Tile ({self._current_tile.tile_y},{self._current_tile.tile_x}) "
            f"timing: {', '.join(parts)}"
        )
        
        self._current_tile = None
    
    @contextmanager
    def stage(self, name: str):
        """
        Context manager to time a named stage within the current tile.
        
        Usage:
            with profiler.stage("threshold"):
                result = apply_threshold(data)
        """
        t0 = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - t0
            if self._current_tile is not None:
                # Accumulate into current tile (stages can be called multiple times)
                self._current_tile.stages[name] = (
                    self._current_tile.stages.get(name, 0.0) + elapsed
                )
            self._stage_totals[name] += elapsed
            self._stage_counts[name] += 1
    
    def record_skip(self) -> None:
        """Record a skipped tile (from checkpoint)."""
        self.tiles_skipped += 1
    
    # -----------------------------------------------------------------
    # Summary & Reporting
    # -----------------------------------------------------------------
    
    @property
    def wall_time(self) -> float:
        """Total wall-clock time for the pipeline."""
        if self._pipeline_start is None:
            return 0.0
        end = self._pipeline_end or time.perf_counter()
        return end - self._pipeline_start
    
    def get_stage_summary(self) -> Dict[str, Dict[str, float]]:
        """
        Per-stage summary statistics across all tiles.
        
        Returns:
            {stage_name: {total, mean, std, min, max, pct_of_total}}
        """
        if not self._tile_timings:
            return {}
        
        total_processing_time = sum(t.total for t in self._tile_timings)
        stage_names = sorted(self._stage_totals.keys())
        
        summary = {}
        for name in stage_names:
            # Collect per-tile times for this stage
            times = [t.stages.get(name, 0.0) for t in self._tile_timings]
            times_arr = np.array(times) if times else np.array([0.0])
            
            summary[name] = {
                "total_s": self._stage_totals[name],
                "mean_s": float(np.mean(times_arr)),
                "std_s": float(np.std(times_arr)),
                "min_s": float(np.min(times_arr)),
                "max_s": float(np.max(times_arr)),
                "pct_of_total": (
                    100.0 * self._stage_totals[name] / total_processing_time
                    if total_processing_time > 0 else 0.0
                ),
            }
        
        return summary
    
    def report(self) -> str:
        """
        Generate and log a formatted profiling report.
        
        Returns:
            The report string.
        """
        lines = []
        lines.append("=" * 70)
        lines.append("PIPELINE PROFILING REPORT")
        lines.append("=" * 70)
        lines.append(f"Wall time:        {self.wall_time:.1f}s ({self.wall_time/60:.1f}m)")
        lines.append(f"Tiles processed:  {self.tiles_processed}")
        lines.append(f"Tiles skipped:    {self.tiles_skipped}")
        
        if self.tiles_processed > 0:
            tile_times = [t.total for t in self._tile_timings]
            mean_tile = np.mean(tile_times)
            lines.append(f"Avg tile time:    {mean_tile:.2f}s")
            lines.append(f"Min/Max tile:     {min(tile_times):.2f}s / {max(tile_times):.2f}s")
        
        summary = self.get_stage_summary()
        if summary:
            lines.append("")
            lines.append(f"{'Stage':<20} {'Total':>10} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8} {'%':>6}")
            lines.append("-" * 70)
            for name, stats in summary.items():
                lines.append(
                    f"{name:<20} "
                    f"{stats['total_s']:>9.1f}s "
                    f"{stats['mean_s']:>7.2f}s "
                    f"{stats['std_s']:>7.2f}s "
                    f"{stats['min_s']:>7.2f}s "
                    f"{stats['max_s']:>7.2f}s "
                    f"{stats['pct_of_total']:>5.1f}%"
                )
        
        lines.append("=" * 70)
        
        report_text = "\n".join(lines)
        logger.info("\n" + report_text)
        return report_text
    
    def to_dict(self) -> dict:
        """
        Serialize profiling data for JSON/metadata storage.
        
        Returns:
            Dict suitable for json.dumps()
        """
        return {
            "wall_time_s": round(self.wall_time, 2),
            "tiles_processed": self.tiles_processed,
            "tiles_skipped": self.tiles_skipped,
            "total_channels": self.total_channels,
            "total_combinations": self.total_combinations,
            "stage_summary": {
                name: {k: round(v, 4) for k, v in stats.items()}
                for name, stats in self.get_stage_summary().items()
            },
            "per_tile": [
                {
                    "tile": (t.tile_y, t.tile_x),
                    "total_s": round(t.total, 3),
                    "stages": {k: round(v, 3) for k, v in t.stages.items()},
                }
                for t in self._tile_timings
            ],
        }