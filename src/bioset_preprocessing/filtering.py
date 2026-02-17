from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Set, Optional
from pathlib import Path
from collections import defaultdict
import heapq

from .stages.overlaps import (
    OverlapTileResult,
    ChannelTileStats,
    PairRow,
    SetRow,
)
from .checkpoint import load_all_checkpoints, save_tile_checkpoint


@dataclass
class FilterConfig:
    """Configuration for filtering tile results."""
    
    # Minimum thresholds (applied per tile)
    min_iou: float = 0.0                    # Min IoU to keep a pair/set in a tile
    min_overlap_coeff: float = 0.0          # Min overlap coefficient
    min_inter_vox: int = 0                  # Min intersection voxels
    
    # Global filters (applied across all tiles)
    min_tiles_present: int = 1              # Combination must appear in at least N tiles
    min_tiles_percent: float = 0.0          # Combination must appear in at least X% of tiles
    
    # Top-K filtering (keep only best tiles per combination)
    top_k_tiles: Optional[int] = None       # Keep top K tiles per combination (by overlap_coeff)
    top_k_percent: Optional[float] = None   # Keep top K% tiles per combination
    
    # Combination limits
    max_pairs_per_tile: Optional[int] = None    # Limit pairs per tile
    max_sets_per_tile: Optional[int] = None     # Limit sets per tile
    
    # Set size filtering
    min_set_size: int = 2                   # Minimum k (2=pairs only, 3=include triplets, etc.)
    max_set_size: int = 10                  # Maximum k


@dataclass
class FilterStats:
    """Statistics from filtering operation."""
    tiles_processed: int = 0
    pairs_before: int = 0
    pairs_after: int = 0
    sets_before: int = 0
    sets_after: int = 0
    unique_pairs_before: int = 0
    unique_pairs_after: int = 0
    unique_sets_before: int = 0
    unique_sets_after: int = 0
    
    @property
    def pairs_removed_pct(self) -> float:
        if self.pairs_before == 0:
            return 0.0
        return 100 * (1 - self.pairs_after / self.pairs_before)
    
    @property
    def sets_removed_pct(self) -> float:
        if self.sets_before == 0:
            return 0.0
        return 100 * (1 - self.sets_after / self.sets_before)
    
    def __str__(self) -> str:
        return (
            f"FilterStats:\n"
            f"  Tiles processed: {self.tiles_processed}\n"
            f"  Pairs: {self.pairs_before:,} → {self.pairs_after:,} "
            f"({self.pairs_removed_pct:.1f}% removed)\n"
            f"  Sets: {self.sets_before:,} → {self.sets_after:,} "
            f"({self.sets_removed_pct:.1f}% removed)\n"
            f"  Unique pairs: {self.unique_pairs_before:,} → {self.unique_pairs_after:,}\n"
            f"  Unique sets: {self.unique_sets_before:,} → {self.unique_sets_after:,}"
        )


def _pair_key(p: PairRow) -> Tuple:
    """Create a hashable key for a pair."""
    return (p.a, p.b, p.r_um)


def _set_key(s: SetRow) -> Tuple:
    """Create a hashable key for a set."""
    return (s.members, s.r_um)


class TileFilter:
    """
    Filter tile results to reduce data size before aggregation.
    
    Example usage:
        filter_cfg = FilterConfig(
            min_overlap_coeff=0.1,      # Drop weak overlaps
            top_k_percent=0.25,         # Keep top 25% tiles per combination
            min_tiles_present=5,        # Combination must appear in 5+ tiles
        )
        
        filterer = TileFilter(filter_cfg)
        filtered_results = filterer.filter_checkpoints(checkpoint_dir)
        stats = filterer.get_stats()
    """
    
    def __init__(self, config: FilterConfig):
        self.config = config
        self._stats = FilterStats()
        
    def get_stats(self) -> FilterStats:
        return self._stats
    
    def filter_checkpoints(
        self,
        checkpoint_dir: Path,
        output_dir: Optional[Path] = None,
    ) -> List[OverlapTileResult]:
        """
        Load checkpoints, filter them, and optionally save filtered versions.
        
        Args:
            checkpoint_dir: Directory containing tile checkpoints
            output_dir: If provided, save filtered checkpoints here
            
        Returns:
            List of filtered OverlapTileResult objects
        """
        print(f"[Filter] Loading checkpoints from {checkpoint_dir}...")
        results = load_all_checkpoints(checkpoint_dir)
        print(f"[Filter] Loaded {len(results)} tiles")
        
        if not results:
            return []
        
        self._stats = FilterStats(tiles_processed=len(results))
        
        for r in results:
            self._stats.pairs_before += len(r.pairs)
            self._stats.sets_before += len(r.sets)
        
        print("[Filter] Applying per-tile filters...")
        filtered_results = [self._filter_single_tile(r) for r in results]
        
        print("[Filter] Computing global statistics...")
        pair_tile_counts, set_tile_counts = self._count_tiles_per_combination(filtered_results)
        pair_best_tiles, set_best_tiles = self._find_best_tiles_per_combination(filtered_results)
        
        print("[Filter] Applying global filters...")
        filtered_results = [
            self._apply_global_filters(
                r, 
                pair_tile_counts, 
                set_tile_counts,
                pair_best_tiles,
                set_best_tiles,
                len(results),
            )
            for r in filtered_results
        ]
        
        unique_pairs: Set[Tuple] = set()
        unique_sets: Set[Tuple] = set()
        for r in filtered_results:
            self._stats.pairs_after += len(r.pairs)
            self._stats.sets_after += len(r.sets)
            for p in r.pairs:
                unique_pairs.add(_pair_key(p))
            for s in r.sets:
                unique_sets.add(_set_key(s))
        
        unique_pairs_before: Set[Tuple] = set()
        unique_sets_before: Set[Tuple] = set()
        for r in results:
            for p in r.pairs:
                unique_pairs_before.add(_pair_key(p))
            for s in r.sets:
                unique_sets_before.add(_set_key(s))
        
        self._stats.unique_pairs_before = len(unique_pairs_before)
        self._stats.unique_pairs_after = len(unique_pairs)
        self._stats.unique_sets_before = len(unique_sets_before)
        self._stats.unique_sets_after = len(unique_sets)
        
        print(f"[Filter] {self._stats}")
        
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            print(f"[Filter] Saving filtered checkpoints to {output_dir}...")
            for r in filtered_results:
                save_tile_checkpoint(output_dir, r)
            print(f"[Filter] Saved {len(filtered_results)} filtered checkpoints")
        
        return filtered_results
    
    def _filter_single_tile(self, result: OverlapTileResult) -> OverlapTileResult:
        cfg = self.config
        
        filtered_pairs = []
        for p in result.pairs:
            if p.iou < cfg.min_iou:
                continue
            if p.overlap_coeff < cfg.min_overlap_coeff:
                continue
            if p.inter_vox < cfg.min_inter_vox:
                continue
            filtered_pairs.append(p)
        
        if cfg.max_pairs_per_tile and len(filtered_pairs) > cfg.max_pairs_per_tile:
            filtered_pairs.sort(key=lambda p: p.overlap_coeff, reverse=True)
            filtered_pairs = filtered_pairs[:cfg.max_pairs_per_tile]
        
        filtered_sets = []
        for s in result.sets:
            if s.k < cfg.min_set_size or s.k > cfg.max_set_size:
                continue
            if s.iou < cfg.min_iou:
                continue
            if s.overlap_coeff < cfg.min_overlap_coeff:
                continue
            if s.inter_vox < cfg.min_inter_vox:
                continue
            filtered_sets.append(s)
        
        if cfg.max_sets_per_tile and len(filtered_sets) > cfg.max_sets_per_tile:
            filtered_sets.sort(key=lambda s: s.overlap_coeff, reverse=True)
            filtered_sets = filtered_sets[:cfg.max_sets_per_tile]
        
        return OverlapTileResult(
            tile_x=result.tile_x,
            tile_y=result.tile_y,
            tile_z=result.tile_z,
            tile_shape=result.tile_shape,
            total_voxels=result.total_voxels,
            radii_um=result.radii_um,
            marker_vox=result.marker_vox,
            channel_stats=result.channel_stats,
            pairs=filtered_pairs,
            sets=filtered_sets,
            n_active_channels=result.n_active_channels,
            n_frequent_pairs=result.n_frequent_pairs,
        )
    
    def _count_tiles_per_combination(
        self,
        results: List[OverlapTileResult],
    ) -> Tuple[Dict[Tuple, int], Dict[Tuple, int]]:
        pair_counts: Dict[Tuple, int] = defaultdict(int)
        set_counts: Dict[Tuple, int] = defaultdict(int)
        
        for r in results:
            for p in r.pairs:
                pair_counts[_pair_key(p)] += 1
            for s in r.sets:
                set_counts[_set_key(s)] += 1
        
        return pair_counts, set_counts
    
    def _find_best_tiles_per_combination(
        self,
        results: List[OverlapTileResult],
    ) -> Tuple[Dict[Tuple, List[Tuple[float, int, int]]], Dict[Tuple, List[Tuple[float, int, int]]]]:
        pair_tiles: Dict[Tuple, List[Tuple[float, int, int]]] = defaultdict(list)
        set_tiles: Dict[Tuple, List[Tuple[float, int, int]]] = defaultdict(list)
        
        for r in results:
            for p in r.pairs:
                key = _pair_key(p)
                pair_tiles[key].append((p.overlap_coeff, r.tile_x, r.tile_y))
            for s in r.sets:
                key = _set_key(s)
                set_tiles[key].append((s.overlap_coeff, r.tile_x, r.tile_y))
        
        return pair_tiles, set_tiles
    
    def _apply_global_filters(
        self,
        result: OverlapTileResult,
        pair_tile_counts: Dict[Tuple, int],
        set_tile_counts: Dict[Tuple, int],
        pair_best_tiles: Dict[Tuple, List[Tuple[float, int, int]]],
        set_best_tiles: Dict[Tuple, List[Tuple[float, int, int]]],
        total_tiles: int,
    ) -> OverlapTileResult:
        cfg = self.config
        
        min_tiles = cfg.min_tiles_present
        if cfg.min_tiles_percent > 0:
            min_tiles = max(min_tiles, int(total_tiles * cfg.min_tiles_percent))
        
        filtered_pairs = []
        for p in result.pairs:
            key = _pair_key(p)
            
            if pair_tile_counts.get(key, 0) < min_tiles:
                continue
            
            if cfg.top_k_tiles is not None or cfg.top_k_percent is not None:
                tiles_for_combo = pair_best_tiles.get(key, [])
                n_tiles = len(tiles_for_combo)
                
                if cfg.top_k_tiles is not None:
                    k = cfg.top_k_tiles
                elif cfg.top_k_percent is not None:
                    k = max(1, int(n_tiles * cfg.top_k_percent))
                else:
                    k = n_tiles
                
                # Get top K tiles for this combination
                top_k_tiles = heapq.nlargest(k, tiles_for_combo, key=lambda x: x[0])
                top_k_coords = {(t[1], t[2]) for t in top_k_tiles}
                
                if (result.tile_x, result.tile_y) not in top_k_coords:
                    continue
            
            filtered_pairs.append(p)
        
        filtered_sets = []
        for s in result.sets:
            key = _set_key(s)
            
            # minimum tiles present
            if set_tile_counts.get(key, 0) < min_tiles:
                continue
            
            # top-K filtering
            if cfg.top_k_tiles is not None or cfg.top_k_percent is not None:
                tiles_for_combo = set_best_tiles.get(key, [])
                n_tiles = len(tiles_for_combo)
                
                if cfg.top_k_tiles is not None:
                    k = cfg.top_k_tiles
                elif cfg.top_k_percent is not None:
                    k = max(1, int(n_tiles * cfg.top_k_percent))
                else:
                    k = n_tiles
                
                top_k_tiles = heapq.nlargest(k, tiles_for_combo, key=lambda x: x[0])
                top_k_coords = {(t[1], t[2]) for t in top_k_tiles}
                
                if (result.tile_x, result.tile_y) not in top_k_coords:
                    continue
            
            filtered_sets.append(s)
        
        return OverlapTileResult(
            tile_x=result.tile_x,
            tile_y=result.tile_y,
            tile_z=result.tile_z,
            tile_shape=result.tile_shape,
            total_voxels=result.total_voxels,
            radii_um=result.radii_um,
            marker_vox=result.marker_vox,
            channel_stats=result.channel_stats,
            pairs=filtered_pairs,
            sets=filtered_sets,
            n_active_channels=result.n_active_channels,
            n_frequent_pairs=result.n_frequent_pairs,
        )


def filter_and_aggregate(
    checkpoint_dir: Path,
    filter_config: FilterConfig,
    pipeline_config, 
    channel_names: List[str],
    save_filtered_checkpoints: bool = False,
) -> Path:
    """
    filter, aggregate and write results to .bioset file.
    Args:
        checkpoint_dir: Directory with tile checkpoints
        filter_config: Filtering configuration
        pipeline_config: Pipeline configuration (for output paths, hierarchy, etc.)
        channel_names: List of channel names
        save_filtered_checkpoints: If True, save filtered checkpoints
        
    Returns:
        Path to output .bioset file
    """
    from .aggregation import HierarchicalAggregator
    from .writer import BiosetWriter
    
    filterer = TileFilter(filter_config)
    
    filtered_checkpoint_dir = None
    if save_filtered_checkpoints:
        filtered_checkpoint_dir = checkpoint_dir.parent / f"{checkpoint_dir.name}_filtered"
    
    filtered_results = filterer.filter_checkpoints(
        checkpoint_dir,
        output_dir=filtered_checkpoint_dir,
    )
    
    if not filtered_results:
        raise RuntimeError("No results after filtering!")
    
    first = filtered_results[0]
    z = first.tile_shape[0]
    max_y = max(r.tile_y for r in filtered_results) + 1
    max_x = max(r.tile_x for r in filtered_results) + 1
    tile_y, tile_x = pipeline_config.tile_xy
    y = max_y * tile_y
    x = max_x * tile_x
    
    print("[Filter] Aggregating filtered results...")
    aggregator = HierarchicalAggregator(
        base_tile_y=tile_y,
        base_tile_x=tile_x,
        n_levels=pipeline_config.hierarchy_levels,
    )
    
    for result in filtered_results:
        aggregator.add_tile_result(result)
    
    levels = aggregator.aggregate()
    
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
        print(f"  Level {lvl.level}: {len(lvl.pairs):,} pairs, {len(lvl.sets):,} sets")
    
    output_dir = Path(pipeline_config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{pipeline_config.output_name}_filtered.bioset"
    
    print(f"[Filter] Writing to {output_path}...")
    
    writer = BiosetWriter(
        output_path=output_path,
        channel_names=channel_names,
        dilation_amounts=pipeline_config.dilate_um,
        volume_shape=(z, y, x),
    )
    
    writer.write_metadata(hierarchy_meta)
    
    for level in levels:
        writer.write_hierarchy_level(level)
    
    final_path = writer.finalize()
    print(f"[Filter] Complete! Output: {final_path}")
    
    return final_path