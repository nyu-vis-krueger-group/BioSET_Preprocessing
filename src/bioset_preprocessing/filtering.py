from __future__ import annotations

import gzip
import json
from dataclasses import dataclass
from typing import Dict, List, Tuple, Set, Optional
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

from .stages.overlaps import (
    OverlapTileResult,
    ChannelTileStats,
    PairRow,
    SetRow,
)


@dataclass
class FilterConfig:
    """Configuration for filtering tile results."""
    
    # Minimum thresholds (applied at max dilation first)
    min_overlap_coeff: float = 0.0
    min_inter_vox: int = 0
    
    # If True, filter by max dilation - if it fails, remove all dilations for that combo
    filter_by_max_dilation: bool = True
    
    # Global: combination must appear in at least N tiles (at max dilation)
    min_tiles_present: int = 1
    
    # Top-K: keep only top X% of tiles per combination (by overlap_coeff at max dilation)
    top_k_percent: Optional[float] = None  # e.g., 0.25 for top 25%
    
    # Set size limits
    min_set_size: int = 2
    max_set_size: int = 10
    
    # Parallelism
    n_workers: int = 8


@dataclass 
class FilterStats:
    tiles_processed: int = 0
    pairs_before: int = 0
    pairs_after: int = 0
    sets_before: int = 0
    sets_after: int = 0
    
    def __str__(self) -> str:
        p_pct = 100 * (1 - self.pairs_after / max(1, self.pairs_before))
        s_pct = 100 * (1 - self.sets_after / max(1, self.sets_before))
        return (
            f"FilterStats:\n"
            f"  Tiles: {self.tiles_processed}\n"
            f"  Pairs: {self.pairs_before:,} → {self.pairs_after:,} ({p_pct:.1f}% removed)\n"
            f"  Sets: {self.sets_before:,} → {self.sets_after:,} ({s_pct:.1f}% removed)"
        )


def _load_checkpoint_raw(filepath: Path) -> dict:
    """Load checkpoint as raw dict (faster than creating dataclasses)."""
    with gzip.open(filepath, 'rt', encoding='utf-8') as f:
        return json.load(f)


def _save_checkpoint_raw(filepath: Path, data: dict) -> None:
    """Save checkpoint from raw dict."""
    with gzip.open(filepath, 'wt', encoding='utf-8') as f:
        json.dump(data, f)


def _get_max_dilation(radii: List[float]) -> float:
    """Get the maximum dilation radius."""
    return max(radii)


# ============================================================================
# PASS 1: Collect statistics (parallel)
# ============================================================================

def _collect_stats_from_file(filepath: Path, max_r: float, cfg: FilterConfig) -> dict:
    """
    Collect statistics from a single checkpoint file.
    Returns dict with pair/set stats at max dilation only.
    """
    data = _load_checkpoint_raw(filepath)
    tile_x, tile_y = data['tile_x'], data['tile_y']
    
    pair_stats = {}  # (a, b) -> {'count': int, 'max_oc': float, 'best_tile': (oc, tx, ty)}
    set_stats = {}   
    
    # Process pairs at max dilation only
    for p in data['pairs']:
        if p['r_um'] != max_r:
            continue
        if p['k'] if 'k' in p else 2 < cfg.min_set_size:
            continue
            
        key = (p['a'], p['b'])
        oc = p['overlap_coeff']
        inter = p['inter_vox']
        
        # Apply per-tile threshold
        if oc < cfg.min_overlap_coeff or inter < cfg.min_inter_vox:
            continue
        
        if key not in pair_stats:
            pair_stats[key] = {'count': 0, 'best_oc': 0.0, 'best_tile': None}
        
        pair_stats[key]['count'] += 1
        if oc > pair_stats[key]['best_oc']:
            pair_stats[key]['best_oc'] = oc
            pair_stats[key]['best_tile'] = (oc, tile_x, tile_y)
    
    # Process sets at max dilation only
    for s in data['sets']:
        if s['r_um'] != max_r:
            continue
        if s['k'] < cfg.min_set_size or s['k'] > cfg.max_set_size:
            continue
            
        key = tuple(s['members'])
        oc = s['overlap_coeff']
        inter = s['inter_vox']
        
        if oc < cfg.min_overlap_coeff or inter < cfg.min_inter_vox:
            continue
        
        if key not in set_stats:
            set_stats[key] = {'count': 0, 'best_oc': 0.0, 'best_tile': None}
        
        set_stats[key]['count'] += 1
        if oc > set_stats[key]['best_oc']:
            set_stats[key]['best_oc'] = oc
            set_stats[key]['best_tile'] = (oc, tile_x, tile_y)
    
    return {
        'filepath': str(filepath),
        'pair_stats': pair_stats,
        'set_stats': set_stats,
    }


def _merge_stats(all_stats: List[dict]) -> Tuple[dict, dict]:
    """Merge statistics from all files."""
    pair_global = {}  # (a,b) -> {'count': total, 'tiles': [(oc, tx, ty), ...]}
    set_global = {}
    
    for stats in all_stats:
        for key, s in stats['pair_stats'].items():
            if key not in pair_global:
                pair_global[key] = {'count': 0, 'tiles': []}
            pair_global[key]['count'] += s['count']
            if s['best_tile']:
                pair_global[key]['tiles'].append(s['best_tile'])
        
        for key, s in stats['set_stats'].items():
            if key not in set_global:
                set_global[key] = {'count': 0, 'tiles': []}
            set_global[key]['count'] += s['count']
            if s['best_tile']:
                set_global[key]['tiles'].append(s['best_tile'])
    
    return pair_global, set_global


def _compute_top_k_tiles(
    global_stats: dict, 
    top_k_percent: float,
) -> Dict[Tuple, Set[Tuple[int, int]]]:
    
    keep_tiles = {}
    
    for key, stats in global_stats.items():
        tiles = stats['tiles']
        if not tiles:
            continue
        
        tiles_sorted = sorted(tiles, key=lambda x: x[0], reverse=True)
        
        k = max(1, int(len(tiles_sorted) * top_k_percent))
        top_tiles = tiles_sorted[:k]
        
        keep_tiles[key] = {(t[1], t[2]) for t in top_tiles}
    
    return keep_tiles

def _filter_single_file(args: Tuple) -> dict:
    """
    Filter a single checkpoint file based on global statistics.
    """
    (filepath, max_r, cfg, valid_pairs, valid_sets, 
     pair_keep_tiles, set_keep_tiles) = args
    
    data = _load_checkpoint_raw(filepath)
    tile_coord = (data['tile_x'], data['tile_y'])
    
    pairs_before = len(data['pairs'])
    sets_before = len(data['sets'])
    
    filtered_pairs = []
    for p in data['pairs']:
        key = (p['a'], p['b'])
        
        if key not in valid_pairs:
            continue
        
        if cfg.filter_by_max_dilation:
            if p['r_um'] == max_r:
                if p['overlap_coeff'] < cfg.min_overlap_coeff:
                    continue
                if p['inter_vox'] < cfg.min_inter_vox:
                    continue
                if pair_keep_tiles and key in pair_keep_tiles:
                    if tile_coord not in pair_keep_tiles[key]:
                        continue
            
        filtered_pairs.append(p)
    
    filtered_sets = []
    for s in data['sets']:
        key = tuple(s['members'])
        
        if s['k'] < cfg.min_set_size or s['k'] > cfg.max_set_size:
            continue
        
        if key not in valid_sets:
            continue
        
        if cfg.filter_by_max_dilation:
            if s['r_um'] == max_r:
                if s['overlap_coeff'] < cfg.min_overlap_coeff:
                    continue
                if s['inter_vox'] < cfg.min_inter_vox:
                    continue
                if set_keep_tiles and key in set_keep_tiles:
                    if tile_coord not in set_keep_tiles[key]:
                        continue
        
        filtered_sets.append(s)
    
    data['pairs'] = filtered_pairs
    data['sets'] = filtered_sets
    
    return {
        'filepath': filepath,
        'data': data,
        'pairs_before': pairs_before,
        'pairs_after': len(filtered_pairs),
        'sets_before': sets_before,
        'sets_after': len(filtered_sets),
    }

class StreamingFilter:
    """
    High-performance streaming filter for large checkpoint datasets.
    
    Usage:
        filter_cfg = FilterConfig(
            min_overlap_coeff=0.05,
            min_tiles_present=10,
            top_k_percent=0.25,
            n_workers=16,
        )
        
        filterer = StreamingFilter(filter_cfg)
        filterer.filter_checkpoints(
            input_dir=Path("checkpoints/melanoma"),
            output_dir=Path("checkpoints/melanoma_filtered"),
        )
    """
    
    def __init__(self, config: FilterConfig):
        self.config = config
        self.stats = FilterStats()
    
    def filter_checkpoints(
        self,
        input_dir: Path,
        output_dir: Path,
    ) -> FilterStats:
       
        cfg = self.config
        filepaths = sorted(input_dir.glob("tile_*.json.gz"))
        n_files = len(filepaths)
        
        if n_files == 0:
            raise RuntimeError(f"No checkpoints found in {input_dir}")
        
        print(f"[StreamingFilter] Found {n_files} checkpoint files")
        print(f"[StreamingFilter] Using {cfg.n_workers} workers")
        
        first_data = _load_checkpoint_raw(filepaths[0])
        max_r = _get_max_dilation(first_data['radii_um'])
        print(f"[StreamingFilter] Max dilation radius: {max_r}")
        
        print(f"\n[Pass 1/2] Collecting statistics at r={max_r}...")
        
        all_stats = []
        with ProcessPoolExecutor(max_workers=cfg.n_workers) as executor:
            futures = {
                executor.submit(_collect_stats_from_file, fp, max_r, cfg): fp 
                for fp in filepaths
            }
            
            completed = 0
            for future in as_completed(futures):
                result = future.result()
                all_stats.append(result)
                completed += 1
                if completed % 500 == 0:
                    print(f"  [{completed}/{n_files}] files processed...")
        
        print(f"  [{n_files}/{n_files}] Pass 1 complete")
        
        print("[Pass 1/2] Merging statistics...")
        pair_global, set_global = _merge_stats(all_stats)
        
        print(f"  Unique pairs at r={max_r}: {len(pair_global):,}")
        print(f"  Unique sets at r={max_r}: {len(set_global):,}")
        
        valid_pairs = {
            k for k, v in pair_global.items() 
            if v['count'] >= cfg.min_tiles_present
        }
        valid_sets = {
            k for k, v in set_global.items()
            if v['count'] >= cfg.min_tiles_present
        }
        
        print(f"  After min_tiles_present={cfg.min_tiles_present}:")
        print(f"    Valid pairs: {len(valid_pairs):,}")
        print(f"    Valid sets: {len(valid_sets):,}")
        
        pair_keep_tiles = None
        set_keep_tiles = None
        
        if cfg.top_k_percent is not None:
            print(f"[Pass 1/2] Computing top {cfg.top_k_percent*100:.0f}% tiles per combination...")
            
            pair_global_valid = {k: v for k, v in pair_global.items() if k in valid_pairs}
            set_global_valid = {k: v for k, v in set_global.items() if k in valid_sets}
            
            pair_keep_tiles = _compute_top_k_tiles(pair_global_valid, cfg.top_k_percent)
            set_keep_tiles = _compute_top_k_tiles(set_global_valid, cfg.top_k_percent)
            
            print(f"    Pairs with top-K tiles: {len(pair_keep_tiles):,}")
            print(f"    Sets with top-K tiles: {len(set_keep_tiles):,}")
        
        del all_stats, pair_global, set_global
        
        print(f"\n[Pass 2/2] Applying filters and writing output...")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        args_list = [
            (fp, max_r, cfg, valid_pairs, valid_sets, pair_keep_tiles, set_keep_tiles)
            for fp in filepaths
        ]
        
        self.stats = FilterStats(tiles_processed=n_files)
        
        with ProcessPoolExecutor(max_workers=cfg.n_workers) as executor:
            futures = {executor.submit(_filter_single_file, args): args[0] for args in args_list}
            
            completed = 0
            for future in as_completed(futures):
                result = future.result()
                
                self.stats.pairs_before += result['pairs_before']
                self.stats.pairs_after += result['pairs_after']
                self.stats.sets_before += result['sets_before']
                self.stats.sets_after += result['sets_after']
                
                out_path = output_dir / Path(result['filepath']).name
                _save_checkpoint_raw(out_path, result['data'])
                
                completed += 1
                if completed % 500 == 0:
                    print(f"  [{completed}/{n_files}] files filtered...")
        
        print(f"  [{n_files}/{n_files}] Pass 2 complete")
        print(f"\n[StreamingFilter] Done!")
        print(self.stats)
        print(f"\nFiltered checkpoints saved to: {output_dir}")
        
        return self.stats

def filter_and_aggregate(
    checkpoint_dir: Path,
    output_checkpoint_dir: Path,
    filter_config: FilterConfig,
    pipeline_config,  
    channel_names: List[str],
) -> Path:
  
    from .aggregation import HierarchicalAggregator
    from .writer import BiosetWriter
    from .checkpoint import load_all_checkpoints
    
    filterer = StreamingFilter(filter_config)
    filterer.filter_checkpoints(
        input_dir=checkpoint_dir,
        output_dir=output_checkpoint_dir,
    )
    
    print("\n[Aggregation] Loading filtered checkpoints...")
    results = load_all_checkpoints(output_checkpoint_dir)
    print(f"  Loaded {len(results)} tiles")
    
    if not results:
        raise RuntimeError("No results after filtering!")
    
    first = results[0]
    z = first.tile_shape[0]
    max_y = max(r.tile_y for r in results) + 1
    max_x = max(r.tile_x for r in results) + 1
    tile_y, tile_x = pipeline_config.tile_xy
    y = max_y * tile_y
    x = max_x * tile_x
    
    print("[Aggregation] Building hierarchy...")
    aggregator = HierarchicalAggregator(
        base_tile_y=tile_y,
        base_tile_x=tile_x,
        n_levels=pipeline_config.hierarchy_levels,
    )
    
    for result in results:
        aggregator.add_tile_result(result)
    
    print("[Aggregation] Computing aggregation...")
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
    
    print(f"[Aggregation] Writing to {output_path}...")
    
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
    print(f"\n[Complete] Output: {final_path}")
    
    return final_path