"""
Checkpoint utilities for saving and loading tile results.
Enables resumable processing and separation of GPU/CPU stages.
"""
from __future__ import annotations

import json
import gzip
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import asdict

from .stages.overlaps import (
    OverlapTileResult, 
    ChannelTileStats, 
    PairRow, 
    SetRow,
)


def _tile_checkpoint_path(checkpoint_dir: Path, tile_x: int, tile_y: int) -> Path:
    """Get the checkpoint file path for a tile."""
    return checkpoint_dir / f"tile_{tile_x:04d}_{tile_y:04d}.json.gz"


def save_tile_checkpoint(
    checkpoint_dir: Path,
    result: OverlapTileResult,
) -> Path:
    """
    Save a single tile result to a compressed JSON file.
    Returns the path to the saved file.
    """
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    filepath = _tile_checkpoint_path(checkpoint_dir, result.tile_x, result.tile_y)
    
    # Convert to serializable dict
    data = {
        "tile_x": result.tile_x,
        "tile_y": result.tile_y,
        "tile_z": result.tile_z,
        "tile_shape": list(result.tile_shape),
        "total_voxels": result.total_voxels,
        "radii_um": result.radii_um,
        "marker_vox": {str(r): v for r, v in result.marker_vox.items()},
        "n_active_channels": result.n_active_channels,
        "n_frequent_pairs": result.n_frequent_pairs,
        "channel_stats": [asdict(cs) for cs in result.channel_stats],
        "pairs": [asdict(p) for p in result.pairs],
        "sets": [
            {
                "tile_x": s.tile_x,
                "tile_y": s.tile_y,
                "r_um": s.r_um,
                "k": s.k,
                "members": list(s.members),
                "inter_vox": s.inter_vox,
                "union_vox": s.union_vox,
                "iou": s.iou,
                "overlap_coeff": s.overlap_coeff,
            }
            for s in result.sets
        ],
    }
    
    with gzip.open(filepath, 'wt', encoding='utf-8') as f:
        json.dump(data, f)
    
    return filepath


def load_tile_checkpoint(filepath: Path) -> OverlapTileResult:
    """Load a tile result from a checkpoint file."""
    with gzip.open(filepath, 'rt', encoding='utf-8') as f:
        data = json.load(f)
    
    # Reconstruct marker_vox with float keys
    marker_vox = {float(r): v for r, v in data["marker_vox"].items()}
    
    # Reconstruct dataclasses
    channel_stats = [
        ChannelTileStats(**cs) for cs in data["channel_stats"]
    ]
    pairs = [PairRow(**p) for p in data["pairs"]]
    sets = [
        SetRow(
            tile_x=s["tile_x"],
            tile_y=s["tile_y"],
            r_um=s["r_um"],
            k=s["k"],
            members=tuple(s["members"]),
            inter_vox=s["inter_vox"],
            union_vox=s["union_vox"],
            iou=s["iou"],
            overlap_coeff=s["overlap_coeff"],
        )
        for s in data["sets"]
    ]
    
    return OverlapTileResult(
        tile_x=data["tile_x"],
        tile_y=data["tile_y"],
        tile_z=data["tile_z"],
        tile_shape=tuple(data["tile_shape"]),
        total_voxels=data["total_voxels"],
        radii_um=data["radii_um"],
        marker_vox=marker_vox,
        channel_stats=channel_stats,
        pairs=pairs,
        sets=sets,
        n_active_channels=data["n_active_channels"],
        n_frequent_pairs=data["n_frequent_pairs"],
    )


def get_completed_tiles(checkpoint_dir: Path) -> Set[Tuple[int, int]]:
    """Get set of (tile_x, tile_y) that have been checkpointed."""
    completed = set()
    if not checkpoint_dir.exists():
        return completed
    
    for filepath in checkpoint_dir.glob("tile_*.json.gz"):
        try:
            # Parse tile coordinates from filename
            parts = filepath.stem.replace(".json", "").split("_")
            tile_x = int(parts[1])
            tile_y = int(parts[2])
            completed.add((tile_x, tile_y))
        except (IndexError, ValueError):
            continue
    
    return completed


def load_all_checkpoints(checkpoint_dir: Path) -> List[OverlapTileResult]:
    """Load all tile checkpoints from a directory."""
    results = []
    if not checkpoint_dir.exists():
        return results
    
    filepaths = sorted(checkpoint_dir.glob("tile_*.json.gz"))
    for filepath in filepaths:
        try:
            result = load_tile_checkpoint(filepath)
            results.append(result)
        except Exception as e:
            print(f"Warning: Failed to load {filepath}: {e}")
    
    return results


def get_checkpoint_stats(checkpoint_dir: Path) -> Dict:
    """Get statistics about checkpointed tiles."""
    completed = get_completed_tiles(checkpoint_dir)
    
    if not completed:
        return {
            "n_completed": 0,
            "tile_x_range": None,
            "tile_y_range": None,
        }
    
    tile_xs = [t[0] for t in completed]
    tile_ys = [t[1] for t in completed]
    
    return {
        "n_completed": len(completed),
        "tile_x_range": (min(tile_xs), max(tile_xs)),
        "tile_y_range": (min(tile_ys), max(tile_ys)),
    }