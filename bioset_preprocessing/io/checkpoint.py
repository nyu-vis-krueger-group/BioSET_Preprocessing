"""
Checkpoint management for resumable processing.

Tracks which tiles have been completed so processing can resume
after interruption.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Set, Tuple

logger = logging.getLogger(__name__)


class CheckpointManager:
    """
    Manages checkpointing for resume capability.
    
    Tracks completed tiles in a JSON file. When processing resumes,
    already-completed tiles can be skipped.
    
    Args:
        output_dir: Directory for checkpoint file
        filename: Name of checkpoint file (default: checkpoint.json)
        
    Example:
        >>> checkpoint = CheckpointManager(Path("./results"))
        >>> if not checkpoint.is_completed(0, 0):
        ...     process_tile(0, 0)
        ...     checkpoint.mark_completed(0, 0)
    """
    
    def __init__(
        self,
        output_dir: Path,
        filename: str = "checkpoint.json",
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.checkpoint_path = self.output_dir / filename
        self.completed_tiles: Set[Tuple[int, int]] = self._load()
    
    def _load(self) -> Set[Tuple[int, int]]:
        """Load completed tiles from checkpoint file."""
        if not self.checkpoint_path.exists():
            self.stage_completion = {}  # ADD THIS
            return set()
        
        try:
            with open(self.checkpoint_path, "r") as f:
                data = json.load(f)
            
            # Convert lists back to tuples for set
            tiles = set(tuple(t) for t in data.get("completed_tiles", []))
            
            # ADD THIS: Load stage completion
            self.stage_completion = {}
            for tile_key, stages in data.get("stage_completion", {}).items():
                self.stage_completion[tile_key] = set(stages)
            
            logger.info(f"Loaded checkpoint: {len(tiles)} tiles completed")
            return tiles
        
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Could not load checkpoint: {e}")
            self.stage_completion = {}  # ADD THIS
            return set()
    
    def _save(self) -> None:
        """Save checkpoint to file."""
        # Convert stage_completion sets to lists for JSON
        stage_completion_json = {}
        if hasattr(self, 'stage_completion'):
            for tile_key, stages in self.stage_completion.items():
                stage_completion_json[tile_key] = list(stages)
        
        data = {
            "completed_tiles": [list(t) for t in sorted(self.completed_tiles)],
            "total_completed": len(self.completed_tiles),
            "stage_completion": stage_completion_json,  # ADD THIS
            "last_updated": datetime.now().isoformat(),
        }
        
        with open(self.checkpoint_path, "w") as f:
            json.dump(data, f, indent=2)
    
    def is_completed(self, tile_y: int, tile_x: int) -> bool:
        """
        Check if a tile has been completed.
        
        Args:
            tile_y: Tile Y index
            tile_x: Tile X index
            
        Returns:
            True if tile has been processed
        """
        return (tile_y, tile_x) in self.completed_tiles
    
    def mark_completed(self, tile_y: int, tile_x: int) -> None:
        """
        Mark a tile as completed and save checkpoint.
        
        Args:
            tile_y: Tile Y index
            tile_x: Tile X index
        """
        self.completed_tiles.add((tile_y, tile_x))
        self._save()

    def mark_stage_completed(self, tile_y: int, tile_x: int, stage: str) -> None:
        """
        Mark a specific stage as completed for a tile.
        
        Args:
            tile_y: Tile Y index
            tile_x: Tile X index
            stage: Stage name (e.g., 'threshold', 'cc_filter', 'dilation_0', 'dilation_5')
        """
        tile_key = f"{tile_y},{tile_x}"
        if not hasattr(self, 'stage_completion'):
            self.stage_completion = {}
        
        if tile_key not in self.stage_completion:
            self.stage_completion[tile_key] = set()
        
        self.stage_completion[tile_key].add(stage)
        self._save()
    
    def is_stage_completed(self, tile_y: int, tile_x: int, stage: str) -> bool:
        """
        Check if a specific stage is completed for a tile.
        
        Args:
            tile_y: Tile Y index
            tile_x: Tile X index
            stage: Stage name
            
        Returns:
            True if stage is completed
        """
        if not hasattr(self, 'stage_completion'):
            return False
        
        tile_key = f"{tile_y},{tile_x}"
        return tile_key in self.stage_completion and stage in self.stage_completion[tile_key]
    
    def reset(self) -> None:
        """Clear all checkpoint data."""
        self.completed_tiles.clear()
        if self.checkpoint_path.exists():
            self.checkpoint_path.unlink()
        logger.info("Checkpoint reset")
    
    @property
    def n_completed(self) -> int:
        """Number of completed tiles."""
        return len(self.completed_tiles)
    
    def __repr__(self) -> str:
        return f"CheckpointManager({self.n_completed} tiles completed)"
