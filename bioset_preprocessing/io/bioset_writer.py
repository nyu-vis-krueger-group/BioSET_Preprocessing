"""
BioSET output writer - gzipped SQLite database (.bioset format).

Replaces the CSV-based ResultSaver with a structured database that supports:
- Per-channel tile statistics (for LLM-queryable tile summaries)
- Pairwise overlap counts + enrichment metrics
- Higher-order combinations with enrichment scores
- Hierarchical tile aggregation
- Multi-dilation analysis
"""

import gzip
import json
import logging
import sqlite3
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# Schema Definition
# =============================================================================

SCHEMA_SQL = """
-- Global metadata (channels, config, physical dimensions, etc.)
CREATE TABLE IF NOT EXISTS metadata (
    key     TEXT PRIMARY KEY,
    value   TEXT NOT NULL  -- JSON-encoded
);

-- Per-channel statistics at each tile (for LLM tile summaries)
CREATE TABLE IF NOT EXISTS channel_stats (
    tile_x0         INTEGER NOT NULL,
    tile_x1         INTEGER NOT NULL,
    tile_y0         INTEGER NOT NULL,
    tile_y1         INTEGER NOT NULL,
    hierarchy_level INTEGER NOT NULL DEFAULT 0,
    channel         TEXT    NOT NULL,  -- channel name
    channel_idx     INTEGER NOT NULL,  -- channel index
    
    -- Thresholding results
    threshold_value REAL,
    active_voxels   INTEGER NOT NULL DEFAULT 0,
    active_fraction REAL    NOT NULL DEFAULT 0.0,
    
    -- Intensity statistics (pre-threshold)
    mean_intensity  REAL,
    max_intensity   REAL,
    
    -- Total voxels in this tile (for normalization)
    total_voxels    INTEGER NOT NULL DEFAULT 0,
    
    PRIMARY KEY (tile_x0, tile_y0, hierarchy_level, channel_idx)
);

-- Combination overlap counts + enrichment metrics
CREATE TABLE IF NOT EXISTS combinations (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    channels        TEXT    NOT NULL,  -- pipe-separated sorted names: "A|B|C"
    channel_indices TEXT    NOT NULL,  -- pipe-separated sorted indices: "0|3|7"
    channel_count   INTEGER NOT NULL,
    dilation        INTEGER NOT NULL DEFAULT 0,  -- dilation in micrometers
    hierarchy_level INTEGER NOT NULL DEFAULT 0,
    
    -- Raw counts
    overlap_count   INTEGER NOT NULL DEFAULT 0,  -- |A ∩ B ∩ ...|
    total_voxels    INTEGER NOT NULL DEFAULT 0,  -- tile total voxels (for reference)
    
    -- Enrichment metrics (computed from pooled counts)
    -- For pairs: enrichment = P(B|A) / P(B) = (|A∩B|/|A|) / (|B|/total)
    -- For triples+: higher_order_enrichment = observed / pairwise-expected
    enrichment_ratio    REAL,   -- E_indep: vs independence baseline
    higher_order_enrichment REAL -- E_higher: vs pairwise-expected (NULL for pairs)
);

-- Spatial tiles for each combination
CREATE TABLE IF NOT EXISTS tiles (
    combination_id  INTEGER NOT NULL REFERENCES combinations(id),
    tile_x0         INTEGER NOT NULL,
    tile_x1         INTEGER NOT NULL,
    tile_y0         INTEGER NOT NULL,
    tile_y1         INTEGER NOT NULL,
    
    -- Raw overlap count in this tile
    count           INTEGER NOT NULL DEFAULT 0,
    
    -- Per-tile enrichment (if applicable)
    enrichment_ratio REAL
);

-- Asymmetric pairwise metrics (conditional probabilities)
-- One row per ORDERED pair per tile per dilation
CREATE TABLE IF NOT EXISTS pairwise_metrics (
    tile_x0         INTEGER NOT NULL,
    tile_x1         INTEGER NOT NULL,
    tile_y0         INTEGER NOT NULL,
    tile_y1         INTEGER NOT NULL,
    hierarchy_level INTEGER NOT NULL DEFAULT 0,
    dilation        INTEGER NOT NULL DEFAULT 0,
    
    channel_a       TEXT    NOT NULL,  -- conditioning channel name
    channel_b       TEXT    NOT NULL,  -- target channel name
    channel_a_idx   INTEGER NOT NULL,
    channel_b_idx   INTEGER NOT NULL,
    
    -- Counts
    count_a         INTEGER NOT NULL DEFAULT 0,  -- |A|
    count_b         INTEGER NOT NULL DEFAULT 0,  -- |B|
    count_ab        INTEGER NOT NULL DEFAULT 0,  -- |A ∩ B|
    
    -- Metrics
    prob_b_given_a  REAL,    -- P(B|A) = |A∩B| / |A|
    prob_a_given_b  REAL,    -- P(A|B) = |A∩B| / |B|
    jaccard         REAL,    -- |A∩B| / |A∪B|
    overlap_coeff   REAL,    -- |A∩B| / min(|A|, |B|)
    enrichment      REAL,    -- P(B|A) / P(B)

    PRIMARY KEY (tile_x0, tile_y0, hierarchy_level, dilation, channel_a_idx, channel_b_idx)
);

-- Indices for fast queries
CREATE INDEX IF NOT EXISTS idx_channel_stats_tile 
    ON channel_stats(tile_x0, tile_y0, hierarchy_level);
CREATE INDEX IF NOT EXISTS idx_channel_stats_channel 
    ON channel_stats(channel_idx, hierarchy_level);

CREATE INDEX IF NOT EXISTS idx_combinations_channels 
    ON combinations(channels);
CREATE INDEX IF NOT EXISTS idx_combinations_dilation 
    ON combinations(dilation);
CREATE INDEX IF NOT EXISTS idx_combinations_level 
    ON combinations(hierarchy_level);
CREATE INDEX IF NOT EXISTS idx_combinations_dilation_level 
    ON combinations(dilation, hierarchy_level);
CREATE INDEX IF NOT EXISTS idx_combinations_enrichment 
    ON combinations(dilation, hierarchy_level, enrichment_ratio);

CREATE INDEX IF NOT EXISTS idx_tiles_combination 
    ON tiles(combination_id);
CREATE INDEX IF NOT EXISTS idx_tiles_spatial 
    ON tiles(tile_x0, tile_y0);

CREATE INDEX IF NOT EXISTS idx_pairwise_tile 
    ON pairwise_metrics(tile_x0, tile_y0, hierarchy_level, dilation);
CREATE INDEX IF NOT EXISTS idx_pairwise_channels 
    ON pairwise_metrics(channel_a_idx, channel_b_idx, dilation, hierarchy_level);
"""


class BiosetWriter:
    """
    Writes analysis results to a .bioset (gzipped SQLite) file.
    
    Usage:
        writer = BiosetWriter("/path/to/output.bioset")
        writer.open()
        
        # Write metadata
        writer.write_metadata(channels=["Hoechst", "MART1", ...], ...)
        
        # Write per-channel stats for a tile
        writer.write_channel_stat(tile_x0=0, ..., channel="Hoechst", ...)
        
        # Write pairwise metrics for a tile  
        writer.write_pairwise_metrics(tile_x0=0, ..., metrics=[...])
        
        # Write combination overlaps
        writer.write_combination(channels="A|B", ..., tiles=[...])
        
        # Finalize
        writer.close()  # compresses to .bioset
    
    The writer uses WAL mode and batched transactions for performance.
    """
    
    def __init__(self, output_path: str | Path):
        self.output_path = Path(output_path)
        if not self.output_path.suffix == ".bioset":
            self.output_path = self.output_path.with_suffix(".bioset")
        
        # We write to a temp .sqlite file, then gzip on close
        self._temp_db_path: Optional[Path] = None
        self._conn: Optional[sqlite3.Connection] = None
        self._combination_cache: Dict[str, int] = {}  # "channels|dilation|level" -> id
        
        # Batching for performance
        self._channel_stats_batch: list = []
        self._pairwise_batch: list = []
        self._tiles_batch: list = []
        self._batch_size = 1000
    
    def open(self) -> "BiosetWriter":
        """Open the database for writing."""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create temp file next to final output
        self._temp_db_path = self.output_path.with_suffix(".sqlite_tmp")
        
        # Remove stale temp file
        if self._temp_db_path.exists():
            self._temp_db_path.unlink()
        
        self._conn = sqlite3.connect(str(self._temp_db_path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA cache_size=-64000")  # 64MB cache
        self._conn.execute("PRAGMA temp_store=MEMORY")
        self._conn.executescript(SCHEMA_SQL)
        self._conn.commit()
        
        logger.info(f"Opened bioset writer: {self._temp_db_path}")
        return self
    
    # -----------------------------------------------------------------
    # Metadata
    # -----------------------------------------------------------------
    
    def write_metadata(
        self,
        channels: List[str],
        channel_indices: List[int],
        hierarchy_levels: List[dict],
        dilation_amounts: List[int],
        volume_bounds: dict,
        physical_dimensions: Optional[dict] = None,
        threshold_method: Optional[str] = None,
        extra: Optional[dict] = None,
    ) -> None:
        """Write global metadata."""
        items = {
            "channels": channels,
            "channel_indices": channel_indices,
            "hierarchy_levels": hierarchy_levels,
            "dilation_amounts": dilation_amounts,
            "volume_bounds": volume_bounds,
        }
        if physical_dimensions:
            items["physical_dimensions"] = physical_dimensions
        if threshold_method:
            items["threshold_method"] = threshold_method
        if extra:
            items.update(extra)
        
        with self._conn:
            for key, value in items.items():
                self._conn.execute(
                    "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
                    (key, json.dumps(value)),
                )
    
    # -----------------------------------------------------------------
    # Channel Stats (per-channel, per-tile)
    # -----------------------------------------------------------------
    
    def write_channel_stat(
        self,
        tile_x0: int, tile_x1: int,
        tile_y0: int, tile_y1: int,
        hierarchy_level: int,
        channel: str,
        channel_idx: int,
        threshold_value: float,
        active_voxels: int,
        active_fraction: float,
        total_voxels: int,
        mean_intensity: Optional[float] = None,
        max_intensity: Optional[float] = None,
    ) -> None:
        """Buffer a channel stat row."""
        self._channel_stats_batch.append((
            tile_x0, tile_x1, tile_y0, tile_y1,
            hierarchy_level, channel, channel_idx,
            threshold_value, active_voxels, active_fraction,
            mean_intensity, max_intensity, total_voxels,
        ))
        if len(self._channel_stats_batch) >= self._batch_size:
            self._flush_channel_stats()
    
    def _flush_channel_stats(self) -> None:
        if not self._channel_stats_batch:
            return
        with self._conn:
            self._conn.executemany(
                """INSERT OR REPLACE INTO channel_stats 
                   (tile_x0, tile_x1, tile_y0, tile_y1,
                    hierarchy_level, channel, channel_idx,
                    threshold_value, active_voxels, active_fraction,
                    mean_intensity, max_intensity, total_voxels)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                self._channel_stats_batch,
            )
        self._channel_stats_batch.clear()
    
    # -----------------------------------------------------------------
    # Pairwise Metrics (asymmetric, per-tile)
    # -----------------------------------------------------------------
    
    def write_pairwise_metric(
        self,
        tile_x0: int, tile_x1: int,
        tile_y0: int, tile_y1: int,
        hierarchy_level: int,
        dilation: int,
        channel_a: str, channel_b: str,
        channel_a_idx: int, channel_b_idx: int,
        count_a: int, count_b: int, count_ab: int,
        total_voxels: int,
    ) -> None:
        """Buffer a pairwise metric row. Computes derived metrics automatically."""
        # Compute metrics
        prob_b_given_a = count_ab / count_a if count_a > 0 else None
        prob_a_given_b = count_ab / count_b if count_b > 0 else None
        union_ab = count_a + count_b - count_ab
        jaccard = count_ab / union_ab if union_ab > 0 else None
        min_ab = min(count_a, count_b)
        overlap_coeff = count_ab / min_ab if min_ab > 0 else None
        
        # Enrichment: P(B|A) / P(B) = (|A∩B|/|A|) / (|B|/total)
        prob_b = count_b / total_voxels if total_voxels > 0 else 0
        enrichment = (prob_b_given_a / prob_b) if (prob_b_given_a is not None and prob_b > 0) else None
        
        self._pairwise_batch.append((
            tile_x0, tile_x1, tile_y0, tile_y1,
            hierarchy_level, dilation,
            channel_a, channel_b,
            channel_a_idx, channel_b_idx,
            count_a, count_b, count_ab,
            prob_b_given_a, prob_a_given_b,
            jaccard, overlap_coeff, enrichment,
        ))
        if len(self._pairwise_batch) >= self._batch_size:
            self._flush_pairwise()
    
    def _flush_pairwise(self) -> None:
        if not self._pairwise_batch:
            return
        with self._conn:
            self._conn.executemany(
                """INSERT OR REPLACE INTO pairwise_metrics
                   (tile_x0, tile_x1, tile_y0, tile_y1,
                    hierarchy_level, dilation,
                    channel_a, channel_b,
                    channel_a_idx, channel_b_idx,
                    count_a, count_b, count_ab,
                    prob_b_given_a, prob_a_given_b,
                    jaccard, overlap_coeff, enrichment)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                self._pairwise_batch,
            )
        self._pairwise_batch.clear()
    
    # -----------------------------------------------------------------
    # Combinations (for UpSet/heatmap visualization + higher-order)
    # -----------------------------------------------------------------
    
    def write_combination_with_tile(
        self,
        channels: str,           # pipe-separated: "A|B|C"
        channel_indices: str,    # pipe-separated: "0|3|7"
        channel_count: int,
        dilation: int,
        hierarchy_level: int,
        overlap_count: int,
        total_voxels: int,
        tile_x0: int, tile_x1: int,
        tile_y0: int, tile_y1: int,
        tile_count: int,
        enrichment_ratio: Optional[float] = None,
        higher_order_enrichment: Optional[float] = None,
        tile_enrichment: Optional[float] = None,
    ) -> None:
        """
        Write a combination + its tile data in one call.
        
        The combination row is created/updated, and a tile row is appended.
        Uses caching to avoid redundant combination lookups.
        """
        # Get or create the combination row
        cache_key = f"{channels}|{dilation}|{hierarchy_level}"
        
        if cache_key not in self._combination_cache:
            cursor = self._conn.execute(
                """INSERT INTO combinations
                   (channels, channel_indices, channel_count, dilation,
                    hierarchy_level, overlap_count, total_voxels,
                    enrichment_ratio, higher_order_enrichment)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                (channels, channel_indices, channel_count, dilation,
                 hierarchy_level, overlap_count, total_voxels,
                 enrichment_ratio, higher_order_enrichment),
            )
            combo_id = cursor.lastrowid
            self._combination_cache[cache_key] = combo_id
        else:
            combo_id = self._combination_cache[cache_key]
            # Update the running total
            self._conn.execute(
                """UPDATE combinations 
                   SET overlap_count = overlap_count + ?,
                       enrichment_ratio = ?,
                       higher_order_enrichment = ?
                   WHERE id = ?""",
                (overlap_count, enrichment_ratio, higher_order_enrichment, combo_id),
            )
        
        # Buffer the tile row
        self._tiles_batch.append((
            combo_id, tile_x0, tile_x1, tile_y0, tile_y1,
            tile_count, tile_enrichment,
        ))
        if len(self._tiles_batch) >= self._batch_size:
            self._flush_tiles()
    
    def _flush_tiles(self) -> None:
        if not self._tiles_batch:
            return
        with self._conn:
            self._conn.executemany(
                """INSERT INTO tiles
                   (combination_id, tile_x0, tile_x1, tile_y0, tile_y1,
                    count, enrichment_ratio)
                   VALUES (?,?,?,?,?,?,?)""",
                self._tiles_batch,
            )
        self._tiles_batch.clear()
    
    # -----------------------------------------------------------------
    # Batch helpers for pipeline integration
    # -----------------------------------------------------------------
    
    def write_tile_pairwise_batch(
        self,
        tile_x0: int, tile_x1: int,
        tile_y0: int, tile_y1: int,
        hierarchy_level: int,
        dilation: int,
        channel_names: Dict[int, str],
        channel_counts: Dict[int, int],
        pairwise_overlaps: Dict[Tuple[int, int], int],
        total_voxels: int,
    ) -> None:
        """
        Write all pairwise metrics for a tile in one call.
        
        Args:
            channel_names: {channel_idx: channel_name}
            channel_counts: {channel_idx: active_voxel_count}
            pairwise_overlaps: {(ch_a, ch_b): overlap_count} (a < b)
            total_voxels: total voxels in tile
        """
        for (ch_a, ch_b), count_ab in pairwise_overlaps.items():
            count_a = channel_counts.get(ch_a, 0)
            count_b = channel_counts.get(ch_b, 0)
            name_a = channel_names.get(ch_a, f"ch{ch_a}")
            name_b = channel_names.get(ch_b, f"ch{ch_b}")
            
            self.write_pairwise_metric(
                tile_x0=tile_x0, tile_x1=tile_x1,
                tile_y0=tile_y0, tile_y1=tile_y1,
                hierarchy_level=hierarchy_level,
                dilation=dilation,
                channel_a=name_a, channel_b=name_b,
                channel_a_idx=ch_a, channel_b_idx=ch_b,
                count_a=count_a, count_b=count_b,
                count_ab=count_ab, total_voxels=total_voxels,
            )
    
    def begin_transaction(self) -> None:
        """Begin an explicit transaction (for tile-level batching)."""
        self._conn.execute("BEGIN IMMEDIATE")
    
    def commit_transaction(self) -> None:
        """Commit and flush all pending batches."""
        self._flush_channel_stats()
        self._flush_pairwise()
        self._flush_tiles()
        self._conn.commit()
    
    # -----------------------------------------------------------------
    # Finalize
    # -----------------------------------------------------------------
    
    def close(self) -> Path:
        """
        Flush all batches, close DB, and gzip to .bioset.
        
        Returns:
            Path to the final .bioset file
        """
        # Flush remaining batches
        self._flush_channel_stats()
        self._flush_pairwise()
        self._flush_tiles()
        self._conn.commit()
        
        # Optimize before closing
        self._conn.execute("PRAGMA optimize")
        self._conn.execute("VACUUM")
        self._conn.close()
        self._conn = None
        
        # Gzip the sqlite file
        logger.info(f"Compressing to {self.output_path}...")
        with open(self._temp_db_path, "rb") as f_in:
            with gzip.open(self.output_path, "wb", compresslevel=6) as f_out:
                while chunk := f_in.read(1024 * 1024):  # 1MB chunks
                    f_out.write(chunk)
        
        # Report sizes
        raw_size = self._temp_db_path.stat().st_size
        compressed_size = self.output_path.stat().st_size
        ratio = compressed_size / raw_size if raw_size > 0 else 0
        logger.info(
            f"Bioset written: {compressed_size / 1e6:.1f}MB "
            f"(compressed from {raw_size / 1e6:.1f}MB, ratio={ratio:.2f})"
        )
        
        # Cleanup temp
        self._temp_db_path.unlink()
        self._temp_db_path = None
        
        return self.output_path
    
    def __enter__(self) -> "BiosetWriter":
        return self.open()
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._conn is not None:
            if exc_type is None:
                self.close()
            else:
                # Error path: close DB but don't gzip
                self._conn.close()
                self._conn = None
                logger.error(f"BiosetWriter closed due to exception: {exc_val}")