# BioSET Preprocessing - Developer Documentation

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Processing Pipeline](#processing-pipeline)
3. [Stage Details](#stage-details)
4. [Data Structures](#data-structures)
5. [Database Schema](#database-schema)
6. [Checkpointing System](#checkpointing-system)
7. [Hierarchical Aggregation](#hierarchical-aggregation)
8. [Query Patterns](#query-patterns)
9. [Extending the Pipeline](#extending-the-pipeline)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           BioSET Preprocessing                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────────┐    ┌─────────────────────────────────────────────────┐   │
│   │   Input     │    │              STAGE 1: GPU Processing             │   │
│   │  OME-Zarr   │───▶│  ┌─────────┐ ┌────┐ ┌────────┐ ┌───────────┐   │   │
│   │  (local or  │    │  │Threshold│─▶│ CC │─▶│Dilation│─▶│  Overlap  │   │   │
│   │   remote)   │    │  └─────────┘ └────┘ └────────┘ │   Mining   │   │   │
│   └─────────────┘    │                                 └─────┬─────┘   │   │
│                      │                                       │         │   │
│                      │                               ┌───────▼───────┐ │   │
│                      │                               │  Checkpoints  │ │   │
│                      │                               │  (per tile)   │ │   │
│                      │                               └───────────────┘ │   │
│                      └─────────────────────────────────────────────────┘   │
│                                                                             │
│                      ┌─────────────────────────────────────────────────┐   │
│                      │              STAGE 2: CPU Aggregation            │   │
│                      │  ┌───────────┐ ┌────────────┐ ┌──────────────┐  │   │
│                      │  │   Load    │─▶│ Hierarchical│─▶│   BioSET    │  │   │
│                      │  │Checkpoints│  │ Aggregation │  │   Writer    │  │   │
│                      │  └───────────┘ └────────────┘ └──────┬───────┘  │   │
│                      └──────────────────────────────────────┼──────────┘   │
│                                                              │              │
│   ┌─────────────┐                                           │              │
│   │   Output    │◀──────────────────────────────────────────┘              │
│   │   .bioset   │                                                          │
│   │  (gzipped   │                                                          │
│   │   SQLite)   │                                                          │
│   └─────────────┘                                                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Module Structure

```
src/bioset_preprocessing/
├── __init__.py
├── config.py           # PipelineConfig, VoxelSizeUM dataclasses
├── pipeline.py         # Main Pipeline class orchestrating all stages
├── io.py               # ZarrPyramid for reading OME-Zarr data
├── tiling.py           # Tile iteration and slicing utilities
├── checkpoint.py       # Save/load tile results for resumability
├── aggregation.py      # HierarchicalAggregator for multi-scale stats
├── writer.py           # BiosetWriter for SQLite output
├── filtering.py        # Post-processing database filters
└── stages/
    ├── threshold.py    # AlphaThreshold: adaptive intensity thresholding
    ├── cc_filter.py    # ConnectedComponentsFilter: remove small objects
    ├── dilation.py     # EDTSweepDilation: morphological expansion
    └── overlaps.py     # OverlapMiner: pairwise/set intersection mining
```

---

## Processing Pipeline

### High-Level Flow

```python
Pipeline.__init__(cfg)
    ├── ZarrPyramid.open()          # Open input data
    ├── AlphaThreshold()            # Initialize thresholder
    ├── ConnectedComponentsFilter() # Initialize CC filter
    ├── EDTSweepDilation()          # Initialize dilator
    └── OverlapMiner()              # Initialize overlap miner

Pipeline.run_full_analysis()
    ├── run_tile_processing()       # STAGE 1: GPU
    │   ├── compute_global_thresholds()
    │   └── for each tile:
    │       ├── _process_single_tile()
    │       └── save_tile_checkpoint()
    │
    └── run_aggregation()           # STAGE 2: CPU
        ├── load_all_checkpoints()
        ├── HierarchicalAggregator.aggregate()
        └── BiosetWriter.finalize()
```

### Tiling Strategy

The volume is divided into XY tiles (default 128×128 pixels) for memory-efficient GPU processing:

```
Volume: (1, C, Z, Y, X) = (1, 70, 194, 9585, 10881)
                                    ↓
                    ┌───────────────────────────────┐
                    │  Tile Grid: 75 × 85 = 6,375   │
                    │  tiles at 128×128 pixels      │
                    │                               │
                    │  Each tile: (Z, 128, 128)     │
                    │  = 194 × 128 × 128 voxels     │
                    └───────────────────────────────┘
```

**Key functions** (`tiling.py`):

```python
def iter_tiles_xy(height, width, tile_y, tile_x) -> Iterator[TileIndex]:
    """Iterate over all tile positions."""

def tile_slices(tile: TileIndex, tile_y, tile_x) -> Tuple[slice, slice]:
    """Convert tile index to array slices."""
```

---

## Stage Details

### Stage 1: Thresholding (`stages/threshold.py`)

**Purpose**: Convert raw intensity data to binary masks.

**Algorithm**: Laplacian-of-histogram adaptive thresholding

```
1. GLOBAL THRESHOLD (computed once per channel from low-res volume):
   - Compute maximum intensity projection (MIP)
   - Trim to 98th percentile to remove outliers
   - Estimate background: μ = median, σ = 1.4826 × MAD
   - Global threshold: t_global = μ + z_α × σ  (z_α from normal distribution)

2. PER-TILE THRESHOLD:
   - Compute local quantile from tile background
   - Final threshold: t_final = max(t_global, q_tile)
   - Ensures no tile has threshold below global
```

**Class**: `AlphaThreshold`

```python
class AlphaThreshold:
    def __init__(self, alpha: float = 0.4, trim_q: float = 0.98):
        """
        alpha: False positive rate (1 - alpha = percentile)
        trim_q: Quantile for outlier trimming
        """
    
    def compute_global(self, lowres_vol: np.ndarray) -> float:
        """Compute global threshold from low-resolution volume."""
    
    def compute_tile_gpu(self, vol_gpu: cp.ndarray, t_global: float) -> ThresholdStats:
        """Compute tile-specific threshold on GPU."""
    
    def apply_gpu(self, vol_gpu: cp.ndarray, t_final: float) -> cp.ndarray:
        """Apply threshold, returning boolean mask."""
```

---

### Stage 2: Connected Components Filtering (`stages/cc_filter.py`)

**Purpose**: Remove noise by filtering small connected components.

**Algorithm**:
1. Label connected components using 6- or 26-connectivity
2. Compute size (voxel count) of each component
3. Remove components smaller than `min_obj_vol_um³`

```python
class ConnectedComponentsFilter:
    def __init__(self, min_obj_vol_um3: float, voxel_vol_um3: float, connectivity: int = 26):
        """
        min_obj_vol_um3: Minimum object volume in cubic microns
        voxel_vol_um3: Volume of one voxel (computed from VoxelSizeUM)
        connectivity: 6 (face) or 26 (face + edge + corner)
        """
    
    def __call__(self, mask_gpu: cp.ndarray) -> Tuple[cp.ndarray, CCStats]:
        """Filter mask, returning cleaned mask and statistics."""
```

**Example**: With `min_obj_vol_um3=1.0` and voxel volume `0.14 × 0.14 × 0.28 = 0.0055 µm³`:
- Minimum voxels = ceil(1.0 / 0.0055) = **182 voxels**

---

### Stage 3: Morphological Dilation (`stages/dilation.py`)

**Purpose**: Expand masks to capture nearby/proximal relationships.

**Algorithm**: Euclidean Distance Transform (EDT) sweep
1. Compute EDT from mask boundary (distance in microns)
2. For each radius r, dilated mask = (EDT ≤ r)

```
Original mask:        Dilated (r=1µm):      Dilated (r=2µm):
    ██                    ████                  ██████
    ██                    ████                  ██████
                          ████                  ██████
                                                ██████
```

```python
class EDTSweepDilation:
    def __init__(self, radii_um: Sequence[float], sampling_zyx_um: Tuple[float, float, float]):
        """
        radii_um: Dilation radii (e.g., [0, 0.5, 1.0, 1.5, 2.0])
        sampling_zyx_um: Physical voxel spacing (z, y, x) in microns
        """
    
    def __call__(self, mask_gpu: cp.ndarray) -> DilationResult:
        """Returns dict mapping radius -> dilated mask."""
```

**Output**: `DilationResult.dilated = {0.0: mask0, 0.5: mask1, 1.0: mask2, ...}`

---

### Stage 4: Overlap Mining (`stages/overlaps.py`)

**Purpose**: Compute intersection/union statistics for all channel pairs and higher-order sets.

**Algorithm**: Apriori-style frequent itemset mining
1. Filter to "active" channels (voxel count ≥ threshold)
2. Compute ALL pairwise intersections at maximum dilation
3. Filter to "frequent pairs" (intersection ≥ min_support)
4. Generate candidate k-sets using Apriori principle
5. Evaluate candidates at all dilation levels (descending)

```python
@dataclass
class PairRow:
    tile_x: int
    tile_y: int
    r_um: float           # Dilation radius
    a: int                # Channel A index
    b: int                # Channel B index
    a_vox: int            # Voxels in A
    b_vox: int            # Voxels in B
    inter_vox: int        # Intersection voxels
    union_vox: int        # Union voxels
    iou: float            # inter_vox / union_vox
    overlap_coeff: float  # inter_vox / min(a_vox, b_vox)

@dataclass
class SetRow:
    # Similar structure for k>2 combinations
    members: Tuple[int, ...]  # Channel indices
    k: int                    # Number of channels
    ...
```

**Mining Parameters**:

| Parameter | Description | Typical Value |
|-----------|-------------|---------------|
| `min_marker_vox` | Min voxels for channel to be "active" | 500-1000 |
| `min_support_pair` | Min intersection for pair to be "frequent" | 100 |
| `min_support_set` | Min intersection for k-set (k≥3) | 50 |
| `max_set_size` | Maximum k for k-sets | 4 |
| `aggressive_stop_on_fail` | Stop at first radius where pair fails | True |

---

## Data Structures

### Configuration

```python
@dataclass
class VoxelSizeUM:
    x: float  # X spacing in microns
    y: float  # Y spacing in microns
    z: float  # Z spacing in microns
    
    @property
    def sampling_zyx(self) -> Tuple[float, float, float]:
        return (self.z, self.y, self.x)
    
    @property
    def voxel_volume_um3(self) -> float:
        return self.x * self.y * self.z

@dataclass
class PipelineConfig:
    # Data source
    zarr_url: str | None = None
    zarr_path: str | None = None
    channels: Sequence[int] = ()
    
    # Tiling
    tile_xy: Tuple[int, int] = (128, 128)
    channel_batch: int = 8
    
    # Thresholding
    alpha: float = 0.4
    trim_q: float = 0.98
    
    # Segmentation
    voxel_size_um: VoxelSizeUM = VoxelSizeUM(0.14, 0.14, 0.28)
    min_obj_vol_um3: float = 1.0
    connectivity: int = 26
    
    # Dilation
    dilate_um: Sequence[float] = (0.0, 1.0, 2.0, 3.0)
    
    # Overlap mining
    max_set_size: int = 4
    min_marker_vox: Dict[float, int] | int = 1000
    min_support_pair: Dict[float, int] | int = 100
    min_support_set: Dict[float, int] | int = 50
    aggressive_stop_on_fail: bool = True
    
    # Aggregation
    hierarchy_levels: int = 4
    
    # Output
    output_dir: str = "results"
    output_name: str = "analysis"
    checkpoint_dir: str = "checkpoints"
```

### Tile Result

```python
@dataclass
class OverlapTileResult:
    tile_x: int                              # Tile X index
    tile_y: int                              # Tile Y index
    tile_z: int                              # Always 0 (full Z processed)
    tile_shape: Tuple[int, int, int]         # (Z, Y, X) in voxels
    total_voxels: int                        # Z × Y × X
    radii_um: List[float]                    # Dilation radii processed
    marker_vox: Dict[float, Dict[int, int]]  # {radius: {channel: voxel_count}}
    channel_stats: List[ChannelTileStats]    # Per-channel statistics
    pairs: List[PairRow]                     # Pairwise overlaps
    sets: List[SetRow]                       # Higher-order overlaps
    n_active_channels: int                   # Channels passing min_marker_vox
    n_frequent_pairs: int                    # Pairs passing min_support
```

---

## Database Schema

### File Format

`.bioset` = gzip-compressed SQLite database

```python
# Reading
import gzip, sqlite3

with gzip.open("analysis.bioset", 'rb') as f:
    db_bytes = f.read()
with open("/tmp/analysis.db", 'wb') as f:
    f.write(db_bytes)
conn = sqlite3.connect("/tmp/analysis.db")
```

### Tables

#### 1. `metadata`

Global configuration stored as JSON strings.

```sql
CREATE TABLE metadata (
    key TEXT PRIMARY KEY,
    value TEXT  -- JSON encoded
);
```

| Key | Type | Example |
|-----|------|---------|
| `channels` | string[] | `["CD8", "MART1", "Hoechst"]` |
| `hierarchy_levels` | object[] | `[{level: 0, tile_size_x: 128, ...}]` |
| `dilation_amounts` | number[] | `[0.0, 0.5, 1.0, 1.5, 2.0]` |
| `volume_bounds` | object | `{x: [0, 10881], y: [0, 9585], z: [0, 194]}` |

#### 2. `channel_stats`

Per-channel, per-tile voxel statistics.

```sql
CREATE TABLE channel_stats (
    channel TEXT,           -- Channel name
    channel_idx INTEGER,    -- Channel index (0-based)
    dilation REAL,          -- Dilation radius in microns
    hierarchy_level INTEGER,-- 0 = finest, 3 = coarsest
    tile_x0 INTEGER,        -- Tile X start INDEX (not voxel!)
    tile_x1 INTEGER,        -- Tile X end INDEX
    tile_y0 INTEGER,        -- Tile Y start INDEX
    tile_y1 INTEGER,        -- Tile Y end INDEX
    voxel_count INTEGER,    -- Positive voxels in tile
    sum_intensity REAL,     -- Sum of intensity values
    mean_intensity REAL     -- voxel_count > 0 ? sum/count : 0
);
```

**Important**: `tile_x0/y0` are **tile indices**, not voxel coordinates!

```
voxel_x = tile_x0 × tile_size[hierarchy_level]
tile_size = {0: 128, 1: 256, 2: 512, 3: 1024}
```

#### 3. `combinations`

Overlap statistics for channel pairs and sets.

```sql
CREATE TABLE combinations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    channels TEXT,          -- Pipe-separated names: "CD8|MART1"
    channel_count INTEGER,  -- 2 for pairs, 3+ for sets
    dilation REAL,          -- Dilation radius
    hierarchy_level INTEGER,
    total_count INTEGER,    -- Intersection voxel count
    total_union INTEGER,    -- Union voxel count
    iou REAL,               -- total_count / total_union
    overlap_coeff REAL      -- total_count / min(channel_voxels)
);
```

**Channel ordering**: Channels are sorted by their index in `metadata.channels`:
```python
# If metadata.channels = ["CD8", "MART1", "Hoechst"]
# Then CD8 (idx 0) + MART1 (idx 1) → "CD8|MART1"
# And MART1 + Hoechst → "Hoechst|MART1" (sorted by idx: 1, 2)
```

#### 4. `tiles`

Spatial distribution linked to combinations.

```sql
CREATE TABLE tiles (
    combination_id INTEGER, -- FK to combinations.id
    tile_x0 INTEGER,
    tile_x1 INTEGER,
    tile_y0 INTEGER,
    tile_y1 INTEGER,
    inter_count INTEGER,    -- Intersection in this tile
    union_count INTEGER,    -- Union in this tile
    FOREIGN KEY (combination_id) REFERENCES combinations(id)
);
```

**Critical**: The schema has a **1:1 relationship** between `combinations` and `tiles`. Each row in `combinations` represents one specific tile, not an aggregated global value.

### Indices

```sql
CREATE INDEX idx_combinations_channels ON combinations(channels);
CREATE INDEX idx_combinations_dilation ON combinations(dilation);
CREATE INDEX idx_combinations_level ON combinations(hierarchy_level);
CREATE INDEX idx_combinations_dilation_level ON combinations(dilation, hierarchy_level);
CREATE INDEX idx_tiles_combination ON tiles(combination_id);
CREATE INDEX idx_tiles_spatial ON tiles(tile_x0, tile_y0);
CREATE INDEX idx_channel_stats_channel ON channel_stats(channel);
CREATE INDEX idx_channel_stats_level ON channel_stats(hierarchy_level);
```

---

## Checkpointing System

### Purpose

Enable resumable processing for long-running jobs. Each tile result is saved immediately after GPU processing.

### File Format

```
checkpoints/
└── {output_name}/
    ├── tile_0000_0000.json.gz
    ├── tile_0001_0000.json.gz
    ├── tile_0002_0000.json.gz
    └── ...
```

Each file is gzip-compressed JSON containing `OverlapTileResult`.

### API

```python
# Save single tile
save_tile_checkpoint(checkpoint_dir, result: OverlapTileResult) -> Path

# Check completed tiles
get_completed_tiles(checkpoint_dir) -> Set[Tuple[int, int]]

# Load all checkpoints
load_all_checkpoints(checkpoint_dir) -> List[OverlapTileResult]

# Get statistics
get_checkpoint_stats(checkpoint_dir) -> Dict
```

### Resume Logic

```python
def run_tile_processing(self, resume: bool = True):
    completed = get_completed_tiles(checkpoint_dir) if resume else set()
    
    for tile in iter_tiles_xy(...):
        if (tile.tx, tile.ty) in completed:
            continue  # Skip already processed
        
        result = self._process_single_tile(tile, ...)
        save_tile_checkpoint(checkpoint_dir, result)
```

---

## Hierarchical Aggregation

### Concept

Aggregate tile-level statistics into multi-scale spatial regions:

```
Level 0: 128×128 tiles (finest - base resolution)
Level 1: 256×256 regions (2×2 tiles aggregated)
Level 2: 512×512 regions (4×4 tiles aggregated)  
Level 3: 1024×1024 regions (8×8 tiles aggregated)
```

### Aggregation Rules

**Channel stats**: Sum voxel counts and intensities
```python
for tile in region_tiles:
    agg["voxel_count"] += tile.voxel_count
    agg["sum_intensity"] += tile.sum_intensity
```

**Pairs/Sets**: Sum intersection and union counts (IoU recomputed)
```python
for tile in region_tiles:
    agg["inter_vox"] += tile.inter_vox
    agg["union_vox"] += tile.union_vox

# Recompute IoU from aggregated values
iou = agg["inter_vox"] / agg["union_vox"]
```

### Class

```python
class HierarchicalAggregator:
    def __init__(self, base_tile_y: int, base_tile_x: int, n_levels: int = 4):
        """
        base_tile_y/x: Base tile size (e.g., 128)
        n_levels: Number of hierarchy levels
        """
    
    def add_tile_result(self, result: OverlapTileResult) -> None:
        """Add a tile result to the aggregator."""
    
    def aggregate(self) -> List[HierarchyLevel]:
        """Compute all hierarchy levels."""
```

---

## Query Patterns

### Get Top Combinations by IoU (Aggregated)

```sql
SELECT 
    channels,
    SUM(total_count) as sum_inter,
    SUM(total_union) as sum_union,
    CAST(SUM(total_count) AS REAL) / SUM(total_union) as global_iou
FROM combinations
WHERE dilation = 2.0 AND hierarchy_level = 0 AND channel_count >= 2
GROUP BY channels
HAVING sum_union > 0
ORDER BY global_iou DESC
LIMIT 50;
```

### Get Channel Voxel Totals

```sql
SELECT channel, SUM(voxel_count) as total
FROM channel_stats
WHERE dilation = 0.0 AND hierarchy_level = 0
GROUP BY channel
ORDER BY total DESC;
```

### Get Tiles for Heatmap (Single Channel)

```sql
SELECT tile_x0, tile_y0, voxel_count
FROM channel_stats
WHERE channel = 'CD8' AND dilation = 0.0 AND hierarchy_level = 0;
```

### Get Tiles for Heatmap (Channel Pair)

```sql
SELECT t.tile_x0, t.tile_y0, t.inter_count
FROM combinations c
JOIN tiles t ON c.id = t.combination_id
WHERE c.channels = 'CD8|MART1' 
  AND c.dilation = 2.0 
  AND c.hierarchy_level = 0;
```

### Filter Self-Pairs

Self-pairs (e.g., `Hoechst|Hoechst`) may exist due to duplicate channel names:

```python
def has_unique_channels(channels_str: str) -> bool:
    parts = channels_str.split("|")
    return len(parts) == len(set(parts))
```

### Dilation Curve for a Combination

```sql
SELECT 
    dilation,
    SUM(total_count) as sum_inter,
    SUM(total_union) as sum_union,
    CAST(SUM(total_count) AS REAL) / SUM(total_union) as iou
FROM combinations
WHERE channels = 'CD8|MART1' AND hierarchy_level = 0
GROUP BY dilation
ORDER BY dilation;
```

---

## Extending the Pipeline

### Adding a New Stage

1. Create a new module in `stages/`:

```python
# stages/my_stage.py
from dataclasses import dataclass
import cupy as cp

@dataclass
class MyStageResult:
    # Output fields
    pass

class MyStage:
    def __init__(self, param1: float, param2: int):
        self.param1 = param1
        self.param2 = param2
    
    def __call__(self, input_data: cp.ndarray) -> MyStageResult:
        # GPU processing
        return MyStageResult(...)
```

2. Add configuration parameters to `PipelineConfig`

3. Initialize in `Pipeline.__init__()`

4. Call in `Pipeline._process_single_tile()`

### Adding Database Fields

1. Modify `writer.py`:

```python
def _setup_schema(self):
    cur.execute("""
        CREATE TABLE my_new_table (
            id INTEGER PRIMARY KEY,
            new_field TEXT,
            ...
        )
    """)

def write_hierarchy_level(self, level):
    # Add INSERT statements for new table
```

2. Update `OverlapTileResult` and `HierarchyLevel` dataclasses

3. Update aggregation logic in `aggregation.py`

### Custom Thresholding

Replace `AlphaThreshold` with your own class:

```python
class MyThreshold:
    def compute_global(self, vol: np.ndarray) -> float:
        # Your global threshold logic
        return threshold
    
    def compute_tile_gpu(self, vol: cp.ndarray, t_global: float) -> ThresholdStats:
        # Your per-tile logic
        return ThresholdStats(...)
    
    def apply_gpu(self, vol: cp.ndarray, t_final: float) -> cp.ndarray:
        return vol > t_final  # Or your logic
```

---

## Performance Considerations

### GPU Memory Management

```python
# In _process_single_tile():
del vol_gpu                              # Free volume immediately
del masks                                # Free masks after use
cp.get_default_memory_pool().free_all_blocks()  # Force cleanup
```

### Batch Processing

Process multiple channels per GPU transfer:

```python
for ch_batch in chunked(channels, channel_batch):
    vol_cpu = A[0, ch_batch, :, ys, xs].compute()  # Single I/O
    vol_gpu = cp.asarray(vol_cpu)                   # Single transfer
    
    for i, ch in enumerate(ch_batch):
        v = vol_gpu[i]  # Slice already on GPU
        # Process channel
```

### I/O Optimization

1. **Local zarr**: 10-100× faster than remote URL
2. **Chunk alignment**: Match tile size to zarr chunks
3. **Rechunking**: Convert to (1, C, Z, tile_y, tile_x) chunks

```python
# Optimal chunk configuration
zarr.open(store, chunks=(1, 1, 194, 128, 128))  # Match tile_xy
```

---

## Troubleshooting Guide

| Symptom | Cause | Solution |
|---------|-------|----------|
| `OutOfMemoryError` | GPU memory exhausted | Reduce `channel_batch` or `tile_xy` |
| Slow tile processing | Network I/O | Use local zarr with SSD |
| `TypeError: dict_keys` | Non-list passed to writer | Convert to `list()` |
| Empty combinations | Thresholds too strict | Lower `min_marker_vox`, `min_support_*` |
| Self-pairs in output | Duplicate channel names | Filter in queries or clean database |
| Inconsistent IoU across levels | Bug in aggregation | Verify SUM(inter)/SUM(union) is used |

