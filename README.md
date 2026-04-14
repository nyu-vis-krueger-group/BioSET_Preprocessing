<p align='center'>
  <img src="assets\icon.jpg" width=150 />
</p>


<h1 align='center'>
  BioSET
  Preprocessing
</h1>
GPU-accelerated spatial overlap analysis pipeline for multi-channel biomedical imaging data. Processes large-scale 3D tissue volumes to identify and quantify co-localization patterns between biomarkers.

## Features

- **GPU-accelerated** processing using CuPy for threshold, segmentation, and overlap computation
- **Resumable checkpointing** - interrupt and resume long-running jobs
- **Hierarchical aggregation** - multi-scale spatial analysis
- **Portable output** - compressed SQLite database (`.bioset` format)

## Requirements

- Python 3.10+
- NVIDIA GPU with CUDA support
- 16GB+ RAM recommended for large datasets

## Installation

```bash
# Clone repository
git clone https://github.com/nyu-vis-krueger-group/BioSET_Preprocessing.git
cd BioSET_Preprocessing

# Create conda environment (recommended)
conda create -n bioset python=3.11
conda activate bioset

# Install package
pip install -e .
```

### Dependencies

```
cupy
numpy
scipy
zarr
dask
requests
ome-types
```

---

## Quick Start

This example shows a real setup using the BiomedVis Challenge 2025 melanoma dataset. We fetch OME-XML to extract channel names, build a name to index mapping (skipping channels marked as "do not use"), read physical voxel size, and configure the pipeline.

> **Tip**: Use `zarr_path` for local scratch/SSD runs (recommended). An example on how to download the remote dataset can be found at [```examples\rechunking.py```](https://github.com/nyu-vis-krueger-group/BioSET_Preprocessing/blob/master/examples/rechunking.py). Due to network latency when fetching from remote databases, the process becomes IO bound, thus only use `zarr_url` for testing.

```python
from bioset_preprocessing.config import PipelineConfig, VoxelSizeUM
from bioset_preprocessing.pipeline import Pipeline

import requests
import ome_types

# 1) Fetch OME metadata (channel names, physical voxel size)
OME_URL = "https://lsp-public-data.s3.amazonaws.com/biomedvis-challenge-2025/Dataset1-LSP13626-melanoma-in-situ/OME/METADATA.ome.xml"
response = requests.get(OME_URL, timeout=60)
response.raise_for_status()

# Some OME exports contain stray encoding artifacts; strip a common one if present
ome_xml = ome_types.from_xml(response.text.replace("Â", ""))

channel_names_all = [c.name for c in ome_xml.images[0].pixels.channels]

# 2) Build a unique list of usable channels (skip "do not use")
cn = {}
for idx, name in enumerate(channel_names_all):
    if name not in cn and "do not use" not in name.lower():
        cn[name] = idx

CHANNEL_NAMES = list(cn.keys())
CHANNEL_INDICES = list(cn.values())

num_channels = len(CHANNEL_INDICES)
channels = list(CHANNEL_INDICES)

# 3) Physical voxel size (µm) from OME metadata
voxel_physical_dims = VoxelSizeUM(
    ome_xml.images[0].pixels.physical_size_x,
    ome_xml.images[0].pixels.physical_size_y,
    ome_xml.images[0].pixels.physical_size_z,
)

print("Num channels:", num_channels)
print("Example channel names:", CHANNEL_NAMES[:10])
print("Voxel size (µm):", voxel_physical_dims)

# 4) Configure pipeline
cfg = PipelineConfig(
    # Local is preferred when available
    zarr_path="/data/rechunked.zarr",
    # Remote URL optional (useful for quick tests)
    zarr_url="https://lsp-public-data.s3.amazonaws.com/biomedvis-challenge-2025/Dataset1-LSP13626-melanoma-in-situ/0",

    channels=channels,
    tile_xy=(128, 128),

    # Segmentation / thresholding
    alpha=0.4,
    trim_q=0.98,
    voxel_size_um=voxel_physical_dims,
    min_obj_vol_um3=1.0,

    # Morphology / proximity
    dilate_um=(0, 0.5, 1.0, 1.5, 2.0),

    # GPU batching
    channel_batch=num_channels,

    # Overlap mining
    max_set_size=4,
    min_marker_vox=500,
    min_support_pair=100,
    min_support_set=50,

    # Hierarchy
    hierarchy_levels=4,

    # Output / checkpoints
    output_dir="results",
    output_name="melanoma_in_situ_unique",
    checkpoint_dir="checkpoints",
)

# 5) Run pipeline
pipe = Pipeline(cfg)
output_path = pipe.run_full_analysis(channel_names=CHANNEL_NAMES, resume=True)
print("Output:", output_path)
```

---

## How Co-localization is Computed

BioSET computes biomarker co-localization by processing the volume in spatial tiles (e.g., 128×128 in XY), while taking the full Z for each tile (these tissue volumes are relatively thin along Z).

For each `(tile_x, tile_y)` and each channel:

### 1. Alpha Thresholding

We threshold the intensity volume to produce a binary channel-tile mask using an adaptive method:
- Compute global threshold from low-resolution MIP using median + MAD
- Compute per-tile threshold from local background
- Use `max(global, local)` to prevent under-thresholding

### 2. Connected Components Filtering

We compute connected components in the binary mask and filter out small objects using `min_obj_vol_um3` (converted to voxels via `voxel_size_um`).

### 3. Multi-radius Dilations

To support proximity-based co-localization, we compute the Euclidean Distance Transform (EDT) once and then threshold it at each requested dilation radius (in µm). This yields multiple "expanded" versions of the mask efficiently.

![Pan Cytokeratin Stages](assets/pan_cytokeratin_stages.png)

### 4. Set Overlap Mining 

Within the tile, we compute overlap statistics for all channel combinations up to `max_set_size` (e.g., up to 4-way overlaps).

Higher-order sets are only evaluated if supporting lower-order masks are sufficiently large:
- A marker must have at least `min_marker_vox` voxels
- Pairs/sets must exceed `min_support_pair` / `min_support_set` voxels

**Metrics computed:**
- **IoU** (Intersection over Union): `intersection / union`
- **Overlap Coefficient**: `intersection / min(channel_a, channel_b)`

---

## Two-Stage Processing

For HPC environments, separate GPU and CPU stages for better resource allocation.

### Stage 1: GPU Tile Processing (GPU node)

```python
from bioset_preprocessing.pipeline import Pipeline

pipe = Pipeline(cfg)

# Per-tile segmentation, dilation masks, checkpoint writing
# Saves after each tile - fully resumable
pipe.run_tile_processing(resume=True)
```

### Stage 2: CPU Aggregation (CPU node)

```python
# Aggregate checkpoints into the .bioset SQLite database
pipe.run_aggregation(channel_names=CHANNEL_NAMES)
```

This separation allows you to:
- Run Stage 1 on expensive GPU nodes with time limits
- Run Stage 2 on cheaper CPU nodes with more memory
- Resume from any interruption point

---

## Configuration Reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `zarr_path` | str | None | Local zarr path (preferred for speed) |
| `zarr_url` | str | None | Remote zarr URL (convenient for testing) |
| `channels` | list[int] | () | Channel indices to process |
| `tile_xy` | tuple | (128, 128) | Tile size (y, x) in pixels |
| `channel_batch` | int | 8 | Channels per GPU batch |
| `alpha` | float | 0.4 | Threshold aggressiveness (higher = more selective) |
| `trim_q` | float | 0.98 | Background trimming quantile |
| `voxel_size_um` | VoxelSizeUM | (0.14, 0.14, 0.28) | Physical voxel size (x, y, z) |
| `min_obj_vol_um3` | float | 1.0 | Minimum object volume (µm³) |
| `connectivity` | int | 26 | CC connectivity (6 or 26) |
| `dilate_um` | tuple | (0, 1, 2, 3) | Dilation radii in microns |
| `max_set_size` | int | 4 | Maximum combination size (2=pairs only) |
| `min_marker_vox` | int/dict | 1000 | Min voxels per marker to be "active" |
| `min_support_pair` | int/dict | 100 | Min intersection voxels for pairs |
| `min_support_set` | int/dict | 50 | Min intersection voxels for sets |
| `hierarchy_levels` | int | 4 | Number of spatial aggregation levels |
| `output_dir` | str | "results" | Output directory |
| `output_name` | str | "analysis" | Output filename (without .bioset) |
| `checkpoint_dir` | str | "checkpoints" | Checkpoint directory |

---

## Choosing Good Parameter Values

A practical approach: start conservative, run on a small ROI (a few tiles), inspect mask quality + top overlaps, then scale up.

### Thresholding / Background Trimming

**`trim_q`** (default ~0.98)
- Use to suppress extreme bright artifacts/background tails before thresholding
- If you see lots of salt-and-pepper positives in background → **increase** (e.g., 0.985–0.995)
- If you are clipping real signal (structures look "eroded") → **decrease** (e.g., 0.95–0.98)

**`alpha`** (default ~0.4)
- Controls threshold aggressiveness. Tune by checking representative markers:
- Too many false positives → **increase** alpha (more selective)
- Missing obvious structures → **decrease** alpha

### Connected-Component Filtering

**`min_obj_vol_um3`**
- Tie this to expected smallest meaningful object
- If markers label cells/nuclei: pick something like a small fraction of a nucleus volume (~1-5 µm³)
- If marker is punctate: keep it lower, but rely on `min_marker_vox` to block tiny noise

### Dilation Radii

**`dilate_um = (0, 0.5, 1.0, 1.5, 2.0)`** (good starter)

Choose based on "interaction distance" you want to count as co-localization:
- **~0–0.5 µm**: Near-direct contact / true overlap
- **~1–2 µm**: Near-neighborhood proximity (e.g., cell-cell contact)

### Overlap Mining Thresholds (Most Important for Runtime)

**`min_marker_vox`** (per-marker gate)
- Set high enough to remove tiny noisy masks
- Heuristic: Start with 500–2000 voxels depending on tile size and voxel spacing
- If missing rare-but-real markers → lower it (use dict overrides for specific radii)

**`min_support_pair` / `min_support_set`**
- Controls how much overlap must exist to keep a combination
- Too many meaningless overlaps → **raise** these
- Higher-order combos never appear → **lower** `min_support_set`, but keep `min_marker_vox` sane

**`max_set_size`**
- `2` = pairs only (cheap)
- `3–4` = usually reasonable
- `≥4` can explode combinatorially unless thresholds are tight

### Performance Knobs

**`tile_xy`**
- 128×128 is a solid default (especially if zarr is rechunked to match)
- Smaller tiles → more overhead/checkpoints
- Larger tiles → more GPU memory, slower random I/O

**`channel_batch`**
- Set as large as your GPU memory allows
- Often "all channels" is fastest (single I/O + transfer)

---

## Output Format

The pipeline produces a `.bioset` file (gzipped SQLite) with four tables:

| Table | Description |
|-------|-------------|
| `metadata` | Channels, dilations, hierarchy levels, volume bounds (JSON) |
| `channel_stats` | Per-channel per-tile voxel counts and intensities |
| `combinations` | Pairwise and multi-channel overlap metrics (IoU, overlap coefficient) |
| `tiles` | Spatial tile coordinates for each combination |

### Reading Output

```python
import gzip
import sqlite3
import json

# Decompress and open
with gzip.open("analysis.bioset", 'rb') as f:
    with open("/tmp/analysis.db", 'wb') as out:
        out.write(f.read())

conn = sqlite3.connect("/tmp/analysis.db")
conn.row_factory = sqlite3.Row

# Get channels
cursor = conn.execute("SELECT value FROM metadata WHERE key = 'channels'")
channels = json.loads(cursor.fetchone()[0])

# Get top combinations by IoU (aggregated across tiles)
cursor = conn.execute('''
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
    LIMIT 20
''')

for row in cursor:
    print(f"{row['channels']}: IoU={row['global_iou']:.4f}")
```

### Database Schema Quick Reference

```sql
-- Metadata (JSON values)
SELECT key, value FROM metadata;
-- Keys: channels, hierarchy_levels, dilation_amounts, volume_bounds

-- Channel voxel totals
SELECT channel, SUM(voxel_count) as total
FROM channel_stats
WHERE dilation = 0.0 AND hierarchy_level = 0
GROUP BY channel
ORDER BY total DESC;

-- Tiles for heatmap (single channel)
SELECT tile_x0, tile_y0, voxel_count
FROM channel_stats
WHERE channel = 'CD8' AND dilation = 0.0 AND hierarchy_level = 0;

-- Tiles for heatmap (channel pair)
SELECT t.tile_x0, t.tile_y0, t.inter_count
FROM combinations c
JOIN tiles t ON c.id = t.combination_id
WHERE c.channels = 'CD8|MART1' AND c.dilation = 2.0 AND c.hierarchy_level = 0;
```

**Note**: `tile_x0/y0` are **tile indices**, not voxel coordinates. Multiply by tile size:
```python
tile_size = {0: 128, 1: 256, 2: 512, 3: 1024}  # per hierarchy level
voxel_x = tile_x0 * tile_size[hierarchy_level]
```

---

## Visualizing Output
The output can be visualized along with a multi-resolution volume rendering of the data by loading the data using the [BioSET Visualizer]().

---

## Performance Tips

1. **Use local zarr**: Copy data to local SSD/scratch for 10-100x faster I/O
2. **Rechunk data**: Align chunks with tile size (128×128) for optimal access patterns
3. **Batch channels**: Set `channel_batch` equal to total channels if GPU memory allows
4. **Monitor checkpoints**: Each tile saves immediately, enabling resume after crashes

### Rechunking Example

A rechunking script is available at [examples\rechunking.py](https://github.com/nyu-vis-krueger-group/BioSET_Preprocessing/blob/master/examples/rechunking.py).
```python
import zarr
import dask.array as da

# Load remote/original zarr
src = zarr.open("https://...")
arr = da.from_zarr(src)

# Rechunk to match tile size
rechunked = arr.rechunk((1, 1, -1, 128, 128))  # (T, C, Z, tile_y, tile_x)

# Save locally
rechunked.to_zarr("rechunked.zarr", overwrite=True)
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `OutOfMemoryError` | Reduce `channel_batch` or `tile_xy` |
| Slow processing | Use local zarr instead of remote URL |
| Missing checkpoints | Check `checkpoint_dir` path and permissions |
| JSON serialization error | Ensure `channel_names` is a `list()`, not `dict.keys()` |
| Empty combinations | Lower `min_marker_vox` or `min_support_*` thresholds |
| Too many meaningless overlaps | Raise `min_support_pair` / `min_support_set` |
| Self-pairs in output (e.g., CD8\|CD8) | Filter in queries or deduplicate channel names |

---

## License

MIT License - see [LICENSE](LICENSE) for details.
