# BioSET Preprocessing

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
git clone https://github.com/yourusername/BioSET_Preprocessing.git
cd BioSET_Preprocessing

# Create conda environment (recommended)
conda create -n bioset python=3.11
conda activate bioset

# Install CUDA-enabled packages
conda install -c conda-forge cupy cudatoolkit=12.0

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

## Quick Start

```python
from bioset_preprocessing import PipelineConfig, VoxelSizeUM, Pipeline

# Configure pipeline
cfg = PipelineConfig(
    zarr_path="/path/to/local/data.zarr",      # Local zarr (faster)
    # zarr_url="https://...",                  # OR remote URL
    channels=list(range(70)),                  # Channel indices to process
    tile_xy=(128, 128),                        # Tile size in pixels
    voxel_size_um=VoxelSizeUM(0.14, 0.14, 0.28),
    dilate_um=(0, 0.5, 1.0, 1.5, 2.0),         # Dilation radii in microns
    output_dir="results",
    output_name="my_analysis",
)

# Run pipeline
pipe = Pipeline(cfg)
output_path = pipe.run_full_analysis(channel_names=["CD8", "MART1", ...])

print(f"Output: {output_path}")  # results/my_analysis.bioset
```

## Two-Stage Processing (HPC)

For HPC environments, separate GPU and CPU stages:

### Stage 1: GPU Tile Processing (GPU node)

```python
pipe = Pipeline(cfg)
pipe.run_tile_processing(resume=True)  # Saves checkpoints after each tile
```

### Stage 2: CPU Aggregation (CPU node)

```python
pipe = Pipeline(cfg)
pipe.run_aggregation(channel_names=CHANNEL_NAMES)  # Loads checkpoints, writes database
```

## Configuration Reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `zarr_path` | str | None | Local zarr path (preferred) |
| `zarr_url` | str | None | Remote zarr URL |
| `channels` | list[int] | () | Channel indices to process |
| `tile_xy` | tuple | (128, 128) | Tile size (y, x) in pixels |
| `channel_batch` | int | 8 | Channels per GPU batch |
| `alpha` | float | 0.4 | Threshold percentile (1 - alpha) |
| `trim_q` | float | 0.98 | Background trimming quantile |
| `voxel_size_um` | VoxelSizeUM | (0.14, 0.14, 0.28) | Physical voxel size (x, y, z) |
| `min_obj_vol_um3` | float | 1.0 | Minimum object volume (µm³) |
| `connectivity` | int | 26 | CC connectivity (6 or 26) |
| `dilate_um` | tuple | (0, 1, 2, 3) | Dilation radii in microns |
| `max_set_size` | int | 4 | Maximum combination size |
| `min_marker_vox` | int/dict | 1000 | Min voxels per marker |
| `min_support_pair` | int/dict | 100 | Min intersection for pairs |
| `min_support_set` | int/dict | 50 | Min intersection for sets |
| `hierarchy_levels` | int | 4 | Number of aggregation levels |
| `output_dir` | str | "results" | Output directory |
| `output_name` | str | "analysis" | Output filename (without .bioset) |
| `checkpoint_dir` | str | "checkpoints" | Checkpoint directory |

## Output Format

The pipeline produces a `.bioset` file (gzipped SQLite) with four tables:

| Table | Description |
|-------|-------------|
| `metadata` | Channels, dilations, hierarchy levels, volume bounds |
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

# Get top combinations by IoU
cursor = conn.execute('''
    SELECT channels, iou, total_count
    FROM combinations
    WHERE dilation = 2.0 AND hierarchy_level = 0
    ORDER BY iou DESC
    LIMIT 20
''')
```

## Example: Melanoma Dataset

```python
import requests
import ome_types
from bioset_preprocessing import PipelineConfig, VoxelSizeUM, Pipeline

# Fetch channel names from OME metadata
response = requests.get("https://lsp-public-data.s3.amazonaws.com/.../METADATA.ome.xml")
ome_xml = ome_types.from_xml(response.text)
CHANNEL_NAMES = [c.name for c in ome_xml.images[0].pixels.channels]

cfg = PipelineConfig(
    zarr_path="/scratch/rechunked_full_128.zarr",
    channels=list(range(70)),
    tile_xy=(128, 128),
    voxel_size_um=VoxelSizeUM(0.14, 0.14, 0.28),
    dilate_um=(0, 0.5, 1.0, 1.5, 2.0),
    channel_batch=70,
    max_set_size=4,
    min_marker_vox=500,
    min_support_pair=100,
    hierarchy_levels=4,
    output_name="melanoma_in_situ",
    checkpoint_dir="/scratch/checkpoints",
)

pipe = Pipeline(cfg)
output = pipe.run_full_analysis(channel_names=CHANNEL_NAMES, resume=True)
```

## Performance Tips

1. **Use local zarr**: Copy data to local SSD/scratch for 10-100x faster I/O
2. **Rechunk data**: Align chunks with tile size (128×128) for optimal access
3. **Batch channels**: Set `channel_batch` equal to total channels if GPU memory allows
4. **Monitor checkpoints**: Each tile saves immediately, enabling resume after crashes

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `OutOfMemoryError` | Reduce `channel_batch` or `tile_xy` |
| Slow processing | Use local zarr instead of remote URL |
| Missing checkpoints | Check `checkpoint_dir` path and permissions |
| JSON serialization error | Ensure `channel_names` is a list, not dict_keys |

## License

MIT License - see [LICENSE](LICENSE) for details.
