# BioSET Preprocessing

A Python pipeline for processing large multi-channel volumetric microscopy data, computing thresholds, and analyzing channel co-localization.

## Features

- **Large volume support**: Process datasets that don't fit in memory using tiled processing
- **Multiple threshold methods**: Percentile, Otsu, mean+std, or custom methods
- **Channel overlap analysis**: Compute co-localization for all channel combinations
- **GPU acceleration**: Optional CuPy support for faster processing
- **Checkpoint/resume**: Automatically resume interrupted processing
- **Flexible I/O**: Load from local files or S3, save results as CSV or TIFF

## Installation

```bash
# Basic installation
pip install bioset

# With GPU support (CUDA 12.x)
pip install bioset[gpu]

# From source
git clone https://github.com/Chahat08/BioSET_Preprocessing.git
cd BioSET_Preprocessing
pip install -e .
```

## Quick Start

### In a Notebook

```python
from bioset_preprocessing import Pipeline, Config

# Simple usage with URL
pipeline = Pipeline.from_url(
    "https://example.com/data.zarr",
    component="2",  # Resolution level
)
results = pipeline.run(channels=[0, 1, 2])

# With config file
pipeline = Pipeline.from_config("config.yaml")
results = pipeline.run()

# Programmatic configuration
config = Config(
    zarr_url="https://example.com/data.zarr",
    channels=[0, 1, 2],
    threshold_method="otsu",
    output_dir="./my_results",
)
pipeline = Pipeline(config)
results = pipeline.run()
```

### From Command Line

```bash
# Using config file
bioset run --config config.yaml

# With arguments
bioset run \
    --url "https://example.com/data.zarr" \
    --channels 0 1 2 \
    --threshold otsu \
    --output ./results

# Show dataset info
bioset info --url "https://example.com/data.zarr"

# List available threshold methods
bioset methods
```

## Configuration

Create a `config.yaml` file (see `config.example.yaml`):

```yaml
zarr_url: "https://example.com/data.zarr"
zarr_component: "0"
channels: [0, 1, 2]
threshold_method: "percentile_95"
tile_size: null  # Auto-calculate
output_dir: "./results"
resume: true
```

## Threshold Methods

| Method | Description |
|--------|-------------|
| `percentile_95` | Top 5% brightest voxels |
| `percentile_90` | Top 10% brightest voxels |
| `percentile_99` | Top 1% brightest voxels |
| `otsu` | Automatic optimal threshold |
| `mean_2std` | Mean + 2 standard deviations |
| `mean_3std` | Mean + 3 standard deviations |

### Custom Threshold Methods

```python
from bioset_preprocessing import register_threshold_method, ThresholdResult
import numpy as np

def my_threshold(data):
    thresh = np.median(data) * 2
    mask = (data > thresh).astype(np.uint8)
    return ThresholdResult(
        mask=mask,
        threshold_value=thresh,
        active_voxels=int(np.sum(mask)),
        active_fraction=np.sum(mask) / mask.size,
        method="my_custom"
    )

register_threshold_method("my_custom", my_threshold)
```

## Output Files

The pipeline creates:

- `thresholds.csv`: Per-channel threshold values for each tile
- `overlaps.csv`: Channel combination overlap counts for each tile  
- `checkpoint.json`: Progress tracking for resume capability
- `*.tiff`: Binary masks (if `save_masks: true`)

## API Reference

### Pipeline Class

```python
from bioset_preprocessing import Pipeline

# Create from config file
pipeline = Pipeline.from_config("config.yaml")

# Create from URL
pipeline = Pipeline.from_url("https://...", component="0")

# Access data info
print(pipeline.array_info)
print(pipeline.tiling_scheme)

# Run processing
results = pipeline.run(channels=[0, 1, 2])
```

### Low-Level API

```python
from bioset_preprocessing import (
    load_zarr_array,
    create_tiling_scheme,
    apply_threshold,
    compute_overlaps,
)

# Load data
arr, info = load_zarr_array("https://...", component="0")

# Create tiling
scheme = create_tiling_scheme(info, tile_size=1024)

# Process a tile
for tile in scheme.iter_tiles():
    data = arr[0, channel, :, tile.y_slice, tile.x_slice].compute()
    result = apply_threshold(data, method="otsu")
    # ...
```

## License

MIT License
