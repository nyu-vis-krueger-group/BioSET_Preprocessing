#!/usr/bin/env python
"""
STAGE 2: CPU aggregation and database writing.

This script loads checkpointed tiles, aggregates them, and writes
the final .bioset database. No GPU required.

Run this on CPU-only nodes after GPU processing is complete.
"""
import time
import requests
import ome_types
from bioset_preprocessing.config import PipelineConfig, VoxelSizeUM
from bioset_preprocessing.pipeline import Pipeline

# Fetch channel names
print("Fetching channel metadata...")
response = requests.get(
    "https://lsp-public-data.s3.amazonaws.com/biomedvis-challenge-2025/"
    "Dataset1-LSP13626-melanoma-in-situ/OME/METADATA.ome.xml"
)
ome_xml = ome_types.from_xml(response.text.replace("Â", ""))
CHANNEL_NAMES = [c.name for c in ome_xml.images[0].pixels.channels]
print(f"Found {len(CHANNEL_NAMES)} channels")

cfg = PipelineConfig(
    zarr_path="/scratch/ck4106/rechunked_full_128.zarr",
    zarr_url="https://lsp-public-data.s3.amazonaws.com/biomedvis-challenge-2025/Dataset1-LSP13626-melanoma-in-situ/0",
    channels=list(range(len(CHANNEL_NAMES))),
    tile_xy=(128, 128),
    alpha=0.4,
    trim_q=0.98,
    voxel_size_um=VoxelSizeUM(0.14, 0.14, 0.28),
    min_obj_vol_um3=1.0,
    dilate_um=(0, 0.5, 1.0, 1.5, 2.0),
    channel_batch=len(CHANNEL_NAMES),
    max_set_size=4,
    min_marker_vox=100,
    min_support_pair=50,
    min_support_set=10,
    hierarchy_levels=4,
    output_dir="/scratch/ck4106/BioSET_Preprocessing/results",
    output_name="melanoma_in_situ",
    checkpoint_dir="/scratch/ck4106/BioSET_Preprocessing/checkpoints",
)

print("\n" + "="*60)
print("STAGE 2: CPU Aggregation & Database Writing")
print("="*60)

start_time = time.time()
pipe = Pipeline(cfg)
output_path = pipe.run_aggregation(channel_names=CHANNEL_NAMES)
elapsed = time.time() - start_time

print(f"\n{'='*60}")
print(f"CPU STAGE COMPLETE")
print(f"Output: {output_path}")
print(f"Time: {elapsed/60:.1f} minutes")
print(f"{'='*60}")