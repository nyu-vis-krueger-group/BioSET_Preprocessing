from bioset_preprocessing.config import PipelineConfig, VoxelSizeUM
from bioset_preprocessing.pipeline import Pipeline
import requests
import ome_types

response = requests.get("https://lsp-public-data.s3.amazonaws.com/biomedvis-challenge-2025/Dataset1-LSP13626-melanoma-in-situ/OME/METADATA.ome.xml")
data = response.text
ome_xml = ome_types.from_xml(response.text.replace("Â",""))
CHANNEL_NAMES = [c.name for c in ome_xml.images[0].pixels.channels]

cfg = PipelineConfig(
    zarr_path="data/rechunk_mis_128_2048_5120.zarr",
    zarr_url="https://lsp-public-data.s3.amazonaws.com/biomedvis-challenge-2025/Dataset1-LSP13626-melanoma-in-situ/0",
    channels=list(range(70)),
    tile_xy=(128, 128),
    alpha=0.4,
    trim_q=0.98,
    voxel_size_um=VoxelSizeUM(0.14, 0.14, 0.28),
    min_obj_vol_um3=1.0,
    dilate_um=(0,0.5,1.0,1.5,2.0),
    channel_batch=70,
    # Overlap mining
    max_set_size=4,
    min_marker_vox=100,
    min_support_pair=50,
    min_support_set=10,
    # Hierarchy
    hierarchy_levels=4,
    # Output
    output_dir="results",
    output_name="melanoma_in_situ",
)

pipe = Pipeline(cfg)
output_path = pipe.run_full_analysis(channel_names=CHANNEL_NAMES)

print(f"\nAnalysis complete! File saved to: {output_path}")