from bioset_preprocessing.config import PipelineConfig, VoxelSizeUM
from bioset_preprocessing.pipeline import Pipeline
import cupy as cp

cfg = PipelineConfig(
    zarr_path="data/rechunk_mis_128_2048_5120.zarr",
    zarr_url="https://lsp-public-data.s3.amazonaws.com/biomedvis-challenge-2025/Dataset1-LSP13626-melanoma-in-situ/0",
    metadata_url="https://lsp-public-data.s3.amazonaws.com/biomedvis-challenge-2025/Dataset1-LSP13626-melanoma-in-situ/OME/METADATA.ome.xml",
    channels=list(range(70)),
    tile_xy=(128,128),
    alpha=0.4,
    trim_q=0.98,
    voxel_size_um=VoxelSizeUM(0.14,0.14,0.28),
    min_obj_vol_um3=1.0,
    dilate_um=(0,0.5,1.0,1.5,2.0),
    channel_batch=70,
    max_set_size=3,           
    min_marker_vox=100,        
    min_support_pair=50,      
    min_support_set=10,        
)

pipe = Pipeline(cfg)
pipe.compute_global_thresholds()

for i, out in enumerate(pipe.iter_tile_overlap_outputs()):
    print(f"Tile ({out.tile_x}, {out.tile_y}): "
          f"active_ch={out.n_active_channels}, "
          f"freq_pairs={out.n_frequent_pairs}, "
          f"pairs={len(out.pairs)}, "
          f"sets={len(out.sets)}")
    
    # Print first 3 pairs for this tile
    for pr in out.pairs[:3]:
        print(f"  Pair ch{pr.a}-ch{pr.b} @ r={pr.r_um}µm: IoU={pr.iou:.3f}, overlap_coeff={pr.overlap_coeff:.3f}")
    
    if i >= 5:  # Just process a few tiles for testing
        break

print("\nDone!")
