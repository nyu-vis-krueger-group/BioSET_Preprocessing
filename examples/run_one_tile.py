from bioset_preprocessing.config import PipelineConfig, VoxelSizeUM
from bioset_preprocessing.pipeline import Pipeline
import cupy as cp

# ─── GPU Diagnostics ───────────────────────────────────────────────────────────
def print_gpu_info():
    try:
        device = cp.cuda.Device()
        props = cp.cuda.runtime.getDeviceProperties(device.id)
        print(f"GPU Device: {props['name'].decode()}")
        print(f"  Compute Capability: {props['major']}.{props['minor']}")
        print(f"  Total Memory: {props['totalGlobalMem'] / 1e9:.2f} GB")
        
        mempool = cp.get_default_memory_pool()
        print(f"  CuPy Memory Pool Used: {mempool.used_bytes() / 1e6:.2f} MB")
        print(f"  CuPy Memory Pool Total: {mempool.total_bytes() / 1e6:.2f} MB")
        
        # Quick test to confirm GPU computation works
        a = cp.arange(1000000, dtype=cp.float32)
        b = cp.sum(a)
        cp.cuda.Stream.null.synchronize()  # Force GPU sync
        print(f"  GPU Test (sum of 1M floats): {float(b):.0f} ✓")
    except Exception as e:
        print(f"GPU ERROR: {e}")

print_gpu_info()
print()

cfg = PipelineConfig(
    zarr_url="https://lsp-public-data.s3.amazonaws.com/biomedvis-challenge-2025/Dataset1-LSP13626-melanoma-in-situ/0",
    metadata_url="https://lsp-public-data.s3.amazonaws.com/biomedvis-challenge-2025/Dataset1-LSP13626-melanoma-in-situ/OME/METADATA.ome.xml",
    #channels=[0,3,39],
    channels=list(range(70)),
    tile_xy=(128,128),
    alpha=0.4,
    trim_q=0.98,
    voxel_size_um=VoxelSizeUM(0.14,0.14,0.28),
    min_obj_vol_um3=1.0,
    dilate_um=(0,1,2,3),
    channel_batch=70
)

pipe = Pipeline(cfg)
pipe.compute_global_thresholds()

# iterate a few outputs
for i, out in enumerate(pipe.iter_tile_outputs()):
    mempool = cp.get_default_memory_pool()
    gpu_mb = mempool.used_bytes() / 1e6
    print(f"[GPU: {gpu_mb:.1f}MB] ch={out.channel} tile={out.tile} t={out.threshold.t_final:.1f} cc={out.cc.n_components} masks={list(out.masks.keys())}")
    if i > 7:
        break
