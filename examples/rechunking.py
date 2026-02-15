import dask.array as da
import zarr
from rechunker import rechunk
import shutil, os
import time
from ome_zarr.io import parse_url

path = "https://lsp-public-data.s3.amazonaws.com/biomedvis-challenge-2025/Dataset1-LSP13626-melanoma-in-situ/0"
root = parse_url(path, mode="r")
store = root.store
print(zarr.__version__)

A = da.from_zarr(store, component="0")  # shape (1,70,194,5508,10908)
A_sub = A[:, :, :, 2048:4096, 4096:5120]

print(f"Source shape: {A_sub.shape}, dtype: {A_sub.dtype}, chunks: {A_sub.chunksize}")

target_path = "data/rechunk_mis_128_2048_5120.zarr"
temp_path   = "data/rechunk_tmp.zarr"

# Ensure parent directory exists
os.makedirs(os.path.dirname(target_path), exist_ok=True)

for p in [target_path, temp_path]:
    if os.path.exists(p):
        shutil.rmtree(p)

target_chunks = (1, 1, 194, 128, 128)
target_store = zarr.NestedDirectoryStore(target_path)
temp_store   = zarr.NestedDirectoryStore(temp_path)

plan = rechunk(
    A_sub,
    target_chunks=target_chunks,
    max_mem="4GB",          
    target_store=target_store,
    temp_store=temp_store,
)

print("Starting rechunk...")
start_time = time.perf_counter()

plan.execute()

elapsed = time.perf_counter() - start_time
print(f"Rechunk completed in {elapsed:.2f}s ({elapsed/60:.2f} min)")

# Cleanup temp
shutil.rmtree(temp_path, ignore_errors=True)

# Verify output
out = zarr.open(target_path, mode="r")
print(f"Output shape: {out.shape}, chunks: {out.chunks}")