import argparse
from .config import PipelineConfig, VoxelSizeUM
from .pipeline import Pipeline

def parse_int_list(s: str):
    return [int(x.strip()) for x in s.split(",") if x.strip() != ""]

def parse_tile(s: str):
    if "x" in s:
        a, b = s.lower().split("x")
        return (int(a), int(b))
    n = int(s)
    return (n, n)

def main():
    p = argparse.ArgumentParser(prog="bioset")
    sub = p.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="Run threshold->CC->dilation pipeline (no outputs written yet)")

    run.add_argument("--zarr-url", help="Remote OME-Zarr root (https://... or s3://...)")
    run.add_argument("--zarr-path", help="Local OME-Zarr root directory (e.g. /scratch/.../rechunked.zarr)")
    run.add_argument("--meta", required=True, help="OME-XML metadata URL")

    run.add_argument("--channels", required=True, help="Comma-separated channel indices, e.g. 0,3,39")
    run.add_argument("--tile", default="128", help="Tile size: N or NyxNx, e.g. 128 or 128x256")
    run.add_argument("--batch", type=int, default=8, help="Channels per batch on GPU")

    run.add_argument("--alpha", type=float, default=0.4)
    run.add_argument("--trim-q", type=float, default=0.98)
    run.add_argument("--min-vol-um3", type=float, default=1.0)
    run.add_argument("--conn", type=int, default=26)

    run.add_argument("--dilate-um", default="0,1,2,3")
    run.add_argument("--vox", default="0.14,0.14,0.28", help="Voxel size x,y,z in µm")
    run.add_argument("--float64-dist", action="store_true", help="Use float64 EDT distances (slower, more precise)")

    run.add_argument("--max-set-size", type=int, default=4)
    run.add_argument("--min-marker-vox", type=int, default=100)
    run.add_argument("--min-support-pair", type=int, default=50)
    run.add_argument("--min-support-set", type=int, default=10)
    run.add_argument("--hierarchy-levels", type=int, default=4)
    run.add_argument("--output-dir", default="results")
    run.add_argument("--output-name", default="analysis")

    args = p.parse_args()
    if not args.zarr_url and not args.zarr_path:
        run.error("Provide at least one of --zarr-url or --zarr-path")
    
    channels = parse_int_list(args.channels)
    tile_xy = parse_tile(args.tile)
    dilate_um = [float(x) for x in args.dilate_um.split(",") if x.strip() != ""]
    vx, vy, vz = [float(x) for x in args.vox.split(",")]

    cfg = PipelineConfig(
        zarr_url=args.zarr_url,          
        zarr_path=args.zarr_path,        
        metadata_url=args.meta,          
        channels=channels,
        tile_xy=tile_xy,
        channel_batch=args.batch,
        alpha=args.alpha,
        trim_q=args.trim_q,
        voxel_size_um=VoxelSizeUM(vx, vy, vz),
        min_obj_vol_um3=args.min_vol_um3,
        connectivity=args.conn,
        dilate_um=tuple(dilate_um),
        float64_distances=args.float64_dist,
        max_set_size=args.max_set_size,
        min_marker_vox=args.min_marker_vox,
        min_support_pair=args.min_support_pair,
        min_support_set=args.min_support_set,
        hierarchy_levels=args.hierarchy_levels,
        output_dir=args.output_dir,
        output_name=args.output_name,
    )

    pipe = Pipeline(cfg)
    output_path = pipe.run_full_analysis()
    print(f"Done! Output: {output_path}")
