from dataclasses import dataclass
from typing import Sequence, Tuple, Union

@dataclass(frozen=True)
class VoxelSizeUM:
    x: float
    y: float
    z: float
    @property
    def sampling_zyx(self) -> Tuple[float, float, float]:
        return (self.z, self.y, self.x)
    @property
    def voxel_volume_um3(self) -> float:
        return self.x * self.y * self.z

@dataclass
class PipelineConfig:
    zarr_url: str
    metadata_url: str
    channels: Sequence[int]  

    tile_xy: Tuple[int, int] = (128, 128)   # (tile_y, tile_x)
    channel_batch: int = 8                  # how many channels per batch

    alpha: float = 0.4
    trim_q: float = 0.98

    voxel_size_um: VoxelSizeUM = VoxelSizeUM(0.14, 0.14, 0.28)
    min_obj_vol_um3: float = 1.0
    connectivity: int = 26

    dilate_um: Sequence[float] = (0.0, 1.0, 2.0, 3.0)

    prefer_lowest_res_for_global: bool = True
    float64_distances: bool = False      # for edt   
