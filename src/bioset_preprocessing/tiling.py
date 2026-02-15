from __future__ import annotations
from dataclasses import dataclass
from typing import Iterator, Tuple

@dataclass(frozen=True)
class TileIndex:
    tx: int
    ty: int

def iter_tiles_xy(shape_y: int, shape_x: int, tile_y: int, tile_x: int) -> Iterator[TileIndex]:
    nx = (shape_x + tile_x - 1) // tile_x
    ny = (shape_y + tile_y - 1) // tile_y
    for ty in range(ny):
        for tx in range(nx):
            yield TileIndex(tx=tx, ty=ty)

def tile_slices(tile: TileIndex, tile_y: int, tile_x: int) -> Tuple[slice, slice]:
    y0 = tile.ty * tile_y
    x0 = tile.tx * tile_x
    return (slice(y0, y0 + tile_y), slice(x0, x0 + tile_x))
