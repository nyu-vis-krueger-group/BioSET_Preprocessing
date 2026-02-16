from __future__ import annotations
from dataclasses import dataclass
import os
from typing import Dict, List, Optional, Tuple
import zarr
import dask.array as da
from ome_zarr.io import parse_url  

def _is_remote_location(loc: str) -> bool:
    return loc.startswith(("http://", "https://", "s3://", "gs://"))

def _open_store(location: str):
    if _is_remote_location(location):
        root = parse_url(location, mode="r")
        return root.store
    if not os.path.exists(location):
        raise FileNotFoundError(f"Local zarr path not found: {location}")
    return zarr.NestedDirectoryStore(location)  

@dataclass
class ZarrPyramid:
    location: str
    arrays: Dict[str, da.Array]  

    @staticmethod
    def open(location: str, components: Optional[List[str]] = None) -> "ZarrPyramid":
        store = _open_store(location)
        arrays: Dict[str, da.Array] = {}

        try:
            A_root = da.from_zarr(store)  
            arrays["root"] = A_root
            return ZarrPyramid(location=location, arrays=arrays)
        except Exception:
            pass

        # If components not provided, try 0..6
        if components is None:
            components = [str(i) for i in range(7)]

        for c in components:
            try:
                arrays[c] = da.from_zarr(store, component=c)
            except Exception:
                pass

        if not arrays:
            raise RuntimeError(f"No readable zarr components found at {location}")
        return ZarrPyramid(location=location, arrays=arrays)

    def highest_res(self) -> Tuple[str, da.Array]:
        # assume "0" is highest if present; else pick max shape
        if "0" in self.arrays:
            return "0", self.arrays["0"]
        return max(self.arrays.items(), key=lambda kv: kv[1].size)

    def lowest_res(self) -> Tuple[str, da.Array]:
        # For numeric keys, pick highest number (lowest resolution)
        numeric_keys = [k for k in self.arrays.keys() if k.isdigit()]
        if numeric_keys:
            key = max(numeric_keys, key=int)
            return key, self.arrays[key]
        # Fallback: pick smallest by size
        return min(self.arrays.items(), key=lambda kv: kv[1].size)

    def is_multiplexed(self) -> bool:
        _, A = self.highest_res()
        return A.shape[1] > 1
