from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import dask.array as da
from ome_zarr.io import parse_url  

@dataclass
class ZarrPyramid:
    url: str
    arrays: Dict[str, da.Array]  

    @staticmethod
    def open(url: str, components: Optional[List[str]] = None) -> "ZarrPyramid":
        root = parse_url(url, mode="r")
        store = root.store

        # If components not provided, try 0..6
        if components is None:
            components = [str(i) for i in range(7)]

        arrays: Dict[str, da.Array] = {}
        for c in components:
            try:
                arrays[c] = da.from_zarr(store, component=c)
            except Exception:
                pass

        if not arrays:
            raise RuntimeError(f"No readable zarr components found at {url}")
        return ZarrPyramid(url=url, arrays=arrays)

    def highest_res(self) -> Tuple[str, da.Array]:
        # assume "0" is highest if present; else pick max shape
        if "0" in self.arrays:
            return "0", self.arrays["0"]
        return max(self.arrays.items(), key=lambda kv: kv[1].size)

    def lowest_res(self) -> Tuple[str, da.Array]:
        # assume highest index is lowest; else pick min shape
        if "6" in self.arrays:
            return "6", self.arrays["6"]
        return min(self.arrays.items(), key=lambda kv: kv[1].size)

    def is_multiplexed(self) -> bool:
        _, A = self.highest_res()
        return A.shape[1] > 1
