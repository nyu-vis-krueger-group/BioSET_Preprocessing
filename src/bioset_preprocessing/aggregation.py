from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Iterator
from collections import defaultdict

from .stages.overlaps import OverlapTileResult, PairRow, SetRow, ChannelTileStats


@dataclass
class AggregatedChannelStats:
    channel: int
    r_um: float
    hierarchy_level: int
    tile_x0: int
    tile_x1: int
    tile_y0: int
    tile_y1: int
    voxel_count: int
    sum_intensity: float
    
    @property
    def mean_intensity(self) -> float:
        return self.sum_intensity / self.voxel_count if self.voxel_count > 0 else 0.0


@dataclass
class AggregatedPair:
    a: int
    b: int
    r_um: float
    hierarchy_level: int
    tile_x0: int
    tile_x1: int
    tile_y0: int
    tile_y1: int
    a_vox: int
    b_vox: int
    inter_vox: int
    union_vox: int
    
    @property
    def iou(self) -> float:
        return self.inter_vox / self.union_vox if self.union_vox > 0 else 0.0
    
    @property
    def overlap_coeff(self) -> float:
        min_vox = min(self.a_vox, self.b_vox)
        return self.inter_vox / min_vox if min_vox > 0 else 0.0


@dataclass
class AggregatedSet:
    members: Tuple[int, ...]
    r_um: float
    hierarchy_level: int
    tile_x0: int
    tile_x1: int
    tile_y0: int
    tile_y1: int
    member_voxels: Dict[int, int] 
    inter_vox: int
    union_vox: int
    
    @property
    def k(self) -> int:
        return len(self.members)
    
    @property
    def iou(self) -> float:
        return self.inter_vox / self.union_vox if self.union_vox > 0 else 0.0
    
    @property
    def overlap_coeff(self) -> float:
        if not self.member_voxels:
            return 0.0
        min_vox = min(self.member_voxels.values())
        return self.inter_vox / min_vox if min_vox > 0 else 0.0


@dataclass
class HierarchyLevel:
    level: int
    tile_size_x: int
    tile_size_y: int
    channels: List[AggregatedChannelStats] = field(default_factory=list)
    pairs: List[AggregatedPair] = field(default_factory=list)
    sets: List[AggregatedSet] = field(default_factory=list)


class HierarchicalAggregator:
    def __init__(
        self,
        base_tile_y: int,
        base_tile_x: int,
        n_levels: int = 4,
    ):
        self.base_tile_y = base_tile_y
        self.base_tile_x = base_tile_x
        self.n_levels = n_levels
        
        self._tile_results: Dict[Tuple[int, int], OverlapTileResult] = {}
    
    def add_tile_result(self, result: OverlapTileResult) -> None:
        key = (result.tile_x, result.tile_y)
        self._tile_results[key] = result
    
    def aggregate(self) -> List[HierarchyLevel]:

        levels: List[HierarchyLevel] = []
        
        for level in range(self.n_levels):
            scale = 2 ** level  # 1, 2, 4, 8, ...
            tile_size_x = self.base_tile_x * scale
            tile_size_y = self.base_tile_y * scale
            
            hierarchy_level = HierarchyLevel(
                level=level,
                tile_size_x=tile_size_x,
                tile_size_y=tile_size_y,
            )
            
            regions: Dict[Tuple[int, int], List[OverlapTileResult]] = defaultdict(list)
            
            for (tx, ty), result in self._tile_results.items():
                region_x = tx // scale
                region_y = ty // scale
                regions[(region_x, region_y)].append(result)
            
            for (region_x, region_y), tile_results in regions.items():
                self._aggregate_region(
                    hierarchy_level,
                    level,
                    region_x,
                    region_y,
                    scale,
                    tile_results,
                )
            
            levels.append(hierarchy_level)
        
        return levels
    
    def _aggregate_region(
        self,
        hierarchy_level: HierarchyLevel,
        level: int,
        region_x: int,
        region_y: int,
        scale: int,
        tile_results: List[OverlapTileResult],
    ) -> None:
        if not tile_results:
            return
        
        tile_x0 = region_x * scale
        tile_x1 = tile_x0 + scale
        tile_y0 = region_y * scale
        tile_y1 = tile_y0 + scale
        
        radii = tile_results[0].radii_um
        
        channel_agg: Dict[Tuple[int, float], Dict] = defaultdict(
            lambda: {"voxel_count": 0, "sum_intensity": 0.0}
        )
        
        for result in tile_results:
            for cs in result.channel_stats:
                key = (cs.channel, cs.r_um)
                channel_agg[key]["voxel_count"] += cs.voxel_count
                channel_agg[key]["sum_intensity"] += cs.sum_intensity
        
        for (ch, r), agg in channel_agg.items():
            hierarchy_level.channels.append(AggregatedChannelStats(
                channel=ch,
                r_um=r,
                hierarchy_level=level,
                tile_x0=tile_x0,
                tile_x1=tile_x1,
                tile_y0=tile_y0,
                tile_y1=tile_y1,
                voxel_count=agg["voxel_count"],
                sum_intensity=agg["sum_intensity"],
            ))
        
        pair_agg: Dict[Tuple[int, int, float], Dict] = defaultdict(
            lambda: {"a_vox": 0, "b_vox": 0, "inter_vox": 0, "union_vox": 0}
        )
        
        for result in tile_results:
            for pr in result.pairs:
                key = (pr.a, pr.b, pr.r_um)
                pair_agg[key]["a_vox"] += pr.a_vox
                pair_agg[key]["b_vox"] += pr.b_vox
                pair_agg[key]["inter_vox"] += pr.inter_vox
                pair_agg[key]["union_vox"] += pr.union_vox
        
        for (a, b, r), agg in pair_agg.items():
            hierarchy_level.pairs.append(AggregatedPair(
                a=a,
                b=b,
                r_um=r,
                hierarchy_level=level,
                tile_x0=tile_x0,
                tile_x1=tile_x1,
                tile_y0=tile_y0,
                tile_y1=tile_y1,
                a_vox=agg["a_vox"],
                b_vox=agg["b_vox"],
                inter_vox=agg["inter_vox"],
                union_vox=agg["union_vox"],
            ))
        
        set_agg: Dict[Tuple[Tuple[int, ...], float], Dict] = defaultdict(
            lambda: {"member_voxels": defaultdict(int), "inter_vox": 0, "union_vox": 0}
        )
        
        for result in tile_results:
            for sr in result.sets:
                key = (sr.members, sr.r_um)
                set_agg[key]["inter_vox"] += sr.inter_vox
                set_agg[key]["union_vox"] += sr.union_vox
                for ch in sr.members:
                    ch_vox = result.marker_vox.get(sr.r_um, {}).get(ch, 0)
                    set_agg[key]["member_voxels"][ch] += ch_vox
        
        for (members, r), agg in set_agg.items():
            hierarchy_level.sets.append(AggregatedSet(
                members=members,
                r_um=r,
                hierarchy_level=level,
                tile_x0=tile_x0,
                tile_x1=tile_x1,
                tile_y0=tile_y0,
                tile_y1=tile_y1,
                member_voxels=dict(agg["member_voxels"]),
                inter_vox=agg["inter_vox"],
                union_vox=agg["union_vox"],
            ))