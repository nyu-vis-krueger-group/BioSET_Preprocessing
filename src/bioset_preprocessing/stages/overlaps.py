from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Dict, List, Sequence, Tuple

import cupy as cp

@dataclass
class ChannelTileStats:
    tile_x: int
    tile_y: int
    channel: int
    r_um: float
    voxel_count: int          
    sum_intensity: float      

@dataclass
class PairRow:
    tile_x: int
    tile_y: int
    r_um: float
    a: int
    b: int
    a_vox: int
    b_vox: int
    inter_vox: int
    union_vox: int
    iou: float
    overlap_coeff: float

@dataclass
class SetRow:
    tile_x: int
    tile_y: int
    r_um: float
    k: int
    members: Tuple[int, ...]
    inter_vox: int
    union_vox: int
    iou: float
    overlap_coeff: float


@dataclass
class OverlapTileResult:
    tile_x: int
    tile_y: int
    tile_z: int
    tile_shape: Tuple[int, int, int]
    total_voxels: int
    radii_um: List[float]
    marker_vox: Dict[float, Dict[int, int]]
    channel_stats: List[ChannelTileStats]
    pairs: List[PairRow]
    sets: List[SetRow]
    n_active_channels: int = 0
    n_frequent_pairs: int = 0


def _compute_pairwise_intersections_batched(
    masks: Dict[int, cp.ndarray],
    pairs: List[Tuple[int, int]],
) -> cp.ndarray:
    if not pairs:
        return cp.array([], dtype=cp.int64)
    
    n_pairs = len(pairs)
    counts = cp.zeros(n_pairs, dtype=cp.int64)
    
    # Process in chunks to avoid memory spikes
    chunk_size = 256
    for start in range(0, n_pairs, chunk_size):
        end = min(start + chunk_size, n_pairs)
        chunk_pairs = pairs[start:end]
        
        for i, (a, b) in enumerate(chunk_pairs):
            inter = masks[a] & masks[b]
            counts[start + i] = cp.count_nonzero(inter)
    
    return counts


class OverlapMiner:
    def __init__(
        self,
        radii_um: Sequence[float],
        max_set_size: int = 5,
        min_marker_vox: Dict[float, int] | int = 0,
        min_support_pair: Dict[float, int] | int = 0,
        min_support_set: Dict[float, int] | int = 0,
        aggressive_stop_on_fail: bool = True,
    ):
        self.radii = sorted(float(r) for r in radii_um)
        self.radii_desc = sorted(self.radii, reverse=True)
        self.r_max = self.radii_desc[0]
        self.max_set_size = int(max_set_size)
        self.min_marker_vox = min_marker_vox
        self.min_support_pair = min_support_pair
        self.min_support_set = min_support_set
        self.aggressive_stop_on_fail = aggressive_stop_on_fail

    def _thr(self, spec: Dict[float, int] | int, r: float) -> int:
        if isinstance(spec, dict):
            return int(spec.get(r, 0))
        return int(spec)

    def run(
        self,
        *,
        tile_x: int,
        tile_y: int,
        tile_shape: Tuple[int, int, int],
        total_voxels: int,
        masks: Dict[float, Dict[int, cp.ndarray]],
        marker_vox: Dict[float, Dict[int, int]],
        sum_intensity: Dict[float, Dict[int, float]],
    ) -> OverlapTileResult:
        channels = sorted(next(iter(masks.values())).keys()) if masks else []

        # Filter to active markers per radius
        active: Dict[float, List[int]] = {}
        channel_stats: List[ChannelTileStats] = []
        for r in self.radii:
            active[r] = []
            min_vox_thr = self._thr(self.min_marker_vox, r)
            for ch in channels:
                vox_count = marker_vox[r][ch]
                channel_stats.append(ChannelTileStats(
                    tile_x=tile_x,
                    tile_y=tile_y,
                    channel=ch,
                    r_um=r,
                    voxel_count=vox_count,
                    sum_intensity=sum_intensity[r][ch],
                ))
                if vox_count >= min_vox_thr:
                    active[r].append(ch)

        active_max = active[self.r_max]
        n_active = len(active_max)
        
        if n_active < 2:
            return OverlapTileResult(
                tile_x=tile_x,
                tile_y=tile_y,
                tile_z=0,
                tile_shape=tile_shape,
                total_voxels=total_voxels,
                radii_um=list(self.radii),
                marker_vox=marker_vox,
                channel_stats=channel_stats,
                pairs=[],
                sets=[],
                n_active_channels=n_active,
                n_frequent_pairs=0,
            )

        # Compute ALL pairwise intersections at r_max
        all_pairs = list(combinations(active_max, 2))
        ms_pair_max = self._thr(self.min_support_pair, self.r_max)
        
        inter_counts = _compute_pairwise_intersections_batched(
            masks[self.r_max], all_pairs
        )
        inter_counts_cpu = cp.asnumpy(inter_counts)
        
        freq_pairs: List[Tuple[int, int]] = []
        freq_pair_inters: Dict[Tuple[int, int], int] = {}
        
        for i, (a, b) in enumerate(all_pairs):
            if inter_counts_cpu[i] >= ms_pair_max:
                pair = (a, b)
                freq_pairs.append(pair)
                freq_pair_inters[pair] = int(inter_counts_cpu[i])
        
        n_freq_pairs = len(freq_pairs)
        
        if n_freq_pairs == 0:
            return OverlapTileResult(
                tile_x=tile_x,
                tile_y=tile_y,
                tile_z=0,
                tile_shape=tile_shape,
                total_voxels=total_voxels,
                radii_um=list(self.radii),
                marker_vox=marker_vox,
                channel_stats=channel_stats,
                pairs=[],
                sets=[],
                n_active_channels=n_active,
                n_frequent_pairs=0,
            )
        
        # Generate higher-order candidates using Apriori
        cand_sets: List[Tuple[int, ...]] = []
        
        if self.max_set_size >= 3 and n_freq_pairs <= 5000:
            freq_pair_set = set(freq_pairs)
            items = sorted({i for p in freq_pairs for i in p})
            
            for k in range(3, self.max_set_size + 1):
                if len(items) > 50 and k > 3:
                    break
                    
                for comb in combinations(items, k):
                    all_freq = True
                    for a, b in combinations(comb, 2):
                        if (a, b) not in freq_pair_set and (b, a) not in freq_pair_set:
                            all_freq = False
                            break
                    if all_freq:
                        cand_sets.append(comb)
                
                if len(cand_sets) > 10000:
                    cand_sets = cand_sets[:10000]
                    break

        # Evaluate pairs across all radii
        all_pairs_out: List[PairRow] = []
        
        for (a, b) in freq_pairs:
            for r in self.radii_desc:
                if a not in active[r] or b not in active[r]:
                    if self.aggressive_stop_on_fail:
                        break
                    continue
                
                # Use cached value for r_max, compute for others
                if r == self.r_max:
                    inter = freq_pair_inters[(a, b)]
                else:
                    inter = int(cp.count_nonzero(masks[r][a] & masks[r][b]))
                
                ms = self._thr(self.min_support_pair, r)
                if inter < ms:
                    if self.aggressive_stop_on_fail:
                        break
                    continue
                
                uni = int(cp.count_nonzero(masks[r][a] | masks[r][b]))
                a_vox = marker_vox[r][a]
                b_vox = marker_vox[r][b]
                
                iou = inter / uni if uni > 0 else 0.0
                oc = inter / min(a_vox, b_vox) if min(a_vox, b_vox) > 0 else 0.0
                
                all_pairs_out.append(PairRow(
                    tile_x=tile_x,
                    tile_y=tile_y,
                    r_um=float(r),
                    a=a,
                    b=b,
                    a_vox=a_vox,
                    b_vox=b_vox,
                    inter_vox=inter,
                    union_vox=uni,
                    iou=iou,
                    overlap_coeff=oc,
                ))

        # Evaluate higher-order sets
        all_sets_out: List[SetRow] = []
        
        for comb in cand_sets:
            for r in self.radii_desc:
                if any(ch not in active[r] for ch in comb):
                    if self.aggressive_stop_on_fail:
                        break
                    continue
                
                # Compute intersection
                inter_mask = masks[r][comb[0]]
                for ch in comb[1:]:
                    inter_mask = inter_mask & masks[r][ch]
                inter = int(cp.count_nonzero(inter_mask))
                
                ms = self._thr(self.min_support_set, r)
                if inter < ms:
                    if self.aggressive_stop_on_fail:
                        break
                    continue
                
                # Compute union
                uni_mask = masks[r][comb[0]]
                for ch in comb[1:]:
                    uni_mask = uni_mask | masks[r][ch]
                uni = int(cp.count_nonzero(uni_mask))
                
                iou = inter / uni if uni > 0 else 0.0
                
                member_voxels = [marker_vox[r][ch] for ch in comb]
                min_member = min(member_voxels) if member_voxels else 0
                overlap_coeff = inter / min_member if min_member > 0 else 0.0
                
                all_sets_out.append(SetRow(
                    tile_x=tile_x,
                    tile_y=tile_y,
                    r_um=float(r),
                    k=len(comb),
                    members=comb,
                    inter_vox=inter,
                    union_vox=uni,
                    iou=float(iou),
                    overlap_coeff=float(overlap_coeff), 
                ))

        return OverlapTileResult(
            tile_x=tile_x,
            tile_y=tile_y,
            tile_z=0,  # full z
            tile_shape=tile_shape,
            total_voxels=total_voxels,
            radii_um=list(self.radii),
            marker_vox=marker_vox,
            channel_stats=channel_stats,
            pairs=all_pairs_out,
            sets=all_sets_out,
            n_active_channels=n_active,
            n_frequent_pairs=n_freq_pairs,
        )