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


def _stack_masks(masks_ch: Dict[int, cp.ndarray], channels: List[int]) -> cp.ndarray:
    """Stack channel masks into (n_channels, Z, Y, X) array for vectorized ops."""
    return cp.stack([masks_ch[ch] for ch in channels], axis=0)


def _compute_pairwise_batched_stacked(
    stacked: cp.ndarray,
    ch_to_idx: Dict[int, int],
    pairs: List[Tuple[int, int]],
) -> Tuple[cp.ndarray, cp.ndarray]:
    """
    Compute intersections and unions for all pairs using stacked masks.
    Returns (inter_counts, union_counts) as GPU arrays.
    """
    n_pairs = len(pairs)
    if n_pairs == 0:
        return cp.array([], dtype=cp.int64), cp.array([], dtype=cp.int64)
    
    inter_counts = cp.zeros(n_pairs, dtype=cp.int64)
    union_counts = cp.zeros(n_pairs, dtype=cp.int64)
    
    for i, (a, b) in enumerate(pairs):
        mask_a = stacked[ch_to_idx[a]]
        mask_b = stacked[ch_to_idx[b]]
        inter_counts[i] = cp.count_nonzero(mask_a & mask_b)
        union_counts[i] = cp.count_nonzero(mask_a | mask_b)
    
    return inter_counts, union_counts


def _compute_set_intersection_stacked(
    stacked: cp.ndarray,
    ch_to_idx: Dict[int, int],
    members: Tuple[int, ...],
) -> Tuple[int, int]:
    """Compute intersection and union for a set of channels."""
    indices = [ch_to_idx[ch] for ch in members]
    
    # Intersection: AND all masks
    inter_mask = stacked[indices[0]]
    for idx in indices[1:]:
        inter_mask = inter_mask & stacked[idx]
    
    # Union: OR all masks
    union_mask = stacked[indices[0]]
    for idx in indices[1:]:
        union_mask = union_mask | stacked[idx]
    
    # Return GPU arrays (will sync later)
    return cp.count_nonzero(inter_mask), cp.count_nonzero(union_mask)


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

        # Build channel index mapping for stacked arrays
        ch_to_idx = {ch: i for i, ch in enumerate(channels)}

        # Pre-stack masks for each radius (enables vectorized operations)
        stacked_masks: Dict[float, cp.ndarray] = {}
        for r in self.radii:
            if channels:
                stacked_masks[r] = _stack_masks(masks[r], channels)

        # Filter to active markers per radius and build channel stats
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

        # Compute ALL pairwise intersections at r_max using stacked masks
        all_pairs = list(combinations(active_max, 2))
        ms_pair_max = self._thr(self.min_support_pair, self.r_max)
        
        inter_counts_gpu, union_counts_gpu = _compute_pairwise_batched_stacked(
            stacked_masks[self.r_max], ch_to_idx, all_pairs
        )
        
        # Single sync for all pair counts at r_max
        inter_counts_cpu = cp.asnumpy(inter_counts_gpu)
        union_counts_cpu = cp.asnumpy(union_counts_gpu)
        
        # Filter to frequent pairs
        freq_pairs: List[Tuple[int, int]] = []
        freq_pair_data: Dict[Tuple[int, int], Dict] = {}  # Cache r_max results
        
        for i, (a, b) in enumerate(all_pairs):
            inter = int(inter_counts_cpu[i])
            if inter >= ms_pair_max:
                pair = (a, b)
                freq_pairs.append(pair)
                freq_pair_data[pair] = {
                    'inter': inter,
                    'union': int(union_counts_cpu[i]),
                }
        
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
        # Batch compute for each radius, then sync once per radius
        all_pairs_out: List[PairRow] = []
        
        for r in self.radii_desc:
            # Get pairs active at this radius
            pairs_at_r = [(a, b) for (a, b) in freq_pairs 
                          if a in active[r] and b in active[r]]
            
            if not pairs_at_r:
                if self.aggressive_stop_on_fail:
                    break
                continue
            
            ms = self._thr(self.min_support_pair, r)
            
            if r == self.r_max:
                # Use cached values
                for (a, b) in pairs_at_r:
                    data = freq_pair_data[(a, b)]
                    inter = data['inter']
                    uni = data['union']
                    
                    if inter < ms:
                        continue
                    
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
            else:
                # Batch compute for this radius
                inter_gpu, union_gpu = _compute_pairwise_batched_stacked(
                    stacked_masks[r], ch_to_idx, pairs_at_r
                )
                # Single sync per radius
                inter_cpu = cp.asnumpy(inter_gpu)
                union_cpu = cp.asnumpy(union_gpu)
                
                for i, (a, b) in enumerate(pairs_at_r):
                    inter = int(inter_cpu[i])
                    uni = int(union_cpu[i])
                    
                    if inter < ms:
                        continue
                    
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
        # Batch per radius: collect all GPU results, then sync
        all_sets_out: List[SetRow] = []
        
        for r in self.radii_desc:
            # Get sets active at this radius
            sets_at_r = [comb for comb in cand_sets 
                         if all(ch in active[r] for ch in comb)]
            
            if not sets_at_r:
                if self.aggressive_stop_on_fail:
                    break
                continue
            
            ms = self._thr(self.min_support_set, r)
            
            # Batch compute intersections and unions
            inter_results = []
            union_results = []
            
            for comb in sets_at_r:
                inter_gpu, union_gpu = _compute_set_intersection_stacked(
                    stacked_masks[r], ch_to_idx, comb
                )
                inter_results.append(inter_gpu)
                union_results.append(union_gpu)
            
            # Single sync for all sets at this radius
            inter_cpu = [int(x.get()) for x in inter_results]
            union_cpu = [int(x.get()) for x in union_results]
            
            for i, comb in enumerate(sets_at_r):
                inter = inter_cpu[i]
                uni = union_cpu[i]
                
                if inter < ms:
                    continue
                
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

        # Free stacked masks
        del stacked_masks

        return OverlapTileResult(
            tile_x=tile_x,
            tile_y=tile_y,
            tile_z=0,
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