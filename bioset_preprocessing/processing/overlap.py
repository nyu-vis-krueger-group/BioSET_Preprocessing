"""
Channel overlap computation for co-localization analysis.

Key changes from the original:
- Matrix-multiply for pairwise overlaps (5-20x faster than looping)
- Enrichment ratio computation (symmetric, for UpSet visualization)
- Adaptive higher-order combo selection (prune by pairwise enrichment)
- Higher-order enrichment: observed vs pairwise-expected
"""

import itertools
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .thresholding import _init_gpu, _GPU_AVAILABLE, _cp

logger = logging.getLogger(__name__)


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class PairwiseResult:
    """
    Result of pairwise overlap computation for a single tile.
    
    Attributes:
        overlap_matrix: (n_channels, n_channels) symmetric matrix.
                        Diagonal = per-channel active voxel counts.
                        Off-diagonal = pairwise overlap counts.
        channel_counts: {channel_idx: active_voxel_count}
        channels: sorted list of channel indices
        total_voxels: total voxels in tile
    """
    overlap_matrix: np.ndarray          # (n, n) int64
    channel_counts: Dict[int, int]
    channels: List[int]
    total_voxels: int
    
    def get_overlap(self, ch_a: int, ch_b: int) -> int:
        """Get pairwise overlap count."""
        i = self.channels.index(ch_a)
        j = self.channels.index(ch_b)
        return int(self.overlap_matrix[i, j])
    
    def get_count(self, ch: int) -> int:
        """Get active voxel count for a channel."""
        return self.channel_counts.get(ch, 0)


@dataclass
class EnrichmentResult:
    """
    Enrichment ratios for pairwise overlaps.
    
    enrichment_matrix[i,j] = |A∩B| * total / (|A| * |B|)
    
    SYMMETRIC: enrichment_matrix[i,j] == enrichment_matrix[j,i].
    Diagonal = 0 (self-enrichment is not meaningful).
    
    Interpretation: E=1 means random, E>1 means enriched, E<1 means avoidance.
    """
    enrichment_matrix: np.ndarray       # (n, n) float64, symmetric
    channels: List[int]
    
    def get_enrichment(self, ch_a: int, ch_b: int) -> float:
        i = self.channels.index(ch_a)
        j = self.channels.index(ch_b)
        return float(self.enrichment_matrix[i, j])
    
    def get_top_pairs(self, n: int = 20) -> List[Tuple[int, int, float]]:
        """Get top N enriched pairs as (ch_a, ch_b, enrichment)."""
        mat = self.enrichment_matrix.copy()
        np.fill_diagonal(mat, 0)
        triu_idx = np.triu_indices_from(mat, k=1)
        values = mat[triu_idx]
        top_idx = np.argsort(values)[::-1][:n]
        result = []
        for idx in top_idx:
            i, j = triu_idx[0][idx], triu_idx[1][idx]
            result.append((self.channels[i], self.channels[j], float(values[idx])))
        return result


@dataclass
class HigherOrderResult:
    """
    Results for higher-order (3+) combinations.
    Only contains combinations selected by enrichment threshold.
    """
    overlaps: Dict[Tuple[int, ...], int]           # raw overlap count
    enrichment_indep: Dict[Tuple[int, ...], float]  # vs independence
    enrichment_higher: Dict[Tuple[int, ...], float] # vs pairwise-expected
    n_candidates: int = 0       # how many combos were candidates
    n_computed: int = 0         # how many actually computed
    n_pruned_empty: int = 0     # how many had zero overlap


# =============================================================================
# Pairwise Overlap: Matrix Multiply (the main speedup)
# =============================================================================

def compute_pairwise_fast(
    masks: Dict[int, np.ndarray],
) -> PairwiseResult:
    """
    Compute all pairwise overlaps using matrix multiplication.
    
    Stacks all masks into (n_channels, n_voxels) and computes
    overlap_matrix = M @ M.T in memory-controlled chunks.
    
    For 70 channels this replaces 2,415 individual AND+sum operations
    with BLAS-optimized matmul calls, yielding 5-20x speedup.
    
    Args:
        masks: {channel_idx: binary_mask_array}
    
    Returns:
        PairwiseResult with overlap matrix and channel counts
    """
    _init_gpu()
    
    channels = sorted(masks.keys())
    first_mask = masks[channels[0]]
    total_voxels = first_mask.size
    
    if _GPU_AVAILABLE:
        return _pairwise_matmul_gpu(masks, channels, total_voxels)
    else:
        return _pairwise_matmul_cpu(masks, channels, total_voxels)


def _pairwise_matmul_cpu(
    masks: Dict[int, np.ndarray],
    channels: List[int],
    total_voxels: int,
    chunk_size: int = 2_000_000,
) -> PairwiseResult:
    """
    CPU matmul-based pairwise computation.
    
    Chunks the voxel dimension to control memory:
    70 channels × 2M voxels × 4 bytes = 560MB per chunk (float32)
    """
    n_channels = len(channels)
    
    # Stack: (n_channels, n_voxels) uint8
    stacked = np.stack([masks[ch].ravel() for ch in channels])
    if stacked.dtype != np.uint8:
        stacked = stacked.astype(np.uint8)
    
    n_voxels = stacked.shape[1]
    overlap_matrix = np.zeros((n_channels, n_channels), dtype=np.int64)
    
    for start in range(0, n_voxels, chunk_size):
        end = min(start + chunk_size, n_voxels)
        chunk = stacked[:, start:end].astype(np.float32)
        overlap_matrix += (chunk @ chunk.T).astype(np.int64)
    
    channel_counts = {ch: int(overlap_matrix[i, i]) for i, ch in enumerate(channels)}
    
    del stacked
    
    return PairwiseResult(
        overlap_matrix=overlap_matrix,
        channel_counts=channel_counts,
        channels=channels,
        total_voxels=total_voxels,
    )


def _pairwise_matmul_gpu(
    masks: Dict[int, np.ndarray],
    channels: List[int],
    total_voxels: int,
    chunk_size: int = 5_000_000,
) -> PairwiseResult:
    """GPU matmul-based pairwise computation (larger chunks, cuBLAS)."""
    n_channels = len(channels)
    
    stacked = np.stack([masks[ch].ravel() for ch in channels])
    if stacked.dtype != np.uint8:
        stacked = stacked.astype(np.uint8)
    
    n_voxels = stacked.shape[1]
    overlap_matrix_gpu = _cp.zeros((n_channels, n_channels), dtype=_cp.int64)
    
    for start in range(0, n_voxels, chunk_size):
        end = min(start + chunk_size, n_voxels)
        chunk_gpu = _cp.asarray(stacked[:, start:end], dtype=_cp.float32)
        overlap_matrix_gpu += _cp.matmul(chunk_gpu, chunk_gpu.T).astype(_cp.int64)
        del chunk_gpu
    
    overlap_matrix = _cp.asnumpy(overlap_matrix_gpu)
    del overlap_matrix_gpu
    _cp.get_default_memory_pool().free_all_blocks()
    
    channel_counts = {ch: int(overlap_matrix[i, i]) for i, ch in enumerate(channels)}
    
    del stacked
    
    return PairwiseResult(
        overlap_matrix=overlap_matrix,
        channel_counts=channel_counts,
        channels=channels,
        total_voxels=total_voxels,
    )


# =============================================================================
# Enrichment Computation
# =============================================================================

def compute_enrichment(pairwise: PairwiseResult) -> EnrichmentResult:
    """
    Compute pairwise enrichment ratios from the overlap matrix.
    
    E(A,B) = |A∩B| * total / (|A| * |B|)
    
    This is the symmetric metric used for UpSet plot bars.
    E = 1 means random co-occurrence.
    E > 1 means enrichment (markers co-localize more than chance).
    E < 1 means avoidance (markers spatially exclude each other).
    """
    total = pairwise.total_voxels
    overlap = pairwise.overlap_matrix.astype(np.float64)
    
    counts = np.diag(overlap).copy()  # (n,) per-channel counts
    denom = np.outer(counts, counts)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        enrichment = np.where(denom > 0, overlap * total / denom, 0.0)
    
    np.fill_diagonal(enrichment, 0.0)
    
    return EnrichmentResult(
        enrichment_matrix=enrichment,
        channels=pairwise.channels,
    )


# =============================================================================
# Higher-Order: Adaptive Selection + Computation
# =============================================================================

def select_higher_order_combos(
    enrichment: EnrichmentResult,
    max_size: int = 4,
    enrichment_threshold: float = 1.5,
) -> List[Tuple[int, ...]]:
    """
    Select higher-order combinations where ALL pairwise enrichments
    exceed the threshold.
    
    For a triple (A,B,C), we require:
        E(A,B) > threshold AND E(A,C) > threshold AND E(B,C) > threshold
    
    This prunes the combinatorial space dramatically:
    70 channels → 54,740 possible triples, typically < 1000 pass.
    
    Returns:
        List of channel index tuples to compute overlaps for
    """
    channels = enrichment.channels
    mat = enrichment.enrichment_matrix
    n = len(channels)
    
    # Build adjacency set of enriched pairs (by matrix index)
    enriched_pairs = set()
    for i in range(n):
        for j in range(i + 1, n):
            if mat[i, j] >= enrichment_threshold:
                enriched_pairs.add((i, j))
    
    n_total_pairs = n * (n - 1) // 2
    logger.info(
        f"  Enriched pairs: {len(enriched_pairs)} / {n_total_pairs} "
        f"(threshold={enrichment_threshold})"
    )
    
    selected = []
    
    for size in range(3, max_size + 1):
        count_before = len(selected)
        for combo_indices in itertools.combinations(range(n), size):
            # Check ALL pairs within this combo
            all_enriched = True
            for i, j in itertools.combinations(combo_indices, 2):
                if (i, j) not in enriched_pairs:
                    all_enriched = False
                    break
            
            if all_enriched:
                combo = tuple(channels[idx] for idx in combo_indices)
                selected.append(combo)
        
        count_new = len(selected) - count_before
        from math import comb
        total_possible = comb(n, size)
        pct = 100 * count_new / total_possible if total_possible > 0 else 0
        logger.info(
            f"  Size-{size} combos: {count_new} / {total_possible} selected ({pct:.1f}%)"
        )
    
    return selected


def compute_selected_overlaps(
    masks: Dict[int, np.ndarray],
    combinations: List[Tuple[int, ...]],
) -> Dict[Tuple[int, ...], int]:
    """
    Compute overlap counts for specific (pre-selected) combinations.
    Uses AND-reduction for each combination.
    """
    if not combinations:
        return {}
    
    _init_gpu()
    
    if _GPU_AVAILABLE:
        return _selected_overlaps_gpu(masks, combinations)
    else:
        return _selected_overlaps_cpu(masks, combinations)


def _selected_overlaps_cpu(
    masks: Dict[int, np.ndarray],
    combinations: List[Tuple[int, ...]],
) -> Dict[Tuple[int, ...], int]:
    """CPU path for selected higher-order overlaps."""
    flat = {ch: masks[ch].ravel() for ch in masks}
    
    overlaps = {}
    for combo in combinations:
        result = flat[combo[0]]
        for ch in combo[1:]:
            result = result & flat[ch]
        overlaps[combo] = int(np.count_nonzero(result))
    
    return overlaps


def _selected_overlaps_gpu(
    masks: Dict[int, np.ndarray],
    combinations: List[Tuple[int, ...]],
) -> Dict[Tuple[int, ...], int]:
    """GPU path for selected higher-order overlaps."""
    needed_channels = set()
    for combo in combinations:
        needed_channels.update(combo)
    
    gpu_masks = {
        ch: _cp.asarray(masks[ch].ravel(), dtype=_cp.uint8)
        for ch in needed_channels
    }
    
    overlaps = {}
    for combo in combinations:
        result = gpu_masks[combo[0]]
        for ch in combo[1:]:
            result = result & gpu_masks[ch]
        overlaps[combo] = int(_cp.sum(result))
    
    del gpu_masks
    _cp.get_default_memory_pool().free_all_blocks()
    
    return overlaps


def compute_higher_order_enrichment(
    combo: Tuple[int, ...],
    overlap_count: int,
    pairwise: PairwiseResult,
) -> Tuple[Optional[float], Optional[float]]:
    """
    Compute enrichment metrics for a higher-order combination.
    
    Returns (E_indep, E_higher):
    
    E_indep: enrichment vs independence baseline.
        For (A,B,C): |A∩B∩C| * total^2 / (|A| * |B| * |C|)
        E_indep > 1 means more co-occurrence than random.
    
    E_higher: enrichment vs pairwise-expected baseline.
        For (A,B,C): |A∩B∩C| * |A| * |B| * |C| / (|A∩B| * |A∩C| * |B∩C|)
        E_higher > 1 means genuine higher-order interaction beyond pairwise.
        E_higher ~ 1 means fully explained by pairwise structure.
    """
    if overlap_count == 0:
        return (0.0, None)
    
    k = len(combo)
    total = pairwise.total_voxels
    
    # Individual counts
    counts = []
    for ch in combo:
        c = pairwise.channel_counts.get(ch, 0)
        if c == 0:
            return (0.0, None)
        counts.append(c)
    
    # E_indep = |intersection| * total^(k-1) / product(counts)
    product_counts = 1.0
    for c in counts:
        product_counts *= c
    
    e_indep = (overlap_count * (total ** (k - 1))) / product_counts if product_counts > 0 else None
    
    # E_higher = |intersection| * product(counts) / product(pairwise_overlaps)
    product_pairwise = 1.0
    for ch_a, ch_b in itertools.combinations(combo, 2):
        pw_count = pairwise.get_overlap(ch_a, ch_b)
        if pw_count == 0:
            return (e_indep, None)
        product_pairwise *= pw_count
    
    e_higher = (overlap_count * product_counts) / product_pairwise if product_pairwise > 0 else None
    
    return (e_indep, e_higher)


# =============================================================================
# High-Level Orchestrator
# =============================================================================

def compute_all_overlaps(
    masks: Dict[int, np.ndarray],
    max_higher_order_size: int = 4,
    enrichment_threshold: float = 1.5,
    compute_higher_order: bool = True,
) -> Tuple[PairwiseResult, EnrichmentResult, HigherOrderResult]:
    """
    Complete overlap analysis for a single tile at a single dilation.
    
    1. Compute pairwise overlaps via matrix multiply (fast)
    2. Compute enrichment ratios
    3. Select higher-order combos by enrichment threshold
    4. Compute selected higher-order overlaps
    5. Compute higher-order enrichment metrics
    
    Args:
        masks: {channel_idx: binary_mask_array}
        max_higher_order_size: max combination size for higher-order (3 or 4)
        enrichment_threshold: minimum pairwise enrichment to include triple
        compute_higher_order: whether to compute triples+ at all
    
    Returns:
        (PairwiseResult, EnrichmentResult, HigherOrderResult)
    """
    # Step 1: Fast pairwise via matmul
    pairwise = compute_pairwise_fast(masks)
    
    # Step 2: Enrichment ratios (symmetric, for UpSet)
    enrichment = compute_enrichment(pairwise)
    
    # Step 3-5: Higher-order (optional)
    higher_order = HigherOrderResult(
        overlaps={}, enrichment_indep={}, enrichment_higher={},
    )
    
    if compute_higher_order and max_higher_order_size >= 3 and len(pairwise.channels) >= 3:
        # Select candidates
        selected = select_higher_order_combos(
            enrichment,
            max_size=max_higher_order_size,
            enrichment_threshold=enrichment_threshold,
        )
        higher_order.n_candidates = len(selected)
        
        if selected:
            # Compute overlaps for selected combos
            ho_overlaps = compute_selected_overlaps(masks, selected)
            higher_order.n_computed = len(ho_overlaps)
            
            # Compute enrichment for each
            n_pruned = 0
            for combo, count in ho_overlaps.items():
                if count == 0:
                    n_pruned += 1
                    continue
                
                e_indep, e_higher = compute_higher_order_enrichment(
                    combo, count, pairwise
                )
                
                higher_order.overlaps[combo] = count
                if e_indep is not None:
                    higher_order.enrichment_indep[combo] = e_indep
                if e_higher is not None:
                    higher_order.enrichment_higher[combo] = e_higher
            
            higher_order.n_pruned_empty = n_pruned
        
        logger.info(
            f"  Higher-order: {higher_order.n_candidates} candidates, "
            f"{len(higher_order.overlaps)} with overlap, "
            f"{higher_order.n_pruned_empty} pruned (zero overlap)"
        )
    
    return pairwise, enrichment, higher_order


# =============================================================================
# Backward-compatible wrappers
# =============================================================================

def get_channel_combinations(
    channels: List[int],
    min_size: int = 2,
    max_size: int = None,
) -> List[Tuple[int, ...]]:
    """Generate all combinations of channels (kept for backward compat)."""
    if max_size is None:
        max_size = len(channels)
    combinations = []
    for r in range(min_size, max_size + 1):
        combinations.extend(itertools.combinations(sorted(channels), r))
    return combinations


@dataclass
class OverlapResult:
    """Legacy result type. Prefer PairwiseResult + HigherOrderResult."""
    overlaps: Dict[Tuple[int, ...], int]
    combinations: List[Tuple[int, ...]]
    total_voxels: int

    def get_overlap(self, *channels: int) -> int:
        key = tuple(sorted(channels))
        return self.overlaps.get(key, 0)

    def get_overlap_fraction(self, *channels: int) -> float:
        return self.get_overlap(*channels) / self.total_voxels

    def to_dict(self) -> Dict[str, int]:
        return {"_".join(map(str, k)): v for k, v in self.overlaps.items()}


def compute_overlaps(
    masks: Dict[int, np.ndarray],
    combinations: List[Tuple[int, ...]] = None,
) -> OverlapResult:
    """Legacy interface. For new code use compute_all_overlaps()."""
    channels = sorted(masks.keys())
    if combinations is None:
        combinations = get_channel_combinations(channels)

    first_mask = masks[channels[0]]
    total_voxels = first_mask.size

    _init_gpu()

    pair_combos = [c for c in combinations if len(c) == 2]
    other_combos = [c for c in combinations if len(c) > 2]

    overlaps = {}

    if pair_combos:
        pairwise = compute_pairwise_fast(masks)
        for combo in pair_combos:
            overlaps[combo] = pairwise.get_overlap(combo[0], combo[1])

    if other_combos:
        ho = compute_selected_overlaps(masks, other_combos)
        overlaps.update(ho)

    return OverlapResult(
        overlaps=overlaps,
        combinations=combinations,
        total_voxels=total_voxels,
    )