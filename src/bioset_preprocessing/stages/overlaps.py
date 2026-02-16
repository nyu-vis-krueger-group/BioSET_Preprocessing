from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Dict, Iterable, List, Sequence, Tuple

import cupy as cp


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


@dataclass
class OverlapTileResult:
    """All overlap stats for a (tile_x, tile_y) across radii."""

    tile_x: int
    tile_y: int
    radii_um: List[float]
    # marker_vox[r][ch] = count
    marker_vox: Dict[float, Dict[int, int]]
    pairs: List[PairRow]
    sets: List[SetRow]


def _inter_count_gpu(masks_ch: Dict[int, cp.ndarray], chs: Sequence[int]) -> int:
    it = iter(chs)
    inter = masks_ch[next(it)]
    for ch in it:
        inter = inter & masks_ch[ch]
        if not bool(cp.any(inter)):
            return 0
    return int(cp.count_nonzero(inter))


def _union_count_gpu(masks_ch: Dict[int, cp.ndarray], chs: Sequence[int]) -> int:
    it = iter(chs)
    uni = masks_ch[next(it)]
    for ch in it:
        uni = uni | masks_ch[ch]
    return int(cp.count_nonzero(uni))


def _pair_metrics(a_vox: int, b_vox: int, inter_vox: int, union_vox: int) -> Tuple[float, float]:
    iou = (inter_vox / union_vox) if union_vox > 0 else 0.0
    denom = min(a_vox, b_vox)
    overlap_coeff = (inter_vox / denom) if denom > 0 else 0.0
    return float(iou), float(overlap_coeff)


def _apriori_candidates_from_pairs(freq_pairs: Iterable[Tuple[int, int]], k: int) -> List[Tuple[int, ...]]:
    pair_set = set(tuple(sorted(p)) for p in freq_pairs)
    items = sorted({i for p in pair_set for i in p})

    cands: List[Tuple[int, ...]] = []
    for comb in combinations(items, k):
        ok = True
        for a, b in combinations(comb, 2):
            if (a, b) not in pair_set:
                ok = False
                break
        if ok:
            cands.append(comb)
    return cands


class OverlapMiner:
    """Mine pairwise + higher-order overlaps per tile, across dilation radii.

    Inputs are the boolean masks produced by threshold/CC/dilation stages:
        masks[r_um][ch] -> cp.bool_ array (Z,Y,X)

    The miner implements:
      1) marker presence filtering per radius
      2) frequent pair mining at max radius (for candidate generation)
      3) apriori-like candidate generation for k>=3 (every pair must be frequent)
      4) dilation-sweep pruning: evaluate from largest->smallest radius and
         stop early when failing support thresholds.
    """

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
        masks: Dict[float, Dict[int, cp.ndarray]],
        marker_vox: Dict[float, Dict[int, int]],
    ) -> OverlapTileResult:
        channels = sorted(next(iter(masks.values())).keys()) if masks else []

        # Active markers per radius
        active: Dict[float, List[int]] = {}
        for r in self.radii:
            mm = self._thr(self.min_marker_vox, r)
            active[r] = [ch for ch in channels if marker_vox[r][ch] >= mm]

        active_max = active[self.r_max]
        if len(active_max) < 2:
            return OverlapTileResult(
                tile_x=tile_x,
                tile_y=tile_y,
                radii_um=list(self.radii),
                marker_vox=marker_vox,
                pairs=[],
                sets=[],
            )

        # Frequent pairs at r_max
        freq_pairs: List[Tuple[int, int]] = []
        ms_pair_max = self._thr(self.min_support_pair, self.r_max)
        for a, b in combinations(active_max, 2):
            inter = _inter_count_gpu(masks[self.r_max], (a, b))
            if inter >= ms_pair_max:
                freq_pairs.append((a, b))

        # Candidate sets upto max set size 
        cand_sets: List[Tuple[int, ...]] = []
        for k in range(3, self.max_set_size + 1):
            cand_sets.extend(_apriori_candidates_from_pairs(freq_pairs, k))

        def sweep_eval(ch_tuple: Tuple[int, ...]) -> Tuple[List[PairRow], List[SetRow]]:
            out_pairs: List[PairRow] = []
            out_sets: List[SetRow] = []

            # must be active at max dilation
            if any(ch not in active[self.r_max] for ch in ch_tuple):
                return out_pairs, out_sets

            for r in self.radii_desc:
                # skip if any member absent at this dilation
                if any(ch not in active[r] for ch in ch_tuple):
                    if r == self.r_max:
                        return [], []
                    if self.aggressive_stop_on_fail:
                        break
                    continue

                inter = _inter_count_gpu(masks[r], ch_tuple)
                min_support = (
                    self._thr(self.min_support_set, r)
                    if len(ch_tuple) >= 3
                    else self._thr(self.min_support_pair, r)
                )

                # if fails at max dilation, will fail at smaller dilations, skip
                if r == self.r_max and inter < min_support:
                    return [], []

                if inter >= min_support:
                    uni = _union_count_gpu(masks[r], ch_tuple)

                    if len(ch_tuple) == 2:
                        a, b = ch_tuple
                        a_vox = int(marker_vox[r][a])
                        b_vox = int(marker_vox[r][b])
                        iou, oc = _pair_metrics(a_vox, b_vox, inter, uni)
                        out_pairs.append(
                            PairRow(
                                tile_x=tile_x,
                                tile_y=tile_y,
                                r_um=float(r),
                                a=a,
                                b=b,
                                a_vox=a_vox,
                                b_vox=b_vox,
                                inter_vox=int(inter),
                                union_vox=int(uni),
                                iou=iou,
                                overlap_coeff=oc,
                            )
                        )
                    else:
                        iou = (inter / uni) if uni > 0 else 0.0
                        out_sets.append(
                            SetRow(
                                tile_x=tile_x,
                                tile_y=tile_y,
                                r_um=float(r),
                                k=len(ch_tuple),
                                members=tuple(ch_tuple),
                                inter_vox=int(inter),
                                union_vox=int(uni),
                                iou=float(iou),
                            )
                        )
                else:
                    if self.aggressive_stop_on_fail:
                        break

            return out_pairs, out_sets

        # evaluate pairs and sets
        all_pairs: List[PairRow] = []
        all_sets: List[SetRow] = []

        for p in freq_pairs:
            pr, sr = sweep_eval(tuple(sorted(p)))
            all_pairs.extend(pr)
            all_sets.extend(sr)

        for s in cand_sets:
            pr, sr = sweep_eval(s)
            all_pairs.extend(pr)
            all_sets.extend(sr)

        return OverlapTileResult(
            tile_x=tile_x,
            tile_y=tile_y,
            radii_um=list(self.radii),
            marker_vox=marker_vox,
            pairs=all_pairs,
            sets=all_sets,
        )