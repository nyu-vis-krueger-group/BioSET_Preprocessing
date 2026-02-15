from dataclasses import dataclass
import numpy as np
import cupy as cp
from scipy.stats import norm

@dataclass
class ThresholdStats:
    t_global: float
    q_tile: float
    t_final: float
    alpha_eff: float

class AlphaThreshold:
    def __init__(self, alpha: float = 0.4, trim_q: float = 0.98):
        self.alpha = alpha
        self.trim_q = trim_q

    @staticmethod
    def _mad_cpu(x: np.ndarray) -> float:
        x = x.astype(np.float32, copy=False).ravel()
        m = np.median(x)
        return float(np.median(np.abs(x - m)) + 1e-6)

    def compute_global(self, lowres_vol_cpu: np.ndarray) -> float:
        mip = lowres_vol_cpu.max(axis=0)
        cut = np.quantile(mip, self.trim_q)
        bg = mip[mip <= cut]
        mu = float(np.median(bg))
        sigma = float(1.4826 * self._mad_cpu(bg))
        z = float(norm.ppf(1.0 - self.alpha))
        return mu + z * sigma

    def compute_tile_gpu(self, vol_gpu: cp.ndarray, t_global: float) -> ThresholdStats:
        mip = cp.max(vol_gpu, axis=0).astype(cp.float32)

        cut = cp.quantile(mip, self.trim_q)                     
        bg = mip[mip <= cut]

        n = int(bg.size)
        alpha_eff = max(self.alpha, 1.0 / max(1, n))
        q_tile = cp.quantile(bg, 1.0 - alpha_eff)

        q_tile_f = float(q_tile.get())
        t_final = float(max(t_global, q_tile_f))

        return ThresholdStats(
            t_global=float(t_global),
            q_tile=q_tile_f,
            t_final=t_final,
            alpha_eff=float(alpha_eff),
        )

    def apply_gpu(self, vol_gpu: cp.ndarray, t_final: float) -> cp.ndarray:
        return vol_gpu > t_final
