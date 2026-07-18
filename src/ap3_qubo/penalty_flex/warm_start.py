"""
PenaltyFlex warm-start λ 缓存。

ParetoZoom 提出新权重组时，查询最近邻已收敛权重的 λ* 值
作为 PenaltyFlex 的初始值，加速收敛。

缓存键: (w1, w2, w3) 权重元组
缓存值: (λ_carbide, λ_ccr)
"""

from typing import Dict, Optional, Tuple

import numpy as np


class LambdaCache:
    """λ 值缓存，支持精确查找和最近邻查找。

    使用示例:
        >>> cache = LambdaCache()
        >>> cache.store((0.5, 0.3, 0.2), 0.12, 0.08)
        >>> lam = cache.get((0.5, 0.3, 0.2))
        >>> nearest = cache.find_nearest((0.5, 0.35, 0.15))
    """

    def __init__(self):
        self._cache: Dict[Tuple[float, float, float], Tuple[float, float]] = {}

    def store(
        self,
        weights: Tuple[float, float, float],
        lambda_carbide: float,
        lambda_ccr: float,
    ) -> None:
        """存储一组权重对应的收敛 λ 值。

        Args:
            weights: (w1, w2, w3) 偏好权重。
            lambda_carbide: P1 收敛 λ。
            lambda_ccr: P2 收敛 λ。
        """
        # 确保权重已归一化
        w = np.array(weights, dtype=float)
        w = w / w.sum()
        key = (float(w[0]), float(w[1]), float(w[2]))
        self._cache[key] = (float(lambda_carbide), float(lambda_ccr))

    def get(
        self, weights: Tuple[float, float, float]
    ) -> Optional[Tuple[float, float]]:
        """精确查找权重对应的 λ 值。

        Args:
            weights: (w1, w2, w3) 偏好权重。

        Returns:
            (λ_carbide, λ_ccr) 或 None（未找到）。
        """
        w = np.array(weights, dtype=float)
        w = w / w.sum()
        key = (float(w[0]), float(w[1]), float(w[2]))
        return self._cache.get(key)

    def find_nearest(
        self,
        weights: Tuple[float, float, float],
        max_distance: float = 0.5,
    ) -> Optional[Tuple[float, float]]:
        """查找最近邻权重对应的 λ 值（欧氏距离）。

        Args:
            weights: (w1, w2, w3) 目标权重。
            max_distance: 最大搜索距离，超过此距离返回 None。

        Returns:
            (λ_carbide, λ_ccr) 或 None（无足够近的邻居）。
        """
        if not self._cache:
            return None

        w_target = np.array(weights, dtype=float)
        w_target = w_target / w_target.sum()

        best_distance = float("inf")
        best_lambdas: Optional[Tuple[float, float]] = None

        for cached_w, cached_lam in self._cache.items():
            dist = np.linalg.norm(w_target - np.array(cached_w))
            if dist < best_distance:
                best_distance = dist
                best_lambdas = cached_lam

        if best_distance > max_distance:
            return None

        return best_lambdas

    def clear(self) -> None:
        """清空缓存。"""
        self._cache.clear()

    def __len__(self) -> int:
        return len(self._cache)

    def __contains__(self, weights: Tuple[float, float, float]) -> bool:
        return self.get(weights) is not None

    def __repr__(self) -> str:
        return f"LambdaCache({len(self._cache)} entries)"
