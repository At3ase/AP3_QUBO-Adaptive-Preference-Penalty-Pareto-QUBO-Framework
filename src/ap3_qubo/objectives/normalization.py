"""
目标归一化层。

提供物理先验归一化和动态归一化两种策略。
"""

from abc import ABC, abstractmethod
from typing import Dict, Tuple

import numpy as np

from ..physical_params import F1_NORM_DENOM, F2_NORM_DENOM, F3_COST_MAX


class Normalizer(ABC):
    """归一化器抽象基类。"""

    @abstractmethod
    def normalize(self, value: float) -> float:
        """将原始目标值归一化到 O(1) 量级。"""
        ...

    @abstractmethod
    def denormalize(self, normalized: float) -> float:
        """将归一化值还原为原始目标值。"""
        ...


class PhysicalPriorNormalizer(Normalizer):
    """物理先验归一化 (用于粗网格阶段)。

    f₁^norm = f₁ / 30
    f₂^norm = f₂ / 10
    f₃^norm = f₃ / c_max
    """

    def __init__(self, f1_denom: float = F1_NORM_DENOM,
                 f2_denom: float = F2_NORM_DENOM,
                 f3_denom: float = F3_COST_MAX):
        # f3 默认分母统一取 physical_params.F3_COST_MAX (=6545.0)，
        # 消除历史上 6543 vs 6545 双源不一致 (评审报告 §四-11)；
        # 与 qubo/builder.py:442 的 f₃ 归一化保持同一数据源。
        self._denoms = np.array([f1_denom, f2_denom, f3_denom])

    def normalize(self, values: np.ndarray) -> np.ndarray:
        """归一化目标向量。

        Args:
            values: shape=(3,) 原始目标值 [f₁, f₂, f₃]。

        Returns:
            shape=(3,) 归一化目标值。
        """
        return values / self._denoms

    def denormalize(self, normalized: np.ndarray) -> np.ndarray:
        return normalized * self._denoms

    def normalize_single(self, obj_idx: int, value: float) -> float:
        """归一化单个目标值。"""
        return value / self._denoms[obj_idx]

    def denormalize_single(self, obj_idx: int, normalized: float) -> float:
        return normalized * self._denoms[obj_idx]


class DynamicNormalizer(Normalizer):
    """动态归一化 (用于 ParetoZoom 阶段, 基于当前存档的目标范围)。

    f_i^norm = (f_i - f_i^min) / (f_i^max - f_i^min)
    """

    def __init__(self, f_mins: np.ndarray, f_maxs: np.ndarray):
        """
        Args:
            f_mins: shape=(3,) 各目标当前最小值。
            f_maxs: shape=(3,) 各目标当前最大值。
        """
        self._mins = np.array(f_mins)
        self._maxs = np.array(f_maxs)
        self._ranges = self._maxs - self._mins
        # 防止除零
        self._ranges[self._ranges < 1e-9] = 1.0

    @classmethod
    def from_archive(
        cls, archive_points: np.ndarray
    ) -> "DynamicNormalizer":
        """从 Pareto 存档点构造动态归一化器。

        Args:
            archive_points: shape=(N,3) 的目标值数组。

        Returns:
            DynamicNormalizer 实例。
        """
        f_mins = archive_points.min(axis=0)
        f_maxs = archive_points.max(axis=0)
        # 增加 10% 边距
        margins = (f_maxs - f_mins) * 0.10
        margins[margins < 1e-6] = 1.0
        return cls(f_mins - margins, f_maxs + margins)

    def normalize(self, values: np.ndarray) -> np.ndarray:
        return (values - self._mins) / self._ranges

    def denormalize(self, normalized: np.ndarray) -> np.ndarray:
        return normalized * self._ranges + self._mins


class WeightedObjective:
    """加权组合目标: H_obj = w₁·f₁^norm + w₂·f₂^norm + w₃·f₃^norm。"""

    def __init__(self, weights: Tuple[float, float, float],
                 normalizer: Normalizer | None = None):
        """
        Args:
            weights: (w₁, w₂, w₃) 偏好权重, 和应为 1。
            normalizer: 归一化策略, 默认物理先验。
        """
        self.weights = np.array(weights, dtype=float)
        if not np.isclose(self.weights.sum(), 1.0):
            raise ValueError(f"Weights must sum to 1, got {self.weights.sum()}")
        self.normalizer = normalizer or PhysicalPriorNormalizer()

    def evaluate(self, f_raw: np.ndarray) -> float:
        """计算加权归一化目标值。

        Args:
            f_raw: shape=(3,) [f₁, f₂, f₃] 原始值。

        Returns:
            加权归一化标量值。
        """
        f_norm = self.normalizer.normalize(f_raw)
        return float(np.dot(self.weights, f_norm))
