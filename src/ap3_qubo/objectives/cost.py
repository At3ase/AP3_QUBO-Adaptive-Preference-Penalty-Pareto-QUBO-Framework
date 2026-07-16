"""
f₃: 成本指数。

f_cost = Σ w_k · c_k
纯线性函数, 映射为 QUBO h_i 偏置场。
"""

from typing import Dict

import numpy as np

from ..physical_params import ELEM, ALL_ELEMENTS, F3_COST_MAX
from .normalization import Normalizer


class WeightedCost:
    """元素成本指数计算器。

    f_cost = Σ w_k · c_k_at%
    其中 w_k 为基于市场价格的相对权重。
    """

    def __init__(self):
        self._weights = ELEM.cost_weights

    def evaluate(self, fractions: Dict[str, float]) -> float:
        """计算给定成分的成本指数。

        Args:
            fractions: 元素 → at% (0~100)。

        Returns:
            成本指数 (相对单位)。
        """
        cost = 0.0
        for elem in ALL_ELEMENTS:
            cost += fractions.get(elem, 0.0) * self._weights[elem]
        return cost

    def evaluate_array(self, c_array: np.ndarray) -> float:
        """从成分数组 [Al,Co,Cr,Fe,Ni,C] 中含 at% 计算成本。

        Args:
            c_array: 元素 at% 值 (0~100 尺度)。
        """
        cost = 0.0
        for i, elem in enumerate(ALL_ELEMENTS):
            cost += c_array[i] * self._weights[elem]
        return cost

    def weight_of(self, elem: str) -> float:
        return self._weights[elem]

    @property
    def max_cost(self) -> float:
        return F3_COST_MAX

    def normalize(self, value: float) -> float:
        """物理先验归一化: f_cost / c_max。"""
        return value / F3_COST_MAX


class CostNormalizer(Normalizer):
    """成本归一化器: f_cost / c_max。"""

    def normalize(self, value: float) -> float:
        return value / F3_COST_MAX

    def denormalize(self, normalized: float) -> float:
        return normalized * F3_COST_MAX
