"""
f₂: 密度 (Vegard 定律)。

ρ = Σ c_i · ρ_i
纯线性函数, 在 QUBO 中映射为 h_i 偏置场。
"""

from typing import Dict

import numpy as np

from ..physical_params import ELEM, ALL_ELEMENTS, F2_NORM_DENOM
from .normalization import Normalizer


class VegardDensity:
    """Vegard 定律密度计算器。

    ρ = Σ_{i} c_i · ρ_i
    其中 c_i 为原子比例 (0~1), ρ_i 为纯元素密度 (g/cm³)。
    """

    def __init__(self):
        self._densities = ELEM.densities

    def evaluate(self, fractions: Dict[str, float]) -> float:
        """计算给定成分的 Vegard 密度。

        Args:
            fractions: 元素 → at% (0~100)。

        Returns:
            密度 (g/cm³)。
        """
        rho = 0.0
        for elem in ALL_ELEMENTS:
            rho += (fractions.get(elem, 0.0) / 100.0) * self._densities[elem]
        return rho

    def evaluate_array(self, c_array: np.ndarray) -> float:
        """从成分数组 [Al,Co,Cr,Fe,Ni,C] ∈ [0,1] 计算密度。"""
        rho = 0.0
        for i, elem in enumerate(ALL_ELEMENTS):
            rho += c_array[i] * self._densities[elem]
        return rho

    def linear_coefficient(self, elem: str) -> float:
        """返回该元素的密度线性系数 ρ_i/100 (转换为 at% 输入)。

        密度 = Σ (c_i_at%/100) × ρ_i
        对于 c_i_at% = 20.0, 贡献 = (20.0/100) × ρ_i
        """
        return self._densities[elem] / 100.0

    def normalize(self, value: float) -> float:
        """物理先验归一化: ρ / 10。"""
        return value / F2_NORM_DENOM

    @property
    def equiatomic_value(self) -> float:
        """等原子比 AlCoCrFeNi (C=0) 的密度 ≈ 7.11 g/cm³。"""
        return sum(self._densities[e] for e in ["Al", "Co", "Cr", "Fe", "Ni"]) / 5.0


class DensityNormalizer(Normalizer):
    """密度归一化器: ρ / 10 (物理先验)。"""

    def normalize(self, value: float) -> float:
        return value / F2_NORM_DENOM

    def denormalize(self, normalized: float) -> float:
        return normalized * F2_NORM_DENOM
