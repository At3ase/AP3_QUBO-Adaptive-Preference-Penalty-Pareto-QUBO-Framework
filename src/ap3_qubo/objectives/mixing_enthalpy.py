"""
f₁: 混合焓代理 (Miedema 模型)。

ΔH_mix = ΔH_substitutional + ΔH_C_interstitial
天然 QUBO 二次型，无需近似或辅助变量。
"""

from typing import Dict

import numpy as np

from ..physical_params import MIEDEMA, MAIN_ELEMENTS, ALL_ELEMENTS, F1_NORM_DENOM
from .normalization import Normalizer


class MixingEnthalpy:
    """Miedema 混合焓计算器。

    ΔH_mix = 4 Σ_{i<j} ΔH_ij · c_i · c_j  (置换项)
           + 4 Σ_i γ · ΔH_{i,C} · c_i · c_C  (间隙 C 折扣项)
    """

    def __init__(self, gamma: float | None = None):
        self.gamma = gamma if gamma is not None else MIEDEMA.gamma_discount

    def evaluate(self, fractions: Dict[str, float]) -> float:
        """计算给定成分的混合焓 (kJ/mol)。

        Args:
            fractions: 元素 → at% (0~100 尺度, 如 Al=20.0 表示 20 at%)。

        Returns:
            ΔH_mix (kJ/mol), 负值越负表示越稳定。
        """
        c = {e: fractions.get(e, 0.0) / 100.0 for e in ALL_ELEMENTS}  # → [0,1]

        # 置换式主元混合焓
        dh_sub = 0.0
        n_main = len(MAIN_ELEMENTS)
        for i in range(n_main):
            for j in range(i + 1, n_main):
                ei, ej = MAIN_ELEMENTS[i], MAIN_ELEMENTS[j]
                dh_sub += MIEDEMA.get_dh(ei, ej) * c[ei] * c[ej]

        # 间隙 C 的有效混合焓 (折扣后)
        dh_int = 0.0
        for elem in MAIN_ELEMENTS:
            dh_int += MIEDEMA.get_dh("C", elem) * c[elem] * c["C"]

        return 4.0 * dh_sub + 4.0 * dh_int

    def evaluate_array(self, c_array: np.ndarray) -> float:
        """从成分数组计算混合焓。

        Args:
            c_array: shape=(6,) 元素比例 [0,1]。

        Returns:
            ΔH_mix (kJ/mol)。
        """
        dh_sub = 0.0
        for i in range(5):
            for j in range(i + 1, 5):
                dh_sub += MIEDEMA.get_dh(MAIN_ELEMENTS[i], MAIN_ELEMENTS[j]) * c_array[i] * c_array[j]

        dh_int = 0.0
        for i in range(5):
            dh_int += MIEDEMA.get_dh("C", MAIN_ELEMENTS[i]) * c_array[i] * c_array[5]

        return 4.0 * dh_sub + 4.0 * dh_int

    def normalize(self, value: float) -> float:
        """物理先验归一化: f₁ / 30。"""
        return value / F1_NORM_DENOM

    def qubo_linear_coefficient(
        self, elem: str, bit_pos: int, step: float
    ) -> float:
        """计算 f₁ 中某变量的线性系数 (归一化后)。

        由 c_i = base_i + step × Σ 2^j·b_{i,j} 展开后，
        自平方项 x_i² = x_i 贡献线性系数。
        """
        # 此方法用于逐项组装 QUBO, 返回归一化后的 h_i 贡献
        # 具体展开见 qubo/builder.py 中的 assemble
        return 0.0  # 占位; 实际系数在 builder 中统一计算

    @property
    def equiatomic_value(self) -> float:
        """等原子比 AlCoCrFeNi (C=0) 的参考值。"""
        return MIEDEMA.dh_equiatomic  # -12.32 kJ/mol


class MixingEnthalpyNormalizer(Normalizer):
    """混合焓归一化器: f₁ / 30 (物理先验)。"""

    def normalize(self, value: float) -> float:
        return value / F1_NORM_DENOM

    def denormalize(self, normalized: float) -> float:
        return normalized * F1_NORM_DENOM
