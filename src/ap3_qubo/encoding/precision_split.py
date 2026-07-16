"""
PrecisionSplit 分层精度二进制编码器/解码器。

将 6 元素成分 (Al, Co, Cr, Fe, Ni, C) 编码为 38 个二元变量:
  - 5 主元 × 7 比特 = 35
  - C × 3 比特 = 3
所有元素共享统一步长 0.25 at%，确保成分和=100% 数学可精确满足。
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

import numpy as np

from ..physical_params import (
    ENCODING,
    MAIN_ELEMENTS,
    INTERSTITIAL_ELEMENT,
    ALL_ELEMENTS,
)


@dataclass
class Composition:
    """一个高熵合金成分点。

    Attributes:
        fractions: 元素名 -> 原子百分比 (at%) 的映射。
    """

    fractions: Dict[str, float]

    def __post_init__(self):
        # 确保所有元素都存在
        for elem in ALL_ELEMENTS:
            if elem not in self.fractions:
                self.fractions[elem] = 0.0

    def __getitem__(self, elem: str) -> float:
        return self.fractions.get(elem, 0.0)

    @property
    def total(self) -> float:
        return sum(self.fractions.values())

    @property
    def is_feasible(self, tolerance: float = 1.0) -> bool:
        """检查成分和是否在 100% ± tolerance 范围内。"""
        return abs(self.total - 100.0) <= tolerance

    @property
    def main_elements_only(self) -> Dict[str, float]:
        """仅返回 5 种主元的成分 (用于 VEC/δ/Δχ 计算)。"""
        return {e: self.fractions[e] for e in MAIN_ELEMENTS}

    @property
    def carbon_fraction(self) -> float:
        return self.fractions["C"]

    def to_array(self) -> np.ndarray:
        """返回 6 元素数组 [Al, Co, Cr, Fe, Ni, C]。"""
        return np.array([self.fractions[e] for e in ALL_ELEMENTS], dtype=float)

    def __repr__(self) -> str:
        parts = ", ".join(f"{e}={self.fractions[e]:.2f}%" for e in ALL_ELEMENTS)
        return f"Composition({parts})"


class PrecisionSplitEncoder:
    """将连续成分编码为 38 比特二进制向量。

    编码公式:
      c_i = base_i + step × Σ 2^j × b_{i,j}
    """

    def __init__(self):
        self._params = ENCODING

    @property
    def num_variables(self) -> int:
        return self._params.total_variables  # 38

    @property
    def num_bits_main(self) -> int:
        return self._params.bits_main  # 7

    @property
    def num_bits_carbon(self) -> int:
        return self._params.bits_carbon  # 3

    def encode(self, composition: Composition) -> np.ndarray:
        """将成分编码为 38 比特二进制向量。

        Args:
            composition: 合金成分。

        Returns:
            shape=(38,) 的整数数组 (0 或 1)。

        Raises:
            ValueError: 如果某元素成分无法精确表示为步长的整数倍。
        """
        bits = np.zeros(self.num_variables, dtype=np.int8)
        step = self._params.step_main

        for i, elem in enumerate(MAIN_ELEMENTS):
            base = self._params.base_main
            c = composition[elem]
            k = round((c - base) / step)
            if abs(k * step + base - c) > 1e-9:
                raise ValueError(
                    f"{elem} 成分 {c:.4f}% 无法表示为步长 {step}% 的整数倍 "
                    f"(最接近: {base + k * step:.4f}%)"
                )
            if not (0 <= k < (1 << self.num_bits_main)):
                raise ValueError(
                    f"{elem} 编码值 k={k} 超出 [0, {self._params.main_levels - 1}] 范围"
                )
            # 写入 7 比特
            start = i * self.num_bits_main
            for j in range(self.num_bits_main):
                bits[start + j] = (k >> j) & 1

        # C 元素 (3 比特)
        c_c = composition[INTERSTITIAL_ELEMENT]
        base_c = self._params.base_carbon
        k_c = round((c_c - base_c) / step)
        if abs(k_c * step + base_c - c_c) > 1e-9:
            raise ValueError(
                f"C 成分 {c_c:.4f}% 无法表示为步长 {step}% 的整数倍"
            )
        if not (0 <= k_c < (1 << self.num_bits_carbon)):
            raise ValueError(
                f"C 编码值 k_C={k_c} 超出 [0, {self._params.carbon_levels - 1}] 范围"
            )
        start_c = len(MAIN_ELEMENTS) * self.num_bits_main
        for j in range(self.num_bits_carbon):
            bits[start_c + j] = (k_c >> j) & 1

        return bits

    def encode_value(self, elem: str, k_value: int) -> np.ndarray:
        """给定某元素的编码整数值 k，返回该元素的比特片段。

        Args:
            elem: 元素名。
            k_value: 编码整数值 (主元 0~127, C 0~7)。

        Returns:
            该元素对应的比特数组。
        """
        if elem in MAIN_ELEMENTS:
            n_bits = self.num_bits_main
        elif elem == INTERSTITIAL_ELEMENT:
            n_bits = self.num_bits_carbon
        else:
            raise ValueError(f"Unknown element: {elem}")

        bits = np.zeros(n_bits, dtype=np.int8)
        for j in range(n_bits):
            bits[j] = (k_value >> j) & 1
        return bits

    def element_range(self, elem: str) -> Tuple[float, float]:
        """返回某元素的编码取值范围 (min%, max%)。"""
        if elem in MAIN_ELEMENTS:
            return (self._params.main_min, self._params.main_max)
        elif elem == INTERSTITIAL_ELEMENT:
            return (self._params.carbon_min, self._params.carbon_max)
        raise ValueError(f"Unknown element: {elem}")


class PrecisionSplitDecoder:
    """将 38 比特二进制向量解码为 Composition。"""

    def __init__(self):
        self._params = ENCODING

    def decode(self, bits: np.ndarray) -> Composition:
        """从 38 比特二进制数组解码成分。

        Args:
            bits: shape=(38,) 的 0/1 数组。

        Returns:
            Composition 对象。
        """
        if len(bits) != self._params.total_variables:
            raise ValueError(
                f"Expected {self._params.total_variables} bits, got {len(bits)}"
            )

        fractions = {}
        step = self._params.step_main

        for i, elem in enumerate(MAIN_ELEMENTS):
            base = self._params.base_main
            start = i * self._params.bits_main
            k = 0
            for j in range(self._params.bits_main):
                if bits[start + j]:
                    k += (1 << j)
            fractions[elem] = base + step * k

        # C
        base_c = self._params.base_carbon
        start_c = len(MAIN_ELEMENTS) * self._params.bits_main
        k_c = 0
        for j in range(self._params.bits_carbon):
            if bits[start_c + j]:
                k_c += (1 << j)
        fractions[INTERSTITIAL_ELEMENT] = base_c + step * k_c

        return Composition(fractions=fractions)

    def decode_ising_solution(self, spin_vector: np.ndarray) -> Composition:
        """从 Ising 自旋配置 (+1/-1) 解码成分。

        Ising spin σ_i ∈ {+1, -1} → QUBO bit x_i = (σ_i + 1) / 2.
        """
        bits = ((np.array(spin_vector) + 1) // 2).astype(np.int8)
        return self.decode(bits)

    def decode_kaiwu_solution(
        self, solution_dict: Dict, variables: List
    ) -> Composition:
        """从 kaiwu SDK 解字典解码成分。

        Args:
            solution_dict: kaiwu 返回的 {variable: value} 字典。
            variables: 变量列表 (x_0 ~ x_37)。

        Returns:
            Composition 对象。
        """
        bits = np.zeros(self._params.total_variables, dtype=np.int8)
        for i, var in enumerate(variables):
            bits[i] = int(solution_dict.get(var, 0))
        return self.decode(bits)
