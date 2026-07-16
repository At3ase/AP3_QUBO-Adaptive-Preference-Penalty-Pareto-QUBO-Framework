"""
P1 软约束: 碳化物抑制。

H_carbide = (c_C - 0.8%)²

采用纯二次型 (而非分段 max(0, c_C-0.8)²), 可精确嵌入 QUBO。
代价: c_C < 0.8% 时引入额外正惩罚 (将优化器推向 c_C≈0.8%),
但该额外惩罚量级 (0.09~0.64) 远小于 P0 硬约束 (>10),
且低 C 成分在 ΔH_mix 中本身有利。

c_C = 0.0% + 0.25% × Σ_{j=0}^{2} 2^j × x_{35+j}
"""

import numpy as np

from ..physical_params import CONSTRAINT, ENCODING
from ..encoding.variable_index import VariableMapper


class CarbideConstraint:
    """碳化物抑制软约束 (P1)。

    惩罚形式: (c_C - 0.8)²  (c_C 以 at% 数值代入, 不含%号)
    展开: c_C² - 1.6·c_C + 0.64
    """

    def __init__(self):
        self._mapper = VariableMapper()
        self._step = ENCODING.step_carbon  # 0.25
        self._soft_upper = CONSTRAINT.carbide_soft_upper  # 0.8
        self._c_indices = self._mapper.get_carbon_indices()  # [35, 36, 37]

    def get_qubo_terms(self, lambda_carbide: float, omega: float | None = None) -> tuple[np.ndarray, np.ndarray]:
        """计算 P1 约束的 QUBO 矩阵贡献。

        c_C = 0.25 × Σ 2^k × x_{35+k}

        H_carbide = λ_carbide × ω × (c_C - 0.8)²
                  = λω × [c_C² - 1.6·c_C + 0.64]

        c_C² 展开:
          - 自平方: (0.25×2^k)² × x_k = 0.0625 × 2^{2k} × x_k → 线性项
          - 交叉: 2 × 0.25² × 2^{k+m} × x_k × x_m = 0.125 × 2^{k+m} × x_k × x_m → 二次项

        -1.6·c_C: -1.6 × 0.25 × 2^k × x_k = -0.4 × 2^k × x_k → 线性项

        常数 0.64 → offset, 忽略。

        Args:
            lambda_carbide: PenaltyFlex 输出的自适应 λ。
            omega: 外部偏好权重, 默认使用 CONSTRAINT.omega_carbide。

        Returns:
            (h_vec, Q_mat): shape=(38,) 和 (38,38)。
        """
        if omega is None:
            omega = CONSTRAINT.omega_carbide

        weight = lambda_carbide * omega
        n = self._mapper.total_variables
        h_vec = np.zeros(n, dtype=float)
        Q_mat = np.zeros((n, n), dtype=float)

        # c_C² 自平方: λω × 0.0625 × 2^{2k}
        for idx, k in enumerate(self._c_indices):
            two_pow_k = 1 << idx  # 2^k
            h_vec[k] += weight * 0.0625 * (two_pow_k ** 2)

        # -1.6·c_C: λω × (-0.4) × 2^k
        for idx, k in enumerate(self._c_indices):
            two_pow_k = 1 << idx
            h_vec[k] += weight * (-0.4) * two_pow_k

        # c_C² 交叉项: λω × 0.125 × 2^{k+m}
        n_c = len(self._c_indices)
        for i in range(n_c):
            ki = self._c_indices[i]
            for j in range(i + 1, n_c):
                kj = self._c_indices[j]
                coeff = weight * 0.125 * (1 << i) * (1 << j)
                Q_mat[ki, kj] += coeff

        return h_vec, Q_mat

    def evaluate_penalty(self, c_carbon: float, lambda_carbide: float,
                         omega: float | None = None) -> float:
        """直接评估 P1 惩罚值 (at% 尺度)。

        Args:
            c_carbon: C 含量 (at%, 如 0.5)。
            lambda_carbide: 当前自适应 λ。
            omega: 偏好权重。

        Returns:
            惩罚值。
        """
        if omega is None:
            omega = CONSTRAINT.omega_carbide
        return lambda_carbide * omega * (c_carbon - self._soft_upper) ** 2

    @property
    def soft_upper(self) -> float:
        return self._soft_upper
