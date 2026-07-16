"""
P2 软约束: C-Cr 耦合最小化。

H_CCr = c_C × c_Cr / H_max  (归一化到 [0,1])
H_max = 1.75 × 36.75 = 64.3125  (%²)

展开 (以 at% 数值代入):
  c_C = 0.25 × Σ_{k=0}^{2} 2^k × x_{35+k}      ∈ [0, 1.75]
  c_Cr = 5.0 + 0.25 × Σ_{j=0}^{6} 2^j × x_{14+j}  ∈ [5.0, 36.75]

  c_C × c_Cr = 1.25·Σ(2^k x_{35+k}) + 0.0625·ΣΣ(2^{k+j} x_{35+k} x_{14+j})

归一化后:
  H_CCr = 0.01944·Σ 2^k x_{35+k}   (线性项, C 的 h_i)
        + 0.000972·ΣΣ 2^{k+j} x_{35+k} x_{14+j}  (交叉项, Q[35+k][14+j])

CIM 精度适配: 交叉项系数 < threshold 的项截断为 0。
默认 threshold = 0.01 (CIM 噪声 floor)。
"""

import numpy as np

from ..physical_params import CONSTRAINT, ENCODING
from ..encoding.variable_index import VariableMapper


class CCrCouplingConstraint:
    """C-Cr 耦合最小化软约束 (P2)。"""

    def __init__(self):
        self._mapper = VariableMapper()
        self._h_max = CONSTRAINT.ccr_h_max  # 64.3125
        self._cr_start = self._mapper.cr_start  # 14
        self._c_start = self._mapper.carbon_start  # 35
        self._step = ENCODING.step_carbon  # 0.25

    def get_qubo_terms(
        self,
        lambda_ccr: float,
        omega: float | None = None,
        cim_mode: bool = False,
        cim_threshold: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """计算 P2 约束的 QUBO 贡献。

        Args:
            lambda_ccr: PenaltyFlex 输出的自适应 λ。
            omega: 外部偏好权重, 默认 0.5。
            cim_mode: 是否启用 CIM 精度截断 (系数 < threshold 的项置 0)。
            cim_threshold: 截断阈值, 默认 CONSTRAINT.cim_noise_floor (0.01)。

        Returns:
            (h_vec, Q_mat): shape=(38,) 和 (38,38)。
        """
        if omega is None:
            omega = CONSTRAINT.omega_ccr
        if cim_threshold is None:
            cim_threshold = CONSTRAINT.cim_noise_floor

        weight = lambda_ccr * omega
        n = self._mapper.total_variables

        h_vec = np.zeros(n, dtype=float)
        Q_mat = np.zeros((n, n), dtype=float)

        # ===== 线性项 (C 的 h_i) =====
        # 系数: 1.25 × 2^k / 64.3125 = 0.01944 × 2^k
        for k_idx in range(3):
            flat_idx = self._c_start + k_idx
            two_pow_k = 1 << k_idx
            coeff = weight * 1.25 * two_pow_k / self._h_max
            h_vec[flat_idx] += coeff

        # Cr 线性项 (来自 c_Cr 的基线 5.0):
        # c_C × 5.0 / H_max → 对每个 C 变量, h_i += λω × 5.0 × 0.25 × 2^k / H_max
        for k_idx in range(3):
            flat_idx = self._c_start + k_idx
            two_pow_k = 1 << k_idx
            coeff = weight * 5.0 * 0.25 * two_pow_k / self._h_max
            h_vec[flat_idx] += coeff

        # ===== 交叉项 (Q[35+k][14+j]) =====
        # 系数: 0.0625 × 2^{k+j} / 64.3125 = 0.000972 × 2^{k+j}
        base_coeff = 0.0625 / self._h_max  # ≈ 0.000972

        for k_idx in range(3):  # C bits: 0,1,2
            c_flat = self._c_start + k_idx
            two_pow_k = 1 << k_idx
            for j_idx in range(7):  # Cr bits: 0,...,6
                cr_flat = self._cr_start + j_idx
                two_pow_j = 1 << j_idx
                coeff = weight * base_coeff * two_pow_k * two_pow_j

                if cim_mode and coeff < cim_threshold:
                    continue  # CIM 精度截断

                # 确保上三角格式
                if c_flat < cr_flat:
                    Q_mat[c_flat, cr_flat] += coeff
                else:
                    Q_mat[cr_flat, c_flat] += coeff

        return h_vec, Q_mat

    def evaluate_penalty(self, c_carbon: float, c_cr: float,
                         lambda_ccr: float,
                         omega: float | None = None) -> float:
        """直接评估 P2 惩罚值。

        Args:
            c_carbon: C 含量 (at%, 如 0.5)。
            c_cr: Cr 含量 (at%, 如 20.0)。
            lambda_ccr: 当前自适应 λ。
            omega: 偏好权重。

        Returns:
            归一化惩罚值 λω × c_C × c_Cr / 64.3125。
        """
        if omega is None:
            omega = CONSTRAINT.omega_ccr
        return lambda_ccr * omega * c_carbon * c_cr / self._h_max

    @property
    def h_max(self) -> float:
        return self._h_max
