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

CIM 精度适配: 未加权交叉项系数 < threshold 的项截断为 0。
默认 threshold = 0.01 (CIM 噪声 floor)。
注意: 截断判定基于未加权系数 (0.000972×2^{k+j}), 与 PenaltyFlex
自适应 λ 解耦 —— 见 get_qubo_terms 交叉项循环处的 B-04 修复注释。
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
            cim_mode: 是否启用 CIM 精度截断 (未加权系数 < threshold 的交叉项置 0)。
            cim_threshold: 截断阈值, 默认 CONSTRAINT.cim_noise_floor (0.01)。
                判定与 λω 权重解耦, 详见交叉项循环处的 B-04 修复注释。

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
        # 方案 C-06: c_C × c_Cr 展开后线性项仅一项 1.25·2^k/H_max
        # (1.25 = 5.0 × 0.25, 来自 c_Cr 基线 5.0 与 c_C 编码步长 0.25 的乘积),
        # 系数: 1.25 × 2^k / 64.3125 = 0.01944 × 2^k
        for k_idx in range(3):
            flat_idx = self._c_start + k_idx
            two_pow_k = 1 << k_idx
            coeff = weight * 1.25 * two_pow_k / self._h_max
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
                # 未加权基数系数 (方案 hea_encoding_scheme_v1.13 C-Cr 系数表):
                # base_term = 0.000972 × 2^{k+j}
                base_term = base_coeff * two_pow_k * two_pow_j

                # B-04 修复: CIM 截断判定与【未加权系数】比较, 而非加权后系数。
                # 依据 plan/CIM_Fusion_Evaluation_Report_Revised.md §3.2 方案A:
                # 其截断影响分析表 (threshold=0.01 → 仅损失最小项, 影响 <5%)
                # 基于未加权系数表 (0.000972×2^{k+j}), 即截断集合是约束的
                # 静态结构属性。原实现用加权后系数判定: 初始 λ=0.05、ω=0.5
                # 时最大交叉系数 0.0062 < 0.01, 21 项全灭, P2 违反度反馈恒 0
                # 误导 PenaltyFlex 持续下调 λ (死循环)。与 λ 解耦后, 默认
                # threshold=0.01 下保留 k+j≥4 的 12 项, 截断最小的 9 项
                # (0.000972~0.00778), 与方案影响分析一致。
                if cim_mode and base_term < cim_threshold:
                    continue  # CIM 精度截断 (仅未加权系数低于阈值的小项)

                coeff = weight * base_term

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
