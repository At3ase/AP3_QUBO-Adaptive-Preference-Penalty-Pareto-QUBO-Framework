"""
P0 硬约束: 成分和 = 100% (质量守恒律)。

H_sum = λ_sum × (S_var / 75 - 1)²
     = (λ_sum / 5625) × [S_var² - 150·S_var + 5625]

其中 S_var = 0.25% × Σ 2^j·x_j (变量贡献部分),
S_var' = S_var / 75 是归一化值。

等原子比 (K=300) 时 S_var = 75, S_var' = 1, 惩罚 = 0.

P0 为硬约束: λ_sum 固定为常数 (10~20), 不参与 PenaltyFlex 自适应。
"""

import numpy as np

from ..physical_params import CONSTRAINT, ENCODING, MAIN_ELEMENTS, INTERSTITIAL_ELEMENT
from ..encoding.variable_index import VariableMapper


class SumTo100Constraint:
    """成分和=100% 硬约束 (P0)。

    返回 QUBO 矩阵 (h_vec, Q_mat) 的惩罚贡献。
    """

    def __init__(self, lambda_sum: float | None = None):
        """
        Args:
            lambda_sum: P0 固定惩罚系数, 默认使用 CONSTRAINT.lambda_sum_fixed。
        """
        self.lambda_sum = lambda_sum if lambda_sum is not None else CONSTRAINT.lambda_sum_fixed
        self._mapper = VariableMapper()
        self._step = ENCODING.step_main  # 0.25

    @property
    def scale_factor(self) -> float:
        """λ_sum / 5625: 惩罚项系数缩放因子。"""
        return self.lambda_sum / 5625.0

    def get_qubo_terms(self) -> tuple[np.ndarray, np.ndarray]:
        """计算 P0 约束的 QUBO 矩阵贡献。

        展开 H_sum = (λ/5625) × [S_var² - 150·S_var + 5625]

        S_var = 0.25 × Σ coef_i × x_i
        其中 coef_i = 2^{bit_pos} (该变量在其元素内的位权重)

        S_var² 展开后:
          - 自平方项: x_i·x_i = x_i → 线性项, 系数 = (λ/5625) × (0.25 × coef_i)²
          - 交叉项: 2·coef_i·coef_j·x_i·x_j → QUBO 二次项,
                    系数 = (λ/5625) × 2 × (0.25)² × coef_i × coef_j

        -150·S_var → 线性项, 系数 = (λ/5625) × (-150) × 0.25 × coef_i

        常数 5625 项 → offset, 不影响优化方向, 可忽略。

        Returns:
            (h_vec, Q_mat): h_vec shape=(38,), Q_mat shape=(38,38) 上三角。
        """
        n = self._mapper.total_variables
        h_vec = np.zeros(n, dtype=float)
        Q_mat = np.zeros((n, n), dtype=float)

        sf = self.scale_factor
        step = self._step

        # 预计算每个变量的 coefficient (0.25 × 2^bit_pos)
        variable_coeffs = np.zeros(n, dtype=float)
        for i in range(n):
            elem, bit_pos = self._mapper.flat_to_element_bit(i)
            variable_coeffs[i] = step * (1 << bit_pos)

        # ===== 线性项: S_var² 的自平方项 + (-150·S_var) 项 =====
        for i in range(n):
            ci = variable_coeffs[i]
            # 自平方: (λ/5625) × ci²
            # -150 项: (λ/5625) × (-150) × ci
            h_vec[i] = sf * (ci * ci - 150.0 * ci)

        # ===== 二次项: S_var² 的交叉项 =====
        for i in range(n):
            ci = variable_coeffs[i]
            for j in range(i + 1, n):
                cj = variable_coeffs[j]
                Q_mat[i, j] = sf * 2.0 * ci * cj

        # 常数为 sf * 5625, 仅影响 offset, 忽略

        return h_vec, Q_mat

    def evaluate_penalty(self, fractions: dict[str, float]) -> float:
        """直接评估 P0 惩罚值 (用于诊断)。

        Args:
            fractions: 元素 → at% 字典。

        Returns:
            λ_sum × (Σc_k/100 - 1)² (使用比例形式)。
        """
        total = sum(fractions.values()) / 100.0  # → [0, ~1.6]
        return self.lambda_sum * (total - 1.0) ** 2

    def evaluate_normalized(self, fractions: dict[str, float]) -> float:
        """评估归一化形式的 P0 惩罚。

        S_var' = Σ c_k / 75 (扣除基线 25% 后)
        惩罚 = λ_sum × (S_var' - 1)²
        """
        total = sum(fractions.values())  # at%
        s_var = total - 5 * ENCODING.base_main  # 扣除 5×5.0=25% baseline
        s_var_norm = s_var / 75.0
        return self.lambda_sum * (s_var_norm - 1.0) ** 2
