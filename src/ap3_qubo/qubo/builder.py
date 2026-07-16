"""
QUBO 模型构建器 (基于 kaiwu SDK 原生 API)。

使用 kaiwu SDK 的 Binary 变量、quicksum 和 QuboModel 构建完整的
AP³-QUBO 优化模型:

H_total = w1*f1_norm + w2*f2_norm + w3*f3_norm
        + lambda_sum*H_sum(P0)
        + omega_carbide*lambda_carbide*H_carbide(P1)
        + omega_CCr*lambda_CCr*H_CCr(P2)

kaiwu SDK 负责表达式展开 → QUBO 矩阵的完整流程。
"""

import re
from dataclasses import dataclass
from typing import Tuple, List, Optional

import numpy as np
import kaiwu as kw
from kaiwu import quicksum

from ..physical_params import (
    ENCODING, MIEDEMA, ELEM, CONSTRAINT,
    MAIN_ELEMENTS, INTERSTITIAL_ELEMENT, ALL_ELEMENTS,
    F1_NORM_DENOM, F2_NORM_DENOM, F3_COST_MAX,
)

# Kaiwu variable naming scheme (avoids x_0..x_37 collision bug):
#   e{ei}_b{bj}  where ei=element index(0-5), bj=bit position(0-6 for main, 0-2 for C)
# Solution dict key pattern: e0_b0, e1_b3, ...
_VAR_PATTERN = re.compile(r"e(\d+)_b(\d+)")


@dataclass
class QUBOMatrix:
    """QUBO 矩阵数据类。

    Attributes:
        h: shape=(N,) 线性系数向量 (对角线)。
        Q: shape=(N,N) 上三角二次系数矩阵。
        constant_offset: 常数偏移量。
    """
    h: np.ndarray
    Q: np.ndarray
    constant_offset: float = 0.0

    @property
    def num_variables(self) -> int:
        return len(self.h)

    def get_full_matrix(self) -> np.ndarray:
        """返回完整矩阵: diag(h) + Q (上三角, kaiwu 标准格式)。"""
        full = self.Q.copy()
        np.fill_diagonal(full, self.h)
        return full

    def compute_energy(self, bits: np.ndarray) -> float:
        """计算给定比特串的 QUBO 能量。

        E(x) = offset + Σ h_i·x_i + Σ_{i<j} Q_ij·x_i·x_j
        """
        e = float(np.dot(self.h, bits)) + self.constant_offset
        n = self.num_variables
        for i in range(n):
            for j in range(i + 1, n):
                if self.Q[i, j] != 0:
                    e += self.Q[i, j] * bits[i] * bits[j]
        return e


class QUBOBuilder:
    """AP³ QUBO 模型构建器 (kaiwu SDK 原生)。

    使用 kaiwu 的 Binary 变量和表达式 API 构建优化模型。
    kaiwu 自动完成表达式展开到 QUBO 矩阵的转换。
    """

    def __init__(self):
        self._params = ENCODING
        self._step = self._params.step_main  # 0.25
        self._n = self._params.total_variables  # 38

    # =========================================================================
    # 变量创建与命名
    # =========================================================================

    def _create_variables(self) -> List[kw.Binary]:
        """创建 38 个 kaiwu Binary 变量 (安全命名)。

        命名格式: e{元素索引}_b{比特编号}
          e0_b0 ~ e0_b6 → Al bits 0~6
          e1_b0 ~ e1_b6 → Co bits 0~6
          e2_b0 ~ e2_b6 → Cr bits 0~6
          e3_b0 ~ e3_b6 → Fe bits 0~6
          e4_b0 ~ e4_b6 → Ni bits 0~6
          e5_b0 ~ e5_b2 → C  bits 0~2

        避免使用 x_0~x_37 (kaiwu SDK 下划线+数字后缀存在索引碰撞 bug)。
        """
        xs = []
        for ei in range(5):  # Main elements
            for bj in range(self._params.bits_main):
                xs.append(kw.Binary(f"e{ei}_b{bj}"))
        for bj in range(self._params.bits_carbon):  # Carbon
            xs.append(kw.Binary(f"e5_b{bj}"))
        return xs

    @staticmethod
    def parse_var_name(name: str) -> int:
        """将 kaiwu 变量名解析为 flat index (0~37)。

        Args:
            name: 变量名, 如 "e0_b2" 或 "e2_b0"。

        Returns:
            flat index (0~37)。

        Raises:
            ValueError: 无法解析的变量名。
        """
        m = _VAR_PATTERN.match(str(name))
        if not m:
            raise ValueError(f"Cannot parse variable name: {name}")
        ei = int(m.group(1))
        bj = int(m.group(2))
        if ei < 5:
            return ei * 7 + bj
        else:
            return 35 + bj

    @staticmethod
    def var_name_for(elem_idx: int, bit_pos: int) -> str:
        """生成 kaiwu 变量名。

        Args:
            elem_idx: 元素索引 (0=Al, 1=Co, 2=Cr, 3=Fe, 4=Ni, 5=C)。
            bit_pos: 比特位置。

        Returns:
            变量名字符串, 如 "e2_b3"。
        """
        return f"e{elem_idx}_b{bit_pos}"

    # =========================================================================
    # 成分表达式
    # =========================================================================

    def _build_composition_exprs(
        self, xs: List[kw.Binary]
    ) -> dict[str, kw.BinaryExpression]:
        """构建各元素成分的 kaiwu 表达式。

        c_elem = base + step × Σ 2^j × e{ei}_b{j}

        Returns:
            elem → kaiwu BinaryExpression 的字典。
        """
        c_expr = {}
        for i, elem in enumerate(MAIN_ELEMENTS):
            start = i * self._params.bits_main
            terms = quicksum(
                (1 << j) * xs[start + j]
                for j in range(self._params.bits_main)
            )
            c_expr[elem] = self._params.base_main + self._step * terms

        # C element (element index 5)
        start_c = 5 * self._params.bits_main  # 35
        terms_c = quicksum(
            (1 << j) * xs[start_c + j]
            for j in range(self._params.bits_carbon)
        )
        c_expr[INTERSTITIAL_ELEMENT] = (
            self._params.base_carbon + self._step * terms_c
        )

        return c_expr

    # =========================================================================
    # f₁: 混合焓 (Miedema)
    # =========================================================================

    def _build_f1_expr(
        self, c_expr: dict[str, kw.BinaryExpression]
    ) -> kw.BinaryExpression:
        """构建 f₁ (Miedema 混合焓) 的 kaiwu 表达式。

        ΔH_mix = 4·Σ_{i<j} ΔH_ij·c_i·c_j / 10000

        除以 10000 将 at% (如 20.0) 转换为原子比例 (0.20):
        c_i×c_j / 10000 = (c_i_at%/100) × (c_j_at%/100) = c_i × c_j (比例单位)
        """
        expr = None

        # 置换式主元间 (5×5, 10 pairs)
        for a in range(len(MAIN_ELEMENTS)):
            for b in range(a + 1, len(MAIN_ELEMENTS)):
                ea, eb = MAIN_ELEMENTS[a], MAIN_ELEMENTS[b]
                dh = MIEDEMA.get_dh(ea, eb)
                term = 4.0 * dh * c_expr[ea] * c_expr[eb] / 10000.0
                expr = term if expr is None else expr + term

        # 间隙 C-主元交叉项 (5 pairs, 折扣后)
        for elem in MAIN_ELEMENTS:
            dh = MIEDEMA.get_dh("C", elem)  # 已含 γ 折扣
            term = 4.0 * dh * c_expr["C"] * c_expr[elem] / 10000.0
            expr = expr + term

        return expr

    # =========================================================================
    # f₂: 密度 (Vegard)
    # =========================================================================

    def _build_f2_expr(
        self, c_expr: dict[str, kw.BinaryExpression]
    ) -> kw.BinaryExpression:
        """构建 f₂ (Vegard 密度) 的 kaiwu 表达式。

        ρ = Σ c_elem_at% × ρ_elem / 100
        """
        return quicksum(
            c_expr[elem] * ELEM.density_of(elem) / 100.0
            for elem in ALL_ELEMENTS
        )

    # =========================================================================
    # f₃: 成本指数
    # =========================================================================

    def _build_f3_expr(
        self, c_expr: dict[str, kw.BinaryExpression]
    ) -> kw.BinaryExpression:
        """构建 f₃ (成本指数) 的 kaiwu 表达式。

        f_cost = Σ c_elem_at% × w_elem
        """
        return quicksum(
            c_expr[elem] * ELEM.cost_weight_of(elem)
            for elem in ALL_ELEMENTS
        )

    # =========================================================================
    # 约束惩罚项
    # =========================================================================

    def _build_P0_expr(
        self, c_expr: dict[str, kw.BinaryExpression]
    ) -> kw.BinaryExpression:
        """构建 P0 硬约束: λ_sum × (Σc_k - 100)²。

        使用 kaiwu 的 PenaltyMethodConstraint 或手动展开。

        (Σc - 100)² = Σc² - 200·Σc + 10000
        其中 Σc = Σ c_elem, Σc² = (Σc)² 自动展开为二次型。
        """
        sum_c = quicksum(c_expr[elem] for elem in ALL_ELEMENTS)
        return (sum_c - 100.0) * (sum_c - 100.0)

    def _build_P1_expr(
        self, c_expr: dict[str, kw.BinaryExpression]
    ) -> kw.BinaryExpression:
        """构建 P1 软约束: (c_C - 0.8)²。

        纯二次型，精确嵌入 QUBO。
        c_C < 0.8% 时引入的额外惩罚量级远小于 P0, 优化器仍能找到低 C 有效解。
        """
        cc = c_expr[INTERSTITIAL_ELEMENT]
        soft_upper = CONSTRAINT.carbide_soft_upper
        return (cc - soft_upper) * (cc - soft_upper)

    def _build_P2_expr(
        self, c_expr: dict[str, kw.BinaryExpression]
    ) -> kw.BinaryExpression:
        """构建 P2 软约束: c_C × c_Cr / H_max。

        H_max = 1.75 × 36.75 = 64.3125 (%²), 归一化到 [0,1]。
        """
        cc = c_expr[INTERSTITIAL_ELEMENT]
        ccr = c_expr["Cr"]
        return cc * ccr / CONSTRAINT.ccr_h_max

    # =========================================================================
    # 主构建方法
    # =========================================================================

    def build_model(
        self,
        weights: Tuple[float, float, float],
        lambda_carbide: float | None = None,
        lambda_ccr: float | None = None,
        lambda_sum: float | None = None,
    ) -> kw.QuboModel:
        """使用 kaiwu SDK 构建完整的 QuboModel。

        H = w1·f1_norm + w2·f2_norm + w3·f3_norm
          + λ_sum·H_P0 + ω₁·λ₁·H_P1 + ω₂·λ₂·H_P2

        Args:
            weights: (w1, w2, w3) 偏好权重, 和应为 1。
            lambda_carbide: P1 惩罚系数。
            lambda_ccr: P2 惩罚系数。
            lambda_sum: P0 固定惩罚系数。

        Returns:
            kaiwu QuboModel, 可直接用于 QuboSolver.solve_qubo()。
        """
        if lambda_carbide is None:
            lambda_carbide = CONSTRAINT.lambda_carbide_init
        if lambda_ccr is None:
            lambda_ccr = CONSTRAINT.lambda_ccr_init
        if lambda_sum is None:
            lambda_sum = CONSTRAINT.lambda_sum_fixed

        # 归一化权重
        w = np.array(weights, dtype=float)
        if abs(w.sum() - 1.0) > 1e-9:
            w = w / w.sum()

        # ===== Step 1: 创建变量 =====
        xs = self._create_variables()

        # ===== Step 2: 构建成分表达式 =====
        c_expr = self._build_composition_exprs(xs)

        # ===== Step 3: 构建目标函数表达式 =====
        f1_raw = self._build_f1_expr(c_expr)
        f2_raw = self._build_f2_expr(c_expr)
        f3_raw = self._build_f3_expr(c_expr)

        # 归一化 + 加权
        obj_expr = (
            w[0] * f1_raw / F1_NORM_DENOM
            + w[1] * f2_raw / F2_NORM_DENOM
            + w[2] * f3_raw / F3_COST_MAX
        )

        # ===== Step 4: 构建约束惩罚表达式 =====
        p0_expr = lambda_sum * self._build_P0_expr(c_expr)
        p1_expr = (
            CONSTRAINT.omega_carbide
            * lambda_carbide
            * self._build_P1_expr(c_expr)
        )
        p2_expr = (
            CONSTRAINT.omega_ccr
            * lambda_ccr
            * self._build_P2_expr(c_expr)
        )

        # ===== Step 5: 组装 QuboModel =====
        total_expr = obj_expr + p0_expr + p1_expr + p2_expr
        model = kw.QuboModel()
        model.set_objective(total_expr)

        return model

    # =========================================================================
    # 便捷方法: 直接构建 QUBOMatrix (用于分析和调试)
    # =========================================================================

    def build(
        self,
        weights: Tuple[float, float, float],
        lambda_carbide: float | None = None,
        lambda_ccr: float | None = None,
        lambda_sum: float | None = None,
        cim_mode: bool = False,
        cim_threshold: float | None = None,
    ) -> QUBOMatrix:
        """构建 QUBOMatrix (便捷方法, 内部通过 kaiwu QuboModel)。

        Args:
            weights: (w1, w2, w3) 偏好权重。
            lambda_carbide: P1 惩罚系数。
            lambda_ccr: P2 惩罚系数。
            lambda_sum: P0 固定惩罚系数。
            cim_mode: 是否启用 CIM 精度截断。
            cim_threshold: P2 截断阈值。

        Returns:
            QUBOMatrix(h, Q, offset)。
        """
        model = self.build_model(
            weights=weights,
            lambda_carbide=lambda_carbide,
            lambda_ccr=lambda_ccr,
            lambda_sum=lambda_sum,
        )

        full_mat = model.get_matrix()
        n = full_mat.shape[0]

        # 提取 h (对角线) 和 Q (严格上三角)
        h = np.diag(full_mat).copy()
        Q = np.triu(full_mat, 1)

        # CIM 截断 (如果需要)
        if cim_mode:
            if cim_threshold is None:
                cim_threshold = CONSTRAINT.cim_noise_floor
            # 对 Q 矩阵中系数 < threshold 的项置零
            mask = np.abs(Q) < cim_threshold
            Q[mask] = 0.0
            # 同步更新 h (对角线截断)
            mask_h = np.abs(h) < cim_threshold
            h[mask_h] = 0.0

        offset = float(model.get_offset())

        return QUBOMatrix(h=h, Q=Q, constant_offset=offset)

    def get_variable_names(self) -> List[str]:
        """返回 kaiwu 变量名列表 ['e0_b0', 'e0_b1', ..., 'e5_b2']."""
        names = []
        for ei in range(5):
            for bj in range(self._params.bits_main):
                names.append(f"e{ei}_b{bj}")
        for bj in range(self._params.bits_carbon):
            names.append(f"e5_b{bj}")
        return names
