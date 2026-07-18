"""
QUBO 模型构建器 (基于 kaiwu SDK 原生 API)。

使用 kaiwu SDK 的 Binary 变量、quicksum 和 QuboModel 构建完整的
AP³-QUBO 优化模型:

H_total = w1*f1_norm + w2*f2_norm + w3*f3_norm
        + lambda_sum*H_sum(P0)
        + omega_carbide*lambda_carbide*H_carbide(P1)
        + omega_CCr*lambda_CCr*H_CCr(P2)

kaiwu SDK 负责表达式展开 → QUBO 矩阵的完整流程。

P0-7 修复：kaiwu 采用延迟导入，实验包可在无 kaiwu 环境下正常导入；
仅在首次调用 QUBO 构建/求解方法时才要求 kaiwu 已安装。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Tuple, List, Optional

import numpy as np

from ..physical_params import (
    ENCODING, MIEDEMA, ELEM, CONSTRAINT,
    MAIN_ELEMENTS, INTERSTITIAL_ELEMENT, ALL_ELEMENTS,
    F1_NORM_DENOM, F2_NORM_DENOM, F3_COST_MAX,
)

# ---- kaiwu 延迟导入（P0-7 修复）------------------------------------------
# 模块级不 import kaiwu，避免硬依赖阻断 ap3_qubo.experiments 等上层
# 包的导入链。首次调用 _ensure_kaiwu() 时完成实际导入。
_kw = None
_quicksum = None


def _ensure_kaiwu():
    """延迟导入 kaiwu SDK。

    Raises:
        ImportError: kaiwu 未安装时给出明确安装指引。
    """
    global _kw, _quicksum
    if _kw is None:
        try:
            import kaiwu as _kaiwu_mod  # noqa: F811
            _kw = _kaiwu_mod
            _quicksum = _kaiwu_mod.quicksum
        except ImportError:
            raise ImportError(
                "kaiwu SDK 未安装。QUBO 模型构建与求解需要 kaiwu>=1.3。\n"
                "安装命令：pip install kaiwu>=1.3\n"
                "（实验模块可在无 kaiwu 环境下导入和检查，"
                "但调用 build_model() / solve() 前必须先安装。）"
            )
    return _kw, _quicksum

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

    支持三种编码模式:
      - "precision_split_38": AP³ 默认，主元各 7 比特 + C 3 比特（38 变量）
      - "unified_48": 统一编码，6 元素各 8 比特（48 变量）
      - "unified_38": 统一编码，6 元素用满 38 比特（主元 6 比特 + C 8 比特）
    """

    # 编码配置表
    _ENCODING_CONFIGS = {
        "precision_split_38": {
            "n_vars": 38,
            "bits_main": 7,
            "bits_carbon": 3,
            "base_main": 5.0,
            "base_carbon": 0.0,
            "step_main": 0.25,
            "step_carbon": 0.25,
            "element_ranges": "precision",  # element-specific bounds
        },
        "unified_48": {
            "n_vars": 48,
            "bits_main": 8,
            "bits_carbon": 8,
            "base_main": 5.0,
            "base_carbon": 5.0,
            "step_main": 30.0 / 255.0,      # (35-5)/(2^8-1) ≈ 0.1176
            "step_carbon": 30.0 / 255.0,
            "element_ranges": "hea_uniform",  # all in [5, 35]
        },
        "unified_38": {
            "n_vars": 38,
            "bits_main": 6,
            "bits_carbon": 8,
            "base_main": 5.0,
            "base_carbon": 5.0,
            "step_main": 30.0 / 63.0,       # (35-5)/(2^6-1) ≈ 0.476
            "step_carbon": 30.0 / 255.0,
            "element_ranges": "hea_uniform",
        },
    }

    def __init__(
        self,
        encoding_type: str = "precision_split_38",
        gamma_discount: float | None = None,
    ):
        if encoding_type not in self._ENCODING_CONFIGS:
            raise ValueError(
                f"Unknown encoding_type: {encoding_type}. "
                f"Supported: {list(self._ENCODING_CONFIGS.keys())}"
            )
        self._encoding_type = encoding_type
        cfg = self._ENCODING_CONFIGS[encoding_type]
        self._n_vars: int = cfg["n_vars"]
        self._bits_main: int = cfg["bits_main"]
        self._bits_carbon: int = cfg["bits_carbon"]
        self._base_main: float = cfg["base_main"]
        self._base_carbon: float = cfg["base_carbon"]
        self._step_main: float = cfg["step_main"]
        self._step_carbon: float = cfg["step_carbon"]
        self._element_ranges: str = cfg["element_ranges"]
        # γ 折扣因子（None 使用 MIEDEMA 默认值，否则覆盖）
        self._gamma_discount: float | None = gamma_discount

    # =========================================================================
    # 变量创建与命名
    # =========================================================================

    def _create_variables(self) -> list:
        """创建 kaiwu Binary 变量（安全命名）。

        根据 encoding_type 创建不同数量的变量:
          - precision_split_38: 5 主元 × 7 bits + C × 3 bits = 38
          - unified_48: 6 元素 × 8 bits = 48
          - unified_38: 5 主元 × 6 bits + C × 8 bits = 38

        命名格式: e{元素索引}_b{比特编号}
        避免使用 x_0~x_37 (kaiwu SDK 下划线+数字后缀存在索引碰撞 bug)。
        """
        kw, _ = _ensure_kaiwu()
        xs = []
        for ei in range(5):  # Main elements (Al, Co, Cr, Fe, Ni)
            for bj in range(self._bits_main):
                xs.append(kw.Binary(f"e{ei}_b{bj}"))
        for bj in range(self._bits_carbon):  # Carbon
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
        self, xs: list
    ) -> dict:
        """构建各元素成分的 kaiwu 表达式。

        c_elem = base + step × Σ 2^j × e{ei}_b{j}

        Returns:
            elem → kaiwu BinaryExpression 的字典。
        """
        _, quicksum = _ensure_kaiwu()
        c_expr = {}
        for i, elem in enumerate(MAIN_ELEMENTS):
            start = i * self._bits_main
            terms = quicksum(
                (1 << j) * xs[start + j]
                for j in range(self._bits_main)
            )
            c_expr[elem] = self._base_main + self._step_main * terms

        # C element (element index 5)
        start_c = 5 * self._bits_main
        terms_c = quicksum(
            (1 << j) * xs[start_c + j]
            for j in range(self._bits_carbon)
        )
        c_expr[INTERSTITIAL_ELEMENT] = (
            self._base_carbon + self._step_carbon * terms_c
        )

        return c_expr

    # =========================================================================
    # f₁: 混合焓 (Miedema)
    # =========================================================================

    def _build_f1_expr(
        self, c_expr: dict
    ) -> ...:
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
        gamma = self._gamma_discount if self._gamma_discount is not None else MIEDEMA.gamma_discount
        for elem in MAIN_ELEMENTS:
            dh_raw = MIEDEMA.dh_carbon[elem]
            dh = gamma * dh_raw
            term = 4.0 * dh * c_expr["C"] * c_expr[elem] / 10000.0
            expr = expr + term

        return expr

    # =========================================================================
    # f₂: 密度 (Vegard)
    # =========================================================================

    def _build_f2_expr(
        self, c_expr: dict
    ) -> ...:
        """构建 f₂ (Vegard 密度) 的 kaiwu 表达式。

        ρ = Σ c_elem_at% × ρ_elem / 100
        """
        _, quicksum = _ensure_kaiwu()
        return quicksum(
            c_expr[elem] * ELEM.density_of(elem) / 100.0
            for elem in ALL_ELEMENTS
        )

    # =========================================================================
    # f₃: 成本指数
    # =========================================================================

    def _build_f3_expr(
        self, c_expr: dict
    ) -> ...:
        """构建 f₃ (成本指数) 的 kaiwu 表达式。

        f_cost = Σ c_elem_at% × w_elem
        """
        _, quicksum = _ensure_kaiwu()
        return quicksum(
            c_expr[elem] * ELEM.cost_weight_of(elem)
            for elem in ALL_ELEMENTS
        )

    # =========================================================================
    # 约束惩罚项
    # =========================================================================

    def _build_P0_expr(
        self, c_expr: dict
    ) -> ...:
        """构建 P0 硬约束表达式: (Σc_k - 100)² / 5625, 即 (S_var' - 1)²。

        方案 C-02 量级归一化 (hea_encoding_scheme_v1.13.md §3.1):
            H_penalty = λ_sum × (S_var/75 - 1)²
                      = (λ_sum/5625) × (Σc - 100)²
        5625 的来源: 5625 = 75², 75 为变量步长贡献的目标量级 —
            S_var = Σc - S_const, S_const = 5×5.0 + 0.0 = 25.0%,
            等原子比目标 S_var = 100 - 25 = 75 (%), 故归一化分母取 75。
        λ_sum 系数在 build_model 中乘入 (CONSTRAINT.lambda_sum_fixed = 15)。

        归一化后 P0 惩罚与 w·f 目标项同数量级, 避免目标函数被淹没、
        偏好权重失效及 CIM 系数动态范围溢出 (评审报告 P0-1)。
        与规范实现 constraints/sum_constraint.py:38-39
        (scale_factor = λ_sum/5625) 保持一致, 可对账。

        (Σc - 100)² = Σc² - 200·Σc + 10000
        其中 Σc = Σ c_elem, Σc² = (Σc)² 由 kaiwu 自动展开为二次型。
        """
        _, quicksum = _ensure_kaiwu()
        sum_c = quicksum(c_expr[elem] for elem in ALL_ELEMENTS)
        # C-02: 除以 5625 (= 75², S_var 目标量级平方) 做量级归一化
        return (sum_c - 100.0) * (sum_c - 100.0) / 5625.0

    def _build_P1_expr(
        self, c_expr: dict
    ) -> ...:
        """构建 P1 软约束: (c_C - 0.8)²。

        纯二次型，精确嵌入 QUBO。
        c_C < 0.8% 时引入的额外惩罚量级远小于 P0, 优化器仍能找到低 C 有效解。
        """
        cc = c_expr[INTERSTITIAL_ELEMENT]
        soft_upper = CONSTRAINT.carbide_soft_upper
        return (cc - soft_upper) * (cc - soft_upper)

    def _build_P2_expr(
        self, c_expr: dict
    ) -> ...:
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
    ):
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
        kw, _ = _ensure_kaiwu()
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
            cim_mode: 是否启用 CIM 精度截断 (仅截断 P2 注入项, 见 B-05 修复注释)。
            cim_threshold: P2 截断阈值, 按 |未加权系数| 判定, 默认
                CONSTRAINT.cim_noise_floor (0.01)。

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

        # CIM 截断 (如果需要)
        if cim_mode:
            if cim_threshold is None:
                cim_threshold = CONSTRAINT.cim_noise_floor
            # B-05 修复: 截断范围限定在 P2 注入项, 不再对全矩阵 |·|<threshold
            # 置零。依据 plan/CIM_Fusion_Evaluation_Report_Revised.md §3.1
            # 结论: "P2 交叉项是唯一的真机不可分辨区域, P0/f1/f2/f3 必须保留"。
            # 原全矩阵截断在 P0-1 归一化 (f1 经 w1/30 缩放) 后误清 230 个
            # 目标函数小系数 (实测, 默认初始 λ、权重 (0.4,0.3,0.3)), 等原子比
            # 点能量失真 -0.436, 远超 P2 自身最大贡献量级 (~0.12)。
            #
            # 做法: 以 lambda_ccr=0 重建模型, 差分隔离 P2 的精确贡献
            # (h 线性项 + Q 交叉项), 仅对 P2 项按 |未加权系数| < threshold
            # 截断 —— 与 constraints/ccr_coupling.py 的 B-04 修复同一规则,
            # 判定与 PenaltyFlex 自适应 λ 解耦, 截断集合为静态结构属性;
            # 再叠加回无 P2 基座。P2 之外的所有系数 (f1/f2/f3/P0/P1)
            # 因此完全不受截断影响, 目标函数零误伤。
            lambda_ccr_resolved = (
                lambda_ccr if lambda_ccr is not None else CONSTRAINT.lambda_ccr_init
            )
            base_model = self.build_model(
                weights=weights,
                lambda_carbide=lambda_carbide,
                lambda_ccr=0.0,
                lambda_sum=lambda_sum,
            )
            base_full = base_model.get_matrix()
            p2_full = full_mat - base_full  # P2 精确贡献 (含 h 与 Q 区块)
            weight_p2 = CONSTRAINT.omega_ccr * lambda_ccr_resolved
            if weight_p2 > 0:
                # 未加权系数 = P2 贡献 / (ω·λ); |未加权| < threshold 的项置零。
                # 非 P2 区域的差分浮点残差 (~1e-20) 同样远低于阈值被置零,
                # 叠加后精确还原基座。
                p2_full = np.where(
                    np.abs(p2_full / weight_p2) < cim_threshold, 0.0, p2_full
                )
            else:
                p2_full = np.zeros_like(p2_full)  # λ=0 → 无 P2, 截断为空操作
            full_mat = base_full + p2_full

        # 提取 h (对角线) 和 Q (严格上三角)
        h = np.diag(full_mat).copy()
        Q = np.triu(full_mat, 1)

        offset = float(model.get_offset())

        return QUBOMatrix(h=h, Q=Q, constant_offset=offset)

    def get_variable_names(self) -> List[str]:
        """返回 kaiwu 变量名列表 ['e0_b0', 'e0_b1', ..., 'e5_b{max}']."""
        names = []
        for ei in range(5):
            for bj in range(self._bits_main):
                names.append(f"e{ei}_b{bj}")
        for bj in range(self._bits_carbon):
            names.append(f"e5_b{bj}")
        return names

    @property
    def encoding_type(self) -> str:
        """返回当前编码类型。"""
        return self._encoding_type

    @property
    def num_variables(self) -> int:
        """返回当前编码的变量总数。"""
        return self._n_vars
