"""
物理合理性过滤器。

对每个合金成分计算并标记：
  - VEC 探针（价电子浓度）
  - δ 判据（原子尺寸差异）
  - Ω 判据（热力学稳定性分级）
  - Δχ 判据（电负性差异）
  - HEA 定义范围检查
  - ΔH_mix 合理区间检查
  - 碳化物风险三级标记
  - C-Cr 耦合风险标记
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

import numpy as np

from ..physical_params import (
    ELEM,
    MIEDEMA,
    MAIN_ELEMENTS,
    INTERSTITIAL_ELEMENT,
    ALL_ELEMENTS,
    VEC_LOWER,
    VEC_UPPER,
    DELTA_THRESHOLD,
    OMEGA_STABLE,
    OMEGA_METASTABLE,
    DELTA_CHI_THRESHOLD,
    HEA_ELEMENT_MIN,
    HEA_ELEMENT_MAX,
    DH_MIX_LOWER,
    DH_MIX_UPPER,
    CARBIDE_RISK_ABSOLUTE,
    CARBIDE_RISK_WARNING,
    CCR_COUPLING_RISK,
    SUM_TOLERANCE,
)

# 理想气体常数 (J/(mol·K))
R_GAS = 8.314


class RiskLevel(Enum):
    """物理风险等级。"""
    PASS = "✅"       # 通过
    WARNING = "⚠️"    # 警告（接近边界或亚稳态）
    FAIL = "❌"       # 不通过（明确违反物理规律）


@dataclass
class PhysicalFilterResult:
    """单个解的物理过滤器评估结果。

    Attributes:
        vec: 价电子浓度值。
        delta: 原子尺寸差异 (%)。
        omega: 热力学参数 Ω。
        delta_chi: 电负性差异。
        vec_pass: VEC 是否在 [7.0, 7.6] 范围内。
        delta_pass: δ 是否 < 6.6%。
        omega_level: Ω 分级 ("stable" | "metastable" | "unstable")。
        delta_chi_pass: Δχ 是否 < 0.133。
        is_hea: 是否满足 HEA 定义（每元素 5%~35%）。
        dh_mix_in_range: ΔH_mix 是否在 [-15, +10] kJ/mol 内。
        carbide_risk: 碳化物风险等级。
        ccr_coupling_risk: C-Cr 耦合是否超阈值。
        sum_deviation: 成分和偏离 100% 的绝对值。
        sum_pass: 成分和是否在容差内。
        all_pass: 所有关键过滤器是否通过。
        risk_flags: 人类可读的风险描述列表。
        metrics: 所有计算的物理指标字典。
    """
    vec: float
    delta: float
    omega: float
    delta_chi: float
    vec_pass: bool
    delta_pass: bool
    omega_level: str  # "stable" | "metastable" | "unstable"
    delta_chi_pass: bool
    is_hea: bool
    dh_mix_in_range: bool
    carbide_risk: str  # "none" | "warning" | "high"
    ccr_coupling_risk: bool
    sum_deviation: float
    sum_pass: bool
    all_pass: bool
    risk_flags: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)


class PhysicalFilter:
    """物理合理性过滤器。

    对合金成分运行所有物理探针检查，输出三级标记结果。

    使用示例:
        >>> filt = PhysicalFilter()
        >>> result = filt.evaluate(composition, dh_mix=-12.32)
        >>> if result.all_pass:
        ...     print("Composition passes all physical filters")
    """

    def __init__(self):
        self._elem_props = ELEM

    # =========================================================================
    # 单成分评估
    # =========================================================================

    def evaluate(
        self,
        fractions: Dict[str, float],
        dh_mix: float,
    ) -> PhysicalFilterResult:
        """对单个成分运行所有物理过滤器。

        Args:
            fractions: 元素 → at% 字典（0~100 尺度）。
            dh_mix: 混合焓 (kJ/mol)。

        Returns:
            PhysicalFilterResult。
        """
        # 归一化到 [0, 1]
        c = {e: fractions.get(e, 0.0) / 100.0 for e in ALL_ELEMENTS}
        # c_i' 主元归一化（hea_encoding_scheme_v1.13.md :281：
        # "c_i' = c_i / (1 − c_C)  // 主元归一化（扣除C的间隙占比）"）。
        # VEC/δ/ΔS_mix/Δχ 的 Σ c_i' 均以主元归一化到和为 1 为方案口径；
        # C 为间隙原子，不占据晶格置换位（v1.13 :284 物理说明）。
        # 原实现直接用 c_i（未归一化），C>0 时 VEC 低估 ~c_C、δ/Δχ/Ω
        # 系统性偏移（C=1.75at% 实测偏差：VEC −1.75%、δ +6.0%、
        # Ω −4.5%、Δχ +2.4%，可致边界判定翻转）。
        main_total = sum(c[e] for e in MAIN_ELEMENTS)
        if main_total > 1e-12:
            c_main = {e: c[e] / main_total for e in MAIN_ELEMENTS}
        else:
            c_main = {e: 0.0 for e in MAIN_ELEMENTS}

        # 计算各项指标
        vec = self._compute_vec(c_main)
        delta = self._compute_delta(c_main)
        omega = self._compute_omega(c_main, dh_mix, c_all=c)
        delta_chi = self._compute_delta_chi(c_main)

        # 检查
        sum_dev = abs(sum(fractions.values()) - 100.0)
        risk_flags = []

        # VEC
        vec_pass = VEC_LOWER <= vec <= VEC_UPPER
        if not vec_pass:
            risk_flags.append(f"VEC={vec:.2f} 不在 [{VEC_LOWER}, {VEC_UPPER}]")

        # δ
        delta_pass = delta < DELTA_THRESHOLD
        if not delta_pass:
            risk_flags.append(f"δ={delta:.2f}% ≥ {DELTA_THRESHOLD}%")

        # Ω 分级
        if omega >= OMEGA_STABLE:
            omega_level = "stable"
        elif omega >= OMEGA_METASTABLE:
            omega_level = "metastable"
            risk_flags.append(f"Ω={omega:.2f} 亚稳态")
        else:
            omega_level = "unstable"
            risk_flags.append(f"Ω={omega:.2f} 不稳定")

        # Δχ
        delta_chi_pass = delta_chi < DELTA_CHI_THRESHOLD
        if not delta_chi_pass:
            risk_flags.append(f"Δχ={delta_chi:.4f} ≥ {DELTA_CHI_THRESHOLD}")

        # HEA 范围
        is_hea = all(
            HEA_ELEMENT_MIN <= fractions.get(e, 0.0) <= HEA_ELEMENT_MAX
            for e in MAIN_ELEMENTS
        )
        if not is_hea:
            risk_flags.append("元素成分超出 HEA 定义范围 [5%, 35%]")

        # ΔH_mix 范围
        dh_mix_in_range = DH_MIX_LOWER <= dh_mix <= DH_MIX_UPPER
        if not dh_mix_in_range:
            risk_flags.append(
                f"ΔH_mix={dh_mix:.1f} kJ/mol 不在 [{DH_MIX_LOWER}, {DH_MIX_UPPER}]"
            )

        # 碳化物风险
        c_carbon = fractions.get(INTERSTITIAL_ELEMENT, 0.0)
        if c_carbon >= CARBIDE_RISK_ABSOLUTE:
            carbide_risk = "high"
            risk_flags.append(f"C={c_carbon:.2f}% ≥ {CARBIDE_RISK_ABSOLUTE}% (碳化物高风险)")
        elif c_carbon >= CARBIDE_RISK_WARNING:
            carbide_risk = "warning"
            risk_flags.append(f"C={c_carbon:.2f}% ≥ {CARBIDE_RISK_WARNING}% (碳化物警告)")
        else:
            carbide_risk = "none"

        # C-Cr 耦合
        c_cr = fractions.get("Cr", 0.0)
        coupling = (c_carbon * c_cr) / (1.75 * 36.75)  # 归一化
        ccr_risk = coupling > CCR_COUPLING_RISK
        if ccr_risk:
            risk_flags.append(
                f"C-Cr 耦合={coupling:.3f} > {CCR_COUPLING_RISK}"
            )

        # 成分和
        sum_pass = sum_dev <= SUM_TOLERANCE
        if not sum_pass:
            risk_flags.append(f"成分和偏差={sum_dev:.2f}% > {SUM_TOLERANCE}%")

        # 综合判定
        all_pass = (
            vec_pass
            and delta_pass
            and omega_level != "unstable"
            and delta_chi_pass
            and is_hea
            and dh_mix_in_range
            and carbide_risk != "high"
            and sum_pass
        )

        metrics = {
            "vec": vec,
            "delta": delta,
            "omega": omega,
            "delta_chi": delta_chi,
            "dh_mix": dh_mix,
            "c_carbon": c_carbon,
            "sum_total": sum(fractions.values()),
            "ccr_coupling": coupling,
        }

        return PhysicalFilterResult(
            vec=vec,
            delta=delta,
            omega=omega,
            delta_chi=delta_chi,
            vec_pass=vec_pass,
            delta_pass=delta_pass,
            omega_level=omega_level,
            delta_chi_pass=delta_chi_pass,
            is_hea=is_hea,
            dh_mix_in_range=dh_mix_in_range,
            carbide_risk=carbide_risk,
            ccr_coupling_risk=ccr_risk,
            sum_deviation=sum_dev,
            sum_pass=sum_pass,
            all_pass=all_pass,
            risk_flags=risk_flags,
            metrics=metrics,
        )

    def evaluate_batch(
        self,
        compositions: List[Dict[str, float]],
        dh_mix_values: List[float],
    ) -> List[PhysicalFilterResult]:
        """批量评估（向量化）。

        Args:
            compositions: 成分字典列表。
            dh_mix_values: 对应的 ΔH_mix 值列表。

        Returns:
            PhysicalFilterResult 列表。
        """
        return [
            self.evaluate(comp, dh)
            for comp, dh in zip(compositions, dh_mix_values)
        ]

    def summary(
        self, results: List[PhysicalFilterResult]
    ) -> Dict[str, float]:
        """汇总过滤器通过率。

        Args:
            results: 物理过滤器结果列表。

        Returns:
            各过滤器通过率和统计信息。
        """
        n = len(results)
        if n == 0:
            return {}

        return {
            "vec_pass_rate": sum(1 for r in results if r.vec_pass) / n,
            "delta_pass_rate": sum(1 for r in results if r.delta_pass) / n,
            "omega_stable_rate": sum(1 for r in results if r.omega_level == "stable") / n,
            "omega_acceptable_rate": sum(
                1 for r in results if r.omega_level in ("stable", "metastable")
            ) / n,
            "delta_chi_pass_rate": sum(1 for r in results if r.delta_chi_pass) / n,
            "hea_rate": sum(1 for r in results if r.is_hea) / n,
            "dh_mix_pass_rate": sum(1 for r in results if r.dh_mix_in_range) / n,
            "carbide_safe_rate": sum(1 for r in results if r.carbide_risk == "none") / n,
            "sum_pass_rate": sum(1 for r in results if r.sum_pass) / n,
            "all_pass_rate": sum(1 for r in results if r.all_pass) / n,
        }

    # =========================================================================
    # 物理量计算
    # =========================================================================

    @staticmethod
    def _compute_vec(c_main: Dict[str, float]) -> float:
        """价电子浓度 VEC = Σ c_i' · VEC_i（c_i' 主元归一化，v1.13 :278）。"""
        return sum(c_main[e] * ELEM.vec_of(e) for e in MAIN_ELEMENTS)

    def _compute_delta(self, c_main: Dict[str, float]) -> float:
        """原子尺寸差异 δ = sqrt(Σ c_i' · (1 − r_i/r̄)²) × 100%，
        r̄ = Σ c_i'·r_i（c_i' 主元归一化，v1.13 :683-684）。"""
        radii = {e: ELEM.radius_of(e) for e in MAIN_ELEMENTS}
        r_bar = sum(c_main[e] * radii[e] for e in MAIN_ELEMENTS)
        if r_bar < 1e-9:
            return 0.0
        variance = sum(
            c_main[e] * (1.0 - radii[e] / r_bar) ** 2
            for e in MAIN_ELEMENTS
        )
        return float(np.sqrt(max(variance, 0.0)) * 100.0)

    def _compute_omega(
        self,
        c_main: Dict[str, float],
        dh_mix: float,
        c_all: Optional[Dict[str, float]] = None,
    ) -> float:
        """热力学参数 Ω = T_m · ΔS_mix / |ΔH_mix|。

        方案口径（hea_encoding_scheme_v1.13.md :690-691）：
          - ΔS_mix = −R·Σ c_i'·ln(c_i')，仅 5 主元，c_i' 主元归一化
            （C 为间隙原子不贡献位形熵）；
          - T_m = Σ_{i∈主元∪{C}} c_i·T_{m,i}，混合熔点**含 C**
            （c_i 为原始原子分数，未归一化）。

        当 ΔH_mix ≈ 0 时返回一个大值（表示高度稳定）。

        Args:
            c_main: 主元归一化成分（Σ=1）。
            dh_mix: 混合焓 (kJ/mol)。
            c_all: 含 C 的原始原子分数字典（T_m 用）；缺省退化为
                仅用 c_main（仅供无 C 场景的旧调用兼容）。
        """
        # 理想混合熵 ΔS_mix = −R · Σ c_i' · ln(c_i')（仅主元）
        delta_s = 0.0
        for e in MAIN_ELEMENTS:
            ci = c_main[e]
            if ci > 1e-10:
                delta_s -= ci * np.log(ci)
        delta_s *= R_GAS  # J/(mol·K)

        # 加权熔点 T_m（含 C，v1.13 :690）
        if c_all is not None:
            t_m = sum(
                c_all.get(e, 0.0) * ELEM.melting_point_of(e)
                for e in ALL_ELEMENTS
            )
        else:
            t_m = sum(
                c_main[e] * ELEM.melting_point_of(e) for e in MAIN_ELEMENTS
            )

        abs_dh = abs(dh_mix)
        if abs_dh < 1e-9:
            return 100.0  # 极大值，表示高度稳定

        return float(t_m * delta_s / (abs_dh * 1000.0))  # kJ → J

    @staticmethod
    def _compute_delta_chi(c_main: Dict[str, float]) -> float:
        """电负性差异 Δχ = sqrt(Σ c_i' · (χ_i − χ̄)²)，
        χ̄ = Σ c_i'·χ_i（c_i' 主元归一化，v1.13 :698-699）。"""
        chi = {e: ELEM.en_of(e) for e in MAIN_ELEMENTS}
        chi_bar = sum(c_main[e] * chi[e] for e in MAIN_ELEMENTS)
        variance = sum(
            c_main[e] * (chi[e] - chi_bar) ** 2
            for e in MAIN_ELEMENTS
        )
        return float(np.sqrt(max(variance, 0.0)))
