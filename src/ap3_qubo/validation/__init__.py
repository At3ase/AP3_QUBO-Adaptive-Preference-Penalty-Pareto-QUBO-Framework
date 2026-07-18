"""
物理验证与 Pareto 分析模块。

提供:
  - PhysicalFilter: 物理合理性过滤器（VEC/δ/Ω/Δχ/碳化物风险/HEA 范围）
  - ParetoSort: 非支配排序
  - Hypervolume: HV 计算
  - ValidatedSolution: 解决方案与其物理验证结果的组合体
  - ConvergenceMetrics: 探索收敛追踪
"""

from dataclasses import dataclass

from .physical_filters import PhysicalFilter, PhysicalFilterResult, RiskLevel
from .pareto import ParetoSort, SolutionRecord
from .hypervolume import HypervolumeCalculator


@dataclass
class ValidatedSolution:
    """解决方案与其物理验证结果的组合体。

    将 SolutionRecord（成分、目标值、权重、能量等）与
    PhysicalFilterResult（VEC、δ、Ω、Δχ、碳化物风险等）绑定在一起，
    提供统一的"已验证解"视图。

    Attributes:
        solution: 原始解记录（成分、目标值、权重等）。
        validation: 物理过滤器结果（所有物理探针值 + 通过/失败标记）。
    """

    solution: SolutionRecord
    validation: PhysicalFilterResult

    @property
    def is_physically_valid(self) -> bool:
        """该解是否通过所有物理过滤器？"""
        return self.validation.all_pass

    @property
    def risk_summary(self) -> dict:
        """汇总风险标志。"""
        return {
            "vec_pass": self.validation.vec_pass,
            "delta_pass": self.validation.delta_pass,
            "omega_class": self.validation.omega_level,
            "delta_chi_pass": self.validation.delta_chi_pass,
            "hea_range_pass": self.validation.is_hea,
            "dh_mix_pass": self.validation.dh_mix_in_range,
            "carbide_risk": self.validation.carbide_risk,
            "ccr_risk": self.validation.ccr_coupling_risk,
        }


__all__ = [
    "PhysicalFilter",
    "PhysicalFilterResult",
    "RiskLevel",
    "ParetoSort",
    "SolutionRecord",
    "HypervolumeCalculator",
    "ValidatedSolution",
]
