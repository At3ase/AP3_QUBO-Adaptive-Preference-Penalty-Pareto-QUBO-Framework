"""
约束与惩罚模块。

三层约束体系:
  - P0 硬约束: 成分和=100%（质量守恒，固定惩罚 λ_sum=15，不参与 PenaltyFlex 自适应）
  - P1 软约束: 碳化物抑制 c_C < 0.8%（PenaltyFlex 控制 λ_carbide）
  - P2 软约束: C-Cr 耦合最小化（PenaltyFlex 控制 λ_ccr）

参考实现:
  - SumTo100Constraint: 通过 λ·(Σc−100)²/5625 做 C-02 量级归一化（规范路径）
  - CarbideConstraint: 纯二次型 (c_C − 0.8)²
  - CCrCouplingConstraint: c_C × c_Cr / H_max 交叉项 + 线性项

注意: 主路径 QUBO 构建在 qubo/builder.py 中，约束模块提供独立的参考实现
与诊断函数，用于对账和验证。
"""

from .sum_constraint import SumTo100Constraint
from .carbide_constraint import CarbideConstraint
from .ccr_coupling import CCrCouplingConstraint

__all__ = [
    "SumTo100Constraint",
    "CarbideConstraint",
    "CCrCouplingConstraint",
]
