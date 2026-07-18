"""
目标函数模块（第二层）。

三目标优化:
  - f₁: Miedema 混合焓 ΔH_mix（天然二次型，精确嵌入 QUBO）
  - f₂: Vegard 定律密度 ρ（纯线性，映射为 QUBO 偏置场）
  - f₃: 加权成本指数（纯线性，映射为 QUBO 偏置场）

归一化层:
  - PhysicalPriorNormalizer: 基于物理先验的固定归一化
  - DynamicNormalizer: 基于数据范围的动态归一化
  - WeightedObjective: 加权组合多目标
"""

from .mixing_enthalpy import MixingEnthalpy
from .density import VegardDensity
from .cost import WeightedCost
from .normalization import (
    PhysicalPriorNormalizer,
    DynamicNormalizer,
    WeightedObjective,
)

__all__ = [
    "MixingEnthalpy",
    "VegardDensity",
    "WeightedCost",
    "PhysicalPriorNormalizer",
    "DynamicNormalizer",
    "WeightedObjective",
]
