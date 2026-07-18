"""
ParetoZoom 动态前沿探索模块（第四层）。

五阶段算法:
  A: 粗网格初始化（12 组权重 G1~G12）
  B: 间隙检测（相邻非支配解间插入新权重）
  C: HV 热点导向微扰（高斯扰动高贡献区域）
  D: QUBO 求解（CIM 真机 / 内置 SA 后端）+ 存档更新
  E: HV 收敛判定

双层自适应架构:
  - 外层 ParetoZoom：控制权重 (w1, w2, w3)
  - 内层 PenaltyFlex：控制惩罚系数 λ
  - warm-start：新权重继承最近邻的 λ*
"""

from .pareto_zoom import ParetoZoom, ParetoZoomRound
from .archive import Archive
from .weight_utils import (
    WeightGenerator,
    normalize_weights,
    deduplicate_weights,
    midpoint_weights,
    clamp_weights,
)

__all__ = [
    "ParetoZoom",
    "ParetoZoomRound",
    "Archive",
    "WeightGenerator",
    "normalize_weights",
    "deduplicate_weights",
    "midpoint_weights",
    "clamp_weights",
]
