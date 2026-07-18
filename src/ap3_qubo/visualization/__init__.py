"""
可视化模块。

提供:
  - Pareto 前沿 2D/3D 可视化
  - HV 收敛曲线
  - 成分分布热力图
  - λ 轨迹图
"""

from .pareto_plots import plot_pareto_2d, plot_pareto_3d, plot_hv_progression
from .composition_plots import plot_composition_heatmap, plot_element_distribution
from .convergence_plots import plot_lambda_trajectory, plot_constraint_violation

__all__ = [
    "plot_pareto_2d",
    "plot_pareto_3d",
    "plot_hv_progression",
    "plot_composition_heatmap",
    "plot_element_distribution",
    "plot_lambda_trajectory",
    "plot_constraint_violation",
]
