"""
收敛过程可视化。

提供 λ 轨迹图和约束违反度变化图。
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes


def plot_lambda_trajectory(
    lambda_history: List[Tuple[float, float]],
    phase_transitions: List[int] | None = None,
    title: str = "PenaltyFlex Lambda Trajectory",
    ax: Optional[Axes] = None,
    figsize: Tuple[int, int] = (10, 5),
    **kwargs,
) -> Figure:
    """绘制 PenaltyFlex 的 λ 随迭代轮数的变化轨迹。

    Args:
        lambda_history: [(λ_carbide, λ_ccr), ...] 每轮的值。
        phase_transitions: 阶段转换的轮数列表。
        title: 图标题。
        ax: 可选 Axes。
        figsize: 图大小。

    Returns:
        matplotlib Figure。
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    rounds = list(range(len(lambda_history)))
    lc = [x[0] for x in lambda_history]
    lccr = [x[1] for x in lambda_history]

    ax.plot(rounds, lc, "o-", color="steelblue", linewidth=2, markersize=5,
            label="λ_carbide (P1)")
    ax.plot(rounds, lccr, "s--", color="darkorange", linewidth=2, markersize=5,
            label="λ_ccr (P2)")

    # 阶段分隔线
    if phase_transitions:
        for pt in phase_transitions:
            if 0 < pt < len(rounds):
                ax.axvline(x=rounds[pt], color="gray", linestyle=":", alpha=0.5,
                           linewidth=1)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Lambda Value")
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    return fig


def plot_constraint_violation(
    violation_history: List[Tuple[float, float]],
    title: str = "Constraint Violation vs Iteration",
    ax: Optional[Axes] = None,
    figsize: Tuple[int, int] = (10, 5),
    **kwargs,
) -> Figure:
    """绘制约束违反度随迭代的变化。

    Args:
        violation_history: [(v_carbide, v_ccr), ...] 每轮的违反度。
        title: 图标题。
        ax: 可选 Axes。
        figsize: 图大小。

    Returns:
        matplotlib Figure。
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    rounds = list(range(len(violation_history)))
    vc = [x[0] for x in violation_history]
    vccr = [x[1] for x in violation_history]

    ax.plot(rounds, vc, "o-", color="crimson", linewidth=2, markersize=5,
            label="Violation P1 (carbide)")
    ax.plot(rounds, vccr, "s--", color="purple", linewidth=2, markersize=5,
            label="Violation P2 (C-Cr coupling)")

    # 目标区域（期望偏离度 ε）
    ax.axhline(y=0.02, color="green", linestyle=":", alpha=0.4,
               label="ε_explore = 0.02")
    ax.axhline(y=0.0, color="gray", linestyle="-.", alpha=0.3,
               label="ε_consolidate = 0")

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Average Violation")
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    return fig
