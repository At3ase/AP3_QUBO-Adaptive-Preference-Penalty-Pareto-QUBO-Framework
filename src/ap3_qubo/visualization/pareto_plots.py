"""
Pareto 前沿可视化。

支持 2D 投影、3D 散点、HV 收敛曲线。
"""

from typing import List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes


# 目标名称和单位
_OBJ_LABELS = [
    ("ΔH_mix", "kJ/mol"),
    ("Density", "g/cm³"),
    ("Cost Index", "rel."),
]


def plot_pareto_2d(
    objectives: np.ndarray,
    x_obj: int = 0,
    y_obj: int = 1,
    title: str = "Pareto Frontier (2D Projection)",
    nondominated_mask: np.ndarray | None = None,
    highlight_indices: List[int] | None = None,
    ax: Optional[Axes] = None,
    figsize: Tuple[int, int] = (8, 6),
    **kwargs,
) -> Figure:
    """绘制 Pareto 前沿的 2D 投影。

    Args:
        objectives: shape=(N, 3) 目标值矩阵。
        x_obj: x 轴目标索引 (0=f1, 1=f2, 2=f3)。
        y_obj: y 轴目标索引。
        title: 图标题。
        nondominated_mask: shape=(N,) bool 数组，标记非支配解。
        highlight_indices: 高亮解的索引列表。
        ax: 可选 matplotlib Axes。
        figsize: 图大小。

    Returns:
        matplotlib Figure。
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    n = len(objectives)

    # 分类点
    if nondominated_mask is not None:
        front_mask = nondominated_mask
    else:
        # 简单帕累托过滤
        front_mask = np.ones(n, dtype=bool)
        for i in range(n):
            for j in range(n):
                if i != j and np.all(objectives[j] <= objectives[i]) and np.any(objectives[j] < objectives[i]):
                    front_mask[i] = False
                    break

    dominated_mask = ~front_mask

    # 被支配点（灰色，较小的点）
    if np.any(dominated_mask):
        ax.scatter(
            objectives[dominated_mask, x_obj],
            objectives[dominated_mask, y_obj],
            c="lightgray", s=20, alpha=0.4, label="Dominated",
        )

    # 非支配点（着色，较大的点）
    ax.scatter(
        objectives[front_mask, x_obj],
        objectives[front_mask, y_obj],
        c="steelblue", s=60, alpha=0.8, edgecolors="darkblue", linewidth=0.5,
        label="Pareto Front",
    )

    # 高亮特定点
    if highlight_indices is not None:
        hl_pts = objectives[highlight_indices]
        ax.scatter(
            hl_pts[:, x_obj], hl_pts[:, y_obj],
            c="red", s=100, marker="*", edgecolors="darkred",
            label="Highlighted",
            zorder=5,
        )

    ax.set_xlabel(f"{_OBJ_LABELS[x_obj][0]} ({_OBJ_LABELS[x_obj][1]})")
    ax.set_ylabel(f"{_OBJ_LABELS[y_obj][0]} ({_OBJ_LABELS[y_obj][1]})")
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    return fig


def plot_pareto_3d(
    objectives: np.ndarray,
    title: str = "3D Pareto Frontier",
    nondominated_mask: np.ndarray | None = None,
    elev: float = 25.0,
    azim: float = 45.0,
    figsize: Tuple[int, int] = (10, 8),
    **kwargs,
) -> Figure:
    """绘制 Pareto 前沿的 3D 散点图。

    Args:
        objectives: shape=(N, 3) 目标值矩阵。
        title: 图标题。
        nondominated_mask: 非支配解标记。
        elev: 仰角。
        azim: 方位角。
        figsize: 图大小。

    Returns:
        matplotlib Figure。
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    n = len(objectives)

    if nondominated_mask is None:
        front_mask = np.ones(n, dtype=bool)
        for i in range(n):
            for j in range(n):
                if i != j and np.all(objectives[j] <= objectives[i]) and np.any(objectives[j] < objectives[i]):
                    front_mask[i] = False
                    break
    else:
        front_mask = nondominated_mask

    # 被支配点
    dominated_mask = ~front_mask
    if np.any(dominated_mask):
        ax.scatter(
            objectives[dominated_mask, 0],
            objectives[dominated_mask, 1],
            objectives[dominated_mask, 2],
            c="lightgray", s=15, alpha=0.3,
        )

    # 非支配点
    ax.scatter(
        objectives[front_mask, 0],
        objectives[front_mask, 1],
        objectives[front_mask, 2],
        c="steelblue", s=50, alpha=0.8, edgecolors="darkblue", linewidth=0.3,
    )

    ax.set_xlabel(f"{_OBJ_LABELS[0][0]} ({_OBJ_LABELS[0][1]})")
    ax.set_ylabel(f"{_OBJ_LABELS[1][0]} ({_OBJ_LABELS[1][1]})")
    ax.set_zlabel(f"{_OBJ_LABELS[2][0]} ({_OBJ_LABELS[2][1]})")
    ax.set_title(title)
    ax.view_init(elev=elev, azim=azim)

    return fig


def plot_hv_progression(
    hv_history: List[float],
    title: str = "Hypervolume Convergence",
    threshold: float = 0.01,
    ax: Optional[Axes] = None,
    figsize: Tuple[int, int] = (8, 5),
    **kwargs,
) -> Figure:
    """绘制 HV 随 ParetoZoom 轮数的变化曲线。

    Args:
        hv_history: 每轮 HV 值列表。
        title: 图标题。
        threshold: 收敛阈值（画水平虚线）。
        ax: 可选 Axes。
        figsize: 图大小。

    Returns:
        matplotlib Figure。
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    rounds = list(range(len(hv_history)))

    ax.plot(rounds, hv_history, "o-", color="steelblue", linewidth=2, markersize=6)
    ax.set_xlabel("ParetoZoom Round")
    ax.set_ylabel("Hypervolume (HV)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    # 收敛指示
    if len(hv_history) >= 3:
        # 标注 ΔHV < 1% 的区域
        deltas = []
        for i in range(1, len(hv_history)):
            if hv_history[i-1] > 1e-12:
                deltas.append(abs(hv_history[i] - hv_history[i-1]) / hv_history[i-1])
            else:
                deltas.append(1.0)

        conv_start = None
        for i in range(len(deltas) - 1):
            if deltas[i] < threshold and deltas[i+1] < threshold:
                conv_start = i + 1
                break

        if conv_start is not None and conv_start < len(rounds):
            ax.axvline(x=rounds[conv_start], color="green", linestyle="--", alpha=0.5,
                       label=f"Converged (ΔHV < {threshold*100:.0f}%)")
            ax.legend()

    return fig
