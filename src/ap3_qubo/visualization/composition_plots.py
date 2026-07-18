"""
成分分布可视化。

提供成分热力图和元素分布直方图。
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from ..physical_params import ALL_ELEMENTS, MAIN_ELEMENTS


_ELEM_COLORS = {
    "Al": "#1f77b4",
    "Co": "#ff7f0e",
    "Cr": "#2ca02c",
    "Fe": "#d62728",
    "Ni": "#9467bd",
    "C": "#8c564b",
}


def plot_element_distribution(
    compositions: List[Dict[str, float]],
    elements: List[str] | None = None,
    title: str = "Element Distribution on Pareto Front",
    ax: Optional[Axes] = None,
    figsize: Tuple[int, int] = (10, 5),
    **kwargs,
) -> Figure:
    """绘制 Pareto 前沿上各元素的成分分布（箱线图）。

    Args:
        compositions: 成分字典列表 [{elem: at%, ...}, ...]。
        elements: 要绘制的元素列表（默认 ALL_ELEMENTS）。
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

    if elements is None:
        elements = list(ALL_ELEMENTS)

    data = []
    labels = []
    colors = []
    for elem in elements:
        values = [c.get(elem, 0.0) for c in compositions]
        if values:
            data.append(values)
            labels.append(elem)
            colors.append(_ELEM_COLORS.get(elem, "#333333"))

    bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.6)

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.set_ylabel("Composition (at%)")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)

    return fig


def plot_composition_heatmap(
    compositions: List[Dict[str, float]],
    elements: List[str] | None = None,
    title: str = "Composition Heatmap",
    figsize: Tuple[int, int] = (10, 6),
    cmap: str = "YlOrRd",
    **kwargs,
) -> Figure:
    """绘制成分热力图（每行一个解，每列一个元素）。

    Args:
        compositions: 成分字典列表。
        elements: 要绘制的元素列表。
        title: 图标题。
        figsize: 图大小。
        cmap: 颜色映射。

    Returns:
        matplotlib Figure。
    """
    if elements is None:
        elements = list(ALL_ELEMENTS)

    if not compositions:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return fig

    n_comps = len(compositions)
    n_elems = len(elements)

    matrix = np.zeros((n_comps, n_elems))
    for i, comp in enumerate(compositions):
        for j, elem in enumerate(elements):
            matrix[i, j] = comp.get(elem, 0.0)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(matrix.T, aspect="auto", cmap=cmap, interpolation="nearest")

    ax.set_xticks(range(n_comps))
    ax.set_xticklabels([f"#{i+1}" for i in range(n_comps)], rotation=45, fontsize=7)
    ax.set_yticks(range(n_elems))
    ax.set_yticklabels(elements)
    ax.set_xlabel("Pareto Solution #")
    ax.set_title(title)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("at%")

    return fig
