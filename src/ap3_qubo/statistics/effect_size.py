"""
效应量计算。

Cohen's d: 衡量两组之间差异的标准化大小。
"""

import numpy as np


def cohens_d(group_a: np.ndarray, group_b: np.ndarray) -> float:
    """Cohen's d 效应量。

    d = (mean_a − mean_b) / pooled_std

    解释:
      |d| > 0.8: 大效应
      |d| > 0.5: 中效应
      |d| > 0.2: 小效应
      |d| ≤ 0.2: 可忽略

    Args:
        group_a: 第一组数据。
        group_b: 第二组数据。

    Returns:
        Cohen's d 值。
    """
    a = np.asarray(group_a, dtype=float)
    b = np.asarray(group_b, dtype=float)

    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]

    if len(a) < 2 or len(b) < 2:
        return 0.0

    mean_a = float(np.mean(a))
    mean_b = float(np.mean(b))

    # 合并标准差
    n_a, n_b = len(a), len(b)
    var_a = np.var(a, ddof=1)
    var_b = np.var(b, ddof=1)

    pooled_std = np.sqrt(
        ((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2)
    )

    if pooled_std < 1e-12:
        return 0.0

    return float((mean_a - mean_b) / pooled_std)


def interpret_cohens_d(d: float) -> str:
    """解释 Cohen's d 的大小。

    Args:
        d: Cohen's d 值。

    Returns:
        "large" | "medium" | "small" | "negligible"。
    """
    abs_d = abs(d)
    if abs_d > 0.8:
        return "large"
    elif abs_d > 0.5:
        return "medium"
    elif abs_d > 0.2:
        return "small"
    else:
        return "negligible"
