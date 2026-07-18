"""
假设检验工具。

提供非参数检验和多重比较校正。
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from scipy import stats


@dataclass
class StatResult:
    """统计检验结果。

    Attributes:
        statistic: 检验统计量。
        p_value: p 值。
        significant: 是否显著（p < alpha）。
        ci_lower: 置信区间下界。
        ci_upper: 置信区间上界。
        test_name: 检验名称。
    """
    statistic: float
    p_value: float
    significant: bool
    ci_lower: float = 0.0
    ci_upper: float = 0.0
    test_name: str = "Mann-Whitney U"


def mann_whitney_u_test(
    group_a: np.ndarray,
    group_b: np.ndarray,
    alternative: str = "two-sided",
    alpha: float = 0.05,
) -> StatResult:
    """Mann-Whitney U 检验（非参数，无正态性假设）。

    Args:
        group_a: 第一组数据。
        group_b: 第二组数据。
        alternative: "two-sided" | "less" | "greater"。
        alpha: 显著性水平。

    Returns:
        StatResult。

    Raises:
        ValueError: 如果两组数据均为空。
    """
    a = np.asarray(group_a, dtype=float)
    b = np.asarray(group_b, dtype=float)

    if len(a) == 0 or len(b) == 0:
        raise ValueError("Both groups must have at least one element")

    # 移除 NaN
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]

    statistic, p_value = stats.mannwhitneyu(a, b, alternative=alternative)

    # Bootstrap 95% CI (差值中位数的置信区间)
    n_bootstrap = 2000
    diffs = []
    rng = np.random.default_rng(42)
    for _ in range(n_bootstrap):
        a_sample = rng.choice(a, size=len(a), replace=True)
        b_sample = rng.choice(b, size=len(b), replace=True)
        diffs.append(np.median(a_sample) - np.median(b_sample))
    diffs = np.sort(diffs)
    ci_lower = float(diffs[int(n_bootstrap * 0.025)])
    ci_upper = float(diffs[int(n_bootstrap * 0.975)])

    return StatResult(
        statistic=float(statistic),
        p_value=float(p_value),
        significant=p_value < alpha,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        test_name="Mann-Whitney U",
    )


def bonferroni_correction(
    p_values: List[float],
    n_comparisons: int | None = None,
) -> List[float]:
    """Bonferroni 多重比较校正。

    adjusted = min(p × n_comparisons, 1.0)

    Args:
        p_values: 原始 p 值列表。
        n_comparisons: 总比较次数。若为 None，默认为 len(p_values)。
                       显式传入可应对仅对 family-wise 检验子集进行校正的场景。

    Returns:
        校正后的 p 值列表。
    """
    n = n_comparisons if n_comparisons is not None else len(p_values)
    if n == 0 or len(p_values) == 0:
        return []
    return [min(p * n, 1.0) for p in p_values]


def confidence_interval(
    data: np.ndarray,
    confidence: float = 0.95,
) -> Tuple[float, float]:
    """计算均值的置信区间。

    n ≥ 30: 正态近似（z 分布）
    n < 30: t 分布

    Args:
        data: 数据数组。
        confidence: 置信水平（默认 0.95）。

    Returns:
        (lower, upper)。
    """
    d = np.asarray(data, dtype=float)
    d = d[~np.isnan(d)]

    if len(d) == 0:
        return (0.0, 0.0)

    mean = float(np.mean(d))
    n = len(d)

    if n >= 30:
        # z 分布（正态近似）
        z = stats.norm.ppf(1.0 - (1.0 - confidence) / 2.0)
        sem = float(np.std(d, ddof=1) / np.sqrt(n))
        return (mean - z * sem, mean + z * sem)
    else:
        # t 分布
        se = stats.sem(d)
        if se is None or np.isnan(se):
            return (mean, mean)
        ci = stats.t.interval(
            confidence, df=n - 1, loc=mean, scale=se
        )
        return (float(ci[0]), float(ci[1]))
