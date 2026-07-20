"""
假设检验工具。

提供非参数检验和多重比较校正。

审计 D-6 补齐（方案 AP3_QUBO_Validation_Scheme_v1.1 §5.1/§5.2/§4.2）：
  - wilcoxon_signed_rank：实验 0（消融）配对 Wilcoxon 符号秩检验
  - friedman_test：实验 4（γ 敏感性）Friedman 多组比较
  - variance_f_test：实验 2 超参数敏感度 σ 的 F 检验方差比较
"""

import warnings
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
        effect_size: 效应量（不适用时为 0.0；含义见 effect_size_name）。
        effect_size_name: 效应量名称（如 "rank-biserial r"、
            "Kendall's W"、"variance ratio"；空串表示未计算）。
    """
    statistic: float
    p_value: float
    significant: bool
    ci_lower: float = 0.0
    ci_upper: float = 0.0
    test_name: str = "Mann-Whitney U"
    effect_size: float = 0.0
    effect_size_name: str = ""


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


def wilcoxon_signed_rank(
    group_a: np.ndarray,
    group_b: np.ndarray,
    alternative: str = "two-sided",
    alpha: float = 0.01,
) -> StatResult:
    """Wilcoxon 符号秩检验（配对样本，非参数，无正态性假设）。

    方案出处：AP3_QUBO_Validation_Scheme_v1.1 §5.1 —— 实验 0（消融）
    显著性检验指定"配对 Wilcoxon 符号秩检验"，阈值 p < 0.01，
    并配合 Bonferroni 多重比较校正（见 bonferroni_correction）。

    配对语义：a[i] 与 b[i] 为同一重复（同 seed）下两配置的配对观测，
    检验对象为差值 d = a − b；差值为 0 的配对按 scipy 默认
    zero_method="wilcox" 在检验内部丢弃（秩次不计）。

    效应量：matched-pairs rank-biserial correlation
        r_rb = (W+ − W−) / (W+ + W−) ∈ [−1, 1]
    其中 W+ / W− 为非零差值 |d| 的秩次按符号分组求和；r_rb > 0 表示
    a 系统性高于 b，|r_rb| 越接近 1 配对差异越大。全部差值为 0 时
    无差异方向，效应量记 0.0。

    Args:
        group_a: 第一组配对数据（如各 Abl 配置逐 rep HV）。
        group_b: 第二组配对数据（如 Full 配置逐 rep HV）。
        alternative: "two-sided" | "less" | "greater"。
        alpha: 显著性水平（方案实验 0 口径 0.01）。

    Returns:
        StatResult（effect_size 为 rank-biserial correlation）。

    Raises:
        ValueError: 两组长度不一致，或 NaN 成对清洗后无有效配对。
    """
    a = np.asarray(group_a, dtype=float)
    b = np.asarray(group_b, dtype=float)

    if a.shape != b.shape:
        raise ValueError(
            f"Paired test requires equal-length groups, "
            f"got {a.shape} vs {b.shape}"
        )

    # 成对移除 NaN：任一侧缺测则该配对作废
    valid = ~(np.isnan(a) | np.isnan(b))
    a, b = a[valid], b[valid]

    if len(a) == 0:
        raise ValueError("No valid paired observations after NaN removal")

    with warnings.catch_warnings():
        # scipy 1.15 在 n 很小或差值全零时，内部正态近似的 z 计算
        # 触发 RuntimeWarning（invalid value in scalar divide），
        # 但最终返回的 exact p 值有效；此处仅抑制告警噪声。
        warnings.simplefilter("ignore", RuntimeWarning)
        statistic, p_value = stats.wilcoxon(a, b, alternative=alternative)

    # 效应量：rank-biserial correlation（零差值与检验同口径剔除）
    d = a - b
    d_nz = d[d != 0.0]
    if len(d_nz) > 0:
        ranks = stats.rankdata(np.abs(d_nz))
        w_plus = float(ranks[d_nz > 0.0].sum())
        w_minus = float(ranks[d_nz < 0.0].sum())
        r_rb = (w_plus - w_minus) / (w_plus + w_minus)
    else:
        r_rb = 0.0

    return StatResult(
        statistic=float(statistic),
        p_value=float(p_value),
        significant=bool(p_value < alpha),
        test_name="Wilcoxon signed-rank",
        effect_size=float(r_rb),
        effect_size_name="rank-biserial r",
    )


def friedman_test(
    groups: List[np.ndarray],
    alpha: float = 0.01,
) -> StatResult:
    """Friedman 检验（k ≥ 3 组相关样本的非参数多组比较）。

    方案出处：AP3_QUBO_Validation_Scheme_v1.1 §5.1 —— 实验 4
    （γ 敏感性）指定"Friedman 检验（多组比较）"：每次完整 pipeline
    重复为一个区组（block），比较 k 个 γ 取值下指标分布是否一致。

    效应量：Kendall's W = χ² / (n·(k−1)) ∈ [0, 1]，n 为区组数、
    k 为组数；W → 0 表示各 γ 组间一致（对 γ 不敏感，方案期望方向），
    W → 1 表示组间强分化。存在 ties 时为与 scipy 统计量同源的近似
    口径。

    Args:
        groups: k 组等长数据（k ≥ 3），groups[j][i] 为第 i 个区组
            在第 j 个处理（γ 值）下的观测。
        alpha: 显著性水平。

    Returns:
        StatResult（effect_size 为 Kendall's W）。

    Raises:
        ValueError: 组数 < 3、各组长度不一致，或 NaN 区组清洗后
            无有效区组。
    """
    if len(groups) < 3:
        raise ValueError(
            f"Friedman test requires ≥ 3 groups, got {len(groups)}"
        )

    cleaned = [np.asarray(g, dtype=float) for g in groups]
    n = len(cleaned[0])
    if any(len(g) != n for g in cleaned):
        raise ValueError(
            "Friedman test requires equal-length groups (blocked design)"
        )

    # 区组级 NaN 移除：任一处理缺测则该区组作废
    mat = np.vstack(cleaned)  # (k, n)
    valid = ~np.isnan(mat).any(axis=0)
    mat = mat[:, valid]
    if mat.shape[1] == 0:
        raise ValueError("No valid blocks after NaN removal")

    statistic, p_value = stats.friedmanchisquare(*list(mat))
    k, n_valid = mat.shape
    kendalls_w = float(statistic) / (n_valid * (k - 1))

    return StatResult(
        statistic=float(statistic),
        p_value=float(p_value),
        significant=bool(p_value < alpha),
        test_name="Friedman",
        effect_size=kendalls_w,
        effect_size_name="Kendall's W",
    )


def variance_f_test(
    group_a: np.ndarray,
    group_b: np.ndarray,
    alpha: float = 0.01,
) -> StatResult:
    """F 检验（两方差齐性比较，假设总体近似正态）。

    方案出处：AP3_QUBO_Validation_Scheme_v1.1 §4.2 评价指标表 ——
    实验 2 超参数敏感度 σ（多次运行 f* 的标准差）指定"20 次重复，
    F 检验方差比较"，期望 PenaltyFlex 的 σ 小于固定 λ。

    统计量 F = s_a² / s_b²（ddof=1），自由度 (n_a−1, n_b−1)；
    双侧 p = 2·min(P(F ≤ f), P(F ≥ f))（截断到 1.0）。效应量即方差
    比 variance ratio = s_a² / s_b²（与 F 统计量同值）：< 1 表示
    a 组波动更小（更稳定）。b 组零方差而 a 组有方差时方差比发散，
    F 记 inf、p → 0。

    Args:
        group_a: 第一组数据（如 PenaltyFlex 的逐 rep f*）。
        group_b: 第二组数据（如固定 λ 的逐 rep f*）。
        alpha: 显著性水平。

    Returns:
        StatResult（effect_size 为方差比）。

    Raises:
        ValueError: 任一组有效样本 < 2，或两组方差均为 0（F 无定义）。
    """
    a = np.asarray(group_a, dtype=float)
    b = np.asarray(group_b, dtype=float)
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]

    if len(a) < 2 or len(b) < 2:
        raise ValueError("F test requires ≥ 2 observations per group")

    var_a = float(np.var(a, ddof=1))
    var_b = float(np.var(b, ddof=1))

    if var_a == 0.0 and var_b == 0.0:
        raise ValueError("Both groups have zero variance; F is undefined")

    if var_b == 0.0:
        f_stat = np.inf  # a 有方差、b 无方差 → 方差比发散
    else:
        f_stat = var_a / var_b

    df_a, df_b = len(a) - 1, len(b) - 1
    p_value = 2.0 * min(
        stats.f.cdf(f_stat, df_a, df_b),
        stats.f.sf(f_stat, df_a, df_b),
    )
    p_value = min(p_value, 1.0)

    return StatResult(
        statistic=float(f_stat),
        p_value=float(p_value),
        significant=bool(p_value < alpha),
        test_name="F test (variance ratio)",
        effect_size=float(f_stat),
        effect_size_name="variance ratio",
    )
