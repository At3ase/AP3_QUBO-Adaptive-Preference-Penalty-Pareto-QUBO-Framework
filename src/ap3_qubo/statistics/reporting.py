"""
统计报告生成。

将实验结果格式化为可发表的统计报告。
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from .hypothesis_tests import mann_whitney_u_test, confidence_interval, StatResult
from .effect_size import cohens_d, interpret_cohens_d


@dataclass
class ExperimentStats:
    """单次实验的统计结果容器。

    Attributes:
        name: 实验名称。
        group_names: 各组名称列表。
        metrics: {指标名 → {组名 → [值列表]}}。
        comparisons: {指标名 → StatResult}。
    """
    name: str
    group_names: List[str] = field(default_factory=list)
    metrics: Dict[str, Dict[str, List[float]]] = field(default_factory=dict)
    comparisons: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def add_metric(self, metric_name: str, group_name: str, values: List[float]) -> None:
        """添加一组指标值（A-1 修复后：追加/累积语义）。

        A-1 修复（缺口扫描致命聚合缺陷）：原为覆盖式赋值
        `self.metrics[metric_name][group_name] = list(values)`，
        comparison.py 三个实验在 rep 循环内每次以单元素列表调用
        （comparison.py:67-68 / :132-133 / :195-196），n_repetitions
        次重复跑完每组只剩最后一次的样本，Mean±SD/CI/假设检验全部
        失真。现改为 extend 追加：逐 rep 累积，每组样本数 = 重复次数。
        兼容性：单组单次调用方（run_experiments._stats_to_report_md
        一次性传完整 rep 列表、summarize_experiment）在全新
        ExperimentStats 上 extend 等价于 set，行为不变；
        run_experiments 驱动层"n_repetitions=1 逐 rep 自行聚合"的
        绕开方式同样保持正确。

        Args:
            metric_name: 指标名称（如 "HV"）。
            group_name: 组名称（如 "ParetoZoom"）。
            values: 本次添加的值列表（追加到该组已有值之后）。
        """
        if metric_name not in self.metrics:
            self.metrics[metric_name] = {}
        self.metrics[metric_name].setdefault(group_name, []).extend(values)

        if group_name not in self.group_names:
            self.group_names.append(group_name)

    def compare(
        self,
        metric_name: str,
        group_a: str,
        group_b: str,
    ) -> StatResult:
        """比较两组在指定指标上的差异。

        Args:
            metric_name: 指标名称。
            group_a: 组 A 名称。
            group_b: 组 B 名称。

        Returns:
            StatResult。
        """
        vals_a = np.array(self.metrics.get(metric_name, {}).get(group_a, []))
        vals_b = np.array(self.metrics.get(metric_name, {}).get(group_b, []))

        return mann_whitney_u_test(vals_a, vals_b)


def report_results(
    stats: ExperimentStats,
    baseline_group: str | None = None,
) -> str:
    """生成 Markdown 格式的统计报告。

    Args:
        stats: 实验统计数据。
        baseline_group: 基线组名称（用于 Cohen's d 比较）。

    Returns:
        Markdown 格式的报告字符串。
    """
    lines = []
    lines.append(f"## {stats.name}")
    lines.append("")

    for metric_name, groups in stats.metrics.items():
        lines.append(f"### {metric_name}")
        lines.append("")
        lines.append(
            "| Group | Mean ± SD | 95% CI | p-value | Cohen's d |"
        )
        lines.append(
            "|-------|-----------|--------|---------|-----------|"
        )

        baseline_values = None
        if baseline_group and baseline_group in groups:
            baseline_values = np.array(groups[baseline_group])

        for group_name in stats.group_names:
            if group_name not in groups:
                continue
            values = np.array(groups[group_name])
            if len(values) == 0:
                continue

            mean = float(np.mean(values))
            sd = float(np.std(values, ddof=1))
            ci = confidence_interval(values)

            # vs baseline 比较
            if baseline_values is not None and group_name != baseline_group:
                try:
                    stat_result = mann_whitney_u_test(values, baseline_values)
                    p_str = f"{stat_result.p_value:.4f}"
                    if stat_result.significant:
                        p_str += " *"
                    d = cohens_d(values, baseline_values)
                    d_str = f"{d:.3f} ({interpret_cohens_d(d)})"
                except ValueError:
                    p_str = "N/A"
                    d_str = "N/A"
            else:
                p_str = "—"
                d_str = "—"

            lines.append(
                f"| {group_name} | {mean:.4f} ± {sd:.4f} | "
                f"[{ci[0]:.4f}, {ci[1]:.4f}] | {p_str} | {d_str} |"
            )

        lines.append("")

    # 注释
    lines.append("---")
    lines.append("*p < 0.05 (未校正)")

    return "\n".join(lines)


def summarize_experiment(
    results: Dict[str, List[float]],
    name: str = "Experiment",
) -> str:
    """快速实验总结（便捷函数）。

    Args:
        results: {group_name: [values]}。
        name: 实验名称。

    Returns:
        格式化的报告字符串。
    """
    stats = ExperimentStats(name=name)
    # 将所有指标合并为一个 "combined" 指标
    for group_name, values in results.items():
        stats.add_metric("combined", group_name, values)

    baseline = list(results.keys())[0] if results else None
    return report_results(stats, baseline_group=baseline)
