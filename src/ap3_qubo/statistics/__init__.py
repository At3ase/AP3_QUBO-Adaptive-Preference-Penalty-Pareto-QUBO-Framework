"""
统计严谨性框架。

提供:
  - Mann-Whitney U 检验
  - Cohen's d 效应量
  - Bonferroni 多重比较校正
  - 95% 置信区间
  - 实验统计报告生成
"""

from .hypothesis_tests import (
    mann_whitney_u_test,
    bonferroni_correction,
    confidence_interval,
    StatResult,
)
from .effect_size import cohens_d, interpret_cohens_d
from .reporting import ExperimentStats, report_results

__all__ = [
    "mann_whitney_u_test",
    "bonferroni_correction",
    "confidence_interval",
    "StatResult",
    "cohens_d",
    "interpret_cohens_d",
    "ExperimentStats",
    "report_results",
]
