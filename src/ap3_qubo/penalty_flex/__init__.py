"""
PenaltyFlex: 自适应惩罚系数学习（第三层）。

根据当前解的质量反馈，动态调整 P1（碳化物抑制）和 P2（C-Cr 耦合）
的惩罚强度 λ。P0（成分和=100%）为硬约束，不参与自适应。

核心组件:
  - PenaltyFlex: 自适应 λ 控制器
  - PenaltyFlexState: 单轮状态快照
  - LambdaCache: λ 缓存（用于 warm-start）
  - FeedbackReport: TOP-K 解反馈分析报告
  - AdaptiveSchedule: 两阶段调度器
"""

from .adaptive_penalty import PenaltyFlex, PenaltyFlexState, FeedbackReport
from .warm_start import LambdaCache
from .schedule import AdaptiveSchedule

__all__ = [
    "PenaltyFlex",
    "PenaltyFlexState",
    "FeedbackReport",
    "LambdaCache",
    "AdaptiveSchedule",
]
