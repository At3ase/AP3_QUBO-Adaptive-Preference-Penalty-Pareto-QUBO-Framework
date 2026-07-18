"""
PenaltyFlex 两阶段调度器。

- 加性启动阶段（前 t_add 轮）：使用较小学习率稳定探索
- 乘性演化阶段：使用指数更新加速收敛
- 从探索期（ε=2%）到固化期（ε=0%）的 ε 衰减
"""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class ScheduleStep:
    """调度器单步输出。

    Attributes:
        t: 当前轮数。
        alpha: 当前学习率。
        epsilon: 当前期望偏离度。
        phase: 当前阶段 ("additive" | "multiplicative")。
        allow_exploration: 是否允许探索奖励。
    """
    t: int
    alpha: float
    epsilon: float
    phase: str
    allow_exploration: bool


class AdaptiveSchedule:
    """PenaltyFlex 两阶段自适应调度器。

    阶段划分:
      轮 0 ~ t_add-1: 加性启动（alpha_add，epsilon_explore）
      轮 t_add ~ t_transition: 乘性探索（alpha_mult，epsilon_explore）
      轮 t_transition ~ t_max: 乘性固化（alpha_mult，epsilon_consolidate）

    使用示例:
        >>> schedule = AdaptiveSchedule(t_add=2, t_max=15)
        >>> for t in range(15):
        ...     step = schedule.get_step(t)
        ...     print(f"t={t}: α={step.alpha}, ε={step.epsilon}, phase={step.phase}")
    """

    def __init__(
        self,
        alpha_add: float = 0.5,
        alpha_mult: float = 0.8,
        epsilon_explore: float = 0.02,
        epsilon_consolidate: float = 0.0,
        t_add: int = 2,
        t_max: int = 15,
        t_transition: int | None = None,
    ):
        """
        Args:
            alpha_add: 加性阶段学习率。
            alpha_mult: 乘性阶段学习率。
            epsilon_explore: 探索期期望偏离度。
            epsilon_consolidate: 固化期期望偏离度。
            t_add: 加性启动轮数。
            t_max: 最大轮数。
            t_transition: 从探索到固化的过渡轮数，默认 t_add + 2。
        """
        self._alpha_add = alpha_add
        self._alpha_mult = alpha_mult
        self._epsilon_explore = epsilon_explore
        self._epsilon_consolidate = epsilon_consolidate
        self._t_add = t_add
        self._t_max = t_max
        self._t_transition = (
            t_transition if t_transition is not None else t_add + 2
        )

    def get_step(self, t: int) -> ScheduleStep:
        """返回第 t 轮的调度参数。

        Args:
            t: 当前轮数 (0-indexed)。

        Returns:
            ScheduleStep。
        """
        if t < 0:
            raise ValueError(f"t must be >= 0, got {t}")

        if t < self._t_add:
            return ScheduleStep(
                t=t,
                alpha=self._alpha_add,
                epsilon=self._epsilon_explore,
                phase="additive",
                allow_exploration=True,
            )
        elif t < self._t_transition:
            return ScheduleStep(
                t=t,
                alpha=self._alpha_mult,
                epsilon=self._epsilon_explore,
                phase="multiplicative",
                allow_exploration=True,
            )
        else:
            return ScheduleStep(
                t=t,
                alpha=self._alpha_mult,
                epsilon=self._epsilon_consolidate,
                phase="multiplicative",
                allow_exploration=False,
            )

    def get_schedule(self) -> List[ScheduleStep]:
        """返回完整调度表（0 ~ t_max-1）。"""
        return [self.get_step(t) for t in range(self._t_max)]

    def epsilon_decay_linear(
        self,
        t: int,
        t_start: int = 2,
        t_end: int = 4,
    ) -> float:
        """线性衰减 ε 从 ε_explore 到 ε_consolidate。

        Args:
            t: 当前轮数。
            t_start: 衰减起始轮数。
            t_end: 衰减结束轮数。

        Returns:
            衰减后的 ε 值。
        """
        if t < t_start:
            return self._epsilon_explore
        if t >= t_end:
            return self._epsilon_consolidate

        ratio = (t - t_start) / max(t_end - t_start, 1)
        return self._epsilon_explore + ratio * (
            self._epsilon_consolidate - self._epsilon_explore
        )

    @property
    def t_add(self) -> int:
        return self._t_add

    @property
    def t_max(self) -> int:
        return self._t_max

    @property
    def t_transition(self) -> int:
        return self._t_transition
