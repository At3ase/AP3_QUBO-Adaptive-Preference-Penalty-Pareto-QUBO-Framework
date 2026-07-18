"""
PenaltyFlex 核心自适应惩罚算法。

管理 P1 (λ_carbide) 和 P2 (λ_ccr) 的自适应更新。
P0 (λ_sum) 为固定硬约束，不参与自适应。

更新公式（加性启动阶段，t ≤ T_add，方案 E-04 分支①）:
    λ_j^(t+1) = λ_j^(t) + α_add · v̄_j        （v̄_j ≥ 0 恒成立，只升不降）

更新公式（乘性演化阶段，t > T_add，方案 E-04 分支③）:
    λ_j^(t+1) = λ_j^(t) × exp(α · tanh(v̄_j − ε_j))

振荡抑制（连续两轮反向调整，方案 E-04 分支②）:
    λ_j^(t+1) = √(λ_j^(t) · λ_j^(t−1))       （当前值与上一轮值的几何平均）

公式出处: plan/hea_encoding_scheme_v1.13.md :306-345（PenaltyFlex 完整算法伪代码）

其中:
  - v̄_j: TOP-10 解的平均约束 j 违反度
  - ε_j: 期望偏离度（探索期 2%，固化期 0%）
  - α: 学习率（加性启动阶段 α_add，乘性演化阶段 α_mult）

四种反馈模式:
  - 探索奖励: 最优解违反但目标极优 → 维持/降低 λ
  - 收紧信号: 所有解满足约束但目标平庸 → 增大 λ
  - 边界聚焦: Pareto 前沿恰在约束边界 → 微调 λ
  - 振荡抑制: λ 连续两轮反向调整 → 改用几何平均
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

import numpy as np

from ..physical_params import CONSTRAINT


@dataclass
class FeedbackReport:
    """TOP-K 解的反馈分析报告。

    Attributes:
        v_carbide: P1 平均违反度 (c_C − 0.8)² 的均值。
        v_ccr: P2 平均违反度 c_C·c_Cr / H_max 的均值。
        best_objective: 最优目标值。
        worst_objective: 最差目标值（TOP-K 内）。
        mean_objective: 平均目标值。
        num_feasible: P0 可行解数量。
        num_total: 总解数量。
        is_oscillating: 是否检测到振荡（连续两轮反向调整）。
    """
    v_carbide: float
    v_ccr: float
    best_objective: float
    worst_objective: float = 0.0
    mean_objective: float = 0.0
    num_feasible: int = 0
    num_total: int = 0
    is_oscillating: bool = False


@dataclass
class PenaltyFlexState:
    """PenaltyFlex 单轮状态快照（不可变）。

    Attributes:
        t: 当前迭代轮数。
        lambda_carbide: P1 当前 λ。
        lambda_ccr: P2 当前 λ。
        epsilon: 当前期望偏离度。
        phase: 当前阶段 ("additive" | "multiplicative")。
        v_carbide: 本轮 P1 平均违反度。
        v_ccr: 本轮 P2 平均违反度。
        delta_carbide: λ_carbide 相对变化。
        delta_ccr: λ_ccr 相对变化。
        is_converged: 是否已收敛。
    """
    t: int
    lambda_carbide: float
    lambda_ccr: float
    epsilon: float
    phase: str
    v_carbide: float
    v_ccr: float
    delta_carbide: float = 0.0
    delta_ccr: float = 0.0
    is_converged: bool = False


class PenaltyFlex:
    """自适应 λ 学习控制器。

    仅控制 P1（碳化物抑制）和 P2（C-Cr 耦合）。
    P0（成分和=100%）固定为 CONSTraint.lambda_sum_fixed，不参与自适应。

    两阶段策略:
      - 加性启动（前 t_add 轮）：用较小学习率稳定探索
      - 乘性演化：用指数更新加速收敛

    使用示例:
        >>> pf = PenaltyFlex()
        >>> for t in range(15):
        ...     feedback = analyze_solutions(solutions)
        ...     state = pf.step(feedback)
        ...     if state.is_converged:
        ...         break
    """

    def __init__(
        self,
        lambda_carbide_init: float | None = None,
        lambda_ccr_init: float | None = None,
        alpha_add: float | None = None,
        alpha_mult: float | None = None,
        epsilon_explore: float | None = None,
        epsilon_consolidate: float | None = None,
        gamma_convergence: float | None = None,
        t_add: int | None = None,
        t_max: int | None = None,
    ):
        """
        Args:
            lambda_carbide_init: P1 初始 λ，默认 CONSTRAINT.lambda_carbide_init (0.05)。
            lambda_ccr_init: P2 初始 λ，默认 CONSTRAINT.lambda_ccr_init (0.05)。
            alpha_add: 加性阶段学习率，默认 CONSTRAINT.alpha_add (0.5)。
            alpha_mult: 乘性阶段学习率，默认 CONSTRAINT.alpha_mult (0.8)。
            epsilon_explore: 探索期期望偏离度，默认 0.02。
            epsilon_consolidate: 固化期期望偏离度，默认 0.0。
            gamma_convergence: λ 相对变化收敛阈值，默认 0.1。
            t_add: 加性启动轮数，默认 2。
            t_max: 最大迭代轮数，默认 15。
        """
        self._lambda_carbide = (
            lambda_carbide_init
            if lambda_carbide_init is not None
            else CONSTRAINT.lambda_carbide_init
        )
        self._lambda_ccr = (
            lambda_ccr_init
            if lambda_ccr_init is not None
            else CONSTRAINT.lambda_ccr_init
        )
        self._alpha_add = alpha_add if alpha_add is not None else CONSTRAINT.alpha_add
        self._alpha_mult = (
            alpha_mult if alpha_mult is not None else CONSTRAINT.alpha_mult
        )
        self._epsilon_explore = (
            epsilon_explore
            if epsilon_explore is not None
            else CONSTRAINT.epsilon_explore
        )
        self._epsilon_consolidate = (
            epsilon_consolidate
            if epsilon_consolidate is not None
            else CONSTRAINT.epsilon_consolidate
        )
        self._gamma = (
            gamma_convergence
            if gamma_convergence is not None
            else CONSTRAINT.gamma_convergence
        )
        self._t_add = t_add if t_add is not None else CONSTRAINT.t_add
        self._t_max = t_max if t_max is not None else CONSTRAINT.t_max

        # 状态追踪
        self._t = 0
        self._prev_lambda_carbide: float | None = None
        self._prev_lambda_ccr: float | None = None
        # 上上一轮 λ 历史槽位：方案 E-04 分支②振荡几何平均 √(λ_t · λ_{t-1})
        # 需要"当前值与上一轮值"两步历史；_prev_* 在 step() 开头即被赋为当前值，
        # 单靠它无法取到 λ_{t-1}，故增加 _prev_prev_* 做历史上移
        self._prev_prev_lambda_carbide: float | None = None
        self._prev_prev_lambda_ccr: float | None = None
        self._converged_count = 0
        self._oscillation_count = 0
        self._last_direction: Tuple[int, int] = (0, 0)  # (+1/-1 for each λ)

    # =========================================================================
    # 属性
    # =========================================================================

    @property
    def lambda_carbide(self) -> float:
        return self._lambda_carbide

    @property
    def lambda_ccr(self) -> float:
        return self._lambda_ccr

    @property
    def current_phase(self) -> str:
        # 方案 E-04（hea_encoding_scheme_v1.13.md :334）：t ≤ T_add 为加性启动阶段，
        # 含 t = T_add 本身（T_add=2 时 t=1,2 均走加性分支）。
        return "additive" if self._t <= self._t_add else "multiplicative"

    @property
    def is_converged(self) -> bool:
        return self._converged_count >= 2

    @property
    def current_epsilon(self) -> float:
        """探索期用 ε_explore，固化期用 ε_consolidate。"""
        return (
            self._epsilon_explore
            if self._t < self._t_add + 2
            else self._epsilon_consolidate
        )

    @property
    def t(self) -> int:
        return self._t

    # =========================================================================
    # 核心更新逻辑
    # =========================================================================

    def step(self, feedback: FeedbackReport) -> PenaltyFlexState:
        """处理一轮反馈并更新 λ。

        Args:
            feedback: 本轮 TOP-K 解的反馈分析报告。

        Returns:
            更新后的 PenaltyFlexState。
        """
        self._t += 1
        # λ 历史上移：_prev_prev_* ← 上一轮值 ← 当前值。
        # _prev_* 供下方相对变化计算；_prev_prev_* 供振荡几何平均取 λ_{t-1}（方案 E-04 分支②）
        self._prev_prev_lambda_carbide = self._prev_lambda_carbide
        self._prev_prev_lambda_ccr = self._prev_lambda_ccr
        self._prev_lambda_carbide = self._lambda_carbide
        self._prev_lambda_ccr = self._lambda_ccr

        # 检测振荡
        if self._detect_oscillation(feedback):
            feedback = FeedbackReport(
                v_carbide=feedback.v_carbide,
                v_ccr=feedback.v_ccr,
                best_objective=feedback.best_objective,
                worst_objective=feedback.worst_objective,
                mean_objective=feedback.mean_objective,
                num_feasible=feedback.num_feasible,
                num_total=feedback.num_total,
                is_oscillating=True,
            )

        # 计算新的 λ
        alpha = self._alpha_add if self.current_phase == "additive" else self._alpha_mult
        eps = self.current_epsilon

        # 分支优先级同方案 E-04 伪代码（:334-340）：加性启动 → 振荡几何平均 → 乘性演化
        if self.current_phase == "additive":
            # 方案 E-04 分支①（hea_encoding_scheme_v1.13.md :334-335）：
            #   λ_j^(t+1) = λ_j^(t) + α_add · v̄_j，t ≤ T_add（T_add=2）
            # v̄_j ≥ 0 恒成立（P1 为平方项、P2 为非负归一化项），故加性阶段只升不降
            new_lc = self._lambda_carbide + alpha * feedback.v_carbide
            new_lccr = self._lambda_ccr + alpha * feedback.v_ccr
        elif feedback.is_oscillating:
            # 方案 E-04 分支②（hea_encoding_scheme_v1.13.md :336-338）：
            #   λ_j^(t+1) = √(λ_j^(t) · λ_j^(t−1))
            # 即"当前值与上一轮值"的几何平均，而非当前值与乘性候选的几何平均（审查 P1-1 修复）。
            # _prev_prev_* 即 λ_{t-1}；本分支仅在 t > T_add ≥ 2 时可达，其必已被赋值
            new_lc = float(
                np.sqrt(self._lambda_carbide * self._prev_prev_lambda_carbide)
            )
            new_lccr = float(
                np.sqrt(self._lambda_ccr * self._prev_prev_lambda_ccr)
            )
        else:
            # 方案 E-04 分支③（hea_encoding_scheme_v1.13.md :339-340）：标准乘性更新
            new_lc = self._lambda_carbide * np.exp(
                alpha * np.tanh(feedback.v_carbide - eps)
            )
            new_lccr = self._lambda_ccr * np.exp(
                alpha * np.tanh(feedback.v_ccr - eps)
            )

        # 边界约束
        self._lambda_carbide = self._clamp(new_lc)
        self._lambda_ccr = self._clamp(new_lccr)

        # 计算相对变化
        delta_c = (
            abs(self._lambda_carbide - self._prev_lambda_carbide)
            / max(self._prev_lambda_carbide, 1e-12)
            if self._prev_lambda_carbide is not None
            else 0.0
        )
        delta_ccr = (
            abs(self._lambda_ccr - self._prev_lambda_ccr)
            / max(self._prev_lambda_ccr, 1e-12)
            if self._prev_lambda_ccr is not None
            else 0.0
        )

        # 收敛判定（方案 E-07，hea_encoding_scheme_v1.13.md :342-343）：
        #   MAX_j |Δλ_j|/λ_j < γ AND t > T_add —— 加性启动阶段（t ≤ T_add）不判收敛，
        #   门控缺失时加性阶段 v̄≈0 会误触发收敛（审查 P1-2 修复）。
        # 注：要求"连续两轮"满足 < γ（is_converged 属性 :_converged_count >= 2）
        #     是在方案基础上的加严，方案原文为单轮满足即 BREAK。
        max_delta = max(delta_c, delta_ccr)
        if self._t > self._t_add and max_delta < self._gamma:
            self._converged_count += 1
        else:
            self._converged_count = 0

        is_conv = self.is_converged or self._t >= self._t_max

        return PenaltyFlexState(
            t=self._t,
            lambda_carbide=self._lambda_carbide,
            lambda_ccr=self._lambda_ccr,
            epsilon=eps,
            phase=self.current_phase,
            v_carbide=feedback.v_carbide,
            v_ccr=feedback.v_ccr,
            delta_carbide=delta_c,
            delta_ccr=delta_ccr,
            is_converged=is_conv,
        )

    def _detect_oscillation(self, feedback: FeedbackReport) -> bool:
        """检测 λ 是否在连续两轮反向调整。"""
        if self._t < 2:
            return False

        current_dir_c = 1 if feedback.v_carbide > self.current_epsilon else -1
        current_dir_ccr = 1 if feedback.v_ccr > self.current_epsilon else -1

        prev_dir_c, prev_dir_ccr = self._last_direction

        oscillates_c = (
            current_dir_c != 0
            and prev_dir_c != 0
            and current_dir_c != prev_dir_c
        )
        oscillates_ccr = (
            current_dir_ccr != 0
            and prev_dir_ccr != 0
            and current_dir_ccr != prev_dir_ccr
        )

        self._last_direction = (current_dir_c, current_dir_ccr)

        if oscillates_c or oscillates_ccr:
            self._oscillation_count += 1
            return True
        self._oscillation_count = 0
        return False

    def _clamp(self, value: float, min_val: float = 0.005, max_val: float = 5.0) -> float:
        """将 λ 限制在合理范围内 [min_val, max_val]。"""
        return max(min_val, min(value, max_val))

    def get_current_lambdas(self) -> Tuple[float, float]:
        """返回当前 (λ_carbide, λ_ccr)。"""
        return (self._lambda_carbide, self._lambda_ccr)

    def warm_start_from(self, lambdas: Tuple[float, float]) -> None:
        """用邻居权重组已收敛的 λ 值初始化当前控制器。

        这避免了从默认 λ_init 开始——利用相似权重组
        （在权重空间中相邻）应该有相似最优 λ 的直觉。

        Args:
            lambdas: (lambda_carbide, lambda_ccr) 从邻居权重组的
                     PenaltyFlex 内循环收敛后取得。
        """
        self._lambda_carbide = float(lambdas[0])
        self._lambda_ccr = float(lambdas[1])
        # 将内部状态重置到干净起点，但保留 warm-start λ
        self._t = 0
        self._prev_lambda_carbide = None
        self._prev_lambda_ccr = None
        self._prev_prev_lambda_carbide = None
        self._prev_prev_lambda_ccr = None
        self._converged_count = 0
        self._oscillation_count = 0
        self._last_direction = (0, 0)

    def reset(self) -> None:
        """重置到初始状态（用于新的权重组）。"""
        self._t = 0
        self._prev_lambda_carbide = None
        self._prev_lambda_ccr = None
        self._prev_prev_lambda_carbide = None
        self._prev_prev_lambda_ccr = None
        self._converged_count = 0
        self._oscillation_count = 0
        self._last_direction = (0, 0)

    # =========================================================================
    # 静态工具方法
    # =========================================================================

    @staticmethod
    def compute_violation_p1(c_carbon: float) -> float:
        """计算 P1 违反度: (c_C − 0.8)²（at% 尺度）。"""
        return float((c_carbon - CONSTRAINT.carbide_soft_upper) ** 2)

    @staticmethod
    def compute_violation_p2(c_carbon: float, c_cr: float) -> float:
        """计算 P2 违反度: c_C × c_Cr / H_max。"""
        return float(c_carbon * c_cr / CONSTRAINT.ccr_h_max)

    @staticmethod
    def analyze_top_k(
        solutions_data: List[Dict],
        k: int = 10,
    ) -> FeedbackReport:
        """从 TOP-K 解的字典列表生成反馈报告。

        Args:
            solutions_data: 每个解为 dict，需包含:
                - "c_carbon": C 含量 (at%)
                - "c_cr": Cr 含量 (at%)
                - "objective": QUBO 目标值（最小化问题，越小越优）
                - "is_feasible": 是否满足 P0 (bool)
                列表无需预排序，内部按 "objective" 升序排序后取前 k 个。
            k: 取 TOP-k 解进行分析。

        Returns:
            FeedbackReport。
        """
        if not solutions_data:
            return FeedbackReport(
                v_carbide=0.0,
                v_ccr=0.0,
                best_objective=0.0,
                num_feasible=0,
                num_total=0,
            )

        # 方案 E（hea_encoding_scheme_v1.13.md :310,333）：TOP-K 反馈应取目标值最优的 k 个解。
        # 先按 objective 升序排序再取前 k 个，而非直接取列表头 k 个
        # （审查 P1-3 修复：不排序时 TOP-K 语义失真，结果取决于传入顺序）。
        top_k = sorted(solutions_data, key=lambda s: s["objective"])[:k]

        violations_c = [
            PenaltyFlex.compute_violation_p1(s["c_carbon"]) for s in top_k
        ]
        violations_ccr = [
            PenaltyFlex.compute_violation_p2(s["c_carbon"], s["c_cr"])
            for s in top_k
        ]
        objectives = [s["objective"] for s in top_k]
        feasible = [s.get("is_feasible", True) for s in top_k]

        v_carbide = float(np.mean(violations_c))
        v_ccr = float(np.mean(violations_ccr))

        return FeedbackReport(
            v_carbide=v_carbide,
            v_ccr=v_ccr,
            best_objective=float(min(objectives)),
            worst_objective=float(max(objectives)),
            mean_objective=float(np.mean(objectives)),
            num_feasible=sum(1 for f in feasible if f),
            num_total=len(top_k),
        )
