"""
ParetoZoom 动态前沿探索算法（第四层核心）。

五阶段流程:
  A: 粗网格初始化（12 组权重 G1~G12 + PenaltyFlex 内循环）
  B: 间隙检测 → 在相邻非支配解间插入新权重组
  C: HV 热点导向微扰 → 在高贡献区域高斯扰动
  D: QUBO 求解（CIM 真机 / 内置 SA 后端）+ 存档更新（warm-start PenaltyFlex）
  E: HV 收敛判定 → ΔHV < 1% 连续 2 轮 → 停止

双层自适应架构:
  - 外层 ParetoZoom：控制 (w1, w2, w3) → 前沿质量
  - 内层 PenaltyFlex：控制 λ_carbide, λ_ccr → 约束满足
  - warm-start：新权重组继承最近邻已收敛的 λ*
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..physical_params import (
    COARSE_WEIGHTS,
    PARETO_ZOOM,
    CONSTRAINT,
    ALL_ELEMENTS,
    MAIN_ELEMENTS,
    INTERSTITIAL_ELEMENT,
)
from ..encoding.precision_split import PrecisionSplitDecoder, Composition
from ..qubo.builder import QUBOBuilder
from ..solver.base import AbstractSolver, Solution as SolverSolution
from ..solver.kaiwu_solver import KaiwuSolver
from ..objectives.mixing_enthalpy import MixingEnthalpy
from ..objectives.density import VegardDensity
from ..objectives.cost import WeightedCost
from ..objectives.normalization import PhysicalPriorNormalizer
from ..penalty_flex.adaptive_penalty import PenaltyFlex, FeedbackReport
from ..penalty_flex.warm_start import LambdaCache
from ..validation.pareto import SolutionRecord as ValidatedRecord
from ..validation.hypervolume import HypervolumeCalculator, set_unified_reference
from ..validation.physical_filters import PhysicalFilter, PhysicalFilterResult

from .archive import Archive
from .weight_utils import WeightGenerator, deduplicate_weights

logger = logging.getLogger(__name__)

# 方案 BASE-4（AP3_QUBO_Validation_Scheme_v1.1 §4.2 :160）Grid-search 网格：
# 在 {0.1, 0.5, 1, 5, 10, 50, 100} 上穷举搜索最优固定 λ。
GRID_SEARCH_LAMBDA_GRID: Tuple[float, ...] = (0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0)


@dataclass
class ParetoZoomRound:
    """单轮 ParetoZoom 的结果。

    Attributes:
        round_id: 轮数编号。
        weights_solved: 本轮求解的权重组。
        new_solutions: 本轮产生的新解。
        hv_before: 本轮前 HV。
        hv_after: 本轮后 HV。
        delta_hv: HV 相对变化。
        gaps_detected: 检测到的间隙数。
        perturbations_generated: 生成的微扰权重数。
    """
    round_id: int
    weights_solved: List[Tuple[float, float, float]] = field(default_factory=list)
    new_solutions: List[ValidatedRecord] = field(default_factory=list)
    hv_before: float = 0.0
    hv_after: float = 0.0
    delta_hv: float = 0.0
    gaps_detected: int = 0
    perturbations_generated: int = 0


class ParetoZoom:
    """ParetoZoom 动态前沿探索器。

    使用示例:
        >>> pz = ParetoZoom()
        >>> archive, rounds = pz.run()
        >>> front = archive.get_fractions_of_front()
        >>> for comp in front:
        ...     print(comp)
    """

    def __init__(
        self,
        initial_weights: List[Tuple[float, float, float]] | None = None,
        solver: AbstractSolver | None = None,
        builder: QUBOBuilder | None = None,
        encoding_type: str = "precision_split_38",
        gamma_discount: float | None = None,
        penalty_strategy: str = "adaptive",
        penalty_fixed_lambda: float | None = None,
        exploration_strategy: str = "pareto_zoom",
        uniform_grid_n: int = 50,
        archive_feasible_tol: float = 1.0,
        seed: int | None = None,
    ):
        """
        Args:
            initial_weights: 初始权重组（默认 COARSE_WEIGHTS 12 组）。
            solver: QUBO 求解器（默认 KaiwuSolver(auto)，当前解析为内置经典 SA 后端）。
            builder: QUBO 构建器（默认根据 encoding_type 创建）。
            encoding_type: 编码方案（"precision_split_38", "unified_48", "unified_38"）。
            gamma_discount: C-主元 γ 折扣因子（None 使用 MIEDEMA 默认值 0.3）。
            penalty_strategy: 惩罚策略（"adaptive", "fixed", "linear", "grid_search"）。
            penalty_fixed_lambda: fixed 策略使用的固定 λ 值。
            exploration_strategy: 探索策略（"pareto_zoom", "uniform_grid", "random"）。
            uniform_grid_n: uniform_grid 策略的权重网格大小。
            archive_feasible_tol: 入档可行性容差 |Σc−100| ≤ tol（百分数点，默认 1.0 即 1%）。
                任务 C 容差统一（2026-07-19，诊断报告
                reports/feasible_hv_diagnostic_2026-07-19 §2.4 根因 3 /
                §5 决策 2）：原默认 2.0%（方案 §4.2.3 阶段D 步骤35
                "constraints_satisfied(c, tol=2%)"）与物理过滤器
                SUM_TOLERANCE=1.0%（physical_params.py）不一致，
                会在 1–2% 夹层产生"入档合法但过滤非法"的灰区解，
                故以更严格的过滤容差为准统一为 1.0%；
                与解码器 Composition.is_feasible 默认 tolerance=1.0
                亦一致。需复现旧口径时可显式传 2.0。
            seed: 随机种子（默认 None 保持旧行为：求解器每次吃 OS 熵源）。
                仅在未注入 solver 时生效——透传到内部默认构造的
                KaiwuSolver(mode="auto", seed=seed)，使同一 (seed, 模型,
                num_reads) 的 SA 采样逐位一致（B-1 修复）。权重微扰 /
                Dirichlet 采样用全局 np.random，由实验层
                np.random.seed(seed+rep) 覆盖，此处不重复接线。
        """
        self._params = PARETO_ZOOM
        self._initial_weights = initial_weights or list(COARSE_WEIGHTS)
        self._solver = solver or KaiwuSolver(mode="auto", seed=seed)
        self._builder = builder or QUBOBuilder(
            encoding_type=encoding_type,
            gamma_discount=gamma_discount,
        )
        self._encoding_type = encoding_type
        self._gamma_discount = gamma_discount
        self._penalty_strategy = penalty_strategy
        self._penalty_fixed_lambda = penalty_fixed_lambda
        self._exploration_strategy = exploration_strategy
        self._uniform_grid_n = uniform_grid_n
        self._archive_feasible_tol = archive_feasible_tol

        # 内部组件
        self._archive = Archive()
        self._weight_gen = WeightGenerator(
            sigma=self._params.sigma_perturb,
            weight_min=self._params.weight_min_bound,
        )
        self._hv_calc = HypervolumeCalculator()
        self._lambda_cache = LambdaCache()
        self._decoder = PrecisionSplitDecoder()
        self._filter = PhysicalFilter()

        # 目标函数
        self._dh_calc = MixingEnthalpy()
        self._density_calc = VegardDensity()
        self._cost_calc = WeightedCost()
        self._normalizer = PhysicalPriorNormalizer()

        # 状态
        self._rounds: List[ParetoZoomRound] = []
        self._explored_weights: set = set()
        self._failed_weights: List[Tuple] = []
        # P0-9/F-04：粗网格后 CONVEXITY_TEST_3D 报告（方案 §4.2.2）
        self._convexity_report: Dict = {}
        # 第 3 批（Feasible Rate 真实统计）：逐解可行性记录
        # (feasible_1pct, archive_feasible)，语义见 decode_stats()。
        self._feas_records: List[Tuple[bool, bool]] = []
        # BASE-4 grid_search 策略：全局固定最优 λ（首次使用时试跑选定一次，
        # 之后所有权重一律复用；明细见 _grid_search_detail）。
        self._grid_search_lambda: Optional[float] = None
        self._grid_search_detail: Dict = {}

    # =========================================================================
    # 主入口
    # =========================================================================

    def run(self) -> Tuple[Archive, List[ParetoZoomRound]]:
        """运行完整探索算法（策略由 exploration_strategy 决定）。

        Returns:
            (archive, round_history)。
        """
        if self._exploration_strategy == "pareto_zoom":
            return self._run_pareto_zoom()
        elif self._exploration_strategy == "uniform_grid":
            return self._run_uniform_grid()
        elif self._exploration_strategy == "random":
            return self._run_random_search()
        else:
            raise ValueError(f"Unknown exploration_strategy: {self._exploration_strategy}")

    def _run_pareto_zoom(self) -> Tuple[Archive, List[ParetoZoomRound]]:
        """标准 ParetoZoom 5 阶段算法。"""
        # ===== Phase A: 粗网格初始化 =====
        self._phase_a_initialize()

        # ===== 主循环 =====
        for round_id in range(self._params.t_max_rounds):
            round_result = self._run_round(round_id)
            self._rounds.append(round_result)

            # Phase E: 收敛判定
            # P1 修复（评审 四.4）：判据取绝对值 |ΔHV|/HV < ε（方案 §4.2.3 阶段E 步骤41），
            # 防止 HV 大幅下降（负 delta）被误判为收敛而提前 BREAK。
            if abs(round_result.delta_hv) < self._params.epsilon_hv:
                # 检查是否连续 2 轮
                if len(self._rounds) >= 2:
                    prev_delta = self._rounds[-2].delta_hv
                    if abs(prev_delta) < self._params.epsilon_hv:
                        break

        return self._archive, self._rounds

    def _run_uniform_grid(self) -> Tuple[Archive, List[ParetoZoomRound]]:
        """均匀网格探索策略：系统地在权重单纯形上采样 N 个点。"""
        n = self._uniform_grid_n
        # 在 3-simplex 上生成均匀网格点
        weights_list = self._generate_simplex_grid(n)
        self._explored_weights.clear()

        for i, w in enumerate(weights_list):
            w_key = self._weight_key(w)
            if w_key in self._explored_weights:
                continue
            self._explored_weights.add(w_key)

            try:
                sols = self._solve_weight_with_penalty_flex(w)
                if sols:
                    self._archive.insert_batch(sols)
            except Exception:
                self._failed_weights.append(w)
                continue

            # 记录 HV
            obj_mat = self._archive.get_objective_matrix()
            if len(obj_mat) > 0:
                if self._hv_calc.reference_point is None:
                    self._hv_calc.set_reference_from_data(obj_mat)
                hv = self._hv_calc.compute(obj_mat)
                self._archive.record_hv(hv)

            self._rounds.append(ParetoZoomRound(
                round_id=i,
                weights_solved=[w],
                new_solutions=sols if 'sols' in dir() else [],
            ))

        return self._archive, self._rounds

    def _run_random_search(self) -> Tuple[Archive, List[ParetoZoomRound]]:
        """随机搜索探索策略：随机采样 n 个权重组。"""
        n = self._uniform_grid_n  # reuse grid_n as sample count
        self._explored_weights.clear()

        for i in range(n):
            # 在单纯形上随机采样：Dirichlet 分布
            w_raw = np.random.dirichlet([1.0, 1.0, 1.0])
            w = (float(w_raw[0]), float(w_raw[1]), float(w_raw[2]))

            w_key = self._weight_key(w)
            if w_key in self._explored_weights:
                continue
            self._explored_weights.add(w_key)

            try:
                sols = self._solve_weight_with_penalty_flex(w)
                if sols:
                    self._archive.insert_batch(sols)
            except Exception:
                self._failed_weights.append(w)
                continue

            obj_mat = self._archive.get_objective_matrix()
            if len(obj_mat) > 0:
                if self._hv_calc.reference_point is None:
                    self._hv_calc.set_reference_from_data(obj_mat)
                hv = self._hv_calc.compute(obj_mat)
                self._archive.record_hv(hv)

            self._rounds.append(ParetoZoomRound(
                round_id=i,
                weights_solved=[w],
                new_solutions=sols if 'sols' in dir() else [],
            ))

        return self._archive, self._rounds

    @staticmethod
    def _generate_simplex_grid(n: int) -> List[Tuple[float, float, float]]:
        """在 3-simplex 上生成 ~n 个均匀分布的权重组。

        使用等间距扫描 w1, w2 ∈ [0.02, 0.96] 且 w3 = 1 − w1 − w2 ≥ 0.02。
        """
        weights = []
        # 步长选择使总点数接近 n
        step = max(0.05, 0.94 / (np.sqrt(n) + 1))
        w1_vals = np.arange(0.02, 0.97, step)
        for w1 in w1_vals:
            for w2 in np.arange(0.02, 0.98 - w1, step):
                w3 = 1.0 - w1 - w2
                if w3 >= 0.02:
                    weights.append((float(w1), float(w2), float(w3)))
        return weights

    # =========================================================================
    # Phase A: 粗网格初始化
    # =========================================================================

    def _phase_a_initialize(self) -> None:
        """运行所有初始权重组，每组建一个 PenaltyFlex 内循环。"""
        for i, weights in enumerate(self._initial_weights):
            w_key = self._weight_key(weights)
            if w_key in self._explored_weights:
                continue
            self._explored_weights.add(w_key)

            # 运行 PenaltyFlex 内循环
            solutions = self._solve_weight_with_penalty_flex(weights)
            if solutions:
                self._archive.insert_batch(solutions)

            # 记录 HV
            obj_mat = self._archive.get_objective_matrix()
            if len(obj_mat) > 0:
                if self._hv_calc.reference_point is None:
                    self._hv_calc.set_reference_from_data(obj_mat)
                hv = self._hv_calc.compute(obj_mat)
                self._archive.record_hv(hv)

        # ===== 粗网格后凸性检验（P0-9/F-04，方案 §4.2.2 CONVEXITY_TEST_3D）=====
        self._convexity_report = self._convexity_test_3d()

    # =========================================================================
    # Phase A 收尾: CONVEXITY_TEST_3D（方案 §4.2.2）
    # =========================================================================

    def _convexity_test_3d(self) -> Dict:
        """三投影平面凸性检验（P0-9/F-04，方案 §4.2.2 CONVEXITY_TEST_3D）。

        投影平面与排序轴（方案指定，三平面排序轴均为平面第一轴）:
            (f1,f2) 按 f1 排序；(f1,f3) 按 f1 排序；(f2,f3) 按 f2 排序。
        每平面投影后重算 2D 非支配集（方案注：投影后 |P_proj| ≥ |P0|，
        原 3D 被支配解可能变为 2D 非支配）。排序后相邻三点按方案叉积公式
        cross<0 计非凸点，ratio = 非凸点数 / |P_proj|，取最坏平面比例；
        max_ratio > convexity_warning(默认 10%) 触发 WARNING（日志 + 结果字段），
        建议信息含 ε-约束法 / 切比雪夫标量化提示。
        点在物理先验归一化空间（objectives_norm）取值，与间隙检测口径一致。

        Returns:
            dict: {ratios: {平面: 非凸比例}, max_ratio, warning, message}
        """
        report: Dict = {"ratios": {}, "max_ratio": 0.0, "warning": False, "message": ""}
        front = self._archive.front
        if len(front) < 3:
            report["message"] = "前沿点数 < 3，跳过凸性检验"
            return report

        points = np.array([r.objectives_norm for r in front])

        # (平面两轴的原始目标索引, 平面名称)
        projections = [
            ((0, 1), "(f1,f2)"),
            ((0, 2), "(f1,f3)"),
            ((1, 2), "(f2,f3)"),
        ]
        for (a, b), name in projections:
            proj = points[:, [a, b]]
            proj_nd = self._filter_nondominated_2d(proj)
            n = len(proj_nd)
            count = 0
            if n >= 3:
                # 方案指定排序轴均为平面第一轴（(f1,f2)/(f1,f3) 按 f1，(f2,f3) 按 f2）
                order = np.argsort(proj_nd[:, 0], kind="stable")
                sp = proj_nd[order]
                for i in range(1, n - 1):
                    # 方案 §4.2.2 叉积公式（cross < 0 计非凸点）:
                    # cross = (y_i.b − y_{i-1}.b)(y_{i+1}.a − y_{i-1}.a)
                    #       − (y_i.a − y_{i-1}.a)(y_{i+1}.b − y_{i-1}.b)
                    cross = (
                        (sp[i, 1] - sp[i - 1, 1]) * (sp[i + 1, 0] - sp[i - 1, 0])
                        - (sp[i, 0] - sp[i - 1, 0]) * (sp[i + 1, 1] - sp[i - 1, 1])
                    )
                    if cross < 0:
                        count += 1
            ratio = count / n if n > 0 else 0.0  # 方案：nonconvex_ratio = count / |P_proj|
            report["ratios"][name] = ratio
            report["max_ratio"] = max(report["max_ratio"], ratio)

        if report["max_ratio"] > self._params.convexity_warning:
            report["warning"] = True
            report["message"] = (
                f"Pareto前沿非凸比例{report['max_ratio']:.1%}，"
                "建议切换为ε-约束法或切比雪夫标量化"
            )
            logger.warning("CONVEXITY_TEST_3D: %s", report["message"])
        return report

    @staticmethod
    def _filter_nondominated_2d(points: np.ndarray) -> np.ndarray:
        """2D 投影平面的非支配过滤（双目标均最小化）。"""
        n = len(points)
        if n <= 1:
            return points
        keep = np.ones(n, dtype=bool)
        for i in range(n):
            if not keep[i]:
                continue
            for j in range(n):
                if i == j or not keep[j]:
                    continue
                if (
                    points[j, 0] <= points[i, 0]
                    and points[j, 1] <= points[i, 1]
                    and (points[j, 0] < points[i, 0] or points[j, 1] < points[i, 1])
                ):
                    keep[i] = False
                    break
        return points[keep]

    # =========================================================================
    # Phase B-D: 单轮处理
    # =========================================================================

    def _run_round(self, round_id: int) -> ParetoZoomRound:
        """执行一轮完整的 B→C→D→E 流程。"""
        obj_before = self._archive.get_objective_matrix()
        hv_before = self._hv_calc.compute(obj_before) if len(obj_before) > 0 else 0.0

        # ===== Phase B: 间隙检测 =====
        gap_weights = self._detect_gaps()

        # ===== Phase C: HV 热点微扰 =====
        perturb_weights = self._generate_perturbations()

        # ===== 合并 + 去重 + 截断（性能优化）=====
        # 维持间隙权重优先语义：前半为 gap_weights（距离排序），
        # 后半为 perturb_weights；去重后取前 max_new_weights_per_round 个。
        new_weights = gap_weights + perturb_weights
        new_weights = deduplicate_weights(new_weights, tolerance=self._params.weight_dedup_tol)

        # 审计修复（Penalty_Auditor P-D2）：补齐方案 §4.2.3 阶段C 步骤25
        # "REMOVE w from G_new if ANY(wᵢ < 0.02)" 边界保护。
        # 间隙中点插值可产生 0 分量权重（如 G1(1,0,0)×G4(0.5,0.5,0) 的
        # 中点 (0.75,0.25,0)）；微扰路径 clip 后再归一化亦不严格保证 ≥0.02。
        w_min = self._params.weight_min_bound
        new_weights = [w for w in new_weights if min(w) >= w_min]

        max_per_round = self._params.max_new_weights_per_round
        if max_per_round > 0 and len(new_weights) > max_per_round:
            new_weights = new_weights[:max_per_round]

        # 过滤已探索的权重
        new_weights = [
            w for w in new_weights
            if self._weight_key(w) not in self._explored_weights
        ]

        # ===== Phase D: 求解 =====
        all_new_solutions = []
        for w in new_weights:
            self._explored_weights.add(self._weight_key(w))
            try:
                sols = self._solve_weight_with_penalty_flex(w)
                if sols:
                    all_new_solutions.extend(sols)
                    self._archive.insert_batch(sols)
            except Exception:
                self._failed_weights.append(w)
                continue

        # ===== Phase E: HV 收敛判定 =====
        obj_after = self._archive.get_objective_matrix()
        if len(obj_after) > 0:
            # 更新参考点
            self._hv_calc.set_reference_from_data(obj_after)
            # P1 修复（评审 四.4）：hv_before 必须在重设后的同一参考点下重算，
            # 否则 hv_before/hv_after 参考点不同，ΔHV 被参考点漂移污染。
            hv_before = (
                self._hv_calc.compute(obj_before) if len(obj_before) > 0 else 0.0
            )
            hv_after = self._hv_calc.compute(obj_after)
            self._archive.record_hv(hv_after)
            delta_hv = self._hv_calc.compute_delta(hv_before, hv_after)
        else:
            hv_after = hv_before
            delta_hv = 0.0

        return ParetoZoomRound(
            round_id=round_id,
            weights_solved=new_weights,
            new_solutions=all_new_solutions,
            hv_before=hv_before,
            hv_after=hv_after,
            delta_hv=delta_hv,
            gaps_detected=len(gap_weights),
            perturbations_generated=len(perturb_weights),
        )

    # =========================================================================
    # PenaltyFlex 内循环
    # =========================================================================

    def _solve_weight_with_penalty_flex(
        self,
        weights: Tuple[float, float, float],
        warm_start_lambdas: Optional[Tuple[float, float]] = None,
    ) -> List[ValidatedRecord]:
        """对单个权重组运行惩罚内循环（策略由 penalty_strategy 决定）。

        Args:
            weights: (w1, w2, w3) 偏好权重。
            warm_start_lambdas: 可选的 warm-start λ 值（仅 adaptive 策略使用）。

        Returns:
            所有迭代中产生的有效解列表。
        """
        if self._penalty_strategy == "adaptive":
            return self._solve_adaptive(weights, warm_start_lambdas)
        elif self._penalty_strategy == "fixed":
            return self._solve_fixed_lambda(weights)
        elif self._penalty_strategy == "linear":
            return self._solve_linear_schedule(weights)
        elif self._penalty_strategy == "grid_search":
            return self._solve_grid_search(weights)
        else:
            raise ValueError(f"Unknown penalty_strategy: {self._penalty_strategy}")

    def _solve_adaptive(
        self,
        weights: Tuple[float, float, float],
        warm_start_lambdas: Optional[Tuple[float, float]] = None,
    ) -> List[ValidatedRecord]:
        """自适应 PenaltyFlex 策略（原实现）。"""
        # warm-start λ
        if warm_start_lambdas is not None:
            lc_init, lccr_init = warm_start_lambdas
        else:
            nearest = self._lambda_cache.find_nearest(weights)
            if nearest is not None:
                lc_init, lccr_init = nearest
            else:
                lc_init = CONSTRAINT.lambda_carbide_init
                lccr_init = CONSTRAINT.lambda_ccr_init

        pf = PenaltyFlex(
            lambda_carbide_init=lc_init,
            lambda_ccr_init=lccr_init,
        )

        all_records: List[ValidatedRecord] = []
        converged_lambdas: Optional[Tuple[float, float]] = None

        for t in range(CONSTRAINT.t_max):
            lc, lccr = pf.get_current_lambdas()

            records, sol_data = self._solve_single_iteration(weights, lc, lccr, len(self._rounds))
            all_records.extend(records)

            if not sol_data:
                continue

            # 方案 E（hea_encoding_scheme_v1.13.md :310,333）：反馈基于"本轮"TOP-10 解。
            # analyze_top_k 内部按 objective 升序排序后取前 k 个（见 adaptive_penalty.py），
            # 因此直接传入本轮 sol_data 即可，不再取累积列表尾部 solutions_data[-10:]。
            feedback = PenaltyFlex.analyze_top_k(sol_data, k=10)
            state = pf.step(feedback)

            if state.is_converged:
                converged_lambdas = (state.lambda_carbide, state.lambda_ccr)
                break

        # 缓存收敛的 λ
        if converged_lambdas is not None:
            self._lambda_cache.store(weights, converged_lambdas[0], converged_lambdas[1])
        else:
            lc, lccr = pf.get_current_lambdas()
            self._lambda_cache.store(weights, lc, lccr)

        return all_records

    def _solve_fixed_lambda(
        self, weights: Tuple[float, float, float]
    ) -> List[ValidatedRecord]:
        """固定 λ 惩罚策略。"""
        lam_val = self._penalty_fixed_lambda or CONSTRAINT.lambda_carbide_init
        records, _ = self._solve_single_iteration(weights, lam_val, lam_val, len(self._rounds))
        return records

    def _solve_linear_schedule(
        self, weights: Tuple[float, float, float]
    ) -> List[ValidatedRecord]:
        """线性调度 λ 策略：从 initial 线性增长到 10× initial。"""
        l_init = CONSTRAINT.lambda_carbide_init
        l_max = l_init * 10.0
        all_records: List[ValidatedRecord] = []

        for t in range(CONSTRAINT.t_max):
            progress = t / max(CONSTRAINT.t_max - 1, 1)
            lam = l_init + (l_max - l_init) * progress
            records, _ = self._solve_single_iteration(weights, lam, lam, len(self._rounds))
            all_records.extend(records)

        return all_records

    def _solve_grid_search(
        self, weights: Tuple[float, float, float]
    ) -> List[ValidatedRecord]:
        """网格搜索 λ 策略（方案 BASE-4，Validation Scheme v1.1 §4.2 :160）。

        审计修正（原 :593,597-611 三处偏离 BASE-4）：
          1. 网格改为方案 7 档 {0.1, 0.5, 1, 5, 10, 50, 100}；
          2. 全部 λ 候选的试跑解集合并后定统一参考点，在同一参考点下
             比 HV（P0-5 同类问题：原实现逐 λ 各自 set_reference_from_data，
             参考点漂移导致跨 λ HV 不可比）；
          3. 全局固定最优 λ——首次调用时在初始权重组上试跑选定一次并缓存，
             之后所有权重一律用该固定 λ，不再逐权重各自选 λ。
        """
        if self._grid_search_lambda is None:
            self._select_grid_search_lambda()
        lam_val = self._grid_search_lambda
        records, _ = self._solve_single_iteration(weights, lam_val, lam_val, len(self._rounds))
        return records

    def _select_grid_search_lambda(self) -> None:
        """BASE-4 试跑：在初始权重组上穷举 7 档 λ，统一参考点下取 HV 最大者。

        选中 λ 缓存到 self._grid_search_lambda（全局固定）；试跑明细存入
        self._grid_search_detail 供实验追溯与外部探针校验。
        全部 λ 试跑均无有效解时回退 CONSTRAINT.lambda_carbide_init。
        """
        lambda_grid = list(GRID_SEARCH_LAMBDA_GRID)
        trial_weights = list(self._initial_weights)

        # 试跑目标矩阵（原始物理目标，与 ablation._grid_search_fixed_lambda 同口径）
        obj_per_lambda: Dict[float, np.ndarray] = {}
        for lam_val in lambda_grid:
            trial_records: List[ValidatedRecord] = []
            for w in trial_weights:
                records, _ = self._solve_single_iteration(
                    w, lam_val, lam_val, len(self._rounds)
                )
                trial_records.extend(records)
            if trial_records:
                obj_per_lambda[lam_val] = np.array(
                    [r.objectives for r in trial_records], dtype=float
                )

        if not obj_per_lambda:
            # 试跑全部失败 → 回退默认 λ（与 fixed 策略缺省口径一致）
            self._grid_search_lambda = CONSTRAINT.lambda_carbide_init
            self._grid_search_detail = {
                "scheme": "BASE-4 (AP3_QUBO_Validation_Scheme_v1.1 §4.2 :160)",
                "grid": lambda_grid,
                "fallback": True,
                "best_lambda": float(self._grid_search_lambda),
            }
            return

        # P0-5 口径：合并全部候选 λ 的试跑解集，统一定参考点后再比 HV。
        ref = set_unified_reference(
            {repr(lam): m for lam, m in obj_per_lambda.items()}, margin=0.10
        )
        calc = HypervolumeCalculator(reference_point=ref)
        hv_per_lambda = {
            lam: calc.compute(m) for lam, m in obj_per_lambda.items()
        }
        best_lambda = max(hv_per_lambda, key=hv_per_lambda.get)

        self._grid_search_lambda = float(best_lambda)
        self._grid_search_detail = {
            "scheme": "BASE-4 (AP3_QUBO_Validation_Scheme_v1.1 §4.2 :160)",
            "grid": lambda_grid,
            "reference_point": [float(x) for x in ref],
            "hv_per_lambda": {
                float(lam): float(hv) for lam, hv in hv_per_lambda.items()
            },
            "best_lambda": float(best_lambda),
            "trial_weights": len(trial_weights),
            # 试跑原始目标矩阵，供探针独立复核"同一参考点"口径
            "objectives_per_lambda": obj_per_lambda,
            "fallback": False,
        }

    def _solve_single_iteration(
        self,
        weights: Tuple[float, float, float],
        lambda_carbide: float,
        lambda_ccr: float,
        round_id: int,
    ) -> Tuple[List[ValidatedRecord], List[Dict]]:
        """执行单次 QUBO 构建+求解+解码+评估。

        Returns:
            (records, solutions_data) — records 为 SolutionRecord 列表，
            solutions_data 为反馈字典列表。
        """
        records: List[ValidatedRecord] = []
        solutions_data: List[Dict] = []

        try:
            model = self._builder.build_model(
                weights=weights,
                lambda_carbide=lambda_carbide,
                lambda_ccr=lambda_ccr,
            )
            result = self._solver.solve_from_model(model, n_vars=self._builder.num_variables)
        except (NotImplementedError, RuntimeError):
            return records, solutions_data

        if not result.solutions:
            return records, solutions_data

        for sol in result.solutions[:10]:
            try:
                comp = self._decoder.decode(sol.bits)
            except ValueError:
                # 第 3 批（Feasible Rate 真实统计）：解码失败计入分母，
                # 两个可行性标志均记 False（语义见 decode_stats()）。
                self._feas_records.append((False, False))
                continue

            dh = self._dh_calc.evaluate(comp.fractions)
            density = self._density_calc.evaluate(comp.fractions)
            cost = self._cost_calc.evaluate(comp.fractions)
            f_norm = self._normalizer.normalize(np.array([dh, density, cost]))

            # P0-8/F-09 入档可行性过滤（方案 §4.2.3 阶段D 步骤35：
            # "IF constraints_satisfied(c, tol=2%)" 才执行 nondominated_update）。
            # 判定 |Σc − 100| ≤ archive_feasible_tol（任务 C 起默认 1%，
            # 与物理过滤器 SUM_TOLERANCE=1.0% 统一，构造参数可配）；
            # 与解码器 Composition.is_feasible 的 tolerance=1.0 同口径。
            # 不满足的解仍记录进 solutions_data 供 PenaltyFlex 反馈用，但不入 Archive。
            archive_ok = abs(comp.total - 100.0) <= self._archive_feasible_tol

            # 第 3 批（Feasible Rate 真实统计）：持久化每个已解码解的
            # 可行性判定，供 comparison.compare_penalty 真实统计。
            self._feas_records.append((bool(comp.is_feasible), bool(archive_ok)))

            solutions_data.append({
                "c_carbon": comp.fractions.get(INTERSTITIAL_ELEMENT, 0.0),
                "c_cr": comp.fractions.get("Cr", 0.0),
                "objective": sol.energy,
                "is_feasible": comp.is_feasible,
                "archive_feasible": archive_ok,  # P0-8: 2% 入档准则判定结果
            })

            if not archive_ok:
                continue  # 反馈用但不入 Archive

            records.append(ValidatedRecord(
                fractions=comp.fractions.copy(),
                bits=sol.bits.copy(),
                objectives=(dh, density, cost),
                objectives_norm=(float(f_norm[0]), float(f_norm[1]), float(f_norm[2])),
                weights=weights,
                lambdas=(lambda_carbide, lambda_ccr),
                energy=sol.energy,
                round_id=round_id,
                is_quantum=True,
            ))

        return records, solutions_data

    # =========================================================================
    # Phase B: 间隙检测
    # =========================================================================

    def _detect_gaps(self) -> List[Tuple[float, float, float]]:
        """检测 Pareto 前沿中的间隙并生成填充权重。"""
        front = self._archive.front
        if len(front) < 2:
            return []

        points = np.array([r.objectives_norm for r in front])
        gaps = self._weight_gen.from_gaps(
            [r.weights for r in front],
            []  # 使用默认间隙检测
        )

        # 直接使用目标空间距离检测
        if len(points) >= 2:
            # 按 f1 排序
            order = np.argsort(points[:, 0])
            sorted_pts = points[order]
            sorted_weights = [front[i].weights for i in order]

            # P0/F-05 间隙阈值接线（方案 §4.2.3 输入行：
            # d_threshold = 0.15 × max_edge_length(P)，
            # 参数 physical_params.ParetoZoomParams.d_threshold_factor=0.15）。
            # max_edge = 当前非支配前沿在归一化目标空间的最大相邻边长；
            # 距离在物理先验归一化空间（objectives_norm）计算，与方案一致。
            adj_dists = [
                float(np.linalg.norm(sorted_pts[i + 1] - sorted_pts[i]))
                for i in range(len(sorted_pts) - 1)
            ]
            max_edge = max(adj_dists)
            d_threshold = self._params.d_threshold_factor * max_edge

            gap_pairs = []
            for i, dist in enumerate(adj_dists):
                if dist > d_threshold:
                    gap_pairs.append((i, i + 1))

            gaps = self._weight_gen.from_gaps(sorted_weights, gap_pairs)

        return gaps

    # =========================================================================
    # Phase C: HV 热点微扰
    # =========================================================================

    def _generate_perturbations(self) -> List[Tuple[float, float, float]]:
        """对 HV 高贡献区域生成微扰权重。"""
        front = self._archive.front
        if len(front) < 2:
            return []

        # 审计修复（Penalty_Auditor P-D3）：HV 贡献必须在原始目标空间计算——
        # self._hv_calc 的参考点由 get_objective_matrix()（原始尺度）设定
        # （_phase_a_initialize / _run_round Phase E），原先用归一化矩阵
        # 调 marginal_contribution 造成"归一化点 × 原始参考点"空间错配。
        points = self._archive.get_objective_matrix()
        if len(points) < 2:
            return []

        # 计算各点 HV 边际贡献
        try:
            contributions = [
                self._hv_calc.marginal_contribution(points, i)
                for i in range(len(points))
            ]
        except Exception:
            contributions = None

        front_weights = self._archive.get_weights_of_front()
        return self._weight_gen.perturb_hotspots(front_weights, contributions)

    # =========================================================================
    # 工具方法
    # =========================================================================

    @staticmethod
    def _weight_key(w: Tuple[float, float, float]) -> Tuple[float, float, float]:
        """生成权重的去重键（四舍五入到 4 位小数）。"""
        return tuple(round(x, 4) for x in w)

    # =========================================================================
    # 结果访问
    # =========================================================================

    def get_front_compositions(self) -> List[Dict[str, float]]:
        """返回前沿上的成分字典列表。"""
        return self._archive.get_fractions_of_front()

    def get_front_metrics(self) -> Dict:
        """汇总前沿指标。

        Returns:
            dict 包含: hv, front_size, obj_ranges, num_explored_weights。
        """
        obj_mat = self._archive.get_objective_matrix()
        return {
            "hv": self._archive.get_latest_hv(),
            "front_size": self._archive.front_size,
            "total_records": self._archive.size,
            "num_explored_weights": len(self._explored_weights),
            "num_failed_weights": len(self._failed_weights),
            "num_rounds": len(self._rounds),
            # P0-9/F-04：CONVEXITY_TEST_3D 报告（方案 §4.2.2 结果字段）
            "convexity": self._convexity_report,
            "obj_ranges": {
                "f1": (float(obj_mat[:, 0].min()), float(obj_mat[:, 0].max()))
                if len(obj_mat) > 0 else (0, 0),
                "f2": (float(obj_mat[:, 1].min()), float(obj_mat[:, 1].max()))
                if len(obj_mat) > 0 else (0, 0),
                "f3": (float(obj_mat[:, 2].min()), float(obj_mat[:, 2].max()))
                if len(obj_mat) > 0 else (0, 0),
            },
        }

    def decode_stats(self) -> Dict[str, int]:
        """第 3 批（Feasible Rate 真实统计）：全部解码解的可行性计数。

        在 _solve_single_iteration 中对每个 TOP-10 求解结果逐条记录：
          - 解码失败（ValueError）→ 计入 total_decoded，两个标志均 False；
          - 解码成功 → (feasible_1pct, archive_feasible)，其中
            feasible_1pct = Composition.is_feasible（|Σc−100| ≤ 1%，
            解码器兜底口径，encoding/precision_split.py:47-49；
            与 KaiwuSolver.FEASIBILITY_TOL_PCT=1.0 同一阈值），
            archive_feasible = |Σc−100| ≤ archive_feasible_tol
            （默认 2%，方案 §4.2.3 阶段D 步骤35 入档准则）。

        Returns:
            dict: {total_decoded, feasible_1pct, archive_feasible}。
            供 comparison 实验按方法×重复聚合 Feasible Rate。
        """
        return {
            "total_decoded": len(self._feas_records),
            "feasible_1pct": sum(1 for f1, _ in self._feas_records if f1),
            "archive_feasible": sum(1 for _, fa in self._feas_records if fa),
        }

    def validate_front(self) -> List[PhysicalFilterResult]:
        """对前沿上的每个解运行物理过滤器。

        Returns:
            PhysicalFilterResult 列表。
        """
        results = []
        front = self._archive.front
        for record in front:
            dh = record.objectives[0]
            result = self._filter.evaluate(record.fractions, dh)
            results.append(result)
        return results

    def filter_summary(self) -> Dict[str, float]:
        """前沿物理过滤器通过率汇总。"""
        results = self.validate_front()
        return self._filter.summary(results)

    @property
    def archive(self) -> Archive:
        return self._archive

    @property
    def round_history(self) -> List[ParetoZoomRound]:
        return self._rounds
