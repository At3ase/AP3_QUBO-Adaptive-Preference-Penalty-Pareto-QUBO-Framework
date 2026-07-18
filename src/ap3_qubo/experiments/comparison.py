"""
对比实验（实验 1~3）。

- 实验 1: PrecisionSplit vs 统一编码(48) vs 统一编码(38)
- 实验 2: PenaltyFlex vs Grid-Search λ vs Linear schedule vs Fixed λ
- 实验 3: ParetoZoom vs 均匀网格 vs NSGA-II vs Random Search
"""

from typing import Dict, List, Optional, Tuple

import random

import numpy as np

from ..physical_params import COARSE_WEIGHTS, PARETO_ZOOM
from ..exploration.pareto_zoom import ParetoZoom, ParetoZoomRound
from ..validation.hypervolume import HypervolumeCalculator, set_unified_reference
from ..statistics.reporting import ExperimentStats


def compare_encoding(
    # 方案 Validation Scheme 实验 1 要求重复 30 次
    # （审查报告 Code_Completion_Review_2026-07-18 P1-6：原默认 20 次不足）
    n_repetitions: int = 30,
    seed: int = 42,
) -> ExperimentStats:
    """实验 1: PrecisionSplit 编码效率对比。

    三组对比:
      - PrecisionSplit(38): 38 变量（AP³ 方案）— 主元 7 bits + C 3 bits
      - Unified(48): 6 元素 × 8 比特 = 48 变量
      - Unified(38): 6 元素共享 38 变量（主元 6 bits + C 8 bits）

    Args:
        n_repetitions: 重复次数。
        seed: 随机种子。

    Returns:
        ExperimentStats。
    """
    stats = ExperimentStats(name="实验 1: PrecisionSplit 编码效率")

    encoding_configs = [
        ("PrecisionSplit(38)", "precision_split_38"),
        ("Unified(48)", "unified_48"),
        ("Unified(38)", "unified_38"),
    ]

    for rep in range(n_repetitions):
        np.random.seed(seed + rep)

        # P0-5 修复：先收集本次重复全部编码方案的 archive 目标矩阵，
        # 再统一设定参考点计算 HV（见文件尾部 _compute_hv_unified）。
        obj_mats: Dict[str, np.ndarray] = {}
        front_sizes: Dict[str, float] = {}

        for label, enc_type in encoding_configs:
            try:
                # B-1：seed+rep 贯通到求解器 SA 采样（ParetoZoom 默认构造
                # KaiwuSolver(seed=...)），同 seed 结果逐位一致
                pz = ParetoZoom(encoding_type=enc_type, seed=seed + rep)
                archive, _ = pz.run()
                obj_mats[label] = archive.get_objective_matrix()
                front_sizes[label] = float(archive.front_size)
            except (NotImplementedError, RuntimeError):
                obj_mats[label] = np.zeros((0, 3))
                front_sizes[label] = 0.0

        hv_vals = _compute_hv_unified(obj_mats)
        for label, _ in encoding_configs:
            stats.add_metric("HV", label, [hv_vals[label]])
            stats.add_metric("Front Size", label, [front_sizes[label]])

    return stats


def compare_penalty(
    n_repetitions: int = 20,
    seed: int = 42,
) -> ExperimentStats:
    """实验 2: PenaltyFlex 自适应惩罚验证。

    五组对比:
      - PenaltyFlex:     自适应 λ（α·tanh 更新）
      - Grid-Search:     网格搜索最优固定 λ ∈ {0.01, 0.05, 0.1, 0.5, 1.0, 5.0}
      - Linear:          线性调度 λ(t) = λ_init → 10×λ_init
      - Fixed(λ=1/10/100): 方案 Baseline-1/2/3 固定 λ 基线

    Args:
        n_repetitions: 重复次数。
        seed: 随机种子。

    Returns:
        ExperimentStats。
    """
    stats = ExperimentStats(name="实验 2: PenaltyFlex 自适应惩罚")

    strategy_configs = [
        ("PenaltyFlex", "adaptive", None),
        ("Grid-Search", "grid_search", None),
        ("Linear", "linear", None),
        # 方案 Validation Scheme 实验 2 固定 λ 档位 Baseline-1/2/3
        # （审查报告 Code_Completion_Review_2026-07-18 P1-6：
        #   原 {0.05, 1.0} 与方案不符，应为 {1, 10, 100}）
        ("Fixed(λ=1)", "fixed", 1.0),
        ("Fixed(λ=10)", "fixed", 10.0),
        ("Fixed(λ=100)", "fixed", 100.0),
    ]

    for rep in range(n_repetitions):
        np.random.seed(seed + rep)

        # P0-5 修复：先收集本次重复全部策略的 archive 目标矩阵，
        # 再统一设定参考点计算 HV。
        obj_mats: Dict[str, np.ndarray] = {}
        feas_stats: Dict[str, Dict[str, int]] = {}

        for label, strategy, fixed_lam in strategy_configs:
            try:
                # B-1：seed+rep 贯通到求解器（同 compare_encoding）
                pz = ParetoZoom(
                    penalty_strategy=strategy,
                    penalty_fixed_lambda=fixed_lam,
                    seed=seed + rep,
                )
                archive, _ = pz.run()
                obj_mats[label] = archive.get_objective_matrix()
                # 第 3 批修复（Feasible Rate 真实统计）：原 :122 硬编码
                # 1.0 为占位数据。现取 ParetoZoom.decode_stats() 持久化
                # 的全部解码解可行性判定（pareto_zoom.py，逐解记录）。
                feas_stats[label] = pz.decode_stats()
            except (NotImplementedError, RuntimeError):
                obj_mats[label] = np.zeros((0, 3))
                feas_stats[label] = {}

        hv_vals = _compute_hv_unified(obj_mats)
        for label, _, _ in strategy_configs:
            stats.add_metric("HV", label, [hv_vals[label]])
            stats.add_metric(
                "Feasible Rate", label,
                [_feasible_rate_1pct(feas_stats[label])],
            )

    return stats


def compare_exploration(
    n_repetitions: int = 20,
    seed: int = 42,
) -> ExperimentStats:
    """实验 3: ParetoZoom 前沿质量验证。

    四组对比:
      - ParetoZoom:   动态加密探索 — 5 阶段 (粗网格 → 间隙 → 热点 → 求解 → 收敛)
      - Uniform Grid:  均匀 50 组权重网格（无自适应加密）
      - NSGA-II:       经典多目标进化算法 (DEAP，连续成分空间)
      - Random:        随机 50 组权重采样

    Args:
        n_repetitions: 重复次数。
        seed: 随机种子。

    Returns:
        ExperimentStats。
    """
    stats = ExperimentStats(name="实验 3: ParetoZoom 前沿探索")

    for rep in range(n_repetitions):
        np.random.seed(seed + rep)
        # B-1：deap 的 SBX 交叉/多项式变异/selTournamentDCD 内部用 stdlib
        # random（非 np.random），必须同步播种，否则 NSGA-II 组不可复现
        random.seed(seed + rep)

        # P0-5 修复（审查报告 Code_Completion_Review_2026-07-18）：
        # 先收集本次重复全部四种方法的目标矩阵，再统一设定参考点
        # 计算 HV——跨方法 HV 才可比。NSGA-II 分支不再使用其内部
        # 自定参考点的 hv，改取原始目标矩阵走同一统一路径。
        obj_mats: Dict[str, np.ndarray] = {}

        for method in ["ParetoZoom", "Uniform Grid", "NSGA-II", "Random"]:
            try:
                if method == "NSGA-II":
                    from .nsga2_baseline import NSGA2Optimizer
                    optimizer = NSGA2Optimizer()
                    front = optimizer.optimize()
                    obj_mats[method] = optimizer.evaluate_front(front)
                elif method == "Uniform Grid":
                    # B-1：seed+rep 贯通到求解器（同 compare_encoding）
                    pz = ParetoZoom(exploration_strategy="uniform_grid", uniform_grid_n=50,
                                    seed=seed + rep)
                    archive, _ = pz.run()
                    obj_mats[method] = archive.get_objective_matrix()
                elif method == "Random":
                    pz = ParetoZoom(exploration_strategy="random", uniform_grid_n=50,
                                    seed=seed + rep)
                    archive, _ = pz.run()
                    obj_mats[method] = archive.get_objective_matrix()
                else:  # ParetoZoom (default)
                    pz = ParetoZoom(exploration_strategy="pareto_zoom",
                                    seed=seed + rep)
                    archive, _ = pz.run()
                    obj_mats[method] = archive.get_objective_matrix()
            except (NotImplementedError, RuntimeError):
                obj_mats[method] = np.zeros((0, 3))

        hv_vals = _compute_hv_unified(obj_mats)
        for method in ["ParetoZoom", "Uniform Grid", "NSGA-II", "Random"]:
            stats.add_metric("HV", method, [hv_vals[method]])
            stats.add_metric("Front Size", method, [float(len(obj_mats[method]))])

    return stats


def _compute_hv_unified(obj_mats: Dict[str, np.ndarray]) -> Dict[str, float]:
    """P0-5 修复（审查报告 Code_Completion_Review_2026-07-18）：
    跨方法统一参考点 HV 计算。

    原 _compute_hv（:170-177）对每个方法自己的 archive 单独
    set_reference_from_data，违反 HV-1 固定参考点前提，组间 HV
    差异混入参考点漂移。现合并全部方法的解集后统一定参考点，
    再用同一参考点分别计算各方法 HV。空解集记 0.0；
    全部为空时各方法均记 0.0。
    """
    if not any(len(m) > 0 for m in obj_mats.values()):
        return {label: 0.0 for label in obj_mats}
    ref = set_unified_reference(obj_mats, margin=0.10)
    hv_calc = HypervolumeCalculator(reference_point=ref)
    return {
        label: (hv_calc.compute(mat) if len(mat) > 0 else 0.0)
        for label, mat in obj_mats.items()
    }


def _feasible_rate_1pct(feas_stats: Dict[str, int]) -> float:
    """第 3 批修复（Feasible Rate 真实统计）：可行解占全部解码解比例。

    口径（任务书第 3 批 A 项）：可行 = |Σc − 100| ≤ 1%，与解码器兜底
    口径一致（Composition.is_feasible 默认 tolerance=1.0，
    encoding/precision_split.py:47-49；求解器侧同阈值
    KaiwuSolver.FEASIBILITY_TOL_PCT）。分母为全部解码解（含解码失败
    与未过 2% 入档准则的解），数据源为 ParetoZoom.decode_stats()。
    按方法×重复聚合：每次重复产生一个比率值，由 ExperimentStats
    跨重复汇总 Mean ± SD。无统计数据（异常分支或无解码解）返回 0.0
    ——不再硬编码 1.0 占位。
    """
    total = feas_stats.get("total_decoded", 0)
    if total <= 0:
        return 0.0
    return feas_stats.get("feasible_1pct", 0) / total
