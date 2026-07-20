"""
对比实验（实验 1~3）。

- 实验 1: PrecisionSplit vs 统一编码(48) vs 统一编码(38)
- 实验 2: PenaltyFlex vs Grid-Search λ vs Linear schedule vs Fixed λ
- 实验 3: ParetoZoom vs 均匀网格 vs NSGA-II vs Random Search

Fairness_Reporter（审计 D-2/D-3 公平性披露，报告层）：
- 四口径预算披露：每组逐 rep 记录 #solves / #samples /
  #objective-evals / T_wall 四项 Budget 指标，随 stats.metrics
  流入 driver 聚合的 report.md / records.csv / results.json；
- 跨方法 Mann-Whitney 的 Bonferroni 校正 p 值与方差 F 检验：
  驱动层逐 rep 以 n_repetitions=1 调用，单调用内每组仅 1 个
  样本，假设检验须跨调用累计（_FAIRNESS_SAMPLES，进程内按实验
  键累计）；每次调用以累计样本重算并注入 Fairness 指标，
  末次 rep 即全样本口径。结构化明细存 stats.comparisons["fairness"]。
- NSGA-II 投影修复贴界率（0.25% 容差）在实验 3 中按组披露。
"""

from typing import Dict, List, Optional, Tuple

import random
import time

import numpy as np
from scipy import stats as _scipy_stats

from ..physical_params import COARSE_WEIGHTS, PARETO_ZOOM
from ..exploration.pareto_zoom import ParetoZoom, ParetoZoomRound
from ..validation.hypervolume import HypervolumeCalculator, set_unified_reference
from ..statistics.reporting import ExperimentStats
from ..statistics.hypothesis_tests import bonferroni_correction


# =========================================================================
# Fairness_Reporter：四口径预算披露（审计 D-2）
# =========================================================================

# Budget 指标名（report.md 新栏目；四口径）
M_SOLVES = "Budget #solves (QUBO)"
M_SAMPLES = "Budget #samples (reads)"
M_OBJEVALS = "Budget #objective-evals"
M_TWALL = "Budget T_wall (s)"

_ZERO_BUDGET: Dict[str, float] = {
    "n_solves": 0.0,
    "n_samples": 0.0,
    "n_objective_evals": 0.0,
    "t_wall_sec": 0.0,
}


class _SolveBudgetProbe:
    """QUBO 求解预算探针（Fairness_Reporter，审计 D-2）。

    包装 ParetoZoom 内部注入的 solver（不改 pareto_zoom / kaiwu_solver
    源码），逐次统计：
      - n_solves:  QUBO 求解调用次数（solve_from_model）；
      - n_samples: 采样总数——按 SolverResult.num_reads 实记，兼容
        驱动层 _DriverSolver 对 num_reads 的覆盖（run_experiments.py
        :144-148，调用方不可见的默认值以 result.num_reads 为准）。
    其余属性经 __getattr__ 透传，不改变求解行为。
    """

    def __init__(self, inner):
        self._inner = inner
        self.n_solves = 0
        self.n_samples = 0

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def solve_from_model(self, model, n_vars: int = 38, **kwargs):
        result = self._inner.solve_from_model(model, n_vars=n_vars, **kwargs)
        self.n_solves += 1
        self.n_samples += int(getattr(result, "num_reads", 0) or 0)
        return result


def _run_pz_with_budget(pz) -> Tuple[object, object, Dict[str, float]]:
    """运行 ParetoZoom 并测量四口径预算（#solves/#samples/#objective-evals/T_wall）。

    objective-evals 口径：decode_stats()["total_decoded"] —— 求解结果
    TOP-10 逐条解码并做 3 目标评估的候选处理数（含解码失败条目，
    上界 ≤ 10×#solves，pareto_zoom.py:655-678）。与 NSGA-II 的
    适应度评估计数并置，回答审计 D-2 "评估预算未对齐" 的披露要求。
    """
    probe = _SolveBudgetProbe(pz._solver)
    pz._solver = probe
    t0 = time.perf_counter()
    archive, rounds = pz.run()
    t_wall = time.perf_counter() - t0
    budget = {
        "n_solves": float(probe.n_solves),
        "n_samples": float(probe.n_samples),
        "n_objective_evals": float(pz.decode_stats().get("total_decoded", 0)),
        "t_wall_sec": t_wall,
    }
    return archive, rounds, budget


def _add_budget_metrics(stats: ExperimentStats, label: str,
                        budget: Dict[str, float]) -> None:
    """把一组四口径预算值登记为 Budget 指标（随 driver 聚合入报告）。"""
    stats.add_metric(M_SOLVES, label, [budget["n_solves"]])
    stats.add_metric(M_SAMPLES, label, [budget["n_samples"]])
    stats.add_metric(M_OBJEVALS, label, [budget["n_objective_evals"]])
    stats.add_metric(M_TWALL, label, [budget["t_wall_sec"]])


# =========================================================================
# Fairness_Reporter：跨方法假设检验披露层（审计 D-2，方案 §5.2）
# =========================================================================

# 跨调用累计样本登记：run_experiments 驱动层逐 rep 以 n_repetitions=1
# 调用 compare_*（run_experiments.py:488），单调用内每组仅 1 个样本，
# 跨方法假设检验须跨调用累计。键为实验标识，值为 {指标: {组: [样本]}}。
_FAIRNESS_SAMPLES: Dict[str, Dict[str, Dict[str, List[float]]]] = {}


def _reset_fairness_samples(exp_key: str | None = None) -> None:
    """清空累计样本（测试/探针用；实验驱动单进程单实验不受影响）。"""
    if exp_key is None:
        _FAIRNESS_SAMPLES.clear()
    else:
        _FAIRNESS_SAMPLES.pop(exp_key, None)


def _f_test_variance(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    """两样本方差 F 检验（two-sided；方案 §4.2 σ 超参数敏感度口径）。

    F = s_a²/s_b²（ddof=1），df = (n_a−1, n_b−1)，
    p = 2·min(P(F≤f), P(F≥f))。双零方差（两组均恒定）记 (1.0, 1.0)；
    单方零方差时 F 塌缩到 0 或 inf，p → 0（方差差异必然显著）。
    要求 n_a ≥ 2 且 n_b ≥ 2（调用方保证）。
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    v1 = float(a.var(ddof=1))
    v2 = float(b.var(ddof=1))
    if v1 == 0.0 and v2 == 0.0:
        return 1.0, 1.0
    f_stat = v1 / v2 if v2 > 0.0 else float("inf")
    df1, df2 = len(a) - 1, len(b) - 1
    p = 2.0 * min(
        float(_scipy_stats.f.cdf(f_stat, df1, df2)),
        float(_scipy_stats.f.sf(f_stat, df1, df2)),
    )
    return f_stat, float(min(p, 1.0))


def _append_fairness_layer(
    stats: ExperimentStats,
    exp_key: str,
    baseline: str,
    quality_metrics: List[str],
) -> None:
    """跨方法 Mann-Whitney（Bonferroni 校正）+ 方差 F 检验披露层。

    样本口径：本进程内该实验全部已完成 rep 的累计样本（驱动层
    n_repetitions=1 逐 rep 调用模式下，末次调用即全样本；直接以
    n_repetitions≥2 单次调用时即本调用全样本）。任一组样本 <2 时
    该比较跳过（故首个 rep 无输出）。

    Bonferroni 族大小 n = 实际执行的 MW 检验数（质量指标 × 非基线
    组）。方案 §5.2 的名义族（实验 2: 4 指标×5 对照=20；实验 3:
    4×3=12）含 C_metric/Spread 等本实现未覆盖的指标，此处按已实现
    指标口径校正，并在指标名与 stats.comparisons["fairness"] 中
    披露实际 n。

    注入两类指标（流入 driver 聚合的 report.md）：
      "Fairness[{m}] MW p Bonferroni×{n}"  —— 校正后 p 值（末 rep=全样本）
      "Fairness[{m}] F-test p (variance)"  —— 方差 F 检验 p 值
    披露层为只增不改的附加统计：任何异常静默跳过，不阻断实验主流程。
    """
    try:
        acc = _FAIRNESS_SAMPLES.setdefault(exp_key, {})
        for m in quality_metrics:
            for g, vals in stats.metrics.get(m, {}).items():
                acc.setdefault(m, {}).setdefault(g, []).extend(
                    float(v) for v in vals)

        details: List[Dict[str, object]] = []
        for m in quality_metrics:
            groups = acc.get(m, {})
            base_vals = np.asarray(
                [v for v in groups.get(baseline, []) if not np.isnan(v)],
                dtype=float)
            if len(base_vals) < 2:
                continue
            for g in stats.group_names:
                if g == baseline or g not in groups:
                    continue
                vals = np.asarray(
                    [v for v in groups[g] if not np.isnan(v)], dtype=float)
                if len(vals) < 2:
                    continue
                try:
                    p_raw = float(_scipy_stats.mannwhitneyu(
                        vals, base_vals, alternative="two-sided").pvalue)
                except ValueError:
                    p_raw = float("nan")
                f_stat, p_f = _f_test_variance(vals, base_vals)
                details.append({
                    "metric": m, "group": g,
                    "n_group": int(len(vals)), "n_baseline": int(len(base_vals)),
                    "p_mw_raw": p_raw, "F": f_stat, "p_f": p_f,
                })

        if not details:
            return

        n_family = len(details)
        p_list = [d["p_mw_raw"] if not np.isnan(d["p_mw_raw"]) else 1.0
                  for d in details]
        p_adj = bonferroni_correction(p_list, n_family)
        for d, pa in zip(details, p_adj):
            d["p_mw_bonferroni"] = pa
            d["significant_bonferroni"] = bool(pa < 0.05)
            stats.add_metric(
                f"Fairness[{d['metric']}] MW p Bonferroni×{n_family}",
                str(d["group"]), [pa])
            stats.add_metric(
                f"Fairness[{d['metric']}] F-test p (variance)",
                str(d["group"]), [d["p_f"]])

        stats.comparisons["fairness"] = {
            "baseline": baseline,
            "n_comparisons": n_family,
            "sample_scope": ("进程内跨 rep 累计样本；末次 rep 的指标值 "
                             "= 全样本统计（records.csv 逐 rep 可追溯）"),
            "tests": details,
        }
    except Exception:
        # 披露层失败不阻断实验主流程（统计附加层，非实验本体）
        return


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
        # Fairness_Reporter：逐组预算（失败组记零值，与既有 HV=0 口径一致）
        budgets: Dict[str, Dict[str, float]] = {}

        for label, enc_type in encoding_configs:
            try:
                # B-1：seed+rep 贯通到求解器 SA 采样（ParetoZoom 默认构造
                # KaiwuSolver(seed=...)），同 seed 结果逐位一致
                pz = ParetoZoom(encoding_type=enc_type, seed=seed + rep)
                archive, _, budgets[label] = _run_pz_with_budget(pz)
                obj_mats[label] = archive.get_objective_matrix()
                front_sizes[label] = float(archive.front_size)
            except (NotImplementedError, RuntimeError):
                obj_mats[label] = np.zeros((0, 3))
                front_sizes[label] = 0.0
                budgets[label] = dict(_ZERO_BUDGET)

        hv_vals = _compute_hv_unified(obj_mats)
        for label, _ in encoding_configs:
            stats.add_metric("HV", label, [hv_vals[label]])
            stats.add_metric("Front Size", label, [front_sizes[label]])
            _add_budget_metrics(stats, label, budgets[label])

    # Fairness_Reporter：跨方法 MW(Bonferroni) + F 检验（累计样本口径）
    _append_fairness_layer(
        stats, "exp1_encoding", "PrecisionSplit(38)", ["HV", "Front Size"])

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
        # Fairness_Reporter：逐组预算（四口径；Grid-Search 的多 λ 穷举
        # 预算经 #solves/#samples 自然显形，正是审计 D-2 关注点）
        budgets: Dict[str, Dict[str, float]] = {}

        for label, strategy, fixed_lam in strategy_configs:
            try:
                # B-1：seed+rep 贯通到求解器（同 compare_encoding）
                pz = ParetoZoom(
                    penalty_strategy=strategy,
                    penalty_fixed_lambda=fixed_lam,
                    seed=seed + rep,
                )
                archive, _, budgets[label] = _run_pz_with_budget(pz)
                obj_mats[label] = archive.get_objective_matrix()
                # 第 3 批修复（Feasible Rate 真实统计）：原 :122 硬编码
                # 1.0 为占位数据。现取 ParetoZoom.decode_stats() 持久化
                # 的全部解码解可行性判定（pareto_zoom.py，逐解记录）。
                feas_stats[label] = pz.decode_stats()
            except (NotImplementedError, RuntimeError):
                obj_mats[label] = np.zeros((0, 3))
                feas_stats[label] = {}
                budgets[label] = dict(_ZERO_BUDGET)

        hv_vals = _compute_hv_unified(obj_mats)
        for label, _, _ in strategy_configs:
            stats.add_metric("HV", label, [hv_vals[label]])
            stats.add_metric(
                "Feasible Rate", label,
                [_feasible_rate_1pct(feas_stats[label])],
            )
            _add_budget_metrics(stats, label, budgets[label])

    # Fairness_Reporter：跨方法 MW(Bonferroni) + 方差 F 检验
    # （方案 §4.2 σ 口径 + §5.2 多重比较校正；累计样本口径）
    _append_fairness_layer(
        stats, "exp2_penalty", "PenaltyFlex", ["HV", "Feasible Rate"])

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
        # Fairness_Reporter：逐方法预算（审计 D-2）与 NSGA-II 投影
        # 修复贴界统计（审计 D-3）
        budgets: Dict[str, Dict[str, float]] = {}
        nsga_boundary: Dict[str, object] | None = None

        for method in ["ParetoZoom", "Uniform Grid", "NSGA-II", "Random"]:
            try:
                if method == "NSGA-II":
                    from .nsga2_baseline import NSGA2Optimizer
                    optimizer = NSGA2Optimizer()
                    # T_wall 口径与 ParetoZoom 系一致：优化 + 前沿目标提取
                    t0 = time.perf_counter()
                    front = optimizer.optimize()
                    obj_mats[method] = optimizer.evaluate_front(front)
                    t_wall = time.perf_counter() - t0
                    b = optimizer.budget_stats()
                    budgets[method] = {
                        "n_solves": float(b["n_solves"]),
                        "n_samples": float(b["n_samples"]),
                        "n_objective_evals": float(b["n_objective_evals"]),
                        "t_wall_sec": t_wall,
                    }
                    nsga_boundary = optimizer.repair_boundary_stats()
                elif method == "Uniform Grid":
                    # B-1：seed+rep 贯通到求解器（同 compare_encoding）
                    pz = ParetoZoom(exploration_strategy="uniform_grid", uniform_grid_n=50,
                                    seed=seed + rep)
                    archive, _, budgets[method] = _run_pz_with_budget(pz)
                    obj_mats[method] = archive.get_objective_matrix()
                elif method == "Random":
                    pz = ParetoZoom(exploration_strategy="random", uniform_grid_n=50,
                                    seed=seed + rep)
                    archive, _, budgets[method] = _run_pz_with_budget(pz)
                    obj_mats[method] = archive.get_objective_matrix()
                else:  # ParetoZoom (default)
                    pz = ParetoZoom(exploration_strategy="pareto_zoom",
                                    seed=seed + rep)
                    archive, _, budgets[method] = _run_pz_with_budget(pz)
                    obj_mats[method] = archive.get_objective_matrix()
            except (NotImplementedError, RuntimeError):
                obj_mats[method] = np.zeros((0, 3))
                budgets[method] = dict(_ZERO_BUDGET)

        hv_vals = _compute_hv_unified(obj_mats)
        for method in ["ParetoZoom", "Uniform Grid", "NSGA-II", "Random"]:
            stats.add_metric("HV", method, [hv_vals[method]])
            stats.add_metric("Front Size", method, [float(len(obj_mats[method]))])
            _add_budget_metrics(stats, method, budgets[method])

        # Fairness_Reporter（审计 D-3）：NSGA-II 投影修复贴界率披露
        # （0.25% 容差；修复前/后总体 + 主元/C 分组，仅 NSGA-II 组）
        if nsga_boundary is not None:
            stats.add_metric("NSGA-II 贴界率 修复前（总体）", "NSGA-II",
                             [float(nsga_boundary["rate_before"])])
            stats.add_metric("NSGA-II 贴界率 修复后（总体）", "NSGA-II",
                             [float(nsga_boundary["rate_after"])])
            stats.add_metric("NSGA-II 贴界率 修复后（主元）", "NSGA-II",
                             [float(nsga_boundary["rate_after_main"])])
            stats.add_metric("NSGA-II 贴界率 修复后（C）", "NSGA-II",
                             [float(nsga_boundary["rate_after_carbon"])])

    # Fairness_Reporter：跨方法 MW(Bonferroni) + 方差 F 检验（累计样本口径）
    _append_fairness_layer(
        stats, "exp3_exploration", "ParetoZoom", ["HV", "Front Size"])

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
