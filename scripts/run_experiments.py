#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""AP³-QUBO 实验驱动脚本（第 3 批）：跑实验 → 落盘 → 出图。

端到端入口，覆盖保底实验顺序（方案 Validation Scheme 执行序
0 → 2 → 3，见 src/ap3_qubo/experiments/__init__.py 注释）；实验 1 / 4
为补充实验，已接入 dispatch 但需单独指定（不并入 all，保底序不变）：

  - 实验 0: 消融实验（AblationRunner，量化三创新各自贡献）
  - 实验 1: PrecisionSplit 编码效率对比（compare_encoding，
            PrecisionSplit(38) vs Unified(48) vs Unified(38)）
  - 实验 2: PenaltyFlex 对比（compare_penalty，6 组惩罚策略）
  - 实验 3: ParetoZoom 对比（compare_exploration，4 种探索方法 + NSGA-II）
  - 实验 4: γ 敏感性分析（SensitivityAnalyzer，5 个 γ 值，基准 0.3）

用法（必须 Python310 环境，默认 python3.12 无 kaiwu）：

  C:/Users/At3ase/AppData/Local/Programs/Python/Python310/python.exe \\
      scripts/run_experiments.py --experiment all --reps 3

  # 冒烟/链路验证（3 权重 + 小采样，分钟级）：
  ... --experiment 2 --quick --reps 1
  ... --experiment 1 --quick --reps 1   # 补充实验 1
  ... --experiment 4 --quick --reps 1   # 补充实验 4

接口事实（2026-07-18 第 3 批核对源码确认，驱动层据此设计）：

  1. compare_penalty / compare_exploration / AblationRunner 均不暴露
     solver 注入口，内部直接构造 ParetoZoom()。ParetoZoom 本身支持
     solver= 注入（pareto_zoom.py:89），且默认 KaiwuSolver(mode="auto")
     按方案 D-04 门禁解析为 simulator 后端（kaiwu_solver.py:238-241）。
     本脚本在自身进程内对 ablation/comparison 模块命名空间中的
     ParetoZoom 打"工厂补丁"，显式注入 KaiwuSolver(mode="simulator")，
     不修改任何 src 源码文件；等价于默认 auto 行为，但更明确。
  2. ExperimentStats.add_metric 已修复为追加/累积语义（A-1 修复，
     statistics/reporting.py:31-53），compare_* 逐 rep 聚合无失真。
     本脚本仍在驱动层外循环 reps、每次以 n_repetitions=1 调用并自行
     聚合——保留该绕法是为让驱动层完全掌控逐 rep 原始记录
     （records.csv）与失败 rep 容错，且与 comparison.py 单次调用内
     逐 rep 统一参考点的语义一致，不改源码。H-04 正式实验
     （reps≥20/30）同样适用。
  3. quick 模式的缩放手法与 smoke_simulator_e2e.py 一致：
     3 组权重（顶点+中心）+ dataclasses.replace 压缩 t_max_rounds；
     另压缩 num_reads / sa_sweeps / PenaltyFlex t_max / uniform_grid_n /
     NSGA-II pop×gen。quick 仅用于验证链路，不用于结论数据。
  4. NSGA-II 在 compare_exploration 内为函数内局部 import
     （comparison.py:168），补丁需打在 nsga2_baseline 模块属性上。

产出（--out 目录，默认 data/results/<时间戳>/）：

  run_log.txt                全程日志（控制台 tee）
  summary.json               全部实验汇总 + 运行配置 + 耗时
  exp{0,1,2,3,4}/results.json    指标 + 统计（mean/std/CI95/min/max）
  exp{0,1,2,3,4}/records.csv     逐 rep 原始记录
  exp{0,1,2,3,4}/report.md       统计报告（0/1/2/3 用 statistics.reporting；
                                 4 用 SensitivityAnalyzer.report 的 γ 表格）
  exp0/fronts/{config}_rep{NN}.npz  逐 rep 逐配置 Pareto 前沿（任务 C：
                                 objectives + fractions + 硬口径可行掩码，
                                 供 feasible-HV 分析与物理核查）
  exp{0,1,2,3,4}/representative_front.npz         代表性 run 前沿/HV/λ 原始数据
  exp{0,1,2,3,4}/representative_compositions.csv  前沿成分表
  exp{0,1,2,3,4}/*.png           Pareto 2D/3D、HV 收敛、λ 轨迹、成分热力图、
                                 元素分布箱线图、组间 HV 箱线图
"""

from __future__ import annotations

import argparse
import dataclasses
import io
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

# ---------------------------------------------------------------------------
# 路径与后端初始化（必须在 import ap3_qubo 之前）
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent  # D:\QUBO
sys.path.insert(0, str(ROOT / "src"))

import matplotlib

matplotlib.use("Agg")  # 无显示环境出图
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# quick 模式缩放常量（仿 smoke_simulator_e2e.py；仅验证链路用）
# ---------------------------------------------------------------------------
QUICK_WEIGHTS: List[Tuple[float, float, float]] = [
    (1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0),
    (1 / 3, 1 / 3, 1 / 3),
]
QUICK_GRID_N = 6          # uniform_grid / random 的权重点数（正式 50）
QUICK_T_MAX_ROUNDS = 1    # ParetoZoom 加密轮数（正式 5，冒烟同款 1）
QUICK_PENALTY_T_MAX = 6   # PenaltyFlex 内循环上限（正式 15）
QUICK_NSGA_POP = 30       # NSGA-II 种群（正式 100）
QUICK_NSGA_GEN = 30       # NSGA-II 代数（正式 200）

EXPERIMENT_ORDER = ["0", "2", "3"]  # 保底顺序：消融 → PenaltyFlex → ParetoZoom


# ---------------------------------------------------------------------------
# 日志 tee：控制台 + run_log.txt 双写
# ---------------------------------------------------------------------------
class _Tee(io.TextIOBase):
    def __init__(self, *streams):
        self._streams = streams

    def write(self, s):
        for st in self._streams:
            st.write(s)
        return len(s)

    def flush(self):
        for st in self._streams:
            st.flush()


def log(msg: str = "") -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# 求解器与 ParetoZoom 工厂补丁（驱动层进程内补丁，不改 src 源码）
# ---------------------------------------------------------------------------
def _build_solver(args, seed: int | None = None):
    """构造驱动专用 KaiwuSolver（simulator 模式 + CLI num_reads 贯通）。

    仿 smoke_simulator_e2e.py 的 FastSolver：ParetoZoom 内部调用
    solve_from_model 不传 num_reads（走默认 1000），此处覆盖默认值，
    把 CLI --num-reads 贯通到后端采样循环（方案 D-05）。

    B-1：seed 贯通 —— 实验层（comparison/ablation/sensitivity）经
    ParetoZoom(seed=seed+rep) 传入时由工厂以该 rep 级种子构造求解器；
    驱动层自建 ParetoZoom（_representative_run 等未传 seed 的路径）
    回退 args.seed，保证 --seed 控制下全进程可复现。
    """
    from ap3_qubo.solver.kaiwu_solver import KaiwuSolver

    class _DriverSolver(KaiwuSolver):
        def solve_from_model(self, model, n_vars=38, num_reads=1000, top_k=None, **kw):
            return super().solve_from_model(
                model, n_vars=n_vars,
                num_reads=args.num_reads, top_k=top_k, **kw,
            )

    return _DriverSolver(mode=args.solver_mode, sa_sweeps=args.sa_sweeps,
                         seed=args.seed if seed is None else seed)


def _build_pz_factory(args):
    """返回 ParetoZoom 工厂：注入 simulator 求解器 + quick 缩放。

    打补丁位置：ablation / comparison 模块命名空间中的 ParetoZoom 名字
    （两模块均 from ..exploration.pareto_zoom import ParetoZoom）。
    """
    from ap3_qubo.exploration.pareto_zoom import ParetoZoom as _OriginalPZ

    def factory(*f_args, **f_kwargs):
        if args.quick:
            # 3 权重 + 小网格（compare_exploration 显式传 uniform_grid_n=50，
            # quick 下须强制覆盖；initial_weights 各调用方均不传）
            f_kwargs["initial_weights"] = list(QUICK_WEIGHTS)
            f_kwargs["uniform_grid_n"] = QUICK_GRID_N
        # B-1：实验层 B-1 修复后 compare_*/ablation 会向 ParetoZoom 传
        # seed=seed+rep；工厂以 rep 级种子构造注入的求解器（quick 与非
        # quick 同一路径），未传 seed 的调用回退 args.seed。
        f_kwargs["solver"] = _build_solver(args, seed=f_kwargs.get("seed"))
        inst = _OriginalPZ(*f_args, **f_kwargs)
        if args.quick:
            # 与 smoke_simulator_e2e.py 同款实例级压缩
            inst._params = dataclasses.replace(
                inst._params, t_max_rounds=QUICK_T_MAX_ROUNDS
            )
        return inst

    return factory


def apply_driver_patches(args) -> None:
    """应用全部驱动层补丁（幂等，进程级一次性）。"""
    import ap3_qubo.experiments.ablation as abl_mod
    import ap3_qubo.experiments.comparison as cmp_mod
    import ap3_qubo.experiments.sensitivity as sens_mod

    factory = _build_pz_factory(args)
    abl_mod.ParetoZoom = factory
    cmp_mod.ParetoZoom = factory
    # 实验 4：SensitivityAnalyzer.run 内 ParetoZoom(gamma_discount=γ, seed=...)
    # 同样经工厂注入 simulator 求解器 + quick 缩放，与 0/1/2/3 同一约定
    sens_mod.ParetoZoom = factory

    if args.quick:
        # PenaltyFlex/Linear 内循环上限（pareto_zoom 模块级 CONSTRAINT 引用，
        # 仅影响本进程；physical_params.CONSTRAINT 本体不动）
        import ap3_qubo.exploration.pareto_zoom as pz_mod

        pz_mod.CONSTRAINT = dataclasses.replace(
            pz_mod.CONSTRAINT, t_max=QUICK_PENALTY_T_MAX
        )
        # NSGA-II quick 缩放：compare_exploration 为函数内局部 import
        # （comparison.py:168），须补丁在 nsga2_baseline 模块属性上
        import ap3_qubo.experiments.nsga2_baseline as nsga_mod

        _OrigNSGA = nsga_mod.NSGA2Optimizer

        class _QuickNSGA2Optimizer(_OrigNSGA):
            """quick 模式 NSGA-II 缩放补丁（子类化，非 lambda 工厂）。

            必须保持真实类身份：nsga2_baseline 内部经模块全局名调用
            NSGA2Optimizer._setup_bounds() / _project_to_box_simplex()
            （:73/:152/:172/:232 等），补丁后该全局名解析到本子类，
            静态方法经继承照常可用；lambda/工厂函数无这些方法，
            __init__ 内即 AttributeError（exp3 quick 路径崩溃点）。
            """

            def __init__(self, *a, **kw):
                # quick 规模强制覆盖（compare_exploration 无参调用；
                # 防御性弹出避免与显式传入冲突）
                kw.pop("pop_size", None)
                kw.pop("generations", None)
                super().__init__(
                    *a,
                    pop_size=QUICK_NSGA_POP,
                    generations=QUICK_NSGA_GEN,
                    **kw,
                )

        nsga_mod.NSGA2Optimizer = _QuickNSGA2Optimizer


# ---------------------------------------------------------------------------
# 统计聚合工具
# ---------------------------------------------------------------------------
def _summarize(values: List[float]) -> Dict[str, Any]:
    arr = np.asarray(values, dtype=float)
    n = int(arr.size)
    mean = float(arr.mean()) if n else 0.0
    sd = float(arr.std(ddof=1)) if n > 1 else 0.0
    ci95 = 1.96 * sd / math.sqrt(n) if n > 1 else 0.0
    return {
        "values": [float(v) for v in arr],
        "n": n,
        "mean": mean,
        "std": sd,
        "ci95": [mean - ci95, mean + ci95],
        "min": float(arr.min()) if n else 0.0,
        "max": float(arr.max()) if n else 0.0,
    }


def _write_csv(path: Path, header: List[str], rows: List[List[Any]]) -> None:
    lines = [",".join(header)]
    for r in rows:
        lines.append(",".join(str(x) for x in r))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _stats_to_report_md(agg: Dict[str, Dict[str, List[float]]],
                        name: str, baseline: str | None) -> str:
    """用现有 statistics.reporting 生成 Markdown 统计报告。

    每组一次性 add_metric 完整 rep 列表（单组单次调用；reporting.py
    已为 A-1 追加语义，全新 ExperimentStats 上 extend 等价于 set）。
    """
    from ap3_qubo.statistics.reporting import ExperimentStats, report_results

    stats = ExperimentStats(name=name)
    for metric, groups in agg.items():
        for g, vals in groups.items():
            stats.add_metric(metric, g, list(vals))
    return report_results(stats, baseline_group=baseline)


# ---------------------------------------------------------------------------
# 代表性 run + 出图
# ---------------------------------------------------------------------------
def _representative_run(args) -> Tuple[Any, Any, Any]:
    """跑一次 AP³ 完整管线（PenaltyFlex 自适应 + ParetoZoom），

    取回 archive / rounds / pz 实例用于出图与原始数据落盘。
    """
    factory = _build_pz_factory(args)
    pz = factory(penalty_strategy="adaptive", exploration_strategy="pareto_zoom")
    archive, rounds = pz.run()
    return archive, rounds, pz


def _lambda_history_from_records(archive) -> List[Tuple[float, float]]:
    """从入档记录重建 λ 轨迹（SolutionRecord.lambdas，pareto_zoom.py:663）。

    同一内循环迭代最多产生 10 条同 λ 记录，折叠连续重复值得到逐迭代轨迹。
    """
    history: List[Tuple[float, float]] = []
    for rec in archive.all_records():
        lam = (float(rec.lambdas[0]), float(rec.lambdas[1]))
        if not history or history[-1] != lam:
            history.append(lam)
    return history


def _make_plots(args, exp_dir: Path, tag: str, archive,
                agg: Dict[str, Dict[str, List[float]]],
                hv_metric_groups: Dict[str, List[float]]) -> List[str]:
    """调用 visualization 包现有函数出图；单张失败不阻断其余。"""
    from ap3_qubo import visualization as viz

    saved: List[str] = []

    def _save(fig, fname: str) -> None:
        p = exp_dir / fname
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved.append(fname)

    objectives = archive.get_objective_matrix()
    comps = archive.get_fractions_of_front()
    hv_hist = archive.get_hv_history()
    lam_hist = _lambda_history_from_records(archive)

    plotters = []
    if len(objectives) > 0:
        plotters += [
            ("pareto_front_2d.png",
             lambda: viz.plot_pareto_2d(objectives, title=f"Pareto Front 2D — {tag}")),
            ("pareto_front_3d.png",
             lambda: viz.plot_pareto_3d(objectives, title=f"Pareto Front 3D — {tag}")),
        ]
    if len(hv_hist) >= 2:
        plotters.append(
            ("hv_convergence.png",
             lambda: viz.plot_hv_progression(hv_hist, title=f"HV Convergence — {tag}"))
        )
    if len(lam_hist) >= 2:
        plotters.append(
            ("lambda_trajectory.png",
             lambda: viz.plot_lambda_trajectory(lam_hist, title=f"λ Trajectory — {tag}"))
        )
    if comps:
        plotters += [
            ("composition_heatmap.png",
             lambda: viz.plot_composition_heatmap(comps, title=f"Composition Heatmap — {tag}")),
            ("element_distribution.png",
             lambda: viz.plot_element_distribution(comps, title=f"Element Distribution — {tag}")),
        ]

    for fname, fn in plotters:
        try:
            _save(fn(), fname)
            log(f"    [plot] {fname}")
        except Exception as exc:  # 单图失败不阻断
            log(f"    [plot] {fname} 失败（跳过）: {exc!r}")

    # 组间 HV 箱线图（visualization 包无此函数，驱动层用 matplotlib 直绘）
    groups = [(g, v) for g, v in hv_metric_groups.items() if len(v) > 0]
    if groups:
        try:
            fig, ax = plt.subplots(figsize=(max(6, 1.6 * len(groups)), 5))
            ax.boxplot([v for _, v in groups],
                       tick_labels=[g for g, _ in groups], showmeans=True)
            ax.set_ylabel("Hypervolume (HV)")
            ax.set_title(f"HV by Group — {tag}")
            ax.grid(True, axis="y", alpha=0.3)
            plt.setp(ax.get_xticklabels(), rotation=20, ha="right", fontsize=8)
            _save(fig, "hv_boxplot.png")
            log("    [plot] hv_boxplot.png")
        except Exception as exc:
            log(f"    [plot] hv_boxplot.png 失败（跳过）: {exc!r}")

    return saved


def _dump_representative(exp_dir: Path, archive, rounds) -> None:
    """代表性 run 原始数据落盘（npz + 成分 csv）。"""
    objectives = archive.get_objective_matrix()
    lam_hist = _lambda_history_from_records(archive)
    np.savez(
        exp_dir / "representative_front.npz",
        objectives=objectives,
        objectives_norm=archive.get_objective_matrix_norm(),
        hv_history=np.asarray(archive.get_hv_history(), dtype=float),
        lambda_history=np.asarray(lam_hist, dtype=float)
        if lam_hist else np.zeros((0, 2)),
        explored_weights=np.asarray(
            archive.get_explored_weights(), dtype=float
        ).reshape(-1, 3),
        round_hv_after=np.asarray(
            [r.hv_after for r in rounds], dtype=float
        ),
    )
    comps = archive.get_fractions_of_front()
    if comps:
        elements = list(comps[0].keys())
        rows = []
        for i, (rec, comp) in enumerate(zip(archive.front, comps)):
            f = rec.objectives
            rows.append([i] + [f"{comp.get(e, 0.0):.4f}" for e in elements]
                        + [f"{f[0]:.4f}", f"{f[1]:.4f}", f"{f[2]:.4f}"])
        _write_csv(exp_dir / "representative_compositions.csv",
                   ["idx"] + elements + ["f1_dH_mix", "f2_density", "f3_cost"],
                   rows)


# ---------------------------------------------------------------------------
# 三个实验的驱动函数
# ---------------------------------------------------------------------------
def run_experiment_0(args, out_root: Path) -> Dict[str, Any]:
    """实验 0：消融（AblationRunner）。

    AblationRunner.run 内部逐 rep append（reporting.py 已为 A-1 追加
    语义，聚合无失真），直接传 n_repetitions=reps。
    """
    from ap3_qubo.experiments.ablation import AblationRunner

    exp_dir = out_root / "exp0"
    exp_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    log(f"  [exp0] AblationRunner.run(n_repetitions={args.reps}, seed={args.seed}) ...")

    runner = AblationRunner()
    results = runner.run(n_repetitions=args.reps, seed=args.seed,
                         fronts_dir=exp_dir / "fronts")
    contributions = runner.compute_contributions(results)

    agg: Dict[str, Dict[str, List[float]]] = {"HV": {}, "Front Size": {}}
    rows: List[List[Any]] = []
    n_errors = 0
    for config in runner.config_names:
        for rep, r in enumerate(results[config]):
            agg["HV"].setdefault(config, []).append(float(r.hv))
            agg["Front Size"].setdefault(config, []).append(float(r.front_size))
            err = r.extra.get("error", "")
            if err:
                n_errors += 1
            # 任务 C：软指标（VEC/δ/Ω/ΔH_mix 各窗口单独通过率，
            # 单列不进门槛）；未计算时为空字符串
            soft = r.extra.get("soft_pass_rates", {})
            soft_cols = [
                f"{soft[k]:.4f}" if k in soft else ""
                for k in ("vec", "delta", "omega", "dh_mix")
            ]
            rows.append([config, rep, f"{r.hv:.6f}", r.front_size,
                         f"{r.feasible_rate:.4f}", f"{r.physical_pass_rate:.4f}",
                         f"{r.feasible_hv:.6f}"] + soft_cols + [err])
    _write_csv(exp_dir / "records.csv",
               ["config", "rep", "hv", "front_size", "feasible_rate",
                "physical_pass_rate", "feasible_hv",
                "vec_pass_rate", "delta_pass_rate", "omega_pass_rate",
                "dh_mix_pass_rate", "error"], rows)

    log("  [exp0] 代表性 run（Full 配置）+ 出图 ...")
    archive, rounds, _ = _representative_run(args)
    _dump_representative(exp_dir, archive, rounds)
    # 图题用 ASCII（matplotlib 默认 DejaVu Sans 无 CJK 字形，中文标题会出方框）
    saved = _make_plots(args, exp_dir, "Exp0 Ablation (Full config)",
                        archive, agg, agg["HV"])

    runtime = time.perf_counter() - t0
    result = {
        "name": "实验 0: 消融实验（PrecisionSplit / PenaltyFlex / ParetoZoom）",
        "n_reps": args.reps,
        "configs": runner.config_names,
        "metrics": {m: {g: _summarize(v) for g, v in groups.items()}
                    for m, groups in agg.items()},
        "contributions_AR_Synergy": contributions,
        "n_config_rep_errors": n_errors,
        "plots": saved,
        "runtime_sec": round(runtime, 2),
    }
    (exp_dir / "results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    # 任务 C：报告追加消融贡献指标口径说明（含旧绝对值口径废弃声明）
    report_md = _stats_to_report_md(agg, result["name"], baseline="Full")
    report_md += (
        "\n\n## 消融贡献指标口径说明（任务 C，2026-07-19）\n\n"
        "- **AR / Synergy（当前口径，带符号）**："
        "AR_i = (HV_Full − HV_i) / HV_Full × 100%，正值 = 组件有正贡献"
        "（去掉它 HV 下降），负值 = 组件在该口径下为负贡献。\n"
        "- **`*_abs_legacy`（已废弃）**：旧绝对值口径 |ΔHV_i| / HV_Full，"
        "会把「去掉组件 HV 反升」伪装成正贡献（formal_exp0_reps20 的 HV "
        "方向性矛盾即源于此，见 reports/feasible_hv_diagnostic_2026-07-19）；"
        "仅保留供与旧数据对照，**不得用于结论**。\n"
        "- **Feasible_AR_* / Feasible_Synergy**：基于 feasible_hv"
        "（硬口径可行子集 HV：仅排除 Ω 不稳定 / 碳化物高风险 / 成分和超差）"
        "的带符号口径；空可行集记 NaN。\n"
        "- **feasible_rate**：硬口径可行率；**physical_pass_rate** 为 "
        "strict all_pass 通过率（VEC 窗口结构性不可达，该口径仅作记录，"
        "不作门槛）；vec/delta/omega/dh_mix_pass_rate 为各窗口单独"
        "通过率（软指标，不进门槛）。\n"
        "- 逐 rep 逐配置前沿已落盘 `fronts/{config}_rep{NN}.npz`。\n"
    )
    (exp_dir / "report.md").write_text(report_md, encoding="utf-8")
    log(f"  [exp0] 完成，耗时 {runtime:.1f}s | contributions={contributions}")
    return result


def _run_comparison_experiment(args, out_root: Path, exp_id: str,
                               compare_fn, name: str,
                               baseline: str, tag: str) -> Dict[str, Any]:
    """实验 2/3 公共驱动：驱动层外循环 reps（reporting.py 已为 A-1
    追加语义；外循环绕法保留以掌控逐 rep 记录与失败容错，
    见模块 docstring 接口事实 2），逐 rep 聚合。"""
    exp_dir = out_root / f"exp{exp_id}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    agg: Dict[str, Dict[str, List[float]]] = {}
    rows: List[List[Any]] = []
    for rep in range(args.reps):
        rep_seed = args.seed + rep
        log(f"  [exp{exp_id}] rep {rep + 1}/{args.reps} (seed={rep_seed}) ...")
        rep_t0 = time.perf_counter()
        stats = compare_fn(n_repetitions=1, seed=rep_seed)
        for metric, groups in stats.metrics.items():
            for g, vals in groups.items():
                agg.setdefault(metric, {}).setdefault(g, []).extend(vals)
                for v in vals:
                    rows.append([g, rep, metric, f"{v:.6f}"])
        log(f"  [exp{exp_id}] rep {rep + 1} 完成，"
            f"耗时 {time.perf_counter() - rep_t0:.1f}s")

    _write_csv(exp_dir / "records.csv",
               ["group", "rep", "metric", "value"], rows)

    log(f"  [exp{exp_id}] 代表性 run（AP³ 完整管线）+ 出图 ...")
    archive, rounds, _ = _representative_run(args)
    _dump_representative(exp_dir, archive, rounds)
    saved = _make_plots(args, exp_dir, tag,
                        archive, agg, agg.get("HV", {}))

    runtime = time.perf_counter() - t0
    result = {
        "name": name,
        "n_reps": args.reps,
        "groups": list(next(iter(agg.values()), {}).keys()),
        "metrics": {m: {g: _summarize(v) for g, v in groups.items()}
                    for m, groups in agg.items()},
        "plots": saved,
        "runtime_sec": round(runtime, 2),
    }
    (exp_dir / "results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (exp_dir / "report.md").write_text(
        _stats_to_report_md(agg, name, baseline=baseline), encoding="utf-8")
    hv = result["metrics"].get("HV", {})
    brief = {g: f"{s['mean']:.4f}±{s['std']:.4f}" for g, s in hv.items()}
    log(f"  [exp{exp_id}] 完成，耗时 {runtime:.1f}s | HV mean±std: {brief}")
    return result


def run_experiment_1(args, out_root: Path) -> Dict[str, Any]:
    """实验 1：PrecisionSplit(38) vs Unified(48) vs Unified(38) 编码效率。

    compare_encoding 与 compare_penalty / compare_exploration 同签名
    （n_repetitions, seed）-> ExperimentStats，直接复用实验 2/3 公共驱动；
    quick/formal 缩放经 apply_driver_patches 的 cmp_mod.ParetoZoom
    工厂补丁贯通（encoding_type 由实验层逐组传入，工厂不改写）。
    """
    from ap3_qubo.experiments.comparison import compare_encoding

    return _run_comparison_experiment(
        args, out_root, "1", compare_encoding,
        "实验 1: PrecisionSplit 编码效率", baseline="PrecisionSplit(38)",
        tag="Exp1 PrecisionSplit Encoding Comparison")


def run_experiment_2(args, out_root: Path) -> Dict[str, Any]:
    """实验 2：PenaltyFlex vs Grid-Search / Linear / Fixed(1/10/100)。"""
    from ap3_qubo.experiments.comparison import compare_penalty

    return _run_comparison_experiment(
        args, out_root, "2", compare_penalty,
        "实验 2: PenaltyFlex 自适应惩罚", baseline="PenaltyFlex",
        tag="Exp2 PenaltyFlex Comparison")


def run_experiment_3(args, out_root: Path) -> Dict[str, Any]:
    """实验 3：ParetoZoom vs Uniform Grid / NSGA-II / Random。"""
    from ap3_qubo.experiments.comparison import compare_exploration

    return _run_comparison_experiment(
        args, out_root, "3", compare_exploration,
        "实验 3: ParetoZoom 前沿探索", baseline="ParetoZoom",
        tag="Exp3 ParetoZoom Comparison")


def run_experiment_4(args, out_root: Path) -> Dict[str, Any]:
    """实验 4：γ 敏感性分析（SensitivityAnalyzer，基准 γ=0.3）。

    SensitivityAnalyzer 为类接口（非 compare_* 函数签名），驱动层单独
    实现，落盘结构与 0/1/2/3 对齐：records.csv 记逐 γ×rep 原始 HV，
    report.md 用实验层自带的 γ 表格报告，代表性 run + 出图沿用公共工具。
    γ 列表保持默认 5 值 [0.1, 0.2, 0.3, 0.4, 0.5]（基准 γ=0.3，即
    MIEDEMA.gamma_discount；quick 与 formal 结构一致，仅单次 run 规模
    缩放，与实验 2 quick 保留 6 组策略同约定）；reps 经 CLI --reps 贯通。
    """
    from ap3_qubo.experiments.sensitivity import SensitivityAnalyzer

    exp_dir = out_root / "exp4"
    exp_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    log(f"  [exp4] SensitivityAnalyzer.run(n_repetitions={args.reps}, "
        f"seed={args.seed}) ...")

    analyzer = SensitivityAnalyzer()
    results = analyzer.run(n_repetitions=args.reps, seed=args.seed)

    # 逐 γ×rep 原始记录 + 聚合（agg 供组间 HV 箱线图复用）
    agg: Dict[str, Dict[str, List[float]]] = {"HV": {}}
    rows: List[List[Any]] = []
    for gamma in sorted(results.keys()):
        g = f"gamma={gamma:.2f}"
        for rep, hv in enumerate(results[gamma].hv_values):
            agg["HV"].setdefault(g, []).append(float(hv))
            rows.append([g, rep, "HV", f"{float(hv):.6f}"])
    _write_csv(exp_dir / "records.csv",
               ["group", "rep", "metric", "value"], rows)

    log("  [exp4] 代表性 run（AP³ 完整管线）+ 出图 ...")
    archive, rounds, _ = _representative_run(args)
    _dump_representative(exp_dir, archive, rounds)
    saved = _make_plots(args, exp_dir, "Exp4 Gamma Sensitivity",
                        archive, agg, agg["HV"])

    runtime = time.perf_counter() - t0
    result = {
        "name": "实验 4: γ 敏感性分析",
        "n_reps": args.reps,
        "gamma_values": sorted(results.keys()),
        "baseline_gamma": analyzer._baseline_gamma,
        "per_gamma": {
            f"{gamma:.2f}": {
                "hv_mean": results[gamma].hv_mean,
                "hv_std": results[gamma].hv_std,
                "hv_values": [float(v) for v in results[gamma].hv_values],
                "top10_overlap": results[gamma].top10_overlap,
                "n_top10_snapshots": len(results[gamma].top10_snapshots),
            }
            for gamma in sorted(results.keys())
        },
        "metrics": {m: {g: _summarize(v) for g, v in groups.items()}
                    for m, groups in agg.items()},
        "plots": saved,
        "runtime_sec": round(runtime, 2),
    }
    (exp_dir / "results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    # 统计报告：实验层自带 γ 表格（含 HV 偏差与 TOP10 重叠率判定）
    (exp_dir / "report.md").write_text(
        analyzer.report(results), encoding="utf-8")
    brief = {g: f"{s['mean']:.4f}±{s['std']:.4f}"
             for g, s in result["metrics"]["HV"].items()}
    log(f"  [exp4] 完成，耗时 {runtime:.1f}s | HV mean±std: {brief}")
    return result


# ---------------------------------------------------------------------------
# CLI 与主流程
# ---------------------------------------------------------------------------
def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="AP³-QUBO 实验驱动：跑实验 → 落盘 → 出图（保底序 0→2→3）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--experiment", choices=["0", "1", "2", "3", "4", "all"],
                   default="all",
                   help="实验编号；all = 按 0→2→3 保底顺序全跑"
                        "（实验 1/4 为补充实验，需单独指定）")
    p.add_argument("--reps", type=int, default=3,
                   help="重复次数（默认小规模 3 用于冒烟；正式实验按方案 20/30）")
    p.add_argument("--num-reads", type=int, default=None,
                   help="每次 QUBO 求解采样数（默认：正式 500 / quick 120）")
    p.add_argument("--out", type=str, default=None,
                   help="输出目录（默认 data/results/<时间戳>/，相对项目根）")
    p.add_argument("--quick", action="store_true",
                   help="链路验证模式：3 权重 + 小采样 + 压缩轮数，不用于结论")
    p.add_argument("--seed", type=int, default=42, help="随机种子基数")
    p.add_argument("--solver-mode", choices=["simulator", "auto"], default="simulator",
                   help="求解器模式（方案 D-04：auto 亦解析为 simulator 后端）")
    p.add_argument("--sa-sweeps", type=int, default=None,
                   help="模拟退火每 read 扫描步数（默认：正式 500 / quick 200）")
    args = p.parse_args(argv)
    if args.num_reads is None:
        # 性能优化（2026-07-18）：默认从 1000 降至 500。
        # 500 reads 与 1000 reads 的 SA 采样质量差异通常 <5%（同 seed），
        # 但时间减半。需更高精度时通过 --num-reads 1000 显式覆盖。
        args.num_reads = 120 if args.quick else 500
    if args.sa_sweeps is None:
        args.sa_sweeps = 200 if args.quick else 500
    return args


def main(argv: List[str] | None = None) -> int:
    # 强制 stdout UTF-8，避免重定向管道时 GBK 编码错误（如 AP³ → \xb3）
    sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args(argv)
    total_t0 = time.perf_counter()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = (Path(args.out) if args.out
                else ROOT / "data" / "results" / ts)
    if not out_root.is_absolute():
        out_root = ROOT / out_root
    out_root.mkdir(parents=True, exist_ok=True)

    # 日志 tee 到 run_log.txt
    log_path = out_root / "run_log.txt"
    log_file = open(log_path, "w", encoding="utf-8")
    sys.stdout = _Tee(sys.__stdout__, log_file)

    log("=" * 72)
    log("AP³-QUBO 实验驱动（第 3 批）— run_experiments.py")
    log(f"  时间        : {datetime.now().isoformat(timespec='seconds')}")
    log(f"  Python      : {sys.executable}")
    log(f"  输出目录    : {out_root}")
    log(f"  实验        : {args.experiment} | reps={args.reps} | seed={args.seed}")
    log(f"  求解器      : KaiwuSolver(mode='{args.solver_mode}') "
        f"num_reads={args.num_reads} sa_sweeps={args.sa_sweeps} "
        f"seed={args.seed}（B-1：实验层逐 rep 传 seed+rep，可复现）")
    log(f"  quick 模式  : {args.quick}"
        + (f"（3 权重/网格 {QUICK_GRID_N}/轮数 {QUICK_T_MAX_ROUNDS}/"
           f"PenaltyFlex t_max {QUICK_PENALTY_T_MAX}/"
           f"NSGA {QUICK_NSGA_POP}×{QUICK_NSGA_GEN}，仅验证链路）"
           if args.quick else ""))
    log("=" * 72)

    apply_driver_patches(args)
    log("[init] 驱动层补丁已应用（simulator 注入"
        + (" + quick 缩放" if args.quick else "") + "）")

    selected = EXPERIMENT_ORDER if args.experiment == "all" else [args.experiment]
    dispatch = {"0": run_experiment_0, "1": run_experiment_1,
                "2": run_experiment_2, "3": run_experiment_3,
                "4": run_experiment_4}

    summary: Dict[str, Any] = {
        "metadata": {
            "script": str(Path(__file__).resolve()),
            "python": sys.executable,
            "timestamp": ts,
            "args": {k: v for k, v in vars(args).items()},
            "note": "quick 模式仅验证链路，结论数据须关闭 quick 并按方案 reps 运行",
        },
        "experiments": {},
    }

    failed: List[str] = []
    for exp_id in selected:
        log(f"\n----- 实验 {exp_id} 开始 " + "-" * 50)
        try:
            res = dispatch[exp_id](args, out_root)
            summary["experiments"][f"experiment_{exp_id}"] = res
        except Exception as exc:
            failed.append(exp_id)
            log(f"  [exp{exp_id}] ★ 失败: {exc!r}")
            summary["experiments"][f"experiment_{exp_id}"] = {
                "error": repr(exc)}

    total = time.perf_counter() - total_t0
    summary["metadata"]["total_runtime_sec"] = round(total, 2)
    summary["metadata"]["failed_experiments"] = failed
    (out_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    log("\n" + "=" * 72)
    log(f"全部结束，总耗时 {total:.1f}s | 输出: {out_root}")
    if failed:
        log(f"★ 失败实验: {failed}（详见 run_log.txt 与各 exp*/results.json）")
    else:
        log("全部实验成功。")
    log("=" * 72)

    log_file.close()
    sys.stdout = sys.__stdout__
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
