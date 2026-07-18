#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""B-1 可复现性验证：求解器随机种子逐层贯通。

必须用 Python310 环境运行（kaiwu + scipy + deap）：
  C:/Users/At3ase/AppData/Local/Programs/Python/Python310/python.exe \\
      scripts/verify_seed_reproducibility.py

三层验证（全部进程内小规模补丁控制耗时，不改 src 语义）：
  [A] 求解器层：同一 (seed, 同一模型, num_reads=100) 两次求解结果
      逐位一致；不同 seed 有差异。附带验证 ParetoZoom(seed=...) →
      KaiwuSolver 的透传接线与 seed=None 旧行为保持。
  [B] pipeline 层：3 权重 + num_reads=100 + 1 个 PenaltyFlex 迭代
      序列（t_max=1）+ 1 轮 ParetoZoom，同一 seed 两次最终 archive
      目标矩阵逐位一致；不同 seed 有差异。
  [C] 实验层：compare_penalty(n_repetitions=2, seed=42) 连跑两次
      （quick 化补丁：3 权重 / t_max=3 / 1 轮 / num_reads=100 /
      sa_sweeps=100），两次各组 HV 序列完全一致。
"""

from __future__ import annotations

import dataclasses
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

PASS = "[PASS]"
FAIL = "[FAIL]"

QUICK_WEIGHTS = [(1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (1 / 3, 1 / 3, 1 / 3)]

failures = []


def check(ok: bool, label: str) -> None:
    print(f"  {PASS if ok else FAIL} {label}")
    if not ok:
        failures.append(label)


# ===========================================================================
# [A] 求解器层
# ===========================================================================
print("=" * 70)
print("[A] 求解器层：同一 (seed, 模型, num_reads) 逐位一致")
print("=" * 70)

from ap3_qubo.qubo.builder import QUBOBuilder
from ap3_qubo.solver.kaiwu_solver import KaiwuSolver
from ap3_qubo.exploration.pareto_zoom import ParetoZoom

_builder = QUBOBuilder()
_model = _builder.build_model(weights=(0.4, 0.3, 0.3))


def _solve_once(seed):
    solver = KaiwuSolver(mode="simulator", seed=seed, sa_sweeps=100)
    res = solver.solve_from_model(_model, n_vars=38, num_reads=100, top_k=10)
    bits = np.array([s.bits for s in res.solutions])
    energies = np.array([s.energy for s in res.solutions])
    return bits, energies


b1, e1 = _solve_once(42)
b2, e2 = _solve_once(42)
check(
    np.array_equal(b1, b2) and np.array_equal(e1, e2),
    f"seed=42 两次求解逐位一致（{len(e1)} 个 TOP-K 解）",
)

b3, e3 = _solve_once(7)
same_shape = b1.shape == b3.shape and e1.shape == e3.shape
check(
    not (same_shape and np.array_equal(b1, b3) and np.array_equal(e1, e3)),
    "seed=7 与 seed=42 结果有差异",
)

# ParetoZoom → KaiwuSolver 透传接线 + seed=None 旧行为保持
pz_seeded = ParetoZoom(seed=123)
check(
    pz_seeded._solver._config.get("seed") == 123,
    "ParetoZoom(seed=123) 透传到内部 KaiwuSolver（_config['seed']==123）",
)
pz_legacy = ParetoZoom()
check(
    pz_legacy._solver._config.get("seed") is None,
    "ParetoZoom() 默认 seed=None 保持旧行为",
)

# ===========================================================================
# [B] pipeline 层：3 权重 + num_reads=100 + 1 个 PenaltyFlex 迭代序列
# ===========================================================================
print("\n" + "=" * 70)
print("[B] pipeline 层：同一 seed 两次 archive 目标矩阵逐位一致")
print("=" * 70)

import ap3_qubo.exploration.pareto_zoom as pz_mod


class _FastKaiwu(KaiwuSolver):
    """进程内补丁：强制 num_reads=100（ParetoZoom 内部调用不传 num_reads，
    走默认 1000 会拖慢验证；其余求解语义不变）。ParetoZoom 内部以
    KaiwuSolver(mode="auto", seed=seed) 构造，模块名补丁后解析到本子类，
    借此同时验证 ParetoZoom(seed=...) 的真实透传路径。"""

    def solve_from_model(self, model, n_vars=38, num_reads=1000, top_k=None):
        return super().solve_from_model(model, n_vars=n_vars,
                                        num_reads=100, top_k=top_k)


_OrigKaiwu = pz_mod.KaiwuSolver
_OrigConstraint = pz_mod.CONSTRAINT
pz_mod.KaiwuSolver = _FastKaiwu
pz_mod.CONSTRAINT = dataclasses.replace(_OrigConstraint, t_max=1)  # 1 个 PenaltyFlex 迭代序列

try:
    def _run_pipeline(seed: int) -> np.ndarray:
        # 与实验层等价动作：全局 np.random 覆盖权重微扰/Dirichlet 采样
        np.random.seed(seed)
        pz = ParetoZoom(initial_weights=list(QUICK_WEIGHTS), seed=seed)
        pz._params = dataclasses.replace(pz._params, t_max_rounds=1)
        archive, _ = pz.run()
        return archive.get_objective_matrix()

    t0 = time.perf_counter()
    m1 = _run_pipeline(42)
    m2 = _run_pipeline(42)
    m3 = _run_pipeline(123)
    dt = time.perf_counter() - t0
    print(f"  3 次小规模 pipeline 耗时 {dt:.1f}s")

    check(m1.size > 0, f"archive 非空（目标矩阵 {m1.shape}）")
    check(
        np.array_equal(m1, m2),
        "seed=42 两次 pipeline archive 目标矩阵逐位一致",
    )
    check(
        not (m1.shape == m3.shape and np.array_equal(m1, m3)),
        "seed=123 与 seed=42 archive 有差异",
    )
finally:
    pz_mod.KaiwuSolver = _OrigKaiwu
    pz_mod.CONSTRAINT = _OrigConstraint

# ===========================================================================
# [C] 实验层：compare_penalty(n_repetitions=2, seed=42) 连跑两次
# ===========================================================================
print("\n" + "=" * 70)
print("[C] 实验层：compare_penalty 两次各组 HV 序列完全一致")
print("=" * 70)

import ap3_qubo.experiments.comparison as cmp_mod
from ap3_qubo.experiments.comparison import compare_penalty


def _fast_solver(seed, num_reads=100, sa_sweeps=100):
    class _Fast(KaiwuSolver):
        def solve_from_model(self, model, n_vars=38, num_reads_=1000, top_k=None, **kw):
            return super().solve_from_model(
                model, n_vars=n_vars, num_reads=num_reads, top_k=top_k, **kw
            )

    return _Fast(mode="simulator", sa_sweeps=sa_sweeps, seed=seed)


def _pz_factory(*f_args, **f_kwargs):
    """quick 化补丁（仿 scripts/run_experiments.py 工厂）：3 权重 + 小网格 +
    注入带 rep 级 seed 的小预算求解器 + 1 轮 ParetoZoom。"""
    f_kwargs["initial_weights"] = list(QUICK_WEIGHTS)
    f_kwargs["uniform_grid_n"] = 6
    # B-1 修复后 compare_penalty 传入 seed=seed+rep，工厂据此构造注入求解器
    f_kwargs["solver"] = _fast_solver(seed=f_kwargs.get("seed"))
    inst = _OrigPZ(*f_args, **f_kwargs)
    inst._params = dataclasses.replace(inst._params, t_max_rounds=1)
    return inst


_OrigPZ = cmp_mod.ParetoZoom
_OrigConstraint2 = pz_mod.CONSTRAINT
cmp_mod.ParetoZoom = _pz_factory
pz_mod.CONSTRAINT = dataclasses.replace(_OrigConstraint2, t_max=3)

try:
    t0 = time.perf_counter()
    stats_run1 = compare_penalty(n_repetitions=2, seed=42)
    stats_run2 = compare_penalty(n_repetitions=2, seed=42)
    dt = time.perf_counter() - t0
    print(f"  2 次 compare_penalty(quick 化) 耗时 {dt:.1f}s")

    for metric in ("HV", "Feasible Rate"):
        g1 = stats_run1.metrics.get(metric, {})
        g2 = stats_run2.metrics.get(metric, {})
        same = g1.keys() == g2.keys() and all(
            v1 == v2 for v1, v2 in zip(g1.values(), g2.values())
        )
        check(same, f"{metric}：两次运行各组序列完全一致（{len(g1)} 组）")
        for g, vals in g1.items():
            print(f"      {metric} | {g}: {['%.6f' % v for v in vals]}")

    hv1 = stats_run1.metrics.get("HV", {})
    nonzero = any(any(v != 0.0 for v in vals) for vals in hv1.values())
    check(nonzero, "quick 化运行产出非占位 HV（链路真实在跑）")
finally:
    cmp_mod.ParetoZoom = _OrigPZ
    pz_mod.CONSTRAINT = _OrigConstraint2

# ===========================================================================
# 汇总
# ===========================================================================
print("\n" + "=" * 70)
if failures:
    print(f"★ B-1 可复现性验证失败 {len(failures)} 项: {failures}")
    sys.exit(1)
print("B-1 可复现性验证全部通过（求解器层 / pipeline 层 / 实验层）。")
print("=" * 70)
