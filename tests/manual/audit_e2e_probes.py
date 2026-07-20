# -*- coding: utf-8 -*-
"""Validation_Auditor 端到端探针（进程内 monkeypatch，不改任何源码文件）。

A. kaiwu 1.3.1 顶层 API 补齐（仅本进程）→ 小规模真实链路跑
   AblationRunner Full vs Abl-2（验证本次 ablation.py 修复未破坏链路）。
B. γ 有效性探针：QUBOBuilder(gamma_discount=0.1 vs 0.5) 的 Q 矩阵
   必须不同（实验 4 的 γ 旋钮真实生效）。
C. NSGA-II 投影修复算子偏差量化：box 均匀采样 → _project_to_box_simplex
   vs 可行域均匀参考（Dirichlet 拒绝采样），比较边际分布。

运行：PYTHONPATH=src "$DAIMON_USER_PYTHON" tests/manual/audit_e2e_probes.py
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

# =============================================================================
# A0. kaiwu 1.3.1 顶层 API monkeypatch（仅本进程生效；修复属 solver 负责人）
# =============================================================================
import kaiwu
import kaiwu.qubo
import kaiwu.conversion
import kaiwu.core

if not hasattr(kaiwu, "qubo_matrix_to_qubo_model"):
    kaiwu.qubo_matrix_to_qubo_model = kaiwu.qubo.qubo_matrix_to_qubo_model
if not hasattr(kaiwu, "qubo_model_to_ising_model"):
    kaiwu.qubo_model_to_ising_model = kaiwu.conversion.qubo_model_to_ising_model
if not hasattr(kaiwu, "get_sol_dict"):
    kaiwu.get_sol_dict = kaiwu.core.get_sol_dict
if not hasattr(kaiwu, "get_sorted_solutions"):
    from ap3_qubo.solver.kaiwu_solver import _get_sorted_solutions_shim
    kaiwu.get_sorted_solutions = _get_sorted_solutions_shim
print("[A0] kaiwu 1.3.1 顶层 API 已在本进程内补齐（monkeypatch）")

from ap3_qubo.experiments.ablation import AblationRunner, GRID_SEARCH_LAMBDA_GRID
from ap3_qubo.solver.kaiwu_solver import KaiwuSolver

PASS = 0
FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    tag = "PASS" if cond else "FAIL"
    if cond:
        PASS += 1
    else:
        FAIL += 1
    print(f"  [{tag}] {name}" + (f" | {detail}" if detail else ""))


# =============================================================================
# A. 小规模真实链路：Full vs Abl-2（3 权重、num_reads=60、1 次重复）
# =============================================================================
print()
print("=" * 72)
print("[A] 小规模真实链路 AblationRunner Full vs Abl-2")
print("=" * 72)


class SmallBudgetSolver(KaiwuSolver):
    def __init__(self, reads, **kw):
        super().__init__(**kw)
        self._reads_cap = reads

    def solve_from_model(self, model, n_vars: int = 38,
                         num_reads: int = 1000, top_k=None):
        return super().solve_from_model(
            model, n_vars=n_vars,
            num_reads=min(num_reads, self._reads_cap), top_k=top_k,
        )


weights3 = [(0.8, 0.1, 0.1), (0.1, 0.8, 0.1), (0.1, 0.1, 0.8)]
t0 = time.time()
runner = AblationRunner(
    grid_search_reads=60,
    grid_search_weights=weights3,
    pareto_zoom_kwargs={
        "solver": SmallBudgetSolver(reads=60, mode="simulator", sa_sweeps=150, seed=7),
        "initial_weights": weights3,
        "uniform_grid_n": 3,
    },
)
results = runner.run(n_repetitions=1, seed=42, configs=["Full", "Abl-2"])
elapsed = time.time() - t0

for name in ["Full", "Abl-2"]:
    r = results[name][0]
    err = r.extra.get("error", "")
    print(f"  [{name}] HV={r.hv:.4f} front={r.front_size}"
          + (f" ERROR={err}" if err else ""))
    gs = r.extra.get("grid_search")
    if gs:
        print(f"    BASE-4 grid-search: best λ={gs['best_lambda']}, "
              f"hv_per_lambda={ {k: round(v, 2) for k, v in gs['hv_per_lambda'].items()} }")

check("Full HV > 0（真实链路产出）", results["Full"][0].hv > 0)
check("Abl-2 HV > 0（grid-search 链路产出）", results["Abl-2"][0].hv > 0)
gs2 = results["Abl-2"][0].extra.get("grid_search")
check("Abl-2 含 BASE-4 grid-search 详情", gs2 is not None)
if gs2:
    check("选中 λ ∈ BASE-4 网格", gs2["best_lambda"] in GRID_SEARCH_LAMBDA_GRID,
          f"best={gs2['best_lambda']}")
contrib = runner.compute_contributions(
    {k: results.get(k, []) for k in ["Full", "Abl-2"]}
)
print(f"  compute_contributions(缺 Abl-1/3/4) → {contrib}")
check("缺配置 AR 为 NaN 不误报", np.isnan(contrib.get("AR_PrecisionSplit", 0.0)))
print(f"  耗时 {elapsed:.1f}s")

# =============================================================================
# B. γ 有效性探针（实验 4 前置条件）
# =============================================================================
print()
print("=" * 72)
print("[B] γ 有效性：QUBOBuilder(gamma_discount=0.1 vs 0.5) Q 矩阵必须不同")
print("=" * 72)

from ap3_qubo.qubo.builder import QUBOBuilder

b1 = QUBOBuilder(gamma_discount=0.1)
b2 = QUBOBuilder(gamma_discount=0.5)
w = (0.5, 0.3, 0.2)
m1 = b1.build_model(weights=w, lambda_carbide=0.05, lambda_ccr=0.05)
m2 = b2.build_model(weights=w, lambda_carbide=0.05, lambda_ccr=0.05)
q1 = np.asarray(m1.get_matrix() if hasattr(m1, "get_matrix") else m1, dtype=float)
q2 = np.asarray(m2.get_matrix() if hasattr(m2, "get_matrix") else m2, dtype=float)
diff = np.abs(q1 - q2).max()
check("γ=0.1 vs 0.5 的 Q 矩阵不同", diff > 1e-9, f"max|ΔQ|={diff:.4f}")
# γ 只应影响 C-主元交叉项：主对角与主元-主元项应一致
check("γ 差异量级合理（|ΔQ| 有限）", diff < 1e3)

# MixingEnthalpy 实例级 γ（父代理通报的修复点复核）
from ap3_qubo.objectives.mixing_enthalpy import MixingEnthalpy

comp = {"Al": 19.8, "Co": 19.8, "Cr": 19.8, "Fe": 19.8, "Ni": 19.8, "C": 1.0}
dh1 = MixingEnthalpy(gamma=0.1).evaluate(comp)
dh2 = MixingEnthalpy(gamma=0.5).evaluate(comp)
check("MixingEnthalpy 实例 γ 生效", abs(dh1 - dh2) > 1e-9,
      f"γ=0.1: {dh1:.4f}, γ=0.5: {dh2:.4f}")

# =============================================================================
# C. NSGA-II 投影修复算子偏差量化
# =============================================================================
print()
print("=" * 72)
print("[C] _project_to_box_simplex 初始化偏差量化")
print("=" * 72)

from ap3_qubo.experiments.nsga2_baseline import NSGA2Optimizer
from ap3_qubo.physical_params import ALL_ELEMENTS, ENCODING

bounds = NSGA2Optimizer._setup_bounds()
lows = np.array([b[0] for b in bounds])
highs = np.array([b[1] for b in bounds])

rng = np.random.default_rng(7)
n_s = 20000
raw = rng.uniform(lows, highs, size=(n_s, len(bounds)))
proj = np.array([
    NSGA2Optimizer._project_to_box_simplex(list(v), list(lows), list(highs))
    for v in raw
])

at_bound_proj = np.mean(
    (np.abs(proj - lows) < 1e-6) | (np.abs(proj - highs) < 1e-6), axis=0
)
mean_proj = proj.mean(axis=0)

# 参考：可行域近均匀（Dirichlet(1)×100 拒绝采样到 box）
n_d = 400000
d = rng.dirichlet(np.ones(6), size=n_d) * 100.0
mask = np.all((d >= lows) & (d <= highs), axis=1)
ref = d[mask]
at_bound_ref = np.mean(
    (np.abs(ref - lows) < 1e-6) | (np.abs(ref - highs) < 1e-6), axis=0
)
mean_ref = ref.mean(axis=0)

print(f"  可行域拒绝采样接受率: {mask.mean():.3%}（{mask.sum()}/{n_d}）")
print("  元素      box均匀→投影 P(贴界)  mean   |  可行域均匀 P(贴界)  mean")
for i, e in enumerate(ALL_ELEMENTS):
    print(f"  {e:>4}      {at_bound_proj[i]:>10.3%}  {mean_proj[i]:6.2f}   |"
          f"      {at_bound_ref[i]:>10.3%}  {mean_ref[i]:6.2f}")

check("投影后全部满足边界+Σc=100", 
      np.all(np.abs(proj.sum(axis=1) - 100.0) < 1e-6)
      and np.all(proj >= lows - 1e-9) and np.all(proj <= highs + 1e-9))
check("参考集可行", len(ref) > 1000)
print("  → P(贴界) 差异即修复算子引入的分布偏差（连续投影每代重复施加）")

# =============================================================================
# 汇总
# =============================================================================
print()
print("=" * 72)
print(f"端到端探针汇总: PASS={PASS}, FAIL={FAIL}")
print("=" * 72)
sys.exit(0 if FAIL == 0 else 1)
