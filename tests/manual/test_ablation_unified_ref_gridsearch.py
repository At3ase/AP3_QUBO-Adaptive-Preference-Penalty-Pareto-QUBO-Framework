# -*- coding: utf-8 -*-
"""第 3 批小规模实测：消融实验统一参考点 + BASE-4 Grid-search。

验证点：
  1. _compute_hv_unified 确实走 set_unified_reference 统一参考点；
  2. AR_i / Synergy 为方案口径 |ΔHV|/HV_Full（符号修正）；
  3. _UnifiedGridDecoder 与 QUBOBuilder 配置表位布局一致；
  4. 真实模拟器链路小规模跑 Full vs Abl-2（均 precision_split_38）：
     grid-search 选出合理 λ、统一参考点下两配置 HV 可比；
  5. Abl-4（unified_48）记录在案：受 kaiwu_solver._parse_var_index
     硬编码 38 布局的上游缺陷影响，预期空档/占位（非本文件问题）。

运行环境：Python 3.10 + kaiwu（C:\\Users\\At3ase\\AppData\\Local\\Programs\\Python\\Python310\\python.exe）
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import numpy as np

from ap3_qubo.experiments.ablation import (
    AblationRunner,
    AblationResult,
    _compute_hv_unified,
    _build_decoder,
    GRID_SEARCH_LAMBDA_GRID,
)
from ap3_qubo.qubo.builder import QUBOBuilder
from ap3_qubo.solver.kaiwu_solver import KaiwuSolver
from ap3_qubo.validation.hypervolume import (
    HypervolumeCalculator,
    set_unified_reference,
)

print("=" * 70)
print("[1] 统一参考点单测：_compute_hv_unified vs 手动 set_unified_reference")
print("=" * 70)
rng = np.random.default_rng(0)
mats = {"A": rng.random((5, 3)) * 10.0, "B": rng.random((4, 3)) * 10.0 + 5.0}
hv = _compute_hv_unified(mats)
ref = set_unified_reference(mats, margin=0.10)
calc = HypervolumeCalculator(reference_point=ref)
assert abs(hv["A"] - calc.compute(mats["A"])) < 1e-9, "A 配置 HV 未用统一参考点"
assert abs(hv["B"] - calc.compute(mats["B"])) < 1e-9, "B 配置 HV 未用统一参考点"
print(f"  统一参考点 ref = {np.round(ref, 3)}")
print(f"  HV_A = {hv['A']:.4f}, HV_B = {hv['B']:.4f}  → 同参考点口径一致 ✓")

print()
print("=" * 70)
print("[2] AR_i / Synergy 符号口径单测（合成数据）")
print("=" * 70)
synth = {
    "Full": [AblationResult("Full", hv=100.0, front_size=10)],
    "Abl-1": [AblationResult("Abl-1", hv=90.0, front_size=9)],
    "Abl-2": [AblationResult("Abl-2", hv=80.0, front_size=8)],
    "Abl-3": [AblationResult("Abl-3", hv=95.0, front_size=9)],
    "Abl-4": [AblationResult("Abl-4", hv=70.0, front_size=7)],
}
contrib = AblationRunner().compute_contributions(synth)
# 方案口径 AR_i = |ΔHV_i|/HV_Full：|90-100|/100=10%，|80-100|=20%，|95-100|=5%
assert contrib["AR_PrecisionSplit"] == 10.0, contrib
assert contrib["AR_PenaltyFlex"] == 20.0, contrib
assert contrib["AR_ParetoZoom"] == 5.0, contrib
# Synergy = Σ|AR| − |70−100|/100 = 35 − 30 = 5
assert contrib["Synergy"] == 5.0, contrib
assert all(v >= 0 for k, v in contrib.items() if k.startswith("AR_")), "AR 必须非负"
print(f"  contributions = {contrib}  → AR 全部非负、口径 |ΔHV|/HV_Full ✓")

print()
print("=" * 70)
print("[3] _UnifiedGridDecoder 位布局单测（对照 QUBOBuilder 配置表）")
print("=" * 70)
cfg48 = QUBOBuilder._ENCODING_CONFIGS["unified_48"]
dec48 = _build_decoder("unified_48")
bits = np.zeros(48, dtype=np.int8)
bits[0] = 1          # 元素0 k=1 → 5 + step_main
bits[8] = 1; bits[9] = 1   # 元素1 k=3
bits[40] = 1; bits[47] = 1  # C: k = 1 + 128 = 129
comp = dec48.decode(bits)
from ap3_qubo.physical_params import MAIN_ELEMENTS, INTERSTITIAL_ELEMENT
assert abs(comp.fractions[MAIN_ELEMENTS[0]] - (cfg48["base_main"] + cfg48["step_main"] * 1)) < 1e-9
assert abs(comp.fractions[MAIN_ELEMENTS[1]] - (cfg48["base_main"] + cfg48["step_main"] * 3)) < 1e-9
assert abs(comp.fractions[INTERSTITIAL_ELEMENT] - (cfg48["base_carbon"] + cfg48["step_carbon"] * 129)) < 1e-9
try:
    dec48.decode(np.zeros(38, dtype=np.int8))
    raise SystemExit("应拒绝 38 bits")
except ValueError:
    pass
print(f"  unified_48 解码与配置表一致（base={cfg48['base_main']}, "
      f"step={cfg48['step_main']:.6f}）；非法位长正确拒绝 ✓")


class SmallBudgetSolver(KaiwuSolver):
    """实测专用：封顶 num_reads 的小预算求解器（不改项目代码）。"""

    def __init__(self, reads: int, **kwargs):
        super().__init__(**kwargs)
        self._reads_cap = reads

    def solve_from_model(self, model, n_vars: int = 38,
                         num_reads: int = 1000, top_k=None):
        return super().solve_from_model(
            model, n_vars=n_vars,
            num_reads=min(num_reads, self._reads_cap), top_k=top_k,
        )


weights3 = [(0.8, 0.1, 0.1), (0.1, 0.8, 0.1), (0.1, 0.1, 0.8)]


def make_runner():
    return AblationRunner(
        grid_search_reads=60,
        grid_search_weights=weights3,
        pareto_zoom_kwargs={
            "solver": SmallBudgetSolver(
                reads=60, mode="simulator", sa_sweeps=150, seed=7
            ),
            "initial_weights": weights3,
            "uniform_grid_n": 3,
        },
    )


def show(name, r):
    print(f"\n  [{name}] HV = {r.hv:.4f}, front_size = {r.front_size}")
    if "error" in r.extra:
        print(f"    ERROR: {r.extra['error']}")
    if r.extra.get("solver_available") is False:
        print("    solver_available = False（占位结果）")
    gs = r.extra.get("grid_search")
    if gs:
        print(f"    BASE-4 网格: {gs['grid']}")
        for lam in gs["grid"]:
            mark = "  ← 选中" if lam == gs["best_lambda"] else ""
            print(f"      λ={lam:>6}: 试跑 HV = {gs['hv_per_lambda'][lam]:.4f}{mark}")
        print(f"    试跑预算 num_reads={gs['grid_search_reads']}, "
              f"试跑权重数={gs['trial_weights']}")
    return gs


print()
print("=" * 70)
print("[4] 小规模实测：Full vs Abl-2（3 权重、num_reads=60、1 次重复）")
print("=" * 70)
t0 = time.time()
results = make_runner().run(n_repetitions=1, seed=42, configs=["Full", "Abl-2"])
elapsed = time.time() - t0
gs2 = None
for name in ["Full", "Abl-2"]:
    gs2 = show(name, results[name][0]) or gs2

assert results["Full"][0].hv > 0, "Full 配置 HV 应 > 0"
assert results["Abl-2"][0].hv > 0, "Abl-2 配置 HV 应 > 0"
assert gs2 is not None, "Abl-2 应含 grid_search 详情"
assert gs2["best_lambda"] in GRID_SEARCH_LAMBDA_GRID, "选中 λ 必须在 BASE-4 网格内"
print(f"\n  耗时 {elapsed:.1f}s；两配置 HV 均在统一参考点下计算且 > 0 ✓")
print(f"  grid-search 选中 λ = {gs2['best_lambda']}（∈ BASE-4 网格）✓")

print()
print("=" * 70)
print("[5] Abl-4（unified_48）在案验证——预期受上游 solver 位映射缺陷影响")
print("=" * 70)
t0 = time.time()
res4 = make_runner().run(n_repetitions=1, seed=42, configs=["Abl-4"])
elapsed4 = time.time() - t0
show("Abl-4", res4["Abl-4"][0])
print(f"\n  耗时 {elapsed4:.1f}s。说明：kaiwu_solver._parse_var_index 把 "
      f"e{{ei}}_b{{bj}} 映射硬编码为 38 布局，unified_48 比特错位，")
print("  绝大多数解被 2% 入档过滤拒绝 → 空档/占位属上游缺陷预期表现，")
print("  需 solver 负责人修复后 Abl-1/Abl-4 才能产出真实前沿。")

print("\n全部实测检查通过。")
