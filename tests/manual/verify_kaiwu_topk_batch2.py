# -*- coding: utf-8 -*-
"""第 2 批修复验证脚本：kaiwu_solver TOP-K + 模拟器门禁 + 硬件自检。

运行：C:\\Users\\At3ase\\AppData\\Local\\Programs\\Python\\Python310\\python.exe
"""
import sys, time, itertools, logging
import numpy as np

sys.path.insert(0, r"D:\QUBO\src")
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

from ap3_qubo.solver.kaiwu_solver import KaiwuSolver

print("=" * 70)
print("[1] 模拟器模式 + TOP-K：38 变量 QUBO（矩阵路径）")
print("=" * 70)
rng = np.random.default_rng(0)
Q = np.triu(rng.normal(0, 1, (38, 38)))
solver = KaiwuSolver(mode="simulator", seed=42)
t0 = time.perf_counter()
res = solver.solve(Q, num_reads=200, top_k=20)
dt = time.perf_counter() - t0
print(f"num_reads={res.num_reads}, 返回解数={len(res.solutions)}, "
      f"耗时={res.timing_ms:.0f} ms (wall {dt:.2f}s)")
energies = [s.energy for s in res.solutions]
print("TOP-5 energy:", [round(e, 4) for e in energies[:5]])
assert energies == sorted(energies), "解列表未按能量升序！"
assert len(res.solutions) == 20
# 能量一致性：bits @ Q @ bits 复算
for s in res.solutions[:5]:
    e_check = float(s.bits @ Q @ s.bits)
    assert abs(e_check - s.energy) < 1e-6, (e_check, s.energy)
print("能量复算 bits@Q@bits 与 SDK hamilton+bias 一致 ✓")
# 唯一性
seen = {s.bits.tobytes() for s in res.solutions}
assert len(seen) == len(res.solutions), "存在重复解！"
print("TOP-20 全部唯一 ✓")

print()
print("=" * 70)
print("[2] 最优性校验：12 变量蛮力对照")
print("=" * 70)
Q2 = np.triu(rng.normal(0, 1, (12, 12)))
res2 = KaiwuSolver(mode="simulator", seed=1).solve(Q2, num_reads=50, top_k=10)
bf = min(float(x @ Q2 @ x) for x in itertools.product([0, 1], repeat=12))
print(f"SA best={res2.best_energy:.6f}  蛮力最优={bf:.6f}")
assert abs(res2.best_energy - bf) < 1e-6
print("模拟器达到蛮力最优 ✓")

print()
print("=" * 70)
print("[3] num_reads 贯通：1000 次采样（方案 D-05）")
print("=" * 70)
t0 = time.perf_counter()
res3 = KaiwuSolver(mode="simulator", seed=7).solve(Q, num_reads=1000, top_k=50)
print(f"num_reads={res3.num_reads}, 唯一解数={len(res3.solutions)}, "
      f"wall={time.perf_counter()-t0:.2f}s")
print("best energy:", round(res3.best_energy, 4),
      "| 200-reads best:", round(res.best_energy, 4))
assert res3.best_energy <= res.best_energy + 1e-9

print()
print("=" * 70)
print("[4] 模式分支与门禁")
print("=" * 70)
print("is_available('auto')      =", KaiwuSolver.is_available("auto"))
print("is_available('simulator') =", KaiwuSolver.is_available("simulator"))
print("is_available('cim')       =", KaiwuSolver.is_available("cim"))
try:
    KaiwuSolver(mode="cim").solve(Q2, num_reads=4)
    print("!! CIM 模式未报错")
except RuntimeError as e:
    print("CIM 模式显式 RuntimeError ✓  指引摘录:",
          str(e).splitlines()[0])
try:
    KaiwuSolver(mode="bogus")
except ValueError as e:
    print("非法 mode 显式 ValueError ✓")

print()
print("=" * 70)
print("[5] 硬件自检 D-06")
print("=" * 70)
rep = KaiwuSolver(mode="simulator").check_hardware_compatibility(38, 703)
print("38v/703c:", {k: rep[k] for k in ("vars_ok", "couplers_ok",
      "within_scheme_budget", "warnings")})
rep2 = KaiwuSolver(mode="simulator").check_hardware_compatibility(600, 200000)
print("600v/200000c warnings:")
for w in rep2["warnings"]:
    print("  -", w)
assert rep2["warnings"], "超限未产生 WARNING"

print()
print("=" * 70)
print("[6] 构建器路径 + is_feasible 真实检查（QUBOBuilder + sum 约束）")
print("=" * 70)
from ap3_qubo.qubo.builder import QUBOBuilder
builder = QUBOBuilder()
model = builder.build_model(weights=(1 / 3, 1 / 3, 1 / 3))
print("QuboModel 构建成功, 变量数:", builder.num_variables)
t0 = time.perf_counter()
res4 = KaiwuSolver(mode="simulator", seed=3).solve_from_model(
    model, n_vars=38, num_reads=300, top_k=20)
print(f"wall={time.perf_counter()-t0:.2f}s, 解数={len(res4.solutions)}")
from ap3_qubo.encoding.precision_split import PrecisionSplitDecoder
dec = PrecisionSplitDecoder()
print(f"{'rank':>4} {'energy':>10} {'Σc':>8} {'|Δ|':>6} feasible")
for i, s in enumerate(res4.solutions[:10]):
    comp = dec.decode(s.bits)
    tot = sum(comp.fractions.values())
    print(f"{i:>4} {s.energy:>10.4f} {tot:>8.2f} {abs(tot-100):>6.2f} "
          f"{s.is_feasible}  ({s.metadata.get('feasibility_check')})")
# is_feasible 必须与手工复算一致
for s in res4.solutions:
    tot = sum(dec.decode(s.bits).fractions.values())
    assert s.is_feasible == (abs(tot - 100.0) <= 1.0)
print("is_feasible 与解码复算逐一一致 ✓")
n_feas = len(res4.filter_feasible())
print(f"TOP-20 中可行解数: {n_feas}")

print()
print("ALL CHECKS PASSED")
