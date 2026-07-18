"""第 3 批修复验证：NSGA-II 基线可行性约束（修复算子保和）。

用 Python 3.10 + deap 环境运行：
  C:\\Users\\At3ase\\AppData\\Local\\Programs\\Python\\Python310\\python.exe verify_batch3_nsga2_feasibility.py

验证项（小规模真 deap NSGA-II, pop=30, gen=10）：
1. 投影算子单元检查：任意向量投影后同时满足边界与 |Σc-100|<1e-9；
2. 端到端 optimize_and_evaluate 跑通，结果字典含 feasibility 字段；
3. 前沿所有解 |Σc-100| <= 1.0（修复前实测低至 60.3）；
4. 前沿所有个体基因均在 [low, up] 边界内；
5. 目标矩阵 objectives 全部有限且无复数，HV 为有限正数。
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import numpy as np

from ap3_qubo.experiments.nsga2_baseline import NSGA2Optimizer
from ap3_qubo.physical_params import ALL_ELEMENTS, ENCODING

# --- 1. 投影算子单元检查 ---
bounds = NSGA2Optimizer._setup_bounds()
lows = [b[0] for b in bounds]
highs = [b[1] for b in bounds]
rng = np.random.default_rng(7)
for trial in range(200):
    v = rng.uniform(-20.0, 80.0, size=len(bounds)).tolist()  # 含越界乱值
    p = NSGA2Optimizer._project_to_box_simplex(v, lows, highs)
    assert abs(sum(p) - 100.0) < 1e-6, f"投影后 Σc={sum(p)}"
    for x, lo, hi in zip(p, lows, highs):
        assert lo - 1e-9 <= x <= hi + 1e-9, f"投影后越界: {x} not in [{lo},{hi}]"
print("[OK] 投影算子 200 组随机向量（含越界）均收敛到 box∩simplex")

# --- 2~5. 小规模真 deap NSGA-II 端到端 ---
np.random.seed(42)
opt = NSGA2Optimizer(pop_size=30, generations=10)
result = opt.optimize_and_evaluate()

front = result["front"]
objs = result["objectives"]
hv = result["hv"]
feas = result["feasibility"]

assert len(front) > 0, "前沿为空"
sums = np.array([sum(c[e] for e in ALL_ELEMENTS) for c in front])
dev = np.abs(sums - 100.0)
assert np.all(dev <= 1.0), f"不可行解混入前沿: max|Σc-100|={dev.max():.4f}"

for comp in front:
    for i, e in enumerate(ALL_ELEMENTS):
        lo, hi = bounds[i]
        assert lo - 1e-9 <= comp[e] <= hi + 1e-9, f"{e}={comp[e]} 越界 [{lo},{hi}]"

assert objs.shape == (len(front), 3), f"objectives 形状异常 {objs.shape}"
assert np.isfinite(objs).all(), "objectives 含 NaN/inf"
assert np.isreal(objs).all(), "objectives 含复数"
assert np.isfinite(hv) and hv > 0, f"HV 异常: {hv}"
assert feas["sum_min"] >= 99.0 and feas["sum_max"] <= 101.0, f"feasibility 异常: {feas}"

print(f"[OK] 前沿解数 = {len(front)}")
print(f"[OK] Σc: min={sums.min():.10f}, max={sums.max():.10f}, "
      f"mean={sums.mean():.10f}, max|dev|={dev.max():.2e}")
print(f"[OK] feasibility 字段 = {feas}")
print(f"[OK] objectives 有限实数, ΔH∈[{objs[:,0].min():.3f},{objs[:,0].max():.3f}], "
      f"ρ∈[{objs[:,1].min():.3f},{objs[:,1].max():.3f}], "
      f"cost∈[{objs[:,2].min():.3f},{objs[:,2].max():.3f}]")
print(f"[OK] HV = {hv:.6f} (有限正数)")
print("=== 第 3 批 NSGA-II 可行性修复验证全部通过 ===")
