"""第 2 批修复验证脚本（Python 3.10 + kaiwu + deap 环境运行）。

验证项：
1. 真实 NSGA-II（DEAP）小规模通路：pop=20, gen=5，结果字典含 algorithm 标注；
2. deap 缺失时显式 ImportError（通过屏蔽 deap 导入模拟）；
3. set_unified_reference 合并多方法解集统一定参考点；
4. set_reference_from_data 零 range 相对兜底（不再出现 1.0 绝对值）。
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import numpy as np

# --- 3. set_unified_reference ---
from ap3_qubo.validation.hypervolume import (
    HypervolumeCalculator,
    set_unified_reference,
)

rng = np.random.default_rng(0)
a = rng.uniform(low=[-15, 6.5, 50], high=[-8, 7.5, 120], size=(10, 3))
b = rng.uniform(low=[-12, 6.8, 80], high=[-5, 8.0, 200], size=(12, 3))
ref = set_unified_reference({"m1": a, "m2": b}, margin=0.10)
merged = np.vstack([a, b])
expected = np.max(merged, axis=0) + 0.10 * (np.max(merged, axis=0) - np.min(merged, axis=0))
assert np.allclose(ref, expected), f"统一参考点不符: {ref} vs {expected}"
calc = HypervolumeCalculator(reference_point=ref)
hv_a, hv_b = calc.compute(a), calc.compute(b)
assert hv_a > 0 and hv_b > 0
print(f"[OK] set_unified_reference ref={np.round(ref, 3)}, HV(m1)={hv_a:.3f}, HV(m2)={hv_b:.3f}")

try:
    set_unified_reference({"x": np.zeros((0, 3))})
    raise AssertionError("空解集应抛 ValueError")
except ValueError:
    print("[OK] 全空解集显式 ValueError")

# --- 4. 零 range 相对兜底 ---
c = np.array([[-10.0, 7.0, 100.0], [-8.0, 7.0, 150.0]])  # 密度零 range
calc2 = HypervolumeCalculator()
calc2.set_reference_from_data(c, margin_factor=0.10)
# 旧逻辑兜底 1.0 绝对值 → ref[1]=8.0；新逻辑为 |nadir|*0.1=0.7 → ref[1]=7.7
assert abs(calc2.reference_point[1] - 7.7) < 1e-9, calc2.reference_point
print(f"[OK] 零 range 相对兜底 ref={np.round(calc2.reference_point, 3)}")

# --- 1. 真实 NSGA-II 小规模 ---
from ap3_qubo.experiments.nsga2_baseline import NSGA2Optimizer

np.random.seed(42)
opt = NSGA2Optimizer(pop_size=20, generations=5)
result = opt.optimize_and_evaluate()
assert isinstance(result, dict) and "algorithm" in result
assert "DEAP" in result["algorithm"]
front, hv, objs = result["front"], result["hv"], result["objectives"]
assert len(front) > 0 and hv > 0 and objs.shape == (len(front), 3)
sums = [sum(f.values()) for f in front]
print(f"[OK] 真 NSGA-II: algorithm={result['algorithm']}")
print(f"     front_size={len(front)}, hv={hv:.4f}, 成分和范围=[{min(sums):.1f}, {max(sums):.1f}]")

# --- 2. deap 缺失 → 显式 ImportError ---
import builtins
real_import = builtins.__import__
def no_deap(name, *args, **kwargs):
    if name == "deap" or name.startswith("deap."):
        raise ImportError("No module named 'deap'")
    return real_import(name, *args, **kwargs)

builtins.__import__ = no_deap
try:
    NSGA2Optimizer(pop_size=4, generations=1).optimize()
    raise AssertionError("deap 缺失时应抛 ImportError")
except ImportError as e:
    assert "pip install deap" in str(e)
    print(f"[OK] deap 缺失显式报错: {str(e)[:60]}...")
finally:
    builtins.__import__ = real_import

print("\nALL CHECKS PASSED")
