# -*- coding: utf-8 -*-
"""端到端冒烟测试（快速版）：模拟器跑通完整 AP3-QUBO pipeline。
3 组权重 + num_reads=150 + t_max_rounds=1，验证全链路可用性。
运行: Python310 python smoke_simulator_e2e.py
"""
import sys, time, dataclasses
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ap3_qubo.exploration.pareto_zoom import ParetoZoom
from ap3_qubo.solver.kaiwu_solver import KaiwuSolver


class FastSolver(KaiwuSolver):
    """冒烟测试专用：压缩采样规模以缩短耗时。"""

    def solve_from_model(self, model, n_vars, num_reads=1000, top_k=None, **kw):
        return super().solve_from_model(model, n_vars=n_vars, num_reads=150, top_k=20, **kw)


t0 = time.time()
weights = [(1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (1/3, 1/3, 1/3)]

solver = FastSolver(mode="simulator")
print(f"[init] solver available(simulator) = {solver.is_available('simulator')}", flush=True)

pz = ParetoZoom(initial_weights=weights, solver=solver)
pz._params = dataclasses.replace(pz._params, t_max_rounds=1)  # 冒烟只跑 1 轮加密

archive, rounds = pz.run()

front = archive.front
metrics = pz.get_front_metrics()

print("=" * 60)
print(f"[done] 耗时 {time.time()-t0:.1f}s | 探索轮数 = {len(rounds)}")
print(f"[archive] 入档记录数 = {len(archive)}")
print(f"[front] 非支配解数 = {len(front)}")
print(f"[metrics] {metrics}")

assert len(archive) > 0, "FAIL: archive 为空"
assert len(front) > 0, "FAIL: 非支配前沿为空"

comps = pz.get_front_compositions()
print("-" * 60)
for i, (rec, comp) in enumerate(zip(front[:5], comps[:5])):
    f = rec.objectives
    total = sum(comp.values())
    print(f"#{i} f1={f[0]:.3f} f2={f[1]:.3f} f3={f[2]:.1f} | "
          f"Al={comp['Al']:.2f} Co={comp['Co']:.2f} Cr={comp['Cr']:.2f} "
          f"Fe={comp['Fe']:.2f} Ni={comp['Ni']:.2f} C={comp['C']:.2f} | sum={total:.2f}")
print("=" * 60)
print("SMOKE_TEST_PASS")
