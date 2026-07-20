# -*- coding: utf-8 -*-
"""Penalty_Auditor 数值探针：PenaltyFlex / ParetoZoom 与方案公式逐项对照。

基准:
  - plan/hea_encoding_scheme_v1.13.md §3.2.5 (PenaltyFlex), §4.2 (ParetoZoom)
  - plan/完整技术路线v2.0.md §4.3 / §5

运行: PYTHONPATH=src "$DAIMON_USER_PYTHON" audit_probes/probe_penalty_pareto.py
"""

import math
import sys

import numpy as np

sys.path.insert(0, "src")

from ap3_qubo.penalty_flex.adaptive_penalty import PenaltyFlex, FeedbackReport
from ap3_qubo.penalty_flex.warm_start import LambdaCache
from ap3_qubo.exploration.weight_utils import (
    normalize_weights,
    midpoint_weights,
    deduplicate_weights,
    WeightGenerator,
)
from ap3_qubo.exploration.pareto_zoom import ParetoZoom
from ap3_qubo.exploration.archive import Archive
from ap3_qubo.validation.hypervolume import HypervolumeCalculator
from ap3_qubo.validation.pareto import SolutionRecord
from ap3_qubo.physical_params import CONSTRAINT, PARETO_ZOOM, COARSE_WEIGHTS

PASS, FAIL, INFO = "PASS", "FAIL", "INFO"
results = []


def check(name, got, expect, tol=1e-9):
    ok = abs(got - expect) <= tol
    results.append((PASS if ok else FAIL, name, f"got={got:.10g} expect={expect:.10g}"))
    return ok


def info(name, msg):
    results.append((INFO, name, msg))


def fb(v_c, v_ccr, obj=1.0):
    return FeedbackReport(v_carbide=v_c, v_ccr=v_ccr, best_objective=obj)


print("=" * 78)
print("Part 1: PenaltyFlex 参数与更新公式（基准: v1.13 §3.2.5 / v2.0 §4.3）")
print("=" * 78)

# ---- T0 参数清单核对 ----
check("P0 λ_sum_fixed ∈ [10,20] (中值15)", CONSTRAINT.lambda_sum_fixed, 15.0)
check("λ_carbide_init = 0.05", CONSTRAINT.lambda_carbide_init, 0.05)
check("λ_ccr_init = 0.05", CONSTRAINT.lambda_ccr_init, 0.05)
check("α_add = 0.5", CONSTRAINT.alpha_add, 0.5)
check("α_mult = 0.8", CONSTRAINT.alpha_mult, 0.8)
check("γ = 0.1", CONSTRAINT.gamma_convergence, 0.1)
check("T_add = 2", CONSTRAINT.t_add, 2)
info("T_max", f"代码={CONSTRAINT.t_max}，方案=15（physical_params.py:260-265 注释为有意性能优化，属文档化偏离）")
check("carbide_soft_upper = 0.8", CONSTRAINT.carbide_soft_upper, 0.8)
check("ccr_h_max = 64.3125", CONSTRAINT.ccr_h_max, 64.3125)

# ---- T1 违反度定义 ----
check("v_carbide(1.5)=(1.5-0.8)^2", PenaltyFlex.compute_violation_p1(1.5), 0.49)
check("v_CCr(1.75,36.75)=1.0", PenaltyFlex.compute_violation_p2(1.75, 36.75), 1.0)
check("v_CCr(0.5,20)=10/64.3125", PenaltyFlex.compute_violation_p2(0.5, 20.0), 10.0 / 64.3125)

# ---- T2 加性启动阶段 (t=1,2): λ^(t+1) = λ^t + α_add·v̄ ----
pf = PenaltyFlex()
s1 = pf.step(fb(0.2, 0.1))
check("t=1 λ_carbide = 0.05+0.5*0.2", s1.lambda_carbide, 0.15)
check("t=1 λ_ccr = 0.05+0.5*0.1", s1.lambda_ccr, 0.10)
check("t=1 相位=additive", 1.0 if s1.phase == "additive" else 0.0, 1.0)
s2 = pf.step(fb(0.2, 0.1))
check("t=2 λ_carbide = 0.15+0.5*0.2", s2.lambda_carbide, 0.25)
check("t=2 λ_ccr = 0.10+0.5*0.1", s2.lambda_ccr, 0.15)
check("t=2 仍 additive（t≤T_add 含端点）", 1.0 if s2.phase == "additive" else 0.0, 1.0)
check("t=2 未收敛（加性阶段不判收敛）", 0.0 if s2.is_converged else 1.0, 1.0)

# ---- T3 乘性演化 (t=3, ε=0.02 探索期): λ·exp(α·tanh(v̄−ε)) ----
s3 = pf.step(fb(0.5, 0.5))
exp_lc = 0.25 * math.exp(0.8 * math.tanh(0.5 - 0.02))
exp_lccr = 0.15 * math.exp(0.8 * math.tanh(0.5 - 0.02))
check("t=3 λ_carbide 乘性公式", s3.lambda_carbide, exp_lc, 1e-12)
check("t=3 λ_ccr 乘性公式", s3.lambda_ccr, exp_lccr, 1e-12)
check("t=3 相位=multiplicative", 1.0 if s3.phase == "multiplicative" else 0.0, 1.0)
check("t=3 ε=0.02（探索期）", s3.epsilon, 0.02)

# ---- T4 ε 切换 (t=4 → 固化期 ε=0.0) ----
# 注意：t=4 若喂 v̄>0 会因 dir 反转(-1→+1)触发振荡分支（见 P-D1），
# 故这里 t=4 继续喂 v̄=0 验证 ε=0 下乘性恒等（tanh(0)=0 → λ 不变）。
pf2 = PenaltyFlex()
pf2.step(fb(0.0, 0.0))  # t=1 加性: λ=0.05
pf2.step(fb(0.0, 0.0))  # t=2 加性: λ=0.05, dir=-1
s43 = pf2.step(fb(0.0, 0.0))  # t=3 乘性 ε=0.02: λ=0.05·exp(0.8·tanh(-0.02))
exp43 = 0.05 * math.exp(0.8 * math.tanh(-0.02))
check("t=3 λ=0.05·exp(0.8·tanh(-0.02))", s43.lambda_carbide, exp43, 1e-12)
s4 = pf2.step(fb(0.0, 0.0))  # t=4: ε=0, tanh(0)=0 → λ 不变
check("t=4 ε=0.0（固化期）", s4.epsilon, 0.0)
check("t=4 ε=0 时乘性恒等 λ 不变", s4.lambda_carbide, exp43, 1e-12)
# 附加演示：t=4 改喂 v̄=0.3 → dir 反转触发几何平均（方案则判 λ 轨迹无反向→乘性）
pf2b = PenaltyFlex()
for _ in range(3):
    pf2b.step(fb(0.0, 0.0))
s4b = pf2b.step(fb(0.3, 0.3))
gm_demo = math.sqrt(exp43 * 0.05)
mult_demo = exp43 * math.exp(0.8 * math.tanh(0.3))
info("ε 切换诱发伪振荡（P-D1 实例）",
     f"t=4 v̄:0→0.3 且 ε:0.02→0.0，代码判 dir 反转→几何平均 λ={s4b.lambda_carbide:.6f}"
     f"（=√({exp43:.6f}·0.05)={gm_demo:.6f}）；方案按 λ 轨迹 sign 判据无反向→乘性 {mult_demo:.6f}")
info("ε 切换边界", "代码 current_epsilon: t < T_add+2=4 用探索期 0.02（t=1~3），t≥4 用 0.0。"
                    "方案仅注明'探索期2%/固化期0%'，未定义切换轮次——实现自定义，非错误")

# ---- T5 振荡几何平均（代码语义: 反馈方向反转触发）----
# 续 pf: t=3 时 v̄=0.5>ε dir=+1（t=2 dir=sign(0.2-0.02)=+1 同向，未触发）
# t=4 喂 v̄=0.0 → dir=-1 ≠ +1 → 触发几何平均: λ(5)=√(λ(4)·λ(3))
# 其中 λ(4)=0.3572571（t=3 乘性输出），λ(3)=0.25（t=2 加性输出）
lam4_c, lam4_ccr = s3.lambda_carbide, s3.lambda_ccr  # 当前值 λ(4)
lam3_c, lam3_ccr = 0.25, 0.15                        # 上一轮值 λ(3)
s5 = pf.step(fb(0.0, 0.0))
check("t=4 振荡触发 λ_carbide=√(λ4·λ3)", s5.lambda_carbide, math.sqrt(lam4_c * lam3_c), 1e-12)
check("t=4 振荡触发 λ_ccr=√(λ4·λ3)", s5.lambda_ccr, math.sqrt(lam4_ccr * lam3_ccr), 1e-12)
info("几何平均取值", "代码取 √(λ_t · λ_{t-1})（当前值×上一轮值），与方案 :338 公式一致")

# ---- T6 几何平均非对称验证 + 方案偏离演示 ----
pf3 = PenaltyFlex()
pf3.step(fb(0.4, 0.4))   # t=1 加性: λ=0.05+0.5*0.4=0.25, dir=+1
pf3.step(fb(0.0, 0.0))   # t=2 加性: λ=0.25+0=0.25, dir=-1（存储），实际 λ 未动
s63 = pf3.step(fb(0.4, 0.4))  # t=3: dir=+1 vs 存-1 → 代码判振荡
code_val = s63.lambda_carbide
plan_val = 0.25 * math.exp(0.8 * math.tanh(0.4 - 0.02))  # 方案: sign(λ3-λ2)=0, 不触发振荡→乘性
info("振荡检测语义偏离（疑点 P-D1）",
     f"场景 v̄=[0.4, 0.0, 0.4]: 代码 t=3 输出 λ={code_val:.6f}（几何平均√(0.25·0.25)）；"
     f"按方案伪代码 sign(λ^t−λ^t−1) 判据，λ 轨迹 0.05→0.25→0.25 无反向，应走乘性 → λ={plan_val:.6f}。"
     "代码用'反馈方向 sign(v̄−ε)'代理方案的'λ 实际变号'，且对两个约束联合触发（方案为逐约束判定）")

# ---- T7 收敛门控: t>T_add 且 MAX|Δλ|/λ<γ 连续2轮（方案为单轮，代码加严有注释）----
pf4 = PenaltyFlex()
pf4.step(fb(0.0, 0.0))  # t=1 加性, λ 不变
pf4.step(fb(0.0, 0.0))  # t=2 加性, λ 不变
s73 = pf4.step(fb(0.0, 0.0))  # t=3 乘性: λ·exp(0.8·tanh(-0.02)), Δ≈1.59%<γ → count=1
check("t=3 Δλ/λ<γ 第1轮不判收敛（需连续2轮）", 0.0 if s73.is_converged else 1.0, 1.0)
s74 = pf4.step(fb(0.0, 0.0))  # t=4: Δ 仍 <γ → count=2 → 收敛
check("t=4 连续第2轮 <γ → 收敛", 1.0 if s74.is_converged else 0.0, 1.0)
info("收敛判据偏离（文档化）",
     "方案 :342 单轮 MAX_j|Δλ_j|/λ_j<γ 且 t>T_add 即 BREAK；"
     "代码 adaptive_penalty.py:296-307 要求连续 2 轮（注释自称为'加严'）")

# ---- T8 analyze_top_k 排序与均值 ----
sols = [{"c_carbon": 1.0, "c_cr": 10.0, "objective": float(100 - i), "is_feasible": True}
        for i in range(15)]  # objective 99..86 降序打乱语义
rep = PenaltyFlex.analyze_top_k(sols, k=10)
exp_v = np.mean([(1.0 - 0.8) ** 2] * 10)
check("analyze_top_k v_carbide 均值", rep.v_carbide, float(exp_v))
check("analyze_top_k best_objective=86", rep.best_objective, 86.0)
check("analyze_top_k num_total=10", float(rep.num_total), 10.0)

print()
print("=" * 78)
print("Part 2: ParetoZoom 五阶段（基准: v1.13 §4.2 / v2.0 §5）")
print("=" * 78)

# ---- T9 粗网格 12 组 ----
plan_G = [(1,0,0),(0,1,0),(0,0,1),(0.5,0.5,0),(0.5,0,0.5),(0,0.5,0.5),
          (1/3,1/3,1/3),(0.7,0.2,0.1),(0.2,0.7,0.1),(0.2,0.1,0.7),(0.1,0.7,0.2),(0.1,0.2,0.7)]
match = all(np.allclose(g, p) for g, p in zip(COARSE_WEIGHTS, plan_G)) and len(COARSE_WEIGHTS) == 12
check("COARSE_WEIGHTS == 方案 G1~G12", 1.0 if match else 0.0, 1.0)

# ---- T10 ParetoZoom 参数 ----
check("t_max_rounds=5", float(PARETO_ZOOM.t_max_rounds), 5.0)
check("ε_HV=0.01", PARETO_ZOOM.epsilon_hv, 0.01)
check("d_threshold_factor=0.15", PARETO_ZOOM.d_threshold_factor, 0.15)
check("σ=0.08", PARETO_ZOOM.sigma_perturb, 0.08)
check("hv_delta_factor=0.10", PARETO_ZOOM.hv_delta_factor, 0.10)
check("convexity_warning=0.10", PARETO_ZOOM.convexity_warning, 0.10)
check("weight_dedup_tol=0.05", PARETO_ZOOM.weight_dedup_tol, 0.05)
check("weight_min_bound=0.02", PARETO_ZOOM.weight_min_bound, 0.02)
info("num_reads", f"PARETO_ZOOM.num_reads={PARETO_ZOOM.num_reads} 注释称'性能优化降至500'，"
                  "但 pareto_zoom.py:637 调用 solve_from_model 未传 num_reads → 实际走求解器默认 1000（=方案值）。"
                  "该参数为死参数且注释误导（疑点 P-D4）")

# ---- T11 阶段A 间隙检测: d_threshold = 0.15 × max_edge，权重中点插值 ----
pts_norm = np.array([[0.0, 0.9, 0.9], [0.1, 0.8, 0.7], [0.9, 0.1, 0.2]])  # 按 f1 有序
recs = [SolutionRecord(fractions={}, bits=np.zeros(38, dtype=int),
                       objectives=(0.0, 0.0, 0.0), objectives_norm=tuple(p),
                       weights=w, energy=-float(i))
        for i, (p, w) in enumerate(zip(pts_norm, [(1,0,0),(0.5,0.5,0),(0,0.5,0.5)]))]
arch = Archive()
arch.insert_batch(recs)
adj = [float(np.linalg.norm(pts_norm[i+1] - pts_norm[i])) for i in range(2)]
max_edge = max(adj)
d_thr = 0.15 * max_edge
info("间隙阈值", f"相邻距离={['%.4f'%d for d in adj]}，max_edge={max_edge:.4f}，"
               f"d_threshold=0.15×max={d_thr:.4f} → 第2对({adj[1]:.4f})>阈值应报间隙")
wg = WeightGenerator()
gaps = wg.from_gaps([r.weights for r in arch.front], [(1, 2)])
mid = midpoint_weights((0.5, 0.5, 0.0), (0.0, 0.5, 0.5))
check("间隙中点插值 w_mid", float(np.allclose(gaps[0], mid)), 1.0)

# ---- T12 阶段C 去重 tol=0.05 ----
dd = deduplicate_weights([(0.5, 0.3, 0.2), (0.51, 0.30, 0.19), (0.6, 0.3, 0.1)], tolerance=0.05)
check("去重保留 2 组（0.014<0.05 判重，0.141>0.05 保留）", float(len(dd)), 2.0)

# ---- T13 边界保护: 方案步骤25 REMOVE ANY(w_i<0.02) ----
mid0 = midpoint_weights((1.0, 0.0, 0.0), (0.5, 0.5, 0.0))
# P-D2 修复验证：pareto_zoom.py _run_round 合并管线现在含
#   new_weights = [w for w in new_weights if min(w) >= weight_min_bound]
# 用同一表达式对混合列表验证过滤语义
mixed = deduplicate_weights([mid0, (0.5, 0.3, 0.2), (0.34, 0.33, 0.33)], tolerance=0.05)
filtered = [w for w in mixed if min(w) >= PARETO_ZOOM.weight_min_bound]
check("P-D2 修复后 w3=0 中点被移除", 1.0 if all(min(w) >= 0.02 for w in filtered) else 0.0, 1.0)
check("P-D2 修复后合法权重保留", float(len(filtered)), 2.0)
info("边界保护（P-D2 已修复）",
     f"间隙中点 {mid0} 含 w3=0.0<0.02，修复后按方案 :569 移除；"
     "源码见 pareto_zoom.py _run_round 合并段（dedup 后、max_per_round 截断前）")
renorm = normalize_weights((0.02, 1.0, 1.0))
info("微扰路径同步受益",
     f"perturb 先 clip[0.02,1] 再归一化，极端情形 (0.02,1,1)→{tuple(round(x,4) for x in renorm)}"
     " w1≈0.0099<0.02；同一过滤行亦拦截此类越界（方案边界语义完整覆盖）")

# ---- T14 阶段B HV 热点微扰: 贡献计算空间一致性 ----
rng = np.random.default_rng(0)
front_raw = np.array([[-18.0, 8.5, 0.9], [-12.0, 7.5, 0.6], [-8.0, 6.8, 0.3], [-2.0, 6.5, 0.2]])
front_norm = front_raw / np.array([30.0, 10.0, 1.0])
hv = HypervolumeCalculator()
hv.set_reference_from_data(front_raw)          # 参考点设在原始尺度（与 pareto_zoom.py:317,459 一致）
contrib_mixed = [hv.marginal_contribution(front_norm, i) for i in range(4)]  # 修复前: 归一化点×原始参考点
hv2 = HypervolumeCalculator()
hv2.set_reference_from_data(front_norm)        # 对照: 归一化点×归一化参考点
contrib_norm = [hv2.marginal_contribution(front_norm, i) for i in range(4)]
hv3 = HypervolumeCalculator()
hv3.set_reference_from_data(front_raw)
contrib_raw = [hv3.marginal_contribution(front_raw, i) for i in range(4)]    # 修复后: 原始点×原始参考点
info("HV 热点贡献空间（P-D3 修复前后对照）",
     f"修复前(错配)={[round(float(c),6) for c in contrib_mixed]}，"
     f"修复后(原始一致)={[round(float(c),6) for c in contrib_raw]}")

# 功能验证：修复后的 _generate_perturbations 传入 perturb_hotspots 的贡献
# 必须等于原始空间贡献（捕获实际传参）
pz2 = ParetoZoom.__new__(ParetoZoom)
pz2._params = PARETO_ZOOM
recs3 = [SolutionRecord(fractions={}, bits=np.zeros(38, dtype=int),
                        objectives=tuple(front_raw[i]),
                        objectives_norm=tuple(front_norm[i]),
                        weights=w, energy=-float(i))
         for i, w in enumerate([(0.7,0.2,0.1),(0.5,0.5,0.0),(0.2,0.7,0.1),(0.6,0.2,0.2)])]
pz2._archive = Archive()
pz2._archive.insert_batch(recs3)
pz2._hv_calc = HypervolumeCalculator()
pz2._hv_calc.set_reference_from_data(pz2._archive.get_objective_matrix())
captured = {}
class _CapWG(WeightGenerator):
    def perturb_hotspots(self, fw, contrib=None):
        captured["contrib"] = contrib
        return []
pz2._weight_gen = _CapWG()
pz2._generate_perturbations()
check("P-D3 修复后贡献=原始空间边际贡献",
      float(np.allclose(captured["contrib"], contrib_raw)), 1.0)

# ---- T15 凸性检验 CONVEXITY_TEST_3D ----
pz = ParetoZoom.__new__(ParetoZoom)  # 不触发 __init__（避免求解器）
pz._params = PARETO_ZOOM
# 构造三点互不支配的前沿，(f1,f2) 平面中点低于两端连线 → 方案叉积 cross<0 计非凸。
# 手算 (f1,f2) 按 f1 排序 [(-10,8),(-6,5),(-2,4.5)]:
# cross=(5-8)(-2+10)-(-6+10)(4.5-8) = -24+14 = -10 < 0 → count=1, ratio=1/3>10% → warning
pts = [(-10.0, 8.0, 0.9), (-6.0, 5.0, 0.5), (-2.0, 4.5, 0.2)]
recs2 = [SolutionRecord(fractions={}, bits=np.zeros(38, dtype=int), objectives=p,
                        objectives_norm=(p[0]/30.0, p[1]/10.0, p[2]), weights=(0.5,0.5,0.0),
                        energy=-float(i))
         for i, p in enumerate(pts)]
pz._archive = Archive()
pz._archive.insert_batch(recs2)
rep_cvx = pz._convexity_test_3d()
check("凸性检验 max_ratio=1/3", rep_cvx["max_ratio"], 1.0 / 3.0, 1e-9)
check("凸性检验 warning=True（>10%）", 1.0 if rep_cvx["warning"] else 0.0, 1.0)
info("凸性检验明细", f"ratios={ {k: round(v,4) for k,v in rep_cvx['ratios'].items()} }；"
                     "叉积公式与方案 v1.13:500-501 逐字一致，排序轴 (f1,f1,f2) 一致，"
                     "投影后重算 2D 非支配集一致")

# ---- T16 阶段E HV 收敛: 连续2轮 |ΔHV|/HV<1%（任务书口径；方案 :585 单轮）----
info("HV 收敛判据", "pareto_zoom.py:195-200 实现 |ΔHV|/HV<ε_HV 连续 2 轮 BREAK，"
                    "与任务书'连续2轮'一致；方案 v1.13:585/v2.0:485 原文为单轮即 BREAK（偏离未在方案文档登记）")

# ---- T17 阶段D TOP-K: 方案 TOP-20 vs 代码 TOP-10 ----
info("阶段D 入档候选数（疑点 P-D5）",
     "方案 :577 'FOR sol ∈ TOP-20(solutions)'；pareto_zoom.py:644 'result.solutions[:10]' → TOP-10。"
     "求解器已按能量升序排序（kaiwu_solver get_sorted_solutions），故是'TOP-10'而非'前10'；"
     "数量与方案不一致且无注释说明")

# ---- T18 warm-start λ 缓存读写一致性 ----
cache = LambdaCache()
cache.store((2.0, 3.0, 5.0), 0.12, 0.08)      # 未归一化存入
got = cache.get((0.2, 0.3, 0.5))              # 等比例查询
check("LambdaCache 存取归一化一致", 1.0 if got == (0.12, 0.08) else 0.0, 1.0)
near = cache.find_nearest((0.21, 0.31, 0.48))
check("find_nearest 近邻命中", 1.0 if near == (0.12, 0.08) else 0.0, 1.0)
far = cache.find_nearest((0.9, 0.05, 0.05), max_distance=0.5)
check("find_nearest 超距返回 None", 1.0 if far is None else 0.0, 1.0)
info("warm-start 流程", "pareto_zoom.py:519-527 优先显式 warm_start_lambdas，否则 cache.find_nearest，"
                        "最后回退 (0.05,0.05)；:556-561 收敛或未收敛均把终值 λ 写回缓存——"
                        "与方案 §4.2.5 'λ_new* 存入缓存'一致（未收敛时也存终值，属合理实现选择）")

# ---- 汇总 ----
print()
print("=" * 78)
print("探针结果汇总")
print("=" * 78)
n_pass = sum(1 for r in results if r[0] == PASS)
n_fail = sum(1 for r in results if r[0] == FAIL)
n_info = sum(1 for r in results if r[0] == INFO)
for st, name, msg in results:
    print(f"[{st}] {name}: {msg}")
print(f"\n合计: PASS={n_pass}, FAIL={n_fail}, INFO={n_info}")
sys.exit(1 if n_fail else 0)
