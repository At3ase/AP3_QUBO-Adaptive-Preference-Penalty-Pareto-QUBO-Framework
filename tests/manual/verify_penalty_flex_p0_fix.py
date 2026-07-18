"""
PenaltyFlex P0-4 + P1-1/2/3 修复的纯 numpy 模拟验证。

不依赖 kaiwu / scipy：直接以违反度序列驱动 PenaltyFlex.step()，
对照手算公式断言：
  A. 加性启动阶段（t ≤ T_add=2）λ += α_add·v_bar，单调不降且数值正确
  B. 收敛判定带 t > T_add 门控（加性阶段 v_bar=0 不得误收敛）
  C. 振荡时几何平均 √(λ_t · λ_{t-1})，与手算值逐位对照
  D. analyze_top_k 先按 objective 升序排序再取前 k 个
  E. 加性阶段与 _clamp 上界 [0.005, 5.0] 兼容（单调性不被破坏）

运行: python verify_penalty_flex_p0_fix.py
"""

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from ap3_qubo.penalty_flex.adaptive_penalty import (  # noqa: E402
    FeedbackReport,
    PenaltyFlex,
)

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, ok, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}  {detail}")


def fb(v_c: float, v_ccr: float) -> FeedbackReport:
    return FeedbackReport(v_carbide=v_c, v_ccr=v_ccr, best_objective=0.0)


# =============================================================================
# A. 加性启动：λ += α_add·v_bar（α_add=0.5），t=1,2 只升不降
# =============================================================================
print("== A. 加性启动阶段（方案 E-04 分支①） ==")
pf = PenaltyFlex()
lam_c, lam_ccr = [0.05], [0.05]
# 手算：λ_c: 0.05 → 0.05+0.5*0.3=0.20 → 0.20+0.5*0.3=0.35
#       λ_ccr: 0.05 → 0.05+0.5*0.2=0.15 → 0.15+0.5*0.2=0.25
expected_c = [0.20, 0.35]
expected_ccr = [0.15, 0.25]
phases = []
for t in range(2):
    st = pf.step(fb(0.3, 0.2))
    lam_c.append(st.lambda_carbide)
    lam_ccr.append(st.lambda_ccr)
    phases.append(st.phase)

check("A1 t=1 λ_carbide = 0.05+0.5×0.3 = 0.20",
      np.isclose(lam_c[1], expected_c[0]), f"got {lam_c[1]:.6f}")
check("A2 t=2 λ_carbide = 0.20+0.5×0.3 = 0.35",
      np.isclose(lam_c[2], expected_c[1]), f"got {lam_c[2]:.6f}")
check("A3 t=1,2 λ_ccr = 0.15, 0.25",
      np.isclose(lam_ccr[1], expected_ccr[0]) and np.isclose(lam_ccr[2], expected_ccr[1]),
      f"got {lam_ccr[1]:.6f}, {lam_ccr[2]:.6f}")
check("A4 加性阶段单调不降",
      lam_c[1] >= lam_c[0] and lam_c[2] >= lam_c[1]
      and lam_ccr[1] >= lam_ccr[0] and lam_ccr[2] >= lam_ccr[1])
check("A5 t=1,2 phase 均为 additive", phases == ["additive", "additive"], f"got {phases}")

# t=3 进入乘性阶段（无振荡，同向）：λ = 0.35·exp(0.8·tanh(0.3−0.02))
st = pf.step(fb(0.3, 0.2))
hand = 0.35 * math.exp(0.8 * math.tanh(0.3 - 0.02))
check("A6 t=3 乘性分支数值 = 0.35·exp(0.8·tanh(0.28))",
      np.isclose(st.lambda_carbide, hand), f"got {st.lambda_carbide:.8f}, hand {hand:.8f}")
check("A7 t=3 phase = multiplicative", st.phase == "multiplicative", f"got {st.phase}")

# =============================================================================
# B. 收敛门控：t ≤ T_add 时 v_bar=0 不得收敛（方案 E-07）
# =============================================================================
print("== B. 收敛判定 t > T_add 门控（方案 E-07） ==")
pf = PenaltyFlex()
conv_flags = []
for t in range(4):
    st = pf.step(fb(0.0, 0.0))
    conv_flags.append(st.is_converged)

check("B1 t=1 不收敛（加性阶段 Δλ=0 也不判）", not conv_flags[0])
check("B2 t=2 不收敛（旧逻辑此处会误判收敛）", not conv_flags[1])
check("B3 t=3 不收敛（连续两轮加严：仅第 1 轮 <γ）", not conv_flags[2])
check("B4 t=4 收敛（t>T_add 且连续两轮 <γ）", conv_flags[3],
      "乘性 v_bar=0 时 Δλ/λ≈1.59% < γ=0.1")

# =============================================================================
# C. 振荡几何平均：√(λ_t · λ_{t-1}) 手算对照（方案 E-04 分支②）
# =============================================================================
print("== C. 振荡几何平均操作数（方案 E-04 分支②） ==")
pf = PenaltyFlex()
# t=1,2 加性：λ_c: 0.05→0.10→0.15；λ_ccr: 0.05→0.15→0.25
pf.step(fb(0.10, 0.20))
pf.step(fb(0.10, 0.20))  # t=2 记录方向 (+1,+1)
# t=3：v_c=0.0 → dir_c 翻转为 -1 → 触发振荡；v_ccr=0.20 → 同向
st = pf.step(fb(0.0, 0.20))

hand_c = math.sqrt(0.15 * 0.10)     # √(λ_c^(2)·λ_c^(1)) = √0.015
hand_ccr = math.sqrt(0.25 * 0.15)   # √(λ_ccr^(2)·λ_ccr^(1)) = √0.0375
check("C1 振荡时 λ_carbide = √(0.15×0.10) = √0.015",
      np.isclose(st.lambda_carbide, hand_c),
      f"got {st.lambda_carbide:.10f}, hand {hand_c:.10f}")
check("C2 振荡时 λ_ccr = √(0.25×0.15) = √0.0375",
      np.isclose(st.lambda_ccr, hand_ccr),
      f"got {st.lambda_ccr:.10f}, hand {hand_ccr:.10f}")
# 排除旧错误公式 √(λ_t·λ_乘性候选)：旧式会给出 √[0.15·0.15·exp(0.8·tanh(-0.02))]
wrong = math.sqrt(0.15 * 0.15 * math.exp(0.8 * math.tanh(0.0 - 0.02)))
check("C3 结果不等于旧错误公式 √(λ_t·λ_乘性候选)",
      not np.isclose(st.lambda_carbide, wrong),
      f"old-wrong would give {wrong:.10f}")

# =============================================================================
# D. analyze_top_k：按 objective 升序排序后取前 k 个
# =============================================================================
print("== D. analyze_top_k 排序语义 ==")
rng = np.random.default_rng(42)
objs = rng.permutation(np.arange(15, dtype=float))  # 打乱顺序的 0..14
sols = [
    {
        "c_carbon": 0.8 + 0.01 * i,  # 违反度与 objective 无关联，检验选择依据
        "c_cr": 10.0,
        "objective": float(o),
        "is_feasible": True,
    }
    for i, o in enumerate(objs)
]
rep = PenaltyFlex.analyze_top_k(sols, k=10)
# 期望：objective 最小的 10 个，即 objective ∈ {0..9}
expected_mask = objs < 10
expected_v = float(np.mean([(s["c_carbon"] - 0.8) ** 2
                            for s, m in zip(sols, expected_mask) if m]))
check("D1 num_total = 10", rep.num_total == 10, f"got {rep.num_total}")
check("D2 best_objective = 全局最小 0.0",
      np.isclose(rep.best_objective, 0.0), f"got {rep.best_objective}")
check("D3 v_carbide 为按目标值排序后前 10 个的均值",
      np.isclose(rep.v_carbide, expected_v),
      f"got {rep.v_carbide:.8f}, expected {expected_v:.8f}")
head10_v = float(np.mean([(s["c_carbon"] - 0.8) ** 2 for s in sols[:10]]))
check("D4 结果 ≠ 未排序的列表头 10 个均值",
      not np.isclose(rep.v_carbide, head10_v),
      f"head-10 mean {head10_v:.8f}")

# =============================================================================
# E. 加性阶段与 _clamp 兼容（上界 5.0 不破坏单调性）
# =============================================================================
print("== E. 加性阶段 _clamp 兼容 ==")
pf = PenaltyFlex()
s1 = pf.step(fb(100.0, 100.0))  # 0.05 + 50 → clamp 5.0
s2 = pf.step(fb(100.0, 100.0))  # 仍 clamp 5.0
check("E1 大违反度下 λ 被钳到上界 5.0",
      np.isclose(s1.lambda_carbide, 5.0) and np.isclose(s1.lambda_ccr, 5.0),
      f"got {s1.lambda_carbide}, {s1.lambda_ccr}")
check("E2 clamp 后仍单调不降", s2.lambda_carbide >= s1.lambda_carbide)

# =============================================================================
print("=" * 60)
n_fail = sum(1 for _, ok, _ in RESULTS if not ok)
print(f"汇总: {len(RESULTS) - n_fail}/{len(RESULTS)} 通过, {n_fail} 失败")
sys.exit(1 if n_fail else 0)
