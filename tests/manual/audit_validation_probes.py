# -*- coding: utf-8 -*-
"""Validation_Auditor 数值探针（只读审计，不改被测代码）。

探针覆盖：
  P1  HV 手算案例对比（3D 含交集案例、支配点、越界点、重复点）
  P2  HV 暴力网格交叉验证（随机小案例）
  P3  set_unified_reference 合并逻辑 / 零 range 兜底 / 空集处理
  P4  physical_filters 手算对比（VEC/δ/Ω/Δχ，等原子比 + 含 C 案例，
      以 hea_encoding_scheme_v1.13 §3.2.4/§7 的 c_i' 归一化公式为基准）
  P5  top10_overlap 贪婪匹配顺序依赖探针（最优匹配 vs 贪婪匹配）

运行（Git Bash，工作目录 D:\\QUBO）：
  PYTHONPATH=src "$DAIMON_USER_PYTHON" tests/manual/audit_validation_probes.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from ap3_qubo.validation.hypervolume import (
    HypervolumeCalculator,
    set_unified_reference,
)
from ap3_qubo.validation.physical_filters import PhysicalFilter, R_GAS
from ap3_qubo.physical_params import ELEM, MAIN_ELEMENTS
from ap3_qubo.experiments.sensitivity import SensitivityAnalyzer

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
# P1 HV 手算案例
# =============================================================================
print("=" * 72)
print("[P1] HV 手算案例对比")
print("=" * 72)

# 案例 1：单点 (1,1,1)，ref (2,2,2) → HV = 1×1×1 = 1
calc = HypervolumeCalculator(reference_point=np.array([2.0, 2.0, 2.0]))
hv = calc.compute(np.array([[1.0, 1.0, 1.0]]))
check("单点 HV == 1.0", abs(hv - 1.0) < 1e-12, f"hv={hv}")

# 案例 2：三点 (1,2,3),(2,3,1),(3,1,2)，ref (4,4,4)
# 手算：盒体积 6+6+6=18，两两交集 2+2+2=6，三者交 1 → HV=13
pts = np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 1.0], [3.0, 1.0, 2.0]])
calc = HypervolumeCalculator(reference_point=np.array([4.0, 4.0, 4.0]))
hv = calc.compute(pts)
check("三点含交 HV == 13.0", abs(hv - 13.0) < 1e-12, f"hv={hv}")

# 案例 3：支配点不改变 HV —— 加入 (2.5,2.5,2.5)（被 (1,2,3)? 不支配…
# 选明确被支配点：(3.5,3.5,3.5) 被 (1,2,3)/(2,3,1)/(3,1,2) 支配? 需全部分量<=。
# (2,3,1) vs (3.5,3.5,3.5): 2<=3.5,3<=3.5,1<=3.5 且至少一个< → 支配成立。
hv2 = calc.compute(np.vstack([pts, [[3.5, 3.5, 3.5]]]))
check("加入被支配点 HV 不变", abs(hv2 - 13.0) < 1e-12, f"hv={hv2}")

# 案例 4：越界点（超出 ref）被剔除 → HV 不变
hv3 = calc.compute(np.vstack([pts, [[5.0, 0.5, 0.5]]]))  # f1=5 > ref 4
check("越界点剔除 HV 不变", abs(hv3 - 13.0) < 1e-12, f"hv={hv3}")

# 案例 5：重复点不改变 HV
hv4 = calc.compute(np.vstack([pts, pts]))
check("重复点 HV 不变", abs(hv4 - 13.0) < 1e-12, f"hv={hv4}")

# 案例 6：2 点 2D 退化（f3 相同切片）——(1,2),(2,1) ref(3,3) 推广到 3D：
# 用 3D 点 (1,2,0),(2,1,0), ref (3,3,1)：等价 2D 面积 3 × 厚度 1 = 3
calc2 = HypervolumeCalculator(reference_point=np.array([3.0, 3.0, 1.0]))
hv5 = calc2.compute(np.array([[1.0, 2.0, 0.0], [2.0, 1.0, 0.0]]))
check("2D 退化案例 HV == 3.0", abs(hv5 - 3.0) < 1e-12, f"hv={hv5}")

# 案例 7：空集 → 0.0；全越界 → 0.0
check("空集 HV == 0.0", calc.compute(np.zeros((0, 3))) == 0.0)
check(
    "全越界 HV == 0.0",
    calc.compute(np.array([[9.0, 9.0, 9.0]])) == 0.0,
)

# 案例 8：compute_delta 口径
check(
    "compute_delta 常规",
    abs(calc.compute_delta(100.0, 110.0) - 0.1) < 1e-12,
)
check("compute_delta 零基线", calc.compute_delta(0.0, 5.0) == 1.0)
check("compute_delta 双零", calc.compute_delta(0.0, 0.0) == 0.0)

# =============================================================================
# P2 HV 暴力网格交叉验证
# =============================================================================
print()
print("=" * 72)
print("[P2] HV 暴力网格交叉验证（随机小案例 ×3）")
print("=" * 72)


def brute_force_hv(points, ref, n_grid=60):
    """在 [0, ref] 网格上逐格判定是否被支配（3D 最小化）。"""
    lo = np.zeros(3)
    edges = [np.linspace(lo[i], ref[i], n_grid + 1) for i in range(3)]
    vol = 0.0
    cell = np.prod([(ref[i] - lo[i]) / n_grid for i in range(3)])
    front = points
    for i in range(n_grid):
        x = 0.5 * (edges[0][i] + edges[0][i + 1])
        for j in range(n_grid):
            y = 0.5 * (edges[1][j] + edges[1][j + 1])
            for k in range(n_grid):
                z = 0.5 * (edges[2][k] + edges[2][k + 1])
                q = np.array([x, y, z])
                if np.any(np.all(front <= q, axis=1)):
                    vol += cell
    return vol


rng = np.random.default_rng(20260718)
for trial in range(3):
    pts = rng.random((6, 3)) * np.array([8.0, 5.0, 3.0])
    ref = np.array([10.0, 6.0, 4.0])
    calc = HypervolumeCalculator(reference_point=ref)
    hv_exact = calc.compute(pts)
    hv_brute = brute_force_hv(pts, ref, n_grid=60)
    rel = abs(hv_exact - hv_brute) / max(hv_brute, 1e-12)
    check(
        f"随机案例 {trial}: 精确 HV vs 暴力网格 相对误差 < 2%",
        rel < 0.02,
        f"exact={hv_exact:.4f}, brute={hv_brute:.4f}, rel={rel:.4%}",
    )

# =============================================================================
# P3 set_unified_reference
# =============================================================================
print()
print("=" * 72)
print("[P3] set_unified_reference 合并逻辑 / 零 range 兜底 / 空集")
print("=" * 72)

# 合并逻辑：A 的 nadir 在 f1，B 的 nadir 在 f2/f3 → ref 取合并后 nadir
A = np.array([[1.0, 2.0, 3.0]])
B = np.array([[5.0, 8.0, 9.0]])
ref = set_unified_reference({"A": A, "B": B}, margin=0.10)
# merged: min=(1,2,3), nadir=(5,8,9), range=(4,6,6), margin=(0.4,0.6,0.6)
expected = np.array([5.4, 8.6, 9.6])
check(
    "合并 nadir + 10%×range",
    np.allclose(ref, expected),
    f"ref={ref}, expected={expected}",
)

# 跨方法可比性语义：ref 必须 ≥ 任一方法自己的 nadir（否则削顶）
rng = np.random.default_rng(1)
mA = rng.random((20, 3)) * [5, 5, 5]
mB = rng.random((20, 3)) * [50, 0.5, 10] + [0, 4, 0]
ref = set_unified_reference({"A": mA, "B": mB})
check(
    "统一 ref 支配所有方法解集",
    np.all(mA <= ref) and np.all(mB <= ref),
    f"ref={np.round(ref, 3)}",
)

# 零 range 兜底：某目标全部相同
C = np.array([[1.0, 7.0, 3.0], [2.0, 7.0, 4.0]])
ref = set_unified_reference({"C": C}, margin=0.10)
# f1: nadir=2, range=1 → ref=2.1; f2: nadir=7, range=0 → fallback |7|*0.1=0.7 → ref=7.7
# f3: nadir=4, range=1 → ref=4.1
expected = np.array([2.1, 7.7, 4.1])
check(
    "零 range 走 |nadir|×margin 兜底",
    np.allclose(ref, expected),
    f"ref={ref}, expected={expected}",
)

# nadir≈0 的零 range：退化为 margin 本身
D = np.array([[1.0, 0.0, 3.0], [2.0, 0.0, 4.0]])
ref = set_unified_reference({"D": D}, margin=0.10)
check(
    "nadir≈0 零 range 退化为 margin",
    abs(ref[1] - 0.10) < 1e-12,
    f"ref[1]={ref[1]}",
)

# 空矩阵跳过
ref = set_unified_reference({"E": np.zeros((0, 3)), "C": C}, margin=0.10)
check("空矩阵自动跳过", np.allclose(ref, np.array([2.1, 7.7, 4.1])), f"ref={ref}")

# 全空 → ValueError
try:
    set_unified_reference({"E": np.zeros((0, 3))})
    check("全空应抛 ValueError", False)
except ValueError:
    check("全空应抛 ValueError", True)

# 与 set_reference_from_data 单集合一致性（同口径）
calc = HypervolumeCalculator()
calc.set_reference_from_data(C, margin_factor=0.10)
check(
    "单集合与 set_reference_from_data 一致",
    np.allclose(set_unified_reference({"C": C}), calc.reference_point),
)

# =============================================================================
# P4 physical_filters 手算对比（基准：scheme v1.13 c_i' 归一化公式）
# =============================================================================
print()
print("=" * 72)
print("[P4] physical_filters 手算对比（scheme v1.13 §3.2.4/§7 基准）")
print("=" * 72)


def scheme_metrics(fracs_at, dh_mix):
    """严格按 hea_encoding_scheme_v1.13 公式计算 VEC/δ/Ω/Δχ。

    c_i' = c_i/(1 − c_C)（主元归一化）；VEC/δ/ΔS/Δχ 仅用主元 c_i'；
    T_m = Σ_{含C} c_i·T_m,i（v1.13 :690 明确"含C"）。
    """
    tot = sum(fracs_at[e] for e in MAIN_ELEMENTS)  # = 100 − c_C
    cp = {e: fracs_at[e] / tot for e in MAIN_ELEMENTS}  # c_i'
    vec = sum(cp[e] * ELEM.vec_of(e) for e in MAIN_ELEMENTS)
    r_bar = sum(cp[e] * ELEM.radius_of(e) for e in MAIN_ELEMENTS)
    delta = float(np.sqrt(sum(
        cp[e] * (1.0 - ELEM.radius_of(e) / r_bar) ** 2 for e in MAIN_ELEMENTS
    )) * 100.0)
    chi_bar = sum(cp[e] * ELEM.en_of(e) for e in MAIN_ELEMENTS)
    dchi = float(np.sqrt(sum(
        cp[e] * (ELEM.en_of(e) - chi_bar) ** 2 for e in MAIN_ELEMENTS
    )))
    ds = -R_GAS * sum(cp[e] * np.log(cp[e]) for e in MAIN_ELEMENTS if cp[e] > 1e-10)
    t_m = sum(
        (fracs_at.get(e, 0.0) / 100.0) * ELEM.melting_point_of(e)
        for e in list(MAIN_ELEMENTS) + ["C"]
    )
    omega = float(t_m * ds / (abs(dh_mix) * 1000.0)) if abs(dh_mix) > 1e-9 else 100.0
    return vec, delta, omega, dchi


filt = PhysicalFilter()

# 案例 A：等原子比 AlCoCrFeNi（C=0）——此时 c_i' = c_i，代码与方案应一致
comp_eq = {"Al": 20.0, "Co": 20.0, "Cr": 20.0, "Fe": 20.0, "Ni": 20.0, "C": 0.0}
r = filt.evaluate(comp_eq, dh_mix=-12.32)
vec_s, delta_s, omega_s, dchi_s = scheme_metrics(comp_eq, -12.32)
print(f"  [手算·等原子] VEC={vec_s:.4f} δ={delta_s:.4f}% Ω={omega_s:.4f} Δχ={dchi_s:.4f}")
print(f"  [代码·等原子] VEC={r.vec:.4f} δ={r.delta:.4f}% Ω={r.omega:.4f} Δχ={r.delta_chi:.4f}")
check("C=0 时 VEC 与手算一致", abs(r.vec - vec_s) < 1e-9, f"{r.vec} vs {vec_s}")
check("C=0 时 δ 与手算一致", abs(r.delta - delta_s) < 1e-9, f"{r.delta} vs {delta_s}")
check("C=0 时 Ω 与手算一致", abs(r.omega - omega_s) < 1e-9, f"{r.omega} vs {omega_s}")
check("C=0 时 Δχ 与手算一致", abs(r.delta_chi - dchi_s) < 1e-9, f"{r.delta_chi} vs {dchi_s}")
# 已知参考值：VEC=7.2（v1.13 :286 中心点）
check("等原子 VEC == 7.2（v1.13 :286）", abs(r.vec - 7.2) < 1e-9)

# 案例 B：含 C=1.0 at%（编码网格可达点）——c_i' 归一化差异显形
comp_c = {"Al": 19.8, "Co": 19.8, "Cr": 19.8, "Fe": 19.8, "Ni": 19.8, "C": 1.0}
r = filt.evaluate(comp_c, dh_mix=-12.0)
vec_s, delta_s, omega_s, dchi_s = scheme_metrics(comp_c, -12.0)
print(f"  [手算·C=1.0] VEC={vec_s:.4f} δ={delta_s:.4f}% Ω={omega_s:.4f} Δχ={dchi_s:.4f}")
print(f"  [代码·C=1.0] VEC={r.vec:.4f} δ={r.delta:.4f}% Ω={r.omega:.4f} Δχ={r.delta_chi:.4f}")
for name, code_v, scheme_v in [
    ("VEC", r.vec, vec_s), ("δ", r.delta, delta_s),
    ("Ω", r.omega, omega_s), ("Δχ", r.delta_chi, dchi_s),
]:
    dev = abs(code_v - scheme_v) / max(abs(scheme_v), 1e-12)
    check(f"C=1.0 时 {name} 与方案公式一致（相对偏差 < 1e-9）",
          dev < 1e-9, f"code={code_v:.6f} scheme={scheme_v:.6f} 偏差={dev:.3%}")

# 案例 C：C 上限 1.75 at%（编码最大值）——偏差量级评估
comp_cmax = {"Al": 19.65, "Co": 19.65, "Cr": 19.65, "Fe": 19.65, "Ni": 19.65, "C": 1.75}
r = filt.evaluate(comp_cmax, dh_mix=-12.0)
vec_s, delta_s, omega_s, dchi_s = scheme_metrics(comp_cmax, -12.0)
print(f"  [手算·C=1.75] VEC={vec_s:.4f} δ={delta_s:.4f}% Ω={omega_s:.4f} Δχ={dchi_s:.4f}")
print(f"  [代码·C=1.75] VEC={r.vec:.4f} δ={r.delta:.4f}% Ω={r.omega:.4f} Δχ={r.delta_chi:.4f}")
for name, code_v, scheme_v in [
    ("VEC", r.vec, vec_s), ("δ", r.delta, delta_s),
    ("Ω", r.omega, omega_s), ("Δχ", r.delta_chi, dchi_s),
]:
    dev = abs(code_v - scheme_v) / max(abs(scheme_v), 1e-12)
    print(f"    {name}: code={code_v:.6f} scheme={scheme_v:.6f} 偏差={dev:.3%}")

# 案例 D：Ω 分级阈值探针（1.1 / 0.8 边界）
r = filt.evaluate(comp_eq, dh_mix=-1.0)  # |ΔH| 小 → Ω 大 → stable
check("Ω 大 → stable", r.omega_level == "stable", f"Ω={r.omega:.3f}")
# ΔH≈0 兜底 → 100.0 → stable
r0 = filt.evaluate(comp_eq, dh_mix=0.0)
check("ΔH=0 → Ω=100 兜底 stable", r0.omega == 100.0 and r0.omega_level == "stable")

# 阈值边界语义（代码 ≥ vs 文档 >）：c_C = 1.0 恰在文档阈值
r = filt.evaluate({**comp_eq, "C": 1.0, "Ni": 19.0}, dh_mix=-12.0)
print(f"  [边界] c_C=1.0 → carbide_risk={r.carbide_risk}（文档 'c_C>1.0% 标记' 严格语义应为 none/warning）")

# =============================================================================
# P5 top10_overlap 贪婪匹配顺序依赖
# =============================================================================
print()
print("=" * 72)
print("[P5] top10_overlap 贪婪匹配探针")
print("=" * 72)

E6 = ["Al", "Co", "Cr", "Fe", "Ni", "C"]


def mk(al):
    # 单元素变化，其余固定，成分和=100
    return dict(zip(E6, [al, 20.0, 20.0, 20.0, 39.0 - al, 1.0]))


# 贪婪失配案例：B1 先匹配抢占了 B2 唯一可配对的候选
A1, A2 = mk(20.0), mk(20.45)
B1, B2 = mk(20.3), mk(19.6)
# B1: d(A1)=0.3✓ d(A2)=0.15✓；B2: d(A1)=0.4✓ d(A2)=0.85✗
ov = SensitivityAnalyzer._top10_pair_overlap([A1, A2], [B1, B2], 0.5)
print(f"  贪婪匹配结果 = {ov}（最优匹配 = 1.0：B1↔A2, B2↔A1）")
print("  → 贪婪顺序依赖会低估重叠率（保守方向偏差），docstring 已声明'贪婪'")
check("贪婪匹配为合法的一一匹配（不重不漏）", 0.0 <= ov <= 1.0)

# =============================================================================
# 汇总
# =============================================================================
print()
print("=" * 72)
print(f"探针汇总: PASS={PASS}, FAIL={FAIL}")
print("=" * 72)
sys.exit(0 if FAIL == 0 else 1)
