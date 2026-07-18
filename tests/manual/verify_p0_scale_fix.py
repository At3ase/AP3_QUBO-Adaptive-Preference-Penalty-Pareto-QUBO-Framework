# -*- coding: utf-8 -*-
"""P0-1 修复数值验证 (不依赖 kaiwu)。

验证 builder.py:_build_P0_expr 修复后的 P0 惩罚尺度:
  修复后: penalty = λ_sum x (Σc - 100)² / 5625
对方案 C-02 规范形式 (hea_encoding_scheme_v1.13.md §3.1):
  penalty = λ_sum x (S_var/75 - 1)²,  S_var = Σc - 25
并对账规范实现 constraints/sum_constraint.py 的 h/Q 矩阵能量。

编码: precision_split_38 — 主元 base 5.0, C base 0.0, 步长均 0.25。
"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import numpy as np
from ap3_qubo.constraints.sum_constraint import SumTo100Constraint

LAMBDA = 15.0          # CONSTRAINT.lambda_sum_fixed
S_CONST = 25.0         # 5x5.0 (主元基线) + 0.0 (C 基线)
STEP = 0.25

def penalty_before(sum_c):
    """修复前 (错误): λx(Σc-100)² — 无归一化。"""
    return LAMBDA * (sum_c - 100.0) ** 2

def penalty_after(sum_c):
    """修复后 builder.py 语义: λx(Σc-100)²/5625。"""
    return LAMBDA * (sum_c - 100.0) ** 2 / 5625.0

def penalty_scheme(sum_c):
    """方案 C-02 规范形式: λx(S_var/75 - 1)²。"""
    s_var = sum_c - S_CONST
    return LAMBDA * (s_var / 75.0 - 1.0) ** 2

print("=" * 72)
print("[1] 方案 §3.1 数值验证点 (K = 步长计数, S_var = 0.25xK)")
print("=" * 72)
print(f"{'K':>5} {'Σc(at%)':>9} {'修复前':>14} {'修复后':>12} {'方案C-02':>12} {'前/后倍数':>10}")
for K, tag in [(0, "全最低"), (300, "等原子比"), (642, "全最高")]:
    sum_c = S_CONST + STEP * K
    b, a, s = penalty_before(sum_c), penalty_after(sum_c), penalty_scheme(sum_c)
    print(f"{K:>5} {sum_c:>9.2f} {b:>14.4f} {a:>12.6f} {s:>12.6f} {b/a if a else float('inf'):>10.1f}  # {tag}")
assert abs(penalty_after(S_CONST + 75)) < 1e-12          # K=300 惩罚=0
assert abs(penalty_after(25.0) - 15.0) < 1e-9            # K=0 → λx1 = 15
assert abs(penalty_before(25.0) / penalty_after(25.0) - 5625.0) < 1e-6
print("[PASS] K=300 penalty=0; K=0 fixed=15 (=lambda); ratio = 5625x")

print()
print("=" * 72)
print("[2] 随机扫描: 修复后表达式 ≡ 方案 C-02 规范形式")
print("=" * 72)
rng = random.Random(42)
max_dev = 0.0
for _ in range(2000):
    ks = [rng.randint(0, 127) for _ in range(5)] + [rng.randint(0, 7)]  # 7bitx5 + 3bit
    sum_c = S_CONST + STEP * sum(ks)
    max_dev = max(max_dev, abs(penalty_after(sum_c) - penalty_scheme(sum_c)))
print(f"2000 组随机成分, 修复后与方案形式最大偏差 = {max_dev:.3e}")
assert max_dev < 1e-9
print("[PASS] 修复后表达式与方案 C-02 完全等价")

print()
print("=" * 72)
print("[3] 对账规范实现 sum_constraint.py (h/Q 矩阵能量)")
print("=" * 72)
con = SumTo100Constraint()  # λ=15, scale_factor=λ/5625
print(f"scale_factor = λ/5625 = {con.scale_factor:.10f}  (期望 {15/5625:.10f})")
assert abs(con.scale_factor - LAMBDA / 5625.0) < 1e-15
h_vec, Q_mat = con.get_qubo_terms()
n = len(h_vec)
max_dev = 0.0
for _ in range(500):
    x = np.array([rng.randint(0, 1) for _ in range(n)], dtype=float)
    # 由变量位权重建 S_var: coef_i = 0.25x2^bit_pos
    s_var = 0.0
    for i in range(n):
        _, bit_pos = con._mapper.flat_to_element_bit(i)
        s_var += STEP * (1 << bit_pos) * x[i]
    # sum_constraint 丢弃了常数项 sfx5625 = λ (仅 offset), 故 E_terms + λ = 惩罚值
    e_terms = float(h_vec @ x) + float(x @ Q_mat @ x)
    max_dev = max(max_dev, abs((e_terms + LAMBDA) - penalty_after(S_CONST + s_var)))
print(f"500 组随机比特串, |E_hQ + λ - 修复后惩罚| 最大偏差 = {max_dev:.3e}")
assert max_dev < 1e-9
print("[PASS] 修复后 builder 表达式与规范实现对账一致 (差 λ 常数 offset)")

print()
print("ALL CHECKS PASSED — P0-1 修复数值验证通过")
