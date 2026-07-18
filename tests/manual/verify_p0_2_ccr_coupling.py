"""
P0-2 修复验证: C-Cr 耦合 QUBO 线性项双倍计数。

方案出处: plan/hea_encoding_scheme_v1.13.md 第 C-06 节
  H_CCr = c_C * c_Cr / 64.3125
  c_C  = 0.25 * sum(2^k * x_{35+k}),  k=0..2
  c_Cr = 5.0 + 0.25 * sum(2^j * x_{14+j}), j=0..6

验证内容:
  1. 修正后线性项 h[35..37] == [0.01944, 0.03887, 0.07775] (容差 1e-4)
  2. 对若干成分点, QUBO 能量 == 解析 c_C*c_Cr/H_max (权重 lambda*omega)
  3. get_qubo_terms() 与 evaluate_penalty() 自洽
  4. 交叉项非零个数 == 21 (cim_mode 关闭时)

本模块不依赖 kaiwu, 可直接运行。
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from ap3_qubo.constraints.ccr_coupling import CCrCouplingConstraint  # noqa: E402

H_MAX = 64.3125
TOL = 1e-4


def qubo_energy(x: np.ndarray, h: np.ndarray, Q: np.ndarray) -> float:
    """E = sum_i h_i x_i + sum_{i<j} Q_ij x_i x_j  (Q 上三角)"""
    return float(x @ h + x @ Q @ x)


def main() -> int:
    c = CCrCouplingConstraint()
    failures = []

    # ---- 检查 1: 线性项数值 (weight=1) ----
    h, Q = c.get_qubo_terms(lambda_ccr=1.0, omega=1.0)
    expected_h = np.array([1.25 * (1 << k) / H_MAX for k in range(3)])
    got_h = h[35:38]
    ok1 = np.allclose(got_h, expected_h, atol=TOL)
    print(f"[1] h[35..37] = {got_h}")
    print(f"    期望      = {expected_h}  ([0.01944, 0.03887, 0.07775])")
    print(f"    结果: {'PASS' if ok1 else 'FAIL'}")
    if not ok1:
        failures.append("linear term values")

    # 非线性项索引不应有 h
    other_h = np.delete(h, [35, 36, 37])
    ok1b = np.all(other_h == 0.0)
    print(f"    其余索引 h 全为 0: {'PASS' if ok1b else 'FAIL'}")
    if not ok1b:
        failures.append("unexpected h outside 35..37")

    # ---- 检查 2: 交叉项个数 (cim_mode=False 应为 21) ----
    nz = int(np.count_nonzero(Q))
    ok2 = nz == 21
    print(f"[2] 交叉项非零个数 = {nz} (期望 21): {'PASS' if ok2 else 'FAIL'}")
    if not ok2:
        failures.append("cross term count")

    # ---- 检查 3: 成分点能量一致性 (QUBO == 解析 == evaluate_penalty) ----
    # (c_bits, cr_bits) 若干成分点, 覆盖边界与随机情形
    cases = [
        ([0, 0, 0], [0, 0, 0, 0, 0, 0, 0]),   # c_C=0,    c_Cr=5.0
        ([1, 1, 1], [1, 1, 1, 1, 1, 1, 1]),   # c_C=1.75, c_Cr=36.75 (H=1)
        ([1, 0, 0], [1, 0, 0, 0, 0, 0, 0]),   # c_C=0.25, c_Cr=5.25
        ([0, 1, 0], [0, 1, 0, 1, 0, 1, 0]),   # c_C=0.5,  c_Cr=5.0+0.25*42=15.5
        ([1, 0, 1], [0, 1, 1, 0, 0, 1, 1]),   # c_C=1.25, c_Cr=5.0+0.25*102=30.5
    ]
    rng = np.random.default_rng(42)
    for _ in range(5):
        cases.append((list(rng.integers(0, 2, 3)), list(rng.integers(0, 2, 7))))

    lam, om = 1.7, 0.5  # 非平凡权重
    h2, Q2 = c.get_qubo_terms(lambda_ccr=lam, omega=om)
    ok3 = True
    for c_bits, cr_bits in cases:
        x = np.zeros(c._mapper.total_variables)
        for k in range(3):
            x[35 + k] = c_bits[k]
        for j in range(7):
            x[14 + j] = cr_bits[j]

        c_C = 0.25 * sum((1 << k) * c_bits[k] for k in range(3))
        c_Cr = 5.0 + 0.25 * sum((1 << j) * cr_bits[j] for j in range(7))

        e_qubo = qubo_energy(x, h2, Q2)
        e_analytic = lam * om * c_C * c_Cr / H_MAX
        e_penalty = c.evaluate_penalty(c_C, c_Cr, lam, om)

        match = abs(e_qubo - e_analytic) < 1e-10 and abs(e_penalty - e_analytic) < 1e-12
        ok3 &= match
        print(f"    c_C={c_C:5.2f}, c_Cr={c_Cr:6.2f}: QUBO={e_qubo:.8f} "
              f"解析={e_analytic:.8f} penalty={e_penalty:.8f} {'OK' if match else 'MISMATCH'}")
    print(f"[3] 能量一致性 (10 个成分点): {'PASS' if ok3 else 'FAIL'}")
    if not ok3:
        failures.append("energy consistency")

    print()
    if failures:
        print(f"FAILED: {failures}")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
