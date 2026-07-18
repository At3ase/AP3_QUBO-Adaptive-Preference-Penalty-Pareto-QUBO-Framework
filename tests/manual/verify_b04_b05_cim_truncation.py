# -*- coding: utf-8 -*-
"""
B-04 / B-05 CIM 截断修复验证脚本 (前后对比用)。

运行环境: C:\\Users\\At3ase\\AppData\\Local\\Programs\\Python\\Python310\\python.exe
          (该环境已装 kaiwu, 勿用默认 python 3.12)

验证点:
  [1] 孤儿路径 constraints/ccr_coupling.py (B-04):
      默认初始 lambda_ccr=0.05, omega=0.5, cim_mode=True 时
      P2 的 21 个交叉项是否被全灭 (修复前: 0 存活)。
  [2] 主路径 qubo/builder.py (B-05):
      全模型 cim_mode=True 是否误清 f1 等目标函数小系数;
      C-bit x Cr-bit 区块 (21 个交叉位置) 在 CIM 模式下是否存活。
  [3] 能量探针: 等原子比点与高 C-Cr 点的 full/cim 能量差。

本脚本只读, 不修改任何源码。修复前后各运行一次以对比。
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from ap3_qubo.constraints.ccr_coupling import CCrCouplingConstraint
from ap3_qubo.physical_params import CONSTRAINT
from ap3_qubo.qubo.builder import QUBOBuilder

W = (0.4, 0.3, 0.3)                       # 与 tests/verify_kaiwu_integration.py 一致
LAM = CONSTRAINT.lambda_ccr_init          # 0.05 (PenaltyFlex 初始值)
OMG = CONSTRAINT.omega_ccr                # 0.5
THR = CONSTRAINT.cim_noise_floor          # 0.01

SEP = "=" * 72


def bits_for(values):
    """values: dict elem_idx(0-5) -> 整数位值, 返回 38 位 0/1 数组。"""
    bits = np.zeros(38, dtype=float)
    for ei, v in values.items():
        n_bits = 7 if ei < 5 else 3
        base = ei * 7 if ei < 5 else 35
        for bj in range(n_bits):
            bits[base + bj] = (v >> bj) & 1
    return bits


def main():
    print(SEP)
    print(f"参数: weights={W}, lambda_ccr={LAM}, omega_ccr={OMG}, threshold={THR}")
    print(f"P2 加权因子 lambda*omega = {LAM * OMG}")
    print(f"P2 交叉项未加权系数范围: {0.0625 / CONSTRAINT.ccr_h_max:.6f} ~ "
          f"{0.0625 / CONSTRAINT.ccr_h_max * 256:.6f}")
    print(f"P2 交叉项加权后系数范围:   {0.0625 / CONSTRAINT.ccr_h_max * LAM * OMG:.6f} ~ "
          f"{0.0625 / CONSTRAINT.ccr_h_max * 256 * LAM * OMG:.6f}")
    print(SEP)

    # ------------------------------------------------------------------
    # [0] kaiwu get_matrix() 格式探查
    # ------------------------------------------------------------------
    print("\n[0] kaiwu QuboModel.get_matrix() 格式探查")
    b = QUBOBuilder()
    model = b.build_model(weights=W)
    fm = model.get_matrix()
    sym = np.allclose(fm, fm.T)
    upper_only = np.allclose(fm, np.triu(fm))
    print(f"    shape={fm.shape}, 对称={sym}, 纯上三角={upper_only}, "
          f"offset={model.get_offset():.6f}")

    # ------------------------------------------------------------------
    # [1] B-04: 孤儿路径 ccr_coupling.py
    # ------------------------------------------------------------------
    print("\n[1] B-04 孤儿路径 CCrCouplingConstraint.get_qubo_terms (cim_mode=True)")
    c = CCrCouplingConstraint()
    h_off, Q_off = c.get_qubo_terms(LAM, OMG, cim_mode=False)
    h_on, Q_on = c.get_qubo_terms(LAM, OMG, cim_mode=True)
    nz_off = int(np.count_nonzero(Q_off))
    nz_on = int(np.count_nonzero(Q_on))
    print(f"    交叉项非零数 (cim OFF) = {nz_off}  (期望 21)")
    print(f"    交叉项非零数 (cim ON)  = {nz_on}  "
          f"{'<<< B-04: 全灭!' if nz_on == 0 else '(存活)'}")
    if nz_on:
        kept = [(14 + j, 35 + k) for k in range(3) for j in range(7)
                if Q_on[14 + j, 35 + k] != 0]
        print(f"    存活位置 (Cr_bit, C_bit): {kept}")
    print(f"    h[35..37] (cim ON)  = {h_on[35:38]}")
    print(f"    h[35..37] (cim OFF) = {h_off[35:38]}")

    # ------------------------------------------------------------------
    # [2] B-05: 主路径 builder.build(cim_mode=True)
    # ------------------------------------------------------------------
    print("\n[2] B-05 主路径 QUBOBuilder.build (全模型, cim_mode=True vs False)")
    mat_full = b.build(W, cim_mode=False)
    mat_cim = b.build(W, cim_mode=True)
    Qf, Qc = mat_full.Q, mat_cim.Q
    hf, hc = mat_full.h, mat_cim.h

    # 2a. C-bit x Cr-bit 区块存活 (上三角位置 [14+j, 35+k])
    block_idx = [(14 + j, 35 + k) for k in range(3) for j in range(7)]
    blk_full = sum(1 for i, j in block_idx if Qf[i, j] != 0)
    blk_cim = sum(1 for i, j in block_idx if Qc[i, j] != 0)
    print(f"    [2a] C×Cr 区块 (21 位) 非零: full={blk_full}/21, cim={blk_cim}/21")

    # 2b. 区块之外被 cim 清掉的目标/约束系数 (f1/f2/f3/P0/P1 误伤统计)
    block_set = set(block_idx)
    killed_q = [(i, j) for i in range(38) for j in range(i + 1, 38)
                if (i, j) not in block_set and Qf[i, j] != 0 and Qc[i, j] == 0]
    altered_q = [(i, j) for i in range(38) for j in range(i + 1, 38)
                 if (i, j) not in block_set and abs(Qf[i, j] - Qc[i, j]) > 1e-15]
    killed_h = [i for i in range(38) if hf[i] != 0 and hc[i] == 0]
    altered_h = [i for i in range(38) if abs(hf[i] - hc[i]) > 1e-15]
    print(f"    [2b] 区块外 Q 被清零数 = {len(killed_q)}  (B-05 误伤 f1/P0 等)")
    print(f"         区块外 Q 被改动数 = {len(altered_q)}")
    print(f"         h 被清零数 = {len(killed_h)}, h 被改动数 = {len(altered_h)}")
    n_small_obj = sum(1 for i in range(38) for j in range(i + 1, 38)
                      if (i, j) not in block_set and 0 < abs(Qf[i, j]) < THR)
    print(f"         (参考) 区块外 |Q|<{THR} 的非零项 = {n_small_obj} "
          f"<- 全矩阵截断会误清的数量级")

    # 2c. 区块内: cim 与 full 的差异 (P2 小项截断是预期行为)
    blk_diff = [(i, j) for i, j in block_idx if abs(Qf[i, j] - Qc[i, j]) > 1e-15]
    print(f"    [2c] 区块内被改动数 = {len(blk_diff)} "
          f"(预期: P2 未加权系数 < {THR} 的小项被截断)")

    # 2d. 整体非零计数
    print(f"    [2d] Q 非零总数: full={int(np.count_nonzero(Qf))}, "
          f"cim={int(np.count_nonzero(Qc))}")

    # ------------------------------------------------------------------
    # [3] 能量探针
    # ------------------------------------------------------------------
    print("\n[3] 能量探针: E(cim) - E(full)")
    probes = {
        "等原子比 (主元各19.75, C=1.25, Σ=100)": bits_for(
            {0: 59, 1: 59, 2: 59, 3: 59, 4: 59, 5: 5}),
        "高 C-Cr (Cr=36.75, C=1.75, 其余15.25)": bits_for(
            {0: 41, 1: 41, 2: 127, 3: 41, 4: 41, 5: 7}),
    }
    for name, bits in probes.items():
        ef = mat_full.compute_energy(bits)
        ec = mat_cim.compute_energy(bits)
        print(f"    {name}")
        print(f"      E(full)={ef:.8f}  E(cim)={ec:.8f}  diff={ec - ef:+.8f}")

    # ------------------------------------------------------------------
    # [4] P2 自身能量信号 (截断后 P2 是否仍提供违反度反馈)
    # ------------------------------------------------------------------
    print("\n[4] P2 信号存活检验 (PenaltyFlex 反馈前提)")
    print("    同一点位, cim 模型能量对 C 含量单调性 (Cr 拉满, C 从 0 增):")
    prev = None
    mono = True
    for c_bits_val in range(8):
        bits = bits_for({0: 41, 1: 41, 2: 127, 3: 41, 4: 41, 5: c_bits_val})
        e = mat_cim.compute_energy(bits)
        tag = ""
        if prev is not None:
            tag = f"(Δ={e - prev:+.6f})"
            if e < prev:
                mono = False
        print(f"      C bits={c_bits_val} (c_C={0.25 * c_bits_val:.2f}%): "
              f"E_cim={e:.6f} {tag}")
        prev = e
    print(f"    P2 惩罚随 C 含量单调不减: {mono}")


if __name__ == "__main__":
    main()
