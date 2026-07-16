"""
QUBO ↔ Ising 模型转换。

QUBO: x_i ∈ {0, 1},  H_QUBO = Σ q_i·x_i + Σ_{i<j} Q_ij·x_i·x_j
Ising: σ_i ∈ {+1, -1}, H_Ising = Σ h_i·σ_i + Σ_{i<j} J_ij·σ_i·σ_j

转换关系 (x_i = (σ_i + 1) / 2):
  J_ij = Q_ij / 4
  h_i = q_i / 2 + Σ_{j≠i} Q_ij / 4
  constant_offset = Σ_i q_i / 2 + Σ_{i<j} Q_ij / 4
"""

import numpy as np


def qubo_to_ising(
    h_qubo: np.ndarray, Q_qubo: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    """将 QUBO (h, Q) 转换为 Ising (h_ising, J_ising)。

    Args:
        h_qubo: shape=(N,) QUBO 线性系数。
        Q_qubo: shape=(N,N) QUBO 二次系数 (上三角)。

    Returns:
        (h_ising, J_ising, constant_offset):
          h_ising: shape=(N,) Ising 局域场。
          J_ising: shape=(N,N) Ising 耦合矩阵 (上三角)。
          constant_offset: 转换产生的常数偏移。
    """
    n = len(h_qubo)
    J_ising = np.zeros((n, n), dtype=float)

    # J_ij = Q_ij / 4
    for i in range(n):
        for j in range(i + 1, n):
            if Q_qubo[i, j] != 0:
                J_ising[i, j] = Q_qubo[i, j] / 4.0

    # h_i = q_i / 2 + Σ_{j≠i} (Q_ij + Q_ji) / 4
    # 由于 Q_qubo 是上三角: Q_ji = 0 for j > i
    h_ising = h_qubo.copy() / 2.0
    for i in range(n):
        for j in range(i + 1, n):
            contrib = Q_qubo[i, j] / 4.0
            h_ising[i] += contrib
            h_ising[j] += contrib  # Q_ij = Q_ji in upper-tri representation

    # 常数偏移
    offset = np.sum(h_qubo) / 2.0
    for i in range(n):
        for j in range(i + 1, n):
            offset += Q_qubo[i, j] / 4.0

    return h_ising, J_ising, offset


def ising_to_qubo(
    h_ising: np.ndarray, J_ising: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    """将 Ising (h, J) 转换为 QUBO (h, Q)。

    σ_i = 2·x_i - 1, x_i ∈ {0, 1}
    """
    n = len(h_ising)
    Q_qubo = np.zeros((n, n), dtype=float)

    # Q_ij = 4 × J_ij
    for i in range(n):
        for j in range(i + 1, n):
            Q_qubo[i, j] = 4.0 * J_ising[i, j]

    # q_i = 2·h_i - 2·Σ_{j≠i} J_ij
    h_qubo = 2.0 * h_ising.copy()
    for i in range(n):
        for j in range(i + 1, n):
            h_qubo[i] -= 2.0 * J_ising[i, j]
            h_qubo[j] -= 2.0 * J_ising[i, j]

    # 常数偏移
    offset = -np.sum(h_ising)
    for i in range(n):
        for j in range(i + 1, n):
            offset += J_ising[i, j]

    return h_qubo, Q_qubo, offset


def spins_to_bits(spins: np.ndarray) -> np.ndarray:
    """Ising 自旋 → QUBO 比特: x = (σ + 1) / 2。"""
    return ((np.array(spins) + 1) // 2).astype(np.int8)


def bits_to_spins(bits: np.ndarray) -> np.ndarray:
    """QUBO 比特 → Ising 自旋: σ = 2·x - 1。"""
    return (2 * np.array(bits) - 1).astype(np.int8)
