"""
QUBO 模型构建模块。

基于 kaiwu SDK 原生 Binary 变量和 quicksum API 构建完整的 AP³-QUBO 优化模型:

  H_total = w1·f1_norm + w2·f2_norm + w3·f3_norm
          + λ_sum·H_sum(P0)
          + ω_carbide·λ_carbide·H_carbide(P1)
          + ω_CCr·λ_CCr·H_CCr(P2)

kaiwu SDK 负责表达式展开 → QUBO 矩阵的完整流程。

核心类型:
  - QUBOBuilder: 主构建器，支持三种编码模式 (precision_split_38 / unified_48 / unified_38)
  - QUBOMatrix: (h, Q, offset) 三元组，封装 QUBO 矩阵的标准表示
  - IsingConverter: QUBO ↔ Ising 模型转换

kaiwu 采用延迟导入：模块级不 import kaiwu，避免硬依赖阻断上层包的导入链。
仅首次调用 build_model() / build() 时才完成实际导入。
"""

from .builder import QUBOBuilder, QUBOMatrix

__all__ = [
    "QUBOBuilder",
    "QUBOMatrix",
]
