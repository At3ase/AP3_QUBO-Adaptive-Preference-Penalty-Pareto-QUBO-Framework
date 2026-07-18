"""
PrecisionSplit 分层精度编码模块（第一层）。

将 Al-Co-Cr-Fe-Ni-C 六元成分空间编码为 38 个二元 QUBO 变量:
  - 5 主元 × 7 比特 = 35 变量
  - C 间隙元素 × 3 比特 = 3 变量
所有元素共享统一步长 0.25 at%，成分和=100% 可精确满足。

核心类型:
  - Composition: 高熵合金成分点（元素 → at%）
  - PrecisionSplitEncoder: 连续成分 → 38 比特二进制向量
  - PrecisionSplitDecoder: 38 比特 → Composition
  - VariableMapper: (元素, 比特位) ↔ flat index [0..37] 双向映射
"""

from .precision_split import (
    Composition,
    PrecisionSplitEncoder,
    PrecisionSplitDecoder,
)
from .variable_index import VariableMapper

__all__ = [
    "Composition",
    "PrecisionSplitEncoder",
    "PrecisionSplitDecoder",
    "VariableMapper",
]
