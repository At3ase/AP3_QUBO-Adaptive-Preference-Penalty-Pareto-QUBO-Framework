"""
求解器模块。

提供统一的抽象求解器接口 + Kaiwu SDK 具体实现。

抽象层（base.py）:
  - AbstractSolver: 所有求解器的抽象基类
  - Solution: 单个 QUBO 解（bits, energy, is_feasible, metadata）
  - SolverResult: 单次求解的完整结果（TOP-K 解列表 + 耗时 + 元数据）

具体实现（kaiwu_solver.py）:
  - KaiwuSolver: kaiwu SDK 求解器，支持三种模式:
      * "simulator": 内置模拟退火后端（离线可用，真实优化计算）
      * "cim":      CIM 光量子真机（需完整版 kaiwu SDK + 玻色量子授权）
      * "auto":     模拟器优先门禁（= simulator）

关键修复（第 2 批）:
  - TOP-K 完整链路: Ising 采样 → get_sorted_solutions 排序去重 → TOP-K 解码
  - 变量名兼容: 同时支持 builder 格式 (e0_b0) 和矩阵格式 (b[00])
  - 位布局推断: 自动识别 precision_split_38 / unified_48 / unified_38
  - 硬件自检 D-06: 变量数/耦合数 vs 550W 真机上限
"""

from .base import AbstractSolver, Solution, SolverResult
from .kaiwu_solver import KaiwuSolver

__all__ = [
    "AbstractSolver",
    "Solution",
    "SolverResult",
    "KaiwuSolver",
]
