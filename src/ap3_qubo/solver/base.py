"""
求解器抽象层。

提供统一的 AbstractSolver 接口 + Solution/SolverResult 数据类型。
具体求解器实现见 kaiwu_solver.py。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any

import numpy as np


@dataclass
class Solution:
    """单个 QUBO 解。

    Attributes:
        bits: shape=(N,) 二元变量向量 (0/1)。
        energy: QUBO 能量值。
        is_feasible: 是否满足约束。
        metadata: 额外元数据。
    """
    bits: np.ndarray
    energy: float
    is_feasible: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def num_variables(self) -> int:
        return len(self.bits)

    def __repr__(self) -> str:
        return f"Solution(energy={self.energy:.4f}, feasible={self.is_feasible})"


@dataclass
class SolverResult:
    """单次求解的完整结果。

    Attributes:
        solutions: 解列表 (按能量升序排列)。
        num_reads: 总采样数。
        solver_name: 求解器名称。
        timing_ms: 求解耗时 (毫秒)。
    """
    solutions: List[Solution]
    num_reads: int
    solver_name: str = "unknown"
    timing_ms: float = 0.0

    @property
    def best_solution(self) -> Solution | None:
        if self.solutions:
            return self.solutions[0]
        return None

    @property
    def best_energy(self) -> float | None:
        best = self.best_solution
        return best.energy if best else None

    def get_top_k(self, k: int = 10) -> List[Solution]:
        return self.solutions[:k]

    def filter_feasible(self) -> List[Solution]:
        return [s for s in self.solutions if s.is_feasible]


class AbstractSolver(ABC):
    """求解器抽象基类。

    所有求解器 (kaiwu 真机 / 内置经典 SA 模拟后端 / 第三方) 必须实现此接口。
    """

    @abstractmethod
    def solve(
        self,
        qubo_matrix: np.ndarray,
        num_reads: int = 1000,
    ) -> SolverResult:
        """求解 QUBO 问题。

        Args:
            qubo_matrix: shape=(N,N) QUBO 矩阵 (上三角格式, diag=h_i)。
            num_reads: 采样/退火次数。

        Returns:
            SolverResult 包含排序后的解列表。
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """求解器名称 (用于日志和结果标记)。"""
        ...

    @property
    def is_quantum(self) -> bool:
        """是否为量子硬件求解器。默认 False。"""
        return False
