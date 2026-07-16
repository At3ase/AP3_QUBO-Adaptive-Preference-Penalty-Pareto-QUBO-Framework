"""
kaiwu SDK 求解器实现。

封装 kaiwu SDK 的 QuboModel 构建 + QuboSolver 求解接口。
后端可为 CIM 光量子真机或 kaiwu 云端模拟器。

支持两种变量命名格式:
  - 构建器格式: e0_b0, e0_b1, ..., e5_b2 (QUBOBuilder)
  - 矩阵格式:   b[0], b[1], ..., b[N-1] (qubo_matrix_to_qubo_model)
"""

import re
import time
from typing import Dict, Any, List

import numpy as np

from .base import AbstractSolver, SolverResult, Solution


class KaiwuSolver(AbstractSolver):
    """kaiwu SDK 求解器。

    支持三种模式:
      - "auto": 自动选择可用后端
      - "cim": CIM 光量子真机
      - "simulator": kaiwu 云端模拟器

    若后端不可用，solve() 将抛出 RuntimeError。
    """

    # 变量名解析正则
    _VAR_BRACKET = re.compile(r"b\[(\d+)\]")       # b[0], b[1], ...
    _VAR_BUILDER = re.compile(r"e(\d+)_b(\d+)")     # e0_b0, e1_b3, ...

    def __init__(self, mode: str = "auto", **kwargs):
        """
        Args:
            mode: 求解模式 ("auto" / "cim" / "simulator")。
            **kwargs: 其他 kaiwu 配置参数。
        """
        self._mode = mode
        self._config = kwargs

    @property
    def name(self) -> str:
        return f"kaiwu-{self._mode}"

    @property
    def is_quantum(self) -> bool:
        return self._mode == "cim"

    def solve_from_model(
        self,
        model,
        n_vars: int = 38,
    ) -> SolverResult:
        """从 kaiwu QuboModel 求解 (推荐方式)。

        使用 QUBOBuilder.build_model() 构建的模型直接传入。
        变量名格式: e0_b0, ..., e5_b2。

        Args:
            model: kaiwu QuboModel。
            n_vars: 变量总数。

        Returns:
            SolverResult。
        """
        import kaiwu as kw

        t_start = time.perf_counter()

        solver = kw.QuboSolver()
        solution_dict, energy_val = solver.solve_qubo(model)

        t_end = time.perf_counter()

        if solution_dict is None:
            return SolverResult(
                solutions=[],
                num_reads=1,
                solver_name=self.name,
                timing_ms=(t_end - t_start) * 1000,
            )

        bits = self._parse_solution(solution_dict, n_vars)
        energy = float(energy_val) if energy_val is not None else 0.0

        solution = Solution(
            bits=bits,
            energy=energy,
            is_feasible=True,
            metadata={"solver": self.name, "mode": self._mode},
        )

        return SolverResult(
            solutions=[solution],
            num_reads=1,
            solver_name=self.name,
            timing_ms=(t_end - t_start) * 1000,
        )

    def solve(
        self,
        qubo_matrix: np.ndarray,
        num_reads: int = 1000,
    ) -> SolverResult:
        """从 numpy QUBO 矩阵求解 (兼容旧接口)。

        使用 qubo_matrix_to_qubo_model 转换矩阵 → 求解。
        变量名格式: b[0], b[1], ..., b[N-1]。

        Args:
            qubo_matrix: shape=(N,N) QUBO 矩阵 (上三角, diag=h_i)。
            num_reads: 采样数。

        Returns:
            SolverResult。
        """
        import kaiwu as kw

        n = qubo_matrix.shape[0]
        t_start = time.perf_counter()

        model = kw.qubo_matrix_to_qubo_model(qubo_matrix)

        solver = kw.QuboSolver()
        solution_dict, energy_val = solver.solve_qubo(model)

        t_end = time.perf_counter()

        if solution_dict is None:
            return SolverResult(
                solutions=[],
                num_reads=num_reads,
                solver_name=self.name,
                timing_ms=(t_end - t_start) * 1000,
            )

        bits = self._parse_solution(solution_dict, n)
        energy = float(energy_val) if energy_val is not None else 0.0

        solution = Solution(
            bits=bits,
            energy=energy,
            is_feasible=True,
            metadata={"solver": self.name, "mode": self._mode},
        )

        return SolverResult(
            solutions=[solution],
            num_reads=num_reads,
            solver_name=self.name,
            timing_ms=(t_end - t_start) * 1000,
        )

    def _parse_solution(
        self, solution_dict: Dict, n_vars: int
    ) -> np.ndarray:
        """解析 kaiwu 返回的解字典 → numpy 比特数组。

        自动检测变量命名格式:
          - e0_b0 格式 (QUBOBuilder)
          - b[0] 格式 (qubo_matrix_to_qubo_model)
        """
        bits = np.zeros(n_vars, dtype=np.int8)

        for var_name, val in solution_dict.items():
            var_str = str(var_name)
            idx = self._parse_var_index(var_str, n_vars)
            if idx is not None:
                bits[idx] = int(val)

        return bits

    @classmethod
    def _parse_var_index(cls, var_str: str, n_vars: int) -> int | None:
        """从变量名字符串解析 flat index。

        Returns:
            flat index (0 ~ n_vars-1), 或 None (无法解析)。
        """
        # 尝试 builder 格式: e{ei}_b{bj}
        m = cls._VAR_BUILDER.match(var_str)
        if m:
            ei, bj = int(m.group(1)), int(m.group(2))
            if ei < 5:
                idx = ei * 7 + bj
            else:
                idx = 35 + bj
            return idx if idx < n_vars else None

        # 尝试 bracket 格式: b[N]
        m = cls._VAR_BRACKET.match(var_str)
        if m:
            idx = int(m.group(1))
            return idx if idx < n_vars else None

        return None

    @staticmethod
    def is_available() -> bool:
        """检查 kaiwu SDK 后端是否可用。"""
        try:
            solver = KaiwuSolver(mode="auto")
            mat = np.array([[1.0, 0.0], [0.0, 1.0]])
            result = solver.solve(mat, num_reads=1)
            return len(result.solutions) > 0
        except (NotImplementedError, RuntimeError):
            return False
        except Exception:
            return False
