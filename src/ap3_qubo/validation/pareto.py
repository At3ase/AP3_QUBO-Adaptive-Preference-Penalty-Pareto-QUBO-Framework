"""
Pareto 非支配排序与存档管理。

支持 3 目标最小化的快速非支配排序。
提供 SolutionRecord 统一数据类和 ParetoSort 排序器。
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class SolutionRecord:
    """Pareto 分析的统一解记录。

    Attributes:
        fractions: 元素 → at% 字典。
        bits: 38 维二元向量 (0/1)。
        objectives: (f1_raw, f2_raw, f3_raw) 原始尺度目标值。
        objectives_norm: (f1_norm, f2_norm, f3_norm) 归一化目标值。
        weights: (w1, w2, w3) 生成该解使用的权重。
        lambdas: (λ_carbide, λ_ccr) 求解时的 λ 值。
        energy: QUBO 能量值。
        round_id: ParetoZoom 轮数。
        is_quantum: 是否由 CIM 真机后端产生（内置 SA 后端为 False）。
        metadata: 额外元数据。
    """
    fractions: Dict[str, float]
    bits: np.ndarray
    objectives: Tuple[float, float, float]
    objectives_norm: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    weights: Tuple[float, float, float] = (0.33, 0.33, 0.34)
    lambdas: Tuple[float, float] = (0.05, 0.05)
    energy: float = 0.0
    round_id: int = 0
    is_quantum: bool = True
    metadata: Dict = field(default_factory=dict)

    def objectives_array(self) -> np.ndarray:
        """返回目标值数组 shape=(3,)。"""
        return np.array(self.objectives, dtype=float)

    def objectives_norm_array(self) -> np.ndarray:
        """返回归一化目标值数组 shape=(3,)。"""
        return np.array(self.objectives_norm, dtype=float)

    def __repr__(self) -> str:
        o = self.objectives
        return (
            f"SolutionRecord(f1={o[0]:.2f}, f2={o[1]:.2f}, "
            f"f3={o[2]:.1f}, energy={self.energy:.4f})"
        )


class ParetoSort:
    """3 目标 Pareto 非支配排序器。

    解 A 支配解 B 当且仅当:
      - A_i ≤ B_i 对所有 i ∈ {0,1,2}
      - A_i < B_i 对至少一个 i

    使用示例:
        >>> sorter = ParetoSort()
        >>> points = np.array([[1.0, 2.0, 3.0], [0.5, 1.5, 4.0], [1.2, 2.1, 2.8]])
        >>> front = sorter.pareto_front(points)
    """

    def is_dominated(self, a: np.ndarray, b: np.ndarray) -> bool:
        """判断 A 是否支配 B（A 在所有目标上 ≤ B 且至少一个严格 <）。"""
        return bool(np.all(a <= b) and np.any(a < b))

    def pareto_front(self, points: np.ndarray) -> np.ndarray:
        """返回 Pareto 前沿点的索引。

        Args:
            points: shape=(N, M) 目标值矩阵（全部最小化）。

        Returns:
            Pareto 最优解的索引数组。

        算法: O(N²)，对 N < 1000 足够高效。
        """
        if len(points) == 0:
            return np.array([], dtype=int)

        n = len(points)
        dominated = np.zeros(n, dtype=bool)

        for i in range(n):
            if dominated[i]:
                continue
            for j in range(n):
                if i == j or dominated[j]:
                    continue
                if self.is_dominated(points[i], points[j]):
                    dominated[j] = True
                elif self.is_dominated(points[j], points[i]):
                    dominated[i] = True
                    break

        return np.where(~dominated)[0]

    def non_dominated_sort(self, points: np.ndarray) -> List[List[int]]:
        """完整非支配排序，返回所有前沿层。

        Args:
            points: shape=(N, M) 目标值矩阵。

        Returns:
            前沿列表，fronts[0] 为 Pareto 前沿。
        """
        if len(points) == 0:
            return []

        n = len(points)
        # 支配计数 + 被支配列表
        domination_count = np.zeros(n, dtype=int)
        dominated_by = {i: [] for i in range(n)}

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if self.is_dominated(points[i], points[j]):
                    dominated_by[i].append(j)
                elif self.is_dominated(points[j], points[i]):
                    domination_count[i] += 1

        # 第一前沿
        fronts = []
        current_front = [i for i in range(n) if domination_count[i] == 0]

        while current_front:
            fronts.append(current_front)
            next_front = []
            for i in current_front:
                for j in dominated_by[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        next_front.append(j)
            current_front = next_front

        return fronts

    def update_archive(
        self,
        archive: List[SolutionRecord],
        new_solutions: List[SolutionRecord],
    ) -> List[SolutionRecord]:
        """将新解加入存档并移除非支配解。

        Args:
            archive: 现有存档。
            new_solutions: 新解列表。

        Returns:
            更新后的存档（仅保留非支配解）。
        """
        combined = archive + new_solutions
        if not combined:
            return []

        points = np.array([s.objectives for s in combined])
        front_idx = self.pareto_front(points)
        return [combined[i] for i in front_idx]

    @staticmethod
    def objective_distance(
        a: np.ndarray,
        b: np.ndarray,
        normalizer: Optional[object] = None,
    ) -> float:
        """计算目标空间中两点之间的欧氏距离。

        Args:
            a: shape=(3,) 目标值。
            b: shape=(3,) 目标值。
            normalizer: 可选归一化器（需有 normalize 方法）。

        Returns:
            欧氏距离。
        """
        if normalizer is not None and hasattr(normalizer, 'normalize'):
            a_n = normalizer.normalize(a)
            b_n = normalizer.normalize(b)
            return float(np.linalg.norm(a_n - b_n))
        return float(np.linalg.norm(a - b))

    def find_gaps(
        self,
        front_points: np.ndarray,
        threshold_factor: float = 0.15,
    ) -> List[Tuple[int, int, float]]:
        """检测 Pareto 前沿中相邻点之间的间隙。

        Args:
            front_points: shape=(N, 3) Pareto 前沿目标值（假设已排序）。
            threshold_factor: 间隙阈值因子（× 最大边长）。

        Returns:
            [(idx_a, idx_b, distance), ...] 超过阈值的间隙列表。
        """
        if len(front_points) < 2:
            return []

        # 按 f1 排序
        order = np.argsort(front_points[:, 0])
        sorted_pts = front_points[order]

        # 计算相邻点间距离
        gaps = []
        all_dists = []
        for i in range(len(sorted_pts) - 1):
            dist = float(np.linalg.norm(sorted_pts[i + 1] - sorted_pts[i]))
            all_dists.append(dist)

        if not all_dists:
            return []

        max_dist = max(all_dists)
        threshold = threshold_factor * max_dist

        # 额外使用绝对最小阈值防止除零
        abs_min = 0.01
        threshold = max(threshold, abs_min)

        for i, dist in enumerate(all_dists):
            if dist > threshold:
                gaps.append((int(order[i]), int(order[i + 1]), dist))

        return gaps
