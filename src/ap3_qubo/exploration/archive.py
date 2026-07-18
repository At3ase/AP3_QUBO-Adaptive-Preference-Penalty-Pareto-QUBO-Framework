"""
Pareto 存档管理器。

存储所有已探索解，维护非支配前沿缓存。
支持插入、剪枝、序列化/反序列化。
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

from ..validation.pareto import SolutionRecord, ParetoSort


class Archive:
    """ParetoZoom 探索存档。

    存储所有 SolutionRecord，自动维护非支配前沿。

    使用示例:
        >>> archive = Archive()
        >>> archive.insert_batch(solutions)
        >>> front = archive.front
        >>> obj_matrix = archive.get_objective_matrix()
    """

    def __init__(self):
        self._records: List[SolutionRecord] = []
        self._sorter = ParetoSort()
        self._front_cache: List[SolutionRecord] | None = None
        self._hv_history: List[float] = []

    # =========================================================================
    # 插入与维护
    # =========================================================================

    def insert(self, record: SolutionRecord) -> None:
        """插入一个解记录（使前沿缓存失效）。

        Args:
            record: 解记录。
        """
        self._records.append(record)
        self._front_cache = None

    def insert_batch(self, records: List[SolutionRecord]) -> None:
        """批量插入后剪枝（移除非支配解）。

        Args:
            records: 解记录列表。
        """
        self._records.extend(records)
        self._front_cache = None

    def prune(self) -> None:
        """移除非支配解，更新前沿缓存。"""
        if not self._records:
            self._front_cache = []
            return

        points = np.array([r.objectives for r in self._records])
        front_idx = self._sorter.pareto_front(points)
        self._records = [self._records[i] for i in front_idx]
        self._front_cache = self._records.copy()

    # =========================================================================
    # 前沿访问
    # =========================================================================

    @property
    def front(self) -> List[SolutionRecord]:
        """当前 Pareto 前沿（惰性计算 + 缓存）。

        Returns:
            非支配解列表。
        """
        if self._front_cache is None:
            self.prune()
        return self._front_cache or []

    @property
    def front_size(self) -> int:
        return len(self.front)

    @property
    def size(self) -> int:
        return len(self._records)

    def get_objective_matrix(self) -> np.ndarray:
        """返回前沿点的目标值矩阵 (N, 3)。"""
        front = self.front
        if not front:
            return np.empty((0, 3))
        return np.array([r.objectives for r in front])

    def get_objective_matrix_norm(self) -> np.ndarray:
        """返回前沿点的归一化目标值矩阵 (N, 3)。"""
        front = self.front
        if not front:
            return np.empty((0, 3))
        return np.array([r.objectives_norm for r in front])

    def get_weights_of_front(self) -> List[Tuple[float, float, float]]:
        """返回前沿各解对应的权重组。"""
        return [r.weights for r in self.front]

    def get_fractions_of_front(self) -> List[Dict[str, float]]:
        """返回前沿各解对应的成分字典。"""
        return [r.fractions for r in self.front]

    def all_records(self) -> List[SolutionRecord]:
        """返回所有记录（包括被支配的）。"""
        return self._records

    def get_top_k(self, k: int = 10) -> List[SolutionRecord]:
        """按 QUBO 能量升序返回 TOP-K 个记录。"""
        sorted_records = sorted(self._records, key=lambda r: r.energy)
        return sorted_records[:k]

    # =========================================================================
    # 权重查找
    # =========================================================================

    def find_nearest_weight(
        self, target_weight: Tuple[float, float, float]
    ) -> Optional[Tuple[float, float, float]]:
        """查找存档中与目标权重最近邻的权重组（用于 warm-start）。

        Args:
            target_weight: 目标权重组。

        Returns:
            最近邻权重组，或 None（存档为空）。
        """
        if not self._records:
            return None

        target = np.array(target_weight, dtype=float)
        best_dist = float("inf")
        best_weight = None

        for r in self._records:
            dist = np.linalg.norm(np.array(r.weights) - target)
            if dist < best_dist:
                best_dist = dist
                best_weight = r.weights

        return best_weight

    def get_explored_weights(self) -> List[Tuple[float, float, float]]:
        """返回所有已探索的权重组（去重）。"""
        seen = set()
        result = []
        for r in self._records:
            # 四舍五入到 3 位小数做去重
            key = tuple(round(w, 3) for w in r.weights)
            if key not in seen:
                seen.add(key)
                result.append(r.weights)
        return result

    # =========================================================================
    # HV 历史追踪
    # =========================================================================

    def record_hv(self, hv: float) -> None:
        """记录一轮的 HV 值。

        Args:
            hv: 当前 HV 值。
        """
        self._hv_history.append(hv)

    def get_hv_history(self) -> List[float]:
        """返回 HV 序列（用于收敛曲线）。"""
        return self._hv_history

    def get_latest_hv(self) -> float | None:
        """返回最新 HV 值。"""
        if self._hv_history:
            return self._hv_history[-1]
        return None

    # =========================================================================
    # 序列化
    # =========================================================================

    def to_dict(self) -> Dict:
        """序列化为字典（用于断点续跑）。

        Returns:
            可 JSON 序列化的字典（含 numpy 数组转换）。
        """
        return {
            "num_records": len(self._records),
            "hv_history": self._hv_history,
            "front_size": self.front_size,
        }

    def __len__(self) -> int:
        return len(self._records)

    def __repr__(self) -> str:
        return (
            f"Archive(records={len(self._records)}, "
            f"front={self.front_size}, "
            f"hv_rounds={len(self._hv_history)})"
        )
