"""
Hypervolume（HV）计算器。

用于 3 目标 Pareto 前沿质量评估。
使用递归切片算法（LebMeasure），精确计算 HV。

HV = 被 Pareto 前沿支配的参考点以上区域的体积。
值越大，前沿质量越高（覆盖更广、更接近理想值）。
"""

from typing import Dict, List, Optional, Tuple

import numpy as np


def set_unified_reference(
    archives_dict: Dict[str, np.ndarray],
    margin: float = 0.10,
) -> np.ndarray:
    """P0-5 修复（审查报告 Code_Completion_Review_2026-07-18）：
    合并所有方法的解集后统一定参考点，保证跨方法 HV 可比。

    HV 对比的前提是固定参考点（方案 HV-1）。若每个方法对自己的
    archive 单独 set_reference_from_data，组间 HV 差异会混入参考点
    漂移而非前沿质量。本函数将全部方法的目标矩阵合并为一个大集合，
    按"nadir + margin × range"（零 range 走相对兜底，同
    set_reference_from_data 的修正逻辑）计算唯一参考点。

    Args:
        archives_dict: {方法标签: shape=(N, 3) 目标值矩阵}。
                       空矩阵（N=0）自动跳过。
        margin: 边距因子（0.10 = 10%）。

    Returns:
        shape=(3,) 的统一参考点，可直接传给
        ``HypervolumeCalculator(reference_point=...)``。

    Raises:
        ValueError: 所有方法的解集均为空时抛出。
    """
    mats = [np.asarray(m, dtype=float) for m in archives_dict.values() if len(m) > 0]
    if not mats:
        raise ValueError(
            "set_unified_reference: 所有方法的解集均为空，无法定参考点。"
        )
    merged = np.vstack(mats)
    nadir = np.max(merged, axis=0)
    ranges = nadir - np.min(merged, axis=0)
    margins = ranges * margin
    fallback = np.abs(nadir) * margin
    fallback = np.where(fallback < 1e-12, margin, fallback)
    margins = np.where(margins < 1e-12, fallback, margins)
    return nadir + margins


def compute_feasible_hv(
    objectives: np.ndarray,
    feasible_mask: np.ndarray,
    reference_point: np.ndarray,
) -> float:
    """feasible-HV：仅对前沿中通过硬口径物理过滤的子集计算 HV。

    任务 C（2026-07-19，诊断报告 reports/feasible_hv_diagnostic_2026-07-19）
    新增，不改动本文件任何现有函数行为。

    口径约定：
      - 参考点必须由调用方在「所有配置的合并前沿」上用
        set_unified_reference(margin=0.10) 定标后传入（方案 HV-1
        固定参考点前提，保证组间可比）；本函数不自行定标。
      - 可行子集为空时返回 NaN（指标不可计算，须与 0.0 区分：
        0.0 表示可行点存在但对参考点无支配体积，NaN 表示无可行点）。
      - 可行点超出参考点的部分由 HypervolumeCalculator.compute
        内部按既有口径过滤。

    Args:
        objectives: shape=(N, 3) 目标值矩阵（某配置某 rep 的 Pareto 前沿）。
        feasible_mask: shape=(N,) 布尔掩码，True = 通过硬口径过滤。
        reference_point: shape=(3,) 统一参考点。

    Returns:
        可行子集的 HV；无可行点或输入为空时返回 NaN。
    """
    pts = np.asarray(objectives, dtype=float)
    mask = np.asarray(feasible_mask, dtype=bool)
    if pts.ndim != 2 or len(pts) == 0 or len(mask) != len(pts):
        return float("nan")
    feasible = pts[mask]
    if len(feasible) == 0:
        return float("nan")
    calc = HypervolumeCalculator(
        reference_point=np.asarray(reference_point, dtype=float)
    )
    return float(calc.compute(feasible))


class HypervolumeCalculator:
    """3 目标最小化问题的 Hypervolume 计算器。

    HV = volume({q ∈ R³ | ∃p ∈ P: p ≤ q ≤ ref})

    使用示例:
        >>> calc = HypervolumeCalculator()
        >>> calc.set_reference_from_data(points)
        >>> hv = calc.compute(points)
    """

    def __init__(self, reference_point: np.ndarray | None = None):
        """
        Args:
            reference_point: shape=(3,) 参考点。默认从数据自动设置。
        """
        self._ref = np.array(reference_point, dtype=float) if reference_point is not None else None

    def set_reference_from_data(
        self, points: np.ndarray, margin_factor: float = 0.10
    ) -> None:
        """从数据设置参考点：每目标最大值 + margin。

        Args:
            points: shape=(N, 3) 目标值。
            margin_factor: 边距因子 (0.10 = 10%)。
        """
        nadir = np.max(points, axis=0)
        ranges = np.max(points, axis=0) - np.min(points, axis=0)
        margins = ranges * margin_factor
        # P0-5 附带修复（审查报告 Code_Completion_Review_2026-07-18，
        # 原 hypervolume.py:46）：零 range 目标原先用 1.0 绝对值兜底，
        # 对成本（~10²）尺度过小、对密度（~7）尺度过大，量级失真。
        # 改为相对兜底：取该目标 nadir 绝对值的 margin_factor 倍；
        # 若 nadir 也接近 0（目标值全在 0 附近），退化为 margin_factor 本身。
        fallback = np.abs(nadir) * margin_factor
        fallback = np.where(fallback < 1e-12, margin_factor, fallback)
        margins = np.where(margins < 1e-12, fallback, margins)
        self._ref = nadir + margins

    @property
    def reference_point(self) -> np.ndarray | None:
        # 未设置时返回 None（调用方如 ParetoZoom 依赖 `is None` 判断），
        # 需要真实参考点的计算路径（compute 等）自行做空值守卫。
        return self._ref

    def compute(self, points: np.ndarray) -> float:
        """计算 HV。

        Args:
            points: shape=(N, 3) 目标值矩阵。

        Returns:
            HV 值。若 points 为空返回 0.0。
        """
        if len(points) == 0:
            return 0.0

        ref = self.reference_point
        if ref is None:
            raise ValueError("Reference point not set. Call set_reference_from_data first.")

        # 只保留非支配解
        front = self._filter_nondominated(points)

        # 过滤掉超出参考点的解
        valid = np.all(front <= ref, axis=1)
        front = front[valid]

        if len(front) == 0:
            return 0.0

        # 3D HV 递归算法
        return self._hv3d_recursive(front, ref)

    def compute_delta(
        self,
        hv_before: float,
        hv_after: float,
    ) -> float:
        """计算相对 HV 增长。

        Args:
            hv_before: 之前 HV。
            hv_after: 之后 HV。

        Returns:
            (HV_after - HV_before) / HV_before，若 hv_before=0 返回 1.0。
        """
        if hv_before < 1e-12:
            return 1.0 if hv_after > 1e-12 else 0.0
        return (hv_after - hv_before) / hv_before

    def marginal_contribution(
        self, points: np.ndarray, idx: int
    ) -> float:
        """计算移除第 idx 个点导致的 HV 下降量。

        Args:
            points: shape=(N, 3) 目标值。
            idx: 要评估的点索引。

        Returns:
            边际贡献 = HV(全集) - HV(去除此点)。
        """
        if len(points) <= 1:
            return 0.0

        hv_full = self.compute(points)
        mask = np.ones(len(points), dtype=bool)
        mask[idx] = False
        hv_without = self.compute(points[mask])
        return max(0.0, hv_full - hv_without)

    def compute_c_metric(
        self, set_a: np.ndarray, set_b: np.ndarray
    ) -> float:
        """C-metric: B 中被 A 中的至少一个解支配的比例。

        C(A, B) = |{b ∈ B | ∃a ∈ A: a dominates b}| / |B|

        Args:
            set_a: shape=(NA, 3) 方法 A 的解集。
            set_b: shape=(NB, 3) 方法 B 的解集。

        Returns:
            C(A, B) ∈ [0, 1]，越高表示 A 越好。
        """
        if len(set_b) == 0:
            return 0.0

        # A 的 Pareto 前沿
        front_a = self._filter_nondominated(set_a)

        dominated_count = 0
        for b in set_b:
            for a in front_a:
                if np.all(a <= b) and np.any(a < b):
                    dominated_count += 1
                    break

        return dominated_count / len(set_b)

    # =========================================================================
    # 内部方法
    # =========================================================================

    @staticmethod
    def _filter_nondominated(points: np.ndarray) -> np.ndarray:
        """快速过滤出非支配解。"""
        if len(points) <= 1:
            return points

        n = len(points)
        dominated = np.zeros(n, dtype=bool)

        for i in range(n):
            if dominated[i]:
                continue
            for j in range(n):
                if i == j or dominated[j]:
                    continue
                if np.all(points[i] <= points[j]) and np.any(points[i] < points[j]):
                    dominated[j] = True
                elif np.all(points[j] <= points[i]) and np.any(points[j] < points[i]):
                    dominated[i] = True
                    break

        return points[~dominated]

    @staticmethod
    def _hv3d_recursive(points: np.ndarray, ref: np.ndarray) -> float:
        """3D 递归 HV 计算（LebMeasure 算法）。

        对 3 目标问题:
          1. 按 f1 降序排列
          2. 递归计算 2D 面积，按 f1 间距加权

        Args:
            points: shape=(N, 3) 已过滤的非支配解。
            ref: shape=(3,) 参考点。

        Returns:
            HV 值。
        """
        n = len(points)
        if n == 0:
            return 0.0

        # 按第一个目标降序排列
        order = np.argsort(-points[:, 0])
        sorted_pts = points[order]

        hv = 0.0
        # 对于 2 目标子问题的参考线
        prev_f1 = ref[0]

        for i in range(n):
            # 当前点贡献的 f1 切片体积
            f1_slice = prev_f1 - sorted_pts[i, 0]

            if f1_slice > 0:
                # 计算剩余点的 2D 面积 (f2, f3)
                remaining = sorted_pts[i:, 1:3]
                area_2d = HypervolumeCalculator._area2d(remaining, ref[1:3])
                hv += f1_slice * area_2d

            prev_f1 = sorted_pts[i, 0]

        return hv

    @staticmethod
    def _area2d(points_2d: np.ndarray, ref_2d: np.ndarray) -> float:
        """计算 2D 被支配面积（用于 3D HV 的递归基）。

        算法（Validation_Auditor 修复，原实现系统性低估）：
        按 f2 升序扫描，切片 x ∈ [f2_i, f2_{i+1}] 的覆盖高度
        = ref3 − min{f3 : f2 ≤ f2_i}（running-min 必须取自已处理
        的"更小 f2"点集，即切片支配集）。原实现按 f2 降序 + 前缀
        running-min，把高度算成"相邻前沿点之间"的阶梯差，凡 2D
        切片集合不是单点全局最小 f3 即漏算矩形条带（手算案例
        {(1,2),(2,1)}, ref(3,3)：正确 3.0，旧实现 2.0；3D 手算
        案例 13.0 → 8.0；随机前沿暴力网格交叉验证低估 10%~50%）。

        Args:
            points_2d: shape=(N, 2) (f2, f3) 值。
            ref_2d: shape=(2,) 参考点。

        Returns:
            被支配的 2D 面积。
        """
        if len(points_2d) == 0:
            return 0.0

        # 过滤
        valid = np.all(points_2d <= ref_2d, axis=1)
        pts = points_2d[valid]
        if len(pts) == 0:
            return 0.0

        # 按 f2 升序排列
        order = np.argsort(pts[:, 0])
        pts = pts[order]

        area = 0.0
        min_f3 = ref_2d[1]  # running-min：已处理点集（f2 ≤ 当前切片）的最小 f3

        for i in range(len(pts)):
            min_f3 = min(min_f3, pts[i, 1])
            # 切片右端：下一个点的 f2，末尾延伸到 ref2
            right = pts[i + 1, 0] if i + 1 < len(pts) else ref_2d[0]
            width = right - pts[i, 0]
            height = ref_2d[1] - min_f3
            if width > 0 and height > 0:
                area += width * height

        return area
