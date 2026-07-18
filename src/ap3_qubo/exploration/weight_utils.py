"""
权重生成、微扰与去重工具。

ParetoZoom 的 Phase B/C 依赖此模块生成新权重组。
"""

from typing import List, Tuple

import numpy as np


def normalize_weights(w: Tuple[float, float, float] | np.ndarray) -> Tuple[float, float, float]:
    """归一化权重到和为 1。

    Args:
        w: (w1, w2, w3) 权重。

    Returns:
        归一化后的三元组。

    Raises:
        ValueError: 如果权重和 ≤ 0。
    """
    arr = np.array(w, dtype=float)
    s = arr.sum()
    if s <= 1e-12:
        raise ValueError(f"Weight sum must be > 0, got {s}")
    arr = arr / s
    return (float(arr[0]), float(arr[1]), float(arr[2]))


def clamp_weights(
    w: Tuple[float, float, float] | np.ndarray,
    min_val: float = 0.02,
) -> Tuple[float, float, float]:
    """将权重各分量限制在下界以上后重新归一化。

    Args:
        w: (w1, w2, w3) 权重。
        min_val: 每个权重分量的下界。

    Returns:
        限制并归一化后的三元组。
    """
    arr = np.array(w, dtype=float)
    arr = np.clip(arr, min_val, 1.0)
    return normalize_weights(arr)


def midpoint_weights(
    w_a: Tuple[float, float, float],
    w_b: Tuple[float, float, float],
) -> Tuple[float, float, float]:
    """计算两组权重的线性中点并重新归一化。

    Args:
        w_a: 第一组权重。
        w_b: 第二组权重。

    Returns:
        中点权重三元组。
    """
    arr = (np.array(w_a) + np.array(w_b)) / 2.0
    return normalize_weights(arr)


def deduplicate_weights(
    weights: List[Tuple[float, float, float]],
    tolerance: float = 0.05,
) -> List[Tuple[float, float, float]]:
    """移除欧氏距离 < tolerance 的重复权重组。

    Args:
        weights: 权重列表。
        tolerance: 重复判定距离阈值。

    Returns:
        去重后的权重列表（保持首次出现顺序）。
    """
    if not weights:
        return []

    unique: List[Tuple[float, float, float]] = []
    for w in weights:
        arr_w = np.array(w)
        is_dup = False
        for u in unique:
            if np.linalg.norm(arr_w - np.array(u)) < tolerance:
                is_dup = True
                break
        if not is_dup:
            unique.append(w)

    return unique


class WeightGenerator:
    """权重组生成器（用于 ParetoZoom Phase B 间隙填充和 Phase C 微扰）。

    使用示例:
        >>> gen = WeightGenerator(sigma=0.08, weight_min=0.02)
        >>> new_w = gen.perturb((0.5, 0.3, 0.2))
        >>> gaps = gen.from_gaps(front_weights, gap_pairs)
    """

    def __init__(
        self,
        sigma: float = 0.08,
        weight_min: float = 0.02,
        n_perturbations: int = 3,
    ):
        """
        Args:
            sigma: 高斯微扰标准差。
            weight_min: 权重分量下界。
            n_perturbations: 每个热点生成的微扰数。
        """
        self._sigma = sigma
        self._min = weight_min
        self._n_pert = n_perturbations

    def perturb(
        self, center: Tuple[float, float, float]
    ) -> Tuple[float, float, float]:
        """围绕中心权重生成一个高斯微扰后的新权重。

        对每个分量加 N(0, σ²) 噪声，然后限制下界并重新归一化。

        Args:
            center: 中心权重 (w1, w2, w3)。

        Returns:
            微扰后的归一化权重。
        """
        arr = np.array(center, dtype=float)
        noise = np.random.normal(0.0, self._sigma, size=3)
        perturbed = arr + noise
        # 限制下界
        perturbed = np.clip(perturbed, self._min, 1.0)
        return normalize_weights(perturbed)

    def perturb_batch(
        self, center: Tuple[float, float, float]
    ) -> List[Tuple[float, float, float]]:
        """为单个中心生成 n_perturbations 个微扰权重。

        Args:
            center: 中心权重。

        Returns:
            微扰后的归一化权重列表。
        """
        weights = [self.perturb(center) for _ in range(self._n_pert)]
        return deduplicate_weights(weights, tolerance=0.05)

    def perturb_hotspots(
        self,
        front_weights: List[Tuple[float, float, float]],
        hv_contributions: List[float] | None = None,
    ) -> List[Tuple[float, float, float]]:
        """对 Pareto 前沿上的所有权重（优先高 HV 贡献）生成微扰。

        Args:
            front_weights: 前沿上各解对应的权重。
            hv_contributions: 可选，各权重的 HV 边际贡献，用于加权采样。

        Returns:
            微扰后的去重权重列表。
        """
        if not front_weights:
            return []

        all_perturbed: List[Tuple[float, float, float]] = []

        if hv_contributions is not None and len(hv_contributions) == len(front_weights):
            # 按 HV 贡献加权：高贡献 → 更多微扰
            contribs = np.array(hv_contributions, dtype=float)
            contribs = np.clip(contribs, 0.0, None)
            total = contribs.sum()
            if total > 1e-12:
                probs = contribs / total
                # 对高贡献权重增加微扰数
                for w, prob in zip(front_weights, probs):
                    n = max(1, int(prob * self._n_pert * len(front_weights)))
                    for _ in range(min(n, self._n_pert)):
                        all_perturbed.append(self.perturb(w))
            else:
                for w in front_weights:
                    all_perturbed.extend(self.perturb_batch(w))
        else:
            for w in front_weights:
                all_perturbed.extend(self.perturb_batch(w))

        return deduplicate_weights(all_perturbed, tolerance=0.05)

    def from_gaps(
        self,
        front_weights: List[Tuple[float, float, float]],
        gap_pairs: List[Tuple[int, int]],
    ) -> List[Tuple[float, float, float]]:
        """从检测到的间隙生成填充权重（中点插值）。

        Args:
            front_weights: 前沿权重列表。
            gap_pairs: [(idx_a, idx_b), ...] 间隙对。

        Returns:
            中点权重的去重列表。
        """
        new_weights = []
        for ia, ib in gap_pairs:
            if 0 <= ia < len(front_weights) and 0 <= ib < len(front_weights):
                mid = midpoint_weights(front_weights[ia], front_weights[ib])
                new_weights.append(mid)
        return deduplicate_weights(new_weights)

    @property
    def sigma(self) -> float:
        return self._sigma

    @property
    def weight_min(self) -> float:
        return self._min
