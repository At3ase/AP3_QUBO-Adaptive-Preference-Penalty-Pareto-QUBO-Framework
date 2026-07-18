"""
γ 敏感性分析（实验 4）。

测试碳折扣因子 γ ∈ {0.1, 0.2, 0.25, 0.3, 0.4, 0.5} 对结果稳定性的影响。
每个 γ 值独立运行 ParetoZoom，γ 通过 QUBOBuilder.gamma_discount 传入。

预期: Pareto 前沿形状稳健，HV 偏差 < 10%，TOP10 解一致性 > 60%。
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from ..physical_params import PARETO_ZOOM, ALL_ELEMENTS
from ..exploration.pareto_zoom import ParetoZoom
from ..validation.hypervolume import HypervolumeCalculator, set_unified_reference
from ..statistics.reporting import ExperimentStats


@dataclass
class SensitivityResult:
    """单个 γ 值的敏感性分析结果。

    Attributes:
        gamma: 碳折扣因子。
        hv_mean: 平均 HV。
        hv_std: HV 标准差。
        front_size_mean: 平均前沿大小。
        top10_overlap: 与基准 (γ=0.25) 的 TOP10 解重叠率。
            P0-10 修复后：真实的解集重叠率 —— 两组成分快照按
            成分空间容差（各元素 at% 差均 < tolerance_at）一对一
            贪婪匹配，重叠率 = |交集| / 基准 TOP10 大小，跨重复取均值。
            基准自身恒为 1.0（与自身重叠，定义性取值）。
        top10_snapshots: 每次重复的 TOP10 成分快照
            （List[ repetition -> List[ {元素: at%} ] ]），
            按 Archive.get_top_k(10)（QUBO 能量升序）采集。
    """
    gamma: float
    hv_mean: float = 0.0
    hv_std: float = 0.0
    front_size_mean: float = 0.0
    top10_overlap: float = 0.0
    hv_values: List[float] = field(default_factory=list)
    top10_snapshots: List[List[Dict[str, float]]] = field(default_factory=list)


class SensitivityAnalyzer:
    """γ 敏感性分析器。

    测试 γ ∈ {0.1, 0.2, 0.25, 0.3, 0.4, 0.5}，
    基准 γ = 0.25（MIEDEMA 默认值）。

    每个 γ 值通过 QUBOBuilder(gamma_discount=γ) 传入，
    ParetoZoom 内循环使用该 γ 构建 C-主元交叉项。

    使用示例:
        >>> analyzer = SensitivityAnalyzer()
        >>> results = analyzer.run(n_repetitions=10)
        >>> analyzer.report(results)
    """

    def __init__(
        self,
        gamma_values: List[float] | None = None,
        tolerance_at: float = 0.5,
    ):
        """
        Args:
            gamma_values: 测试的 γ 值列表
                          （默认 [0.1, 0.2, 0.25, 0.3, 0.4, 0.5]）。
            tolerance_at: TOP10 重叠判定的成分容差 (at%)。
                          两个解在所有元素上的 at% 差均 < tolerance_at
                          时视为同一解（默认 0.5 at%，即 2 个编码步长，
                          对应 PrecisionSplit 统一步长 0.25 at%，
                          见方案 v1.13 §2 PrecisionSplit 编码）。
        """
        self._gamma_values = gamma_values or [0.1, 0.2, 0.25, 0.3, 0.4, 0.5]
        self._baseline_gamma = 0.25
        self._tolerance_at = tolerance_at

    def run(
        self,
        n_repetitions: int = 10,
        seed: int = 42,
    ) -> Dict[float, SensitivityResult]:
        """运行敏感性分析。

        对每个 γ 值，创建带有该 γ 折扣因子的 QUBOBuilder，
        通过 ParetoZoom(gamma_discount=γ) 运行完整探索流程。

        Args:
            n_repetitions: 每个 γ 值的重复次数。
            seed: 随机种子基数。

        Returns:
            {gamma: SensitivityResult}。
        """
        results: Dict[float, SensitivityResult] = {}
        # P0-5 修复（审查报告 Code_Completion_Review_2026-07-18）：
        # 原实现在每次重复内对单个 archive 单独 set_reference_from_data
        # （原 :150-154），各 γ 之间参考点漂移，跨 γ 的 HV 偏差结论
        # 失真。现先收集全部 γ × 全部重复的目标矩阵，再统一设定
        # 参考点后计算 HV。
        obj_mats: Dict[float, List[np.ndarray]] = {}

        for gamma in self._gamma_values:
            sr = SensitivityResult(gamma=gamma)
            obj_mats[gamma] = []
            for rep in range(n_repetitions):
                np.random.seed(seed + int(gamma * 100) + rep)
                try:
                    # 核心修复：通过 gamma_discount 参数真正切换 γ
                    # B-1：seed 同步贯通到求解器（与 np.random 同一基数）
                    pz = ParetoZoom(gamma_discount=gamma,
                                    seed=seed + int(gamma * 100) + rep)
                    archive, _ = pz.run()
                    obj_mats[gamma].append(archive.get_objective_matrix())
                    # P0-10 修复（审查报告 Code_Completion_Review_2026-07-18）：
                    # 采集本次重复的 TOP10 成分快照，供真实重叠率计算。
                    # 排序准则与 Archive.get_top_k 一致（QUBO 能量升序），
                    # 基准与各 γ 使用同一准则，保证可比性。
                    sr.top10_snapshots.append(
                        [dict(r.fractions) for r in archive.get_top_k(10)]
                    )
                except (NotImplementedError, RuntimeError):
                    obj_mats[gamma].append(np.zeros((0, 3)))

            results[gamma] = sr

        # 统一参考点：合并全部 γ 的解集后定 ref，再逐矩阵计算 HV。
        all_mats = {
            f"γ={gamma}#rep{i}": mat
            for gamma, mats in obj_mats.items()
            for i, mat in enumerate(mats)
        }
        if any(len(m) > 0 for m in all_mats.values()):
            ref = set_unified_reference(all_mats, margin=0.10)
            hv_calc = HypervolumeCalculator(reference_point=ref)
            for gamma in self._gamma_values:
                results[gamma].hv_values = [
                    hv_calc.compute(mat) if len(mat) > 0 else 0.0
                    for mat in obj_mats[gamma]
                ]

        for gamma in self._gamma_values:
            sr = results[gamma]
            if sr.hv_values:
                vals = np.array(sr.hv_values)
                sr.hv_mean = float(np.mean(vals[vals > 0])) if any(vals > 0) else 0.0
                sr.hv_std = float(np.std(vals[vals > 0])) if any(vals > 0) else 0.0

        # P0-10 修复：计算真实的 TOP10 解重叠率（基准 γ=0.25）。
        # 旧实现用 min(hv_mean/baseline_hv, 1.0) 冒充重叠率，名实不符；
        # 现改为成分空间容差匹配（各元素 at% 差均 < tolerance_at 视为
        # 同一解），重叠率 = |交集| / 基准 TOP10 大小，跨重复取均值。
        # 对应方案 v1.13 验收预期"TOP10 解一致性 > 60%"。
        baseline = results.get(self._baseline_gamma)
        if baseline and baseline.top10_snapshots:
            for gamma, sr in results.items():
                if gamma == self._baseline_gamma:
                    sr.top10_overlap = 1.0  # 与自身重叠，定义性取值
                elif sr.top10_snapshots:
                    sr.top10_overlap = self._mean_top10_overlap(
                        sr.top10_snapshots,
                        baseline.top10_snapshots,
                        self._tolerance_at,
                    )

        return results

    def report(
        self, results: Dict[float, SensitivityResult]
    ) -> str:
        """生成敏感性分析报告（Markdown 表格）。

        Args:
            results: run() 输出。

        Returns:
            Markdown 格式的报告。
        """
        lines = [
            "## 实验 4: γ 敏感性分析",
            "",
            "| γ | HV (Mean ± SD) | vs Baseline (γ=0.25) | TOP10 重叠率 |",
            "|---|-----------------|----------------------|--------------|",
        ]
        baseline_hv = results.get(self._baseline_gamma, SensitivityResult(gamma=0.25)).hv_mean

        for gamma in sorted(results.keys()):
            sr = results[gamma]
            ratio = (sr.hv_mean / baseline_hv - 1.0) * 100 if baseline_hv > 1e-12 else 0.0
            marker = " ✅ 基准" if gamma == self._baseline_gamma else ""
            lines.append(
                f"| {gamma:.2f} | {sr.hv_mean:.4f} ± {sr.hv_std:.4f} | "
                f"{ratio:+.1f}%{marker} | {sr.top10_overlap:.0%} |"
            )

        lines.append("")
        lines.append(
            "**结论**: 若各 γ 值 HV 偏差 < 10% 且 TOP10 重叠率 > 60%，"
            "则结果对碳折扣因子不敏感。"
        )
        lines.append(
            f"（TOP10 重叠率：与基准 γ=0.25 的成分空间容差匹配，"
            f"容差 {self._tolerance_at} at%，跨重复取均值。）"
        )
        return "\n".join(lines)

    # NOTE(P0-5)：原 _compute_hv 对单个 archive 单独定参考点，
    # 已随统一参考点改造移除；HV 计算收敛到 run() 内的
    # set_unified_reference 统一路径。

    # =========================================================================
    # P0-10：TOP10 解重叠率（成分空间容差匹配）
    # =========================================================================

    @staticmethod
    def _compositions_match(
        frac_a: Dict[str, float],
        frac_b: Dict[str, float],
        tolerance_at: float,
    ) -> bool:
        """判定两个成分是否代表同一解。

        判据（方案 v1.13 成分空间，ALL_ELEMENTS = Al/Co/Cr/Fe/Ni/C）：
        所有元素的 at% 差均 < tolerance_at 时视为同一解。

        Args:
            frac_a: 成分字典 {元素: at%}。
            frac_b: 成分字典 {元素: at%}。
            tolerance_at: 成分容差 (at%)。

        Returns:
            True 表示两成分在容差内一致。
        """
        for elem in ALL_ELEMENTS:
            if abs(frac_a.get(elem, 0.0) - frac_b.get(elem, 0.0)) >= tolerance_at:
                return False
        return True

    @classmethod
    def _top10_pair_overlap(
        cls,
        top_a: List[Dict[str, float]],
        top_b: List[Dict[str, float]],
        tolerance_at: float,
    ) -> float:
        """两组成分快照的一对一贪婪匹配重叠率。

        top_b 为基准侧 TOP10：对基准侧每个解，在 top_a 中找一个
        尚未匹配且成分容差内一致的解配对（防止两个基准解重复匹配
        同一个候选解）。重叠率 = |交集| / len(top_b)；标准情形
        top_b 为 TOP10，即 |交集| / 10。

        Args:
            top_a: 候选 γ 的 TOP10 成分快照。
            top_b: 基准 γ=0.25 的 TOP10 成分快照。
            tolerance_at: 成分容差 (at%)。

        Returns:
            重叠率 ∈ [0, 1]；基准侧为空时返回 0.0。
        """
        if not top_b:
            return 0.0
        matched_a = [False] * len(top_a)
        n_overlap = 0
        for fb in top_b:
            for i, fa in enumerate(top_a):
                if not matched_a[i] and cls._compositions_match(fa, fb, tolerance_at):
                    matched_a[i] = True
                    n_overlap += 1
                    break
        return n_overlap / len(top_b)

    @classmethod
    def _mean_top10_overlap(
        cls,
        snapshots_a: List[List[Dict[str, float]]],
        snapshots_b: List[List[Dict[str, float]]],
        tolerance_at: float,
    ) -> float:
        """跨重复的两两 TOP10 重叠率均值。

        对候选 γ 的每次重复快照与基准 γ=0.25 的每次重复快照做
        笛卡尔配对，取所有配对重叠率的算术平均；任一为空返回 0.0。
        """
        overlaps = [
            cls._top10_pair_overlap(snap_a, snap_b, tolerance_at)
            for snap_a in snapshots_a
            for snap_b in snapshots_b
        ]
        return float(np.mean(overlaps)) if overlaps else 0.0
