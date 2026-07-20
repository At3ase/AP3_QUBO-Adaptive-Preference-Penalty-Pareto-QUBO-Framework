"""
NSGA-II 基线实现（实验 3 对照组）。

使用 DEAP 库实现经典 NSGA-II 算法:
  - SBX 模拟二进制交叉
  - 多项式变异
  - 非支配排序 + 拥挤距离选择

在连续成分空间中优化，绕开 QUBO 编码层。
约束: 成分和 = 100%，每元素在规定范围内。
"""

from typing import Dict, List, Tuple

import numpy as np

from ..physical_params import (
    MAIN_ELEMENTS,
    INTERSTITIAL_ELEMENT,
    ALL_ELEMENTS,
    ENCODING,
    PARETO_ZOOM,
)
from ..objectives.mixing_enthalpy import MixingEnthalpy
from ..objectives.density import VegardDensity
from ..objectives.cost import WeightedCost
from ..validation.hypervolume import HypervolumeCalculator, set_unified_reference
from ..validation.pareto import ParetoSort


class NSGA2Optimizer:
    """NSGA-II 多目标进化优化器。

    直接在连续成分空间 [base, max] 中优化。
    成分和 = 100% 约束采用"修复算子保和"为主、惩罚为辅：
    初始化与每次交叉/变异后均投影到 box∩simplex 可行域
    （见 _project_to_box_simplex，第 3 批修复），f1 惩罚仅
    兜底浮点残差。

    使用示例:
        >>> opt = NSGA2Optimizer(pop_size=100, generations=200)
        >>> front = opt.optimize()
        >>> for comp in front:
        ...     print(comp)
    """

    def __init__(
        self,
        pop_size: int | None = None,
        generations: int | None = None,
        crossover_prob: float = 0.9,
        mutation_prob: float = 0.1,
        eta_crossover: float = 20.0,
        eta_mutation: float = 20.0,
    ):
        """
        Args:
            pop_size: 种群大小（默认 PARETO_ZOOM.nsga_pop_size = 100）。
            generations: 迭代代数（默认 PARETO_ZOOM.nsga_generations = 200）。
            crossover_prob: 交叉概率。
            mutation_prob: 变异概率。
            eta_crossover: SBX 交叉分布指数。
            eta_mutation: 多项式变异分布指数。
        """
        self._pop_size = pop_size or PARETO_ZOOM.nsga_pop_size
        self._generations = generations or PARETO_ZOOM.nsga_generations
        self._cx_prob = crossover_prob
        self._mut_prob = mutation_prob
        self._eta_cx = eta_crossover
        self._eta_mut = eta_mutation

        # 变量边界
        self._bounds = NSGA2Optimizer._setup_bounds()

        # 目标函数
        self._dh_calc = MixingEnthalpy()
        self._density_calc = VegardDensity()
        self._cost_calc = WeightedCost()
        self._sorter = ParetoSort()
        self._hv_calc = HypervolumeCalculator()

        # Fairness_Reporter（审计 D-2/D-3 公平性披露）：
        # 评估预算与投影修复贴界统计。_n_objective_evals 仅计
        # optimize() 内适应度评估（pop + Σ 各代子代），不含
        # evaluate_front 的事后重评估（口径见 budget_stats）。
        self._n_objective_evals = 0
        self._repair_stats: Dict[str, object] = {
            "n_repair_calls": 0,
            "n_coords": 0,
            "before_at_bound": 0,
            "after_at_bound": 0,
            "per_element_coords": {e: 0 for e in ALL_ELEMENTS},
            "per_element_before": {e: 0 for e in ALL_ELEMENTS},
            "per_element_after": {e: 0 for e in ALL_ELEMENTS},
        }

    # 贴界判定容差（Fairness_Reporter，审计 D-3）：坐标到最近边界的
    # 距离 ≤ BOUNDARY_TOL_REL × (hi − lo) 即判定为"贴界"。0.25% 口径下
    # 主元 tol ≈ 0.0794 at%（range 31.75）、C tol ≈ 0.0044 at%（range 1.75）。
    BOUNDARY_TOL_REL = 0.0025

    @staticmethod
    def _setup_bounds() -> List[Tuple[float, float]]:
        """每个元素的成分边界 (at%)。"""
        bounds = []
        for _ in MAIN_ELEMENTS:
            bounds.append((ENCODING.main_min, ENCODING.main_max))
        bounds.append((ENCODING.carbon_min, ENCODING.carbon_max))
        return bounds

    @staticmethod
    def _boundary_flags(
        vec: List[float],
        lows: List[float],
        highs: List[float],
        tol_rel: float | None = None,
    ) -> List[bool]:
        """逐坐标贴界判定：min(x − lo, hi − x) ≤ tol_rel × (hi − lo)。

        Fairness_Reporter（审计 D-3）：越界坐标（x < lo 或 x > hi）
        距离为负、判定为贴界——它们经投影恰被裁剪到边界上，是投影
        修复贴界富集的直接来源，修复前统计必须将其计入。
        """
        rel = NSGA2Optimizer.BOUNDARY_TOL_REL if tol_rel is None else tol_rel
        flags = []
        for x, lo, hi in zip(vec, lows, highs):
            tol = rel * (hi - lo)
            flags.append((float(x) - lo) <= tol or (hi - float(x)) <= tol)
        return flags

    def _repair_and_record(
        self, ind: List[float], lows: List[float], highs: List[float]
    ) -> List[float]:
        """投影修复 + 贴界统计（Fairness_Reporter，审计 D-3）。

        对 _project_to_box_simplex 的输入（修复前）与输出（修复后）
        逐坐标做贴界判定并累计计数，供 repair_boundary_stats() 汇总。
        修复算法本体不变（第 3 批修复的投影算子语义保持原样）。
        """
        before = NSGA2Optimizer._boundary_flags(ind, lows, highs)
        repaired = NSGA2Optimizer._project_to_box_simplex(ind, lows, highs)
        after = NSGA2Optimizer._boundary_flags(repaired, lows, highs)
        st = self._repair_stats
        st["n_repair_calls"] += 1
        st["n_coords"] += len(repaired)
        st["before_at_bound"] += sum(before)
        st["after_at_bound"] += sum(after)
        for e, fb, fa in zip(ALL_ELEMENTS, before, after):
            st["per_element_coords"][e] += 1
            st["per_element_before"][e] += int(fb)
            st["per_element_after"][e] += int(fa)
        return repaired

    @staticmethod
    def _project_to_box_simplex(
        ind: List[float],
        lows: List[float],
        highs: List[float],
        tol: float = 1e-9,
        max_iter: int = 200,
    ) -> List[float]:
        """将个体投影到 box∩simplex 可行域（各元素边界 ∩ Σc=100）。

        第 3 批修复（NSGA-II 可行性约束缺陷）：原实现仅在初始化时
        投影一次，SBX 交叉 / 多项式变异后的子代不保成分和，不可行解
        （实测前沿 Σc 低至 60.3）混入 Pareto 前沿，使基线 HV 虚高、
        对比实验失真（违反方案 BASE-9 公平性"相同后处理"）。现提取
        为公共投影算子，初始化与每次交叉/变异后均调用，使种群中
        每个个体天然满足 Σc=100（f1 惩罚降为辅助兜底）。

        采用"裁剪到边界 → 重新归一化到 Σc=100"迭代：
        单纯归一化会把个别元素压出边界（如碳被稀释到 carbon_min
        以下），而 SBX 交叉要求双亲基因严格在 [low, up] 内（否则
        beta<0 经分数次幂产生复数，deap cxSimulatedBinaryBounded
        前置条件）；单纯裁剪又会破坏成分和=100。两者迭代至同时
        满足——box∩simplex 非空（本问题 5×[5.0,36.75] + C[0,1.75]
        显然非空，如 5×19.65+1.75=100），迭代必收敛。进化过程中的
        修复对象是近可行子代（1~2 次迭代即收敛），max_iter=200 仅
        为极端输入兜底。
        """
        x = [min(max(float(v), lo), hi) for v, lo, hi in zip(ind, lows, highs)]
        for _ in range(max_iter):
            total = sum(x)
            if abs(total - 100.0) < tol:
                break
            x = [v * 100.0 / total for v in x]
            x = [min(max(v, lo), hi) for v, lo, hi in zip(x, lows, highs)]
        return x

    def optimize(self) -> List[Dict[str, float]]:
        """运行 NSGA-II 优化。

        Returns:
            Pareto 前沿成分列表 [{element: at%, ...}, ...]。

        Raises:
            ImportError: deap 未安装时显式抛出。
                P0-6 修复（审查报告 Code_Completion_Review_2026-07-18）：
                原实现 deap 不可用时静默回退 _optimize_simple（无拥挤
                距离、无 selNSGA-II 选择），结果仍以 "NSGA-II" 标签
                上报，伪结果会流入对比报告。现改为显式报错，只允许
                真实 NSGA-II（DEAP）结果进入实验结论。
        """
        # 必须使用 DEAP 的真 NSGA-II（SBX + 多项式变异 + selNSGA2）
        try:
            from deap import base, creator, tools
        except ImportError as exc:
            raise ImportError(
                "NSGA-II 基线依赖 deap 库（方案 H-04 必需实验对照组），"
                "当前环境未安装。请执行: pip install deap "
                "（或 pip install .[experiments]）。"
                "为避免伪 NSGA-II 结果流入报告，已禁用静默回退实现。"
            ) from exc

        bounds = NSGA2Optimizer._setup_bounds()

        # 创建适应度和个体类型
        try:
            creator.create("FitnessMin3", base.Fitness, weights=(-1.0, -1.0, -1.0))
            creator.create("Individual", list, fitness=creator.FitnessMin3)
        except RuntimeError:
            # 类型已存在，跳过
            pass

        toolbox = base.Toolbox()

        # 变量初始化（各元素在 bound 内随机）
        lows = [b[0] for b in bounds]
        highs = [b[1] for b in bounds]

        def init_individual():
            ind = [np.random.uniform(lo, hi) for lo, hi in bounds]
            # 初始个体投影到 box∩simplex 可行域（投影必要性与收敛性
            # 论证见 _project_to_box_simplex docstring）。
            # Fairness_Reporter：经 _repair_and_record 走同一投影，
            # 同时累计修复前/后贴界统计（审计 D-3）。
            return self._repair_and_record(ind, lows, highs)

        toolbox.register("individual", tools.initIterate, creator.Individual, init_individual)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        # 评估函数
        def evaluate(ind):
            # Fairness_Reporter（审计 D-2）：适应度评估计数（四口径之
            # #objective-evals；正式规模 pop100×gen200 ≈ 20,100 次）。
            self._n_objective_evals += 1
            fractions = {elem: ind[i] for i, elem in enumerate(ALL_ELEMENTS)}
            dh = self._dh_calc.evaluate(fractions)
            density = self._density_calc.evaluate(fractions)
            cost = self._cost_calc.evaluate(fractions)
            # 成分和惩罚（第 3 批起降为辅助保险：修复算子已使种群
            # 个体天然满足 Σc=100，此处仅兜底浮点残差；目标函数定义
            # 不变，仍复用 MixingEnthalpy/VegardDensity/WeightedCost）
            total = sum(ind)
            penalty = (total - 100.0) ** 2 * 1000.0
            return (dh + penalty, density, cost)

        toolbox.register("evaluate", evaluate)
        toolbox.register("mate", tools.cxSimulatedBinaryBounded,
                         eta=self._eta_cx, low=[b[0] for b in bounds],
                         up=[b[1] for b in bounds])
        toolbox.register("mutate", tools.mutPolynomialBounded,
                         eta=self._eta_mut, low=[b[0] for b in bounds],
                         up=[b[1] for b in bounds], indpb=1.0/6.0)
        toolbox.register("select", tools.selNSGA2)

        # 运行进化
        pop = toolbox.population(n=self._pop_size)

        # 初始评估
        invalid_ind = [ind for ind in pop if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # 初代先做一次 selNSGA2：为非支配排序并给每个个体写入
        # crowding_dist 属性——后续 selTournamentDCD 的拥挤距离
        # 锦标赛依赖该属性（deap.tools.emo.selTournamentDCD 官方
        # 用法，见 DEAP NSGA-II 示例；缺失会抛 AttributeError）。
        pop = toolbox.select(pop, k=self._pop_size)

        # NSGA-II 主循环
        for gen in range(self._generations):
            # 选择 + 交叉 + 变异
            # deap 前置条件：selTournamentDCD 在 k == len(individuals)
            # 时要求 k 被 4 整除（deap.tools.emo.selTournamentDCD 源码
            # ValueError）。pop_size 非 4 倍数（如小规模验证 pop=30）时
            # 向下取整到 4 的倍数，亲本+子代合并选择仍恢复 pop_size；
            # 默认 pop_size=100 时行为与原来完全一致。
            n_offspring = len(pop) - (len(pop) % 4)
            offspring = tools.selTournamentDCD(pop, n_offspring)
            offspring = [toolbox.clone(ind) for ind in offspring]

            for i in range(1, len(offspring), 2):
                if np.random.random() < self._cx_prob:
                    toolbox.mate(offspring[i-1], offspring[i])
                    # 第 3 批修复：SBX 只保边界不保成分和，交叉后立即
                    # 投影回 box∩simplex 可行域（修复算子保和，杜绝
                    # 不可行子代进入评估与前沿）。
                    # Fairness_Reporter：经 _repair_and_record 累计贴界统计。
                    offspring[i-1][:] = self._repair_and_record(
                        offspring[i-1], lows, highs)
                    offspring[i][:] = self._repair_and_record(
                        offspring[i], lows, highs)
                    del offspring[i-1].fitness.values
                    del offspring[i].fitness.values

            for ind in offspring:
                if np.random.random() < self._mut_prob:
                    toolbox.mutate(ind)
                    # 第 3 批修复：多项式变异同样不保成分和，变异后投影。
                    # Fairness_Reporter：经 _repair_and_record 累计贴界统计。
                    ind[:] = self._repair_and_record(ind, lows, highs)
                    del ind.fitness.values

            # 重新评估
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # 选择下一代
            pop = toolbox.select(pop + offspring, k=self._pop_size)

        # 提取 Pareto 前沿
        fronts = self._sorter.non_dominated_sort(
            np.array([ind.fitness.values for ind in pop])
        )
        if not fronts:
            return []

        front_indices = fronts[0]
        front = []
        for idx in front_indices:
            ind = pop[idx]
            fractions = {elem: ind[i] for i, elem in enumerate(ALL_ELEMENTS)}
            front.append(fractions)

        return front

    # NOTE(P0-6, 审查报告 Code_Completion_Review_2026-07-18)：
    # 原 _optimize_simple（无拥挤距离、无 selNSGA2 的"伪 NSGA-II"
    # 静默回退）已整体删除——deap 不可用时 optimize() 显式抛
    # ImportError，杜绝伪 NSGA-II 结果以 "NSGA-II" 标签流入报告。

    def evaluate_front(
        self, front: List[Dict[str, float]]
    ) -> np.ndarray:
        """提取 Pareto 前沿的目标值矩阵 (N, 3) = (ΔH_mix, ρ, cost)。

        供跨方法统一 HV 参考点（P0-5）使用：comparison.py 先收集
        各方法的目标矩阵，再统一设定参考点计算 HV。

        Args:
            front: optimize() 返回的成分列表。

        Returns:
            shape=(N, 3) 目标值矩阵；front 为空时返回 shape=(0, 3)。
        """
        obj_vals = []
        for comp in front:
            dh = self._dh_calc.evaluate(comp)
            density = self._density_calc.evaluate(comp)
            cost = self._cost_calc.evaluate(comp)
            obj_vals.append([dh, density, cost])
        return np.array(obj_vals, dtype=float).reshape(-1, 3)

    def repair_boundary_stats(self) -> Dict[str, object]:
        """投影修复贴界率汇总（Fairness_Reporter，审计 D-3 披露）。

        贴界判定：坐标到最近边界距离 ≤ 0.25%×(hi−lo)（见
        BOUNDARY_TOL_REL / _boundary_flags；越界坐标计为贴界）。
        rate_after > rate_before 的富集源于投影裁剪——均值无偏，
        但坐标分布向边界富集，须在结果 metadata 与对比报告中披露。

        Returns:
            dict: tol_rel、n_repair_calls、n_coords、修复前/后总体
            贴界率、主元/C 分组贴界率、逐元素修复后贴界率。
        """
        st = self._repair_stats
        n = max(int(st["n_coords"]), 1)
        main = [e for e in ALL_ELEMENTS if e != INTERSTITIAL_ELEMENT]

        def _group_rate(count_key: str, elems: List[str]) -> float:
            coords = sum(int(st["per_element_coords"][e]) for e in elems)
            if coords == 0:
                return 0.0
            return sum(int(st[count_key][e]) for e in elems) / coords

        return {
            "tol_rel": self.BOUNDARY_TOL_REL,
            "n_repair_calls": int(st["n_repair_calls"]),
            "n_coords": int(st["n_coords"]),
            "rate_before": int(st["before_at_bound"]) / n,
            "rate_after": int(st["after_at_bound"]) / n,
            "rate_before_main": _group_rate("per_element_before", main),
            "rate_after_main": _group_rate("per_element_after", main),
            "rate_before_carbon": _group_rate(
                "per_element_before", [INTERSTITIAL_ELEMENT]),
            "rate_after_carbon": _group_rate(
                "per_element_after", [INTERSTITIAL_ELEMENT]),
            "per_element_after": {
                e: (int(st["per_element_after"][e])
                    / int(st["per_element_coords"][e])
                    if int(st["per_element_coords"][e]) else 0.0)
                for e in ALL_ELEMENTS
            },
        }

    def budget_stats(self) -> Dict[str, int]:
        """NSGA-II 评估预算（Fairness_Reporter，审计 D-2 四口径披露）。

        口径：#solves = 0、#samples = 0（NSGA-II 不经 QUBO 求解链路）；
        #objective-evals = optimize() 内适应度评估次数
        （pop + Σ 各代子代；正式规模 100×200 ≈ 20,100），
        不含 evaluate_front 对前沿的事后重评估。
        """
        return {
            "n_solves": 0,
            "n_samples": 0,
            "n_objective_evals": self._n_objective_evals,
        }

    def optimize_and_evaluate(self) -> Dict[str, object]:
        """优化并计算 HV（独立运行入口）。

        Returns:
            结果字典:
              - "algorithm": 实际算法标注（P0-6：恒为真 NSGA-II/DEAP，
                deap 缺失时根本不会走到这里）；
              - "front": Pareto 前沿成分列表；
              - "objectives": shape=(N, 3) 目标值矩阵（供外部做
                跨方法统一参考点 HV）；
              - "hv": 独立运行时的 HV（单方法场景下以自身解集定
                参考点，走 set_unified_reference 统一代码路径）；
              - "feasibility": 前沿可行性统计（第 3 批新增），
                {"sum_min", "sum_max", "sum_mean"} —— 前沿各解
                Σc (at%) 的最小/最大/均值，修复算子保和后应紧贴 100。
              - "boundary_repair": 投影修复贴界率统计
                （Fairness_Reporter，审计 D-3；repair_boundary_stats）。
              - "budget": 评估预算四口径之三
                （Fairness_Reporter，审计 D-2；budget_stats）。
        """
        front = self.optimize()
        result: Dict[str, object] = {
            "algorithm": "NSGA-II (DEAP: SBX + PolyMutation + selNSGA2)",
            "front": front,
            "objectives": np.zeros((0, 3)),
            "hv": 0.0,
            "feasibility": {"sum_min": 0.0, "sum_max": 0.0, "sum_mean": 0.0},
            "boundary_repair": self.repair_boundary_stats(),
            "budget": self.budget_stats(),
        }
        if not front:
            return result

        points = self.evaluate_front(front)
        result["objectives"] = points

        # 第 3 批：前沿可行性统计（Σc 应紧贴 100，供 BASE-9 公平性核查）
        sums = np.array(
            [sum(comp[e] for e in ALL_ELEMENTS) for comp in front], dtype=float
        )
        result["feasibility"] = {
            "sum_min": float(sums.min()),
            "sum_max": float(sums.max()),
            "sum_mean": float(sums.mean()),
        }

        # P0-5 修复（审查报告 Code_Completion_Review_2026-07-18）：
        # 原 :287 对自身解集直接 set_reference_from_data；现统一走
        # set_unified_reference 路径。独立运行时只有本方法一个解集，
        # 跨方法对比由 comparison.py 收集全部 archive 后另行统一
        # 定参考点（不使用本函数返回的 hv）。
        ref = set_unified_reference({"NSGA-II": points}, margin=0.10)
        self._hv_calc = HypervolumeCalculator(reference_point=ref)
        result["hv"] = self._hv_calc.compute(points)

        return result
