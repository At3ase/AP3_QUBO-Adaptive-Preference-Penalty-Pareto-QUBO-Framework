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

    @staticmethod
    def _setup_bounds() -> List[Tuple[float, float]]:
        """每个元素的成分边界 (at%)。"""
        bounds = []
        for _ in MAIN_ELEMENTS:
            bounds.append((ENCODING.main_min, ENCODING.main_max))
        bounds.append((ENCODING.carbon_min, ENCODING.carbon_max))
        return bounds

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
            return NSGA2Optimizer._project_to_box_simplex(ind, lows, highs)

        toolbox.register("individual", tools.initIterate, creator.Individual, init_individual)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        # 评估函数
        def evaluate(ind):
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
                    offspring[i-1][:] = NSGA2Optimizer._project_to_box_simplex(
                        offspring[i-1], lows, highs)
                    offspring[i][:] = NSGA2Optimizer._project_to_box_simplex(
                        offspring[i], lows, highs)
                    del offspring[i-1].fitness.values
                    del offspring[i].fitness.values

            for ind in offspring:
                if np.random.random() < self._mut_prob:
                    toolbox.mutate(ind)
                    # 第 3 批修复：多项式变异同样不保成分和，变异后投影。
                    ind[:] = NSGA2Optimizer._project_to_box_simplex(
                        ind, lows, highs)
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
        """
        front = self.optimize()
        result: Dict[str, object] = {
            "algorithm": "NSGA-II (DEAP: SBX + PolyMutation + selNSGA2)",
            "front": front,
            "objectives": np.zeros((0, 3)),
            "hv": 0.0,
            "feasibility": {"sum_min": 0.0, "sum_max": 0.0, "sum_mean": 0.0},
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
