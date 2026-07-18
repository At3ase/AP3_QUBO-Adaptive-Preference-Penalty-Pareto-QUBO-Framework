"""
消融实验（实验 0）—— 最高优先级。

系统移除每个创新组件，量化其独立贡献：
  - Full: PrecisionSplit(38) + PenaltyFlex + ParetoZoom
  - Abl-1: 统一编码(48) + PenaltyFlex + ParetoZoom
  - Abl-2: PrecisionSplit(38) + Grid-search 最优固定 λ + ParetoZoom
  - Abl-3: PrecisionSplit(38) + PenaltyFlex + 均匀网格
  - Abl-4: 统一编码(48) + Grid-search 最优固定 λ + 均匀网格

指标:
  - ΔHV_i: HV 差值
  - AR_i: 消融贡献率 |ΔHV_i| / HV_Full × 100%（目标 > 3%）
  - Synergy: 协同效应

第 3 批修复（审查报告 Code_Completion_Review_2026-07-18）：
  - P0-5：每次重复内 5 配置全部跑完后，用 set_unified_reference
    统一定参考点再算各配置 HV（方案 HV-1 固定参考点前提，
    与 comparison.py 的 _compute_hv_unified 同口径）。
  - P1-6：Abl-2/Abl-4 的固定 λ 由硬编码 0.05 改为 BASE-4
    Grid-search（方案 AP3_QUBO_Validation_Scheme_v1.1 §4.2 :160：
    "在 {0.1,0.5,1,5,10,50,100} 上穷举搜索最优固定 λ"）。
  - AR_i 口径修正为 |ΔHV_i| / HV_Full（方案 完整技术路线v2.0 :545），
    原实现带负号与方案不符。
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..physical_params import COARSE_WEIGHTS, MAIN_ELEMENTS, INTERSTITIAL_ELEMENT
from ..exploration.pareto_zoom import ParetoZoom
from ..exploration.archive import Archive
from ..encoding.precision_split import Composition, PrecisionSplitDecoder
from ..validation.hypervolume import HypervolumeCalculator, set_unified_reference

# 方案 BASE-4 网格（AP3_QUBO_Validation_Scheme_v1.1 §4.2 :160，
# "Grid-search最优λ：在{0.1,0.5,1,5,10,50,100}上穷举搜索最优固定λ"，
# 即"固定 λ 的上限"基线）。Abl-2 / Abl-4 的惩罚策略按方案
# 消融表（同文件 :94, :96）均为 "Grid-search 最优固定 λ"。
GRID_SEARCH_LAMBDA_GRID: Tuple[float, ...] = (0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0)


@dataclass
class AblationResult:
    """消融实验单次运行结果。

    Attributes:
        config_name: 配置名称。
        hv: Hypervolume 值（统一参考点口径，同一次重复内 5 配置共享
            同一参考点，跨重复参考点随当次解集重定）。
        front_size: 前沿解数量。
        feasible_rate: 可行解比例。
        physical_pass_rate: 物理过滤器通过率。
    """
    config_name: str
    hv: float
    front_size: int
    feasible_rate: float = 1.0
    physical_pass_rate: float = 1.0
    extra: Dict = field(default_factory=dict)


def _compute_hv_unified(obj_mats: Dict[str, np.ndarray]) -> Dict[str, float]:
    """统一参考点 HV 计算（P0-5 修复，审查报告 Code_Completion_Review_2026-07-18）。

    与 experiments/comparison.py 的 _compute_hv_unified 同一模式：
    合并全部配置的解集 → set_unified_reference(margin=0.10) 统一定
    参考点 → 同一参考点分别计算各配置 HV。空解集记 0.0；
    全部为空时各配置均记 0.0（不抛异常，保证框架验证可跑）。
    方案 HV-1：跨配置 HV 对比的前提是固定参考点。
    """
    if not any(len(m) > 0 for m in obj_mats.values()):
        return {label: 0.0 for label in obj_mats}
    ref = set_unified_reference(obj_mats, margin=0.10)
    hv_calc = HypervolumeCalculator(reference_point=ref)
    return {
        label: (hv_calc.compute(mat) if len(mat) > 0 else 0.0)
        for label, mat in obj_mats.items()
    }


class _UnifiedGridDecoder:
    """统一编码（unified_48 / unified_38）解码器，接口对齐
    PrecisionSplitDecoder.decode。

    位布局出处：qubo/builder.py :127-147 的 _ENCODING_CONFIGS
    （每主元 bits_main 位、base_main + k×step_main；C 为 bits_carbon
    位、base_carbon + k×step_carbon）。bit j 为第 j 低位
    （k += 1<<j），与 PrecisionSplitDecoder.decode 的位序约定一致。

    背景：PrecisionSplitDecoder 仅接受 38 bits
    （encoding/precision_split.py:188-191），统一编码路径在
    ParetoZoom 内会全部 ValueError 并被静默跳过
    （exploration/pareto_zoom.py:631 的 except ValueError: continue），
    导致 Abl-1/Abl-4 空档。本类仅在 ablation.py 内使用
    （试跑解码 + 注入本文件构造的 ParetoZoom 实例），
    不改动 encoding/precision_split.py 与 pareto_zoom.py。

    注意（第 3 批发现的越界前置缺陷，需 solver 负责人修复）：
    solver/kaiwu_solver.py:490-504 的 _parse_var_index 把
    e{ei}_b{bj} → flat index 的映射硬编码为 precision_split_38
    布局（ei*7+bj / 35+bj），对 unified_48/unified_38 会产生
    索引碰撞与比特错位。该 bug 修复前，统一编码配置即使解码器
    正确也只能得到错位比特——多数解 |Σc−100| 远超 2% 入档容差，
    表现为空档/占位，这是上游问题而非本文件逻辑问题。
    """

    def __init__(self, encoding_type: str):
        from ..qubo.builder import QUBOBuilder

        cfg = QUBOBuilder._ENCODING_CONFIGS[encoding_type]
        self._n_vars = cfg["n_vars"]
        self._bits_main = cfg["bits_main"]
        self._bits_carbon = cfg["bits_carbon"]
        self._base_main = cfg["base_main"]
        self._base_carbon = cfg["base_carbon"]
        self._step_main = cfg["step_main"]
        self._step_carbon = cfg["step_carbon"]

    def decode(self, bits: np.ndarray) -> Composition:
        if len(bits) != self._n_vars:
            raise ValueError(
                f"Expected {self._n_vars} bits, got {len(bits)}"
            )
        fractions = {}
        for i, elem in enumerate(MAIN_ELEMENTS):
            k = 0
            start = i * self._bits_main
            for j in range(self._bits_main):
                if bits[start + j]:
                    k += (1 << j)
            fractions[elem] = self._base_main + self._step_main * k
        k_c = 0
        start_c = len(MAIN_ELEMENTS) * self._bits_main
        for j in range(self._bits_carbon):
            if bits[start_c + j]:
                k_c += (1 << j)
        fractions[INTERSTITIAL_ELEMENT] = (
            self._base_carbon + self._step_carbon * k_c
        )
        return Composition(fractions=fractions)


def _build_decoder(encoding_type: str):
    """按编码类型返回解码器。

    precision_split_38 → PrecisionSplitDecoder（原路径）；
    unified_48 / unified_38 → _UnifiedGridDecoder（本文件最小补充，
    位布局以 QUBOBuilder._ENCODING_CONFIGS 为准）。
    """
    if encoding_type == "precision_split_38":
        return PrecisionSplitDecoder()
    return _UnifiedGridDecoder(encoding_type)


class AblationRunner:
    """消融实验执行器。

    五种配置通过模块开关控制：
      - use_precision_split: True → 38 变量, False → 48 变量统一编码
      - use_penalty_flex: True → PenaltyFlex, False → Grid-search 最优固定 λ
      - use_pareto_zoom: True → ParetoZoom, False → 均匀 50 组网格

    使用示例:
        >>> runner = AblationRunner()
        >>> results = runner.run(n_repetitions=20)
        >>> runner.compute_contributions(results)
    """

    def __init__(
        self,
        grid_search_reads: int = 200,
        grid_search_weights: Optional[List[Tuple[float, float, float]]] = None,
        pareto_zoom_kwargs: Optional[Dict] = None,
    ):
        """
        Args:
            grid_search_reads: BASE-4 grid-search 试跑的采样预算
                （每次试跑求解的 num_reads）。方案未规定试跑预算，
                默认 200（完整流程 num_reads=1000 的 1/5，试跑只用于
                选 λ，不作为实验数据）；正式实验可调大。
            grid_search_weights: grid-search 试跑权重组（默认
                COARSE_WEIGHTS 12 组）；小规模实测可传 2-3 组。
            pareto_zoom_kwargs: 透传给 ParetoZoom 构造器的额外参数
                （如 initial_weights / solver），供小规模实测注入小预算。
        """
        if grid_search_reads < 1:
            raise ValueError(f"grid_search_reads 必须 ≥ 1，得到 {grid_search_reads}")
        self._grid_search_reads = int(grid_search_reads)
        self._grid_search_weights = grid_search_weights
        self._pz_kwargs = dict(pareto_zoom_kwargs or {})

        self._configs = {
            "Full": {
                "use_precision_split": True,
                "use_penalty_flex": True,
                "use_pareto_zoom": True,
                "description": "AP³ 完整版",
            },
            "Abl-1": {
                "use_precision_split": False,
                "use_penalty_flex": True,
                "use_pareto_zoom": True,
                "description": "去掉 PrecisionSplit（统一 48 变量编码）",
            },
            "Abl-2": {
                "use_precision_split": True,
                "use_penalty_flex": False,
                "use_pareto_zoom": True,
                "description": "去掉 PenaltyFlex（Grid-search 最优固定 λ）",
            },
            "Abl-3": {
                "use_precision_split": True,
                "use_penalty_flex": True,
                "use_pareto_zoom": False,
                "description": "去掉 ParetoZoom（均匀 50 组网格）",
            },
            "Abl-4": {
                "use_precision_split": False,
                "use_penalty_flex": False,
                "use_pareto_zoom": False,
                "description": "三创新全部去掉（性能下界）",
            },
        }

    @property
    def config_names(self) -> List[str]:
        return list(self._configs.keys())

    def run(
        self,
        n_repetitions: int = 20,
        seed: int = 42,
        configs: Optional[List[str]] = None,
    ) -> Dict[str, List[AblationResult]]:
        """运行完整消融实验。

        Args:
            n_repetitions: 每种配置的重复次数（≥ 20）。
            seed: 随机种子基数。
            configs: 可选，只运行指定配置子集（小规模调试用）；
                默认 None 运行全部 5 配置。

        Returns:
            {config_name: [AblationResult, ...]}。
        """
        names = list(configs) if configs else list(self._configs.keys())
        for name in names:
            if name not in self._configs:
                raise ValueError(f"未知配置: {name!r}，可选 {list(self._configs.keys())}")

        all_results: Dict[str, List[AblationResult]] = {name: [] for name in names}

        for rep in range(n_repetitions):
            np.random.seed(seed + rep)

            # P0-5 修复（审查报告 Code_Completion_Review_2026-07-18）：
            # 每次重复内先跑完全部配置、收集目标矩阵，再用统一参考点
            # 一次性计算各配置 HV（方案 HV-1 固定参考点前提；
            # 原实现在 _run_single_config 内各自 set_reference_from_data，
            # 组间 HV 差异会混入参考点漂移）。
            obj_mats: Dict[str, np.ndarray] = {}
            meta: Dict[str, Tuple[int, Dict]] = {}

            for config_name in names:
                try:
                    # B-1：seed+rep 贯通到 ParetoZoom 默认求解器与
                    # BASE-4 试跑求解器，同 seed 结果逐位一致
                    mat, front_size, extra = self._run_single_config(
                        config_name, self._configs[config_name],
                        rep_seed=seed + rep,
                    )
                except Exception as e:
                    # 记录失败但不中断
                    mat, front_size, extra = (
                        np.zeros((0, 3)),
                        0,
                        {"error": str(e)},
                    )
                obj_mats[config_name] = mat
                meta[config_name] = (front_size, extra)

            hv_vals = _compute_hv_unified(obj_mats)

            for config_name in names:
                front_size, extra = meta[config_name]
                all_results[config_name].append(
                    AblationResult(
                        config_name=config_name,
                        hv=hv_vals[config_name],
                        front_size=front_size,
                        extra=extra,
                    )
                )

        return all_results

    def _run_single_config(
        self, name: str, config: Dict, rep_seed: Optional[int] = None
    ) -> Tuple[np.ndarray, int, Dict]:
        """运行单个配置，返回 (目标矩阵 (N,3), 前沿解数, 附加信息)。

        根据 config 中的 use_precision_split / use_penalty_flex / use_pareto_zoom
        标志，创建不同配置的 ParetoZoom 实例。

        HV 不在此计算——由 run() 收集齐本次重复全部配置的目标矩阵后，
        用统一参考点一次性计算（P0-5 / 方案 HV-1）。

        rep_seed: 本次重复的随机种子（seed+rep，B-1），透传到 ParetoZoom
            默认求解器与 BASE-4 试跑求解器；pareto_zoom_kwargs 显式给定
            seed 时不覆盖。
        """
        use_precision = config.get("use_precision_split", True)
        use_penalty_flex = config.get("use_penalty_flex", True)
        use_pareto_zoom = config.get("use_pareto_zoom", True)

        # 编码类型
        encoding_type = "precision_split_38" if use_precision else "unified_48"

        # 惩罚策略
        extra: Dict = {}
        if use_penalty_flex:
            penalty_strategy = "adaptive"
            penalty_fixed = None
        else:
            # P1-6 修复（审查报告 Code_Completion_Review_2026-07-18）：
            # 方案消融表（Validation Scheme v1.1 :94, :96）要求 Abl-2 与
            # Abl-4 使用 "Grid-search 最优固定 λ"（BASE-4 网格穷举），
            # 原硬编码 0.05 仅为占位。先试跑选 λ，再用该 λ 跑完整流程。
            penalty_strategy = "fixed"
            try:
                penalty_fixed, gs_detail = self._grid_search_fixed_lambda(
                    encoding_type, seed=rep_seed,
                )
            except (NotImplementedError, RuntimeError, ImportError):
                # 求解器不可用 / 试跑全失败 → 与主流程同口径走占位
                extra["solver_available"] = False
                return np.zeros((0, 3)), 0, extra
            extra["grid_search"] = gs_detail

        # 探索策略
        if use_pareto_zoom:
            exploration_strategy = "pareto_zoom"
        else:
            exploration_strategy = "uniform_grid"

        try:
            # B-1：seed 透传（setdefault 语义：pareto_zoom_kwargs 显式
            # 指定 seed 时不覆盖，保证小规模实测注入优先）
            pz_kwargs = dict(self._pz_kwargs)
            pz_kwargs.setdefault("seed", rep_seed)
            pz = ParetoZoom(
                encoding_type=encoding_type,
                penalty_strategy=penalty_strategy,
                penalty_fixed_lambda=penalty_fixed,
                exploration_strategy=exploration_strategy,
                **pz_kwargs,
            )
            # 统一编码解码器注入（见 _UnifiedGridDecoder 类注释）：
            # ParetoZoom 内部固定使用 38-bit PrecisionSplitDecoder，
            # 对 unified_48/unified_38 会全部解码失败被静默跳过。
            # 此处仅替换本文件所建实例的 _decoder，不改动 pareto_zoom.py。
            if encoding_type != "precision_split_38":
                pz._decoder = _build_decoder(encoding_type)
            archive, rounds = pz.run()
            return archive.get_objective_matrix(), archive.front_size, extra
        except (NotImplementedError, RuntimeError, ImportError):
            # 求解器不可用 → 返回空矩阵，由 run() 记占位结果用于框架验证
            extra["solver_available"] = False
            return np.zeros((0, 3)), 0, extra

    def _grid_search_fixed_lambda(
        self, encoding_type: str, seed: Optional[int] = None
    ) -> Tuple[float, Dict]:
        """方案 BASE-4（Validation Scheme v1.1 §4.2 :160）：
        在 {0.1, 0.5, 1, 5, 10, 50, 100} 上穷举搜索最优固定 λ。

        流程：对每个 λ，在试跑权重组（默认 COARSE_WEIGHTS 12 组）上
        以小预算（grid_search_reads 次采样）构建 → 求解 → 解码 →
        评估，收集试跑目标矩阵；全部 λ 的试跑解集合并后按统一参考点
        （set_unified_reference, margin=0.10）算 HV，取 HV 最大者
        作为本次的最优固定 λ，随后用该 λ 跑完整 ParetoZoom 流程。

        复用说明：exploration/pareto_zoom.py:573 已有
        penalty_strategy='grid_search' 分支，但其网格为实验 2 的
        {0.01, 0.05, 0.1, 0.5, 1.0, 5.0}（comparison.py:81），与
        BASE-4 口径不符，且无试跑预算参数；pareto_zoom.py 由其他
        代理负责，故此处用现有组件（QUBOBuilder / KaiwuSolver /
        解码器 / 目标计算器）实现 BASE-4 最小版本，不改动
        pareto_zoom.py。试跑的 TOP-10 截取与 2% 可行性过滤同
        ParetoZoom._solve_single_iteration 口径
        （方案 §4.2.3 阶段D 步骤35）。

        Args:
            encoding_type: 编码方案（决定 QUBOBuilder 变量布局）。
            seed: 试跑求解器随机种子（B-1，run() 逐 rep 传 seed+rep）；
                None 保持旧行为（每次随机）。

        Returns:
            (best_lambda, detail)：detail 含网格、各 λ 试跑 HV、
            最优 λ 与试跑预算，供实验报告追溯。

        Raises:
            NotImplementedError/ImportError: 求解器不可用时上抛，
                由 _run_single_config 捕获走占位结果。
            RuntimeError: 全部 λ 试跑均无有效解时抛出。
        """
        # 延迟导入：保持模块导入开销最小，且避免与 pareto_zoom.py 的
        # 组件构造分叉（试跑必须与完整流程用同一套物理/编码组件）。
        from ..qubo.builder import QUBOBuilder
        from ..solver.kaiwu_solver import KaiwuSolver
        from ..objectives.mixing_enthalpy import MixingEnthalpy
        from ..objectives.density import VegardDensity
        from ..objectives.cost import WeightedCost

        solver = self._pz_kwargs.get("solver") or KaiwuSolver(mode="auto", seed=seed)
        builder = QUBOBuilder(
            encoding_type=encoding_type,
            gamma_discount=self._pz_kwargs.get("gamma_discount"),
        )
        # 编码感知解码（precision_split_38 走原 PrecisionSplitDecoder；
        # unified_* 走本文件 _UnifiedGridDecoder，见该类注释）
        decoder = _build_decoder(encoding_type)
        dh_calc = MixingEnthalpy()
        density_calc = VegardDensity()
        cost_calc = WeightedCost()

        weights_list = (
            list(self._grid_search_weights)
            if self._grid_search_weights
            else list(COARSE_WEIGHTS)
        )

        mats: Dict[float, np.ndarray] = {}
        for lam in GRID_SEARCH_LAMBDA_GRID:
            pts: List[Tuple[float, float, float]] = []
            for w in weights_list:
                try:
                    model = builder.build_model(
                        weights=w,
                        lambda_carbide=lam,
                        lambda_ccr=lam,
                    )
                    result = solver.solve_from_model(
                        model,
                        n_vars=builder.num_variables,
                        num_reads=self._grid_search_reads,
                    )
                except (NotImplementedError, ImportError):
                    raise  # 求解器不可用 → 整个配置走占位
                except RuntimeError:
                    continue  # 单 (λ, w) 试跑失败不中断网格
                for sol in result.solutions[:10]:  # TOP-10，同 ParetoZoom
                    try:
                        comp = decoder.decode(sol.bits)
                    except ValueError:
                        continue
                    # 2% 可行性过滤（方案 §4.2.3 阶段D 步骤35，
                    # 与 ParetoZoom 入档容差 archive_feasible_tol=2.0 同口径）
                    if abs(comp.total - 100.0) > 2.0:
                        continue
                    pts.append((
                        dh_calc.evaluate(comp.fractions),
                        density_calc.evaluate(comp.fractions),
                        cost_calc.evaluate(comp.fractions),
                    ))
            mats[lam] = np.asarray(pts, dtype=float).reshape(-1, 3)

        if not any(len(m) > 0 for m in mats.values()):
            raise RuntimeError("BASE-4 grid-search 试跑全部失败：无有效解")

        # 全部 λ 的试跑解集合并统一定参考点，同一参考点比 HV
        # （P0-5 口径；试跑选 λ 也须固定参考点才可比）
        hv_per_lambda = _compute_hv_unified(
            {repr(lam): m for lam, m in mats.items()}
        )
        best_lambda = max(
            GRID_SEARCH_LAMBDA_GRID,
            key=lambda l: hv_per_lambda[repr(l)],
        )
        detail = {
            "scheme": "BASE-4 (AP3_QUBO_Validation_Scheme_v1.1 §4.2 :160)",
            "grid": list(GRID_SEARCH_LAMBDA_GRID),
            "hv_per_lambda": {
                float(lam): hv_per_lambda[repr(lam)]
                for lam in GRID_SEARCH_LAMBDA_GRID
            },
            "best_lambda": float(best_lambda),
            "grid_search_reads": self._grid_search_reads,
            "trial_weights": len(weights_list),
        }
        return float(best_lambda), detail

    def compute_contributions(
        self,
        results: Dict[str, List[AblationResult]],
    ) -> Dict[str, float]:
        """计算各创新的消融贡献率 AR_i 和协同效应。

        方案口径（完整技术路线v2.0 :544-546 / 第一性原理解释版 :409-411）：
          ΔHV_i   = HV_Abl-i - HV_Full（越负，贡献越大）
          AR_i    = |ΔHV_i| / HV_Full × 100%（目标：每个 AR_i > 3%）
          Synergy = Σ|AR_i| - |HV_Abl-4 - HV_Full| / HV_Full × 100%
                    （> 0 表示三创新存在正协同）

        第 3 批修正：AR 原实现为带负号的 (HV_Abl - HV_Full) / HV_Full，
        与方案 |ΔHV| 口径不符，已修正为绝对值。

        Args:
            results: run() 的输出。

        Returns:
            {"AR_PrecisionSplit": ..., "AR_PenaltyFlex": ..., "AR_ParetoZoom": ..., "Synergy": ...}
        """
        hv_full = np.mean([r.hv for r in results.get("Full", []) if r.hv > 0])
        if hv_full < 1e-12:
            return {}

        hv_abl = {}
        for name in ["Abl-1", "Abl-2", "Abl-3", "Abl-4"]:
            vals = [r.hv for r in results.get(name, []) if r.hv > 0]
            hv_abl[name] = np.mean(vals) if vals else 0.0

        ar_mapping = {
            "AR_PrecisionSplit": "Abl-1",
            "AR_PenaltyFlex": "Abl-2",
            "AR_ParetoZoom": "Abl-3",
        }

        contributions = {}
        sum_ar = 0.0
        for ar_name, abl_name in ar_mapping.items():
            # 方案口径：AR_i = |ΔHV_i| / HV_Full × 100%
            ar = abs(hv_abl[abl_name] - hv_full) / hv_full * 100.0
            contributions[ar_name] = round(ar, 2)
            sum_ar += ar

        # 协同效应：Σ|AR_i| − |ΔHV_Abl-4| / HV_Full × 100%
        synergy = sum_ar - abs(hv_abl["Abl-4"] - hv_full) / hv_full * 100.0
        contributions["Synergy"] = round(synergy, 2)

        return contributions
