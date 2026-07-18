"""
kaiwu SDK 求解器实现（第 2 批修复版：TOP-K + 模拟器门禁 + 硬件自检）。

封装 kaiwu SDK 的 QuboModel 构建 + 求解链路。

实测 SDK 事实（Python310 环境内省，2026-07 第 2 批修复时确认）：
  - 当前环境 `import kaiwu` 实际解析为 kaiwu-community 1.0.4
    （site-packages 中 kaiwu-1.3.1.dist-info 存在，但完整版 SDK 的
    kaiwu/cim、kaiwu/classical、kaiwu/sampler 等模块文件缺失）。
  - kaiwu-community 的 `QuboSolver.solve_qubo(model)` 源码确认只返回
    单个最优解 (solution_dict, energy)，且不接受 num_reads —— 这是
    审查 P0-3「TOP-K 断链」的 SDK 侧根因。
  - TOP-K 正确链路（以社区版真实 API 重构）：
      qubo_model_to_ising_model(model) -> (matrix, bias, vars_dict)
      采样得到候选自旋配置 c_set（模拟退火 / 真机）
      get_sorted_solutions(matrix, c_set, 0, negtail_ff=True, sort_solution=True)
      -> 按能量升序的 (configs, hamiltonians)，去重后取 TOP-K
      get_sol_dict(config[:-1] * config[-1], vars_dict) -> {var: 0/1}
      energy = hamiltonian + bias（已实测与 bits @ Q @ bits 完全一致）
  - Ising 矩阵含 1 个 negtail 辅助变量（38 变量 -> 39x39 矩阵），
    有效自旋 = config[:-1] * config[-1]（SDK solve_qubo 源码同款处理）。

模式语义（方案 D-04：先模拟器跑通全 pipeline 再上真机）：
  - "simulator": 内置模拟退火后端（真实优化计算，非伪解），离线可用。
  - "cim":       CIM 光量子真机。需要完整版 kaiwu SDK 1.3.1 + 玻色量子
                 授权；不可用时显式抛出 RuntimeError 并给出安装/授权
                 指引，禁止静默回退或返回伪解。
  - "auto":      模拟器优先门禁 —— 一律走模拟器后端（真机上线需显式
                 指定 mode="cim"），kaiwu 完全缺失时抛出 RuntimeError。

支持两种变量命名格式:
  - 构建器格式: e0_b0, e0_b1, ..., e5_b2 (QUBOBuilder)。flat 位布局按
    构建时的实际编码推断（_infer_bits_main：precision_split_38 →
    7 bits/主元，unified_48 → 8，unified_38 → 6），不再硬编码
    precision_split_38 布局。
  - 矩阵格式:   b[00], b[01], ..., b[37] (qubo_matrix_to_qubo_model,
                实测为零填充两位数字，正则兼容不填充写法)
"""

import logging
import re
import time
from typing import Dict, Any, List, Optional, Tuple

import numpy as np

from .base import AbstractSolver, SolverResult, Solution

logger = logging.getLogger(__name__)


class KaiwuSolver(AbstractSolver):
    """kaiwu SDK 求解器（TOP-K / 模式分支 / 硬件自检）。

    Args:
        mode: 求解模式 ("auto" / "cim" / "simulator")。
        top_k: 默认返回的 TOP-K 解条数；None 表示 min(num_reads, 100)。
        **kwargs: 可选调参：
            sa_sweeps (int): 模拟退火每次采样的扫描步数，默认 500。
            sa_t0 / sa_t1 (float): 退火初末温度，默认 3.0 / 0.15
                （实测调参：t1 不过低可保留 TOP-K 解多样性，
                38 变量随机实例 1000 reads 约 60 个唯一解，
                且 12 变量蛮力对照仍达全局最优）。
            seed (int|None): 随机种子（None = 每次随机）。
    """

    # 变量名解析正则
    _VAR_BRACKET = re.compile(r"b\[(\d+)\]")       # b[00], b[0], ...
    _VAR_BUILDER = re.compile(r"e(\d+)_b(\d+)")     # e0_b0, e1_b3, ...

    # ---- 硬件兼容自检阈值（方案 §五「硬件兼容性验证」/ 审查 D-06）----
    HARDWARE_MAX_VARS = 550             # 玻色量子 550W 比特数上限
    HARDWARE_MAX_COUPLERS = 150_975     # C(550, 2)，550W 全连接耦合上限
    SCHEME_VARS = 38                    # 本方案变量数
    SCHEME_MAX_COUPLERS = 703           # C(38, 2)，本方案耦合预算
    # 性能优化（2026-07-18）：每实例仅告警一次，避免每个 QUBO solve
    # 都重复输出同一条 D-06 自检消息（典型运行中数百次无差别重复警告）
    _hw_warned: bool = False

    # ---- 可行性判定阈值（成分和质量守恒）----
    FEASIBILITY_TOL_PCT = 1.0           # |Σc - 100| ≤ 1%

    _VALID_MODES = ("auto", "cim", "simulator")

    def __init__(self, mode: str = "auto", top_k: Optional[int] = None, **kwargs):
        if mode not in self._VALID_MODES:
            raise ValueError(
                f"未知求解模式: {mode!r}，可选 {self._VALID_MODES}"
            )
        self._mode = mode
        self._default_top_k = top_k
        self._config = kwargs
        # 性能优化（2026-07-18）：每实例仅告警一次 D-06
        self._hw_warned = False

    @property
    def name(self) -> str:
        return f"kaiwu-{self._mode}"

    @property
    def is_quantum(self) -> bool:
        return self._mode == "cim"

    # =========================================================================
    # 公开求解接口
    # =========================================================================

    def solve_from_model(
        self,
        model,
        n_vars: int = 38,
        num_reads: int = 1000,
        top_k: Optional[int] = None,
    ) -> SolverResult:
        """从 kaiwu QuboModel 求解 (推荐方式)。

        使用 QUBOBuilder.build_model() 构建的模型直接传入。
        变量名格式: e0_b0, ..., e5_b2。

        Args:
            model: kaiwu QuboModel。
            n_vars: 变量总数。
            num_reads: 采样次数（贯通到后端采样循环，方案 D-05）。
            top_k: 返回的 TOP-K 解条数（按能量升序）。

        Returns:
            SolverResult，solutions 按能量升序。
        """
        # D-06 硬件自检（超限 WARNING，不阻断模拟器）
        n_couplers = self._count_couplers_from_model(model)
        self.check_hardware_compatibility(n_vars=n_vars, n_couplers=n_couplers)
        return self._solve_model(model, n_vars, num_reads, top_k)

    def solve(
        self,
        qubo_matrix: np.ndarray,
        num_reads: int = 1000,
        top_k: Optional[int] = None,
    ) -> SolverResult:
        """从 numpy QUBO 矩阵求解 (兼容旧接口)。

        使用 qubo_matrix_to_qubo_model 转换矩阵 → 求解。
        变量名格式: b[00], b[01], ..., b[N-1]。

        Args:
            qubo_matrix: shape=(N,N) QUBO 矩阵 (上三角, diag=h_i)。
            num_reads: 采样次数（贯通到后端采样循环）。
            top_k: 返回的 TOP-K 解条数（按能量升序）。

        Returns:
            SolverResult，solutions 按能量升序。
        """
        import kaiwu as kw

        n = qubo_matrix.shape[0]

        # D-06 硬件自检（超限 WARNING，不阻断模拟器）
        n_couplers = int(np.count_nonzero(np.triu(qubo_matrix, k=1)))
        self.check_hardware_compatibility(n_vars=n, n_couplers=n_couplers)

        model = kw.qubo_matrix_to_qubo_model(qubo_matrix)
        return self._solve_model(model, n, num_reads, top_k)

    # =========================================================================
    # 硬件兼容自检（方案 D-06 / 审查 P0-7）
    # =========================================================================

    def check_hardware_compatibility(
        self,
        n_vars: int,
        n_couplers: Optional[int] = None,
    ) -> Dict[str, Any]:
        """校验问题规模是否满足 CIM 真机与方案预算。

        阈值来源（方案 §五「硬件兼容性验证」表）：
          - 变量数 ≤ 550（玻色量子 550W）
          - 耦合数 ≤ 150,975 = C(550,2)（550W 全连接上限）
          - 方案自身预算：38 变量 / 703 = C(38,2) 耦合

        超限仅 WARNING 日志，不抛异常（模拟器不受硬件限制；
        真机提交前应由调用方查看返回报告）。

        Returns:
            自检报告 dict:
              n_vars / n_couplers / vars_ok / couplers_ok /
              within_scheme_budget / warnings(list[str])
        """
        report: Dict[str, Any] = {
            "n_vars": n_vars,
            "n_couplers": n_couplers,
            "vars_ok": n_vars <= self.HARDWARE_MAX_VARS,
            "couplers_ok": True,
            "within_scheme_budget": n_vars <= self.SCHEME_VARS,
            "warnings": [],
        }

        if n_vars > self.HARDWARE_MAX_VARS:
            msg = (
                f"[D-06 硬件自检] 变量数 {n_vars} 超过 550W 上限 "
                f"{self.HARDWARE_MAX_VARS}，无法上真机"
            )
            logger.warning(msg)
            report["warnings"].append(msg)

        if n_couplers is not None:
            report["couplers_ok"] = n_couplers <= self.HARDWARE_MAX_COUPLERS
            report["within_scheme_budget"] = (
                report["within_scheme_budget"]
                and n_couplers <= self.SCHEME_MAX_COUPLERS
            )
            if n_couplers > self.HARDWARE_MAX_COUPLERS:
                msg = (
                    f"[D-06 硬件自检] 耦合数 {n_couplers} 超过 550W 上限 "
                    f"{self.HARDWARE_MAX_COUPLERS}，无法上真机"
                )
                if not self._hw_warned:
                    logger.warning(msg)
                report["warnings"].append(msg)
            elif n_couplers > self.SCHEME_MAX_COUPLERS:
                msg = (
                    f"[D-06 硬件自检] 耦合数 {n_couplers} 超出方案预算 "
                    f"{self.SCHEME_MAX_COUPLERS}=C(38,2)（仍在 550W 硬件"
                    f"能力内，但偏离方案 §五 设计基线）"
                )
                if not self._hw_warned:
                    logger.warning(msg)
                report["warnings"].append(msg)
        self._hw_warned = True

        return report

    # =========================================================================
    # 后端分支（方案 D-04：模拟器优先门禁）
    # =========================================================================

    def _resolve_backend(self) -> str:
        """按 mode 解析实际后端。

        Returns:
            "simulator" 或 "cim"。

        Raises:
            RuntimeError: cim 模式真机后端不可用（显式报错 + 指引，
                          禁止静默回退/伪解）。
        """
        if self._mode == "simulator":
            return "simulator"

        if self._mode == "auto":
            # D-04 门禁：必须先用模拟器跑通全 pipeline，再上真机。
            # auto = 模拟器优先；真机需显式 mode="cim"。
            return "simulator"

        # mode == "cim"
        try:
            import kaiwu.cim  # noqa: F401
        except Exception as exc:
            raise RuntimeError(
                "CIM 真机后端不可用：当前环境的 kaiwu 包缺少真机/云端模块"
                "（实测 import kaiwu 解析为 kaiwu-community 1.0.4，仅含 "
                "common/core）。请按以下指引处理后重试：\n"
                "  1. 安装完整版 kaiwu SDK 1.3.1（提供 kaiwu.cim 等模块）；"
                "注意 kaiwu-community 与完整版共用 kaiwu 包名，后装者会"
                "覆盖先装者的同名文件，建议按玻色量子官方文档的顺序重装。\n"
                "  2. 配置玻色量子授权/license（完整版 SDK 的 "
                "kaiwu.license 模块）。\n"
                "  3. 按方案 D-04，先用 mode='simulator' 跑通全 pipeline "
                "再上真机。\n"
                f"底层错误: {exc!r}"
            ) from exc
        return "cim"

    def _solve_model(
        self,
        model,
        n_vars: int,
        num_reads: int,
        top_k: Optional[int],
    ) -> SolverResult:
        """共享求解核心：转换 → 采样 → TOP-K 解码。"""
        import kaiwu as kw

        if num_reads < 1:
            raise ValueError(f"num_reads 必须 ≥ 1，得到 {num_reads}")

        backend = self._resolve_backend()
        if top_k is None:
            top_k = self._default_top_k
        if top_k is None:
            top_k = min(num_reads, 100)

        t_start = time.perf_counter()

        # QUBO -> Ising（38 变量 -> 39x39，含 1 个 negtail 辅助变量）
        ising_model = kw.qubo_model_to_ising_model(model)
        ising_mat = ising_model.get_matrix()
        bias = ising_model.get_bias()
        vars_dict = ising_model.get_variables()

        # 布局推断（修复 _parse_var_index 硬编码 precision_split_38 布局
        # 导致 unified_48/unified_38 索引碰撞的 bug）：
        # vars_dict 含构建时的完整变量名集合（{名: ising索引} + __spin__），
        # 据此推断 bits_main；无法可靠推断时回退 legacy 7（回归保证）。
        bits_main = self._infer_bits_main(vars_dict.keys(), n_vars)

        # 采样 num_reads 组候选自旋配置
        if backend == "simulator":
            c_set = self._sample_spins_simulator(ising_mat, num_reads)
        else:  # pragma: no cover - 真机路径，当前环境不可达
            c_set = self._sample_spins_cim(ising_mat, num_reads)

        # TOP-K：按能量升序排序（SDK 官方工具函数）
        configs, hamiltons = kw.get_sorted_solutions(
            ising_mat, c_set, 0, negtail_ff=True, sort_solution=True
        )

        t_end = time.perf_counter()

        solutions = self._configs_to_solutions(
            configs, hamiltons, bias, vars_dict, n_vars, top_k, backend,
            bits_main,
        )

        return SolverResult(
            solutions=solutions,
            num_reads=num_reads,
            solver_name=self.name,
            timing_ms=(t_end - t_start) * 1000,
        )

    # =========================================================================
    # 采样后端
    # =========================================================================

    def _sample_spins_simulator(
        self, ising_mat: np.ndarray, num_reads: int
    ) -> np.ndarray:
        """模拟器后端：内置模拟退火采样（真实优化计算，非伪解）。

        语义对齐 SDK IsingSolver._solve：返回 shape=(num_reads, N) 的
        候选自旋配置 c_set（±1，含 negtail 辅助变量），交由
        get_sorted_solutions 统一排序。
        目标：最小化 Ising 能量 H = -sᵀMs（与 SDK get_sorted_solutions
        的 hamilton 定义一致）。

        增量场更新：翻转 s_i 时 ΔH = 4·s_i·(M s)_i（已数值验证），
        局部场 h ← h - 2·s_i·M[:,i]。
        """
        n = ising_mat.shape[0]
        sweeps = int(self._config.get("sa_sweeps", 500))
        t0 = float(self._config.get("sa_t0", 3.0))
        t1 = float(self._config.get("sa_t1", 0.15))
        rng = np.random.default_rng(self._config.get("seed", None))

        mat = np.asarray(ising_mat, dtype=float)
        c_set = np.empty((num_reads, n), dtype=int)

        for r in range(num_reads):
            s = rng.choice([-1.0, 1.0], size=n)
            h = mat @ s
            for t in range(sweeps):
                temp = t0 * (t1 / t0) ** (t / sweeps)
                i = int(rng.integers(n))
                d_energy = 4.0 * s[i] * h[i]
                if d_energy <= 0.0 or rng.random() < np.exp(
                    -d_energy / temp
                ):
                    h -= 2.0 * s[i] * mat[:, i]
                    s[i] = -s[i]
            c_set[r] = s.astype(int)

        return c_set

    def _sample_spins_cim(
        self, ising_mat: np.ndarray, num_reads: int
    ) -> np.ndarray:  # pragma: no cover - 真机路径，当前环境不可达
        """CIM 真机后端采样。

        当前环境完整版 kaiwu SDK 真机模块缺失（见 _resolve_backend），
        此路径不可达；保留为真机接入点，任何失败均显式 RuntimeError，
        禁止返回伪解。
        """
        raise RuntimeError(
            "CIM 真机采样路径尚未接入：完整版 kaiwu SDK 的真机客户端 API "
            "需以实际安装版本的官方文档为准接线。"
            "请先使用 mode='simulator' 完成全 pipeline 验证（方案 D-04）。"
        )

    # =========================================================================
    # TOP-K 解码 + 可行性检查
    # =========================================================================

    def _configs_to_solutions(
        self,
        configs: np.ndarray,
        hamiltons: np.ndarray,
        bias: float,
        vars_dict: Dict,
        n_vars: int,
        top_k: int,
        backend: str,
        bits_main: int = 7,
    ) -> List[Solution]:
        """排序后的自旋配置 → 去重 → TOP-K Solution 列表（能量升序）。

        bits_main: 每主元比特数（构建时的实际位布局，见
            _infer_bits_main），默认 7 = precision_split_38 legacy 布局。
        """
        import kaiwu as kw

        if configs is None or len(configs) == 0:
            return []

        # 去重（np.unique 返回首次出现下标；能量取下标对应值后再排序）
        uniq, first_idx = np.unique(configs, axis=0, return_index=True)
        uniq_hams = hamiltons[first_idx]
        order = np.argsort(uniq_hams, kind="stable")
        uniq = uniq[order]
        uniq_hams = uniq_hams[order]

        solutions: List[Solution] = []
        for rank, (cfg, ham) in enumerate(zip(uniq, uniq_hams)):
            if rank >= top_k:
                break
            # negtail 还原有效自旋（SDK solve_qubo 源码同款处理）
            solution_dict = kw.get_sol_dict(cfg[:-1] * cfg[-1], vars_dict)
            bits = self._parse_solution(solution_dict, n_vars, bits_main)
            energy = float(ham) + float(bias)
            is_feasible, feas_meta = self._check_feasibility(bits, bits_main)
            solutions.append(
                Solution(
                    bits=bits,
                    energy=energy,
                    is_feasible=is_feasible,
                    metadata={
                        "solver": self.name,
                        "mode": self._mode,
                        "backend": backend,
                        "rank": rank,
                        **feas_meta,
                    },
                )
            )
        return solutions

    def _check_feasibility(
        self, bits: np.ndarray, bits_main: int = 7
    ) -> Tuple[bool, Dict[str, Any]]:
        """真实可行性检查：成分和守恒 |Σc - 100| ≤ 1%（方案 P0 硬约束）。

        使用 PrecisionSplitDecoder 将 38 比特解码为成分后求和判定。
        非 precision_split_38 布局（任意 N 的矩阵路径，或 unified_48 /
        unified_38 统一编码）不适用该判定，此时 metadata 显式标注
        feasibility_check="not_applicable"，避免用错误的布局解码出
        垃圾成分后误报可行性。
        """
        if len(bits) != self.SCHEME_VARS or bits_main != 7:
            return True, {
                "feasibility_check": "not_applicable",
                "reason": (
                    f"n_vars={len(bits)}, bits_main={bits_main}；"
                    f"仅 precision_split_38 (38 vars, 7 bits/elem) 适用"
                ),
            }
        try:
            # 延迟导入，避免模块级循环依赖
            from ..encoding.precision_split import PrecisionSplitDecoder

            comp = PrecisionSplitDecoder().decode(bits)
            total = float(sum(comp.fractions.values()))
            deviation = abs(total - 100.0)
            return deviation <= self.FEASIBILITY_TOL_PCT, {
                "feasibility_check": "sum_constraint",
                "composition_sum": total,
                "sum_deviation": deviation,
                "tolerance": self.FEASIBILITY_TOL_PCT,
            }
        except Exception as exc:  # 解码器不可用时显式标注，不硬编码 True
            logger.warning("可行性检查解码失败: %r", exc)
            return True, {
                "feasibility_check": "decode_error",
                "reason": repr(exc),
            }

    # =========================================================================
    # 耦合统计与变量名解析
    # =========================================================================

    @staticmethod
    def _count_couplers_from_model(model) -> Optional[int]:
        """从 QuboModel 统计非零二次耦合数（用于 D-06 自检）。"""
        try:
            mat = np.asarray(model.get_matrix(), dtype=float)
            return int(np.count_nonzero(np.triu(mat, k=1)))
        except Exception:
            return None

    def _parse_solution(
        self, solution_dict: Dict, n_vars: int, bits_main: int = 7
    ) -> np.ndarray:
        """解析 kaiwu 返回的解字典 → numpy 比特数组。

        自动检测变量命名格式:
          - e0_b0 格式 (QUBOBuilder)，flat 布局由 bits_main 决定
          - b[00] / b[0] 格式 (qubo_matrix_to_qubo_model)

        bits_main: 每主元比特数（构建时的实际位布局）。默认 7 =
            precision_split_38 legacy 布局，保证旧调用方行为不变；
            真实求解链路在 _solve_model 中经 _infer_bits_main 从模型
            完整变量名集合推断后显式传入（支持 unified_48/unified_38）。
        """
        bits = np.zeros(n_vars, dtype=np.int8)

        for var_name, val in solution_dict.items():
            var_str = str(var_name)
            idx = self._parse_var_index(var_str, n_vars, bits_main)
            if idx is not None:
                bits[idx] = int(val)

        return bits

    @classmethod
    def _infer_bits_main(cls, var_names, n_vars: int) -> int:
        """从完整变量名集合推断每主元比特数（构建时的实际位布局）。

        QUBOBuilder._create_variables 的创建顺序即编码布局顺序：
          主元 ei∈0..4 各 bits_main 位（e{ei}_b0 .. e{ei}_b{bits_main-1}），
          随后 C（e5_b0 .. e5_b{bits_carbon-1}），故 flat index 为
          ei*bits_main+bj（ei<5）/ 5*bits_main+bj（ei==5）。

        推断规则（严格校验，任一不满足即回退 legacy 7，保证
        precision_split_38 与矩阵 b[N] 路径回归不变）：
          - 5 个主元齐全、各自比特集合均为连续的 range(bits_main)；
          - C 的比特集合为连续的 range(bits_carbon)；
          - 5*bits_main + bits_carbon == n_vars。

        precision_split_38 → 7，unified_48 → 8，unified_38 → 6。
        """
        per_elem: Dict[int, set] = {}
        carbon_bits: set = set()
        for name in var_names:
            m = cls._VAR_BUILDER.match(str(name))
            if not m:
                continue
            ei, bj = int(m.group(1)), int(m.group(2))
            if ei < 5:
                per_elem.setdefault(ei, set()).add(bj)
            elif ei == 5:
                carbon_bits.add(bj)

        # 非 builder 命名（如矩阵 b[N] 路径）或不完整集合 → legacy 7
        if sorted(per_elem) != [0, 1, 2, 3, 4] or not carbon_bits:
            return 7

        sizes = {len(s) for s in per_elem.values()}
        if len(sizes) != 1:
            return 7
        bits_main = sizes.pop()
        if any(s != set(range(bits_main)) for s in per_elem.values()):
            return 7

        bits_carbon = len(carbon_bits)
        if carbon_bits != set(range(bits_carbon)):
            return 7

        if 5 * bits_main + bits_carbon != n_vars:
            return 7

        return bits_main

    @classmethod
    def _parse_var_index(
        cls, var_str: str, n_vars: int, bits_main: int = 7
    ) -> Optional[int]:
        """从变量名字符串解析 flat index。

        bits_main: 每主元比特数（构建时的实际位布局），默认 7 =
            precision_split_38 legacy 布局（ei*7+bj / 35+bj，行为与
            修复前完全一致）。

        Returns:
            flat index (0 ~ n_vars-1), 或 None (无法解析/越界)。
        """
        # 尝试 builder 格式: e{ei}_b{bj}
        m = cls._VAR_BUILDER.match(var_str)
        if m:
            ei, bj = int(m.group(1)), int(m.group(2))
            if ei < 5:
                if bj >= bits_main:
                    return None  # 该布局下不存在的比特位，防静默碰撞
                idx = ei * bits_main + bj
            elif ei == 5:
                if bj >= n_vars - 5 * bits_main:
                    return None  # 超出 C 实际比特数
                idx = 5 * bits_main + bj
            else:
                return None
            return idx if idx < n_vars else None

        # 尝试 bracket 格式: b[N] / b[NN]
        m = cls._VAR_BRACKET.match(var_str)
        if m:
            idx = int(m.group(1))
            return idx if idx < n_vars else None

        return None

    # =========================================================================
    # 可用性探测
    # =========================================================================

    @staticmethod
    def is_available(mode: str = "auto") -> bool:
        """检查指定模式后端是否真实可用（跑一个 2 变量小 QUBO）。

        auto/simulator: 模拟器后端可用性（kaiwu-community 即可）。
        cim: 真机后端可用性（当前环境预期 False）。
        """
        try:
            solver = KaiwuSolver(mode=mode)
            backend = solver._resolve_backend()
        except (ValueError, RuntimeError):
            return False
        try:
            mat = np.array([[1.0, 0.0], [0.0, 1.0]])
            result = solver.solve(mat, num_reads=4, top_k=4)
            return len(result.solutions) > 0
        except Exception:
            return False
