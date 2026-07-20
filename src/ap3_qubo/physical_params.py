"""
物理参数与配置常量 —— AP³-QUBO 框架的唯一数据源。

所有硬编码的物理常数、编码参数、约束参数、优化参数均定义在此模块中。
其他任何模块不得重复定义这些值。
"""

from dataclasses import dataclass, field
from typing import Dict, Tuple

# =============================================================================
# 元素清单
# =============================================================================

MAIN_ELEMENTS: Tuple[str, ...] = ("Al", "Co", "Cr", "Fe", "Ni")
INTERSTITIAL_ELEMENT: str = "C"
ALL_ELEMENTS: Tuple[str, ...] = MAIN_ELEMENTS + (INTERSTITIAL_ELEMENT,)

# =============================================================================
# 一、PrecisionSplit 编码参数
# =============================================================================


@dataclass(frozen=True)
class EncodingParams:
    """PrecisionSplit 分层精度编码参数。"""

    # 主元 (Al, Co, Cr, Fe, Ni)
    bits_main: int = 7
    base_main: float = 5.0  # at%
    step_main: float = 0.25  # at%

    # 间隙元素 C
    bits_carbon: int = 3
    base_carbon: float = 0.0  # at%
    step_carbon: float = 0.25  # at%

    # 导出量
    @property
    def total_variables(self) -> int:
        return len(MAIN_ELEMENTS) * self.bits_main + self.bits_carbon  # 38

    @property
    def main_levels(self) -> int:
        return 1 << self.bits_main  # 128

    @property
    def carbon_levels(self) -> int:
        return 1 << self.bits_carbon  # 8

    @property
    def main_min(self) -> float:
        return self.base_main  # 5.0%

    @property
    def main_max(self) -> float:
        return self.base_main + self.step_main * (self.main_levels - 1)  # 36.75%

    @property
    def carbon_min(self) -> float:
        return self.base_carbon  # 0.0%

    @property
    def carbon_max(self) -> float:
        return self.base_carbon + self.step_carbon * (self.carbon_levels - 1)  # 1.75%


ENCODING = EncodingParams()


# =============================================================================
# 二、Miedema 二元混合焓参数 (kJ/mol)
# =============================================================================


@dataclass(frozen=True)
class MiedemaParams:
    """Miedema 模型二元混合焓参数。

    ΔH_ij 矩阵 (5×5 对称，置换式主元间):
        来源: Takeuchi-Inoue 表
        单位: kJ/mol
    """

    # 置换式主元间混合焓 (5×5 对称矩阵下三角)
    dh_main: Dict[Tuple[str, str], float] = field(default_factory=lambda: {
        # 按字母序排序键
        ("Al", "Co"): -19.0,
        ("Al", "Cr"): -10.0,
        ("Al", "Fe"): -11.0,
        ("Al", "Ni"): -22.0,
        ("Co", "Cr"): -4.0,
        ("Co", "Fe"): -1.0,
        ("Co", "Ni"): 0.0,
        ("Cr", "Fe"): -1.0,
        ("Cr", "Ni"): -7.0,
        ("Fe", "Ni"): -2.0,
    })

    # C-TM 间隙-置换交叉混合焓 (γ 折扣前)
    # 来源: Takeuchi & Inoue 2005, Mater. Trans. 46:2817, Table 2
    # （Miedema 模型等原子比二元液态混合焓，C 行逐格直读；
    #   裁定记录见 references/cni_enthalpy_verdict.md）
    # 2026-07-19 修正: C-Ni −46→−39（原 −46 无任何文献出处，
    #   判定为邻元平均 (C-Co+C-Fe)/2=−46 的传抄错误）；
    #   C-Al −39→−36（原 −39 为相邻格 C-Si 串格）。
    # 锚点 ΔH_mix(等原子比 AlCoCrFeNi)=−12.32 不含 C 项，不受影响。
    dh_carbon: Dict[str, float] = field(default_factory=lambda: {
        "Al": -36.0,
        "Co": -42.0,
        "Cr": -61.0,
        "Fe": -50.0,
        "Ni": -39.0,
    })

    # C 的唯象折扣因子 (间隙占位使有效混合焓仅为置换混合的分数)
    # 2026-07-19 修正: 默认 0.25→0.3 —— 0.3 是敏感性网格
    # {0.1,0.2,0.3,0.4,0.5} 的中心点，默认值必须被敏感性实验覆盖
    # （方法学一致性，裁定记录见 references/gamma_literature_review.md）；
    # 0.3 仍落在方案文档区间 [0.2, 0.3] 内（上界）。
    gamma_discount: float = 0.3  # 推荐 0.2~0.3

    def get_dh(self, elem_a: str, elem_b: str) -> float:
        """获取两元素间的 Miedema 混合焓 (kJ/mol)。自动处理排序和 C 折扣。"""
        if elem_a == elem_b:
            return 0.0

        # C 与主元的交叉项 (折扣后)
        if elem_a == "C" and elem_b in MAIN_ELEMENTS:
            return self.gamma_discount * self.dh_carbon[elem_b]
        if elem_b == "C" and elem_a in MAIN_ELEMENTS:
            return self.gamma_discount * self.dh_carbon[elem_a]

        # 主元间
        key = tuple(sorted([elem_a, elem_b]))
        if key in self.dh_main:
            return self.dh_main[key]
        raise KeyError(f"Unknown element pair: ({elem_a}, {elem_b})")

    @property
    def dh_equiatomic(self) -> float:
        """等原子比 AlCoCrFeNi (C=0) 的预测 ΔH_mix (kJ/mol)。"""
        #  = 4 × Σ_{i<j} ΔH_ij × (0.2)²
        # 10 对: sum = -19-10-11-22-4-1+0-1-7-2 = -77
        return 4.0 * (-77.0) * 0.04  # = -12.32 kJ/mol


MIEDEMA = MiedemaParams()


# =============================================================================
# 三、元素物理性质表
# =============================================================================


@dataclass(frozen=True)
class ElementProperties:
    """各元素的物理性质常量。"""

    # 纯元素密度 (g/cm³) — f₂ Vegard 定律
    densities: Dict[str, float] = field(default_factory=lambda: {
        "Al": 2.70,
        "Co": 8.86,
        "Cr": 7.19,
        "Fe": 7.87,
        "Ni": 8.91,
        "C": 2.27,  # 石墨值近似
    })

    # 原子半径 (Å) — δ 判据 (仅主元; C 不参与)
    atomic_radii: Dict[str, float] = field(default_factory=lambda: {
        "Al": 1.43,
        "Co": 1.25,
        "Cr": 1.28,
        "Fe": 1.26,
        "Ni": 1.24,
    })

    # 熔点 (K) — Ω 判据
    melting_points: Dict[str, float] = field(default_factory=lambda: {
        "Al": 933,
        "Co": 1768,
        "Cr": 2180,
        "Fe": 1811,
        "Ni": 1728,
        "C": 3800,
    })

    # 价电子浓度 VEC_i — VEC 后处理探针 (仅主元)
    vec_values: Dict[str, int] = field(default_factory=lambda: {
        "Al": 3,
        "Co": 9,
        "Cr": 6,
        "Fe": 8,
        "Ni": 10,
    })

    # Pauling 电负性 — Δχ 判据 (仅主元)
    electronegativity: Dict[str, float] = field(default_factory=lambda: {
        "Al": 1.61,
        "Co": 1.88,
        "Cr": 1.66,
        "Fe": 1.83,
        "Ni": 1.91,
    })

    # 成本相对权重 — f₃ 成本指数
    cost_weights: Dict[str, float] = field(default_factory=lambda: {
        "Co": 100.0,
        "Ni": 40.0,
        "Cr": 30.0,
        "Al": 6.0,
        "Fe": 2.0,
        "C": 2.0,
    })

    def density_of(self, elem: str) -> float:
        return self.densities[elem]

    def radius_of(self, elem: str) -> float:
        return self.atomic_radii[elem]

    def melting_point_of(self, elem: str) -> float:
        return self.melting_points[elem]

    def vec_of(self, elem: str) -> int:
        return self.vec_values[elem]

    def en_of(self, elem: str) -> float:
        return self.electronegativity[elem]

    def cost_weight_of(self, elem: str) -> float:
        return self.cost_weights[elem]


ELEM = ElementProperties()

# 等原子比 AlCoCrFeNi 密度 (手工验算: 7.106 g/cm³ ≈ 7.11)
DENSITY_EQUIATOMIC = sum(ELEM.densities[e] for e in MAIN_ELEMENTS) / 5.0  # 7.106


# =============================================================================
# 四、约束参数
# =============================================================================


@dataclass(frozen=True)
class ConstraintParams:
    """约束与惩罚参数。"""

    # P0 硬约束: 成分和=100% (质量守恒, 不可权衡, PenaltyFlex 不控制)
    lambda_sum_fixed: float = 15.0  # 推荐 10~20

    # P1 软约束: 碳化物抑制
    omega_carbide: float = 0.2  # 偏好权重 0.1~0.3
    lambda_carbide_init: float = 0.05  # PenaltyFlex 初始值
    carbide_soft_upper: float = 0.8  # at% 软上限

    # P2 软约束: C-Cr 耦合最小化
    omega_ccr: float = 0.5  # 偏好权重 0.3~1.0
    lambda_ccr_init: float = 0.05  # PenaltyFlex 初始值
    ccr_h_max: float = 64.3125  # = 1.75 × 36.75 (%²) 归一化分母

    # PenaltyFlex 算法参数
    alpha_add: float = 0.5  # 加性启动阶段学习率
    alpha_mult: float = 0.8  # 乘性演化阶段学习率
    epsilon_explore: float = 0.02  # 探索期期望偏离度
    epsilon_consolidate: float = 0.0  # 固化期期望偏离度
    gamma_convergence: float = 0.1  # λ 相对变化收敛阈值
    t_add: int = 2  # 加性启动轮数
    # 性能优化（2026-07-18）：从 15 降至 8。
    # 实际收敛行为观察：自适应策略在 t=5~7 时 Δλ/λ 已进入 γ=0.1 以下，
    # t=15 的后半段多数为无效迭代（λ 已夹紧在 [0.005, 5.0] 边界）。
    # 降至 8 后单权重 PenaltyFlex 内循环时间减半，且仍保留 3 轮
    # early-stop 余量（is_converged 需连续 2 轮 <γ）。
    t_max: int = 8  # 最大迭代轮数（原 15）

    # CIM 精度适配
    cim_noise_floor: float = 0.01  # P2 交叉项截断阈值


CONSTRAINT = ConstraintParams()


# =============================================================================
# 五、ParetoZoom 探索参数
# =============================================================================


@dataclass(frozen=True)
class ParetoZoomParams:
    """ParetoZoom 动态前沿探索参数。"""

    coarse_grid_size: int = 12  # 粗网格初始权重数
    t_max_rounds: int = 5  # 最大探索轮数
    epsilon_hv: float = 0.01  # HV 增长收敛阈值 (1%)
    d_threshold_factor: float = 0.15  # 间隙检测阈值因子 (×max_edge_length)
    sigma_perturb: float = 0.08  # 高斯微扰标准差
    num_reads: int = 500  # 每次 QUBO 求解采样数（当前默认后端为内置经典 SA；原 1000，性能优化降至 500）
    hv_delta_factor: float = 0.10  # 参考点边距 (10%)
    convexity_warning: float = 0.10  # 非凸比例告警阈值
    weight_dedup_tol: float = 0.05  # 权重去重容差
    weight_min_bound: float = 0.02  # 权重下界 (避免退化)
    # 性能优化（2026-07-18）：每轮新增权重上限。
    # 诊断发现：仅 1 个初始权重即可在一轮内生成 19 个间隙权重，
    # 每个需跑完整 PenaltyFlex 内循环（t_max=8，约 1~2s/权重）。
    # 无上限时前沿越大→间隙越多→权重爆炸→运行时间不可控。
    # 10 个上限按间隙距离降序取 top-N，保证优先填充最大间隙，
    # 且限制每轮最多新增 10×8=80 次 QUBO 求解。
    max_new_weights_per_round: int = 10  # 每轮新增权重数上限（0=不限制）
    nsga_pop_size: int = 100  # NSGA-II 种群大小
    nsga_generations: int = 200  # NSGA-II 迭代代数


PARETO_ZOOM = ParetoZoomParams()


# =============================================================================
# 六、粗网格权重 (12 组 G1~G12)
# =============================================================================

COARSE_WEIGHTS: Tuple[Tuple[float, float, float], ...] = (
    # G1~G3: 顶点 (纯目标)
    (1.0, 0.0, 0.0),  # G1: 纯混合焓
    (0.0, 1.0, 0.0),  # G2: 纯密度
    (0.0, 0.0, 1.0),  # G3: 纯成本
    # G4~G6: 边中点
    (0.5, 0.5, 0.0),  # G4: 混合焓-密度
    (0.5, 0.0, 0.5),  # G5: 混合焓-成本
    (0.0, 0.5, 0.5),  # G6: 密度-成本
    # G7: 中心点
    (1 / 3, 1 / 3, 1 / 3),  # G7: 三目标均衡
    # G8~G12: 非对称方向
    (0.7, 0.2, 0.1),  # G8: 混合焓强 + 密度次
    (0.2, 0.7, 0.1),  # G9: 密度强 + 混合焓次
    (0.2, 0.1, 0.7),  # G10: 成本强 + 混合焓次
    (0.1, 0.7, 0.2),  # G11: 密度强 + 成本次
    (0.1, 0.2, 0.7),  # G12: 成本强 + 密度次
)


# =============================================================================
# 七、目标归一化常数 (物理先验)
# =============================================================================

# f₁: ΔH_mix 范围 ~[-20, +5] kJ/mol, 跨度~25 → 归一化分母 30
F1_NORM_DENOM = 30.0
# f₂: 密度范围 ~[6.0, 9.0] g/cm³, 跨度~3 → 归一化分母 10
F2_NORM_DENOM = 10.0
# f₃: 成本指数, c_max 预计算 = max(各 c_i × w_i) 的合理上限
# 最贵情况: Co 36.75% × 100 + Ni 36.75% × 40 + Cr 36.75% × 30 + Al 36.75% × 6 + Fe 36.75% × 2 + C 1.75% × 2
F3_COST_MAX = (
    36.75 * 100 + 36.75 * 40 + 36.75 * 30
    + 36.75 * 6 + 36.75 * 2 + 1.75 * 2
)  # = 6545.0 (超立方体最大成本, 归一化上界)

# =============================================================================
# 八、物理过滤器阈值
# =============================================================================

# VEC 探针区间
VEC_LOWER: float = 7.0
VEC_UPPER: float = 7.6

# δ 判据 (Hume-Rothery)
DELTA_THRESHOLD: float = 6.6  # %

# Ω 判据分级
OMEGA_STABLE: float = 1.1  # ≥: 热力学稳定
OMEGA_METASTABLE: float = 0.8  # [0.8, 1.1): 亚稳态

# Δχ 判据
DELTA_CHI_THRESHOLD: float = 0.133

# HEA 定义范围
HEA_ELEMENT_MIN: float = 5.0  # at%
HEA_ELEMENT_MAX: float = 35.0  # at%

# 混合焓合理区间 (Takeuchi-Inoue)
DH_MIX_LOWER: float = -15.0  # kJ/mol
DH_MIX_UPPER: float = 10.0  # kJ/mol

# 成分和容差
SUM_TOLERANCE: float = 1.0  # %

# 碳化物风险阈值
CARBIDE_RISK_ABSOLUTE: float = 1.0  # at% C
CARBIDE_RISK_WARNING: float = 0.5  # at% C
CCR_COUPLING_RISK: float = 0.3  # c_C × c_Cr / 64.3125
