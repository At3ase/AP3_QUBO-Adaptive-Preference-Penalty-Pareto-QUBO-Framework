# CLAUDE.md — AP³-QUBO 项目指南

## 项目概述

AP³-QUBO（Adaptive Preference-Penalty Pareto QUBO）是面向 CIM 光量子计算机的高熵合金 Al-Co-Cr-Fe-Ni-C 六元成分多目标优化框架。

- **编码**: PrecisionSplit 38 比特（5 主元 × 7 bits + C × 3 bits），步长 0.25 at%
- **目标**: ΔH_mix（Miedema 混合焓）、ρ（Vegard 密度）、成本指数，三目标最小化
- **约束**: P0 成分和=100%（硬约束）、P1 碳化物抑制（软约束）、P2 C-Cr 耦合（软约束）
- **求解**: kaiwu SDK → Ising 采样 → TOP-K 解码，当前后端为内置模拟退火
- **算法**: PenaltyFlex 自适应罚函数（内层 λ 控制）+ ParetoZoom 五阶段动态前沿探索（外层 w 控制）

## 环境

- **Python 3.10**（硬性要求，kaiwu SDK 兼容版本）
- kaiwu ≥ 1.3（QUBO 构建与求解）
- numpy, scipy, matplotlib
- deap ≥ 1.4（NSGA-II 基线，实验 3 必需）

```bash
pip install .            # 基础安装
pip install .[experiments]  # 含 NSGA-II 基线
pip install .[dev]       # 含 pytest
```

## 目录结构

```
D:\QUBO\
├── src/ap3_qubo/           # 主包
│   ├── physical_params.py  # ★ 唯一数据源：所有物理参数、编码参数、配置常量
│   ├── encoding/           # L1: PrecisionSplit 编码/解码
│   ├── objectives/         # L2: 三目标函数 + 归一化
│   ├── constraints/        # 约束参考实现（对账用；主路径在 qubo/builder.py）
│   ├── qubo/               # L3: QUBO 构建器（kaiwu SDK 原生 API）
│   ├── penalty_flex/       # L4: 自适应罚函数（内层 λ 控制）
│   ├── exploration/        # L5: ParetoZoom 五阶段前沿探索（外层 w 控制）
│   ├── solver/             # 求解器抽象 + kaiwu 实现
│   ├── validation/         # 物理过滤 / Pareto 排序 / HV 计算
│   ├── statistics/         # 假设检验 / 效应量 / 报告生成
│   ├── experiments/        # 实验矩阵（消融 / 对比 / γ 敏感性 / NSGA-II 基线）
│   └── visualization/      # 前沿 2D/3D / HV 收敛 / 成分热力图 / λ 轨迹
├── tests/                  # 集成验证 + 冒烟测试 + 手动验证脚本
├── scripts/                # 实验驱动脚本 (run_experiments.py) + 启动脚本
├── plan/                   # ★ 方案文档与审查报告（权威参考）
├── data/                   # 实验运行数据（results/ 子目录不跟踪）
├── pyproject.toml          # 项目配置
└── README.md               # 用户文档
```

## 核心架构（分层数据流）

```
physical_params.py  ←── 唯一数据源，所有模块从此导入参数
        │
   ┌────┴────┐
   │ encoding │  ←── L1: Composition ↔ 38 bits
   └────┬────┘
   ┌────┴─────┐
   │objectives│  ←── L2: f₁(ΔH_mix), f₂(ρ), f₃(cost) + 归一化
   └────┬─────┘
   ┌────┴────┐
   │  qubo   │  ←── L3: kaiwu Binary 表达式 → QUBOMatrix
   └────┬────┘
   ┌────┴──────┐
   │penalty_flex│  ←── L4: 内层自适应 λ_carbide, λ_ccr
   └────┬──────┘
   ┌────┴──────┐
   │exploration│  ←── L5: 外层 ParetoZoom (w₁,w₂,w₃) 五阶段探索
   └────┬──────┘
   ┌────┴────┐
   │ solver  │  ←── kaiwu CIM / 内置 SA → TOP-K Solution
   └─────────┘
```

## 编码规范

### 物理参数: 单点真理源
`physical_params.py` 是所有物理常数和配置的唯一定义位置。其他模块不得重复定义这些值。使用 `@dataclass(frozen=True)` 确保不可变性。

### kaiwu 延迟导入
`qubo/builder.py` 和 `solver/kaiwu_solver.py` 采用延迟导入 kaiwu SDK：
- 模块级 `_kw = None`，首次调用 `_ensure_kaiwu()` 时完成 `import kaiwu`
- 允许实验包在无 kaiwu 环境下正常导入（ImportError 仅在调用求解/构建时抛出）
- 错误信息必须包含安装指引

### 求解器模式
- `"simulator"`: 内置 SA，离线可用（当前默认）
- `"cim"`: CIM 真机（需完整版 kaiwu SDK + 玻色量子授权）
- `"auto"`: 模拟器优先门禁（D-04 方案要求先跑通模拟器再上真机）
- 禁止静默回退或返回伪解

### 验证与对账
约束模块 (`constraints/`) 提供独立的参考实现，用于与主路径 `qubo/builder.py` 对账。两处实现应保持一致的数学语义。

## 常用命令

```bash
# 环境：必须使用 Python 3.10（默认 python3.12 无 kaiwu）
PY="C:/Users/At3ase/AppData/Local/Programs/Python/Python310/python.exe"

# 冒烟测试（快速验证全链路，~30s）
$PY tests/smoke_simulator_e2e.py

# 跑全部实验（保底顺序 0→2→3），正式规模
$PY scripts/run_experiments.py --experiment all --reps 20

# 链路验证模式（3 权重 + 小采样，分钟级）
$PY scripts/run_experiments.py --experiment all --quick --reps 1

# 单个实验
$PY scripts/run_experiments.py --experiment 2 --reps 3

# 跑单元测试
$PY -m pytest tests/ -v
```

## 实验矩阵

| 实验 | 模块 | 说明 |
|------|------|------|
| 0 | `experiments/ablation.py` | 消融：量化三创新各自贡献 |
| 1 | `experiments/comparison.py` | PrecisionSplit vs 统一编码 |
| 2 | `experiments/comparison.py` | PenaltyFlex vs Grid-Search/Linear/Fixed |
| 3 | `experiments/comparison.py` | ParetoZoom vs Uniform/NSGA-II/Random |
| 4 | `experiments/sensitivity.py` | γ 折扣因子敏感性分析 |
| 5 | (未实现) | Pycalphad 相图验证层 |

保底执行顺序: 0 → 2 → 3（消融 → PenaltyFlex → ParetoZoom）

## 注意事项

- 方案文档在 `plan/` 目录，是功能设计与公式的权威参考
- 代码审查报告: `plan/Code_Completion_Review_2026-07-18.md`（含 P0 级问题清单）
- 所有 `scratch/` 和 `data/results/` 不纳入版本控制
- `tests/manual/` 中的 verify 脚本为一次性手动验证，非自动化测试
