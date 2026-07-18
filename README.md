# AP³-QUBO：高熵合金成分多目标优化的量子 QUBO 框架

AP³-QUBO（Adaptive Preference-Penalty Pareto QUBO）面向 CIM 光量子计算机的高熵合金成分优化框架。将 Al-Co-Cr-Fe-Ni-C 六元成分空间编码为 38 比特 QUBO，以混合焓 ΔH_mix、密度 ρ、成本为三目标，通过 PenaltyFlex 自适应罚函数与 ParetoZoom 动态加密探索求解 Pareto 前沿，并与经典 NSGA-II 基线做公平对比。

## 环境要求

- **Python 3.10**（kaiwu SDK 兼容版本；本项目在 Python 3.10 + kaiwu 1.3.1 下验证）
- [kaiwu](https://kaiwu.qboson.com/) ≥ 1.3（玻色量子 SDK，QUBO 求解与 CIM 真机接入；真机可用前由内置经典 SA 后端离线占位）
- numpy ≥ 1.24、scipy ≥ 1.10、matplotlib ≥ 3.7
- deap ≥ 1.4（实验 3 的 NSGA-II 对照组，**方案必需实验**，缺失时基线显式报错而非静默降级）

## 安装

```bash
# 基础安装
pip install .

# 跑对比/消融实验（含 deap）
pip install .[experiments]

# 开发（pytest）
pip install .[dev]

# Pycalphad 相图验证层（实验 5）
pip install .[calphad]
```

## 快速开始

```python
from ap3_qubo.exploration.pareto_zoom import ParetoZoom

# 运行 AP³ 五阶段探索，得到 Pareto 存档
pz = ParetoZoom(encoding_type="precision_split_38")
archive, rounds = pz.run()
print(archive.get_objective_matrix())

# NSGA-II 基线对照（需 pip install .[experiments]）
from ap3_qubo.experiments.nsga2_baseline import NSGA2Optimizer
result = NSGA2Optimizer(pop_size=40, generations=20).optimize_and_evaluate()
print(result["algorithm"], result["hv"])
```

对比实验入口见 `src/ap3_qubo/experiments/`：`comparison.py`（实验 1~3）、`sensitivity.py`（实验 4 γ 敏感性）、`ablation.py`（消融实验）。跨方法 HV 统一参考点由 `validation.hypervolume.set_unified_reference` 保证。

## 目录结构

```
src/ap3_qubo/
├── encoding/        # PrecisionSplit 38 比特编码与变量映射
├── objectives/      # 三目标：ΔH_mix（Miedema）、Vegard 密度、加权成本
├── constraints/     # P0 成分和 / 碳化物 / C-Cr 耦合约束（参考实现）
├── qubo/            # QUBO 构建主路径（builder）
├── penalty_flex/    # PenaltyFlex 自适应罚函数
├── exploration/     # ParetoZoom 五阶段探索与 Archive
├── solver/          # kaiwu CIM 求解器接口
├── validation/      # HV / Pareto 排序 / 物理过滤（Ω、δ、VEC）
├── statistics/      # 统计检验与报告
├── experiments/     # 实验 1~4 与 NSGA-II 基线
└── visualization/   # 前沿可视化
tests/               # 集成验证脚本
plan/                # 方案文档与代码审查报告
```
