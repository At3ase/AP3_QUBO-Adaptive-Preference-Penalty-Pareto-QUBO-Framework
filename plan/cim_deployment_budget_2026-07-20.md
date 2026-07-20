# CIM 真机部署预算评估

**日期**: 2026-07-20
**状态**: 评估报告（待真机窗口确认）
**前置依赖**: 模拟器三实验（Exp 0/2/3 全部完成，见 `data/results/formal_exp*_reps20/`）

---

## 一、为什么不需要全量重跑

CIM 在 AP³-QUBO 框架中的角色是 **Ising 采样器的物理替代**——它不改变：

- QUBO 构建（`qubo/builder.py`）
- PenaltyFlex 自适应 λ 逻辑（`penalty_flex/`）
- ParetoZoom 五阶段探索策略（`exploration/`）
- 物理过滤管线（`validation/physical_filters.py`）

因此，CIM 真机只需验证三个核心问题：

| 验证目标 | 对应问题 |
|----------|----------|
| 硬件可行性 | 38-bit QUBO 能否在真机上正确映射和采样？ |
| 解质量 | CIM 采样质量 vs 模拟器 SA，差距多大？ |
| 物理过滤 | CIM 产生的解能否通过物理可行性检查？ |

这些问题 **只需少量关键配置的对比实验**，不需要 20 reps × 多配置的全量重跑。

---

## 二、硬件约束前提

### 2.1 QUBO 规模

```
比特数: 38（5 主元 × 7 bit + C × 3 bit）
耦合预算: C(38,2) = 703 条（方案 §五 设计基线）
实际耦合: 703 条（理论推导；P0 惩罚 (Σc−100)² 使 38 变量全稠密，恰为 C(38,2)）
```

> ✅ 口径勘误（2026-07-20）：此前记录的"实际耦合 1128 条、超预算 60%"系误读。
> 1128 = C(48,2)，实测自 Exp 0 中 Abl-1/Abl-4 的 unified_48 消融编码（48 变量，
> P0 惩罚导致全稠密），并非部署目标 Full 配置（precision_split_38）。证据：
> `data/results/exp0_bg_run.log:15` 的 D-06 告警仅出现于 Exp 0（含 48-bit 配置），
> Exp 2 / Exp 3（纯 38-bit）零告警；D-06 告警文本未报告被测实例的变量数，是误读根源。
> **38-bit Full 配置耦合数恰为预算上限 703，未超预算**，真机部署无稀疏化前置需求。
> 附带结论：PrecisionSplit 编码将耦合数从 1128（48-bit）降至 703（38-bit），降幅 38%，
> 这是其对 CIM 可部署性的核心贡献。
> ⚠️ 残余风险：703 为全稠密耦合，仍需确认目标 CIM 芯片的耦合拓扑能否原生支持
> 全连接（或需 embedding 及 chain 长度评估）。后续方向：constraint-by-construction
> 编码以消除 P0 全连接结构。

### 2.2 求解器模式切换

```python
# 模拟器（当前默认）
KaiwuSolver(mode="simulator", sa_sweeps=500)

# CIM 真机（需完整版 kaiwu SDK + license）
KaiwuSolver(mode="cim")
# 需: kaiwu.cim 模块 + license.lic（platform.qboson.com）
```

---

## 三、运行预算分级方案

### 方案 A：最低可行（Proof of Life）

| 项目 | 数值 |
|------|------|
| 内容 | Phase A 粗网格 12 权重 × 1 次求解 |
| λ 参数 | 模拟器预收敛的最优 λ（跳过 PenaltyFlex 搜索） |
| num_reads | 200 |
| **QUBO 提交次数** | **12** |
| **物理退火总次数** | **2,400** |
| 预计真机时间 | ~1-2 分钟 |
| 能回答的问题 | 硬件能否运行？解是否物理可行？ |

### 方案 B：标准验证（★ 推荐）

| 项目 | 数值 |
|------|------|
| Phase A 基础覆盖 | 12 权重 × 1 次 = 12 次 |
| 代表性 Focus 权重 | 5 个 × 1 次 = 5 次 |
| CIM vs 模拟器同种子对比 | (12 + 5) 组配对的模拟器重复 = 17 次 |
| 每组 λ 参数 | 模拟器预收敛 |
| num_reads | 200 |
| **QUBO 提交次数** | **35** |
| **物理退火总次数** | **7,000** |
| 预计真机时间 | ~3-5 分钟 |

对比分析能力：
- CIM vs SA 的配对 Wilcoxon 检验
- HV、前沿点数、物理可行性率的三维对比
- CIM 解的成分分布 vs 模拟器解的分布

### 方案 C：完整验证（论文级）

| 项目 | 数值 |
|------|------|
| Full ParetoZoom 1 rep（Full 配置 + 预收敛 λ） | ~50 次 |
| 3 组消融对比（−PenaltyFlex / −ParetoZoom / Baseline） | ~20 次 |
| CIM vs 模拟器同配置对比 | 各 1 rep |
| num_reads | 200 |
| **QUBO 提交次数** | **70** |
| **物理退火总次数** | **14,000** |
| 预计真机时间 | ~5-10 分钟 |

产出：CIM vs Simulator 完整消融对比图（`pareto_front_2d/3d`, `hv_boxplot`, `composition_heatmap`）

---

## 四、省钱策略

| 策略 | 节省比例 | 实施方式 |
|------|----------|----------|
| **λ 预收敛** | ~90% | PenaltyFlex 的 15 轮 λ 搜索全在模拟器完成，真机只跑最终收敛的 λ 值 |
| **降低 num_reads** | 50-80% | 模拟器 500 reads → CIM 100-200 reads（CIM 单次采样质量通常高于 SA） |
| **不做统计重复** | ~95% | 模拟器 20 reps → CIM 1 rep（CIM 的物理噪声本质上提供了重复） |
| **只跑 Full 配置** | ~60% | 消融实验的编码/惩罚/探索对比在模拟器上已完成 |

### 省钱逻辑说明

```
模拟器的角色：算法开发 + 超参搜索 + 统计显著性检验
    ↓ （全在离线完成）
CIM 真机的角色：物理验证 + 质量对比（目标：确认 CIM ≥ SA）
    ↓
只提交「已验证有效」的最小 QUBO 集合
```

---

## 五、操作流程

### 5.1 真机前置准备（离线完成）

1. 从模拟器运行中提取最优 PenaltyFlex λ 值
2. 固化 ParetoZoom 权重序列（不依赖在线 HV 反馈）
3. 将 QUBO 矩阵序列化保存（跳过在线构建开销）
4. 编写真机专用启动脚本（mode="cim"，预加载 λ 和权重）

### 5.2 真机运行日

```
1. 部署 license.lic
2. 验证 kaiwu.cim 导入
3. 运行 proof-of-life：1 次小采样 → 确认结果非空
4. 运行方案 B/C 提交序列
5. 收集 SolverResult → 本地落盘
6. 离线后处理：物理过滤 → Pareto 排序 → HV 计算 → 对比统计
```

### 5.3 后处理（离线完成）

- 配对 Wilcoxon：CIM vs SA（同一 QUBO + 同一 seed）
- HV 对比：CIM 前沿 vs SA 前沿
- 物理可行性率：CIM 解的约束满足比例
- 成分分布：CIM 解的 Al-Co-Cr-Fe-Ni-C 分布 vs 理论预测

---

## 六、风险与缓解

| 风险 | 概率 | 缓解措施 |
|------|------|----------|
| 全稠密耦合（703 条）超 CIM 芯片拓扑承载能力 | 中 | 提前确认芯片耦合拓扑；必要时评估 embedding chain 长度或稀疏化弱耦合 |
| CIM 采样质量低于 SA | 低 | 增大 num_reads 或做多轮融合 |
| license 不可用 | 低 | 提前 1 周确认玻色量子授权 |
| CIM 返回非法成分（和=100% 违反） | 中 | 物理过滤层已有 P0 硬约束检查，自动剔除 |

---

## 七、参考

- 方案文档：`plan/完整技术路线v2.0.md`（§五：硬件适配）
- 求解器实现：`src/ap3_qubo/solver/kaiwu_solver.py`
- 问题规模校验：`kaiwu_solver.py:_validate_problem_scale()` (L295)
- CIM 采样路径：`kaiwu_solver.py:_sample_spins_cim()` (L658)
- 实验驱动：`scripts/run_experiments.py`
- 物理参数：`src/ap3_qubo/physical_params.py`
