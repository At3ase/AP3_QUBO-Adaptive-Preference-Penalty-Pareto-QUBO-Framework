# scripts/run_experiments.py 使用说明

AP³-QUBO 实验驱动脚本（第 3 批）：**跑实验 → 落盘 → 出图**，一条命令产出初步实验结果。

## 环境要求（硬性）

必须用 Python 3.10 环境（已装 kaiwu 1.3.1 + scipy + numpy + deap 1.4）：

```
C:\Users\At3ase\AppData\Local\Programs\Python\Python310\python.exe scripts\run_experiments.py ...
```

默认 `python`（3.12）**没有 kaiwu**，会在求解阶段失败。

## 常用命令

```bash
# 链路冒烟（分钟级）：quick = 3 权重 + 小采样 + 压缩轮数，仅验证链路
python310 scripts/run_experiments.py --experiment 2 --quick --reps 1

# 小规模初步结果（默认 reps=3）
python310 scripts/run_experiments.py --experiment all

# 正式实验（方案 H-04 口径：实验 0/2 重复 20+ 次，关闭 quick，num_reads=1000）
python310 scripts/run_experiments.py --experiment 0 --reps 20
python310 scripts/run_experiments.py --experiment 2 --reps 20
python310 scripts/run_experiments.py --experiment 3 --reps 30
```

## CLI 参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--experiment {0,2,3,all}` | `all` | `all` 按保底顺序 实验0消融 → 实验2 PenaltyFlex对比 → 实验3 ParetoZoom对比 |
| `--reps N` | `3` | 重复次数（默认小规模冒烟；正式按方案 20/30） |
| `--num-reads N` | 1000（quick 120） | 每次 QUBO 求解的采样数，贯通到后端采样循环（方案 D-05） |
| `--out DIR` | `data/results/<时间戳>/` | 输出目录（相对路径基于项目根 D:\QUBO） |
| `--quick` | 关 | 链路验证模式：3 权重、网格 6、加密轮数 1、PenaltyFlex t_max 6、NSGA-II 30×30。**仅验证链路，不用于结论** |
| `--seed` | 42 | 随机种子基数（逐 rep 递增） |
| `--solver-mode` | `simulator` | 求解器模式；`auto` 按方案 D-04 门禁同样解析为 simulator 后端 |
| `--sa-sweeps` | 500（quick 200） | 模拟退火每 read 扫描步数 |

## 产出清单

```
<out>/
├── run_log.txt                        # 全程日志（与控制台同步）
├── summary.json                       # 全部实验汇总 + 运行配置 + 耗时
└── exp0/ exp2/ exp3/
    ├── results.json                   # 指标 + 统计（mean/std/CI95/min/max）
    ├── records.csv                    # 逐 rep 原始记录
    ├── report.md                      # 统计报告（Mann-Whitney + Cohen's d）
    ├── representative_front.npz       # 代表性 run：前沿目标矩阵/HV 历史/λ 轨迹/权重
    ├── representative_compositions.csv# 前沿成分表
    └── *.png                          # pareto_front_2d/3d, hv_convergence,
                                       # lambda_trajectory, composition_heatmap,
                                       # element_distribution, hv_boxplot
```

## 关键实现说明（驱动层补丁，不改 src 源码）

1. **simulator 求解器注入**：`compare_penalty` / `compare_exploration` / `AblationRunner`
   均不暴露 solver 参数，内部直接构造 `ParetoZoom()`。本脚本在自身进程内对
   `ablation` / `comparison` 模块命名空间中的 `ParetoZoom` 打工厂补丁，显式注入
   `KaiwuSolver(mode='simulator')`。等价于默认 `mode='auto'` 行为（方案 D-04 门禁
   解析为 simulator 后端），但更明确、且贯通了 `--num-reads` / `--sa-sweeps`。
2. **reps 在驱动层外循环**：`ExperimentStats.add_metric` 已修复为追加/累积语义
   （A-1 修复，`statistics/reporting.py:31-53`），`compare_*` 内部逐 rep 聚合已无失真。
   本脚本仍逐 rep 以 `n_repetitions=1` 调用并自行聚合——保留该结构是为让驱动层
   完全掌控逐 rep 原始记录（`records.csv`）与失败 rep 容错，语义与单次调用内
   逐 rep 统一 HV 参考点一致（P0-5 修复口径不变）。
3. **quick 缩放手法**与 `smoke_simulator_e2e.py` 一致（3 权重 +
   `dataclasses.replace` 压缩 `t_max_rounds`），另压缩 PenaltyFlex `t_max`、
   `uniform_grid_n`、NSGA-II pop×gen（NSGA-II 为 `comparison.py` 函数内局部
   import，补丁打在 `nsga2_baseline` 模块属性上；采用**子类化补丁**——继承真
   `NSGA2Optimizer` 并在 `__init__` 强制覆盖 pop_size/generations 为 quick 规模，
   以保证 `nsga2_baseline` 内部经模块全局名调用的
   `NSGA2Optimizer._setup_bounds()` / `_project_to_box_simplex()` 等静态方法
   在补丁后仍可用；lambda/工厂函数补丁会因缺少这些方法而 AttributeError）。
4. **λ 轨迹**由入档记录 `SolutionRecord.lambdas` 折叠连续重复值重建
   （PenalyFlex 自适应策略下逐迭代变化）；HV 收敛曲线取 `archive.get_hv_history()`。
5. 每个实验额外跑一次**代表性 run**（AP³ 完整管线：PenaltyFlex 自适应 +
   ParetoZoom）用于出图与 npz 原始数据，与统计 reps 相互独立。
