# Penalty_Auditor 审计报告：罚函数与探索逻辑

- 审计基准：`plan/完整技术路线v2.0.md` §4.3/§5；`plan/hea_encoding_scheme_v1.13.md` §3.2.5（PenaltyFlex）、§4.2（ParetoZoom）
- 审计对象：`src/ap3_qubo/penalty_flex/adaptive_penalty.py`、`warm_start.py`；`src/ap3_qubo/exploration/pareto_zoom.py`、`weight_utils.py`、`archive.py`
- 方法：逐行对照 + 数值探针 `audit_probes/probe_penalty_pareto.py`（52 PASS / 0 FAIL / 15 INFO）+ mock solver 端到端冒烟
- 日期锚：2026-07-18；运行环境 `"$DAIMON_USER_PYTHON"`（Python 3.10 + kaiwu 1.3.1），`PYTHONPATH=src`

---

## 一、与方案一致项（数值探针验证通过）

### 1.1 PenaltyFlex（基准 v1.13 §3.2.5 :314-345）

| 项目 | 方案 | 代码 | 探针 |
|---|---|---|---|
| P0 固定 λ_sum | 10~20，不参与自适应 | `physical_params.py:241` = 15.0 | PASS |
| 初始 λ_soft | (0.05, 0.05) | `:245/:250` | PASS |
| 加性启动 | t ≤ T_add=2（含端点），λ+α_add·v̄，α_add=0.5 | `adaptive_penalty.py:191-194, 252-257` | PASS（t=1,2 手算一致；t=2 仍 additive） |
| 乘性更新 | λ·exp(α·tanh(v̄−ε))，α=0.8 | `:269-276` | PASS（t=3 手算 0.3572571412 逐位一致） |
| 振荡几何平均 | √(λ_t·λ_{t-1}) | `:258-268`（`_prev_prev_*` 历史上移正确） | PASS（√(0.3572571×0.25)=0.298854957 一致） |
| 收敛门控 | MAX_j\|Δλ_j\|/λ_j < γ=0.1 且 t > T_add | `:296-307`（门控 t>T_add 正确；轮数见偏离 2.2） | PASS |
| 违反度定义 | v_carbide=(c_C−0.8)²；v_CCr=c_C·c_Cr/64.3125 | `:396-404` | PASS |
| TOP-10 反馈 | 目标值最优前 10 取均值 | `analyze_top_k :407-460`（先排序再取 k） | PASS |

### 1.2 ParetoZoom（基准 v1.13 §4.2 / v2.0 §5）

| 阶段 | 方案 | 代码 | 探针 |
|---|---|---|---|
| 12 组粗网格 | G1~G12 精确值 | `physical_params.py:311-328` | PASS（逐组匹配） |
| 凸性检验 | 三投影平面、排序轴 (f1,f1,f2)、叉积 cross<0、ratio=count/\|P_proj\|、max>10% 告警 | `pareto_zoom.py:328-387` | PASS（构造凹陷三点 max_ratio=1/3、warning=True 与手算一致） |
| 间隙检测 | sort by f1，d_threshold=0.15×max_edge，中点插值 | `:698-736` + `weight_utils.from_gaps` | PASS |
| HV 热点微扰 | contribution=HV(P)−HV(P_wo)，高斯 σ=0.08、投影单纯形 | `:742-762` + `weight_utils.perturb`（修复后空间一致，见 3.1） | PASS |
| 去重 | tol=0.05 欧氏距离 | `weight_utils.deduplicate_weights` | PASS |
| 边界保护 | REMOVE ANY(w_i<0.02) | **修复后补齐**（见 3.2） | PASS |
| HV 收敛 | \|ΔHV\|/HV < 1% | `:192-200`（连续 2 轮，见偏离 2.3）；hv_before 同参考点重算（:460-464） | PASS |
| 参考点 | r = max + 10%×range | `validation/hypervolume.py:74-94` | PASS |
| 入档可行性 | constraints_satisfied tol=2% | `:663`（\|Σc−100\| ≤ 2.0） | PASS |

### 1.3 warm-start λ 缓存

- 读写均归一化键（`warm_start.py:42-46, 58-62`），等比例权重 (2,3,5)/(0.2,0.3,0.5) 命中一致：PASS。
- 查找链：显式 warm_start → `find_nearest`（欧氏距离，max_distance=0.5）→ 回退 (0.05,0.05)（`pareto_zoom.py:519-527`），终值写回（`:556-561`），与方案 §4.2.5 步骤 3-5 一致。
- 端到端冒烟：3 组初始 + 3 轮探索共 24 组权重，缓存 24 条，逐权重一条，无串键。

---

## 二、文档化偏离（代码注释已声明，非错误）

| # | 偏离 | 位置 | 说明 |
|---|---|---|---|
| 2.1 | T_max 15 → 8 | `physical_params.py:260-265` | 注释声明性能优化，观察 λ 在 t=5~7 已进入 γ 以下；保留 3 轮 early-stop 余量 |
| 2.2 | PenaltyFlex 收敛需连续 2 轮 <γ（方案单轮 BREAK） | `adaptive_penalty.py:296-307` | 注释自称"加严" |
| 2.3 | ParetoZoom HV 收敛连续 2 轮（方案 v1.13:585/v2.0:485 为单轮） | `pareto_zoom.py:9, 195-200` | 与任务书"HV<1% 连续 2 轮"口径一致；**方案文档未登记此变更** |
| 2.4 | λ clamp [0.005, 5.0] | `adaptive_penalty.py:351-353` | 方案无此项，安全边界 |
| 2.5 | ε 切换边界 t < T_add+2=4 用探索期 0.02 | `adaptive_penalty.py:201-207` | 方案仅"探索期2%/固化期0%"，未定义切换轮次 |
| 2.6 | max_new_weights_per_round=10 截断 | `physical_params.py:293-299`、`pareto_zoom.py:432-434` | 方案无上限；间隙优先排序后取 top-10，注释声明性能优化 |

## 三、本轮最小修复（2 处，均已实测验证）

### 3.1 P-D3：HV 热点贡献计算空间错配（明确逻辑错误，已修复）

- **问题**：`_generate_perturbations` 原用 `get_objective_matrix_norm()`（归一化）调 `marginal_contribution`，而 `self._hv_calc` 参考点由 `get_objective_matrix()`（原始尺度）设定（`pareto_zoom.py:317, 459`）→ "归一化点 × 原始参考点"空间错配。
- **探针实证**：修复前贡献 [0.1099, 0, 0, 0]（热点区分度被压没），修复后 [0.084, 1.72, 4.974, 0.6736]。
- **修复**：`pareto_zoom.py:755` 一行改 `get_objective_matrix()`。验证：捕获传入 `perturb_hotspots` 的贡献向量与手算原始空间边际贡献逐位一致（PASS）；mock solver 冒烟 3 轮微扰正常生成。
- **影响面**：仅 hotspot 加权采样的微扰数分配（1~3 个/权重），不改存档/前沿语义。

### 3.2 P-D2：边界保护步骤缺失（方案步骤遗漏，已修复）

- **问题**：方案 v1.13 §4.2.3 阶段C 步骤 25 `REMOVE w from G_new if ANY(wᵢ < 0.02)` 在合并管线中缺失。间隙中点可产生 0 分量权重：G1(1,0,0)×G4(0.5,0.5,0) 的中点 = (0.75, 0.25, 0.0)；微扰路径 clip[0.02,1] 后再归一化亦不严格保证 ≥0.02（探针：(0.02,1,1)→w1≈0.0099）。
- **修复**：`pareto_zoom.py:432-438`，dedup 后、`max_per_round` 截断前加 `new_weights = [w for w in new_weights if min(w) >= w_min]`（w_min=weight_min_bound=0.02），位置与方案步骤顺序一致。
- **验证**：探针构造混合权重列表过滤语义 PASS；mock 冒烟 3 轮 21 组新权重 0 越界。

## 四、疑点清单（未修复，建议上级决策）

| # | 级别 | 疑点 | 证据 |
|---|---|---|---|
| P-D1 | P1 | **振荡检测判据与方案字面不符**：方案 :336-338 用 λ 轨迹变号 `sign(λ^t−λ^{t-1}) ≠ sign(λ^{t-1}−λ^{t-2})` 且逐约束判定；`adaptive_penalty.py:322-349` 用反馈方向 `sign(v̄−ε)` 代理，且两约束任一振荡即联合几何平均。纯乘性阶段二者等价，但在 ①加性→乘性切换（t=3）、②ε 切换（t=4）、③clamp 饱和时分歧。探针实例：v̄=[0.4,0,0.4] 时代码 0.250 vs 方案乘性 0.334；t=4 v̄:0→0.3 且 ε:0.02→0 时代码几何平均 0.0496 vs 方案乘性 0.0621。修复需按 λ 实际 Δ 符号逐约束追踪，属设计决策而非一行修复 | `adaptive_penalty.py:322-349`；探针 T6/T4-附加 |
| P-D4 | P2 | **num_reads 死参数且注释误导**：`physical_params.py:288` 注释"性能优化降至 500"，但 `pareto_zoom.py:644` 调 `solve_from_model` 未传 num_reads → 实际走求解器默认 1000（=方案值）。行为正确，注释/参数失实 | grep 全仓 num_reads 引用 |
| P-D5 | P1 | **阶段D 候选数 TOP-20 → TOP-10**：方案 v1.13:577 `FOR sol ∈ TOP-20(solutions)`；`pareto_zoom.py:651` `result.solutions[:10]`。求解器按能量升序排序（`kaiwu_solver.py:313-316`），故语义是"TOP-10"无误，但数量与方案不符且无注释。涉及 exp0 已产数据可比性，建议实验系列完成后统一对齐 | `pareto_zoom.py:651` |
| P-D6 | P2 | 方案偏离 2.3（HV 连续 2 轮收敛）未回写方案文档；若方案以文档为唯一基准，建议在 plan 文档补登记（本代理不改 plan/） | `pareto_zoom.py:9` |
| P-D7 | P3 | `_detect_gaps` 首句 `from_gaps(..., [])` 空调用为死代码（返回 [] 后被覆盖），无害可清理 | `pareto_zoom.py:705-708` |

## 五、范围外发现（转交对应代理）

- `solver/kaiwu_solver.py:700`：`kw.get_sol_dict` 在 kaiwu 1.3.1 下 `AttributeError`（mock 冒烟绕过真实求解路径后发现）。真实求解链路当前不可用 → 转 Solver/kaiwu 集成负责代理。本代理未改 solver/。

## 六、环境干扰确认

- exp0（PID 31144）与 exp2（PID 29628）均为单进程 `scripts/run_experiments.py --reps 1`（PowerShell 命令行确认），源码内无 subprocess/multiprocessing 派生 → 运行进程持有旧模块内存映像，本轮源码修复不影响在跑批次；未 kill 任何进程、未提交 git、未改 plan/ 与 solver/、qubo/、experiments/。
