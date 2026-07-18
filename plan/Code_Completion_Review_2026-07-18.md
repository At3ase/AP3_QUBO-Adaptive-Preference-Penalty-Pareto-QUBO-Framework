# AP³-QUBO 代码完成度审查报告

**审查日期**：2026-07-18
**审查基准**：`hea_encoding_scheme_v1.13.md`、`完整技术路线v2.0.md`、`AP3_QUBO_P0P1补充修改文档.md`、`AP3_QUBO_Validation_Scheme_v1.1.md`、`AP3_Validation_Review_Report_v1.0.md`、`integration_plan.md`
**代码范围**：`src/ap3_qubo/` 全部 13 个子模块（约 6,500 行）+ `tests/verify_kaiwu_integration.py`
**审查方式**：2 个方案解析代理提取约 130 项需求清单 → 7 个模块审查代理逐项对照代码（file:line 证据）

---

## 一、总体结论

**整体完成度约 60%**：架构骨架完整、编码层与物理参数表高度忠实于方案，但存在 **2 处 P0 级公式错误**、**1 条断链（TOP-K）**，且验证层/实验层/统计层完成度不足一半，**当前状态不具备跑对比实验和上真机的条件**。

| 模块组 | 完成度 | 状态概述 |
|---|---|---|
| 编码 + 物理参数（A、数据表） | ~90% | 忠实实现，比特方向/边界/精度严谨 |
| 目标函数 + 归一化（B） | ~80% | 公式全部正确，存在死参数/死代码/口径分裂 |
| 约束 + QUBO 构建 + 求解器（C、D） | ~65% | 双实现路径互不一致，P0 尺度错 5625 倍 |
| PenaltyFlex 罚函数（E） | ~65% | 参数体系完整，加性分支缺失、振荡公式用错操作数 |
| ParetoZoom 探索（F） | ~70% | 五阶段骨架完整，凸性检验缺失、3 个 P0 级偏离 |
| 验证 + 统计 + 物理过滤（G、HV、STAT） | ~45% | 基础探针可用，Ω 公式违背 P0 修正条款 |
| 实验 + 可视化 + 工程化（H、VIS、K） | ~45% | 开关式骨架合格，实验 5 缺失、公平性架空 |

---

## 二、方案核心修正条款的落实核查（重点）

| 修正条款 | 状态 | 证据 |
|---|---|---|
| C-04：λ_sum 固定 10~20、剥离 PenaltyFlex | ✅ 已落实 | `physical_params.py:241` λ=15.0；`adaptive_penalty.py:82-86` 仅自适应 λ_carbide/λ_CCr |
| C-05：碳化物放弃 max(0,·)² 改纯二次型 | ✅ 已落实 | `carbide_constraint.py:23-24`、`builder.py:326-328` |
| C-07：ω_CCr 默认 0.5 | ✅ 已落实 | `physical_params.py:249` |
| C-08：VEC 移出 QUBO 仅后处理 | ✅ 已落实 | builder 目标仅 f1/f2/f3；VEC 仅在 `physical_filters.py` |
| E-02：γ 收敛阈值收紧为 0.1 | ✅ 已落实 | `physical_params.py:258` |
| F-04：凸性检验升级为 CONVEXITY_TEST_3D | ❌ 未实现 | 参数 `convexity_warning=0.10` 是死配置，无任何消费方 |
| G-09：Ω 三级输出 + T_m 含 C + ΔS_mix 五主元归一化 | ⚠️ 三级输出有，公式两处偏离 | `physical_filters.py:321-345`：T_m 未含 C（:337-339 仅遍历主元）、ΔS_mix/T_m 未做五主元归一化重标定 |
| G-09 附则：Ω<1.1 不进推荐列表 | ⚠️ 矛盾 | `physical_filters.py:213` all_pass 放行 metastable（0.8≤Ω<1.1） |
| G-10：δ-Ω 联合判据 2×2 矩阵 | ❌ 未实现 | 全库无组合判定逻辑 |
| G-11：温度敏感性 4 条标记 | ❌ 基本未实现 | 仅 c_C·c_Cr/64.3125>0.3 数值形式存在；快冷/时效/双重风险三条缺失 |
| I-01~04：Pycalphad 升级为必做验证层（PSR≥70%） | ❌ 未实现 | 仅 docstring 提及，无任何接口/代码 |

---

## 三、P0 级问题清单（修复前不可跑实验、不可上真机）

### P0-1 P0 约束双实现相差 5625 倍，且生效的是错误一方
- 主路径 `qubo/builder.py:316`：`(Σc − 100)²` × λ=15 → 实际惩罚 `15·(Σc−100)²`
- 规范路径 `constraints/sum_constraint.py:38-39`：`λ·(Σc−100)²/5625`（C-02 量级归一化）
- **后果**：主路径 P0 强度是方案值的 5625 倍，f1/f2/f3 在哈密顿量中被淹没，偏好权重 w 实际失效；CIM 上系数动态范围爆炸。
- 附带：`sum_constraint.py:92-102` 的 `evaluate_penalty` 又是第三种尺度（差 10000 倍），诊断函数与 QUBO 项不可对账。

### P0-2 C-Cr 耦合 QUBO 线性项双倍计数（公式错误）
- `constraints/ccr_coupling.py:66-80`：`1.25·2^k/H_max`（= 5.0×0.25，Cr 基线贡献）被加了两次。
- 实测 h[35..37] = [0.03887, 0.07775, 0.15549]，恰为规范值 [0.01944, 0.03887, 0.07775] 的 **2 倍**。
- 注：主路径 `builder.py:337-339` 的 P2 实现是正确的——孤儿路径错公式、主路径错尺度，两处各错一处，无法互相印证。

### P0-3 TOP-K 断链：求解器只返回单解
- `kaiwu_solver.py:74-75,128-129`：`solve_qubo` 仅返回单个最优解；`num_reads` 记录但从未传给 SDK。
- 下游全部断粮：PenaltyFlex 的 TOP-10 反馈（`adaptive_penalty.py:373`）、ParetoZoom 的 TOP-20 解码（`pareto_zoom.py:520`）、D-05 的 num_reads=1000（`physical_params.py:283` 死配置）。

### P0-4 PenaltyFlex 加性启动分支整体缺失
- 方案 E-04 分支① `λ += α_add·v̄`（t≤T_add）在代码中不存在；`adaptive_penalty.py:229,244-251` 无论阶段均走乘性更新，"加性阶段"名不副实（乘性在 v̄<ε 时 λ 下降，纯加性只升不降，数学上不等价）。

### P0-5 跨方法 HV 参考点不统一，对比实验结论失真
- `comparison.py:175-177`、`sensitivity.py:150-154`、`nsga2_baseline.py:287`：每个方法对自己的 archive 单独 `set_reference_from_data`，违反 HV-1 固定参考点前提。实验 1/2/3/4 的组间 HV 差异部分来自参考点漂移而非前沿质量。**当前跑出的任何 HV 对比结论不可写入报告。**

### P0-6 deap 缺失时静默降级为"伪 NSGA-II"
- `nsga2_baseline.py:97-99`：deap 不可用时回退 `_optimize_simple`（无拥挤距离、无 selNSGA2），结果仍以 "NSGA-II" 标签上报。当前环境 deap 未安装，必然走回退分支。

### P0-7 真机前置条件全部缺失（D-04/D-06）
- `KaiwuSolver` 的 `mode ∈ {auto, cim, simulator}` 参数空转，无模拟器优先门禁、无 fallback 重试链（`kaiwu_solver.py:36-43,74,128`）。
- 硬件兼容自检（38≤550、耦合≤703）全库零命中。
- 本机环境 `import kaiwu` 失败，`experiments` 包顶层经 `builder.py:20` 传递依赖 kaiwu → 无 kaiwu 环境下整个实验包不可导入，设计的 ImportError 优雅降级不可达。

### P0-8 入档无可行性过滤（F-09）
- `pareto_zoom.py:520-548` 对所有解码解无条件入档，"约束满足 tol=2% 才入档"未实现，违反约束的解直接污染 Archive。

### P0-9 凸性检验 CONVEXITY_TEST_3D 整体缺失（F-04）
- 三投影平面、cross<0 计数、10% 阈值 WARNING 均无代码；`convexity_warning` 参数无消费方。

### P0-10 `top10_overlap` 指标造假嫌疑
- `sensitivity.py:110-111`：用 `min(hv_mean/baseline_hv, 1.0)` 冒充"与基准的 TOP10 解重叠率"，docstring 宣称的"重叠率>60%"无法成立。

---

## 四、P1 级重要缺口（影响结论可信度）

1. **PenaltyFlex 振荡几何平均用错操作数**：`adaptive_penalty.py:234-238` 算的是 √(λ_t·λ_乘性候选)，方案要求 √(λ_t·λ_{t-1})；`_prev_lambda_*` 保存后未用（:212-213）。
2. **收敛判定缺 T_add 门控**：`adaptive_penalty.py:271-277` 加性阶段可触发收敛，且引入方案外"连续两轮"条件；`is_converged` 与 T_max 预算耗尽混用同一标志。
3. **TOP-10 反馈语义失真**：集成点传 `solutions_data[-10:]`（累积列表尾部，非按目标排序的 TOP-10），`pareto_zoom.py:420,425`。
4. **HV 收敛判据缺绝对值**：`hypervolume.py:98` HV 下降 50% 时 delta=−0.5<0.01 会被误判收敛提前 BREAK；且 hv_before/hv_after 在不同参考点下计算（`pareto_zoom.py:306,341`），收敛信号混入参考点漂移。
5. **间隙检测阈值硬编码 0.02**：方案 `d_threshold=0.15×max_edge` 未接线（`pareto_zoom.py:578` vs `physical_params.py:281` 死参数）。
6. **实验配置与方案不符**：实验 2 固定 λ={0.05,1.0} 应为 {1,10,100}（`comparison.py:89-90`）；Abl-2 未做 Grid-search 硬编码 0.05（`ablation.py:159`）；实验 1 重复次数默认 20 < 方案要求 30（`comparison.py:21`）。
7. **统计框架缺一半**：配对 Wilcoxon、Friedman、Cohen's κ、F 检验、中位数+IQR 均未实现；Bonferroni 函数存在但 α'=0.0025/0.0042 无常量无调用，报告层输出未校正 p 值（`reporting.py:135`）。
8. **MET 三层指标大部分无计算逻辑**：HV_ratio>1.10、FR_diff>10%、SR>1.30、CAR≥80%、NDR=100%、LAR≥75% 全库无实现；C-metric 函数存在但无实验调用。
9. **BASE-9 公平性缺口**：评估次数无对齐（NSGA-II ≈20,000 次 vs ParetoZoom 口径不明）、wall-clock 无实验层计时。
10. **F-07 热点微扰偏差**：">0.2×均值"筛选门槛未实现；微扰次数默认 3 应为 2（`weight_utils.py:110,175-189`）。
11. **单位口径分裂**：`evaluate_array` 接口中 f₁/f₂ 用 [0,1] 分数、f₃ 用 at%，同数组喂三个计算器成本差 100 倍（`cost.py:41-44`）；c_max 双源不一致 6545 vs 6543（`physical_params.py:329` vs `normalization.py:36`）。
12. **README.md 缺失致 `pip install .` 构建失败**（`pyproject.toml:9`）；deap 被错归入 dev extra（H-04 是必需实验）。

---

## 五、死代码 / 养护缺口（不阻断但需清理）

- `schedule.py` 整体死代码：调度逻辑被 PenaltyFlex 内部重新实现，默认值双份硬编码（`schedule.py:50-55` vs `physical_params.py:254-260`），漂移风险。
- `DynamicNormalizer` 死代码 + docstring 虚假宣传（声称用于 ParetoZoom，`pareto_zoom.py:136` 实际全程物理先验）；B-10 动态归一化切换 + λ 重置逻辑未接线。
- `MixingEnthalpy(gamma=…)` 实例参数被静默忽略（`mixing_enthalpy.py:23-24`），恒用全局 0.25——γ 敏感性实验若走此路径将失效（当前实验走 builder 路径，正确）。
- `constraints/` 三个模块的 `get_qubo_terms()` 主流程无调用方（验证脚本自己承认是 REFERENCE 实现）——**这正是 P0-1/P0-2 双实现不一致的结构性根源，建议二选一收敛**。
- `warm_start_from()` 死代码、`qubo_linear_coefficient` 占位死方法、`_oscillation_count` 只累计不消费。
- C-Cr H_max=64.3125 硬编码字面量（方案审查意见 12 要求从编码参数实时导出，`physical_params.py:251`）。
- 可视化与实验层无连线：图 7（消融堆叠柱状）、图 8（NSGA-II 叠加）、图 5 C-metric 热力小图、图 6、图 S1/S2 均无函数；λ 轨迹图有函数但无 lambda_history 数据生产者。
- 无端到端入口：`scripts/`、`notebooks/`、`data/` 全空，H-10 保底执行顺序（0→2→3→1→5→4）仅存在于 docstring。
- 等原子比基准（ΔH≈−12.32、ρ≈7.11、f₃=35.6）无任何测试断言锚定。

---

## 六、做得好的部分（值得肯定）

1. **编码层质量高**：PrecisionSplit 比特方向全链路统一（LSB 低位索引）、边界检查完整、0.25/5.0 二进制精确无浮点陷阱；变量映射与方案 §2.4 逐字一致。
2. **物理参数单一数据源**：`physical_params.py` frozen dataclass 集中管理，ΔH/密度/成本/半径/VEC/电负性/熔点表与方案附录全部核对一致。
3. **P0P1 修正条款的核心项已落实**：λ_sum 固定剥离、纯二次型碳化物、ω_CCr=0.5、VEC 移出 QUBO、γ=0.1 收紧——方案的"修正灵魂"已进入代码。
4. **目标函数公式经实跑验证正确**：等原子比 ΔH_mix=−12.32 复现；γ 折扣单次施加无双重折扣；无 η_form 残留。
5. **warm-start 双层协同闭环成立**：LambdaCache 最近邻查询 → PenaltyFlex 注入 → 收敛回写（`pareto_zoom.py:389-437`）。
6. **消融实验开关式拆卸**：三布尔开关组合 5 配置，符合方案"模块开关不需从头实现"的时间优化设计。
7. **NSGA-II 公平性的一半做到了**：复用同一套 f₁f₂f₃ 目标类与相同编码边界。

---

## 七、修复路线建议（按方案保底执行顺序对齐）

**第 1 批（公式正确性，1~2 天）——不修则一切实验结论无效**
1. P0-1：builder P0 按 C-02 做 /5625 归一化（或收敛双路径为单一实现）
2. P0-2：删除 `ccr_coupling.py:74-80` 的重复线性项
3. P0-4 + P1-1/2：PenaltyFlex 补加性分支、修正振荡几何平均操作数、收敛加 t>T_add 门控
4. P0-10：top10_overlap 改为真实解集重叠率

**第 2 批（链路贯通，2~3 天）——不修则真机与对比实验跑不了**
5. P0-3：kaiwu TOP-K 采样链路（num_reads 贯通），同步补模拟器门禁 + fallback（P0-7）
6. P0-5：跨方法统一 HV 参考点（合并全部解集后统一定 ref）
7. P0-6：deap 缺失时显式报错 + 结果标注；deap 移入主依赖
8. P0-8/P0-9：入档 2% 可行性过滤 + CONVEXITY_TEST_3D

**第 3 批（验证层补齐）——对照 P0P1 修正文档与 Validation Scheme**
9. G-09 公式修正（T_m 含 C、五主元归一化）+ all_pass 收紧；G-10/G-11 补全
10. 统计层补 Wilcoxon/Friedman/κ/F检验/中位数IQR + Bonferroni α' 落地
11. MET 指标计算层（HV_ratio/FR_diff/SR/AR/CAR/NDR/LAR）
12. 实验 5（Pycalphad PSR≥70% + 负对照 NDR=100%）接口与实现

**第 4 批（工程收尾）**
13. README、端到端入口（scripts/）、测试断言锚定等原子比基准、死代码清理（schedule.py/constraints 双路径二选一）

---

## 八、一句话总结

**方案设计的"形"已经立起来（60%），但两处公式错误和一条断链让"神"还跑不通；验证与实验这两层决定答辩弹药的部分完成度不足一半。建议先修第 1、2 批共 8 项，再上模拟器跑通全 pipeline，之后按保底顺序 0→2→3→1→5→4 推进实验。**

---

## 九、修复进度更新（2026-07-18 13:30）

**第 1 批（公式正确性）✅ 已完成**：P0-1 builder /5625 归一化、P0-2 C-Cr 线性项去重、P0-4 PenaltyFlex 加性分支+振荡操作数+收敛门控+TOP-10 真排序、P0-10 真实 TOP10 重叠率。各自带数值验证脚本，全部通过。

**第 2 批（链路贯通 + 公平性）✅ 已完成**：kaiwu_solver 整体重写（TOP-K 贯通、模拟器/真机门禁、硬件自检 D-06、真实 is_feasible）、CIM 截断 B-04/B-05 修复（P2 存活 21/21、f1 误清 0 项）、统一 HV 参考点（`set_unified_reference`）、deap 显式化、实验参数对齐方案、README + pyproject 修复、ParetoZoom 四项（2% 入档过滤、d_threshold 接线、CONVEXITY_TEST_3D、HV 收敛绝对值+参考点漂移）。

**端到端冒烟测试 ✅ 通过**（`smoke_simulator_e2e.py`，Python3.10 + kaiwu 模拟器）：41.5s 跑通 构建→求解TOP-K→解码→PenaltyFlex→过滤入档→非支配前沿→HV→凸性检验 全链路；67 条入档记录、HV=835.6、凸性告警正常触发、前沿成分物理合理。

**环境结论**：Python 3.10（`C:\Users\At3ase\...\Python310\python.exe`）为项目运行环境，kaiwu 1.3.1 + scipy + deap 已就绪；注意 kaiwu-community 覆盖问题导致 CIM 真机模块暂不可导入，上真机前需重装完整版 SDK。

**距离"模拟器初步实验结果"的剩余缺口**：
1. NSGA-II 可行性约束缺陷（前沿混入 Σc≠100 解，最低 60.3——基线 HV 可信度，实验 3 前置）
2. 消融实验统一参考点接入 + Abl-2 Grid-search（实验 0 前置）
3. `comparison.py` Feasible Rate 硬编码 1.0 需改为真实统计
4. 实验驱动脚本（scripts/ 为空，无端到端跑批+落盘+出图入口）
5. 过时测试更新：`tests/verify_kaiwu_integration.py` Check 5 断言的是旧的错误实现
6. （答辩级才需要）统计层 Wilcoxon/Friedman/κ、MET 指标、可视化连线、G-09~G-11

**整体完成度估计：~75%**（公式层可信、模拟器链路已通、实验编排骨架可用；物理验证层与统计层仍是主要缺口）。

---

## 十、第 3 批进度更新（2026-07-18 15:55）——里程碑：模拟器初步实验结果 ✅ 达成

**第 3 批（实验前置 + 驱动脚本）✅ 全部完成**：
1. NSGA-II 可行性缺陷修复（修复算子保和投影，前沿 Σc 偏差 <1e-9，BASE-9 公平性恢复）
2. 消融实验统一参考点 + Abl-2 真 Grid-search（BASE-4 网格 {0.1…100}，选中 λ 可追溯）+ AR 符号口径修正
3. Feasible Rate 真实统计（不再恒 1.0；实测 PenaltyFlex 0.375 > Fixed(λ=1) 0.150，趋势合理）
4. **A-1 致命聚合缺陷修复**（add_metric 覆盖→追加，20/30 次重复不再退化为 1 个样本）
5. 实验驱动脚本 `scripts/run_experiments.py`（CLI：--experiment {0,2,3,all} --reps --num-reads --quick；跑实验→落盘→7 张图）
6. `verify_kaiwu_integration.py` 重写为运行时探针，**8/8 全过**
7. 追加修复两个新发现阻断点：`kaiwu_solver._parse_var_index` 布局硬编码（unified_48/38 此前产出垃圾解）、驱动脚本 quick 模式 NSGA2 补丁 bug

**验收实测**：`--experiment 0 --quick --reps 1` 端到端 122.9s 跑通，5 配置全部产出、0 错误，结果落盘 `data/results/exp0_acceptance/`（results.json + records.csv + report.md + 7 图）。注意：quick 模式（3 权重/120 reads/1 rep）数字仅供链路验证，量级与方向不可作物理解读。

**距离"答辩级结果"的剩余缺口**：
- 正式跑批：12 权重 + num_reads=1000 + reps=20（实验 0 约 8~16 小时，可并行）
- B-1 求解器 seed 未接线（结果不可复现）
- 统计层：配对 Wilcoxon（实验 0）、Friedman（实验 4）、Bonferroni 接线
- MET 指标计算层（HV_ratio/FR_diff/SR/C_metric 实验调用）
- 图 7（消融堆叠柱状）、图 8（NSGA-II 叠加）专用绘图函数
- 物理验证层 G-09~G-11 + Pycalphad（实验 5）
