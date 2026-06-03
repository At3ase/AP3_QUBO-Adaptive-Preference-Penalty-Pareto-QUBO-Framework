# 高熵合金成分优化 · 编码方案 v1.6
## AlCoCrFeNi-C 体系 | 玻色量子 kaiwu SDK

---

## 一、体系特征分析

| 元素 | 角色 | 典型含量范围 | 精度要求 |
|------|------|-------------|---------|
| Al | 主元 | 15~25 at% | 中等 |
| Co | 主元 | 15~25 at% | 中等 |
| Cr | 主元 | 15~25 at% | 中等 |
| Fe | 主元 | 15~25 at% | 中等 |
| Ni | 主元 | 15~25 at% | 中等 |
| C  | 间隙元素 | 0~1.5 at% | **高** |

**关键洞察**：C与其他主元含量差1~2个数量级。统一编码精度会导致主元比特数爆炸或C精度不足。采用**分层精度二进制编码（PrecisionSplit）**解决。

---

## 二、分层精度二进制编码（PrecisionSplit）

### 2.1 主元编码（Al, Co, Cr, Fe, Ni）

每种主元分配 **7个二元变量**：

```
b_{i,0}, b_{i,1}, ..., b_{i,6}  ∈ {0,1}
```

**解码公式**：
```
c_i = base_main + step_main × Σ_{j=0}^{6} 2^j × b_{i,j}
```

**参数**：
- `base_main = 5.0%`（最低成分，防止元素缺失导致相不稳定）
- `step_main = 0.25%`（步长）
- **取值范围**：`c_i ∈ [5.0%, 36.75%]`，共128个离散等级

> 等原子比（20%）落在中间区域，编码对称性好。

### 2.2 间隙元素C编码

C分配 **3个二元变量**：

```
b_{C,0}, b_{C,1}, b_{C,2} ∈ {0,1}
```

**解码公式**：
```
c_C = base_C + step_C × Σ_{j=0}^{2} 2^j × b_{C,j}
```

**参数**：
- `base_C = 0.0%`
- `step_C = 0.25%`（与主元统一步长）
- **取值范围**：`c_C ∈ [0.0%, 1.75%]`，共8个离散等级

> 1.75%上限覆盖C的固溶极限及轻度碳化物形成区。如需更高可调为4比特。

### 2.3 变量汇总

| 元素组 | 元素数 | 每元素比特 | 小计 |
|--------|--------|-----------|------|
| 主元 | 5 | 7 | **35** |
| C | 1 | 3 | **3** |
| **总计** | **6** | — | **38 个二元变量** |

> **分层编码保留**：主元范围大→7比特，C范围小→3比特。步长统一为0.25%确保数学上可精确满足成分和=100%。

### 2.4 变量映射

```
x_0  ~ x_6   → Al: b_0 ~ b_6
x_7  ~ x_13  → Co: b_0 ~ b_6
x_14 ~ x_20  → Cr: b_0 ~ b_6
x_21 ~ x_27  → Fe: b_0 ~ b_6
x_28 ~ x_34  → Ni: b_0 ~ b_6
x_35 ~ x_37  → C:  b_0 ~ b_2
```

### 2.5 成分解码的线性表达式

```
c_Al = 5.0 + 0.25×(x_0 + 2x_1 + 4x_2 + 8x_3 + 16x_4 + 32x_5 + 64x_6)
c_Co = 5.0 + 0.25×(x_7 + 2x_8 + 4x_9 + 8x_10 + 16x_11 + 32x_12 + 64x_13)
c_Cr = 5.0 + 0.25×(x_14 + 2x_15 + 4x_16 + 8x_17 + 16x_18 + 32x_19 + 64x_20)
c_Fe = 5.0 + 0.25×(x_21 + 2x_22 + 4x_23 + 8x_24 + 16x_25 + 32x_26 + 64x_27)
c_Ni = 5.0 + 0.25×(x_28 + 2x_29 + 4x_30 + 8x_31 + 16x_32 + 32x_33 + 64x_34)
c_C  = 0.0 + 0.25×(x_35 + 2x_36 + 4x_37)
```

**统一步长验证**：
所有成分均为 0.25% 的整数倍。设各元素编码值为 k_i（主元）和 k_C（C）：
```
总成分 = 5×5.0% + 0.25% × (Σk_i + k_C) = 25% + 0.25% × K
```
要精确等于100%，需 K = 300。主元 k_i ∈ [0,127]，C 的 k_C ∈ [0,7]，最大可达 5×127+7=642 > 300，最小为0，因此 **K=300 完全可达**（例如等原子比成分：各主元 k_i=60、C 的 k_C=0，即 60×5+0=300，对应各主元 20%、C 为 0%，均为 0.25% 的整数倍）。编码空间中存在大量精确满足成分和=100%的整数组合，QUBO硬约束在数学上**可精确满足**。

---

## 三、约束处理与惩罚项设计

### 3.1 硬约束：成分和 = 100%（含C）

六种元素（Al, Co, Cr, Fe, Ni, C）的原子百分比之和严格等于100%。C虽为间隙元素且含量低，但在约束方程中占据完整权重。

**惩罚项形式**：
```
H_sum = λ_sum × ( Σ_{k∈{Al,Co,Cr,Fe,Ni,C}} c_k − 100 )²
```

展开为QUBO：

定义 `S = Σ c_k = Σ(base_k) + Σ(step_k × Σ 2^j x_{k,j})`

常数基线：`S_const = 5×5.0 + 0.0 = 25.0%`（主元最低值之和）

变量部分：`S_var = S − S_const`

**量级归一化**：为避免惩罚项系数与ΔH_mix（kJ/mol量级）相差过大，对S_var做无量纲化处理：
```
S_var' = S_var / 75  // 归一化，使 S_var' ∈ [−1/3, ~8.56]

H_penalty = λ_sum × (S_var' − 1)²
          = λ_sum × [(S_var/75) − 1]²
          = λ_sum × [S_var²/5625 − 2·S_var/75 + 1]
          = (λ_sum/5625) × [S_var² − 150·S_var + 5625]
```

归一化后惩罚项系数与目标函数（ΔH_mix ~ −5~−15 kJ/mol）处于同一数量级，避免量子退火器动态范围溢出。

其中 `S_var²` 展开后：
- 自平方项：`x_i² = x_i`（二元变量）→ 线性项，系数 = (λ_sum/5625) × coef_i
- 交叉项：`2×coef_i×coef_j×x_i×x_j` → QUBO二次项，系数 = (λ_sum/5625) × 2×coef_i×coef_j

### 3.2 Feedback-Driven Penalty Adaptation (PenaltyFlex)

**核心思想**：惩罚项系数根据**当前解的质量反馈**和**用户对不同约束的偏好权重**动态调整。经典-量子混合迭代：量子退火器负责单次QUBO采样，经典后处理层分析结果并调整下一轮参数，再通过kaiwu SDK重新提交。

#### 3.2.1 分层约束优先级

| 约束层级 | 约束内容 | 偏好权重 ω | 物理意义 |
|---------|---------|-----------|---------|
| **P0（刚性）** | 成分和 = 100% | ω_sum = 1.0（固定最高）| 质量守恒，不可违反 |
| **P1（强偏好）** | 碳化物抑制（C含量上限） | ω_carbide = 0.1~0.3 | 防止间隙C过量析出碳化物 |
| **P2（弱偏好）** | 成本/稀缺性约束 | ω_cost = 0.01~0.1 | 倾向低成本元素组合 |

> **VEC约束移除说明**：原方案将VEC（价电子浓度）作为P1软约束嵌入QUBO，但VEC与f₁的ΔH_mix在物理上存在耦合（如VEC降低常伴随d电子数减少，影响混合焓），导致QUBO内部隐性对抗。现依据评审建议，**VEC仅作为后处理相结构探针**（见第7节物理过滤器），不再参与QUBO惩罚项，避免与f₁目标函数冲突。

总惩罚项：
```
H_penalty = ω_sum·λ_sum·H_sum + ω_carbide·λ_carbide·H_carbide + ω_cost·λ_cost·H_cost
```

#### 3.2.2 P1约束：碳化物抑制项

**核心物理**：AlCoCrFeNi-C体系中，C > 1.0 at% 时易析出M₂₃C₆或M₇C₃碳化物（Gorr et al., 2015），破坏固溶体结构。将碳化物抑制作为P1软约束，引导优化器向低C固溶区探索。

```
H_carbide = max(0, c_C − 0.8%)²  // 软上限0.8 at%（保守固溶区）
```

权重：`ω_carbide = 0.1~0.3`，可调。

> **与后处理VEC探针的分工**：
> - **P1 H_carbide**：参与QUBO优化，主动抑制高C成分
> - **后处理VEC检查**（第7节）：优化完成后独立验证相结构，不参与QUBO，避免与f₁冲突

#### 3.2.3 VEC后处理探针（已移出QUBO约束）

**VEC（价电子浓度）—— 仅针对置换式主元**：
```
VEC = Σ_{i∈{Al,Co,Cr,Fe,Ni}} c_i' · VEC_i
VEC_i: Al=3, Co=9, Cr=6, Fe=8, Ni=10

c_i' = c_i / (1 − c_C)  // 主元归一化（扣除C的间隙占比）
```

> **VEC物理说明**：C为间隙原子，不占据晶格置换位，依据Guo et al. (2011)与Yang et al. (2020)的原始定义，VEC仅统计置换式主元。主元成分需归一化以保证VEC反映真实晶格电子结构。

VEC不再嵌入QUBO惩罚项，仅在第7节后处理中作为**相结构风险提示**使用（目标区间7.0~7.6，中心点7.2对应5主元等原子比）。

#### 3.2.3 自适应更新规则

每轮量子采样后，经典层分析TOP-K解的（目标值，约束违反度），按以下规则调整λ：

| 反馈模式 | 判定条件 | 自适应动作 | 目的 |
|---------|---------|-----------|------|
| **探索奖励** | 最优解约束违反 > 5% 但目标值极优 | 维持或降低该约束的λ | 给优秀方向更多空间 |
| **收紧信号** | 所有TOP解约束满足但目标平庸 | 增大λ_push | 迫使跳出舒适区 |
| **边界聚焦** | Pareto前沿恰在约束边界上 | 微调λ使前沿分辨率提高 | 获取更密的Pareto点 |
| **振荡抑制** | λ连续两轮反向调整 | 改用几何平均而非算术更新 | 防止 hunting |

**更新公式**：
```
λ_sum^(t+1) = λ_sum^(t) × exp( α · tanh( violation_avg − target_violation ) )
```
- `violation_avg`：本轮TOP-10解的平均成分和偏离度
- `target_violation`：期望偏离度（探索期2%，固化期0%）
- `α`：学习率（0.5~1.0）

**完整PenaltyFlex算法（8步）**：
```
算法：PenaltyFlex (Feedback-Driven Penalty Adaptation)
────────────────────────────────────
输入：初始λ⁽⁰⁾ = (0.1, 0.0, 0.0)  // (sum, carbide, cost)
      α=0.8, γ=0.5, T_max=15
      偏好权重 ω = (1.0, ω_carbide, ω_cost)
输出：优化解 x*，自适应λ*，收敛历史

1. FOR t = 1, ..., T_max:
2.     组装QUBO：H = H_obj + ω·λ⁽ᵗ⁾·H_penalty
3.     量子求解：solutions = kaiwu.solve(H)
4.     TOP-10解解码：{(c_k, v_k)}，v_k = 约束违反度
5.     目标值评估：f_k = (f₁(c_k), f₂(c_k), f₃(c_k))
6.     
7.     FOR 每个约束j ∈ {sum, carbide, cost}:
8.         v̄_j = MEAN(v_j for k=1..10)
9.         IF t > 1 AND sign(λ_j⁽ᵗ⁾ − λ_j⁽ᵗ⁻¹⁾) ≠ sign(λ_j⁽ᵗ⁻¹⁾ − λ_j⁽ᵗ⁻²⁾):
10.            λ_j⁽ᵗ⁺¹⁾ = √(λ_j⁽ᵗ⁾ · λ_j⁽ᵗ⁻¹⁾)  // 几何平均抑制振荡
11.        ELSE:
12.            IF j == "sum":  // P0硬保护：只增不减
                λ_sum^(t+1) = MAX( λ_sum^(t), λ_sum^(t) × exp(α · tanh(v̄_sum)) )
            ELSE:  // carbide, cost 正常自适应
                λ_j^(t+1) = λ_j^(t) × exp( α · tanh(v̄_j − ε_j) )
13.   
14.    IF MAX_j |λ_j⁽ᵗ⁺¹⁾ − λ_j⁽ᵗ⁾| / λ_j⁽ᵗ⁾ < γ:
15.        BREAK  // 收敛
16. RETURN best_solution, λ_final
```

**warm-start复用**：相邻权重组合的Pareto解通常接近，复用上一轮最优解作为初态。自适应λ序列只需在第一组权重完整跑一遍，后续权重继承最终λ值作为起点。

---

## 四、三目标Pareto优化框架

### 4.1 目标函数设计

| 目标 | 符号 | 优化方向 | 物理意义 | QUBO可实现性 |
|------|------|---------|---------|-------------|
| **f₁** | 形成能 | 最小化 | 合金热力学稳定性 | **✅ 天然二次型**（ΔH_mix） |
| **f₂** | 密度 | 最小化 | 合金宏观物理性质 | **✅ 线性→可构造二次型**（Vegard） |
| **f₃** | 成本 | 最小化 | 元素稀缺性与市场价格 | **✅ 纯线性** |

#### f₁：形成能（ΔH_mix代理）

```
ΔH_mix = 4 Σ_{i<j} ΔH_{ij}^{AB} · c_i · c_j
H_form = η_form × ΔH_mix
```
- 天然为QUBO二次型，无需近似或辅助变量
- 优化方向：最小化（使ΔH_mix更负，合金更稳定）
- η_form：缩放系数，使形成能量级与惩罚项匹配

> **与相稳定性的关系**：f₁直接负责热力学稳定性（ΔH_mix最小化），因此P1约束中不再重复包含ΔH_mix软引导。VEC保留为独立相结构判据。后处理层（6.3节）对极端ΔH_mix值做物理合理性检查。

#### f₂：密度（Vegard定律）

**代理模型**：Vegard定律线性近似。HEA中预测误差约0.5~1%，对成分优化足够精确。

```
ρ = Σ_{i∈{Al,Co,Cr,Fe,Ni,C}} c_i · ρ_i
```

| 元素 | 纯元素密度 ρ_i (g/cm³) |
|------|------------------------|
| Al | 2.70 |
| Co | 8.86 |
| Cr | 7.19 |
| Fe | 7.87 |
| Ni | 8.91 |
| C  | 2.27 |

等原子比AlCoCrFeNi预测密度 ≈ **7.11 g/cm³**。

**QUBO嵌入**：
```
H_density = ρ = Σ c_i · ρ_i   // 纯线性，最小化方向
```
- 不预设固定ρ_target，让Pareto前沿自然涌现密度梯度
- Al↑→密度↓，但Al↑同时导致BCC↑、韧性↓——三目标自然竞争

#### f₃：成本指数

```
f_cost = Σ w_k · c_k
```
- Co权重较高（战略资源）
- Ni权重中等
- Al、Fe、C权重较低
- 纯线性项，直接映射为QUBO的h_i偏置场

### 4.2 Pareto前沿生成：Hypervolume-Guided Pareto Exploration (ParetoZoom) + 粗网格初始化

**用户确认方案2**：先以6组粗网格建立初始前沿轮廓，再由ParetoZoom动态填补间隙与热点加密。

#### 4.2.1 理论背景

固定权重网格局限：
- **非凸前沿遗漏**：weighted-sum只能恢复凸包，凹陷区域需特定权重采样
- **均匀采样浪费**：平坦区过度采样，热点区不足
- **无反馈终止**：必须跑完全部预设权重

ParetoZoom优势：
- kaiwu SDK单次求解毫秒级，动态增删权重组合在wall-clock上完全可行
- 超体积（HV）作为统一收敛指标，同时惩罚收敛不足和分布不均

#### 4.2.2 粗网格初始化（6组）

启动权重集合 `G_init` 覆盖权重单纯形的顶点与边中点：

| 编号 | w₁ (形成能) | w₂ (密度) | w₃ (成本) | 物理意义 |
|------|------------|----------|----------|---------|
| G1 | 1.0 | 0.0 | 0.0 | 纯形成能优先（最稳定） |
| G2 | 0.0 | 1.0 | 0.0 | 纯密度优先（最轻） |
| G3 | 0.0 | 0.0 | 1.0 | 纯成本优先（最便宜） |
| G4 | 0.5 | 0.5 | 0.0 | 形成能-密度权衡 |
| G5 | 0.5 | 0.0 | 0.5 | 形成能-成本权衡 |
| G6 | 0.0 | 0.5 | 0.5 | 密度-成本权衡 |

对每组权重：
1. 用PenaltyFlex跑完整自适应λ序列，获取最优可行解
2. 解码并计算 (f₁, f₂, f₃)
3. 送入Pareto存档P，执行非支配筛选

首轮保障：`|P| ≥ 1`，HV(P) > 0。

#### 4.2.3 ParetoZoom算法

```
算法：Hypervolume-Guided Pareto Exploration (ParetoZoom)
────────────────────────────────────
输入：粗网格存档 P₀
      参考点 r = (max f₁ + δ₁, max f₂ + δ₂, max f₃ + δ₃)，δ_i = 10% × (max f_i − min f_i)
      ε_HV = 1%, d_threshold = 0.15·max_edge_length(P)
      σ = 0.08, T_max = 5轮, num_reads = 1000
输出：近似Pareto前沿 P*

1. P ← P₀, G ← G_init
2. FOR t = 1, ..., T_max:
3.    G_new ← ∅
4.    
5.    // 阶段A：间隙检测与加密
6.    SORT P by f₁
7.    FOR 存档P中每对相邻非支配解 (y_i, y_{i+1}):
8.        d_obj ← ||y_i − y_{i+1}||₂
9.        IF d_obj > d_threshold:
10.           w_mid ← interpolate_weights(y_i, y_{i+1})
11.           G_new ← G_new ∪ {w_mid}
12.   
13.   // 阶段B：HV热点导向微扰
14.   FOR 每个已探索权重 w ∈ G:
15.       P_wo ← P \ {w产生的解}
16.       contribution ← HV(P, r) − HV(P_wo, r)
17.       IF contribution > 0.2 × mean_contributions:
18.           FOR k = 1, 2:
19.               w_perturb ← w + N(0, σ·I)
20.               PROJECT w_perturb onto simplex  // Σwᵢ=1, wᵢ≥0
21.               G_new ← G_new ∪ {w_perturb}
22.   
23.   // 阶段C：去重与边界保护
24.   G_new ← DEDUPLICATE(G_new, tol=0.05)
25.   REMOVE w from G_new if ANY(wᵢ < 0.02)
26.   IF G_new = ∅: BREAK
27.   
28.   // 阶段D：量子求解与存档更新
29.   FOR w ∈ G_new:
30.       H_single ← w₁·f₁ + w₂·f₂ + w₃·f₃ + H_penalty(λ_PenaltyFlex_final)
31.       // λ_PenaltyFlex_final：复用最近邻权重已跑过的λ终值（warm-start）
32.       solutions ← kaiwu.solve(H_single, num_reads=1000)
33.       FOR sol ∈ TOP-20(solutions):
34.           c ← decode(sol)
35.           IF constraints_satisfied(c, tol=2%):
36.               y ← (f₁(c), f₂(c), f₃(c))
37.               P ← nondominated_update(P, y)
38.   
39.   // 阶段E：HV收敛判定
40.   HV⁽ᵗ⁾ ← HV(P, r)
41.   IF |HV⁽ᵗ⁾ − HV⁽ᵗ⁻¹⁾| / HV⁽ᵗ⁾ < ε_HV: BREAK
42. 
43. RETURN P
```

#### 4.2.4 ParetoZoom关键设计说明

| 组件 | 功能 | 物理意义 |
|------|------|---------|
| **间隙检测** | 发现目标空间"空洞" | 两个解距离太远 = 中间可能有更好的权衡解 |
| **权重插值** | 单纯形上生成中点权重 | 保持w₁+w₂+w₃=1 |
| **HV贡献度** | 识别"高价值权重方向" | 某些权重恰好产出扩展HV的关键解 |
| **高斯微扰** | 在热点附近加密 | 微调探索半径，不改变方向本质 |
| **边界保护** | 删除wᵢ<0.02的极端权重 | 避免单目标退化数值不稳定 |
| **HV参考点** | r = (max f_i + δ_i)，δ_i = 10%目标范围 | 确保参考点严格支配所有解，HV > 0 |

#### 4.2.5 双层自适应架构

| 层次 | 算法 | 控制对象 | 作用域 |
|------|------|---------|--------|
| **外层ParetoZoom** | 4.2.3节 | 权重 (w₁,w₂,w₃) | **Pareto前沿质量**（覆盖度、密度） |
| **内层PenaltyFlex** | 3.2节 | 惩罚系数 (λ_sum,λ_carbide,λ_cost) | **约束满足率**（可行性、精确度） |

协同流程：
1. ParetoZoom提出新权重 `w_new`
2. 查找最近邻 `w_neighbor`（已跑过PenaltyFlex）
3. 以 `w_neighbor` 的PenaltyFlex终值 `λ*` 为 `w_new` 的初值（warm-start）
4. PenaltyFlex跑自适应λ序列，微调至 `λ_new*`
5. `λ_new*` 存入缓存，供后续相邻权重复用

#### 4.2.6 理论预期

| 特性 | 预期 |
|------|------|
| 粗网格调用 | 6组QUBO，每组PenaltyFlex≈10-15轮 → 总计60-90次量子提交 |
| ParetoZoom迭代 | 每轮新增3-8组权重，2-4轮收敛 → 总计20-40次量子提交 |
| 总wall-clock | <10秒（模拟器）/ <1分钟（真机含通信开销） |
| 非凸前沿恢复 | 依赖粗网格是否覆盖凹陷区方向；6组顶点+边中点提供基础方向多样性 |

---

## 五、硬件兼容性验证

| 指标 | 本方案需求 | 玻色量子550W能力 | 余量 |
|------|-----------|----------------|------|
| 变量数 | 38 | 550 | **14.5×** |
| 耦合链接 | ≤ 703 | 150,975 | **214×** |
| 单次求解复杂度 | 轻量 | 毫秒级 | — |

**结论**：38变量在550比特真机上仍属于"sandbox级别"，资源极其充裕。

**可选升级**：
- 主元→8比特（步长0.125%）：总变量 = 43
- C→4比特（步长0.25%）：总变量 = 39
- 仍远低于550上限，可随时升级。

---

## 六、与kaiwu SDK的对接

| SDK功能 | 本方案使用方式 |
|---------|---------------|
| QUBO模型构建 | Python组装38×38 Q矩阵 |
| 伊辛模型转化 | 自动转换（如SDK支持） |
| 模拟器验证 | **必须先用模拟器跑通全pipeline（粗网格+ParetoZoom）**，再上真机 |
| 真机采样 | 每轮QUBO提交，返回TOP-K解 |
| 多轮迭代 | Python循环控制：PenaltyFlex（λ自适应）+ ParetoZoom（w自适应）双层嵌套 |
| warm-start缓存 | Python dict存储 {weight_tuple: λ_final}，供相邻权重复用 |

---

## 七、后处理物理过滤器（输出层）

Pareto解输出后，经典层执行物理合理性检查，不消耗量子资源：

```
过滤器：物理合理性检查
────────────────────────
输入：Pareto解 (c₁, c₂, ..., c₆) 及其 (f₁, f₂, f₃)
输出：标记后的推荐列表

IF ΔH_mix < −20 或 ΔH_mix > 5:
    标记 ⚠️ "偏离HEA稳定区间"  // Miedema判据：-15~5 kJ/mol为合理

IF VEC < 7.0 或 VEC > 7.6:
    标记 ⚠️ "相结构风险"  // 偏离FCC+BCC双相区

IF c_C > 1.0%:
    标记 ⚠️ "碳化物形成风险"  // 间隙C过高可能析出M₂C/M₇C₃

IF ANY(c_i > 35%):
    标记 ⚠️ "偏离HEA定义"  // 传统HEA定义：各元素5~35%

IF |Σc_k − 100%| > 1%:
    标记 ⚠️ "成分和非100%"  // 硬约束应已消除，保留兜底

IF 标记数 = 0:
    标记 ✅ "可直接实验"
```

**作用**：不修改QUBO结构，不消耗量子资源，仅作为输出层清洗。用户看到结果时，高风险解自动降权或折叠。

---

## 八、参数确认清单（v1.5 FINAL）

| 参数 | 取值 | 状态 |
|------|------|------|
| base_main | 5.0% | **✅ 确认** |
| step_main | 0.25% | **✅ 确认** |
| base_C | 0.0% | **✅ 确认** |
| step_C | 0.25% | **✅ 确认** |
| 总变量数 | 38 | **✅ 确认** |
| PenaltyFlex策略 | 反馈驱动自适应（3.2节） | **✅ 确认** |
| f₁（形成能） | ΔH_mix最小化 | **✅ 确认** |
| f₂（密度） | 最小化方向（无固定target） | **✅ 确认** |
| f₃（成本） | 线性权重最小化 | **✅ 确认** |
| VEC探针 | 区间7.0~7.6，仅后处理（已移出QUBO） | **✅ 确认** |
| P1约束 | 碳化物抑制 c_C ≤ 0.8% | **✅ 确认** |
| Pareto生成策略 | 方案2：粗网格+ParetoZoom | **✅ 确认** |
| 粗网格组数 | 6组（顶点+边中点） | **✅ 确认** |
| ParetoZoom动态步长 | 动态自适应（无固定） | **✅ 确认** |
| 面向领域 | 航天高温结构件/热障涂层 | **✅ 确认** |

---

## 九、参考文献

[1] Halfmann P, Trebing M. Penalty Factor Optimization for Quantum Annealing - A Multiobjective Approach. EURO 2025.
[2] Quantum Integer Programming (QuIP) 47-779: Lecture Notes. Carnegie Mellon University. arXiv:2012.11382.
[3] Kurebayashi Y, Yamashita Y, Tobe Y. Optimization of Penalty Weights in Quantum Annealing for Dynamic Spectrum Allocation. IEICE, 2025.
[4] Miedema A R, de Chatel P F, de Boer F R. Cohesion in alloys – fundamentals of a semi-empirical model. Physica B+C, 1980.
[5] Guo S, Ng C, Lu J, Liu CT. Effect of valence electron concentration on stability of fcc or bcc phase in high entropy alloys. J. Appl. Phys. 109:103505, 2011.
[6] Yang S, Lu J, Xing F, Zhang L, Du Y. Revisit the VEC rule in high entropy alloys with high-throughput CALPHAD approach. Acta Mater. 192:11-19, 2020.
[7] Lu Y, Wang Z, Li S, et al. Learning to Optimize Multi-Objective Alignment Through Dynamic Reward Weighting. arXiv:2509.11452, 2025.
[8] Quantum Multi-Objective Optimization. Emergent Mind, 2026. (综述：QA/QAOA多目标优化硬件实现与benchmark)
[9] Fonseca C M, Paquete L, Lopez-Ibanez M. An improved dimension-sweep algorithm for the hypervolume indicator. IEEE Congress on Evolutionary Computation, 2006.
[10] Senkov O N, Scott J M, Senkova S V, et al. Microstructure and room temperature properties of a high-entropy TaNbHfZrTi alloy. J. Alloys Compd. 509:6043-6048, 2011. (LWHEA综述)
[11] Gorr B, Azim M, Christ H J, et al. Phase equilibria, microstructure, and creep behavior of Al-Co-Cr-Fe-Ni high entropy alloys. Acta Mater. 85:234-251, 2015. (高温抗氧化)
[12] Li R, Niu P, Yuan Y, et al. Selective laser melting of an equiatomic AlCoCrFeNi high-entropy alloy: Processability, non-equilibrium microstructure and mechanical property. Mater. Des. 118:95-108, 2017. (SLM增材制造)
