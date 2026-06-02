# AP³-QUBO 分步开发计划与 Claude Code Prompts

> **版本**: v1.0  
> **基于文档**: 《AP³-QUBO: Adaptive Preference-Penalty Pareto QUBO Framework 技术实现规格说明书》  
> **总步数**: 15步（9个Phase）  
> **开发原则**: 先基线后创新、先经典后量子、先单元后集成

---

## 整体架构概览

```
┌─────────────────────────────────────────────────────────────┐
│                    AP³-QUBO 框架架构                         │
├─────────────────────────────────────────────────────────────┤
│  Phase 1: 基础设施 (环境 + 项目骨架)                          │
│  Phase 2: QUBO基础建模层 (变量编码 + 目标函数 + 约束惩罚)        │
│  Phase 3: 求解器集成 (SA + CIM接口)                           │
│  Phase 4: APPS模块 (自适应偏好引导Pareto搜索)                   │
│  Phase 5: CVD-AP模块 (约束违反度自适应惩罚)                    │
│  Phase 6: APPC模块 (偏好-惩罚协同演化)                        │
│  Phase 7: 主算法集成 (AP³-QUBO主循环)                         │
│  Phase 8: 实验验证 (5个系统实验)                              │
│  Phase 9: 论文与竞赛材料 (论文 + PPT + Demo)                   │
└─────────────────────────────────────────────────────────────┘
```

---

## Phase 1: 基础设施与环境搭建

### Step 1: 项目骨架与依赖管理

**工作内容**:
1. 创建项目目录结构
2. 初始化Python虚拟环境
3. 创建 `requirements.txt`，包含：numpy, scipy, matplotlib, pytest, jupyter
4. 创建基础配置文件 `config.py`（元素参数、默认超参数）
5. 设置Git仓库并提交初始代码

**Claude Code Prompt**:

```text
请为我创建一个完整的Python项目骨架，项目名称为 "ap3-qubo"，用于实现面向高熵合金多目标设计的AP³-QUBO优化框架。具体要求：

1. 创建以下目录结构：
   ```
   ap3-qubo/
   ├── src/
   │   ├── __init__.py
   │   ├── config.py          # 全局配置参数
   │   ├── modeling/          # QUBO建模层
   │   │   ├── __init__.py
   │   │   ├── encoding.py    # 决策变量编码
   │   │   ├── objectives.py  # 三目标函数
   │   │   ├── constraints.py # 约束惩罚项
   │   │   └── qubo_matrix.py # QUBO矩阵构建
   │   ├── solvers/           # 求解器层
   │   │   ├── __init__.py
   │   │   ├── sa_solver.py   # 模拟退火求解器
   │   │   └── cim_solver.py  # CIM求解器接口
   │   ├── modules/           # 三大创新模块
   │   │   ├── __init__.py
   │   │   ├── apps.py        # APPS: 自适应偏好引导
   │   │   ├── cvd_ap.py      # CVD-AP: 自适应惩罚
   │   │   └── appc.py        # APPC: 协同演化
   │   ├── core/              # 核心算法
   │   │   ├── __init__.py
   │   │   ├── main_loop.py   # AP³-QUBO主循环
   │   │   └── pareto.py      # Pareto解集管理
   │   └── utils/             # 工具函数
   │       ├── __init__.py
   │       └── helpers.py
   ├── tests/                 # 单元测试
   ├── experiments/           # 实验脚本
   ├── notebooks/             # Jupyter Notebook
   ├── requirements.txt
   ├── setup.py
   └── README.md
   ```

2. 在 `src/config.py` 中定义以下常量（来自AlCoCrFeNi高熵合金系统）：
   - 元素列表: ['Al', 'Co', 'Cr', 'Fe', 'Ni']
   - 密度参数: [2.70, 8.86, 7.19, 7.87, 8.90] g/cm³
   - 成本参数: [2.2, 32.5, 12.0, 0.5, 16.5] USD/kg
   - VEC参数: [3.0, 9.0, 6.0, 8.0, 10.0]
   - Miedema混合焓矩阵 (5×5对称矩阵, 对角线为0)
   - 默认编码参数: 5元素 × 5bit = 25bit, Z = 31
   - 默认权重: [1/3, 1/3, 1/3]
   - 默认惩罚系数: [10.0, 100.0, 50.0]
   - 浓度范围约束: c_min=0.05, c_max=0.35
   - VEC约束范围: [6.87, 8.0]

3. `requirements.txt` 包含: numpy>=1.21, scipy>=1.7, matplotlib>=3.4, pytest>=7.0, jupyter>=1.0, pymoo>=0.6.0

4. 创建 `setup.py` 使得可以通过 `pip install -e .` 安装项目

5. 确保所有 `__init__.py` 文件正确导出子模块

请一次性创建所有文件，确保项目可以直接运行 `python -m pytest` 不报错。
```

---

### Step 2: 元素参数与物性数据验证

**工作内容**:
1. 实现元素物性参数的数据结构
2. 验证混合焓矩阵的对称性和完整性
3. 编写单元测试确保参数正确加载
4. 实现浓度编码/解码的基本工具函数

**Claude Code Prompt**:

```text
在已创建的项目骨架基础上，请实现高熵合金物性参数管理和基本编码工具。具体要求：

1. 在 `src/config.py` 中完善以下内容：
   - 定义 `ELEMENT_NAMES = ['Al', 'Co', 'Cr', 'Fe', 'Ni']`
   - 定义 `DENSITIES = np.array([2.70, 8.86, 7.19, 7.87, 8.90])`
   - 定义 `COSTS = np.array([2.2, 32.5, 12.0, 0.5, 16.5])`
   - 定义 `VECS = np.array([3.0, 9.0, 6.0, 8.0, 10.0])`
   - 定义完整的Miedema混合焓矩阵 `MIX_ENTHALPY` (5×5):
     Al-Co:-19, Al-Cr:-10, Al-Fe:-11, Al-Ni:-22, Co-Cr:-4, Co-Fe:-1, Co-Ni:0, Cr-Fe:-1, Cr-Ni:-7, Fe-Ni:-2
     对角线为0，矩阵对称
   - 定义 `BITS_PER_ELEMENT = 5`, `N_ELEMENTS = 5`, `N_BITS = 25`, `Z = 31.0`
   - 定义 `C_MIN = 0.05`, `C_MAX = 0.35`, `VEC_MIN = 6.87`, `VEC_MAX = 8.0`
   - 定义归一化范围: `R = np.array([15.0, 6.2, 32.0])` 对应形成能/密度/成本
   - 定义 `IDEAL_POINT_DEFAULT = np.array([-15.0, 2.7, 0.5])`

2. 在 `src/utils/helpers.py` 中实现以下工具函数：
   - `binary_to_concentration(x: np.ndarray) -> np.ndarray`: 将25维二进制向量解码为5维浓度向量
     公式: c_e = sum_b(x[e,b] * 2^b) / Z_actual，其中 Z_actual = sum_e sum_b(x[e,b] * 2^b) 为动态归一化因子
   - `concentration_to_objectives(c: np.ndarray) -> np.ndarray`: 从浓度计算三目标函数值
     f1 = sum_{i<j} c_i * c_j * dH_ij  (形成能)
     f2 = sum_e c_e * density_e         (密度)
     f3 = sum_e c_e * cost_e            (成本)
   - `normalize_objectives(obj: np.ndarray) -> np.ndarray`: 目标函数归一化到[0,1]
     tilde_f_m = (f_m - f_m^min) / R_m

3. 在 `tests/test_config.py` 中编写单元测试：
   - 测试混合焓矩阵对称: MIX_ENTHALPY[i,j] == MIX_ENTHALPY[j,i]
   - 测试对角线为0
   - 测试所有10个非对角独立元素值正确
   - 测试参数维度一致性(所有数组长度=5)

4. 在 `tests/test_helpers.py` 中编写单元测试：
   - 测试 binary_to_concentration: 输入全1向量(25个1)，输出应为 [31/155, 31/155, 31/155, 31/155, 31/155]（动态归一化: Z_actual=5*31=155）
   - 测试浓度之和为1（动态归一化天然保证）
   - 测试非全1向量: 如仅第一个元素全1，Z_actual=31，c=[1,0,0,0,0]，验证动态归一化正确性
   - 测试 concentration_to_objectives 返回形状为(3,)的数组

请确保所有测试通过，函数有完整的docstring和类型注解。
```

---

## Phase 2: QUBO基础建模层

### Step 3: 决策变量编码与浓度计算

**工作内容**:
1. 实现25位二进制变量到5元素浓度的完整编码/解码
2. 实现归一化浓度计算（含Z=31的归一化因子）
3. 验证编码的数值稳定性
4. 实现批量编码/解码（支持求解器返回的多个解）

**Claude Code Prompt**:

```text
请实现AP³-QUBO框架的决策变量编码模块。基于已完成的 `src/config.py` 和 `src/utils/helpers.py`，在 `src/modeling/encoding.py` 中实现以下功能：

核心数学要求：
- 25维二进制向量 x ∈ {0,1}^25，组织为5×5矩阵 x[e,b]，e∈{0..4}为元素索引，b∈{0..4}为比特位
- 元素原始计数: tilde_c_e = sum_{b=0}^{4} x[e,b] * 2^b
- 动态归一化因子: Z_actual = sum_e tilde_c_e（依赖于具体编码x）
- 归一化浓度（解码时用）: c_e = tilde_c_e / Z_actual，满足 sum_e c_e = 1
- QUBO近似常数: Z_qubo = 31（用于QUBO矩阵构建的固定近似值）
- 注意: QUBO中使用固定常数Z_qubo=31来近似二次型，但实际解码时采用动态归一化Z_actual。
  这意味着QUBO优化目标与真实目标之间存在近似误差，成分和约束（sum c_e = 1）
  需要通过Q_pen1惩罚项显式强制满足，而非天然满足

请实现以下函数（全部带完整docstring和类型注解）：

1. `encode_concentration(x: np.ndarray, Z_fixed: float = 31.0) -> np.ndarray`:
   - 输入: x 形状 (25,) 或 (N, 25)，元素为0或1
   - 输出: c 形状 (5,) 或 (N, 5)，归一化浓度（使用动态归一化 Z_actual = sum_e sum_b x[e,b]*2^b）
   - 处理标量和批量两种情况
   - 数值保护: 如果Z_actual=0，返回等浓度分布 [0.2, 0.2, 0.2, 0.2, 0.2]
   - Z_fixed参数仅保留接口兼容性，实际计算使用动态Z_actual

2. `decode_concentration(c: np.ndarray, bits: int = 5) -> np.ndarray`:
   - 输入: c 形状 (5,)，目标浓度（可能不完全精确对应离散级别）
   - 输出: x 形状 (25,)，最接近的离散浓度对应的二进制编码
   - 方法: 将每个 c_e * Z 四舍五入到最接近的整数，然后转为二进制表示

3. `batch_encode(solutions: list[np.ndarray], Z: float = 31.0) -> np.ndarray`:
   - 输入: 解列表，每个解形状 (25,)
   - 输出: 浓度矩阵，形状 (N, 5)
   - 使用向量化操作提高效率

4. `validate_encoding(x: np.ndarray) -> bool`:
   - 检查输入是否为有效的25维二进制向量
   - 检查所有元素是否在{0, 1}中
   - 检查总比特数是否正确

在 `tests/test_encoding.py` 中编写以下测试：
- test_single_encode: 单个25维向量的编码正确性
- test_batch_encode: 批量编码返回正确形状
- test_all_ones: 全1向量编码为等浓度 [1/5, 1/5, 1/5, 1/5, 1/5]（验证动态归一化: Z_actual=155, c_e=31/155=1/5）
- test_single_element: 仅第一个元素所有位为1，其余为0，验证 Al 浓度=1（Z_actual=31, c_0=31/31=1）
- test_normalization: 任意随机二进制向量的浓度之和严格为1（动态归一化天然保证）
- test_qubo_approximation_gap: 验证当Z_actual != 31时，QUBO能量与实际目标值存在偏差（近似误差的来源说明）
- test_decode_roundtrip: 编码->解码->再编码 的一致性

确保所有测试通过。
```

---

### Step 4: 三目标函数实现

**工作内容**:
1. 实现形成能目标 f1（Miedema混合焓模型）
2. 实现密度目标 f2（浓度加权平均）
3. 实现成本目标 f3（浓度加权平均）
4. 实现目标函数的归一化版本
5. 实现批量评估函数（用于评估求解器返回的解集）

**Claude Code Prompt**:

```text
请实现AP³-QUBO框架的三目标函数模块。在 `src/modeling/objectives.py` 中实现以下功能：

物性参数（从config导入）：
- MIX_ENTHALPY: 5×5 对称矩阵，单位 kJ/mol
  Al-Co:-19, Al-Cr:-10, Al-Fe:-11, Al-Ni:-22, Co-Cr:-4, Co-Fe:-1, Co-Ni:0, Cr-Fe:-1, Cr-Ni:-7, Fe-Ni:-2
- DENSITIES: [2.70, 8.86, 7.19, 7.87, 8.90] g/cm³
- COSTS: [2.2, 32.5, 12.0, 0.5, 16.5] USD/kg
- 归一化范围 R = [15.0, 6.2, 32.0]（用于将目标值映射到[0,1]）
- 理想点: f_min = [-15.0, 2.7, 0.5]

请实现以下函数：

1. `formation_energy(c: np.ndarray) -> float`:
   计算形成能: f1 = sum_{i<j} c_i * c_j * dH_ij
   - 输入: c 形状 (5,)，浓度向量
   - 输出: 标量，单位 kJ/mol
   - 使用上三角求和避免重复计算

2. `alloy_density(c: np.ndarray) -> float`:
   计算合金密度: f2 = sum_e c_e * density_e
   - 输入: c 形状 (5,)
   - 输出: 标量，单位 g/cm³

3. `alloy_cost(c: np.ndarray) -> float`:
   计算合金成本: f3 = sum_e c_e * cost_e
   - 输入: c 形状 (5,)
   - 输出: 标量，单位 USD/kg

4. `evaluate_objectives(c: np.ndarray) -> np.ndarray`:
   同时计算三个目标，返回形状 (3,) 的数组 [f1, f2, f3]

5. `normalize_objectives(obj: np.ndarray) -> np.ndarray`:
   将目标值归一化到 [0, 1] 区间:
   tilde_f_m = (f_m - f_m^min) / R_m
   - 输入: obj 形状 (3,) 或 (N, 3)
   - 输出: 归一化后的目标值，同形状

6. `batch_evaluate_objectives(concentrations: np.ndarray) -> np.ndarray`:
   批量评估多个浓度向量的目标函数
   - 输入: concentrations 形状 (N, 5)
   - 输出: objectives 形状 (N, 3)
   - 使用向量化numpy操作，避免Python循环

7. `objectives_from_binary(x: np.ndarray, Z: float = 31.0) -> np.ndarray`:
   直接从二进制编码计算目标函数（编码+评估的复合函数）
   - 输入: x 形状 (25,) 或 (N, 25)
   - 输出: obj 形状 (3,) 或 (N, 3)

在 `tests/test_objectives.py` 中编写测试：
- test_formation_energy_pure_al: 纯Al (c=[1,0,0,0,0]) 的形成能为0（无混合）
- test_formation_energy_symmetric: 两种浓度顺序不同的相同合金应有相同形成能
- test_density_weighted_average: 密度是严格加权平均，验证 c=[0.5,0.5,0,0,0] 时 density = 0.5*2.7+0.5*8.86
- test_cost_weighted_average: 同理验证成本
- test_normalization_range: 归一化后的值在[0,1]范围内
- test_batch_evaluate_shape: 批量评估返回正确形状
- test_from_binary: 从二进制直接计算目标函数

请确保数值精度（使用np.isclose进行浮点比较）。
```

---

### Step 5: 三类约束与惩罚项实现

**工作内容**:
1. 实现成分和约束（硬约束）的违反度计算与惩罚项
2. 实现浓度范围约束（边界约束）的违反度计算与惩罚项
3. 实现VEC相稳定性约束（软约束）的违反度计算与惩罚项
4. 实现约束违反度评估函数（用于CVD-AP模块）
5. 实现惩罚项的QUBO矩阵编码

**Claude Code Prompt**:

```text
请实现AP³-QUBO框架的约束评估与惩罚项模块。在 `src/modeling/constraints.py` 中实现以下功能：

约束定义：
1. 成分和约束 (硬约束): sum_e c_e = 1。注意：编码函数中采用动态归一化保证解码后 sum c_e = 1，
   但QUBO矩阵中使用固定常数 Z_qubo=31 进行近似，导致QUBO搜索空间中 sum c_e 不一定等于1。
   因此必须通过 Q_pen1 惩罚项在QUBO层面显式强制该约束。
2. 浓度范围约束 (硬约束): 0.05 <= c_e <= 0.35, forall e
3. VEC相稳定性约束 (软约束): 6.87 <= VEC(c) <= 8.0, 其中 VEC(c) = sum_e c_e * VEC_e

默认惩罚系数: lambda_sum=10.0, lambda_range=100.0, lambda_vec=50.0

请实现以下函数：

1. `constraint_sum_violation(c: np.ndarray) -> float`:
   成分和约束违反度: v_sum = |sum_e c_e - 1|
   - 在编码函数（动态归一化）中天然满足，但QUBO层面（固定Z_qubo=31近似）可能违反
   - 该约束用于: (a) QUBO优化中的惩罚项构建; (b) 最终解码后可行性校验

2. `constraint_range_violation(c: np.ndarray) -> float`:
   浓度范围约束违反度: v_range = sum_e [max(0, c_e - 0.35)^2 + max(0, 0.05 - c_e)^2]
   - 对每种元素分别计算上下界违反量的平方和

3. `constraint_vec_violation(c: np.ndarray) -> float`:
   VEC约束违反度: v_vec = max(0, 6.87 - VEC)^2 + max(0, VEC - 8.0)^2
   - 先计算 VEC = sum_e c_e * VEC_e
   - 再计算超出允许区间的偏离量平方和

4. `evaluate_all_violations(c: np.ndarray) -> np.ndarray`:
   同时评估三个约束的违反度
   - 输入: c 形状 (5,)
   - 输出: v 形状 (3,)，分别为 [v_sum, v_range, v_vec]

5. `penalty_sum(c: np.ndarray, lambda_sum: float = 10.0) -> float`:
   成分和惩罚项: P_sum = lambda_sum * v_sum

6. `penalty_range(c: np.ndarray, lambda_range: float = 100.0) -> float`:
   浓度范围惩罚项: P_range = lambda_range * v_range

7. `penalty_vec(c: np.ndarray, lambda_vec: float = 50.0) -> float`:
   VEC惩罚项: P_vec = lambda_vec * v_vec

8. `total_penalty(c: np.ndarray, lambdas: np.ndarray) -> float`:
   总惩罚: P_total = lambdas[0]*P_sum + lambdas[1]*P_range + lambdas[2]*P_vec
   - lambdas 形状 (3,)

9. `batch_evaluate_violations(concentrations: np.ndarray) -> np.ndarray`:
   批量评估多个浓度的约束违反度
   - 输入: concentrations 形状 (N, 5)
   - 输出: violations 形状 (N, 3)

10. `is_feasible(c: np.ndarray, tol: float = 1e-6) -> bool`:
    判断浓度向量是否可行（所有约束违反度小于tol）

在 `tests/test_constraints.py` 中编写测试：
- test_sum_constraint_perfect: 等浓度 [0.2,0.2,0.2,0.2,0.2] 的 v_sum = 0
- test_range_constraint_satisfied: 等浓度的范围违反度为0
- test_range_constraint_violated: c=[0.5,0.5,0,0,0] 时两个元素超出上界
- test_vec_constraint_satisfied: 验证等浓度的VEC在允许范围内
- test_vec_constraint_violated: c=[1,0,0,0,0] (纯Al, VEC=3.0) 应违反VEC约束
- test_feasibility: 等浓度分布应该是可行的
- test_batch_violations_shape: 批量评估返回形状 (N, 3)

注意: 所有约束评估函数应直接使用浓度向量c作为输入（而非二进制x），以保持模块独立性。上层调用者负责从x解码到c。
```

---

### Step 6: QUBO矩阵构建

**工作内容**:
1. 实现三个目标函数的QUBO矩阵编码（Q_obj1, Q_obj2, Q_obj3）
2. 实现三个约束惩罚项的QUBO矩阵编码（Q_pen1, Q_pen2, Q_pen3）
3. 实现参数化的QUBO矩阵组合函数 build_qubo_matrix(weights, lambdas)
4. 验证QUBO矩阵的上三角形式
5. 验证QUBO矩阵的对称性和正定性

**Claude Code Prompt**:

```text
请实现AP³-QUBO框架的核心QUBO矩阵构建模块。在 `src/modeling/qubo_matrix.py` 中实现以下功能：

这是整个框架最核心的计算模块，将多目标函数和约束惩罚统一编码为25×25的QUBO矩阵。

数学基础：
- 决策变量: x ∈ {0,1}^25, x[e,b] 表示元素e的第b个二进制位
- 浓度编码: c_e = (sum_b x[e,b] * 2^b) / Z, Z = 31
- QUBO标准形式: F(x) = x^T Q x + q^T x + c0
- 最终矩阵为25×25上三角实矩阵

请实现以下函数：

1. `build_q_obj1(mix_enthalpy: np.ndarray, Z: float = 31.0) -> np.ndarray`:
   目标函数1（形成能）的QUBO矩阵:
   f1 = sum_{i<j} c_i * c_j * dH_ij
   将c_i = sum_b x[i,b]*2^b/Z 代入，展开为x的二次型
   - 输入: mix_enthalpy 形状 (5, 5)
   - 输出: Q_obj1 形状 (25, 25)
   - 仅填充上三角部分(i*5+b1 <= j*5+b2)

2. `build_q_obj2(densities: np.ndarray, Z: float = 31.0) -> np.ndarray`:
   目标函数2（密度）的QUBO矩阵:
   f2 = sum_e c_e * density_e
   由于f2是c_e的线性函数，QUBO中只有对角项:
   Q_obj2[e*5+b, e*5+b] = density_e * 2^b / Z
   - 输入: densities 形状 (5,)
   - 输出: Q_obj2 形状 (25, 25)

3. `build_q_obj3(costs: np.ndarray, Z: float = 31.0) -> np.ndarray`:
   目标函数3（成本）的QUBO矩阵，与obj2类似结构

4. `build_q_pen1(Z: float = 31.0) -> np.ndarray`:
   成分和约束惩罚的QUBO矩阵:
   P_sum = (sum_e c_e - 1)^2，其中编码使用固定Z_qubo=31
   展开后包含: sum c_e^2 + 2*sum_{i<j} c_i*c_j - 2*sum c_e + 1
   - 输出: Q_pen1 形状 (25, 25)
   - 必要性说明: 由于QUBO中采用固定Z_qubo=31进行近似，解码时才用动态归一化，
     QUBO搜索空间中的解x对应的浓度和不严格等于1。该惩罚项将和约束显式编码到QUBO中，
     迫使求解器找到满足 sum c_e = 1 的解。lambda_sum 建议取较大值（如10~100）。

5. `build_q_pen2(c_min: float = 0.05, c_max: float = 0.35, Z: float = 31.0) -> np.ndarray`:
   浓度范围约束惩罚的QUBO矩阵（近似）:
   采用对角近似: Q_pen2[e*5+b, e*5+b] = ((2^b/Z) - center)^2, center=(c_min+c_max)/2
   - 输出: Q_pen2 形状 (25, 25)

6. `build_q_pen3(vecs: np.ndarray, target_vec: float = 7.435, Z: float = 31.0) -> np.ndarray`:
   VEC约束惩罚的QUBO矩阵:
   P_vec = (VEC(c) - target_vec)^2
   展开为c的二次型，再转为x的二次型
   - target_vec 取区间 [6.87, 8.0] 的中点 7.435，确保 QUBO 优化方向与约束校验口径一致
   - 最终约束合规性仍以区间 [6.87, 8.0] 的硬性检验为准
   - 输入: vecs 形状 (5,)
   - 输出: Q_pen3 形状 (25, 25)

7. `build_qubo_matrix(weights: np.ndarray, lambdas: np.ndarray, element_params: dict, mix_enthalpy: np.ndarray, Z: float = 31.0) -> np.ndarray`:
   组合所有项的参数化QUBO矩阵:
   Q = w1*Q_obj1 + w2*Q_obj2 + w3*Q_obj3 + lambda1*Q_pen1 + lambda2*Q_pen2 + lambda3*Q_pen3
   - weights 形状 (3,): [w_energy, w_density, w_cost]
   - lambdas 形状 (3,): [lambda_sum, lambda_range, lambda_vec]
   - element_params: dict 含 'density', 'cost', 'vec' 键
   - 返回上三角QUBO矩阵（确保下三角为0）

8. `qubo_energy(x: np.ndarray, Q: np.ndarray) -> float`:
   计算给定解x的QUBO能量: E = x^T Q x
   - 输入: x 形状 (25,), Q 形状 (25, 25)
   - 输出: 标量能量值

在 `tests/test_qubo_matrix.py` 中编写测试：
- test_q_obj1_shape: Q_obj1 形状为 (25, 25)
- test_q_obj2_diagonal: Q_obj2 只有对角项非零
- test_qubo_symmetric: build_qubo_matrix 返回对称矩阵
- test_qubo_upper_triangular: 下三角严格为0
- test_qubo_energy_consistency: 对同一x，手工计算能量与qubo_energy函数结果一致
- test_parameterized_qubo: 不同权重产生不同的Q矩阵
- test_zero_weights: 当所有权重和惩罚为0时，Q矩阵应为零矩阵

关键要求: 确保所有矩阵构建函数的正确性——这是整个算法的基础。建议先用小规模(2元素×2bit=4bit)手算验证，再扩展到25bit。
```

---

## Phase 3: 求解器集成

### Step 7: 模拟退火(SA)求解器实现

**工作内容**:
1. 实现基于Kaiwu SDK的SA求解器封装
2. 实现一个独立的SA求解器（作为Kaiwu不可用的fallback）
3. 实现SA参数配置（温度、冷却率、迭代次数）
4. 实现求解结果的标准化输出格式

**Claude Code Prompt**:

```text
请实现AP³-QUBO框架的求解器层。由于Kaiwu SDK可能不可用，我们需要实现一个独立的模拟退火求解器作为核心求解器，同时保留Kaiwu SDK的接口封装。

在 `src/solvers/sa_solver.py` 中实现：

1. `class SimulatedAnnealingSolver:`
   独立的SA求解器，不依赖任何外部SDK。

   初始化参数:
   - temperature: float = 1e8      (初始温度)
   - alpha: float = 0.99           (冷却系数)
   - cutoff: float = 0.01          (截止温度)
   - iter_per_t: int = 200         (每个温度下的迭代次数)
   - n_bits: int = 25              (决策变量维度)

   方法:
   - `__init__(self, **kwargs)`: 初始化参数
   - `solve(self, Q: np.ndarray, num_reads: int = 1000) -> dict`:
     主求解函数，返回标准化结果字典
     
     SA核心算法:
     对每个read (共num_reads次):
       1. 随机初始化 x ∈ {0,1}^25
       2. 当前温度 T = temperature
       3. while T > cutoff:
            重复 iter_per_t 次:
              a. 随机选择一个比特位翻转
              b. 计算能量差 delta_E (利用QUBO矩阵的增量更新)
              c. 如果 delta_E < 0: 接受翻转
              d. 否则以概率 exp(-delta_E / T) 接受翻转
            T = T * alpha
       4. 记录最终解和能量
     
     返回: {'solutions': list[np.ndarray], 'energies': list[float], 'best_idx': int}
   
   - `_flip_energy_delta(self, x: np.ndarray, Q: np.ndarray, bit_idx: int) -> float`:
     计算翻转第bit_idx位带来的能量变化（增量更新，O(n)而非O(n^2)）
     delta_E = (1 - 2*x[bit_idx]) * (Q[bit_idx, bit_idx] + 2*sum_{j!=bit_idx} Q[bit_idx, j]*x[j])
     注意: 由于Q是上三角矩阵，求和时需要特殊处理

   - `set_params(self, **kwargs)`: 动态修改SA参数

2. `solve_qubo(Q: np.ndarray, num_reads: int = 1000, **sa_kwargs) -> list[np.ndarray]`:
   便捷函数，一行调用求解QUBO
   - 输入: Q (25, 25) 上三角QUBO矩阵
   - 输出: solutions 列表，每个元素形状 (25,)

在 `src/solvers/__init__.py` 中:
- 导出 SimulatedAnnealingSolver 和 solve_qubo
- 如果Kaiwu SDK可用，也导出Kaiwu的封装器（用try-except导入）

在 `tests/test_sa_solver.py` 中编写测试：
- test_solver_initialization: 正确初始化SA求解器
- test_single_solve_shape: solve返回正确数量的解
- test_energy_consistency: 返回的解的能量值与手工计算一致
- test_flip_delta_correctness: 增量能量计算与全量计算一致
- test_deterministic_seed: 固定随机种子时结果可复现（通过np.random.seed）
- test_small_problem: 在小规模QUBO问题上验证SA能找到已知最优解
  例如: Q = np.diag([1, 2, 3])，最优解应为 [0, 0, 0]（能量=0）

性能要求:
- 25bit问题、num_reads=1000时，求解时间应在5-10秒内
- 使用增量能量更新而非每次重新计算
- 提供进度条显示（可选，使用tqdm）
```

---

### Step 8: CIM求解器接口与求解器工厂

**工作内容**:
1. 实现CIM（相干伊辛机）求解器的接口封装
2. 实现求解器工厂模式（根据配置自动选择SA或CIM）
3. 实现求解结果的标准化数据结构
4. 实现求解器性能监控

**Claude Code Prompt**:

```text
请实现AP³-QUBO框架的CIM求解器接口和求解器工厂。在 `src/solvers/cim_solver.py` 和 `src/solvers/factory.py` 中实现以下内容：

1. 在 `src/solvers/cim_solver.py` 中实现 `class CIMSolver:`:
   
   CIM（相干伊辛机）求解器的接口封装。由于CIM真机可能不可用，需要提供一个模拟模式。
   
   初始化参数:
   - platform_url: str = "https://platform.qboson.com"
   - api_key: str = "" (留空时使用模拟模式)
   - shots: int = 100
   - use_simulation: bool = True (默认使用模拟模式)

   方法:
   - `__init__(self, **kwargs)`
   - `solve(self, Q: np.ndarray, num_reads: int = None) -> dict`:
     如果 use_simulation=True:
       使用高斯噪声模拟CIM采样（作为占位实现）:
       1. 先用SA找到一个近似最优解x_best
       2. 然后围绕x_best生成num_reads个带噪声的样本:
          对每个样本，以30%概率翻转x_best的每一位
       3. 返回这些样本
     如果 use_simulation=False:
       尝试调用Kaiwu SDK（用try-except包裹）:
       ```python
       try:
           import kaiwu as kw
           ising = kw.qubo.qubo_matrix_to_ising_matrix(Q)
           optimizer = kw.cim.CIMOptimizer(...)
           results = optimizer.solve(ising, shots=num_reads)
           solutions = [np.array(s) for s in results.samples]
           return {'solutions': solutions, ...}
       except ImportError:
           raise RuntimeError("Kaiwu SDK not available, set use_simulation=True")
       ```

2. 在 `src/solvers/factory.py` 中实现求解器工厂:

   `create_solver(solver_type: str = "sa", **kwargs) -> BaseSolver`:
   - solver_type="sa": 返回 SimulatedAnnealingSolver 实例
   - solver_type="cim": 返回 CIMSolver 实例
   - solver_type="auto": 如果Kaiwu SDK可用返回CIM，否则返回SA

   `class BaseSolver(ABC):`:
   抽象基类，定义统一接口:
   - `solve(Q, num_reads) -> dict`: 必须实现
   - `set_params(**kwargs)`: 可选实现

3. 在 `src/solvers/__init__.py` 中统一导出:
   - SimulatedAnnealingSolver
   - CIMSolver
   - create_solver
   - BaseSolver

在 `tests/test_solvers.py` 中编写测试:
- test_factory_sa: 工厂正确创建SA求解器
- test_factory_cim: 工厂正确创建CIM求解器（模拟模式）
- test_cim_simulation_mode: CIM模拟模式返回正确数量的解
- test_cim_solution_shape: 所有返回的解形状为 (25,)
- test_solver_interface_consistency: SA和CIM返回结果格式一致
- test_base_solver_abstract: 不能直接实例化BaseSolver

注意: CIM求解器目前仅作为接口占位，实际CIM调用将在W7实验阶段完善。重点是确保接口一致性和可扩展性。
```

---

## Phase 4: APPS模块（自适应偏好引导Pareto搜索）

### Step 9: APPS核心算法实现

**工作内容**:
1. 实现Pareto覆盖度度量（网格密度估计）
2. 实现稀疏区域识别与引导权重计算
3. 实现权重向量的凸组合更新
4. 实现epsilon-greedy探索-利用平衡策略
5. 实现概率单纯形投影算法

**Claude Code Prompt**:

```text
请实现AP³-QUBO框架的第一个核心创新模块——APPS（Adaptive Preference-guided Pareto Search）。

在 `src/modules/apps.py` 中实现以下完整功能：

**数学原理回顾**:
- 输入: 当前Pareto解集 P, 当前权重 w^(k), 理想点 z*
- 输出: 更新后的权重 w^(k+1)
- 核心机制: 在三维归一化目标空间[0,1]^3中划分B×B×B网格，识别稀疏区域，向稀疏区域引导搜索

请实现以下函数：

1. `compute_density_grid(pareto_set: list[np.ndarray], num_bins: int = 10) -> np.ndarray`:
   在三维目标空间中进行网格密度估计:
   - 将每个维度划分为 num_bins 个区间
   - 统计每个超立方体单元中的Pareto解数量
   - 返回 density_grid 形状 (num_bins, num_bins, num_bins)
   - 归一化: d_ell = count_ell / |P|

2. `identify_sparse_cells(density_grid: np.ndarray, total_solutions: int) -> list[np.ndarray]`:
   识别稀疏单元:
   - 阈值: tau_d = 0.5 / (num_bins^3)
   - 稀疏条件: d_ell < tau_d
   - 对每个稀疏单元，计算其几何中心:
     center = ((i+0.5)/num_bins, (j+0.5)/num_bins, (k+0.5)/num_bins)
   - 返回稀疏单元中心列表，每个中心形状 (3,)

3. `compute_guidance_direction(sparse_cells: list[np.ndarray], ideal_point: np.ndarray) -> np.ndarray`:
   计算稀疏区域引导方向:
   - 对每个稀疏单元，计算到理想点的方向向量: dir_ell = center_ell - z*
   - 距离: d_ell = ||dir_ell||_2
   - 距离加权权重: alpha_ell = (1/d_ell) / sum(1/d_ell')
   - 加权方向向量: guided_dir = sum_ell alpha_ell * dir_ell
   - L1归一化: guided_dir = guided_dir / np.sum(guided_dir)
   - 设计原理: 距离理想点越远的稀疏单元获得越高的引导优先级；
     使用方向向量的加权平均替代标量-向量混合运算，确保数学维度和物理意义一致

4. `update_weights_apps(pareto_set: list[np.ndarray], current_weights: np.ndarray, ideal_point: np.ndarray, num_bins: int = 10, gamma: float = 0.3) -> np.ndarray`:
   APPS权重更新（利用阶段）:
   - 调用上述函数计算密度网格和稀疏单元
   - 计算引导方向
   - 凸组合更新: w_new = (1-gamma)*w_current + gamma*guided_dir
   - 归一化: w_new = w_new / sum(w_new)
   - 如果Pareto解集少于3个，返回current_weights不变
   - 如果没有稀疏单元，返回current_weights不变

5. `epsilon_greedy_explore(iteration: int, epsilon_0: float = 0.2, alpha: float = 0.95) -> np.ndarray`:
   epsilon-greedy探索:
   - 探索概率: epsilon_k = epsilon_0 * alpha^iteration
   - 如果random < epsilon_k:
       从Dirichlet分布 Dir([1,1,1]) 采样权重向量
       归一化后返回
   - 否则返回None（表示不探索，进入利用阶段）

6. `apps_adaptive_weights(pareto_set: list[np.ndarray], current_weights: np.ndarray, ideal_point: np.ndarray, num_bins: int = 10, gamma: float = 0.3, epsilon_0: float = 0.2, alpha: float = 0.95, iteration: int = 0) -> np.ndarray`:
   APPS完整算法入口:
   - 先检查epsilon-greedy探索条件
   - 如果不探索，调用update_weights_apps进行利用更新
   - 返回最终权重向量（确保非负且和为1）

7. `project_onto_simplex(v: np.ndarray) -> np.ndarray`:
   投影到概率单纯形（Duchi et al.算法）:
   - 输入: v ∈ R^M
   - 输出: w* ∈ Delta^{M-1}（满足w_m >= 0, sum w_m = 1）
   - 算法:
     1. u = sort(v, descending)
     2. cssv = cumsum(u) - 1
     3. ind = arange(1, M+1)
     4. rho = max{j: u_j + (1 - sum_{i=1}^j u_i)/j > 0}
     5. tau = (1 - sum_{i=1}^rho u_i) / rho
     6. w* = max(v - tau, 0)

在 `tests/test_apps.py` 中编写测试:
- test_density_grid_empty: 空Pareto集返回全零网格
- test_density_grid_uniform: 均匀分布的Pareto解产生相对均匀的密度
- test_sparse_cells_threshold: 稀疏单元识别阈值正确
- test_guidance_direction_shape: 引导方向形状为(3,)，且和为1
- test_weights_sum_to_one: 更新后的权重严格和为1
- test_weights_nonnegative: 更新后的权重非负
- test_epsilon_decay: 探索概率随迭代递减
- test_projection_simplex: 投影到单纯形后的向量满足约束
- test_projection_known_case: 已知输入的投影结果可验证
  例如: v=[0.5, 0.5, 0.5] -> 投影应为 [1/3, 1/3, 1/3]（各分量减去 (sum(v)-1)/dim = (1.5-1)/3 = 1/6）
- test_apps_with_small_pareto_set: Pareto解少于3个时返回原权重

关键验证: 确保权重向量始终在概率单纯形上，探索概率正确衰减。
```

---

## Phase 5: CVD-AP模块（约束违反度自适应惩罚）

### Step 10: CVD-AP核心算法实现

**工作内容**:
1. 实现约束违反度评估函数
2. 实现tanh有界更新规则
3. 实现惩罚系数的上界保护
4. 实现收敛判断逻辑
5. 实现完整的CVD-AP更新流程

**Claude Code Prompt**:

```text
请实现AP³-QUBO框架的第二个核心创新模块——CVD-AP（Constraint Violation Degree Adaptive Penalty）。

在 `src/modules/cvd_ap.py` 中实现以下完整功能：

**数学原理回顾**:
- 输入: 求解器返回的解集 S^(k), 当前惩罚系数 lambda^(k), 约束函数列表
- 输出: 更新后的惩罚系数 lambda^(k+1), 归一化违反度 v_tilde, 收敛标志
- 核心机制: 根据约束违反度的实时反馈，动态调整各约束的独立惩罚系数

请实现以下函数:

1. `evaluate_constraint_violations(solutions: list[np.ndarray], constraint_funcs: list[callable]) -> np.ndarray`:
   评估解集中各约束的原始平均违反度:
   - solutions: 列表，每个元素形状(25,)，共N个解
   - constraint_funcs: 列表，每个函数接收浓度向量c(形状(5,))返回标量违反度
   - 对每个约束c，计算: v_c = (1/N) * sum_{x in S} max(0, g_c(c(x)))
   - 返回 raw_violations 形状 (C,)，C为约束数量

2. `normalize_violations(raw_violations: np.ndarray) -> np.ndarray`:
   归一化违反度（消除量纲差异）:
   - v_tilde_c = v_c / (max(v) + 1e-10)
   - 返回 [0, 1] 区间的归一化违反度
   - 最大违反约束的归一化值为1

3. `tanh_update_rule(current_lambdas: np.ndarray, normalized_violations: np.ndarray, alpha: float = 0.2, beta: float = 5.0) -> np.ndarray`:
   tanh有界更新规则:
   - lambda_c^(k+1) = lambda_c^(k) * [1 + alpha * tanh(beta * v_tilde_c)]
   - 当v_tilde=0时，更新因子=1（保持不变）
   - 当v_tilde=1时，更新因子≈1+alpha（最大增幅）
   - alpha控制最大单次增幅（默认20%）
   - beta控制tanh曲线的陡峭度（默认5.0）

4. `apply_upper_bound(lambdas: np.ndarray, lambda_max: float = 10000.0) -> np.ndarray`:
   惩罚系数上界保护:
   - lambda_c = min(lambda_c, lambda_max)
   - 防止数值溢出和惩罚项过度支配

5. `check_convergence(normalized_violations: np.ndarray, tol: float = 1e-6) -> bool`:
   收敛判断:
   - 如果所有 v_tilde_c < tol，返回True
   - 表示所有约束均已满足，后续可跳过CVD-AP更新

6. `cvd_ap_update(solutions: list[np.ndarray], constraint_funcs: list[callable], current_lambdas: np.ndarray, alpha: float = 0.2, beta: float = 5.0, lambda_max: float = 10000.0, convergence_tol: float = 1e-6) -> tuple[np.ndarray, np.ndarray, bool]`:
   CVD-AP完整更新流程:
   - 步骤1: 评估原始违反度
   - 步骤2: 归一化
   - 步骤3: 收敛判断（如果收敛直接返回）
   - 步骤4: tanh有界更新
   - 步骤5: 上界保护
   - 返回: (new_lambdas, normalized_violations, converged)

7. 预定义约束函数（适配AlCoCrFeNi系统）:
   ```python
   def make_constraint_functions(element_params: dict, Z: float = 31.0):
       """创建三个约束评估函数"""
       vecs = element_params['vec']
       
       def g_sum(x):
           c = encode_concentration(x, Z)
           return abs(np.sum(c) - 1.0)
       
       def g_range(x):
           c = encode_concentration(x, Z)
           violation = 0.0
           for ce in c:
               violation += max(0, ce - 0.35)**2 + max(0, 0.05 - ce)**2
           return violation
       
       def g_vec(x):
           c = encode_concentration(x, Z)
           vec = np.dot(c, vecs)
           return max(0, 6.87 - vec)**2 + max(0, vec - 8.0)**2
       
       return [g_sum, g_range, g_vec]
   ```

在 `tests/test_cvd_ap.py` 中编写测试:
- test_violation_evaluation: 正确评估解集的约束违反度
- test_normalization: 归一化后最大值为1
- test_tanh_bounded: tanh更新后的lambda不超过 (1+alpha)倍原值
- test_upper_bound: 上界保护正确截断
- test_convergence_all_satisfied: 所有违反度为0时判定收敛
- test_convergence_not_satisfied: 有违反度时未收敛
- test_cvd_ap_shape: 返回的new_lambdas形状与输入一致
- test_zero_violation_no_change: 违反度全为0时lambda不变
- test_high_violation_increases: 高违反度导致lambda增加
- test_lambda_increase_monotonic: 违反度越高，lambda增幅越大（tanh单调性）

注意: constraint_funcs接收的是二进制向量x(形状(25,))而非浓度c，需要在函数内部解码。这与Step 5中的约束评估函数（直接接收c）是不同的接口层次。
```

---

## Phase 6: APPC模块（偏好-惩罚协同演化）

### Step 11: APPC协同机制实现

**工作内容**:
1. 实现P2W耦合函数（惩罚到偏好）
2. 实现W2P耦合函数（偏好到惩罚）
3. 实现联合更新方程
4. 实现APPC的完整入口函数
5. 确保APPC正确调用APPS和CVD-AP

**Claude Code Prompt**:

```text
请实现AP³-QUBO框架的第三个核心创新模块——APPC（Adaptive Preference-Penalty Coevolution）。这是整个框架最关键的创新点，负责建立偏好引导与惩罚控制之间的双向协同。

在 `src/modules/appc.py` 中实现以下完整功能：

**数学原理回顾**:
- 协同状态: s^(k) = [w^(k); lambda^(k); v^(k)]
- P2W耦合: 惩罚偏离驱动权重微调，缓解约束压力
- W2P耦合: 权重变化预判新方向的约束难度，预调惩罚
- 联合更新: w^(k+1) = Proj[w^(k) + eta_w * (delta_w_APPS + delta_w_pen)]
            lambda^(k+1) = Clip[lambda^(k) + delta_lambda_CVD + rho_co * delta_lambda_pref]

请实现以下函数:

1. `compute_p2w_coupling(current_lambdas: np.ndarray, violation_degrees: np.ndarray, rho_co: float = 0.5, epsilon: float = 1e-10) -> np.ndarray`:
   P2W耦合函数（惩罚到偏好）:
   - lambda_mean = mean(lambda)
   - lambda_dev = lambda - lambda_mean
   - lambda_dev_norm = ||lambda_dev|| + epsilon
   - v_norm = ||violation_degrees|| + epsilon
   - delta_w_pen = -rho_co * (lambda_dev / lambda_dev_norm) * (violation_degrees / v_norm)
   - 返回形状 (C,)，截断至前3维用于权重更新
   
   设计原理: 惩罚显著偏高且违反度大的约束产生更强的反向推动力，
   负号表示将搜索方向轻微推离该约束的敏感区域。

2. `compute_w2p_coupling(weight_change: np.ndarray, violation_degrees: np.ndarray, rho_co: float = 0.5, lambda_scale: float = 10.0, n_lambdas: int = 3) -> np.ndarray`:
   W2P耦合函数（偏好到惩罚）:
   - effective_dim = min(len(weight_change), n_lambdas)
   - delta_lambda_pref = zeros(n_lambdas)
   - delta_lambda_pref[:effective_dim] = rho_co * lambda_scale * |weight_change[:effective_dim]| * violation_degrees[:effective_dim]
   - 返回形状 (n_lambdas,)
   
   设计原理: 权重变化幅度大意味着搜索方向显著偏移，
   新区域可能激活此前未成为瓶颈的约束。

3. `appc_coevolution_update(pareto_set: list[np.ndarray], current_weights: np.ndarray, ideal_point: np.ndarray, solutions: list[np.ndarray], constraint_funcs: list[callable], current_lambdas: np.ndarray, rho_co: float = 0.5, eta_w: float = 0.3, eta_lambda: float = 0.2, lambda_scale: float = 10.0, lambda_min: float = 0.1, lambda_max: float = 10000.0, **kwargs) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]`:
   
   APPC完整协同更新入口函数:
   
   步骤1: 独立更新（调用子模块）
   - 1a. 调用 `apps_adaptive_weights` 计算独立权重更新方向 delta_w_APPS
   - 1b. 调用 `cvd_ap_update` 计算独立惩罚更新 delta_lambda_CVD
   
   步骤2: 协同耦合计算
   - 2a. 调用 compute_p2w_coupling 计算 P2W 耦合增量
   - 2b. 调用 compute_w2p_coupling 计算 W2P 耦合增量
   
   步骤3: 联合更新
   - 3a. 权重: w_tilde = current_weights + eta_w * delta_w_APPS + eta_w * delta_w_pen[:3]
          new_weights = project_onto_simplex(w_tilde)
   - 3b. 惩罚: new_lambdas = current_lambdas + delta_lambda_CVD + rho_co * delta_lambda_pref
          new_lambdas = clip(new_lambdas, lambda_min, lambda_max)
   
   返回: (new_weights, new_lambdas, violation_degrees, converged)

4. `apps_compute_weight_delta(pareto_set, current_weights, ideal_point, **kwargs) -> np.ndarray`:
   计算APPS独立权重更新量（封装apps_adaptive_weights的逻辑）:
   - 返回 delta_w = w_new - w_current（而非绝对权重）

在 `tests/test_appc.py` 中编写测试:
- test_p2w_shape: P2W耦合输出形状正确
- test_p2w_direction: 当某个约束惩罚远高于平均时，P2W产生负向推动
- test_w2p_scale: W2P耦合的量级与lambda_scale成正比
- test_weighted_weights_sum_to_one: 联合更新后的权重和为1
- test_lambdas_within_bounds: 更新后的惩罚在[lambda_min, lambda_max]内
- test_rho_co_zero: rho_co=0时APPC退化为独立运行（P2W和W2P输出为0）
- test_rho_co_one: rho_co=1时协同信号全额注入
- test_convergence_propagation: CVD-AP判定收敛时，APPC也返回converged=True
- test_appc_integration: 完整调用appc_coevolution_update，验证输出格式正确

在 `src/modules/__init__.py` 中导出所有三个模块的公共接口。

关键验证: 当rho_co=0时，APPC应完全退化为APPS和CVD-AP独立运行，输出应与分别调用两个模块一致。请在测试中验证这一点。
```

---

## Phase 7: 主算法集成

### Step 12: Pareto解集管理与主循环

**工作内容**:
1. 实现非支配排序算法
2. 实现Pareto解集的增量更新
3. 实现主循环的完整逻辑
4. 实现结果收集与历史记录
5. 实现收敛判断与提前终止

**Claude Code Prompt**:

```text
请实现AP³-QUBO框架的Pareto解集管理和主算法循环。这是将所有模块集成为完整算法的关键步骤。

**A. Pareto解集管理** — 在 `src/core/pareto.py` 中实现:

1. `dominates(obj_a: np.ndarray, obj_b: np.ndarray) -> bool`:
   判断解a是否支配解b:
   - 对所有目标: obj_a[m] <= obj_b[m]
   - 至少一个目标: obj_a[m] < obj_b[m]
   - 输入: obj_a, obj_b 形状 (3,)（归一化目标值，越小越好）

2. `update_pareto_set(pareto_set: list[np.ndarray], pareto_objs: list[np.ndarray], new_x: np.ndarray, new_obj: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]`:
   增量更新Pareto解集:
   - 如果new_x被现有任何解支配: 不添加
   - 如果new_x支配某些现有解: 移除被支配的，添加new_x
   - 如果互不支配: 添加new_x
   - 返回更新后的 (pareto_set, pareto_objs)

3. `batch_update_pareto(pareto_set: list[np.ndarray], pareto_objs: list[np.ndarray], new_solutions: list[np.ndarray], new_objectives: list[np.ndarray]) -> tuple[list[np.ndarray], list[np.ndarray]]`:
   批量更新Pareto解集（处理求解器返回的一批解）

4. `compute_hypervolume(pareto_objs: list[np.ndarray], reference_point: np.ndarray) -> float`:
   计算Hypervolume指标（使用 pymoo.indicators.hv.HV 标准实现）:
   - reference_point: 参考点，通常取各目标最大值
   - 调用 pymoo 的精确HV计算，确保与学术界通用标准对标
   - 返回Pareto前沿与参考点围成的超体积

5. `compute_spread(pareto_objs: list[np.ndarray]) -> float`:
   计算Spread指标（解集分布均匀性）:
   - 计算相邻解之间的距离标准差

**B. 主算法循环** — 在 `src/core/main_loop.py` 中实现:

`ap3_qubo_optimize(element_names, bits_per_element, densities, costs, vecs, mix_enthalpy, max_iterations, num_reads, initial_weights, initial_lambdas, apps_kwargs, cvdap_kwargs, appc_kwargs, use_cim, **solver_kwargs) -> dict`:

完整主算法，遵循"构建-求解-评估-演化"四阶段循环:

```python
def ap3_qubo_optimize(
    element_names=None, bits_per_element=5,
    densities=None, costs=None, vecs=None, mix_enthalpy=None,
    max_iterations=50, num_reads=1000,
    initial_weights=None, initial_lambdas=None,
    apps_kwargs=None, cvdap_kwargs=None, appc_kwargs=None,
    use_cim=False, **solver_kwargs
) -> dict:
    # 1. 参数初始化
    # 2. 求解器初始化 (SA or CIM)
    # 3. 历史记录初始化
    # 4. 主循环 (for k in range(max_iterations)):
    #    Stage 1: build_qubo_matrix(w^(k), lambda^(k))
    #    Stage 2: solver.solve(Q, num_reads)
    #    Stage 3: batch_update_pareto()
    #    Stage 4: appc_coevolution_update()
    #    Stage 5: 记录历史
    #    Stage 6: 收敛判断
    # 5. 结果组装与返回
```

返回结果字典:
```python
{
    'pareto_set': list[np.ndarray],        # Pareto解集（二进制向量）
    'pareto_objectives': list[np.ndarray],  # 对应的三目标函数值
    'weight_history': list[np.ndarray],     # 权重演化轨迹
    'lambda_history': list[np.ndarray],     # 惩罚系数演化轨迹
    'violation_history': list[np.ndarray],  # 约束违反度演化轨迹
    'ideal_point': np.ndarray,              # 最终理想点
    'num_iterations': int,                  # 实际执行迭代次数
    'hypervolume_history': list[float],     # HV指标历史
}
```

在 `tests/test_pareto.py` 中编写测试:
- test_dominates_basic: a=[1,1,1] 支配 b=[2,2,2]
- test_dominates_false: a=[1,2,3] 不支配 b=[2,1,3]（互不占优）
- test_pareto_update_add: 新解不被支配时正确添加
- test_pareto_update_remove: 新解支配旧解时正确移除被支配解
- test_pareto_no_duplicate: 相同解不重复添加
- test_hypervolume_positive: HV指标为正
- test_hypervolume_monotonic: Pareto集扩大时HV不减

在 `tests/test_main_loop.py` 中编写测试:
- test_main_loop_runs: 主循环能正常运行不报错
- test_main_loop_returns_correct_keys: 返回字典包含所有预期键
- test_main_loop_pareto_nonempty: 返回的Pareto集非空
- test_main_loop_history_length: 历史记录长度等于迭代次数+1
- test_main_loop_weights_normalized: 所有历史权重和为1
- test_main_loop_lambdas_positive: 所有历史惩罚系数为正
- test_convergence_early_stop: 收敛时提前终止（将convergence_tol设大测试）
- test_small_iteration: 少量迭代（如5次）也能完成完整流程

**重要**: 主循环的每次迭代应打印进度信息，格式:
```
Iter 01/50 | Pareto: 42 | HV: 0.8523 | Weights: [0.33,0.33,0.34] | Lambdas: [10.0,100.0,50.0] | Converged: False
```

请确保所有模块正确导入和调用，主循环是各模块的集成测试。
```

---

## Phase 8: 实验验证

### Step 13: 实验一 — APPS模块验证

**工作内容**:
1. 实现固定权重Baseline求解器
2. 实现实验一的对比组（Baseline vs APPS）
3. 实现评价指标计算（HV, Spread, Pareto数量, Coverage Rate）
4. 运行实验并可视化结果

**Claude Code Prompt**:

```text
请实现AP³-QUBO框架的实验一——APPS模块验证实验。在 `experiments/exp01_apps_validation.py` 中实现:

**实验目的**: 验证APPS自适应偏好引导策略相比固定权重策略在Pareto前沿覆盖质量上的提升。

对比组设计:
| 组别 | 权重策略 |
|------|----------|
| Baseline | 固定权重 (0.33, 0.33, 0.34)，全程不变 |
| APPS | 自适应动态权重，根据覆盖度调整 |

评价指标:
1. Hypervolume (HV): Pareto前沿与参考点围成的超体积，越大越好
2. Spread (SP): 相邻解距离的标准差，越小分布越均匀
3. Pareto解集数量 |P|: 非支配解个数，越多越好
4. Coverage Rate (CR): 前沿覆盖比例，越多越好

实验设置:
- 运行10次独立实验取平均值
- 每次实验: 50次迭代，每次迭代1000次采样
- 参考点: z_ref = [0, 0, 0]（归一化目标空间，越小越好，所以参考点取原点反向）
  实际上应取各目标在单目标优化中的最大值

请实现以下函数:

1. `run_baseline_trial(max_iterations=50, num_reads=1000, random_seed=None) -> dict`:
   固定权重Baseline的一次实验运行:
   - 权重始终为 [1/3, 1/3, 1/3]
   - 其他参数与主算法一致
   - 调用 ap3_qubo_optimize 但禁用APPS（设置固定权重）
   - 返回完整结果字典

2. `run_apps_trial(max_iterations=50, num_reads=1000, random_seed=None) -> dict`:
   APPS组的一次实验运行:
   - 启用APPS自适应权重
   - 其他参数一致
   - 返回完整结果字典

3. `compute_metrics(result: dict) -> dict`:
   计算实验评价指标:
   - HV: 使用 src.core.pareto.compute_hypervolume
   - SP: 使用 src.core.pareto.compute_spread
   - |P|: len(result['pareto_set'])
   - CR: 自定义计算（Pareto解覆盖的目标空间比例）

4. `run_experiment(n_trials=10, **kwargs) -> dict`:
   完整实验流程:
   - 运行10次Baseline和10次APPS
   - 收集所有指标
   - 计算均值和标准差
   - 进行Wilcoxon秩和检验（scipy.stats.wilcoxon）
   - 返回统计结果

5. `plot_results(results: dict, save_path: str = None)`:
   可视化实验结果:
   - 图1: 两组HV对比柱状图（带误差棒）
   - 图2: 两组SP对比柱状图
   - 图3: 两组|P|对比柱状图
   - 图4: 典型运行的Pareto前沿3D散点图（Baseline vs APPS）

6. `print_latex_table(results: dict)`:
   打印LaTeX格式的结果表格

在实验脚本底部添加 `if __name__ == "__main__"` 块，运行完整实验并保存结果到 `results/exp01/` 目录。

输出要求:
- 实验结果保存为 JSON 格式
- 图表保存为 PNG 格式（300 DPI）
- 在控制台打印统计摘要
- 如果APPS相比Baseline的HV提升超过15%，打印 "✓ APPS验证通过"

运行命令: `python experiments/exp01_apps_validation.py`
```

---

### Step 14: 实验二 — CVD-AP模块验证

**工作内容**:
1. 实现固定小惩罚和固定大惩罚的Baseline
2. 实现CVD-AP组的实验流程
3. 实现评价指标（可行解率、收敛速度、最优目标值、平均违反度）
4. 运行实验并可视化

**Claude Code Prompt**:

```text
请实现AP³-QUBO框架的实验二——CVD-AP模块验证实验。在 `experiments/exp02_cvdap_validation.py` 中实现:

**实验目的**: 验证CVD-AP自适应惩罚机制相比固定惩罚系数在约束满足性上的优势。

对比组设计:
| 组别 | 惩罚策略 | 惩罚系数 |
|------|----------|----------|
| Baseline 1 | 固定小惩罚 | lambda = 10（过小） |
| Baseline 2 | 固定大惩罚 | lambda = 1000（过大） |
| CVD-AP | 自适应惩罚 | 初始100，按tanh规则动态调整 |

评价指标:
1. 可行解率 FR: 可行解数/总采样数 × 100%，越大越好
2. 收敛速度 CS: 可行解率达到90%所需迭代次数，越小越好
3. 最优目标值 f_best: 可行解中的最优标量化值，越小越好
4. 平均约束违反度 v_bar: 所有解的违反度算术平均，越小越好

实验设置:
- 固定权重 w = [0.4, 0.3, 0.3]
- SA求解器，每次迭代1000次采样，50次迭代
- CVD-AP参数: alpha=0.2, beta=5.0, lambda_max=10000

请实现以下函数:

1. `run_fixed_small_trial(**kwargs) -> dict`:
   固定小惩罚Baseline实验，lambdas=[10.0, 10.0, 10.0]

2. `run_fixed_large_trial(**kwargs) -> dict`:
   固定大惩罚Baseline实验，lambdas=[1000.0, 1000.0, 1000.0]

3. `run_cvdap_trial(**kwargs) -> dict`:
   CVD-AP实验，初始lambdas=[100.0, 100.0, 50.0]，启用自适应更新

4. `compute_feasibility_rate(solutions: list[np.ndarray], tol: float = 1e-6) -> float`:
   计算可行解率

5. `compute_convergence_speed(violation_history: list[np.ndarray], threshold: float = 0.9) -> int`:
   计算达到目标可行解率所需的迭代次数

6. `run_experiment(n_trials=10, **kwargs) -> dict`:
   完整实验流程:
   - 运行10次三组实验
   - 收集可行解率、收敛速度、最优值、违反度
   - 计算统计值
   - Wilcoxon秩和检验

7. `plot_convergence_curves(results: dict, save_path: str = None)`:
   绘制收敛曲线:
   - 图1: 三组的平均可行解率随迭代的变化曲线
   - 图2: 三组的平均约束违反度随迭代的变化曲线
   - 图3: 三组的惩罚系数演化轨迹对比（CVD-AP组）

8. `print_latex_table(results: dict)`:
   打印LaTeX结果表格

预期结果判定:
- CVD-AP可行解率稳定在90%以上: "✓ FR目标达成"
- CVD-AP 30次迭代内达到90%: "✓ CS目标达成"
- Baseline 2最优值劣于CVD-AP至少20%: "✓ 无目标淹没"

运行命令: `python experiments/exp02_cvdap_validation.py`
```

---

### Step 15: 实验三 — APPC协同验证 + 实验四准备

**工作内容**:
1. 实现实验三的四个Variant对比（A/B/C/D）
2. 实现APPC协同机制的增量价值验证
3. 实现CIM求解器接口的初步测试
4. 为实验五（NSGA-II基准对比）准备AP³-QUBO的标准化输出接口
5. 生成所有实验的汇总报告

**Claude Code Prompt**:

```text
请实现AP³-QUBO框架的实验三——APPC协同验证实验，以及实验四的CIM可迁移性准备。在 `experiments/exp03_appc_validation.py` 和 `experiments/exp04_cim_migration.py` 中实现:

**实验三: APPC协同验证**

对比组设计:
| 组别 | APPS | CVD-AP | APPC协同 |
|------|------|--------|----------|
| Variant A | 开 | 关(固定λ=100) | 关 |
| Variant B | 关(固定权重) | 开 | 关 |
| Variant C | 开 | 开 | 关 |
| Variant D | 开 | 开 | 开 |

评价指标:
- Pareto覆盖率 PCR
- 可行Pareto率 FPR
- Hypervolume HV
- 综合目标值 g^te (Tchebycheff标量化)

在 `experiments/exp03_appc_validation.py` 中实现:

1. `run_variant_a_trial(**kwargs) -> dict`:
   Variant A: APPS + 固定惩罚
   - 启用APPS，禁用CVD-AP（rho_co=0）
   - 固定惩罚系数 lambda=[100, 100, 100]

2. `run_variant_b_trial(**kwargs) -> dict`:
   Variant B: 固定权重 + CVD-AP
   - 禁用APPS（权重固定为[1/3,1/3,1/3]），启用CVD-AP
   - rho_co=0

3. `run_variant_c_trial(**kwargs) -> dict`:
   Variant C: APPS + CVD-AP 无协同
   - 两个模块都启用，但 rho_co=0

4. `run_variant_d_trial(**kwargs) -> dict`:
   Variant D: 完整框架
   - 两个模块都启用，rho_co=0.5

5. `compute_tchebycheff(objectives: np.ndarray, weights: np.ndarray, ideal_point: np.ndarray) -> float`:
   Tchebycheff标量化函数:
   g^te = max_m { w_m * |f_m(x) - z_m*| }

6. `run_experiment(n_trials=10, **kwargs) -> dict`:
   完整实验:
   - 运行10次四个Variant
   - 关键对比: Variant C vs D（APPC增量价值）
   - 统计检验

7. `plot_comparison(results: dict)`:
   - 雷达图: 四个Variant的多指标对比
   - 柱状图: Variant C vs D的核心指标对比（突出增量价值）

预期结果判定:
- Variant D相比C的HV提升>10%: "✓ APPC HV提升验证通过"
- Variant D相比C的FPR提升>15%: "✓ APPC FPR提升验证通过"

**实验四: CIM可迁移性准备**

在 `experiments/exp04_cim_migration.py` 中实现:

1. `test_cim_interface()`:
   测试CIM求解器接口:
   - 创建小规模QUBO问题
   - 调用CIMSolver（模拟模式）
   - 验证返回结果格式正确
   - 对比SA和CIM模拟模式的解质量

2. `benchmark_solver_comparison()`:
   SA vs CIM模拟对比:
   - 同一QUBO问题
   - 相同num_reads
   - 对比最优能量、成功率、求解时间
   - 生成对比表格

3. `generate_cim_experiment_template()`:
   生成CIM真机实验模板:
   - shots=50/100/500三种配置
   - 数据记录格式
   - 结果分析脚本框架

实验结果保存到 `results/` 目录:
- exp03/ : APPC实验结果
- exp04/ : CIM迁移准备结果
- exp05_nsga2_benchmark.py : NSGA-II外部基准对比
- summary/ : 所有实验汇总报告

运行命令:
```bash
python experiments/exp03_appc_validation.py
python experiments/exp04_cim_migration.py
```

最后，请创建一个 `experiments/run_all.py` 脚本，按顺序运行所有五个实验:
```python
# run_all.py
import subprocess
import sys

experiments = [
    "exp01_apps_validation.py",
    "exp02_cvdap_validation.py",
    "exp03_appc_validation.py",
    "exp04_cim_migration.py",
    "exp05_nsga2_benchmark.py",
]

for exp in experiments:
    print(f"\n{'='*60}")
    print(f"Running {exp}...")
    print(f"{'='*60}")
    result = subprocess.run([sys.executable, f"experiments/{exp}"])
    if result.returncode != 0:
        print(f"ERROR: {exp} failed!")
        break
    print(f"✓ {exp} completed successfully")

print("\n" + "="*60)
print("All experiments completed!")
print("="*60)
```
```

---

### Step 16: 实验五 — NSGA-II外部基准对比

> **新增实验**: 与经典多目标进化算法 NSGA-II 进行横向对比，验证 AP³-QUBO 相比传统方法的优势。
> **必要性**: 仅有内部模块对比（Baseline vs APPS vs APPC）不足以证明"使用QUBO建模"的必要性，
> 必须通过与 NSGA-II、MOEA/D 等经典算法的横向对比，回答核心问题："为什么要用QUBO？"

**工作内容**:
1. 实现 NSGA-II 基准求解器（调用 pymoo 库）
2. 统一优化问题建模（相同的决策变量、目标函数、约束条件）
3. 实现评价指标对比（HV、运行时间、可行解率、Pareto解数量）
4. 运行实验并生成对比图表

**Claude Code Prompt**:

```text
请实现AP³-QUBO框架的实验五——NSGA-II外部基准对比实验。在 `experiments/exp05_nsga2_benchmark.py` 中实现:

**实验目的**: 通过与经典多目标进化算法 NSGA-II 的横向对比，验证 AP³-QUBO 在以下维度上的竞争力。

对比组设计:
| 组别 | 算法类型 | 求解方法 |
|------|----------|----------|
| AP³-QUBO | 量子启发式 | QUBO建模 + SA求解 + APPS/CVD-AP/APPC自适应模块 |
| NSGA-II | 进化算法 | 遗传算子（交叉+变异）+ 非支配排序 |

统一问题建模:
- 决策变量: 5维连续浓度向量 c ∈ [0,1]^5，满足 sum c_e = 1
- 目标函数: f1=形成能, f2=密度, f3=成本（与AP³-QUBO完全一致）
- 约束条件: 0.05 <= c_e <= 0.35, 6.87 <= VEC(c) <= 8.0（完全一致）
- NSGA-II采用 pymoo 的 Problem 接口统一封装

AP³-QUBO实验设置:
- 完整启用 APPS + CVD-AP + APPC
- max_iterations=50, num_reads=1000
- 运行10次取平均

NSGA-II实验设置:
- 使用 pymoo.algorithms.moo.nsga2.NSGA2
- 种群大小 pop_size=100
- 迭代代数 n_generations=500（总评估次数=50,000，与AP³-QUBO的50×1000=50,000对齐）
- 交叉概率 0.9，变异概率 0.1
- 运行10次取平均

评价指标:
1. Hypervolume (HV): 相同参考点下的超体积对比，越高越好
2. 运行时间 RT: 单次实验 wall-clock 时间，越低越好
3. 可行解率 FR: 最终解集中满足所有约束的比例，越高越好
4. Pareto解数量 |P|: 非支配解个数
5. IGD (Inverted Generational Distance): 到真实前沿的距离，越低越好

请实现以下函数:

1. `class HEAProblem(ElementwiseProblem):`
   高熵合金多目标优化问题封装（pymoo Problem接口）:
   - n_var=5, n_obj=3, n_ieq_constr=2
   - xl=np.array([0.05]*5), xu=np.array([0.35]*5)
   - 自动施加 sum c_e = 1 约束（通过归一化或等式约束）
   - _evaluate 方法计算三个目标和约束违反度

2. `run_nsga2_trial(random_seed=None, **kwargs) -> dict:`
   运行一次NSGA-II实验:
   - 初始化 HEAProblem
   - 创建 NSGA2 算法实例
   - 调用 minimize 求解
   - 提取 Pareto 解集、目标值、运行时间
   - 返回标准化结果字典

3. `run_ap3qubo_trial(random_seed=None, **kwargs) -> dict:`
   运行一次AP³-QUBO完整实验:
   - 调用 ap3_qubo_optimize（启用全部模块）
   - 返回标准化结果字典

4. `compute_benchmark_metrics(results: list[dict]) -> dict:`
   计算两组实验的对比指标:
   - 对每组10次实验计算各指标的均值和标准差
   - 进行Wilcoxon秩和检验
   - 返回统计对比结果

5. `plot_benchmark_comparison(results: dict, save_path: str = None):`
   生成对比图表:
   - 图1: HV对比箱线图（AP³-QUBO vs NSGA-II，10次运行）
   - 图2: 运行时间对比柱状图
   - 图3: 可行解率对比
   - 图4: 两组Pareto前沿的3D散点叠加图
   - 图5: HV收敛曲线对比（AP³-QUBO迭代 vs NSGA-II代数）

6. `print_benchmark_latex_table(results: dict):`
   打印LaTeX格式的对比表格

预期结果判定（AP³-QUBO相对NSGA-II）:
- HV差距在 ±10% 以内: "✓ HV竞争力达标"
- 运行时间不超过NSGA-II的3倍: "✓ 时间效率可接受"
- 可行解率不低于NSGA-II: "✓ 约束处理能力达标"
- 若三项均达标，则AP³-QUBO可作为QUBO方法的valid baseline

运行命令: `python experiments/exp05_nsga2_benchmark.py`
```

---

## 总结：完整开发路线图

```
Phase 1: 基础设施 ─────────────────────────────────────────
  Step 1: 项目骨架与依赖管理
  Step 2: 元素参数与物性数据验证

Phase 2: QUBO基础建模层 ───────────────────────────────────
  Step 3: 决策变量编码与浓度计算
  Step 4: 三目标函数实现
  Step 5: 三类约束与惩罚项实现
  Step 6: QUBO矩阵构建

Phase 3: 求解器集成 ──────────────────────────────────────
  Step 7: 模拟退火(SA)求解器实现
  Step 8: CIM求解器接口与求解器工厂

Phase 4: APPS模块 ─────────────────────────────────────────
  Step 9: APPS核心算法实现

Phase 5: CVD-AP模块 ──────────────────────────────────────
  Step 10: CVD-AP核心算法实现

Phase 6: APPC模块 ─────────────────────────────────────────
  Step 11: APPC协同机制实现

Phase 7: 主算法集成 ──────────────────────────────────────
  Step 12: Pareto解集管理与主循环

Phase 8: 实验验证 ─────────────────────────────────────────
  Step 13: 实验一 — APPS验证
  Step 14: 实验二 — CVD-AP验证
  Step 15: 实验三+四 — APPC协同 + CIM迁移
  Step 16: 实验五 — NSGA-II外部基准对比
```

**关键依赖关系**:
- W2依赖W1（环境就绪）
- W3依赖W2（QUBO模型可运行）
- W4和W5可并行（APPS与CVD-AP独立）
- W6依赖W4和W5（需要APPS和CVD-AP完成）
- W7依赖W6（完整APPC算法）

**每周对应步骤建议**:
| 周次 | 步骤 | 核心交付物 |
|------|------|-----------|
| W1 | Step 1-2 | 项目骨架 + 参数验证通过 |
| W2 | Step 3-6 | QUBO建模层完整可运行 |
| W3 | Step 7-8 | SA+CIM求解器集成 |
| W4 | Step 9 | APPS模块 + 单元测试 |
| W5 | Step 10 | CVD-AP模块 + 单元测试 |
| W6 | Step 11-12 | APPC模块 + 主循环集成 |
| W7 | Step 13-16 | 五个实验全部完成 |
| W8 | - | 论文 + PPT + Demo（人工完成）|

---

**使用说明**:

1. 将每个Step的Prompt完整复制到Claude Code中
2. Claude Code会按Prompt要求生成对应代码
3. 每步完成后运行 `python -m pytest` 确保测试通过
4. 按顺序执行，不要跳步（后续步骤依赖前面模块）
5. 如果遇到问题，可以在Prompt末尾追加 "请详细解释" 或 "请添加更多调试信息"
