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