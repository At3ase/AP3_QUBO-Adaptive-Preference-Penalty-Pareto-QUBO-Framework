"""
P0-10 独立验证：γ 敏感性实验 TOP10 重叠率计算逻辑。

背景（Code_Completion_Review_2026-07-18 P0-10）：
    旧实现用 min(hv_mean/baseline_hv, 1.0) 冒充"与基准 TOP10 解重叠率"。
    修复后改为成分空间容差匹配：基准 γ=0.25 与各 γ 的 TOP10 成分快照
    （Archive.get_top_k(10)，QUBO 能量升序）做一对一贪婪匹配，
    各元素 at% 差均 < tolerance_at 视为同一解，重叠率 = |交集| / 10。

环境限制：
    kaiwu SDK 与 scipy 未安装，sensitivity 的 import 链
    （experiments/__init__ → ablation → pareto_zoom → qubo.builder → kaiwu；
     statistics.reporting → hypothesis_tests → scipy）无法直接 import。
    本测试通过 sys.modules 打桩隔离重依赖后，用 importlib 加载
    真实的 sensitivity.py 源码进行验证 —— 被测逻辑为线上真实代码。

验证用例：
    1. 全同快照        → 重叠率 1.0
    2. 全不同快照      → 重叠率 0.0
    3. 部分重叠 (4/10) → 重叠率 0.4
    4. 容差边界：差 = 0.5 at% 不匹配；差 = 0.49 at% 匹配
    5. 一对一匹配防重复计数（基准两个相同解只能匹配候选一个解）
    6. 跨重复均值 _mean_top10_overlap
    7. 集成：FakeParetoZoom + 人工 archive 端到端跑 run()/report()

运行：
    python tests/verify_top10_overlap.py
"""

import sys
import types
import importlib.util
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))


# =============================================================================
# 打桩：隔离 kaiwu / scipy 重依赖链（被测 sensitivity.py 本身为真实源码）
# =============================================================================

def _stub_pkg(name: str) -> types.ModuleType:
    """注册空包占位，阻断包级 __init__ 的重导入链。"""
    mod = types.ModuleType(name)
    mod.__path__ = []
    sys.modules[name] = mod
    return mod


import ap3_qubo  # noqa: E402  真实包（__init__ 为空，安全）
import ap3_qubo.physical_params  # noqa: E402  真实常量模块（纯 dataclasses/typing）

# experiments/__init__ 会级联导入 ablation/comparison/nsga2 → kaiwu/deap
_stub_pkg("ap3_qubo.experiments")

# exploration.pareto_zoom → qubo.builder → kaiwu（模块级 import）
_stub_pkg("ap3_qubo.exploration")
_pz_stub = types.ModuleType("ap3_qubo.exploration.pareto_zoom")


class _ParetoZoomPlaceholder:
    """占位类：单元测试不实例化；集成测试时替换为 FakeParetoZoom。"""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError("ParetoZoom placeholder — 测试中应被 Fake 替换")


_pz_stub.ParetoZoom = _ParetoZoomPlaceholder
sys.modules["ap3_qubo.exploration.pareto_zoom"] = _pz_stub

# statistics.reporting → hypothesis_tests → scipy（未安装）
_stub_pkg("ap3_qubo.statistics")
_rep_stub = types.ModuleType("ap3_qubo.statistics.reporting")


class ExperimentStats:  # noqa: D401 — 占位即可，sensitivity 未实际调用
    """占位类。"""


_rep_stub.ExperimentStats = ExperimentStats
sys.modules["ap3_qubo.statistics.reporting"] = _rep_stub

# validation.hypervolume 本身仅需 numpy，但其包 __init__ 牵一发动全身，统一打桩
_stub_pkg("ap3_qubo.validation")
_hv_stub = types.ModuleType("ap3_qubo.validation.hypervolume")


class HypervolumeCalculator:  # 占位：集成测试用空目标矩阵走 _compute_hv 早退分支
    pass


def set_unified_reference(*args, **kwargs):  # P0-5 新增导出，占位即可
    return np.array([1.0, 1.0, 1.0])


_hv_stub.HypervolumeCalculator = HypervolumeCalculator
_hv_stub.set_unified_reference = set_unified_reference
sys.modules["ap3_qubo.validation.hypervolume"] = _hv_stub

# =============================================================================
# 用 importlib 加载真实的 sensitivity.py（被测对象）
# =============================================================================

_spec = importlib.util.spec_from_file_location(
    "ap3_qubo.experiments.sensitivity",
    SRC / "ap3_qubo" / "experiments" / "sensitivity.py",
)
sensitivity = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = sensitivity
_spec.loader.exec_module(sensitivity)

SensitivityAnalyzer = sensitivity.SensitivityAnalyzer

# =============================================================================
# 人工成分数据构造
# =============================================================================

ELEMS = ["Al", "Co", "Cr", "Fe", "Ni", "C"]


def make_frac(al, co, cr, fe, ni, c):
    """构造成分字典 {元素: at%}。"""
    return dict(zip(ELEMS, [al, co, cr, fe, ni, c]))


# 基准 TOP10：10 个互不相同的解（成分和均 = 100 at%）
BASELINE_TOP10 = [make_frac(20 + i, 20, 20, 20, 19 - i, 1.0) for i in range(10)]

TOL = 0.5  # 默认容差 (at%)，与 SensitivityAnalyzer 默认值一致

_passed = 0


def check(name: str, cond: bool) -> None:
    global _passed
    assert cond, f"FAIL: {name}"
    _passed += 1
    print(f"  PASS  {name}")


# =============================================================================
# 用例 1：全同 → 1.0
# =============================================================================

print("[1] 全同快照 → 重叠率应为 1.0")
identical = [dict(f) for f in BASELINE_TOP10]
ov = SensitivityAnalyzer._top10_pair_overlap(identical, BASELINE_TOP10, TOL)
check("全同重叠率 == 1.0", ov == 1.0)

# =============================================================================
# 用例 2：全不同 → 0.0
# =============================================================================

print("[2] 全不同快照 → 重叠率应为 0.0")
disjoint = [make_frac(35, 15, 15, 15, 18, 2.0) for _ in range(10)]
ov = SensitivityAnalyzer._top10_pair_overlap(disjoint, BASELINE_TOP10, TOL)
check("全不同重叠率 == 0.0", ov == 0.0)

# =============================================================================
# 用例 3：部分重叠 (4/10) → 0.4
# =============================================================================

print("[3] 部分重叠（基准 10 解中 4 个有容差内匹配）→ 重叠率应为 0.4")
partial = (
    # 前 4 个与基准前 4 解各元素差 0.1 at%（< 0.5 容差）→ 匹配
    [make_frac(20 + i + 0.1, 20, 20, 20, 19 - i - 0.1, 1.0) for i in range(4)]
    # 后 6 个远离基准所有解 → 不匹配
    + [make_frac(35, 15, 15, 15, 18, 2.0) for _ in range(6)]
)
ov = SensitivityAnalyzer._top10_pair_overlap(partial, BASELINE_TOP10, TOL)
check("部分重叠率 == 0.4", abs(ov - 0.4) < 1e-9)

# =============================================================================
# 用例 4：容差边界（严格 <）
# =============================================================================

print("[4] 容差边界：差 = 0.5 at% 不匹配；差 = 0.49 at% 匹配")
a = make_frac(20, 20, 20, 20, 19, 1.0)
b_edge = make_frac(20.5, 20, 20, 20, 18.5, 1.0)   # Al/Ni 差恰为 0.5
b_in = make_frac(20.49, 20, 20, 20, 18.51, 1.0)   # 差 0.49 < 0.5
check("差 0.5 at% → 不匹配", not SensitivityAnalyzer._compositions_match(a, b_edge, TOL))
check("差 0.49 at% → 匹配", SensitivityAnalyzer._compositions_match(a, b_in, TOL))
check("缺失元素按 0.0 处理", SensitivityAnalyzer._compositions_match(
    {"Al": 20.0}, {"Al": 20.1, "C": 0.0}, TOL))

# =============================================================================
# 用例 5：一对一匹配防重复计数
# =============================================================================

print("[5] 基准两个相同解 vs 候选一个匹配解 → 重叠率应为 1/2（禁止重复匹配）")
base_two = [make_frac(20, 20, 20, 20, 19, 1.0) for _ in range(2)]
cand_one = [make_frac(20.1, 20, 20, 20, 18.9, 1.0)]
ov = SensitivityAnalyzer._top10_pair_overlap(cand_one, base_two, TOL)
check("一对一匹配重叠率 == 0.5", abs(ov - 0.5) < 1e-9)

# =============================================================================
# 用例 6：跨重复均值
# =============================================================================

print("[6] 跨重复均值：候选 1 次重复 × 基准 2 次重复（1.0 与 0.0）→ 0.5")
snaps_a = [[dict(f) for f in BASELINE_TOP10]]
snaps_b = [
    [dict(f) for f in BASELINE_TOP10],                       # 与候选全同 → 1.0
    [make_frac(35, 15, 15, 15, 18, 2.0) for _ in range(10)],  # 与候选全不同 → 0.0
]
mean_ov = SensitivityAnalyzer._mean_top10_overlap(snaps_a, snaps_b, TOL)
check("跨重复均值 == 0.5", abs(mean_ov - 0.5) < 1e-9)
check("空快照均值 == 0.0",
      SensitivityAnalyzer._mean_top10_overlap([], snaps_b, TOL) == 0.0)

# =============================================================================
# 用例 7：集成 —— FakeParetoZoom + 人工 archive 端到端 run()/report()
# =============================================================================

print("[7] 集成：FakeParetoZoom 端到端（基准→1.0，γ=0.1→0.4）")


class FakeRecord:
    """模拟 SolutionRecord，仅携带 fractions（archive.get_top_k 的最小接口）。"""

    def __init__(self, fractions):
        self.fractions = fractions


class FakeArchive:
    """模拟 Archive：提供 get_top_k / get_objective_matrix 最小接口。"""

    def __init__(self, top10):
        self._top10 = top10

    def get_top_k(self, k=10):
        return self._top10[:k]

    def get_objective_matrix(self):
        # 空矩阵 → sensitivity._compute_hv 早退返回 0.0，避开 HV 依赖
        return np.empty((0, 3))


class FakeParetoZoom:
    """按 gamma_discount 返回确定性的预构 archive。"""

    def __init__(self, gamma_discount=None, seed=None):
        self._gamma = gamma_discount

    def run(self):
        if self._gamma == 0.25:
            top = [FakeRecord(dict(f)) for f in BASELINE_TOP10]
        else:
            # 非基准 γ：4 个与基准容差内一致 + 6 个完全不同
            top = [FakeRecord(dict(f)) for f in partial]
        return FakeArchive(top), []


sensitivity.ParetoZoom = FakeParetoZoom  # 替换模块内引用（最小侵入、仅测试侧）

analyzer = sensitivity.SensitivityAnalyzer(gamma_values=[0.25, 0.1])
results = analyzer.run(n_repetitions=2, seed=42)

check("基准 γ=0.25 top10_overlap == 1.0", results[0.25].top10_overlap == 1.0)
check("γ=0.1 top10_overlap == 0.4", abs(results[0.1].top10_overlap - 0.4) < 1e-9)
check("快照已采集（2 次重复 × 10 解）",
      len(results[0.1].top10_snapshots) == 2
      and len(results[0.1].top10_snapshots[0]) == 10)

report_md = analyzer.report(results)
check("报告含 TOP10 重叠率列", "TOP10 重叠率" in report_md)
check("报告数值名实相符（含 40%）", "40%" in report_md)

# 反向校验：修复后的指标不再随 HV 造假 —— 本集成中所有 HV 均为 0，
# 旧实现会得到 0/除零保护值，新实现给出真实成分重叠率 0.4。
check("HV 全 0 时重叠率仍为真实值 0.4（与 HV 脱钩）",
      results[0.1].hv_mean == 0.0 and abs(results[0.1].top10_overlap - 0.4) < 1e-9)

# =============================================================================
# 汇总
# =============================================================================

print()
print(f"All {_passed} checks PASSED - P0-10 fix verified")
