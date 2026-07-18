"""
实验矩阵模块。

六项实验:
  - 实验 0: 消融实验（量化三创新各自贡献）
  - 实验 1: PrecisionSplit vs 统一编码
  - 实验 2: PenaltyFlex vs 固定 λ / Grid-Search / Linear
  - 实验 3: ParetoZoom vs 均匀网格 / NSGA-II / Random
  - 实验 4: γ 敏感性分析
  - 实验 5: Pycalphad + 文献回测 + 负对照

执行顺序: 实验0 → 实验2 → 实验3 → 实验1 → 实验5 → 实验4
"""

from .ablation import AblationRunner, AblationResult
from .comparison import (
    compare_encoding,
    compare_penalty,
    compare_exploration,
)
from .sensitivity import SensitivityAnalyzer, SensitivityResult
from .nsga2_baseline import NSGA2Optimizer

__all__ = [
    "AblationRunner",
    "AblationResult",
    "compare_encoding",
    "compare_penalty",
    "compare_exploration",
    "SensitivityAnalyzer",
    "SensitivityResult",
    "NSGA2Optimizer",
]
