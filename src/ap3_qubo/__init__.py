"""
AP³-QUBO: Adaptive Preference-Penalty Pareto QUBO Framework.

面向 CIM 光量子计算机的高熵合金 Al-Co-Cr-Fe-Ni-C 六元成分多目标优化框架。
将成分空间编码为 38 比特 QUBO，以混合焓 ΔH_mix、密度 ρ、成本为三目标，
通过 PenaltyFlex 自适应罚函数与 ParetoZoom 动态加密探索求解 Pareto 前沿。

核心入口:
    >>> from ap3_qubo.exploration import ParetoZoom
    >>> pz = ParetoZoom()
    >>> archive, rounds = pz.run()
"""

from .encoding import (
    Composition,
    PrecisionSplitEncoder,
    PrecisionSplitDecoder,
    VariableMapper,
)
from .objectives import (
    MixingEnthalpy,
    VegardDensity,
    WeightedCost,
    PhysicalPriorNormalizer,
)
from .solver import (
    AbstractSolver,
    Solution,
    SolverResult,
    KaiwuSolver,
)
from .exploration import ParetoZoom, Archive
from .qubo import QUBOBuilder, QUBOMatrix
from .physical_params import (
    EncodingParams,
    MiedemaParams,
    ConstraintParams,
)

__all__ = [
    # encoding
    "Composition",
    "PrecisionSplitEncoder",
    "PrecisionSplitDecoder",
    "VariableMapper",
    # objectives
    "MixingEnthalpy",
    "VegardDensity",
    "WeightedCost",
    "PhysicalPriorNormalizer",
    # solver
    "AbstractSolver",
    "Solution",
    "SolverResult",
    "KaiwuSolver",
    # exploration
    "ParetoZoom",
    "Archive",
    # qubo
    "QUBOBuilder",
    "QUBOMatrix",
    # physical params
    "EncodingParams",
    "MiedemaParams",
    "ConstraintParams",
]
