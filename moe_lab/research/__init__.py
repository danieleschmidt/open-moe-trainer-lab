"""Research components for novel MoE algorithms and experimental frameworks."""

from .experimental_routers import (
    AdaptiveRouter,
    HierarchicalRouter,
    LearnedSparseRouter,
    DynamicTopKRouter,
    ContextAwareRouter
)

from .baseline_comparisons import (
    BaselineComparison,
    DenseBaseline,
    RouterComparison,
    PerformanceBenchmark
)

from .ablation_studies import (
    AblationStudy,
    RouterAblation,
    ExpertAblation,
    ArchitectureAblation
)

from .experimental_framework import (
    ExperimentRunner,
    ExperimentConfig,
    ResultsAnalyzer,
    StatisticalValidator
)

__all__ = [
    "AdaptiveRouter",
    "HierarchicalRouter", 
    "LearnedSparseRouter",
    "DynamicTopKRouter",
    "ContextAwareRouter",
    "BaselineComparison",
    "DenseBaseline",
    "RouterComparison",
    "PerformanceBenchmark",
    "AblationStudy",
    "RouterAblation",
    "ExpertAblation",
    "ArchitectureAblation",
    "ExperimentRunner",
    "ExperimentConfig",
    "ResultsAnalyzer",
    "StatisticalValidator"
]