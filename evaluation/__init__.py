"""
AEGIS Evaluation Module
=======================
Metrics collection and analysis for AEGIS experiments.
"""

from .metrics import MetricsCalculator, ExperimentResult
from .collector import ResultsCollector
from .statistics import (
    StatisticalResult,
    ComparisonResult,
    calculate_confidence_interval,
    calculate_proportion_ci,
    compare_conditions,
    compare_proportions,
    format_ci,
    format_comparison,
    bootstrap_ci
)

__all__ = [
    "MetricsCalculator",
    "ExperimentResult",
    "ResultsCollector",
    "StatisticalResult",
    "ComparisonResult",
    "calculate_confidence_interval",
    "calculate_proportion_ci",
    "compare_conditions",
    "compare_proportions",
    "format_ci",
    "format_comparison",
    "bootstrap_ci"
]
