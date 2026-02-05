"""
Statistical Analysis Utilities
==============================
Provides statistical rigor for experiment evaluation including
confidence intervals, significance tests, and effect size calculations.
"""

import math
import statistics
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
from scipy import stats
import numpy as np


@dataclass
class StatisticalResult:
    """Result of statistical analysis"""
    mean: float
    std: float
    ci_lower: float
    ci_upper: float
    n: int
    confidence_level: float = 0.95


@dataclass
class ComparisonResult:
    """Result of comparing two conditions"""
    condition_a: StatisticalResult
    condition_b: StatisticalResult
    difference: float
    p_value: float
    is_significant: bool
    effect_size: float  # Cohen's d
    effect_interpretation: str


def calculate_confidence_interval(
    data: List[float],
    confidence: float = 0.95
) -> StatisticalResult:
    """
    Calculate mean and confidence interval for a dataset.

    Args:
        data: List of numeric values
        confidence: Confidence level (default 0.95 for 95% CI)

    Returns:
        StatisticalResult with mean, std, and CI bounds
    """
    n = len(data)
    if n == 0:
        return StatisticalResult(0, 0, 0, 0, 0, confidence)

    mean = statistics.mean(data)

    if n == 1:
        return StatisticalResult(mean, 0, mean, mean, n, confidence)

    std = statistics.stdev(data)
    se = std / math.sqrt(n)

    # Use t-distribution for small samples
    t_value = stats.t.ppf((1 + confidence) / 2, n - 1)
    margin = t_value * se

    return StatisticalResult(
        mean=mean,
        std=std,
        ci_lower=mean - margin,
        ci_upper=mean + margin,
        n=n,
        confidence_level=confidence
    )


def calculate_proportion_ci(
    successes: int,
    total: int,
    confidence: float = 0.95
) -> StatisticalResult:
    """
    Calculate confidence interval for a proportion (e.g., success rate).
    Uses Wilson score interval for better coverage.

    Args:
        successes: Number of successes
        total: Total number of trials
        confidence: Confidence level

    Returns:
        StatisticalResult with proportion and CI bounds
    """
    if total == 0:
        return StatisticalResult(0, 0, 0, 0, 0, confidence)

    p = successes / total
    z = stats.norm.ppf((1 + confidence) / 2)

    # Wilson score interval
    denominator = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denominator
    spread = z * math.sqrt((p * (1 - p) + z**2 / (4 * total)) / total) / denominator

    return StatisticalResult(
        mean=p,
        std=math.sqrt(p * (1 - p) / total) if total > 0 else 0,
        ci_lower=max(0, center - spread),
        ci_upper=min(1, center + spread),
        n=total,
        confidence_level=confidence
    )


def compare_conditions(
    data_a: List[float],
    data_b: List[float],
    paired: bool = False,
    alpha: float = 0.05
) -> ComparisonResult:
    """
    Compare two conditions using appropriate statistical test.

    Args:
        data_a: Data from condition A
        data_b: Data from condition B
        paired: Whether the data is paired (same subjects)
        alpha: Significance level

    Returns:
        ComparisonResult with test statistics and interpretation
    """
    stats_a = calculate_confidence_interval(data_a)
    stats_b = calculate_confidence_interval(data_b)

    difference = stats_b.mean - stats_a.mean

    # Choose appropriate test
    if paired and len(data_a) == len(data_b):
        statistic, p_value = stats.ttest_rel(data_a, data_b)
    else:
        statistic, p_value = stats.ttest_ind(data_a, data_b)

    # Effect size (Cohen's d)
    pooled_std = math.sqrt(
        ((len(data_a) - 1) * stats_a.std**2 + (len(data_b) - 1) * stats_b.std**2) /
        (len(data_a) + len(data_b) - 2)
    ) if (stats_a.std > 0 or stats_b.std > 0) else 1

    effect_size = difference / pooled_std if pooled_std > 0 else 0

    # Interpret effect size
    abs_effect = abs(effect_size)
    if abs_effect < 0.2:
        effect_interpretation = "negligible"
    elif abs_effect < 0.5:
        effect_interpretation = "small"
    elif abs_effect < 0.8:
        effect_interpretation = "medium"
    else:
        effect_interpretation = "large"

    return ComparisonResult(
        condition_a=stats_a,
        condition_b=stats_b,
        difference=difference,
        p_value=p_value,
        is_significant=p_value < alpha,
        effect_size=effect_size,
        effect_interpretation=effect_interpretation
    )


def compare_proportions(
    successes_a: int,
    total_a: int,
    successes_b: int,
    total_b: int,
    alpha: float = 0.05
) -> ComparisonResult:
    """
    Compare two proportions using chi-square test.

    Args:
        successes_a: Successes in condition A
        total_a: Total in condition A
        successes_b: Successes in condition B
        total_b: Total in condition B
        alpha: Significance level

    Returns:
        ComparisonResult with test statistics
    """
    stats_a = calculate_proportion_ci(successes_a, total_a)
    stats_b = calculate_proportion_ci(successes_b, total_b)

    # Chi-square test
    observed = np.array([
        [successes_a, total_a - successes_a],
        [successes_b, total_b - successes_b]
    ])

    if observed.min() >= 0:
        chi2, p_value, dof, expected = stats.chi2_contingency(observed)
    else:
        p_value = 1.0

    difference = stats_b.mean - stats_a.mean

    # Effect size (Cohen's h for proportions)
    h = 2 * (math.asin(math.sqrt(stats_b.mean)) - math.asin(math.sqrt(stats_a.mean)))

    abs_h = abs(h)
    if abs_h < 0.2:
        effect_interpretation = "negligible"
    elif abs_h < 0.5:
        effect_interpretation = "small"
    elif abs_h < 0.8:
        effect_interpretation = "medium"
    else:
        effect_interpretation = "large"

    return ComparisonResult(
        condition_a=stats_a,
        condition_b=stats_b,
        difference=difference,
        p_value=p_value,
        is_significant=p_value < alpha,
        effect_size=h,
        effect_interpretation=effect_interpretation
    )


def format_ci(result: StatisticalResult, as_percentage: bool = False) -> str:
    """Format confidence interval for display"""
    multiplier = 100 if as_percentage else 1
    suffix = "%" if as_percentage else ""

    return (f"{result.mean * multiplier:.1f}{suffix} "
            f"[{result.ci_lower * multiplier:.1f}, {result.ci_upper * multiplier:.1f}] "
            f"(n={result.n})")


def format_comparison(result: ComparisonResult, as_percentage: bool = False) -> str:
    """Format comparison result for display"""
    multiplier = 100 if as_percentage else 1
    suffix = "%" if as_percentage else ""

    sig_marker = "*" if result.is_significant else ""

    return (f"A: {result.condition_a.mean * multiplier:.1f}{suffix} vs "
            f"B: {result.condition_b.mean * multiplier:.1f}{suffix} "
            f"(diff: {result.difference * multiplier:+.1f}{suffix}, "
            f"p={result.p_value:.4f}{sig_marker}, "
            f"effect: {result.effect_interpretation})")


def bootstrap_ci(
    data: List[float],
    statistic_func=np.mean,
    n_bootstrap: int = 10000,
    confidence: float = 0.95
) -> Tuple[float, float, float]:
    """
    Calculate bootstrap confidence interval.

    Useful for non-normal distributions or small samples.

    Args:
        data: Input data
        statistic_func: Function to compute statistic (default: mean)
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level

    Returns:
        Tuple of (statistic, ci_lower, ci_upper)
    """
    data = np.array(data)
    n = len(data)

    # Bootstrap resampling
    bootstrap_stats = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=n, replace=True)
        bootstrap_stats.append(statistic_func(sample))

    # Calculate percentile CI
    alpha = 1 - confidence
    ci_lower = np.percentile(bootstrap_stats, alpha / 2 * 100)
    ci_upper = np.percentile(bootstrap_stats, (1 - alpha / 2) * 100)

    return statistic_func(data), ci_lower, ci_upper
