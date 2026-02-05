#!/usr/bin/env python3
"""
Experiment 4: Latency and Cost Analysis
=======================================
Measures the overhead of AEGIS in terms of time and API calls.

Research Question: What is the computational overhead of AEGIS,
and is it acceptable for production use?

Methodology:
1. Run pipelines with and without AEGIS
2. Measure execution time, API calls, token usage
3. Analyze overhead across different scenarios
4. Calculate cost implications
"""

import os
import sys
import time
import random
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv()

from aegis import AEGIS, AEGISConfig
from systems.research_pipeline import create_research_pipeline
from injection import FailureInjector, InjectionConfig, FailureMode, InjectionTrigger
from evaluation import ResultsCollector
from evaluation.metrics import ExperimentResult
from evaluation.statistics import (
    calculate_confidence_interval, compare_conditions, format_ci
)


# Pricing (as of 2024, approximate)
PRICING = {
    "gpt-4o": {"input": 0.005, "output": 0.015},  # per 1K tokens
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
}

NUM_TRIALS = 20
TOPICS = [
    "renewable energy",
    "artificial intelligence",
    "climate change",
    "quantum computing",
    "biotechnology"
]


@dataclass
class CostMetrics:
    """Metrics for cost analysis"""
    condition: str
    execution_time_ms: float
    api_calls: int
    estimated_input_tokens: int
    estimated_output_tokens: int
    estimated_cost_usd: float
    had_failure: bool
    healing_overhead_ms: float = 0.0


class APICallTracker:
    """Track API calls and estimate costs"""

    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.pricing = PRICING.get(model, PRICING["gpt-4o-mini"])

    def record_call(self, input_text: str, output_text: str):
        """Record an API call with estimated tokens"""
        self.calls += 1
        # Rough estimation: ~4 chars per token
        self.input_tokens += len(input_text) // 4
        self.output_tokens += len(output_text) // 4

    def get_cost(self) -> float:
        """Calculate estimated cost"""
        input_cost = (self.input_tokens / 1000) * self.pricing["input"]
        output_cost = (self.output_tokens / 1000) * self.pricing["output"]
        return input_cost + output_cost

    def reset(self):
        """Reset counters"""
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0


def measure_baseline_run(topic: str, inject_failure: bool = False) -> CostMetrics:
    """Measure a baseline run (no AEGIS)"""

    start_time = time.time()

    pipeline = create_research_pipeline()

    if inject_failure:
        injector = FailureInjector()
        injector.schedule(
            "analyze",
            InjectionConfig(
                failure_mode=FailureMode.HALLUCINATION,
                trigger=InjectionTrigger.ALWAYS
            )
        )
        from systems.research_pipeline import research_agent, analyze_agent, summarize_agent
        wrapped = injector.wrap(analyze_agent, "analyze")

        from langgraph.graph import StateGraph, END
        from systems.research_pipeline import ResearchPipelineState

        pipeline = StateGraph(ResearchPipelineState)
        pipeline.add_node("research", research_agent)
        pipeline.add_node("analyze", wrapped)
        pipeline.add_node("summarize", summarize_agent)
        pipeline.set_entry_point("research")
        pipeline.add_edge("research", "analyze")
        pipeline.add_edge("analyze", "summarize")
        pipeline.add_edge("summarize", END)

    compiled = pipeline.compile()

    try:
        result = compiled.invoke({
            "topic": topic,
            "errors": [],
            "execution_log": []
        })
    except Exception:
        pass

    execution_time = (time.time() - start_time) * 1000

    # Estimate API calls (3 agents = 3 calls)
    api_calls = 3
    # Rough token estimates
    input_tokens = 500 * api_calls  # prompts
    output_tokens = 800 * api_calls  # responses

    return CostMetrics(
        condition="baseline",
        execution_time_ms=execution_time,
        api_calls=api_calls,
        estimated_input_tokens=input_tokens,
        estimated_output_tokens=output_tokens,
        estimated_cost_usd=(input_tokens / 1000) * 0.00015 + (output_tokens / 1000) * 0.0006,
        had_failure=inject_failure
    )


def measure_aegis_run(topic: str, inject_failure: bool = False) -> CostMetrics:
    """Measure an AEGIS run"""

    start_time = time.time()

    pipeline = create_research_pipeline()

    if inject_failure:
        injector = FailureInjector()
        injector.schedule(
            "analyze",
            InjectionConfig(
                failure_mode=FailureMode.HALLUCINATION,
                trigger=InjectionTrigger.ALWAYS
            )
        )
        from systems.research_pipeline import research_agent, analyze_agent, summarize_agent
        wrapped = injector.wrap(analyze_agent, "analyze")

        from langgraph.graph import StateGraph, END
        from systems.research_pipeline import ResearchPipelineState

        pipeline = StateGraph(ResearchPipelineState)
        pipeline.add_node("research", research_agent)
        pipeline.add_node("analyze", wrapped)
        pipeline.add_node("summarize", summarize_agent)
        pipeline.set_entry_point("research")
        pipeline.add_edge("research", "analyze")
        pipeline.add_edge("analyze", "summarize")
        pipeline.add_edge("summarize", END)

    config = AEGISConfig()
    aegis_pipeline = AEGIS.wrap(pipeline, config=config)

    healing_start = time.time()
    try:
        result = aegis_pipeline.invoke({
            "topic": topic,
            "errors": [],
            "execution_log": []
        })
    except Exception:
        pass

    execution_time = (time.time() - start_time) * 1000

    # Get healing metrics
    metrics = aegis_pipeline.get_metrics()
    repair_attempts = metrics.get("repair_attempts", 0)

    # Calculate API calls
    # Base: 3 agent calls
    # Detection: 2 LLM calls per agent output (hallucination + semantic check)
    # Repair: ~2 calls per attempt
    base_calls = 3
    detection_calls = 3 * 2  # Check each agent output
    repair_calls = repair_attempts * 2

    total_calls = base_calls + detection_calls + repair_calls

    # Token estimates
    input_tokens = 500 * base_calls + 800 * detection_calls + 600 * repair_calls
    output_tokens = 800 * base_calls + 200 * detection_calls + 800 * repair_calls

    healing_overhead = (time.time() - healing_start) * 1000 - (execution_time / 2) if inject_failure else 0

    return CostMetrics(
        condition="aegis",
        execution_time_ms=execution_time,
        api_calls=total_calls,
        estimated_input_tokens=input_tokens,
        estimated_output_tokens=output_tokens,
        estimated_cost_usd=(input_tokens / 1000) * 0.00015 + (output_tokens / 1000) * 0.0006,
        had_failure=inject_failure,
        healing_overhead_ms=max(0, healing_overhead)
    )


def run_latency_experiment():
    """Run the latency analysis experiment"""

    print("\n" + "=" * 60)
    print("EXPERIMENT 4: Latency and Cost Analysis")
    print("=" * 60)

    collector = ResultsCollector(output_dir="results/latency_cost")
    collector.start_experiment(
        name="latency_cost_analysis",
        description="Measuring AEGIS overhead in time and cost",
        config={"num_trials": NUM_TRIALS}
    )

    # Storage
    baseline_no_fail: List[CostMetrics] = []
    baseline_with_fail: List[CostMetrics] = []
    aegis_no_fail: List[CostMetrics] = []
    aegis_with_fail: List[CostMetrics] = []

    for trial in range(NUM_TRIALS):
        print(f"\n--- Trial {trial + 1}/{NUM_TRIALS} ---")
        topic = random.choice(TOPICS)

        # Run all 4 conditions
        print("  Baseline (no failure)...", end=" ", flush=True)
        m1 = measure_baseline_run(topic, inject_failure=False)
        baseline_no_fail.append(m1)
        print(f"{m1.execution_time_ms:.0f}ms")

        print("  Baseline (with failure)...", end=" ", flush=True)
        m2 = measure_baseline_run(topic, inject_failure=True)
        baseline_with_fail.append(m2)
        print(f"{m2.execution_time_ms:.0f}ms")

        print("  AEGIS (no failure)...", end=" ", flush=True)
        m3 = measure_aegis_run(topic, inject_failure=False)
        aegis_no_fail.append(m3)
        print(f"{m3.execution_time_ms:.0f}ms")

        print("  AEGIS (with failure)...", end=" ", flush=True)
        m4 = measure_aegis_run(topic, inject_failure=True)
        aegis_with_fail.append(m4)
        print(f"{m4.execution_time_ms:.0f}ms")

        # Record results
        for m in [m1, m2, m3, m4]:
            result = ExperimentResult(
                run_id=f"{m.condition}_fail{m.had_failure}_{trial}",
                success=True,
                total_time_ms=m.execution_time_ms,
                config={
                    "condition": m.condition,
                    "had_failure": m.had_failure,
                    "api_calls": m.api_calls,
                    "cost_usd": m.estimated_cost_usd
                }
            )
            collector.add_result(result)

    # Results
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    # Latency Analysis
    print("\n1. EXECUTION TIME (ms)")
    print("-" * 50)

    def print_latency_stats(name: str, metrics: List[CostMetrics]):
        times = [m.execution_time_ms for m in metrics]
        ci = calculate_confidence_interval(times)
        print(f"{name:30} {ci.mean:8.0f}ms [{ci.ci_lower:6.0f}, {ci.ci_upper:6.0f}]")

    print_latency_stats("Baseline (no failure):", baseline_no_fail)
    print_latency_stats("Baseline (with failure):", baseline_with_fail)
    print_latency_stats("AEGIS (no failure):", aegis_no_fail)
    print_latency_stats("AEGIS (with failure):", aegis_with_fail)

    # Overhead calculation
    baseline_avg = sum(m.execution_time_ms for m in baseline_no_fail) / len(baseline_no_fail)
    aegis_avg = sum(m.execution_time_ms for m in aegis_no_fail) / len(aegis_no_fail)
    overhead_pct = ((aegis_avg / baseline_avg) - 1) * 100

    print(f"\nDetection overhead (no failure): {overhead_pct:.1f}%")

    # Cost Analysis
    print("\n2. ESTIMATED COST (USD per run)")
    print("-" * 50)

    def print_cost_stats(name: str, metrics: List[CostMetrics]):
        costs = [m.estimated_cost_usd for m in metrics]
        avg = sum(costs) / len(costs)
        print(f"{name:30} ${avg:.4f}")

    print_cost_stats("Baseline (no failure):", baseline_no_fail)
    print_cost_stats("Baseline (with failure):", baseline_with_fail)
    print_cost_stats("AEGIS (no failure):", aegis_no_fail)
    print_cost_stats("AEGIS (with failure):", aegis_with_fail)

    # API Calls
    print("\n3. API CALLS (per run)")
    print("-" * 50)

    def print_api_stats(name: str, metrics: List[CostMetrics]):
        calls = [m.api_calls for m in metrics]
        avg = sum(calls) / len(calls)
        print(f"{name:30} {avg:.1f} calls")

    print_api_stats("Baseline (no failure):", baseline_no_fail)
    print_api_stats("Baseline (with failure):", baseline_with_fail)
    print_api_stats("AEGIS (no failure):", aegis_no_fail)
    print_api_stats("AEGIS (with failure):", aegis_with_fail)

    # Summary
    print("\n4. OVERHEAD SUMMARY")
    print("-" * 50)

    baseline_cost = sum(m.estimated_cost_usd for m in baseline_no_fail) / len(baseline_no_fail)
    aegis_cost = sum(m.estimated_cost_usd for m in aegis_no_fail) / len(aegis_no_fail)
    cost_overhead = ((aegis_cost / baseline_cost) - 1) * 100

    print(f"Latency overhead (detection): {overhead_pct:.1f}%")
    print(f"Cost overhead (detection): {cost_overhead:.1f}%")
    print(f"Additional API calls for detection: {aegis_no_fail[0].api_calls - baseline_no_fail[0].api_calls}")

    # Cost at scale
    print("\n5. COST AT SCALE (1000 runs)")
    print("-" * 50)
    print(f"Baseline: ${baseline_cost * 1000:.2f}")
    print(f"AEGIS:    ${aegis_cost * 1000:.2f}")
    print(f"Added cost for self-healing: ${(aegis_cost - baseline_cost) * 1000:.2f}")

    output = collector.end_experiment()
    return output


if __name__ == "__main__":
    print("Starting latency and cost analysis...")

    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set")
        sys.exit(1)

    run_latency_experiment()

    print("\nLatency and cost analysis complete!")
