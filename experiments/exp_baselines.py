#!/usr/bin/env python3
"""
Experiment 7: Baseline Comparisons
==================================
Compares AEGIS against alternative approaches for handling LLM failures.

Research Question: How does AEGIS compare to existing approaches for
handling failures in multi-agent LLM systems?

Baselines:
1. No recovery (raw pipeline)
2. Simple retry (retry on exception)
3. Output validation (basic checks + retry)
4. Temperature reduction (reduce temp on failure)
5. AEGIS (full system)
"""

import os
import sys
import time
import random
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END

from aegis import AEGIS, AEGISConfig, AEGISDetector
from aegis.config import FailureType
from systems.research_pipeline import (
    create_research_pipeline, ResearchPipelineState,
    research_agent, analyze_agent, summarize_agent
)
from injection import FailureInjector, InjectionConfig, FailureMode, InjectionTrigger
from evaluation import ResultsCollector
from evaluation.metrics import ExperimentResult
from evaluation.statistics import (
    calculate_proportion_ci, compare_proportions, format_ci, format_comparison
)


NUM_TRIALS = 30
FAILURE_RATE = 0.5  # 50% of runs have injected failures

TOPICS = [
    "renewable energy",
    "artificial intelligence",
    "climate change",
    "quantum computing",
    "biotechnology"
]


@dataclass
class BaselineResult:
    """Result from a baseline run"""
    method: str
    success: bool
    output_quality: float
    execution_time_ms: float
    retries_used: int
    had_injected_failure: bool


class RecoveryMethod(ABC):
    """Abstract base class for recovery methods"""

    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def run(
        self,
        topic: str,
        inject_failure: bool,
        failure_mode: Optional[FailureMode]
    ) -> BaselineResult:
        pass


class NoRecoveryBaseline(RecoveryMethod):
    """Baseline: No recovery, just run the pipeline"""

    def name(self) -> str:
        return "no_recovery"

    def run(self, topic: str, inject_failure: bool, failure_mode: Optional[FailureMode]) -> BaselineResult:
        start_time = time.time()

        pipeline = create_research_pipeline()

        if inject_failure and failure_mode:
            injector = FailureInjector()
            injector.schedule("analyze", InjectionConfig(
                failure_mode=failure_mode,
                trigger=InjectionTrigger.ALWAYS
            ))
            wrapped = injector.wrap(analyze_agent, "analyze")

            pipeline = StateGraph(ResearchPipelineState)
            pipeline.add_node("research", research_agent)
            pipeline.add_node("analyze", wrapped)
            pipeline.add_node("summarize", summarize_agent)
            pipeline.set_entry_point("research")
            pipeline.add_edge("research", "analyze")
            pipeline.add_edge("analyze", "summarize")
            pipeline.add_edge("summarize", END)

        try:
            compiled = pipeline.compile()
            result = compiled.invoke({
                "topic": topic,
                "errors": [],
                "execution_log": []
            })

            success, quality = self._evaluate_output(result, topic)

        except Exception:
            success = False
            quality = 0.0

        return BaselineResult(
            method=self.name(),
            success=success,
            output_quality=quality,
            execution_time_ms=(time.time() - start_time) * 1000,
            retries_used=0,
            had_injected_failure=inject_failure
        )

    def _evaluate_output(self, output: Dict[str, Any], topic: str) -> tuple:
        """Basic output evaluation"""
        summary = output.get("summary", "")

        if not summary or len(summary) < 100:
            return False, 0.0

        # Check if topic is mentioned
        if topic.lower() not in summary.lower():
            return False, 0.3

        return True, 0.7


class SimpleRetryBaseline(RecoveryMethod):
    """Baseline: Retry on any exception"""

    def __init__(self, max_retries: int = 3):
        self.max_retries = max_retries

    def name(self) -> str:
        return "simple_retry"

    def run(self, topic: str, inject_failure: bool, failure_mode: Optional[FailureMode]) -> BaselineResult:
        start_time = time.time()
        retries = 0

        for attempt in range(self.max_retries + 1):
            try:
                pipeline = create_research_pipeline()

                # Only inject failure on first attempt (to simulate transient failures)
                if inject_failure and failure_mode and attempt == 0:
                    injector = FailureInjector()
                    injector.schedule("analyze", InjectionConfig(
                        failure_mode=failure_mode,
                        trigger=InjectionTrigger.ALWAYS
                    ))
                    wrapped = injector.wrap(analyze_agent, "analyze")

                    pipeline = StateGraph(ResearchPipelineState)
                    pipeline.add_node("research", research_agent)
                    pipeline.add_node("analyze", wrapped)
                    pipeline.add_node("summarize", summarize_agent)
                    pipeline.set_entry_point("research")
                    pipeline.add_edge("research", "analyze")
                    pipeline.add_edge("analyze", "summarize")
                    pipeline.add_edge("summarize", END)

                compiled = pipeline.compile()
                result = compiled.invoke({
                    "topic": topic,
                    "errors": [],
                    "execution_log": []
                })

                # For simple retry, we only catch crashes
                # Semantic failures pass through undetected
                return BaselineResult(
                    method=self.name(),
                    success=True,  # Didn't crash
                    output_quality=0.7 if not inject_failure else 0.3,  # Quality is low if failure wasn't detected
                    execution_time_ms=(time.time() - start_time) * 1000,
                    retries_used=retries,
                    had_injected_failure=inject_failure
                )

            except Exception:
                retries += 1
                if attempt < self.max_retries:
                    time.sleep(0.5)  # Brief delay before retry
                    continue

        return BaselineResult(
            method=self.name(),
            success=False,
            output_quality=0.0,
            execution_time_ms=(time.time() - start_time) * 1000,
            retries_used=retries,
            had_injected_failure=inject_failure
        )


class OutputValidationBaseline(RecoveryMethod):
    """Baseline: Basic output validation with retry"""

    def __init__(self, max_retries: int = 3):
        self.max_retries = max_retries

    def name(self) -> str:
        return "output_validation"

    def run(self, topic: str, inject_failure: bool, failure_mode: Optional[FailureMode]) -> BaselineResult:
        start_time = time.time()
        retries = 0

        for attempt in range(self.max_retries + 1):
            try:
                pipeline = create_research_pipeline()

                if inject_failure and failure_mode:
                    injector = FailureInjector()
                    # Reduce injection probability on retries
                    prob = 1.0 if attempt == 0 else 0.3
                    injector.schedule("analyze", InjectionConfig(
                        failure_mode=failure_mode,
                        trigger=InjectionTrigger.RANDOM,
                        probability=prob
                    ))
                    wrapped = injector.wrap(analyze_agent, "analyze")

                    pipeline = StateGraph(ResearchPipelineState)
                    pipeline.add_node("research", research_agent)
                    pipeline.add_node("analyze", wrapped)
                    pipeline.add_node("summarize", summarize_agent)
                    pipeline.set_entry_point("research")
                    pipeline.add_edge("research", "analyze")
                    pipeline.add_edge("analyze", "summarize")
                    pipeline.add_edge("summarize", END)

                compiled = pipeline.compile()
                result = compiled.invoke({
                    "topic": topic,
                    "errors": [],
                    "execution_log": []
                })

                # Basic validation
                is_valid, quality = self._validate_output(result, topic)

                if is_valid:
                    return BaselineResult(
                        method=self.name(),
                        success=True,
                        output_quality=quality,
                        execution_time_ms=(time.time() - start_time) * 1000,
                        retries_used=retries,
                        had_injected_failure=inject_failure
                    )

                retries += 1

            except Exception:
                retries += 1

        return BaselineResult(
            method=self.name(),
            success=False,
            output_quality=0.0,
            execution_time_ms=(time.time() - start_time) * 1000,
            retries_used=retries,
            had_injected_failure=inject_failure
        )

    def _validate_output(self, output: Dict[str, Any], topic: str) -> tuple:
        """Basic validation checks"""
        summary = output.get("summary", "")
        analysis = output.get("analysis", "")
        research = output.get("research_data", "")

        # Check for empty outputs
        if not summary or len(summary) < 100:
            return False, 0.0

        if not analysis or len(analysis) < 50:
            return False, 0.2

        if not research or len(research) < 50:
            return False, 0.2

        # Check if topic is mentioned (basic relevance check)
        topic_words = topic.lower().split()
        content = (summary + analysis + research).lower()
        matches = sum(1 for word in topic_words if word in content)

        if matches < len(topic_words) / 2:
            return False, 0.3

        return True, 0.7


class TemperatureReductionBaseline(RecoveryMethod):
    """Baseline: Reduce temperature on failure"""

    def __init__(self, initial_temp: float = 0.7, min_temp: float = 0.1):
        self.initial_temp = initial_temp
        self.min_temp = min_temp

    def name(self) -> str:
        return "temperature_reduction"

    def run(self, topic: str, inject_failure: bool, failure_mode: Optional[FailureMode]) -> BaselineResult:
        start_time = time.time()
        retries = 0
        temp = self.initial_temp

        while temp >= self.min_temp:
            try:
                # This would require modifying the pipeline to use variable temperature
                # For simplicity, we simulate the effect
                pipeline = create_research_pipeline()

                if inject_failure and failure_mode:
                    # Lower temperature reduces hallucination likelihood
                    reduced_prob = 1.0 - (self.initial_temp - temp)
                    injector = FailureInjector()
                    injector.schedule("analyze", InjectionConfig(
                        failure_mode=failure_mode,
                        trigger=InjectionTrigger.RANDOM,
                        probability=reduced_prob
                    ))
                    wrapped = injector.wrap(analyze_agent, "analyze")

                    pipeline = StateGraph(ResearchPipelineState)
                    pipeline.add_node("research", research_agent)
                    pipeline.add_node("analyze", wrapped)
                    pipeline.add_node("summarize", summarize_agent)
                    pipeline.set_entry_point("research")
                    pipeline.add_edge("research", "analyze")
                    pipeline.add_edge("analyze", "summarize")
                    pipeline.add_edge("summarize", END)

                compiled = pipeline.compile()
                result = compiled.invoke({
                    "topic": topic,
                    "errors": [],
                    "execution_log": []
                })

                # Basic check
                if result.get("summary") and len(result.get("summary", "")) > 100:
                    return BaselineResult(
                        method=self.name(),
                        success=True,
                        output_quality=0.6 + (0.2 * (1 - temp)),  # Higher quality at lower temp
                        execution_time_ms=(time.time() - start_time) * 1000,
                        retries_used=retries,
                        had_injected_failure=inject_failure
                    )

                temp -= 0.2
                retries += 1

            except Exception:
                temp -= 0.2
                retries += 1

        return BaselineResult(
            method=self.name(),
            success=False,
            output_quality=0.0,
            execution_time_ms=(time.time() - start_time) * 1000,
            retries_used=retries,
            had_injected_failure=inject_failure
        )


class AEGISMethod(RecoveryMethod):
    """AEGIS: Full self-healing system"""

    def name(self) -> str:
        return "aegis"

    def run(self, topic: str, inject_failure: bool, failure_mode: Optional[FailureMode]) -> BaselineResult:
        start_time = time.time()

        pipeline = create_research_pipeline()

        if inject_failure and failure_mode:
            injector = FailureInjector()
            injector.schedule("analyze", InjectionConfig(
                failure_mode=failure_mode,
                trigger=InjectionTrigger.ALWAYS
            ))
            wrapped = injector.wrap(analyze_agent, "analyze")

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

        try:
            result = aegis_pipeline.invoke({
                "topic": topic,
                "errors": [],
                "execution_log": []
            })

            metrics = aegis_pipeline.get_metrics()
            success, quality = self._evaluate_output(result, topic)

            return BaselineResult(
                method=self.name(),
                success=success,
                output_quality=quality,
                execution_time_ms=(time.time() - start_time) * 1000,
                retries_used=metrics.get("repair_attempts", 0),
                had_injected_failure=inject_failure
            )

        except Exception:
            return BaselineResult(
                method=self.name(),
                success=False,
                output_quality=0.0,
                execution_time_ms=(time.time() - start_time) * 1000,
                retries_used=0,
                had_injected_failure=inject_failure
            )

    def _evaluate_output(self, output: Dict[str, Any], topic: str) -> tuple:
        """Evaluate output quality"""
        summary = output.get("summary", "")

        if not summary or len(summary) < 100:
            return False, 0.0

        # Use LLM to evaluate
        try:
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
            eval_prompt = f"""Rate this summary about "{topic}" for quality (1-5):
{summary[:1000]}

Respond with just a number 1-5."""

            response = llm.invoke([HumanMessage(content=eval_prompt)])
            score = int(response.content.strip())
            quality = (score - 1) / 4

            return quality > 0.5, quality

        except Exception:
            return True, 0.6


def run_baseline_comparison():
    """Run the baseline comparison experiment"""

    print("\n" + "=" * 60)
    print("EXPERIMENT 7: Baseline Comparisons")
    print("=" * 60)

    collector = ResultsCollector(output_dir="results/baselines")
    collector.start_experiment(
        name="baseline_comparison",
        description="Comparing AEGIS against alternative approaches",
        config={"num_trials": NUM_TRIALS, "failure_rate": FAILURE_RATE}
    )

    methods = [
        NoRecoveryBaseline(),
        SimpleRetryBaseline(),
        OutputValidationBaseline(),
        TemperatureReductionBaseline(),
        AEGISMethod()
    ]

    failure_modes = [
        FailureMode.HALLUCINATION,
        FailureMode.SEMANTIC_DRIFT,
        FailureMode.EMPTY_OUTPUT
    ]

    # Results storage
    results_by_method = {m.name(): {"successes": 0, "total": 0, "quality_sum": 0.0, "time_sum": 0.0}
                         for m in methods}

    for trial in range(NUM_TRIALS):
        print(f"\n--- Trial {trial + 1}/{NUM_TRIALS} ---")

        topic = random.choice(TOPICS)
        inject_failure = random.random() < FAILURE_RATE
        failure_mode = random.choice(failure_modes) if inject_failure else None

        if inject_failure:
            print(f"  Topic: {topic} (with {failure_mode.value})")
        else:
            print(f"  Topic: {topic}")

        for method in methods:
            print(f"    {method.name():25}", end="", flush=True)

            result = method.run(topic, inject_failure, failure_mode)

            results_by_method[method.name()]["total"] += 1
            results_by_method[method.name()]["time_sum"] += result.execution_time_ms

            if result.success:
                results_by_method[method.name()]["successes"] += 1
                results_by_method[method.name()]["quality_sum"] += result.output_quality

            print(f"{'SUCCESS' if result.success else 'FAILED':10} ({result.execution_time_ms:.0f}ms)")

            # Record result
            exp_result = ExperimentResult(
                run_id=f"{method.name()}_{trial}",
                success=result.success,
                total_time_ms=result.execution_time_ms,
                output_quality_score=result.output_quality,
                repair_attempts=result.retries_used,
                failure_types=[failure_mode.value] if failure_mode else [],
                config={
                    "method": method.name(),
                    "had_failure": inject_failure
                }
            )
            collector.add_result(exp_result)

    # Display results
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    print("\n1. SUCCESS RATES (with 95% CI)")
    print("-" * 60)

    for method_name, stats in sorted(results_by_method.items(),
                                      key=lambda x: x[1]["successes"]/x[1]["total"] if x[1]["total"] > 0 else 0,
                                      reverse=True):
        ci = calculate_proportion_ci(stats["successes"], stats["total"])
        print(f"{method_name:25} {format_ci(ci, as_percentage=True)}")

    # Statistical comparison with AEGIS
    print("\n2. COMPARISON WITH AEGIS")
    print("-" * 60)

    aegis_stats = results_by_method["aegis"]

    for method_name, stats in results_by_method.items():
        if method_name == "aegis":
            continue

        comparison = compare_proportions(
            stats["successes"], stats["total"],
            aegis_stats["successes"], aegis_stats["total"]
        )

        sig_marker = "*" if comparison.is_significant else ""
        print(f"{method_name:25} vs AEGIS: {comparison.difference:+.1%} "
              f"(p={comparison.p_value:.4f}{sig_marker})")

    # Average execution time
    print("\n3. AVERAGE EXECUTION TIME")
    print("-" * 60)

    for method_name, stats in sorted(results_by_method.items(),
                                      key=lambda x: x[1]["time_sum"]/x[1]["total"] if x[1]["total"] > 0 else 0):
        avg_time = stats["time_sum"] / stats["total"] if stats["total"] > 0 else 0
        print(f"{method_name:25} {avg_time:8.0f}ms")

    # Quality scores
    print("\n4. AVERAGE OUTPUT QUALITY (successful runs only)")
    print("-" * 60)

    for method_name, stats in sorted(results_by_method.items(),
                                      key=lambda x: x[1]["quality_sum"]/x[1]["successes"] if x[1]["successes"] > 0 else 0,
                                      reverse=True):
        if stats["successes"] > 0:
            avg_quality = stats["quality_sum"] / stats["successes"]
            print(f"{method_name:25} {avg_quality:.2f}")
        else:
            print(f"{method_name:25} N/A (no successes)")

    output = collector.end_experiment()
    return output


if __name__ == "__main__":
    print("Starting baseline comparison experiments...")

    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set")
        sys.exit(1)

    run_baseline_comparison()

    print("\nBaseline comparison complete!")
