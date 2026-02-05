#!/usr/bin/env python3
"""
Experiment 3: End-to-End Pipeline Evaluation
=============================================
Tests complete pipeline success rate with and without AEGIS.

Research Question: Does AEGIS improve end-to-end task success rate
in multi-agent workflows?

Methodology:
1. Run pipeline without AEGIS (baseline)
2. Run pipeline with AEGIS enabled
3. Inject controlled failures at various rates
4. Measure task completion success and output quality
5. Compare with statistical significance
"""

import os
import sys
import time
import random
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from aegis import AEGIS, AEGISConfig
from systems.research_pipeline import create_research_pipeline, ResearchPipelineState
from injection import (
    FailureInjector, InjectionConfig, FailureMode, InjectionTrigger
)
from evaluation import ResultsCollector
from evaluation.metrics import ExperimentResult
from evaluation.statistics import (
    calculate_proportion_ci, compare_proportions, format_ci, format_comparison,
    calculate_confidence_interval
)


# Configuration
NUM_TRIALS = 30  # Per condition for statistical power
FAILURE_INJECTION_RATE = 0.5  # 50% of runs have injected failures
TOPICS = [
    "renewable energy adoption in developing countries",
    "artificial intelligence ethics and regulation",
    "climate change impact on agriculture",
    "quantum computing for drug discovery",
    "sustainable urban transportation",
    "biodiversity conservation strategies",
    "mental health in the digital age",
    "space exploration commercialization",
    "food security and technology",
    "cybersecurity in critical infrastructure"
]


@dataclass
class PipelineRun:
    """Result of a single pipeline run"""
    topic: str
    condition: str  # "baseline" or "aegis"
    success: bool
    had_injected_failure: bool
    failure_type: Optional[str]
    output_quality: float
    total_time_ms: float
    healing_attempts: int
    healing_successes: int
    final_output: Optional[str]
    error_message: Optional[str]


def evaluate_output_quality(
    output: Dict[str, Any],
    topic: str
) -> Tuple[bool, float, List[str]]:
    """
    Evaluate the quality of pipeline output using LLM judge.

    Returns:
        Tuple of (is_successful, quality_score, issues)
    """

    summary = output.get("summary", "")
    research_data = output.get("research_data", "")
    analysis = output.get("analysis", "")

    # Basic checks
    if not summary or len(summary) < 100:
        return False, 0.0, ["Summary missing or too short"]

    if not research_data or len(research_data) < 100:
        return False, 0.3, ["Research data missing or insufficient"]

    if not analysis or len(analysis) < 100:
        return False, 0.4, ["Analysis missing or insufficient"]

    # Use LLM to evaluate quality
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    eval_prompt = f"""Evaluate this research pipeline output about "{topic}".

SUMMARY:
{summary[:1500]}

RESEARCH DATA (excerpt):
{research_data[:1000]}

ANALYSIS (excerpt):
{analysis[:1000]}

Rate the output on these criteria (1-5 each):
1. Relevance: Does it address the topic?
2. Accuracy: Is information factual (no obvious hallucinations)?
3. Completeness: Are key aspects covered?
4. Coherence: Is the output well-structured and logical?
5. Usefulness: Would this be useful for someone researching the topic?

Respond with JSON only:
{{"relevance": 1-5, "accuracy": 1-5, "completeness": 1-5, "coherence": 1-5, "usefulness": 1-5, "overall_success": true/false, "issues": ["list of issues if any"]}}"""

    try:
        response = llm.invoke([HumanMessage(content=eval_prompt)])
        import json

        # Extract JSON from response
        content = response.content
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]

        result = json.loads(content.strip())

        # Calculate average score
        scores = [result.get(k, 3) for k in
                  ["relevance", "accuracy", "completeness", "coherence", "usefulness"]]
        avg_score = sum(scores) / len(scores)
        quality = (avg_score - 1) / 4  # Normalize to 0-1

        return (
            result.get("overall_success", quality > 0.6),
            quality,
            result.get("issues", [])
        )

    except Exception as e:
        # Fallback to basic evaluation
        quality = 0.6 if len(summary) > 200 else 0.3
        return quality > 0.5, quality, [f"LLM evaluation failed: {e}"]


def run_baseline_pipeline(
    topic: str,
    inject_failure: bool = False,
    failure_mode: Optional[FailureMode] = None
) -> PipelineRun:
    """
    Run pipeline without AEGIS (baseline condition).
    """

    start_time = time.time()

    try:
        pipeline = create_research_pipeline()

        # Optionally inject failure
        if inject_failure and failure_mode:
            injector = FailureInjector()
            # Inject into analyze agent (middle of pipeline)
            injector.schedule(
                agent_name="analyze",
                config=InjectionConfig(
                    failure_mode=failure_mode,
                    trigger=InjectionTrigger.ALWAYS
                )
            )

            # Get nodes and wrap them
            from systems.research_pipeline import research_agent, analyze_agent, summarize_agent

            wrapped_analyze = injector.wrap(analyze_agent, "analyze")

            # Rebuild pipeline with wrapped agent
            from langgraph.graph import StateGraph, END
            pipeline = StateGraph(ResearchPipelineState)
            pipeline.add_node("research", research_agent)
            pipeline.add_node("analyze", wrapped_analyze)
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

        total_time = (time.time() - start_time) * 1000

        # Evaluate output
        is_success, quality, issues = evaluate_output_quality(result, topic)

        return PipelineRun(
            topic=topic,
            condition="baseline",
            success=is_success,
            had_injected_failure=inject_failure,
            failure_type=failure_mode.value if failure_mode else None,
            output_quality=quality,
            total_time_ms=total_time,
            healing_attempts=0,
            healing_successes=0,
            final_output=result.get("summary", "")[:500],
            error_message=None if is_success else "; ".join(issues)
        )

    except Exception as e:
        total_time = (time.time() - start_time) * 1000
        return PipelineRun(
            topic=topic,
            condition="baseline",
            success=False,
            had_injected_failure=inject_failure,
            failure_type=failure_mode.value if failure_mode else None,
            output_quality=0.0,
            total_time_ms=total_time,
            healing_attempts=0,
            healing_successes=0,
            final_output=None,
            error_message=str(e)
        )


def run_aegis_pipeline(
    topic: str,
    inject_failure: bool = False,
    failure_mode: Optional[FailureMode] = None
) -> PipelineRun:
    """
    Run pipeline with AEGIS enabled.
    """

    start_time = time.time()

    try:
        pipeline = create_research_pipeline()

        # Optionally inject failure
        if inject_failure and failure_mode:
            injector = FailureInjector()
            injector.schedule(
                agent_name="analyze",
                config=InjectionConfig(
                    failure_mode=failure_mode,
                    trigger=InjectionTrigger.ALWAYS
                )
            )

            from systems.research_pipeline import research_agent, analyze_agent, summarize_agent

            wrapped_analyze = injector.wrap(analyze_agent, "analyze")

            from langgraph.graph import StateGraph, END
            pipeline = StateGraph(ResearchPipelineState)
            pipeline.add_node("research", research_agent)
            pipeline.add_node("analyze", wrapped_analyze)
            pipeline.add_node("summarize", summarize_agent)
            pipeline.set_entry_point("research")
            pipeline.add_edge("research", "analyze")
            pipeline.add_edge("analyze", "summarize")
            pipeline.add_edge("summarize", END)

        # Wrap with AEGIS
        config = AEGISConfig()
        aegis_pipeline = AEGIS.wrap(pipeline, config=config)

        result = aegis_pipeline.invoke({
            "topic": topic,
            "errors": [],
            "execution_log": []
        })

        total_time = (time.time() - start_time) * 1000

        # Get healing metrics
        metrics = aegis_pipeline.get_metrics()
        healing_log = aegis_pipeline.get_healing_log()

        # Evaluate output
        is_success, quality, issues = evaluate_output_quality(result, topic)

        return PipelineRun(
            topic=topic,
            condition="aegis",
            success=is_success,
            had_injected_failure=inject_failure,
            failure_type=failure_mode.value if failure_mode else None,
            output_quality=quality,
            total_time_ms=total_time,
            healing_attempts=metrics.get("repair_attempts", 0),
            healing_successes=metrics.get("repair_successes", 0),
            final_output=result.get("summary", "")[:500],
            error_message=None if is_success else "; ".join(issues)
        )

    except Exception as e:
        total_time = (time.time() - start_time) * 1000
        return PipelineRun(
            topic=topic,
            condition="aegis",
            success=False,
            had_injected_failure=inject_failure,
            failure_type=failure_mode.value if failure_mode else None,
            output_quality=0.0,
            total_time_ms=total_time,
            healing_attempts=0,
            healing_successes=0,
            final_output=None,
            error_message=str(e)
        )


def run_end_to_end_experiment():
    """Run the main end-to-end comparison experiment"""

    print("\n" + "=" * 60)
    print("EXPERIMENT 3: End-to-End Pipeline Evaluation")
    print("=" * 60)

    collector = ResultsCollector(output_dir="results/end_to_end")
    collector.start_experiment(
        name="end_to_end_comparison",
        description="Comparing pipeline success with and without AEGIS",
        config={
            "num_trials": NUM_TRIALS,
            "failure_injection_rate": FAILURE_INJECTION_RATE,
            "topics": len(TOPICS)
        }
    )

    # Storage for results
    baseline_runs: List[PipelineRun] = []
    aegis_runs: List[PipelineRun] = []

    failure_modes = [
        FailureMode.HALLUCINATION,
        FailureMode.SEMANTIC_DRIFT,
        FailureMode.EMPTY_OUTPUT
    ]

    for trial in range(NUM_TRIALS):
        print(f"\n--- Trial {trial + 1}/{NUM_TRIALS} ---")

        topic = random.choice(TOPICS)
        inject_failure = random.random() < FAILURE_INJECTION_RATE
        failure_mode = random.choice(failure_modes) if inject_failure else None

        if inject_failure:
            print(f"  Topic: {topic[:40]}... (with {failure_mode.value} injection)")
        else:
            print(f"  Topic: {topic[:40]}... (no injection)")

        # Run baseline
        print("  Running baseline...", end=" ", flush=True)
        baseline_result = run_baseline_pipeline(topic, inject_failure, failure_mode)
        baseline_runs.append(baseline_result)
        print(f"{'SUCCESS' if baseline_result.success else 'FAILED'}")

        # Run AEGIS
        print("  Running AEGIS...", end=" ", flush=True)
        aegis_result = run_aegis_pipeline(topic, inject_failure, failure_mode)
        aegis_runs.append(aegis_result)
        print(f"{'SUCCESS' if aegis_result.success else 'FAILED'}")

        # Record results
        for run in [baseline_result, aegis_result]:
            result = ExperimentResult(
                run_id=f"{run.condition}_{trial}",
                success=run.success,
                failures_injected=1 if run.had_injected_failure else 0,
                failures_detected=1 if run.had_injected_failure and run.condition == "aegis" else 0,
                repair_attempts=run.healing_attempts,
                repair_successes=run.healing_successes,
                total_time_ms=run.total_time_ms,
                output_quality_score=run.output_quality,
                failure_types=[run.failure_type] if run.failure_type else [],
                config={
                    "condition": run.condition,
                    "topic": run.topic,
                    "had_failure": run.had_injected_failure
                }
            )
            collector.add_result(result)

    # Calculate and display results
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    # Overall success rates
    baseline_successes = sum(1 for r in baseline_runs if r.success)
    aegis_successes = sum(1 for r in aegis_runs if r.success)

    print("\n1. OVERALL SUCCESS RATES")
    print("-" * 40)

    baseline_ci = calculate_proportion_ci(baseline_successes, len(baseline_runs))
    aegis_ci = calculate_proportion_ci(aegis_successes, len(aegis_runs))

    print(f"Baseline: {format_ci(baseline_ci, as_percentage=True)}")
    print(f"AEGIS:    {format_ci(aegis_ci, as_percentage=True)}")

    comparison = compare_proportions(
        baseline_successes, len(baseline_runs),
        aegis_successes, len(aegis_runs)
    )
    print(f"\n{format_comparison(comparison, as_percentage=True)}")
    print(f"Statistically significant: {'YES' if comparison.is_significant else 'NO'}")

    # Success rates when failures were injected
    print("\n2. SUCCESS RATES WITH INJECTED FAILURES")
    print("-" * 40)

    baseline_with_failure = [r for r in baseline_runs if r.had_injected_failure]
    aegis_with_failure = [r for r in aegis_runs if r.had_injected_failure]

    baseline_fail_successes = sum(1 for r in baseline_with_failure if r.success)
    aegis_fail_successes = sum(1 for r in aegis_with_failure if r.success)

    if baseline_with_failure:
        baseline_fail_ci = calculate_proportion_ci(baseline_fail_successes, len(baseline_with_failure))
        aegis_fail_ci = calculate_proportion_ci(aegis_fail_successes, len(aegis_with_failure))

        print(f"Baseline (with failures): {format_ci(baseline_fail_ci, as_percentage=True)}")
        print(f"AEGIS (with failures):    {format_ci(aegis_fail_ci, as_percentage=True)}")

        fail_comparison = compare_proportions(
            baseline_fail_successes, len(baseline_with_failure),
            aegis_fail_successes, len(aegis_with_failure)
        )
        print(f"\nImprovement: {format_comparison(fail_comparison, as_percentage=True)}")

    # Output quality comparison
    print("\n3. OUTPUT QUALITY SCORES")
    print("-" * 40)

    baseline_quality = [r.output_quality for r in baseline_runs if r.success]
    aegis_quality = [r.output_quality for r in aegis_runs if r.success]

    if baseline_quality and aegis_quality:
        baseline_qual_ci = calculate_confidence_interval(baseline_quality)
        aegis_qual_ci = calculate_confidence_interval(aegis_quality)

        print(f"Baseline quality: {baseline_qual_ci.mean:.3f} [{baseline_qual_ci.ci_lower:.3f}, {baseline_qual_ci.ci_upper:.3f}]")
        print(f"AEGIS quality:    {aegis_qual_ci.mean:.3f} [{aegis_qual_ci.ci_lower:.3f}, {aegis_qual_ci.ci_upper:.3f}]")

    # Timing comparison
    print("\n4. EXECUTION TIME")
    print("-" * 40)

    baseline_times = [r.total_time_ms for r in baseline_runs]
    aegis_times = [r.total_time_ms for r in aegis_runs]

    baseline_time_ci = calculate_confidence_interval(baseline_times)
    aegis_time_ci = calculate_confidence_interval(aegis_times)

    print(f"Baseline: {baseline_time_ci.mean:.0f}ms [{baseline_time_ci.ci_lower:.0f}, {baseline_time_ci.ci_upper:.0f}]")
    print(f"AEGIS:    {aegis_time_ci.mean:.0f}ms [{aegis_time_ci.ci_lower:.0f}, {aegis_time_ci.ci_upper:.0f}]")
    print(f"Overhead: {((aegis_time_ci.mean / baseline_time_ci.mean) - 1) * 100:.1f}%")

    # Save results
    output = collector.end_experiment()

    return output


def run_failure_rate_sensitivity():
    """Test AEGIS effectiveness at different failure injection rates"""

    print("\n" + "=" * 60)
    print("FAILURE RATE SENSITIVITY ANALYSIS")
    print("=" * 60)

    failure_rates = [0.0, 0.25, 0.5, 0.75, 1.0]
    trials_per_rate = 10

    results = {}

    for rate in failure_rates:
        print(f"\nTesting with {rate:.0%} failure rate...")

        baseline_successes = 0
        aegis_successes = 0

        for trial in range(trials_per_rate):
            topic = random.choice(TOPICS)
            inject_failure = random.random() < rate
            failure_mode = FailureMode.HALLUCINATION if inject_failure else None

            baseline = run_baseline_pipeline(topic, inject_failure, failure_mode)
            aegis = run_aegis_pipeline(topic, inject_failure, failure_mode)

            if baseline.success:
                baseline_successes += 1
            if aegis.success:
                aegis_successes += 1

        results[rate] = {
            "baseline": baseline_successes / trials_per_rate,
            "aegis": aegis_successes / trials_per_rate,
            "improvement": (aegis_successes - baseline_successes) / trials_per_rate
        }

        print(f"  Baseline: {results[rate]['baseline']:.0%}")
        print(f"  AEGIS: {results[rate]['aegis']:.0%}")
        print(f"  Improvement: {results[rate]['improvement']:+.0%}")

    return results


if __name__ == "__main__":
    print("Starting end-to-end experiments...")

    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set")
        sys.exit(1)

    # Run main experiment
    run_end_to_end_experiment()

    # Run sensitivity analysis
    run_failure_rate_sensitivity()

    print("\nEnd-to-end experiments complete!")
