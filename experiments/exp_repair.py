#!/usr/bin/env python3
"""
Experiment 2: Repair Effectiveness
==================================
Tests AEGIS repair capabilities across different failure types.

Research Question: How effectively can AEGIS repair detected failures
to produce correct outputs?

Methodology:
1. Inject failures of each type
2. Detect the failure
3. Attempt repair
4. Verify the repaired output is correct
5. Measure repair success rate and quality
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv()

from aegis import AEGIS, AEGISConfig, AEGISDetector
from aegis.repair import AEGISRepair
from aegis.state import FailureInfo
from aegis.config import FailureType
from injection import FailureMode, InjectionConfig, InjectionTrigger
from evaluation import ResultsCollector
from evaluation.metrics import ExperimentResult
from evaluation.statistics import (
    calculate_proportion_ci, compare_proportions, format_ci, format_comparison
)

# Configuration
NUM_TRIALS = 30  # Per failure type for statistical significance
FAILURE_MODES = [
    (FailureMode.HALLUCINATION, FailureType.HALLUCINATION),
    (FailureMode.SEMANTIC_DRIFT, FailureType.SEMANTIC_DRIFT),
    (FailureMode.EMPTY_OUTPUT, FailureType.QUALITY),
    (FailureMode.MALFORMED_OUTPUT, FailureType.FORMAT_ERROR),
]


@dataclass
class RepairTestCase:
    """A test case for repair evaluation"""
    failure_mode: FailureMode
    failure_type: FailureType
    original_task: str
    input_state: Dict[str, Any]
    failed_output: Any
    expected_characteristics: List[str]  # What the repaired output should have


def create_test_cases() -> List[RepairTestCase]:
    """Create test cases for repair evaluation"""

    test_cases = []
    topics = [
        "renewable energy trends",
        "artificial intelligence in healthcare",
        "climate change mitigation strategies",
        "quantum computing applications",
        "sustainable agriculture practices"
    ]

    for topic in topics:
        # Hallucination case
        test_cases.append(RepairTestCase(
            failure_mode=FailureMode.HALLUCINATION,
            failure_type=FailureType.HALLUCINATION,
            original_task=f"Research and summarize key facts about {topic}",
            input_state={"topic": topic},
            failed_output=f"""According to the prestigious Thornberry Institute (founded 2019),
            recent studies show that {topic} has improved outcomes by exactly 847.3% globally.
            Dr. Margaret Fictitious, lead researcher, states: "These unprecedented findings
            will revolutionize everything." The Global Commission has allocated $4.7 trillion
            for further investigation based on these groundbreaking results.""",
            expected_characteristics=[
                "factual information only",
                "no made-up statistics",
                "no fictional sources",
                "addresses the topic"
            ]
        ))

        # Semantic drift case
        test_cases.append(RepairTestCase(
            failure_mode=FailureMode.SEMANTIC_DRIFT,
            failure_type=FailureType.SEMANTIC_DRIFT,
            original_task=f"Analyze the impact of {topic}",
            input_state={"topic": topic},
            failed_output="""The history of cheese-making dates back thousands of years.
            Ancient civilizations in Mesopotamia discovered that milk could be
            preserved through fermentation. Today, France produces over 400 varieties
            of cheese, each with unique characteristics and aging requirements.""",
            expected_characteristics=[
                "addresses the original topic",
                "relevant to the task",
                "analytical content"
            ]
        ))

        # Empty output case
        test_cases.append(RepairTestCase(
            failure_mode=FailureMode.EMPTY_OUTPUT,
            failure_type=FailureType.QUALITY,
            original_task=f"Provide a summary of {topic}",
            input_state={"topic": topic},
            failed_output="",
            expected_characteristics=[
                "non-empty response",
                "substantive content",
                "addresses the topic"
            ]
        ))

        # Malformed output case
        test_cases.append(RepairTestCase(
            failure_mode=FailureMode.MALFORMED_OUTPUT,
            failure_type=FailureType.FORMAT_ERROR,
            original_task=f"Provide analysis of {topic} in a structured format",
            input_state={"topic": topic},
            failed_output="```json\n{\"incomplete\": true, \"data\":",
            expected_characteristics=[
                "complete response",
                "readable format",
                "addresses the topic"
            ]
        ))

    return test_cases


def verify_repair_quality(
    repaired_output: Any,
    test_case: RepairTestCase,
    detector: AEGISDetector
) -> Dict[str, Any]:
    """
    Verify the quality of a repaired output.

    Returns dict with:
    - is_valid: bool - whether repair produced valid output
    - quality_score: float - 0-1 quality rating
    - issues: list - any remaining issues
    """

    if repaired_output is None or repaired_output == "":
        return {
            "is_valid": False,
            "quality_score": 0.0,
            "issues": ["Repair produced empty output"]
        }

    # Check if the repaired output still has failures
    recheck = detector.detect(
        agent_name="repair_verification",
        agent_output=repaired_output,
        original_task=test_case.original_task,
        input_state=test_case.input_state
    )

    if recheck.is_failure:
        return {
            "is_valid": False,
            "quality_score": 0.3,
            "issues": [f"Repaired output still has failure: {recheck.failure_type}"]
        }

    # Basic quality checks
    output_str = str(repaired_output)
    issues = []
    quality_score = 1.0

    # Check length
    if len(output_str) < 50:
        issues.append("Output too short")
        quality_score -= 0.2

    # Check if topic is mentioned
    topic = test_case.input_state.get("topic", "")
    if topic and topic.lower() not in output_str.lower():
        # Check if at least some topic words are present
        topic_words = topic.lower().split()
        matches = sum(1 for word in topic_words if word in output_str.lower())
        if matches < len(topic_words) / 2:
            issues.append("Topic not adequately addressed")
            quality_score -= 0.3

    return {
        "is_valid": len(issues) == 0,
        "quality_score": max(0, quality_score),
        "issues": issues
    }


def run_repair_experiment():
    """Run the repair effectiveness experiment"""

    print("\n" + "=" * 60)
    print("EXPERIMENT 2: Repair Effectiveness")
    print("=" * 60)

    # Initialize
    config = AEGISConfig()
    detector = AEGISDetector(config)
    repair = AEGISRepair(config, detector)
    collector = ResultsCollector(output_dir="results/repair")

    collector.start_experiment(
        name="repair_effectiveness",
        description="Testing AEGIS repair capabilities across failure types",
        config={
            "num_trials": NUM_TRIALS,
            "failure_types": [f[0].value for f in FAILURE_MODES]
        }
    )

    # Create test cases
    all_test_cases = create_test_cases()

    # Results storage
    results_by_type = {ft.value: {"attempts": 0, "successes": 0, "quality_scores": []}
                       for _, ft in FAILURE_MODES}

    # Run trials
    for trial in range(NUM_TRIALS):
        print(f"\n--- Trial {trial + 1}/{NUM_TRIALS} ---")

        for test_case in all_test_cases:
            failure_type = test_case.failure_type.value

            # Create failure info
            failure_info = FailureInfo(
                is_failure=True,
                failure_type=test_case.failure_type,
                agent_name="test_agent",
                error_message=f"Detected {test_case.failure_mode.value}",
                confidence=0.9,
                evidence=f"Test case for {test_case.failure_mode.value}",
                input_state=test_case.input_state,
                output=test_case.failed_output
            )

            # Attempt repair
            start_time = time.time()
            repair_result = repair.repair(
                failure_info=failure_info,
                agent_func=lambda x: x,  # Placeholder
                original_prompt=test_case.original_task,
                input_state=test_case.input_state,
                original_task=test_case.original_task
            )
            repair_time = (time.time() - start_time) * 1000

            results_by_type[failure_type]["attempts"] += 1

            # Verify repair quality
            if repair_result.success and repair_result.new_output:
                verification = verify_repair_quality(
                    repair_result.new_output,
                    test_case,
                    detector
                )

                if verification["is_valid"]:
                    results_by_type[failure_type]["successes"] += 1

                results_by_type[failure_type]["quality_scores"].append(
                    verification["quality_score"]
                )

                success = verification["is_valid"]
                quality = verification["quality_score"]
            else:
                success = False
                quality = 0.0
                results_by_type[failure_type]["quality_scores"].append(0.0)

            # Record result
            result = ExperimentResult(
                run_id=f"{failure_type}_{trial}",
                success=success,
                failures_injected=1,
                failures_detected=1,
                repair_attempts=repair_result.attempts if repair_result else 0,
                repair_successes=1 if success else 0,
                total_time_ms=repair_time,
                recovery_time_ms=repair_time,
                output_quality_score=quality,
                failure_types=[failure_type],
                config={
                    "failure_type": failure_type,
                    "trial": trial,
                    "strategy": repair_result.strategy_used if repair_result else None
                }
            )
            collector.add_result(result)

    # Calculate and display results
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    print("\nRepair Success Rate by Failure Type (with 95% CI):")
    print("-" * 50)

    for failure_type, stats in results_by_type.items():
        ci = calculate_proportion_ci(stats["successes"], stats["attempts"])
        avg_quality = (sum(stats["quality_scores"]) / len(stats["quality_scores"])
                      if stats["quality_scores"] else 0)

        print(f"\n{failure_type}:")
        print(f"  Success rate: {format_ci(ci, as_percentage=True)}")
        print(f"  Avg quality score: {avg_quality:.2f}")

    # Overall
    total_successes = sum(s["successes"] for s in results_by_type.values())
    total_attempts = sum(s["attempts"] for s in results_by_type.values())
    overall_ci = calculate_proportion_ci(total_successes, total_attempts)

    print(f"\nOverall repair success rate: {format_ci(overall_ci, as_percentage=True)}")

    # Save results
    output = collector.end_experiment()

    return output


def run_repair_strategy_comparison():
    """Compare different repair strategies"""

    print("\n" + "=" * 60)
    print("REPAIR STRATEGY COMPARISON")
    print("=" * 60)

    # This would compare different repair approaches
    # For now, we test with different configurations

    strategies = {
        "default": AEGISConfig(),
        "aggressive": AEGISConfig(),
        "conservative": AEGISConfig()
    }

    # Modify configurations
    strategies["aggressive"].repair.max_repair_attempts = 5
    strategies["aggressive"].repair.enable_temperature_adjustment = True

    strategies["conservative"].repair.max_repair_attempts = 1
    strategies["conservative"].repair.enable_temperature_adjustment = False

    results = {}

    for strategy_name, config in strategies.items():
        print(f"\nTesting strategy: {strategy_name}")

        detector = AEGISDetector(config)
        repair = AEGISRepair(config, detector)

        successes = 0
        total = 0

        # Quick test with hallucination cases
        test_cases = [tc for tc in create_test_cases()
                      if tc.failure_type == FailureType.HALLUCINATION][:5]

        for test_case in test_cases:
            failure_info = FailureInfo(
                is_failure=True,
                failure_type=test_case.failure_type,
                agent_name="test_agent",
                error_message=f"Detected {test_case.failure_mode.value}",
                confidence=0.9,
                input_state=test_case.input_state,
                output=test_case.failed_output
            )

            repair_result = repair.repair(
                failure_info=failure_info,
                agent_func=lambda x: x,
                original_prompt=test_case.original_task,
                input_state=test_case.input_state,
                original_task=test_case.original_task
            )

            total += 1
            if repair_result.success:
                verification = verify_repair_quality(
                    repair_result.new_output, test_case, detector
                )
                if verification["is_valid"]:
                    successes += 1

        results[strategy_name] = {
            "successes": successes,
            "total": total,
            "rate": successes / total if total > 0 else 0
        }

        print(f"  Success rate: {results[strategy_name]['rate']:.1%}")

    return results


if __name__ == "__main__":
    print("Starting repair experiments...")

    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set")
        sys.exit(1)

    # Run experiments
    run_repair_experiment()
    run_repair_strategy_comparison()

    print("\nRepair experiments complete!")
