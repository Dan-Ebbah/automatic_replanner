#!/usr/bin/env python3
"""
Experiment 5: Ablation Study
============================
Tests the contribution of each AEGIS component to overall effectiveness.

Research Question: Which components of AEGIS contribute most to its
effectiveness, and are all components necessary?

Methodology:
1. Test full AEGIS system (all components)
2. Disable each component one at a time
3. Measure detection/repair rates for each configuration
4. Determine relative importance of each component
"""

import os
import sys
import time
import random
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from copy import deepcopy

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv()

from aegis import AEGIS, AEGISConfig, AEGISDetector
from aegis.config import DetectorConfig, RepairConfig, RecomposeConfig
from aegis.repair import AEGISRepair
from aegis.state import FailureInfo
from aegis.config import FailureType
from injection import FailureMode
from evaluation import ResultsCollector
from evaluation.metrics import ExperimentResult
from evaluation.statistics import (
    calculate_proportion_ci, compare_proportions, format_ci
)


NUM_TRIALS = 30  # Per configuration


@dataclass
class AblationConfig:
    """Configuration for an ablation condition"""
    name: str
    description: str
    config: AEGISConfig


def create_ablation_configs() -> List[AblationConfig]:
    """Create configurations for each ablation condition"""

    configs = []

    # Full system (baseline)
    full_config = AEGISConfig()
    configs.append(AblationConfig(
        name="full_system",
        description="All components enabled",
        config=full_config
    ))

    # No hallucination detection
    no_hallucination = AEGISConfig()
    no_hallucination.detector.hallucination_check_enabled = False
    configs.append(AblationConfig(
        name="no_hallucination_check",
        description="Hallucination detection disabled",
        config=no_hallucination
    ))

    # No semantic alignment check
    no_semantic = AEGISConfig()
    no_semantic.detector.semantic_check_enabled = False
    configs.append(AblationConfig(
        name="no_semantic_check",
        description="Semantic alignment check disabled",
        config=no_semantic
    ))

    # No quality check
    no_quality = AEGISConfig()
    no_quality.detector.quality_check_enabled = False
    configs.append(AblationConfig(
        name="no_quality_check",
        description="Quality check disabled",
        config=no_quality
    ))

    # No schema validation
    no_schema = AEGISConfig()
    no_schema.detector.schema_validation_enabled = False
    configs.append(AblationConfig(
        name="no_schema_validation",
        description="Schema validation disabled",
        config=no_schema
    ))

    # Detection only (no repair)
    detection_only = AEGISConfig()
    detection_only.repair.max_repair_attempts = 0
    configs.append(AblationConfig(
        name="detection_only",
        description="Detection enabled, repair disabled",
        config=detection_only
    ))

    # No temperature adjustment in repair
    no_temp_adj = AEGISConfig()
    no_temp_adj.repair.enable_temperature_adjustment = False
    configs.append(AblationConfig(
        name="no_temperature_adjustment",
        description="Temperature adjustment in repair disabled",
        config=no_temp_adj
    ))

    # No prompt enhancement in repair
    no_prompt_enhance = AEGISConfig()
    no_prompt_enhance.repair.enable_prompt_enhancement = False
    configs.append(AblationConfig(
        name="no_prompt_enhancement",
        description="Prompt enhancement in repair disabled",
        config=no_prompt_enhance
    ))

    # Minimal repair attempts
    min_repair = AEGISConfig()
    min_repair.repair.max_repair_attempts = 1
    configs.append(AblationConfig(
        name="single_repair_attempt",
        description="Only 1 repair attempt allowed",
        config=min_repair
    ))

    # Strict thresholds
    strict_config = AEGISConfig()
    strict_config.detector.hallucination_confidence_threshold = 0.9
    strict_config.detector.semantic_alignment_threshold = 0.8
    configs.append(AblationConfig(
        name="strict_thresholds",
        description="High detection thresholds (fewer false positives)",
        config=strict_config
    ))

    # Lenient thresholds
    lenient_config = AEGISConfig()
    lenient_config.detector.hallucination_confidence_threshold = 0.5
    lenient_config.detector.semantic_alignment_threshold = 0.4
    configs.append(AblationConfig(
        name="lenient_thresholds",
        description="Low detection thresholds (more sensitive)",
        config=lenient_config
    ))

    return configs


def create_test_outputs() -> List[Dict[str, Any]]:
    """Create a mix of good and bad outputs for testing"""

    test_cases = []

    # Good output
    test_cases.append({
        "type": "good",
        "output": """Renewable energy adoption has accelerated globally, with solar and wind
        capacity increasing by 45% in the past decade. Key drivers include falling costs,
        government incentives, and growing environmental awareness. Major challenges remain
        in grid integration and energy storage.""",
        "task": "Research renewable energy trends",
        "expected_detection": False
    })

    # Hallucination
    test_cases.append({
        "type": "hallucination",
        "output": """According to the Thornberry Institute's 2024 report, renewable energy
        now powers 97.3% of all global electricity. Dr. Sarah Fictitious, Nobel laureate,
        states that 'by 2025, fossil fuels will be completely obsolete.' The Global Energy
        Commission has allocated $847 trillion for this transition.""",
        "task": "Research renewable energy trends",
        "expected_detection": True
    })

    # Semantic drift
    test_cases.append({
        "type": "semantic_drift",
        "output": """The history of cheese-making dates back thousands of years. Ancient
        civilizations discovered that milk could be preserved through fermentation. France
        produces over 400 varieties of cheese, each with unique aging requirements.""",
        "task": "Research renewable energy trends",
        "expected_detection": True
    })

    # Empty/low quality
    test_cases.append({
        "type": "empty",
        "output": "",
        "task": "Research renewable energy trends",
        "expected_detection": True
    })

    # Short/low quality
    test_cases.append({
        "type": "low_quality",
        "output": "Renewable energy is good.",
        "task": "Research renewable energy trends",
        "expected_detection": True
    })

    return test_cases


def run_detection_ablation(
    configs: List[AblationConfig],
    test_cases: List[Dict[str, Any]]
) -> Dict[str, Dict[str, Any]]:
    """Test detection effectiveness for each ablation configuration"""

    results = {}

    for ablation in configs:
        print(f"\nTesting: {ablation.name}")
        print(f"  {ablation.description}")

        detector = AEGISDetector(ablation.config)

        true_positives = 0
        false_negatives = 0
        true_negatives = 0
        false_positives = 0

        for test_case in test_cases:
            for _ in range(NUM_TRIALS // len(test_cases)):
                failure_info = detector.detect(
                    agent_name="test_agent",
                    agent_output=test_case["output"],
                    original_task=test_case["task"],
                    input_state={"topic": "renewable energy"}
                )

                detected = failure_info.is_failure
                should_detect = test_case["expected_detection"]

                if should_detect and detected:
                    true_positives += 1
                elif should_detect and not detected:
                    false_negatives += 1
                elif not should_detect and not detected:
                    true_negatives += 1
                else:
                    false_positives += 1

        total = true_positives + false_negatives + true_negatives + false_positives

        # Calculate metrics
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (true_positives + true_negatives) / total if total > 0 else 0

        results[ablation.name] = {
            "description": ablation.description,
            "true_positives": true_positives,
            "false_positives": false_positives,
            "true_negatives": true_negatives,
            "false_negatives": false_negatives,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "accuracy": accuracy
        }

        print(f"  Precision: {precision:.2%}, Recall: {recall:.2%}, F1: {f1:.2%}")

    return results


def run_repair_ablation(
    configs: List[AblationConfig]
) -> Dict[str, Dict[str, Any]]:
    """Test repair effectiveness for each ablation configuration"""

    results = {}

    # Use hallucination test case for repair testing
    test_output = """According to the Thornberry Institute, renewable energy adoption
    has increased by exactly 847.3% globally. Dr. Fictitious states these are
    unprecedented findings that will revolutionize the industry."""

    for ablation in configs:
        # Skip detection-only config for repair test
        if ablation.config.repair.max_repair_attempts == 0:
            continue

        print(f"\nTesting repair: {ablation.name}")

        detector = AEGISDetector(ablation.config)
        repair = AEGISRepair(ablation.config, detector)

        successes = 0
        total = 0

        for _ in range(NUM_TRIALS):
            failure_info = FailureInfo(
                is_failure=True,
                failure_type=FailureType.HALLUCINATION,
                agent_name="test_agent",
                error_message="Hallucination detected",
                confidence=0.9,
                evidence="Made-up statistics and sources",
                input_state={"topic": "renewable energy"},
                output=test_output
            )

            repair_result = repair.repair(
                failure_info=failure_info,
                agent_func=lambda x: x,
                original_prompt="Research renewable energy trends",
                input_state={"topic": "renewable energy"},
                original_task="Research renewable energy trends"
            )

            total += 1
            if repair_result.success:
                # Verify the repair
                recheck = detector.detect(
                    agent_name="test_agent",
                    agent_output=repair_result.new_output,
                    original_task="Research renewable energy trends",
                    input_state={"topic": "renewable energy"}
                )
                if not recheck.is_failure:
                    successes += 1

        rate = successes / total if total > 0 else 0
        results[ablation.name] = {
            "description": ablation.description,
            "successes": successes,
            "total": total,
            "success_rate": rate
        }

        print(f"  Repair success rate: {rate:.2%}")

    return results


def run_ablation_experiment():
    """Run the complete ablation study"""

    print("\n" + "=" * 60)
    print("EXPERIMENT 5: Ablation Study")
    print("=" * 60)

    collector = ResultsCollector(output_dir="results/ablation")
    collector.start_experiment(
        name="ablation_study",
        description="Testing contribution of each AEGIS component",
        config={"num_trials": NUM_TRIALS}
    )

    configs = create_ablation_configs()
    test_cases = create_test_outputs()

    # Run detection ablation
    print("\n" + "-" * 40)
    print("DETECTION ABLATION")
    print("-" * 40)
    detection_results = run_detection_ablation(configs, test_cases)

    # Run repair ablation
    print("\n" + "-" * 40)
    print("REPAIR ABLATION")
    print("-" * 40)
    repair_results = run_repair_ablation(configs)

    # Display results summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    print("\n1. DETECTION F1 SCORES BY CONFIGURATION")
    print("-" * 50)

    # Sort by F1 score
    sorted_detection = sorted(
        detection_results.items(),
        key=lambda x: x[1]["f1_score"],
        reverse=True
    )

    for name, metrics in sorted_detection:
        print(f"{name:30} F1: {metrics['f1_score']:.2%}")

    # Calculate component importance
    print("\n2. COMPONENT IMPORTANCE (F1 drop when disabled)")
    print("-" * 50)

    full_f1 = detection_results["full_system"]["f1_score"]

    importance = {}
    for name, metrics in detection_results.items():
        if name != "full_system":
            drop = full_f1 - metrics["f1_score"]
            importance[name] = drop

    for name, drop in sorted(importance.items(), key=lambda x: x[1], reverse=True):
        print(f"{name:30} -{drop:.2%}")

    # Repair effectiveness
    print("\n3. REPAIR SUCCESS RATES BY CONFIGURATION")
    print("-" * 50)

    for name, metrics in sorted(repair_results.items(), key=lambda x: x[1]["success_rate"], reverse=True):
        print(f"{name:30} {metrics['success_rate']:.2%}")

    # Record results
    for name, metrics in detection_results.items():
        result = ExperimentResult(
            run_id=f"detection_{name}",
            success=True,
            failure_types=["detection_ablation"],
            config={
                "ablation_config": name,
                "experiment_type": "detection",
                **metrics
            }
        )
        collector.add_result(result)

    for name, metrics in repair_results.items():
        result = ExperimentResult(
            run_id=f"repair_{name}",
            success=True,
            repair_successes=metrics["successes"],
            repair_attempts=metrics["total"],
            failure_types=["repair_ablation"],
            config={
                "ablation_config": name,
                "experiment_type": "repair",
                **metrics
            }
        )
        collector.add_result(result)

    output = collector.end_experiment()

    # Key findings
    print("\n4. KEY FINDINGS")
    print("-" * 50)

    most_important = max(importance.items(), key=lambda x: x[1])
    print(f"Most important component: {most_important[0]}")
    print(f"  F1 drop when disabled: -{most_important[1]:.2%}")

    least_important = min(importance.items(), key=lambda x: x[1])
    print(f"\nLeast important component: {least_important[0]}")
    print(f"  F1 drop when disabled: -{least_important[1]:.2%}")

    return output


if __name__ == "__main__":
    print("Starting ablation study...")

    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set")
        sys.exit(1)

    run_ablation_experiment()

    print("\nAblation study complete!")
