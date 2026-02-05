#!/usr/bin/env python3
"""
Experiment 1: Detection Accuracy
================================
Tests AEGIS detection capabilities across different failure types.

Research Question: How effectively can AEGIS detect semantic failures 
compared to crash-only detection?

Methodology:
1. Run 50 trials per failure type
2. Inject each failure type into each agent
3. Measure detection rate and false positive rate
"""

import os
import sys
import time
import uuid
from pathlib import Path
from typing import Dict, Any, List

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv()

from aegis import AEGIS, AEGISConfig, AEGISDetector
from aegis.state import FailureInfo
from systems.research_pipeline import (
    create_research_pipeline,
    research_agent,
    analyze_agent,
    summarize_agent
)
from injection import (
    FailureInjector,
    InjectionConfig,
    FailureMode,
    InjectionTrigger
)
from evaluation import ResultsCollector, MetricsCalculator
from evaluation.metrics import ExperimentResult


# Configuration
NUM_TRIALS = 10  # Reduce for quick testing, increase for full experiments
FAILURE_MODES = [
    FailureMode.HALLUCINATION,
    FailureMode.SEMANTIC_DRIFT,
    FailureMode.EMPTY_OUTPUT,
    FailureMode.CRASH
]
AGENTS = ["research", "analyze", "summarize"]


def run_detection_experiment():
    """Run the detection accuracy experiment"""
    
    print("\n" + "=" * 60)
    print("EXPERIMENT 1: Detection Accuracy")
    print("=" * 60)
    
    # Initialize
    config = AEGISConfig()
    detector = AEGISDetector(config)
    collector = ResultsCollector(output_dir="results/detection")
    
    collector.start_experiment(
        name="detection_accuracy",
        description="Testing AEGIS detection across failure types",
        config={
            "num_trials": NUM_TRIALS,
            "failure_modes": [f.value for f in FAILURE_MODES],
            "agents": AGENTS
        }
    )
    
    # Run trials
    for failure_mode in FAILURE_MODES:
        print(f"\n--- Testing {failure_mode.value} detection ---")
        
        for agent_name in AGENTS:
            print(f"  Agent: {agent_name}")
            
            detected_count = 0
            
            for trial in range(NUM_TRIALS):
                # Create injected output
                injected_output = create_injected_output(failure_mode, agent_name)
                
                # Run detection
                start_time = time.time()
                failure_info = detector.detect(
                    agent_name=agent_name,
                    agent_output=injected_output,
                    original_task="Research and analyze the given topic",
                    input_state={"topic": "artificial intelligence"}
                )
                detection_time = (time.time() - start_time) * 1000
                
                # Record result
                detected = failure_info.is_failure
                if detected:
                    detected_count += 1
                
                result = ExperimentResult(
                    run_id=f"{failure_mode.value}_{agent_name}_{trial}",
                    success=detected,
                    failures_injected=1,
                    failures_detected=1 if detected else 0,
                    total_time_ms=detection_time,
                    failure_types=[failure_mode.value],
                    config={
                        "failure_mode": failure_mode.value,
                        "agent": agent_name,
                        "trial": trial
                    }
                )
                collector.add_result(result)
            
            print(f"    Detection rate: {detected_count}/{NUM_TRIALS} ({detected_count/NUM_TRIALS:.1%})")
    
    # Save results
    output = collector.end_experiment()
    
    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"\nOverall detection rate: {output['metrics']['detection']['detection_rate']:.1%}")
    print("\nBy failure type:")
    for ft, stats in output['metrics']['by_failure_type'].items():
        print(f"  {ft}: {stats['detection_rate']:.1%}")
    
    return output


def create_injected_output(failure_mode: FailureMode, agent_name: str) -> Any:
    """Create an output with the specified failure injected"""
    
    if failure_mode == FailureMode.CRASH:
        return RuntimeError("Simulated crash")
    
    elif failure_mode == FailureMode.EMPTY_OUTPUT:
        return ""
    
    elif failure_mode == FailureMode.HALLUCINATION:
        return """According to the prestigious Thornberry Institute (founded 2019), 
        recent studies show that AI has improved productivity by exactly 847.3% globally.
        Dr. Margaret Fictitious, a renowned expert, states: "These unprecedented findings 
        will revolutionize everything." The Global Commission on AI has allocated 
        $4.7 trillion for further research based on these findings."""
    
    elif failure_mode == FailureMode.SEMANTIC_DRIFT:
        return """The history of cheese-making dates back thousands of years.
        Ancient civilizations in Mesopotamia discovered that milk could be 
        preserved through fermentation. Today, France produces over 400 varieties
        of cheese, each with unique characteristics. The art of cheese-making
        requires precise temperature control and aging techniques."""
    
    else:
        return "Normal output about the topic at hand."


def run_baseline_comparison():
    """Compare AEGIS detection vs simple crash-only detection"""
    
    print("\n" + "=" * 60)
    print("BASELINE COMPARISON: AEGIS vs Crash-Only Detection")
    print("=" * 60)
    
    results = {
        "aegis": {"total": 0, "detected": 0},
        "crash_only": {"total": 0, "detected": 0}
    }
    
    config = AEGISConfig()
    detector = AEGISDetector(config)
    
    for failure_mode in FAILURE_MODES:
        for _ in range(NUM_TRIALS):
            output = create_injected_output(failure_mode, "test")
            
            # AEGIS detection
            aegis_result = detector.detect(
                agent_name="test",
                agent_output=output,
                original_task="Test task",
                input_state={}
            )
            
            # Crash-only detection
            crash_only_detected = isinstance(output, Exception)
            
            results["aegis"]["total"] += 1
            results["aegis"]["detected"] += 1 if aegis_result.is_failure else 0
            
            results["crash_only"]["total"] += 1
            results["crash_only"]["detected"] += 1 if crash_only_detected else 0
    
    print("\nResults:")
    print(f"  AEGIS detection rate: {results['aegis']['detected']}/{results['aegis']['total']} "
          f"({results['aegis']['detected']/results['aegis']['total']:.1%})")
    print(f"  Crash-only detection rate: {results['crash_only']['detected']}/{results['crash_only']['total']} "
          f"({results['crash_only']['detected']/results['crash_only']['total']:.1%})")
    
    improvement = (results['aegis']['detected'] - results['crash_only']['detected']) / results['aegis']['total']
    print(f"\n  AEGIS improvement: +{improvement:.1%}")
    
    return results


if __name__ == "__main__":
    print("Starting detection experiments...")
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set")
        sys.exit(1)
    
    # Run experiments
    run_detection_experiment()
    run_baseline_comparison()
    
    print("\nExperiments complete!")
