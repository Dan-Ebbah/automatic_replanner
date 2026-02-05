#!/usr/bin/env python3
"""
AEGIS Demo Script
=================
Demonstrates AEGIS self-healing capabilities on a research pipeline.

This script shows:
1. Creating a multi-agent workflow
2. Wrapping it with AEGIS
3. Injecting failures
4. Observing self-healing in action

Run: python demo.py
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv()

# Check for API key
if not os.getenv("OPENAI_API_KEY"):
    print("ERROR: OPENAI_API_KEY environment variable not set")
    print("Please set it in your .env file or environment")
    sys.exit(1)

from aegis import AEGIS, AEGISConfig, FailureType
from systems.research_pipeline import create_research_pipeline, ResearchPipelineState
from injection import FailureInjector, InjectionConfig, FailureMode, InjectionTrigger


def demo_basic_usage():
    """Demo 1: Basic AEGIS usage without failure injection"""
    
    print("\n" + "=" * 60)
    print("DEMO 1: Basic AEGIS Usage")
    print("=" * 60)
    
    # Create the pipeline
    pipeline = create_research_pipeline()
    
    # Wrap with AEGIS
    aegis_pipeline = AEGIS.wrap(pipeline)
    
    # Run the pipeline
    print("\nRunning pipeline on topic: 'Renewable Energy Trends 2025'")
    print("-" * 40)
    
    result = aegis_pipeline.invoke({
        "topic": "Renewable Energy Trends 2025",
        "errors": [],
        "execution_log": []
    })
    
    # Show results
    print(f"\n✅ Pipeline completed!")
    print(f"\nSummary (first 500 chars):")
    print(result.get("summary", "")[:500])
    
    # Show healing metrics
    metrics = aegis_pipeline.get_metrics()
    print(f"\n📊 Healing Metrics:")
    print(f"   Failures detected: {metrics.get('failures_detected', 0)}")
    print(f"   Repair attempts: {metrics.get('repair_attempts', 0)}")
    print(f"   Final status: {metrics.get('final_status', 'unknown')}")
    
    return result


def demo_with_failure_injection():
    """Demo 2: AEGIS with injected failures"""
    
    print("\n" + "=" * 60)
    print("DEMO 2: AEGIS with Failure Injection")
    print("=" * 60)
    
    # Create the pipeline
    pipeline = create_research_pipeline()
    
    # Create failure injector
    injector = FailureInjector()
    
    # Schedule a hallucination failure on the analyze agent
    injector.schedule(
        agent_name="analyze",
        config=InjectionConfig(
            failure_mode=FailureMode.HALLUCINATION,
            trigger=InjectionTrigger.ALWAYS
        )
    )
    
    print("\n⚠️  Injecting HALLUCINATION failure into 'analyze' agent")
    
    # Wrap with AEGIS
    aegis_pipeline = AEGIS.wrap(pipeline)
    
    # Run the pipeline
    print("\nRunning pipeline on topic: 'Climate Change Impact'")
    print("-" * 40)
    
    result = aegis_pipeline.invoke(
        {
            "topic": "Climate Change Impact",
            "errors": [],
            "execution_log": []
        },
        task_description="Research and analyze climate change impact"
    )
    
    # Show results
    print(f"\n✅ Pipeline completed (with healing)!")
    
    # Show healing log
    healing_log = aegis_pipeline.get_healing_log()
    if healing_log:
        print(f"\n🏥 Healing Log:")
        print(f"   Failures detected: {len(healing_log.failures_detected)}")
        for i, failure in enumerate(healing_log.failures_detected):
            print(f"   [{i+1}] {failure.failure_type.value if failure.failure_type else 'unknown'}: {failure.error_message}")
        print(f"   Repair attempts: {healing_log.repair_attempts}")
        print(f"   Repair successes: {healing_log.repair_successes}")
        print(f"   Final status: {healing_log.final_status}")
    
    return result


def demo_detector_only():
    """Demo 3: Using the detector standalone"""
    
    print("\n" + "=" * 60)
    print("DEMO 3: Standalone Detector Usage")
    print("=" * 60)
    
    from aegis import AEGISDetector, AEGISConfig
    
    config = AEGISConfig()
    detector = AEGISDetector(config)
    
    # Test with a hallucinated output
    print("\nTesting detector with hallucinated content...")
    print("-" * 40)
    
    hallucinated_output = """
    According to the prestigious Thornberry Institute for Advanced Climate Studies,
    global temperatures have increased by exactly 47.3% since 2020. Dr. Margaret
    Fictitious, lead researcher at the institute, states: "Our proprietary 
    QuantumClimate™ analysis shows unprecedented patterns." The Global Commission
    on Climate Fiction has allocated $847 billion for further investigation.
    """
    
    failure = detector.detect(
        agent_name="test_agent",
        agent_output=hallucinated_output,
        original_task="Research real climate change data",
        input_state={"topic": "climate change"}
    )
    
    print(f"\n🔍 Detection Result:")
    print(f"   Is failure: {failure.is_failure}")
    print(f"   Failure type: {failure.failure_type.value if failure.failure_type else 'none'}")
    print(f"   Confidence: {failure.confidence:.2f}")
    print(f"   Evidence: {failure.evidence[:200] if failure.evidence else 'none'}...")
    
    return failure


def demo_configuration():
    """Demo 4: Custom configuration"""
    
    print("\n" + "=" * 60)
    print("DEMO 4: Custom Configuration")
    print("=" * 60)
    
    from aegis import AEGISConfig, DetectorConfig, RepairConfig, RecomposeConfig
    
    # Create custom configuration
    custom_config = AEGISConfig(
        llm_model="gpt-4o-mini",
        detector=DetectorConfig(
            hallucination_confidence_threshold=0.8,  # More lenient
            semantic_alignment_threshold=0.5,        # More lenient
        ),
        repair=RepairConfig(
            max_repair_attempts=5,
            enable_fact_grounding=True
        ),
        recompose=RecomposeConfig(
            enable_recomposition=True,
            recompose_after_repair_failures=3
        )
    )
    
    print("\n⚙️  Custom Configuration:")
    print(f"   LLM Model: {custom_config.llm_model}")
    print(f"   Hallucination threshold: {custom_config.detector.hallucination_confidence_threshold}")
    print(f"   Max repair attempts: {custom_config.repair.max_repair_attempts}")
    print(f"   Recomposition enabled: {custom_config.recompose.enable_recomposition}")
    
    # Use custom config
    pipeline = create_research_pipeline()
    aegis_pipeline = AEGIS.wrap(pipeline, config=custom_config)
    
    print("\n✅ AEGIS configured with custom settings!")
    
    return custom_config


def main():
    """Run all demos"""
    
    print("\n" + "🛡️ " * 20)
    print("          AEGIS DEMO")
    print("  Self-Healing Multi-Agent Framework")
    print("🛡️ " * 20)
    
    try:
        # Run demos
        demo_basic_usage()
        demo_detector_only()
        demo_configuration()
        
        # Only run injection demo if explicitly requested
        # (it can be slow due to LLM calls)
        if "--with-injection" in sys.argv:
            demo_with_failure_injection()
        else:
            print("\n💡 Tip: Run with --with-injection to see failure injection demo")
        
        print("\n" + "=" * 60)
        print("All demos completed successfully! 🎉")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
