"""
Basic Tests for AEGIS
=====================
Quick sanity checks to ensure modules import correctly.
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_imports():
    """Test that all main modules can be imported"""
    from aegis import (
        AEGIS,
        AEGISConfig,
        AEGISDetector,
        AEGISRepair,
        AEGISRecompose,
        AgentRegistry,
        FailureType,
        RecoveryStrategy
    )
    
    assert AEGIS is not None
    assert AEGISConfig is not None
    assert AEGISDetector is not None


def test_config_creation():
    """Test configuration creation"""
    from aegis import AEGISConfig, DetectorConfig
    
    config = AEGISConfig()
    assert config.llm_model == "gpt-4o"
    
    custom_config = AEGISConfig(
        llm_model="gpt-4o-mini",
        detector=DetectorConfig(
            hallucination_confidence_threshold=0.9
        )
    )
    assert custom_config.llm_model == "gpt-4o-mini"
    assert custom_config.detector.hallucination_confidence_threshold == 0.9


def test_agent_registry():
    """Test agent registry"""
    from aegis import AgentRegistry
    
    registry = AgentRegistry()
    
    # Check default agents are registered
    agents = registry.list_agents()
    assert len(agents) > 0
    
    # Check specific agent
    fact_checker = registry.get_agent("fact_checker")
    assert fact_checker is not None
    assert fact_checker.role == "validator"


def test_failure_info():
    """Test FailureInfo dataclass"""
    from aegis import FailureInfo, FailureType
    
    failure = FailureInfo(
        is_failure=True,
        failure_type=FailureType.HALLUCINATION,
        agent_name="test_agent",
        confidence=0.85
    )
    
    assert failure.is_failure == True
    assert failure.failure_type == FailureType.HALLUCINATION
    assert failure.confidence == 0.85
    
    # Test to_dict
    d = failure.to_dict()
    assert d["is_failure"] == True
    assert d["failure_type"] == "hallucination"


def test_injection_framework():
    """Test failure injection framework"""
    from injection import (
        FailureInjector,
        InjectionConfig,
        FailureMode,
        InjectionTrigger
    )
    
    injector = FailureInjector()
    
    # Schedule a failure
    injector.schedule(
        agent_name="test",
        config=InjectionConfig(
            failure_mode=FailureMode.EMPTY_OUTPUT,
            trigger=InjectionTrigger.ALWAYS
        )
    )
    
    assert "test" in injector.schedules


def test_systems_import():
    """Test that test systems can be imported"""
    from systems import (
        ResearchPipelineState,
        research_agent,
        analyze_agent,
        summarize_agent,
        create_research_pipeline
    )
    
    assert create_research_pipeline is not None


def test_evaluation_metrics():
    """Test evaluation metrics calculator"""
    from evaluation import MetricsCalculator
    from evaluation.metrics import ExperimentResult
    
    calc = MetricsCalculator()
    
    # Add some results
    calc.add_result(ExperimentResult(
        run_id="test_1",
        success=True,
        failures_injected=1,
        failures_detected=1,
        repair_attempts=1,
        repair_successes=1
    ))
    
    calc.add_result(ExperimentResult(
        run_id="test_2",
        success=False,
        failures_injected=1,
        failures_detected=0,
        repair_attempts=0,
        repair_successes=0
    ))
    
    metrics = calc.calculate_all()
    
    assert metrics["summary"]["total_runs"] == 2
    assert metrics["summary"]["success_rate"] == 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
