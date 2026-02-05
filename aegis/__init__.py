"""
AEGIS: Autonomous Error-handling and Graph-recomposition 
for Intelligent agent Systems
=======================================================

A self-healing framework for LangGraph multi-agent workflows.

Quick Start:
    from aegis import AEGIS
    
    # Wrap any LangGraph workflow
    aegis_workflow = AEGIS.wrap(your_workflow)
    
    # Run with self-healing
    result = aegis_workflow.invoke({"input": "your input"})

Components:
    - AEGIS: Main wrapper class
    - AEGISDetector: Failure detection
    - AEGISRepair: Agent repair strategies
    - AEGISRecompose: Workflow recomposition
    - AgentRegistry: Available agents for recomposition
"""

from .config import (
    AEGISConfig,
    DetectorConfig,
    RepairConfig,
    RecomposeConfig,
    FailureType,
    RecoveryStrategy,
    LLMProvider,
    default_config
)

from .state import (
    FailureInfo,
    RepairResult,
    RecomposeResult,
    HealingLog,
    HealingEvent,
    AgentInfo,
    AEGISState,
    create_aegis_state
)

from .detector import AEGISDetector
from .repair import AEGISRepair
from .recompose import AEGISRecompose
from .registry import AgentRegistry, default_registry
from .wrapper import AEGIS, AEGISWorkflow, with_aegis_monitoring

from .events import EventBus, Event, EventType, WeatherChangedEvent, ReplanRequestedEvent, PlanUpdatedEvent
from .agent import BaseAgent, AgentConfig
from .mcp_adapter import MCPAgentAdapter, MCPToolSpec

__version__ = "0.1.0"
__author__ = "Your Name"

__all__ = [
    # Main classes
    "AEGIS",
    "AEGISWorkflow",
    
    # Components
    "AEGISDetector",
    "AEGISRepair",
    "AEGISRecompose",
    "AgentRegistry",
    
    # Configuration
    "AEGISConfig",
    "DetectorConfig",
    "RepairConfig",
    "RecomposeConfig",
    "default_config",
    
    # State and results
    "FailureInfo",
    "RepairResult",
    "RecomposeResult",
    "HealingLog",
    "HealingEvent",
    "AgentInfo",
    "AEGISState",
    "create_aegis_state",
    
    # Enums
    "FailureType",
    "RecoveryStrategy",
    "LLMProvider",
    
    # Utilities
    "default_registry",
    "with_aegis_monitoring",

    # Event system
    "EventBus",
    "Event",
    "EventType",
    "WeatherChangedEvent",
    "ReplanRequestedEvent",
    "PlanUpdatedEvent",

    # Agent framework
    "BaseAgent",
    "AgentConfig",
    "MCPAgentAdapter",
    "MCPToolSpec",
]
