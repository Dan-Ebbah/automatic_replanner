"""
AEGIS Configuration
===================
Central configuration for the AEGIS self-healing framework.
"""

from dataclasses import dataclass, field
from typing import Optional, List
from enum import Enum
import os
from dotenv import load_dotenv

load_dotenv()


class LLMProvider(Enum):
    """Supported LLM providers"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


class RecoveryStrategy(Enum):
    """Available recovery strategies"""
    REPAIR = "repair"           # Fix the agent's output
    REPLACE = "replace"         # Swap agent with alternative
    RECOMPOSE = "recompose"     # Restructure the workflow
    ESCALATE = "escalate"       # Human intervention needed


class FailureType(Enum):
    """Types of failures AEGIS can detect"""
    CRASH = "crash"                     # Exception/error
    TIMEOUT = "timeout"                 # Agent took too long
    HALLUCINATION = "hallucination"     # Made up information
    SEMANTIC_DRIFT = "semantic_drift"   # Misunderstood the task
    FORMAT_ERROR = "format_error"       # Wrong output format
    CASCADE = "cascade"                 # Failure propagated from another agent
    QUALITY = "quality"                 # Output below quality threshold


@dataclass
class DetectorConfig:
    """Configuration for the failure detector"""
    
    # Hallucination detection
    hallucination_check_enabled: bool = True
    hallucination_confidence_threshold: float = 0.7
    
    # Semantic alignment detection
    semantic_check_enabled: bool = True
    semantic_alignment_threshold: float = 0.6
    
    # Timeout settings
    timeout_enabled: bool = True
    default_timeout_seconds: float = 60.0
    
    # Schema validation
    schema_validation_enabled: bool = True
    
    # Quality checks
    quality_check_enabled: bool = True
    min_output_length: int = 10
    max_output_length: int = 50000


@dataclass
class RepairConfig:
    """Configuration for the repair module"""
    
    # Retry settings
    max_repair_attempts: int = 3
    repair_backoff_seconds: float = 1.0
    
    # Prompt enhancement
    enable_prompt_enhancement: bool = True
    
    # Grounding
    enable_fact_grounding: bool = True
    max_grounding_facts: int = 5
    
    # Temperature adjustment
    enable_temperature_adjustment: bool = True
    retry_temperatures: List[float] = field(default_factory=lambda: [0.3, 0.1, 0.0])


@dataclass
class RecomposeConfig:
    """Configuration for the workflow recomposition module"""
    
    # Recomposition triggers
    enable_recomposition: bool = True
    recompose_after_repair_failures: int = 2  # Trigger recompose after N repair failures
    
    # Recomposition constraints
    max_agents_to_add: int = 3
    max_workflow_depth: int = 10
    allow_parallel_branches: bool = True
    
    # Safety
    require_human_approval: bool = False
    max_recomposition_attempts: int = 2


@dataclass
class AEGISConfig:
    """Main AEGIS configuration"""
    
    # LLM Settings
    llm_provider: LLMProvider = LLMProvider.OPENAI
    llm_model: str = "gpt-4o"
    llm_temperature: float = 0.2
    
    # API Keys (from environment)
    openai_api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY")
    )
    anthropic_api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("ANTHROPIC_API_KEY")
    )
    
    # Module configs
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    repair: RepairConfig = field(default_factory=RepairConfig)
    recompose: RecomposeConfig = field(default_factory=RecomposeConfig)
    
    # Logging
    log_level: str = "INFO"
    log_all_outputs: bool = True
    
    # Recovery strategy order
    recovery_order: List[RecoveryStrategy] = field(
        default_factory=lambda: [
            RecoveryStrategy.REPAIR,
            RecoveryStrategy.REPLACE,
            RecoveryStrategy.RECOMPOSE,
            RecoveryStrategy.ESCALATE
        ]
    )
    
    def validate(self) -> bool:
        """Validate configuration"""
        if self.llm_provider == LLMProvider.OPENAI and not self.openai_api_key:
            raise ValueError("OpenAI API key required but not set")
        if self.llm_provider == LLMProvider.ANTHROPIC and not self.anthropic_api_key:
            raise ValueError("Anthropic API key required but not set")
        return True


# Default configuration instance
default_config = AEGISConfig()
