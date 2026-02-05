"""
Failure Injection Framework
===========================
Injects controlled failures into agent workflows for testing AEGIS.
"""

import random
import time
import functools
from typing import Dict, Any, Optional, List, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import os


class InjectionTrigger(Enum):
    """When to trigger the failure"""
    ALWAYS = "always"                    # Always inject
    RANDOM = "random"                    # Random with probability
    AFTER_N_CALLS = "after_n_calls"      # After N successful calls
    BEFORE_N_CALLS = "before_n_calls"    # Only for first N calls
    CONDITIONAL = "conditional"          # Based on state condition


class FailureMode(Enum):
    """Types of failures to inject"""
    CRASH = "crash"                      # Raise an exception
    TIMEOUT = "timeout"                  # Simulate timeout
    HALLUCINATION = "hallucination"      # Return fabricated information
    SEMANTIC_DRIFT = "semantic_drift"    # Return off-topic content
    EMPTY_OUTPUT = "empty_output"        # Return empty response
    MALFORMED_OUTPUT = "malformed_output"  # Return badly formatted output
    PARTIAL_OUTPUT = "partial_output"    # Return incomplete response


@dataclass
class InjectionConfig:
    """Configuration for a failure injection"""
    
    failure_mode: FailureMode
    trigger: InjectionTrigger = InjectionTrigger.ALWAYS
    probability: float = 1.0              # For RANDOM trigger
    n_calls: int = 1                      # For AFTER_N_CALLS / BEFORE_N_CALLS
    condition: Optional[Callable] = None  # For CONDITIONAL trigger
    custom_message: Optional[str] = None  # Custom error message
    delay_seconds: float = 0.0            # Delay before failure


@dataclass
class InjectionRecord:
    """Record of an injection event"""
    
    agent_name: str
    failure_mode: FailureMode
    triggered: bool
    timestamp: datetime = field(default_factory=datetime.now)
    details: Dict[str, Any] = field(default_factory=dict)


class FailureInjector:
    """
    Injects failures into agent functions for testing.
    
    Usage:
        injector = FailureInjector()
        
        # Schedule a failure
        injector.schedule(
            agent_name="analyze",
            config=InjectionConfig(
                failure_mode=FailureMode.HALLUCINATION,
                trigger=InjectionTrigger.ALWAYS
            )
        )
        
        # Wrap agent function
        wrapped_agent = injector.wrap(analyze_agent, "analyze")
        
        # Use wrapped agent in workflow
    """
    
    def __init__(self):
        self.schedules: Dict[str, InjectionConfig] = {}
        self.call_counts: Dict[str, int] = {}
        self.injection_log: List[InjectionRecord] = []
        self._llm = None
    
    @property
    def llm(self):
        """Lazy-load LLM for generating bad outputs"""
        if self._llm is None:
            self._llm = ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0.9,
                api_key=os.getenv("OPENAI_API_KEY")
            )
        return self._llm
    
    def schedule(
        self,
        agent_name: str,
        config: InjectionConfig
    ) -> None:
        """Schedule a failure for an agent"""
        self.schedules[agent_name] = config
        self.call_counts[agent_name] = 0
    
    def unschedule(self, agent_name: str) -> None:
        """Remove scheduled failure for an agent"""
        if agent_name in self.schedules:
            del self.schedules[agent_name]
    
    def clear_all(self) -> None:
        """Clear all scheduled failures"""
        self.schedules.clear()
        self.call_counts.clear()
    
    def wrap(
        self,
        agent_func: Callable,
        agent_name: str
    ) -> Callable:
        """
        Wrap an agent function to inject failures.
        
        Args:
            agent_func: The original agent function
            agent_name: Name of the agent (for matching scheduled failures)
        
        Returns:
            Wrapped function that may inject failures
        """
        
        @functools.wraps(agent_func)
        def wrapped(state: Dict[str, Any]) -> Dict[str, Any]:
            # Increment call count
            self.call_counts[agent_name] = self.call_counts.get(agent_name, 0) + 1
            
            # Check if we should trigger failure
            if self._should_trigger(agent_name, state):
                return self._inject_failure(agent_name, state)
            
            # Normal execution
            return agent_func(state)
        
        return wrapped
    
    def _should_trigger(
        self,
        agent_name: str,
        state: Dict[str, Any]
    ) -> bool:
        """Determine if failure should be triggered"""
        
        if agent_name not in self.schedules:
            return False
        
        config = self.schedules[agent_name]
        call_count = self.call_counts.get(agent_name, 0)
        
        if config.trigger == InjectionTrigger.ALWAYS:
            return True
        
        elif config.trigger == InjectionTrigger.RANDOM:
            return random.random() < config.probability
        
        elif config.trigger == InjectionTrigger.AFTER_N_CALLS:
            return call_count > config.n_calls
        
        elif config.trigger == InjectionTrigger.BEFORE_N_CALLS:
            return call_count <= config.n_calls
        
        elif config.trigger == InjectionTrigger.CONDITIONAL:
            if config.condition:
                return config.condition(state)
            return False
        
        return False
    
    def _inject_failure(
        self,
        agent_name: str,
        state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Inject the configured failure"""
        
        config = self.schedules[agent_name]
        
        # Optional delay
        if config.delay_seconds > 0:
            time.sleep(config.delay_seconds)
        
        # Log the injection
        self.injection_log.append(InjectionRecord(
            agent_name=agent_name,
            failure_mode=config.failure_mode,
            triggered=True,
            details={"state_keys": list(state.keys())}
        ))
        
        # Inject based on failure mode
        if config.failure_mode == FailureMode.CRASH:
            return self._inject_crash(config)
        
        elif config.failure_mode == FailureMode.TIMEOUT:
            return self._inject_timeout(config)
        
        elif config.failure_mode == FailureMode.HALLUCINATION:
            return self._inject_hallucination(state, agent_name)
        
        elif config.failure_mode == FailureMode.SEMANTIC_DRIFT:
            return self._inject_semantic_drift(state, agent_name)
        
        elif config.failure_mode == FailureMode.EMPTY_OUTPUT:
            return self._inject_empty_output(agent_name)
        
        elif config.failure_mode == FailureMode.MALFORMED_OUTPUT:
            return self._inject_malformed_output(agent_name)
        
        elif config.failure_mode == FailureMode.PARTIAL_OUTPUT:
            return self._inject_partial_output(state, agent_name)
        
        else:
            raise ValueError(f"Unknown failure mode: {config.failure_mode}")
    
    def _inject_crash(self, config: InjectionConfig) -> Dict[str, Any]:
        """Inject a crash (raise exception)"""
        message = config.custom_message or "Injected crash failure"
        raise RuntimeError(message)
    
    def _inject_timeout(self, config: InjectionConfig) -> Dict[str, Any]:
        """Inject a timeout"""
        # Sleep for a very long time (will likely be killed by actual timeout)
        time.sleep(300)  # 5 minutes
        return {}
    
    def _inject_hallucination(
        self,
        state: Dict[str, Any],
        agent_name: str
    ) -> Dict[str, Any]:
        """Inject hallucinated content"""
        
        topic = state.get("topic", "the subject")
        
        prompt = f"""Generate a plausible-sounding but COMPLETELY FABRICATED response about "{topic}".

Include:
- Made-up statistics with specific numbers
- Invented expert quotes with fake names
- Fictional organizations or studies
- False historical claims

Make it sound authoritative but ensure the information is NOT REAL.
Write 2-3 paragraphs."""

        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            hallucinated_content = response.content
        except:
            hallucinated_content = f"""According to the prestigious Johnson Institute for Advanced Studies, 
            recent research shows that {topic} has increased by 847% since 2019. 
            Dr. Margaret Thornberry, lead researcher, states: "These findings are unprecedented 
            and will reshape our understanding entirely." The Global Commission on {topic} 
            has allocated $4.7 billion for further investigation."""
        
        # Return in expected format based on agent name
        output_key = self._get_output_key(agent_name)
        return {output_key: hallucinated_content}
    
    def _inject_semantic_drift(
        self,
        state: Dict[str, Any],
        agent_name: str
    ) -> Dict[str, Any]:
        """Inject semantically drifted content (off-topic)"""
        
        topic = state.get("topic", "the subject")
        
        # Generate completely off-topic content
        off_topics = [
            "the history of cheese-making in medieval France",
            "underwater basket weaving techniques",
            "the mating habits of Antarctic penguins",
            "the philosophy of ancient Sumerian pottery",
            "competitive speed knitting championships"
        ]
        
        off_topic = random.choice(off_topics)
        
        prompt = f"""Write a detailed, informative response about {off_topic}.
        
Do NOT mention "{topic}" at all.
Write 2-3 paragraphs about {off_topic} instead."""

        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            drifted_content = response.content
        except:
            drifted_content = f"""The art of {off_topic} has fascinated scholars for centuries. 
            Recent developments in this field have shown remarkable progress..."""
        
        output_key = self._get_output_key(agent_name)
        return {output_key: drifted_content}
    
    def _inject_empty_output(self, agent_name: str) -> Dict[str, Any]:
        """Inject empty output"""
        output_key = self._get_output_key(agent_name)
        return {output_key: ""}
    
    def _inject_malformed_output(self, agent_name: str) -> Dict[str, Any]:
        """Inject malformed output"""
        output_key = self._get_output_key(agent_name)
        
        # Return various types of malformed content
        malformed_options = [
            "```json\n{\"incomplete\": true",  # Unclosed JSON
            "ERROR: NULL POINTER EXCEPTION at 0x7fff",
            "<xml><broken><tag>",
            "{'python': 'dict' 'missing': 'comma'}",
            "[REDACTED] [REDACTED] [REDACTED]",
        ]
        
        return {output_key: random.choice(malformed_options)}
    
    def _inject_partial_output(
        self,
        state: Dict[str, Any],
        agent_name: str
    ) -> Dict[str, Any]:
        """Inject partial/incomplete output"""
        
        topic = state.get("topic", "the subject")
        
        partial_content = f"""Regarding {topic}, there are several important points to consider:

1. The first key aspect is that... [CONTENT TRUNCATED]

2. Additionally, research has shown... 

ERROR: Maximum token limit reached. Response incomplete."""
        
        output_key = self._get_output_key(agent_name)
        return {output_key: partial_content}
    
    def _get_output_key(self, agent_name: str) -> str:
        """Get the expected output key for an agent"""
        
        # Map agent names to their output keys
        output_keys = {
            "research": "research_data",
            "analyze": "analysis",
            "summarize": "summary",
            "research_agent": "research_data",
            "analyze_agent": "analysis",
            "summarize_agent": "summary"
        }
        
        return output_keys.get(agent_name, "output")
    
    def get_injection_log(self) -> List[InjectionRecord]:
        """Get the log of all injections"""
        return self.injection_log
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get injection statistics"""
        
        total = len(self.injection_log)
        by_mode = {}
        by_agent = {}
        
        for record in self.injection_log:
            mode = record.failure_mode.value
            by_mode[mode] = by_mode.get(mode, 0) + 1
            by_agent[record.agent_name] = by_agent.get(record.agent_name, 0) + 1
        
        return {
            "total_injections": total,
            "by_failure_mode": by_mode,
            "by_agent": by_agent
        }


# ============================================================================
# Convenience Functions
# ============================================================================

def create_hallucination_injection(probability: float = 1.0) -> InjectionConfig:
    """Create a hallucination injection config"""
    return InjectionConfig(
        failure_mode=FailureMode.HALLUCINATION,
        trigger=InjectionTrigger.RANDOM if probability < 1.0 else InjectionTrigger.ALWAYS,
        probability=probability
    )


def create_crash_injection(message: str = "Simulated crash") -> InjectionConfig:
    """Create a crash injection config"""
    return InjectionConfig(
        failure_mode=FailureMode.CRASH,
        trigger=InjectionTrigger.ALWAYS,
        custom_message=message
    )


def create_random_failure(probability: float = 0.5) -> InjectionConfig:
    """Create a random failure injection"""
    failure_mode = random.choice(list(FailureMode))
    return InjectionConfig(
        failure_mode=failure_mode,
        trigger=InjectionTrigger.RANDOM,
        probability=probability
    )
