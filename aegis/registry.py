"""
AEGIS Agent Registry
====================
Manages a registry of available agents that can be used in workflows.
"""

from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
import json

from .state import AgentInfo, AgentTool, Action


class ToolRegistry:
    def __init__(self):
        self.tools: dict[str, AgentTool] = {}
        self.tools_by_agent: dict[str, list[AgentTool]] = {}
        self.available_agents: set[str] = set()
        self.capability_providers: dict[str, set[str]] = {}  # capability -> agent_ids

    def register_tool(self, tool: AgentTool):
        """Register a tool and track its capability provider"""
        self.tools[tool.name] = tool

        if tool.agent_id not in self.tools_by_agent:
            self.tools_by_agent[tool.agent_id] = []
        self.tools_by_agent[tool.agent_id].append(tool)

        self.available_agents.add(tool.agent_id)

        # Track capability providers for automatic substitution
        if tool.capability:
            if tool.capability not in self.capability_providers:
                self.capability_providers[tool.capability] = set()
            self.capability_providers[tool.capability].add(tool.agent_id)

    def get_planning_actions(self) -> list[Action]:
        actions = []
        for agent_id in self.available_agents:
            for tool in self.tools_by_agent.get(agent_id, []):
                actions.append(tool.to_planning_action())
        return actions

    def get_llm_tools(self) -> list[dict]:
        specs = []
        for agent_id in self.available_agents:
            for tool in self.tools_by_agent.get(agent_id, []):
                specs.append(tool.to_llm_tool_spec())
        return specs

    def mark_agent_unavailable(self, agent_id: str):
        self.available_agents.discard(agent_id)

    def mark_agent_available(self, agent_id: str):
        if agent_id in self.tools_by_agent:
            self.available_agents.add(agent_id)

    def find_alternative_agents(self, failed_agent_id: str) -> Dict[str, set[str]]:
        """
        Find agents that can substitute for a failed agent's capabilities.

        Returns a dict mapping each lost capability to the set of alternative
        agent IDs that can provide it.

        This is used by graph planning to find alternative paths when an agent fails.
        """
        alternatives: Dict[str, set[str]] = {}

        failed_tools = self.tools_by_agent.get(failed_agent_id, [])
        failed_capabilities = {
            tool.capability for tool in failed_tools if tool.capability
        }

        for capability in failed_capabilities:
            providers = self.capability_providers.get(capability, set())
            # Find available agents that provide this capability (excluding failed one)
            alternative_agents = providers & self.available_agents - {failed_agent_id}
            if alternative_agents:
                alternatives[capability] = alternative_agents

        return alternatives

    def get_agents_for_capability(self, capability: str) -> set[str]:
        """Get all available agents that can provide a specific capability"""
        providers = self.capability_providers.get(capability, set())
        return providers & self.available_agents

    async def execute_tool(self, tool_name: str, **kwargs) -> dict[str, Any]:
        tool = self.tools.get(tool_name)
        if tool is None:
            raise ValueError(f"Unknown tool: {tool_name}")

        if tool.agent_id not in self.available_agents:
            raise RuntimeError(f"Agent {tool.agent_id} is not available")

        return await tool.execute(**kwargs)

class AgentRegistry:
    """
    Registry of available agents for workflow composition.
    
    Agents can be registered with:
    - Name and description
    - Role (researcher, analyzer, writer, validator, etc.)
    - The actual agent function
    - Input/output schemas
    - Performance metadata
    """
    
    def __init__(self):
        self._agents: Dict[str, AgentInfo] = {}
        self._register_default_agents()
    
    def _register_default_agents(self):
        """Register default utility agents"""
        
        # Fact Checker agent
        self.register(AgentInfo(
            name="fact_checker",
            description="Validates facts and detects hallucinations in content",
            role="validator",
            default_prompt="""You are a fact-checker. Review the provided content and identify any claims that appear to be:
1. Unverifiable or made-up
2. Contradicting known facts
3. Missing proper attribution

For each issue found, explain why it's problematic."""
        ))
        
        # Semantic Validator agent
        self.register(AgentInfo(
            name="semantic_validator",
            description="Checks if output aligns with the original task",
            role="validator",
            default_prompt="""You are a semantic validator. Check if the provided output properly addresses the original task.

Rate alignment from 1-5:
1 = Completely off-topic
5 = Perfect alignment

If alignment is below 4, explain what's missing or wrong."""
        ))
        
        # Format Fixer agent
        self.register(AgentInfo(
            name="format_fixer",
            description="Fixes format issues and ensures schema compliance",
            role="validator",
            default_prompt="""You are a format fixer. The provided content has format issues.
Convert it to the required format while preserving all important information."""
        ))
        
        # Summarizer agent
        self.register(AgentInfo(
            name="summarizer",
            description="Creates concise summaries of content",
            role="processor",
            default_prompt="""You are a summarizer. Create a concise summary of the provided content,
capturing all key points in a clear and organized manner."""
        ))
        
        # Generic Processor agent
        self.register(AgentInfo(
            name="generic_processor",
            description="General-purpose content processor",
            role="processor",
            default_prompt="""Process the provided content according to the task requirements.
Be thorough and accurate in your response."""
        ))
        
        # Checkpoint agent (passthrough with state saving)
        self.register(AgentInfo(
            name="checkpoint",
            description="Creates a checkpoint for rollback capability",
            role="utility",
            default_prompt=None,  # Special handling
            agent_func=lambda state: state  # Passthrough
        ))
        
        # Error Handler agent
        self.register(AgentInfo(
            name="error_handler",
            description="Handles errors gracefully and provides fallback responses",
            role="utility",
            default_prompt="""An error occurred in the workflow. Analyze the situation and provide:
1. A graceful fallback response
2. Explanation of what went wrong
3. Suggestions for resolution"""
        ))
    
    def register(self, agent_info: AgentInfo) -> None:
        """Register an agent"""
        self._agents[agent_info.name] = agent_info
    
    def register_function(
        self,
        name: str,
        func: Callable,
        description: str = "",
        role: str = "processor",
        **kwargs
    ) -> None:
        """Register an agent from a function"""
        
        agent_info = AgentInfo(
            name=name,
            description=description,
            role=role,
            agent_func=func,
            **kwargs
        )
        self.register(agent_info)
    
    def get_agent(self, name: str) -> Optional[AgentInfo]:
        """Get an agent by name"""
        return self._agents.get(name)
    
    def list_agents(self) -> List[AgentInfo]:
        """List all registered agents"""
        return list(self._agents.values())
    
    def list_by_role(self, role: str) -> List[AgentInfo]:
        """List agents by role"""
        return [a for a in self._agents.values() if a.role == role]
    
    def describe_all(self) -> str:
        """Get description of all agents (for LLM prompts)"""
        descriptions = []
        for agent in self._agents.values():
            descriptions.append(
                f"- {agent.name} ({agent.role}): {agent.description}"
            )
        return "\n".join(descriptions)
    
    def to_dict(self) -> Dict[str, Any]:
        """Export registry as dictionary"""
        return {
            name: info.to_dict()
            for name, info in self._agents.items()
        }
    
    def update_performance(
        self,
        agent_name: str,
        latency_ms: float,
        success: bool
    ) -> None:
        """Update agent performance metrics"""
        
        agent = self._agents.get(agent_name)
        if not agent:
            return
        
        # Update running average of latency
        if agent.avg_latency_ms == 0:
            agent.avg_latency_ms = latency_ms
        else:
            # Exponential moving average
            agent.avg_latency_ms = 0.9 * agent.avg_latency_ms + 0.1 * latency_ms
        
        # Update success rate
        if success:
            agent.success_rate = 0.95 * agent.success_rate + 0.05
        else:
            agent.success_rate = 0.95 * agent.success_rate


# Global default registry
default_registry = AgentRegistry()
