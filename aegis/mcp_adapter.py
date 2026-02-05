"""
AEGIS MCP Agent Adapter
========================
Wraps any MCP server as an AEGIS agent so it can publish/subscribe on the EventBus.

Usage::

    async def my_mcp_call(tool_name, params):
        # call a real MCP server here
        return {"status": "ok", "data": ...}

    adapter = MCPAgentAdapter(
        config=AgentConfig(agent_id="traffic", name="Traffic MCP"),
        event_bus=bus,
        mcp_call_fn=my_mcp_call,
        tool_specs=[
            MCPToolSpec(
                name="get_traffic",
                description="Fetch live traffic conditions",
                input_schema={"location": "str"},
                output_event_type=EventType.REPLAN_REQUESTED,
                output_transformer=lambda data: {"reason": f"traffic: {data}"},
            )
        ],
    )
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, Dict, List, Optional

from .agent import AgentConfig, BaseAgent
from .events import Event, EventBus, EventType

logger = logging.getLogger(__name__)


@dataclass
class MCPToolSpec:
    """Describes one MCP tool and how its output maps to an AEGIS event."""
    name: str
    description: str = ""
    input_schema: Dict[str, Any] = field(default_factory=dict)
    output_event_type: EventType = EventType.CUSTOM
    output_transformer: Optional[Callable[[Any], Dict[str, Any]]] = None


# Type alias for the MCP call seam
MCPCallFn = Callable[[str, Dict[str, Any]], Coroutine[Any, Any, Any]]


class MCPAgentAdapter(BaseAgent):
    """Adapter that wraps an MCP server as an AEGIS agent.

    Supports two patterns:
    * **Polling** — periodically calls each tool, detects changes, publishes events.
    * **Reactive** — subscribes to trigger events, calls tools in response.
    """

    def __init__(
        self,
        config: AgentConfig,
        event_bus: EventBus,
        mcp_call_fn: MCPCallFn,
        tool_specs: List[MCPToolSpec] | None = None,
        trigger_events: List[EventType] | None = None,
    ) -> None:
        super().__init__(config, event_bus)
        self._mcp_call_fn = mcp_call_fn
        self._tool_specs = tool_specs or []
        self._trigger_events = trigger_events or []
        # Last known outputs per tool (for change detection in polling mode)
        self._last_outputs: Dict[str, Any] = {}

    # -- lifecycle overrides --------------------------------------------------

    async def setup(self) -> None:
        for et in self._trigger_events:
            self.subscribe_to(et)

    async def run(self) -> None:
        """Poll MCP tools at the configured interval."""
        while self._running:
            for spec in self._tool_specs:
                await self._poll_tool(spec)
            await asyncio.sleep(self.config.poll_interval_seconds)

    async def handle_event(self, event: Event) -> None:
        """Reactive mode: call all tools when a trigger event arrives."""
        for spec in self._tool_specs:
            try:
                result = await self._mcp_call_fn(spec.name, event.payload)
                await self._maybe_publish(spec, result)
            except Exception:
                logger.exception(
                    "MCP tool %s failed on trigger event %s",
                    spec.name, event.event_id,
                )

    # -- internals ------------------------------------------------------------

    async def _poll_tool(self, spec: MCPToolSpec) -> None:
        try:
            result = await self._mcp_call_fn(spec.name, spec.input_schema)
        except Exception:
            logger.exception("MCP poll failed for tool %s", spec.name)
            return

        # Only publish when output changes
        if result != self._last_outputs.get(spec.name):
            self._last_outputs[spec.name] = result
            await self._maybe_publish(spec, result)

    async def _maybe_publish(self, spec: MCPToolSpec, result: Any) -> None:
        payload = (
            spec.output_transformer(result)
            if spec.output_transformer
            else {"mcp_tool": spec.name, "result": result}
        )
        event = Event(
            event_type=spec.output_event_type,
            source_agent_id=self.agent_id,
            payload=payload,
        )
        await self.publish(event)
