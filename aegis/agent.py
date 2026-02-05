"""
AEGIS Base Agent
================
Abstract base class for all AEGIS agents with lifecycle management.

Lifecycle:  start() -> setup() -> run() -> stop()
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

from .events import Event, EventBus, EventType

logger = logging.getLogger(__name__)


@dataclass
class AgentConfig:
    """Configuration for an AEGIS agent."""
    agent_id: str
    name: str
    description: str = ""
    poll_interval_seconds: float = 5.0
    enabled: bool = True


class BaseAgent(ABC):
    """Abstract base agent with event-bus integration and lifecycle hooks."""

    def __init__(self, config: AgentConfig, event_bus: EventBus) -> None:
        self.config = config
        self.event_bus = event_bus
        self._running = False
        self._task: Optional[asyncio.Task] = None

    # -- properties -----------------------------------------------------------

    @property
    def agent_id(self) -> str:
        return self.config.agent_id

    @property
    def name(self) -> str:
        return self.config.name

    # -- lifecycle ------------------------------------------------------------

    async def start(self) -> None:
        """Initialise the agent: run setup, register with the bus, start run loop."""
        if not self.config.enabled:
            logger.info("Agent %s is disabled, skipping start.", self.agent_id)
            return

        await self.setup()
        self._running = True
        self._task = asyncio.create_task(self._run_wrapper())
        logger.info("Agent %s started.", self.agent_id)

    async def stop(self) -> None:
        """Signal the agent to stop and wait for its task to finish."""
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Agent %s stopped.", self.agent_id)

    # -- hooks for subclasses -------------------------------------------------

    async def setup(self) -> None:
        """Called once before run(). Override to subscribe to events."""

    @abstractmethod
    async def run(self) -> None:
        """Main agent loop. Must respect self._running to allow clean shutdown."""

    @abstractmethod
    async def handle_event(self, event: Event) -> None:
        """Called by the EventBus when a subscribed event arrives."""

    # -- convenience helpers --------------------------------------------------

    def subscribe_to(self, event_type: EventType) -> None:
        """Register interest in *event_type* (call in setup())."""
        self.event_bus.subscribe(event_type, self.handle_event)

    async def publish(self, event: Event) -> None:
        """Publish an event, auto-setting source_agent_id."""
        event.source_agent_id = self.agent_id
        await self.event_bus.publish(event)

    # -- internal -------------------------------------------------------------

    async def _run_wrapper(self) -> None:
        """Wraps run() so one agent crash doesn't kill the system."""
        try:
            await self.run()
        except asyncio.CancelledError:
            raise  # propagate cancellation cleanly
        except Exception:
            logger.exception("Agent %s crashed in run().", self.agent_id)
