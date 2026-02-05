"""
AEGIS Event Bus & Event Types
==============================
Async pub/sub system for inter-agent communication.

Event flow:
    Agent publishes Event -> EventBus dispatches to subscribers -> Handlers called concurrently
"""

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Coroutine, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


# =============================================================================
# Event Types
# =============================================================================

class EventType(Enum):
    WEATHER_CHANGED = auto()
    REPLAN_REQUESTED = auto()
    PLAN_UPDATED = auto()
    ACTION_COMPLETED = auto()
    ACTION_FAILED = auto()
    AGENT_REGISTERED = auto()
    AGENT_UNAVAILABLE = auto()
    CUSTOM = auto()


# =============================================================================
# Event Data Classes
# =============================================================================

@dataclass
class Event:
    """Base event published through the EventBus."""
    event_type: EventType
    source_agent_id: str
    payload: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    def __str__(self) -> str:
        return (
            f"[{self.event_type.name}] from={self.source_agent_id} "
            f"id={self.event_id} at={self.timestamp:%H:%M:%S}"
        )


@dataclass
class WeatherChangedEvent(Event):
    """Published when a weather agent detects a condition change."""
    previous_condition: Optional[str] = None
    new_condition: Optional[str] = None
    location: Optional[str] = None

    def __post_init__(self):
        self.event_type = EventType.WEATHER_CHANGED


@dataclass
class ReplanRequestedEvent(Event):
    """Published when an agent requests the planner to replan."""
    reason: str = ""

    def __post_init__(self):
        self.event_type = EventType.REPLAN_REQUESTED


@dataclass
class PlanUpdatedEvent(Event):
    """Published after the planner successfully replans."""
    plan_step_count: int = 0
    trigger_event_id: Optional[str] = None

    def __post_init__(self):
        self.event_type = EventType.PLAN_UPDATED


# =============================================================================
# Event Bus
# =============================================================================

# Handler signature: async (Event) -> None
EventHandler = Callable[[Event], Coroutine[Any, Any, None]]


class EventBus:
    """Async publish/subscribe bus for inter-agent communication."""

    def __init__(self) -> None:
        self._subscribers: Dict[EventType, List[EventHandler]] = {}
        self._event_log: List[Event] = []

    # -- subscriptions --------------------------------------------------------

    def subscribe(self, event_type: EventType, handler: EventHandler) -> None:
        self._subscribers.setdefault(event_type, []).append(handler)
        logger.debug("Subscribed handler %s to %s", handler, event_type.name)

    def unsubscribe(self, event_type: EventType, handler: EventHandler) -> None:
        handlers = self._subscribers.get(event_type, [])
        if handler in handlers:
            handlers.remove(handler)

    # -- publishing -----------------------------------------------------------

    async def publish(self, event: Event) -> None:
        """Publish an event, dispatching to all subscribers concurrently."""
        self._event_log.append(event)
        logger.info("Event published: %s", event)

        handlers = self._subscribers.get(event.event_type, [])
        if not handlers:
            return

        # Dispatch to all handlers concurrently
        results = await asyncio.gather(
            *(h(event) for h in handlers),
            return_exceptions=True,
        )
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(
                    "Handler %s raised %s for event %s",
                    handlers[i], result, event.event_id,
                )

    # -- introspection --------------------------------------------------------

    @property
    def event_log(self) -> List[Event]:
        return list(self._event_log)

    def get_events_by_type(self, event_type: EventType) -> List[Event]:
        return [e for e in self._event_log if e.event_type == event_type]

    def print_event_log(self) -> None:
        """Print the full event communication trace."""
        print("\n--- Event Log ---")
        for event in self._event_log:
            print(f"  {event}")
        print(f"--- {len(self._event_log)} events total ---")
