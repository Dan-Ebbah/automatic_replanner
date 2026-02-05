"""
PlannerAgent
============
Manages itinerary creation, step-by-step execution, and event-driven replanning.

Subscribes to:
  - WEATHER_CHANGED  — triggers replan with new weather constraints
  - REPLAN_REQUESTED — triggers replan for any external reason (e.g., traffic)

Publishes:
  - PLAN_UPDATED — after a successful replan
  - ACTION_COMPLETED — after each action step executes
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Set

from aegis.agent import AgentConfig, BaseAgent
from aegis.events import (
    Event,
    EventBus,
    EventType,
    PlanUpdatedEvent,
    WeatherChangedEvent,
)
from aegis.state import Action, ActionStatus, ExecutionState, PlanningGraph
from demos.activities import ActivityAction, WeatherCondition

logger = logging.getLogger(__name__)


@dataclass
class PlannerState:
    """Internal mutable state for the PlannerAgent."""
    execution_state: ExecutionState = field(default_factory=ExecutionState)
    current_plan: Optional[List[Set[Action]]] = None
    current_weather: WeatherCondition = WeatherCondition.SUNNY
    all_activities: List[ActivityAction] = field(default_factory=list)
    location: str = "Montreal"


class PlannerAgent(BaseAgent):
    """Event-driven planner that creates, executes, and replans itineraries."""

    def __init__(
        self,
        config: AgentConfig,
        event_bus: EventBus,
        all_activities: List[ActivityAction],
        initial_weather: WeatherCondition = WeatherCondition.SUNNY,
        location: str = "Montreal",
    ) -> None:
        super().__init__(config, event_bus)
        self.state = PlannerState(
            all_activities=list(all_activities),
            current_weather=initial_weather,
            location=location,
        )
        self._replan_signal = asyncio.Event()
        # Stores the event that triggered the latest replan request
        self._trigger_event: Optional[Event] = None
        # Set by the demo to define initial state and goal
        self._initial_state: Set[str] = set()
        self._goal: Set[str] = set()
        # Signals when the agent finishes execution (for awaiting in demos)
        self.done_event = asyncio.Event()
        self.success = False

    # -- lifecycle ------------------------------------------------------------

    async def setup(self) -> None:
        self.subscribe_to(EventType.WEATHER_CHANGED)
        self.subscribe_to(EventType.REPLAN_REQUESTED)

    async def run(self) -> None:
        """Create initial plan, then execute step-by-step, checking replan signal."""
        try:
            if not self.create_plan(self._initial_state, self._goal):
                print("  [PlannerAgent] Could not create initial plan!")
                self.success = False
                return

            self._print_plan("INITIAL PLAN")
            self.success = await self._execute_plan()

            if self.success:
                print("\n  [PlannerAgent] SUCCESS: All goals achieved!")
            else:
                print("\n  [PlannerAgent] INCOMPLETE: Not all goals met.")
        finally:
            self.done_event.set()

    async def _execute_plan(self) -> bool:
        """Execute the current plan, handling replan signals between steps."""
        executed = set()  # track action names already executed

        step_num = 0
        while step_num < len(self.state.current_plan) and self._running:
            # Check for pending replan signal before each step
            if self._replan_signal.is_set():
                self._replan_signal.clear()
                if not await self._do_replan():
                    print("  [PlannerAgent] Replan failed!")
                    return False
                step_num = 0
                executed = set()  # reset — the replan provides a fresh plan from current state
                continue

            actions = self.state.current_plan[step_num]
            # Filter out actions already completed (GraphPlan layers may overlap)
            new_actions = [a for a in actions if a.name not in executed]

            if new_actions:
                print(f"\n  [PlannerAgent] --- Step {step_num + 1} ---")
                for action in new_actions:
                    print(f"    Executing: {action.name} (agent: {action.agent_id})")
                    self.state.execution_state.action_statuses[action.name] = ActionStatus.EXECUTING
                    self.state.execution_state.current_propositions = action.apply(
                        self.state.execution_state.current_propositions
                    )
                    self.state.execution_state.action_statuses[action.name] = ActionStatus.COMPLETED
                    executed.add(action.name)
                    print(f"      -> Achieved: {action.positive_effects}")

                    await self.publish(Event(
                        event_type=EventType.ACTION_COMPLETED,
                        source_agent_id=self.agent_id,
                        payload={"action": action.name},
                    ))

            step_num += 1

            # Yield to let the event loop process incoming events
            await asyncio.sleep(0.2)

        # Check for a final replan signal after the last step
        if self._replan_signal.is_set():
            self._replan_signal.clear()
            if not await self._do_replan():
                return False
            return await self._execute_plan()

        return self._goal.issubset(self.state.execution_state.current_propositions)

    # -- event handling -------------------------------------------------------

    async def handle_event(self, event: Event) -> None:
        if event.event_type == EventType.WEATHER_CHANGED:
            new_cond = event.payload.get("new_condition", "")
            try:
                self.state.current_weather = WeatherCondition(new_cond)
            except ValueError:
                logger.warning("Unknown weather condition: %s", new_cond)
                return
            print(f"  [PlannerAgent] Received weather change -> {new_cond}")
            self._trigger_event = event
            self._replan_signal.set()

        elif event.event_type == EventType.REPLAN_REQUESTED:
            reason = event.payload.get("reason", "external request")
            print(f"  [PlannerAgent] Replan requested: {reason}")
            self._trigger_event = event
            self._replan_signal.set()

    # -- planning helpers -----------------------------------------------------

    def create_plan(self, initial_state: Set[str], goal: Set[str]) -> bool:
        self._initial_state = initial_state
        self._goal = goal
        self.state.execution_state = ExecutionState(
            current_propositions=initial_state.copy(),
            goal=goal,
        )

        available = [
            a for a in self.state.all_activities
            if a.weather_requirement == self.state.current_weather
        ]
        print(
            f"  [PlannerAgent] Planning with {len(available)} activities "
            f"for {self.state.current_weather.value} weather"
        )

        graph = PlanningGraph(
            initial_state=initial_state,
            goal=goal,
            available_actions=available,
        )
        if not graph.build_graph():
            return False

        plan = graph.extract_plan()
        if plan is None:
            return False

        self.state.current_plan = plan
        for step in plan:
            for action in step:
                self.state.execution_state.action_statuses[action.name] = ActionStatus.PENDING
        return True

    async def _do_replan(self) -> bool:
        """Rebuild the plan from the current execution state."""
        print("\n  [PlannerAgent] === REPLANNING ===")
        current_state = self.state.execution_state.current_propositions

        available = [
            a for a in self.state.all_activities
            if a.weather_requirement == self.state.current_weather
        ]
        print(
            f"  [PlannerAgent] Available actions for {self.state.current_weather.value}: "
            f"{[a.name for a in available]}"
        )

        graph = PlanningGraph(
            initial_state=current_state,
            goal=self._goal,
            available_actions=available,
        )
        if not graph.build_graph():
            print("  [PlannerAgent] Replan FAILED: no path to goal.")
            return False

        plan = graph.extract_plan()
        if plan is None:
            print("  [PlannerAgent] Replan FAILED: could not extract plan.")
            return False

        self.state.current_plan = plan
        for step in plan:
            for action in step:
                if action.name not in self.state.execution_state.action_statuses:
                    self.state.execution_state.action_statuses[action.name] = ActionStatus.PENDING

        self._print_plan("REPLANNED ITINERARY")

        # Publish PlanUpdatedEvent
        trigger_id = self._trigger_event.event_id if self._trigger_event else None
        await self.publish(PlanUpdatedEvent(
            event_type=None,  # overwritten by __post_init__
            source_agent_id=self.agent_id,
            plan_step_count=len(plan),
            trigger_event_id=trigger_id,
            payload={"steps": [[a.name for a in step] for step in plan]},
        ))
        return True

    # -- display helpers ------------------------------------------------------

    def _print_plan(self, title: str) -> None:
        if not self.state.current_plan:
            return
        print(f"\n  [{title}]")
        seen = set()
        step_counter = 0
        for step in self.state.current_plan:
            for action in step:
                if action.name not in seen:
                    seen.add(action.name)
                    step_counter += 1
                    print(f"    {step_counter}. {action.name.replace('_', ' ').title()}")
