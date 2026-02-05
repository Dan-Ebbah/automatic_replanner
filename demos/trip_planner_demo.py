from dataclasses import dataclass, field
from typing import Set, Dict, Any, Optional, List

import sys
import os

from demos.activities import ActivityAction
from demos.weather_service import MockWeatherService, WeatherData

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aegis.state import (
    Action, AgentTool, ExecutionState, PlanningGraph, ActionStatus
)
from aegis.registry import ToolRegistry


def create_trip_planner_registry() -> ToolRegistry:
    """
    Create a registry with multiple travel agents that have overlapping capabilities.
    This allows the system to find alternatives when one agent fails.
    """
    registry = ToolRegistry()

    # -------------------------------------------------------------------------
    # Expedia Agent: Full-service travel agent
    # -------------------------------------------------------------------------
    registry.register_tool(AgentTool(
        name="expedia_search_flights",
        agent_id="expedia",
        description="Search for flights on Expedia",
        pre_conditions=frozenset({"trip_request", "travel_dates"}),
        positive_effects=frozenset({"flight_options", "expedia_session"}),
        negative_effects=frozenset(),
        capability="flight_search"
    ))

    registry.register_tool(AgentTool(
        name="expedia_book_flight",
        agent_id="expedia",
        description="Book a flight on Expedia",
        pre_conditions=frozenset({"flight_options", "expedia_session", "payment_info"}),
        positive_effects=frozenset({"flight_booked", "flight_confirmation"}),
        negative_effects=frozenset({"flight_options"}),
        capability="flight_booking"
    ))

    registry.register_tool(AgentTool(
        name="expedia_search_hotels",
        agent_id="expedia",
        description="Search for hotels on Expedia",
        pre_conditions=frozenset({"trip_request", "travel_dates"}),
        positive_effects=frozenset({"hotel_options", "expedia_session"}),
        negative_effects=frozenset(),
        capability="hotel_search"
    ))

    registry.register_tool(AgentTool(
        name="expedia_book_hotel",
        agent_id="expedia",
        description="Book a hotel on Expedia",
        pre_conditions=frozenset({"hotel_options", "expedia_session", "payment_info"}),
        positive_effects=frozenset({"hotel_booked", "hotel_confirmation"}),
        negative_effects=frozenset({"hotel_options"}),
        capability="hotel_booking"
    ))

    # -------------------------------------------------------------------------
    # Kayak Agent: Flight specialist
    # -------------------------------------------------------------------------
    registry.register_tool(AgentTool(
        name="kayak_search_flights",
        agent_id="kayak",
        description="Search for flights on Kayak",
        pre_conditions=frozenset({"trip_request", "travel_dates"}),
        positive_effects=frozenset({"flight_options", "kayak_session"}),
        negative_effects=frozenset(),
        capability="flight_search"
    ))

    registry.register_tool(AgentTool(
        name="kayak_book_flight",
        agent_id="kayak",
        description="Book a flight through Kayak",
        pre_conditions=frozenset({"flight_options", "kayak_session", "payment_info"}),
        positive_effects=frozenset({"flight_booked", "flight_confirmation"}),
        negative_effects=frozenset({"flight_options"}),
        capability="flight_booking"
    ))

    # -------------------------------------------------------------------------
    # Booking.com Agent: Hotel specialist
    # -------------------------------------------------------------------------
    registry.register_tool(AgentTool(
        name="bookingcom_search_hotels",
        agent_id="booking_com",
        description="Search for hotels on Booking.com",
        pre_conditions=frozenset({"trip_request", "travel_dates"}),
        positive_effects=frozenset({"hotel_options", "bookingcom_session"}),
        negative_effects=frozenset(),
        capability="hotel_search"
    ))

    registry.register_tool(AgentTool(
        name="bookingcom_book_hotel",
        agent_id="booking_com",
        description="Book a hotel on Booking.com",
        pre_conditions=frozenset({"hotel_options", "bookingcom_session", "payment_info"}),
        positive_effects=frozenset({"hotel_booked", "hotel_confirmation"}),
        negative_effects=frozenset({"hotel_options"}),
        capability="hotel_booking"
    ))

    # -------------------------------------------------------------------------
    # Enterprise Agent: Car rental
    # -------------------------------------------------------------------------
    registry.register_tool(AgentTool(
        name="enterprise_search_cars",
        agent_id="enterprise",
        description="Search for rental cars at Enterprise",
        pre_conditions=frozenset({"trip_request", "travel_dates"}),
        positive_effects=frozenset({"car_options", "enterprise_session"}),
        negative_effects=frozenset(),
        capability="car_search"
    ))

    registry.register_tool(AgentTool(
        name="enterprise_book_car",
        agent_id="enterprise",
        description="Book a rental car at Enterprise",
        pre_conditions=frozenset({"car_options", "enterprise_session", "payment_info"}),
        positive_effects=frozenset({"car_booked", "car_confirmation"}),
        negative_effects=frozenset({"car_options"}),
        capability="car_booking"
    ))

    # -------------------------------------------------------------------------
    # Hertz Agent: Alternative car rental
    # -------------------------------------------------------------------------
    registry.register_tool(AgentTool(
        name="hertz_search_cars",
        agent_id="hertz",
        description="Search for rental cars at Hertz",
        pre_conditions=frozenset({"trip_request", "travel_dates"}),
        positive_effects=frozenset({"car_options", "hertz_session"}),
        negative_effects=frozenset(),
        capability="car_search"
    ))

    registry.register_tool(AgentTool(
        name="hertz_book_car",
        agent_id="hertz",
        description="Book a rental car at Hertz",
        pre_conditions=frozenset({"car_options", "hertz_session", "payment_info"}),
        positive_effects=frozenset({"car_booked", "car_confirmation"}),
        negative_effects=frozenset({"car_options"}),
        capability="car_booking"
    ))

    # -------------------------------------------------------------------------
    # Trip Summary Agent: Creates final itinerary
    # -------------------------------------------------------------------------
    registry.register_tool(AgentTool(
        name="create_itinerary",
        agent_id="itinerary_agent",
        description="Create a complete trip itinerary",
        pre_conditions=frozenset({"flight_confirmation", "hotel_confirmation", "car_confirmation"}),
        positive_effects=frozenset({"trip_itinerary", "trip_complete"}),
        negative_effects=frozenset(),
        capability="itinerary_creation"
    ))

    return registry


# =============================================================================
# Plan Execution Simulator
# =============================================================================

@dataclass
class TripPlannerController:
    """Simulates execution of trip planning with self-healing on agent failure"""

    registry: ToolRegistry
    execution_state: ExecutionState = field(default_factory=ExecutionState)
    current_plan: Optional[List[Set[Action]]] = None

    def create_plan(self, initial_state: Set[str], goal: Set[str]) -> bool:
        self.execution_state = ExecutionState(
            current_propositions=initial_state.copy(),
            goal=goal
        )

        available_actions = self.registry.get_planning_actions()

        graph = PlanningGraph(
            initial_state=initial_state,
            goal=goal,
            available_actions=available_actions
        )

        if not graph.build_graph():
            print("ERROR: No plan exists to reach the goal!")
            return False

        self.current_plan = graph.extract_plan()
        if self.current_plan is None:
            print("ERROR: Plan extraction failed!")
            return False

        # Initialize action statuses
        for step in self.current_plan:
            for action in step:
                self.execution_state.action_statuses[action.name] = ActionStatus.PENDING

        return True

    def execute_plan(self) -> bool:
        """Execute the plan step by step"""
        if not self.current_plan:
            return False

        step_num = 0
        while step_num < len(self.current_plan):
            step = self.current_plan[step_num]
            print(f"\n--- Step {step_num + 1} ---")
            print(f"Actions to execute: {[a.name for a in step]}")

            # Check agent availability
            for action in step:
                if action.agent_id not in self.registry.available_agents:
                    print(f"FAILURE: Agent '{action.agent_id}' is unavailable!")
                    if not self._handle_failure(action.agent_id):
                        return False
                    step_num = 0  # Restart with new plan
                    continue

            # Execute all actions in this step
            for action in step:
                self._execute_action(action)

            step_num += 1

        # Check if goal achieved
        if self.execution_state.goal.issubset(self.execution_state.current_propositions):
            print("\n" + "=" * 60)
            print("SUCCESS: Trip planning completed!")
            print("=" * 60)
            return True

        return False

    def _execute_action(self, action: Action):
        """Simulate executing an action"""
        self.execution_state.action_statuses[action.name] = ActionStatus.EXECUTING
        print(f"  Executing: {action.name} (agent: {action.agent_id})")

        # Apply effects
        self.execution_state.current_propositions = action.apply(
            self.execution_state.current_propositions
        )
        self.execution_state.action_statuses[action.name] = ActionStatus.COMPLETED
        print(f"    -> Achieved: {action.positive_effects}")

    def _handle_failure(self, failed_agent_id: str) -> bool:
        """Handle agent failure by replanning"""
        print(f"\n{'!' * 60}")
        print(f"HANDLING FAILURE: Agent '{failed_agent_id}' went offline")
        print(f"{'!' * 60}")

        # Mark agent unavailable
        self.registry.mark_agent_unavailable(failed_agent_id)

        # Find alternatives
        alternatives = self.registry.find_alternative_agents(failed_agent_id)
        if alternatives:
            print(f"Alternative agents found: {alternatives}")
        else:
            print("No direct alternatives, attempting full replan...")

        # Replan from current state
        print(f"\nCurrent state: {self.execution_state.current_propositions}")
        print(f"Goal: {self.execution_state.goal}")

        available_actions = self.registry.get_planning_actions()
        print(f"Available actions: {[a.name for a in available_actions]}")

        graph = PlanningGraph(
            initial_state=self.execution_state.current_propositions,
            goal=self.execution_state.goal,
            available_actions=available_actions
        )

        if not graph.build_graph():
            print("REPAIR FAILED: No alternative path to goal!")
            return False

        new_plan = graph.extract_plan()
        if new_plan is None:
            print("REPAIR FAILED: Could not extract plan!")
            return False

        print(f"\nREPAIR SUCCESSFUL: Found {len(new_plan)}-step alternative plan")
        self.current_plan = new_plan

        # Update action statuses
        for step in new_plan:
            for action in step:
                if action.name not in self.execution_state.action_statuses:
                    self.execution_state.action_statuses[action.name] = ActionStatus.PENDING

        return True

@dataclass
class ItineraryController(TripPlannerController):
    """Controller for activity-based itineraries with weather awareness"""
    weather_service: MockWeatherService = None
    location: str = "Montreal"
    activity_actions: List[ActivityAction] = field(default_factory=list)  # Store ActivityAction objects

    def set_activities(self, activities: List[ActivityAction]):
        """Set the available activity actions for planning"""
        self.activity_actions = activities

    def create_activity_plan(self, initial_state: Set[str], goal: Set[str]) -> bool:
        """Create a plan using ActivityAction objects directly"""
        self.execution_state = ExecutionState(
            current_propositions=initial_state.copy(),
            goal=goal
        )

        # Filter activities by current weather
        current_weather = self.weather_service.get_current_weather(self.location)
        available_activities = [
            a for a in self.activity_actions
            if a.weather_requirement == current_weather.condition
        ]

        print(f"Planning with {len(available_activities)} activities for {current_weather.condition.value} weather")

        graph = PlanningGraph(
            initial_state=initial_state,
            goal=goal,
            available_actions=available_activities
        )

        if not graph.build_graph():
            print("ERROR: No plan exists to reach the goal!")
            return False

        self.current_plan = graph.extract_plan()
        if self.current_plan is None:
            print("ERROR: Plan extraction failed!")
            return False

        # Initialize action statuses
        for step in self.current_plan:
            for action in step:
                self.execution_state.action_statuses[action.name] = ActionStatus.PENDING

        return True

    def execute_with_weather_monitoring(self) -> bool:
        """Execute plan with weather checks before each step"""
        if not self.current_plan:
            return False

        step_num = 0
        replan_count = 0
        max_replans = 5  # Prevent infinite replan loops

        while step_num < len(self.current_plan):
            actions = self.current_plan[step_num]
            print(f"\n--- Step {step_num + 1} ---")
            print(f"Actions to execute: {[a.name for a in actions]}")

            needs_replan = False

            for action in actions:
                current_weather = self.weather_service.get_current_weather(location=self.location)

                # Check if action's weather requirement is satisfied
                if isinstance(action, ActivityAction):
                    if not action.is_applicable_with_weather(
                        self.execution_state.current_propositions,
                        current_weather.condition
                    ):
                        print(f"WEATHER CONFLICT: '{current_weather.condition.value}' not suitable for '{action.name}'")

                        if replan_count >= max_replans:
                            print("ERROR: Maximum replan attempts reached!")
                            return False

                        if not self._replan_for_weather(failed_action=action, weather=current_weather):
                            print("ERROR: Could not find alternative plan!")
                            return False

                        replan_count += 1
                        needs_replan = True
                        break

            if needs_replan:
                step_num = 0  # Restart with new plan
                continue

            # Execute all actions in this step (weather is OK)
            for action in actions:
                self._execute_action(action)

            step_num += 1

        # Check if goal achieved
        if self.execution_state.goal.issubset(self.execution_state.current_propositions):
            print("\n" + "=" * 60)
            print("SUCCESS: Itinerary completed with weather adaptation!")
            print("=" * 60)
            return True

        return False

    def _replan_for_weather(self, failed_action: Action, weather: WeatherData) -> bool:
        """Replan when weather constraint is violated"""
        print(f"\n{'!' * 60}")
        print(f"HANDLING WEATHER CHANGE: {failed_action.name} cannot proceed")
        print(f"Current weather: {weather.condition.value}, temp: {weather.temperature}°C")
        print(f"{'!' * 60}")

        # Get current state
        current_state = self.execution_state.current_propositions
        print(f"\nCurrent state: {current_state}")
        print(f"Goal: {self.execution_state.goal}")

        # Filter ActivityActions by current weather
        available_actions = [
            a for a in self.activity_actions
            if a.weather_requirement == weather.condition
        ]

        print(f"Available actions for {weather.condition.value} weather: {[a.name for a in available_actions]}")

        # Build new planning graph from current state
        graph = PlanningGraph(
            initial_state=current_state,
            goal=self.execution_state.goal,
            available_actions=available_actions
        )

        if not graph.build_graph():
            print("REPLAN FAILED: No path to goal with current weather!")
            return False

        new_plan = graph.extract_plan()
        if new_plan is None:
            print("REPLAN FAILED: Could not extract plan!")
            return False

        print(f"\nREPLAN SUCCESSFUL: Found {len(new_plan)}-step alternative plan")
        print("New itinerary:")
        for i, step in enumerate(new_plan):
            for action in step:
                print(f"  {i+1}. {action.name.replace('_', ' ').title()}")

        self.current_plan = new_plan

        # Update action statuses for new plan
        for step in new_plan:
            for action in step:
                if action.name not in self.execution_state.action_statuses:
                    self.execution_state.action_statuses[action.name] = ActionStatus.PENDING

        return True



# =============================================================================
# Demo Scenarios
# =============================================================================

def print_header(title: str):
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


def print_plan(plan: List[Set[Action]]):
    """Pretty print a plan"""
    for i, step in enumerate(plan):
        actions = [f"{a.name} ({a.agent_id})" for a in step]
        print(f"  Step {i + 1}: {actions}")


def demo_successful_trip():
    print_header("DEMO 1: Successful Trip Planning")

    registry = create_trip_planner_registry()
    controller = TripPlannerController(registry=registry)

    # User's initial information
    initial_state = {
        "trip_request",
        "travel_dates",
        "payment_info",
    }

    # Goal: Complete trip with all confirmations
    goal = {"trip_complete"}

    print("\nInitial state:", initial_state)
    print("Goal:", goal)
    print("\nAvailable agents:", registry.available_agents)

    # Create and show initial plan
    if controller.create_plan(initial_state, goal):
        print("\nInitial Plan:")
        print_plan(controller.current_plan)

        print("\n--- Executing Plan ---")
        controller.execute_plan()


def demo_agent_failure_recovery():
    """Demo 2: Agent failure and automatic recovery"""
    print_header("DEMO 2: Agent Failure & Recovery")

    registry = create_trip_planner_registry()
    controller = TripPlannerController(registry=registry)

    initial_state = {
        "trip_request",
        "travel_dates",
        "payment_info",
    }
    goal = {"trip_complete"}

    print("\nInitial state:", initial_state)
    print("Goal:", goal)

    if controller.create_plan(initial_state, goal):
        print("\nInitial Plan:")
        print_plan(controller.current_plan)

        # Simulate Expedia going down before execution
        print("\n" + "!" * 70)
        print(" SIMULATING FAILURE: Expedia service goes offline!")
        print("!" * 70)
        registry.mark_agent_unavailable("expedia")

        print("\n--- Executing Plan (with failure handling) ---")
        controller.execute_plan()


def demo_multiple_failures():
    """Demo 3: Multiple agent failures"""
    print_header("DEMO 3: Multiple Agent Failures")

    registry = create_trip_planner_registry()
    controller = TripPlannerController(registry=registry)

    initial_state = {
        "trip_request",
        "travel_dates",
        "payment_info",
    }
    goal = {"trip_complete"}

    print("\nInitial state:", initial_state)
    print("Goal:", goal)

    if controller.create_plan(initial_state, goal):
        print("\nInitial Plan:")
        print_plan(controller.current_plan)

        # Simulate multiple failures
        print("\n" + "!" * 70)
        print(" SIMULATING FAILURES: Expedia AND Enterprise go offline!")
        print("!" * 70)
        registry.mark_agent_unavailable("expedia")
        registry.mark_agent_unavailable("enterprise")

        print("\nRemaining agents:", registry.available_agents)

        print("\n--- Executing Plan (with multiple failure handling) ---")
        controller.execute_plan()


def demo_unrecoverable_failure():
    """Demo 4: Unrecoverable failure (no alternatives)"""
    print_header("DEMO 4: Unrecoverable Failure")

    registry = create_trip_planner_registry()
    controller = TripPlannerController(registry=registry)

    initial_state = {
        "trip_request",
        "travel_dates",
        "payment_info",
    }
    goal = {"trip_complete"}

    print("\nInitial state:", initial_state)
    print("Goal:", goal)

    if controller.create_plan(initial_state, goal):
        print("\nInitial Plan:")
        print_plan(controller.current_plan)

        # Simulate ALL car rental agents going down
        print("\n" + "!" * 70)
        print(" SIMULATING CATASTROPHIC FAILURE: ALL car rental agents offline!")
        print("!" * 70)
        registry.mark_agent_unavailable("enterprise")
        registry.mark_agent_unavailable("hertz")

        # Also take down itinerary agent
        registry.mark_agent_unavailable("itinerary_agent")

        print("\nRemaining agents:", registry.available_agents)

        print("\n--- Executing Plan ---")
        success = controller.execute_plan()

        if not success:
            print("\n" + "=" * 60)
            print("EXPECTED: System correctly identified unrecoverable failure")
            print("=" * 60)


def demo_capability_analysis():
    """Demo 5: Show capability providers"""
    print_header("DEMO 5: Capability Analysis")

    registry = create_trip_planner_registry()

    print("\nCapability Providers:")
    print("-" * 40)
    for capability, providers in sorted(registry.capability_providers.items()):
        print(f"  {capability}: {providers}")

    print("\n\nIf 'expedia' fails, alternatives are:")
    alternatives = registry.find_alternative_agents("expedia")
    for capability, alt_agents in alternatives.items():
        print(f"  {capability}: {alt_agents}")


def demo_day_itinerary_with_weather():
    """
    Demo 6: Full Day Itinerary with Sudden Weather Change

    Simulates a tourist's day in Montreal with multiple outdoor activities planned.
    Midway through, rain starts and the system must replan to indoor alternatives.
    """
    print_header("DEMO 6: Day Itinerary with Sudden Rain")

    from demos.activities import ActivityAction, WeatherCondition

    # Create a fresh registry (not used for activities, but needed by parent class)
    registry = ToolRegistry()

    print("\n" + "=" * 60)
    print(" SCENARIO: A Tourist's Day in Montreal")
    print("=" * 60)
    print("""
    You're planning a perfect summer day in Montreal:

    MORNING (Sunny):
      - Breakfast at outdoor cafe
      - Walk through Old Montreal
      - Visit Mount Royal Park

    AFTERNOON (Rain starts!):
      - Picnic in the park  <-- BLOCKED by rain!
      - Outdoor concert     <-- BLOCKED by rain!

    The system must REPLAN to indoor alternatives:
      - Visit Montreal Museum of Fine Arts
      - Explore Underground City
      - Indoor food market
    """)

    # =========================================================================
    # Define Activities (Outdoor - require SUNNY weather)
    # =========================================================================

    outdoor_activities = [
        ActivityAction(
            name="breakfast_outdoor_cafe",
            agent_id="morning_planner",
            preconditions=frozenset({"at_hotel", "morning"}),
            positive_effects=frozenset({"had_breakfast", "in_old_montreal"}),
            negative_effects=frozenset({"at_hotel"}),
            weather_requirement=WeatherCondition.SUNNY,
            duration_minutes=60
        ),
        ActivityAction(
            name="walk_old_montreal",
            agent_id="morning_planner",
            preconditions=frozenset({"had_breakfast", "in_old_montreal"}),
            positive_effects=frozenset({"explored_old_montreal", "morning_activity_done"}),
            negative_effects=frozenset(),
            weather_requirement=WeatherCondition.SUNNY,
            duration_minutes=90
        ),
        ActivityAction(
            name="hike_mount_royal",
            agent_id="morning_planner",
            preconditions=frozenset({"morning_activity_done"}),
            positive_effects=frozenset({"visited_mount_royal", "exercised"}),
            negative_effects=frozenset({"morning"}),
            weather_requirement=WeatherCondition.SUNNY,
            duration_minutes=120
        ),
        ActivityAction(
            name="picnic_in_park",
            agent_id="afternoon_planner",
            preconditions=frozenset({"visited_mount_royal", "exercised"}),
            positive_effects=frozenset({"had_lunch", "relaxed"}),
            negative_effects=frozenset(),
            weather_requirement=WeatherCondition.SUNNY,
            duration_minutes=60
        ),
        ActivityAction(
            name="outdoor_jazz_concert",
            agent_id="afternoon_planner",
            preconditions=frozenset({"had_lunch", "relaxed"}),
            positive_effects=frozenset({"entertainment_done", "cultural_experience"}),
            negative_effects=frozenset(),
            weather_requirement=WeatherCondition.SUNNY,
            duration_minutes=120
        ),
    ]

    # =========================================================================
    # Define Activities (Indoor - work in RAINY weather)
    # =========================================================================

    indoor_activities = [
        ActivityAction(
            name="breakfast_hotel_restaurant",
            agent_id="morning_planner",
            preconditions=frozenset({"at_hotel", "morning"}),
            positive_effects=frozenset({"had_breakfast", "in_old_montreal"}),
            negative_effects=frozenset({"at_hotel"}),
            weather_requirement=WeatherCondition.RAINY,
            duration_minutes=45
        ),
        ActivityAction(
            name="explore_underground_city",
            agent_id="morning_planner",
            preconditions=frozenset({"had_breakfast", "in_old_montreal"}),
            positive_effects=frozenset({"explored_old_montreal", "morning_activity_done"}),
            negative_effects=frozenset(),
            weather_requirement=WeatherCondition.RAINY,
            duration_minutes=90
        ),
        ActivityAction(
            name="visit_biodome",
            agent_id="morning_planner",
            preconditions=frozenset({"morning_activity_done"}),
            positive_effects=frozenset({"visited_mount_royal", "exercised"}),  # Same effects as outdoor
            negative_effects=frozenset({"morning"}),
            weather_requirement=WeatherCondition.RAINY,
            duration_minutes=120
        ),
        ActivityAction(
            name="lunch_at_food_market",
            agent_id="afternoon_planner",
            preconditions=frozenset({"visited_mount_royal", "exercised"}),
            positive_effects=frozenset({"had_lunch", "relaxed"}),
            negative_effects=frozenset(),
            weather_requirement=WeatherCondition.RAINY,
            duration_minutes=60
        ),
        ActivityAction(
            name="visit_fine_arts_museum",
            agent_id="afternoon_planner",
            preconditions=frozenset({"had_lunch", "relaxed"}),
            positive_effects=frozenset({"entertainment_done", "cultural_experience"}),
            negative_effects=frozenset(),
            weather_requirement=WeatherCondition.RAINY,
            duration_minutes=120
        ),
    ]

    # Final activity (works in any weather - it's at the hotel)
    final_activity = ActivityAction(
        name="evening_dinner_hotel",
        agent_id="evening_planner",
        preconditions=frozenset({"entertainment_done", "cultural_experience"}),
        positive_effects=frozenset({"had_dinner", "day_complete"}),
        negative_effects=frozenset(),
        weather_requirement=WeatherCondition.SUNNY,  # Start with sunny plan
        duration_minutes=90
    )

    final_activity_rainy = ActivityAction(
        name="evening_dinner_restaurant",
        agent_id="evening_planner",
        preconditions=frozenset({"entertainment_done", "cultural_experience"}),
        positive_effects=frozenset({"had_dinner", "day_complete"}),
        negative_effects=frozenset(),
        weather_requirement=WeatherCondition.RAINY,
        duration_minutes=90
    )

    # =========================================================================
    # Collect all activities
    # =========================================================================

    all_activities = outdoor_activities + indoor_activities + [final_activity, final_activity_rainy]

    # =========================================================================
    # Setup weather and controller
    # =========================================================================

    weather_service = MockWeatherService()
    # Start with sunny weather
    weather_service.inject_weather_change("Montreal", WeatherCondition.SUNNY)

    controller = ItineraryController(
        registry=registry,
        weather_service=weather_service,
        location="Montreal"
    )

    # Set the activities for weather-aware planning
    controller.set_activities(all_activities)

    # Initial state: Tourist just woke up at the hotel
    initial_state = {"at_hotel", "morning"}

    # Goal: Complete a full day of activities
    goal = {"day_complete", "had_dinner", "cultural_experience"}

    print("\n" + "-" * 60)
    print("PLANNING PHASE")
    print("-" * 60)
    print(f"Initial state: {initial_state}")
    print(f"Goal: {goal}")
    print(f"Weather: {weather_service.get_current_weather('Montreal')}")
    print(f"Total activities defined: {len(all_activities)}")
    print(f"  - Outdoor (sunny): {len(outdoor_activities) + 1}")
    print(f"  - Indoor (rainy):  {len(indoor_activities) + 1}")

    # Create initial plan using weather-aware method (should use outdoor activities)
    if not controller.create_activity_plan(initial_state, goal):
        print("ERROR: Could not create initial plan!")
        return

    print("\n INITIAL ITINERARY (Sunny Day Plan):")
    print("-" * 40)
    for i, step in enumerate(controller.current_plan):
        for action in step:
            print(f"  {i+1}. {action.name.replace('_', ' ').title()}")

    # =========================================================================
    # Simulate weather change mid-execution
    # =========================================================================

    print("\n" + "!" * 60)
    print(" WEATHER ALERT: Rain starting in Montreal!")
    print(" Forecast: Heavy rain for the rest of the day")
    print("!" * 60)

    weather_service.inject_weather_change("Montreal", WeatherCondition.RAINY)
    print(f"\nNew weather: {weather_service.get_current_weather('Montreal')}")

    # =========================================================================
    # Execute with weather monitoring (will trigger replanning)
    # =========================================================================

    print("\n" + "-" * 60)
    print("EXECUTION PHASE (with weather monitoring)")
    print("-" * 60)

    success = controller.execute_with_weather_monitoring()

    # =========================================================================
    # Results
    # =========================================================================

    print("\n" + "=" * 60)
    if success:
        print(" DAY COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("""
    Despite the sudden rain, the system automatically:
    1. Detected weather-incompatible outdoor activities
    2. Found indoor alternatives with equivalent outcomes
    3. Replanned the itinerary to use indoor activities
    4. Completed all goals for the day
        """)
    else:
        print(" DAY COULD NOT BE COMPLETED")
        print("=" * 60)
        print("The system could not find suitable alternatives.")

    print(f"Final state: {controller.execution_state.current_propositions}")
    print(f"Goals achieved: {goal.issubset(controller.execution_state.current_propositions)}")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print(" AEGIS Graph Planning Recomposition Demo")
    print(" Multi-Agent Trip Planner with Self-Healing")
    print("=" * 70)

    # Run all demos
    # demo_capability_analysis()
    # demo_successful_trip()
    # demo_agent_failure_recovery()
    # demo_multiple_failures()
    # demo_unrecoverable_failure()
    demo_day_itinerary_with_weather()

    print("\n" + "=" * 70)
    print(" All demos completed!")
    print("=" * 70)