"""
Multi-Agent Trip Planner Demo
==============================
Demonstrates the event-driven multi-agent architecture:

1. demo_weather_driven_replan()
   - PlannerAgent + WeatherAgent
   - Weather changes mid-execution, triggers event-driven replan

2. demo_mcp_traffic_agent()
   - Mock traffic MCP agent publishes REPLAN_REQUESTED events
"""

import asyncio

from aegis.agent import AgentConfig
from aegis.events import EventBus, EventType, Event
from aegis.mcp_adapter import MCPAgentAdapter, MCPToolSpec
from demos.activities import ActivityAction, WeatherCondition
from demos.weather_service import MockWeatherService
from demos.agents.weather_agent import WeatherAgent
from demos.agents.planner_agent import PlannerAgent


# =============================================================================
# Shared activity definitions (same data as trip_planner_demo)
# =============================================================================

def _build_activities():
    """Return (outdoor, indoor, all) activity lists."""

    outdoor = [
        ActivityAction(
            name="breakfast_outdoor_cafe",
            agent_id="morning_planner",
            preconditions=frozenset({"at_hotel", "morning"}),
            positive_effects=frozenset({"had_breakfast", "in_old_montreal"}),
            negative_effects=frozenset({"at_hotel"}),
            weather_requirement=WeatherCondition.SUNNY,
            duration_minutes=60,
        ),
        ActivityAction(
            name="walk_old_montreal",
            agent_id="morning_planner",
            preconditions=frozenset({"had_breakfast", "in_old_montreal"}),
            positive_effects=frozenset({"explored_old_montreal", "morning_activity_done"}),
            negative_effects=frozenset(),
            weather_requirement=WeatherCondition.SUNNY,
            duration_minutes=90,
        ),
        ActivityAction(
            name="hike_mount_royal",
            agent_id="morning_planner",
            preconditions=frozenset({"morning_activity_done"}),
            positive_effects=frozenset({"visited_mount_royal", "exercised"}),
            negative_effects=frozenset({"morning"}),
            weather_requirement=WeatherCondition.SUNNY,
            duration_minutes=120,
        ),
        ActivityAction(
            name="picnic_in_park",
            agent_id="afternoon_planner",
            preconditions=frozenset({"visited_mount_royal", "exercised"}),
            positive_effects=frozenset({"had_lunch", "relaxed"}),
            negative_effects=frozenset(),
            weather_requirement=WeatherCondition.SUNNY,
            duration_minutes=60,
        ),
        ActivityAction(
            name="outdoor_jazz_concert",
            agent_id="afternoon_planner",
            preconditions=frozenset({"had_lunch", "relaxed"}),
            positive_effects=frozenset({"entertainment_done", "cultural_experience"}),
            negative_effects=frozenset(),
            weather_requirement=WeatherCondition.SUNNY,
            duration_minutes=120,
        ),
    ]

    indoor = [
        ActivityAction(
            name="breakfast_hotel_restaurant",
            agent_id="morning_planner",
            preconditions=frozenset({"at_hotel", "morning"}),
            positive_effects=frozenset({"had_breakfast", "in_old_montreal"}),
            negative_effects=frozenset({"at_hotel"}),
            weather_requirement=WeatherCondition.RAINY,
            duration_minutes=45,
        ),
        ActivityAction(
            name="explore_underground_city",
            agent_id="morning_planner",
            preconditions=frozenset({"had_breakfast", "in_old_montreal"}),
            positive_effects=frozenset({"explored_old_montreal", "morning_activity_done"}),
            negative_effects=frozenset(),
            weather_requirement=WeatherCondition.RAINY,
            duration_minutes=90,
        ),
        ActivityAction(
            name="visit_biodome",
            agent_id="morning_planner",
            preconditions=frozenset({"morning_activity_done"}),
            positive_effects=frozenset({"visited_mount_royal", "exercised"}),
            negative_effects=frozenset({"morning"}),
            weather_requirement=WeatherCondition.RAINY,
            duration_minutes=120,
        ),
        ActivityAction(
            name="lunch_at_food_market",
            agent_id="afternoon_planner",
            preconditions=frozenset({"visited_mount_royal", "exercised"}),
            positive_effects=frozenset({"had_lunch", "relaxed"}),
            negative_effects=frozenset(),
            weather_requirement=WeatherCondition.RAINY,
            duration_minutes=60,
        ),
        ActivityAction(
            name="visit_fine_arts_museum",
            agent_id="afternoon_planner",
            preconditions=frozenset({"had_lunch", "relaxed"}),
            positive_effects=frozenset({"entertainment_done", "cultural_experience"}),
            negative_effects=frozenset(),
            weather_requirement=WeatherCondition.RAINY,
            duration_minutes=120,
        ),
    ]

    finals = [
        ActivityAction(
            name="evening_dinner_hotel",
            agent_id="evening_planner",
            preconditions=frozenset({"entertainment_done", "cultural_experience"}),
            positive_effects=frozenset({"had_dinner", "day_complete"}),
            negative_effects=frozenset(),
            weather_requirement=WeatherCondition.SUNNY,
            duration_minutes=90,
        ),
        ActivityAction(
            name="evening_dinner_restaurant",
            agent_id="evening_planner",
            preconditions=frozenset({"entertainment_done", "cultural_experience"}),
            positive_effects=frozenset({"had_dinner", "day_complete"}),
            negative_effects=frozenset(),
            weather_requirement=WeatherCondition.RAINY,
            duration_minutes=90,
        ),
    ]

    all_activities = outdoor + indoor + finals
    return outdoor, indoor, all_activities


# =============================================================================
# Demo 1: Weather-Driven Replan
# =============================================================================

async def demo_weather_driven_replan():
    """
    PlannerAgent creates a sunny-day itinerary.
    WeatherAgent detects a weather change mid-execution.
    PlannerAgent replans to indoor activities via the EventBus.
    """
    print("=" * 70)
    print(" DEMO 1: Weather-Driven Replan (Event-Driven Multi-Agent)")
    print("=" * 70)

    _, _, all_activities = _build_activities()

    # -- shared services ------------------------------------------------------
    weather_service = MockWeatherService()
    weather_service.inject_weather_change("Montreal", WeatherCondition.SUNNY)
    event_bus = EventBus()

    # -- agents ---------------------------------------------------------------
    weather_agent = WeatherAgent(
        config=AgentConfig(
            agent_id="weather-agent",
            name="Weather Monitor",
            description="Polls weather, publishes WeatherChangedEvent",
            poll_interval_seconds=0.1,
        ),
        event_bus=event_bus,
        weather_service=weather_service,
        location="Montreal",
    )

    planner_agent = PlannerAgent(
        config=AgentConfig(
            agent_id="planner-agent",
            name="Trip Planner",
            description="Creates and executes itinerary, replans on events",
            poll_interval_seconds=1.0,
        ),
        event_bus=event_bus,
        all_activities=all_activities,
        initial_weather=WeatherCondition.SUNNY,
        location="Montreal",
    )

    initial_state = {"at_hotel", "morning"}
    goal = {"day_complete", "had_dinner", "cultural_experience"}
    planner_agent._initial_state = initial_state
    planner_agent._goal = goal

    print(f"\n  Initial state : {initial_state}")
    print(f"  Goal          : {goal}")
    print(f"  Weather       : {weather_service.get_current_weather('Montreal').condition.value}")
    print()

    # -- schedule the weather change after the planner executes a few steps ---
    async def inject_rain():
        # Wait for planner to get ~2 steps in (each step ~0.2s)
        await asyncio.sleep(0.6)
        print("\n  !! WEATHER ALERT: Injecting rain into Montreal !!")
        weather_agent.inject_weather_change(WeatherCondition.RAINY)

    # -- run ------------------------------------------------------------------
    await weather_agent.start()
    await planner_agent.start()
    asyncio.create_task(inject_rain())

    # Wait for the planner to finish (with a timeout)
    try:
        await asyncio.wait_for(planner_agent.done_event.wait(), timeout=30.0)
    except asyncio.TimeoutError:
        print("  [TIMEOUT] Demo timed out!")

    await weather_agent.stop()
    await planner_agent.stop()

    # -- results --------------------------------------------------------------
    print("\n" + "=" * 70)
    if planner_agent.success:
        print(" DEMO 1 RESULT: SUCCESS")
    else:
        print(" DEMO 1 RESULT: FAILED")
    print("=" * 70)
    print(f"  Final state: {planner_agent.state.execution_state.current_propositions}")
    print(f"  Goals met  : {goal.issubset(planner_agent.state.execution_state.current_propositions)}")

    event_bus.print_event_log()


# =============================================================================
# Demo 2: MCP Traffic Agent
# =============================================================================

async def demo_mcp_traffic_agent():
    """
    A mock traffic MCP agent publishes REPLAN_REQUESTED when it detects congestion.
    The PlannerAgent receives the event and replans.
    """
    print("\n" + "=" * 70)
    print(" DEMO 2: MCP Traffic Agent (REPLAN_REQUESTED via MCP Adapter)")
    print("=" * 70)

    _, _, all_activities = _build_activities()

    event_bus = EventBus()

    # -- mock MCP call function -----------------------------------------------
    _call_count = 0

    async def mock_traffic_mcp(tool_name: str, params: dict):
        nonlocal _call_count
        _call_count += 1
        # Simulate: on the 3rd poll, detect heavy traffic
        if _call_count >= 3:
            return {"congestion_level": "heavy", "delay_minutes": 45}
        return {"congestion_level": "normal", "delay_minutes": 0}

    traffic_adapter = MCPAgentAdapter(
        config=AgentConfig(
            agent_id="traffic-mcp",
            name="Traffic MCP Agent",
            description="Monitors traffic via MCP, publishes replan events",
            poll_interval_seconds=0.3,
        ),
        event_bus=event_bus,
        mcp_call_fn=mock_traffic_mcp,
        tool_specs=[
            MCPToolSpec(
                name="get_traffic",
                description="Fetch live traffic conditions",
                input_schema={"location": "Montreal"},
                output_event_type=EventType.REPLAN_REQUESTED,
                output_transformer=lambda data: {
                    "reason": f"Traffic congestion: {data['congestion_level']} "
                              f"(+{data['delay_minutes']} min delay)",
                },
            ),
        ],
    )

    planner_agent = PlannerAgent(
        config=AgentConfig(
            agent_id="planner-agent",
            name="Trip Planner",
            description="Creates and executes itinerary, replans on events",
            poll_interval_seconds=1.0,
        ),
        event_bus=event_bus,
        all_activities=all_activities,
        initial_weather=WeatherCondition.SUNNY,
        location="Montreal",
    )

    initial_state = {"at_hotel", "morning"}
    goal = {"day_complete", "had_dinner", "cultural_experience"}
    planner_agent._initial_state = initial_state
    planner_agent._goal = goal

    print(f"\n  Initial state : {initial_state}")
    print(f"  Goal          : {goal}")
    print(f"  Weather       : sunny (constant)")
    print()

    # -- run ------------------------------------------------------------------
    await traffic_adapter.start()
    await planner_agent.start()

    try:
        await asyncio.wait_for(planner_agent.done_event.wait(), timeout=30.0)
    except asyncio.TimeoutError:
        print("  [TIMEOUT] Demo timed out!")

    await traffic_adapter.stop()
    await planner_agent.stop()

    # -- results --------------------------------------------------------------
    print("\n" + "=" * 70)
    if planner_agent.success:
        print(" DEMO 2 RESULT: SUCCESS")
    else:
        print(" DEMO 2 RESULT: FAILED (expected — replan stays sunny)")
    print("=" * 70)
    print(f"  Final state: {planner_agent.state.execution_state.current_propositions}")
    print(f"  Goals met  : {goal.issubset(planner_agent.state.execution_state.current_propositions)}")

    event_bus.print_event_log()


# =============================================================================
# Main
# =============================================================================

async def main():
    print("=" * 70)
    print(" AEGIS Multi-Agent Trip Planner Demo")
    print(" Event-Driven Architecture with EventBus")
    print("=" * 70)

    await demo_weather_driven_replan()
    await demo_mcp_traffic_agent()

    print("\n" + "=" * 70)
    print(" All multi-agent demos completed!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
