import asyncio
import json
import logging

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

from aegis.config import AEGISConfig, LLMProvider
from aegis.agent import BaseAgent, AgentConfig
from aegis.events import (
    Event,
    EventBus,
    EventType,
    WeatherChangedEvent,
    PlanUpdatedEvent,
)
from aegis.state import PlanningGraph
from app.data.extract_data import get_place, build_weather_and_alternatives
from demos.activities import ActivityAction, WeatherCondition
from app.chat.state import ChatState, Intent, DayPlan, Message
from app.services.activity_adapter import (
    build_actions_for_day_with_weather,
    build_indoor_alternatives,
)
from app.data.sample_trip import SAMPLE_TRIP, WEATHER_MAPS, INDOOR_ALTERNATIVES
from app.prompts import INTENT_SYSTEM_PROMPT, RESPONSE_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


class TripChatBot(BaseAgent):
    def __init__(
        self,
        config: AEGISConfig,
        event_bus: EventBus | None = None,
        agent_config: AgentConfig | None = None,
    ):
        agent_cfg = agent_config or AgentConfig(
            agent_id="trip_chatbot",
            name="TripChatBot",
            description="Interactive trip-planning chatbot",
        )
        bus = event_bus or EventBus()
        super().__init__(agent_cfg, bus)

        # Rename to avoid collision with BaseAgent.config (AgentConfig)
        self.aegis_config = config
        self.llm = self._create_llm()
        self.state = ChatState()
        self._load_sample_trip()

        self.notification_queue: asyncio.Queue[str] = asyncio.Queue()
        self._event_connected = event_bus is not None

    # ------------------------------------------------------------------
    # BaseAgent lifecycle
    # ------------------------------------------------------------------

    async def setup(self) -> None:
        self.subscribe_to(EventType.WEATHER_CHANGED)
        self.subscribe_to(EventType.PLAN_UPDATED)

    async def run(self) -> None:
        while self._running:
            await asyncio.sleep(1.0)

    async def handle_event(self, event: Event) -> None:
        if event.source_agent_id == self.agent_id:
            return

        if event.event_type == EventType.WEATHER_CHANGED:
            await self._on_external_weather_change(event)
        elif event.event_type == EventType.PLAN_UPDATED:
            self.notification_queue.put_nowait(
                f"Plan updated by agent '{event.source_agent_id}' "
                f"({event.plan_step_count} steps)."
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def handle_message(self, user_input: str) -> str:
        logger.info("Received message: %s", user_input[:80])
        self.state.add_message("user", user_input)

        intent, params = await self._parse_intent(user_input)
        logger.info("Parsed intent=%s, params=%s", intent, params)

        result, events = self._dispatch_intent(intent, params, user_input)
        logger.debug("Dispatch result length=%d, events=%d", len(result), len(events))

        for evt in events:
            logger.debug("Publishing event: %s", evt.event_type)
            await self._publish_if_connected(evt)

        llm_response = await self._format_response(intent, result)
        logger.debug("LLM response length=%d", len(llm_response))
        self.state.add_message("bot", llm_response)
        return llm_response

    # ------------------------------------------------------------------
    # LLM creation
    # ------------------------------------------------------------------

    def _create_llm(self):
        if self.aegis_config.llm_provider == LLMProvider.OPENAI:
            return ChatOpenAI(
                model=self.aegis_config.llm_model,
                temperature=0.1,
                api_key=self.aegis_config.openai_api_key,
            )
        elif self.aegis_config.llm_provider == LLMProvider.ANTHROPIC:
            return ChatAnthropic(
                model=self.aegis_config.llm_model,
                temperature=0.1,
                api_key=self.aegis_config.anthropic_api_key,
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {self.aegis_config.llm_provider}")

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_sample_trip(self):
        logger.info("Loading trip data from MongoDB...")
        self.trip_data = get_place()
        # Fallback to SAMPLE_TRIP if MongoDB connection fails or data is empty
        if self.trip_data is None:
            logger.warning("MongoDB returned None — falling back to SAMPLE_TRIP")
            self.trip_data = SAMPLE_TRIP
        elif not self.trip_data.get("days") or all(
            not day.get("plan") for day in self.trip_data.get("days", [])
        ):
            logger.warning("MongoDB data has no activities — falling back to SAMPLE_TRIP")
            self.trip_data = SAMPLE_TRIP
        else:
            days = self.trip_data["days"]
            total = sum(len(d.get("plan", [])) for d in days)
            logger.info("Loaded trip from MongoDB: %d days, %d activities", len(days), total)

        # Use dynamic maps from MongoDB when trip data came from DB,
        # fall back to hardcoded maps only for the SAMPLE_TRIP fallback.
        if self.trip_data is SAMPLE_TRIP:
            self.weather_maps = WEATHER_MAPS
            self.indoor_alternatives = INDOOR_ALTERNATIVES
            logger.info("Using hardcoded WEATHER_MAPS and INDOOR_ALTERNATIVES (sample trip)")
        else:
            self.weather_maps, self.indoor_alternatives = build_weather_and_alternatives(self.trip_data)
            logger.info("Using dynamic weather maps and MongoDB indoor alternatives")

    # ------------------------------------------------------------------
    # Context / intent parsing
    # ------------------------------------------------------------------

    def _build_context(self) -> str:
        recent = self.state.conversation[-5:]
        lines = []
        for msg in recent:
            lines.append(f"{msg.role}: {msg.content}")
        return "\n".join(lines)

    async def _parse_intent(self, user_input: str):
        context = self._build_context()
        response = await self.llm.ainvoke([
            SystemMessage(content=INTENT_SYSTEM_PROMPT),
            HumanMessage(
                content=f"Conversation context:\n{context}\n\nUser message: {user_input}"
            ),
        ])
        if not response.content:
            raise ValueError("LLM response is empty. Unable to parse intent.")
        try:
            parsed = json.loads(self._strip_code_fences(response.content))
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to decode JSON from LLM response: {response.content}") from e

        return parsed.get("intent"), parsed.get("params", {})

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def _dispatch_intent(
        self, intent_str: str, params: dict, user_input: str
    ) -> tuple[str, list[Event]]:
        intent_map = {
            "plan_trip": Intent.PLAN_TRIP,
            "show_itinerary": Intent.SHOW_ITINERARY,
            "weather_change": Intent.WEATHER_CHANGE,
            "swap_activity": Intent.SWAP_ACTIVITY,
            "list_alternatives": Intent.LIST_ACTIVITIES,
            "help": Intent.HELP,
            "general": Intent.GENERAL,
        }
        intent = intent_map.get(intent_str, Intent.GENERAL)
        params = params or {}

        if intent == Intent.PLAN_TRIP:
            return self._handle_plan_trip(params)
        elif intent == Intent.SHOW_ITINERARY:
            return self._handle_show_itinerary(params), []
        elif intent == Intent.WEATHER_CHANGE:
            return self._handle_weather_change(params)
        elif intent == Intent.SWAP_ACTIVITY:
            return self._handle_swap_activity(params)
        elif intent == Intent.LIST_ACTIVITIES:
            return self._handle_list_alternatives(params), []
        elif intent == Intent.HELP:
            return self._handle_help(), []
        else:
            return self._handle_general(user_input), []

    # ------------------------------------------------------------------
    # Response formatting
    # ------------------------------------------------------------------

    async def _format_response(self, intent_str: str, result: str) -> str:
        response = await self.llm.ainvoke([
            SystemMessage(content=RESPONSE_SYSTEM_PROMPT),
            HumanMessage(content=f"Intent: {intent_str}\n\nResult:\n{result}"),
        ])
        return response.content

    # ------------------------------------------------------------------
    # Event helpers
    # ------------------------------------------------------------------

    async def _publish_if_connected(self, event: Event) -> None:
        if self._event_connected:
            await self.publish(event)

    async def _on_external_weather_change(self, event: Event) -> None:
        location = getattr(event, "location", None) or event.payload.get("location")
        new_cond_str = (
            getattr(event, "new_condition", None) or event.payload.get("new_condition")
        )
        if not location or not new_cond_str:
            return

        weather_map = {"rainy": WeatherCondition.RAINY, "sunny": WeatherCondition.SUNNY}
        weather = weather_map.get(new_cond_str, WeatherCondition.RAINY)

        if not self.state.trip_planned:
            self.notification_queue.put_nowait(
                f"Weather alert: {location} is now {new_cond_str}, "
                f"but no trip is planned yet."
            )
            return

        affected_days = []
        for day_idx, day_data in enumerate(self.trip_data["days"]):
            if day_data.get("city", "").lower() == location.lower():
                actions = self._plan_day(day_idx, weather)
                if actions is not None:
                    day_num = day_data.get("day", day_idx + 1)
                    self.state.day_plans[day_idx] = DayPlan(
                        day_number=day_num,
                        city=day_data.get("city", "Unknown"),
                        actions=actions,
                        weather=weather,
                    )
                    affected_days.append(day_num)

        if affected_days:
            days_str = ", ".join(str(d) for d in affected_days)
            self.notification_queue.put_nowait(
                f"Weather alert! {location} is now {new_cond_str}. "
                f"Auto-replanned day(s) {days_str} with {new_cond_str}-weather activities."
            )
        else:
            self.notification_queue.put_nowait(
                f"Weather alert: {location} is now {new_cond_str}, "
                f"but no matching days found in the trip."
            )

    # ------------------------------------------------------------------
    # Planning helper
    # ------------------------------------------------------------------

    def _plan_day(self, day_idx: int, weather: WeatherCondition = WeatherCondition.SUNNY):
        day_data = self.trip_data["days"][day_idx]
        weather_map = self.weather_maps.get(day_idx, {})

        all_actions = build_actions_for_day_with_weather(day_data, weather_map)

        if weather == WeatherCondition.RAINY:
            pool = [a for a in all_actions if a.weather_requirement == WeatherCondition.ANY]
            outdoor = [a for a in all_actions if a.weather_requirement == WeatherCondition.SUNNY]
            alts = self.indoor_alternatives.get(day_idx, {})
            pool.extend(build_indoor_alternatives(outdoor, alts))
        else:
            pool = [
                a for a in all_actions
                if a.weather_requirement in (WeatherCondition.SUNNY, WeatherCondition.ANY)
            ]

        graph = PlanningGraph(
            initial_state={"day_started"},
            goal={"day_completed", "had_dinner"},
            available_actions=pool,
        )
        if not graph.build_graph():
            return None
        plan = graph.extract_plan()
        if plan is None:
            return None

        actions = []
        for step in plan:
            actions.extend(step)
        actions.sort(key=lambda a: next(
            (int(e.split("_")[1]) for e in a.positive_effects
             if e.startswith("slot_") and e.endswith("_completed")), 0
        ))
        return actions

    @staticmethod
    def _format_actions(actions: list[ActivityAction]) -> str:
        lines = []
        for i, action in enumerate(actions, 1):
            weather_tag = ""
            if action.weather_requirement == WeatherCondition.RAINY:
                weather_tag = " [indoor replacement]"
            lines.append(
                f"{i}. {action.name.replace('_', ' ').title()} "
                f"({action.duration_minutes} min){weather_tag}"
            )
        return "\n".join(lines)

    @staticmethod
    def _slot_index(action: ActivityAction) -> int:
        for eff in action.positive_effects:
            if eff.startswith("slot_") and eff.endswith("_completed"):
                try:
                    return int(eff.split("_")[1])
                except ValueError:
                    continue
        return -1

    # ------------------------------------------------------------------
    # Intent handlers
    # ------------------------------------------------------------------

    def _handle_plan_trip(self, params: dict) -> tuple[str, list[Event]]:
        # Safety check: ensure trip_data is available
        if not self.trip_data or "days" not in self.trip_data or not self.trip_data["days"]:
            return "Error: No trip data available. Please check the configuration.", []

        results = []
        total_steps = 0
        for day_idx in range(len(self.trip_data["days"])):
            day_data = self.trip_data["days"][day_idx]
            city = day_data.get("city", "Unknown")
            day_num = day_data.get("day", day_idx + 1)

            actions = self._plan_day(day_idx)
            if actions is None:
                results.append(f"Day {day_num} ({city}): Could not find a valid plan.")
                continue

            self.state.day_plans[day_idx] = DayPlan(
                day_number=day_num,
                city=city,
                actions=actions,
                weather=WeatherCondition.SUNNY,
            )
            total_steps += len(actions)
            results.append(f"Day {day_num} ({city}):\n{self._format_actions(actions)}")

        self.state.trip_planned = True
        result = "Trip planned!\n\n" + "\n\n".join(results)

        events = [
            PlanUpdatedEvent(
                event_type=EventType.PLAN_UPDATED,
                source_agent_id=self.agent_id,
                plan_step_count=total_steps,
                payload={"action": "plan_trip", "days": len(self.trip_data["days"])},
            )
        ]
        return result, events

    def _handle_show_itinerary(self, params: dict) -> str:
        if not self.state.trip_planned:
            return "No trip planned yet. Say 'plan my trip' to get started."

        day = params.get("day")
        if day is not None:
            day_idx = day - 1
            plan = self.state.day_plans.get(day_idx)
            if plan is None:
                return f"No plan found for day {day}."
            return (
                f"Day {plan.day_number} ({plan.city}) — "
                f"Weather: {plan.weather.value}\n{self._format_actions(plan.actions)}"
            )

        # Show all days
        parts = []
        for day_idx in sorted(self.state.day_plans):
            plan = self.state.day_plans[day_idx]
            parts.append(
                f"Day {plan.day_number} ({plan.city}) — "
                f"Weather: {plan.weather.value}\n{self._format_actions(plan.actions)}"
            )
        return "\n\n".join(parts)

    def _handle_weather_change(self, params: dict) -> tuple[str, list[Event]]:
        if not self.state.trip_planned:
            return "No trip planned yet. Plan the trip first.", []

        day = params.get("day")
        weather_str = params.get("weather", "rainy")
        weather_map = {"rainy": WeatherCondition.RAINY, "snowy": WeatherCondition.RAINY,
                       "sunny": WeatherCondition.SUNNY}
        weather = weather_map.get(weather_str, WeatherCondition.RAINY)

        if day is None:
            # Default to last focused day or day 3
            day = self.state.current_day_focus or len(self.trip_data["days"])

        day_idx = day - 1
        if day_idx < 0 or day_idx >= len(self.trip_data["days"]):
            return f"Invalid day number: {day}", []

        self.state.current_day_focus = day

        old_plan = self.state.day_plans.get(day_idx)
        old_actions_str = self._format_actions(old_plan.actions) if old_plan else "None"

        actions = self._plan_day(day_idx, weather)
        if actions is None:
            return f"Could not replan day {day} for {weather_str} weather.", []

        day_data = self.trip_data["days"][day_idx]
        city = day_data.get("city", "Unknown")
        self.state.day_plans[day_idx] = DayPlan(
            day_number=day_data.get("day", day),
            city=city,
            actions=actions,
            weather=weather,
        )

        new_actions_str = self._format_actions(actions)

        # Identify swaps
        swaps = []
        if old_plan:
            old_by_slot = {self._slot_index(a): a for a in old_plan.actions}
            new_by_slot = {self._slot_index(a): a for a in actions}
            for slot in sorted(set(old_by_slot) | set(new_by_slot)):
                old_a = old_by_slot.get(slot)
                new_a = new_by_slot.get(slot)
                if old_a and new_a and old_a.name != new_a.name:
                    swaps.append(
                        f"  - {old_a.name.replace('_', ' ').title()} -> "
                        f"{new_a.name.replace('_', ' ').title()}"
                    )

        result = f"Replanned day {day} for {weather_str} weather.\n\n{new_actions_str}"
        if swaps:
            result += "\n\nSwaps made:\n" + "\n".join(swaps)

        events = [
            WeatherChangedEvent(
                event_type=EventType.WEATHER_CHANGED,
                source_agent_id=self.agent_id,
                previous_condition=old_plan.weather.value if old_plan else "sunny",
                new_condition=weather_str,
                location=city,
                payload={"day": day, "weather": weather_str},
            ),
            PlanUpdatedEvent(
                event_type=EventType.PLAN_UPDATED,
                source_agent_id=self.agent_id,
                plan_step_count=len(actions),
                payload={"action": "weather_replan", "day": day, "weather": weather_str},
            ),
        ]
        return result, events

    def _handle_swap_activity(self, params: dict) -> tuple[str, list[Event]]:
        if not self.state.trip_planned:
            return "No trip planned yet. Plan the trip first.", []

        from_name = params.get("from_activity", "")
        to_name = params.get("to_activity", "")
        day = params.get("day")

        if not from_name or not to_name:
            return "Please specify which activity to swap and what to replace it with.", []

        if day is None:
            day = self.state.current_day_focus or len(self.trip_data["days"])
        day_idx = day - 1

        plan = self.state.day_plans.get(day_idx)
        if plan is None:
            return f"No plan found for day {day}.", []

        from_slug = from_name.lower().replace(" ", "_")
        to_slug = to_name.lower().replace(" ", "_")

        # Find the action to replace
        target_action = None
        target_idx = None
        for i, action in enumerate(plan.actions):
            if from_slug in action.name or action.name in from_slug:
                target_action = action
                target_idx = i
                break

        if target_action is None:
            return f"Could not find activity '{from_name}' in day {day}.", []

        # Create replacement action with same preconditions/effects
        replacement = ActivityAction(
            name=to_slug,
            agent_id="itinerary_planner",
            preconditions=target_action.preconditions,
            positive_effects=target_action.positive_effects,
            negative_effects=frozenset(),
            weather_requirement=WeatherCondition.ANY,
            duration_minutes=target_action.duration_minutes,
        )

        plan.actions[target_idx] = replacement
        result = (
            f"Swapped '{from_name}' with '{to_name}' on day {day}.\n\n"
            f"Updated itinerary:\n{self._format_actions(plan.actions)}"
        )

        events = [
            PlanUpdatedEvent(
                event_type=EventType.PLAN_UPDATED,
                source_agent_id=self.agent_id,
                plan_step_count=len(plan.actions),
                payload={
                    "action": "swap_activity",
                    "day": day,
                    "from": from_name,
                    "to": to_name,
                },
            )
        ]
        return result, events

    def _handle_list_alternatives(self, params: dict) -> str:
        day = params.get("day")
        slot = params.get("slot")

        if day is None:
            day = self.state.current_day_focus or len(self.trip_data["days"])
        day_idx = day - 1

        alts = self.indoor_alternatives.get(day_idx, {})
        if not alts:
            return f"No alternatives available for day {day}."

        if slot is not None and slot in alts:
            alt_list = alts[slot]
            lines = [f"Alternatives for slot {slot} on day {day}:"]
            for alt in alt_list:
                lines.append(f"  - {alt['name']} ({alt.get('duration_minutes', '?')} min)")
            return "\n".join(lines)

        # Show all alternatives for the day
        lines = [f"Available alternatives for day {day}:"]
        for s, alt_list in sorted(alts.items()):
            lines.append(f"\n  Slot {s}:")
            for alt in alt_list:
                lines.append(f"    - {alt['name']} ({alt.get('duration_minutes', '?')} min)")
        return "\n".join(lines)

    def _handle_help(self) -> str:
        return (
            "I can help you with trip planning! Here's what I can do:\n"
            "- Plan your trip: 'Plan my trip'\n"
            "- Show itinerary: 'Show me day 1' or 'Show the itinerary'\n"
            "- Weather changes: 'What if it rains on day 3?'\n"
            "- Swap activities: 'Swap Mount Royal Park with Montreal Biodome on day 3'\n"
            "- List alternatives: 'What are the alternatives for day 3?'\n"
            "- General questions: Ask me anything about the trip!"
        )

    def _handle_general(self, user_input: str) -> str:
        return f"User asked: {user_input}\n\nThis is a general question about the trip."


    @staticmethod
    def _strip_code_fences(text: str) -> str:
        text = text.strip()
        if text.startswith("```"):
            first_line = text.index("\n")
            text = text[first_line + 1:]
        if text.endswith("```"):
            text = text[:-3]
        return text.strip()
