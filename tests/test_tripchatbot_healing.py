"""
Tests for AEGIS healing loop and TripChatBot self-healing integration.

Three layers:
  1. AEGIS.check_and_heal loop — pure logic, no external calls
  2. TripChatBot._replan_affected_days — trip planning logic, no LLM
  3. TripChatBot._on_external_weather_change — full event → notification pipeline
"""
import pytest
from unittest.mock import patch, MagicMock

from aegis import AEGIS, AEGISHealing
from aegis.config import AEGISConfig, LLMProvider, FailureType
from aegis.events import EventBus, WeatherChangedEvent, EventType
from demos.activities import WeatherCondition
from app.chat.state import DayPlan


# ---------------------------------------------------------------------------
# Patch targets
# ---------------------------------------------------------------------------
# Every place that calls ChatOpenAI() at construction time
_OPENAI_PATCHES = [
    "aegis.detector.ChatOpenAI",
    "aegis.repair.ChatOpenAI",
    "aegis.recompose.ChatOpenAI",
    "app.chat.bot.ChatOpenAI",
]
_GET_PLACE = "app.chat.bot.get_place"


# ---------------------------------------------------------------------------
# Shared factories
# ---------------------------------------------------------------------------

def _make_aegis() -> AEGIS:
    """AEGIS instance with LLM construction patched out."""
    with patch("aegis.detector.ChatOpenAI"), \
         patch("aegis.repair.ChatOpenAI"), \
         patch("aegis.recompose.ChatOpenAI"):
        config = AEGISConfig(
            llm_provider=LLMProvider.OPENAI,
            openai_api_key="test-key",
        )
        return AEGIS(config=config)


def _make_bot():
    """TripChatBot with all LLM calls and MongoDB patched out."""
    from app.chat.bot import TripChatBot

    config = AEGISConfig(
        llm_provider=LLMProvider.OPENAI,
        openai_api_key="test-key",
    )
    with patch("aegis.detector.ChatOpenAI"), \
         patch("aegis.repair.ChatOpenAI"), \
         patch("aegis.recompose.ChatOpenAI"), \
         patch("app.chat.bot.ChatOpenAI"), \
         patch(_GET_PLACE, return_value=None):   # forces SAMPLE_TRIP fallback
        return TripChatBot(config=config)


# ===========================================================================
# Layer 1: AEGIS.check_and_heal loop
# ===========================================================================

class TestCheckAndHeal:

    @pytest.mark.asyncio
    async def test_no_registered_healings_returns_no_match(self):
        aegis = _make_aegis()
        result = await aegis.check_and_heal({"anything": True})
        assert result.healed is False
        assert result.reason == "no matching condition"

    @pytest.mark.asyncio
    async def test_condition_not_met_returns_no_match(self):
        aegis = _make_aegis()
        aegis.declare_healing(AEGISHealing(
            name="never_triggered",
            failure_condition=lambda s: s.get("broken") is True,
            repair_fn=lambda s: {"broken": False},
            failure_type=FailureType.CRASH,
        ))
        result = await aegis.check_and_heal({"broken": False})
        assert result.healed is False
        assert result.reason == "no matching condition"

    @pytest.mark.asyncio
    async def test_successful_repair_on_first_attempt(self):
        aegis = _make_aegis()
        aegis.declare_healing(AEGISHealing(
            name="instant_fix",
            failure_condition=lambda s: s.get("broken") is True,
            repair_fn=lambda s: {"broken": False},
            failure_type=FailureType.SEMANTIC_DRIFT,
        ))
        result = await aegis.check_and_heal({"broken": True})
        assert result.healed is True
        assert result.healing_name == "instant_fix"
        assert result.attempts == 1
        assert result.result == {"broken": False}

    @pytest.mark.asyncio
    async def test_repair_fn_called_with_original_state(self):
        aegis = _make_aegis()
        received_states = []

        def capture_state(s):
            received_states.append(dict(s))
            return {"fixed": True}

        aegis.declare_healing(AEGISHealing(
            name="capture",
            failure_condition=lambda s: not s.get("fixed"),
            repair_fn=capture_state,
            failure_type=FailureType.CRASH,
        ))
        state = {"fixed": False, "location": "Montreal"}
        await aegis.check_and_heal(state)

        assert received_states[0]["location"] == "Montreal"
        assert received_states[0]["fixed"] is False

    @pytest.mark.asyncio
    async def test_retries_when_repair_returns_none(self):
        aegis = _make_aegis()
        call_count = []

        def flaky(s):
            call_count.append(1)
            return None

        aegis.declare_healing(AEGISHealing(
            name="flaky",
            failure_condition=lambda s: True,
            repair_fn=flaky,
            failure_type=FailureType.CRASH,
            max_attempts=3,
        ))
        result = await aegis.check_and_heal({})
        assert result.healed is False
        assert result.reason == "max attempts reached"
        assert len(call_count) == 3   # retried exactly max_attempts times

    @pytest.mark.asyncio
    async def test_succeeds_on_second_attempt(self):
        aegis = _make_aegis()
        attempts = []

        def repair_on_second(s):
            attempts.append(1)
            if len(attempts) >= 2:
                return {"fixed": True}
            return None

        aegis.declare_healing(AEGISHealing(
            name="retry_once",
            failure_condition=lambda s: not s.get("fixed"),
            repair_fn=repair_on_second,
            failure_type=FailureType.CRASH,
            max_attempts=3,
        ))
        result = await aegis.check_and_heal({"fixed": False})
        assert result.healed is True
        assert result.attempts == 2

    @pytest.mark.asyncio
    async def test_max_attempts_respected(self):
        aegis = _make_aegis()
        call_count = []

        aegis.declare_healing(AEGISHealing(
            name="always_fails",
            failure_condition=lambda s: True,
            repair_fn=lambda s: call_count.append(1) or None,
            failure_type=FailureType.CRASH,
            max_attempts=5,
        ))
        result = await aegis.check_and_heal({})
        assert result.healed is False
        assert result.attempts == 5
        assert len(call_count) == 5

    @pytest.mark.asyncio
    async def test_repair_that_does_not_resolve_condition_retries(self):
        """repair_fn returns data but condition is still True → must retry."""
        aegis = _make_aegis()
        call_count = []

        # Condition checks key "ok"; repair always sets ok=False (wrong repair)
        def bad_repair(s):
            call_count.append(1)
            return {"ok": False}   # condition still active after this

        aegis.declare_healing(AEGISHealing(
            name="bad_repair",
            failure_condition=lambda s: not s.get("ok"),
            repair_fn=bad_repair,
            failure_type=FailureType.SEMANTIC_DRIFT,
            max_attempts=3,
        ))
        result = await aegis.check_and_heal({"ok": False})
        assert result.healed is False
        assert len(call_count) == 3

    @pytest.mark.asyncio
    async def test_repair_fn_exception_is_caught_and_retried(self):
        aegis = _make_aegis()
        call_count = []

        def boom(s):
            call_count.append(1)
            raise RuntimeError("planning exploded")

        aegis.declare_healing(AEGISHealing(
            name="boom",
            failure_condition=lambda s: True,
            repair_fn=boom,
            failure_type=FailureType.CRASH,
            max_attempts=2,
        ))
        result = await aegis.check_and_heal({})
        assert result.healed is False
        assert len(call_count) == 2   # tried both times despite exception

    @pytest.mark.asyncio
    async def test_declare_healing_registers_in_list(self):
        aegis = _make_aegis()
        assert len(aegis._healings) == 0
        aegis.declare_healing(AEGISHealing(
            name="a",
            failure_condition=lambda s: False,
            repair_fn=lambda s: None,
            failure_type=FailureType.CRASH,
        ))
        aegis.declare_healing(AEGISHealing(
            name="b",
            failure_condition=lambda s: False,
            repair_fn=lambda s: None,
            failure_type=FailureType.CRASH,
        ))
        assert len(aegis._healings) == 2
        assert aegis._healings[0].name == "a"
        assert aegis._healings[1].name == "b"

    @pytest.mark.asyncio
    async def test_first_matching_healing_is_used(self):
        """Only the first triggered healing runs; the second is not evaluated."""
        aegis = _make_aegis()
        ran = []

        # Condition checks "done"; repair sets it True so the condition resolves.
        aegis.declare_healing(AEGISHealing(
            name="first",
            failure_condition=lambda s: not s.get("done"),
            repair_fn=lambda s: ran.append("first") or {"done": True},
            failure_type=FailureType.CRASH,
        ))
        aegis.declare_healing(AEGISHealing(
            name="second",
            failure_condition=lambda s: not s.get("done"),
            repair_fn=lambda s: ran.append("second") or {"done": True},
            failure_type=FailureType.CRASH,
        ))
        result = await aegis.check_and_heal({"done": False})
        assert result.healed is True
        assert result.healing_name == "first"
        assert ran == ["first"]


# ===========================================================================
# Layer 2: TripChatBot._replan_affected_days
# ===========================================================================

class TestReplanAffectedDays:

    def test_matching_city_returns_day_plans(self):
        bot = _make_bot()
        # Montreal appears on day index 0 (day 1) and day index 2 (day 3)
        with patch.object(bot, "_plan_day", return_value=[MagicMock()]) as mock_plan:
            result = bot._replan_affected_days({
                "location": "Montreal",
                "new_weather": WeatherCondition.RAINY,
            })

        assert result is not None
        assert len(result["affected_days"]) == 2          # day 1 and day 3
        assert set(result["affected_days"]) == {1, 3}
        assert 0 in result["day_plans"]                   # day index 0
        assert 2 in result["day_plans"]                   # day index 2

    def test_unrecognised_city_returns_none(self):
        bot = _make_bot()
        result = bot._replan_affected_days({
            "location": "Tokyo",
            "new_weather": WeatherCondition.RAINY,
        })
        assert result is None

    def test_empty_location_returns_none(self):
        bot = _make_bot()
        result = bot._replan_affected_days({
            "location": "",
            "new_weather": WeatherCondition.RAINY,
        })
        assert result is None

    def test_case_insensitive_city_match(self):
        bot = _make_bot()
        with patch.object(bot, "_plan_day", return_value=[MagicMock()]):
            result = bot._replan_affected_days({
                "location": "MONTREAL",
                "new_weather": WeatherCondition.RAINY,
            })
        assert result is not None
        assert len(result["affected_days"]) == 2

    def test_returned_day_plan_has_correct_weather(self):
        bot = _make_bot()
        with patch.object(bot, "_plan_day", return_value=[MagicMock()]):
            result = bot._replan_affected_days({
                "location": "Montreal",
                "new_weather": WeatherCondition.RAINY,
            })
        for day_plan in result["day_plans"].values():
            assert day_plan.weather == WeatherCondition.RAINY

    def test_returned_day_plan_has_correct_city(self):
        bot = _make_bot()
        with patch.object(bot, "_plan_day", return_value=[MagicMock()]):
            result = bot._replan_affected_days({
                "location": "Montreal",
                "new_weather": WeatherCondition.SUNNY,
            })
        for day_plan in result["day_plans"].values():
            assert day_plan.city == "Montreal"

    def test_plan_day_failure_skips_that_day(self):
        """If _plan_day returns None for one day, skip it; succeed on others."""
        bot = _make_bot()
        calls = []

        def plan_day_side_effect(day_idx, weather):
            calls.append(day_idx)
            # Fail for day index 0 (Montreal day 1), succeed for day index 2 (Montreal day 3)
            return None if day_idx == 0 else [MagicMock()]

        with patch.object(bot, "_plan_day", side_effect=plan_day_side_effect):
            result = bot._replan_affected_days({
                "location": "Montreal",
                "new_weather": WeatherCondition.RAINY,
            })

        assert result is not None
        assert result["affected_days"] == [3]    # only day 3 succeeded
        assert 2 in result["day_plans"]
        assert 0 not in result["day_plans"]

    def test_all_days_fail_returns_none(self):
        bot = _make_bot()
        with patch.object(bot, "_plan_day", return_value=None):
            result = bot._replan_affected_days({
                "location": "Montreal",
                "new_weather": WeatherCondition.RAINY,
            })
        assert result is None

    def test_plan_day_called_with_correct_weather(self):
        bot = _make_bot()
        call_args = []

        def capture(day_idx, weather):
            call_args.append((day_idx, weather))
            return [MagicMock()]

        with patch.object(bot, "_plan_day", side_effect=capture):
            bot._replan_affected_days({
                "location": "Quebec City",
                "new_weather": WeatherCondition.RAINY,
            })

        # Quebec City is only day index 1
        assert len(call_args) == 1
        assert call_args[0] == (1, WeatherCondition.RAINY)


# ===========================================================================
# Layer 3: _on_external_weather_change — event → AEGIS → notification
# ===========================================================================

def _weather_event(location: str, new_condition: str) -> WeatherChangedEvent:
    return WeatherChangedEvent(
        event_type=EventType.WEATHER_CHANGED,
        source_agent_id="weather_agent",
        previous_condition="sunny",
        new_condition=new_condition,
        location=location,
        payload={"location": location, "new_condition": new_condition},
    )


class TestOnExternalWeatherChange:

    @pytest.mark.asyncio
    async def test_no_trip_planned_sends_warning_notification(self):
        bot = _make_bot()
        bot.state.trip_planned = False

        await bot.setup()
        await bot._on_external_weather_change(_weather_event("Montreal", "rainy"))

        note = bot.notification_queue.get_nowait()
        assert "no trip is planned yet" in note

    @pytest.mark.asyncio
    async def test_no_trip_planned_skips_healing(self):
        bot = _make_bot()
        bot.state.trip_planned = False

        await bot.setup()
        with patch.object(bot, "_replan_affected_days") as mock_replan:
            await bot._on_external_weather_change(_weather_event("Montreal", "rainy"))

        mock_replan.assert_not_called()

    @pytest.mark.asyncio
    async def test_missing_location_produces_no_notification(self):
        bot = _make_bot()
        bot.state.trip_planned = True

        await bot.setup()
        bad_event = WeatherChangedEvent(
            event_type=EventType.WEATHER_CHANGED,
            source_agent_id="weather_agent",
            previous_condition="sunny",
            new_condition="rainy",
            location=None,          # missing
            payload={},
        )
        await bot._on_external_weather_change(bad_event)
        assert bot.notification_queue.empty()

    @pytest.mark.asyncio
    async def test_missing_condition_produces_no_notification(self):
        bot = _make_bot()
        bot.state.trip_planned = True

        await bot.setup()
        bad_event = WeatherChangedEvent(
            event_type=EventType.WEATHER_CHANGED,
            source_agent_id="weather_agent",
            previous_condition="sunny",
            new_condition=None,     # missing
            location="Montreal",
            payload={},
        )
        await bot._on_external_weather_change(bad_event)
        assert bot.notification_queue.empty()

    @pytest.mark.asyncio
    async def test_successful_heal_sends_replanned_notification(self):
        bot = _make_bot()
        bot.state.trip_planned = True

        fake_plan = MagicMock(spec=DayPlan)
        # weather_changed: False tells AEGIS the condition is now resolved.
        with patch.object(
            bot,
            "_replan_affected_days",
            return_value={
                "day_plans": {0: fake_plan},
                "affected_days": [1],
                "weather_changed": False,
            },
        ):
            await bot.setup()
            await bot._on_external_weather_change(_weather_event("Montreal", "rainy"))

        note = bot.notification_queue.get_nowait()
        assert "Auto-replanned" in note
        assert "Montreal" in note
        assert "rainy" in note
        assert "1" in note          # day number

    @pytest.mark.asyncio
    async def test_successful_heal_updates_bot_state(self):
        bot = _make_bot()
        bot.state.trip_planned = True

        fake_plan = MagicMock(spec=DayPlan)
        with patch.object(
            bot,
            "_replan_affected_days",
            return_value={
                "day_plans": {0: fake_plan},
                "affected_days": [1],
                "weather_changed": False,
            },
        ):
            await bot.setup()
            await bot._on_external_weather_change(_weather_event("Montreal", "rainy"))

        assert bot.state.day_plans[0] is fake_plan

    @pytest.mark.asyncio
    async def test_failed_heal_sends_manual_review_notification(self):
        bot = _make_bot()
        bot.state.trip_planned = True

        with patch.object(bot, "_replan_affected_days", return_value=None):
            await bot.setup()
            await bot._on_external_weather_change(_weather_event("Montreal", "rainy"))

        note = bot.notification_queue.get_nowait()
        assert "Manual review needed" in note
        assert "Montreal" in note

    @pytest.mark.asyncio
    async def test_failed_heal_does_not_modify_bot_state(self):
        bot = _make_bot()
        bot.state.trip_planned = True
        original_plans = dict(bot.state.day_plans)

        with patch.object(bot, "_replan_affected_days", return_value=None):
            await bot.setup()
            await bot._on_external_weather_change(_weather_event("Montreal", "rainy"))

        assert bot.state.day_plans == original_plans

    @pytest.mark.asyncio
    async def test_aegis_retries_repair_before_giving_up(self):
        """Verify repair is attempted max_attempts times, not just once."""
        bot = _make_bot()
        bot.state.trip_planned = True
        call_count = []

        def flaky_replan(s):
            call_count.append(1)
            return None

        with patch.object(bot, "_replan_affected_days", side_effect=flaky_replan):
            await bot.setup()
            await bot._on_external_weather_change(_weather_event("Montreal", "rainy"))

        healing = bot.aegis._healings[0]
        assert len(call_count) == healing.max_attempts

    @pytest.mark.asyncio
    async def test_event_from_self_is_ignored(self):
        """Events originating from TripChatBot itself must be silently dropped."""
        bot = _make_bot()
        bot.state.trip_planned = True

        await bot.setup()
        self_event = _weather_event("Montreal", "rainy")
        self_event.source_agent_id = bot.agent_id   # came from us

        await bot.handle_event(self_event)
        assert bot.notification_queue.empty()

    @pytest.mark.asyncio
    async def test_setup_registers_exactly_one_healing(self):
        bot = _make_bot()
        assert len(bot.aegis._healings) == 0
        await bot.setup()
        assert len(bot.aegis._healings) == 1
        assert bot.aegis._healings[0].name == "weather_change"

    @pytest.mark.asyncio
    async def test_rainy_weather_uses_rainy_condition_enum(self):
        bot = _make_bot()
        bot.state.trip_planned = True
        captured_state = []

        def capture_and_succeed(s):
            captured_state.append(s)
            return {"day_plans": {0: MagicMock()}, "affected_days": [1], "weather_changed": False}

        with patch.object(bot, "_replan_affected_days", side_effect=capture_and_succeed):
            await bot.setup()
            await bot._on_external_weather_change(_weather_event("Montreal", "rainy"))

        assert captured_state[0]["new_weather"] == WeatherCondition.RAINY

    @pytest.mark.asyncio
    async def test_sunny_weather_uses_sunny_condition_enum(self):
        bot = _make_bot()
        bot.state.trip_planned = True
        captured_state = []

        def capture_and_succeed(s):
            captured_state.append(s)
            return {"day_plans": {0: MagicMock()}, "affected_days": [1], "weather_changed": False}

        with patch.object(bot, "_replan_affected_days", side_effect=capture_and_succeed):
            await bot.setup()
            await bot._on_external_weather_change(_weather_event("Montreal", "sunny"))

        assert captured_state[0]["new_weather"] == WeatherCondition.SUNNY
