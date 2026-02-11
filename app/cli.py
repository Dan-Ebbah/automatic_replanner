import asyncio

from aegis.config import AEGISConfig
from aegis.agent import AgentConfig
from aegis.events import EventBus
from demos.activities import WeatherCondition
from demos.weather_service import MockWeatherService
from demos.agents.weather_agent import WeatherAgent
from app.chat import TripChatBot

BANNER = """
========================================
  AEGIS Trip Planner (Event-Driven)
========================================
  Agents active:
    - TripChatBot   (plans & replans)
    - WeatherAgent  (monitors weather)

  Commands:
    plan my trip     — generate itinerary
    show itinerary   — view current plan
    inject rain      — simulate rainy weather
    inject sunny     — simulate sunny weather
    help             — list capabilities
    quit / exit      — shut down
========================================
"""

POLL_INTERVAL = 2.0  # seconds between WeatherAgent polls


def _drain_notifications(bot: TripChatBot) -> None:
    """Print and clear all queued notifications."""
    while not bot.notification_queue.empty():
        try:
            msg = bot.notification_queue.get_nowait()
            print(f"\n  [AEGIS Notification] {msg}")
        except asyncio.QueueEmpty:
            break


async def main():
    # -- shared infrastructure ------------------------------------------------
    event_bus = EventBus()
    weather_service = MockWeatherService()
    aegis_config = AEGISConfig()

    # -- agents ---------------------------------------------------------------
    bot = TripChatBot(
        config=aegis_config,
        event_bus=event_bus,
        agent_config=AgentConfig(
            agent_id="trip_chatbot",
            name="TripChatBot",
            description="Interactive trip-planning chatbot",
        ),
    )

    weather_agent = WeatherAgent(
        config=AgentConfig(
            agent_id="weather_agent",
            name="WeatherAgent",
            description="Monitors weather and publishes change events",
            poll_interval_seconds=POLL_INTERVAL,
        ),
        event_bus=event_bus,
        weather_service=weather_service,
        location="Montreal",
    )

    # -- start agents ---------------------------------------------------------
    await bot.start()
    await weather_agent.start()

    print(BANNER)

    loop = asyncio.get_event_loop()

    try:
        while True:
            # Drain any async notifications that arrived while waiting
            _drain_notifications(bot)

            try:
                user_input = await loop.run_in_executor(None, input, "You: ")
                user_input = user_input.strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break

            if user_input.lower() in ("quit", "exit"):
                print("Goodbye!")
                break
            if not user_input:
                continue

            # -- demo inject commands -----------------------------------------
            if user_input.lower().startswith("inject "):
                condition_str = user_input.lower().split("inject ", 1)[1].strip()
                cond_map = {
                    "rain": WeatherCondition.RAINY,
                    "rainy": WeatherCondition.RAINY,
                    "sunny": WeatherCondition.SUNNY,
                    "sun": WeatherCondition.SUNNY,
                }
                condition = cond_map.get(condition_str)
                if condition is None:
                    print(f"  Unknown condition '{condition_str}'. Use: rain, sunny")
                    continue

                print(f"  Injecting {condition.value} weather into MockWeatherService...")
                weather_agent.inject_weather_change(condition)

                # Wait for the agent to poll and dispatch the event
                await asyncio.sleep(POLL_INTERVAL + 0.5)
                _drain_notifications(bot)
                print()
                continue

            # -- normal chat --------------------------------------------------
            response = await bot.handle_message(user_input)
            print(f"\nAEGIS: {response}\n")

            # Drain notifications that may have been generated during handling
            _drain_notifications(bot)

    finally:
        await bot.stop()
        await weather_agent.stop()


if __name__ == "__main__":
    asyncio.run(main())
