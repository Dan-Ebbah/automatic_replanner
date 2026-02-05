"""
WeatherAgent
============
Polls a MockWeatherService and publishes WeatherChangedEvent when conditions change.
This is a publish-only agent (no subscriptions).
"""

import asyncio
import logging

from aegis.agent import AgentConfig, BaseAgent
from aegis.events import Event, EventBus, WeatherChangedEvent
from demos.weather_service import MockWeatherService

logger = logging.getLogger(__name__)


class WeatherAgent(BaseAgent):
    """Monitors weather and publishes change events."""

    def __init__(
        self,
        config: AgentConfig,
        event_bus: EventBus,
        weather_service: MockWeatherService,
        location: str = "Montreal",
    ) -> None:
        super().__init__(config, event_bus)
        self._weather_service = weather_service
        self._location = location
        self._last_condition = None

    # -- lifecycle ------------------------------------------------------------

    async def run(self) -> None:
        # Capture the initial condition so the first real change is detected
        initial = self._weather_service.get_current_weather(self._location)
        self._last_condition = initial.condition
        logger.info(
            "WeatherAgent: initial condition for %s is %s",
            self._location, self._last_condition.value,
        )

        while self._running:
            await asyncio.sleep(self.config.poll_interval_seconds)
            weather = self._weather_service.get_current_weather(self._location)

            if weather.condition != self._last_condition:
                prev = self._last_condition
                self._last_condition = weather.condition
                print(
                    f"  [WeatherAgent] Detected change: "
                    f"{prev.value} -> {weather.condition.value}"
                )
                await self.publish(WeatherChangedEvent(
                    event_type=None,  # overwritten by __post_init__
                    source_agent_id=self.agent_id,
                    previous_condition=prev.value,
                    new_condition=weather.condition.value,
                    location=self._location,
                    payload={
                        "previous_condition": prev.value,
                        "new_condition": weather.condition.value,
                        "temperature": weather.temperature,
                        "humidity": weather.humidity,
                        "is_raining": weather.is_raining,
                    },
                ))

    async def handle_event(self, event: Event) -> None:
        # No subscriptions — nothing to handle.
        pass

    # -- demo convenience -----------------------------------------------------

    def inject_weather_change(self, condition) -> None:
        """Convenience for demos: inject a weather change into the underlying service."""
        self._weather_service.inject_weather_change(self._location, condition)
