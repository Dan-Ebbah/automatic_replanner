from dataclasses import dataclass
from typing import Dict

from demos.activities import WeatherCondition


@dataclass
class WeatherData:
    condition: WeatherCondition
    temperature: float
    humidity: float
    is_raining: bool

    def is_suitable_for_outdoor(self) -> bool:
        return self.condition in [WeatherCondition.SUNNY, WeatherCondition.CLOUDY] and not self.is_raining


class MockWeatherService:
    def __init__(self):
        self._weather: Dict[str, WeatherData] = {}
        self._default = WeatherData(
            condition=WeatherCondition.SUNNY,
            temperature=22.0,
            humidity=50.0,
            is_raining=False
        )

    def get_current_weather(self, location: str) -> WeatherData:
        return self._weather.get(location, self._default)


    def inject_weather_change(self, location: str, condition: WeatherCondition):
        if condition == WeatherCondition.SUNNY:
            self._weather[location] = WeatherData(
                condition=WeatherCondition.SUNNY,
                temperature=25.0,
                humidity=40.0,
                is_raining=False
            )
        elif condition == WeatherCondition.RAINY:
            self._weather[location] = WeatherData(
                condition=WeatherCondition.RAINY,
                temperature=18.0,
                humidity=90.0,
                is_raining=True
            )
        elif condition == WeatherCondition.CLOUDY:
            self._weather[location] = WeatherData(
                condition=WeatherCondition.CLOUDY,
                temperature=20.0,
                humidity=60.0,
                is_raining=False
            )


def quick_demo():
    weather_service = MockWeatherService()
    location = "Central Park"

    print(f"Initial weather in {location}: {weather_service.get_current_weather(location)}")

    weather_service.inject_weather_change(location, WeatherCondition.RAINY)
    print(f"After injecting rain, weather in {location}: {weather_service.get_current_weather(location)}")

    weather_service.inject_weather_change(location, WeatherCondition.SUNNY)
    print(f"After injecting sun, weather in {location}: {weather_service.get_current_weather(location)}")


if __name__ == "__main__":
    quick_demo()