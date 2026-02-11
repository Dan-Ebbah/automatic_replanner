from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional, List, Dict, FrozenSet, Set

from aegis.state import Action


class WeatherCondition(Enum):
    SUNNY = "sunny"
    RAINY = "rainy"
    CLOUDY = "cloudy"
    SNOWY = "snowy"
    WINDY = "windy"
    STORMY = "stormy"
    ANY = "any"

class ActivityLocation(Enum):
    INDOOR = "indoor"
    OUTDOOR = "outdoor"
    VIRTUAL = "virtual"

@dataclass
class Activity:
    name: str
    description: str
    duration_minutes: int
    location: str
    weather: WeatherCondition
    category: ActivityLocation
    preconditions: FrozenSet[str] = frozenset()
    positive_effects: FrozenSet[str] = frozenset()
    negative_effects: FrozenSet[str] = frozenset()
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

    def to_action(self) -> Action:
        return Action(
            name=self.name,
            agent_id="activity_planner",
            preconditions=self.preconditions,
            positive_effects=self.positive_effects,
            negative_effects=self.negative_effects
        )



@dataclass(frozen=True)
class ActivityAction(Action):
    weather_requirement: WeatherCondition = WeatherCondition.SUNNY
    duration_minutes: int = 60

    def is_applicable_with_weather(self, state: Set[str], weather: WeatherCondition) -> bool:
        return self._is_applicable(state) and (weather == self.weather_requirement or self.weather_requirement == WeatherCondition.ANY)




class ActivityRegistry:
    def __init__(self):
        self.activities: Dict[str, Activity] = {}

    def register(self, activity: Activity):
        self.activities[activity.name] = activity

    def get_outdoor_activities(self) -> List[Activity]:
        return [activity for activity in self.activities.values() if activity.category == ActivityLocation.OUTDOOR]

    def get_alternatives(self, activity_name: str) -> List[Activity]:
        target_activity = self.activities.get(activity_name)
        if not target_activity:
            return []
        return [
            activity for activity in self.activities.values()
            if activity.category == target_activity.category and activity.name != activity_name
        ]
def test_activity_action():
    park_action = ActivityAction(
        name="Visit Park",
        agent_id="activity_planner",
        preconditions=frozenset({"good_weather"}),
        positive_effects=frozenset({"relaxed"}),
        negative_effects=frozenset(),
        weather_requirement=WeatherCondition.SUNNY,
        duration_minutes=120
    )

    state = {"good_weather"}
    print(park_action.is_applicable_with_weather(state, WeatherCondition.SUNNY))
    print(park_action.is_applicable_with_weather(state, WeatherCondition.RAINY))

def quick_demo():
    hiking_activity = Activity(name="Hiking", description="A refreshing hike through the mountains.",
                               duration_minutes=180,
                               location="Blue Ridge Mountains", weather=WeatherCondition.SUNNY,
                               category=ActivityLocation.OUTDOOR)

    reading_activity = Activity(name="Reading", description="Reading a novel in a cozy cafe.", duration_minutes=120,
                                location="Downtown Cafe", weather=WeatherCondition.CLOUDY,
                                category=ActivityLocation.INDOOR)

    cycling_activity = Activity(name="Cycling", description="Cycling along the river trail.", duration_minutes=90,
                                location="River Trail", weather=WeatherCondition.WINDY,
                                category=ActivityLocation.OUTDOOR)

    swimming_activity = Activity(name="Swimming", description="Swimming in the community pool.", duration_minutes=60,
                                 location="Community Pool", weather=WeatherCondition.SUNNY,
                                 category=ActivityLocation.INDOOR)

    yoga_activity = Activity(name="Yoga", description="Morning yoga session in the studio.", duration_minutes=75,
                             location="Central Park", weather=WeatherCondition.SNOWY, category=ActivityLocation.INDOOR)

    registry = ActivityRegistry()
    registry.register(hiking_activity)
    registry.register(reading_activity)
    registry.register(cycling_activity)
    registry.register(swimming_activity)
    registry.register(yoga_activity)

    outdoor_activities = registry.get_outdoor_activities()
    print(f"Found {len(outdoor_activities)} outdoor activities:")


if __name__ == "__main__":
    quick_demo()
