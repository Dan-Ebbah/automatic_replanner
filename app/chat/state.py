from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional
from datetime import datetime

from demos.activities import ActivityAction, WeatherCondition

class Intent(Enum):
    PLAN_TRIP = auto()
    SHOW_ITINERARY = auto()
    WEATHER_CHANGE = auto()
    SWAP_ACTIVITY = auto()
    LIST_ACTIVITIES = auto()
    HELP = auto()
    GENERAL = auto()

@dataclass
class Message:
    role: str
    content: str
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class DayPlan:
    day_number: int
    city: str
    actions: list[ActivityAction]
    weather: WeatherCondition = WeatherCondition.SUNNY

@dataclass
class ChatState:
    conversation: list[Message] = field(default_factory=list)
    trip_planned: bool = False
    day_plans: dict[int, DayPlan] = field(default_factory=dict)
    current_day_focus: Optional[int] = None

    def add_message(self, role: str, content: str):
        self.conversation.append(Message(role=role, content=content))