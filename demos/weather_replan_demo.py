"""
AEGIS Weather Replan Demo
=========================
Demonstrates AEGIS self-healing by replanning Day 3 of a Montreal/Quebec City
trip when the weather changes from sunny to rainy.

Run:  python -m demos.weather_replan_demo
"""

import sys
import os
from typing import List, Set, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aegis.state import Action, PlanningGraph
from demos.activities import ActivityAction, WeatherCondition
from app.services.activity_adapter import (
    build_actions_for_day_with_weather,
    build_indoor_alternatives,
)

# ============================================================================
# Day 3 itinerary data (real Montreal trip)
# ============================================================================

DAY3_DATA = {
    "day": 3,
    "city": "Montreal",
    "plan": [
        {
            "location": "Chez Hailar",
            "activity": "Breakfast",
            "start_time": "8:00 AM",
            "end_time": "9:00 AM",
        },
        {
            "location": "Montreal Museum of Fine Arts",
            "activity": "Sightseeing",
            "start_time": "9:30 AM",
            "end_time": "11:30 AM",
        },
        {
            "location": "Restaurant L'Autre Saison",
            "activity": "Lunch",
            "start_time": "12:00 PM",
            "end_time": "1:00 PM",
        },
        {
            "location": "Mount Royal Park",
            "activity": "Sightseeing",
            "start_time": "1:30 PM",
            "end_time": "3:30 PM",
        },
        {
            "location": "Saint Joseph's Oratory",
            "activity": "Sightseeing",
            "start_time": "4:00 PM",
            "end_time": "5:30 PM",
        },
        {
            "location": "Gaspar Brasserie Francaise",
            "activity": "Dinner",
            "start_time": "6:30 PM",
            "end_time": "8:00 PM",
        },
    ],
}

# Weather suitability for each location
WEATHER_MAP = {
    "Chez Hailar": "any",
    "Montreal Museum of Fine Arts": "any",
    "Restaurant L'Autre Saison": "any",
    "Mount Royal Park": "sunny",
    "Saint Joseph's Oratory": "sunny",
    "Gaspar Brasserie Francaise": "any",
}

# Indoor alternatives keyed by slot index
INDOOR_ALTERNATIVES = {
    3: [  # Replacements for Mount Royal Park (slot 3)
        {"name": "Montreal Biodome", "duration_minutes": 120},
        {"name": "Pointe-a-Calliere Museum", "duration_minutes": 120},
    ],
    4: [  # Replacements for Saint Joseph's Oratory (slot 4)
        {"name": "Underground City", "duration_minutes": 90},
        {"name": "McCord Stewart Museum", "duration_minutes": 90},
    ],
}

# Display names (slug -> pretty name) for output
DISPLAY_NAMES = {
    "chez_hailar": "Chez Hailar",
    "montreal_museum_of_fine_arts": "Montreal Museum of Fine Arts",
    "restaurant_l'autre_saison": "Restaurant L'Autre Saison",
    "mount_royal_park": "Mount Royal Park",
    "saint_joseph's_oratory": "Saint Joseph's Oratory",
    "gaspar_brasserie_francaise": "Gaspar Brasserie Francaise",
    "montreal_biodome": "Montreal Biodome",
    "pointe-a-calliere_museum": "Pointe-a-Calliere Museum",
    "underground_city": "Underground City",
    "mccord_stewart_museum": "McCord Stewart Museum",
}

OUTDOOR_SLOTS = {3, 4}  # Slot indices of outdoor activities


# ============================================================================
# Helpers
# ============================================================================

def _display(action: ActivityAction) -> str:
    """Pretty-print name for an action."""
    return DISPLAY_NAMES.get(action.name, action.name)


def _activity_type(action: ActivityAction) -> str:
    """Infer the activity type from semantic tags in positive_effects."""
    tag_map = {
        "had_breakfast": "Breakfast",
        "had_lunch": "Lunch",
        "had_dinner": "Dinner",
        "sightseeing_done": "Sightseeing",
        "activity_done": "Activity",
        "checked_in": "Hotel Check-in",
        "checked_out": "Hotel Checkout",
    }
    for eff in action.positive_effects:
        if eff in tag_map:
            return tag_map[eff]
    return "Activity"


def _slot_index(action: ActivityAction) -> int:
    """Extract the slot index from an action's positive_effects."""
    for eff in action.positive_effects:
        if eff.startswith("slot_") and eff.endswith("_completed"):
            try:
                return int(eff.split("_")[1])
            except ValueError:
                continue
    return -1


def flatten_plan(plan: Optional[List[Set[Action]]]) -> List[ActivityAction]:
    """Flatten a GraphPlan parallel-step plan into a slot-ordered action list."""
    if plan is None:
        return []
    actions = []
    for step in plan:
        actions.extend(step)
    actions.sort(key=_slot_index)
    return actions


# ============================================================================
# Pool builders
# ============================================================================

def build_sunny_pool() -> List[ActivityAction]:
    """Build the action pool for sunny weather (original itinerary)."""
    all_actions = build_actions_for_day_with_weather(DAY3_DATA, WEATHER_MAP)
    return [
        a for a in all_actions
        if a.weather_requirement in (WeatherCondition.SUNNY, WeatherCondition.ANY)
    ]


def build_rainy_pool() -> List[ActivityAction]:
    """Build the action pool for rainy weather (swap outdoor for indoor)."""
    all_actions = build_actions_for_day_with_weather(DAY3_DATA, WEATHER_MAP)

    # Keep only ANY-weather actions (meals, indoor sightseeing)
    rainy_actions = [
        a for a in all_actions
        if a.weather_requirement == WeatherCondition.ANY
    ]

    # Identify the outdoor actions that need replacing
    outdoor_actions = [
        a for a in all_actions
        if a.weather_requirement == WeatherCondition.SUNNY
    ]

    # Build indoor alternatives with matching slot pre/pos
    indoor_alts = build_indoor_alternatives(outdoor_actions, INDOOR_ALTERNATIVES)

    # Add RAINY alternatives to the pool
    rainy_actions.extend(indoor_alts)

    return rainy_actions


# ============================================================================
# Planning
# ============================================================================

def run_plan(actions: List[ActivityAction]) -> List[ActivityAction]:
    """Run PlanningGraph and return the flattened, ordered plan."""
    initial_state = {"day_started"}
    goal = {"day_completed", "had_dinner"}

    graph = PlanningGraph(initial_state, goal, actions)
    success = graph.build_graph()
    if not success:
        print("  ERROR: PlanningGraph could not find a valid plan!")
        return []

    plan = graph.extract_plan()
    return flatten_plan(plan)


# ============================================================================
# Output
# ============================================================================

def print_plan(title: str, actions: List[ActivityAction], highlight_outdoor=False):
    print(f"\n{title}")
    for i, action in enumerate(actions):
        name = _display(action)
        atype = _activity_type(action)
        slot = _slot_index(action)
        tag = ""
        if highlight_outdoor and slot in OUTDOOR_SLOTS:
            if action.weather_requirement == WeatherCondition.SUNNY:
                tag = "  <-- OUTDOOR"
            elif action.weather_requirement == WeatherCondition.RAINY:
                tag = "  <-- INDOOR replacement"
        print(f"  {i+1}. {name} ({atype}){tag}")


def print_swaps(sunny: List[ActivityAction], rainy: List[ActivityAction]):
    print("\nSWAPPED:")
    sunny_by_slot = {_slot_index(a): a for a in sunny}
    rainy_by_slot = {_slot_index(a): a for a in rainy}
    for slot in sorted(OUTDOOR_SLOTS):
        old = sunny_by_slot.get(slot)
        new = rainy_by_slot.get(slot)
        if old and new and old.name != new.name:
            print(f"  - {_display(old)} -> {_display(new)}")


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 60)
    print("  AEGIS Weather Replan Demo")
    print("  Day 3 — Montreal")
    print("=" * 60)

    # --- Sunny plan ---
    sunny_pool = build_sunny_pool()
    sunny_plan = run_plan(sunny_pool)
    print_plan("=== ORIGINAL ITINERARY (Sunny Day) ===", sunny_plan, highlight_outdoor=True)

    # --- Weather alert ---
    print("\n!!! WEATHER ALERT: Rain starting in Montreal !!!")

    # --- Rainy plan ---
    rainy_pool = build_rainy_pool()
    rainy_plan = run_plan(rainy_pool)
    print_plan("=== REPLANNED ITINERARY (Rainy Day) ===", rainy_plan, highlight_outdoor=True)

    # --- Comparison ---
    print_swaps(sunny_plan, rainy_plan)

    # --- Verification ---
    print("\n--- Verification ---")
    sunny_state = {"day_started"}
    for a in sunny_plan:
        sunny_state = a.apply(sunny_state)
    rainy_state = {"day_started"}
    for a in rainy_plan:
        rainy_state = a.apply(rainy_state)

    checks = [
        ("Sunny plan has 6 activities", len(sunny_plan) == 6),
        ("Rainy plan has 6 activities", len(rainy_plan) == 6),
        ("Sunny plan achieves day_completed", "day_completed" in sunny_state),
        ("Sunny plan achieves had_dinner", "had_dinner" in sunny_state),
        ("Rainy plan achieves day_completed", "day_completed" in rainy_state),
        ("Rainy plan achieves had_dinner", "had_dinner" in rainy_state),
    ]
    all_pass = True
    for label, ok in checks:
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_pass = False
        print(f"  [{status}] {label}")

    if all_pass:
        print("\nAll checks passed!")
    else:
        print("\nSome checks FAILED.")
        sys.exit(1)


if __name__ == "__main__":
    main()
