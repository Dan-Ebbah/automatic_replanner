from datetime import datetime

from demos.activities import ActivityAction, WeatherCondition


def _semantic_tag(activity_type):
    return {
        "breakfast": "had_breakfast",
        "lunch": "had_lunch",
        "dinner": "had_dinner",
        "sightseeing": "sightseeing_done",
        "activity": "activity_done",
        "hotel check-in": "checked_in",
        "hotel checkout": "checked_out"
    }.get(activity_type, "activity_done")


def _slugify(param):
    return param.lower().replace(" ", "_")


def _map_weather(param):
    mapping = {
        "sunny": WeatherCondition.SUNNY,
        "rainy": WeatherCondition.RAINY,
        "snowy": WeatherCondition.SNOWY,
        "any": WeatherCondition.ANY
    }
    return mapping.get(param.lower(), WeatherCondition.ANY)


def _calc_duration(param, param1):
    fmt = "%I:%M %p"
    start = datetime.strptime(param, fmt)
    end = datetime.strptime(param1, fmt)
    return int((end - start).total_seconds() / 60)


def build_actions_for_day(day: dict) -> list[ActivityAction]:
    actions = []
    plan = day["plan"]

    for i, slot in enumerate(plan):
        activity_type = slot["activity"].lower()

        if i == 0:
            preconditions = frozenset({"day_started"})
        else:
            preconditions = frozenset({f"slot_{i-1}_completed"})

        semantic_tag = _semantic_tag(activity_type)
        is_last = (i == len(plan) - 1)

        positive_effects = frozenset({f"slot_{i}_completed", semantic_tag})
        if is_last:
            positive_effects = positive_effects.union({"day_completed"})

        actions.append(ActivityAction(
            name=_slugify(slot["location"]),
            agent_id="itinerary_planner",
            preconditions=preconditions,
            positive_effects=positive_effects,
            negative_effects=frozenset(),
            weather_requirement=_map_weather(slot.get("weather_suitability", "any")),
            duration_minutes=_calc_duration(slot["start_time"], slot["end_time"]),
        ))

    return actions


def build_actions_for_day_with_weather(
    day: dict, weather_map: dict[str, str]
) -> list[ActivityAction]:
    """Build actions for a day, using an external weather-suitability map.

    Args:
        day: The day JSON dict (same format as build_actions_for_day).
        weather_map: Maps activity location name → weather category string
                     (e.g. {"Mount Royal Park": "sunny", "Chez Hailar": "any"}).
                     Locations not in the map default to "any".
    """
    actions = []
    plan = day["plan"]

    for i, slot in enumerate(plan):
        activity_type = slot["activity"].lower()
        location = slot["location"]

        if i == 0:
            preconditions = frozenset({"day_started"})
        else:
            preconditions = frozenset({f"slot_{i-1}_completed"})

        semantic_tag = _semantic_tag(activity_type)
        is_last = (i == len(plan) - 1)

        positive_effects = frozenset({f"slot_{i}_completed", semantic_tag})
        if is_last:
            positive_effects = positive_effects.union({"day_completed"})

        weather_str = weather_map.get(location, "any")

        actions.append(ActivityAction(
            name=_slugify(location),
            agent_id="itinerary_planner",
            preconditions=preconditions,
            positive_effects=positive_effects,
            negative_effects=frozenset(),
            weather_requirement=_map_weather(weather_str),
            duration_minutes=_calc_duration(slot["start_time"], slot["end_time"]),
        ))

    return actions


def build_indoor_alternatives(
    outdoor_actions: list[ActivityAction],
    alternatives: dict[int, list[dict]],
) -> list[ActivityAction]:
    """Build indoor replacement actions that share the same slot pre/pos.

    Args:
        outdoor_actions: The original outdoor ActivityAction objects to replace.
        alternatives: Maps slot index → list of alternative venue dicts.
                      Each dict has keys: "name" (display name),
                      "duration_minutes" (int, optional – defaults to original).

    Returns:
        A list of ActivityAction objects with weather_requirement=RAINY and the
        same preconditions/positive_effects as the originals they replace.
    """
    result = []
    for action in outdoor_actions:
        # Find the slot index from the action's positive_effects
        slot_idx = None
        for eff in action.positive_effects:
            if eff.startswith("slot_") and eff.endswith("_completed"):
                try:
                    slot_idx = int(eff.split("_")[1])
                except ValueError:
                    continue

        if slot_idx is None or slot_idx not in alternatives:
            continue

        for alt in alternatives[slot_idx]:
            result.append(ActivityAction(
                name=_slugify(alt["name"]),
                agent_id="itinerary_planner",
                preconditions=action.preconditions,
                positive_effects=action.positive_effects,
                negative_effects=frozenset(),
                weather_requirement=WeatherCondition.RAINY,
                duration_minutes=alt.get("duration_minutes", action.duration_minutes),
            ))

    return result
