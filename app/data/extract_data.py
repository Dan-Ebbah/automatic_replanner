import logging
import os

from dotenv import load_dotenv
from pymongo import MongoClient
from bson import ObjectId

load_dotenv()

logger = logging.getLogger(__name__)

MONGO_URI = os.environ.get("MONGO_URI", "mongodb+srv://test:12345@cluster0.wfmgf.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
# Add timeout to fail fast if MongoDB is unreachable
logger.debug("Connecting to MongoDB (host: %s)", MONGO_URI.split("@")[-1].split("/")[0] if "@" in MONGO_URI else "unknown")
client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=2000)
DB_NAME = "plan_info_db"
db = client[DB_NAME]
places = db["plans_template"]
logger.info("MongoDB client initialised — db=%s, collection=plans_template", DB_NAME)

# Indoor activity type keywords for filtering places
_INDOOR_ACTIVITY_TYPES = {"museum", "cultural", "dining", "shopping", "entertainment", "art", "historical"}
# Indoor location keywords that override outdoor activity types
_INDOOR_LOCATION_KEYWORDS = {"museum", "basilica", "church", "oratory", "cathedral", "gallery", "centre", "center"}
# Outdoor location keywords
_OUTDOOR_LOCATION_KEYWORDS = {"park", "falls", "old port", "plains", "garden", "beach", "trail"}


def _infer_weather_suitability(activity: dict) -> str:
    """Determine weather suitability from activity type and location keywords."""
    act_type = activity.get("activity", "").lower()
    location = activity.get("location", "").lower()

    # Meals and hotels are always fine
    if act_type in ("breakfast", "lunch", "dinner", "hotel check-in", "hotel checkout", "museum"):
        return "any"

    # For sightseeing/parks, check location keywords
    if any(kw in location for kw in _INDOOR_LOCATION_KEYWORDS):
        return "any"
    if any(kw in location for kw in _OUTDOOR_LOCATION_KEYWORDS):
        return "sunny"

    # Default: outdoor sightseeing
    if act_type in ("sightseeing", "park", "historic site"):
        return "sunny"

    return "any"


def get_indoor_places(city: str) -> list[dict]:
    """Query the places collection for indoor venues in the given city."""
    try:
        places_collection = db["places"]
        # category is a singular string field; match case-insensitively
        indoor_categories_re = "^(Restaurant|Museum|Attraction|museum|tourist_attraction|Dinner|Lunch|Breakfast)$"
        query = {
            "city": {"$regex": f"^{city}$", "$options": "i"},
            "category": {
                "$regex": indoor_categories_re,
                "$options": "i",
            },
        }

        results = []
        for doc in places_collection.find(query):
            # Skip parks — category can be a string or a list
            raw_cat = doc.get("category", "")
            cats = raw_cat if isinstance(raw_cat, list) else [raw_cat]
            if any(c.lower() in ("park", "sightseeing") for c in cats if c):
                continue
            # Require indoor activity_types — skip docs without them
            activity_types = doc.get("activity_types") or []
            if isinstance(activity_types, str):
                activity_types = [activity_types]
            if not any(t.lower() in _INDOOR_ACTIVITY_TYPES for t in activity_types):
                continue
            results.append({
                "name": doc.get("name", "Unknown"),
                "duration_minutes": doc.get("duration_minutes", 90),
            })

        logger.info("Found %d indoor places for city=%s", len(results), city)
        return results
    except Exception as e:
        logger.error("Error querying indoor places for %s: %s", city, e, exc_info=True)
        return []


def build_weather_and_alternatives(trip_data: dict) -> tuple[dict, dict]:
    """Build weather_maps and indoor_alternatives from trip data and MongoDB places.

    Returns:
        (weather_maps, indoor_alternatives) — same structure as sample_trip.py constants.
    """
    weather_maps = {}
    indoor_alternatives = {}

    for day_idx, day in enumerate(trip_data.get("days", [])):
        city = day.get("city", "")
        day_weather = {}
        day_alts = {}

        for slot_idx, activity in enumerate(day.get("plan", [])):
            location = activity.get("location", "")
            suitability = _infer_weather_suitability(activity)
            day_weather[location] = suitability

            # If outdoor, provide indoor alternatives from MongoDB
            if suitability == "sunny":
                indoor = get_indoor_places(city)
                if indoor:
                    # Exclude the current location from alternatives
                    filtered = [p for p in indoor if p["name"].lower() != location.lower()]
                    if filtered:
                        day_alts[slot_idx] = filtered

        weather_maps[day_idx] = day_weather
        if day_alts:
            indoor_alternatives[day_idx] = day_alts

    logger.info(
        "Built weather maps for %d days, indoor alternatives for %d days",
        len(weather_maps), len(indoor_alternatives),
    )
    return weather_maps, indoor_alternatives


def get_place():
    """Fetch trip data from MongoDB. Returns None if connection fails."""
    try:
        result_json = {"days": []}
        example_place_id = "68518162cd354dcd6c65cb47"
        logger.info("Querying MongoDB for place_id=%s", example_place_id)
        result = places.find_one({"_id": ObjectId(example_place_id)})

        if not result:
            logger.warning("No document found for place_id=%s", example_place_id)
            return None

        logger.info("Found document: city=%s, duration_days=%s", result.get("city"), result.get("duration_days"))

        for day_num in range(1, result['duration_days'] + 1):
            day_key = str(day_num)
            if 'plan' in result and day_key in result['plan']:
                activities = [
                    {
                        "activity": activity['activity'],
                        "location": activity['location'],
                        "start_time": activity['start_time'],
                        "end_time": activity['end_time'],
                        "weather_suitability": activity.get('weather_suitability', "sunny")
                    }
                    for activity in result['plan'][day_key]
                ]
            else:
                activities = []

            logger.debug("Day %d: %d activities loaded", day_num, len(activities))
            result_json["days"].append({
                "day": day_num,
                "city": result['city'],
                "plan": activities
            })

        total = sum(len(d["plan"]) for d in result_json["days"])
        logger.info("MongoDB load complete — %d days, %d total activities", len(result_json["days"]), total)
        return result_json
    except Exception as e:
        logger.error("Could not fetch data from MongoDB: %s", e, exc_info=True)
        return None


