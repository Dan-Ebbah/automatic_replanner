SAMPLE_TRIP = {
    "days": [
        {
            "day": 1,
            "city": "Montreal",
            "plan": [
                {
                    "activity": "sightseeing",
                    "location": "Old Montreal",
                    "start_time": "9:00 AM",
                    "end_time": "12:00 PM",
                    "weather_suitability": "sunny"
                },
                {
                    "activity": "museum",
                    "location": "Montreal Museum of Fine Arts",
                    "start_time": "1:00 PM",
                    "end_time": "3:00 PM",
                    "weather_suitability": "any"
                },
                {
                    "activity": "park",
                    "location": "Mount Royal Park",
                    "start_time": "4:00 PM",
                    "end_time": "6:00 PM",
                    "weather_suitability": "sunny"
                },
                {
                    "activity": "dinner",
                    "location": "Schwartz's Deli",
                    "start_time": "7:00 PM",
                    "end_time": "8:30 PM",
                    "weather_suitability": "any"
                }
            ]
        },
        {
            "day": 2,
            "city": "Quebec City",
            "plan": [
                {
                    "activity": "historic site",
                    "location": "Plains of Abraham",
                    "start_time": "9:00 AM",
                    "end_time": "11:00 AM",
                    "weather_suitability": "sunny"
                },
                {
                    "activity": "sightseeing",
                    "location": "Old Quebec",
                    "start_time": "12:00 PM",
                    "end_time": "3:00 PM",
                    "weather_suitability": "sunny"
                },
                {
                    "activity": "museum",
                    "location": "Musee de la Civilisation",
                    "start_time": "3:30 PM",
                    "end_time": "5:30 PM",
                    "weather_suitability": "any"
                },
                {
                    "activity": "dinner",
                    "location": "Chez Muffy",
                    "start_time": "6:00 PM",
                    "end_time": "8:00 PM",
                    "weather_suitability": "any"
                }
            ]
        },
        {
            "day": 3,
            "city": "Montreal",
            "plan": [
                {
                    "location": "Chez Hailar",
                    "activity": "Breakfast",
                    "start_time": "8:00 AM",
                    "end_time": "9:00 AM",
                    "weather_suitability": "any"
                },
                {
                    "location": "Montreal Museum of Fine Arts",
                    "activity": "Sightseeing",
                    "start_time": "9:30 AM",
                    "end_time": "11:30 AM",
                    "weather_suitability": "any"
                },
                {
                    "location": "Restaurant L'Autre Saison",
                    "activity": "Lunch",
                    "start_time": "12:00 PM",
                    "end_time": "1:00 PM",
                    "weather_suitability": "any"
                },
                {
                    "location": "Mount Royal Park",
                    "activity": "Sightseeing",
                    "start_time": "1:30 PM",
                    "end_time": "3:30 PM",
                    "weather_suitability": "sunny"
                },
                {
                    "location": "Saint Joseph's Oratory",
                    "activity": "Sightseeing",
                    "start_time": "4:00 PM",
                    "end_time": "5:30 PM",
                    "weather_suitability": "sunny"
                },
                {
                    "location": "Gaspar Brasserie Francaise",
                    "activity": "Dinner",
                    "start_time": "6:30 PM",
                    "end_time": "8:00 PM",
                    "weather_suitability": "any"
                },
            ]
        }
    ]
}

# Weather suitability maps per day index (location -> weather string)
WEATHER_MAPS = {
    0: {  # Day 1 - Montreal
        "Old Montreal": "sunny",
        "Montreal Museum of Fine Arts": "any",
        "Mount Royal Park": "sunny",
        "Schwartz's Deli": "any",
    },
    1: {  # Day 2 - Quebec City
        "Plains of Abraham": "sunny",
        "Old Quebec": "sunny",
        "Musee de la Civilisation": "any",
        "Chez Muffy": "any",
    },
    2: {  # Day 3 - Montreal
        "Chez Hailar": "any",
        "Montreal Museum of Fine Arts": "any",
        "Restaurant L'Autre Saison": "any",
        "Mount Royal Park": "sunny",
        "Saint Joseph's Oratory": "sunny",
        "Gaspar Brasserie Francaise": "any",
    },
}

# Indoor alternatives per day index, keyed by slot index
INDOOR_ALTERNATIVES = {
    0: {  # Day 1
        0: [  # Replacements for Old Montreal (outdoor sightseeing)
            {"name": "Pointe-a-Calliere Museum", "duration_minutes": 120},
            {"name": "Montreal Science Centre", "duration_minutes": 120},
        ],
        2: [  # Replacements for Mount Royal Park
            {"name": "Montreal Biodome", "duration_minutes": 120},
            {"name": "Underground City", "duration_minutes": 90},
        ],
    },
    1: {  # Day 2
        0: [  # Replacements for Plains of Abraham
            {"name": "Musee National des Beaux-Arts du Quebec", "duration_minutes": 120},
        ],
        1: [  # Replacements for Old Quebec (outdoor sightseeing)
            {"name": "Musee de la Civilisation", "duration_minutes": 120},
            {"name": "Chateau Frontenac Tour", "duration_minutes": 90},
        ],
    },
    2: {  # Day 3
        3: [  # Replacements for Mount Royal Park (slot 3)
            {"name": "Montreal Biodome", "duration_minutes": 120},
            {"name": "Pointe-a-Calliere Museum", "duration_minutes": 120},
        ],
        4: [  # Replacements for Saint Joseph's Oratory (slot 4)
            {"name": "Underground City", "duration_minutes": 90},
            {"name": "McCord Stewart Museum", "duration_minutes": 90},
        ],
    },
}
