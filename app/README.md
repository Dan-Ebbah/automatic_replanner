# AEGIS Trip Planner — `app/` Package

Interactive, event-driven trip planning chatbot with LLM-powered intent parsing, weather-aware itinerary generation, and real-time notifications.

## Directory Structure

```
app/
├── api.py                  # FastAPI server (HTTP + WebSocket)
├── cli.py                  # Interactive CLI interface
├── prompts.py              # LLM system prompts (intent parsing, response formatting)
├── chat/
│   ├── bot.py              # TripChatBot agent (core logic)
│   └── state.py            # State models (ChatState, DayPlan, Intent, Message)
├── data/
│   ├── extract_data.py     # MongoDB data extraction + weather map builder
│   └── sample_trip.py      # Hardcoded 3-day Montreal/Quebec trip (fallback)
└── services/
    └── activity_adapter.py # Converts trip data → ActivityAction objects for planning
```

## Interfaces

### FastAPI Server (`api.py`)

```bash
uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload
```

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/chat` | Send a message, receive bot response + inline notifications |
| `GET` | `/api/itinerary` | Get current trip plan state |
| `POST` | `/api/weather` | Inject a weather change (`"rainy"` or `"sunny"`) |
| `WS` | `/api/ws` | Real-time push channel for weather alerts and replan events |

CORS is configured for `localhost:3000` (React/Next.js dev server).

### CLI (`cli.py`)

```bash
python -m app.cli
```

Commands: `plan my trip`, `show itinerary`, `inject rain`, `inject sunny`, `help`, `quit`.

## Architecture

```
         ┌──────────────┐    ┌──────────────┐
         │  FastAPI      │    │  CLI         │
         │  (api.py)     │    │  (cli.py)    │
         └──────┬───────┘    └──────┬───────┘
                │                   │
                └─────────┬─────────┘
                          ▼
                 ┌─────────────────┐
                 │  TripChatBot    │  ← BaseAgent (aegis framework)
                 │  (chat/bot.py)  │
                 └────────┬────────┘
                          │
           ┌──────────────┼──────────────┐
           ▼              ▼              ▼
    ┌─────────────┐ ┌──────────┐ ┌──────────────────┐
    │ LLM Intent  │ │ ChatState│ │ Activity Adapter  │
    │ Parsing     │ │ DayPlan  │ │ (services/)       │
    │ (prompts.py)│ │ (state)  │ │                   │
    └─────────────┘ └──────────┘ └────────┬─────────┘
                                          ▼
                              ┌───────────────────────┐
                              │ PlanningGraph          │
                              │ (aegis.state)          │
                              │ precondition/effect    │
                              │ based plan extraction  │
                              └───────────────────────┘
```

### Event Flow

The system uses an async `EventBus` for inter-agent communication:

1. **WeatherAgent** polls `MockWeatherService`, detects condition changes, publishes `WEATHER_CHANGED`
2. **TripChatBot** subscribes to `WEATHER_CHANGED` and `PLAN_UPDATED` events
3. On weather change, affected days are automatically replanned with indoor alternatives
4. Notifications are pushed to the `notification_queue` and broadcast to WebSocket clients

## Core Components

### TripChatBot (`chat/bot.py`)

The main agent. Handles user messages through an LLM-powered pipeline:

1. **Intent parsing** — LLM classifies the message into one of: `PLAN_TRIP`, `SHOW_ITINERARY`, `WEATHER_CHANGE`, `SWAP_ACTIVITY`, `LIST_ACTIVITIES`, `HELP`, `GENERAL`
2. **Dispatch** — Routes to the appropriate handler
3. **Planning** — Uses `PlanningGraph` (precondition/effect solver) to build valid itineraries
4. **Response formatting** — LLM formats the raw result into a natural language response
5. **Event publishing** — Publishes `PLAN_UPDATED` / `WEATHER_CHANGED` events to the bus

### Data Layer (`data/`)

- **`extract_data.py`** — Connects to MongoDB to fetch trip data, builds weather suitability maps and indoor alternative lookups dynamically
- **`sample_trip.py`** — Hardcoded 3-day Montreal/Quebec City trip used as fallback when MongoDB is unavailable

### Activity Adapter (`services/activity_adapter.py`)

Converts raw trip data into `ActivityAction` objects compatible with the planning graph. Handles slot-based scheduling, weather requirements, and indoor alternative substitution.

## State Models (`chat/state.py`)

| Model | Fields | Purpose |
|-------|--------|---------|
| `ChatState` | `conversation`, `trip_planned`, `day_plans`, `current_day_focus` | Session state |
| `DayPlan` | `day_number`, `city`, `actions`, `weather` | Single day's itinerary |
| `Message` | `role`, `content`, `timestamp` | Conversation turn |
| `Intent` | enum | Classified user intention |

## Configuration

| Env Variable | Default | Description |
|-------------|---------|-------------|
| `OPENAI_API_KEY` | — | Required if using OpenAI LLM provider |
| `ANTHROPIC_API_KEY` | — | Required if using Anthropic LLM provider |
| `LOG_LEVEL` | `INFO` | Logging verbosity |
| `LOG_FILE` | `aegis.log` | Log output file |

LLM provider and model are configured via `AEGISConfig` (see `aegis/config.py`).

## Dependencies

External packages: `fastapi`, `uvicorn`, `langchain-core`, `langchain-openai`, `langchain-anthropic`, `pymongo`, `pydantic`, `python-dotenv`

Framework packages (from `aegis/`): `BaseAgent`, `AgentConfig`, `EventBus`, `PlanningGraph`, `AEGISConfig`

Demo packages (from `demos/`): `WeatherAgent`, `MockWeatherService`, `ActivityAction`, `WeatherCondition`
