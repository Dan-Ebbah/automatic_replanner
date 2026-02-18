"""
AEGIS Trip Planner — FastAPI + WebSocket API
=============================================
Connects the multi-agent chatbot to a React/Next.js frontend.

Run:
    uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload
"""

import asyncio
import logging
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

load_dotenv()

# ---------------------------------------------------------------------------
# Logging (mirrors cli.py)
# ---------------------------------------------------------------------------
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
LOG_FILE = os.environ.get("LOG_FILE", "aegis.log")

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.FileHandler(LOG_FILE, mode="a")],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Domain imports
# ---------------------------------------------------------------------------
from aegis.config import AEGISConfig  # noqa: E402
from aegis.agent import AgentConfig  # noqa: E402
from aegis.events import EventBus  # noqa: E402
from demos.activities import WeatherCondition  # noqa: E402
from demos.weather_service import MockWeatherService  # noqa: E402
from demos.agents.weather_agent import WeatherAgent  # noqa: E402
from app.chat import TripChatBot  # noqa: E402

# ---------------------------------------------------------------------------
# Pydantic request / response models
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)


class ChatResponse(BaseModel):
    response: str
    notifications: list[str] = []


class WeatherInjectRequest(BaseModel):
    condition: str = Field(..., pattern=r"^(rainy|sunny|rain|sun)$")


class WeatherInjectResponse(BaseModel):
    ok: bool
    injected: str


class DayPlanResponse(BaseModel):
    day_number: int
    city: str
    weather: str
    actions: list[dict]


class ItineraryResponse(BaseModel):
    trip_planned: bool
    days: list[DayPlanResponse] = []


# ---------------------------------------------------------------------------
# App state container
# ---------------------------------------------------------------------------

class AppState:
    """Holds shared singletons populated during lifespan."""

    def __init__(self) -> None:
        self.bot: TripChatBot | None = None
        self.weather_agent: WeatherAgent | None = None
        self.event_bus: EventBus | None = None
        self.weather_service: MockWeatherService | None = None


app_state = AppState()

POLL_INTERVAL = 2.0  # seconds between WeatherAgent polls


# ---------------------------------------------------------------------------
# WebSocket connection manager
# ---------------------------------------------------------------------------

class ConnectionManager:
    def __init__(self) -> None:
        self._active: set[WebSocket] = set()

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self._active.add(ws)
        logger.info("WebSocket client connected (%d total)", len(self._active))

    def disconnect(self, ws: WebSocket) -> None:
        self._active.discard(ws)
        logger.info("WebSocket client disconnected (%d remaining)", len(self._active))

    async def broadcast(self, message: str) -> None:
        dead: list[WebSocket] = []
        for ws in self._active:
            try:
                await ws.send_text(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self._active.discard(ws)

    @property
    def count(self) -> int:
        return len(self._active)


manager = ConnectionManager()


# ---------------------------------------------------------------------------
# Queue-to-WebSocket bridge
# ---------------------------------------------------------------------------

async def _drain_queue_to_ws() -> None:
    """Long-lived task: awaits notifications from the bot and broadcasts to WS clients."""
    while True:
        notification = await app_state.bot.notification_queue.get()
        logger.info("Notification from queue: %s", notification)
        if manager.count > 0:
            await manager.broadcast(notification)
        app_state.bot.notification_queue.task_done()


# ---------------------------------------------------------------------------
# Lifespan (startup / shutdown)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    # -- startup --------------------------------------------------------------
    event_bus = EventBus()
    weather_service = MockWeatherService()
    aegis_config = AEGISConfig()

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

    await bot.start()
    await weather_agent.start()

    app_state.bot = bot
    app_state.weather_agent = weather_agent
    app_state.event_bus = event_bus
    app_state.weather_service = weather_service

    bridge_task = asyncio.create_task(_drain_queue_to_ws())
    logger.info("AEGIS API started — log_level=%s, llm=%s", LOG_LEVEL, aegis_config.llm_provider)

    yield

    # -- shutdown -------------------------------------------------------------
    bridge_task.cancel()
    try:
        await bridge_task
    except asyncio.CancelledError:
        pass
    await bot.stop()
    await weather_agent.stop()
    logger.info("AEGIS API shut down")


# ---------------------------------------------------------------------------
# FastAPI app + CORS
# ---------------------------------------------------------------------------

app = FastAPI(title="AEGIS Trip Planner API", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """Send a message to the trip-planning bot."""
    try:
        response = await app_state.bot.handle_message(req.message)
    except Exception as exc:
        logger.exception("Error handling chat message")
        raise HTTPException(status_code=500, detail=str(exc))

    # Drain any inline notifications generated during handling
    notifications: list[str] = []
    while not app_state.bot.notification_queue.empty():
        try:
            notifications.append(app_state.bot.notification_queue.get_nowait())
            app_state.bot.notification_queue.task_done()
        except asyncio.QueueEmpty:
            break

    return ChatResponse(response=response, notifications=notifications)


@app.websocket("/api/ws")
async def websocket_endpoint(ws: WebSocket):
    """Real-time notification push channel."""
    await manager.connect(ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(ws)


@app.post("/api/weather", response_model=WeatherInjectResponse)
async def inject_weather(req: WeatherInjectRequest):
    """Inject a weather change (for demo / testing)."""
    cond_map = {
        "rain": WeatherCondition.RAINY,
        "rainy": WeatherCondition.RAINY,
        "sunny": WeatherCondition.SUNNY,
        "sun": WeatherCondition.SUNNY,
    }
    condition = cond_map.get(req.condition)
    if condition is None:
        raise HTTPException(status_code=400, detail=f"Unknown condition: {req.condition}")

    app_state.weather_agent.inject_weather_change(condition)
    logger.info("Injected weather change: %s", condition.value)
    return WeatherInjectResponse(ok=True, injected=condition.value)


@app.get("/api/itinerary", response_model=ItineraryResponse)
async def get_itinerary():
    """Return the current trip plan state."""
    bot = app_state.bot
    if not bot.state.trip_planned:
        return ItineraryResponse(trip_planned=False)

    days: list[DayPlanResponse] = []
    for day_idx in sorted(bot.state.day_plans):
        plan = bot.state.day_plans[day_idx]
        actions = [
            {
                "name": a.name,
                "duration_minutes": a.duration_minutes,
                "weather_requirement": a.weather_requirement.value,
            }
            for a in plan.actions
        ]
        days.append(
            DayPlanResponse(
                day_number=plan.day_number,
                city=plan.city,
                weather=plan.weather.value,
                actions=actions,
            )
        )

    return ItineraryResponse(trip_planned=True, days=days)
