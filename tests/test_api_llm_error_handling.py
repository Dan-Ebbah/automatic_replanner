"""Tests for OpenAI error handling in the chat API."""

from pathlib import Path
import sys

import pytest
from fastapi import HTTPException
from httpx import Request, Response
from openai import RateLimitError
from unittest.mock import patch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from aegis.config import AEGISConfig
from app.api import ChatRequest, app_state, chat
from app.chat.bot import TripAgentLLMError, TripChatBot


class _FakeLLM:
    async def ainvoke(self, messages):
        raise self._error

    def __init__(self, error):
        self._error = error


class _FakeBot:
    def __init__(self, exc):
        self.exc = exc

    async def handle_message(self, user_input: str) -> str:
        raise self.exc


@pytest.mark.asyncio
async def test_chat_endpoint_passes_through_structured_llm_error(monkeypatch):
    error = TripAgentLLMError("OpenAI quota was exhausted while parsing intent.", status_code=429)
    monkeypatch.setattr(app_state, "bot", _FakeBot(error))

    with pytest.raises(HTTPException) as exc_info:
        await chat(ChatRequest(message="Plan my trip"))

    http_exc = exc_info.value
    assert getattr(http_exc, "status_code", None) == 429
    assert getattr(http_exc, "detail", "") == "OpenAI quota was exhausted while parsing intent."


@pytest.mark.asyncio
async def test_trip_chatbot_translates_openai_rate_limit_error():
    request = Request("POST", "https://api.openai.com/v1/chat/completions")
    response = Response(status_code=429, request=request, content=b'{"error":{"code":"insufficient_quota"}}')
    rate_limit_error = RateLimitError(
        "You exceeded your current quota.",
        response=response,
        body={"error": {"code": "insufficient_quota"}},
    )

    with patch("app.chat.bot.AEGIS") as mock_aegis, \
         patch.object(TripChatBot, "_load_sample_trip", autospec=True, return_value=None), \
         patch.object(TripChatBot, "_create_llm", return_value=_FakeLLM(rate_limit_error)):
        mock_aegis.return_value.declare_healing.return_value = None
        bot = TripChatBot(
            config=AEGISConfig(openai_api_key="test-key"),
        )

    with pytest.raises(TripAgentLLMError) as exc_info:
        await bot._ainvoke_llm([], purpose="parsing intent")

    assert exc_info.value.status_code == 429
    assert "quota" in exc_info.value.detail.lower()
    assert "billing/credits" in exc_info.value.detail


