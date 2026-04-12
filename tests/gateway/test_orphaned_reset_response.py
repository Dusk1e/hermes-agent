"""Regression tests for orphaned agent responses after /reset or /stop."""

import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.platforms.base import MessageEvent, MessageType
from gateway.session import SessionSource, SessionStore


def _make_event(text: str = "hello") -> MessageEvent:
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="12345",
        chat_type="dm",
        user_id="user-1",
    )
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=source,
        message_id="mid-1",
    )


def _make_runner(tmp_path):
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    adapter = MagicMock()
    adapter.send = AsyncMock()
    adapter.send_typing = AsyncMock()
    adapter.stop_typing = AsyncMock()
    adapter.get_pending_message = MagicMock(return_value=None)
    adapter.has_pending_interrupt = MagicMock(return_value=False)
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._pending_messages = {}
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._agent_cache = {}
    runner._agent_cache_lock = None
    runner._session_model_overrides = {}
    runner._pending_model_notes = {}
    runner._background_tasks = set()
    runner._voice_mode = {}
    runner._prefill_messages = []
    runner._ephemeral_system_prompt = ""
    runner._reasoning_config = None
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._session_db = None
    runner._show_reasoning = False
    runner._draining = False
    runner._busy_input_mode = "interrupt"
    runner._MAX_INTERRUPT_DEPTH = 3
    runner._async_flush_memories = AsyncMock()
    runner._deliver_media_from_response = AsyncMock()
    runner._send_voice_reply = AsyncMock()
    runner._inject_watch_notification = AsyncMock()
    runner._should_send_voice_reply = lambda *args, **kwargs: False
    runner._format_session_info = lambda: ""
    runner._set_session_env = lambda context: []
    runner._clear_session_env = lambda tokens: None
    runner._load_reasoning_config = lambda: None
    runner._load_service_tier = lambda: None
    runner._is_intentional_model_switch = lambda *args, **kwargs: False
    runner._evict_cached_agent = lambda *args, **kwargs: None
    runner._status_action_label = lambda: "restart"
    runner._status_action_gerund = lambda: "restarting"
    runner._queue_during_drain_enabled = lambda: False
    runner._update_runtime_status = lambda *args, **kwargs: None
    runner.hooks = SimpleNamespace(
        emit=AsyncMock(),
        loaded_hooks=False,
    )
    runner.config = GatewayConfig()
    runner.session_store = SessionStore(tmp_path / "sessions.json", runner.config)
    runner._handle_message_with_agent = GatewayRunner._handle_message_with_agent.__get__(runner, GatewayRunner)
    runner._discard_turn_result_reason = GatewayRunner._discard_turn_result_reason.__get__(runner, GatewayRunner)
    return runner


@pytest.mark.asyncio
async def test_handle_message_with_agent_discards_orphaned_result(tmp_path):
    runner = _make_runner(tmp_path)
    event = _make_event()
    source = event.source
    session_entry = runner.session_store.get_or_create_session(source)
    session_key = session_entry.session_key

    runner._prepare_inbound_message_text = AsyncMock(return_value="hello")
    runner._run_agent = AsyncMock(return_value={
        "discarded": True,
        "final_response": "",
        "messages": [],
        "api_calls": 1,
    })
    runner.session_store.append_to_transcript = MagicMock()
    runner.session_store.update_session = MagicMock()

    old_home = os.environ.get("TELEGRAM_HOME_CHANNEL")
    os.environ["TELEGRAM_HOME_CHANNEL"] = "set"
    try:
        result = await runner._handle_message_with_agent(event, source, session_key)
    finally:
        if old_home is None:
            os.environ.pop("TELEGRAM_HOME_CHANNEL", None)
        else:
            os.environ["TELEGRAM_HOME_CHANNEL"] = old_home

    assert result is None
    runner._run_agent.assert_awaited_once()
    runner.adapters[Platform.TELEGRAM].send.assert_not_awaited()
    runner.session_store.append_to_transcript.assert_not_called()
    runner.session_store.update_session.assert_not_called()


def test_discard_turn_result_reason_detects_reset_and_suspension(tmp_path):
    runner = _make_runner(tmp_path)
    source = _make_event().source
    entry = runner.session_store.get_or_create_session(source)
    session_key = entry.session_key
    old_session_id = entry.session_id

    assert runner._discard_turn_result_reason(session_key, old_session_id) is None

    runner.session_store.reset_session(session_key)
    assert runner._discard_turn_result_reason(session_key, old_session_id) == "reset"

    refreshed = runner.session_store.get_session_entry(session_key)
    runner.session_store.suspend_session(session_key)
    assert runner._discard_turn_result_reason(session_key, refreshed.session_id) == "suspended"
