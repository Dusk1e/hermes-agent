"""Shared constants for Hermes Agent.

Import-safe module with no dependencies — can be imported from anywhere
without risk of circular imports.
"""

import os
from pathlib import Path


def get_hermes_home() -> Path:
    """Return the Hermes home directory (default: ~/.hermes).

    Reads HERMES_HOME env var, falls back to ~/.hermes.
    This is the single source of truth — all other copies should import this.
    """
    return Path(os.getenv("HERMES_HOME", Path.home() / ".hermes"))


def get_optional_skills_dir(default: Path | None = None) -> Path:
    """Return the optional-skills directory, honoring package-manager wrappers.

    Packaged installs may ship ``optional-skills`` outside the Python package
    tree and expose it via ``HERMES_OPTIONAL_SKILLS``.
    """
    override = os.getenv("HERMES_OPTIONAL_SKILLS", "").strip()
    if override:
        return Path(override)
    if default is not None:
        return default
    return get_hermes_home() / "optional-skills"


def get_hermes_dir(new_subpath: str, old_name: str) -> Path:
    """Resolve a Hermes subdirectory with backward compatibility.

    New installs get the consolidated layout (e.g. ``cache/images``).
    Existing installs that already have the old path (e.g. ``image_cache``)
    keep using it — no migration required.

    Args:
        new_subpath: Preferred path relative to HERMES_HOME (e.g. ``"cache/images"``).
        old_name: Legacy path relative to HERMES_HOME (e.g. ``"image_cache"``).

    Returns:
        Absolute ``Path`` — old location if it exists on disk, otherwise the new one.
    """
    home = get_hermes_home()
    old_path = home / old_name
    if old_path.exists():
        return old_path
    return home / new_subpath


def display_hermes_home() -> str:
    """Return a user-friendly display string for the current HERMES_HOME.

    Uses ``~/`` shorthand for readability::

        default:  ``~/.hermes``
        profile:  ``~/.hermes/profiles/coder``
        custom:   ``/opt/hermes-custom``

    Use this in **user-facing** print/log messages instead of hardcoding
    ``~/.hermes``.  For code that needs a real ``Path``, use
    :func:`get_hermes_home` instead.
    """
    home = get_hermes_home()
    try:
        return "~/" + str(home.relative_to(Path.home()))
    except ValueError:
        return str(home)


VALID_REASONING_EFFORTS = ("xhigh", "high", "medium", "low", "minimal")


def parse_reasoning_effort(effort: str) -> dict | None:
    """Parse a reasoning effort level into a config dict.

    Valid levels: "xhigh", "high", "medium", "low", "minimal", "none".
    Returns None when the input is empty or unrecognized (caller uses default).
    Returns {"enabled": False} for "none".
    Returns {"enabled": True, "effort": <level>} for valid effort levels.
    """
    if not effort or not effort.strip():
        return None
    effort = effort.strip().lower()
    if effort == "none":
        return {"enabled": False}
    if effort in VALID_REASONING_EFFORTS:
        return {"enabled": True, "effort": effort}
    return None


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_MODELS_URL = f"{OPENROUTER_BASE_URL}/models"
OPENROUTER_CHAT_URL = f"{OPENROUTER_BASE_URL}/chat/completions"

AI_GATEWAY_BASE_URL = "https://ai-gateway.vercel.sh/v1"
AI_GATEWAY_MODELS_URL = f"{AI_GATEWAY_BASE_URL}/models"
AI_GATEWAY_CHAT_URL = f"{AI_GATEWAY_BASE_URL}/chat/completions"

NOUS_API_BASE_URL = "https://inference-api.nousresearch.com/v1"
NOUS_API_CHAT_URL = f"{NOUS_API_BASE_URL}/chat/completions"

import contextvars

# Thread-safe context variables for session state
_SESSION_PLATFORM = contextvars.ContextVar("HERMES_SESSION_PLATFORM", default="")
_SESSION_CHAT_ID = contextvars.ContextVar("HERMES_SESSION_CHAT_ID", default="")
_SESSION_CHAT_NAME = contextvars.ContextVar("HERMES_SESSION_CHAT_NAME", default="")
_SESSION_THREAD_ID = contextvars.ContextVar("HERMES_SESSION_THREAD_ID", default="")
_SESSION_KEY = contextvars.ContextVar("HERMES_SESSION_KEY", default="")
_YOLO_MODE = contextvars.ContextVar("HERMES_YOLO_MODE", default="")
_CRON_AUTO_DELIVER_PLATFORM = contextvars.ContextVar("HERMES_CRON_AUTO_DELIVER_PLATFORM", default="")
_CRON_AUTO_DELIVER_CHAT_ID = contextvars.ContextVar("HERMES_CRON_AUTO_DELIVER_CHAT_ID", default="")
_CRON_AUTO_DELIVER_THREAD_ID = contextvars.ContextVar("HERMES_CRON_AUTO_DELIVER_THREAD_ID", default="")

_CONTEXT_VAR_MAP = {
    "HERMES_SESSION_PLATFORM": _SESSION_PLATFORM,
    "HERMES_SESSION_CHAT_ID": _SESSION_CHAT_ID,
    "HERMES_SESSION_CHAT_NAME": _SESSION_CHAT_NAME,
    "HERMES_SESSION_THREAD_ID": _SESSION_THREAD_ID,
    "HERMES_SESSION_KEY": _SESSION_KEY,
    "HERMES_YOLO_MODE": _YOLO_MODE,
    "HERMES_CRON_AUTO_DELIVER_PLATFORM": _CRON_AUTO_DELIVER_PLATFORM,
    "HERMES_CRON_AUTO_DELIVER_CHAT_ID": _CRON_AUTO_DELIVER_CHAT_ID,
    "HERMES_CRON_AUTO_DELIVER_THREAD_ID": _CRON_AUTO_DELIVER_THREAD_ID,
}

def get_session_env(key: str, default=None):
    """Get a session env var from contextvars, falling back to os.environ for CLI/tests."""
    if key in _CONTEXT_VAR_MAP:
        val = _CONTEXT_VAR_MAP[key].get()
        if val:
            return val
    return os.environ.get(key, default)

def set_session_env(key: str, value: str):
    """Set a session env var in contextvars only (does not pollute global os.environ)."""
    if key in _CONTEXT_VAR_MAP:
        if value is None:
            _CONTEXT_VAR_MAP[key].set("")
        else:
            _CONTEXT_VAR_MAP[key].set(str(value))
    else:
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = str(value)
            
def clear_session_env():
    """Clear all session contextvars."""
    for cvar in _CONTEXT_VAR_MAP.values():
        cvar.set("")
