"""Parity tests for per-provider max_output_tokens across CLI surfaces.

Companion to ``tests/gateway/test_max_tokens_propagation.py``. The gateway
already honored a custom provider's ``max_output_tokens`` cap (#20741); these
cover the shared ``resolve_effective_max_tokens`` resolver and the ``hermes -q``
/ oneshot path so the same config caps output identically on every surface.

Precedence (highest first):
    HERMES_MAX_TOKENS env  >  model.max_tokens  >
    per-provider max_output_tokens  >  None
"""

import importlib
import os
import textwrap
from pathlib import Path

import pytest


def _runtime_provider():
    return importlib.import_module("hermes_cli.runtime_provider")


def test_resolve_effective_max_tokens_precedence(monkeypatch):
    """Provider cap fills in only when no global cap is configured."""
    rp = _runtime_provider()
    monkeypatch.delenv("HERMES_MAX_TOKENS", raising=False)

    # Per-provider cap applies when no global model.max_tokens is set.
    assert rp.resolve_effective_max_tokens({"max_output_tokens": 12000}, model_cfg={}) == 12000
    # Documented global model.max_tokens wins over the per-provider cap.
    assert (
        rp.resolve_effective_max_tokens(
            {"max_output_tokens": 12000}, model_cfg={"max_tokens": 16384}
        )
        == 16384
    )
    # Nothing configured anywhere -> None (no spurious limit).
    assert rp.resolve_effective_max_tokens({}, model_cfg={}) is None
    assert rp.resolve_effective_max_tokens(None, model_cfg={}) is None
    # Non-positive provider cap is ignored.
    assert rp.resolve_effective_max_tokens({"max_output_tokens": 0}, model_cfg={}) is None
    # A bool global is garbage and is rejected, falling through to the cap.
    assert (
        rp.resolve_effective_max_tokens(
            {"max_output_tokens": 12000}, model_cfg={"max_tokens": True}
        )
        == 12000
    )


def test_resolve_effective_max_tokens_env_override(monkeypatch):
    """HERMES_MAX_TOKENS is the highest-priority override."""
    rp = _runtime_provider()
    monkeypatch.setenv("HERMES_MAX_TOKENS", "2048")
    assert (
        rp.resolve_effective_max_tokens(
            {"max_output_tokens": 12000}, model_cfg={"max_tokens": 16384}
        )
        == 2048
    )
    # An unparseable env value is ignored and resolution falls through.
    monkeypatch.setenv("HERMES_MAX_TOKENS", "not-an-int")
    assert rp.resolve_effective_max_tokens({"max_output_tokens": 12000}, model_cfg={}) == 12000


def test_oneshot_forwards_provider_max_output_tokens(monkeypatch):
    """oneshot ``_run_agent`` caps ``AIAgent.max_tokens`` with a custom
    provider's ``max_output_tokens`` when no global cap is set — the bug was
    that oneshot dropped max_tokens entirely while the gateway honored it."""
    home = Path(os.environ["HERMES_HOME"])
    (home / "config.yaml").write_text(
        textwrap.dedent(
            """
            model:
              default: glm-5.1
              provider: mylocal
            providers:
              mylocal:
                api: http://localhost:11434/v1
                api_key: sk-test
                default_model: glm-5.1
                max_output_tokens: 12000
            """
        )
    )
    monkeypatch.delenv("HERMES_MAX_TOKENS", raising=False)

    captured: dict = {}

    class _FakeAgent:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def chat(self, prompt):
            return "ok"

    import run_agent

    monkeypatch.setattr(run_agent, "AIAgent", _FakeAgent)

    from hermes_cli import oneshot

    result = oneshot._run_agent("hi", provider="mylocal", use_config_toolsets=False)

    assert result == "ok"
    assert captured["max_tokens"] == 12000
