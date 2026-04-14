import logging
from types import SimpleNamespace

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.run import GatewayRunner


def _runner_with_api_server(port=8642):
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={
            Platform.API_SERVER: PlatformConfig(
                enabled=True,
                extra={"port": port},
            )
        }
    )
    runner._tailscale_serve_enabled = True
    runner._tailscale_serve_active = False
    runner._tailscale_serve_port = None
    runner._tailscale_serve_url = None
    return runner


def test_start_tailscale_serve_requires_api_server():
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig()
    runner._tailscale_serve_enabled = True
    runner._tailscale_serve_active = False
    runner._tailscale_serve_port = None
    runner._tailscale_serve_url = None

    ok, reason = GatewayRunner._start_tailscale_serve(runner)

    assert ok is False
    assert "API server" in reason


def test_start_tailscale_serve_configures_https_proxy(monkeypatch, caplog):
    runner = _runner_with_api_server(port=8642)
    monkeypatch.delenv("API_SERVER_KEY", raising=False)
    monkeypatch.setattr("gateway.run.shutil.which", lambda name: "/usr/bin/tailscale")

    calls = []

    def fake_run(cmd, capture_output=True, text=True, timeout=20.0):
        calls.append(cmd)
        if cmd[1:] == ["status", "--json"]:
            return SimpleNamespace(
                returncode=0,
                stdout='{"Self":{"DNSName":"hermes.tailnet.ts.net."}}',
                stderr="",
            )
        if cmd[1:] == ["serve", "--bg", "--https=8642", "http://127.0.0.1:8642"]:
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        raise AssertionError(f"Unexpected command: {cmd}")

    monkeypatch.setattr("gateway.run.subprocess.run", fake_run)

    with caplog.at_level(logging.WARNING):
        ok, reason = GatewayRunner._start_tailscale_serve(runner)

    assert ok is True
    assert reason is None
    assert runner._tailscale_serve_active is True
    assert runner._tailscale_serve_port == 8642
    assert runner._tailscale_serve_url == "https://hermes.tailnet.ts.net:8642"
    assert calls == [
        ["/usr/bin/tailscale", "status", "--json"],
        ["/usr/bin/tailscale", "serve", "--bg", "--https=8642", "http://127.0.0.1:8642"],
    ]
    assert "without API_SERVER_KEY" in caplog.text


def test_stop_tailscale_serve_turns_off_same_https_port(monkeypatch):
    runner = _runner_with_api_server(port=8642)
    runner._tailscale_serve_active = True
    runner._tailscale_serve_port = 8642
    runner._tailscale_serve_url = "https://hermes.tailnet.ts.net:8642"

    monkeypatch.setattr("gateway.run.shutil.which", lambda name: "/usr/bin/tailscale")
    calls = []

    def fake_run(cmd, capture_output=True, text=True, timeout=20.0):
        calls.append(cmd)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("gateway.run.subprocess.run", fake_run)

    GatewayRunner._stop_tailscale_serve(runner)

    assert calls == [
        ["/usr/bin/tailscale", "serve", "--https=8642", "off"],
    ]
    assert runner._tailscale_serve_active is False
    assert runner._tailscale_serve_port is None
    assert runner._tailscale_serve_url is None
