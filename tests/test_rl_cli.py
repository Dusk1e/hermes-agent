import importlib
import sys
import types


def test_rl_cli_import_bootstraps_hermes_home_without_nameerror(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    home.mkdir()

    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = object

    fake_rl_tool = types.ModuleType("tools.rl_training_tool")
    fake_rl_tool.get_missing_keys = lambda: []

    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)
    monkeypatch.setitem(sys.modules, "tools.rl_training_tool", fake_rl_tool)

    sys.modules.pop("rl_cli", None)
    module = importlib.import_module("rl_cli")

    assert module._hermes_home == home
