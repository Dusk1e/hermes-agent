"""Regression tests for repo .env loading in Hermes Atropos environments."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_PATH = REPO_ROOT / "environments" / "hermes_base_env.py"


def _stub_module(name: str, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


def _load_module_with_stubs(monkeypatch, tmp_path, load_dotenv):
    repo_root = tmp_path / "repo"
    env_dir = repo_root / "environments"
    env_dir.mkdir(parents=True)

    module_path = env_dir / "hermes_base_env.py"
    module_path.write_text(SOURCE_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    (repo_root / ".env").write_bytes(b"OPENROUTER_API_KEY=test\x96value\n")

    class _BaseEnv:
        pass

    class _BaseEnvConfig:
        pass

    class _ScoredDataGroup:
        pass

    class _ScoredDataItem:
        pass

    class _APIServerConfig:
        pass

    class _ServerBaseline:
        pass

    class _ServerManager:
        pass

    class _AgentResult:
        pass

    class _HermesAgentLoop:
        pass

    class _ToolContext:
        pass

    class _Item:
        pass

    stub_modules = {
        "dotenv": _stub_module("dotenv", load_dotenv=load_dotenv),
        "pydantic": _stub_module("pydantic", Field=lambda *args, **kwargs: kwargs.get("default")),
        "environments": _stub_module("environments"),
        "environments.patches": _stub_module("environments.patches", apply_patches=lambda: None),
        "environments.agent_loop": _stub_module(
            "environments.agent_loop",
            AgentResult=_AgentResult,
            HermesAgentLoop=_HermesAgentLoop,
        ),
        "environments.tool_context": _stub_module(
            "environments.tool_context",
            ToolContext=_ToolContext,
        ),
        "atroposlib": _stub_module("atroposlib"),
        "atroposlib.envs": _stub_module("atroposlib.envs"),
        "atroposlib.envs.base": _stub_module(
            "atroposlib.envs.base",
            BaseEnv=_BaseEnv,
            BaseEnvConfig=_BaseEnvConfig,
            ScoredDataGroup=_ScoredDataGroup,
            ScoredDataItem=_ScoredDataItem,
        ),
        "atroposlib.envs.server_handling": _stub_module("atroposlib.envs.server_handling"),
        "atroposlib.envs.server_handling.server_manager": _stub_module(
            "atroposlib.envs.server_handling.server_manager",
            APIServerConfig=_APIServerConfig,
            ServerBaseline=_ServerBaseline,
            ServerManager=_ServerManager,
        ),
        "atroposlib.type_definitions": _stub_module(
            "atroposlib.type_definitions",
            Item=_Item,
        ),
        "tools": _stub_module("tools"),
        "tools.budget_config": _stub_module(
            "tools.budget_config",
            DEFAULT_RESULT_SIZE_CHARS=100_000,
            DEFAULT_TURN_BUDGET_CHARS=200_000,
            DEFAULT_PREVIEW_SIZE_CHARS=1_500,
        ),
        "model_tools": _stub_module("model_tools", get_tool_definitions=lambda *args, **kwargs: []),
        "toolset_distributions": _stub_module(
            "toolset_distributions",
            sample_toolsets_from_distribution=lambda *args, **kwargs: [],
        ),
    }

    stub_modules["environments"].patches = stub_modules["environments.patches"]
    stub_modules["environments"].agent_loop = stub_modules["environments.agent_loop"]
    stub_modules["environments"].tool_context = stub_modules["environments.tool_context"]
    stub_modules["atroposlib"].envs = stub_modules["atroposlib.envs"]
    stub_modules["atroposlib"].type_definitions = stub_modules["atroposlib.type_definitions"]
    stub_modules["atroposlib.envs"].base = stub_modules["atroposlib.envs.base"]
    stub_modules["atroposlib.envs"].server_handling = stub_modules["atroposlib.envs.server_handling"]
    stub_modules["atroposlib.envs.server_handling"].server_manager = stub_modules[
        "atroposlib.envs.server_handling.server_manager"
    ]
    stub_modules["tools"].budget_config = stub_modules["tools.budget_config"]

    for name, module in stub_modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    spec = importlib.util.spec_from_file_location("test_hermes_base_env_module", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_hermes_base_env_falls_back_to_latin1_for_project_dotenv(monkeypatch, tmp_path):
    calls = []

    def fake_load_dotenv(*, dotenv_path=None, encoding=None, **kwargs):
        calls.append(
            {
                "dotenv_path": Path(dotenv_path) if dotenv_path is not None else None,
                "encoding": encoding,
                "kwargs": kwargs,
            }
        )
        if encoding == "utf-8":
            raise UnicodeDecodeError("utf-8", b"\x96", 0, 1, "invalid start byte")
        return True

    _load_module_with_stubs(monkeypatch, tmp_path, fake_load_dotenv)

    assert [call["encoding"] for call in calls] == ["utf-8", "latin-1"]
    assert calls[0]["dotenv_path"] == calls[1]["dotenv_path"]
    assert calls[0]["dotenv_path"].name == ".env"
