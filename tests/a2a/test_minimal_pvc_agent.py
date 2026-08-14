"""Tests for minimal_pvc_agent configuration (isolated from heavy deps).

The minimal_pvc_agent package pulls in heavy deps at import time
(langgraph, a2a, sqlalchemy, ...). We bypass that by pre-registering a
fake ``minimal_pvc_agent`` package and loading configuration.py
directly by file path (same approach as test_weather_secret_redaction.py).
"""

import importlib.util
import pathlib
import sys
from types import ModuleType

_fake_pkg = ModuleType("minimal_pvc_agent")
_fake_pkg.__path__ = []  # type: ignore[attr-defined]
sys.modules.setdefault("minimal_pvc_agent", _fake_pkg)

_BASE = (
    pathlib.Path(__file__).parent.parent.parent
    / "a2a"
    / "minimal_pvc_agent"
    / "src"
    / "minimal_pvc_agent"
)


def _load_module(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules[name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


_config_mod = _load_module("minimal_pvc_agent.configuration", _BASE / "configuration.py")
Configuration = _config_mod.Configuration


class TestConfiguration:
    def test_defaults(self):
        config = Configuration()
        assert config.llm_model == "qwen3:4b"
        assert config.llm_api_base == "http://host.docker.internal:11434/v1"
        assert config.llm_api_key == "dummy"
        assert config.checkpoint_path == "/shared/checkpoints.db"
        assert config.task_store_path == "/shared/tasks.db"

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("LLM_MODEL", "gpt-4o")
        monkeypatch.setenv("LLM_API_BASE", "https://api.openai.com/v1")
        monkeypatch.setenv("CHECKPOINT_PATH", "/tmp/lg.db")
        monkeypatch.setenv("TASK_STORE_PATH", "/tmp/tasks.db")
        config = Configuration()
        assert config.llm_model == "gpt-4o"
        assert config.llm_api_base == "https://api.openai.com/v1"
        assert config.checkpoint_path == "/tmp/lg.db"
        assert config.task_store_path == "/tmp/tasks.db"

    def test_empty_paths_allowed_for_memory_mode(self, monkeypatch):
        monkeypatch.setenv("CHECKPOINT_PATH", "")
        monkeypatch.setenv("TASK_STORE_PATH", "")
        config = Configuration()
        assert config.checkpoint_path == ""
        assert config.task_store_path == ""

    def test_has_valid_api_key_local_ollama_with_dummy(self, monkeypatch):
        monkeypatch.setenv("LLM_API_BASE", "http://localhost:11434/v1")
        monkeypatch.setenv("LLM_API_KEY", "dummy")
        assert Configuration().has_valid_api_key is True

    def test_has_valid_api_key_localhost_127(self, monkeypatch):
        monkeypatch.setenv("LLM_API_BASE", "http://127.0.0.1:8080/v1")
        monkeypatch.setenv("LLM_API_KEY", "dummy")
        assert Configuration().has_valid_api_key is True

    def test_has_valid_api_key_remote_dummy_invalid(self, monkeypatch):
        monkeypatch.setenv("LLM_API_BASE", "https://api.openai.com/v1")
        monkeypatch.setenv("LLM_API_KEY", "dummy")
        assert Configuration().has_valid_api_key is False

    def test_has_valid_api_key_remote_empty_invalid(self, monkeypatch):
        monkeypatch.setenv("LLM_API_BASE", "https://api.openai.com/v1")
        monkeypatch.setenv("LLM_API_KEY", "")
        assert Configuration().has_valid_api_key is False

    def test_has_valid_api_key_remote_placeholder_invalid(self, monkeypatch):
        monkeypatch.setenv("LLM_API_BASE", "https://api.openai.com/v1")
        for key in ["changeme", "your-api-key-here"]:
            monkeypatch.setenv("LLM_API_KEY", key)
            assert Configuration().has_valid_api_key is False, f"'{key}' should be invalid"

    def test_has_valid_api_key_remote_real_valid(self, monkeypatch):
        monkeypatch.setenv("LLM_API_BASE", "https://api.openai.com/v1")
        monkeypatch.setenv("LLM_API_KEY", "sk-real-key-123")
        assert Configuration().has_valid_api_key is True
