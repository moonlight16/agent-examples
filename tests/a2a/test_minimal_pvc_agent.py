"""Tests for minimal_pvc_agent — configuration and persistence (isolated from heavy deps).

The minimal_pvc_agent package may pull in heavy deps at import time (langgraph,
a2a, opentelemetry, etc.). We bypass this by pre-registering a fake
``minimal_pvc_agent`` package and loading each target module directly by file
path via ``importlib`` (same approach as test_weather_secret_redaction.py).
"""

import importlib.util
import json
import logging
import pathlib
import sys
from types import ModuleType
from unittest.mock import MagicMock

# --- Isolation setup (must happen before any minimal_pvc_agent imports) ---
_fake_pkg = ModuleType("minimal_pvc_agent")
_fake_pkg.__path__ = []  # type: ignore[attr-defined]
sys.modules.setdefault("minimal_pvc_agent", _fake_pkg)
sys.modules.setdefault("minimal_pvc_agent.observability", MagicMock())

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
_persistence_mod = _load_module("minimal_pvc_agent.persistence", _BASE / "persistence.py")

Configuration = _config_mod.Configuration
sanitize_context_id = _persistence_mod.sanitize_context_id
context_file = _persistence_mod.context_file
append_turn = _persistence_mod.append_turn
read_history = _persistence_mod.read_history


# --- Tests ---


class TestConfiguration:
    """Test minimal_pvc_agent configuration defaults, env overrides, and key validation."""

    def test_defaults(self):
        config = Configuration()
        assert config.llm_model == "qwen3:4b"
        assert config.llm_api_base == "http://host.docker.internal:11434/v1"
        assert config.llm_api_key == "dummy"
        assert config.context_dir == "/shared/minimal-pvc-agent"
        assert config.task_store_db_url == ""
        assert config.checkpoint_db_url == ""

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("LLM_MODEL", "gpt-4o")
        monkeypatch.setenv("LLM_API_BASE", "https://api.openai.com/v1")
        monkeypatch.setenv("CONTEXT_DIR", "/tmp/custom-ctx")
        monkeypatch.setenv("CHECKPOINT_DB_URL", "postgresql://u:p@h/db")
        monkeypatch.setenv("TASK_STORE_DB_URL", "postgresql://u:p@h/tasks")
        config = Configuration()
        assert config.llm_model == "gpt-4o"
        assert config.llm_api_base == "https://api.openai.com/v1"
        assert config.context_dir == "/tmp/custom-ctx"
        assert config.checkpoint_db_url == "postgresql://u:p@h/db"
        assert config.task_store_db_url == "postgresql://u:p@h/tasks"

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


class TestSanitizeContextId:
    """Test sanitize_context_id and its use by context_file."""

    def test_uuid_preserved(self):
        uid = "550e8400-e29b-41d4-a716-446655440000"
        assert sanitize_context_id(uid) == uid

    def test_path_separators_replaced(self):
        result = sanitize_context_id("foo/bar\\baz")
        assert "/" not in result
        assert "\\" not in result
        assert result == "foo_bar_baz"

    def test_dotdot_traversal_neutralized(self):
        result = sanitize_context_id("../etc/passwd")
        assert ".." not in result
        assert "/" not in result

    def test_empty_returns_unknown(self):
        assert sanitize_context_id(None) == "unknown"
        assert sanitize_context_id("") == "unknown"

    def test_only_special_returns_unknown(self):
        assert sanitize_context_id("///") == "unknown"

    def test_length_capped(self):
        result = sanitize_context_id("a" * 5000)
        assert len(result) <= 128

    def test_context_file_uses_sanitized_stem(self, tmp_path):
        result = context_file(tmp_path, "foo/bar")
        assert result == tmp_path / "foo_bar.jsonl"


class TestAppendTurnAndReadHistory:
    """Test append_turn and read_history persistence functions."""

    def test_appends_single_turn(self, tmp_path):
        append_turn(tmp_path, "ctx-1", "task-1", "hello", "hi there")
        path = tmp_path / "ctx-1.jsonl"
        assert path.exists()
        lines = path.read_text().splitlines()
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["context_id"] == "ctx-1"
        assert record["task_id"] == "task-1"
        assert record["user_input"] == "hello"
        assert record["agent_reply"] == "hi there"
        assert "ts" in record

    def test_appends_multiple_turns_same_context(self, tmp_path):
        append_turn(tmp_path, "ctx-1", "task-1", "first", "reply1")
        append_turn(tmp_path, "ctx-1", "task-2", "second", "reply2")
        path = tmp_path / "ctx-1.jsonl"
        lines = path.read_text().splitlines()
        assert len(lines) == 2
        rec1 = json.loads(lines[0])
        rec2 = json.loads(lines[1])
        assert rec1["user_input"] == "first"
        assert rec2["user_input"] == "second"

    def test_separate_contexts_separate_files(self, tmp_path):
        append_turn(tmp_path, "ctx-a", "t1", "u1", "r1")
        append_turn(tmp_path, "ctx-b", "t2", "u2", "r2")
        assert (tmp_path / "ctx-a.jsonl").exists()
        assert (tmp_path / "ctx-b.jsonl").exists()
        assert len((tmp_path / "ctx-a.jsonl").read_text().splitlines()) == 1
        assert len((tmp_path / "ctx-b.jsonl").read_text().splitlines()) == 1

    def test_sanitized_filename(self, tmp_path):
        append_turn(tmp_path, "foo/bar", "t1", "u1", "r1")
        assert (tmp_path / "foo_bar.jsonl").exists()

    def test_creates_missing_root_directory(self, tmp_path):
        nested = tmp_path / "nested" / "dir"
        assert not nested.exists()
        append_turn(nested, "ctx-1", "t1", "u1", "r1")
        assert (nested / "ctx-1.jsonl").exists()

    def test_io_failure_swallowed_returns_none(self, tmp_path, caplog):
        # Create a file where a directory is expected — mkdir will fail
        blocker = tmp_path / "blocker"
        blocker.write_text("not a directory")
        bad_root = blocker / "subdir"
        with caplog.at_level(logging.WARNING):
            result = append_turn(bad_root, "ctx-1", "t1", "u1", "r1")
        assert result is None

    def test_read_history_empty_when_no_file(self, tmp_path):
        result = read_history(tmp_path, "nonexistent-ctx")
        assert result == []

    def test_read_history_skips_malformed_lines(self, tmp_path):
        path = tmp_path / "ctx-1.jsonl"
        good1 = json.dumps(
            {
                "ts": "2026-01-01T00:00:00Z",
                "context_id": "ctx-1",
                "task_id": "t1",
                "user_input": "u1",
                "agent_reply": "r1",
            }
        )
        good2 = json.dumps(
            {
                "ts": "2026-01-02T00:00:00Z",
                "context_id": "ctx-1",
                "task_id": "t2",
                "user_input": "u2",
                "agent_reply": "r2",
            }
        )
        path.write_text(good1 + "\n" + "this is not json {{{" + "\n" + good2 + "\n")
        records = read_history(tmp_path, "ctx-1")
        assert len(records) == 2
        assert records[0]["user_input"] == "u1"
        assert records[1]["user_input"] == "u2"

    def test_read_history_returns_chronological_order(self, tmp_path):
        append_turn(tmp_path, "ctx-1", "t1", "first", "r1")
        append_turn(tmp_path, "ctx-1", "t2", "second", "r2")
        append_turn(tmp_path, "ctx-1", "t3", "third", "r3")
        records = read_history(tmp_path, "ctx-1")
        assert len(records) == 3
        assert [r["user_input"] for r in records] == ["first", "second", "third"]
