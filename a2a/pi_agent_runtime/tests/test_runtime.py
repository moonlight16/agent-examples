import json
from pathlib import Path

import pytest
from pi_agent_runtime.runtime import PiConfig, PiRunner


def config(tmp_path: Path, executable: str) -> PiConfig:
    return PiConfig(
        workspace=tmp_path / "workspace",
        agent_dir=tmp_path / "agent",
        executable=executable,
        provider="test-provider",
        model="test-model",
        base_url="http://model.example/v1",
        api="openai-completions",
        api_key="test-key",
    )


def test_writes_pi_model_config(tmp_path: Path) -> None:
    value = config(tmp_path, "pi")
    value.write_model_config()

    document = json.loads((value.agent_dir / "models.json").read_text())
    provider = document["providers"]["test-provider"]
    assert provider["baseUrl"] == "http://model.example/v1"
    assert provider["models"] == [{"id": "test-model"}]


@pytest.mark.asyncio
async def test_runs_pi_in_workspace(tmp_path: Path) -> None:
    fake_pi = tmp_path / "pi"
    fake_pi.write_text("#!/bin/sh\nprintf 'response from pi'\n", encoding="utf-8")
    fake_pi.chmod(0o755)

    result = await PiRunner(config(tmp_path, str(fake_pi))).run("hello")

    assert result == "response from pi"
    assert (tmp_path / "workspace").is_dir()


@pytest.mark.asyncio
async def test_rejects_empty_prompt(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        await PiRunner(config(tmp_path, "pi")).run("  ")
