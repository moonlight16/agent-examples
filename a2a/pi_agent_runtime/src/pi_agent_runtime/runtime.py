"""Run one isolated Pi turn against a configurable model endpoint."""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PiConfig:
    workspace: Path
    agent_dir: Path
    executable: str
    provider: str
    model: str
    base_url: str
    api: str
    api_key: str

    @classmethod
    def from_env(cls) -> "PiConfig":
        return cls(
            workspace=Path(os.getenv("PI_WORKSPACE", "/shared")),
            agent_dir=Path(os.getenv("PI_AGENT_DIR", "/home/node/.pi/agent")),
            executable=os.getenv("PI_EXECUTABLE", "pi"),
            provider=os.getenv("PI_MODEL_PROVIDER", "llmd"),
            model=os.getenv("PI_MODEL", "meta-llama/Llama-3.3-70B-Instruct"),
            base_url=os.getenv("PI_MODEL_BASE_URL", ""),
            api=os.getenv("PI_MODEL_API", "anthropic-messages"),
            api_key=os.getenv("PI_API_KEY", "unused"),
        )

    def validate(self) -> None:
        if not self.base_url:
            raise ValueError("PI_MODEL_BASE_URL is required")

    def write_model_config(self) -> None:
        self.validate()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.agent_dir.mkdir(parents=True, exist_ok=True)
        model_config = {
            "providers": {
                self.provider: {
                    "baseUrl": self.base_url.rstrip("/"),
                    "api": self.api,
                    "apiKey": self.api_key,
                    "models": [{"id": self.model}],
                }
            }
        }
        (self.agent_dir / "models.json").write_text(json.dumps(model_config, indent=2) + "\n", encoding="utf-8")


class PiRunner:
    """Serialize Pi turns that share one writable workspace."""

    def __init__(self, config: PiConfig):
        self.config = config
        self._lock = asyncio.Lock()

    async def run(self, prompt: str) -> str:
        if not prompt.strip():
            raise ValueError("prompt must not be empty")

        async with self._lock:
            self.config.write_model_config()
            process = await asyncio.create_subprocess_exec(
                self.config.executable,
                "--print",
                "--no-session",
                "--provider",
                self.config.provider,
                "--model",
                self.config.model,
                prompt,
                cwd=self.config.workspace,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await process.communicate()
            if process.returncode != 0:
                detail = stderr.decode(errors="replace").strip()
                raise RuntimeError(f"Pi exited with status {process.returncode}: {detail}")
            return stdout.decode(errors="replace").strip()
