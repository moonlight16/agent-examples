from urllib.parse import urlparse

from pydantic_settings import BaseSettings


class Configuration(BaseSettings):
    """Environment-driven config for the minimal PVC agent."""

    llm_model: str = "qwen3:4b"
    llm_api_base: str = "http://host.docker.internal:11434/v1"
    llm_api_key: str = "dummy"

    # Where per-turn JSONL is written. Default matches the Kagenti
    # StatefulSet PVC mount path.
    context_dir: str = "/shared/minimal-pvc-agent"

    # Optional Postgres DSNs. When unset, the agent falls back to
    # in-memory task storage and the LangGraph MemorySaver.
    task_store_db_url: str = ""
    checkpoint_db_url: str = ""

    @property
    def has_valid_api_key(self) -> bool:
        """Mirror weather_service: dummy keys are fine for localhost LLMs."""
        key = (self.llm_api_key or "").strip()
        base = (self.llm_api_base or "").strip().lower()
        host = urlparse(base).hostname or ""
        is_local = host in {"localhost", "127.0.0.1", "host.docker.internal"} or host.endswith(".local")
        if is_local:
            return True
        if not key or key.lower() in {"dummy", "changeme", "your-api-key-here"}:
            return False
        return True
