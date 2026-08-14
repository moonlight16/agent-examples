from urllib.parse import urlparse

from pydantic_settings import BaseSettings


class Configuration(BaseSettings):
    """Environment-driven config for the minimal PVC agent.

    Persistence paths default to a Kagenti StatefulSet PVC mount at
    /shared. Both the LangGraph checkpointer and the A2A task store use
    SQLite files at those paths so agent state survives pod restarts
    without any additional infrastructure.
    """

    llm_model: str = "qwen3:4b"
    llm_api_base: str = "http://host.docker.internal:11434/v1"
    llm_api_key: str = "dummy"

    # LangGraph checkpointer SQLite file. When empty, MemorySaver is
    # used (state does NOT survive pod restart) — useful for tests.
    checkpoint_path: str = "/shared/checkpoints.db"

    # A2A task store SQLite file. When empty, InMemoryTaskStore is used.
    task_store_path: str = "/shared/tasks.db"

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
