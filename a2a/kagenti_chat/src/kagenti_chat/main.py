"""Main entry point for Kagenti Chat."""

import logging
import sys

import uvicorn

from kagenti_chat.a2a_server import create_app
from kagenti_chat.config import Settings

logger = logging.getLogger(__name__)


def setup_logging(settings: Settings) -> None:
    logging.basicConfig(
        level=settings.LOG_LEVEL,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stdout,
    )


def run() -> None:
    settings = Settings()  # type: ignore[call-arg]
    setup_logging(settings)

    logger.info("Starting Kagenti Chat")
    logger.info("LLM model: %s", settings.LLM_MODEL)
    if settings.LLM_BASE_URL:
        logger.info("LLM base URL: %s", settings.LLM_BASE_URL)

    try:
        app = create_app(settings)
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to create A2A app: %s", exc, exc_info=True)
        sys.exit(1)

    logger.info("A2A server listening on %s:%s", settings.A2A_HOST, settings.A2A_PORT)
    uvicorn.run(
        app,
        host=settings.A2A_HOST,
        port=settings.A2A_PORT,
        log_level=settings.LOG_LEVEL.lower(),
    )


if __name__ == "__main__":
    run()
