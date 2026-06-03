"""Bearer token authentication middleware for the A2A server."""

from __future__ import annotations

import logging
import secrets

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp

logger = logging.getLogger(__name__)

# Paths that bypass auth entirely. Agent discovery must remain public so clients
# can fetch the AgentCard before they have credentials.
PUBLIC_PATHS = (
    "/.well-known/agent.json",
    "/.well-known/agent-card.json",
)


class BearerAuthMiddleware(BaseHTTPMiddleware):
    """Require `Authorization: Bearer <token>` matching the configured API key.

    If `api_key` is None or empty, the middleware is a no-op (auth disabled).
    """

    def __init__(self, app: ASGIApp, api_key: str | None) -> None:
        super().__init__(app)
        self.api_key = api_key or None

    async def dispatch(self, request: Request, call_next) -> Response:
        if self.api_key is None:
            return await call_next(request)

        if request.url.path in PUBLIC_PATHS:
            return await call_next(request)

        # Allow CORS preflights through; the browser will retry with auth.
        if request.method == "OPTIONS":
            return await call_next(request)

        auth = request.headers.get("authorization", "")
        if not auth.lower().startswith("bearer "):
            return JSONResponse(
                {"error": "missing or invalid Authorization header"},
                status_code=401,
                headers={"WWW-Authenticate": 'Bearer realm="kagenti-chat"'},
            )

        provided = auth[7:].strip()
        if not secrets.compare_digest(provided, self.api_key):
            logger.warning("Rejected request with bad API key from %s", request.client.host if request.client else "?")
            return JSONResponse({"error": "invalid API key"}, status_code=403)

        return await call_next(request)
