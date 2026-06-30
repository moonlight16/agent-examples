"""Per-turn JSONL writer keyed by sanitized A2A context_id.

This is the simple, see-with-your-own-eyes proof of persistence: every
chat turn appends one JSONL line under <context_dir>/<sanitized>.jsonl.
Best-effort — IO errors are logged at WARNING and swallowed so the
agent stays responsive when /shared is unavailable.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_SANITIZE_RE = re.compile(r"[^A-Za-z0-9._-]")
_MAX_STEM = 128


def sanitize_context_id(context_id: str | None) -> str:
    if not context_id:
        return "unknown"
    sanitized = _SANITIZE_RE.sub("_", context_id).strip("._-")
    if not sanitized:
        return "unknown"
    return sanitized[:_MAX_STEM]


def context_file(root: Path, context_id: str | None) -> Path:
    return Path(root) / f"{sanitize_context_id(context_id)}.jsonl"


def append_turn(
    root: Path,
    context_id: str | None,
    task_id: str | None,
    user_input: str,
    agent_reply: str,
) -> Path | None:
    """Append one turn record. Returns the file path on success, None on failure."""
    target = Path(root)
    path = context_file(target, context_id)
    record: dict[str, Any] = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "context_id": context_id,
        "task_id": task_id,
        "user_input": user_input,
        "agent_reply": agent_reply,
    }
    try:
        target.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        return path
    except OSError as exc:
        logger.warning("minimal-pvc-agent: persist failed at %s: %s", path, exc)
        return None


def read_history(root: Path, context_id: str | None) -> list[dict[str, Any]]:
    """Read all turn records for a context_id. Returns [] if no file exists."""
    path = context_file(Path(root), context_id)
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.warning("minimal-pvc-agent: skipping malformed JSONL line")
        return records
    except OSError as exc:
        logger.warning("minimal-pvc-agent: read failed at %s: %s", path, exc)
        return []
