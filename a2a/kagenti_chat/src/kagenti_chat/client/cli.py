"""Claude/Codex-style CLI client for kagenti-chat A2A agent."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import httpx
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit.styles import Style
from rich.console import Console
from rich.markdown import Markdown
from rich.text import Text

DEFAULT_URL = "https://kagenti-chat.163-75-85-180.sslip.io"
HISTORY_DIR = Path.home() / ".kagenti-chat"
INPUT_HISTORY_FILE = HISTORY_DIR / "input_history"
CHAT_HISTORY_FILE = HISTORY_DIR / "history.jsonl"

console = Console()

PROMPT_STYLE = Style.from_dict(
    {
        "prompt": "ansibrightblue bold",
    }
)


def print_banner(url: str, agent_name: str, model: str | None) -> None:
    console.print()
    console.print(Text("kagenti-chat", style="bold cyan"))
    if model:
        console.print(f"  [dim]{agent_name} • {model}[/]")
    else:
        console.print(f"  [dim]{agent_name}[/]")
    console.print(f"  [dim]{url}[/]")
    console.print(f"  [dim]Type [bold]/help[/] for commands, [bold]/exit[/] to quit[/]")
    console.print()


def print_help() -> None:
    console.print()
    console.print("[bold]Commands[/]")
    console.print("  [cyan]/help[/]      Show this help")
    console.print("  [cyan]/clear[/]     Clear conversation context")
    console.print("  [cyan]/save[/]      Save the current conversation to a file")
    console.print("  [cyan]/info[/]      Show connection info")
    console.print("  [cyan]/exit[/]      Quit (also: /quit, Ctrl-D)")
    console.print()
    console.print("[bold]Tips[/]")
    console.print("  • Use [cyan]Ctrl-C[/] to cancel a streaming response")
    console.print("  • Press [cyan]Up/Down[/] to navigate input history")
    console.print()


def append_history(record: dict) -> None:
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    record["ts"] = datetime.now(timezone.utc).isoformat()
    with CHAT_HISTORY_FILE.open("a") as f:
        f.write(json.dumps(record) + "\n")


async def fetch_agent_card(client: httpx.AsyncClient, url: str) -> dict:
    """Try the modern path first, fall back to the deprecated one.

    The agent card is intentionally unauthenticated, so this never sends the
    bearer token.
    """
    last_error: Exception | None = None
    for path in ("/.well-known/agent-card.json", "/.well-known/agent.json"):
        try:
            r = await client.get(f"{url}{path}", timeout=10.0)
            if r.status_code == 200:
                return r.json()
            last_error = RuntimeError(f"HTTP {r.status_code}")
        except httpx.HTTPError as e:
            last_error = e
            continue
    raise RuntimeError(f"Could not fetch agent card from {url}: {last_error}")


def extract_text_from_part(part: dict) -> str:
    if isinstance(part, dict):
        if "text" in part:
            return part["text"]
        if part.get("kind") == "text" and "text" in part:
            return part["text"]
    return ""


def render_agent_text(text: str) -> None:
    """Render markdown if it looks like markdown, otherwise plain."""
    if any(marker in text for marker in ("```", "**", "##", "- ", "1. ", "* ")):
        console.print(Markdown(text))
    else:
        console.print(text)


async def stream_message(
    client: httpx.AsyncClient,
    url: str,
    user_input: str,
    context_id: str | None,
    api_key: str | None = None,
) -> tuple[str, str | None]:
    """Send a message via JSON-RPC streaming. Returns (final_text, new_context_id)."""
    message_id = uuid4().hex
    payload = {
        "jsonrpc": "2.0",
        "id": uuid4().hex,
        "method": "message/stream",
        "params": {
            "message": {
                "role": "user",
                "parts": [{"kind": "text", "text": user_input}],
                "messageId": message_id,
            }
        },
    }
    if context_id:
        payload["params"]["message"]["contextId"] = context_id

    final_text = ""
    new_context_id = context_id
    streaming_supported = True

    headers = {"Accept": "text/event-stream"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        async with client.stream(
            "POST",
            f"{url}/",
            json=payload,
            headers=headers,
            timeout=httpx.Timeout(connect=10.0, read=120.0, write=10.0, pool=10.0),
        ) as response:
            if response.status_code in (401, 403):
                raise RuntimeError(_auth_error_message(response.status_code))
            content_type = response.headers.get("content-type", "")
            if "event-stream" not in content_type:
                # Server doesn't actually stream; fall back to non-streaming
                streaming_supported = False
                response.aclose
            else:
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data_str = line[5:].strip()
                    if not data_str:
                        continue
                    try:
                        event = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    result = event.get("result", {})
                    if result.get("contextId"):
                        new_context_id = result["contextId"]

                    # Streaming chunks come as task status updates with messages,
                    # or artifact updates with parts.
                    artifacts = result.get("artifacts") or []
                    for artifact in artifacts:
                        for part in artifact.get("parts", []):
                            chunk = extract_text_from_part(part)
                            if chunk and chunk not in final_text:
                                # Avoid duplicating since artifacts re-send full text
                                final_text = chunk

                    # Some servers stream incremental text in status messages
                    status = result.get("status") or {}
                    state = status.get("state")
                    if state == "failed":
                        msg = status.get("message", {})
                        for part in msg.get("parts", []):
                            err = extract_text_from_part(part)
                            if err:
                                raise RuntimeError(err)
    except httpx.HTTPError as e:
        raise RuntimeError(f"Connection error: {e}") from e

    if streaming_supported and final_text:
        return final_text, new_context_id

    # Fallback: non-streaming send
    return await send_message(client, url, user_input, context_id, api_key, message_id)


def _auth_error_message(status_code: int) -> str:
    if status_code == 401:
        return "Authentication required (HTTP 401). Pass --api-key or set KAGENTI_CHAT_API_KEY."
    return "API key rejected (HTTP 403). Check that your --api-key is correct."


async def send_message(
    client: httpx.AsyncClient,
    url: str,
    user_input: str,
    context_id: str | None,
    api_key: str | None = None,
    message_id: str | None = None,
) -> tuple[str, str | None]:
    payload = {
        "jsonrpc": "2.0",
        "id": uuid4().hex,
        "method": "message/send",
        "params": {
            "message": {
                "role": "user",
                "parts": [{"kind": "text", "text": user_input}],
                "messageId": message_id or uuid4().hex,
            }
        },
    }
    if context_id:
        payload["params"]["message"]["contextId"] = context_id

    headers: dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    response = await client.post(
        f"{url}/",
        json=payload,
        headers=headers,
        timeout=httpx.Timeout(connect=10.0, read=120.0, write=10.0, pool=10.0),
    )
    if response.status_code in (401, 403):
        raise RuntimeError(_auth_error_message(response.status_code))
    response.raise_for_status()
    data = response.json()

    if "error" in data:
        raise RuntimeError(data["error"].get("message", "Unknown error"))

    result = data.get("result", {})
    new_context_id = result.get("contextId") or context_id

    if result.get("status", {}).get("state") == "failed":
        for part in result.get("status", {}).get("message", {}).get("parts", []):
            text = extract_text_from_part(part)
            if text:
                raise RuntimeError(text)
        raise RuntimeError("Agent task failed")

    artifacts = result.get("artifacts", [])
    for artifact in artifacts:
        for part in artifact.get("parts", []):
            text = extract_text_from_part(part)
            if text:
                return text, new_context_id

    return "(no response)", new_context_id


async def chat_loop(url: str, insecure: bool, no_stream: bool, api_key: str | None) -> int:
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)

    verify = not insecure
    timeout = httpx.Timeout(connect=10.0, read=120.0, write=10.0, pool=10.0)

    async def _connect(verify_value: bool) -> tuple[httpx.AsyncClient, dict]:
        c = httpx.AsyncClient(verify=verify_value, timeout=timeout, follow_redirects=True)
        try:
            card = await fetch_agent_card(c, url)
            return c, card
        except Exception:
            await c.aclose()
            raise

    try:
        with console.status("[dim]Connecting...[/]", spinner="dots"):
            try:
                client, card = await _connect(verify)
            except (httpx.ConnectError, RuntimeError) as e:
                # Auto-retry with TLS verification disabled on cert errors.
                # Self-signed sslip.io clusters are common.
                msg = str(e).lower()
                if verify and ("ssl" in msg or "certificate" in msg or "self-signed" in msg or "self signed" in msg):
                    console.print("[yellow]TLS cert not trusted; retrying with --insecure[/]")
                    client, card = await _connect(False)
                else:
                    raise
    except Exception as e:
        console.print(f"[red]Failed to connect to {url}: {e}[/]")
        console.print(f"[dim]If the server uses a self-signed cert, retry with: kagenti-chat --insecure[/]")
        return 1

    try:
        agent_name = card.get("name", "Agent")
        # Try to find model in description or skills (best effort)
        model = None
        desc = card.get("description", "")
        if "Llama" in desc:
            model = "Llama"

        print_banner(url, agent_name, model)

        session: PromptSession[str] = PromptSession(
            history=FileHistory(str(INPUT_HISTORY_FILE)),
            style=PROMPT_STYLE,
            multiline=False,
        )

        context_id: str | None = None
        conversation: list[dict] = []

        while True:
            try:
                with patch_stdout():
                    user_input = await session.prompt_async([("class:prompt", "› ")])
            except (EOFError, KeyboardInterrupt):
                console.print()
                console.print("[dim]Goodbye![/]")
                return 0

            user_input = user_input.strip()
            if not user_input:
                continue

            # Slash commands
            if user_input.startswith("/"):
                cmd = user_input.split()[0].lower()
                if cmd in ("/exit", "/quit"):
                    console.print("[dim]Goodbye![/]")
                    return 0
                if cmd == "/help":
                    print_help()
                    continue
                if cmd == "/clear":
                    context_id = None
                    conversation = []
                    console.clear()
                    print_banner(url, agent_name, model)
                    console.print("[dim](conversation cleared)[/]")
                    continue
                if cmd == "/info":
                    console.print()
                    console.print(f"  [bold]URL[/]        {url}")
                    console.print(f"  [bold]Agent[/]      {agent_name}")
                    console.print(f"  [bold]Context[/]    {context_id or '(new conversation)'}")
                    console.print(f"  [bold]Messages[/]   {len(conversation)}")
                    console.print(f"  [bold]History[/]    {CHAT_HISTORY_FILE}")
                    console.print()
                    continue
                if cmd == "/save":
                    out_file = Path.cwd() / f"kagenti-chat-{datetime.now().strftime('%Y%m%d-%H%M%S')}.md"
                    with out_file.open("w") as f:
                        f.write(f"# Kagenti Chat conversation\n\n")
                        f.write(f"_Saved {datetime.now(timezone.utc).isoformat()}_\n\n")
                        for entry in conversation:
                            role = "**You**" if entry["role"] == "user" else "**Agent**"
                            f.write(f"{role}: {entry['text']}\n\n")
                    console.print(f"[dim]Saved to {out_file}[/]")
                    continue
                console.print(f"[red]Unknown command: {cmd}[/] — type /help")
                continue

            # Send to agent
            conversation.append({"role": "user", "text": user_input})
            append_history({"role": "user", "text": user_input, "context_id": context_id})

            try:
                with console.status("[dim]Thinking...[/]", spinner="dots"):
                    if no_stream:
                        text, context_id = await send_message(client, url, user_input, context_id, api_key)
                    else:
                        text, context_id = await stream_message(client, url, user_input, context_id, api_key)
            except KeyboardInterrupt:
                console.print()
                console.print("[dim](cancelled)[/]")
                continue
            except Exception as e:
                console.print(f"[red]Error: {e}[/]")
                continue

            console.print()
            render_agent_text(text)
            console.print()

            conversation.append({"role": "agent", "text": text})
            append_history({"role": "agent", "text": text, "context_id": context_id})
    finally:
        await client.aclose()


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="kagenti-chat",
        description="Claude/Codex-style CLI for the Kagenti Chat A2A agent.",
    )
    parser.add_argument(
        "--url",
        default=os.environ.get("KAGENTI_CHAT_URL", DEFAULT_URL),
        help=f"Agent URL (default: {DEFAULT_URL}, env: KAGENTI_CHAT_URL)",
    )
    parser.add_argument(
        "--insecure",
        "-k",
        action="store_true",
        default=os.environ.get("KAGENTI_CHAT_INSECURE", "").lower() in ("1", "true", "yes"),
        help="Skip TLS certificate verification (default: env KAGENTI_CHAT_INSECURE)",
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Disable streaming responses",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("KAGENTI_CHAT_API_KEY"),
        help="Bearer token for authentication (env: KAGENTI_CHAT_API_KEY)",
    )
    args = parser.parse_args()

    try:
        sys.exit(asyncio.run(chat_loop(args.url, args.insecure, args.no_stream, args.api_key)))
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":
    main()
